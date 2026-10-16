"""Completion-control ablation over local files, with one premature Finish injection."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal

from agent_loop import Case, Catalogue, CatalogueReader, Probe, file_tool
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from weavria.artifacts import InMemoryArtifactStore
from weavria.context import EvidenceMode
from weavria.core import (
    AdmissionNotice,
    BatchRejected,
    CompletionAccept,
    CompletionDecision,
    CompletionRepair,
    CompletionUnverifiable,
    ControlRequest,
    FinishAction,
    ModelTurn,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    TurnRequest,
)
from weavria.models import ModelCapabilities
from weavria.runtime import AgentLoop, CompletionContext, RunLimits, ToolScheduling, reduce_event
from weavria.tools import ToolRegistry, resolve_pointer


class Release(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    release: str
    artifact_id: str
    pointer: Literal["/document/release"]


class Answer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    complete: bool
    answers: dict[str, Release]


class ControlledReader:
    """Same interpreter in both arms; injected premature finish tests intervention.

    No expected answers or recoverability labels enter the provider. Later actions
    still use the catalogue, actual tool feedback and batch denial.
    """

    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self) -> None:
        self._reader = CatalogueReader()
        self._premature = True
        self._next_id = 0
        self._batch_limit = 32

    async def generate_turn(self, request: ControlRequest) -> ModelTurn:
        if isinstance(request.control, AdmissionNotice):
            self._batch_limit = max(1, request.control.proposed_calls // 2)
        if self._premature and any(isinstance(e, ToolExchange) for e in request.history):
            self._premature = False
            return ModelTurn(decision=FinishAction(output='{"complete":true,"answers":{}}'))
        legacy_view = TurnRequest(
            instructions=request.instructions,
            task=request.task,
            history=request.history,
            feedback=request.feedback,
        )
        turn = await self._reader.generate_turn(legacy_view)
        if isinstance(turn.decision, ToolBatch):
            calls: list[ToolCallAction] = []
            for proposed in turn.decision.calls[: self._batch_limit]:
                calls.append(
                    ToolCallAction(
                        call_id=f"source-{self._next_id}",
                        tool_name=proposed.tool_name,
                        arguments=proposed.arguments,
                    )
                )
                self._next_id += 1
            return ModelTurn(decision=ToolBatch(calls=tuple(calls)))
        return turn


class ReleaseEvidencePolicy:
    """Validate coverage and citations against accepted files, without a judge LLM."""

    version = "catalogue_release/1"

    @staticmethod
    def repair(code: str, guidance: str) -> CompletionRepair:
        return CompletionRepair(issue_codes=(code,), guidance=guidance)

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        try:
            answer = Answer.model_validate_json(context.candidate.output)
        except ValidationError:
            return self.repair(
                "invalid_answer", "Return complete and answers with source citations"
            )
        sources = {r.artifact.artifact_id: r for r in context.records if r.artifact is not None}
        catalogues = [
            r
            for r in sources.values()
            if r.exchange.action.arguments.get("name") == "catalogue.json"
        ]
        if not catalogues:
            return CompletionUnverifiable(reason="No accepted catalogue defines the requirements")
        catalogue_record = catalogues[-1]
        assert catalogue_record.artifact is not None
        raw = (await context.read_artifact(catalogue_record.artifact.artifact_id)).value
        assert isinstance(raw, dict)
        catalogue = Catalogue.model_validate_json(json.dumps(raw["document"]))
        required = {c.name: c for c in catalogue.components}
        if set(answer.answers) - required.keys():
            return self.repair(
                "unknown_component", "Only report components in the accepted catalogue"
            )
        for name, release in answer.answers.items():
            record = sources.get(release.artifact_id)
            if record is None:
                return self.repair("unaccepted_citation", "Cite an accepted source artifact")
            component = required[name]
            allowed = {component.source, component.fallback} - {None}
            if record.exchange.action.arguments.get("name") not in allowed:
                return self.repair(
                    "wrong_source", "Use the source or fallback declared for each component"
                )
            snapshot = await context.read_artifact(release.artifact_id)
            if resolve_pointer(snapshot.value, release.pointer) != release.release:
                return self.repair(
                    "citation_mismatch", "Use the release value from the cited source"
                )
        missing = required.keys() - answer.answers.keys()
        if missing:
            # A model's incomplete claim is insufficient: declared alternatives
            # must have actual failure facts before an unverifiable result.
            failed = {
                name
                for r in context.records
                if isinstance(r.exchange.observation, ToolFailedObservation)
                and isinstance(name := r.exchange.action.arguments.get("name"), str)
            }
            unavailable = all(
                {required[n].source, required[n].fallback} - {None} <= failed for n in missing
            )
            if unavailable and not answer.complete:
                return CompletionUnverifiable(
                    reason="All declared sources for missing components failed"
                )
            return self.repair(
                "missing_components",
                "Read catalogue sources and report available releases; explain missing sources",
            )
        if not answer.complete:
            return self.repair(
                "completion_flag", "All components have checked citations; mark complete"
            )
        return CompletionAccept(outcome="verified")


async def evaluate(case: Case, root: Path, checked: bool) -> dict[str, JsonValue]:
    store, probe = InMemoryArtifactStore(), Probe()
    agent = AgentLoop(
        run_id=f"{case.name}-{checked}",
        task=TaskInput(goal="Find each component release from the catalogue"),
        instructions="Start at catalogue.json. Cite source identities; report missing results.",
        model=ControlledReader(),
        artifacts=store,
        tools=ToolRegistry((file_tool(root / case.name, probe),)),
        completion=ReleaseEvidencePolicy() if checked else None,
        mode=EvidenceMode.FULL,
        limits=RunLimits(max_request_bytes=12000),
        scheduling=ToolScheduling(2, frozenset({"read_source"})),
    )
    result = await agent.execute()
    payload = Answer.model_validate_json(result.output or "{}")
    grounded = True
    for release in payload.answers.values():
        snapshot = await store.get(agent.state.run_id, release.artifact_id)
        grounded = grounded and resolve_pointer(snapshot.value, release.pointer) == release.release
    releases = {name: r.release for name, r in payload.answers.items()}
    state = None
    for event in agent.events:
        state = reduce_event(state, event)
    assert probe.active == 0
    return TypeAdapter(dict[str, JsonValue]).validate_python(
        {
            "case": case.name,
            "policy": "evidence" if checked else "none",
            "correct": releases == case.expected and payload.complete == case.recoverable,
            "completed_goal": payload.complete and releases == case.expected and grounded,
            "grounded": grounded,
            "task_outcome": result.task_outcome,
            "execution_status": result.status.value,
            "model_calls": agent.state.model_calls,
            "tool_calls": agent.state.tool_calls,
            "repairs": agent.state.completion_repairs,
            "rejections": sum(isinstance(e.data, BatchRejected) for e in agent.events),
            "input_bytes": sum(p.request_bytes for p in agent.plans),
            "peak_request_bytes": max(p.request_bytes for p in agent.plans),
            "replay_valid": state == agent.state,
            "evidence_calls": list(result.assessment.evidence_call_ids)
            if result.assessment
            else [],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "data" / "loop"
    cases = TypeAdapter(tuple[Case, ...]).validate_json((root / "cases.json").read_bytes())
    manifest = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.json"))
    }

    async def experiment() -> list[dict[str, JsonValue]]:
        return [await evaluate(case, root, checked) for case in cases for checked in (False, True)]

    results = asyncio.run(experiment())
    report = {
        "schema_version": 3,
        "dataset_version": "1",
        "solver_version": "premature_catalogue/1",
        "evaluator_version": "release/1",
        "policy_version": ReleaseEvidencePolicy.version,
        "configuration": {
            "max_request_bytes": 12000,
            "max_model_calls": 16,
            "max_tool_calls": 12,
            "max_completion_repairs": 3,
            "mode": "full",
            "fault": "one premature empty Finish after catalogue read",
            "judge_llm_calls": 0,
        },
        "manifest": manifest,
        "implementations": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("agent_loop.py"))
        },
        "scope": "deterministic mechanism ablation, not real LLM capability",
        "results": results,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for row in results:
            print(row)


if __name__ == "__main__":
    main()
