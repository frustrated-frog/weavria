"""Offline design probe; no production request contract or provider is changed."""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Literal

import tiktoken
from pydantic import JsonValue

from apps.release_readiness import (
    ReleaseDossierDiscoveryPolicy,
    load_release_readiness_dataset,
    release_dossier_evidence_profile,
    release_dossier_tool,
    release_specification,
)
from examples.task_understanding import DatasetInterpreter, registry
from weavria.core import SemanticSnapshot, SemanticVersion, TaskInput
from weavria.harness import (
    CapabilityResolver,
    TaskAwareSessionPreparer,
    TaskSpecification,
    TaskUnderstandingEngine,
    load_need_ledger,
    load_task_frame,
)
from weavria.sessions import SessionTaskContext, SessionTaskPreparationError
from weavria.tools import ToolRegistry


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    family: str
    task: TaskInput
    snapshot: SemanticSnapshot


def encoded(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def candidate_view(
    task: TaskInput, snapshot: SemanticSnapshot, phase: Literal["decision", "synthesis"]
) -> dict[str, JsonValue]:
    """Test a loss boundary over prepared fixtures, never accept or authorize facts."""
    frame = load_task_frame(snapshot.observation.payload)
    ledger = load_need_ledger(snapshot.need_ledger)
    version = snapshot.version
    if (
        task.revision != version.task_revision
        or snapshot.observation.task_revision != version.task_revision
        or snapshot.observation.frame_revision != version.frame_revision
        or frame.request_anchor.task_revision != version.task_revision
        or frame.request_anchor.task_sha256
        != hashlib.sha256(task.model_dump_json().encode()).hexdigest()
        or frame.work_id != snapshot.work_id
        or ledger.work_id != snapshot.work_id
        or frame.revision != version.frame_revision
        or ledger.task_revision != version.task_revision
        or ledger.frame_revision != version.frame_revision
        or frame.capability_epoch != version.capability_epoch
        or frame.evidence_revision != version.evidence_revision
        or ledger.evidence_revision != version.evidence_revision
        or frame.need_state_sha256 != ledger.progress_sha256()
        or {n.need_id for n in frame.requirements} != {s.need_id for s in ledger.states}
    ):
        raise ValueError("candidate view requires one consistent task snapshot")
    needs: list[JsonValue] = []
    for requirement in frame.requirements:
        state = ledger.state_for(requirement.need_id)
        needs.append(
            {
                "id": requirement.need_id,
                "description": requirement.description,
                "required": requirement.required,
                "source": requirement.source,
                "freshness": requirement.freshness,
                "completion_mode": requirement.completion_mode,
                "required_fields": list(requirement.required_fields),
                "capabilities": list(requirement.capability_names),
                "depends_on": list(requirement.depends_on),
                "status": state.status.value,
                "missing_fields": list(state.missing_fields),
                "acquisition_status": state.acquisition_status,
                "can_retry": state.can_retry,
                "failures": [failure.model_dump(mode="json") for failure in state.failures],
                "bindings": [
                    binding.model_dump(mode="json")
                    for binding in ledger.bindings
                    if binding.binding_id in state.evidence_binding_ids
                ],
                "targets": [
                    binding.model_dump(mode="json")
                    for plan in requirement.evidence_match_plans
                    for binding in plan.target_bindings
                ],
            }
        )
    return {
        "schema_version": 1,
        "projection_version": "task-model-view-design/0",
        "phase": phase,
        "source_observation_sha256": hashlib.sha256(
            snapshot.observation.model_dump_json().encode()
        ).hexdigest(),
        "source_ledger_sha256": hashlib.sha256(encoded(snapshot.need_ledger)).hexdigest(),
        "semantic_version": version.model_dump(mode="json"),
        "trust": "model_inference",
        "route": frame.specification.route,
        "interpreted_goal": frame.specification.goal,
        "resolved_outcome_key": frame.specification.resolved_outcome_key,
        "ambiguities": list(frame.specification.ambiguities),
        "selected_question": (
            frame.clarification_decision.question_text if frame.clarification_decision else None
        ),
        "requested_targets": [
            {
                "need_index": index,
                "selection": target.model_dump(mode="json"),
                "text": target.text(task),
            }
            for index, need in enumerate(frame.specification.information_needs)
            for target in need.targets
        ],
        "navigation_context": [item.model_dump(mode="json") for item in frame.application_context],
        "needs": needs,
        "obligations": [obligation.model_dump(mode="json") for obligation in frame.obligations],
        "visible_tools": list(snapshot.visible_tool_names) if phase == "decision" else [],
    }


async def prepared_cases() -> tuple[tuple[PreparedCase, ...], tuple[dict[str, str], ...]]:
    root = (await asyncio.to_thread(Path(__file__).resolve)).parents[1]
    data = json.loads(
        await asyncio.to_thread((root / "examples/data/task_understanding/cases.json").read_text)
    )
    generic = [
        (
            case["id"],
            case["request"],
            TaskSpecification.model_validate_json(json.dumps(case["specification"])),
        )
        for case in data["cases"]
    ]
    release = await asyncio.to_thread(
        load_release_readiness_dataset, root / "tests/fixtures/release_readiness_v1.json"
    )
    release_cases = [
        (case.case_id, case.prompt, release_specification(case)) for case in release.cases
    ]
    cases: list[PreparedCase] = []
    failures: list[dict[str, str]] = []
    for family, inputs, tools, profiles, resolver in (
        ("generic_tools", generic, registry(), (), CapabilityResolver()),
        (
            "release_readiness",
            release_cases,
            ToolRegistry((release_dossier_tool(release.dossier),)),
            (release_dossier_evidence_profile(),),
            CapabilityResolver(policy=ReleaseDossierDiscoveryPolicy()),
        ),
    ):
        for case_id, goal, specification in inputs:
            task = TaskInput(goal=goal)
            preparer = TaskAwareSessionPreparer(
                TaskUnderstandingEngine(
                    DatasetInterpreter({goal: specification}), evidence_profiles=profiles
                ),
                capabilities=resolver,
            )
            try:
                prepared = await preparer.prepare(
                    SessionTaskContext(
                        session_id="design", input_id=case_id, task=task, history=(), tools=tools
                    )
                )
            except SessionTaskPreparationError as exc:
                failures.append({"case": case_id, "family": family, "error": exc.error.message})
                continue
            # A synthetic snapshot of preparation output; no durable acceptance is claimed.
            snapshot = SemanticSnapshot(
                session_id="design",
                work_id=prepared.work_id,
                input_id=case_id,
                run_id="design",
                version=SemanticVersion(
                    task_revision=task.revision,
                    frame_revision=prepared.observation.frame_revision,
                    capability_epoch=prepared.capability_epoch,
                    evidence_revision=prepared.evidence_revision,
                    semantic_sequence=1,
                ),
                observation=prepared.observation,
                visible_tool_names=prepared.visible_tool_names,
                need_ledger=prepared.need_ledger,
            )
            cases.append(PreparedCase(case_id, family, task, snapshot))
    return tuple(cases), tuple(failures)


async def build_report() -> dict[str, object]:
    tokenizer = await asyncio.to_thread(tiktoken.get_encoding, "cl100k_base")
    root = (await asyncio.to_thread(Path(__file__).resolve)).parents[1]
    paths = (
        "examples/data/task_understanding/cases.json",
        "tests/fixtures/release_readiness_v1.json",
    )
    rows: list[dict[str, str | int]] = []
    cases, failures = await prepared_cases()
    for case in cases:
        before = case.snapshot.observation.model_dump_json().encode()
        after = encoded(candidate_view(case.task, case.snapshot, "decision"))
        rows.append(
            {
                "case": case.case_id,
                "family": case.family,
                "before_bytes": len(before),
                "candidate_bytes": len(after),
                "before_tokens": len(tokenizer.encode(before.decode())),
                "candidate_tokens": len(tokenizer.encode(after.decode())),
            }
        )
    digests: dict[str, str] = {}
    for path in paths:
        raw = await asyncio.to_thread((root / path).read_bytes)
        digests[path] = hashlib.sha256(raw).hexdigest()
    return {
        "schema_version": 1,
        "projection_version": "task-model-view-design/0",
        "measurement": "task_component_only_not_transport",
        "tokenizer": "cl100k_base",
        "tiktoken_version": await asyncio.to_thread(package_version, "tiktoken"),
        "scope": (
            "initial prepared fixtures; no tool execution, production projection, or model call"
        ),
        "dataset_sha256": digests,
        "preparation_failures": failures,
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(build_report()), ensure_ascii=False, indent=2))
