"""Frozen historical-discovery and exact-range delivery experiment, without model API calls."""

import argparse
import asyncio
import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from weavria.artifacts import InMemoryArtifactStore
from weavria.context import (
    ContextCompilerConfig,
    ContextDemand,
    HistoryRetrievalConfig,
    SessionContextCompiler,
    SessionTokenCounter,
)
from weavria.core import (
    ControlRequest,
    FinishAction,
    ModelTurn,
    MultiRangeTextView,
    SessionRequest,
    TaskInput,
)
from weavria.models import FakeSessionProvider, OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.sessions import SessionEngine, SQLiteSessionStore
from weavria.tools import ToolRegistry


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    query: str
    first: str
    filler: str
    repeats: int
    last: str
    target_ordinal: int
    expected_phrases: tuple[str, ...]
    source: str = "synthetic"

    @property
    def original(self) -> str:
        return self.first + self.filler * self.repeats + self.last + self.filler * self.repeats


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    dataset_version: str
    evidence_scope: str
    turns: int
    cases: tuple[Case, ...]


@dataclass(frozen=True)
class Result:
    case: str
    policy: str
    expected: int
    delivered: int
    target_candidate: bool
    target_selected: bool
    candidates: int
    source_ranges_valid: bool
    request_sha256: str
    input_tokens: int
    request_bytes: int
    retrieval_complete: bool
    indexed_turns: int
    postings_examined: int
    cold_retrieval_ms: float
    cold_indexed_turns: int
    retrieval_ms: float
    compilation_ms: float


def delivered_text(request: SessionRequest) -> str:
    return "\n".join(
        view.text
        for turn in request.conversation.turns
        for view in (turn.user, turn.assistant)
        if view is not None
    )


async def evaluate(
    dataset: Dataset,
    *,
    counter: SessionTokenCounter,
    window_tokens: int = 5000,
    max_request_bytes: int = 9000,
) -> tuple[Result, ...]:
    """Labels are read only after selection; each candidate set comes from accepted history."""
    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="weavria-history-") as directory:
        for case in dataset.cases:
            path = Path(directory) / f"{case.name}.db"
            original = case.original
            outputs = tuple(
                original if ordinal == case.target_ordinal else f"Lunch notes {ordinal}"
                for ordinal in range(1, dataset.turns + 1)
            )
            model = FakeSessionProvider(
                tuple(ModelTurn(decision=FinishAction(output=text)) for text in outputs)
            )
            identities = iter(f"run-{ordinal}" for ordinal in range(1, dataset.turns + 1))
            async with SessionEngine(
                path, model=model, tools=ToolRegistry(()), id_generator=identities.__next__
            ) as engine:
                await engine.create("s")
                for ordinal in range(1, dataset.turns + 1):
                    await engine.send("s", f"archive entry {ordinal}", input_id=f"input-{ordinal}")
            async with SQLiteSessionStore(path) as store:
                sequence, count = await store.head("s")
                started = time.perf_counter()
                cold = await store.retrieve_history(
                    "s",
                    task_revision=1,
                    before_ordinal=count + 1,
                    through_sequence=sequence,
                    demands=(ContextDemand("current_task", case.query),),
                    config=HistoryRetrievalConfig(recent_turns=2, max_candidates=6),
                )
                cold_ms = (time.perf_counter() - started) * 1000
                for policy, spans in (
                    ("window64", 1),
                    ("retrieval-single", 1),
                    ("retrieval-multi", 4),
                ):
                    started = time.perf_counter()
                    discovery = None
                    if policy == "window64":
                        turns = await store.history("s")
                    else:
                        discovery = await store.retrieve_history(
                            "s",
                            task_revision=1,
                            before_ordinal=count + 1,
                            through_sequence=sequence,
                            demands=(ContextDemand("current_task", case.query),),
                            config=HistoryRetrievalConfig(recent_turns=2, max_candidates=6),
                        )
                        turns = discovery.turns
                    retrieval_ms = (time.perf_counter() - started) * 1000
                    compiler = SessionContextCompiler(
                        InMemoryArtifactStore(),
                        counter=counter,
                        config=ContextCompilerConfig(
                            window_tokens=window_tokens,
                            output_tokens=256,
                            max_history_spans=spans,
                        ),
                    )
                    started = time.perf_counter()
                    compiled = await compiler.compile(
                        run_id="probe",
                        session_id="s",
                        sequence=sequence,
                        base=ControlRequest(
                            instructions="Use original source data.",
                            task=TaskInput(goal=case.query),
                        ),
                        turns=turns,
                        total_turns=count,
                        records=(),
                        max_request_bytes=max_request_bytes,
                        retrieval=discovery.report if discovery else None,
                    )
                    elapsed = (time.perf_counter() - started) * 1000
                    request, report = compiled.request, compiled.compilation
                    valid = True
                    by_run = {turn.run_id: turn for turn in turns}
                    for item in request.conversation.turns:
                        for field in ("user", "assistant"):
                            view = getattr(item, field)
                            text = getattr(by_run[item.run_id], field)
                            if view is None:
                                valid = valid and text is None
                            elif isinstance(view, MultiRangeTextView):
                                valid = valid and all(
                                    p.text == text[p.start : p.end] for p in view.ranges
                                )
                            else:
                                start = getattr(view, "start", 0)
                                valid = valid and view.text == text[start : start + len(view.text)]
                    visible = delivered_text(request)
                    results.append(
                        Result(
                            case=case.name,
                            policy=policy,
                            expected=len(case.expected_phrases),
                            delivered=sum(phrase in visible for phrase in case.expected_phrases),
                            target_candidate=any(t.ordinal == case.target_ordinal for t in turns),
                            target_selected=any(
                                t.ordinal == case.target_ordinal for t in request.conversation.turns
                            ),
                            candidates=len(turns),
                            source_ranges_valid=valid,
                            request_sha256=report.request_sha256,
                            input_tokens=report.input_tokens,
                            request_bytes=report.request_bytes,
                            retrieval_complete=discovery.report.index_complete
                            and discovery.report.search_complete
                            if discovery
                            else True,
                            indexed_turns=discovery.report.indexed_turns if discovery else 0,
                            postings_examined=discovery.report.postings_examined
                            if discovery
                            else 0,
                            cold_retrieval_ms=cold_ms,
                            cold_indexed_turns=cold.report.indexed_turns,
                            retrieval_ms=retrieval_ms,
                            compilation_ms=elapsed,
                        )
                    )
    return tuple(results)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare source discovery and delivery, not LLM task quality"
    )
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--window-tokens", type=int, default=5000)
    parser.add_argument("--max-request-bytes", type=int, default=9000)
    args = parser.parse_args()
    path = Path(__file__).parent / "data/history_context/cases.json"
    dataset = Dataset.model_validate_json(path.read_text())
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="offline-history-probe", max_output_tokens=256),
        encoding=args.encoding,
    )
    results = await evaluate(
        dataset,
        counter=counter,
        window_tokens=args.window_tokens,
        max_request_bytes=args.max_request_bytes,
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": dataset.dataset_version,
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "counter": counter.version,
                "window_tokens": args.window_tokens,
                "max_request_bytes": args.max_request_bytes,
                "evidence_scope": dataset.evidence_scope,
                "results": [asdict(result) for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
