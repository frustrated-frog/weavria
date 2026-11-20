"""重放源码长任务第 16 次调用,核对原响应散列、字段纠错和最终发布。"""

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

from examples.m1_corpus import ReadOutput, SnapshotCorpus
from examples.m1_harness import (
    DATA,
    TrialTransport,
    grade_output,
    load_dataset,
    load_environment,
    verify,
)
from examples.work_note_retry_replay import archived_inputs, frozen_session
from weavria.core import (
    CompletionNotice,
    CompletionUnverifiable,
    FinishAction,
    ModelOutputRejected,
    ModelTurn,
    RunEnded,
    RunStatus,
    SessionContextCompiled,
    SessionRequest,
    SessionRunEvent,
    TurnDecided,
)
from weavria.evaluation import summarize_samples
from weavria.harness import WorkingState
from weavria.models import FakeSessionProvider, ModelOutputError, OpenAICompatibleSessionProvider

BATCH = Path(".weavria/m1/luna-work-note-final-20261120")


class FrozenResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    async def complete(self, body: dict[str, Any]) -> object:
        return {"choices": [{"finish_reason": "stop", "message": {"content": self.content}}]}


class LastCall(FakeSessionProvider):
    """前十五轮原样重放。最后一轮只允许增加宿主声明的交付字段。"""

    def __init__(
        self,
        requests: dict[int, str],
        turns: dict[int, ModelTurn],
        archived: SessionRequest,
        delegate: OpenAICompatibleSessionProvider,
    ) -> None:
        super().__init__(())
        self.digests, self.turns, self.archived, self.delegate = requests, turns, archived, delegate
        self.count = 0

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.count += 1
        original = request
        if self.count == 16:
            notice, old = request.current.control, self.archived.current.control
            if not (
                isinstance(notice, CompletionNotice)
                and isinstance(old, CompletionNotice)
                and notice.decision.json_object_fields
                and old.decision.schema_version == 1
            ):
                raise ValueError("missing declared delivery repair at the fork")
            legacy = notice.decision.model_copy(
                update={"schema_version": 1, "json_object_fields": ()}
            )
            original = request.model_copy(
                update={
                    "current": request.current.model_copy(
                        update={"control": notice.model_copy(update={"decision": legacy})}
                    )
                }
            )
        if (
            hashlib.sha256(original.model_dump_json().encode()).hexdigest()
            != self.digests[self.count]
        ):
            raise ValueError(f"archived input differs at model call {self.count}")
        if self.count == 16:
            return await self.delegate.generate_session(request)
        return self.turns[self.count]


async def replay(batch: Path, live: bool, env_file: Path | None) -> dict[str, object]:
    dataset = load_dataset()
    manifest = json.loads((batch / "manifest.json").read_text())
    expected = verify(dataset)
    if (
        any(manifest[key] != expected[key] for key in ("dataset_sha256", "corpus_sha256"))
        or manifest["model"] != dataset.model.model_dump()
        or manifest["budgets"] != dataset.budgets.model_dump()
    ):
        raise ValueError("source, dataset, model or budget changed")
    case = next(c for c in dataset.cases if c.case_id == "repository_long_context")
    corpus = SnapshotCorpus.load(DATA / "corpus.json", dataset.corpora[case.domain])
    path = batch / f"{case.case_id}-coverage-1" / "session.db"
    events, first, instructions, observations = archived_inputs(path)
    head = first.snapshot
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        archived = SessionRequest.model_validate_json(
            db.execute(
                "SELECT data FROM request_views WHERE run_id=? AND model_call=16", (head.run_id,)
            ).fetchone()[0]
        )
    digests: dict[int, str] = {}
    turns: dict[int, ModelTurn] = {}
    diag = None
    terminal = None
    n = 0
    for event in events:
        e = event.data
        if isinstance(e, SessionContextCompiled):
            n = e.model_call
            digests[n] = e.compilation.request_sha256
        elif isinstance(e, SessionRunEvent):
            if isinstance(e.event.data, TurnDecided):
                turns[n] = e.event.data.turn
            elif n == 16 and isinstance(e.event.data, ModelOutputRejected):
                diag = e.event.data.diagnostic
            elif isinstance(e.event.data, RunEnded):
                terminal = e.event.data.reason
    # 归档只保存响应指纹。仅在候选正文能逐字节复原时使用这个现场,
    # 评分数据仅参与离线身份核对和事后评分,从不进入请求、工具或发布策略。
    raw = json.dumps(
        {**case.expected, "sources": list(case.required_sources)}, separators=(",", ":")
    )
    if (
        diag is None
        or diag.code != "schema_mismatch"
        or diag.finish_reason != "stop"
        or terminal != "model_output_repair_exhausted"
        or len(raw.encode()) != diag.response_bytes
        or hashlib.sha256(raw.encode()).hexdigest() != diag.response_sha256
    ):
        raise ValueError("cannot identify the original rejected response")
    before = turns[15].decision
    assert isinstance(before, FinishAction) and isinstance(
        archived.current.control, CompletionNotice
    )
    if (
        hashlib.sha256(before.model_dump_json().encode()).hexdigest()
        != archived.current.control.candidate_sha256
    ):
        raise ValueError("repair did not refer to the original candidate")
    config = dataset.model.provider_config()
    local = OpenAICompatibleSessionProvider(config, transport=FrozenResponse(raw))
    try:
        await local.generate_session(archived)
    except ModelOutputError as error:
        if error.diagnostic != diag:
            raise ValueError("legacy diagnostic differs") from error
    else:
        raise ValueError("old notice unexpectedly accepted a bare object")
    transport = None
    if live:
        load_environment(env_file)
        if (
            os.getenv("WEAVRIA_MODEL_BASE_URL", "").rstrip("/") != "https://api.openai.com/v1"
            or os.getenv("WEAVRIA_MODEL_NAME") != config.model
        ):
            raise ValueError("live fork requires the frozen model and official OpenAI endpoint")
        transport = TrialTransport(
            "https://api.openai.com/v1",
            os.environ["WEAVRIA_MODEL_API_KEY"],
            dataset.budgets.model_seconds,
        )
    model = LastCall(
        digests,
        turns,
        archived,
        OpenAICompatibleSessionProvider(config, transport=transport) if transport else local,
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            async with frozen_session(
                Path(directory),
                model=model,
                initial=first,
                instructions=instructions,
                dataset=dataset,
                case=case,
                trial="coverage-1",
                corpus=corpus,
            ) as engine:
                result = await engine.send(
                    head.session_id,
                    case.request,
                    input_id=head.input_id,
                    application_context=observations,
                )
                saved = await engine.store.semantic_head(head.session_id, head.work_id)
                assert saved and saved.working_state is not None
                work = WorkingState.model_validate_json(json.dumps(saved.working_state))
                (note,) = work.notes
                snippets: list[str] = []
                for ref in note.sources:
                    doc = ReadOutput.model_validate_json(
                        (await engine.store.get(head.run_id, ref.artifact_id)).content_json
                    )
                    assert ref.pointer == "/text" and ref.start is not None and ref.end is not None
                    if (
                        doc.path not in case.required_sources
                        or corpus.documents[doc.path] != doc.text
                    ):
                        raise ValueError("committed note differs from frozen source")
                    snippets.append(doc.text[ref.start : ref.end])
                if not all(
                    any(marker in text for text in snippets)
                    for marker in (
                        "max_intervention_reconsiderations: int = 2",
                        'call.tool_name != "read_artifact"',
                        "proposed.start != 0",
                        "proposed.start == page.end",
                    )
                ):
                    raise ValueError("committed ranges do not support the four claims")
                grade = grade_output(case, result.report.output)
                if not live and (
                    model.count != 16
                    or result.report.status is not RunStatus.COMPLETED
                    or result.report.output != raw
                    or grade["passed"] is not True
                ):
                    raise ValueError(
                        f"archived answer was not published: {result.report.execution.status}"
                    )
                assessment = result.report.assessment
                return {
                    "arm": "live" if live else "frozen",
                    "old_terminal": terminal,
                    "archived_response_sha256": diag.response_sha256,
                    "model_calls": model.count,
                    "initial_grade": grade_output(case, before.output),
                    "status": result.report.status.value,
                    "task_outcome": result.report.task_outcome,
                    "unverifiable_reason": (
                        assessment.decision.reason
                        if assessment and isinstance(assessment.decision, CompletionUnverifiable)
                        else None
                    ),
                    "grade": grade,
                    "provider": summarize_samples(tuple(transport.samples)) if transport else None,
                }
    finally:
        if transport:
            await transport.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=BATCH)
    parser.add_argument("--live", action="store_true", help="仅在原第 16 次调用访问供应商")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(replay(args.batch, args.live, args.env_file)), ensure_ascii=False))
