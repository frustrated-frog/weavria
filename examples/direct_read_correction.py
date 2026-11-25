"""在冻结的 read_doc 语料中, 从连续的自主单读和持久原件提取离线纠错记录。"""

import hashlib
import sqlite3
from dataclasses import dataclass

from weavria.artifacts import ArtifactSnapshot, resolve_pointer
from weavria.core import (
    ContextSelected,
    ControlEvent,
    ControlRequest,
    FinishAction,
    ReadAssessment,
    ReadCounterquote,
    RunEnded,
    SessionRequest,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolRecord,
    TurnDecided,
    complete_exchange,
)


@dataclass(frozen=True)
class Source:
    name: str
    sha256: str
    tool_version: str


@dataclass(frozen=True)
class Correction:
    task_sha256: str
    original: Source
    replacement: Source
    quote: ReadCounterquote
    original_call_id: str
    replacement_call_id: str


def task_digest(task: TaskInput) -> str:
    return hashlib.sha256(task.model_dump_json().encode()).hexdigest()


def trace_digest(db: sqlite3.Connection, run_id: str) -> str:
    """以有序事件与请求绑定本次运行。相同初始请求可能产生完全不同的后续判断。"""
    digest = hashlib.sha256()
    for sql in (
        "SELECT data FROM run_events WHERE run_id=? ORDER BY sequence",
        "SELECT data FROM request_views WHERE run_id=? ORDER BY model_call",
    ):
        digest.update(b"\x00")
        for (data,) in db.execute(sql, (run_id,)):
            encoded = data.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _delivered(
    db: sqlite3.Connection, run_id: str, action: ToolCallAction, view: ControlRequest
) -> tuple[Source, ArtifactSnapshot] | None:
    feedback = view.feedback
    if (
        action.tool_name != "read_doc"
        or feedback is None
        or feedback.task_revision != view.task.revision
        or len(feedback.calls) != 1
        or feedback.calls[0].call_id != action.call_id
        or feedback.calls[0].status != "succeeded"
    ):
        return None
    row = db.execute(
        """SELECT o.record,a.reference,a.content FROM outcomes o JOIN artifacts a
        ON a.run_id=o.run_id AND a.id=o.artifact_id WHERE o.run_id=? AND o.call_id=?""",
        (run_id, action.call_id),
    ).fetchone()
    if row is None:
        return None
    record = ToolRecord.model_validate_json(row[0])
    artifact = ArtifactSnapshot(row[1], row[2])
    ref, auth = artifact.reference, record.authorization
    if (
        record.exchange.action != action
        or record.artifact != ref
        or ref.run_id != run_id
        or auth is None
        or not auth.allowed
        or not isinstance(name := action.arguments.get("name"), str)
        or auth.normalized_arguments != {"name": name, "digest": ref.sha256}
    ):
        return None
    expected = complete_exchange(action, ref, artifact.value)
    if not any(
        isinstance(e, ToolExchange) and e.model_dump_json() == expected.model_dump_json()
        for e in view.history
    ):
        return None
    return Source(name, ref.sha256, auth.tool_version), artifact


def _verified_quote(
    assessment: ReadAssessment | None, artifact: ArtifactSnapshot
) -> ReadCounterquote | None:
    if (
        assessment is None
        or assessment.quote is None
        or assessment.source_sha256 != artifact.reference.sha256
    ):
        return None
    q = assessment.quote
    try:
        value = resolve_pointer(artifact.value, q.pointer)
    except (KeyError, IndexError, ValueError):
        return None
    if not isinstance(value, str) or q.end > len(value):
        return None
    passage = value[q.start : q.end]
    if (
        not any(c.isalnum() for c in passage)
        or value.find(passage) != q.start
        or value.find(passage, q.start + 1) >= 0
        or hashlib.sha256(passage.encode()).hexdigest() != q.quote_sha256
    ):
        return None
    return q


def extract_correction(db: sqlite3.Connection, run_id: str) -> Correction | None:
    """严格回放三轮已完成的决策。额外决策、重试、任务修订和不完整的原件均弃权。"""
    rows = db.execute(
        """SELECT v.model_call,v.data FROM request_views v JOIN runs r ON r.id=v.run_id
        WHERE r.id=? AND r.status='completed' ORDER BY v.model_call LIMIT 4""",
        (run_id,),
    ).fetchall()
    if [n for n, _ in rows] != [1, 2, 3]:
        return None
    requests = [SessionRequest.model_validate_json(data) for _, data in rows]
    first, second, third = (r.current for r in requests)
    if (
        any(
            v.control is not None or v.task_model_view is not None or v.task_observation is not None
            for v in (first, second, third)
        )
        or first.feedback is not None
        or first.task != second.task
        or first.task != third.task
    ):
        return None
    events = [
        ControlEvent.model_validate_json(row[0]).data
        for row in db.execute(
            """SELECT data FROM run_events WHERE run_id=? AND json_extract(data,'$.data.kind')
        IN ('context_selected','turn_decided','batch_rejected','model_output_rejected',
            'decision_discarded','task_revised','run_ended')
        ORDER BY sequence LIMIT 8""",
            (run_id,),
        )
    ]
    if len(events) != 7 or not isinstance(end := events[-1], RunEnded):
        return None
    turns: list[TurnDecided] = []
    for request, context, decision in zip(requests, events[:-1:2], events[1::2], strict=True):
        if not isinstance(context, ContextSelected) or not isinstance(decision, TurnDecided):
            return None
        if (
            context.request_sha256
            != hashlib.sha256(request.current.model_dump_json().encode()).hexdigest()
        ):
            return None
        manifest = context.delivery_manifest
        if request.current.feedback is not None and (
            manifest is None
            or manifest.transport_request_sha256
            != hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        ):
            return None
        turns.append(decision)
    a, b, c = turns
    old, new, finish = a.turn.decision, b.turn.decision, c.turn.decision
    if not (
        isinstance(old, ToolBatch)
        and len(old.calls) == 1
        and isinstance(new, ToolBatch)
        and len(new.calls) == 1
        and isinstance(finish, FinishAction)
        and end.result.output == finish.output
    ):
        return None
    original = _delivered(db, run_id, old.calls[0], second)
    replacement = _delivered(db, run_id, new.calls[0], third)
    if original is None or replacement is None or original[0].name == replacement[0].name:
        return None
    quote = _verified_quote(b.turn.read_assessment, original[1])
    # 引用是模型在完整送达后选用替代来源的证据。答案正确性另由任务判分器评估。
    if quote is None or not finish.output.strip().endswith(f" [{replacement[0].name}]"):
        return None
    return Correction(
        task_digest(first.task),
        original[0],
        replacement[0],
        quote,
        old.calls[0].call_id,
        new.calls[0].call_id,
    )


def recommend(
    correction: Correction | None, task: TaskInput, candidates: tuple[Source, ...]
) -> Source | None:
    """离线建议只接受同一任务及两份同版候选。调用方按同一会话种子、同一配置配对。"""
    if (
        correction is not None
        and task_digest(task) == correction.task_sha256
        and candidates == (correction.original, correction.replacement)
    ):
        return correction.replacement
    return candidates[0] if candidates else None
