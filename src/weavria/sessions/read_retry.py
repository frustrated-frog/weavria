"""同会话单读经验索引。在事实提交事务中归因, 从持久请求与工具原件核对交付。"""

import hashlib
import json
import sqlite3

from weavria.artifacts import ArtifactSnapshot
from weavria.core import (
    BatchAdmitted,
    BatchRejected,
    ControlEvent,
    ReadRetryOutcome,
    SessionRequest,
    ToolExchange,
    ToolFailedObservation,
    ToolRecord,
    TurnDecided,
    complete_exchange,
    minimum_exchange,
)
from weavria.runtime import RunState

from .history_index import MODEL_VISIBLE_RUN_PREDICATE


def create_read_retry_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE read_retry_trials(
        run_id TEXT NOT NULL REFERENCES runs(id), rejection_sequence INTEGER NOT NULL,
        session_id TEXT NOT NULL REFERENCES sessions(id), ordinal INTEGER NOT NULL,
        profile TEXT NOT NULL, scene TEXT NOT NULL, action TEXT NOT NULL, revision INTEGER NOT NULL,
        model_call INTEGER NOT NULL, call_id TEXT, delivery TEXT
        CHECK(delivery IN ('full','partial','reference','failed')),
        PRIMARY KEY(run_id,model_call), UNIQUE(run_id,rejection_sequence),
        FOREIGN KEY(run_id,rejection_sequence) REFERENCES run_events(run_id,sequence))""")
    conn.execute("""CREATE INDEX retry_scene ON read_retry_trials(
        session_id,profile,scene,ordinal DESC,rejection_sequence DESC)
        WHERE delivery IS NOT NULL""")


def _profile_digest(config: str) -> str:
    # 外层 send 传入的剩余秒数每次不同。实际配置和工具契约参与分组, 预算每次重新验证。
    settings = json.loads(config)
    settings.pop("max_request_duration", None)
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def recall_read_retries(
    conn: sqlite3.Connection, run_id: str, scene: str
) -> tuple[ReadRetryOutcome, ...]:
    # 分叉中可见的对话摘要不转移经验授权。配置摘要进入索引, 检索时无需解析历史配置。
    current = conn.execute(
        "SELECT session_id,ordinal,config FROM runs WHERE id=?", (run_id,)
    ).fetchone()
    if current is None:
        return ()
    rows = conn.execute(
        f"""SELECT t.run_id,t.rejection_sequence,t.model_call+2,t.action,t.delivery
        FROM read_retry_trials t JOIN runs r ON r.id=t.run_id
        WHERE t.session_id=? AND t.profile=? AND t.scene=? AND t.ordinal<?
        AND t.delivery IS NOT NULL AND r.status IN ('completed','budget_exhausted','failed')
        AND {MODEL_VISIBLE_RUN_PREDICATE}
        ORDER BY t.ordinal DESC,t.rejection_sequence DESC LIMIT 16""",
        (current[0], _profile_digest(current[2]), scene, current[1]),
    )
    return tuple(
        ReadRetryOutcome(
            run_id=source,
            rejection_sequence=seq,
            model_call=model_call,
            action_sha256=action,
            delivery=delivery,
        )
        for source, seq, model_call, action, delivery in rows
    )


def project_read_retry(
    conn: sqlite3.Connection, event: ControlEvent, state: RunState, session_id: str, ordinal: int
) -> None:
    data = event.data
    model_call = state.model_calls
    if isinstance(data, BatchRejected) and (trial := data.read_prediction) is not None:
        (config,) = conn.execute("SELECT config FROM runs WHERE id=?", (event.run_id,)).fetchone()
        conn.execute(
            """INSERT INTO read_retry_trials(
            run_id,rejection_sequence,session_id,ordinal,profile,scene,action,revision,model_call
            ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                event.run_id,
                event.sequence,
                session_id,
                ordinal,
                _profile_digest(config),
                trial.scene_sha256,
                trial.selected_sha256,
                state.task.revision,
                model_call,
            ),
        )
    elif isinstance(data, BatchAdmitted) and model_call >= 2 and len(data.feedback_envelopes) == 1:
        envelope = data.feedback_envelopes[0]
        # 按归一化动作匹配, 重提时允许模型更换 call_id。隔轮、改变任务或改走其他路均无归因。
        conn.execute(
            """UPDATE read_retry_trials SET call_id=? WHERE run_id=? AND
            model_call=? AND revision=? AND action=?""",
            (
                envelope.call_id,
                event.run_id,
                model_call - 1,
                state.task.revision,
                envelope.action_sha256,
            ),
        )
    elif isinstance(data, TurnDecided) and model_call >= 3:
        revision = state.task.revision
        row = conn.execute(
            """SELECT t.call_id,v.data FROM read_retry_trials t
            JOIN request_views v ON v.run_id=t.run_id AND v.model_call=t.model_call+2
            WHERE t.run_id=? AND t.model_call=? AND t.revision=? AND t.call_id IS NOT NULL""",
            (event.run_id, model_call - 2, revision),
        ).fetchone()
        if row is None:
            return
        request = SessionRequest.model_validate_json(row[1]).current
        feedback = request.feedback
        if (
            request.task.revision != revision
            or feedback is None
            or feedback.task_revision != revision
            or len(feedback.calls) != 1
            or feedback.calls[0].call_id != row[0]
        ):
            return
        delivery = _observed_delivery(conn, event.run_id, row[0], request.history)
        if delivery is not None:
            conn.execute(
                """UPDATE read_retry_trials SET delivery=?
                WHERE run_id=? AND model_call=?""",
                (delivery, event.run_id, model_call - 2),
            )


def _observed_delivery(
    conn: sqlite3.Connection, run_id: str, call_id: str, history: tuple[object, ...]
) -> str | None:
    exchange = next(
        (e for e in history if isinstance(e, ToolExchange) and e.action.call_id == call_id), None
    )
    if exchange is None:
        return None
    durable = conn.execute(
        "SELECT record FROM outcomes WHERE run_id=? AND call_id=?", (run_id, call_id)
    ).fetchone()
    if durable is None:
        return None
    record = ToolRecord.model_validate_json(durable[0])
    if record.exchange.action != exchange.action:
        return None
    if isinstance(exchange.observation, ToolFailedObservation):
        return "failed" if isinstance(record.exchange.observation, ToolFailedObservation) else None
    output, ref = exchange.observation.output, record.artifact
    if not isinstance(output, dict) or ref is None or output.get("artifact_id") != ref.artifact_id:
        return None
    if output.get("sha256") != ref.sha256 or output.get("size_bytes") != ref.size_bytes:
        return None
    if output.get("representation") == "full":
        # 不能凭工具自报成功或视图标签授予正反馈。验证原件摘要, 然后比较完整类型化交换。
        source = conn.execute(
            "SELECT reference,content FROM artifacts WHERE run_id=? AND id=?",
            (run_id, ref.artifact_id),
        ).fetchone()
        if source is None:
            return None
        snapshot = ArtifactSnapshot(*source)
        expected = complete_exchange(record.exchange.action, snapshot.reference, snapshot.value)
        return "full" if expected.model_dump_json() == exchange.model_dump_json() else None
    if output.get("representation") == "excerpts" and output.get("passages"):
        return "partial"
    return "reference" if exchange == minimum_exchange(record) else None
