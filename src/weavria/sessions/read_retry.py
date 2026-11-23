"""同会话单读经验索引。在事实提交事务中归因, 从持久请求与工具原件核对交付。"""

import hashlib
import json
import sqlite3

from weavria.artifacts import ArtifactSnapshot, resolve_pointer
from weavria.core import (
    BatchAdmitted,
    BatchRejected,
    ControlEvent,
    ReadAssessment,
    ReadRetryOutcome,
    ReadRetryScope,
    SessionRequest,
    TaskInput,
    ToolExchange,
    ToolFailedObservation,
    ToolRecord,
    TurnDecided,
    complete_exchange,
    minimum_exchange,
)
from weavria.runtime import ReadRetryHistory, RunState

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


def create_read_retry_scope_index(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE read_retry_trials ADD COLUMN scope_sha256 TEXT")
    conn.execute("ALTER TABLE read_retry_trials ADD COLUMN scope TEXT")
    conn.execute(
        "ALTER TABLE read_retry_trials ADD COLUMN field_support INTEGER NOT NULL DEFAULT 0"
    )
    conn.execute("""CREATE INDEX retry_scope ON read_retry_trials(
        session_id,profile,scope_sha256,ordinal DESC,rejection_sequence DESC)
        WHERE field_support=1""")


def create_read_retry_review_index(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE INDEX retry_full_history ON read_retry_trials(
        session_id,profile,ordinal DESC,rejection_sequence DESC) WHERE delivery='full'""")


def recall_read_retry_history(
    conn: sqlite3.Connection, run_id: str, tools: tuple[str, ...]
) -> tuple[ReadRetryHistory, ...]:
    """沿已核验回执找原件与当时模型看到的任务, 不从会话摘要恢复正文。"""
    if not 1 <= len(tools) <= 32:
        return ()
    current = conn.execute(
        "SELECT session_id,ordinal,config FROM runs WHERE id=?", (run_id,)
    ).fetchone()
    if current is None:
        return ()
    placeholders = ",".join("?" for _ in tools)
    rows = conn.execute(
        f"""SELECT t.run_id,t.rejection_sequence,t.model_call+2,t.action,
        json_extract(v.data,'$.current.task'),o.record,a.reference,a.content
        FROM read_retry_trials t JOIN runs r ON r.id=t.run_id
        JOIN request_views v ON v.run_id=t.run_id AND v.model_call=t.model_call
        JOIN outcomes o ON o.run_id=t.run_id AND o.call_id=t.call_id
        JOIN artifacts a ON a.run_id=o.run_id AND a.id=o.artifact_id
        WHERE t.session_id=? AND t.profile=? AND t.ordinal<? AND t.delivery='full'
        AND r.status IN ('completed','budget_exhausted','failed')
        AND {MODEL_VISIBLE_RUN_PREDICATE}
        AND json_extract(o.record,'$.exchange.action.tool_name') IN ({placeholders})
        AND length(CAST(a.content AS BLOB))<=2200
        AND length(CAST(json_extract(v.data,'$.current.task') AS BLOB))<=1800
        AND length(CAST(json_extract(o.record,'$.authorization.normalized_arguments') AS BLOB))
            <=1000
        ORDER BY t.ordinal DESC,t.rejection_sequence DESC LIMIT 4""",
        (current[0], _profile_digest(current[2]), current[1], *tools),
    )
    return tuple(
        ReadRetryHistory(
            ReadRetryOutcome(
                run_id=source,
                rejection_sequence=seq,
                model_call=model_call,
                action_sha256=action,
                delivery="full",
            ),
            TaskInput.model_validate_json(task),
            ToolRecord.model_validate_json(record),
            ArtifactSnapshot(reference, content),
        )
        for source, seq, model_call, action, task, record, reference, content in rows
    )


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
        f"""SELECT t.run_id,t.rejection_sequence,t.model_call+2,t.action,t.delivery,t.assessment
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
            schema_version=2 if assessed is not None else 1,
            assessment=ReadAssessment.model_validate_json(assessed)
            if assessed is not None
            else None,
        )
        for source, seq, model_call, action, delivery, assessed in rows
    )


def recall_scoped_read_retries(
    conn: sqlite3.Connection,
    run_id: str,
    scopes: tuple[str, ...],
) -> dict[str, ReadRetryOutcome]:
    """每个当前作用域最多回取一条正向证据; 失败经验不会跨实体扣分。"""
    if len(scopes) > 32:
        raise ValueError("scoped read lookup exceeds the batch limit")
    current = conn.execute(
        "SELECT session_id,ordinal,config FROM runs WHERE id=?", (run_id,)
    ).fetchone()
    if current is None:
        return {}
    session, ordinal, config = current
    profile = _profile_digest(config)
    found: dict[str, ReadRetryOutcome] = {}
    for scope in scopes:
        row = conn.execute(
            f"""SELECT t.run_id,t.rejection_sequence,t.model_call+2,t.action
            FROM read_retry_trials t JOIN runs r ON r.id=t.run_id
            WHERE t.session_id=? AND t.profile=? AND t.scope_sha256=? AND t.ordinal<?
            AND t.field_support=1 AND r.status IN ('completed','budget_exhausted','failed')
            AND {MODEL_VISIBLE_RUN_PREDICATE}
            ORDER BY t.ordinal DESC,t.rejection_sequence DESC LIMIT 1""",
            (session, profile, scope, ordinal),
        ).fetchone()
        if row is not None:
            source, sequence, model_call, action = row
            found[scope] = ReadRetryOutcome(
                run_id=source,
                rejection_sequence=sequence,
                model_call=model_call,
                action_sha256=action,
                delivery="full",
            )
    return found


def project_read_retry(
    conn: sqlite3.Connection, event: ControlEvent, state: RunState, session_id: str, ordinal: int
) -> None:
    data = event.data
    model_call = state.model_calls
    if isinstance(data, BatchRejected) and (trial := data.read_prediction) is not None:
        (config,) = conn.execute("SELECT config FROM runs WHERE id=?", (event.run_id,)).fetchone()
        conn.execute(
            """INSERT INTO read_retry_trials(
            run_id,rejection_sequence,session_id,ordinal,profile,scene,action,revision,model_call,
            scope_sha256,scope
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
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
                trial.scope_sha256,
                trial.scope.model_dump_json() if trial.scope is not None else None,
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
            """SELECT t.call_id,v.data,t.scope FROM read_retry_trials t
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
        scope = ReadRetryScope.model_validate_json(row[2]) if row[2] is not None else None
        observation = _observed_delivery(
            conn, event.run_id, row[0], request.history, scope, data.turn.read_assessment
        )
        if observation is not None:
            delivery, field_support, assessment = observation
            conn.execute(
                """UPDATE read_retry_trials SET delivery=?,field_support=?,assessment=?
                WHERE run_id=? AND model_call=?""",
                (
                    delivery,
                    int(field_support),
                    assessment.model_dump_json() if assessment else None,
                    event.run_id,
                    model_call - 2,
                ),
            )


def _observed_delivery(
    conn: sqlite3.Connection,
    run_id: str,
    call_id: str,
    history: tuple[object, ...],
    scope: ReadRetryScope | None,
    assessment: ReadAssessment | None,
) -> tuple[str, bool, ReadAssessment | None] | None:
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
        return (
            ("failed", False, None)
            if isinstance(record.exchange.observation, ToolFailedObservation)
            else None
        )
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
        original = snapshot.value
        expected = complete_exchange(record.exchange.action, snapshot.reference, original)
        if expected.model_dump_json() != exchange.model_dump_json():
            return None
        supported = False
        if scope is not None:
            try:
                subject = resolve_pointer(original, scope.subject_pointer)
                values = [resolve_pointer(original, field) for field in scope.required_fields]
            except (KeyError, IndexError, ValueError):
                pass
            else:
                supported = (
                    isinstance(subject, str)
                    and scope.subject_digest(subject) == scope.subject_sha256
                    and all(value not in (None, "", [], {}) for value in values)
                )
        # 独立核验原件, 不能信任供应商适配器或自定义 ModelProvider 给出的坐标。
        if assessment is not None and assessment.source_sha256 != ref.sha256:
            assessment = None
        if assessment is not None and (q := assessment.quote) is not None:
            try:
                text = resolve_pointer(original, q.pointer)
            except (ValueError, KeyError, IndexError):
                text = None
            excerpt = text[q.start : q.end] if isinstance(text, str) else ""
            if (
                not isinstance(text, str)
                or q.end > len(text)
                or not any(c.isalnum() for c in excerpt)
                or text.find(excerpt) != q.start
                or text.find(excerpt, q.start + 1) >= 0
                or hashlib.sha256(excerpt.encode()).hexdigest() != q.quote_sha256
            ):
                assessment = None
        return "full", supported, assessment
    if output.get("representation") == "excerpts" and output.get("passages"):
        return "partial", False, None
    return ("reference", False, None) if exchange == minimum_exchange(record) else None
