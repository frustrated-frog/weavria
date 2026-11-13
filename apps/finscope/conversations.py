"""FinScope 本机单用户对话投影与执行所有权.

独立 UI 索引只保存宿主请求和展示结果; 框架事件及 Artifact 仍是执行事实来源.
一个请求只对应一个受跟踪的 Task. 取消与关机必须等待 Harness 完成清理.
"""

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import JsonValue

from weavria.core import ErrorCategory
from weavria.sessions import SessionConflictError, SessionEngine, SessionTaskPreparationError

Payload = dict[str, JsonValue]
Execute = Callable[[str, Payload], Awaitable[Payload]]


class ConversationHost:
    def __init__(self, path: Path, engine: SessionEngine, execute: Execute) -> None:
        self.path = path
        self.engine = engine
        self.execute = execute
        self.tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self.lock = asyncio.Lock()
        self.closing = False

    def _db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    async def open(self) -> None:
        def initialize() -> None:
            with self._db() as db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS ui_sessions(
                      id TEXT PRIMARY KEY, title TEXT NOT NULL, updated_at TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS ui_requests(
                      session_id TEXT NOT NULL, input_id TEXT NOT NULL, payload TEXT NOT NULL,
                      status TEXT NOT NULL, result TEXT, cursor INTEGER NOT NULL,
                      created_at TEXT NOT NULL DEFAULT(strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                      PRIMARY KEY(session_id,input_id));
                    UPDATE ui_requests SET status='interrupted'
                      WHERE status IN ('running','cancelling');
                """)

        await asyncio.to_thread(initialize)

    async def register(self, session_id: str, title: str = "新对话") -> None:
        await self.engine.store.head(session_id)

        def write() -> None:
            with self._db() as db:
                db.execute(
                    "INSERT OR IGNORE INTO ui_sessions "
                    "VALUES(?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                    (session_id, title),
                )

        await asyncio.to_thread(write)

    async def sessions(self) -> list[Payload]:
        def read() -> list[Payload]:
            with self._db() as db:
                return [
                    dict(r)
                    for r in db.execute(
                        "SELECT id AS session_id,title,updated_at FROM ui_sessions ORDER BY "
                        "updated_at DESC,id LIMIT 100"
                    )
                ]

        return await asyncio.to_thread(read)

    async def start(self, session_id: str, payload: Payload) -> Payload:
        async with self.lock:
            if self.closing:
                raise RuntimeError("sidecar is closing")
            await self.register(session_id)
            input_id = str(payload["input_id"])
            cursor, _ = await self.engine.store.head(session_id)
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)

            def claim() -> bool:
                with self._db() as db:
                    prior = db.execute(
                        "SELECT payload FROM ui_requests WHERE session_id=? AND input_id=?",
                        (session_id, input_id),
                    ).fetchone()
                    if prior:
                        if prior[0] != encoded:
                            raise SessionConflictError("input_id payload differs")
                        return False
                    active = db.execute(
                        "SELECT 1 FROM ui_requests WHERE session_id=? AND status IN "
                        "('running','cancelling')",
                        (session_id,),
                    ).fetchone()
                    if active:
                        raise SessionConflictError("session is busy")
                    db.execute(
                        "INSERT INTO ui_requests(session_id,input_id,payload,status,cursor) "
                        "VALUES(?,?,?,'running',?)",
                        (session_id, input_id, encoded, cursor),
                    )
                    db.execute(
                        "UPDATE ui_sessions SET "
                        "title=?,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
                        (str(payload["text"])[:60], session_id),
                    )
                    return True

            # claim 完成后才创建 Task; 取消请求不会留下无人负责的后台模型调用.
            mutation = asyncio.create_task(asyncio.to_thread(claim))
            cancelled = False
            try:
                claimed = await asyncio.shield(mutation)
            except asyncio.CancelledError:
                claimed = await mutation
                cancelled = True
            if claimed:
                self.tasks[(session_id, input_id)] = asyncio.create_task(
                    self._run(session_id, input_id, payload)
                )
            if cancelled:
                raise asyncio.CancelledError
            return await self.get(session_id, input_id)

    async def _finish(
        self, session_id: str, input_id: str, status: str, result: Payload | None
    ) -> None:
        def write() -> None:
            with self._db() as db:
                db.execute(
                    "UPDATE ui_requests SET status=?,result=? WHERE session_id=? AND input_id=? "
                    "AND status IN ('running','cancelling')",
                    (
                        status,
                        json.dumps(result, ensure_ascii=False) if result is not None else None,
                        session_id,
                        input_id,
                    ),
                )

        mutation = asyncio.create_task(asyncio.to_thread(write))
        try:
            await asyncio.shield(mutation)
        except asyncio.CancelledError:
            # 已开始的 SQLite 写入先结算, 再传播取消; 已提交的完成结果不能被停止覆盖.
            await mutation
            raise

    async def _run(self, session_id: str, input_id: str, payload: Payload) -> None:
        try:
            result = await self.execute(session_id, payload)
            await self._finish(session_id, input_id, "finished", result)
        except asyncio.CancelledError:
            await self._finish(
                session_id, input_id, "interrupted" if self.closing else "cancelled", None
            )
            raise
        except SessionTaskPreparationError as exc:
            # 稳定错误类别可形成用户提示, 原始模型输出和异常堆栈仍留在服务器。
            item = await self.get(session_id, input_id)
            run_id = item.get("run_id")
            messages = {
                ErrorCategory.INVALID_MODEL_RESPONSE: "问题理解或任务要求校验未通过",
                ErrorCategory.MODEL_FAILURE: "任务准备时模型服务调用失败",
                ErrorCategory.TIMEOUT: "任务准备超时",
                ErrorCategory.CONTEXT_BUDGET_EXCEEDED: "任务准备所需内容超出了上下文预算",
            }
            message = messages.get(exc.error.category, "任务准备未完成")
            if exc.error.cause_type in {
                "CoverageValidation:source_quote_not_found",
                "CoverageValidation:source_quote_ambiguous",
                "CoverageValidation:source_quote_occurrence_out_of_range",
                "CoverageValidation:invalid_obligation_source",
            }:
                message = "任务要求的原文引用未通过校验"
            await self._finish(
                session_id,
                input_id,
                "failed",
                {
                    "session_id": session_id,
                    "run_id": run_id,
                    "status": "failed",
                    "task_outcome": "unverifiable",
                    "answer": None,
                    "error": f"{message}。请重试。诊断编号 {input_id}",
                    "completion_reason": None,
                    "tools": [],
                    "sources": [],
                    "clarification_id": None,
                }
                if isinstance(run_id, str)
                else None,
            )
        except Exception:
            # 诊断留在服务器日志, 不把供应商响应或本地路径发送给浏览器.
            import logging

            logging.getLogger(__name__).exception(
                "Agent request failed session=%s input=%s", session_id, input_id
            )
            await self._finish(session_id, input_id, "failed", None)
        finally:
            self.tasks.pop((session_id, input_id), None)

    async def get(self, session_id: str, input_id: str) -> Payload:
        def read() -> Payload:
            with self._db() as db:
                row = db.execute(
                    "SELECT input_id,status,result,cursor FROM ui_requests WHERE "
                    "session_id=? AND input_id=?",
                    (session_id, input_id),
                ).fetchone()
                if row is None:
                    raise KeyError("request not found")
                return dict(row)

        item = await asyncio.to_thread(read)
        cursor = int(str(item.pop("cursor")))
        result = item.get("result")
        item["result"] = json.loads(result) if isinstance(result, str) else None
        phase = "正在理解问题"
        tools: dict[str, Payload] = {}
        # 公共事件分页读取, 只投影阶段和工具名称; 不暴露模型决策正文/隐藏推理.
        run_id: str | None = None
        names: dict[str, str] = {}
        while True:
            events = await self.engine.store.events(session_id, after=cursor, limit=256)
            for event in events:
                cursor = event.sequence
                data = event.data
                if data.kind == "input_accepted" and data.input_id == input_id:
                    run_id = data.run_id
                if data.kind != "run_event" or data.event.run_id != run_id:
                    continue
                control = data.event.data
                if control.kind == "model_started":
                    phase = "正在分析证据"
                elif control.kind in ("finalization_started", "verification_started"):
                    phase = "正在整理回答"
                elif control.kind == "tool_recorded":
                    exchange = control.record.exchange
                    tools[exchange.action.call_id] = {
                        "call_id": exchange.action.call_id,
                        "tool_name": exchange.action.tool_name,
                        "status": "succeeded" if control.record.artifact else "failed",
                    }
                    phase = "已读取证据, 正在继续分析"
                elif control.kind == "turn_decided" and control.turn.decision.kind == "tool_batch":
                    names.update(
                        {call.call_id: call.tool_name for call in control.turn.decision.calls}
                    )
                elif control.kind == "tool_started":
                    phase = "正在读取证据"
                    tools[control.call_id] = {
                        "call_id": control.call_id,
                        "tool_name": names.get(control.call_id, "读取证据"),
                        "status": "running",
                    }
            if len(events) < 256:
                break
        item["phase"] = "正在停止并保存执行状态" if item["status"] == "cancelling" else phase
        item["tools"] = list(tools.values())
        item["run_id"] = run_id
        return item

    async def cancel(self, session_id: str, input_id: str) -> Payload:
        async with self.lock:
            task = self.tasks.get((session_id, input_id))
            if task is not None and not task.done():
                await self._finish(session_id, input_id, "cancelling", None)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                item = await self.get(session_id, input_id)
                if item["status"] == "cancelling":
                    await self._finish(session_id, input_id, "cancelled", None)
            return await self.get(session_id, input_id)

    async def history(self, session_id: str, before: int | None = None) -> Payload:
        await self.register(session_id)
        turns = await self.engine.store.history(session_id, before_ordinal=before, limit=30)

        def read() -> list[Payload]:
            with self._db() as db:
                rows = db.execute(
                    "SELECT input_id,payload,status,result FROM ui_requests WHERE "
                    "session_id=? ORDER BY created_at DESC LIMIT 60",
                    (session_id,),
                ).fetchall()
                return [dict(r) for r in reversed(rows)]

        jobs = await asyncio.to_thread(read)
        requests: list[JsonValue] = []
        for job in jobs:
            payload = json.loads(str(job.pop("payload")))
            job["text"] = payload["text"]
            raw = job.get("result")
            job["result"] = json.loads(raw) if isinstance(raw, str) else None
            requests.append(job)
        return {
            "turns": [t.model_dump(mode="json") for t in turns],
            "requests": requests,
            "before": turns[0].ordinal if len(turns) == 30 else None,
        }

    async def close(self) -> None:
        self.closing = True
        owned = tuple(self.tasks.items())
        for _, task in owned:
            task.cancel()
        await asyncio.gather(*(task for _, task in owned), return_exceptions=True)
        for (session_id, input_id), _ in owned:
            item = await self.get(session_id, input_id)
            if item["status"] in ("running", "cancelling"):
                await self._finish(session_id, input_id, "interrupted", None)
