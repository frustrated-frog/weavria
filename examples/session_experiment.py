"""Frozen persistence/context experiment; literal interpreter, no LLM capability claims."""

import asyncio
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

from persistent_session import ConversationInterpreter

from weavria.context import EvidenceMode
from weavria.core import (
    ModelTurn,
    SessionContextPrepared,
    SessionRequest,
    SessionRunEvent,
    ToolStarted,
)
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolRegistry


class Probe(ConversationInterpreter):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(SessionRequest.model_validate_json(request.model_dump_json()))
        return await super().generate_session(request)


async def experiment(root: Path) -> dict[str, object]:
    seed = root / "seed.db"
    prompts = ['options ["checkpoint", "proof", "trace"]'] + [
        f'options ["unrelated-{i}", "other", "last"]' for i in range(10)
    ]
    async with SessionEngine(
        seed,
        model=ConversationInterpreter(),
        tools=ToolRegistry(()),
        limits=RunLimits(max_request_bytes=24000),
        history_budget_bytes=14000,
    ) as engine:
        await engine.create("s")
        for i, prompt in enumerate(prompts):
            await engine.send("s", prompt, input_id=f"seed-{i}")
    results: list[dict[str, object]] = []
    for mode in EvidenceMode:
        path = root / (mode.value + ".db")
        with sqlite3.connect(seed) as src, sqlite3.connect(path) as dst:
            src.backup(dst)
        probe = Probe()
        async with SessionEngine(
            path,
            model=probe,
            tools=ToolRegistry(()),
            limits=RunLimits(max_request_bytes=16000),
            history_budget_bytes=3500,
            history_mode=mode,
            mode=EvidenceMode.FULL,
        ) as engine:
            answer = await engine.send("s", "recall checkpoint", input_id="question")
            facts: list[SessionContextPrepared | SessionRunEvent] = []
            cursor = 0
            while page := await engine.store.events("s", after=cursor):
                facts.extend(
                    e.data
                    for e in page
                    if (
                        isinstance(e.data, SessionContextPrepared)
                        and e.data.run_id == answer.run_id
                    )
                    or (
                        isinstance(e.data, SessionRunEvent) and e.data.event.run_id == answer.run_id
                    )
                )
                cursor = page[-1].sequence
            prepared = [f for f in facts if isinstance(f, SessionContextPrepared)]
            results.append(
                {
                    "policy": mode.value,
                    "status": answer.report.status.value,
                    "correct": json.loads(answer.report.output or "null")
                    == {"choices": ["checkpoint", "proof", "trace"]},
                    "model_calls": len(probe.requests),
                    "tool_calls": sum(
                        isinstance(f, SessionRunEvent) and isinstance(f.event.data, ToolStarted)
                        for f in facts
                    ),
                    "input_bytes": sum(p.request_bytes for p in prepared),
                    "peak_request_bytes": max((p.request_bytes for p in prepared), default=0),
                    "all_snapshots_match": all(
                        hashlib.sha256(r.model_dump_json().encode()).hexdigest() == p.request_sha256
                        for r, p in zip(probe.requests, prepared, strict=True)
                    ),
                }
            )
    return {
        "schema_version": 1,
        "dataset_version": "conversation/1",
        "solver_version": "literal/1",
        "evaluator_version": "recall/1",
        "scope": "deterministic context and persistence mechanism, no real LLM",
        "configuration": {
            "max_request_bytes": 16000,
            "history_budget_bytes": 3500,
            "recent_pins": 2,
        },
        "dataset_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "implementation_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("persistent_session.py"))
        },
        "results": results,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="weavria-session-") as directory:
        print(json.dumps(asyncio.run(experiment(Path(directory))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
