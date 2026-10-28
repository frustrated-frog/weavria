"""Opt-in HTTP regressions over frozen private history and fresh FinScope sessions."""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import httpx

from apps.finscope.config import FinScopeSettings
from apps.finscope.server import MessageOutput, PageContext, create_app


def freeze_database(source: Path, destination: Path) -> None:
    """Include committed WAL contents without writing to the source database."""
    reader = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    writer = sqlite3.connect(destination)
    try:
        reader.backup(writer)
    finally:
        writer.close()
        reader.close()
    destination.chmod(0o600)


async def run_trial(
    settings: FinScopeSettings,
    *,
    seed: Path | None,
    seed_sha256: str | None,
    session_id: str | None,
    question: str,
    context: PageContext,
    repeat: int,
) -> dict[str, object]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="weavria-regression-") as directory:
        database = Path(directory) / "sessions.sqlite3"
        if seed is not None:
            await asyncio.to_thread(shutil.copy2, seed, database)
        app = create_app(replace(settings, state_path=database))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://regression.test"
            ) as client:
                if session_id is None:
                    session_id = (await client.post("/api/sessions")).json()["session_id"]
                response = await client.post(
                    f"/api/sessions/{session_id}/messages",
                    json={
                        "input_id": f"frozen-regression-{repeat}",
                        "text": question,
                        "context": context.model_dump(mode="json"),
                    },
                )
        result: dict[str, object] = {
            "schema_version": 1,
            "case": "historical" if seed is not None else "fresh",
            "repeat": repeat,
            "http_status": response.status_code,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "context_sha256": hashlib.sha256(context.model_dump_json().encode()).hexdigest(),
            "seed_sha256": seed_sha256,
            "model": settings.model_name,
            "execution_reasoning_effort": settings.model_reasoning_effort,
            "understanding_reasoning_effort": settings.understanding_reasoning_effort,
        }
        if response.status_code == 200:
            output = MessageOutput.model_validate_json(response.content)
            result.update(
                status=output.status,
                task_outcome=output.task_outcome,
                answer_chars=len(output.answer or ""),
                completion_reason=output.completion_reason,
                tool_calls=len(output.tools),
                failed_tools=sum(tool.status == "failed" for tool in output.tools),
            )
        # No answers, tool contents, response bodies, model reasoning or secrets are exported.
        return result


async def main() -> None:
    if os.environ.get("WEAVRIA_RUN_LIVE_MODEL_TESTS") != "1":
        raise SystemExit("Set WEAVRIA_RUN_LIVE_MODEL_TESTS=1 for paid model and real backend calls")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--seed-database", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or bool(args.seed_database) != bool(args.session_id):
        parser.error("repeats must be in [1,20]; seed-database and session-id must be paired")
    settings = FinScopeSettings.from_env()
    question = args.question_file.read_text().strip()
    context = PageContext.model_validate_json(args.context_file.read_bytes())
    with tempfile.TemporaryDirectory(prefix="weavria-frozen-") as directory:
        seed = Path(directory) / "seed.sqlite3" if args.seed_database else None
        if seed is not None:
            await asyncio.to_thread(freeze_database, args.seed_database, seed)
        seed_sha256 = (
            hashlib.sha256(await asyncio.to_thread(seed.read_bytes)).hexdigest() if seed else None
        )
        # Refuse to replace an earlier evaluation and keep exported metrics private by default.
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as output:
            cases = ((seed, args.session_id), (None, None)) if seed else ((None, None),)
            for trial_seed, session in cases:
                for repeat in range(1, args.repeats + 1):
                    result = await run_trial(
                        settings,
                        seed=trial_seed,
                        seed_sha256=seed_sha256 if trial_seed else None,
                        session_id=session,
                        question=question,
                        context=context,
                        repeat=repeat,
                    )
                    encoded = json.dumps(result, ensure_ascii=False)
                    output.write(encoded + "\n")
                    output.flush()
                    print(encoded, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
