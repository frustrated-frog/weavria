"""Terminate a real writer process around the transaction boundary, then reopen."""

import asyncio
import sys
from pathlib import Path

import pytest

from weavria.memory import (
    MemoryAccess,
    MemoryAssertion,
    MemoryBatch,
    MemoryMutation,
    MemorySource,
    SQLiteMemoryStore,
)

CHILD = """
import asyncio, json, sys, time
from datetime import UTC, datetime
from pathlib import Path
from weavria.memory import MemoryAccess, MemoryBatch, SQLiteMemoryStore
payload = json.loads(sys.stdin.read())
access = MemoryAccess.model_validate_json(json.dumps(payload["access"]))
batch = MemoryBatch.model_validate_json(json.dumps(payload["batch"]))
def clock():
    if sys.argv[2] == "before":
        print("boundary", flush=True)
        time.sleep(30)
    return datetime(2026, 1, 15, tzinfo=UTC)
async def run():
    async with SQLiteMemoryStore(Path(sys.argv[1]), now=clock) as store:
        await store.commit(access, batch)
        print("boundary", flush=True)
        await asyncio.sleep(30)
asyncio.run(run())
"""


@pytest.mark.parametrize("boundary", ["before", "after"])
async def test_process_death_keeps_atomic_memory(
    tmp_path: Path,
    access: MemoryAccess,
    assertion: MemoryAssertion,
    source: MemorySource,
    boundary: str,
) -> None:
    import json

    path = tmp_path / "memory.db"
    batch = MemoryBatch(
        batch_id="one",
        scope=assertion.scope,
        policy_version="host/1",
        sources=(source,),
        mutations=(MemoryMutation(operation="add", assertion=assertion, reason="explicit"),),
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        CHILD,
        str(path),
        boundary,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(
            json.dumps(
                {"access": access.model_dump(mode="json"), "batch": batch.model_dump(mode="json")}
            ).encode()
        )
        await process.stdin.drain()
        process.stdin.close()
        async with asyncio.timeout(5):
            assert await process.stdout.readline() == b"boundary\n"
        process.kill()
        await process.wait()
        async with SQLiteMemoryStore(path) as store:
            events = await store.events(access, assertion.scope)
            assert len(events) == (1 if boundary == "after" else 0)
            receipt = await store.commit(access, batch)
            assert receipt.duplicate == (boundary == "after")
            assert len(await store.history(access, assertion.scope, assertion.memory_id)) == 1
    finally:
        if process.returncode is None:
            process.kill()
        await process.communicate()
