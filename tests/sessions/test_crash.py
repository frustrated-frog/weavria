"""Real process death at durable boundaries; no simulated catch/finally recovery."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from weavria.core import FinishAction, ModelTurn, SessionRequest
from weavria.models import ModelCapabilities
from weavria.sessions import SessionEngine, SessionInterruptedError
from weavria.tools import ToolRegistry

CHILD = r"""
import asyncio, os, sys
from pathlib import Path
from pydantic import BaseModel
from weavria.core import ModelTurn, FinishAction, ToolBatch, ToolCallAction, RunEnded
from weavria.models import ModelCapabilities
from weavria.sessions import SessionEngine, SQLiteSessionStore
from weavria.tools import ToolRegistry, TypedTool
point=sys.argv[2]
class In(BaseModel):
    value: str
class Out(BaseModel):
    value: str
async def produce(args, context):
    if point == 'tool': os._exit(73)
    return Out(value=args.value)
class CrashStore(SQLiteSessionStore):
    async def begin(self,*args,**kwargs):
        claim=await super().begin(*args,**kwargs)
        if point=='input': os._exit(73)
        return claim
    async def put(self,*args,**kwargs):
        snapshot=await super().put(*args,**kwargs)
        if point=='artifact': os._exit(73)
        return snapshot
    async def append(self,event):
        await super().append(event)
        if point=='finish' and isinstance(event.data,RunEnded): os._exit(73)
class Provider:
    capabilities=ModelCapabilities(True,True)
    async def generate_session(self,request):
        if point=='model': os._exit(73)
        if request.current.history:
            if point=='accepted': os._exit(73)
            return ModelTurn(decision=FinishAction(output='actual answer'))
        action=ToolCallAction(call_id='call',tool_name='produce',
                             arguments={'value':'actual source'})
        return ModelTurn(decision=ToolBatch(calls=(action,)))
async def main():
    tool=TypedTool(name='produce',version='1',description='produce evidence',
                   input_type=In,output_type=Out,execute=produce)
    engine=SessionEngine(Path(sys.argv[1]),model=Provider(),
                         tools=ToolRegistry((tool,)))
    engine.store=CrashStore(Path(sys.argv[1]))
    async with engine:
        await engine.create('s')
        await engine.send('s','original request',input_id='original')
asyncio.run(main())
"""


class Continue:
    capabilities = ModelCapabilities(True, True)

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        assert request.conversation.turns
        return ModelTurn(decision=FinishAction(output="continued with preserved history"))


@pytest.mark.parametrize("point", ["input", "model", "tool", "artifact", "accepted", "finish"])
async def test_reopen_preserves_exact_acceptance_boundary_after_process_death(
    tmp_path: Path, point: str
) -> None:
    path = tmp_path / "session.db"
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", CHILD, str(path), point],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 73, result.stderr
    async with SessionEngine(path, model=Continue(), tools=ToolRegistry(())) as engine:
        history = await engine.store.history("s")
        assert len(history) == 1 and history[0].user == "original request"
        old = history[0]
        if point == "finish":
            assert old.status == "completed" and old.assistant == "actual answer"
            duplicate = await engine.send("s", "original request", input_id="original")
            assert duplicate.run_id == old.run_id and duplicate.report.output == "actual answer"
        else:
            assert old.status == "interrupted" and old.assistant is None
            with pytest.raises(SessionInterruptedError):
                await engine.send("s", "original request", input_id="original")
        if point in ("tool", "artifact"):
            assert old.unfinished_call_ids == ("call",) and old.calls == ()
        if point in ("accepted", "finish"):
            assert len(old.calls) == 1 and old.calls[0].artifact is not None
            reference = old.calls[0].artifact
            source = await engine.store.get(reference.run_id, reference.artifact_id)
            assert source.value == {"value": "actual source"}
        elif point != "finish":
            assert old.calls == ()
        continued = await engine.send("s", "continue", input_id="next")
        assert continued.report.output == "continued with preserved history"
        assert len(await engine.store.history("s")) == 2
