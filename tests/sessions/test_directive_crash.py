import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from weavria.core import FinishAction, ModelTurn, SessionRunEvent, TaskRevised
from weavria.models import FakeSessionProvider
from weavria.sessions import SessionBusyError, SessionEngine
from weavria.tools import ToolRegistry

CHILD = r"""
import asyncio, os, sys
from pathlib import Path
from weavria.core import ModelTurn, FinishAction
from weavria.models import ModelCapabilities
from weavria.sessions import SessionEngine, SQLiteSessionStore
from weavria.tools import ToolRegistry
entered=asyncio.Event()
class Store(SQLiteSessionStore):
    async def append_revision(self,event,update):
        if sys.argv[2]=='before': os._exit(73)
        await super().append_revision(event,update)
        os._exit(73)
class Model:
    capabilities=ModelCapabilities(True,True)
    async def generate_session(self,request):
        entered.set()
        await asyncio.Event().wait()
        return ModelTurn(decision=FinishAction(output='unreachable'))
async def main():
    engine=SessionEngine(Path(sys.argv[1]),model=Model(),tools=ToolRegistry(()))
    engine.store=Store(Path(sys.argv[1]))
    async with engine:
        await engine.create('s')
        run=asyncio.create_task(engine.send('s','work',input_id='original'))
        await entered.wait()
        await engine.steer('s','new direction',input_id='change')
        await run
asyncio.run(main())
"""


@pytest.mark.parametrize("point", ["before", "after"])
async def test_process_death_preserves_both_input_and_revision_or_neither(
    tmp_path: Path,
    point: str,
) -> None:
    path = tmp_path / "db"
    process = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", CHILD, str(path), point],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 73, process.stderr
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="continued")),))
    async with SessionEngine(path, model=model, tools=ToolRegistry(())) as engine:
        turn = (await engine.store.history("s"))[0]
        assert turn.status == "interrupted" and turn.assistant is None
        changes = [
            e.data.event.data
            for e in await engine.store.events("s")
            if isinstance(e.data, SessionRunEvent) and isinstance(e.data.event.data, TaskRevised)
        ]
        assert len(turn.updates) == len(changes) == int(point == "after")
        if point == "after":
            update = await engine.steer("s", "new direction", input_id="change")
            assert update == turn.updates[0] and update.task.revision == 2
        else:
            with pytest.raises(SessionBusyError):
                await engine.steer("s", "new direction", input_id="change")
        await engine.send("s", "continue", input_id="next")
        assert len(model.requests[0].conversation.turns[0].updates) == int(point == "after")
