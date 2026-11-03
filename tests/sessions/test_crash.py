"""Real process death at durable boundaries; no simulated catch/finally recovery."""

import asyncio
import sqlite3
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
from datetime import timedelta
from pathlib import Path
from pydantic import BaseModel
from weavria.core import ModelTurn, FinishAction, ToolBatch, ToolCallAction, RunEnded
from weavria.models import ModelCapabilities
from weavria.sessions import SessionEngine, SQLiteSessionStore
from weavria.tools import PolicyDecision, ToolRegistry, TypedTool
from weavria.runtime import SystemClock
point=sys.argv[2]
class In(BaseModel):
    value: str
class Out(BaseModel):
    value: str
async def produce(args, context):
    if point == 'tool': os._exit(73)
    if point.startswith('resume'): os._exit(74)
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
    async def record_resolution(self,resolution,**kwargs):
        await super().record_resolution(resolution,**kwargs)
        if point=='boundary' and resolution.stage=='execution': os._exit(73)
    async def append(self,event):
        await super().append(event)
        if point=='finish' and isinstance(event.data,RunEnded): os._exit(73)
class Provider:
    capabilities=ModelCapabilities(True,True)
    async def generate_session(self,request):
        if point=='model': os._exit(73)
        if request.current.history:
            if point=='accepted': os._exit(73)
            assert 'actual source' in request.current.model_dump_json()
            return ModelTurn(decision=FinishAction(output='actual answer'))
        action=ToolCallAction(call_id='call',tool_name='produce',
                             arguments={'value':'actual source'})
        return ModelTurn(decision=ToolBatch(calls=(action,)))
class Revoked:
    async def authorize(self,*args):
        return PolicyDecision(False, 'revoked/1', 'revoked')
class ExpiredClock(SystemClock):
    def now(self):
        return super().now()+timedelta(days=1)
async def main():
    tool=TypedTool(name='produce',version='1',description='produce evidence',
                   input_type=In,output_type=Out,execute=produce)
    engine=SessionEngine(Path(sys.argv[1]),model=Provider(),
                         tools=ToolRegistry((tool,)),
                         clock=ExpiredClock() if point=='resume-expired' else None,
                         policy=Revoked() if point=='resume-revoked' else None)
    engine.store=CrashStore(Path(sys.argv[1]), clock=engine._clock)
    async with engine:
        if point.startswith('resume'):
            old, = await engine.store.history('s')
            if point=='resume':
                a,b=await asyncio.gather(engine.resume('s',old.run_id),
                                         engine.resume('s',old.run_id), return_exceptions=True)
                from weavria.sessions import SessionBusyError
                assert isinstance(b, SessionBusyError), b
                if isinstance(a, BaseException): raise a
                assert a.run_id==old.run_id and a.report.output=='actual answer', a
                events=await engine.store.events('s')
                assert sum(e.data.kind=='resume_claimed' for e in events)==1
                facts=[e.data.event.data.kind for e in events if e.data.kind=='run_event']
                assert facts.count('model_started')==2 and facts.count('tool_started')==1
                assert facts.count('run_started')==1
                assert (await engine.store.history('s'))[0].status=='completed'
            else:
                from weavria.sessions import SessionConflictError
                try:
                    await engine.resume('s',old.run_id)
                except SessionConflictError as error:
                    print(error)
                else:
                    raise AssertionError('unsafe resume succeeded')
        else:
            await engine.create('s')
            result=await engine.send('s','original request',input_id='original')
        if point=='baseline':
            events=await engine.store.events('s')
            facts=[e.data.event.data.kind for e in events if e.data.kind=='run_event']
            assert result.report.output=='actual answer'
            assert facts.count('model_started')==2 and facts.count('tool_started')==1
async def semantic_main():
    from weavria.context import ContextCompilerConfig
    from weavria.core import ModelResponse
    from weavria.harness import AgentHarness, CompletionCriterion, HarnessConfig, TaskSpecification
    from weavria.models import FakeModelProvider, OpenAICompatibleConfig
    from weavria.models import OpenAICompatibleTokenCounter
    from weavria.runtime import RunLimits
    original_record=SQLiteSessionStore.record_resolution
    async def crash_at_commit(self,resolution,**kwargs):
        await original_record(self,resolution,**kwargs)
        if resolution.stage=='execution': os._exit(73)
    if point=='semantic-boundary':
        SQLiteSessionStore.record_resolution=crash_at_commit
    class SemanticProvider:
        capabilities=ModelCapabilities(True,True)
        async def generate_session(self,request):
            work=request.current.task_model_view.progress['working_state']
            if point=='semantic-resume':
                assert work['revision']==1 and work['notes'][0]['summary']=='Check second case'
                return ModelTurn(decision=FinishAction(output='resumed with working state'))
            assert work['revision']==0
            return ModelTurn(decision=ToolBatch(calls=(ToolCallAction(
                call_id='work',tool_name='propose_work',arguments={
                    'notes':[{'kind':'plan','summary':'Check second case'}]}),)))
    spec=TaskSpecification(route='direct',intent='Answer',goal='Explain',
        completion_criteria=(CompletionCriterion(kind='response',description='Answer'),))
    reply=ModelResponse(action=ToolCallAction(
        call_id='interpret',tool_name='declare_task_specification',arguments=spec.model_dump(mode='json')))
    reasoning=FakeModelProvider((reply,) if point=='semantic-boundary' else ())
    counter=OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model='fake',max_output_tokens=256,context_window_tokens=64000),
        encoding='cl100k_base')
    profile=HarnessConfig(configuration_id='resume-test/1',working_records=True,
        context=ContextCompilerConfig(window_tokens=64000,output_tokens=256),
        limits=RunLimits(max_request_bytes=64000))
    async with AgentHarness.from_config(Path(sys.argv[1]),config=profile,
            model=SemanticProvider(),tools=ToolRegistry(()),token_counter=counter,
            reasoning_model=reasoning,reasoning_counter=counter) as harness:
        if point=='semantic-boundary':
            await harness.create('s')
            await harness.send('s','Explain the rule',input_id='original')
        else:
            old,=await harness.sessions.store.history('s')
            if point=='semantic-conflict':
                from weavria.sessions import SessionConflictError
                try:
                    await harness.resume('s',old.run_id)
                except SessionConflictError as error:
                    print(error)
                    return
                raise AssertionError('stale semantic head accepted')
            result=await harness.resume('s',old.run_id)
            assert result.session.run_id==old.run_id
            assert result.session.report.output=='resumed with working state', result
            assert not reasoning.requests
asyncio.run(semantic_main() if point.startswith('semantic-') else main())
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


@pytest.mark.parametrize(
    ("crash", "resume", "reason"),
    [
        ("boundary", "resume", ""),
        ("semantic-boundary", "semantic-resume", ""),
        ("semantic-boundary", "semantic-conflict", "resume_task_version_changed"),
        ("tool", "resume-unsafe", "resume_unsafe_boundary"),
        ("artifact", "resume-unsafe", "resume_unsafe_boundary"),
        ("accepted", "resume-unsafe", "resume_unsafe_boundary"),
        ("boundary", "resume-revoked", "resume_source_authorization_revoked"),
        ("boundary", "resume-expired", "resume_budget_exhausted"),
    ],
)
async def test_resume_only_at_durable_settlement_boundary(
    tmp_path: Path, crash: str, resume: str, reason: str
) -> None:
    path = tmp_path / "resume.db"

    async def run(point: str) -> subprocess.CompletedProcess[str]:
        return await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", CHILD, str(path), point],
            capture_output=True,
            text=True,
            timeout=10,
        )

    interrupted = await run(crash)
    assert interrupted.returncode == 73, interrupted.stderr
    if resume == "semantic-conflict":
        # 注入比结算边界更新的语义版本, 验证恢复者不会覆盖当前语义头。
        with sqlite3.connect(path) as conn:
            conn.execute(
                """UPDATE semantic_heads SET data=json_set(data,'$.version.semantic_sequence',
                   json_extract(data,'$.version.semantic_sequence')+1)"""
            )
    resumed = await run(resume)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.stdout.strip() == reason


async def test_uninterrupted_baseline_matches_resume_call_counts(tmp_path: Path) -> None:
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", CHILD, str(tmp_path / "baseline.db"), "baseline"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
