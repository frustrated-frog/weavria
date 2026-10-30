"""Execute both request layouts; measure complete requests, not real-model quality."""

import asyncio
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from pydantic import BaseModel

from weavria.context import ContextCompilerConfig
from weavria.core import (
    FinishAction,
    ModelTurn,
    RunStatus,
    SessionContextCompiled,
    ToolBatch,
    ToolCallAction,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    InformationNeed,
    TaskAwareSessionPreparer,
    TaskModelViewProjector,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
)
from weavria.models import FakeSessionProvider, OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class _Input(BaseModel):
    pass


class _Output(BaseModel):
    data: dict[str, str]


class _Clock:
    def monotonic(self) -> float:
        return monotonic()

    def now(self) -> datetime:
        return datetime(2026, 10, 30, tzinfo=UTC)


class _Interpreter:
    version = "fixed-archive-specification/1"

    def __init__(self, needs: int) -> None:
        self.needs = needs

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        # 固定解释结果用于隔离请求表示差异, 不把规格生成能力算作投影收益。
        return TaskSpecification(
            route="execute" if self.needs else "direct",
            intent="Read archive facts" if self.needs else "Explain assistant role",
            goal="Retrieve the requested archive facts" if self.needs else "Explain assistant role",
            information_needs=tuple(
                InformationNeed(
                    description=f"The archived value of key_{index}",
                    source="external",
                    freshness="stable",
                    required_fields=(f"/data/key_{index}",),
                    capability_names=("read_archive",),
                )
                for index in range(self.needs)
            ),
            capability_needs=(
                CapabilityNeed(name="read_archive", purpose="Read archive", necessity="required"),
            )
            if self.needs
            else (),
            completion_criteria=(
                *(
                    (
                        CompletionCriterion(
                            kind="capability_result",
                            description="Read succeeds",
                            capability="read_archive",
                        ),
                    )
                    if self.needs
                    else ()
                ),
                CompletionCriterion(
                    kind="response",
                    description="Deliver the requested values" if self.needs else "Explain role",
                ),
            ),
        )


class TaskViewCostRow(BaseModel):
    needs: int
    projected: bool
    status: RunStatus
    task_outcome: str
    model_calls: int
    tool_calls: int
    request_bytes: int
    transport_bytes: int
    input_tokens: int
    largest_growth_reserve_bytes: int
    smallest_feedback_budget_bytes: int
    compilation_versions: tuple[str, ...]


class TaskViewCostReport(BaseModel):
    schema_version: int = 1
    experiment_version: str = "task-model-view-requests/1"
    scope: str = (
        "Fixed specifications and scripted decisions; actual Session/SQLite/compiler execution; "
        "no paid model or task-quality claim"
    )
    dataset_sha256: str
    counter_version: str
    projector_version: str = TaskModelViewProjector.version
    interpreter_version: str = _Interpreter.version
    decision_script_version: str = "read-then-finish/1"
    input_window_tokens: int = 128_000
    max_request_bytes: int = 96_000
    max_transport_bytes: int = 192_000
    rows: tuple[TaskViewCostRow, ...]


async def build_report() -> TaskViewCostReport:
    data = {f"key_{index}": f"Frozen archived value {index}" for index in range(8)}

    async def read(arguments: _Input, context: ToolContext) -> _Output:
        return _Output(data=data)

    tools = ToolRegistry(
        (
            TypedTool(
                name="read_archive",
                version="1",
                description="Read archived facts",
                input_type=_Input,
                output_type=_Output,
                execute=read,
            ),
        )
    )
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(
            model="fixed-cost-probe", max_output_tokens=256, max_request_bytes=192_000
        ),
        encoding="cl100k_base",
    )
    rows: list[TaskViewCostRow] = []
    with tempfile.TemporaryDirectory(prefix="weavria-task-view-") as folder:
        for needs in (0, 1, 8):
            for projected in (False, True):
                script = (
                    *(
                        (
                            ModelTurn(
                                decision=ToolBatch(
                                    calls=(
                                        ToolCallAction(
                                            call_id="archive",
                                            tool_name="read_archive",
                                            arguments={},
                                        ),
                                    )
                                )
                            ),
                        )
                        if needs
                        else ()
                    ),
                    ModelTurn(
                        decision=FinishAction(
                            output="Archive values read." if needs else "A configured assistant."
                        )
                    ),
                )
                provider = FakeSessionProvider(script)
                async with SessionEngine(
                    Path(folder) / f"{needs}-{projected}.db",
                    model=provider,
                    tools=tools,
                    task_preparation=TaskAwareSessionPreparer(
                        TaskUnderstandingEngine(_Interpreter(needs))
                    ),
                    task_model_projector=TaskModelViewProjector() if projected else None,
                    context_compiler=ContextCompilerConfig(
                        window_tokens=128_000, output_tokens=256
                    ),
                    token_counter=counter,
                    limits=RunLimits(max_request_bytes=96_000),
                    clock=_Clock(),
                    id_generator=lambda: "fixed-run",
                ) as engine:
                    await engine.create("cost")
                    result = await engine.send(
                        "cost",
                        f"Read {needs} archive values" if needs else "Who are you?",
                        input_id="fixed-input",
                    )
                    reports = [
                        event.data.compilation
                        for event in await engine.store.events("cost")
                        if isinstance(event.data, SessionContextCompiled)
                    ]
                    records = await engine.store.run_records("cost", result.run_id)
                rows.append(
                    TaskViewCostRow(
                        needs=needs,
                        projected=projected,
                        status=result.report.status,
                        task_outcome=result.report.task_outcome,
                        model_calls=len(provider.requests),
                        tool_calls=len(records),
                        request_bytes=sum(report.request_bytes for report in reports),
                        transport_bytes=sum(
                            report.transport_request_bytes or 0 for report in reports
                        ),
                        input_tokens=sum(report.input_tokens for report in reports),
                        largest_growth_reserve_bytes=max(
                            (
                                report.task_model_view.growth_reserve_bytes
                                if report.task_model_view
                                else 0
                                for report in reports
                            ),
                            default=0,
                        ),
                        smallest_feedback_budget_bytes=min(
                            (report.feedback_budget_bytes for report in reports), default=0
                        ),
                        compilation_versions=tuple(report.policy_version for report in reports),
                    )
                )
    return TaskViewCostReport(
        dataset_sha256=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        counter_version=counter.version,
        rows=tuple(rows),
    )


if __name__ == "__main__":
    print(asyncio.run(build_report()).model_dump_json(indent=2))
