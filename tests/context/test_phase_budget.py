"""Phase-specific growth, complete evidence and legacy receipt compatibility."""

import json
from dataclasses import replace

import pytest
from pydantic import BaseModel, ValidationError

from labs.task_model_view_design import PreparedCase, prepared_cases
from weavria.artifacts import InMemoryArtifactStore
from weavria.context import (
    ContextBudgetError,
    ContextCompilerConfig,
    ContextTokenBudgetError,
    SessionContextCompiler,
)
from weavria.core import (
    CONTROL_FEEDBACK_BYTES,
    CompletionNotice,
    CompletionRepair,
    ContextCompilation,
    ControlRequest,
    TaskModelViewReceipt,
    ToolAuthorization,
    ToolCallAction,
    ToolExchange,
    ToolRecord,
    ToolSucceededObservation,
)
from weavria.harness import TaskModelViewProjector
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.runtime import AgentLoop
from weavria.tools import ToolContext, ToolRegistry, TypedTool


@pytest.fixture
async def case() -> PreparedCase:
    cases, failures = await prepared_cases()
    assert not failures
    return next(case for case in cases if case.case_id == "document_lookup")


async def _evidence() -> tuple[InMemoryArtifactStore, ToolRecord]:
    store = InMemoryArtifactStore()
    snapshot = await store.put("run", "read", {"value": 'quoted "\\ 确认🙂 ' * 8})
    ref = snapshot.reference
    record = ToolRecord(
        schema_version=2,
        exchange=ToolExchange(
            action=ToolCallAction(call_id="read", tool_name="read_record", arguments={}),
            observation=ToolSucceededObservation(
                call_id="read",
                tool_name="read_record",
                output={
                    "artifact_id": ref.artifact_id,
                    "sha256": ref.sha256,
                    "size_bytes": ref.size_bytes,
                    "policy_version": "test/1",
                },
            ),
        ),
        artifact=ref,
        authorization=ToolAuthorization(
            allowed=True,
            policy_version="test/1",
            reason="fixture",
            tool_version="1",
            normalized_arguments={},
        ),
        feedback_delivery="complete",
    )
    return store, record


@pytest.mark.parametrize("window", [12_000, 64_000])
@pytest.mark.parametrize("tool_count", [1, 12])
async def test_synthesis_releases_only_exploration_growth_and_keeps_complete_evidence(
    case: PreparedCase,
    window: int,
    tool_count: int,
) -> None:
    class Input(BaseModel):
        key: str

    class Output(BaseModel):
        value: str

    async def read(arguments: Input, context: ToolContext) -> Output:
        return Output(value=arguments.key)

    tools = ToolRegistry(
        tuple(
            TypedTool(
                name=f"read_record_{i}",
                version="1",
                description="Read archival evidence " * 20,
                input_type=Input,
                output_type=Output,
                execute=read,
            )
            for i in range(tool_count)
        )
    )
    instruction_capacity = max(
        len(
            json.dumps(
                AgentLoop.request_template_for(
                    case.task,
                    "Finish using the record",
                    tools,
                    finalizing=finalizing,
                ).instructions,
                ensure_ascii=False,
            ).encode()
        )
        for finalizing in (False, True)
    )
    store, record = await _evidence()
    config = ContextCompilerConfig(window_tokens=window, output_tokens=256, safety_tokens=64)
    http_limit = 24_000 if window == 12_000 else 128_000
    byte_limit = 8_000 if window == 12_000 else 64_000
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="test", max_output_tokens=256, max_request_bytes=http_limit),
        encoding="cl100k_base",
    )
    compiler = SessionContextCompiler(store, config=config, counter=counter)
    view = TaskModelViewProjector().project(
        case.task,
        case.snapshot,
        phase="synthesis",
        max_instruction_bytes=instruction_capacity,
    )
    base = ControlRequest(
        schema_version=5,
        instructions="Finish using the record",
        task=case.task,
        task_model_view=view,
    )
    plan = await compiler.compile(
        run_id="run",
        session_id="s",
        sequence=1,
        base=base,
        turns=(),
        total_turns=0,
        records=(record,),
        required_call_ids=("read",),
        max_request_bytes=byte_limit,
    )
    report = plan.compilation
    receipt = report.task_model_view
    assert report.schema_version == 10 and receipt is not None and receipt.schema_version == 2
    assert receipt.phase == "synthesis" and receipt.growth_reserve_bytes == 0
    # 容量声明没有被缩小。只释放当前阶段不可能发生的增长,必需证据仍完整交付。
    assert receipt.max_view_bytes == view.maximum_bytes
    assert receipt.max_instruction_bytes == instruction_capacity
    assert report.delivery_manifest is not None and report.delivery_manifest.deliveries[0].complete
    assert report.control_reserve_bytes == CONTROL_FEEDBACK_BYTES - len(b"null")
    assert report.input_tokens == counter.count(plan.request)
    assert report.transport_request_bytes == counter.session_request_bytes(plan.request)

    # 在同一预算下换成普通决策,新策略必须继续为工具/进度增长预留,不能以
    # “收尾可装下”推断下一次工具执行也一定能准入。
    decision_view = view.model_copy(update={"phase": "decision"})
    decision = base.model_copy(update={"task_model_view": decision_view})
    if window == 12_000 and tool_count == 12:
        with pytest.raises((ContextBudgetError, ContextTokenBudgetError)):
            await compiler.compile(
                run_id="run",
                session_id="s",
                sequence=1,
                base=decision,
                turns=(),
                total_turns=0,
                records=(record,),
                required_call_ids=("read",),
                max_request_bytes=byte_limit,
            )
        assert (
            report.request_bytes
            + report.control_reserve_bytes
            + view.growth_reserve_bytes(base.instructions)
            > byte_limit
        )
    else:
        ordinary = await compiler.compile(
            run_id="run",
            session_id="s",
            sequence=1,
            base=decision,
            turns=(),
            total_turns=0,
            records=(record,),
            required_call_ids=("read",),
            max_request_bytes=byte_limit,
        )
        assert ordinary.compilation.task_model_view is not None
        assert (
            ordinary.compilation.task_model_view.growth_reserve_bytes
            == (decision_view.growth_reserve_bytes(base.instructions))
            > 0
        )

    # 用已通过准入的收尾请求替换最大长度纠错,直接验证原证明覆盖实际 Token、
    # 规范字节及多一层 JSON 转义后的 HTTP 体,而不只断言 reserve > 0。
    repaired = plan.request.model_copy(
        update={
            "current": plan.request.current.model_copy(
                update={
                    "control": CompletionNotice(
                        candidate_sha256="0" * 64,
                        decision=CompletionRepair(
                            issue_codes=tuple(f"issue_{i}" for i in range(8)),
                            guidance="x" * 256,
                        ),
                    ),
                }
            )
        }
    )
    repaired = type(repaired).model_validate_json(repaired.model_dump_json())
    assert len(repaired.model_dump_json().encode()) <= byte_limit
    assert counter.count(repaired) <= config.input_tokens
    assert counter.session_request_bytes(repaired) <= http_limit


@pytest.mark.parametrize("dimension", ["tokens", "normalized_bytes", "http_bytes"])
async def test_obsolete_growth_cannot_block_a_fitting_synthesis_in_any_budget_dimension(
    case: PreparedCase,
    dimension: str,
) -> None:
    store, record = await _evidence()
    view = TaskModelViewProjector().project(
        case.task,
        case.snapshot,
        phase="synthesis",
        max_instruction_bytes=30_000,
    )
    config = ContextCompilerConfig(window_tokens=64_000, output_tokens=256, safety_tokens=64)
    model_config = OpenAICompatibleConfig(
        model="test", max_output_tokens=256, max_request_bytes=128_000
    )
    counter = OpenAICompatibleTokenCounter(model_config, encoding="cl100k_base")
    base = ControlRequest(
        schema_version=5, instructions="Finish", task=case.task, task_model_view=view
    )
    compiler = SessionContextCompiler(store, config=config, counter=counter)
    wide = await compiler.compile(
        run_id="run",
        session_id="s",
        sequence=1,
        base=base,
        turns=(),
        total_turns=0,
        records=(record,),
        required_call_ids=("read",),
        max_request_bytes=64_000,
    )
    report = wide.compilation
    byte_limit = 64_000
    growth = view.growth_reserve_bytes(base.instructions)
    if dimension == "tokens":
        config = replace(
            config,
            window_tokens=(
                report.input_tokens
                + report.control_reserve_bytes
                + config.output_tokens
                + config.safety_tokens
                + 64
            ),
        )
        assert report.input_tokens + report.control_reserve_bytes + growth > config.input_tokens
    elif dimension == "normalized_bytes":
        byte_limit = report.request_bytes + report.control_reserve_bytes + 64
        assert report.request_bytes + report.control_reserve_bytes + growth > byte_limit
    else:
        assert report.transport_request_bytes is not None
        http_limit = report.transport_request_bytes + 2 * report.control_reserve_bytes + 64
        counter = OpenAICompatibleTokenCounter(
            replace(model_config, max_request_bytes=http_limit),
            encoding="cl100k_base",
        )
        assert (
            report.transport_request_bytes + 2 * (report.control_reserve_bytes + growth)
            > http_limit
        )
    narrow = await SessionContextCompiler(store, config=config, counter=counter).compile(
        run_id="run",
        session_id="s",
        sequence=1,
        base=base,
        turns=(),
        total_turns=0,
        records=(record,),
        required_call_ids=("read",),
        max_request_bytes=byte_limit,
    )
    assert narrow.compilation.request_sha256 == report.request_sha256


async def test_v9_receipt_wire_and_full_capacity_math_remain_readable(case: PreparedCase) -> None:
    view = TaskModelViewProjector().project(
        case.task,
        case.snapshot,
        phase="synthesis",
        max_instruction_bytes=30_000,
    )
    legacy = TaskModelViewReceipt.from_view(view, "Finish", schema_version=1)
    assert legacy.growth_reserve_bytes == view.growth_reserve_bytes("Finish") > 0
    assert "phase" not in legacy.model_dump()
    assert TaskModelViewReceipt.model_validate_json(legacy.model_dump_json()) == legacy
    current = TaskModelViewReceipt.from_view(view, "Finish")
    assert current.growth_reserve_bytes == 0 and current.max_view_bytes == legacy.max_view_bytes

    store = InMemoryArtifactStore()
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="test", max_output_tokens=256, max_request_bytes=128_000),
        encoding="cl100k_base",
    )
    plan = await SessionContextCompiler(
        store,
        config=ContextCompilerConfig(window_tokens=128_000, output_tokens=256),
        counter=counter,
    ).compile(
        run_id="run",
        session_id="s",
        sequence=1,
        turns=(),
        total_turns=0,
        records=(),
        max_request_bytes=96_000,
        base=ControlRequest(
            schema_version=5, instructions="Finish", task=case.task, task_model_view=view
        ),
    )
    old = {
        **plan.compilation.model_dump(mode="json"),
        "schema_version": 9,
        "policy_version": "context-compiler/9[coverage]",
        "task_model_view": legacy.model_dump(mode="json"),
    }
    old_report = ContextCompilation.model_validate_json(json.dumps(old))
    assert old_report.model_dump(mode="json") == old
    with pytest.raises(ValidationError, match="compilation v10"):
        ContextCompilation.model_validate(
            old_report.model_copy(update={"task_model_view": current})
        )
    with pytest.raises(ValidationError, match="compilation v10"):
        ContextCompilation.model_validate(
            plan.compilation.model_copy(update={"task_model_view": legacy})
        )
    with pytest.raises(ValidationError, match="exploration growth"):
        TaskModelViewReceipt.model_validate(current.model_copy(update={"growth_reserve_bytes": 1}))
