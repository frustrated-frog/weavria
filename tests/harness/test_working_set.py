"""选择机制和真实执行接线分别验证;合成需求不代表自然语言规划质量。"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from weavria.core import (
    AdmissionNotice,
    ErrorCategory,
    FinishAction,
    ModelTurn,
    RunError,
    RunStatus,
    SessionSemanticCommitted,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolRecord,
)
from weavria.harness import (
    AdaptiveCapabilityPolicy,
    CapabilityNeed,
    CapabilityResolver,
    CompletionCriterion,
    InformationNeed,
    InformationRequirement,
    NeedFailure,
    NeedLedger,
    NeedState,
    NeedStatus,
    OriginalRequestAnchor,
    TaskAwareSessionPreparer,
    TaskFrame,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
    canonicalize_task_frame,
)
from weavria.harness.context import need_priority_weights
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolFault, ToolRegistry, TypedTool


class Input(BaseModel):
    key: str


class Output(BaseModel):
    value: str


async def execute(arguments: Input, context: ToolContext) -> Output:
    return Output(value=arguments.key)


def tool(
    name: str, *, description: str = "Read requested value", effect: bool = False
) -> TypedTool[Input, Output]:
    return TypedTool(
        name=name,
        version="1",
        description=description,
        input_type=Input,
        output_type=Output,
        execute=execute,
        behavior="side_effect" if effect else "read_only",
    )


def need(identity: str, *names: str, depends_on: tuple[str, ...] = ()) -> InformationRequirement:
    return InformationRequirement(
        need_id=identity,
        description=f"Read {identity} value",
        source="external",
        capability_names=names,
        required_fields=("/value",),
        origin_ref="test",
        depends_on=depends_on,
    )


def problem(requirements: tuple[InformationRequirement, ...]) -> tuple[TaskFrame, NeedLedger]:
    names = tuple(dict.fromkeys(name for n in requirements for name in n.capability_names))
    frame = canonicalize_task_frame(
        TaskFrame(
            frame_id="0" * 64,
            work_id="work",
            revision=1,
            request_anchor=OriginalRequestAnchor(
                input_id="input", task_revision=1, task_sha256="0" * 64
            ),
            interpreter_version="fixture/1",
            catalogue_sha256="0" * 64,
            capability_epoch=1,
            evidence_revision=1,
            specification=TaskSpecification(
                route="execute",
                intent="Read values",
                goal="Read values",
                capability_needs=tuple(
                    CapabilityNeed(name=n, purpose="Read", necessity="required") for n in names
                ),
                completion_criteria=(
                    CompletionCriterion(kind="response", description="Answer"),
                    *(
                        CompletionCriterion(
                            kind="capability_result", description="Read", capability=n
                        )
                        for n in names
                    ),
                ),
            ),
            requirements=requirements,
        )
    )
    ledger = NeedLedger(
        work_id="work",
        task_revision=1,
        frame_revision=1,
        evidence_revision=1,
        states=tuple(NeedState(need_id=n.need_id, status=NeedStatus.PENDING) for n in requirements),
    )
    return frame, ledger


def test_marginal_coverage_diversifies_and_completed_need_releases_slot() -> None:
    frame, ledger = problem((need("alpha", "a_one", "a_two"), need("beta", "b_one")))
    registry = ToolRegistry(tuple(tool(n) for n in ("a_one", "a_two", "b_one")))
    policy = AdaptiveCapabilityPolicy(max_tools=2)
    first = policy.select(
        frame=frame,
        ledger=ledger,
        tools=registry,
        resolver=CapabilityResolver(),
        previous=("a_one", "a_two", "b_one"),
        records=(),
        discover=False,
    )
    assert first.names == ("a_one", "b_one")
    assert first.uncovered_targets == ()
    settled = ledger.model_copy(
        update={
            "states": (
                NeedState(
                    need_id="alpha", status=NeedStatus.SATISFIED, evidence_binding_ids=("accepted",)
                ),
                ledger.states[1],
            )
        }
    )
    second = policy.select(
        frame=frame,
        ledger=settled,
        tools=registry,
        resolver=CapabilityResolver(),
        previous=first.names,
        records=(),
        discover=False,
    )
    assert second.names == ("b_one",)
    assert second.descriptor_bytes < first.descriptor_bytes
    # 需求重新打开后允许恢复工具,退休不是授权撤销或永久封禁。
    assert (
        policy.select(
            frame=frame,
            ledger=ledger,
            tools=registry,
            resolver=CapabilityResolver(),
            previous=second.names,
            records=(),
            discover=False,
        ).names
        == first.names
    )


def test_failure_discount_is_local_and_single_failed_path_remains_selectable() -> None:
    frame, ledger = problem((need("alpha", "a_one", "b_two"),))
    registry = ToolRegistry((tool("a_one"), tool("b_two")))
    failure = NeedFailure(
        call_id="fail", tool_name="a_one", category=ErrorCategory.TOOL_FAILURE, retryable=False
    )
    record = ToolRecord(
        exchange=ToolExchange(
            action=ToolCallAction(call_id="fail", tool_name="a_one", arguments={"key": "alpha"}),
            observation=ToolFailedObservation(
                call_id="fail",
                tool_name="a_one",
                error=RunError(category=ErrorCategory.TOOL_FAILURE, message="unavailable"),
            ),
        )
    )
    ledger = ledger.model_copy(
        update={
            "states": (
                NeedState(need_id="alpha", status=NeedStatus.UNAVAILABLE, failures=(failure,)),
            )
        }
    )
    policy = AdaptiveCapabilityPolicy(max_tools=1)
    selected = policy.select(
        frame=frame,
        ledger=ledger,
        tools=registry,
        resolver=CapabilityResolver(),
        previous=("a_one",),
        records=(record,),
        discover=False,
    )
    assert selected.names == ("b_two",)
    assert policy.select(
        frame=frame,
        ledger=ledger,
        tools=registry.select(("a_one",)),
        resolver=CapabilityResolver(),
        previous=("a_one",),
        records=(record,),
        discover=False,
    ).names == ("a_one",)
    # 不属于当前运行记录的失败不能污染新的工具选择。
    assert policy.select(
        frame=frame,
        ledger=ledger,
        tools=registry,
        resolver=CapabilityResolver(),
        previous=("a_one",),
        records=(),
        discover=False,
    ).names == ("a_one",)


def test_descriptor_budget_reports_uncovered_targets_without_overflow() -> None:
    frame, ledger = problem((need("alpha", "large"), need("beta", "small")))
    registry = ToolRegistry((tool("large", description="source " * 400), tool("small")))
    cost = len(registry.get("small").descriptor.model_dump_json().encode())
    selected = AdaptiveCapabilityPolicy(max_descriptor_bytes=cost).select(
        frame=frame,
        ledger=ledger,
        tools=registry,
        resolver=CapabilityResolver(),
        previous=(),
        records=(),
        discover=False,
    )
    assert selected.names == ("small",)
    assert selected.descriptor_bytes == cost
    assert selected.uncovered_targets == ("need:alpha",)


def test_dependency_frontier_priorities_are_shared_and_cycle_is_rejected() -> None:
    requirements = (need("root", "read_root"), need("leaf", "read_leaf", depends_on=("root",)))
    frame, ledger = problem(requirements)
    weights = need_priority_weights(requirements, ledger)
    assert weights["root"] > weights["leaf"]
    selected = AdaptiveCapabilityPolicy(max_tools=1).select(
        frame=frame,
        ledger=ledger,
        tools=ToolRegistry((tool("read_root"), tool("read_leaf"))),
        resolver=CapabilityResolver(),
        previous=(),
        records=(),
        discover=False,
    )
    assert selected.names == ("read_root",)
    opened = ledger.model_copy(
        update={
            "states": (
                NeedState(
                    need_id="root", status=NeedStatus.SATISFIED, evidence_binding_ids=("accepted",)
                ),
                ledger.states[1],
            )
        }
    )
    assert need_priority_weights(requirements, opened)["leaf"] > weights["leaf"]
    cycle = (requirements[0].model_copy(update={"depends_on": ("leaf",)}), requirements[1])
    with pytest.raises(ValueError, match="acyclic"):
        need_priority_weights(cycle, ledger)


@pytest.mark.parametrize("unified", [False, True])
async def test_session_failure_switches_tool_then_keeps_acquired_material_readable(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    unified: bool,
) -> None:
    from weavria.context import ContextCompilerConfig
    from weavria.harness import AgentHarness, HarnessResult
    from weavria.runtime import RunLimits

    calls: list[str] = []

    async def primary(arguments: Input, context: ToolContext) -> Output:
        calls.append("a_source")
        raise ToolFault(ErrorCategory.TOOL_FAILURE, "unavailable", retryable=False)

    async def backup(arguments: Input, context: ToolContext) -> Output:
        calls.append("b_source")
        # 大结果必须卸载,完成信息需求不意味着模型已经看完原文。
        return Output(value="accepted " * 10000)

    registry = ToolRegistry(
        tuple(
            TypedTool(
                name=n,
                version="1",
                description="Read requested value",
                input_type=Input,
                output_type=Output,
                execute=f,
            )
            for n, f in (("a_source", primary), ("b_source", backup))
        )
    )

    class Interpreter:
        version = "fixture/1"

        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            return TaskSpecification(
                route="execute",
                intent="Read a value",
                goal="Read alpha",
                information_needs=(
                    InformationNeed(description="Read alpha value", source="external"),
                ),
                capability_needs=tuple(
                    CapabilityNeed(name=n, purpose="Read value", necessity="required")
                    for n in ("a_source", "b_source")
                ),
                completion_criteria=(
                    CompletionCriterion(kind="response", description="Answer"),
                    *(
                        CompletionCriterion(
                            kind="capability_result", description="Read", capability=n
                        )
                        for n in ("a_source", "b_source")
                    ),
                ),
            )

    from weavria.core import SessionRequest, ToolSucceededObservation

    class ReadbackModel(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            if isinstance(request.current.control, AdmissionNotice):
                assert request.current.control.reason == "capability_view"
                assert request.current.control.unavailable_calls == ((0, "a_source"),)
                assert calls == ["a_source"]
                assert "b_source" in request.current.instructions
                assert "a_source" not in request.current.instructions
            exchanges = [e for e in request.current.history if isinstance(e, ToolExchange)]
            if any(e.action.call_id == "b" for e in exchanges) and not any(
                e.action.call_id == "readback" for e in exchanges
            ):
                names = request.current.instructions
                assert "a_source" not in names and "b_source" in names
                assert "read_artifact" in names
                source = next(e for e in exchanges if e.action.call_id == "b")
                assert isinstance(source.observation, ToolSucceededObservation)
                output = source.observation.output
                assert isinstance(output, dict)
                assert output["representation"] != "full"
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="readback",
                                tool_name="read_artifact",
                                arguments={
                                    "artifact_id": output["artifact_id"],
                                    "pointer": "/value",
                                    "start": 0,
                                    "length": 100,
                                },
                            ),
                        )
                    )
                )
            if any(e.action.call_id == "readback" for e in exchanges):
                recovered = next(e for e in exchanges if e.action.call_id == "readback")
                assert isinstance(recovered.observation, ToolSucceededObservation)
                assert "accepted" in str(recovered.observation.output)
            return await super().generate_session(request)

    model = ReadbackModel(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="a", tool_name="a_source", arguments={"key": "alpha"}
                        ),
                    )
                )
            ),
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        # 整批拒绝后,重规划只提交仍可见的来源。
                        ToolCallAction(
                            call_id="a-retry", tool_name="a_source", arguments={"key": "alpha"}
                        ),
                        ToolCallAction(
                            call_id="b-rejected", tool_name="b_source", arguments={"key": "alpha"}
                        ),
                    )
                )
            ),
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="b", tool_name="b_source", arguments={"key": "alpha"}
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="accepted")),
        )
    )
    understanding = TaskUnderstandingEngine(Interpreter())
    policy = AdaptiveCapabilityPolicy(max_tools=1)
    engine = (
        AgentHarness(
            tmp_path / "run.db",
            model=model,
            tools=registry,
            understanding=understanding,
            working_set=policy,
            context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
            token_counter=session_counter,
            limits=RunLimits(max_request_bytes=64000),
        )
        if unified
        else SessionEngine(
            tmp_path / "run.db",
            model=model,
            tools=registry,
            task_preparation=TaskAwareSessionPreparer(understanding, working_set=policy),
        )
    )
    async with engine as session:
        await session.create("s")
        result = await session.send("s", "Read alpha")
        sessions = session.sessions if isinstance(session, AgentHarness) else session
        events = await sessions.store.events("s")
    report = result.session.report if isinstance(result, HarnessResult) else result.report
    assert report.status is RunStatus.COMPLETED
    assert calls == ["a_source", "b_source"]
    snapshots = [e.data.snapshot for e in events if isinstance(e.data, SessionSemanticCommitted)]
    assert [s.visible_tool_names for s in snapshots[:3]] == [
        ("a_source",),
        ("b_source",),
        ("b_source",),
    ]
    assert [s.version.capability_epoch for s in snapshots[:3]] == [1, 2, 2]


async def test_steering_changes_view_epoch_with_unchanged_declared_tools() -> None:
    from typing import NoReturn

    from weavria.core import SemanticSnapshot, SemanticVersion, TaskInput
    from weavria.sessions import SessionSemanticContext, SessionTaskContext

    names = ("a_source", "b_source")
    registry = ToolRegistry(tuple(tool(n) for n in names))

    class Interpreter:
        version = "fixture/1"

        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            return TaskSpecification(
                route="execute",
                intent="Read values",
                goal="Read values",
                information_needs=tuple(
                    InformationNeed(
                        description=f"Value from {name}",
                        source="external",
                        capability_names=(name,),
                        required=(i == request.task.revision - 1),
                    )
                    for i, name in enumerate(names)
                ),
                capability_needs=tuple(
                    CapabilityNeed(name=name, purpose="Read", necessity="required")
                    for name in names
                ),
                completion_criteria=(
                    CompletionCriterion(kind="response", description="Answer"),
                    *(
                        CompletionCriterion(
                            kind="capability_result", description="Read", capability=name
                        )
                        for name in names
                    ),
                ),
            )

    async def no_artifact(identity: str) -> NoReturn:
        raise AssertionError(identity)

    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(Interpreter()), working_set=AdaptiveCapabilityPolicy(max_tools=1)
    )
    initial = await preparer.prepare(
        SessionTaskContext(
            session_id="s",
            input_id="i",
            task=TaskInput(goal="Read values"),
            history=(),
            tools=registry,
        )
    )
    base = SemanticSnapshot(
        session_id="s",
        input_id="i",
        run_id="r",
        work_id=initial.work_id,
        version=SemanticVersion(
            task_revision=1,
            frame_revision=initial.observation.frame_revision,
            capability_epoch=initial.capability_epoch,
            evidence_revision=initial.evidence_revision,
            semantic_sequence=1,
        ),
        observation=initial.observation,
        visible_tool_names=initial.visible_tool_names,
        need_ledger=initial.need_ledger,
    )
    changed = await preparer.refresh(
        SessionSemanticContext(
            session_id="s",
            input_id="i",
            task=TaskInput(goal="Read values", revision=2),
            history=(),
            tools=registry,
            base_semantic=base,
            records=(),
            read_artifact=no_artifact,
            trigger="steering",
        )
    )
    assert initial.visible_tool_names == ("a_source",)
    assert changed.visible_tool_names == ("b_source",)
    assert changed.capability_epoch > initial.capability_epoch
    reduced = preparer.semantic_reducer.reduce_observation(base, changed.transition)
    assert reduced == changed.observation


def test_selection_experiment_compares_fixed_budget_and_unbounded_baselines() -> None:
    from examples.capability_working_set import experiment

    # 只检查选择层面的性质:同一合成状态轨迹与工具预算,不推断模型任务收益。
    result = experiment()
    totals = result["totals"]
    assert isinstance(totals, dict)
    assert (
        totals["adaptive"]["uncovered_open_need_steps"]
        < totals["fixed_two"]["uncovered_open_need_steps"]
    )
    assert totals["adaptive"]["count_budget_violations"] == 0
    assert totals["monotonic_union"]["count_budget_violations"] > 0


def test_best_single_tool_can_replace_density_greedy_prefix() -> None:
    frame, ledger = problem((need("alpha", "small", "large"), need("beta", "large")))
    registry = ToolRegistry((tool("small"), tool("large", description="x" * 1400)))
    budget = len(registry.get("large").descriptor.model_dump_json().encode())
    selected = AdaptiveCapabilityPolicy(max_tools=2, max_descriptor_bytes=budget).select(
        frame=frame,
        ledger=ledger,
        tools=registry,
        resolver=CapabilityResolver(),
        previous=(),
        records=(),
        discover=False,
    )
    assert selected.names == ("large",)
    assert selected.uncovered_targets == ()


async def test_discovery_uses_read_only_candidates_and_success_clears_old_failure_discount() -> (
    None
):
    from weavria.artifacts import InMemoryArtifactStore
    from weavria.tools import ReadOnlyPolicy, ToolExecutor

    frame, ledger = problem((need("alpha", "a_source"),))
    registry = ToolRegistry((tool("a_source"), tool("b_source"), tool("c_effect", effect=True)))
    failure = NeedFailure(
        call_id="fail", tool_name="a_source", category=ErrorCategory.TOOL_FAILURE, retryable=False
    )
    record = ToolRecord(
        exchange=ToolExchange(
            action=ToolCallAction(call_id="fail", tool_name="a_source", arguments={"key": "alpha"}),
            observation=ToolFailedObservation(
                call_id="fail",
                tool_name="a_source",
                error=RunError(category=ErrorCategory.TOOL_FAILURE, message="unavailable"),
            ),
        )
    )
    ledger = ledger.model_copy(
        update={
            "states": (
                NeedState(
                    need_id="alpha",
                    status=NeedStatus.PARTIAL,
                    failures=(failure,),
                    missing_fields=("/value",),
                ),
            )
        }
    )
    policy = AdaptiveCapabilityPolicy(max_tools=1)
    selected = policy.select(
        frame=frame,
        ledger=ledger,
        tools=registry,
        resolver=CapabilityResolver(),
        previous=("a_source",),
        records=(record,),
        discover=True,
    )
    assert selected.names == ("b_source",)
    # 形成后续成功事实,不使用伪造的成功标志覆盖旧失败。
    executor = ToolExecutor("run", registry, InMemoryArtifactStore(), ReadOnlyPolicy())
    recovered = await executor.execute(
        ToolCallAction(call_id="ok", tool_name="a_source", arguments={"key": "alpha"}),
        max_duration=1,
    )
    selected = policy.select(
        frame=frame,
        ledger=ledger,
        tools=registry,
        resolver=CapabilityResolver(),
        previous=("a_source",),
        records=(record, recovered),
        discover=True,
    )
    assert selected.names == ("b_source",)  # 没有当前需求关联的成功不算修复。
    from weavria.harness import EvidenceProbe

    assert recovered.artifact is not None
    scoped = ledger.model_copy(
        update={
            "probes": (
                EvidenceProbe(
                    probe_id="scoped-success",
                    need_id="alpha",
                    producer_call_id="ok",
                    tool_name="a_source",
                    artifact_id=recovered.artifact.artifact_id,
                    artifact_sha256=recovered.artifact.sha256,
                    status="partial",
                    matched_fields=("/value",),
                    reason_code="matched_current_need",
                ),
            )
        }
    )
    selected = policy.select(
        frame=frame,
        ledger=scoped,
        tools=registry,
        resolver=CapabilityResolver(),
        previous=("a_source",),
        records=(record, recovered),
        discover=True,
    )
    assert selected.names == ("a_source",)


def test_action_without_information_need_keeps_its_unresolved_obligation() -> None:
    from weavria.harness import CompletionObligation

    frame, ledger = problem((need("alpha", "perform"),))
    frame = canonicalize_task_frame(
        frame.model_copy(
            update={
                "requirements": (),
                "obligations": (
                    CompletionObligation(
                        obligation_id="perform-once",
                        kind="capability_result",
                        capability_name="perform",
                    ),
                ),
            }
        )
    )
    ledger = ledger.model_copy(update={"states": ()})
    selected = AdaptiveCapabilityPolicy().select(
        frame=frame,
        ledger=ledger,
        tools=ToolRegistry((tool("perform", effect=True),)),
        resolver=CapabilityResolver(),
        previous=(),
        records=(),
        discover=False,
    )
    assert selected.names == ("perform",)


@pytest.mark.parametrize("scenario", ["recovers", "backup", "exhausted", "partial"])
async def test_shared_source_outage_uses_settled_turns_and_preserves_partial_delivery(
    tmp_path: Path, scenario: str
) -> None:
    calls: list[tuple[str, str]] = []

    async def primary(arguments: Input, context: ToolContext) -> Output:
        calls.append(("primary", arguments.key))
        if scenario == "recovers" and calls.count(("primary", "one")) == 2:
            return Output(value="fresh value")
        raise ToolFault(ErrorCategory.SOURCE_UNAVAILABLE, "Service unreachable", retryable=True)

    async def backup(arguments: Input, context: ToolContext) -> Output:
        calls.append(("backup", arguments.key))
        return Output(value="independent value")

    class ArchiveOutput(BaseModel):
        timestamp: str

    async def archive(arguments: Input, context: ToolContext) -> ArchiveOutput:
        calls.append(("backup", arguments.key))
        return ArchiveOutput(timestamp="a recorded timestamp")

    use_backup = scenario in {"backup", "partial"}
    registry = ToolRegistry(
        (
            TypedTool(
                name="primary",
                version="1",
                description="Read value",
                input_type=Input,
                output_type=Output,
                execute=primary,
                source_group="primary-service",
            ),
            *(
                (
                    TypedTool(
                        name="backup",
                        version="1",
                        description="Read an independently stored copy. " * 15,
                        input_type=Input,
                        output_type=Output,
                        execute=backup,
                        source_group="backup-service",
                    ),
                )
                if scenario == "backup"
                else (
                    TypedTool(
                        name="backup",
                        version="1",
                        description="Retrieve archival timestamp",
                        input_type=Input,
                        output_type=ArchiveOutput,
                        execute=archive,
                        source_group="backup-service",
                    ),
                )
                if scenario == "partial"
                else ()
            ),
        )
    )

    class Interpreter:
        version = "source-outage-fixture/1"

        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            names = ("primary", "backup") if use_backup else ("primary",)
            needs = (
                InformationNeed(
                    description="Read primary value",
                    source="external",
                    capability_names=("primary",) if scenario == "partial" else names,
                    required_fields=("/value",),
                ),
                *(
                    (
                        InformationNeed(
                            description="Retrieve archival timestamp",
                            source="external",
                            capability_names=("backup",),
                            required_fields=("/timestamp",),
                        ),
                    )
                    if scenario == "partial"
                    else ()
                ),
            )
            return TaskSpecification(
                route="execute",
                intent="Read values",
                goal="Read values",
                information_needs=needs,
                capability_needs=tuple(
                    CapabilityNeed(name=name, purpose="Read value", necessity="required")
                    for name in names
                ),
                completion_criteria=(
                    CompletionCriterion(kind="response", description="Answer"),
                    *(
                        CompletionCriterion(
                            kind="capability_result", description="Read value", capability=name
                        )
                        for name in names
                    ),
                ),
            )

    def batch(*items: tuple[str, str], attempt: int = 1) -> ModelTurn:
        return ModelTurn(
            decision=ToolBatch(
                calls=tuple(
                    ToolCallAction(
                        call_id=f"{name}-{key}" if attempt == 1 else f"{name}-{key}-{attempt}",
                        tool_name=name,
                        arguments={"key": key},
                    )
                    for name, key in items
                )
            )
        )

    initial_calls = (("primary", "one"), ("primary", "two"), ("primary", "three"))
    output = (
        "The archive records a timestamp. The primary source is unavailable."
        if scenario == "partial"
        else "Report what is available."
    )
    model = FakeSessionProvider(
        (
            # 三个兄弟失败仍只记一轮,保留下一轮重试机会。
            batch(*initial_calls, *((("backup", "read"),) if scenario == "partial" else ())),
            batch(("primary", "one" if scenario == "recovers" else "retry"), attempt=2),
            *((batch(("backup", "read")),) if scenario == "backup" else ()),
            *(
                (ModelTurn(decision=FinishAction(output=output)),)
                if scenario != "exhausted"
                else ()
            ),
        )
    )
    preparer = TaskAwareSessionPreparer(
        TaskUnderstandingEngine(Interpreter()), working_set=AdaptiveCapabilityPolicy()
    )
    async with SessionEngine(
        tmp_path / "outage.db", model=model, tools=registry, task_preparation=preparer
    ) as session:
        await session.create("s")
        result = await session.send("s", "Read values")
        events = await session.store.events("s")

    snapshots = [
        item.data.snapshot for item in events if isinstance(item.data, SessionSemanticCommitted)
    ]
    assert "primary" in snapshots[1].visible_tool_names
    assert snapshots[1].need_ledger["source_failure_rounds"] == {"primary-service": 1}
    if scenario == "recovers":
        assert "source_failure_rounds" not in snapshots[2].need_ledger
        assert result.report.status is RunStatus.COMPLETED
        assert calls.count(("primary", "one")) == 2
        assert not any(
            isinstance(request.current.control, AdmissionNotice) for request in model.requests
        )
    else:
        assert "primary" not in snapshots[2].visible_tool_names
        assert snapshots[2].need_ledger["source_failure_rounds"] == {"primary-service": 2}
        if scenario == "exhausted":
            assert result.report.error is not None
            assert result.report.error.category is ErrorCategory.SOURCE_UNAVAILABLE
            assert len(model.requests) == 2
        else:
            assert result.report.status is RunStatus.COMPLETED
            if scenario == "backup":
                assert snapshots[2].visible_tool_names == ("backup",)
                assert calls[-1] == ("backup", "read")
            else:
                assert result.report.task_outcome == "unverifiable"
                assert result.report.output == output
                assert snapshots[2].need_ledger["sources_exhausted"] is True
                assert any(
                    isinstance(item, ToolExchange) and item.action.call_id == "backup-read"
                    for item in model.requests[-1].current.history
                )


async def test_model_replans_from_rejected_call_and_committed_evidence_gap(tmp_path: Path) -> None:
    from weavria.core import SessionRequest

    class Page(BaseModel):
        fields: dict[str, str]

    executed: list[str] = []

    async def read(arguments: Input, context: ToolContext) -> Page:
        executed.append(arguments.key)
        return Page(
            fields={
                "title": "Accepted title",
                **({"detail": "Missing detail now obtained"} if arguments.key == "detail" else {}),
            }
        )

    class Interpreter:
        version = "replan-fixture/1"

        async def interpret(
            self, request: TaskUnderstandingInput, *, max_duration: float
        ) -> TaskSpecification:
            return TaskSpecification(
                route="execute",
                intent="Read a title and its detail",
                goal="Read the title and detail",
                information_needs=(
                    InformationNeed(
                        description="Title and detail",
                        source="external",
                        capability_names=("read_page",),
                        required_fields=("/fields/title", "/fields/detail"),
                    ),
                ),
                capability_needs=(
                    CapabilityNeed(name="read_page", purpose="Read page", necessity="required"),
                ),
                completion_criteria=(
                    CompletionCriterion(kind="response", description="Answer"),
                    CompletionCriterion(
                        kind="capability_result", description="Read page", capability="read_page"
                    ),
                ),
            )

    class ReplanModel(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            control = request.current.control
            if isinstance(control, AdmissionNotice):
                assert control.progress_reason == "repeated_action"
                replan = control.replan
                assert replan is not None
                assert replan.call_index == 0 and replan.tool_name == "read_page"
                prior = replan.prior
                assert prior is not None and prior.source_sha256 is not None
                assert prior.call_id == "second-title"
                assert len(replan.open_needs) == 1
                assert replan.open_needs[0].missing_fields == ("/fields/detail",)
                # 由拒绝反馈决定下一动作。重复提案未实际调用,只补账本里仍缺的字段。
                assert executed == ["title", "title"]
                return proposal("read-detail", "detail")
            return await super().generate_session(request)

    def proposal(call_id: str, key: str) -> ModelTurn:
        return ModelTurn(
            decision=ToolBatch(
                calls=(
                    ToolCallAction(call_id=call_id, tool_name="read_page", arguments={"key": key}),
                )
            )
        )

    model = ReplanModel(
        (
            proposal("first-title", "title"),
            proposal("second-title", "title"),
            proposal("third-title", "title"),
            ModelTurn(decision=FinishAction(output="Accepted title and its detail.")),
        )
    )
    registry = ToolRegistry(
        (
            TypedTool(
                name="read_page",
                version="1",
                description="Read a field on the page",
                input_type=Input,
                output_type=Page,
                execute=read,
            ),
        )
    )
    async with SessionEngine(
        tmp_path / "replan.db",
        model=model,
        tools=registry,
        task_preparation=TaskAwareSessionPreparer(
            TaskUnderstandingEngine(Interpreter()), working_set=AdaptiveCapabilityPolicy()
        ),
    ) as session:
        await session.create("s")
        result = await session.send("s", "Read the title and detail")

    assert result.report.status is RunStatus.COMPLETED, result.report.error
    assert executed == ["title", "title", "detail"]
