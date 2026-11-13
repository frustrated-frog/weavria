"""确定性验证执行与验收边界;Fake 模型不证明发现的真实任务收益。"""

import asyncio
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import JsonValue, TypeAdapter

import examples.live_capability_execution as execution
from examples.capability_execution_environment import (
    EnvironmentSeed,
    LocalEnvironment,
    LocalSnapshotEvidence,
    instant,
    local_environment,
)
from examples.capability_execution_grading import AcceptedCall, ExecutionDataset, grade
from examples.live_capability_execution import (
    FIXTURES,
    VARIANTS,
    harness_config,
    load_fixtures,
    run_case,
)
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    ErrorCategory,
    FinishAction,
    ModelResponse,
    ModelTurn,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
)
from weavria.harness import (
    CapabilityNeed,
    CompletionCriterion,
    HarnessConfig,
    InformationNeed,
    InformationRequirement,
    TaskSpecification,
)
from weavria.models import FakeModelProvider, FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.tools import ReadOnlyPolicy, ToolContext, ToolExecutor, ToolFault


@pytest.fixture
def fixtures() -> tuple[ExecutionDataset, EnvironmentSeed]:
    return load_fixtures(
        FIXTURES / "capability_execution.json", FIXTURES / "capability_execution_environment.json"
    )


@pytest.fixture
def environment(fixtures: tuple[ExecutionDataset, EnvironmentSeed]) -> Generator[LocalEnvironment]:
    with local_environment(fixtures[1]) as value:
        yield value


async def invoke(environment: LocalEnvironment, name: str, **arguments: JsonValue) -> JsonValue:
    tool = environment.registry().get(name)
    context = ToolContext("run", InMemoryArtifactStore())
    try:
        return await tool.invoke(tool.validate_arguments(arguments), context)
    finally:
        context.close()


async def test_tools_read_real_bytes_metadata_and_revision_sources(
    environment: LocalEnvironment, fixtures: tuple[ExecutionDataset, EnvironmentSeed]
) -> None:
    cases = {case.id: case for case in fixtures[0].cases}
    assert len(environment.registry().descriptors) == 91
    result = await invoke(environment, "compute_file_checksum", path="/work/build.bin")
    assert result == cases["file_fingerprint"].expected
    assert (
        await invoke(environment, "read_file_metadata", path="/work/notes.txt")
        == cases["file_properties"].expected
    )
    result = await invoke(environment, "read_archived_text", archive_id="ARC-204")
    assert result == {"text": cases["archive_revision"].expected["current_text"]}
    result = await invoke(environment, "read_document_history", archive_id="ARC-204")
    assert isinstance(result, dict) and len(cast(list[JsonValue], result["revisions"])) == 2
    assert await invoke(environment, "locate_asset", asset_id="release-17") == {
        "path": "/objects/7e49c2a8.bin"
    }
    # 修改真实文件应改变结果,防止工具实现偷偷返回已知标准答案。
    (environment.root / "work/build.bin").write_bytes(b"changed")
    assert (
        await invoke(environment, "compute_file_checksum", path="/work/build.bin")
        != cases["file_fingerprint"].expected
    )
    # 模板干扰项也是不同的实际数据源,不能全部指向循环的最后一个变量。
    for name in ("read_billing_configuration", "read_search_error_logs"):
        result = await invoke(environment, name)
        assert isinstance(result, dict) and result["source"] == name


async def test_free_windows_match_independent_minute_occupancy_oracle(
    environment: LocalEnvironment, fixtures: tuple[ExecutionDataset, EnvironmentSeed]
) -> None:
    result = await invoke(
        environment,
        "find_available_slots",
        person="Lee",
        start="2026-11-16T09:00:00Z",
        end="2026-11-16T13:00:00Z",
        duration_minutes=30,
    )
    case = next(case for case in fixtures[0].cases if case.id == "free_windows")
    assert result == case.expected
    # 与工具的排序/区间合并算法独立:逐分钟涂格,再丢弃不足 30 格的连续片段。
    origin = instant("2026-11-16T09:00:00Z").timestamp()
    busy = set[int]()
    for item in fixtures[1].calendars["Lee"]:
        busy.update(
            range(
                int((instant(item.start).timestamp() - origin) / 60),
                int((instant(item.end).timestamp() - origin) / 60),
            )
        )
    runs: list[list[int]] = [[]]
    for minute in range(240):
        if minute in busy:
            if runs[-1]:
                runs.append([])
        else:
            runs[-1].append(minute)
    expected = {minute for run in runs if len(run) >= 30 for minute in run}
    returned: set[int] = set()
    assert isinstance(result, dict)
    for item in cast(list[dict[str, str]], result["intervals"]):
        returned.update(
            range(
                int((instant(item["start"]).timestamp() - origin) / 60),
                int((instant(item["end"]).timestamp() - origin) / 60),
            )
        )
    assert returned == expected


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "/work/../work/build.bin",
        "//work/build.bin",
        "/work//build.bin",
        "/work/./build.bin",
        "/work/%2e%2e/build.bin",
        "/sources/assets.json",
        "work/build.bin",
    ],
)
async def test_unlisted_and_noncanonical_paths_are_unavailable(
    environment: LocalEnvironment, path: str
) -> None:
    with pytest.raises(ToolFault) as error:
        await invoke(environment, "compute_file_checksum", path=path)
    assert error.value.category is ErrorCategory.SOURCE_UNAVAILABLE


@pytest.mark.parametrize("kind", ["file_link", "parent_link", "fifo", "oversized", "missing"])
async def test_file_boundary_rejects_links_special_files_and_growth(
    environment: LocalEnvironment, tmp_path: Path, kind: str
) -> None:
    target = environment.root / "work/build.bin"
    if kind == "parent_link":
        (environment.root / "work").rename(environment.root / "moved")
        (environment.root / "work").symlink_to(tmp_path, target_is_directory=True)
    else:
        target.unlink()
        if kind == "file_link":
            (tmp_path / "private").write_text("private")
            target.symlink_to(tmp_path / "private")
        elif kind == "fifo":
            os.mkfifo(target)
        elif kind == "oversized":
            target.write_bytes(b"x" * 65537)
    with pytest.raises(ToolFault):
        await invoke(environment, "compute_file_checksum", path="/work/build.bin")


async def test_cancellation_and_exception_remove_environment(
    fixtures: tuple[ExecutionDataset, EnvironmentSeed],
) -> None:
    roots: list[Path] = []
    entered = asyncio.Event()

    async def work() -> None:
        with local_environment(fixtures[1]) as value:
            roots.append(value.root)
            entered.set()
            await invoke(value, "compute_file_checksum", path="/work/build.bin")
            pytest.fail("cancelled invocation must not complete")

    task = asyncio.create_task(work())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not roots[0].exists()
    root = Path("unused")
    with pytest.raises(RuntimeError), local_environment(fixtures[1]) as value:
        root = value.root
        raise RuntimeError("injected")
    assert not root.exists()


def test_grader_requires_result_and_matching_accepted_evidence(
    fixtures: tuple[ExecutionDataset, EnvironmentSeed],
) -> None:
    case = fixtures[0].cases[0]
    correct = AcceptedCall(
        "compute_file_checksum", {"path": "/work/build.bin"}, True, 2, 3, "a" * 64
    )
    answer = json.dumps(case.expected)
    assert grade(case, output=answer, completed=True, calls=(correct,)).task_success
    for calls in (
        (),
        (replace(correct, succeeded=False),),
        (replace(correct, arguments={"path": "/work/notes.txt"}),),
        (replace(correct, started_sequence=0),),
        (replace(correct, artifact_sha256=None),),
    ):
        assert not grade(case, output=answer, completed=True, calls=calls).task_success
    assert not grade(
        case, output='{"sha256":"invented"}', completed=True, calls=(correct,)
    ).task_success
    assert not grade(case, output=answer, completed=False, calls=(correct,)).task_success


def test_chain_must_observe_lookup_before_dependent_call_and_accepts_calendar_alternatives(
    fixtures: tuple[ExecutionDataset, EnvironmentSeed],
) -> None:
    cases = {case.id: case for case in fixtures[0].cases}
    case = cases["locate_fingerprint"]
    first = AcceptedCall("locate_asset", {"asset_id": "release-17"}, True, 2, 5, "a" * 64)
    second = AcceptedCall(
        "compute_file_checksum", {"path": "/objects/7e49c2a8.bin"}, True, 6, 9, "b" * 64
    )
    assert grade(
        case, output=json.dumps(case.expected), completed=True, calls=(first, second)
    ).task_success
    assert not grade(
        case,
        output=json.dumps(case.expected),
        completed=True,
        calls=(first, replace(second, started_sequence=3)),
    ).task_success
    case = cases["free_windows"]
    for path in case.evidence_paths:
        calls = tuple(
            AcceptedCall(probe.name, probe.arguments, True, 1, 2, "a" * 64) for probe in path
        )
        assert grade(
            case, output=json.dumps(case.expected), completed=True, calls=calls
        ).task_success


def test_safe_stop_is_separate_from_success_and_generic_failure(
    fixtures: tuple[ExecutionDataset, EnvironmentSeed],
) -> None:
    case = next(case for case in fixtures[0].cases if case.unavailable)
    held = grade(case, output=None, completed=False, calls=(), discovery_exhausted=True)
    assert held.safe_stop and held.expected_behavior and not held.task_success
    assert not grade(case, output=None, completed=False, calls=()).expected_behavior
    assert not grade(case, output='{"status":"deleted"}', completed=True, calls=()).safe_stop
    assert grade(case, output=json.dumps(case.expected), completed=True, calls=()).task_success
    direct = fixtures[0].cases[-1]
    assert not grade(
        direct,
        output='{"length_identifies_content":0,"hash_depends_on_content":1}',
        completed=True,
        calls=(),
    ).task_success


@pytest.mark.parametrize(
    "variant,evidence_mode",
    [
        ("all_tools", "local_snapshot"),
        ("fit_repair", "local_snapshot"),
        ("fit_repair", "structural"),
    ],
)
async def test_real_runtime_executes_local_tool_then_independent_grader_checks_answer(
    fixtures: tuple[ExecutionDataset, EnvironmentSeed],
    session_counter: OpenAICompatibleTokenCounter,
    variant: Literal["all_tools", "fit_repair"],
    evidence_mode: Literal["structural", "local_snapshot"],
) -> None:
    case, seed = fixtures[0].cases[0], fixtures[1]
    specification = TaskSpecification(
        route="execute",
        intent="Compute a fingerprint",
        goal=case.query,
        information_needs=(
            InformationNeed(
                description="Current checksum",
                source="external",
                freshness="current",
                required_fields=("/sha256",),
                capability_names=("compute_file_checksum",),
            ),
        ),
        capability_needs=(
            CapabilityNeed(
                name="compute_file_checksum", purpose="Read checksum", necessity="required"
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result",
                description="Read checksum",
                capability="compute_file_checksum",
            ),
            CompletionCriterion(kind="response", description="Return requested JSON"),
        ),
    )
    reasoner = FakeModelProvider(
        tuple(
            ModelResponse(
                action=ToolCallAction(
                    call_id=str(position),
                    tool_name=name,
                    arguments=arguments,
                )
            )
            for position, (name, arguments) in enumerate(
                cast(
                    tuple[tuple[str, dict[str, JsonValue]], ...],
                    (
                        ("nominate_capabilities", {"names": ["compute_file_checksum"]}),
                        (
                            "inspect_capability_fit",
                            {
                                "outcome": "ready",
                                "judgments": [
                                    {
                                        "name": "compute_file_checksum",
                                        "decision": "supported",
                                        "source": {"quote": "content fingerprint"},
                                        "support": [
                                            {"pointer": "/description", "quote": "content checksum"}
                                        ],
                                    }
                                ],
                                "gaps": [],
                            },
                        ),
                        ("declare_task_specification", specification.model_dump(mode="json")),
                    ),
                )
            )
        )
    )
    model = FakeSessionProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="digest",
                            tool_name="compute_file_checksum",
                            arguments={"path": "/work/build.bin"},
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output=json.dumps(case.expected))),
        )
    )
    # 这个 Fake 只验证执行接线。为一次扫描缩小目录;真实实验仍使用冻结的 91 项。
    seed = seed.model_copy(update={"domains": (), "views": ()})
    assert variant in {"all_tools", "fit_repair"}
    row = await run_case(
        seed,
        case,
        variant,
        model=model,
        reasoning_model=reasoner,
        counter=session_counter,
        index=None,
        evidence_mode=evidence_mode,
    )
    result = cast(dict[str, object], row["grade"])
    if evidence_mode == "structural":
        assert row["status"] == "failed" and result["task_success"] is False
        states = cast(list[dict[str, object]], row["need_states"])
        assert states[0]["missing_fields"] == ("$freshness",)
    else:
        assert row["status"] == "completed" and result["task_success"] is True
    assert len(cast(list[object], row["calls"])) == 1
    assert len(reasoner.requests) == (0 if variant == "all_tools" else 3)


def test_fixture_environment_is_bound_before_execution(
    fixtures: tuple[ExecutionDataset, EnvironmentSeed],
    tmp_path: Path,
) -> None:
    path = tmp_path / "changed.json"
    path.write_text(fixtures[1].model_dump_json())
    with pytest.raises(ValueError, match="frozen"):
        load_fixtures(FIXTURES / "capability_execution.json", path)


def test_all_experiment_configurations_serialize_nested_discovery_settings() -> None:
    for variant in VARIANTS:
        value = TypeAdapter(HarnessConfig).dump_python(
            harness_config(variant, 4096, 180), mode="json"
        )
        encoded = json.dumps(value)
        assert json.loads(encoded)["semantic_discovery"] is not None or variant == "all_tools"


async def test_local_freshness_does_not_upgrade_missing_fields_or_other_versions(
    environment: LocalEnvironment,
) -> None:
    registry, store = environment.registry(), InMemoryArtifactStore()
    record = await ToolExecutor("run", registry, store, ReadOnlyPolicy()).execute(
        ToolCallAction(
            call_id="digest",
            tool_name="compute_file_checksum",
            arguments={"path": "/work/build.bin"},
        ),
        max_duration=1,
    )
    assert record.artifact is not None and record.authorization is not None
    snapshot = await store.get("run", record.artifact.artifact_id)
    extractor = LocalSnapshotEvidence(registry)
    requirement = InformationRequirement(
        need_id="checksum",
        description="Current digest and missing data",
        source="external",
        freshness="current",
        capability_names=("compute_file_checksum",),
        required_fields=("/sha256", "/missing"),
        origin_ref="test",
    )
    evaluation = await extractor.evaluate(requirement, record, snapshot)
    assert evaluation.freshness == "valid" and evaluation.missing_fields == ("/missing",)
    assert evaluation.relation == "partially_supports" and evaluation.proof_strength == "structural"
    assert requirement.freshness == "current"
    altered = record.model_copy(
        update={
            "authorization": record.authorization.model_copy(
                update={"tool_version": "2"},
            )
        }
    )
    assert not extractor.supports(requirement, altered)
    with pytest.raises(ValueError, match="adapter"):
        await extractor.evaluate(requirement, altered, snapshot)
    unrelated = await store.put("other-run", "digest", {"sha256": "a" * 64})
    with pytest.raises(ValueError, match="adapter"):
        await extractor.evaluate(requirement, record, unrelated)


def test_duplicate_json_keys_cannot_hide_a_false_success(
    fixtures: tuple[ExecutionDataset, EnvironmentSeed],
) -> None:
    case = next(case for case in fixtures[0].cases if case.unavailable)
    assert not grade(
        case, output='{"status":"deleted","status":"unavailable"}', completed=True, calls=()
    ).safe_stop


@pytest.mark.parametrize(
    "output",
    ["[" * 10000 + "]" * 10000, '{"status":NaN}', "not JSON"],
    ids=("deeply_nested", "nonfinite", "invalid_json"),
)
def test_malformed_final_answers_fail_grading_without_aborting_experiment(
    fixtures: tuple[ExecutionDataset, EnvironmentSeed],
    output: str,
) -> None:
    case = next(case for case in fixtures[0].cases if case.unavailable)
    assert not grade(case, output=output, completed=True, calls=()).expected_behavior


@pytest.mark.parametrize("termination", ["cancel", "timeout"])
async def test_run_cleanup_preserves_completed_calls_on_interruption(
    fixtures: tuple[ExecutionDataset, EnvironmentSeed],
    session_counter: OpenAICompatibleTokenCounter,
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
) -> None:
    roots: list[Path] = []

    @contextmanager
    def tracked(seed: EnvironmentSeed) -> Generator[LocalEnvironment]:
        with local_environment(seed) as value:
            roots.append(value.root)
            yield value

    monkeypatch.setattr(execution, "local_environment", tracked)
    entered, closed = asyncio.Event(), asyncio.Event()

    class BlockingFinish(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            if not self.requests:
                return await super().generate_session(request)
            entered.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            finally:
                closed.set()

    model = BlockingFinish(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="digest",
                            tool_name="compute_file_checksum",
                            arguments={"path": "/work/build.bin"},
                        ),
                    )
                )
            ),
        )
    )
    case, seed = fixtures[0].cases[0], fixtures[1].model_copy(update={"domains": (), "views": ()})
    progress: dict[str, object] = {}
    task = asyncio.create_task(
        run_case(
            seed,
            case,
            "all_tools",
            model=model,
            reasoning_model=FakeModelProvider(()),
            counter=session_counter,
            index=None,
            max_duration=1.5,
            progress=progress,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    if termination == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert progress["status"] == "cancelled"
    else:
        await task
        assert progress["error_category"] == "timeout"
    assert closed.is_set() and not roots[0].exists()
    calls = cast(list[dict[str, object]], progress["calls"])
    assert len(calls) == 1 and calls[0]["succeeded"] is True
    assert not cast(dict[str, object], progress["grade"])["task_success"]
