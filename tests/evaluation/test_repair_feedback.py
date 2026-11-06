"""固定起点比较的反例与正式 Harness 回归;Fake 只证明控制变量和失败语义。"""

import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from examples import repair_feedback as experiment
from examples.repair_cases import CASES, RepairCase, SeededRepairModel, seed
from examples.repair_feedback import run_trial
from examples.sandbox_execution import dataset, reference
from weavria.core import (
    FinishAction,
    ModelTurn,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
)
from weavria.evaluation import RepairStrategy, RepairTrial, TokenUsage, compare_repair_trials
from weavria.models import ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.sandbox import (
    DockerPythonSandbox,
    SandboxProfile,
    SandboxRequest,
    SandboxResult,
    TextFile,
)


def trial(strategy: RepairStrategy) -> RepairTrial:
    return RepairTrial(
        case_id="sample",
        repetition=1,
        strategy=strategy,
        comparison_sha256="0" * 64,
        expected_seed_sha256="1" * 64,
        seed_sha256="1" * 64,
        setup_valid=True,
        status="completed",
        verified=True,
        setup_tool_calls=2,
        model_calls=2,
        tool_calls=0,
        unchanged_repairs=0,
        model_rejections=0,
        usage=TokenUsage(total_tokens=200),
    )


def test_pairing_rejects_uncontrolled_or_incomplete_members_and_preserves_unknown_usage() -> None:
    left, right = trial("prefix"), trial("dependency_frontier")
    for members in (
        (left,),
        (left, left, right),
        (left, right.model_copy(update={"comparison_sha256": "2" * 64})),
        (
            left,
            right.model_copy(update={"expected_seed_sha256": "2" * 64, "seed_sha256": "2" * 64}),
        ),
    ):
        with pytest.raises(ValueError):
            compare_repair_trials(members)
    right = right.model_copy(
        update={
            "status": "budget_exhausted",
            "verified": False,
            "model_calls": 5,
            "usage": TokenUsage(),
        }
    )
    pair = compare_repair_trials((right, left))[0]
    assert pair.comparable and pair.baseline_verified and not pair.selected_verified
    assert pair.model_call_delta == 3 and pair.token_delta is None
    assert RepairTrial.model_validate_json(right.model_dump_json()) == right
    failed = right.model_copy(update={"setup_valid": False, "seed_sha256": None})
    pair = compare_repair_trials((left, failed))[0]
    assert (
        not pair.comparable and pair.exclusion == "setup_failed" and pair.model_call_delta is None
    )
    unavailable = right.model_copy(
        update={
            "provider_calls": 1,
            "http_statuses": (402,),
            "finish_reasons": ("request_failed",),
        }
    )
    pair = compare_repair_trials((left, unavailable))[0]
    assert not pair.comparable and pair.exclusion == "provider_unavailable"
    assert pair.token_delta is None and unavailable.provider_block_status == 402


def test_invalid_trial_cannot_claim_success_from_wrong_seed_or_nonterminal_work() -> None:
    for changes in (
        {"seed_sha256": None},
        {"status": "failed"},
        {"setup_valid": False},
        {"unchanged_repairs": 1},
        {"provider_calls": 1},
        {"http_statuses": (200,)},
    ):
        with pytest.raises(ValidationError):
            RepairTrial.model_validate_json(
                trial("prefix").model_copy(update=changes).model_dump_json()
            )


async def test_cli_stops_unavailable_provider_and_retains_partial_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    async def unavailable(path: Path, **arguments: object) -> RepairTrial:
        calls.append(path)
        failed = trial("prefix").model_copy(
            update={
                "status": "failed",
                "verified": False,
                "provider_calls": 1,
                "http_statuses": (402,),
                "finish_reasons": ("request_failed",),
                "usage": TokenUsage(),
            }
        )
        cast(Callable[[RepairTrial], None], arguments["save"])(failed)
        return failed

    # 替换整个运行入口,这里只验证停止/落盘控制,不访问远端或执行容器。
    monkeypatch.setattr(experiment, "run_trial", unavailable)
    for name, value in {
        "WEAVRIA_MODEL_NAME": "fake",
        "WEAVRIA_MODEL_BASE_URL": "http://invalid.local",
        "WEAVRIA_MODEL_API_KEY": "test-placeholder",
    }.items():
        monkeypatch.setenv(name, value)
    output = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        ["repair_feedback", "--run", "--image", "sha256:" + "0" * 64, "--output", str(output)],
    )
    await experiment.main()
    assert len(calls) == 1
    assert len((output / "runs.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "interruption.json").read_text())["http_status"] == 402
    assert not (output / "comparison.json").exists()


class FixtureSandbox:
    """输出已知文件快照,不在单测中本地执行模型代码。真实执行由显式 Docker 覆盖。"""

    profile = SandboxProfile(image="sha256:" + "0" * 64)

    def __init__(self, case: RepairCase, *, broken: bool = False) -> None:
        self.fixture = seed(case)
        self.broken = broken

    async def run(self, request: SandboxRequest) -> SandboxResult:
        if self.broken:
            return SandboxResult(
                status="code_error",
                exit_code=1,
                stdout="",
                stderr="injected",
                profile_sha256=self.profile.sha256,
            )
        if request.code == self.fixture.statistics_code:
            rows = reference(dataset())
            if self.fixture.case == "statistics":
                for row in rows:
                    row["functions"] = int(row["functions"]) + 1
            files = (TextFile(path="statistics.json", text=json.dumps(rows)),)
        else:
            clean = seed("clean")
            if request.code == clean.report_code:
                workspace = clean.expected
            else:
                assert request.code in (
                    self.fixture.report_code,
                    (self.fixture.report_code or "") + "# repeated error\n",
                )
                workspace = self.fixture.expected
            files = tuple(f for f in workspace.files if f.path in request.outputs)
        return SandboxResult(
            status="completed",
            exit_code=0,
            stdout="",
            stderr="",
            files=files,
            profile_sha256=self.profile.sha256,
        )


class FinishModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self) -> None:
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(request)
        return ModelTurn(decision=FinishAction(output="Check the workspace."))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("strategy", ("prefix", "dependency_frontier"))
async def test_frozen_setup_and_independent_grading(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    case: RepairCase,
    strategy: RepairStrategy,
) -> None:
    inner = FinishModel()
    model = SeededRepairModel(seed(case), inner)
    saved: list[RepairTrial] = []
    result = await run_trial(
        tmp_path / "case.sqlite",
        fixture=model.fixture,
        model=model,
        backend=FixtureSandbox(case),
        counter=session_counter,
        strategy=strategy,
        repetition=1,
        comparison_sha256="0" * 64,
        usage=TokenUsage,
        save=saved.append,
    )
    assert result.setup_valid and result.seed_sha256 == model.fixture.expected.revision
    assert saved == [result]
    assert result.verified == (case == "clean")
    assert result.setup_tool_calls == len(model.fixture.call_ids) and result.tool_calls == 0
    assert result.model_calls == len(inner.requests) and result.model_calls > 0
    view = inner.requests[0].current.task_model_view
    assert view is not None
    rows = view.progress["capability_outcomes"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    feedback = rows[0]["feedback"]
    assert isinstance(feedback, dict)
    issues = feedback["issues"]
    assert isinstance(issues, list)
    codes = [item["code"] for item in issues if isinstance(item, dict)]
    if strategy == "dependency_frontier":
        assert (
            codes
            == {
                "statistics": ["statistics_content"],
                "report_format": ["report_content"],
                "preservation": ["statistics_preservation"],
                "clean": [],
            }[case]
        )
    elif case == "statistics":
        assert codes == ["statistics_content", "report_stage", "report_content"]


async def test_bad_setup_does_not_reach_live_model_or_count_as_comparable(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    inner = FinishModel()
    model = SeededRepairModel(seed("statistics"), inner)
    result = await run_trial(
        tmp_path / "broken.sqlite",
        fixture=model.fixture,
        model=model,
        backend=FixtureSandbox("statistics", broken=True),
        counter=session_counter,
        strategy="dependency_frontier",
        repetition=1,
        comparison_sha256="0" * 64,
        usage=TokenUsage,
        save=lambda _: None,
    )
    assert not result.setup_valid and not result.verified and not inner.requests
    assert result.status == "failed" and result.model_calls == 0


@pytest.mark.parametrize("save_failure", (False, True))
async def test_cancellation_is_saved_and_propagates_after_harness_closes(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    save_failure: bool,
) -> None:
    started = asyncio.Event()

    class Waiting(FinishModel):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    model = SeededRepairModel(seed("statistics"), Waiting())
    saved: list[RepairTrial] = []

    def save(record: RepairTrial) -> None:
        if save_failure:
            raise OSError("injected report write failure")
        saved.append(record)

    task = asyncio.create_task(
        run_trial(
            tmp_path / "cancel.sqlite",
            fixture=model.fixture,
            model=model,
            backend=FixtureSandbox("statistics"),
            counter=session_counter,
            strategy="prefix",
            repetition=1,
            comparison_sha256="0" * 64,
            usage=TokenUsage,
            save=save,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await task
        if save_failure:
            assert isinstance(cancellation.value.__cause__, OSError)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    if save_failure:
        assert not saved
    else:
        assert len(saved) == 1 and saved[0].status == "cancelled" and saved[0].setup_valid
        assert not saved[0].verified


async def test_repair_metrics_recheck_complete_results_and_exclude_setup_costs(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    class RepairModel(FinishModel):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.requests.append(request)
            if len(self.requests) > 2:
                return ModelTurn(decision=FinishAction(output="Files verified."))
            parent = next(
                item.action.arguments["base"]
                for item in request.current.history
                if isinstance(item, ToolExchange) and item.action.call_id == "seed_report"
            )
            code = (
                seed("report_format").report_code
                if len(self.requests) == 1
                else seed("clean").report_code
            )
            assert code is not None
            if len(self.requests) == 1:
                code += "# repeated error\n"
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id=f"repair_{len(self.requests)}",
                            tool_name="execute_workspace_python",
                            arguments={"code": code, "base": parent, "outputs": ["report.txt"]},
                        ),
                    )
                )
            )

    model = SeededRepairModel(seed("report_format"), RepairModel())
    result = await run_trial(
        tmp_path / "metrics.sqlite",
        fixture=model.fixture,
        model=model,
        backend=FixtureSandbox("report_format"),
        counter=session_counter,
        strategy="dependency_frontier",
        repetition=1,
        comparison_sha256="0" * 64,
        usage=TokenUsage,
        save=lambda _: None,
    )
    assert result.verified and result.setup_tool_calls == 3 and result.tool_calls == 2
    assert result.unchanged_repairs == 1 and result.model_calls == 3
    assert [step.unmet_conditions for step in result.steps] == [("report_content",), ()]


@pytest.mark.docker
@pytest.mark.skipif(
    not os.getenv("WEAVRIA_TEST_DOCKER_IMAGE"), reason="set WEAVRIA_TEST_DOCKER_IMAGE"
)
@pytest.mark.parametrize("case", CASES)
async def test_real_container_seed_matches_frozen_workspace(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    case: RepairCase,
) -> None:
    model = SeededRepairModel(seed(case), FinishModel())
    result = await run_trial(
        tmp_path / "docker.sqlite",
        fixture=model.fixture,
        model=model,
        backend=DockerPythonSandbox(image=os.environ["WEAVRIA_TEST_DOCKER_IMAGE"]),
        counter=session_counter,
        strategy="dependency_frontier",
        repetition=1,
        comparison_sha256="0" * 64,
        usage=TokenUsage,
        save=lambda _: None,
    )
    assert result.setup_valid and result.seed_sha256 == model.fixture.expected.revision
    assert result.verified == (case == "clean")
