import asyncio
import json
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from examples.live_task_model_view import (
    Case,
    Dataset,
    evaluate,
    grade,
    main,
    prepare_case,
    run_case,
)
from weavria.core import (
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
    InformationNeed,
    TaskSpecification,
)
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    ModelProvider,
    OpenAICompatibleConfig,
)


def dataset() -> Dataset:
    return Dataset.model_validate_json(
        (Path(__file__).parents[1] / "examples/data/task_model_view/cases.json").read_bytes()
    )


def specification(*fields: str) -> TaskSpecification:
    return TaskSpecification(
        route="execute",
        intent="Read archive",
        goal=f"Read {', '.join(fields)}",
        information_needs=tuple(
            InformationNeed(
                description=f"Archived {field}",
                source="external",
                freshness="stable",
                required_fields=(f"/data/{field}",),
                capability_names=("read_archive",),
            )
            for field in fields
        ),
        capability_needs=(
            CapabilityNeed(name="read_archive", purpose="Read", necessity="required"),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result", description="Read", capability="read_archive"
            ),
            CompletionCriterion(kind="response", description="Return value"),
        ),
    )


def read(field: str, source: str = "primary") -> ModelTurn:
    return ModelTurn(
        decision=ToolBatch(
            calls=(
                ToolCallAction(
                    call_id=f"{field}-{source}",
                    tool_name="read_archive",
                    arguments={"section": field, "source": source},
                ),
            )
        )
    )


def test_dataset_and_grader_keep_negative_samples() -> None:
    data = dataset()
    assert len(data.cases) == 5
    case = data.cases[0]
    assert grade(case, json.dumps(case.expected))["correct"] is True
    for output in (
        None,
        "done",
        "[]",
        "{}",
        '{"code":"invented"}',
        '{"code":"EMBER-742","extra":true}',
    ):
        assert grade(case, output)["correct"] is False
    with pytest.raises(ValidationError):
        Case(case_id="bad", request="read", data={"code": "x"}, expected={"code": "y"})


def test_cli_requires_explicit_live_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "must-not-create"
    monkeypatch.setattr("sys.argv", ["probe", "--output-dir", str(target)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not target.exists()


async def test_revision_preparation_preserves_the_original_work_identity() -> None:
    case = next(c for c in dataset().cases if c.steering)
    model = FakeModelProvider(
        tuple(
            ModelResponse(
                action=ToolCallAction(
                    call_id=f"declare-{index}",
                    tool_name="declare_task_specification",
                    arguments=specification(field).model_dump(mode="json"),
                )
            )
            for index, field in enumerate(("title", "revision_token"))
        )
    )
    prepared = await prepare_case(case, model)
    assert set(prepared) == {1, 2}
    assert len(model.requests) == 2
    assert model.requests[1].task.constraints[0] == case.steering


async def test_failed_preparation_keeps_both_arms_in_the_planned_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(case: Case, provider: ModelProvider) -> dict[int, TaskSpecification]:
        raise ValueError("injected preparation failure")

    monkeypatch.setenv("WEAVRIA_MODEL_NAME", "test")
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "http://model.test")
    monkeypatch.setattr("examples.live_task_model_view.prepare_case", fail)
    result = await evaluate(root=tmp_path / "failed", repeats=1, selected=("single_lookup",))
    rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(result["rows"])
    assert len(rows) == result["planned_runs"] == 2
    assert all(row["status"] == "preparation_failed" and not row["task_success"] for row in rows)
    assert (tmp_path / "failed/runner.py").exists()


async def test_reused_intent_must_belong_to_the_same_frozen_dataset(tmp_path: Path) -> None:
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "report.json").write_text('{"dataset_sha256":"wrong"}')
    with pytest.raises(ValueError, match="another dataset"):
        await evaluate(root=tmp_path / "new", repeats=1, selected=(), intent_dir=prior)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("projected", [False, True])
async def test_real_session_path_uses_oracle_only_for_grading(
    tmp_path: Path, projected: bool
) -> None:
    case = dataset().cases[0]
    model = FakeSessionProvider(
        (read("code"), ModelTurn(decision=FinishAction(output=json.dumps(case.expected))))
    )
    row = await run_case(
        root=tmp_path,
        case=case,
        specifications={1: specification("code")},
        projected=projected,
        provider=model,
        repetition=0,
        config=OpenAICompatibleConfig(
            model="test", max_output_tokens=2048, max_request_bytes=128000
        ),
    )
    assert row["task_success"] is True and row["model_calls"] == 2
    assert row["tool_calls"] == 1 and row["all_expected_fields_delivered"] is True
    assert row["task_outcome"] == "unverified"
    assert "EMBER-742" not in model.requests[0].model_dump_json()
    assert "EMBER-742" in model.requests[1].model_dump_json()
    assert row["phases"] == (["decision", "synthesis"] if projected else ["full", "full"])


async def test_failures_remain_in_trace_and_can_be_recovered(tmp_path: Path) -> None:
    case = Case(
        case_id="failure",
        request="Read archived owner, try mirror on failure",
        data={"owner": "Mina-Z"},
        expected={"owner": "Mina-Z"},
        primary_failures=("owner",),
    )
    model = FakeSessionProvider(
        (
            read("owner"),
            read("owner", "mirror"),
            ModelTurn(decision=FinishAction(output='{"owner":"Mina-Z"}')),
        )
    )
    row = await run_case(
        root=tmp_path,
        case=case,
        specifications={1: specification("owner")},
        projected=True,
        provider=model,
        repetition=0,
        config=OpenAICompatibleConfig(
            model="test", max_output_tokens=2048, max_request_bytes=128000
        ),
    )
    assert row["task_success"] is True
    assert row["failed_tool_calls"] == 1 and row["tool_calls"] == 2


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("failed_fields", [("owner",), ("owner", "deadline")])
async def test_successful_sibling_does_not_turn_failed_attempt_into_early_finalization(
    tmp_path: Path, projected: bool, failed_fields: tuple[str, ...]
) -> None:
    # 与真实失败轨迹保持相同边界:同一工具的一批调用中,一项成功、一项永久失败。
    # 失败只否定 owner/primary 这次尝试;宿主没有枚举替代参数,更没有代模型选择 mirror。
    case = next(case for case in dataset().cases if case.case_id == "primary_failure")
    if len(failed_fields) > 1:
        values = {**case.data, "deadline": "2031-06-18"}
        case = Case(
            case_id="multiple_failures",
            request="Read title, owner and deadline, using another source after a failure",
            data=values,
            expected=values,
            primary_failures=failed_fields,
        )
    first = ToolBatch(
        calls=tuple(
            ToolCallAction(
                call_id=f"{field}-primary",
                tool_name="read_archive",
                arguments={"section": field, "source": "primary"},
            )
            for field in ("title", *failed_fields)
        )
    )
    fallback = ToolBatch(
        calls=tuple(
            ToolCallAction(
                call_id=f"{field}-mirror",
                tool_name="read_archive",
                arguments={"section": field, "source": "mirror"},
            )
            for field in failed_fields
        )
    )
    model = FakeSessionProvider(
        (
            ModelTurn(decision=first),
            ModelTurn(decision=fallback),
            ModelTurn(decision=FinishAction(output=json.dumps(case.expected))),
        )
    )
    row = await run_case(
        root=tmp_path,
        case=case,
        specifications={1: specification("title", *failed_fields)},
        projected=projected,
        provider=model,
        repetition=0,
        config=OpenAICompatibleConfig(
            model="test", max_output_tokens=2048, max_request_bytes=128000
        ),
    )
    assert row["task_success"] is True
    assert row["model_calls"] == 3 and row["tool_calls"] == 1 + 2 * len(failed_fields)
    assert row["failed_tool_calls"] == len(failed_fields)
    assert row["all_expected_fields_delivered"] is True
    assert "FINALIZATION" not in model.requests[1].current.instructions
    assert "FINALIZATION" in model.requests[2].current.instructions
    assert row["phases"] == (
        ["decision", "decision", "synthesis"] if projected else ["full", "full", "full"]
    )


async def test_probe_cancellation_joins_owned_model_task(tmp_path: Path) -> None:
    entered, cleaned = asyncio.Event(), asyncio.Event()

    class Blocked(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            raise AssertionError("unreachable")

    owner = asyncio.create_task(
        run_case(
            root=tmp_path,
            case=dataset().cases[0],
            specifications={1: specification("code")},
            projected=True,
            provider=Blocked(()),
            repetition=0,
            config=OpenAICompatibleConfig(
                model="test", max_output_tokens=2048, max_request_bytes=128000
            ),
        )
    )
    await asyncio.wait_for(entered.wait(), 3)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert cleaned.is_set()
