from collections.abc import Callable

import pytest

from apps.finscope.config import FinScopeSettings


def test_parent_origin_rejects_paths() -> None:
    with pytest.raises(ValueError, match="exact origin"):
        FinScopeSettings(
            model_base_url="https://model.example/v1",
            model_name="model",
            model_api_key="",
            allowed_parent_origin="http://localhost:5173/widget",
        )


def test_context_window_requires_output_and_safety_capacity() -> None:
    with pytest.raises(ValueError, match="context window"):
        FinScopeSettings(
            model_base_url="https://model.example/v1",
            model_name="model",
            model_api_key="",
            model_max_output_tokens=4096,
            context_window_tokens=4096 + 256,
        )


def test_context_encoding_is_required() -> None:
    with pytest.raises(ValueError, match="context_encoding"):
        FinScopeSettings(
            model_base_url="https://model.example/v1",
            model_name="model",
            model_api_key="",
            context_encoding=" ",
        )


def test_understanding_reasoning_effort_is_bounded() -> None:
    with pytest.raises(ValueError, match="reasoning effort"):
        FinScopeSettings(
            model_base_url="https://model.example/v1",
            model_name="model",
            model_api_key="",
            understanding_reasoning_effort="extreme",  # type: ignore[arg-type]
        )


def test_understanding_reasoning_effort_is_explicitly_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "https://model.example/v1")
    monkeypatch.setenv("WEAVRIA_MODEL_NAME", "model")
    monkeypatch.setenv("WEAVRIA_UNDERSTANDING_REASONING_EFFORT", "none")

    settings = FinScopeSettings.from_env()

    assert settings.understanding_reasoning_effort == "none"


def test_understanding_reasoning_effort_defaults_to_non_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "https://model.example/v1")
    monkeypatch.setenv("WEAVRIA_MODEL_NAME", "model")
    monkeypatch.delenv("WEAVRIA_UNDERSTANDING_REASONING_EFFORT", raising=False)

    settings = FinScopeSettings.from_env()

    assert settings.understanding_reasoning_effort == "none"


def test_understanding_reasoning_effort_can_be_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "https://model.example/v1")
    monkeypatch.setenv("WEAVRIA_MODEL_NAME", "model")
    monkeypatch.setenv("WEAVRIA_UNDERSTANDING_REASONING_EFFORT", "omit")

    settings = FinScopeSettings.from_env()

    assert settings.understanding_reasoning_effort is None


@pytest.mark.parametrize(
    "factory",
    (
        lambda: FinScopeSettings(
            model_base_url="https://model.example/v1",
            model_name="model",
            model_api_key="",
            run_max_model_calls=0,
        ),
        lambda: FinScopeSettings(
            model_base_url="https://model.example/v1",
            model_name="model",
            model_api_key="",
            run_max_tool_calls=0,
        ),
        lambda: FinScopeSettings(
            model_base_url="https://model.example/v1",
            model_name="model",
            model_api_key="",
            run_max_request_bytes=0,
        ),
        lambda: FinScopeSettings(
            model_base_url="https://model.example/v1",
            model_name="model",
            model_api_key="",
            run_max_duration_seconds=0.0,
        ),
        lambda: FinScopeSettings(
            model_base_url="https://model.example/v1",
            model_name="model",
            model_api_key="",
            understanding_max_duration_seconds=0.0,
        ),
    ),
)
def test_run_budgets_must_be_positive(factory: Callable[[], FinScopeSettings]) -> None:
    with pytest.raises(ValueError, match=r"run counter|timeouts"):
        factory()
