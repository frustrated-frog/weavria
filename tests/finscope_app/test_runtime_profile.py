import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import JsonValue

from apps.finscope.config import FinScopeSettings
from apps.finscope.runtime_profile import build_runtime_profile
from weavria.context import ContextCompilerConfig, HistoryRetrievalConfig
from weavria.core import RecoveryLimits
from weavria.runtime import RunLimits


def profile(
    settings: FinScopeSettings, *, coverage: bool = True, projector_version: str | None = None
) -> dict[str, JsonValue]:
    return build_runtime_profile(
        settings,
        limits=RunLimits(
            max_model_calls=settings.run_max_model_calls,
            finalization_model_calls=settings.finalization_model_calls,
            finalization_seconds=settings.finalization_seconds,
        ),
        context=ContextCompilerConfig(window_tokens=settings.context_window_tokens),
        history=HistoryRetrievalConfig(),
        recovery=RecoveryLimits(),
        completion_version="completion/1",
        finalization_version="finalization/1",
        execution_provider="provider.session",
        understanding_provider="provider.understanding",
        token_counter_version="counter/1",
        task_preparation_version="preparation/1",
        requirement_coverage=coverage,
        evidence_guidance=coverage,
        task_model_projector_version=projector_version,
    )


def test_profile_is_credential_free_and_its_identity_ignores_secrets(tmp_path: Path) -> None:
    settings = FinScopeSettings(
        model_base_url="https://model.example/v1?api_key=PRIVATE-ROUTE",
        model_name="model",
        model_api_key="PRIVATE-API-KEY",
        state_path=tmp_path / "PRIVATE-STATE.sqlite3",
    )
    first = profile(settings)
    changed_secrets = replace(
        settings,
        model_api_key="DIFFERENT-KEY",
        state_path=tmp_path / "OTHER-STATE.db",
        model_base_url="https://other.example/private?token=ANOTHER-TOKEN",
    )
    assert profile(changed_secrets) == first
    assert "PRIVATE" not in json.dumps(first)
    identity = {key: value for key, value in first.items() if key != "sha256"}
    assert (
        first["sha256"]
        == hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize("field", ["reserve", "reasoning", "coverage"])
def test_profile_changes_with_the_actual_policy_configuration(field: str) -> None:
    settings = FinScopeSettings(
        model_base_url="http://model.test", model_name="fake", model_api_key=""
    )
    baseline = profile(settings)
    if field == "reserve":
        candidate = profile(replace(settings, finalization_model_calls=2))
    elif field == "reasoning":
        candidate = profile(replace(settings, model_reasoning_effort="high"))
    else:
        candidate = profile(settings, coverage=False)
    assert candidate["sha256"] != baseline["sha256"]


def test_projection_is_explicitly_fingerprinted_and_legacy_profile_stays_unchanged() -> None:
    settings = FinScopeSettings(
        model_base_url="http://model.test", model_name="fake", model_api_key=""
    )
    baseline, projected = (
        profile(settings),
        profile(settings, projector_version="task-model-view/1"),
    )
    assert baseline["profile_version"] == "finscope-runtime/1"
    assert projected["profile_version"] == "finscope-runtime/2"
    assert projected["sha256"] != baseline["sha256"]
    assert "task_model_projector_version" not in json.dumps(baseline)


def test_example_loads_the_recommended_profile_without_changing_library_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for line in Path("apps/finscope/.env.example").read_text().splitlines():
        if line and not line.startswith("#"):
            name, value = line.split("=", 1)
            monkeypatch.setenv(name, value)
    recommended = FinScopeSettings.from_env()
    assert recommended.finalization_model_calls == 2 and recommended.finalization_seconds == 30
    assert (
        recommended.model_reasoning_effort == recommended.understanding_reasoning_effort == "none"
    )
    assert recommended.json_mode and recommended.run_max_model_calls == 20
    assert recommended.run_max_tool_calls == 20 and recommended.run_max_duration_seconds == 180
    assert RunLimits().finalization_model_calls == 0
    monkeypatch.delenv("WEAVRIA_FINALIZATION_MODEL_CALLS")
    assert FinScopeSettings.from_env().finalization_model_calls == 0
