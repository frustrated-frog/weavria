"""Credential-free configuration evidence for local FinScope experiments."""

import hashlib
import json
from dataclasses import asdict

from pydantic import JsonValue

from weavria.context import ContextCompilerConfig, HistoryRetrievalConfig
from weavria.core import RecoveryLimits
from weavria.runtime import RunLimits

from .config import FinScopeSettings


def build_runtime_profile(
    settings: FinScopeSettings,
    *,
    limits: RunLimits,
    context: ContextCompilerConfig,
    history: HistoryRetrievalConfig | None = None,
    recovery: RecoveryLimits | None = None,
    completion_version: str,
    finalization_version: str | None = None,
    execution_provider: str,
    understanding_provider: str,
    token_counter_version: str,
    task_preparation_version: str | None = None,
    requirement_coverage: bool,
    evidence_guidance: bool,
    task_model_projector_version: str | None = None,
    harness_configuration_id: str | None = None,
) -> dict[str, JsonValue]:
    """Fingerprint an explicit allow-list; URLs, paths and credentials never enter it."""
    configuration: dict[str, JsonValue] = {
        "model_name": settings.model_name,
        "json_mode": settings.json_mode,
        "execution_reasoning_effort": settings.model_reasoning_effort,
        "understanding_reasoning_effort": settings.understanding_reasoning_effort,
        "execution_output_tokens": settings.model_max_output_tokens,
        "understanding_output_tokens": settings.understanding_model_max_output_tokens,
        "context_window_tokens": settings.context_window_tokens,
        "context_encoding": settings.context_encoding,
        "token_counter_version": token_counter_version,
        "model_timeout_seconds": settings.model_timeout_seconds,
        "understanding_seconds": settings.understanding_max_duration_seconds,
        "tool_request_timeout_seconds": settings.request_timeout_seconds,
        "limits": asdict(limits),
        "context_compiler": asdict(context),
        "completion_version": completion_version,
        "request_seconds": (
            settings.understanding_max_duration_seconds + settings.run_max_duration_seconds
        ),
        "model_max_request_bytes": settings.model_max_request_bytes,
        "execution_provider": execution_provider,
        "understanding_provider": understanding_provider,
        "requirement_coverage": requirement_coverage,
        "evidence_guidance": evidence_guidance,
    }
    # 统一入口选择的检索、恢复及策略版本由 SessionEngine 写入 Run 配置。
    if harness_configuration_id is not None:
        configuration["harness_configuration_id"] = harness_configuration_id
    else:
        configuration["history_budget_bytes"] = 8_000
    if history is not None:
        configuration["history_retrieval"] = asdict(history)
    if recovery is not None:
        configuration["context_recovery"] = recovery.model_dump(mode="json")
    if finalization_version is not None:
        configuration["finalization_version"] = finalization_version
    if task_preparation_version is not None:
        configuration["task_preparation_version"] = task_preparation_version
    if task_model_projector_version is not None:
        configuration["task_model_projector_version"] = task_model_projector_version
    identity: dict[str, JsonValue] = {
        "schema_version": 1,
        "profile_version": "finscope-runtime/3"
        if harness_configuration_id is not None
        else "finscope-runtime/2"
        if task_model_projector_version is not None
        else "finscope-runtime/1",
        "configuration": configuration,
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return {**identity, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}
