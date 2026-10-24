"""Environment-owned configuration for the local FinScope sidecar."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class FinScopeSettings:
    model_base_url: str
    model_name: str
    model_api_key: str
    finscope_base_url: str = "http://127.0.0.1:8080"
    allowed_parent_origin: str = "http://localhost:5173"
    state_path: Path = Path(".weavria/finscope-sessions.sqlite3")
    model_timeout_seconds: float = 60.0
    request_timeout_seconds: float = 15.0
    json_mode: bool = False
    model_max_output_tokens: int = 4096
    context_window_tokens: int = 32_768
    context_encoding: str = "cl100k_base"
    run_max_model_calls: int = 20
    run_max_tool_calls: int = 20
    run_max_request_bytes: int = 64_000
    run_max_duration_seconds: float = 180.0

    def __post_init__(self) -> None:
        _validate_http_url("model_base_url", self.model_base_url)
        _validate_http_url("finscope_base_url", self.finscope_base_url)
        _validate_http_url("allowed_parent_origin", self.allowed_parent_origin, origin_only=True)
        if not self.model_name.strip():
            raise ValueError("model_name is required")
        if not self.context_encoding.strip():
            raise ValueError("context_encoding is required")
        if (
            type(self.model_max_output_tokens) is not int
            or type(self.context_window_tokens) is not int
            or self.model_max_output_tokens <= 0
            or self.context_window_tokens <= self.model_max_output_tokens + 256
        ):
            raise ValueError("context window must leave room beyond output and safety reserves")
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.run_max_model_calls,
                self.run_max_tool_calls,
                self.run_max_request_bytes,
            )
        ):
            raise ValueError("run counter and request limits must be positive integers")
        if (
            not math.isfinite(self.model_timeout_seconds)
            or not math.isfinite(self.request_timeout_seconds)
            or not math.isfinite(self.run_max_duration_seconds)
            or self.model_timeout_seconds <= 0
            or self.request_timeout_seconds <= 0
            or self.run_max_duration_seconds <= 0
        ):
            raise ValueError("timeouts must be positive")

    @classmethod
    def from_env(cls) -> "FinScopeSettings":
        return cls(
            model_base_url=_required("WEAVRIA_MODEL_BASE_URL"),
            model_name=_required("WEAVRIA_MODEL_NAME"),
            model_api_key=os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
            finscope_base_url=os.environ.get("WEAVRIA_FINSCOPE_BASE_URL", "http://127.0.0.1:8080"),
            allowed_parent_origin=os.environ.get(
                "WEAVRIA_ALLOWED_PARENT_ORIGIN", "http://localhost:5173"
            ),
            state_path=Path(
                os.environ.get("WEAVRIA_STATE_PATH", ".weavria/finscope-sessions.sqlite3")
            ),
            model_timeout_seconds=float(os.environ.get("WEAVRIA_MODEL_TIMEOUT_SECONDS", "60")),
            request_timeout_seconds=float(os.environ.get("WEAVRIA_FINSCOPE_TIMEOUT_SECONDS", "15")),
            json_mode=os.environ.get("WEAVRIA_MODEL_JSON_MODE", "false").lower()
            in {"1", "true", "yes"},
            model_max_output_tokens=int(os.environ.get("WEAVRIA_MODEL_MAX_OUTPUT_TOKENS", "4096")),
            context_window_tokens=int(os.environ.get("WEAVRIA_CONTEXT_WINDOW_TOKENS", "32768")),
            context_encoding=os.environ.get("WEAVRIA_CONTEXT_ENCODING", "cl100k_base").strip(),
            run_max_model_calls=int(os.environ.get("WEAVRIA_RUN_MAX_MODEL_CALLS", "20")),
            run_max_tool_calls=int(os.environ.get("WEAVRIA_RUN_MAX_TOOL_CALLS", "20")),
            run_max_request_bytes=int(os.environ.get("WEAVRIA_RUN_MAX_REQUEST_BYTES", "64000")),
            run_max_duration_seconds=float(
                os.environ.get("WEAVRIA_RUN_MAX_DURATION_SECONDS", "180")
            ),
        )


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _validate_http_url(name: str, value: str, *, origin_only: bool = False) -> None:
    parsed = urlsplit(value)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} has an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or any(character.isspace() or character in ";," for character in value)
    ):
        raise ValueError(f"{name} must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{name} must not contain credentials")
    if origin_only and (parsed.path or parsed.query or parsed.fragment):
        raise ValueError(f"{name} must be an exact origin without a path")
