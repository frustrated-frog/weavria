"""Local HTTP sidecar for the FinScope application adapter."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi import Path as ApiPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from weavria.context import (
    ContextCompilerConfig,
    HistoryRetrievalConfig,
    ModelRequestContextCompiler,
    ModelRequestContextConfig,
)
from weavria.core import ApplicationObservation, CompletionUnverifiable, RecoveryLimits
from weavria.harness import (
    CapabilityResolver,
    EvidenceResolver,
    ModelRequirementCoverageValidator,
    ModelTaskInterpreter,
    TaskAwareSessionPreparer,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
)
from weavria.models.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
)
from weavria.models.provider import ModelProvider, SessionModelProvider
from weavria.runtime import RunLimits
from weavria.sessions import SessionConflictError, SessionEngine
from weavria.tools import ToolRegistry

from .completion import (
    FinScopeAnswerPolicy,
    FinScopeWorkspaceEvidenceExtractor,
    MarketPulseEvidenceExtractor,
)
from .config import FinScopeSettings
from .market import MarketPulseClient, market_pulse_evidence_profile, market_pulse_tool
from .model import HTTPChatCompletionTransport
from .policy import FinScopeReadOnlyPolicy
from .workspace import (
    FinScopeWorkspaceClient,
    finscope_workspace_evidence_profiles,
    finscope_workspace_tools,
)

_MARKET_VIEWS = frozenset({"panorama", "review", "transition", "breadth", "history"})


class WorkspaceNavigation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    view: str = Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_-]*$")
    label: str = Field(min_length=1, max_length=64)
    hint: str | None = Field(default=None, min_length=1, max_length=160)


class PageContext(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    surface: Literal["market_pulse", "finscope_workspace"]
    business_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    current_view: str = Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_-]*$")
    quality_status: Literal["READY", "PARTIAL", "STALE", "UNAVAILABLE"] | None = None
    generated_at: str | None = Field(default=None, max_length=64)
    navigation: WorkspaceNavigation | None = None

    @model_validator(mode="after")
    def validate_surface_shape(self) -> "PageContext":
        if self.surface == "market_pulse":
            if self.current_view not in _MARKET_VIEWS or self.navigation is not None:
                raise ValueError("market_pulse context has an invalid view or navigation")
            return self
        if self.navigation is None or self.navigation.view != self.current_view:
            raise ValueError("workspace context requires matching navigation metadata")
        if any(
            value is not None
            for value in (self.business_date, self.quality_status, self.generated_at)
        ):
            raise ValueError("workspace context cannot carry Market Pulse evidence fields")
        return self


class MessageInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    input_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=4000)
    context: PageContext
    clarification_id: str | None = Field(default=None, min_length=1, max_length=128)


class SessionCreated(BaseModel):
    session_id: str


class ToolTrace(BaseModel):
    call_id: str
    tool_name: str
    status: Literal["succeeded", "failed"]


class MessageOutput(BaseModel):
    session_id: str
    run_id: str
    status: str
    task_outcome: str
    answer: str | None
    error: str | None
    completion_reason: str | None = None
    tools: tuple[ToolTrace, ...]
    clarification_id: str | None = None


class _Runtime:
    def __init__(
        self,
        engine: SessionEngine,
    ) -> None:
        self.engine = engine


def create_app(
    settings: FinScopeSettings | None = None,
    *,
    provider: SessionModelProvider | None = None,
    interpreter_provider: ModelProvider | None = None,
    market_client: MarketPulseClient | None = None,
    workspace_client: FinScopeWorkspaceClient | None = None,
    requirement_coverage: bool = True,
    evidence_guidance: bool | None = None,
) -> FastAPI:
    """Create a sidecar. Uvicorn should invoke this as an application factory."""
    if type(requirement_coverage) is not bool:
        raise TypeError("requirement_coverage must be a boolean")
    if evidence_guidance is not None and type(evidence_guidance) is not bool:
        raise TypeError("evidence_guidance must be a boolean or None")
    use_evidence_guidance = requirement_coverage if evidence_guidance is None else evidence_guidance
    resolved = settings or FinScopeSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        resolved.state_path.parent.mkdir(parents=True, exist_ok=True)
        created_transport: HTTPChatCompletionTransport | None = None
        if provider is None or interpreter_provider is None:
            created_transport = HTTPChatCompletionTransport(
                resolved.model_base_url,
                api_key=resolved.model_api_key,
                timeout_seconds=resolved.model_timeout_seconds,
            )
        model_config = OpenAICompatibleConfig(
            model=resolved.model_name,
            max_output_tokens=resolved.model_max_output_tokens,
            json_mode=resolved.json_mode,
        )
        understanding_model_config = OpenAICompatibleConfig(
            model=resolved.model_name,
            max_output_tokens=resolved.understanding_model_max_output_tokens,
            json_mode=resolved.json_mode,
            reasoning_effort=resolved.understanding_reasoning_effort,
            context_window_tokens=resolved.context_window_tokens,
            context_encoding=resolved.context_encoding,
            max_request_bytes=resolved.run_max_request_bytes,
        )
        active_provider = provider
        if active_provider is None:
            assert created_transport is not None
            active_provider = OpenAICompatibleSessionProvider(
                model_config,
                transport=created_transport,
            )
        active_interpreter = interpreter_provider
        if active_interpreter is None:
            assert created_transport is not None
            active_interpreter = OpenAICompatibleProvider(
                understanding_model_config,
                transport=created_transport,
            )
        created_market = None
        if market_client is None:
            created_market = MarketPulseClient(
                resolved.finscope_base_url,
                timeout_seconds=resolved.request_timeout_seconds,
            )
        active_market = market_client or created_market
        created_workspace = None
        if workspace_client is None:
            created_workspace = FinScopeWorkspaceClient(
                resolved.finscope_base_url,
                timeout_seconds=resolved.request_timeout_seconds,
            )
        active_workspace = workspace_client or created_workspace
        assert (
            active_provider is not None
            and active_market is not None
            and active_workspace is not None
        )
        access_policy = FinScopeReadOnlyPolicy()
        tools = ToolRegistry(
            (market_pulse_tool(active_market), *finscope_workspace_tools(active_workspace))
        )
        token_counter = (
            active_provider.token_counter(encoding=resolved.context_encoding)
            if isinstance(active_provider, OpenAICompatibleSessionProvider)
            else OpenAICompatibleTokenCounter(
                model_config,
                encoding=resolved.context_encoding,
            )
        )
        understanding_counter = OpenAICompatibleTokenCounter(
            understanding_model_config,
            encoding=resolved.context_encoding,
        )
        understanding_context = ModelRequestContextCompiler(
            ModelRequestContextConfig(
                window_tokens=resolved.context_window_tokens,
                output_tokens=resolved.understanding_model_max_output_tokens,
                max_request_bytes=resolved.run_max_request_bytes,
                max_history_bytes=8_000,
                max_history_entries=16,
            ),
            understanding_counter,
        )
        task_interpreter = ModelTaskInterpreter(
            active_interpreter,
            context_compiler=understanding_context,
        )
        engine = SessionEngine(
            resolved.state_path,
            model=active_provider,
            tools=tools,
            instructions=_instructions(),
            limits=RunLimits(
                max_model_calls=resolved.run_max_model_calls,
                max_tool_calls=resolved.run_max_tool_calls,
                max_request_bytes=resolved.run_max_request_bytes,
                max_duration=resolved.run_max_duration_seconds,
                max_model_duration=resolved.model_timeout_seconds,
                max_completion_duration=5.0,
                max_completion_repairs=2,
            ),
            max_request_duration=(
                resolved.understanding_max_duration_seconds + resolved.run_max_duration_seconds
            ),
            completion=FinScopeAnswerPolicy(),
            policy=access_policy,
            history_budget_bytes=8_000,
            task_preparation=TaskAwareSessionPreparer(
                TaskUnderstandingEngine(
                    task_interpreter,
                    coverage_validator=(
                        ModelRequirementCoverageValidator(
                            active_interpreter,
                            context_compiler=understanding_context,
                        )
                        if requirement_coverage
                        else None
                    ),
                    coverage_repairer=task_interpreter if requirement_coverage else None,
                    evidence_profiles=(
                        (
                            market_pulse_evidence_profile(),
                            *finscope_workspace_evidence_profiles(),
                        )
                        if use_evidence_guidance
                        else ()
                    ),
                    max_duration=resolved.understanding_max_duration_seconds,
                ),
                evidence=EvidenceResolver(
                    (MarketPulseEvidenceExtractor(), FinScopeWorkspaceEvidenceExtractor())
                ),
                capabilities=CapabilityResolver(policy=access_policy),
            ),
            context_compiler=ContextCompilerConfig(
                window_tokens=resolved.context_window_tokens,
                output_tokens=resolved.model_max_output_tokens,
            ),
            token_counter=token_counter,
            history_retrieval=HistoryRetrievalConfig(),
            context_recovery=RecoveryLimits(),
        )
        await engine.__aenter__()
        app.state.runtime = _Runtime(engine)
        try:
            yield
        finally:
            await engine.close()
            if created_transport is not None:
                await created_transport.aclose()
            if created_market is not None:
                await created_market.aclose()
            if created_workspace is not None:
                await created_workspace.aclose()

    app = FastAPI(
        title="Weavria FinScope Adapter",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved.allowed_parent_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(SessionConflictError)
    async def session_conflict(_request: Request, _error: SessionConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": "session request conflicts"})

    @app.exception_handler(TaskUnderstandingError)
    async def task_understanding_failure(
        _request: Request, error: TaskUnderstandingError
    ) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": error.error.message})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/api/sessions", response_model=SessionCreated, status_code=201)
    async def create_session(request: Request) -> SessionCreated:
        runtime = _runtime(request)
        return SessionCreated(session_id=await runtime.engine.create())

    @app.post("/api/sessions/{session_id}/messages", response_model=MessageOutput)
    async def send_message(
        request: Request,
        payload: MessageInput,
        session_id: Annotated[str, ApiPath(min_length=1, max_length=128)],
    ) -> MessageOutput:
        runtime = _runtime(request)
        try:
            result = await runtime.engine.send(
                session_id,
                payload.text.strip(),
                input_id=payload.input_id,
                application_context=(
                    ApplicationObservation(
                        observation_id=f"page-{payload.input_id}",
                        input_id=payload.input_id,
                        provider="finscope",
                        payload_schema="finscope.page-context/1",
                        payload=payload.context.model_dump(mode="json"),
                    ),
                ),
                clarification_id=payload.clarification_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        turn = await runtime.engine.store.turn(session_id, result.run_id)
        tools = tuple(
            ToolTrace(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="succeeded" if call.artifact is not None else "failed",
            )
            for call in (turn.calls if turn else ())
        )
        error = result.report.error.message if result.report.error else None
        completion_reason = (
            result.report.assessment.decision.reason
            if result.report.assessment is not None
            and isinstance(result.report.assessment.decision, CompletionUnverifiable)
            else None
        )
        pending = next(
            (
                item.pending_clarification
                for item in await runtime.engine.store.pending_clarifications(session_id)
                if item.run_id == result.run_id and item.pending_clarification is not None
            ),
            None,
        )
        return MessageOutput(
            session_id=session_id,
            run_id=result.run_id,
            status=result.report.status.value,
            task_outcome=result.report.task_outcome,
            answer=result.report.output,
            error=error,
            completion_reason=completion_reason,
            tools=tools,
            clarification_id=pending.clarification_id if pending is not None else None,
        )

    return app


def _runtime(request: Request) -> _Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, _Runtime):
        raise HTTPException(status_code=503, detail="sidecar is starting")
    return runtime


def _instructions() -> str:
    return (
        "You are a read-only research assistant embedded in the FinScope local research workspace. "
        "Answer in concise Chinese. Page metadata only identifies what the user is viewing; "
        "it is not authoritative evidence. Follow the Task Frame route: answer direct requests "
        "without an evidence read, ask one focused question for clarify requests, and for execute "
        "requests use only the exposed capabilities with the smallest relevant scope. "
        "Never invent a URL, path, HTTP method, write, refresh, generation, SQL, or filesystem "
        "operation. A date explicitly named by the user takes precedence over the Market Pulse "
        "page date; otherwise use the page date. Use more than one read only when the question "
        "requires it. Distinguish observed facts from inference and never turn research candidates "
        "into buy or sell advice. Cite every successful read with the exact source_id marker "
        "[来源:<source_id>]. For Market Pulse evidence, also state 数据质量 explicitly. If "
        "required evidence is unavailable, say what cannot be judged."
    )
