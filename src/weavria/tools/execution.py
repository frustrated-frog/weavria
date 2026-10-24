"""Typed tool dispatch, independent authorization and tracked artifact reads."""

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core import PydanticSerializationError

from weavria.artifacts import ArtifactSnapshot, ArtifactStore
from weavria.core import (
    ArtifactRef,
    ErrorCategory,
    FeedbackBounds,
    FeedbackDelivery,
    FeedbackEnvelopeBound,
    JsonPointer,
    RunError,
    ToolAuthorization,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolRecord,
    ToolSucceededObservation,
    complete_exchange,
    maximum_complete_exchange_bytes,
    maximum_exchange_bytes,
    minimum_exchange,
)


class ToolDescriptor(BaseModel):
    """Frozen registration contract; schemas and metadata are snapshotted by registry."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
        revalidate_instances="always",
    )
    schema_version: Literal[2, 3, 4] = 2
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    intent_signals: tuple[str, ...] = Field(default=(), max_length=32)
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    behavior: Literal["read_only", "side_effect"] = "read_only"
    idempotency: Literal["replay_safe", "unknown"] = "replay_safe"
    risk: Literal["low", "medium", "high"] = "low"
    approval_required: bool = False
    timeout_seconds: float = Field(default=10.0, gt=0)
    max_attempts: Literal[1] = 1
    sensitivity: Literal["public", "private"] = "public"
    context_paths: tuple[JsonPointer, ...] = ()
    feedback_delivery: FeedbackDelivery = "adaptive"
    feedback_bound_version: str | None = Field(default=None, min_length=1)

    @field_validator("schema_version", "max_attempts", mode="before")
    @classmethod
    def validate_literal_integer(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("version and attempts must be integers")
        return value

    @model_serializer(mode="wrap")
    def preserve_legacy_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.schema_version == 2:
            data.pop("feedback_delivery", None)
        if self.schema_version < 4:
            data.pop("feedback_bound_version", None)
        return data

    @model_validator(mode="after")
    def validate_feedback_delivery_version(self) -> "ToolDescriptor":
        if self.schema_version == 2 and self.feedback_delivery != "adaptive":
            raise ValueError("complete feedback delivery requires descriptor schema version 3")
        if self.schema_version < 4 and self.feedback_bound_version is not None:
            raise ValueError("feedback bound metadata requires descriptor schema version 4")
        if self.schema_version == 4 and (
            self.feedback_delivery != "complete" or self.feedback_bound_version is None
        ):
            raise ValueError("descriptor schema version 4 requires a complete feedback bound")
        if self.feedback_delivery == "adaptive" and self.feedback_bound_version is not None:
            raise ValueError("adaptive feedback cannot declare a complete-data bound")
        return self


@dataclass(frozen=True, slots=True)
class ToolActionIdentity:
    """Opaque execution identity derived from one registered tool's validation boundary.

    The identity deliberately excludes ``call_id``. Valid calls hash the normalized argument
    object produced by the tool contract; invalid and unknown calls hash their raw proposal with
    an explicit normalization state so they cannot collide with a valid execution input.
    """

    tool_version: str
    normalization: Literal["validated", "invalid_arguments", "unknown_tool", "validation_failure"]
    canonical_action_sha256: str
    action_family_sha256: str


class ToolFault(Exception):
    """A stable dispatch failure; raw backend errors remain exception causes."""

    def __init__(self, category: ErrorCategory, message: str, *, retryable: bool = False) -> None:
        if type(retryable) is not bool:
            raise TypeError("tool fault retryable flag must be a boolean")
        self.category = category
        self.retryable = retryable
        super().__init__(message)


class ToolExecutionError(RuntimeError):
    """A framework port broke execution identity; cannot be ordinary tool feedback."""


class ToolContext:
    """Invocation-scoped read capability; captures dependencies and closes on exit."""

    def __init__(
        self,
        run_id: str,
        store: ArtifactStore,
        accepted_artifacts: tuple[ArtifactRef, ...] | None = None,
        *,
        call_id: str | None = None,
    ) -> None:
        self._call_id = call_id
        self._run_id = run_id
        self._store = store
        self._reads: list[str] = []
        self._active = True
        self._accepted = (
            None
            if accepted_artifacts is None
            else {ref.artifact_id: ref for ref in accepted_artifacts}
        )

    @property
    def call_id(self) -> str | None:
        return self._call_id

    @property
    def active(self) -> bool:
        return self._active

    @property
    def dependencies(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._reads))

    async def read_artifact(self, artifact_id: str) -> ArtifactSnapshot:
        """Read within the owning run and record the producing call."""
        if not self._active:
            raise RuntimeError("tool invocation context is closed")
        if self._accepted is not None and artifact_id not in self._accepted:
            raise ToolFault(
                ErrorCategory.POLICY_DENIED, "Artifact is outside the accepted turn view"
            )
        snapshot = await self._store.get(self._run_id, artifact_id)
        if not self._active:
            raise RuntimeError("tool invocation context is closed")
        if snapshot.reference.run_id != self._run_id:
            raise ToolExecutionError("artifact store returned another run's result")
        if self._accepted is not None and snapshot.reference != self._accepted[artifact_id]:
            raise ToolExecutionError("artifact store changed an accepted source identity")
        self._reads.append(snapshot.reference.call_id)
        return snapshot

    def close(self) -> None:
        self._active = False


class RegisteredTool(Protocol):
    """Schema validation surrounds every trusted backend implementation."""

    @property
    def descriptor(self) -> ToolDescriptor: ...

    def validate_arguments(self, arguments: dict[str, JsonValue]) -> BaseModel: ...

    def estimate_feedback_data_json_bytes(
        self, arguments: BaseModel, bounds: FeedbackBounds
    ) -> int | None: ...

    async def invoke(self, arguments: BaseModel, context: ToolContext) -> JsonValue: ...


class TypedTool[Input: BaseModel, Output: BaseModel]:
    """Bind a typed asynchronous backend without erasing boundary validation."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        description: str,
        intent_signals: tuple[str, ...] = (),
        input_type: type[Input],
        output_type: type[Output],
        execute: Callable[[Input, ToolContext], Awaitable[Output]],
        behavior: Literal["read_only", "side_effect"] = "read_only",
        idempotency: Literal["replay_safe", "unknown"] = "replay_safe",
        risk: Literal["low", "medium", "high"] = "low",
        approval_required: bool = False,
        timeout_seconds: float = 10.0,
        sensitivity: Literal["public", "private"] = "public",
        context_paths: tuple[str, ...] = (),
        feedback_delivery: FeedbackDelivery = "adaptive",
        feedback_bound_version: str | None = None,
        feedback_data_bound: Callable[[Input, FeedbackBounds], int] | None = None,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds):
            raise ValueError("tool timeout must be finite")
        has_bound = feedback_bound_version is not None and feedback_data_bound is not None
        if feedback_delivery == "complete" and not has_bound:
            raise ValueError("complete feedback delivery requires a versioned data bound")
        if feedback_delivery == "adaptive" and (
            feedback_bound_version is not None or feedback_data_bound is not None
        ):
            raise ValueError("adaptive feedback cannot declare a complete-data bound")
        adapter = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])
        descriptor = ToolDescriptor(
            schema_version=4 if feedback_delivery != "adaptive" else 2,
            name=name,
            version=version,
            description=description,
            intent_signals=intent_signals,
            input_schema=adapter.validate_python(input_type.model_json_schema()),
            output_schema=adapter.validate_python(output_type.model_json_schema()),
            behavior=behavior,
            idempotency=idempotency,
            risk=risk,
            approval_required=approval_required,
            timeout_seconds=timeout_seconds,
            sensitivity=sensitivity,
            context_paths=context_paths,
            feedback_delivery=feedback_delivery,
            feedback_bound_version=feedback_bound_version,
        )
        self._descriptor_json = descriptor.model_dump_json()
        self._input_type = input_type
        self._output_type = output_type
        self._execute = execute
        self._feedback_data_bound = feedback_data_bound

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor.model_validate_json(self._descriptor_json)

    def validate_arguments(self, arguments: dict[str, JsonValue]) -> BaseModel:
        try:
            return self._input_type.model_validate(arguments, strict=True)
        except ValidationError as exc:
            raise ToolFault(
                ErrorCategory.INVALID_TOOL_ARGUMENTS, "Tool arguments are invalid"
            ) from exc

    def estimate_feedback_data_json_bytes(
        self, arguments: BaseModel, bounds: FeedbackBounds
    ) -> int | None:
        if self._feedback_data_bound is None:
            return None
        normalized = self._input_type.model_validate(arguments.model_dump(), strict=True)
        estimated = self._feedback_data_bound(normalized, FeedbackBounds.model_validate(bounds))
        if type(estimated) is not int or estimated <= 0:
            raise ToolExecutionError("tool feedback estimator returned an invalid byte bound")
        return estimated

    async def invoke(self, arguments: BaseModel, context: ToolContext) -> JsonValue:
        normalized = self._input_type.model_validate(arguments.model_dump(), strict=True)
        output = await self._execute(normalized, context)
        try:
            # Revalidate constructed models and isolate caller-owned nested payloads.
            validated = self._output_type.model_validate(
                output.model_dump(warnings="error"), strict=True
            )
            encoded = json.dumps(
                validated.model_dump(mode="json"), ensure_ascii=False, allow_nan=False
            )
            return TypeAdapter[JsonValue](JsonValue).validate_json(encoded)
        except (ValidationError, PydanticSerializationError, AttributeError, ValueError) as exc:
            raise ToolFault(ErrorCategory.INVALID_TOOL_OUTPUT, "Tool output is invalid") from exc


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    policy_version: str
    reason: str


class ToolPolicy(Protocol):
    """Authorize a concrete descriptor and normalized argument snapshot."""

    async def authorize(
        self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
    ) -> PolicyDecision: ...


class ReadOnlyPolicy:
    """Default permission ceiling: public, low-risk, replay-safe reads only."""

    async def authorize(
        self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
    ) -> PolicyDecision:
        allowed = (
            descriptor.behavior == "read_only"
            and descriptor.idempotency == "replay_safe"
            and descriptor.risk == "low"
            and not descriptor.approval_required
            and descriptor.sensitivity == "public"
        )
        return PolicyDecision(
            allowed, "read_only/1", "allowed" if allowed else "permission ceiling"
        )


class ToolRegistry:
    """An immutable registration set; duplicate stable names are rejected."""

    def __init__(self, tools: tuple[RegisteredTool, ...]) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        for tool in tools:
            name = tool.descriptor.name
            if name in self._tools:
                raise ValueError("duplicate tool registration")
            self._tools[name] = tool

    @property
    def descriptors(self) -> tuple[ToolDescriptor, ...]:
        return tuple(tool.descriptor for tool in self._tools.values())

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolFault(ErrorCategory.UNKNOWN_TOOL, "Unknown tool") from exc

    def action_identity(self, action: ToolCallAction) -> ToolActionIdentity:
        """Return a stable identity without authorizing or dispatching the proposed call.

        Validation is intentionally performed before progress admission so omitted defaults and
        tool-specific Pydantic normalization cannot manufacture distinct exploration attempts.
        A malformed proposal remains executable as ordinary safe tool feedback; its identity
        falls back to the raw JSON object and records why normalization was unavailable.
        """
        action = ToolCallAction.model_validate_json(
            ToolCallAction.model_validate(action).model_dump_json()
        )
        tool = self._tools.get(action.tool_name)
        if tool is None:
            tool_version = "unknown"
            normalization: Literal[
                "validated", "invalid_arguments", "unknown_tool", "validation_failure"
            ] = "unknown_tool"
            identity_arguments = action.arguments
        else:
            tool_version = tool.descriptor.version
            try:
                identity_arguments = _normalized_argument_object(
                    tool.validate_arguments(action.arguments)
                )
                normalization = "validated"
            except ToolFault as exc:
                identity_arguments = action.arguments
                normalization = (
                    "invalid_arguments"
                    if exc.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
                    else "validation_failure"
                )
            except Exception:
                # A broken third-party validator must still be classified by ToolExecutor rather
                # than turn progress inspection becoming a new framework-failure surface.
                identity_arguments = action.arguments
                normalization = "validation_failure"
        return _tool_action_identity(
            action,
            tool_version=tool_version,
            normalization=normalization,
            arguments=identity_arguments,
        )

    def feedback_envelope(
        self,
        action: ToolCallAction,
        *,
        identity: ToolActionIdentity,
        bounds: FeedbackBounds,
    ) -> FeedbackEnvelopeBound:
        """Bind a deterministic next-turn feedback ceiling to one normalized proposal."""
        action = ToolCallAction.model_validate_json(
            ToolCallAction.model_validate(action).model_dump_json()
        )
        bounds = FeedbackBounds.model_validate(bounds)
        if self.action_identity(action) != identity:
            raise ToolExecutionError("tool normalization changed while sizing feedback")
        tool = self._tools.get(action.tool_name)
        delivery: FeedbackDelivery = "adaptive"
        estimator_version = "minimum-feedback/1"
        maximum = maximum_exchange_bytes(action, bounds)
        if tool is not None and identity.normalization == "validated":
            descriptor = tool.descriptor
            if descriptor.feedback_delivery == "complete":
                if descriptor.schema_version != 4 or descriptor.feedback_bound_version is None:
                    raise ToolExecutionError(
                        "complete feedback tool does not provide a versioned bound"
                    )
                arguments = tool.validate_arguments(action.arguments)
                data_bound = tool.estimate_feedback_data_json_bytes(arguments, bounds)
                if data_bound is None:
                    raise ToolExecutionError("complete feedback tool omitted its data bound")
                delivery = "complete"
                estimator_version = descriptor.feedback_bound_version
                maximum = maximum_complete_exchange_bytes(
                    action, bounds, min(data_bound, bounds.max_result_bytes)
                )
        return FeedbackEnvelopeBound(
            call_id=action.call_id,
            tool_name=action.tool_name,
            tool_version=identity.tool_version,
            delivery=delivery,
            estimator_version=estimator_version,
            action_sha256=identity.canonical_action_sha256,
            max_exchange_bytes=maximum,
        )

    def select(self, names: tuple[str, ...]) -> "ToolRegistry":
        """Return an ordered least-capability view; unknown or duplicate names fail closed."""
        if len(names) != len(set(names)):
            raise ValueError("selected tool names must be unique")
        try:
            return ToolRegistry(tuple(self._tools[name] for name in names))
        except KeyError as exc:
            raise ValueError("selected tool is not registered") from exc


class ToolExecutor:
    """Dispatch one attempt and enforce any admission-time feedback promise."""

    def __init__(
        self,
        run_id: str,
        registry: ToolRegistry,
        store: ArtifactStore,
        policy: ToolPolicy,
        feedback_bounds: FeedbackBounds | None = None,
    ) -> None:
        self._run_id, self._registry, self._store, self._policy = run_id, registry, store, policy
        self._failures: dict[str, BaseException] = {}
        self._bounds = (
            FeedbackBounds.model_validate(feedback_bounds) if feedback_bounds is not None else None
        )

    def failure_cause(self, call_id: str) -> BaseException | None:
        """Owner-only diagnostics, including exception chains; never serialize to model or trace."""
        return self._failures.get(call_id)

    async def execute(
        self,
        action: ToolCallAction,
        *,
        max_duration: float,
        accepted_artifacts: tuple[ArtifactRef, ...] | None = None,
        expected_identity: ToolActionIdentity | None = None,
        feedback_envelope: FeedbackEnvelopeBound | None = None,
    ) -> ToolRecord:
        """Return a classified exchange; external cancellation always propagates."""
        if isinstance(max_duration, bool) or not math.isfinite(max_duration) or max_duration <= 0:
            raise ValueError("tool duration must be positive and finite")
        action = ToolCallAction.model_validate_json(
            ToolCallAction.model_validate(action).model_dump_json()
        )
        if feedback_envelope is not None:
            if expected_identity is None:
                raise ToolExecutionError("feedback envelope requires an admitted action identity")
            _validate_feedback_envelope(action, expected_identity, feedback_envelope)
        owner = asyncio.current_task()
        cancellations = owner.cancelling() if owner is not None else 0
        context = ToolContext(self._run_id, self._store, accepted_artifacts, call_id=action.call_id)
        authorization: ToolAuthorization | None = None
        retryable = False
        loop = asyncio.get_running_loop()
        deadline: float | None = None
        limit: asyncio.Timeout | None = None

        def attempt_expired() -> bool:
            return deadline is not None and (
                loop.time() >= deadline or (limit is not None and limit.expired())
            )

        try:
            tool = self._registry.get(action.tool_name)
            descriptor = tool.descriptor
            if (
                expected_identity is not None
                and descriptor.version != expected_identity.tool_version
            ):
                raise ToolExecutionError("tool version changed after admission")
            if (
                descriptor.behavior != "read_only"
                or descriptor.idempotency != "replay_safe"
                or descriptor.approval_required
            ):
                raise ToolFault(
                    ErrorCategory.POLICY_DENIED,
                    "This executor supports replay-safe reads without approval only",
                )
            deadline = loop.time() + min(max_duration, descriptor.timeout_seconds)
            limit = asyncio.timeout_at(deadline)
            async with limit:
                try:
                    arguments = tool.validate_arguments(action.arguments)
                except Exception as exc:
                    if (
                        expected_identity is not None
                        and expected_identity.normalization == "validated"
                    ):
                        raise ToolExecutionError(
                            "tool normalization changed after progress admission"
                        ) from exc
                    raise
                normalized = _normalized_argument_object(arguments)
                actual_identity = _tool_action_identity(
                    action,
                    tool_version=descriptor.version,
                    normalization="validated",
                    arguments=normalized,
                )
                if expected_identity is not None and actual_identity != expected_identity:
                    raise ToolExecutionError("tool normalization changed after progress admission")
                if feedback_envelope is not None and (
                    feedback_envelope.delivery != descriptor.feedback_delivery
                    or feedback_envelope.tool_version != descriptor.version
                    or (
                        feedback_envelope.delivery == "complete"
                        and feedback_envelope.estimator_version != descriptor.feedback_bound_version
                    )
                ):
                    raise ToolExecutionError("tool feedback contract changed after admission")
                policy_input = _normalized_argument_object(arguments)
                decision = await self._policy.authorize(self._run_id, descriptor, policy_input)
                authorization = ToolAuthorization(
                    tool_version=descriptor.version,
                    policy_version=decision.policy_version,
                    allowed=decision.allowed,
                    reason=decision.reason,
                    normalized_arguments=normalized,
                )
                if not decision.allowed:
                    raise ToolFault(ErrorCategory.POLICY_DENIED, "Tool execution denied by policy")
                value = await tool.invoke(arguments, context)
                if owner is not None and owner.cancelling() > cancellations:
                    raise asyncio.CancelledError
                if attempt_expired():
                    raise TimeoutError("tool backend returned after its deadline")
                if feedback_envelope is not None and feedback_envelope.delivery == "complete":
                    if self._bounds is None:
                        raise ToolExecutionError(
                            "complete feedback execution requires enforced feedback bounds"
                        )
                    data_bytes = len(TypeAdapter[JsonValue](JsonValue).dump_json(value))
                    if data_bytes > self._bounds.max_result_bytes:
                        raise ToolExecutionError(
                            "tool result exceeded the admitted artifact ceiling"
                        )
                    if (
                        maximum_complete_exchange_bytes(action, self._bounds, data_bytes)
                        > feedback_envelope.max_exchange_bytes
                    ):
                        raise ToolExecutionError(
                            "tool result exceeded its admitted complete-feedback bound"
                        )
                snapshot = await self._store.put(self._run_id, action.call_id, value)
            if attempt_expired():
                raise TimeoutError("tool returned after its deadline")
            if owner is not None and owner.cancelling() > cancellations:
                raise asyncio.CancelledError
            reference = snapshot.reference
            if reference.run_id != self._run_id or reference.call_id != action.call_id:
                raise ToolExecutionError("artifact store violated execution identity")
            if self._bounds is not None and (
                len(TypeAdapter(str).dump_json(reference.artifact_id))
                > self._bounds.max_artifact_id_json_bytes
                or reference.size_bytes > self._bounds.max_result_bytes
            ):
                raise ToolExecutionError("artifact store violated admitted feedback bounds")
            receipt: dict[str, JsonValue] = {
                "artifact_id": reference.artifact_id,
                "sha256": reference.sha256,
                "size_bytes": reference.size_bytes,
                "policy_version": decision.policy_version,
            }
            record = ToolRecord(
                schema_version=2 if descriptor.feedback_delivery != "adaptive" else 1,
                exchange=ToolExchange(
                    action=action,
                    observation=ToolSucceededObservation(
                        call_id=action.call_id,
                        tool_name=action.tool_name,
                        output=receipt,
                    ),
                ),
                artifact=reference,
                dependencies=context.dependencies,
                authorization=authorization,
                context_paths=descriptor.context_paths,
                feedback_delivery=descriptor.feedback_delivery,
            )
            if feedback_envelope is not None:
                projected = (
                    complete_exchange(action, reference, value)
                    if feedback_envelope.delivery == "complete"
                    else minimum_exchange(record)
                )
                if len(projected.model_dump_json().encode()) > feedback_envelope.max_exchange_bytes:
                    raise ToolExecutionError("tool feedback exceeded its admitted envelope")
            return record
        except ToolExecutionError:
            raise
        except ToolFault as exc:
            if attempt_expired():
                category, message, cause = ErrorCategory.TIMEOUT, "Tool call timed out", exc
                retryable = True
            else:
                category, message, cause = exc.category, str(exc), exc.__cause__
                retryable = exc.retryable
        except TimeoutError as exc:
            if owner is not None and owner.cancelling() > cancellations:
                raise asyncio.CancelledError from exc
            category, message, cause = ErrorCategory.TIMEOUT, "Tool call timed out", exc
            retryable = True
        except Exception as exc:
            if owner is not None and owner.cancelling() > cancellations:
                raise asyncio.CancelledError from exc
            if attempt_expired():
                category, message, cause = ErrorCategory.TIMEOUT, "Tool call timed out", exc
            else:
                category, message, cause = ErrorCategory.TOOL_FAILURE, "Tool backend failed", exc
        finally:
            context.close()
        if owner is not None and owner.cancelling() > cancellations:
            raise asyncio.CancelledError
        if cause is not None:
            self._failures[action.call_id] = cause
        record = ToolRecord(
            exchange=ToolExchange(
                action=action,
                observation=ToolFailedObservation(
                    call_id=action.call_id,
                    tool_name=action.tool_name,
                    error=RunError(
                        category=category,
                        message=message,
                        retryable=retryable,
                        cause_type=type(cause).__name__ if cause else None,
                    ),
                ),
            ),
            dependencies=context.dependencies,
            authorization=authorization,
        )
        if feedback_envelope is not None and (
            len(minimum_exchange(record).model_dump_json().encode())
            > feedback_envelope.max_exchange_bytes
        ):
            raise ToolExecutionError("tool failure feedback exceeded its admitted envelope")
        return record


def _validate_feedback_envelope(
    action: ToolCallAction,
    identity: ToolActionIdentity,
    envelope: FeedbackEnvelopeBound,
) -> None:
    if (
        envelope.call_id != action.call_id
        or envelope.tool_name != action.tool_name
        or envelope.tool_version != identity.tool_version
        or envelope.action_sha256 != identity.canonical_action_sha256
    ):
        raise ToolExecutionError("feedback envelope does not bind the dispatched action")


def _normalized_argument_object(arguments: BaseModel) -> dict[str, JsonValue]:
    normalized = TypeAdapter[JsonValue](JsonValue).validate_json(arguments.model_dump_json())
    if not isinstance(normalized, dict):
        raise ToolExecutionError("tool input must normalize to an object")
    return normalized


def _canonical_sha256(value: JsonValue) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tool_action_identity(
    action: ToolCallAction,
    *,
    tool_version: str,
    normalization: Literal["validated", "invalid_arguments", "unknown_tool", "validation_failure"],
    arguments: dict[str, JsonValue],
) -> ToolActionIdentity:
    family_payload: dict[str, JsonValue] = {
        "identity_schema": 1,
        "tool_name": action.tool_name,
        "tool_version": tool_version,
    }
    return ToolActionIdentity(
        tool_version=tool_version,
        normalization=normalization,
        canonical_action_sha256=_canonical_sha256(
            {
                **family_payload,
                "normalization": normalization,
                "arguments": arguments,
            }
        ),
        action_family_sha256=_canonical_sha256(family_payload),
    )
