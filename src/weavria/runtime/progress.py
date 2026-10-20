"""Semantic-frontier progress control independent of domain completion rules."""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Protocol

from weavria.core import TaskInput, TaskObservation, ToolBatch, ToolFailedObservation, ToolRecord
from weavria.tools import ToolRegistry


@dataclass(frozen=True, slots=True)
class ProgressCheck:
    blocked: bool
    repeated: bool
    reason: Literal["novel_action", "repeated_action", "stagnant_action_family"]
    fingerprint_sha256: str
    canonical_tool_key_sha256: str
    action_family_sha256: str
    prior_family_attempts: int


class ProgressPolicy(Protocol):
    @property
    def version(self) -> str: ...

    def inspect(
        self,
        *,
        task: TaskInput,
        observation: TaskObservation | None,
        records: tuple[ToolRecord, ...],
        batch: ToolBatch,
        capability_epoch: int,
    ) -> ProgressCheck: ...

    def record_admission(self, check: ProgressCheck) -> None:
        """Commit one inspection only after its batch admission fact is accepted."""
        ...


class SemanticProgressPolicy:
    """Bound one action family's exploration while the effective task frontier is unchanged.

    ``inspect`` is intentionally side-effect free. Runtime admission and policy state form a
    two-phase protocol: an action is counted only after ``BatchAdmitted`` has been accepted.
    """

    version = "semantic-progress/3"

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        max_stagnant_batches_per_action_family: int = 3,
    ) -> None:
        if (
            type(max_stagnant_batches_per_action_family) is not int
            or max_stagnant_batches_per_action_family < 1
        ):
            raise ValueError("stagnant action-family limit must be a positive integer")
        self._versions = {item.name: item.version for item in tools.descriptors}
        self._max_stagnant_batches = max_stagnant_batches_per_action_family
        self._admitted: dict[tuple[str, str], set[str]] = {}

    def inspect(
        self,
        *,
        task: TaskInput,
        observation: TaskObservation | None,
        records: tuple[ToolRecord, ...],
        batch: ToolBatch,
        capability_epoch: int,
    ) -> ProgressCheck:
        payload = observation.payload if observation is not None else {}
        need_state = payload.get("need_state_sha256")
        effective_facts: object
        if _is_sha256(need_state):
            # A Task-aware Harness has already reduced accepted records into effective Need
            # state. Raw Artifact hashes must not make irrelevant or volatile outputs look like
            # progress. Runtime remains domain-independent and treats the digest as an opaque
            # host-prepared frontier component.
            effective_facts = {"need_state_sha256": need_state}
        else:
            # Standalone AgentLoop users have no semantic reducer. Preserve the conservative v2
            # fallback: new accepted evidence or a new failure class establishes a new frontier.
            effective_facts = {
                "evidence": sorted(
                    {item.artifact.sha256 for item in records if item.artifact is not None}
                ),
                "failures": sorted(
                    {
                        (
                            item.exchange.action.tool_name,
                            item.exchange.observation.error.category.value,
                            item.exchange.observation.error.retryable,
                        )
                        for item in records
                        if isinstance(item.exchange.observation, ToolFailedObservation)
                    }
                ),
            }
        fingerprint = _digest(
            {
                "task_revision": task.revision,
                "frame_semantics_sha256": _frame_semantics_sha256(observation),
                "effective_facts": effective_facts,
                "capability_epoch": capability_epoch,
            }
        )
        canonical = _digest(
            [
                {
                    "tool_name": call.tool_name,
                    "tool_version": self._versions.get(call.tool_name, "unknown"),
                    "arguments": call.arguments,
                }
                for call in batch.calls
            ]
        )
        family = _digest(
            [
                {
                    "tool_name": call.tool_name,
                    "tool_version": self._versions.get(call.tool_name, "unknown"),
                }
                for call in batch.calls
            ]
        )
        admitted = self._admitted.get((fingerprint, family), set())
        repeated = canonical in admitted
        stagnant = not repeated and len(admitted) >= self._max_stagnant_batches
        reason: Literal["novel_action", "repeated_action", "stagnant_action_family"] = (
            "repeated_action"
            if repeated
            else "stagnant_action_family"
            if stagnant
            else "novel_action"
        )
        return ProgressCheck(
            blocked=repeated or stagnant,
            repeated=repeated,
            reason=reason,
            fingerprint_sha256=fingerprint,
            canonical_tool_key_sha256=canonical,
            action_family_sha256=family,
            prior_family_attempts=len(admitted),
        )

    def record_admission(self, check: ProgressCheck) -> None:
        """Record exactly one accepted action against the frontier it actually consumed."""
        if check.blocked:
            raise ValueError("a blocked progress proposal cannot be admitted")
        key = (check.fingerprint_sha256, check.action_family_sha256)
        admitted = self._admitted.setdefault(key, set())
        if check.canonical_tool_key_sha256 in admitted:
            raise ValueError("progress admission was already recorded")
        if len(admitted) != check.prior_family_attempts:
            raise ValueError("progress admission does not match the inspected frontier")
        admitted.add(check.canonical_tool_key_sha256)


def _frame_semantics_sha256(observation: TaskObservation | None) -> str:
    if observation is None:
        return ""
    payload = observation.payload
    if observation.payload_schema in {
        "weavria.task-frame/2",
        "weavria.task-frame/3",
        "weavria.task-frame/4",
        "weavria.task-frame/5",
    }:
        payload = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "frame_id",
                "revision",
                "evidence_revision",
                "need_state_sha256",
                "capability_epoch",
            }
        }
    return _digest(
        {
            "payload_schema": observation.payload_schema,
            "producer_version": observation.producer_version,
            "payload": payload,
        }
    )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
