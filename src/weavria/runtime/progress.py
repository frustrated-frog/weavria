"""Semantic-frontier progress control independent of domain completion rules."""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Protocol

from weavria.core import TaskInput, TaskObservation, ToolBatch, ToolFailedObservation, ToolRecord
from weavria.tools import ToolActionIdentity, ToolRegistry


@dataclass(frozen=True, slots=True)
class ProgressReservation:
    """Unique actions reserved atomically for one frontier/family pair."""

    action_family_sha256: str
    canonical_action_sha256s: tuple[str, ...]
    prior_family_attempts: int

    def __post_init__(self) -> None:
        if not _is_sha256(self.action_family_sha256):
            raise ValueError("progress reservation family must be a SHA-256 digest")
        if not self.canonical_action_sha256s:
            raise ValueError("progress reservation actions must be nonempty")
        if any(not _is_sha256(value) for value in self.canonical_action_sha256s):
            raise ValueError("progress reservation actions must be SHA-256 digests")
        if len(set(self.canonical_action_sha256s)) != len(self.canonical_action_sha256s):
            raise ValueError("progress reservation actions must be unique")
        if type(self.prior_family_attempts) is not int or self.prior_family_attempts < 0:
            raise ValueError("prior family attempts must be nonnegative")


@dataclass(frozen=True, slots=True)
class ProgressCheck:
    blocked: bool
    repeated: bool
    reason: Literal["novel_action", "repeated_action", "stagnant_action_family"]
    fingerprint_sha256: str
    canonical_tool_key_sha256: str
    action_family_sha256: str
    prior_family_attempts: int
    reservations: tuple[ProgressReservation, ...]
    action_identities: tuple[ToolActionIdentity, ...]


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


class ProgressPolicyFactory(Protocol):
    """Create isolated mutable policy state for one Run."""

    @property
    def version(self) -> str: ...

    def create(self, tools: ToolRegistry) -> ProgressPolicy: ...


@dataclass(frozen=True, slots=True)
class SemanticProgressPolicyFactory:
    """Versioned construction boundary for SessionEngine and controlled experiments."""

    max_stagnant_actions_per_family: int = 3

    def __post_init__(self) -> None:
        if (
            type(self.max_stagnant_actions_per_family) is not int
            or self.max_stagnant_actions_per_family < 1
        ):
            raise ValueError("stagnant action-family limit must be a positive integer")

    @property
    def version(self) -> str:
        return (
            "semantic-progress-factory/3["
            f"policy={SemanticProgressPolicy.version};"
            f"stagnation_threshold={self.max_stagnant_actions_per_family}]"
        )

    def create(self, tools: ToolRegistry) -> ProgressPolicy:
        return SemanticProgressPolicy(
            tools,
            max_stagnant_actions_per_family=self.max_stagnant_actions_per_family,
        )


class SemanticProgressPolicy:
    """Bound one action family's exploration while the effective task frontier is unchanged.

    ``inspect`` is intentionally side-effect free. Runtime admission and policy state form a
    two-phase protocol: an action is counted only after ``BatchAdmitted`` has been accepted.
    """

    version = "semantic-progress/5"

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        max_stagnant_actions_per_family: int = 3,
    ) -> None:
        if type(max_stagnant_actions_per_family) is not int or max_stagnant_actions_per_family < 1:
            raise ValueError("stagnant action-family limit must be a positive integer")
        self._tools = tools
        self._max_stagnant_actions = max_stagnant_actions_per_family
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
        action_identities = tuple(self._tools.action_identity(call) for call in batch.calls)
        actions = tuple(
            (
                identity.canonical_action_sha256,
                identity.action_family_sha256,
            )
            for identity in action_identities
        )
        canonical = _digest([action for action, _ in actions])
        proposed: dict[str, list[str]] = {}
        blocked_reason: Literal["repeated_action", "stagnant_action_family"] | None = None
        blocked_action = actions[0][0]
        blocked_family = actions[0][1]
        blocked_prior = len(self._admitted.get((fingerprint, blocked_family), set()))
        for action, family in actions:
            admitted = self._admitted.get((fingerprint, family), set())
            family_proposal = proposed.setdefault(family, [])
            if action in admitted or action in family_proposal:
                blocked_reason = "repeated_action"
            elif len(admitted) >= self._max_stagnant_actions:
                blocked_reason = "stagnant_action_family"
            if blocked_reason is not None:
                blocked_action = action
                blocked_family = family
                blocked_prior = len(admitted)
                break
            family_proposal.append(action)
        reservations = (
            ()
            if blocked_reason is not None
            else tuple(
                ProgressReservation(
                    action_family_sha256=family,
                    canonical_action_sha256s=tuple(family_actions),
                    prior_family_attempts=len(self._admitted.get((fingerprint, family), set())),
                )
                for family, family_actions in proposed.items()
            )
        )
        first_reservation = reservations[0] if reservations else None
        family = (
            blocked_family
            if blocked_reason is not None
            else first_reservation.action_family_sha256
            if first_reservation is not None
            else actions[0][1]
        )
        prior = (
            blocked_prior
            if blocked_reason is not None
            else first_reservation.prior_family_attempts
            if first_reservation is not None
            else 0
        )
        repeated = blocked_reason == "repeated_action"
        reason: Literal["novel_action", "repeated_action", "stagnant_action_family"] = (
            blocked_reason if blocked_reason is not None else "novel_action"
        )
        return ProgressCheck(
            blocked=blocked_reason is not None,
            repeated=repeated,
            reason=reason,
            fingerprint_sha256=fingerprint,
            canonical_tool_key_sha256=(blocked_action if blocked_reason is not None else canonical),
            action_family_sha256=family,
            prior_family_attempts=prior,
            reservations=reservations,
            action_identities=action_identities,
        )

    def record_admission(self, check: ProgressCheck) -> None:
        """Atomically record every action in one accepted batch against its inspected frontier."""
        if check.blocked:
            raise ValueError("a blocked progress proposal cannot be admitted")
        if not check.reservations:
            raise ValueError("progress admission requires at least one family reservation")
        families = [item.action_family_sha256 for item in check.reservations]
        if len(families) != len(set(families)):
            raise ValueError("progress admission families must be unique")
        for reservation in check.reservations:
            admitted = self._admitted.get(
                (check.fingerprint_sha256, reservation.action_family_sha256), set()
            )
            if len(admitted) != reservation.prior_family_attempts:
                raise ValueError("progress admission does not match the inspected frontier")
            if admitted.intersection(reservation.canonical_action_sha256s):
                raise ValueError("progress admission was already recorded")
            if len(admitted) >= self._max_stagnant_actions:
                raise ValueError("progress admission starts from an exhausted action family")
        for reservation in check.reservations:
            admitted = self._admitted.setdefault(
                (check.fingerprint_sha256, reservation.action_family_sha256), set()
            )
            admitted.update(reservation.canonical_action_sha256s)


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
