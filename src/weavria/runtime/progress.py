"""Semantic no-progress detection independent of domain-specific completion rules."""

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from weavria.core import TaskInput, TaskObservation, ToolBatch, ToolFailedObservation, ToolRecord
from weavria.tools import ToolRegistry


@dataclass(frozen=True, slots=True)
class ProgressCheck:
    repeated: bool
    fingerprint_sha256: str
    canonical_tool_key_sha256: str


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


class SemanticProgressPolicy:
    """Detect repeated canonical actions when effective facts and Need state are unchanged."""

    version = "semantic-progress/2"

    def __init__(self, tools: ToolRegistry) -> None:
        self._versions = {item.name: item.version for item in tools.descriptors}
        self._seen: set[tuple[str, str]] = set()

    def inspect(
        self,
        *,
        task: TaskInput,
        observation: TaskObservation | None,
        records: tuple[ToolRecord, ...],
        batch: ToolBatch,
        capability_epoch: int,
    ) -> ProgressCheck:
        evidence = sorted({item.artifact.sha256 for item in records if item.artifact is not None})
        failures = sorted(
            {
                (
                    item.exchange.action.tool_name,
                    item.exchange.observation.error.category.value,
                    item.exchange.observation.error.retryable,
                )
                for item in records
                if isinstance(item.exchange.observation, ToolFailedObservation)
            }
        )
        payload = observation.payload if observation is not None else {}
        need_state = payload.get("need_state_sha256")
        fingerprint = _digest(
            {
                "task_revision": task.revision,
                "frame_semantics_sha256": _frame_semantics_sha256(observation),
                "need_state_sha256": need_state if isinstance(need_state, str) else "",
                "evidence": evidence,
                "failures": failures,
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
        key = (fingerprint, canonical)
        repeated = key in self._seen
        self._seen.add(key)
        return ProgressCheck(repeated, fingerprint, canonical)


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
