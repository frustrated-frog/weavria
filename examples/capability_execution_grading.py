"""执行后的独立判定:标准答案和证据路径从不传给模型或工具实现。"""

import json
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from examples.capability_execution_environment import Data


class EvidenceProbe(Data):
    name: str
    arguments: dict[str, JsonValue]


class ExecutionCase(Data):
    id: str
    query: str
    expected: dict[str, JsonValue]
    # 每条路径是一个足够的证据组合;多条路径允许等价算法/工具组合。
    evidence_paths: tuple[tuple[EvidenceProbe, ...], ...] = ((),)
    ordered: bool = False
    unavailable: bool = False


class ExecutionDataset(Data):
    schema_version: Literal[1]
    version: str
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[ExecutionCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("case identities must be unique")
        for case in self.cases:
            if not case.evidence_paths:
                raise ValueError("every case needs at least one evidence path")
            if case.unavailable and (
                case.expected != {"status": "unavailable"} or case.evidence_paths != ((),)
            ):
                raise ValueError("unavailable cases require an explicit abstention and no tools")
        return self


@dataclass(frozen=True)
class AcceptedCall:
    """仅从同一 Run 的持久化事实投影;不接受模型自报调用成功。"""

    name: str
    arguments: dict[str, JsonValue]
    succeeded: bool
    started_sequence: int
    recorded_sequence: int
    artifact_sha256: str | None = None


@dataclass(frozen=True)
class Grade:
    answer_correct: bool
    evidence_satisfied: bool
    task_success: bool
    safe_stop: bool
    expected_behavior: bool
    unrelated_calls: int


def canonical(value: JsonValue) -> str:
    # JSON 类型敏感:Python 的 True == 1 不能让错误布尔值通过标准答案。
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys are ambiguous")
        result[key] = value
    return result


def _matches(probe: EvidenceProbe, call: AcceptedCall) -> bool:
    return (
        call.succeeded
        and call.artifact_sha256 is not None
        and 0 < call.started_sequence < call.recorded_sequence
        and probe.name == call.name
        and all(
            key in call.arguments and canonical(value) == canonical(call.arguments[key])
            for key, value in probe.arguments.items()
        )
    )


def _path_satisfied(
    probes: tuple[EvidenceProbe, ...], calls: tuple[AcceptedCall, ...], ordered: bool
) -> bool:
    # 每一步消费不同记录;先查位置再读文件要求前次结果已接受才开始下一次调用。
    # 单纯比较 ToolRecorded 次序会误接受同批并发中预猜出的路径。
    def visit(index: int, used: frozenset[int], after: int) -> bool:
        if index == len(probes):
            return True
        return any(
            visit(index + 1, used | {position}, call.recorded_sequence)
            for position, call in enumerate(calls)
            if position not in used
            and (not ordered or call.started_sequence > after)
            and _matches(probes[index], call)
        )

    return visit(0, frozenset(), -1)


def grade(
    case: ExecutionCase,
    *,
    output: str | None,
    completed: bool,
    calls: tuple[AcceptedCall, ...],
    discovery_exhausted: bool = False,
) -> Grade:
    """明确停止和真正完成分开报告;异常、空输出或自报 completed 都不是答案。"""
    answer = False
    try:
        # 只放宽首尾空白;请求已要求 JSON,不从自然语言中猜测用户想交付的部分。
        parsed: JsonValue = (
            json.loads(output, object_pairs_hook=_unique_object) if output is not None else None
        )
        answer = canonical(parsed) == canonical(case.expected)
    except (ValueError, TypeError, RecursionError):
        pass
    evidence = any(_path_satisfied(path, calls, case.ordered) for path in case.evidence_paths)
    relevant = {probe.name for path in case.evidence_paths for probe in path}
    auxiliaries = {
        "read_artifact",
        "read_task_source",
        "read_session_source",
        "read_session_history",
    }
    unrelated = sum(call.name not in relevant | auxiliaries for call in calls)
    # 缺失能力和无需工具任务要求没有任何工具动作。检索停止属于有效保留,
    # 但没有向用户交付答案,不能纳入 task_success。
    no_tools = case.evidence_paths == ((),)
    if no_tools:
        evidence = not calls
    success = completed and answer and evidence
    stopped = case.unavailable and not calls and (discovery_exhausted or success)
    return Grade(answer, evidence, success, stopped, success or stopped, unrelated)
