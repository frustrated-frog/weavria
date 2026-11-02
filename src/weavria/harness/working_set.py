"""按当前工作需求分配有限工具描述空间,不改变工具执行权限。"""

from dataclasses import dataclass
from typing import Protocol

from weavria.core import ToolFailedObservation, ToolRecord
from weavria.tools import ToolRegistry

from .capabilities import CapabilityQuery, CapabilityResolver
from .context import need_priority_weights
from .requirements import NeedLedger, NeedStatus
from .understanding import TaskFrame
from .work import WorkingState, work_need_weights


@dataclass(frozen=True)
class CapabilityWorkingSet:
    """一次纯选择结果。未覆盖目标表示展示缺口,并非任务已不可完成。"""

    names: tuple[str, ...]
    descriptor_bytes: int
    utility: float
    uncovered_targets: tuple[str, ...]


class CapabilityWorkingSetPolicy(Protocol):
    @property
    def version(self) -> str: ...

    def select(
        self,
        *,
        frame: TaskFrame,
        ledger: NeedLedger,
        tools: ToolRegistry,
        resolver: CapabilityResolver,
        previous: tuple[str, ...],
        records: tuple[ToolRecord, ...],
        discover: bool,
        working_state: WorkingState | None = None,
    ) -> CapabilityWorkingSet: ...


@dataclass(frozen=True)
class AdaptiveCapabilityPolicy:
    """双预算下的加权递减覆盖选择。所有分值都是启发式效用,不是概率。

    复用现有发现策略逐需求召回,避免全局 top-k 只覆盖一个热门需求。
    每轮从接受的 Frame/Ledger/Records 重算,不隐藏运行状态或永久封禁工具。
    """

    working_records_version = 1
    max_tools: int = 8
    max_descriptor_bytes: int = 16000

    def __post_init__(self) -> None:
        if type(self.max_tools) is not int or not 1 <= self.max_tools <= 64:
            raise ValueError("working set tool limit must be in [1, 64]")
        if (
            type(self.max_descriptor_bytes) is not int
            or not 256 <= self.max_descriptor_bytes <= 1048576
        ):
            raise ValueError("working set descriptor budget must be in [256, 1048576]")

    @property
    def version(self) -> str:
        return f"adaptive-capabilities/3:tools={self.max_tools}:bytes={self.max_descriptor_bytes}"

    def select(
        self,
        *,
        frame: TaskFrame,
        ledger: NeedLedger,
        tools: ToolRegistry,
        resolver: CapabilityResolver,
        previous: tuple[str, ...],
        records: tuple[ToolRecord, ...],
        discover: bool,
        working_state: WorkingState | None = None,
    ) -> CapabilityWorkingSet:
        if frame.work_id != ledger.work_id or frame.revision != ledger.frame_revision:
            raise ValueError("working set requires the current task frame and ledger")
        if (
            working_state is not None
            and working_state.task_revision != frame.request_anchor.task_revision
        ):
            raise ValueError("working preference belongs to another task revision")
        descriptors = {d.name: d for d in tools.descriptors}
        declared = {n.name for n in frame.specification.capability_needs}
        candidates = set(previous) | declared
        priorities = need_priority_weights(frame.requirements, ledger)
        for need_id, boost in work_need_weights(working_state, ledger).items():
            if need_id in priorities:
                priorities[need_id] = min(8.0, priorities[need_id] * boost)
        # 一次索引全部执行记录,避免每个需求-候选组合重扫长任务历史。
        positions = {r.exchange.action.call_id: i for i, r in enumerate(records)}
        last_success: dict[tuple[str, str], int] = {}
        for probe in ledger.probes:
            index = positions.get(probe.producer_call_id)
            if (
                index is None
                or probe.status not in {"matched", "partial"}
                or probe.freshness == "stale"
            ):
                continue
            record = records[index]
            if record.exchange.action.tool_name != probe.tool_name or isinstance(
                record.exchange.observation, ToolFailedObservation
            ):
                continue
            # 同一个工具在别的目标成功,不能清除当前需求的失败折扣。
            key = (probe.need_id, probe.tool_name)
            last_success[key] = max(index, last_success.get(key, -1))
        weights: dict[str, float] = {}
        coverage: dict[str, dict[str, float]] = {}
        for need in frame.requirements:
            state = ledger.state_for(need.need_id)
            if state.status is NeedStatus.SATISFIED or need.source != "external":
                continue
            target = f"need:{need.need_id}"
            weights[target] = priorities[need.need_id]
            matches = {name: 1.0 for name in need.capability_names if name in candidates}
            penalties: dict[str, float] = {}
            for failure in state.failures:
                if positions.get(failure.call_id, -1) > last_success.get(
                    (need.need_id, failure.tool_name), -1
                ):
                    penalties[failure.tool_name] = penalties.get(failure.tool_name, 0) + (
                        0.25 if failure.retryable else 1.0
                    )
            if discover:
                # 已覆盖字段不应继续主导候选发现。只改查询副本,不修改权威需求。
                missing = tuple(field for field in state.missing_fields if field.startswith("/"))
                query_need = (
                    need.model_copy(update={"required_fields": missing}) if missing else need
                )
                _, choices = resolver.resolve(
                    CapabilityQuery(
                        goal=need.description, requirements=(query_need,), allow_side_effects=False
                    ),
                    tools,
                    epoch=frame.capability_epoch,
                    trigger="partial_need",
                )
                for choice in choices:
                    if choice.decision == "select":
                        candidates.add(choice.name)
                        # 声明映射比词法/Schema 猜测更强,但均不替代实际输出核验。
                        matches[choice.name] = (
                            1.0
                            if choice.name in need.capability_names
                            else (0.6 if need.need_id in choice.intended_need_ids else 0.2)
                        )
            for name, quality in matches.items():
                coverage.setdefault(name, {})[target] = quality / (1.0 + penalties.get(name, 0))

        # 有些动作没有信息需求,但有独立完成义务。它们不能被空 Need 集误删。
        for obligation in frame.obligations:
            name = obligation.capability_name
            if not obligation.required or name is None or obligation.need_ids:
                continue
            if any(
                r.exchange.action.tool_name == name
                and not isinstance(r.exchange.observation, ToolFailedObservation)
                for r in records
            ):
                continue
            target = f"obligation:{obligation.obligation_id}"
            weights[target] = 2.0
            if name in declared:
                candidates.add(name)
                coverage.setdefault(name, {})[target] = 1.0

        costs = {
            name: len(descriptors[name].model_dump_json().encode())
            for name in candidates
            if name in descriptors
            and name in coverage
            and (descriptors[name].behavior == "read_only" or name in declared)
        }
        remaining = {name for name, cost in costs.items() if cost <= self.max_descriptor_bytes}
        chosen: list[str] = []
        residual = dict.fromkeys(weights, 1.0)
        spent = 0

        def gain(name: str, state: dict[str, float]) -> float:
            return sum(weights[t] * state[t] * q for t, q in coverage[name].items())

        # 覆盖函数 F(S)=sum(w * (1-product(1-q))) 是单调次模函数。
        # 每加入一个工具,只递减它覆盖目标的剩余收益,让不同需求有机会进入。
        while remaining and len(chosen) < self.max_tools:
            feasible = [
                name for name in remaining if spent + costs[name] <= self.max_descriptor_bytes
            ]
            if not feasible:
                break
            name = min(
                feasible,
                key=lambda n: (
                    -(gain(n, residual) / costs[n]) * (1.05 if n in previous else 1.0),
                    n not in previous,
                    n,
                ),
            )
            if gain(name, residual) <= 1e-12:
                break
            chosen.append(name)
            spent += costs[name]
            remaining.remove(name)
            for target, quality in coverage[name].items():
                residual[target] *= 1 - quality

        utility = sum(w * (1 - residual[t]) for t, w in weights.items())
        # 密度贪心可能被小工具占满预算而错过一个高价值大工具,保留单例比较。
        feasible = [n for n, cost in costs.items() if cost <= self.max_descriptor_bytes]
        if feasible:
            best = min(feasible, key=lambda n: (-gain(n, dict.fromkeys(weights, 1.0)), n))
            best_utility = gain(best, dict.fromkeys(weights, 1.0))
            if best_utility > utility + 1e-12:
                chosen, spent, utility = [best], costs[best], best_utility
        # 顺序是视图身份的一部分,规范化集合顺序避免仅排序变化推进 epoch。
        names = tuple(sorted(chosen))
        return CapabilityWorkingSet(
            names,
            spent,
            utility,
            tuple(
                sorted(
                    target
                    for target in weights
                    if not any(coverage[name].get(target, 0) > 0 for name in names)
                )
            ),
        )
