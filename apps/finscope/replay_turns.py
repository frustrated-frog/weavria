"""与 FinScope capitalReplayEvents.ts 一致的复盘转折口径, 不产生交易建议."""

import math

from pydantic import JsonValue


def replay_turns(rows: dict[int, dict[str, JsonValue]]) -> list[JsonValue]:
    def value(minute: int, key: str) -> float | None:
        v = rows[minute].get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            return float(v)
        return None

    def price(minute: int) -> float:
        p = value(minute, "price")
        assert p is not None and p > 0
        return p

    def net(minute: int) -> float | None:
        a, b = value(minute, "largeNet"), value(minute, "superLargeNet")
        return a + b if a is not None and b is not None else None

    def total(sample: list[int], field: str) -> float | None:
        values = [net(m) if field == "net" else value(m, field) for m in sample]
        return (
            sum(v for v in values if v is not None)
            if values and all(v is not None for v in values)
            else None
        )

    runs: list[list[int]] = []
    for m in sorted(rows):
        p = value(m, "price")
        if p is None or p <= 0:
            continue
        if not runs or m != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(m)
    events: list[dict[str, JsonValue]] = []
    candidates: list[dict[str, JsonValue]] = []
    magnitudes: list[float] = []

    def describe(
        run: list[int], start: int, end: int, flow_only: bool = False
    ) -> dict[str, JsonValue]:
        sample = [m for m in run if start <= m <= end]
        flow_sample = [m for m in sample if m != 570]
        amount, flow = total(sample, "tradeAmount"), total(flow_sample, "net")
        if flow_only and len(flow_sample) != len(sample):
            flow = None
        prior = [m for m in run if m < start][-len(sample) :]
        baseline = total(prior, "tradeAmount") if len(prior) == len(sample) else None
        ratio = (
            amount / baseline
            if amount is not None and baseline is not None and baseline > 0
            else None
        )
        change = price(end) / price(start) - 1
        up = change > 0
        if flow_only:
            label = "大额净流入集中" if flow is not None and flow > 0 else "大额净流出集中"
        elif (
            flow is not None
            and amount is not None
            and amount > 0
            and abs(flow) / amount >= 0.15
            and (flow > 0) != up
        ):
            label = "上涨伴大额净流出" if up else "下跌伴大额净流入"
        elif ratio is not None and ratio >= 1.5:
            label = "放量上涨" if up else "放量下跌"
        elif end - start <= 10 and abs(change) >= 0.01 - 1e-10:
            label = "快速上涨" if up else "快速下跌"
        else:
            label = "持续走高" if up else "持续走低"
        return {
            "fromTime": f"{start // 60:02}:{start % 60:02}",
            "toTime": f"{end // 60:02}:{end % 60:02}",
            "label": label,
            "direction": "flat" if flow_only else "up" if up else "down",
            "changePct": change * 100,
            "amount": amount,
            "largeNet": flow,
            "amountRatio": ratio,
            "from": start,
            "to": end,
            "flowCount": sum(net(m) is not None for m in flow_sample),
            "minuteCount": len(flow_sample),
        }

    for run in runs:
        anchor = extreme = low = high = run[0]
        direction = 0
        for m in run[1:]:
            if direction == 0:
                if price(m) <= price(low):
                    low = m
                if price(m) >= price(high):
                    high = m
                if price(m) / price(low) - 1 >= 0.006 - 1e-10:
                    anchor, direction, extreme = low, 1, m
                elif price(m) / price(high) - 1 <= -0.006 + 1e-10:
                    anchor, direction, extreme = high, -1, m
                continue
            if direction * (price(m) - price(extreme)) > 0:
                extreme = m
            elif direction * (price(m) / price(extreme) - 1) <= -0.004 + 1e-10:
                if abs(price(extreme) / price(anchor) - 1) >= 0.006 - 1e-10:
                    events.append(describe(run, anchor, extreme))
                anchor, extreme, direction = extreme, m, -direction
        if abs(price(extreme) / price(anchor) - 1) >= 0.006 - 1e-10:
            events.append(describe(run, anchor, extreme))
        for i in range(len(run) - 4):
            event = describe(run, run[i], run[i + 4], True)
            flow, amount = event["largeNet"], event["amount"]
            if isinstance(flow, (int, float)) and isinstance(amount, (int, float)) and amount > 0:
                magnitudes.append(abs(flow))
                if abs(flow) / amount >= 0.15:
                    candidates.append(event)
    magnitudes.sort()
    threshold = magnitudes[len(magnitudes) // 2] * 1.5 if magnitudes else 0
    candidates.sort(key=lambda e: abs(float(str(e["largeNet"]))), reverse=True)
    for event in candidates:
        if abs(float(str(event["largeNet"]))) <= threshold or any(
            int(str(event["from"])) <= int(str(other["to"]))
            and int(str(event["to"])) >= int(str(other["from"]))
            for other in events
        ):
            continue
        events.append(event)
    events.sort(key=lambda e: int(str(e["from"])))
    return list(events)
