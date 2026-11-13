"""分钟证据的确定性区间计算; 缺失值保留为空, 午休不算交易分钟。"""

from pydantic import JsonValue, TypeAdapter

from .replay_turns import replay_turns


def trading_minute(value: str) -> int:
    hour, minute = map(int, value[:5].split(":"))
    result = hour * 60 + minute
    if minute > 59 or not (570 <= result <= 690 or 781 <= result <= 900):
        raise ValueError("time must be a trading minute (09:30-11:30 or 13:01-15:00)")
    return result


def summarize_replay(data: JsonValue, day: str | None, start: str, end: str) -> JsonValue:
    """只对已有、同日且连续的样本计算收益, 不用插值补齐证据。"""
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), list):
        raise ValueError("invalid capital replay envelope")
    sessions = TypeAdapter(list[dict[str, JsonValue]]).validate_python(data["sessions"])
    available = [s.get("date") for s in sessions]
    selected = next((s for s in sessions if day is None or s.get("date") == day), None)
    if not isinstance(selected, dict) or not isinstance(selected.get("minutes"), list):
        return {"availableDates": available, "session": None, "warnings": data.get("warnings", [])}
    lower, upper = trading_minute(start), trading_minute(end)
    minute_rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(selected["minutes"])
    rows: dict[int, dict[str, JsonValue]] = {
        trading_minute(str(p["time"])): p
        for p in minute_rows
        if isinstance(p.get("time"), str) and str(p["time"])[:5] not in ("13:00",)
    }
    slots = [m for m in range(lower, upper + 1) if m <= 690 or m >= 781]
    flow_slots = [m for m in slots if m != 570]

    def number(row: dict[str, JsonValue], key: str) -> float | None:
        v = row.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def total(key: str, expected: list[int]) -> dict[str, JsonValue]:
        values = [v for m in expected if (v := number(rows.get(m, {}), key)) is not None]
        return {
            "value": sum(values) if values else None,
            "covered": len(values),
            "expected": len(expected),
            "complete": len(values) == len(expected),
        }

    large, super_large = total("largeNet", flow_slots), total("superLargeNet", flow_slots)
    complete = bool(large["complete"] and super_large["complete"] and flow_slots)
    net = (number(large, "value") or 0) + (number(super_large, "value") or 0) if complete else None
    # 转折只表示相邻分钟大额净流向变号, 不推断机构意图。
    turns: list[JsonValue] = []
    previous: tuple[int, float] | None = None
    for m in slots:
        row = rows.get(m, {})
        a, b = number(row, "largeNet"), number(row, "superLargeNet")
        value = a + b if a is not None and b is not None else None
        if value is not None and previous and m - previous[0] == 1 and value * previous[1] < 0:
            turns.append(
                {
                    "time": row["time"],
                    "largeNet": value,
                    "direction": "inflow" if value > 0 else "outflow",
                }
            )
        previous = (m, value) if value is not None else None
    trading = [m for m in range(570, 901) if m <= 690 or m >= 781]
    base = number(rows.get(upper, {}), "price")
    followups: list[JsonValue] = []
    for horizon in (5, 15, 30):
        index = trading.index(upper) + horizon
        target = trading[index] if index < len(trading) else None
        path = trading[trading.index(upper) : index + 1]
        covered = target is not None and all(
            number(rows.get(m, {}), "price") is not None for m in path
        )
        price = number(rows.get(target, {}), "price") if target else None
        followups.append(
            {
                "minutes": horizon,
                "complete": covered,
                "returnPct": (price / base - 1) * 100
                if covered and price is not None and base is not None and base > 0
                else None,
            }
        )
    return {
        "availableDates": available,
        "date": selected.get("date"),
        "priceProvider": selected.get("priceProvider"),
        "flowProvider": selected.get("flowProvider"),
        "fromTime": start,
        "toTime": end,
        "amount": total("tradeAmount", slots),
        "large": large,
        "superLarge": super_large,
        "largeNet": net,
        "startPrice": number(rows.get(lower, {}), "price"),
        "endPrice": base,
        "directionChanges": turns,
        "keyTurns": [
            event
            for event in replay_turns(rows)
            if isinstance(event, dict)
            and int(str(event["to"])) >= lower
            and int(str(event["from"])) <= upper
        ],
        "ruleVersion": "capital-replay-events-v1",
        "followups": followups,
        "minutes": [rows[m] for m in slots if m in rows],
        "warnings": data.get("warnings", []),
    }
