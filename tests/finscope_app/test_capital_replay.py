import httpx
import pytest
from pydantic import JsonValue, ValidationError

from apps.finscope.capital_replay import summarize_replay
from apps.finscope.workspace import FinScopeResourceInput, FinScopeWorkspaceClient


async def test_replay_uses_only_readonly_route_and_preserves_coverage() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path.endswith("/40/capital-replay")
        assert request.url.params["refresh"] == "false"
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "sessions": [
                        {
                            "date": "2026-11-12",
                            "minutes": [
                                {
                                    "time": "09:31:00",
                                    "price": 50,
                                    "largeNet": 100,
                                    "superLargeNet": 20,
                                    "tradeAmount": 1000,
                                },
                                {
                                    "time": "09:32:00",
                                    "price": 51,
                                    "largeNet": -100,
                                    "superLargeNet": -20,
                                    "tradeAmount": 2000,
                                },
                            ],
                        }
                    ],
                    "warnings": [],
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = FinScopeWorkspaceClient("http://finscope", client=http)
        result = await client.read_resource(
            FinScopeResourceInput(
                kind="market_capital_replay",
                identifier="40",
                secondary_identifier="2026-11-12",
                from_time="09:31",
                to_time="09:33",
            )
        )
    assert isinstance(result.data, dict)
    assert result.data["largeNet"] is None
    assert result.data["amount"] == {"value": 3000, "covered": 2, "expected": 3, "complete": False}
    assert result.data["directionChanges"] == [
        {"time": "09:32:00", "largeNet": -120, "direction": "outflow"}
    ]


def test_followup_requires_continuous_prices_and_trading_date() -> None:
    rows: list[JsonValue] = [
        {"time": f"09:{m}:00", "price": m, "largeNet": 1, "superLargeNet": 1} for m in range(31, 38)
    ]
    data: JsonValue = {"sessions": [{"date": "2026-11-12", "minutes": rows}], "warnings": []}
    result = summarize_replay(data, "2026-11-12", "09:31", "09:32")
    assert isinstance(result, dict)
    assert isinstance(result["followups"], list)
    assert isinstance(result["followups"][0], dict)
    assert result["followups"][0]["returnPct"] == pytest.approx((37 / 32 - 1) * 100)
    rows.pop(3)
    gap = summarize_replay(data, "2026-11-12", "09:31", "09:32")
    assert isinstance(gap, dict)
    assert isinstance(gap["followups"], list)
    assert isinstance(gap["followups"][0], dict)
    assert gap["followups"][0]["returnPct"] is None
    absent = summarize_replay(data, "2026-11-11", "09:31", "09:32")
    assert isinstance(absent, dict) and absent["session"] is None


@pytest.mark.parametrize("start,end", [("12:00", "15:00"), ("10:00", "09:31"), ("09:99", "15:00")])
def test_replay_rejects_non_trading_ranges(start: str, end: str) -> None:
    with pytest.raises(ValidationError):
        FinScopeResourceInput(
            kind="market_capital_replay", identifier="40", from_time=start, to_time=end
        )


def test_key_turns_follow_price_legs_and_do_not_bridge_gaps() -> None:
    from apps.finscope.replay_turns import replay_turns

    rows: dict[int, dict[str, JsonValue]] = {
        571 + index: {"price": price, "tradeAmount": 10000, "largeNet": 1, "superLargeNet": 1}
        for index, price in enumerate([100, 100.3, 101, 100.4, 100])
    }
    turns = replay_turns(rows)
    assert len(turns) == 2
    assert isinstance(turns[0], dict) and turns[0]["label"] == "快速上涨"
    assert isinstance(turns[1], dict) and turns[1]["label"] == "持续走低"
    assert turns[0]["fromTime"] == "09:31" and turns[0]["toTime"] == "09:33"
    assert replay_turns({571: {"price": 100}, 573: {"price": 110}}) == []
    assert replay_turns({690: {"price": 100}, 781: {"price": 110}}) == []
