"""Exercise the real model adapter/evaluation wiring using a local HTTP fixture."""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from examples import live_memory_formation as experiment


def test_frozen_pilot_and_exact_metrics() -> None:
    path = Path(__file__).parents[1] / "examples/data/memory_formation/cases.json"
    raw = path.read_bytes()
    dataset = experiment.Dataset.model_validate_json(raw)
    assert len(dataset.cases) == 12
    assert len({case.id for case in dataset.cases}) == 12
    assert dataset.dataset_version == "memory-formation-pilot/1"
    check = experiment.Check(day=3, values=("Chinese",), qualified=0)
    assert experiment.metrics(check, ["中文"], 0)["passed"] is True
    assert experiment.metrics(check, ["中文", "中文"], 0)["passed"] is False
    assert experiment.metrics(check, ["中文或英文"], 0)["passed"] is False
    with pytest.raises(ValidationError):
        experiment.Dataset.model_validate_json(
            raw.replace(b'"schema_version": 1', b'"schema_version": 2')
        )


async def test_live_runner_shares_extraction_and_never_sends_expected_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": "http-contract/1",
                "cases": [
                    {
                        "id": "01",
                        "inputs": ["本项目从今天起用中文写技术文档。"],
                        "checks": [{"day": 3, "values": ["Chinese"], "qualified": 0}],
                    }
                ],
            }
        )
    )
    calls: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, Any], json.loads(request.content))
        calls.append(body)
        messages = body["messages"]
        payload = json.loads(json.loads(messages[1]["content"])["task"]["goal"])
        assert "checks" not in payload and "expected" not in payload
        source = payload["input"]
        if "proposals" not in payload:
            name = "propose_memory"
            output = {
                "candidates": [
                    {
                        "subject": "project",
                        "predicate": "documentation_language",
                        "value": "中文",
                        "text": source["documents"][-1]["text"],
                        "scope_kind": "project",
                        "retention": "durable",
                        "time_kind": "current",
                        "citations": [{"source_id": source["primary_source_id"], "quote": "中文"}],
                    }
                ]
            }
        else:
            name = "review_memory"
            output = {
                "judgments": [
                    {
                        "candidate_index": 0,
                        "supported": True,
                        "scope_supported": True,
                        "conditions_preserved": True,
                        "time_supported": True,
                        "relation": "new",
                    }
                ]
            }
        content = json.dumps(
            {
                "action": {
                    "kind": "tool_call",
                    "call_id": "ignored",
                    "tool_name": name,
                    "arguments": output,
                }
            }
        )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        )

    clients: list[httpx.AsyncClient] = []

    class LocalTransport(experiment.Transport):
        def __init__(self, base: str, key: str, timeout: float) -> None:
            self.url = "https://fixture.invalid/chat/completions"
            self.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            self.calls = 0
            clients.append(self.client)

    monkeypatch.setattr(experiment, "Transport", LocalTransport)
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "https://fixture.invalid")
    monkeypatch.setenv("WEAVRIA_MODEL_NAME", "fixture-model")
    monkeypatch.setenv("WEAVRIA_MODEL_API_KEY", "never-report-this")
    result = await experiment.evaluate(1, dataset_path=path)
    encoded = json.dumps(result)
    assert "never-report-this" not in encoded
    assert len(calls) == 2 and all(client.is_closed for client in clients)
    assert result["dataset_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    rows = cast(list[dict[str, Any]], result["rows"])
    assert rows[0]["complete_trial"] is True
    assert rows[0]["extraction_calls_shared"] == rows[0]["review_calls_extra"] == 1
    assert rows[0]["checks"][0]["latest_value"]["passed"] is True
    assert rows[0]["checks"][0]["reviewed"]["passed"] is True
