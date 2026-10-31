"""Exercise actual Session/provider wiring without a network or paid model."""

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from examples import live_memory_recall as experiment
from weavria.models import OpenAICompatibleTokenCounter


def test_scoring_requires_exact_answer_and_complete_citations() -> None:
    data = experiment.Dataset.model_validate_json(
        (Path(__file__).parents[1] / "examples/data/memory_recall/cases.json").read_bytes()
    )
    assert len(data.cases) == 10
    case = data.cases[0]
    assert experiment.score('{"answer":"English","sources":["exception"]}', case)["passed"]
    assert not experiment.score('{"answer":"English","sources":[]}', case)["passed"]
    assert not experiment.score('{"answer":"English or Chinese","sources":["exception"]}', case)[
        "passed"
    ]
    assert not experiment.score('{"answer":"English","sources":["exception","exception"]}', case)[
        "passed"
    ]


async def test_frozen_answers_never_enter_model_input_and_all_arms_use_session_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_counter: OpenAICompatibleTokenCounter
) -> None:
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": "http/1",
                "cases": [
                    {
                        "id": "01",
                        "question": "document language",
                        "answer": "evaluator-only-sentinel",
                        "sources": ["one"],
                        "claims": [
                            {
                                "id": "one",
                                "predicate": "language",
                                "value": "Chinese",
                                "text": "document language Chinese",
                            }
                        ],
                    }
                ],
            }
        )
    )
    calls: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "evaluator-only-sentinel" not in request.content.decode()
        observations = json.loads(body["messages"][1]["content"])
        calls.append(observations)
        assert not observations["conversation"]["turns"]
        return httpx.Response(
            200,
            json={
                "model": "fixture",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": {
                                        "kind": "finish",
                                        "output": '{"answer":"Chinese","sources":["one"]}',
                                    }
                                }
                            )
                        },
                    }
                ],
            },
        )

    real_client = httpx.AsyncClient

    def client(**kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(respond))

    def counter(*args: object, **kwargs: object) -> OpenAICompatibleTokenCounter:
        return session_counter

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(experiment, "OpenAICompatibleTokenCounter", counter)
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "https://fixture.invalid")
    monkeypatch.setenv("WEAVRIA_MODEL_NAME", "fixture")
    monkeypatch.delenv("WEAVRIA_MODEL_API_KEY", raising=False)
    report = await experiment.evaluate(path)
    assert len(calls) == 3
    assert sorted(len(call.get("bundles", [])) for call in calls) == [0, 1, 1]
    rows = report["rows"]
    assert isinstance(rows, list) and len(cast(list[object], rows)) == 3
