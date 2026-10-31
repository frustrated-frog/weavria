"""HTTP 替身验证端到端连接与留证,不证明真实模型的语义判断能力。"""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from examples import live_memory_lifecycle as experiment
from weavria.models import OpenAICompatibleTokenCounter


def test_frozen_lifecycle_has_expiry_conflict_correction_and_retention() -> None:
    raw = (Path(__file__).parents[1] / "examples/data/memory_lifecycle/cases.json").read_bytes()
    dataset = experiment.Dataset.model_validate_json(raw)
    assert {c.id for c in dataset.cases} == {"exception", "expiry", "conflict", "retention"}
    assert sum(len(c.checks) for c in dataset.cases) == 12
    check = dataset.cases[0].checks[0]
    # 正确语言若引用未实际交付的来源仍失败,unknown 也不能冒充该任务的正确答案。
    assert not experiment.score('{"answer":"中文","sources":["invented"]}', check, None)["passed"]
    assert not experiment.score('{"answer":"unknown","sources":[]}', check, None)["passed"]


@pytest.mark.parametrize("family_alignment", [False, True])
@pytest.mark.parametrize("failure", [False, True])
async def test_actual_session_ingestion_reopen_consumption_and_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_counter: OpenAICompatibleTokenCounter,
    failure: bool,
    family_alignment: bool,
) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": "http-lifecycle/1",
                "cases": [
                    {
                        "id": "probe",
                        "inputs": ["本项目技术文档在2026年1月1日至1月10日使用中文。"],
                        "checks": [
                            {
                                "after": 1,
                                "day": 5,
                                "question": "技术文档的语言?",
                                "answer": "Chinese",
                            },
                            {
                                "after": 1,
                                "day": 25,
                                "question": "技术文档的语言?",
                                "answer": "unknown",
                            },
                        ],
                    }
                ],
            }
        )
    )
    calls: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, Any], json.loads(request.content))
        calls.append(body)
        observations = json.loads(body["messages"][1]["content"])
        assert "checks" not in observations and "expected" not in observations
        if "current" in observations:
            assert observations["conversation"]["turns"] == []
            bundles = observations.get("bundles", [])
            source_ids = []
            if bundles:
                member = bundles[0]["content"]["group"]["members"][0]["version"]["assertion"]
                source_ids = member["source_ids"]
                assert source_ids[0].startswith("session:prior:")
                assert member["authority"] == "user_statement"
            result: object = {
                "decision": {
                    "kind": "finish",
                    "output": json.dumps(
                        {
                            "answer": "Chinese" if bundles else "unknown",
                            "sources": source_ids,
                        }
                    ),
                }
            }
        else:
            payload = json.loads(observations["task"]["goal"])
            assert set(payload) <= {"input", "proposals", "memory"}
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
                            "time_kind": "dated",
                            "valid_from": "2026-01-01T00:00:00Z",
                            "valid_to": "2026-01-10T00:00:00Z",
                            "citations": [
                                {"source_id": source["primary_source_id"], "quote": "中文"}
                            ],
                        }
                    ]
                }
            else:
                name = "review_memory"
                output = {
                    "schema_version": 3 if family_alignment else 2,
                    "judgments": [
                        {
                            "candidate_index": 0,
                            "supported": True,
                            "scope_supported": True,
                            "conditions_preserved": True,
                            "time_supported": True,
                            "relation": "new",
                        }
                    ],
                }
            result = {
                "action": {
                    "kind": "tool_call",
                    "call_id": "fixture",
                    "tool_name": name,
                    "arguments": {"candidates": "invalid"} if failure else output,
                }
            }
        return httpx.Response(
            200,
            json={
                "model": "fixture",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(result),
                            "reasoning_content": "private-reasoning",
                        },
                    }
                ],
            },
        )

    clients: list[httpx.AsyncClient] = []

    class Transport(experiment.Transport):
        def __init__(
            self, base: str, key: str, timeout: float, *, options: dict[str, object] | None = None
        ) -> None:
            self.url = "https://fixture.invalid/chat/completions"
            self.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            self.options = options or {}
            self.calls = 0
            self.observations: list[dict[str, object]] = []
            clients.append(self.client)

    monkeypatch.setattr(experiment, "Transport", Transport)

    def counter(*args: object, **kwargs: object) -> OpenAICompatibleTokenCounter:
        return session_counter

    monkeypatch.setattr(experiment, "OpenAICompatibleTokenCounter", counter)
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "https://fixture.invalid")
    monkeypatch.setenv("WEAVRIA_MODEL_NAME", "fixture")
    monkeypatch.setenv("WEAVRIA_MODEL_API_KEY", "never-retain-key")
    report = await experiment.evaluate(path, repeats=1, family_alignment=family_alignment)
    assert report["family_alignment"] is family_alignment
    encoded = json.dumps(report)
    assert "never-retain-key" not in encoded and "private-reasoning" not in encoded
    assert all(c.is_closed for c in clients)
    rows = cast(list[dict[str, Any]], report["rows"])
    formed = next(r for r in rows if r["kind"] == "formation")
    consumed = [r for r in rows if r["kind"] == "consumption"]
    assert len(consumed) == 6 and all(r["prior_history_empty"] for r in consumed)
    assert all(r["status"] == "completed" for r in consumed)
    assert formed["blocked"] == failure
    if failure:
        assert formed["error"] == "MemoryFormationError"
        assert formed["failure_audit"]["pending_job"]["completed_head"] is None
        assert formed["failure_audit"]["pending_job"]["extraction"] is None
        assert all(not r["pipeline_passed"] for r in consumed)
        assert not formed["snapshot"]["versions"]
        assert all(not r["formation_complete"] for r in consumed)
    else:
        # 两次形成调用 + 六次消费;完成作业的显式重放不额外调模型。
        assert len(calls) == 8 and len(formed["jobs"]) == 1
        assert sum(r["score"]["passed"] for r in consumed) == 5
        assert all(r["formation_complete"] for r in consumed)


async def test_evaluator_cancellation_closes_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    entered, cleaned = asyncio.Event(), asyncio.Event()
    clients: list[httpx.AsyncClient] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        finally:
            cleaned.set()

    class Transport(experiment.Transport):
        def __init__(
            self, base: str, key: str, timeout: float, *, options: dict[str, object] | None = None
        ) -> None:
            self.url = "https://fixture.invalid/chat/completions"
            self.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            self.options = options or {}
            self.calls = 0
            self.observations: list[dict[str, object]] = []
            clients.append(self.client)

    monkeypatch.setattr(experiment, "Transport", Transport)

    def counter(*args: object, **kwargs: object) -> OpenAICompatibleTokenCounter:
        return session_counter

    monkeypatch.setattr(experiment, "OpenAICompatibleTokenCounter", counter)
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "https://fixture.invalid")
    monkeypatch.setenv("WEAVRIA_MODEL_NAME", "fixture")
    dataset = Path(__file__).parents[1] / "examples/data/memory_lifecycle/cases.json"
    work = asyncio.create_task(experiment.evaluate(dataset, repeats=1))
    try:
        async with asyncio.timeout(3):
            await entered.wait()
            work.cancel()
            with pytest.raises(asyncio.CancelledError):
                await work
        assert cleaned.is_set() and all(c.is_closed for c in clients)
    finally:
        if not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
