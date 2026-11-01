"""形成搜索与独立消费探针的接线;不以 Fake 输出证明实际任务收益。"""

import json
from pathlib import Path
from typing import Any

import pytest

from examples import live_memory_interpretation as experiment
from weavria.core import ModelResponse, ToolCallAction
from weavria.models import FakeModelProvider


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("captured", [False, True])
async def test_probe_has_no_source_or_expected_answer_and_failure_stays_in_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool, captured: bool
) -> None:
    source = "项目受限资料使用格式A。"
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "split": "validation",
                "cases": [
                    {
                        "id": "fixture",
                        "input": source,
                        "probes": [{"question": "受限资料的格式?", "answer": "A"}],
                    }
                ],
            }
        )
    )
    claim: dict[str, Any] = {
        "subject": "project",
        "predicate": "format",
        "value": "A",
        "text": source,
        "scope_kind": "project",
        "retention": "durable",
        "time_kind": "unknown",
        "modality": "asserted",
        "evidence": [0],
    }
    fixed = dict(claim, qualifiers={"access": "受限"}, evidence=[99] if failure else [0])
    if captured:
        data = json.loads(dataset.read_text())
        data["cases"][0]["initial"] = {
            "schema_version": 1,
            "candidates": [
                {
                    **{k: v for k, v in claim.items() if k not in {"modality", "evidence"}},
                    "citations": [{"source_id": "input:1", "quote": source, "start": 0}],
                }
            ],
        }
        dataset.write_text(json.dumps(data))

    def q(grade: int) -> dict[str, object]:
        return {
            "assessments": [
                {"grade": grade, "defect": "none" if grade == 3 else "missing_condition"}
            ],
            "unsupported": 0,
        }

    actions: list[tuple[str, dict[str, Any]]] = [
        ("propose_memory", {"claims": [claim]}),
        (
            "analyze_meanings",
            {"units": [{"meaning": "受限资料有特定格式", "evidence": [0]}]},
        ),
        ("rebuild_meanings", {"alternatives": [{"claims": [fixed]}]}),
    ]
    if captured:
        actions.pop(0)
    if not failure:
        actions.append(("compare_meanings", {"qualities": [q(2), q(3)]}))
    actions.append(("answer_probe", {"answer": "unknown"}))
    if not failure:
        actions.append(("answer_probe", {"answer": "A"}))
    provider = FakeModelProvider(
        tuple(
            ModelResponse(action=ToolCallAction(call_id=str(i), tool_name=name, arguments=args))
            for i, (name, args) in enumerate(actions)
        )
    )

    def build_provider(*args: object, **kwargs: object) -> FakeModelProvider:
        return provider

    transports: list[experiment.Transport] = []
    transport_class = experiment.Transport

    def transport(*args: Any, **kwargs: Any) -> experiment.Transport:
        result = transport_class(*args, **kwargs)
        transports.append(result)
        return result

    monkeypatch.setattr(experiment, "OpenAICompatibleProvider", build_provider)
    monkeypatch.setattr(experiment, "Transport", transport)
    monkeypatch.setenv("WEAVRIA_MODEL_NAME", "fixture")
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "https://fixture.invalid")
    monkeypatch.setenv("WEAVRIA_MODEL_API_KEY", "test-secret-never-save")
    output = tmp_path / "report.jsonl"
    await experiment.evaluate(dataset, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    probes = [r for r in rows if r.get("stage") == "probe"]
    assert len(probes) == 2 and probes[0]["passed"] is False
    assert probes[1]["passed"] == (not failure)
    if failure:
        assert probes[1]["error"] == "extraction_unavailable"
    for request in provider.requests:
        payload = json.loads(request.task.goal)
        assert "answer" not in payload and "probes" not in payload
        if "facts" in payload:
            assert "input" not in payload and all(
                "text" not in f and "citations" not in f for f in payload["facts"]
            )
    assert all(t.client.is_closed for t in transports)
    assert "test-secret-never-save" not in output.read_text()
