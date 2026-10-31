"""实验协议隔离标签、复用提取、关闭资源,Fake 结果不当成自然语义能力证据。"""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from examples import live_memory_conditions as experiment
from weavria.core import ModelResponse, ToolCallAction
from weavria.models import FakeModelProvider


@pytest.mark.parametrize("failure", [False, True])
async def test_frozen_extraction_two_strategies_and_failure_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    case = {
        "id": "fixture",
        "first": "初始长期设置是 old。",
        "second": "修正该设置为 new。",
        "relationship": "same",
        "old_value": "old",
        "new_value": "new",
    }
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": "fixture/1",
                "split": "development",
                "cases": [case],
            }
        )
    )
    actions: list[ModelResponse] = []
    for step in range(6 if failure else 5):
        if step in (0, 2):
            arguments: dict[str, Any] = {
                "claims": [
                    {
                        "subject": "project",
                        "predicate": "setting",
                        "value": "old" if step == 0 else "new",
                        "qualifiers": {"q0": "first"} if step == 0 else {"q1": "second"},
                        "text": case["first" if step == 0 else "second"],
                        "scope_kind": "project",
                        "retention": "durable",
                        "time_kind": "unknown",
                        "modality": "asserted",
                        "evidence": [0 if step == 0 else 1],
                    }
                ]
            }
            tool = "propose_memory"
        else:
            tool = "review_memory"
            decision: dict[str, Any] = {
                "candidate": 0,
                "verdict": "supported",
                "relation": "new" if step == 1 else "correction",
            }
            if step != 1:
                decision.update(
                    anchor=0,
                    mapping=[["/subject", "/subject"], ["/predicate", "/predicate"]],
                    citations=[0],
                    resolves=[0],
                )
                if step == 3:
                    decision["mapping"].append(["/qualifiers/q1", "/qualifiers/q0"])
                elif not failure:
                    decision["condition_links"] = [
                        ["candidate", "q1", "target", "q0", "equivalent"]
                    ]
            arguments = {"decisions": [decision]}
            if failure and step >= 4:
                arguments = {"decisions": "bad"}
        actions.append(
            ModelResponse(
                action=ToolCallAction(call_id=f"call-{step}", tool_name=tool, arguments=arguments)
            )
        )
    provider = FakeModelProvider(tuple(actions))

    def build_provider(*args: object, **kwargs: object) -> FakeModelProvider:
        return provider

    transports: list[experiment.Transport] = []
    base_transport = experiment.Transport

    def transport(*args: Any, **kwargs: Any) -> experiment.Transport:
        instance = base_transport(*args, **kwargs)
        transports.append(instance)
        return instance

    monkeypatch.setattr(experiment, "OpenAICompatibleProvider", build_provider)
    monkeypatch.setattr(experiment, "Transport", transport)
    monkeypatch.setenv("WEAVRIA_MODEL_NAME", "fixture")
    monkeypatch.setenv("WEAVRIA_MODEL_BASE_URL", "https://fixture.invalid")
    monkeypatch.setenv("WEAVRIA_MODEL_API_KEY", "do-not-record")
    output = tmp_path / "report.jsonl"
    result = await experiment.evaluate(dataset, output)
    rows = cast(list[dict[str, Any]], result["rows"])
    assert len(rows) == 3 and rows[1]["score"] == {"passed": False}
    assert rows[2]["score"] == {"passed": not failure}
    if failure:
        assert rows[2]["error"] == "MemoryFormationError"
        assert rows[2]["pending"]["extraction"] == rows[0]["extraction"]
    else:
        assert rows[2]["job"]["plan"]["review"]["schema_version"] == 6
        assert rows[1]["job"]["extraction"] == rows[2]["job"]["extraction"]
    assert len(provider.requests) == len(actions)
    for request in provider.requests:
        data = json.loads(request.task.goal)
        assert "relationship" not in data and "old_value" not in data and "new_value" not in data
    assert all(t.client.is_closed for t in transports)
    assert "do-not-record" not in output.read_text()
    with pytest.raises(FileExistsError):
        await experiment.evaluate(dataset, output)
    assert len(transports) == 1
