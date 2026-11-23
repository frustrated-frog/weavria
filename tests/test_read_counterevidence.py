"""用受控的供应商 JSON 回放跨 Run 排序, 并检查适配器和存储各自的证据边界。"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from examples.read_counterevidence import (
    CURRENT,
    HISTORICAL,
    NOTICE,
    TABLE,
    VersionedSource,
    replay,
)
from examples.read_retry_experience import Document
from examples.semantic_read_retry import model_config
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    ErrorCategory,
    ModelTurn,
    SessionRequest,
    ToolCallAction,
    ToolFailedObservation,
)
from weavria.models import OpenAICompatibleSessionProvider, OpenAICompatibleTokenCounter
from weavria.tools import ReadOnlyPolicy, ToolContext, ToolExecutor, ToolRegistry, TypedTool


class Script:
    """只验证既定模型判断如何流转。真实模型是否愿意引用、采纳建议另由成对实验验证。"""

    def __init__(self, quote: str = "This timetable has been withdrawn.") -> None:
        self.quote = quote

    async def complete(self, body: dict[str, Any]) -> object:
        current = json.loads(body["messages"][1]["content"])["current"]
        assessment = None
        if current.get("control"):
            call = current["control"]["retry_read"]
            decision = {"kind": "tool_batch", "calls": [call]}
        else:
            output = current["history"][-1]["observation"]["output"]["data"]["text"]
            if "withdrawn." in output and current["task"]["goal"].startswith(CURRENT):
                decision = {
                    "kind": "tool_batch",
                    "calls": [
                        {
                            "kind": "tool_call",
                            "call_id": "next",
                            "tool_name": "read_doc",
                            "arguments": {"name": NOTICE},
                        }
                    ],
                }
                assessment = {"verdict": "inapplicable", "pointer": "/text", "quote": self.quote}
            else:
                answer = (
                    "22:45"
                    if current["task"]["goal"].startswith(HISTORICAL)
                    else "23:17"
                    if "Reinstated" in output
                    else "20:55"
                )
                decision = {"kind": "finish", "output": answer}
                assessment = {"verdict": "applicable"}
        payload: dict[str, object] = {"decision": decision}
        if assessment and "read_assessment" in body["messages"][0]["content"]:
            payload["read_assessment"] = assessment
        return {"choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}]}


@pytest.mark.parametrize("case", ["repeat", "updated", "historical"])
async def test_paired_read_receipts(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, case: str
) -> None:
    model = OpenAICompatibleSessionProvider(model_config(), transport=Script())
    base = await replay(tmp_path / "base.db", case, False, session_counter, model)
    trial = await replay(tmp_path / "trial.db", case, True, session_counter, model)
    assert trial.seed_answer == base.seed_answer == "20:55"
    assert trial.grounded_correct and base.grounded_correct
    if case == "repeat":
        assert base.reads == (TABLE, NOTICE) and trial.reads == (NOTICE,)
        assert trial.seed_verdict == "inapplicable"
        assert trial.target_turns == base.target_turns - 1
    else:
        assert base.reads == trial.reads == (TABLE,)
        assert trial.target_turns == base.target_turns
    with sqlite3.connect(tmp_path / "trial.db") as conn:
        delivery, encoded = conn.execute(
            "SELECT delivery,assessment FROM read_retry_trials WHERE ordinal=1"
        ).fetchone()
        assert delivery == "full" and json.loads(encoded)["quote"]["pointer"] == "/text"
        assert "withdrawn" not in encoded


async def test_invented_quote_does_not_demote_or_repair_the_decision(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    model = OpenAICompatibleSessionProvider(model_config(), transport=Script("Invented quote"))
    row = await replay(tmp_path / "bad.db", "repeat", True, session_counter, model)
    assert row.seed_verdict is None and row.grounded_correct
    assert row.reads == (TABLE, NOTICE) and row.target_turns == 4


async def test_wrong_pinned_version_is_never_published() -> None:
    async def wrong(source: VersionedSource, context: ToolContext) -> Document:
        return Document(text="Different edition")

    tool = TypedTool(
        name="read_doc",
        version="1",
        description="Pinned source",
        execute=wrong,
        input_type=VersionedSource,
        output_type=Document,
        artifact_read_inputs=(),
        result_sha256_parameter="digest",
    )
    executor = ToolExecutor("run", ToolRegistry((tool,)), InMemoryArtifactStore(), ReadOnlyPolicy())
    result = await executor.execute(
        ToolCallAction(
            tool_name="read_doc", call_id="read", arguments={"name": "a", "digest": "0" * 64}
        ),
        max_duration=1,
    )
    assert isinstance(result.exchange.observation, ToolFailedObservation)
    assert result.exchange.observation.error.category == ErrorCategory.INVALID_TOOL_OUTPUT
    assert result.artifact is None


@pytest.mark.parametrize("change", ["wrong_hash", "wrong_source", "uncertain"])
async def test_store_rechecks_independent_provider_assessment(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, change: str
) -> None:
    # 自定义 Provider 可直接产生合法的 v4 对象, 跳过 OpenAI 文本定位。
    # 合法的形状仍须通过持久原件核验才能影响后续排序。
    class OtherProvider(OpenAICompatibleSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            turn = await super().generate_session(request)
            assessment = turn.read_assessment
            if assessment is None or assessment.quote is None:
                return turn
            match change:
                case "wrong_hash":
                    quote = assessment.quote.model_copy(update={"quote_sha256": "0" * 64})
                    assessment = assessment.model_copy(update={"quote": quote})
                case "wrong_source":
                    assessment = assessment.model_copy(update={"source_sha256": "0" * 64})
                case _:
                    assessment = assessment.model_copy(
                        update={"verdict": "uncertain", "quote": None}
                    )
            return turn.model_copy(update={"read_assessment": assessment})

    row = await replay(
        tmp_path / "mismatch.db",
        "repeat",
        True,
        session_counter,
        OtherProvider(model_config(), transport=Script()),
    )
    assert row.seed_verdict == ("uncertain" if change == "uncertain" else None)
    assert row.reads == (TABLE, NOTICE) and row.grounded_correct


async def test_unpinned_source_keeps_delivery_ranking(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    model = OpenAICompatibleSessionProvider(model_config(), transport=Script())
    row = await replay(tmp_path / "unversioned.db", "unversioned", True, session_counter, model)
    assert row.seed_verdict == "inapplicable" and row.grounded_correct
    assert row.advice == TABLE and row.reads == (TABLE, NOTICE)
