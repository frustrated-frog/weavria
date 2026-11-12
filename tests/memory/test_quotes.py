"""引文修复只定位原文,不允许模糊补词或消除来源歧义。"""

import hashlib
import json
from pathlib import Path

import pytest

from examples.memory_sufficiency import Dataset, Fact, bound_supply, seeded_supply
from weavria.context import ContextDemand
from weavria.core import ModelResponse, ToolCallAction
from weavria.harness import JointMemoryEvidenceAssessor, ModelMemoryAnswerer
from weavria.harness.memory_quotes import locate_memory_quote
from weavria.memory import SQLiteMemoryStore
from weavria.models import FakeModelProvider


@pytest.mark.parametrize("quote", ["Do not log tokens.", r"Do not\n  log tokens."])
def test_whitespace_or_escaped_controls_resolve_original_span(quote: str) -> None:
    text = "Prefix: Do not\n  log tokens. Suffix."
    located = locate_memory_quote(text, quote)
    assert located.start == 8 and located.quote == "Do not\n  log tokens."
    assert located.resolution == "source_normalized"
    assert text[located.start : located.start + len(located.quote)] == located.quote


def test_literal_backslash_is_not_unescaped_when_original_contains_it() -> None:
    located = locate_memory_quote(r"The literal token \n is not a line break.", r"\n")
    assert located.quote == r"\n" and located.resolution == "literal"


@pytest.mark.parametrize(
    "text,quote",
    [
        ("Do not log tokens.", "Do log tokens."),
        ("Do not log tokens.", "do not log tokens."),
        ("Do not log tokens.", "Do not log tokens!"),
        ("Do not\nlog tokens. Do not log tokens.", "Do not  log tokens."),
        ("aaaa", "aaa"),
    ],
)
def test_mutation_and_ambiguity_remain_invalid(text: str, quote: str) -> None:
    with pytest.raises(ValueError):
        locate_memory_quote(text, quote)


async def test_evidence_and_answer_keep_restored_source_and_hash(tmp_path: Path) -> None:
    query, text = "Should tokens be logged?", "Never log\n  access tokens."
    async with SQLiteMemoryStore(tmp_path / "memory.db") as store:
        bundles = await seeded_supply(
            Dataset(
                version="test/1",
                description="source quote",
                cases=(),
                facts=(Fact(id="rule", predicate="logging", text=text),),
            ),
            store,
        )
    supply = bound_supply(bundles, query)
    demands = (ContextDemand("current_task", query),)
    provider = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="e",
                    tool_name="assess_memory_evidence_jointly",
                    arguments={
                        "demands": [
                            {
                                "demand_id": "current_task",
                                "parts": [{"text": query, "slot": 0}],
                                "slots": [
                                    {
                                        "roles": ["direct"],
                                        "conditions": [],
                                        "citations": [
                                            {"member": 0, "quote": r"Never log\n  access tokens."}
                                        ],
                                    }
                                ],
                            }
                        ]
                    },
                )
            ),
        )
    )
    assessment = await JointMemoryEvidenceAssessor(provider, configuration_id="test").assess(
        supply, demands
    )
    assert assessment.failure is None and assessment.demands[0].needs[0].status == "supported"
    assert (
        assessment.demands[0].needs[0].checks[0].quote_sha256
        == hashlib.sha256(text.encode()).hexdigest()
    )
    writer = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="a",
                    tool_name="propose_memory_answer",
                    arguments={
                        "claims": [
                            {
                                "text": "Do not log access tokens.",
                                "citations": [{"member": 0, "quote": "Never log access tokens."}],
                            }
                        ]
                    },
                )
            ),
        )
    )
    draft = await ModelMemoryAnswerer(writer, configuration_id="test").propose(supply, demands)
    citation = draft.claims[0].citations[0]
    assert (
        citation.quote == text
        and citation.start == 0
        and citation.resolution == "source_normalized"
    )
    # 输入原文未变化;模型原提案和宿主定位后的输出通过各自摘要区分。
    assert text in json.dumps(bundles[0].content, ensure_ascii=False).replace(r"\n", "\n")
