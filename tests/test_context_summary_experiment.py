"""对照入口冻结相同任务/预算;显式摘要功能验证不得冒充自主任务收益。"""

import json
from dataclasses import replace

from examples.context_summary_experiment import configuration, grade_audit


def test_summary_arms_change_only_feature_flag_and_configuration_identity() -> None:
    baseline, candidate = configuration("extractive"), configuration("hierarchical")
    assert (
        replace(
            baseline,
            configuration_id=candidate.configuration_id,
            context=replace(baseline.context, semantic_compaction=True),
        )
        == candidate
    )


def test_guided_arm_and_audit_grading_do_not_claim_semantic_success() -> None:
    passive, guided = configuration("hierarchical"), configuration("guided")
    assert (
        replace(
            passive,
            configuration_id=guided.configuration_id,
            context=replace(passive.context, summary_guidance=True),
        )
        == guided
    )
    quote = "A caller cancellation preserves results of completed sibling operations."
    documents = {"public.md": quote}
    output = {
        "sources": ["public.md"],
        "claims": [
            {"path": "public.md", "quote": quote, "statement": "Interpretation still needs review."}
        ],
    }
    grade = grade_audit(json.dumps(output), documents)
    assert grade["evidence_passed"] and grade["semantic_review_required"]
    output = {
        "sources": ["public.md"],
        "claims": [
            {
                "path": "public.md",
                "quote": "Invented statement that never appeared in the original.",
                "statement": "Interpretation still needs review.",
            }
        ],
    }
    assert not grade_audit(json.dumps(output), documents)["evidence_passed"]


def test_prepared_arm_keeps_model_and_task_budgets_fixed() -> None:
    guided, prepared = configuration("guided"), configuration("prepared")
    assert (
        replace(
            guided,
            configuration_id=prepared.configuration_id,
            context=replace(guided.context, summary_preparation=True),
        )
        == prepared
    )
