from dataclasses import replace

from weavria.evaluation.context_compiler import ContextProbe, evaluate_context_compiler
from weavria.models import OpenAICompatibleTokenCounter


async def test_expected_answers_are_evaluation_only_and_do_not_influence_selection(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    case = ContextProbe(
        "check",
        "Aurora password",
        ("unrelated " * 100, "archive"),
        ("Aurora password is violet.", "kitchen " * 200),
        ("violet",),
    )
    first = await evaluate_context_compiler(
        (case,), counter=session_counter, window_tokens=6000, max_request_bytes=4000
    )
    changed = await evaluate_context_compiler(
        (replace(case, expected_phrases=("impossible-marker",)),),
        counter=session_counter,
        window_tokens=6000,
        max_request_bytes=4000,
    )
    assert len(first) == len(changed) == 4
    for before, after in zip(first, changed, strict=True):
        assert (
            before.input_tokens,
            before.request_bytes,
            before.evaluations,
            before.selected_sources,
        ) == (after.input_tokens, after.request_bytes, after.evaluations, after.selected_sources)
        assert after.retained == 0
    assert first[-1].retained == 1
