from examples.history_context import Case, Dataset, evaluate
from weavria.models import OpenAICompatibleTokenCounter


async def test_labels_do_not_influence_discovery_or_multi_span_delivery(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    case = Case(
        name="frozen",
        query="Aurora password and Zephyr deadline",
        first="Aurora password is violet. ",
        filler="irrelevant appendix ",
        repeats=700,
        last="Zephyr deadline is Friday. ",
        target_ordinal=1,
        expected_phrases=("password is violet", "deadline is Friday"),
    )
    dataset = Dataset(
        schema_version=1,
        dataset_version="test/1",
        evidence_scope="synthetic",
        turns=6,
        cases=(case,),
    )
    before = await evaluate(dataset, counter=session_counter, window_tokens=10000)
    changed = dataset.model_copy(
        update={"cases": (case.model_copy(update={"expected_phrases": ("impossible",)}),)}
    )
    after = await evaluate(changed, counter=session_counter, window_tokens=10000)
    assert len(before) == len(after) == 3
    assert before[-1].delivered == 2 and before[-1].source_ranges_valid
    for first, second in zip(before, after, strict=True):
        assert first.request_sha256 == second.request_sha256
        assert first.input_tokens == second.input_tokens
        assert first.postings_examined == second.postings_examined
        assert first.candidates == second.candidates
        assert second.delivered == 0
