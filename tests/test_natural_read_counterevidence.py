"""用可预测的决策校准计数器与出处判分。在线实验直接传入正式 Provider。"""

import hashlib
from pathlib import Path

from examples.natural_read_counterevidence import CASES, content_digest, run_series
from examples.read_counterevidence_feedback import read_action
from weavria.core import (
    AdmissionNotice,
    FinishAction,
    ModelTurn,
    ReadAssessment,
    ReadCounterquote,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
)
from weavria.models import ModelCapabilities, OpenAICompatibleTokenCounter


async def test_past_answer_alone_does_not_pass_repeat_task(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    case = CASES[1]

    class Reader:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            if not request.conversation.turns and request.current.feedback is None:
                return read_action(case.current, "lookup")
            return ModelTurn(decision=FinishAction(output=f"{case.answer} [{case.current}]"))

    first, second = await run_series(
        tmp_path / "test.db", case, "feedback", False, Reader(), session_counter, ()
    )
    assert first.grounded_correct and first.reads == first.delivered == (case.current,)
    assert first.tool_calls == 1
    assert second.answer == first.answer and not second.grounded_correct
    assert second.first_proposal == ("finish",) and second.reads == second.delivered == ()
    assert second.shown == second.stored_negatives == second.tool_calls == 0
    assert second.provider["total_tokens"] is None


async def test_only_indexed_observations_count_as_reusable(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    case = CASES[0]

    class Reader:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

        def __init__(self, batch: bool) -> None:
            self.batch = batch

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            control, feedback = request.current.control, request.current.feedback
            if isinstance(control, AdmissionNotice):
                assert control.retry_read is not None
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(control.retry_read.model_copy(update={"call_id": "accepted-read"}),)
                    )
                )
            if feedback is None:
                if self.batch:
                    return ModelTurn(
                        decision=ToolBatch(
                            calls=tuple(
                                ToolCallAction(
                                    call_id=f"proposal-{i}",
                                    tool_name="read_doc",
                                    arguments={"name": name},
                                )
                                for i, name in enumerate((case.old, case.current))
                            )
                        )
                    )
                return read_action(case.old, "old-source")
            if not any(
                isinstance(e, ToolExchange) and e.action.arguments.get("name") == case.current
                for e in request.current.history
            ):
                quote = "This timetable has been withdrawn."
                start = case.old_text.index(quote)
                return ModelTurn(
                    schema_version=4,
                    decision=read_action(case.current, "current-source").decision,
                    read_assessment=ReadAssessment(
                        source_sha256=content_digest(case.old_text),
                        verdict="inapplicable",
                        quote=ReadCounterquote(
                            pointer="/text",
                            start=start,
                            end=start + len(quote),
                            quote_sha256=hashlib.sha256(quote.encode()).hexdigest(),
                        ),
                    ),
                )
            return ModelTurn(decision=FinishAction(output=f"{case.answer} [{case.current}]"))

    direct, later = await run_series(
        tmp_path / "direct.db", case, "feedback", False, Reader(False), session_counter, ()
    )
    assert direct.old_judgments == later.old_judgments == 1
    assert direct.stored_negatives == later.recalled_negative == later.shown == 0
    assert direct.grounded_correct and later.grounded_correct

    seed, repeat = await run_series(
        tmp_path / "batch.db", case, "feedback", False, Reader(True), session_counter, ()
    )
    assert seed.old_judgments == seed.stored_negatives == 1
    assert repeat.recalled_negative == repeat.displacements == repeat.shown == 1
    assert seed.reads == (case.old, case.current) and repeat.reads == (case.current,)
    assert seed.tool_calls == 2 and repeat.tool_calls == 1
    assert seed.grounded_correct and repeat.grounded_correct
