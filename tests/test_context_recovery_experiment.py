"""The real-material probe keeps evaluator labels out of compilation and search."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

from weavria.models import OpenAICompatibleTokenCounter


async def test_frozen_recovery_probe_is_label_blind(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    path = Path(__file__).parents[1] / "examples/context_recovery.py"
    spec = importlib.util.spec_from_file_location("recovery_probe", path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        dataset = module.Dataset.model_validate_json(
            (path.parent / "data/context_recovery/cases.json").read_text()
        )
        dataset = dataset.model_copy(update={"cases": dataset.cases[:1]})
        changed = dataset.model_copy(
            update={
                "cases": (
                    dataset.cases[0].model_copy(
                        update={"expected_phrases": ("unfindable-marker",)}
                    ),
                )
            }
        )
        first = await module.evaluate(dataset, counter=session_counter, window_tokens=24000)
        second = await module.evaluate(changed, counter=session_counter, window_tokens=24000)
        assert len(first) == len(second) == 3
        for before, after in zip(first, second, strict=True):
            assert (
                before.request_sha256,
                before.input_tokens,
                before.recovery_bytes,
                before.tool_calls,
            ) == (after.request_sha256, after.input_tokens, after.recovery_bytes, after.tool_calls)
            assert after.delivered == 0
            assert before.request_bytes <= 12000 and before.input_tokens <= 24000 - 256 - 128
    finally:
        sys.modules.pop(spec.name, None)
