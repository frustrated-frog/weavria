"""来源语境是同字段精确原文;完整性、资源界限和恢复不依赖测试问题。"""

import json

import pytest

from weavria.context import ContextCapacityError
from weavria.context.indexing import bounded_passages


async def test_sections_and_code_scopes_travel_with_exact_unicode_excerpts() -> None:
    text = (
        "===== guide.md =====\n# Configuration\n## Example only\n"
        + "说明 " * 600
        + '\nsetting="basic"\n'
        + "===== module.py =====\nclass Engine:\n    def read(self):\n"
        + "        value = '内容'\n" * 160
    )
    passages = await bounded_passages({"text": text}, max_passages=100, max_bytes=100_000)
    assert "".join(json.loads(p.value_json) for p in passages) == text
    for p in passages:
        assert json.loads(p.value_json) == text[p.start : p.end]
        assert len(text[p.start : p.end].encode()) <= 320
        for c in p.context:
            assert c.value == text[c.start : c.end]
            assert len(c.value.encode()) <= 240
    setting = next(p for p in passages if "setting=" in p.value_json)
    assert any("Example only" in c.value for c in setting.context)
    assert any("guide.md" in c.value for c in setting.context)
    tail = passages[-1]
    assert any("module.py" in c.value for c in tail.context)
    assert any("def read" in c.value for c in tail.context)
    assert all("guide.md" not in c.value for c in tail.context)


async def test_context_overhead_counts_toward_index_limit() -> None:
    text = "# " + "x" * 220 + "\n" + "body " * 300
    with pytest.raises(ContextCapacityError, match="index_bytes"):
        await bounded_passages(text, max_passages=100, max_bytes=len(json.dumps(text).encode()))


async def test_indented_code_comments_do_not_replace_function_scope() -> None:
    text = "class Engine:\n    def recover(self):\n" + (
        "        # 关于返回值的中文注释\n        result = proposed.start\n" * 60
    )
    passages = await bounded_passages(text, max_passages=100, max_bytes=100_000)
    assert len(passages) > 4
    assert all(any("def recover" in c.value for c in p.context) for p in passages[1:])
    assert all(not any(c.value.lstrip().startswith("#") for c in p.context) for p in passages)
