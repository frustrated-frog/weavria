"""本地 XLM-R 交叉编码: 原生 pair tokenization,无生成调用或隐式截断。"""

import math

from pydantic import TypeAdapter

from ._local_onnx import LocalOnnxModel
from .reranking import MemoryRerankError


class OnnxMemoryPairScorer(LocalOnnxModel):
    """支持 pad ID=1、单 logit 输出的 XLM-R pair 分类模型。"""

    _protocol = "onnx-xlm-r-pair/1"
    _error_type = MemoryRerankError

    async def score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float | None, ...]:
        if not 1 <= len(pairs) <= 32 or any(not q.strip() or not p.strip() for q, p in pairs):
            raise MemoryRerankError("pair scoring requires a bounded nonempty batch")
        async with self._lock:
            return await self._run(lambda: self._score(pairs))

    def _score(self, pairs: tuple[tuple[str, str], ...]) -> tuple[float | None, ...]:
        assert self._tokenizer is not None
        indices: list[int] = []
        ids: list[list[int]] = []
        types: list[list[int]] = []
        result: list[float | None] = [None] * len(pairs)
        for index, (query, passage) in enumerate(pairs):
            # 先限字节工作量,再按实际 pair tokenizer 检查总 Token 数。
            # 超限对应 unknown,绝不先裁掉尾部的例外再说它相关。
            if len(query.encode()) > 8192 or len(passage.encode()) > 65536:
                continue
            encoded = self._tokenizer.encode(query, passage)
            if not encoded.ids or len(encoded.ids) > self._max_tokens:
                continue
            if len(encoded.ids) != len(encoded.type_ids):
                raise MemoryRerankError("tokenizer pair segments are inconsistent")
            ids.append(encoded.ids)
            types.append(encoded.type_ids)
            indices.append(index)
        if not ids:
            return tuple(result)
        values = TypeAdapter(list[list[float]]).validate_python(
            self._infer(ids, types), strict=True
        )
        if len(values) != len(indices) or any(
            len(v) != 1 or not math.isfinite(v[0]) for v in values
        ):
            raise MemoryRerankError("cross encoder must produce one finite logit per pair")
        for index, value in zip(indices, values, strict=True):
            result[index] = value[0]
        return tuple(result)
