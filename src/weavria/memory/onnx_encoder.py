"""可选本地 E5 编码: 明确角色前缀、无截断和掩码均值池化。"""

from typing import Literal

from ._local_onnx import LocalOnnxModel
from .task_retrieval import MemoryEncodingError


class OnnxE5MemoryEncoder(LocalOnnxModel):
    """宿主用 async with 管理本地模型;取消会等待当前推理完成。"""

    _protocol = "onnx-e5/1"
    _error_type = MemoryEncodingError

    async def encode(
        self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
    ) -> tuple[tuple[float, ...], ...]:
        if not 1 <= len(texts) <= 32 or any(not t.strip() or len(t.encode()) > 8192 for t in texts):
            raise MemoryEncodingError("local embedding input exceeds batch capacity")
        if role not in {"query", "passage"}:
            raise MemoryEncodingError("invalid embedding role")
        async with self._lock:
            return await self._run(lambda: self._encode(texts, role))

    def _encode(self, texts: tuple[str, ...], role: str) -> tuple[tuple[float, ...], ...]:
        assert self._tokenizer is not None and self._session is not None and self._numpy is not None
        # E5 的检索角色属于模型输入协议;非英文输入也保留前缀。
        ids = [self._tokenizer.encode(f"{role}: {text}").ids for text in texts]
        if any(not row or len(row) > self._max_tokens for row in ids):
            raise MemoryEncodingError("embedding input exceeds tokenizer capacity; no truncation")
        width = max(map(len, ids))
        raw = self._infer(ids)
        # 第三方数组在边界转成标准值并验证,不把 Any 扩散到检索算法。
        from pydantic import TypeAdapter

        hidden = TypeAdapter(list[list[list[float]]]).validate_python(raw, strict=True)
        if len(hidden) != len(ids):
            raise MemoryEncodingError("local encoder changed batch size")
        result: list[tuple[float, ...]] = []
        for row, token_ids in zip(hidden, ids, strict=True):
            if len(row) != width or not row or not 1 <= len(row[0]) <= 4096:
                raise MemoryEncodingError("invalid hidden-state layout")
            dim = len(row[0])
            if any(len(token) != dim for token in row):
                raise MemoryEncodingError("inconsistent hidden-state dimensions")
            result.append(
                tuple(
                    sum(row[t][i] for t in range(len(token_ids))) / len(token_ids)
                    for i in range(dim)
                )
            )
        return tuple(result)
