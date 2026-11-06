"""用可控本地依赖验证编码协议、线程归属和取消;不下载或运行真实模型。"""

import asyncio
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from weavria.memory import MemoryEncodingError, OnnxE5MemoryEncoder, OnnxMemoryPairScorer


class Backend:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.entered, self.release, self.finished = (threading.Event() for _ in range(3))
        self.block = False
        self.long = False
        self.scalar = False

    def encode(self, text: str, pair: str | None = None) -> SimpleNamespace:
        self.texts.append(text if pair is None else text + "|" + pair)
        ids = list(range(513)) if self.long else [0, 5, 2]
        return SimpleNamespace(ids=ids, type_ids=[0] * len(ids))

    def run(self, output_names: object, input_feed: dict[str, list[list[int]]]) -> list[object]:
        self.entered.set()
        try:
            if self.block and not self.release.wait(5):
                raise TimeoutError("fixture worker was not released")
            # 每 token 两维,掩码均值结果固定,便于检查 query/passage 协议。
            if self.scalar:
                scores = [[2.0] for _ in input_feed["input_ids"]]
                return [SimpleNamespace(tolist=lambda: scores)]
            values = [[[1.0, 3.0] for _ in row] for row in input_feed["input_ids"]]
            return [SimpleNamespace(tolist=lambda: values)]
        finally:
            self.finished.set()


@pytest.fixture
def local_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Backend, Path, Path]:
    backend = Backend()
    model, tokenizer = tmp_path / "model.onnx", tmp_path / "tokenizer.json"
    model.write_bytes(b"test model")
    tokenizer.write_text("{}")

    def session(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            get_inputs=lambda: [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
            ],
            run=backend.run,
        )

    def from_str(serialized: str) -> SimpleNamespace:
        return SimpleNamespace(
            encode=backend.encode,
            no_truncation=lambda: None,
            no_padding=lambda: None,
        )

    def array(value: list[list[int]], *, dtype: str) -> list[list[int]]:
        assert dtype == "int64"
        return value

    modules = {
        "onnxruntime": SimpleNamespace(
            SessionOptions=lambda: SimpleNamespace(),
            InferenceSession=session,
        ),
        "tokenizers": SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)),
        "numpy": SimpleNamespace(array=array),
    }
    import importlib

    original = importlib.import_module

    def load(name: str, package: str | None = None) -> object:
        return modules[name] if name in modules else original(name, package)

    monkeypatch.setattr("weavria.memory._local_onnx.importlib.import_module", load)
    return backend, model, tokenizer


async def test_cross_encoder_uses_tokenizer_pair_and_withholds_overlong_group(
    local_backend: tuple[Backend, Path, Path],
) -> None:
    backend, model, tokenizer = local_backend
    backend.scalar = True
    async with OnnxMemoryPairScorer(model, tokenizer) as scorer:
        assert await scorer.score((("question", "all facts and exceptions"),)) == (2.0,)
        assert backend.texts == ["question|all facts and exceptions"]
        backend.long = True
        backend.entered.clear()
        assert await scorer.score((("question", "too many tokens"),)) == (None,)
        assert not backend.entered.is_set()


@pytest.mark.parametrize("role", ["query", "passage"])
async def test_role_prefix_digest_pooling_and_no_truncation(
    local_backend: tuple[Backend, Path, Path], role: Literal["query", "passage"]
) -> None:
    backend, model, tokenizer = local_backend
    async with OnnxE5MemoryEncoder(model, tokenizer) as encoder:
        assert hashlib.sha256(model.read_bytes()).hexdigest() in encoder.version
        assert await encoder.encode(("hello",), role=role) == ((1.0, 3.0),)
        assert backend.texts == [f"{role}: hello"]
        backend.long = True
        with pytest.raises(MemoryEncodingError, match="no truncation"):
            await encoder.encode(("large",), role=role)
    with pytest.raises(MemoryEncodingError, match="closed"):
        await encoder.encode(("hello",), role=role)
    with pytest.raises(MemoryEncodingError, match="digest mismatch"):
        async with OnnxE5MemoryEncoder(model, tokenizer, model_sha256="f" * 64):
            pytest.fail("file digest mismatch must fail before dependencies run")


async def test_cancelled_encode_and_close_join_owned_worker(
    local_backend: tuple[Backend, Path, Path],
) -> None:
    backend, model, tokenizer = local_backend
    backend.block = True
    encoder = await OnnxE5MemoryEncoder(model, tokenizer).__aenter__()
    pending = asyncio.create_task(encoder.encode(("hello",), role="query"))
    try:
        assert await asyncio.to_thread(backend.entered.wait, 2)
        pending.cancel()
        close = asyncio.create_task(encoder.__aexit__())
        await asyncio.sleep(0)
        close.cancel()
        await asyncio.sleep(0)
        assert not pending.done() and not close.done()
    finally:
        backend.release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    with pytest.raises(asyncio.CancelledError):
        await close
    assert backend.finished.is_set()
    with pytest.raises(MemoryEncodingError, match="closed"):
        await encoder.encode(("later",), role="query")
