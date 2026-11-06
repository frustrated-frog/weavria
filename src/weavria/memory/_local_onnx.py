"""本地 ONNX 适配器共享的文件身份、输入和有界工作线程生命周期。"""

import asyncio
import hashlib
import importlib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, Self, cast

from .contracts import MemoryError


class _Encoding(Protocol):
    @property
    def ids(self) -> list[int]: ...

    @property
    def type_ids(self) -> list[int]: ...


class _Tokenizer(Protocol):
    def encode(self, text: str, pair: str | None = None) -> _Encoding: ...
    def no_truncation(self) -> None: ...
    def no_padding(self) -> None: ...


class _TokenizerFactory(Protocol):
    def from_str(self, serialized: str) -> _Tokenizer: ...


class _Tokenizers(Protocol):
    @property
    def Tokenizer(self) -> _TokenizerFactory: ...


class _Input(Protocol):
    @property
    def name(self) -> str: ...


class _Session(Protocol):
    def get_inputs(self) -> Sequence[_Input]: ...
    def run(self, output_names: None, input_feed: dict[str, object]) -> list[object]: ...


class _Options(Protocol):
    intra_op_num_threads: int
    inter_op_num_threads: int


class _Onnx(Protocol):
    def SessionOptions(self) -> _Options: ...
    def InferenceSession(
        self, path: bytes, *, sess_options: _Options, providers: list[str]
    ) -> _Session: ...


class _Numpy(Protocol):
    def array(self, value: list[list[int]], *, dtype: str) -> object: ...


class _Array(Protocol):
    def tolist(self) -> object: ...


class LocalOnnxModel:
    """显式加载本地 ONNX/tokenizer 文件;不下载模型、不持久化正文或向量。

    一个有界 worker 串行执行 CPU 推理。取消时等待已提交批次结束后传播,
    不能把协作式取消说成硬抢占。由 async context manager 关闭 worker。
    """

    _protocol = "local-onnx/1"
    _error_type: type[MemoryError] = MemoryError

    def __init__(
        self,
        model: Path,
        tokenizer: Path,
        *,
        max_tokens: int = 512,
        model_sha256: str | None = None,
        tokenizer_sha256: str | None = None,
    ) -> None:
        if type(max_tokens) is not int or not 8 <= max_tokens <= 512:
            raise ValueError("local model maximum tokens must be in [8,512]")
        self._model_path, self._tokenizer_path = model, tokenizer
        self._expected = (model_sha256, tokenizer_sha256)
        self._max_tokens = max_tokens
        self._pool: ThreadPoolExecutor | None = None
        self._session: _Session | None = None
        self._tokenizer: _Tokenizer | None = None
        self._numpy: _Numpy | None = None
        self._version: str | None = None
        self._lock = asyncio.Lock()

    @property
    def version(self) -> str:
        if self._version is None:
            raise self._error_type("local encoder must be opened before binding retrieval")
        return self._version

    def _open(self) -> None:
        if (
            self._model_path.stat().st_size > 1_073_741_824
            or self._tokenizer_path.stat().st_size > 33_554_432
        ):
            raise self._error_type("local embedding files exceed loading capacity")
        model_bytes, tokenizer_bytes = (
            self._model_path.read_bytes(),
            self._tokenizer_path.read_bytes(),
        )
        digests = (
            hashlib.sha256(model_bytes).hexdigest(),
            hashlib.sha256(tokenizer_bytes).hexdigest(),
        )
        if any(
            expected is not None and actual != expected
            for expected, actual in zip(self._expected, digests, strict=True)
        ):
            raise self._error_type("local embedding file digest mismatch")
        ort = cast(_Onnx, importlib.import_module("onnxruntime"))
        tokenizer_module = cast(_Tokenizers, importlib.import_module("tokenizers"))
        self._numpy = cast(_Numpy, importlib.import_module("numpy"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = tokenizer_module.Tokenizer.from_str(tokenizer_bytes.decode())
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self._version = f"{self._protocol}:{digests[0]}:{digests[1]}:tokens={self._max_tokens}"

    async def __aenter__(self) -> Self:
        if self._pool is not None:
            raise self._error_type("local encoder is already open")
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weavria-embedding")
        try:
            await self._run(self._open)
        except BaseException:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = self._session = self._tokenizer = self._numpy = None
            self._version = None
            raise
        return self

    async def __aexit__(self, *args: object) -> None:
        # close 自身取消也要等待所有已提交推理结束;此任务始终由当前调用 join。
        task = asyncio.create_task(self._close())
        cancelled = False
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        async with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = self._session = self._tokenizer = self._numpy = None
            self._version = None

    async def _run[T](self, operation: Callable[[], T]) -> T:
        if self._pool is None:
            raise self._error_type("local encoder is closed")
        future = asyncio.get_running_loop().run_in_executor(self._pool, operation)
        cancelled = False
        while not future.done():
            try:
                await asyncio.wait({future})
            except asyncio.CancelledError:
                cancelled = True
        try:
            result = future.result()
        except Exception as exc:
            if cancelled:
                raise asyncio.CancelledError from exc
            if isinstance(exc, self._error_type):
                raise
            raise self._error_type("local embedding dependency failed") from exc
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _infer(self, ids: list[list[int]], types: list[list[int]] | None = None) -> object:
        assert self._session is not None and self._numpy is not None
        width = max(map(len, ids))
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in ids]
        values = {
            "input_ids": [row + [1] * (width - len(row)) for row in ids],
            "attention_mask": masks,
            "token_type_ids": [row + [0] * (width - len(row)) for row in types]
            if types is not None
            else [[0] * width for _ in ids],
        }
        inputs = self._session.get_inputs()
        if any(item.name not in values for item in inputs):
            raise self._error_type("unsupported local encoder input layout")
        outputs = self._session.run(
            None,
            {item.name: self._numpy.array(values[item.name], dtype="int64") for item in inputs},
        )
        if not outputs:
            raise self._error_type("local encoder returned no hidden states")
        return cast(_Array, outputs[0]).tolist()
