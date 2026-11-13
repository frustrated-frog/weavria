"""替身只验证检索/更新承诺;跨语言质量另由冻结任务和真实 E5 验证。"""

import asyncio
import hashlib
import json
from typing import Literal

import pytest
from pydantic import BaseModel

from weavria.harness import (
    CapabilityIndexConfig,
    CapabilityIndexError,
    CatalogSnapshot,
    HybridCapabilityIndex,
)
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class Value(BaseModel):
    value: str = ""


def catalogue(*definitions: tuple[str, str]) -> CatalogSnapshot:
    async def execute(arguments: Value, context: ToolContext) -> Value:
        raise AssertionError("retrieval cannot execute a tool")

    return CatalogSnapshot.from_registry(
        ToolRegistry(
            tuple(
                TypedTool(
                    name=name,
                    version="1",
                    description=description,
                    input_type=Value,
                    output_type=Value,
                    execute=execute,
                )
                for name, description in definitions
            )
        )
    )


class Encoder:
    version = "controlled/1"

    def __init__(self) -> None:
        self.calls: list[tuple[Literal["query", "passage"], tuple[str, ...]]] = []
        self.bad: tuple[float, ...] | None = None
        self.fail_query = False

    async def encode(
        self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
    ) -> tuple[tuple[float, ...], ...]:
        self.calls.append((role, texts))
        if self.fail_query and role == "query":
            raise OSError("private upstream content")
        return tuple(
            self.bad
            if self.bad is not None
            else ((1.0, 0.0) if "clock" in text or "几点" in text else (0.0, 1.0))
            for text in texts
        )


async def test_dense_recovers_nonlexical_candidate_and_ranks_are_not_permissions() -> None:
    index = HybridCapabilityIndex(Encoder(), config=CapabilityIndexConfig(candidate_limit=1))
    report = await index.retrieve(
        catalogue(("archive", "Retrieve a document"), ("clock", "Read UTC time")),
        "几点了",
        max_duration=2,
    )
    assert report.lexical_ranking == ()
    assert report.dense_ranking[0] == report.candidate_names[0] == "clock"
    assert "几点" not in report.model_dump_json()
    assert report.encoded_passages == 2


async def test_hot_lookup_incremental_update_removal_and_schema_identity() -> None:
    encoder = Encoder()
    index = HybridCapabilityIndex(encoder)
    first = catalogue(("clock", "Read UTC time"), ("archive", "Retrieve a document"))
    cold = await index.retrieve(first, "time", max_duration=2)
    hot = await index.retrieve(first, "document", max_duration=2)
    assert hot.encoded_passages == 0 and hot.reused_passages == 2
    assert hot.index_sha256 == cold.index_sha256
    changed = catalogue(("clock", "Read current time"), ("archive", "Retrieve a document"))
    update = await index.retrieve(changed, "time", max_duration=2)
    assert update.encoded_passages == update.reused_passages == 1
    assert update.index_sha256 != cold.index_sha256
    # 传入的新目录代表新的可见范围,旧快照不能补回被移除/撤权的工具。
    removed = await index.retrieve(
        catalogue(("archive", "Retrieve a document")), "几点", max_duration=2
    )
    assert removed.candidate_names == ("archive",)
    assert removed.encoded_passages == 0
    # 即使嵌入概览没变,完整协议变化也必须改变索引身份。
    body = changed.model_dump(mode="json")
    body["capabilities"][0]["input_schema"]["title"] = "New schema identity"
    body["catalogue_sha256"] = hashlib.sha256(
        json.dumps(
            body["capabilities"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    amended = CatalogSnapshot.model_validate_json(json.dumps(body))
    revised = await index.retrieve(amended, "time", max_duration=2)
    assert revised.index_sha256 != update.index_sha256


async def test_failure_does_not_publish_partial_build_or_return_stale_catalogue() -> None:
    encoder = Encoder()
    index = HybridCapabilityIndex(encoder)
    first = catalogue(("clock", "Read time"))
    accepted = await index.retrieve(first, "time", max_duration=2)
    encoder.fail_query = True
    changed = catalogue(("clock", "Read time"), ("archive", "Read documents"))
    with pytest.raises(CapabilityIndexError, match="encoding_failed") as caught:
        await index.retrieve(changed, "documents", max_duration=2)
    assert "private" not in str(caught.value)
    encoder.fail_query = False
    assert (
        await index.retrieve(first, "time", max_duration=2)
    ).index_sha256 == accepted.index_sha256
    retried = await index.retrieve(changed, "documents", max_duration=2)
    assert retried.encoded_passages == 1  # 失败构建的片段从未进入缓存。


@pytest.mark.parametrize("bad", [(), (0.0, 0.0), (float("nan"),), (float("inf"),)])
async def test_invalid_vectors_fail_without_cache_publication(bad: tuple[float, ...]) -> None:
    encoder = Encoder()
    encoder.bad = bad
    index = HybridCapabilityIndex(encoder)
    source = catalogue(("clock", "Read time"))
    with pytest.raises(CapabilityIndexError):
        await index.retrieve(source, "time", max_duration=2)
    encoder.bad = None
    assert (await index.retrieve(source, "time", max_duration=2)).encoded_passages == 1


async def test_encoder_identity_and_dimension_changes_are_rejected() -> None:
    encoder = Encoder()
    index = HybridCapabilityIndex(encoder)
    source = catalogue(("clock", "Read time"))
    await index.retrieve(source, "time", max_duration=2)
    encoder.bad = (1.0,)
    with pytest.raises(CapabilityIndexError, match="dimension"):
        await index.retrieve(catalogue(("archive", "Read documents")), "time", max_duration=2)
    encoder.version = "controlled/2"
    with pytest.raises(CapabilityIndexError, match="encoder_version_changed"):
        await index.retrieve(source, "time", max_duration=2)


@pytest.mark.parametrize(
    "config",
    [
        CapabilityIndexConfig(max_tools=1),
        CapabilityIndexConfig(max_text_bytes=1),
        CapabilityIndexConfig(max_passages=1),
        CapabilityIndexConfig(max_vector_elements=1),
        CapabilityIndexConfig(max_dot_products=1),
        CapabilityIndexConfig(max_query_bytes=1),
        CapabilityIndexConfig(max_query_chunks=1, chunk_bytes=64),
    ],
)
async def test_capacity_limits_fail_explicitly(config: CapabilityIndexConfig) -> None:
    index = HybridCapabilityIndex(Encoder(), config=config)
    with pytest.raises(CapabilityIndexError) as caught:
        await index.retrieve(
            catalogue(("clock", "Read time"), ("archive", "Read documents")),
            "long query " * 20,
            max_duration=2,
        )
    assert caught.value.capacity


async def test_long_descriptions_keep_tail_and_deduplicate_capability_ranking() -> None:
    encoder = Encoder()
    index = HybridCapabilityIndex(encoder, config=CapabilityIndexConfig(chunk_bytes=64))
    await index.retrieve(
        catalogue(("archive", "资料" * 100 + "clock tail")), "几点", max_duration=2
    )
    texts = tuple(t for role, batch in encoder.calls if role == "passage" for t in batch)
    assert all(len(t.encode()) <= 64 for t in texts)
    assert "clock tail" in "".join(texts) and "资料" in "".join(texts)
    report = await index.retrieve(
        catalogue(("archive", "资料" * 100 + "clock tail")), "几点", max_duration=2
    )
    assert report.dense_ranking == ("archive",) and report.encoded_passages == 0


async def test_empty_and_reordered_catalogue_have_deterministic_ranking() -> None:
    encoder = Encoder()
    index = HybridCapabilityIndex(encoder)
    empty = await index.retrieve(catalogue(), "time", max_duration=2)
    assert empty.candidate_names == () and not encoder.calls
    a, b = ("alpha", "Read a document"), ("beta", "Read a document")
    first = await index.retrieve(catalogue(a, b), "document", max_duration=2)
    second = await index.retrieve(catalogue(b, a), "document", max_duration=2)
    assert first.candidate_names == second.candidate_names == ("alpha", "beta")
    assert second.encoded_passages == 0


async def test_cancel_timeout_and_waiters_have_no_detached_work() -> None:
    entered, released, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Waiting(Encoder):
        async def encode(
            self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
        ) -> tuple[tuple[float, ...], ...]:
            entered.set()
            try:
                await released.wait()
                return await super().encode(texts, role=role)
            finally:
                cleaned.set()

    index = HybridCapabilityIndex(Waiting())
    source = catalogue(("clock", "Read time"))
    active = asyncio.create_task(index.retrieve(source, "time", max_duration=2))
    await entered.wait()
    with pytest.raises(TimeoutError):
        await index.retrieve(source, "time", max_duration=0.01)  # 期限包含等待构建锁。
    active.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active
    assert cleaned.is_set()
    released.set()
    results = await asyncio.gather(
        *(index.retrieve(source, "time", max_duration=2) for _ in range(2))
    )
    assert [r.encoded_passages for r in results] == [1, 0]


async def test_rrf_preserves_contributions_from_both_channels() -> None:
    # query 的词面偏向 archive,语义向量偏向 clock;两路头名应各有贡献。
    index = HybridCapabilityIndex(
        Encoder(), config=CapabilityIndexConfig(channel_limit=1, candidate_limit=2)
    )
    result = await index.retrieve(
        catalogue(("archive", "document"), ("clock", "UTC")), "document 几点", max_duration=2
    )
    assert result.lexical_ranking == ("archive",)
    assert result.dense_ranking == ("clock",)
    assert result.candidate_names == ("archive", "clock")


async def test_encoding_batch_and_midcall_version_are_validated() -> None:
    class WrongBatch(Encoder):
        async def encode(
            self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
        ) -> tuple[tuple[float, ...], ...]:
            return ()

    source = catalogue(("clock", "Read time"))
    with pytest.raises(CapabilityIndexError, match="batch_size"):
        await HybridCapabilityIndex(WrongBatch()).retrieve(source, "time", max_duration=2)

    class Changed(Encoder):
        async def encode(
            self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
        ) -> tuple[tuple[float, ...], ...]:
            self.version = "changed"
            return await super().encode(texts, role=role)

    with pytest.raises(CapabilityIndexError, match="version_changed"):
        await HybridCapabilityIndex(Changed()).retrieve(source, "time", max_duration=2)


async def test_timeout_during_encoding_cleans_up_and_can_retry() -> None:
    cleaned = asyncio.Event()

    class Waiting(Encoder):
        blocked = True

        async def encode(
            self, texts: tuple[str, ...], *, role: Literal["query", "passage"]
        ) -> tuple[tuple[float, ...], ...]:
            if self.blocked:
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()
            return await super().encode(texts, role=role)

    encoder = Waiting()
    index = HybridCapabilityIndex(encoder)
    source = catalogue(("clock", "Read time"))
    with pytest.raises(TimeoutError):
        await index.retrieve(source, "time", max_duration=0.01)
    assert cleaned.is_set()
    encoder.blocked = False
    assert (await index.retrieve(source, "time", max_duration=2)).encoded_passages == 1
