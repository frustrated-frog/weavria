"""按请求对照可见原文,并在取材边界注入错误、取消和容量收紧。"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from examples.conditional_material import run_case
from weavria.context import ContextCompilerConfig, ContextSourceError
from weavria.core import (
    ContextBundle,
    ContextCompilation,
    FinishAction,
    ModelTurn,
    SessionContextCompiled,
)
from weavria.harness import (
    ConditionalMaterial,
    ExtensionAssembly,
    ExtensionDescriptor,
    ExtensionHooks,
    ExtensionQuery,
    ExtensionScope,
    HarnessExtension,
    HarnessExtensions,
)
from weavria.models import FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.sessions import SessionEngine
from weavria.sessions.extensions import RunExtensions
from weavria.tools import ToolRegistry

_QUERY = ExtensionQuery("delta", (), 1, "decision")
type Reader = Callable[[ExtensionQuery, str | None], Awaitable[ConditionalMaterial]]


def source(reader: Reader, name: str = "docs", *, selector: str | None = None) -> HarnessExtension:
    @asynccontextmanager
    async def mount(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        yield ExtensionHooks(read_material=reader)

    return HarnessExtension(
        ExtensionDescriptor(
            schema_version=3,
            name=name,
            version="1",
            hooks=("before_compile",),
            conditional_material=True,
            uses_selector=selector,
        ),
        mount,
    )


@asynccontextmanager
async def mounted(*plugins: HarnessExtension) -> AsyncGenerator[RunExtensions]:
    assembly = ExtensionAssembly(plugins)
    chain = RunExtensions(assembly.plan)
    try:
        await chain.open(assembly.extensions, ExtensionScope("session", "run"))
        yield chain
    finally:
        await chain.close()


async def test_same_request_bytes_with_partial_invalidation(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    baseline = await run_case(tmp_path, conditional=False, counter=session_counter)
    reuse = await run_case(tmp_path, conditional=True, counter=session_counter)
    assert baseline.full_reads == (4, 4) and reuse.full_reads == (2, 1)
    assert reuse.request_hashes == baseline.request_hashes
    assert reuse.outcomes == (
        ("loaded", "loaded"),
        ("reused", "reused"),
        ("loaded", "reused"),
        ("reused", "reused"),
    )
    old, repeated, updated, last = reuse.delivered
    assert old == repeated and updated == last
    assert len(old) == len(updated) == 2 and old[0] != updated[0] and old[1] == updated[1]


@pytest.mark.parametrize(
    "changed",
    [
        replace(_QUERY, goal="other"),
        replace(_QUERY, constraints=("exact",)),
        replace(_QUERY, revision=2),
        replace(_QUERY, phase="synthesis"),
    ],
)
async def test_validator_is_local_to_one_query_and_mount(changed: ExtensionQuery) -> None:
    offered: list[str | None] = []

    async def read(query: ExtensionQuery, previous: str | None) -> ConditionalMaterial:
        offered.append(previous)
        return ConditionalMaterial("stable", None if previous else ())

    ext = source(read)
    async with mounted(ext) as chain:
        for query in (_QUERY, _QUERY, changed, _QUERY):
            material, _, reads = await chain.collect(query, max_bytes=100)
            assert not material and reads[0].candidate_count == 0
    async with mounted(ext) as other:
        await other.collect(_QUERY, max_bytes=100)
    assert offered == [None, "stable", None, None, None]


async def test_reuse_rechecks_selection_and_protects_snapshot() -> None:
    shared = ContextBundle(bundle_id="p", content={"text": "original"})

    async def read(query: ExtensionQuery, previous: str | None) -> ConditionalMaterial:
        return ConditionalMaterial("v1", None if previous else (shared,))

    decisions = iter((False, True))

    @asynccontextmanager
    async def policy(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        async def select(
            query: ExtensionQuery, bundles: tuple[ContextBundle, ...]
        ) -> tuple[str, ...]:
            (bundle,) = bundles
            assert bundle.content["text"] == "original"
            bundle.content["text"] = "modified by selector"
            return (bundle.bundle_id,) if next(decisions) else ()

        yield ExtensionHooks(select_material=select)

    selector = HarnessExtension(
        ExtensionDescriptor(
            schema_version=2, name="policy", version="1", provides_selector="filter", hooks=()
        ),
        policy,
    )
    async with mounted(source(read, selector="filter"), selector) as chain:
        empty, denied, loaded = await chain.collect(_QUERY, max_bytes=1000)
        assert empty == () and not denied[0].candidates[0].admitted
        shared.content["text"] = "modified by producer"
        accepted, admitted, reused = await chain.collect(_QUERY, max_bytes=1000)
        assert (
            reused[0].outcome == "reused"
            and reused[0].candidates_sha256 == loaded[0].candidates_sha256
        )
        assert admitted[0].candidates[0].admitted and accepted[0].content == {"text": "original"}


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("cold_confirmation", "invalid_snapshot"),
        ("wrong_revision", "invalid_snapshot"),
        ("revision_collision", "invalid_snapshot"),
        ("bad_revision", "invalid_snapshot"),
        ("invalid_unicode", "invalid_snapshot"),
        ("budget", "capacity"),
        ("unavailable", "unavailable"),
        ("revoked", "access_denied"),
    ],
)
async def test_cache_never_masks_source_or_contract_failure(mode: str, reason: str) -> None:
    full = ConditionalMaterial("v1", (ContextBundle(bundle_id="p", content={"text": "first"}),))

    async def read(query: ExtensionQuery, previous: str | None) -> ConditionalMaterial:
        if mode == "cold_confirmation":
            return ConditionalMaterial("v1", None)
        if previous is None:
            return full
        if mode == "wrong_revision":
            return ConditionalMaterial("v2", None)
        if mode == "revision_collision":
            return ConditionalMaterial("v1", ())
        if mode == "bad_revision":
            return ConditionalMaterial(" ", ())
        if mode == "invalid_unicode":
            return ConditionalMaterial(chr(0xD800), ())
        if mode == "unavailable":
            raise OSError("source offline")
        if mode == "revoked":
            raise ContextSourceError("access_denied")
        return ConditionalMaterial(previous, None)

    async with mounted(source(read)) as chain:
        if mode != "cold_confirmation":
            await chain.collect(_QUERY, max_bytes=1024)
        with pytest.raises(ContextSourceError, match=reason):
            await chain.collect(_QUERY, max_bytes=1 if mode == "budget" else 1024)


async def test_cancelled_batch_does_not_publish_partial_refresh() -> None:
    offered: list[str | None] = []
    revision = "old"

    async def changing(query: ExtensionQuery, previous: str | None) -> ConditionalMaterial:
        offered.append(previous)
        return ConditionalMaterial(revision, None if revision == previous else ())

    entered = asyncio.Event()
    block = False

    async def sibling(query: ExtensionQuery, previous: str | None) -> ConditionalMaterial:
        if block:
            entered.set()
            await asyncio.Event().wait()
        return ConditionalMaterial("same", None if previous else ())

    async with mounted(source(changing, "alpha"), source(sibling, "beta")) as chain:
        await chain.collect(_QUERY, max_bytes=100)
        block, revision = True, "new"
        task = asyncio.create_task(chain.collect(_QUERY, max_bytes=100))
        try:
            await asyncio.wait_for(entered.wait(), 2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        block = False
        _, _, reads = await chain.collect(_QUERY, max_bytes=100)
    assert offered == [None, "old", "old"]
    assert [r.outcome for r in reads] == ["loaded", "reused"]


async def test_archive_binds_preselection_snapshot_even_when_all_candidates_are_filtered(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter
) -> None:
    async def read(query: ExtensionQuery, previous: str | None) -> ConditionalMaterial:
        return ConditionalMaterial(
            "internal-tag", (ContextBundle(bundle_id="p", content={"text": "private"}),)
        )

    @asynccontextmanager
    async def mount(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        async def reject(
            query: ExtensionQuery, bundles: tuple[ContextBundle, ...]
        ) -> tuple[str, ...]:
            return ()

        yield ExtensionHooks(select_material=reject)

    policy = HarnessExtension(
        ExtensionDescriptor(
            schema_version=2, name="policy", version="1", hooks=(), provides_selector="filter"
        ),
        mount,
    )
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="ok")),))
    async with SessionEngine(
        tmp_path / "db",
        model=model,
        tools=ToolRegistry(()),
        context_compiler=ContextCompilerConfig(window_tokens=16000, output_tokens=256),
        token_counter=session_counter,
        extensions=HarnessExtensions(ExtensionAssembly((source(read, selector="filter"), policy))),
    ) as engine:
        await engine.create("s")
        result = await engine.send("s", "delta")
        view = await engine.store.explain_extension_request("s", result.run_id, 1)
        (report,) = [
            e.data.compilation
            for e in await engine.store.events("s")
            if isinstance(e.data, SessionContextCompiled)
        ]
    assert not model.requests[0].bundles and not view.choices
    assert (
        view.material_reads[0].candidate_count == 1
        and not view.selections[0].candidates[0].admitted
    )
    assert report.schema_version == 20 and "internal-tag" not in report.model_dump_json()
    forged = view.material_reads[0].model_copy(update={"candidates_sha256": "0" * 64})
    with pytest.raises(ValueError, match="differs from material candidates"):
        ContextCompilation.model_validate_json(
            report.model_copy(update={"material_reads": (forged,)}).model_dump_json()
        )
