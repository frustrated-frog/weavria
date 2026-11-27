"""跨轮次只复用当前工具能独立核验的完整输出。历史材料不自行成为新义务的证据。"""

import asyncio
from pathlib import Path

import pytest
from pydantic import JsonValue

from examples.m1_corpus import SnapshotCorpus
from weavria.artifacts import ArtifactSnapshot
from weavria.core import (
    FinishAction,
    ModelTurn,
    RunStatus,
    ToolAuthorization,
    ToolBatch,
    ToolCallAction,
)
from weavria.models import FakeSessionProvider
from weavria.sessions import SessionEngine
from weavria.tools import PolicyDecision, ToolDescriptor


def reader(*paths: str) -> FakeSessionProvider:
    decisions = [
        ModelTurn(
            decision=ToolBatch(
                calls=(
                    ToolCallAction(
                        call_id=f"read-{i}", tool_name="read_document", arguments={"path": path}
                    ),
                )
            )
        )
        for i, path in enumerate(paths)
    ]
    return FakeSessionProvider((*decisions, ModelTurn(decision=FinishAction(output="done"))))


async def test_reopened_run_reuses_only_requested_current_editions(tmp_path: Path) -> None:
    db = tmp_path / "data.db"
    files = {"docs/base": "old", "docs/lock": "stable", "docs/alias": "same"}
    original = SnapshotCorpus(files)
    async with SessionEngine(
        db, model=reader(*files), tools=original.registry(documents=True)
    ) as e:
        await e.create("s")
        first = await e.send("s", "check original", input_id="one")
        assert first.report.status is RunStatus.COMPLETED
    # 同一文件的后缀别名返回逐字相同的输出, 但当前授权参数不同。
    amended = SnapshotCorpus(
        files | {"docs/base": "new", "docs/new": "additional"}, reuse_receipts=True
    )
    async with SessionEngine(
        db,
        model=reader("docs/lock", "docs/base", "docs/new", "alias"),
        tools=amended.registry(documents=True),
    ) as e:
        second = await e.send("s", "check revised", input_id="two")
        assert second.report.status is RunStatus.COMPLETED
        view = await e.store.turn("s", second.run_id)
        assert view is not None
        assert amended.reused == ["docs/lock"]
        assert [p for _, p in amended.reads] == ["docs/base", "docs/new", "docs/alias"]
        assert len(view.source_uses) == 1
        use = view.source_uses[0]
        assert (use.consumer_run_id, use.consumer_call_id, use.source.run_id) == (
            second.run_id,
            "read-0",
            first.run_id,
        )
        ref = view.calls[0].artifact
        assert ref and ref.run_id == second.run_id and ref.sha256 == use.source.sha256
        assert ref.artifact_id != use.source.artifact_id
        assert await e.send("s", "check revised", input_id="two") == second
        assert amended.reused == ["docs/lock"]


class Policy:
    def __init__(self, version: str, allow: bool):
        self.version, self.allow = version, allow

    async def authorize(
        self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
    ) -> PolicyDecision:
        return PolicyDecision(self.allow, self.version, "test")


@pytest.mark.parametrize(
    ("session", "version", "allow"), [("other", "a", True), ("s", "b", True), ("s", "a", False)]
)
async def test_session_boundary_and_current_policy(
    tmp_path: Path, session: str, version: str, allow: bool
) -> None:
    db = tmp_path / "s.db"
    corpus = SnapshotCorpus({"doc": "stable"}, reuse_receipts=True)
    async with SessionEngine(
        db, model=reader("doc"), tools=corpus.registry(documents=True), policy=Policy("a", True)
    ) as e:
        await e.create("s")
        await e.send("s", "first", input_id="one")
    corpus.reads.clear()
    async with SessionEngine(
        db,
        model=reader("doc"),
        tools=corpus.registry(documents=True),
        policy=Policy(version, allow),
    ) as e:
        if session != "s":
            await e.create(session)
        result = await e.send(session, "second", input_id="two")
        view = await e.store.turn(session, result.run_id)
        assert view and not view.source_uses
        assert corpus.reused == []
        assert len(corpus.reads) == int(allow)


async def test_cancellation_during_verification_never_falls_back_to_external_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "cancel.db"
    corpus = SnapshotCorpus({"doc": "stable"}, reuse_receipts=True)
    async with SessionEngine(db, model=reader("doc"), tools=corpus.registry(documents=True)) as e:
        await e.create("s")
        await e.send("s", "first", input_id="one")
    corpus.reads.clear()
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def block(
        run_id: str, call_id: str, tool_name: str, authorization: ToolAuthorization, digest: str
    ) -> ArtifactSnapshot | None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return None

    async with SessionEngine(db, model=reader("doc"), tools=corpus.registry(documents=True)) as e:
        monkeypatch.setattr(e.store, "prior_result", block)
        running = asyncio.create_task(e.send("s", "revise", input_id="two"))
        await asyncio.wait_for(entered.wait(), timeout=3)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert stopped.is_set() and not corpus.reused and not corpus.reads
        page = await e.store.history("s")
        assert not page[-1].source_uses


async def test_link_is_a_visible_pointer_and_does_not_create_a_source_receipt(
    tmp_path: Path,
) -> None:
    doc = "docs/guide/index.md"
    target = "docs/spec.md"
    text = (
        "[spec](../spec.md#details) [dup](../spec.md) ![img](img.md) "
        "[external](https://example.invalid/spec.md) [self](index.md) "
        "[missing](../private) [invalid](https://[) [absolute](/docs/spec.md)"
    )
    corpus = SnapshotCorpus(
        {doc: text, target: "official", "docs/guide/img.md": "image"}, source_links=True
    )
    async with SessionEngine(
        tmp_path / "links.db", model=reader(doc), tools=corpus.registry(documents=True)
    ) as e:
        await e.create("s")
        result = await e.send("s", "read the guide", input_id="one")
        view = await e.store.turn("s", result.run_id)
        assert view and len(view.calls) == 1 and not view.source_uses
        ref = view.calls[0].artifact
        assert ref
        value = (await e.store.get(result.run_id, ref.artifact_id)).value
        assert isinstance(value, dict) and value["referenced_paths"] == [target]
        assert [path for _, path in corpus.reads] == [doc]
