"""对照逐轮读文件和条件读取。两个来源中只有一个在第三次请求前更新。"""

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from weavria.context import ContextCompilerConfig
from weavria.core import (
    ContextBundle,
    FinishAction,
    ModelTurn,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
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
from weavria.models import FakeSessionProvider, OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.sessions import SessionEngine, SQLiteSessionStore
from weavria.tools import ToolContext, ToolRegistry, TypedTool


@dataclass
class PublishedFile:
    """宿主发布内容寻址的小文件,读者抓取同一个修订的文件名与正文。"""

    name: str
    directory: Path
    revision: str
    full_reads: int = 0

    def extension(self, *, conditional: bool) -> HarnessExtension:
        @asynccontextmanager
        async def mount(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
            async def read(query: ExtensionQuery, previous: str | None) -> ConditionalMaterial:
                revision = self.revision
                if previous == revision:
                    return ConditionalMaterial(revision, None)
                text = await asyncio.to_thread(
                    (self.directory / revision).read_text, encoding="utf-8"
                )
                assert hashlib.sha256(text.encode()).hexdigest() == revision
                self.full_reads += 1
                return ConditionalMaterial(
                    revision, (ContextBundle(bundle_id="passage", content={"text": text}),)
                )

            async def live(query: ExtensionQuery) -> tuple[ContextBundle, ...]:
                result = await read(query, None)
                assert result.bundles is not None
                return result.bundles

            yield (
                ExtensionHooks(read_material=read)
                if conditional
                else ExtensionHooks(before_compile=live)
            )

        return HarnessExtension(
            ExtensionDescriptor(
                schema_version=3,
                name=self.name,
                version="1",
                hooks=("before_compile",),
                conditional_material=conditional,
            ),
            mount,
        )


class Lookup(BaseModel):
    number: int


async def lookup(args: Lookup, context: ToolContext) -> Lookup:
    return args


@dataclass(frozen=True)
class Comparison:
    full_reads: tuple[int, int]
    request_hashes: tuple[str, ...]
    outcomes: tuple[tuple[str, ...], ...]
    delivered: tuple[tuple[str, ...], ...]


async def run_case(
    directory: Path, *, conditional: bool, counter: OpenAICompatibleTokenCounter
) -> Comparison:
    root = Path(__file__).parents[1] / "docs/decisions"
    paragraphs = [
        part
        for part in (root / "0085-只读任务结算边界恢复.md")
        .read_text(encoding="utf-8")
        .split("\n\n")
        if "SessionEngine" in part or "SQLite" in part
    ]
    versions: list[str] = []
    for part in paragraphs[:3]:
        revision = hashlib.sha256(part.encode()).hexdigest()
        (directory / revision).write_text(part, encoding="utf-8")
        versions.append(revision)
    alpha = PublishedFile("manual", directory, versions[0])
    beta = PublishedFile("reference", directory, versions[1])

    class Script(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            action = await super().generate_session(request)
            # 模拟独立发布者在两次调用之间切换指针。工具只读,两组发布时间相同。
            if len(self.requests) == 2:
                alpha.revision = versions[2]
            return action

    model = Script(
        (
            *(
                ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id=f"call-{i}", tool_name="lookup", arguments={"number": i}
                            ),
                        )
                    )
                )
                for i in range(3)
            ),
            ModelTurn(decision=FinishAction(output="ok")),
        )
    )
    db = directory / ("conditional.db" if conditional else "baseline.db")
    registry = ToolRegistry(
        (
            TypedTool(
                name="lookup",
                version="1",
                description="Read a number",
                input_type=Lookup,
                output_type=Lookup,
                execute=lookup,
            ),
        )
    )
    assembly = ExtensionAssembly(
        tuple(source.extension(conditional=conditional) for source in (alpha, beta))
    )
    async with SessionEngine(
        db,
        model=model,
        tools=registry,
        id_generator=lambda: "fixed-run",
        extensions=HarnessExtensions(assembly),
        context_compiler=ContextCompilerConfig(window_tokens=16000, output_tokens=256),
        token_counter=counter,
    ) as engine:
        await engine.create("experiment")
        result = await engine.send("experiment", "SessionEngine SQLite", input_id="input")
        assert result.report.output == "ok" and len(model.requests) == 4

    async with SQLiteSessionStore(db) as store:
        views = [
            await store.explain_extension_request("experiment", result.run_id, call)
            for call in range(1, 5)
        ]
    hashes = tuple(hashlib.sha256(r.model_dump_json().encode()).hexdigest() for r in model.requests)
    assert hashes == tuple(view.request_sha256 for view in views)
    return Comparison(
        (alpha.full_reads, beta.full_reads),
        hashes,
        tuple(tuple(r.outcome for r in view.material_reads) for view in views),
        tuple(tuple(str(b.content["text"]) for b in r.bundles) for r in model.requests),
    )


async def main() -> None:
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="local", max_output_tokens=256), encoding="cl100k_base"
    )
    with TemporaryDirectory() as directory:
        baseline = await run_case(Path(directory), conditional=False, counter=counter)
        reuse = await run_case(Path(directory), conditional=True, counter=counter)
    assert baseline.request_hashes == reuse.request_hashes
    print(
        json.dumps(
            {
                "baseline_full_reads": baseline.full_reads,
                "conditional_full_reads": reuse.full_reads,
                "per_call": reuse.outcomes,
                "identical_requests": len(reuse.request_hashes),
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
