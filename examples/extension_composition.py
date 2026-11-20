"""比较两种宿主装配。实际发送请求后关闭 Harness,用 SQLite 解释材料的去留。"""

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.harness_extensions import document_extension
from weavria.context import ContextCompilerConfig
from weavria.core import ContextBundle, FinishAction, ModelTurn
from weavria.harness import (
    AgentHarness,
    ExtensionAssembly,
    ExtensionDescriptor,
    ExtensionHooks,
    ExtensionQuery,
    ExtensionScope,
    HarnessConfig,
    HarnessExtension,
    HarnessExtensions,
)
from weavria.models import FakeSessionProvider, OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.sessions import SQLiteSessionStore
from weavria.tools import ToolRegistry


def paragraph_selector(*, deduplicate: bool) -> HarnessExtension:
    """同一能力的两个实现。去重只依据此次取材的段落正文。"""

    @asynccontextmanager
    async def mount(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        async def select(
            query: ExtensionQuery, bundles: tuple[ContextBundle, ...]
        ) -> tuple[str, ...]:
            seen: set[str] = set()
            selected: list[str] = []
            for bundle in bundles:
                text = bundle.content.get("text")
                if deduplicate and isinstance(text, str):
                    if text in seen:
                        continue
                    seen.add(text)
                selected.append(bundle.bundle_id)
            return tuple(selected)

        yield ExtensionHooks(select_material=select)

    return HarnessExtension(
        ExtensionDescriptor(
            schema_version=2,
            name="unique" if deduplicate else "all",
            version="1",
            hooks=(),
            provides_selector="paragraphs",
        ),
        mount,
    )


def compose_documents(
    documents: Mapping[str, str], *, provider: str, reverse: bool = False
) -> ExtensionAssembly:
    source = document_extension(documents)
    consumer = HarnessExtension(
        ExtensionDescriptor(
            schema_version=2,
            name=source.descriptor.name,
            version=source.descriptor.version,
            hooks=("before_compile",),
            uses_selector="paragraphs",
            source_byte_limit=100_000,
        ),
        source.mount,
    )
    extensions = (
        consumer,
        paragraph_selector(deduplicate=True),
        paragraph_selector(deduplicate=False),
    )
    return ExtensionAssembly(
        extensions[::-1] if reverse else extensions, select={"paragraphs": provider}
    )


async def main() -> None:
    source = Path(__file__).parents[1] / "docs/decisions/0085-只读任务结算边界恢复.md"
    paragraph = next(part for part in source.read_text().split("\n\n") if "SessionEngine" in part)
    documents = {"manual": paragraph, "mirror": paragraph}
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="ok")),) * 2)
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="local", max_output_tokens=256), encoding="cl100k_base"
    )
    sessions = {key: compose_documents(documents, provider=key) for key in ("all", "unique")}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sessions.db"
        runs: dict[str, str] = {}
        async with AgentHarness.from_config(
            path,
            model=model,
            tools=ToolRegistry(()),
            token_counter=counter,
            config=HarnessConfig(
                context=ContextCompilerConfig(window_tokens=16384, output_tokens=256),
                task_understanding=False,
                configuration_id="composition/1",
            ),
            extensions=HarnessExtensions(sessions=sessions),
        ) as agent:
            for key in sessions:
                await agent.create(key)
                run = await agent.send(key, "SessionEngine")
                runs[key] = run.session.run_id
        async with SQLiteSessionStore(path) as store:
            for key, run_id in runs.items():
                view = await store.explain_extension_request(key, run_id, 1)
                assert view.plan is not None
                print(
                    json.dumps(
                        {
                            "binding": view.plan.bindings[0].model_dump(mode="json"),
                            "admitted": [
                                c.bundle_id for c in view.selections[0].candidates if c.admitted
                            ],
                            "filtered": [
                                c.bundle_id for c in view.selections[0].candidates if not c.admitted
                            ],
                            "delivered": [
                                c.source_id for c in view.choices if c.representation == "raw"
                            ],
                            "input_tokens": view.input_tokens,
                        },
                        ensure_ascii=False,
                    )
                )


if __name__ == "__main__":
    asyncio.run(main())
