"""同一 Harness 中的只读文档扩展和终态观察器。运行后可对照请求及落盘的轨迹。"""

import asyncio
import hashlib
import json
import re
import tempfile
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from weavria.context import ContextCompilerConfig
from weavria.core import ContextBundle, FinishAction, ModelTurn, SessionContextCompiled
from weavria.harness import (
    AgentHarness,
    ExtensionDescriptor,
    ExtensionHooks,
    ExtensionQuery,
    ExtensionScope,
    ExtensionTrace,
    HarnessConfig,
    HarnessExtension,
    HarnessExtensions,
)
from weavria.models import FakeSessionProvider, OpenAICompatibleConfig, OpenAICompatibleTokenCounter
from weavria.tools import ToolRegistry


def document_extension(documents: Mapping[str, str]) -> HarnessExtension:
    """宿主传入当前项目允许读取的小型文档快照。按查询选段,保留源文件和段号。"""
    files = tuple(sorted(documents.items()))
    version = hashlib.sha256(json.dumps(files, ensure_ascii=False).encode()).hexdigest()
    descriptor = ExtensionDescriptor(
        name="project-docs", version=f"paragraph-lookup/1.{version}", hooks=("before_compile",)
    )

    @asynccontextmanager
    async def mount(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        passages = tuple(
            (path, number, text, set(re.findall(r"\w+", text.casefold())))
            for path, contents in files
            for number, text in enumerate(contents.split("\n\n"), 1)
            if text.strip()
        )

        async def retrieve(query: ExtensionQuery) -> tuple[ContextBundle, ...]:
            terms = set(re.findall(r"\w+", " ".join((query.goal, *query.constraints)).casefold()))
            matches = sorted(
                ((len(terms & words), path, n, text) for path, n, text, words in passages),
                key=lambda row: (-row[0], row[1], row[2]),
            )
            return tuple(
                ContextBundle(
                    bundle_id=f"{path}#{n}", content={"source": path, "paragraph": n, "text": text}
                )
                for score, path, n, text in matches[:3]
                if score
            )

        yield ExtensionHooks(before_compile=retrieve)

    return HarnessExtension(descriptor, mount)


def trace_extension(records: list[ExtensionTrace]) -> HarnessExtension:
    @asynccontextmanager
    async def mount(scope: ExtensionScope) -> AsyncGenerator[ExtensionHooks]:
        async def observe(trace: ExtensionTrace) -> None:
            records.append(trace)

        yield ExtensionHooks(after_run=observe)

    return HarnessExtension(
        ExtensionDescriptor(name="trace", version="1", hooks=("after_run",)), mount
    )


async def main() -> None:
    source = Path(__file__).parents[1] / "docs/decisions/0085-只读任务结算边界恢复.md"
    docs = document_extension({source.name: source.read_text(encoding="utf-8")})
    records: list[ExtensionTrace] = []
    observer = trace_extension(records)
    extensions = HarnessExtensions(default=(observer,), sessions={"repository": (docs, observer)})
    config = HarnessConfig(
        context=ContextCompilerConfig(window_tokens=16384, output_tokens=256),
        configuration_id="extension-example/1",
        task_understanding=False,
    )
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="ok")),) * 2)
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="local", max_output_tokens=256), encoding="cl100k_base"
    )
    with tempfile.TemporaryDirectory() as directory:
        async with AgentHarness.from_config(
            Path(directory) / "sessions.db",
            config=config,
            model=model,
            tools=ToolRegistry(()),
            token_counter=counter,
            extensions=extensions,
        ) as harness:
            for session in ("repository", "baseline"):
                await harness.create(session)
                result = await harness.send(session, "SessionEngine resume SQLite")
                trace = await harness.sessions.store.extension_trace(session, result.session.run_id)
                assert records[-1] == trace
                (compiled,) = (
                    event.data.compilation
                    for event in await harness.sessions.store.events(session)
                    if isinstance(event.data, SessionContextCompiled)
                )
                print(
                    json.dumps(
                        {
                            "session": session,
                            "extensions": [d.name for d in trace.extensions],
                            "material": [b.bundle_id for b in model.requests[-1].bundles],
                            "prepared_requests": trace.prepared_requests,
                            "input_tokens": compiled.input_tokens,
                            "request_bytes": compiled.request_bytes,
                            "status": trace.status,
                        },
                        ensure_ascii=False,
                    )
                )


if __name__ == "__main__":
    asyncio.run(main())
