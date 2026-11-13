"""固定实际模型请求,只替换资料展示方式;仅观察下一步决策,不执行工具或计作任务成功。"""

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from pydantic import TypeAdapter

from examples.live_memory_formation import Transport
from examples.work_recheck import Case, Document, load_dataset
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import HistoryEntry, SessionRequest, ToolExchange, ToolSucceededObservation
from weavria.models import (
    OpenAICompatibleConfig,
    OpenAICompatibleSessionProvider,
    OpenAICompatibleTokenCounter,
    call_session_model,
)


async def full_source_view(request: SessionRequest, case: Case) -> SessionRequest:
    """验证冻结材料摘要,只还原已有 read_source 结果;不补尚未读取的 current。

    Task、需求状态、工作图、反馈、工具目录全部不改。即使其中有已知的范围
    缺陷也保留,才能把本探针限定为表示方式的单因素干预。
    """
    store = InMemoryArtifactStore()
    history: list[HistoryEntry] = []
    for item in request.current.history:
        if not isinstance(item, ToolExchange) or item.action.tool_name != "read_source":
            history.append(item)
            continue
        if not isinstance(item.observation, ToolSucceededObservation):
            history.append(item)
            continue
        output = item.observation.output
        version = item.action.arguments.get("version")
        if not isinstance(output, dict) or version not in {"original", "current"}:
            raise ValueError("probe requires a recognized source result")
        assert isinstance(version, str)
        value = Document(version=version, text=getattr(case, version)).model_dump(mode="json")
        snapshot = await store.put("probe", item.action.call_id, value)
        if output.get("sha256") != snapshot.reference.sha256:
            raise ValueError("source digest differs from the frozen corpus")
        raw = {key: output[key] for key in ("artifact_id", "sha256", "size_bytes")}
        raw["data"] = value
        history.append(
            item.model_copy(
                update={"observation": item.observation.model_copy(update={"output": raw})}
            )
        )
    return SessionRequest.model_validate(
        request.model_copy(
            update={"current": request.current.model_copy(update={"history": tuple(history)})}
        )
    )


async def probe(args: argparse.Namespace) -> None:
    source_bytes = await asyncio.to_thread(args.requests.read_bytes)
    requests: tuple[SessionRequest, ...] = TypeAdapter(tuple[SessionRequest, ...]).validate_json(
        source_bytes
    )
    call: int = args.call
    request = requests[call - 1]
    case = next(c for c in load_dataset(args.dataset).cases if c.id == args.case)
    full = await full_source_view(request, case)
    variants = [("captured", request), ("full_original", full)]
    config = OpenAICompatibleConfig(
        model=os.environ.get("WEAVRIA_MODEL_NAME", "offline"),
        max_output_tokens=4096,
        temperature=0,
        context_window_tokens=64000,
        max_request_bytes=128000,
    )
    counter = OpenAICompatibleTokenCounter(config, encoding="cl100k_base")
    report: dict[str, object] = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "call": args.call,
        "model": asdict(config),
        "variants": [
            {
                "name": name,
                "request_sha256": hashlib.sha256(r.model_dump_json().encode()).hexdigest(),
                "input_tokens": counter.count(r),
                "request_bytes": counter.session_request_bytes(r),
            }
            for name, r in variants
        ],
        "scope": "仅下一步模型行动,不执行工具,不证明完成率或引用正确性。",
    }
    rows: list[dict[str, object]] = []
    report["rows"] = rows
    if args.live:
        options: dict[str, object] = {}
        if effort := os.environ.get("WEAVRIA_MODEL_REASONING_EFFORT"):
            options["reasoning_effort"] = effort
        report["options"] = options
        transport = Transport(
            os.environ["WEAVRIA_MODEL_BASE_URL"],
            os.environ.get("WEAVRIA_MODEL_API_KEY", ""),
            60,
            options=options,
        )
        provider = OpenAICompatibleSessionProvider(config, transport=transport)
        try:
            for repeat in range(args.repeats):
                order = variants if repeat % 2 == 0 else list(reversed(variants))
                for name, candidate in order:
                    row: dict[str, object] = {"repeat": repeat, "variant": name}
                    calls = transport.calls
                    try:
                        turn = await call_session_model(provider, candidate, max_duration=60)
                        row["decision"] = turn.model_dump(mode="json")
                    except Exception as exc:
                        row["error"] = type(exc).__name__
                    row["provider_observations"] = transport.observations[calls:]
                    rows.append(row)
                    print(
                        json.dumps(
                            {k: v for k, v in row.items() if k != "provider_observations"},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            await transport.client.aclose()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--call", type=int, choices=range(1, 17), default=2)
    parser.add_argument("--case", default="code_revision")
    parser.add_argument(
        "--dataset", type=Path, default=Path("tests/fixtures/work_recheck/tasks.json")
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    parser.add_argument(
        "--output", type=Path, default=Path(".weavria/work-recheck/context-probe.json")
    )
    asyncio.run(probe(parser.parse_args()))


if __name__ == "__main__":
    main()
