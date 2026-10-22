"""Offline source-delivery experiment with actual BPE costs, no model API calls."""

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from weavria.evaluation.context_compiler import ContextProbe, evaluate_context_compiler
from weavria.models import OpenAICompatibleConfig, OpenAICompatibleTokenCounter


def load_cases(path: Path) -> tuple[ContextProbe, ...]:
    payload: dict[str, Any] = json.loads(path.read_text())
    corpus: dict[str, str] = {key: value["text"] for key, value in payload["corpus"].items()}
    for key, source in payload["corpus"].items():
        if hashlib.sha256(corpus[key].encode()).hexdigest() != source["sha256"]:
            raise ValueError("frozen corpus digest changed")
    return tuple(
        ContextProbe(
            case["name"],
            case["query"],
            tuple(corpus[key] for key in case["history"]),
            tuple(corpus[key] for key in case["documents"]),
            tuple(case["expected_phrases"]),
        )
        for case in payload["cases"]
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare context information delivery, not LLM task success"
    )
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--window-tokens", type=int, default=2600)
    parser.add_argument("--max-request-bytes", type=int, default=13000)
    args = parser.parse_args()
    corpus_path = Path(__file__).parent / "data" / "context_compiler" / "cases.json"
    counter = OpenAICompatibleTokenCounter(
        OpenAICompatibleConfig(model="offline-probe", max_output_tokens=256), encoding=args.encoding
    )
    results = await evaluate_context_compiler(
        load_cases(corpus_path),
        counter=counter,
        window_tokens=args.window_tokens,
        max_request_bytes=args.max_request_bytes,
    )
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for result in results:
        item = totals[result.policy]
        item["cases"] += 1
        item["compiled"] += result.status == "compiled"
        item["fully_retained_cases"] += result.retained == result.expected
        item["retained_phrases"] += result.retained
        item["expected_phrases"] += result.expected
        item["input_tokens"] += result.input_tokens
        item["peak_input_tokens"] = max(item["peak_input_tokens"], result.input_tokens)
        item["elapsed_ms"] += result.elapsed_ms
    print(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "context-source-delivery/1",
                "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                "counter": counter.version,
                "window_tokens": args.window_tokens,
                "max_request_bytes": args.max_request_bytes,
                "evidence_scope": "source phrase delivery only; no LLM task quality claims",
                "summary": dict(totals),
                "results": [asdict(result) for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
