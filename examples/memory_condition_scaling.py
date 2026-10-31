"""条件闭包的独立规模实验;模型解释成本与网络延迟不计入图计算时间。"""

import argparse
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

from weavria.memory import MemoryConditionLink, compare_memory_conditions


def measure(size: int, repeats: int) -> dict[str, object]:
    left = {f"c{i}": f"condition-{i}" for i in range(size)}
    right = {f"t{i}": f"restriction-{i}" for i in range(size)}
    # 稠密等价图覆盖上限边数,验证去重与循环不会增加语义权重。
    links = tuple(
        MemoryConditionLink(
            left_side="candidate",
            left_key=a,
            right_side="target",
            right_key=b,
            relation="equivalent",
        )
        for a in left
        for b in right
    )
    result = compare_memory_conditions(left, right, links)
    elapsed: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = compare_memory_conditions(left, right, links)
        elapsed.append((time.perf_counter() - started) * 1000)
    tracemalloc.start()
    compare_memory_conditions(left, right, links)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "conditions_per_side": size,
        "links": len(links),
        "repeats": repeats,
        "median_ms": statistics.median(elapsed),
        "peak_bytes": peak,
        "relation": result.relation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100000:
        parser.error("repeats must be within [1,100000]")
    source = Path(__file__).resolve().parents[1] / "src/weavria/memory/conditions.py"
    print(
        json.dumps(
            {
                "benchmark": "memory-condition-closure/1",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "rows": [measure(size, args.repeats) for size in (2, 8, 16)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
