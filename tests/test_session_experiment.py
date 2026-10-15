import asyncio
import json
import subprocess
import sys
from pathlib import Path


async def test_frozen_context_ablation_uses_same_durable_seed_and_measures_retrieval_cost() -> None:
    script = Path(__file__).parents[1] / "examples" / "session_experiment.py"
    output = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    report = json.loads(output.stdout)
    assert report["dataset_version"] == "conversation/1" and len(report["dataset_sha256"]) == 64
    rows = {r["policy"]: r for r in report["results"]}
    assert rows["full"]["status"] == "budget_exhausted" and rows["full"]["model_calls"] == 0
    assert rows["recent"]["correct"] and rows["relevant"]["correct"]
    assert rows["recent"]["tool_calls"] == 3 and rows["relevant"]["tool_calls"] == 0
    assert rows["recent"]["input_bytes"] > rows["relevant"]["input_bytes"]
    assert all(r["peak_request_bytes"] <= 16000 and r["all_snapshots_match"] for r in rows.values())
