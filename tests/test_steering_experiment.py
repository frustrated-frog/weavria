import asyncio
import json
import subprocess
import sys
from pathlib import Path


async def test_live_goal_change_reuses_work_and_avoids_obsolete_reads() -> None:
    script = Path(__file__).parents[1] / "examples" / "steering_experiment.py"
    output = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    report = json.loads(output.stdout)
    assert report["dataset_version"] == "focus-change/1"
    baseline, steering = report["results"]
    assert baseline["verified"] and steering["verified"]
    assert baseline["answer"] == steering["answer"]
    assert baseline["file_reads"] == 7 and steering["file_reads"] == 3
    assert steering["obsolete_reads_after_update"] == 0
    assert steering["skipped_calls"] == 4 and steering["replans"] == 1
    assert steering["input_bytes"] < baseline["input_bytes"]
