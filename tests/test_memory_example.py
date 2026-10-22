"""The documented CLI must deliver the temporal facts it reports as correct."""

import asyncio
import json
import sys
from pathlib import Path
from typing import cast

from pydantic import JsonValue


async def test_memory_example_cross_session_temporal_results() -> None:
    script = Path(__file__).parents[1] / "examples" / "memory_kernel.py"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        "--distractors",
        "8",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(10):
            stdout, stderr = await process.communicate()
        assert process.returncode == 0, stderr.decode()
        result = cast(dict[str, JsonValue], json.loads(stdout))
        cases = result["cases"]
        assert isinstance(cases, list) and len(cases) == 3
        assert all(isinstance(case, dict) and case["memory_correct"] is True for case in cases)
        assert (
            sum(
                isinstance(case, dict) and case["latest_value_baseline_correct"] is True
                for case in cases
            )
            == 1
        )
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()
