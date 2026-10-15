import asyncio
import subprocess
import sys
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from weavria.core import SessionResult


async def test_cli_continues_a_conversation_and_reuses_sources_across_separate_processes(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "examples" / "persistent_session.py"

    async def run(prompt: str, create: bool = False) -> SessionResult:
        args = [sys.executable, str(script), "--db", str(tmp_path / "db"), "--prompt", prompt]
        if create:
            args.append("--create")
        output = await asyncio.to_thread(
            subprocess.run, args, check=True, capture_output=True, text=True, timeout=10
        )
        return SessionResult.model_validate_json(output.stdout)

    first = await run('options ["alpha","beta","gamma"]', True)
    second = await run("second")
    assert first.run_id != second.run_id
    assert second.report.output == '"beta"'
    produced = await run("read catalogue.json")
    reused = await run("reuse")
    value = TypeAdapter(dict[str, JsonValue]).validate_json(reused.report.output or "{}")
    source = value["source"]
    assert isinstance(source, dict) and source["run_id"] == produced.run_id
    data = value["value"]
    assert isinstance(data, dict)
    document = data["document"]
    assert isinstance(document, dict) and document["components"]
