"""显式 Docker 集成测试;默认不启动容器、不访问网络或下载镜像。"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import cast

import pytest

from weavria.context import SessionTokenCounter
from weavria.sandbox import DockerPythonSandbox, SandboxLimits, SandboxRequest, TextFile
from weavria.sandbox.process import command

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(
        not os.getenv("WEAVRIA_TEST_DOCKER_IMAGE"),
        reason="set WEAVRIA_TEST_DOCKER_IMAGE to a pinned local image ID",
    ),
]


@pytest.fixture
def backend() -> DockerPythonSandbox:
    return DockerPythonSandbox(
        image=os.environ["WEAVRIA_TEST_DOCKER_IMAGE"], limits=SandboxLimits(wall_seconds=1.0)
    )


async def test_real_input_computation_and_output(backend: DockerPythonSandbox) -> None:
    result = await backend.run(
        SandboxRequest(
            code=(
                "import csv,json\nfrom pathlib import Path\n"
                "rows=list(csv.DictReader(Path('/workspace/inputs/data.csv').open()))\n"
                "value=sum(int(r['n']) for r in rows)\n"
                "Path('/workspace/outputs/result.json').write_text(json.dumps({'total':value}))\n"
                "print(value)"
            ),
            files=(TextFile(path="data.csv", text="n\n2\n3\n"),),
            outputs=("result.json",),
        )
    )
    assert result.status == "completed" and result.stdout == "5\n"
    assert result.files[0].text == '{"total": 5}'


@pytest.mark.parametrize(
    "code,status",
    (
        ("raise ValueError('bad column')", "code_error"),
        ("while True: pass", "timed_out"),
        ("import os\nos.close(1);os.close(2)\nwhile True: pass", "timed_out"),
        ("print('x'*1000000)", "output_limit"),
        ("import os\nos.symlink('/etc/passwd','/workspace/outputs/result.txt')", "export_error"),
        ("import os\nos.mkfifo('/workspace/outputs/result.txt')", "export_error"),
        (
            "from pathlib import Path\nPath('/workspace/outputs/result.txt').write_bytes(b'\\xff')",
            "export_error",
        ),
        (
            "from pathlib import Path\nPath('/workspace/outputs/result.txt').write_text('x'*9000)",
            "export_error",
        ),
    ),
)
async def test_failure_paths_never_publish_files(
    backend: DockerPythonSandbox, code: str, status: str
) -> None:
    result = await backend.run(SandboxRequest(code=code, outputs=("result.txt",)))
    assert result.status == status and not result.files


async def test_container_permissions_and_no_cross_call_state(backend: DockerPythonSandbox) -> None:
    code = """import os,socket
from pathlib import Path
assert os.getuid()==65534
assert not Path('/var/run/docker.sock').exists()
for path in ['/workspace/inputs/data.txt','/usr/forbidden']:
    try: Path(path).write_text('overwrite')
    except OSError: pass
    else: raise AssertionError('writable protected path')
try: socket.create_connection(('1.1.1.1',80), timeout=.1)
except OSError: pass
else: raise AssertionError('network was available')
Path('/workspace/work/leftover').write_text('old state')
print('isolated')
"""
    result = await backend.run(
        SandboxRequest(code=code, files=(TextFile(path="data.txt", text="original"),))
    )
    assert result.status == "completed" and result.stdout == "isolated\n"
    result = await backend.run(
        SandboxRequest(
            code="from pathlib import Path\nassert not Path('/workspace/work/leftover').exists()"
        )
    )
    assert result.status == "completed"


async def test_cancel_removes_container_and_all_children() -> None:
    suffix = uuid.uuid4().hex
    backend = DockerPythonSandbox(
        image=os.environ["WEAVRIA_TEST_DOCKER_IMAGE"],
        limits=SandboxLimits(wall_seconds=30.0),
        id_factory=lambda: suffix,
    )
    task = asyncio.create_task(
        backend.run(
            SandboxRequest(
                code=(
                    "import subprocess,time\n"
                    "subprocess.Popen(['python3','-c','import time; time.sleep(30)'])\n"
                    "time.sleep(30)"
                )
            )
        )
    )
    name = "weavria-code-" + suffix
    try:
        async with asyncio.timeout(5):
            while True:
                found = await command(
                    ("docker", "ps", "--filter", "name=" + name, "--format", "{{.Names}}")
                )
                if name.encode() in found.stdout:
                    break
                await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        found = await command(
            ("docker", "ps", "-a", "--filter", "name=" + name, "--format", "{{.Names}}")
        )
        assert not found.stdout.strip()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_harness_corrects_code_and_verifies_persisted_export(
    tmp_path: Path,
    backend: DockerPythonSandbox,
    session_counter: SessionTokenCounter,
) -> None:
    # 模型替身先写错变量,再根据正式交付的 stderr 改正。代码在真实 Docker 中运行,
    # 结果必须经过正常 ToolRecord/Artifact/SQLite/Completion,不把 Fake 当能力收益。
    import json

    from examples.sandbox_execution import run_task
    from weavria.core import (
        FinishAction,
        ModelTurn,
        SessionRequest,
        ToolBatch,
        ToolCallAction,
        ToolExchange,
    )
    from weavria.models import ModelCapabilities

    class Model:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)
        calls = 0

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            self.calls += 1
            if self.calls == 1:
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="read", tool_name="read_source_bundle", arguments={}
                            ),
                        )
                    )
                )
            exchange = next(
                h
                for h in request.current.history
                if isinstance(h, ToolExchange) and h.action.call_id == "read"
            )
            artifact = json.loads(exchange.model_dump_json())["observation"]["output"][
                "artifact_id"
            ]
            if self.calls == 2:
                code = "undefined_variable"
            elif self.calls == 3:
                assert "NameError" in request.model_dump_json()
                code = """import ast,json
from pathlib import Path
files=json.loads(Path('/workspace/inputs/sources.json').read_text())
result=[]
for file in sorted(files,key=lambda f:f['path']):
    nodes=list(ast.walk(ast.parse(file['text'])))
    result.append({
        'path':file['path'],
        'functions':sum(isinstance(n,ast.FunctionDef) for n in nodes),
        'async_functions':sum(isinstance(n,ast.AsyncFunctionDef) for n in nodes),
        'classes':sum(isinstance(n,ast.ClassDef) for n in nodes),
    })
Path('/workspace/outputs/statistics.json').write_text(json.dumps(result))
"""
            else:
                return ModelTurn(decision=FinishAction(output="statistics.json is ready"))
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id=f"code{self.calls}",
                            tool_name="execute_python",
                            arguments={
                                "code": code,
                                "inputs": [
                                    {
                                        "artifact_id": artifact,
                                        "pointer": "/files",
                                        "path": "sources.json",
                                        "format": "json",
                                    }
                                ],
                                "outputs": ["statistics.json"],
                            },
                        ),
                    )
                )
            )

    model = Model()
    result = await run_task(
        tmp_path / "run.sqlite", backend=backend, model=model, counter=session_counter
    )
    assert result["status"] == "completed"
    attempts = cast(list[dict[str, object]], result["code_attempts"])
    assert [a["status"] for a in attempts] == ["code_error", "completed"]
    assert model.calls == 4


async def test_name_collision_never_removes_a_container_owned_by_another_call() -> None:
    from weavria.sandbox import SandboxCleanupError

    suffix = uuid.uuid4().hex
    name = "weavria-code-" + suffix
    image = os.environ["WEAVRIA_TEST_DOCKER_IMAGE"]
    created = await command(
        (
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            name,
            "--label",
            "weavria.owner=another",
            "--entrypoint=python3",
            image,
            "-c",
            "import time;time.sleep(30)",
        )
    )
    assert created.returncode == 0
    try:
        backend = DockerPythonSandbox(image=image, id_factory=lambda: suffix)
        with pytest.raises(SandboxCleanupError):
            await backend.run(SandboxRequest(code="pass"))
        existing = await command(
            ("docker", "ps", "--filter", "name=" + name, "--format", "{{.Names}}")
        )
        assert existing.stdout.strip() == name.encode()
    finally:
        await command(("docker", "rm", "--force", name))
