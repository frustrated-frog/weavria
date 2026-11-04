"""Docker 一次性执行后端。只暴露固定资源配置,模型不能构造 Docker 参数。"""

import asyncio
import json
import re
import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from .contracts import (
    SandboxCleanupError,
    SandboxError,
    SandboxLimits,
    SandboxProfile,
    SandboxRequest,
    SandboxResult,
)
from .process import command, settle


class DockerPythonSandbox:
    """Execute Python in a pinned local image; requires a trusted Docker daemon."""

    def __init__(
        self,
        *,
        image: str,
        limits: SandboxLimits | None = None,
        executable: str = "docker",
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._profile = SandboxProfile(image=image, limits=limits or SandboxLimits())
        self._executable = executable
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)

    @property
    def profile(self) -> SandboxProfile:
        return self._profile

    async def run(self, request: SandboxRequest) -> SandboxResult:
        request = SandboxRequest.model_validate_json(request.model_dump_json())
        limits = self.profile.limits
        if sum(len(f.text.encode()) for f in request.files) > limits.input_bytes:
            raise ValueError("sandbox inputs exceed the configured byte limit")
        suffix = self._id_factory()
        if len(suffix) != 32 or any(c not in "0123456789abcdef" for c in suffix):
            raise ValueError("sandbox IDs must contain 32 lowercase hexadecimal characters")
        name = "weavria-code-" + suffix
        owner = uuid.uuid4().hex
        # 脚本来自安装包,没有导入时 I/O;镜像、代码和资源身份随结果保存。
        guest = Path(__file__).with_name("_guest.py").read_text(encoding="utf-8")
        argv = (
            self._executable,
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--label",
            "weavria.sandbox=docker-python-1",
            "--label",
            "weavria.owner=" + owner,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--cap-add=SETUID",
            "--cap-add=SETGID",
            "--cap-add=CHOWN",
            "--security-opt=no-new-privileges",
            "--pids-limit",
            str(limits.processes),
            "--memory",
            f"{limits.memory_mb}m",
            "--memory-swap",
            f"{limits.memory_mb}m",
            "--cpus=1",
            "--ulimit",
            "nofile=128:128",
            "--ulimit",
            "core=0:0",
            "--tmpfs",
            f"/workspace:rw,noexec,nosuid,nodev,size={limits.workspace_mb}m,mode=755",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=8m,mode=1777",
            "--user=0:0",
            "--workdir=/workspace",
            "--log-driver=none",
            "-i",
            "--entrypoint=python3",
            self.profile.image,
            "-I",
            "-B",
            "-u",
            "-c",
            guest,
        )
        payload = {
            **request.model_dump(mode="json"),
            "limits": limits.model_dump(mode="json"),
            "profile_sha256": self.profile.sha256,
        }
        result: SandboxResult | None = None
        failure: BaseException | None = None
        try:
            response = await command(
                argv,
                data=json.dumps(payload, ensure_ascii=False).encode(),
                max_duration=limits.wall_seconds + 10,
                capacity=20000 + 6 * (2 * limits.log_bytes + limits.output_bytes),
            )
            if response.returncode == 137:
                result = self._failed("process_killed")
            elif response.returncode != 0:
                raise SandboxError("Docker or its execution supervisor failed")
            else:
                try:
                    result = SandboxResult.model_validate_json(response.stdout)
                except ValidationError as exc:
                    raise SandboxError("sandbox returned an invalid result protocol") from exc
                self._validate_result(request, result)
        except TimeoutError:
            result = self._failed("timed_out")
        except BaseException as exc:
            failure = exc
        # 不以 CLI 的退出代替容器终止。正常 --rm 后也确认不存在;失败时显式清理。
        try:
            await settle(asyncio.create_task(self._remove(name, owner)))
        except BaseException as exc:
            if isinstance(failure, asyncio.CancelledError):
                failure.add_note(f"Sandbox cleanup could not be confirmed: {name}")
                raise failure from exc
            raise
        if failure is not None:
            raise failure
        assert result is not None
        return result

    async def _remove(self, name: str, owner: str) -> None:
        try:
            inspected = await command(
                (
                    self._executable,
                    "container",
                    "inspect",
                    "--format",
                    '{{.Id}} {{ index .Config.Labels "weavria.owner" }}',
                    name,
                ),
                max_duration=5,
            )
            if inspected.returncode != 0:
                if (
                    b"No such container" in inspected.stderr
                    or b"No such object" in inspected.stderr
                ):
                    return
                raise SandboxCleanupError(f"Sandbox cleanup unconfirmed: {name}")
            identity = inspected.stdout.strip().split()
            if (
                len(identity) != 2
                or not re.fullmatch(rb"[a-f0-9]{64}", identity[0])
                or identity[1] != owner.encode()
            ):
                raise SandboxCleanupError(f"Refusing cleanup of another sandbox owner: {name}")
            # inspect 后名称可能被外部重新占用;只删除已核对 owner 的不可变容器 ID。
            response = await command(
                (self._executable, "rm", "--force", identity[0].decode()), max_duration=5
            )
        except (OSError, TimeoutError, SandboxError) as exc:
            raise SandboxCleanupError(f"Sandbox cleanup unconfirmed: {name}") from exc
        if response.returncode != 0 and b"No such container" not in response.stderr:
            raise SandboxCleanupError(f"Sandbox cleanup unconfirmed: {name}")

    def _failed(self, status: str) -> SandboxResult:
        return SandboxResult.model_validate_json(
            json.dumps(
                {
                    "status": status,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "profile_sha256": self.profile.sha256,
                }
            )
        )

    def _validate_result(self, request: SandboxRequest, result: SandboxResult) -> None:
        limits = self.profile.limits
        # 即使镜像/后端异常,容器协议也不能把未声明文件和无界数据写进 Artifact。
        if (
            result.profile_sha256 != self.profile.sha256
            or len(result.stdout.encode()) > 3 * limits.log_bytes
            or len(result.stderr.encode()) > 3 * limits.log_bytes
            or sum(len(f.text.encode()) for f in result.files) > limits.output_bytes
            or (
                result.status == "completed"
                and tuple(f.path for f in result.files) != request.outputs
            )
        ):
            raise SandboxError("sandbox result violated the execution profile")
