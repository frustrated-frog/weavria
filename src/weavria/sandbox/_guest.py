"""固定容器监督程序;作为源码传入镜像,不会在宿主执行生成代码。

监督程序与用户程序 UID 不同。root 只负责输入、降权启动和有界成果导出;
用户程序不能修改监督程序的状态或从其 /proc 文件描述符伪造结果。
"""

import json
import os
import selectors
import signal
import stat
import subprocess
import sys
import time
from typing import TypedDict, cast


class _InputFile(TypedDict):
    path: str
    text: str


class _Limits(TypedDict):
    wall_seconds: float
    log_bytes: int
    output_bytes: int


class _Request(TypedDict):
    code: str
    files: list[_InputFile]
    outputs: list[str]
    limits: _Limits
    profile_sha256: str


def export_file(path: str, limit: int) -> str:
    # 不跟随任何中间目录/最终文件的符号链接。读取时限仍受宿主容器期限约束。
    parts = path.split("/")
    fd = os.open("/workspace/outputs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        output = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            info = os.fstat(output)
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError("export is not a bounded regular file")
            chunks = bytearray()
            while len(chunks) <= limit:
                block = os.read(output, min(8192, limit + 1 - len(chunks)))
                if not block:
                    break
                chunks.extend(block)
            if len(chunks) > limit:
                raise ValueError("export grew beyond its bound")
            return bytes(chunks).decode("utf-8")
        finally:
            os.close(output)
    finally:
        os.close(fd)


def main() -> None:
    # stdin 由宿主已校验协议生成;这份最小程序只依赖镜像中的标准库。
    request = cast(_Request, json.load(sys.stdin))
    limits = request["limits"]
    for folder in ("inputs", "outputs", "work"):
        os.mkdir("/workspace/" + folder, 0o755)
    for folder in ("outputs", "work"):
        os.chown("/workspace/" + folder, 65534, 65534)
    for item in request["files"]:
        path = "/workspace/inputs/" + item["path"]
        os.makedirs(os.path.dirname(path), mode=0o755, exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(item["text"])
        os.chmod(path, 0o444)
    with open("/workspace/program.py", "w", encoding="utf-8") as stream:
        stream.write(request["code"])
    # 根目录和输入均不可写,用户程序只获得 work、outputs 与受限 tmpfs。
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-u", "/workspace/program.py"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/workspace/work",
        user=65534,
        group=65534,
        extra_groups=(),
        start_new_session=True,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/workspace/work", "TMPDIR": "/tmp"},
    )
    selector = selectors.DefaultSelector()
    assert process.stdout is not None and process.stderr is not None
    for pipe, key in ((process.stdout, "stdout"), (process.stderr, "stderr")):
        selector.register(pipe, selectors.EVENT_READ, key)
    logs = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + limits["wall_seconds"]
    status = "completed"
    message = ""
    while selector.get_map():
        if time.monotonic() >= deadline:
            status = "timed_out"
            break
        for key, _ in selector.select(timeout=min(0.05, max(0, deadline - time.monotonic()))):
            block = os.read(key.fd, 8192)
            if not block:
                selector.unregister(key.fileobj)
                continue
            log = logs[key.data]
            remaining = limits["log_bytes"] - len(log)
            log.extend(block[:remaining])
            if len(block) > remaining:
                status = "output_limit"
                break
        if status != "completed":
            break
    # 管道可以被用户主动关闭,进程仍在死循环;wait 同样必须受壁钟期限控制。
    if status == "completed":
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            status = "timed_out"
    if status != "completed":
        # root 不保留 CAP_KILL:切换信号发送子进程到相同 UID 后终止整个进程组。
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import os,signal; os.killpg(int(__import__('sys').argv[1]),signal.SIGKILL)",
                str(process.pid),
            ],
            user=65534,
            group=65534,
            extra_groups=(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
        process.wait(timeout=1)
    selector.close()
    for pipe in (process.stdout, process.stderr):
        if pipe and not pipe.closed:
            pipe.close()
    if status == "completed" and process.returncode != 0:
        # SIGKILL 本身不能证明 OOM,外部终止和主动自杀也可能产生这个退出码。
        status = "process_killed" if process.returncode == -signal.SIGKILL else "code_error"
    files: list[dict[str, str | int]] = []
    if status == "completed":
        remaining = limits["output_bytes"]
        for path in request["outputs"]:
            try:
                text = export_file(path, remaining)
                remaining -= len(text.encode())
                files.append({"schema_version": 1, "path": path, "text": text})
            except (OSError, ValueError) as exc:
                status = "export_error"
                files = []
                message = f"Cannot export {path}: {type(exc).__name__}"
                break
    # 只有可信监督程序写协议 stdout;用户输出来自另一个 UID 的管道。
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "exit_code": process.returncode,
                "stdout": bytes(logs["stdout"]).decode("utf-8", errors="replace"),
                "stderr": bytes(logs["stderr"]).decode("utf-8", errors="replace"),
                "files": files,
                "message": message,
                "profile_sha256": request["profile_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
