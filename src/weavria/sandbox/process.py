"""有界 CLI 通信与所有权清理;不使用 shell,不遗留后台 pipe reader。"""

import asyncio
import os
import signal
from dataclasses import dataclass

from .contracts import SandboxError


async def settle[T](task: asyncio.Task[T]) -> T:
    """延迟传播取消直到持有的任务结束;失败原因不会把取消变成普通错误。"""
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancelled = exc
        except Exception:
            break
    if cancelled is not None:
        if not task.cancelled() and task.exception() is not None:
            raise cancelled from task.exception()
        raise cancelled
    return task.result()


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


async def command(
    argv: tuple[str, ...], *, data: bytes = b"", max_duration: float = 10, capacity: int = 262144
) -> CommandResult:
    """Launch one CLI with bounded pipes, kill and reap it on every failure path."""
    process: asyncio.subprocess.Process | None = None
    readers: list[asyncio.Task[bytes]] = []

    async def read(stream: asyncio.StreamReader) -> bytes:
        chunks = bytearray()
        while block := await stream.read(8192):
            if len(chunks) + len(block) > capacity:
                raise SandboxError("sandbox CLI output exceeded its transport bound")
            chunks.extend(block)
        return bytes(chunks)

    async def cleanup() -> None:
        assert process is not None
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        # CLI 进程组已终止,此时只排空有限的内核/reader缓冲,避免 wait 管道死锁。
        await process.communicate()

    try:
        async with asyncio.timeout(max_duration):
            launch = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            )
            try:
                process = await asyncio.shield(launch)
            except asyncio.CancelledError as cancelled:
                # 取消可以发生在 fork 完成与赋值之间;先取得句柄才能可靠回收。
                try:
                    await settle(launch)
                except BaseException as exc:
                    if launch.done() and not launch.cancelled() and launch.exception() is None:
                        process = launch.result()
                    raise cancelled from exc
                process = launch.result()
                raise
            assert process is not None
            assert process.stdout and process.stderr and process.stdin
            readers = [
                asyncio.create_task(read(process.stdout)),
                asyncio.create_task(read(process.stderr)),
            ]
            try:
                process.stdin.write(data)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            process.stdin.close()
            stdout, stderr = await asyncio.gather(*readers)
            code = await process.wait()
            return CommandResult(code, stdout, stderr)
    finally:
        if process is not None:
            await settle(asyncio.create_task(cleanup()))
