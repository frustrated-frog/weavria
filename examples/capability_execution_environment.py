"""能力发现实验的真实本地工具;环境不接收任务、答案或预编排工作流。"""

import asyncio
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from weavria.artifacts import ArtifactSnapshot
from weavria.core import ErrorCategory, ToolRecord
from weavria.harness import EvidenceEvaluation, InformationRequirement, StructuralEvidenceExtractor
from weavria.tools import RegisteredTool, ToolContext, ToolFault, ToolRegistry, TypedTool


class Data(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def virtual_parts(value: str) -> tuple[str, ...]:
    """只接受规范的虚拟绝对路径;不能把宿主绝对路径交给文件系统。"""
    path = PurePosixPath(value)
    if (
        not value.startswith("/")
        or value.startswith("//")
        or str(path) != value
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or len(path.parts) < 2
    ):
        raise ValueError("expected a canonical virtual file path")
    return path.parts[1:]


def instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("time must explicitly use UTC")
    return parsed.astimezone(UTC)


def timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


class FileSeed(Data):
    hex: str = Field(max_length=131072)
    modified_at: str

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        bytes.fromhex(self.hex)
        instant(self.modified_at)
        return self


class Revision(Data):
    revision: int = Field(ge=1)
    text: str = Field(max_length=4000)


class Interval(Data):
    start: str
    end: str

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        start, end = instant(self.start), instant(self.end)
        if not start < end or end - start > timedelta(days=1):
            raise ValueError("interval must have positive duration of at most one day")
        return self


class EnvironmentSeed(Data):
    schema_version: Literal[1]
    files: dict[str, FileSeed]
    assets: dict[str, str]
    archives: dict[str, tuple[Revision, ...]]
    calendars: dict[str, tuple[Interval, ...]]
    domains: tuple[str, ...]
    views: tuple[str, ...]

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        for path in self.files:
            virtual_parts(path)
        for path in self.assets.values():
            if path not in self.files:
                raise ValueError("asset location must name a fixture file")
        for revisions in self.archives.values():
            ids = [item.revision for item in revisions]
            if not ids or ids != sorted(set(ids)):
                raise ValueError("archive revisions must be nonempty and ordered")
        return self


class FileInput(Data):
    path: str = Field(description="Canonical virtual absolute path of an existing local file.")


class DigestOutput(Data):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of all file bytes.")


class MetadataOutput(Data):
    size_bytes: int = Field(ge=0, description="Byte length, not file content or a checksum.")
    modified_at: str = Field(description="Last modification time in UTC.")


class ArchiveInput(Data):
    archive_id: str = Field(pattern=r"^ARC-[0-9]+$", description="Archive ID, not a file path.")


class TextOutput(Data):
    text: str = Field(description="Full text of the latest saved archive revision.")


class HistoryOutput(Data):
    revisions: tuple[Revision, ...] = Field(description="Saved revisions in ascending order.")


class AssetInput(Data):
    asset_id: str = Field(description="Exact asset identifier whose local location is needed.")


class LocationOutput(Data):
    path: str = Field(description="Current virtual file path; not file content or digest.")


class CalendarInput(Interval):
    person: str
    duration_minutes: int = Field(default=30, ge=1, le=1440)


class CalendarOutput(Data):
    intervals: tuple[Interval, ...]


class SnapshotInput(Data):
    subject: str = Field(default="local", description="Snapshot subject; this fixture has local.")


class SnapshotOutput(Data):
    source: str
    values: dict[str, JsonValue]


class LocalEnvironment:
    """小型只读实验环境,实际文件位于每次运行独立的临时目录。

    模型只能访问 manifest 中的文件。逐级 openat + O_NOFOLLOW 防止文件/祖先
    链接逃逸;不承诺阻止有权限重命名整个根目录的恶意宿主。文件最多 64 KiB,
    同步读取有硬上限,不创建在父请求取消后继续工作的线程。
    """

    def __init__(self, root: Path, seed: EnvironmentSeed) -> None:
        self.root = root
        # 序列化隔离可变字典,避免外部修改 seed 改变当前运行的世界。
        self.seed = EnvironmentSeed.model_validate_json(seed.model_dump_json())
        for path, item in self.seed.files.items():
            target = root.joinpath(*virtual_parts(path))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes.fromhex(item.hex))
            seconds = instant(item.modified_at).timestamp()
            os.utime(target, (seconds, seconds))
        self._documents: dict[str, bytes] = {
            "/sources/assets.json": json.dumps(self.seed.assets).encode(),
            **{
                f"/sources/archive-{index}.json": HistoryOutput(revisions=revisions)
                .model_dump_json()
                .encode()
                for index, revisions in enumerate(self.seed.archives.values())
            },
            **{
                f"/sources/calendar-{index}.json": CalendarOutput(intervals=intervals)
                .model_dump_json()
                .encode()
                for index, intervals in enumerate(self.seed.calendars.values())
            },
        }
        for domain in self.seed.domains:
            for view in self.seed.views:
                name = f"read_{domain}_{view}"
                self._documents[f"/sources/{name}.json"] = (
                    SnapshotOutput(
                        source=name, values={"subject": "local", "domain": domain, "view": view}
                    )
                    .model_dump_json()
                    .encode()
                )
        if set(self._documents) & set(self.seed.files):
            raise ValueError("fixture paths cannot shadow internal data")
        for path, content in self._documents.items():
            if len(content) > 65536:
                raise ValueError("local data exceeds bounded read size")
            target = root.joinpath(*virtual_parts(path))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def _read(self, path: str, *, internal: bool = False) -> tuple[bytes, os.stat_result]:
        try:
            parts = virtual_parts(path)
            allowed = self._documents if internal else self.seed.files
            if path not in allowed:
                raise ValueError("path is outside the manifest")
            descriptors: list[int] = []
            try:
                descriptors.append(os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
                for part in parts[:-1]:
                    descriptors.append(
                        os.open(
                            part,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptors[-1],
                        )
                    )
                descriptor = os.open(
                    parts[-1],
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=descriptors[-1],
                )
                descriptors.append(descriptor)
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 65536:
                    raise ValueError("source is not a bounded regular file")
                # read 可以短读;累计检查也覆盖打开后文件增长的情况。
                data = bytearray()
                while chunk := os.read(descriptor, 65537 - len(data)):
                    data.extend(chunk)
                    if len(data) > 65536:
                        raise ValueError("source exceeded bounded read size")
                return bytes(data), metadata
            finally:
                for descriptor in reversed(descriptors):
                    os.close(descriptor)
        except (OSError, ValueError) as exc:
            raise ToolFault(
                ErrorCategory.SOURCE_UNAVAILABLE, "Local source is unavailable"
            ) from exc

    def _source(self, kind: str, key: str) -> bytes:
        inventory = self.seed.archives if kind == "archive" else self.seed.calendars
        try:
            index = tuple(inventory).index(key)
        except ValueError as exc:
            raise ToolFault(ErrorCategory.SOURCE_UNAVAILABLE, "Unknown local source") from exc
        return self._read(f"/sources/{kind}-{index}.json", internal=True)[0]

    async def digest(self, arguments: FileInput, context: ToolContext) -> DigestOutput:
        await asyncio.sleep(0)
        content, _ = self._read(arguments.path)
        return DigestOutput(sha256=hashlib.sha256(content).hexdigest())

    async def metadata(self, arguments: FileInput, context: ToolContext) -> MetadataOutput:
        await asyncio.sleep(0)
        _, metadata = self._read(arguments.path)
        return MetadataOutput(
            size_bytes=metadata.st_size,
            modified_at=timestamp(datetime.fromtimestamp(metadata.st_mtime, UTC)),
        )

    async def archive(self, arguments: ArchiveInput, context: ToolContext) -> TextOutput:
        await asyncio.sleep(0)
        history = HistoryOutput.model_validate_json(self._source("archive", arguments.archive_id))
        return TextOutput(text=history.revisions[-1].text)

    async def history(self, arguments: ArchiveInput, context: ToolContext) -> HistoryOutput:
        await asyncio.sleep(0)
        return HistoryOutput.model_validate_json(self._source("archive", arguments.archive_id))

    async def locate(self, arguments: AssetInput, context: ToolContext) -> LocationOutput:
        await asyncio.sleep(0)
        # 位置来自独立文件,调用闭包不含任何任务专用分支。
        locations: object = json.loads(self._read("/sources/assets.json", internal=True)[0])
        if not isinstance(locations, dict) or arguments.asset_id not in locations:
            raise ToolFault(ErrorCategory.SOURCE_UNAVAILABLE, "Unknown asset")
        return LocationOutput.model_validate({"path": locations[arguments.asset_id]})

    async def busy(self, arguments: CalendarInput, context: ToolContext) -> CalendarOutput:
        await asyncio.sleep(0)
        intervals = CalendarOutput.model_validate_json(self._source("calendar", arguments.person))
        start, end = instant(arguments.start), instant(arguments.end)
        return CalendarOutput(
            intervals=tuple(
                Interval(
                    start=timestamp(max(start, instant(item.start))),
                    end=timestamp(min(end, instant(item.end))),
                )
                for item in intervals.intervals
                if instant(item.start) < end and instant(item.end) > start
            )
        )

    async def free(self, arguments: CalendarInput, context: ToolContext) -> CalendarOutput:
        occupied = await self.busy(arguments, context)
        # 扫描排序后的忙碌区间;cursor 同时合并重叠/相接事件并裁剪窗口。
        # 输出的是满足时长的最大空闲窗口,不是任意枚举的预约开始时间。
        cursor, end = instant(arguments.start), instant(arguments.end)
        minimum = timedelta(minutes=arguments.duration_minutes)
        free: list[Interval] = []
        for item in sorted(occupied.intervals, key=lambda value: instant(value.start)):
            start, stop = instant(item.start), instant(item.end)
            if start - cursor >= minimum:
                free.append(Interval(start=timestamp(cursor), end=timestamp(start)))
            cursor = max(cursor, stop)
        if end - cursor >= minimum:
            free.append(Interval(start=timestamp(cursor), end=timestamp(end)))
        return CalendarOutput(intervals=tuple(free))

    def registry(self) -> ToolRegistry:
        tools: list[RegisteredTool] = [
            TypedTool(
                name="compute_file_checksum",
                version="1",
                description="Compute a content checksum from an existing file.",
                input_type=FileInput,
                output_type=DigestOutput,
                execute=self.digest,
            ),
            TypedTool(
                name="read_file_metadata",
                version="1",
                description="Inspect basic properties of a local file.",
                input_type=FileInput,
                output_type=MetadataOutput,
                execute=self.metadata,
            ),
            TypedTool(
                name="read_archived_text",
                version="1",
                description="Retrieve a stored archive document's current text.",
                input_type=ArchiveInput,
                output_type=TextOutput,
                execute=self.archive,
            ),
            TypedTool(
                name="read_document_history",
                version="1",
                description="Retrieve the full text of saved archive document revisions.",
                input_type=ArchiveInput,
                output_type=HistoryOutput,
                execute=self.history,
            ),
            TypedTool(
                name="locate_asset",
                version="1",
                description="Resolve an asset identifier to its current local file path.",
                input_type=AssetInput,
                output_type=LocationOutput,
                execute=self.locate,
            ),
            TypedTool(
                name="read_calendar_busy",
                version="1",
                description="Read occupied calendar intervals clipped to a UTC window.",
                input_type=CalendarInput,
                output_type=CalendarOutput,
                execute=self.busy,
            ),
            TypedTool(
                name="find_available_slots",
                version="1",
                description="Find all maximal free calendar windows at least duration_minutes "
                "long, within a UTC window. Merges overlapping busy events.",
                input_type=CalendarInput,
                output_type=CalendarOutput,
                execute=self.free,
            ),
        ]
        for domain in self.seed.domains:
            for view in self.seed.views:
                name = f"read_{domain}_{view}"

                async def snapshot(
                    arguments: SnapshotInput, context: ToolContext, *, source: str = name
                ) -> SnapshotOutput:
                    await asyncio.sleep(0)
                    if arguments.subject != "local":
                        raise ToolFault(ErrorCategory.SOURCE_UNAVAILABLE, "Unknown subject")
                    raw, _ = self._read(f"/sources/{source}.json", internal=True)
                    return SnapshotOutput.model_validate_json(raw)

                tools.append(
                    TypedTool(
                        name=name,
                        version="1",
                        description=f"Read the local {domain} {view} snapshot.",
                        input_type=SnapshotInput,
                        output_type=SnapshotOutput,
                        execute=snapshot,
                    )
                )
        return ToolRegistry(tuple(tools))


@contextmanager
def local_environment(seed: EnvironmentSeed) -> Generator[LocalEnvironment]:
    """成功、异常、超时或取消离开作用域时,统一回收本次执行的所有本地文件。"""
    with tempfile.TemporaryDirectory(prefix="weavria-capability-") as directory:
        yield LocalEnvironment(Path(directory), seed)


class LocalSnapshotEvidence(StructuralEvidenceExtractor):
    """只证明本地适配器无缓存读取的时效性,不证明答案或参数语义正确。

    当前环境在 Run 期间没有写工具,每次读取都访问当前本地快照,不复用前次
    Run 的结果。这个承诺属于此适配器,不能上提成所有成功工具都 fresh。
    所有必需字段仍经过通用结构检查;Need 保持 ACQUIRED 而非语义 SATISFIED。
    """

    key = "local-snapshot-json"
    version = "1"

    def __init__(self, registry: ToolRegistry) -> None:
        self._versions = {item.name: item.version for item in registry.descriptors}

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        authorization = record.authorization
        return (
            super().supports(requirement, record)
            and record.artifact is not None
            and authorization is not None
            and authorization.allowed
            and self._versions.get(record.exchange.action.tool_name) == authorization.tool_version
        )

    async def evaluate(
        self, requirement: InformationRequirement, record: ToolRecord, snapshot: ArtifactSnapshot
    ) -> EvidenceEvaluation:
        if not self.supports(requirement, record) or record.artifact != snapshot.reference:
            raise ValueError("freshness evidence must bind this adapter's accepted result")
        # 副本只传给字段检查器:时效性已经由上面的本地读取契约承担。原任务和
        # 原 Need 的 current 要求不变,缺字段/空值的结果类型也不被提升。
        return await super().evaluate(
            requirement.model_copy(update={"freshness": "stable"}), record, snapshot
        )
