"""两个独立接入场景共用的只读文件工具;资料来自固定提交,不包含评分答案。"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from weavria.core import ErrorCategory
from weavria.tools import ToolContext, ToolFault, ToolRegistry, TypedTool


class ReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, max_length=256)


class ReadOutput(BaseModel):
    path: str
    sha256: str
    text: str


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(default="", max_length=200)


class SearchOutput(BaseModel):
    paths: list[str]


@dataclass
class SnapshotCorpus:
    """真实资料的内存只读快照。校验散列后才允许工具访问;没有任意文件系统读口。"""

    documents: dict[str, str]
    fail_primary: bool = False
    reads: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    failures: int = 0

    @classmethod
    def load(cls, archive: Path, sources: dict[str, str]) -> "SnapshotCorpus":
        stored = TypeAdapter(dict[str, str]).validate_json(archive.read_bytes())
        documents: dict[str, str] = {}
        for path, expected in sources.items():
            # 只读专门维护的冻结原文,不读取工作目录、Git 元数据或宿主密钥。
            if not path.startswith(("docs/", "src/weavria/")) or ".." in Path(path).parts:
                raise ValueError("source is outside the frozen public corpus")
            data = stored[path].encode("utf-8")
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("frozen source digest differs")
            documents[path] = data.decode("utf-8")
        return cls(documents)

    def registry(self, *, documents: bool, bundle: bool = False) -> ToolRegistry:
        """宿主只提供工具;理解、选取、工作状态、记忆与完成判断都由 Harness 执行。"""
        primary = "read_document" if documents else "read_source"
        mirror = primary + "_mirror"
        content = dict(self.documents)
        if bundle:
            # 聚合完整原文,不复制填充、不编造事实。文件身份仍保留在各节标题。
            content["research_bundle.txt"] = "\n\n".join(
                f"===== {path} =====\n{text}" for path, text in sorted(content.items())
            )

        async def search(arguments: SearchInput, context: ToolContext) -> SearchOutput:
            terms = arguments.query.casefold().split()
            return SearchOutput(
                paths=[
                    path
                    for path, text in content.items()
                    if not terms or all(term in (path + "\n" + text).casefold() for term in terms)
                ][:40]
            )

        def reader(name: str) -> TypedTool[ReadInput, ReadOutput]:
            async def read(arguments: ReadInput, context: ToolContext) -> ReadOutput:
                if name == primary and self.fail_primary:
                    self.failures += 1
                    raise ToolFault(
                        ErrorCategory.TOOL_FAILURE,
                        "Primary archive unavailable; mirror remains available",
                        retryable=False,
                    )
                text = content.get(arguments.path)
                if text is None:
                    raise ToolFault(
                        ErrorCategory.TOOL_FAILURE,
                        "Unknown path; search the snapshot catalogue",
                        retryable=False,
                    )
                self.reads.append((name, arguments.path))
                return ReadOutput(
                    path=arguments.path, sha256=hashlib.sha256(text.encode()).hexdigest(), text=text
                )

            return TypedTool(
                name=name,
                version="1",
                input_type=ReadInput,
                output_type=ReadOutput,
                execute=read,
                description=(
                    "Read a complete UTF-8 file from the frozen "
                    + ("primary archive" if name == primary else "independent mirror")
                    + ". Paths come from search_snapshot; large results become artifacts."
                    + (" research_bundle.txt contains all original documents." if bundle else "")
                ),
            )

        return ToolRegistry(
            (
                TypedTool(
                    name="search_snapshot",
                    version="1",
                    description=(
                        "List snapshot paths matching all query words; "
                        "empty query lists all files. "
                        "Does not return file contents."
                    ),
                    input_type=SearchInput,
                    output_type=SearchOutput,
                    execute=search,
                ),
                reader(primary),
                reader(mirror),
            )
        )
