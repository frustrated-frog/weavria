"""两个独立接入场景共用的只读文件工具;资料来自固定提交,不包含评分答案。"""

import hashlib
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from weavria.core import ErrorCategory, task_view_digest
from weavria.tools import ToolContext, ToolFault, ToolRegistry, TypedTool


class ReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, max_length=256)


class ReadOutput(BaseModel):
    path: str
    sha256: str
    text: str


class LinkedReadOutput(ReadOutput):
    # 路径只是可导航的线索, 不能把链接当成目标文件的取证结果。
    referenced_paths: list[str]


def referenced_paths(path: str, text: str, documents: dict[str, str]) -> list[str]:
    """从简式 Markdown 行内链接中解析本快照可见的相对路径, 最多八个。"""
    links: list[str] = []
    if not path.endswith(".md"):
        return links
    for match in re.finditer(r"(?<!!)\[[^\]\n]{1,128}\]\(([^\s()]{1,512})\)", text):
        try:
            ref = urlsplit(match[1])
        except ValueError:
            continue
        relative = unquote(ref.path)
        if ref.scheme or ref.netloc or not relative or relative.startswith("/") or ref.query:
            continue
        target = posixpath.normpath(posixpath.join(posixpath.dirname(path), relative))
        if target != path and target in documents and target not in links:
            links.append(target)
            if len(links) == 8:
                break
    return links


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
    reuse_receipts: bool = False
    source_links: bool = False
    reused: list[str] = field(default_factory=list[str])

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
        source_links, reuse_receipts = self.source_links, self.reuse_receipts
        if bundle:
            # 聚合完整原文,不复制填充、不编造事实。文件身份仍保留在各节标题。
            content["research_bundle.txt"] = "\n\n".join(
                f"===== {path} =====\n{text}" for path, text in sorted(content.items())
            )

        # 此夹具的授权视图是不可变内存快照。模拟生产方在发布版本时准备结果摘要,
        # 度量时只计后端读取次数; 构建清单本身仍遍历正文。
        def file_result(path: str, text: str) -> ReadOutput:
            digest = hashlib.sha256(text.encode()).hexdigest()
            if source_links:
                return LinkedReadOutput(
                    path=path,
                    sha256=digest,
                    text=text,
                    referenced_paths=referenced_paths(path, text, content),
                )
            return ReadOutput(path=path, sha256=digest, text=text)

        digests = (
            {
                path: task_view_digest(file_result(path, text).model_dump(mode="json"))
                for path, text in content.items()
            }
            if reuse_receipts
            else {}
        )
        output_type = LinkedReadOutput if source_links else ReadOutput

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
                path = arguments.path
                if (
                    path not in content
                    and not path.startswith("/")
                    and ".." not in Path(path).parts
                ):
                    # 只在冻结索引中解析唯一的路径后缀, 输出始终保留可引用的完整路径。
                    matches = [item for item in content if item.endswith("/" + path)]
                    if len(matches) == 1:
                        path = matches[0]
                if path not in content:
                    raise ToolFault(
                        ErrorCategory.TOOL_FAILURE,
                        "Unknown path; search the snapshot catalogue",
                        retryable=False,
                    )
                if reuse_receipts:
                    prior = await context.reuse_verified_result(digests[path])
                    if prior is not None:
                        result = output_type.model_validate(prior.value, strict=True)
                        self.reused.append(path)
                        return result
                self.reads.append((name, path))
                return file_result(path, content[path])

            return TypedTool(
                name=name,
                version="2" if source_links else "1",
                input_type=ReadInput,
                output_type=output_type,
                execute=read,
                artifact_read_inputs=(),
                description=(
                    "Read a complete UTF-8 file from the frozen "
                    + ("primary archive" if name == primary else "independent mirror")
                    + ". Paths come from search_snapshot; large results become artifacts."
                    + (
                        " Follow relevant referenced_paths to check the underlying source "
                        "before citing it."
                        if source_links
                        else ""
                    )
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
