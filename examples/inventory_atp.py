"""相同库存问题的双向验收。默认离线, --live 在两个独立会话中调用真实模型。"""

import argparse
import asyncio
import json
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from examples.dependency_migration import unique_object
from examples.inventory_time import ROOT, PromiseReport, documents, inventory_config, live_provider
from examples.m1_corpus import ReadOutput
from examples.upgrade_review import ReviewTask, replay, sha, turn
from weavria.artifacts import ArtifactSnapshot
from weavria.core import ToolRecord
from weavria.harness import (
    EvidenceEvaluation,
    HarnessEvidence,
    InformationRequirement,
    StructuralEvidenceExtractor,
)

AT = datetime(2026, 5, 14, 10, tzinfo=UTC)
REGISTER = "source-register.json"
PUBLICATION = "allocation.json"
PIN = "b807ca68b49d81154e33686a6a8bcfb078989b478ed4064e3cd0eb7ad8e656c9"


class Publication(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    issuer: Literal["warehouse-allocation"]
    sku: Literal["S4"]
    effective: AwareDatetime
    expires: AwareDatetime
    east: int = Field(ge=0)
    west: int = Field(ge=0)


def materials(available: bool) -> dict[str, str]:
    archive = documents(True)
    if available:
        archive[PUBLICATION] = (ROOT / PUBLICATION).read_text()
        if sha(archive[PUBLICATION]) != PIN:
            raise ValueError("frozen allocation publication differs from issuer pin")
    # 由有限夹具枚举整个来源集合。清单只给范围、来源类型和摘要, 不复制件数。
    archive[REGISTER] = json.dumps(
        {
            "as_of": AT.isoformat(),
            "scope": "Complete set of warehouse sources supplied for this request",
            "sources": [
                {
                    "file": name,
                    "kind": "atp_publication" if name == PUBLICATION else "period_close",
                    "sha256": sha(body),
                }
                for name, body in archive.items()
            ],
        },
        sort_keys=True,
    )
    return archive


def read_digest(path: str, body: str) -> str:
    payload = ReadOutput(path=path, text=body, sha256=sha(body))
    return sha(
        json.dumps(payload.model_dump(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def has_receipt(record: ToolRecord, path: str, digest: str) -> bool:
    # ToolRecord 已校验成功回执、调用标识、Artifact 和授权一致。镜像可提供同一原文。
    auth, ref = record.authorization, record.artifact
    return (
        auth is not None
        and auth.allowed
        and auth.tool_version == "1"
        and auth.normalized_arguments == {"path": path}
        and record.exchange.action.tool_name in {"read_document", "read_document_mirror"}
        and ref is not None
        and ref.sha256 == digest
    )


class PublicationEvidence(StructuralEvidenceExtractor):
    """在本夹具的核验时刻, 只给仓储发布方固定摘要所指的原文补充时效证明。"""

    key = "fixture-warehouse-publication"
    version = "1"

    def __init__(self, at: datetime) -> None:
        if at.utcoffset() is None:
            raise ValueError("assessment time must be timezone aware")
        body = materials(True)[PUBLICATION]
        publication = Publication.model_validate_json(body)
        self.digest = read_digest(PUBLICATION, body)
        self.valid = publication.effective <= at < publication.expires

    def supports(self, requirement: InformationRequirement, record: ToolRecord) -> bool:
        return super().supports(requirement, record) and has_receipt(
            record, PUBLICATION, self.digest
        )

    async def evaluate(
        self, requirement: InformationRequirement, record: ToolRecord, snapshot: ArtifactSnapshot
    ) -> EvidenceEvaluation:
        if record.artifact != snapshot.reference or not self.supports(requirement, record):
            raise ValueError("publication does not match the authorized artifact")
        # 继续验证实际 JSON Pointer。有效期只证明取到此刻适用的原文, 不证明模型读对数量。
        structural = await super().evaluate(
            requirement.model_copy(update={"freshness": "stable"}), record, snapshot
        )
        return structural.model_copy(update={"freshness": "valid" if self.valid else "stale"})


def expected(available: bool) -> PromiseReport:
    # 独立人工答案, 禁止从工具输出拷贝期望值。东仓有证明的 0 与未知的 null 不能互换。
    if available:
        return PromiseReport(status="confirmed", east=0, west=31, needed="none")
    return PromiseReport(status="insufficient_evidence", east=None, west=None, needed="current_atp")


def scenario(available: bool, *, pilot: bool = False) -> ReviewTask:
    archive = materials(available)
    wanted = expected(available)
    proof_path = PUBLICATION if available else REGISTER
    proof_digest = read_digest(proof_path, archive[proof_path])

    def errors(output: str | None, revised: bool) -> list[str]:
        try:
            report = PromiseReport.model_validate(
                json.loads(output or "", object_pairs_hook=unique_object)
            )
        except ValueError:
            return ["invalid_report"]
        return [
            key
            for key in PromiseReport.model_fields
            if getattr(report, key) != getattr(wanted, key)
        ]

    def sources(records: tuple[ToolRecord, ...]) -> list[str]:
        if any(has_receipt(record, proof_path, proof_digest) for record in records):
            return []
        return ["publication_not_read" if available else "source_scope_not_read"]

    prompt_file = "promise-question-pilot.txt" if pilot else "promise-question.txt"
    return ReviewTask(
        (ROOT / prompt_file).read_text().strip(),
        "",
        lambda _: archive.copy(),
        errors,
        sources,
    )


async def run(args: argparse.Namespace) -> int:
    if args.replay:
        stages = replay(args.replay, scenario(args.branch == "published", pilot=args.pilot))
        print(
            json.dumps({k: v.model_dump() for k, v in stages.items()}, ensure_ascii=False, indent=2)
        )
        trace = stages.get("original")
        return int(
            set(stages) != {"original"}
            or trace is None
            or trace.delivery_errors != []
            or trace.source_errors != []
        )
    config = inventory_config("inventory-atp/1")
    meta: dict[str, object] = dict(
        version="inventory-atp/1" if args.pilot else "inventory-atp/2",
        question=sha(scenario(False, pilot=args.pilot).initial),
        at=AT.isoformat(),
        repeats=args.repeats,
        config=asdict(config),
        audit_requirements=True,
        extractor=PublicationEvidence.version,
        materials={str(a): {p: sha(t) for p, t in materials(a).items()} for a in (False, True)},
        answers=[sha(expected(a).model_dump_json()) for a in (False, True)],
    )
    if not args.live:
        print(json.dumps(meta, ensure_ascii=False))
        return 0
    args.state.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.state))
    print(directory, flush=True)
    evidence = HarnessEvidence(audit_requirements=True, extractors=(PublicationEvidence(AT),))
    model, transport = live_provider()
    meta.update(model=asdict(model), options=transport.options)
    rows: list[dict[str, object]] = []
    try:
        for pair in range(1, args.repeats + 1):
            # 两臂均从空会话开始。一臂失败也要运行另一臂, 不从成功轨迹抽样。
            for available in (False, True) if pair % 2 else (True, False):
                branch = "published" if available else "archive"
                row = await turn(
                    directory / f"{pair}-{branch}.db",
                    revised=False,
                    config=config,
                    model=model,
                    transport=transport,
                    task=scenario(available, pilot=args.pilot),
                    evidence=evidence,
                )
                row.update(pair=pair, branch=branch)
                rows.append(row)
                pending = directory / "results.json.tmp"
                pending.write_text(
                    json.dumps(dict(metadata=meta, rows=rows), ensure_ascii=False, indent=2)
                )
                pending.replace(directory / "results.json")
                print(pair, branch, row["status"], row["failures"], flush=True)
    finally:
        await transport.client.aclose()
    return int(any(not row["passed"] for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--replay", type=Path)
    parser.add_argument("--pilot", action="store_true", help="使用首批尚未澄清 needed 含义的问题")
    parser.add_argument("--branch", choices=("published", "archive"))
    parser.add_argument("--state", type=Path, default=Path(".weavria/inventory-atp"))
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    if bool(args.replay) != bool(args.branch):
        parser.error("--replay and --branch must be specified together")
    raise SystemExit(asyncio.run(run(args)))
