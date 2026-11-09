"""需求证据审查的可审计投影;语义判断仍是低信任模型推断。"""

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from ._schema import ContractModel

type EvidenceStatus = Literal[
    "supported", "missing", "conditional", "conflict", "unverified", "omitted"
]


class ContextEvidenceCheck(ContractModel):
    """定位被审查的成员,记录确定性条件计算;不记录模型私有推理。"""

    bundle_id: str = Field(min_length=1, max_length=256)
    member_id: str = Field(min_length=1, max_length=256)
    revision: int = Field(ge=1)
    relation: Literal["direct", "background", "conflict"]
    applicability: Literal[
        "equivalent",
        "candidate_entails_target",
        "target_entails_candidate",
        "disjoint",
        "unknown",
        "inconsistent",
    ]
    quote_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    condition_edges: tuple[
        tuple[
            Literal["candidate", "target"],
            str,
            Literal["candidate", "target"],
            str,
            Literal["equivalent", "entails", "disjoint"],
        ],
        ...,
    ] = Field(default=(), max_length=64)


class ContextEvidenceNeed(ContractModel):
    need_id: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=512)
    status: EvidenceStatus
    # 整族闭包是判定依赖,不是可以单独删除的引用装饰。
    bundle_ids: tuple[str, ...] = Field(default=(), max_length=100)
    checks: tuple[ContextEvidenceCheck, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def bindings(self) -> Self:
        if len(set(self.bundle_ids)) != len(self.bundle_ids):
            raise ValueError("evidence repeats a bundle dependency")
        if (
            self.status in {"supported", "conditional", "conflict", "omitted"}
            and not self.bundle_ids
        ):
            raise ValueError("evidence judgment requires complete bundle dependencies")
        if self.status in {"missing", "unverified"} and self.bundle_ids:
            raise ValueError("unverified evidence cannot release bundles")
        keys = [(c.bundle_id, c.member_id) for c in self.checks]
        if len(set(keys)) != len(keys):
            raise ValueError("evidence repeats a member check")
        return self


class ContextEvidenceDemand(ContractModel):
    demand_id: str = Field(min_length=1, max_length=256)
    needs: tuple[ContextEvidenceNeed, ...] = Field(min_length=1, max_length=8)
    # 保存可重算的图前提和局部边,而不是只有一句“适用”。未解决的条件只供诊断。
    given_conditions: dict[str, JsonValue] = Field(default_factory=dict, max_length=16)
    unresolved_conditions: dict[str, JsonValue] = Field(default_factory=dict, max_length=16)


class ContextEvidenceAssessment(ContractModel):
    """供给阶段的判断;依赖裁剪后必须通过 project 重新投影。"""

    assessor_version: str = Field(min_length=1, max_length=256)
    task_revision: int = Field(ge=1)
    demand_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    supplied_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    proposal_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    review_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_calls: int = Field(default=0, ge=0, le=2)
    failure: Literal["invalid_assessment", "unavailable", "capacity"] | None = None
    demands: tuple[ContextEvidenceDemand, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def identities(self) -> Self:
        if len({d.demand_id for d in self.demands}) != len(self.demands):
            raise ValueError("evidence repeats a demand")
        ids = [n.need_id for d in self.demands for n in d.needs]
        if len(set(ids)) != len(ids):
            raise ValueError("evidence repeats a need")
        if self.failure is not None and any(
            n.status != "unverified" for d in self.demands for n in d.needs
        ):
            raise ValueError("failed evidence assessment cannot admit a need")
        return self

    def allowed_bundle_ids(self) -> frozenset[str]:
        return frozenset(b for d in self.demands for n in d.needs for b in n.bundle_ids)

    def project(self, retained: frozenset[str]) -> "ContextEvidenceAssessment":
        """保守依赖闭包:任一判定材料落选就降级,不把旧判断套在残缺证据上。"""
        return self.model_copy(
            update={
                "demands": tuple(
                    d.model_copy(
                        update={
                            "needs": tuple(
                                n.model_copy(update={"status": "omitted"})
                                if n.bundle_ids and not set(n.bundle_ids) <= retained
                                else n
                                for n in d.needs
                            )
                        }
                    )
                    for d in self.demands
                )
            }
        )
