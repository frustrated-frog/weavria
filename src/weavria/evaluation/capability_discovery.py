"""Reproducible capability-discovery comparison against all-tools disclosure."""

from pydantic import BaseModel, ConfigDict, Field

from weavria.harness import CapabilityQuery, CapabilityResolver
from weavria.tools import ToolRegistry


class CapabilityCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str = Field(min_length=1, max_length=128)
    query: CapabilityQuery
    expected_tool_names: tuple[str, ...] = Field(min_length=1)


class CapabilityCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str
    selected_tool_names: tuple[str, ...]
    expected_recall: float = Field(ge=0, le=1)
    visible_count: int = Field(ge=0)
    baseline_visible_count: int = Field(ge=0)


class CapabilityExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    resolver_version: str
    catalogue_size: int = Field(ge=0)
    cases: tuple[CapabilityCaseResult, ...]
    mean_recall: float = Field(ge=0, le=1)
    mean_visible_count: float = Field(ge=0)
    baseline_mean_visible_count: float = Field(ge=0)


def evaluate_capability_discovery(
    cases: tuple[CapabilityCase, ...],
    registry: ToolRegistry,
    resolver: CapabilityResolver,
) -> CapabilityExperiment:
    """Freeze one registry and report recall/exposure without claiming task success."""
    if not cases:
        raise ValueError("capability experiment requires at least one case")
    names = {item.name for item in registry.descriptors}
    results: list[CapabilityCaseResult] = []
    for index, case in enumerate(cases, 1):
        expected = set(case.expected_tool_names)
        if not expected.issubset(names):
            raise ValueError("evaluation case expects an unregistered capability")
        view, _ = resolver.resolve(
            case.query,
            registry,
            epoch=index,
            trigger="initial",
        )
        selected = set(view.allowed_tool_names)
        results.append(
            CapabilityCaseResult(
                case_id=case.case_id,
                selected_tool_names=view.allowed_tool_names,
                expected_recall=len(selected & expected) / len(expected),
                visible_count=len(selected),
                baseline_visible_count=len(names),
            )
        )
    return CapabilityExperiment(
        resolver_version=resolver.version,
        catalogue_size=len(names),
        cases=tuple(results),
        mean_recall=sum(item.expected_recall for item in results) / len(results),
        mean_visible_count=sum(item.visible_count for item in results) / len(results),
        baseline_mean_visible_count=float(len(names)),
    )
