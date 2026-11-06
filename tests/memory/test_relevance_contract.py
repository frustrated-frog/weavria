"""可选检索信号必须版本化并绑定供给;不能借元数据指向未交付的内容。"""

import hashlib
import json

import pytest
from pydantic import ValidationError

from weavria.core import ContextRelevance, ContextSupply, ContextSupplyReport, RetrievalDiagnostics


def report() -> ContextSupplyReport:
    return ContextSupplyReport(
        provider_version="test/1",
        task_revision=1,
        demand_sha256="d" * 64,
        snapshot_sha256="a" * 64,
        supplied_sha256=hashlib.sha256(b"[]").hexdigest(),
        matched_groups=0,
        supplied_groups=0,
        limit_omitted_groups=0,
        budget_omitted_groups=0,
    )


def test_legacy_supply_keeps_its_exact_wire_and_roundtrip() -> None:
    original = report()
    wire = original.model_dump_json()
    assert "relevance" not in wire and "retrieval" not in wire
    assert ContextSupplyReport.model_validate_json(wire) == original
    assert ContextSupply(bundles=(), report=original).report == original


def test_relevance_requires_v2_complete_diagnostics_and_present_bundle() -> None:
    hint = ContextRelevance(bundle_id="absent", demand_id="current_task", score=0.5)
    value = {**report().model_dump(mode="json"), "relevance": [hint.model_dump(mode="json")]}
    with pytest.raises(ValidationError, match="requires source report v2"):
        ContextSupplyReport.model_validate_json(json.dumps(value))
    value["schema_version"] = 2
    with pytest.raises(ValidationError, match="requires retrieval diagnostics"):
        ContextSupplyReport.model_validate_json(json.dumps(value))
    value["retrieval"] = RetrievalDiagnostics(
        mode="dense",
        encoded_chunks=1,
        cache_hits=0,
        selection_steps=1,
    ).model_dump(mode="json")
    new = ContextSupplyReport.model_validate_json(json.dumps(value))
    with pytest.raises(ValidationError, match="absent bundle"):
        ContextSupply(bundles=(), report=new)
    value["relevance"] = [hint.model_dump(mode="json"), hint.model_dump(mode="json")]
    with pytest.raises(ValidationError, match="repeats a relevance binding"):
        ContextSupplyReport.model_validate_json(json.dumps(value))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), 0.0, 1.01])
def test_relevance_is_bounded(score: float) -> None:
    with pytest.raises(ValidationError):
        ContextRelevance(bundle_id="b", demand_id="d", score=score)
