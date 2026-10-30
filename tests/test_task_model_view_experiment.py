from examples.task_model_view import build_report
from weavria.core import RunStatus


async def test_complete_request_comparison_retains_benefits_overhead_and_execution_contract() -> (
    None
):
    report = await build_report()
    assert len(report.dataset_sha256) == 64
    assert len(report.rows) == 6
    for needs in (0, 1, 8):
        plain = next(row for row in report.rows if row.needs == needs and not row.projected)
        projected = next(row for row in report.rows if row.needs == needs and row.projected)
        assert plain.status is projected.status is RunStatus.COMPLETED
        assert plain.task_outcome == projected.task_outcome == "unverified"
        assert plain.model_calls == projected.model_calls == (2 if needs else 1)
        assert plain.tool_calls == projected.tool_calls == (1 if needs else 0)
        assert projected.largest_growth_reserve_bytes > 0
        assert projected.transport_bytes > projected.request_bytes
        if needs == 8:
            assert projected.input_tokens < plain.input_tokens
        if needs == 0:
            # 保留负向样例: 短任务的版本与来源信封可能比原观察更大。
            assert projected.request_bytes > plain.request_bytes
