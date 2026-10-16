import pytest

from weavria.runtime import RootBudget


def test_root_budget_never_grows_child_deadlines() -> None:
    budget = RootBudget.start(10.0, 5.0)
    assert budget.remaining(11.0) == 4.0
    assert budget.child_duration(11.0, 10.0) == 4.0
    assert budget.child_duration(14.5, 1.0) == 0.5
    assert budget.remaining(20.0) == 0.0


@pytest.mark.parametrize("duration", [0.0, -1.0, float("inf"), float("nan")])
def test_root_budget_rejects_invalid_duration(duration: float) -> None:
    with pytest.raises(ValueError):
        RootBudget.start(0.0, duration)
