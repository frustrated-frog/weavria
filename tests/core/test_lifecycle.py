from itertools import product

import pytest

from weavria.core import InvalidStateTransitionError, RunStatus, transition_status


@pytest.mark.parametrize(("current", "target"), list(product(RunStatus, repeat=2)))
def test_complete_lifecycle_transition_matrix(current: RunStatus, target: RunStatus) -> None:
    legal = {
        (RunStatus.CREATED, RunStatus.RUNNING),
        (RunStatus.RUNNING, RunStatus.COMPLETED),
        (RunStatus.RUNNING, RunStatus.FAILED),
        (RunStatus.RUNNING, RunStatus.CANCELLED),
        (RunStatus.RUNNING, RunStatus.BUDGET_EXHAUSTED),
    }
    if (current, target) in legal:
        assert transition_status(current, target) is target
    else:
        with pytest.raises(InvalidStateTransitionError) as raised:
            transition_status(current, target)
        assert raised.value.current is current
        assert raised.value.target is target


def test_completed_run_cannot_be_completed_twice_or_restarted() -> None:
    status = transition_status(RunStatus.CREATED, RunStatus.RUNNING)
    status = transition_status(status, RunStatus.COMPLETED)
    assert status.is_terminal
    for target in RunStatus:
        with pytest.raises(InvalidStateTransitionError):
            transition_status(status, target)
