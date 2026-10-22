from weavria.evaluation import ActionValueTrajectoryCase, evaluate_action_value_policy


def gain_rate_regression_cases() -> tuple[ActionValueTrajectoryCase, ...]:
    return (
        ActionValueTrajectoryCase(
            case_id="cold-start",
            objective="replan",
            outcomes=("neutral",),
            expected_disposition="allow",
        ),
        ActionValueTrajectoryCase(
            case_id="uncertain-neutral",
            objective="replan",
            outcomes=("neutral",) * 3,
            expected_disposition="allow",
        ),
        ActionValueTrajectoryCase(
            case_id="sustained-neutral",
            objective="replan",
            outcomes=("neutral",) * 8,
            expected_disposition="reconsider",
        ),
        *(
            ActionValueTrajectoryCase(
                case_id=f"productive-width-{width}",
                objective="continue",
                outcomes=("positive",) * 3,
                candidate_calls=width,
                gain_units_per_productive_turn=width,
                expected_disposition="allow",
            )
            for width in (1, 3, 4, 5, 8)
        ),
        ActionValueTrajectoryCase(
            case_id="sparse-wide",
            objective="continue",
            outcomes=("positive",) * 8,
            candidate_calls=8,
            expected_disposition="reconsider",
        ),
        ActionValueTrajectoryCase(
            case_id="uncertain-sparse",
            objective="continue",
            outcomes=("positive", "neutral", "neutral"),
            candidate_calls=3,
            expected_disposition="allow",
        ),
        ActionValueTrajectoryCase(
            case_id="sustained-sparse",
            objective="continue",
            outcomes=(
                "positive",
                "neutral",
                "neutral",
                "neutral",
                "neutral",
                "neutral",
                "neutral",
                "neutral",
            ),
            candidate_calls=3,
            expected_disposition="reconsider",
        ),
        ActionValueTrajectoryCase(
            case_id="regression-risk",
            objective="repair",
            outcomes=("regressed",) * 3,
            expected_disposition="reconsider",
        ),
        ActionValueTrajectoryCase(
            case_id="mixed-risk",
            objective="repair",
            outcomes=("mixed",) * 3,
            gain_units_per_productive_turn=4,
            expected_disposition="reconsider",
        ),
        *(
            ActionValueTrajectoryCase(
                case_id=f"gain-unit-{dimension}",
                objective="continue",
                outcomes=("positive",) * 3,
                gain_dimension=dimension,
                gain_units_per_productive_turn=4,
                expected_disposition="allow",
            )
            for dimension in ("obligations", "coverage", "independent_sources")
        ),
    )


async def test_action_value_policy_uses_gain_magnitude_uncertainty_and_risk() -> None:
    report = await evaluate_action_value_policy(gain_rate_regression_cases())

    assert report.schema_version == 2
    assert report.policy_version.startswith("progress-action-intervention/3[")
    assert report.baseline_version == "progress-action-intervention/1"
    assert report.selected_correct == report.total_cases == 16
    assert report.baseline_correct == 11
    assert report.prevented_calls == 14
    assert report.false_interventions == 0
    by_id = {item.case_id: item for item in report.cases}
    for width in (1, 3, 4, 5, 8):
        assert by_id[f"productive-width-{width}"].expected_progress_per_call == 1.0
    uncertain = by_id["uncertain-sparse"]
    assert uncertain.expected_progress_per_call == 0.2
    assert uncertain.optimistic_progress_per_call is not None
    assert uncertain.optimistic_progress_per_call > 0.25
    for dimension in ("obligations", "coverage", "independent_sources"):
        value = by_id[f"gain-unit-{dimension}"].expected_progress_per_call
        assert value is not None and value > 1.0
