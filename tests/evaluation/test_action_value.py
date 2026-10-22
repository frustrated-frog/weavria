from weavria.evaluation import ActionValueTrajectoryCase, evaluate_action_value_policy


async def test_action_value_policy_uses_yield_risk_and_per_call_cost() -> None:
    cases = (
        ActionValueTrajectoryCase(
            case_id="cold-start",
            objective="replan",
            outcomes=("neutral",),
            expected_disposition="allow",
        ),
        ActionValueTrajectoryCase(
            case_id="sustained-neutral",
            objective="replan",
            outcomes=("neutral", "neutral", "neutral"),
            expected_disposition="reconsider",
        ),
        ActionValueTrajectoryCase(
            case_id="productive",
            objective="continue",
            outcomes=("positive", "positive", "positive"),
            expected_disposition="allow",
        ),
        ActionValueTrajectoryCase(
            case_id="productive-three-calls",
            objective="continue",
            outcomes=("positive", "positive", "positive"),
            candidate_calls=3,
            expected_disposition="allow",
        ),
        ActionValueTrajectoryCase(
            case_id="regression-risk",
            objective="repair",
            outcomes=("regressed", "regressed", "regressed"),
            expected_disposition="reconsider",
        ),
        ActionValueTrajectoryCase(
            case_id="moderate-yield-one-call",
            objective="continue",
            outcomes=("positive", "neutral", "neutral"),
            expected_disposition="allow",
        ),
        ActionValueTrajectoryCase(
            case_id="moderate-yield-three-calls",
            objective="continue",
            outcomes=("positive", "neutral", "neutral"),
            candidate_calls=3,
            expected_disposition="reconsider",
        ),
    )

    report = await evaluate_action_value_policy(cases)

    assert report.policy_version.startswith("progress-action-intervention/2[")
    assert report.baseline_version == "progress-action-intervention/1"
    assert report.selected_correct == 7
    assert report.baseline_correct == 4
    assert report.prevented_calls == 5
    assert report.false_interventions == 0
    by_id = {item.case_id: item for item in report.cases}
    one_call_yield = by_id["moderate-yield-one-call"].expected_progress_per_call
    three_call_yield = by_id["moderate-yield-three-calls"].expected_progress_per_call
    assert one_call_yield is not None and three_call_yield is not None
    assert one_call_yield > three_call_yield
    assert (
        by_id["regression-risk"].pessimistic_regression_probability is not None
        and by_id["regression-risk"].pessimistic_regression_probability > 0.35
    )
