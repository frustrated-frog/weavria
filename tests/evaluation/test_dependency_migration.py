"""检验交付判据, 无需模型、网络或历史 Git 对象。"""

import json

from examples.dependency_migration import INVENTORY, MANIFEST, expected, grade


def test_grading_separates_an_answer_from_a_grounded_delivery() -> None:
    answer = expected().model_dump_json()
    assert grade("completed", answer, {MANIFEST, INVENTORY}) == ()
    assert grade("failed", answer, {MANIFEST}) == ("not_completed", "unread:" + INVENTORY)


def test_wrong_version_wrong_group_and_decoy_dependency_fail_independently() -> None:
    report = expected().model_dump()
    report["added"]["snowballstemmer"] = ">=3"
    report["retained"]["numpy"] = report["removed"].pop("numpy")
    report["removed"]["pytest"] = "8.3"
    assert grade("completed", json.dumps(report), {MANIFEST, INVENTORY}) == (
        "added",
        "removed",
        "retained",
    )


def test_duplicate_keys_are_not_a_valid_delivery() -> None:
    output = '{"added":{},' + expected().model_dump_json()[1:]
    assert grade("completed", output, {MANIFEST, INVENTORY}) == ("invalid_delivery_json",)
