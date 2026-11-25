import json
from pathlib import Path
from typing import Any

import pytest

from inspect_testeval.formatting import (
    change_function_name,
    format_test,
    reformat_case_byrules,
    remove_extra,
)

FIXTURE = Path(__file__).parent / "fixtures" / "format_cases.jsonl"


def _cases() -> list[dict[str, Any]]:
    with FIXTURE.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.mark.parametrize(
    "case", _cases(), ids=lambda c: f"{c['model']}-{c['task_num']}-{c['test_index']}"
)
def test_matches_reference_formatter_on_real_replies(case: dict[str, Any]) -> None:
    assert format_test(case["raw"], case["func_name"]) == case["formatted"]


def test_remove_extra_drops_prose_and_later_inputs() -> None:
    raw = (
        "Here is a test:\n"
        "def test_add():\n"
        "    solution=Solution()\n"
        "    assert solution.add(1, 2) == 3\n"
        "    assert solution.add(2, 2) == 4\n"
        "This covers addition."
    )
    assert remove_extra(raw, "add") == (
        "def test_add():\n    solution=Solution()\n    assert solution.add(1, 2) == 3"
    )


def test_remove_extra_keeps_everything_without_def_or_call() -> None:
    raw = "x = 1\ny = 2"
    assert remove_extra(raw, "add") == raw


def test_reformat_drops_incomplete_last_line_and_renames() -> None:
    case = "def check():\n    solution=Solution()\n    assert solution.add(1,"
    assert reformat_case_byrules(case, "test_add") == "def test_add():\n    solution = Solution()"


def test_reformat_dedents_leading_space() -> None:
    case = " def test_add():\n     solution=Solution()\n     assert solution.add(1, 2) == 3"
    assert reformat_case_byrules(case, "test_add") == (
        "def test_add():\n    solution = Solution()\n    assert solution.add(1, 2) == 3"
    )


def test_change_function_name_keeps_unparseable_code() -> None:
    code = "def test_add(:\n    pass"
    assert change_function_name(code, "test_add") == code


def test_change_function_name_renames_only_first_function() -> None:
    code = "def a():\n    pass\n\ndef b():\n    pass"
    assert change_function_name(code, "test_x") == "def test_x():\n    pass\n\ndef b():\n    pass"


def test_change_function_name_round_trips_when_name_already_matches() -> None:
    # Upstream still re-renders through ast.unparse, which normalises spacing.
    assert change_function_name("def test_x():\n    a=1", "test_x") == "def test_x():\n    a = 1"
