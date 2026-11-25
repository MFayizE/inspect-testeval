import random
from pathlib import Path

import pytest

from inspect_testeval import coverage_runner
from inspect_testeval.coverage_runner import cov_at_k, evaluate, split_groups
from inspect_testeval.scorers import COVERAGERC

# 7 statements (class, def, if, return, if, return, return) and 4 branches.
PROGRAM = """\
class Solution:
    def classify(self, x: int) -> str:
        if x > 0:
            return "pos"
        if x < 0:
            return "neg"
        return "zero"
"""

POSITIVE = (
    'def test_classify():\n    solution = Solution()\n    assert solution.classify(3) == "pos"'
)
NEGATIVE = (
    'def test_classify():\n    solution = Solution()\n    assert solution.classify(-3) == "neg"'
)
WRONG_ASSERT = (
    'def test_classify():\n    solution = Solution()\n    assert solution.classify(-3) == "pos"'
)
SYNTAX_ERROR = "def test_classify(:\n    pass"
RUNTIME_ERROR = "def test_classify():\n    solution = Solution()\n    solution.classify(undefined)"
NOT_CALLED = "def test_classify():\n    assert 1 + 1 == 2"
HANGS = (
    "def test_classify():\n    solution = Solution()\n    while True:\n        solution.classify(1)"
)
KILLS_INTERPRETER = (
    "def test_classify():\n    import os\n    solution = Solution()\n"
    "    solution.classify(1)\n    os._exit(3)"
)


class _NoShuffle(random.Random):
    def shuffle(self, x: list[str]) -> None:  # type: ignore[override]  # keep order for exact expectations
        return None


def _job(tests: list[str], ks: list[int] | None = None) -> dict[str, object]:
    return {
        "code": PROGRAM,
        "func_name": "classify",
        "tests": tests,
        "ks": ks or [1, 2],
        "seed": "42:1",
        "coveragerc": COVERAGERC,
    }


def test_classifies_every_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coverage_runner, "EXEC_TIMEOUT_SECONDS", 2)
    tests = [
        POSITIVE,
        WRONG_ASSERT,
        SYNTAX_ERROR,
        RUNTIME_ERROR,
        NOT_CALLED,
        HANGS,
        KILLS_INTERPRETER,
    ]
    result = evaluate(_job(tests), tmp_path / "work")
    outcomes = [t["outcome"] for t in result["tests"]]
    assert outcomes == [
        "ok",
        "assertion_error",
        "syntax_error",
        "error",
        "not_called",
        "timeout",
        "error",
    ]
    assert result["tests"][3]["error"] == "NameError"
    assert result["tests"][6]["error"] == "exit code 3"
    assert result["total_cases"] == 7
    assert result["syntax_correct"] == 6
    # Only the passing test and the failing-assertion test count as executable.
    assert result["exec_correct"] == 2
    assert [t["executed"] for t in result["tests"]] == [True, True] + [False] * 5


def test_failing_assertion_still_contributes_coverage(tmp_path: Path) -> None:
    result = evaluate(_job([POSITIVE, WRONG_ASSERT]), tmp_path / "work")
    # Together they run lines 1-6 (not `return "zero"`) and 3 of 4 branches.
    assert result["line_coverage"] == pytest.approx(6 / 7)
    assert result["branch_coverage"] == pytest.approx(3 / 4)


def test_no_executable_tests_gives_no_coverage(tmp_path: Path) -> None:
    result = evaluate(_job([SYNTAX_ERROR, NOT_CALLED]), tmp_path / "work")
    assert result["exec_correct"] == 0
    assert result["line_coverage"] is None
    assert result["branch_coverage"] is None
    assert result["cov_at_k"] is None


def test_cov_at_k_paper_and_reference_definitions(tmp_path: Path) -> None:
    work = tmp_path / "work"
    evaluate(_job([POSITIVE, NEGATIVE]), work)  # writes under_test.py and test files
    result = cov_at_k(work, ["test_0.py", "test_1.py"], [1, 2], _NoShuffle())
    # k=1: one test per group, both definitions average the single-test coverages.
    for definition in ("paper", "reference"):
        assert result[definition]["1"]["line"] == pytest.approx((4 / 7 + 5 / 7) / 2)
        assert result[definition]["1"]["branch"] == pytest.approx((1 / 4 + 2 / 4) / 2)
    # k=2: paper takes the group's coverage; the reference code averages the
    # cumulative coverage after the first test and after both.
    assert result["paper"]["2"]["line"] == pytest.approx(6 / 7)
    assert result["paper"]["2"]["branch"] == pytest.approx(3 / 4)
    assert result["reference"]["2"]["line"] == pytest.approx((4 / 7 + 6 / 7) / 2)
    assert result["reference"]["2"]["branch"] == pytest.approx((1 / 4 + 3 / 4) / 2)


def test_cov_at_k_is_reproducible_for_a_seed(tmp_path: Path) -> None:
    tests = [POSITIVE, NEGATIVE, POSITIVE, NEGATIVE, WRONG_ASSERT]
    first = evaluate(_job(tests, ks=[2]), tmp_path / "a")
    second = evaluate(_job(tests, ks=[2]), tmp_path / "b")
    assert first["cov_at_k"] == second["cov_at_k"]


@pytest.mark.parametrize(
    "body",
    [
        # Bytes that aren't UTF-8 used to crash the runner when it decoded output.
        "import sys\n    sys.stdout.buffer.write(bytes([255, 10]))",
        # Replacing or closing stdout used to hide the result.
        "import sys, io\n    sys.stdout = io.StringIO()",
        "import sys\n    sys.stdout.close()",
        # A relative result path would break after chdir.
        "import os\n    os.chdir('/')",
        "print('x' * 1_000_000)",
    ],
)
def test_test_side_effects_cannot_hide_or_corrupt_the_result(tmp_path: Path, body: str) -> None:
    test = (
        f"def test_classify():\n    {body}\n    solution = Solution()\n"
        '    assert solution.classify(3) == "pos"'
    )
    result = evaluate(_job([test]), tmp_path / "work")
    assert result["tests"][0]["outcome"] == "ok"
    assert result["exec_correct"] == 1


def test_coverage_run_without_report_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coverage_runner, "pytest_timeout", lambda num_files: 0.001)
    result = evaluate(_job([POSITIVE]), tmp_path / "work")
    assert result["exec_correct"] == 1
    assert result["line_coverage"] is None
    assert result["coverage_failures"][0] == "overall: test_0.py"
    assert any(f.startswith("cov@1: ") for f in result["coverage_failures"])


def test_pytest_timeout_grows_with_tests() -> None:
    assert coverage_runner.pytest_timeout(1) < coverage_runner.pytest_timeout(20)
    # Each test already ran within 5 s; tracing gets three times that per test.
    assert coverage_runner.pytest_timeout(20) == 60 + 15 * 20


@pytest.mark.parametrize(
    ("count", "k", "sizes"),
    [(5, 1, [1, 1, 1, 1, 1]), (5, 2, [2, 2]), (5, 5, [5]), (3, 5, [3]), (10, 5, [5, 5])],
)
def test_split_groups_drops_remainder_and_keeps_short_lists(
    count: int, k: int, sizes: list[int]
) -> None:
    tests = [f"t{i}" for i in range(count)]
    assert [len(g) for g in split_groups(tests, k)] == sizes
