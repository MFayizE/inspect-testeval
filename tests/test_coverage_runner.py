import json
import random
from pathlib import Path

import pytest

from inspect_testeval import coverage_runner
from inspect_testeval.coverage_runner import (
    cov_at_k,
    evaluate,
    evaluate_path,
    evaluate_target,
    split_groups,
)
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


def _target_job(test: str, check_line: int, measure_errors: bool = False) -> dict[str, object]:
    return {
        "code": PROGRAM,
        "func_name": "classify",
        "test": test,
        "check_line": check_line,
        "measure_errors": measure_errors,
        "coveragerc": COVERAGERC,
    }


@pytest.mark.parametrize(
    ("test", "check_line", "hit"),
    [
        (POSITIVE, 4, True),  # return "pos"
        (POSITIVE, 6, False),  # return "neg" never runs for 3
        (NEGATIVE, 6, True),
        (WRONG_ASSERT, 6, True),  # the assertion fails after the line ran
    ],
)
def test_target_reports_whether_the_line_ran(
    tmp_path: Path, test: str, check_line: int, hit: bool
) -> None:
    result = evaluate_target(_target_job(test, check_line), tmp_path / "work")
    assert result["syntax_ok"] is True
    assert result["called"] is True
    assert result["line_hit"] is hit
    assert result["coverage_failure"] is False


@pytest.mark.parametrize(
    ("test", "outcome", "called"),
    [(SYNTAX_ERROR, "syntax_error", False), (NOT_CALLED, "ok", False)],
)
def test_target_not_measured_without_a_compiling_call(
    tmp_path: Path, test: str, outcome: str, called: bool
) -> None:
    result = evaluate_target(_target_job(test, 4), tmp_path / "work")
    assert result["outcome"] == outcome
    assert result["called"] is called
    assert result["line_hit"] is None


# Reaches `return "neg"` (line 6), then raises.
RAISES_AFTER_CALL = (
    "def test_classify():\n    solution = Solution()\n"
    "    solution.classify(-1)\n    raise ValueError('boom')"
)


def test_target_error_measured_only_when_asked(tmp_path: Path) -> None:
    plain = evaluate_target(_target_job(RAISES_AFTER_CALL, 6), tmp_path / "a")
    assert plain["outcome"] == "error"
    assert plain["error"] == "ValueError"
    assert plain["line_hit"] is None
    # eval_branchcov.py counts such a test as executable and measures it.
    measured = evaluate_target(
        _target_job(RAISES_AFTER_CALL, 6, measure_errors=True), tmp_path / "b"
    )
    assert measured["line_hit"] is True


def test_target_timeout_is_never_measured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coverage_runner, "EXEC_TIMEOUT_SECONDS", 1)
    result = evaluate_target(_target_job(HANGS, 4, measure_errors=True), tmp_path / "work")
    assert result["outcome"] == "timeout"
    assert result["line_hit"] is None


def test_target_coverage_run_without_report_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coverage_runner, "pytest_timeout", lambda num_files: 0.001)
    result = evaluate_target(_target_job(POSITIVE, 4), tmp_path / "work")
    assert result["outcome"] == "ok"
    assert result["coverage_failure"] is True
    assert result["line_hit"] is None


def test_main_dispatches_on_job_kind(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    targeted = tmp_path / "targeted" / "job.json"
    targeted.parent.mkdir()
    targeted.write_text(json.dumps(_target_job(POSITIVE, 4)), encoding="utf-8")
    assert coverage_runner.main(["runner", str(targeted)]) == 0
    assert json.loads(capsys.readouterr().out)["line_hit"] is True

    overall = tmp_path / "overall" / "job.json"
    overall.parent.mkdir()
    overall.write_text(json.dumps(_job([POSITIVE])), encoding="utf-8")
    assert coverage_runner.main(["runner", str(overall)]) == 0
    assert json.loads(capsys.readouterr().out)["exec_correct"] == 1


@pytest.mark.parametrize(
    ("generated", "reference", "similarity"),
    [
        (["a", "b", "c"], ["a", "b", "c"], 1.0),
        # The reference appears unbroken inside a longer log: still 1.
        (["x", "a", "b", "y"], ["a", "b"], 1.0),
        # Longest common *contiguous* run, not subsequence: "a", "c" share only runs of 1.
        (["a", "b", "c"], ["a", "c"], 0.5),
        (["b", "a"], ["a", "b"], 0.5),
        ([], ["a"], 0.0),
    ],
)
def test_match_path_is_longest_common_contiguous_run(
    generated: list[str], reference: list[str], similarity: float
) -> None:
    assert coverage_runner.match_path(generated, reference) == pytest.approx(similarity)


# Logs each branch it enters, as the instrumented TestEval programs do.
INSTRUMENTED = """\
class Solution:
    def classify(self, x: int) -> str:
        if x > 0:
            log_function('BRANCH #1\\n')
            return "pos"
        return "other"

def log_function(info_str):
    with open("test_logs/Classify.log", "a") as file:
        file.write(info_str)
"""


def _path_job(test: str, target_path: list[str]) -> dict[str, object]:
    return {
        "code": INSTRUMENTED,
        "func_name": "classify",
        "test": test,
        "log_file": "test_logs/Classify.log",
        "target_path": target_path,
    }


def test_path_reads_the_branches_the_test_entered(tmp_path: Path) -> None:
    result = evaluate_path(_path_job(POSITIVE, ["BRANCH #1\n"]), tmp_path / "work")
    assert result["outcome"] == "ok"
    assert result["path_preview"] == ["BRANCH #1\n"]
    assert result["similarity"] == result["similarity_reference"] == 1.0


def test_path_log_is_emptied_before_each_test(tmp_path: Path) -> None:
    work = tmp_path / "work"
    evaluate_path(_path_job(POSITIVE, ["BRANCH #1\n"]), work)
    result = evaluate_path(_path_job(NEGATIVE, ["BRANCH #1\n"]), work)
    assert result["path_length"] == 0
    assert result["similarity"] == 0.0


def test_empty_target_path_scores_both_definitions(tmp_path: Path) -> None:
    # No branch entered: the port counts it as following the empty path;
    # the reference scorer divides by zero and scores 0.
    followed = evaluate_path(_path_job(NEGATIVE, []), tmp_path / "a")
    assert (followed["similarity"], followed["similarity_reference"]) == (1.0, 0.0)
    strayed = evaluate_path(_path_job(POSITIVE, []), tmp_path / "b")
    assert (strayed["similarity"], strayed["similarity_reference"]) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("test", "outcome"),
    [(SYNTAX_ERROR, "syntax_error"), (RUNTIME_ERROR, "error"), (NOT_CALLED, "ok")],
)
def test_path_not_scored_without_an_executed_call(tmp_path: Path, test: str, outcome: str) -> None:
    result = evaluate_path(_path_job(test, ["BRANCH #1\n"]), tmp_path / "work")
    assert result["outcome"] == outcome
    assert result["similarity"] is None


def test_main_runs_path_jobs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job = tmp_path / "path" / "job.json"
    job.parent.mkdir()
    job.write_text(json.dumps(_path_job(POSITIVE, ["BRANCH #1\n"])), encoding="utf-8")
    assert coverage_runner.main(["runner", str(job)]) == 0
    assert json.loads(capsys.readouterr().out)["similarity"] == 1.0
