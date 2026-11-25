"""Score generated tests inside the sandbox: correctness, coverage and cov@k.

This file is copied into the Docker sandbox and run there with the sandbox's
Python, so it uses only the standard library plus pytest and pytest-cov. It
follows ``check_correctness`` and ``coverage_at_k_sample`` in eval_overall.py
from https://github.com/LLM4SoftwareTesting/TestEval (MIT licence, commit
6f67dee71de6ccda1767b8c219a93f4ae0a7ce90), with two deliberate differences:

* Each test is executed in its own Python process with a fresh namespace and a
  timeout. Upstream calls ``exec`` inside the evaluator's own module under a
  Unix ``SIGALRM``, which shares globals between tests and with the evaluator.
* cov@k is reported twice: ``paper`` follows Algorithm 1 of the paper (the
  coverage of each group of k tests), and ``reference`` reproduces the
  reference code, which averages the cumulative coverage after 1..k tests of
  each group. The two agree for k = 1.

``evaluate_target`` scores one test written to cover a target line or branch,
following ``eval_linecov.py`` and ``eval_branchcov.py`` from the same commit.

Usage: ``python coverage_runner.py <job.json>``; prints one JSON object. A job
with ``check_line`` is a targeted job, any other an overall one.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

EXEC_TIMEOUT_SECONDS = 5  # upstream: execute(test_code, timeout=5)
PYTEST_BASE_TIMEOUT_SECONDS = 60
COVERAGE_JSON = "coverage.json"
COV_COMMAND = [
    sys.executable,
    "-m",
    "pytest",
    "-p",
    "pytest_cov",
    "--cov=under_test",
    "--cov-branch",
    f"--cov-report=json:{COVERAGE_JSON}",
    "-q",
    "-p",
    "no:cacheprovider",
]
# Load only pytest-cov, so results don't depend on other installed plugins
# (and a development environment's plugins don't slow every run).
PYTEST_ENV = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}

# Runs one test the way upstream's execute() classifies it: an AssertionError
# still counts as executable, a timeout or any other exception does not.
# The outcome goes to a result file, not stdout, so a test that replaces or
# closes sys.stdout, writes bytes that aren't UTF-8, changes directory or
# rebinds builtins cannot hide or corrupt it. open() and json.dump are bound
# before the test runs.
EXEC_HARNESS = """\
import json, sys
_open, _dump = open, json.dump
source_path, result_path = sys.argv[1], sys.argv[2]
with _open(source_path, encoding="utf-8") as fh:
    source = fh.read()
try:
    exec(compile(source, "test_case.py", "exec"), {"__name__": "__testeval__"})
    outcome = {"status": "ok"}
except AssertionError:
    outcome = {"status": "assertion_error"}
except BaseException as e:
    outcome = {"status": "error", "error": type(e).__name__}
with _open(result_path, "w", encoding="utf-8") as fh:
    _dump(outcome, fh)
"""


def run_test(workdir: Path, index: int, test_code: str) -> dict[str, Any]:
    """Execute one test in a fresh interpreter and report how it ended."""
    script = workdir / f"exec_{index}.py"
    result_file = (workdir / f"exec_{index}.result.json").resolve()
    script.write_text(test_code, encoding="utf-8")
    result_file.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", EXEC_HARNESS, str(script.resolve()), str(result_file)],
            cwd=workdir,
            # The test's output is never read, so it can't fail to decode or fill memory.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=EXEC_TIMEOUT_SECONDS,
            check=False,  # a failing test is an outcome to record, not an error
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    finally:
        script.unlink(missing_ok=True)
    try:
        return json.loads(result_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        # The test ended the interpreter (os._exit, a crash) before reporting.
        return {"status": "error", "error": f"exit code {proc.returncode}"}
    finally:
        result_file.unlink(missing_ok=True)


def pytest_timeout(num_files: int) -> int:
    """Time allowed for one coverage run over ``num_files`` tests.

    Each test already finished within EXEC_TIMEOUT_SECONDS without tracing;
    coverage tracing slows code down, so allow three times that per test.
    """
    return PYTEST_BASE_TIMEOUT_SECONDS + 3 * EXEC_TIMEOUT_SECONDS * num_files


def coverage_report(workdir: Path, test_files: list[str]) -> dict[str, Any] | None:
    """Run pytest with branch coverage over ``test_files``; ``None`` if no report is produced.

    Like upstream, the coverage report is read even when tests fail: a failing
    assertion still executes the program under test. Any earlier report is
    deleted first, so a run that writes none can't be read as an older one.
    """
    report_path = workdir / COVERAGE_JSON
    report_path.unlink(missing_ok=True)
    try:
        subprocess.run(
            [*COV_COMMAND, *test_files],
            cwd=workdir,
            env=PYTEST_ENV,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=pytest_timeout(len(test_files)),
            check=False,  # pytest exits non-zero when a test's assertion fails
        )
    except subprocess.TimeoutExpired:
        return None
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def measure(workdir: Path, test_files: list[str]) -> dict[str, float] | None:
    """Line and branch coverage of ``test_files``; ``None`` if it can't be measured."""
    report = coverage_report(workdir, test_files)
    if report is None:
        return None
    totals = report["totals"]
    if totals["num_statements"] == 0 or totals["num_branches"] == 0:
        return None
    return {
        "line": totals["covered_lines"] / totals["num_statements"],
        "branch": totals["covered_branches"] / totals["num_branches"],
    }


def split_groups(tests: list[str], k: int) -> list[list[str]]:
    """Split shuffled tests into floor(M / k) groups of k; one group of all M if M < k."""
    if len(tests) >= k:
        return [tests[i * k : (i + 1) * k] for i in range(len(tests) // k)]
    return [tests]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def cov_at_k(
    workdir: Path,
    passed: list[str],
    ks: list[int],
    rng: random.Random,
    failures: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float | None]]]:
    """cov@k under both definitions.

    As in upstream, the list of passing tests is shuffled in place once per k,
    in the order of ``ks``; the same grouping feeds both definitions. Runs that
    produce no coverage report are left out of the averages and listed in
    ``failures``.
    """
    failures = failures if failures is not None else []

    def measured(k: int, files: list[str]) -> dict[str, float] | None:
        result = measure(workdir, files)
        if result is None:
            failures.append(f"cov@{k}: {' '.join(files)}")
        return result

    result: dict[str, dict[str, dict[str, float | None]]] = {"paper": {}, "reference": {}}
    tests = list(passed)
    for k in ks:
        rng.shuffle(tests)
        groups = split_groups(tests, k)

        paper_line, paper_branch = [], []
        reference_line, reference_branch = [], []
        for group in groups:
            whole = measured(k, group)
            if whole is not None:
                paper_line.append(whole["line"])
                paper_branch.append(whole["branch"])
            prefix_line, prefix_branch = [], []
            for end in range(1, len(group) + 1):
                prefix = whole if end == len(group) else measured(k, group[:end])
                if prefix is not None:
                    prefix_line.append(prefix["line"])
                    prefix_branch.append(prefix["branch"])
            if prefix_line:
                reference_line.append(mean(prefix_line))
                reference_branch.append(mean(prefix_branch))

        result["paper"][str(k)] = {
            "line": mean(paper_line) if paper_line else None,
            "branch": mean(paper_branch) if paper_branch else None,
        }
        result["reference"][str(k)] = {
            "line": mean(reference_line) if reference_line else None,
            "branch": mean(reference_branch) if reference_branch else None,
        }
    return result


def evaluate(
    job: dict[str, Any], workdir: Path, rng: random.Random | None = None
) -> dict[str, Any]:
    """Score every generated test for one program (one sample).

    ``rng`` defaults to ``random.Random(job["seed"])``; tests pass their own to
    fix the cov@k grouping.
    """
    func_name: str = job["func_name"]
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "under_test.py").write_text(job["code"], encoding="utf-8")
    (workdir / ".coveragerc").write_text(job["coveragerc"], encoding="utf-8")

    tests: list[dict[str, Any]] = []
    passed: list[str] = []
    for index, testcase in enumerate(job["tests"]):
        record: dict[str, Any] = {"index": index, "syntax_ok": False, "executed": False}
        tests.append(record)
        try:
            compile(testcase, "<string>", "exec")
        except Exception:  # upstream counts any compile failure as a syntax error
            record["outcome"] = "syntax_error"
            continue
        record["syntax_ok"] = True

        test_code = "from under_test import Solution\n" + testcase + f"\ntest_{func_name}()"
        outcome = run_test(workdir, index, test_code)
        record["outcome"] = outcome["status"]
        if "error" in outcome:
            record["error"] = outcome["error"]
        if outcome["status"] not in ("ok", "assertion_error"):
            continue
        if test_code.find(f"solution.{func_name}") == -1:
            record["outcome"] = "not_called"
            continue
        record["executed"] = True
        test_file = f"test_{index}.py"
        (workdir / test_file).write_text(
            "from under_test import Solution\n" + testcase, encoding="utf-8"
        )
        passed.append(test_file)

    failures: list[str] = []
    overall = measure(workdir, passed) if passed else None
    if passed and overall is None:
        failures.append(f"overall: {' '.join(passed)}")
    at_k = (
        cov_at_k(workdir, passed, job["ks"], rng or random.Random(job["seed"]), failures)
        if passed
        else None
    )
    return {
        "total_cases": len(tests),
        "syntax_correct": sum(t["syntax_ok"] for t in tests),
        "exec_correct": len(passed),
        "line_coverage": overall["line"] if overall else None,
        "branch_coverage": overall["branch"] if overall else None,
        "cov_at_k": at_k,
        # Coverage runs that produced no report (timeout, no data); each counts as 0.
        "coverage_failures": failures,
        "tests": tests,
    }


def evaluate_target(job: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """Run one test written to cover a target and report whether it reaches ``check_line``.

    Follows eval_linecov.py and eval_branchcov.py: a test that compiles, runs
    (a failing assertion counts) and calls the function under test is run under
    coverage on its own, and the target counts as covered if ``check_line`` is
    among the executed lines. ``check_line`` is the target line, or the line
    after a branch's first line.

    With ``measure_errors``, a test that raised another exception is measured
    too, because eval_branchcov.py counts it as executable (it tests
    ``res != False``); the scorer decides which outcomes count.
    """
    func_name: str = job["func_name"]
    testcase: str = job["test"]
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "under_test.py").write_text(job["code"], encoding="utf-8")
    (workdir / ".coveragerc").write_text(job["coveragerc"], encoding="utf-8")

    result: dict[str, Any] = {
        "syntax_ok": False,
        "outcome": "syntax_error",
        "called": False,
        "line_hit": None,
        "coverage_failure": False,
    }
    try:
        compile(testcase, "<string>", "exec")
    except Exception:  # upstream counts any compile failure as a syntax error
        return result
    result["syntax_ok"] = True

    test_code = "from under_test import Solution\n" + testcase + f"\ntest_{func_name}()"
    outcome = run_test(workdir, 0, test_code)
    result["outcome"] = outcome["status"]
    if "error" in outcome:
        result["error"] = outcome["error"]
    result["called"] = test_code.find(f"solution.{func_name}") != -1

    measured = (
        ("ok", "assertion_error", "error") if job["measure_errors"] else ("ok", "assertion_error")
    )
    if not result["called"] or result["outcome"] not in measured:
        return result
    (workdir / "test_target.py").write_text(
        "from under_test import Solution\n" + testcase, encoding="utf-8"
    )
    report = coverage_report(workdir, ["test_target.py"])
    file_report = (report or {}).get("files", {}).get("under_test.py")
    if file_report is None:
        result["coverage_failure"] = True
        return result
    result["line_hit"] = job["check_line"] in file_report["executed_lines"]
    return result


def main(argv: list[str]) -> int:
    job_path = Path(argv[1])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    workdir = job_path.parent / "work"
    result = evaluate_target(job, workdir) if "check_line" in job else evaluate(job, workdir)
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
