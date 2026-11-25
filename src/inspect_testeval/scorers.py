"""Score generated tests by running them in the sandbox, as eval_overall.py does."""

import json
from collections.abc import Sequence
from importlib.resources import files
from typing import Any

from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    grouped,
    mean,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from inspect_testeval.formatting import format_test
from inspect_testeval.solvers import GeneratedTests

# .coveragerc from the TestEval repository (commit 6f67dee), verbatim: coverage
# ignores the import lines every program under test starts with.
COVERAGERC = (
    "[report]\n"
    ";exclude import statements\n"
    "exclude_also =\n"
    "    import math\n"
    "    import itertools\n"
    "    import bisect\n"
    "    import collections\n"
    "    import string\n"
    "    import heapq\n"
    "    import functools\n"
    "    import sortedcontainers\n"
    "    import operator\n"
    "    from typing import List, Dict, Tuple, Iterator*\n"
    "    from math import sqrt\n"
    "    from bisect import bisect_left*\n"
    "    from dataclasses import dataclass\n"
    "    import *\n"
    "    from * import *"
)

DEFAULT_KS = (1, 2, 5)  # eval_overall.py --ks default
DEFAULT_SEED = 42  # eval_overall.py seeds random with 42
# Upper bound for scoring one program: up to 20 tests at 5 s each, plus one
# pytest run for overall coverage and one per cov@k group and prefix.
RUNNER_TIMEOUT_SECONDS = 3600


def runner_source() -> str:
    """Source of the script that runs inside the sandbox."""
    return files("inspect_testeval").joinpath("coverage_runner.py").read_text(encoding="utf-8")


def coverage_keys(ks: Sequence[int]) -> list[str]:
    keys = ["line_coverage", "branch_coverage"]
    for k in ks:
        keys += [f"line_cov@{k}", f"branch_cov@{k}"]
        keys += [f"line_cov@{k}_reference", f"branch_cov@{k}_reference"]
    return keys


def _ratio_of_sums(scores: list[SampleScore], numerator: str, denominator: str) -> float:
    """Divide summed counts across samples, as eval_overall.py does over all test cases."""
    num = den = 0.0
    for sample_score in scores:
        value = sample_score.score.value
        if not isinstance(value, dict):
            raise ValueError(f"expected a dict score value, got {type(value).__name__}")
        num += float(value[numerator] or 0.0)
        den += float(value[denominator] or 0.0)
    return num / den if den else float("nan")


@metric
def syntax_correctness() -> Metric:
    """Share of all generated tests that compile."""

    def compute(scores: list[SampleScore]) -> float:
        return _ratio_of_sums(scores, "syntax_correct", "total_cases")

    return compute


@metric
def execution_correctness() -> Metric:
    """Share of all generated tests that run and call the function under test.

    A failing assertion still counts as executed, as in the paper.
    """

    def compute(scores: list[SampleScore]) -> float:
        return _ratio_of_sums(scores, "exec_correct", "total_cases")

    return compute


def score_value(result: dict[str, Any], ks: Sequence[int]) -> dict[str, float]:
    """Flatten the runner's result into the score value.

    A program with no executable test, or whose coverage could not be measured,
    scores 0 coverage: eval_overall.py sums coverage over programs that have
    some and divides by the number of all programs.
    """
    value: dict[str, float] = {
        "total_cases": float(result["total_cases"]),
        "syntax_correct": float(result["syntax_correct"]),
        "exec_correct": float(result["exec_correct"]),
        "line_coverage": float(result["line_coverage"] or 0.0),
        "branch_coverage": float(result["branch_coverage"] or 0.0),
    }
    cov_at_k = result["cov_at_k"] or {}
    for k in ks:
        for definition, suffix in (("paper", ""), ("reference", "_reference")):
            entry = cov_at_k.get(definition, {}).get(str(k), {})
            value[f"line_cov@{k}{suffix}"] = float(entry.get("line") or 0.0)
            value[f"branch_cov@{k}{suffix}"] = float(entry.get("branch") or 0.0)
    return value


def overall_metrics(ks: Sequence[int] = DEFAULT_KS) -> list[Metric | dict[str, list[Metric]]]:
    """Metrics for the overall task; the coverage keys depend on ``ks``."""
    return [
        syntax_correctness(),
        execution_correctness(),
        {key: [mean()] for key in coverage_keys(ks)},
    ]


@scorer(metrics=overall_metrics())
def overall_coverage(ks: Sequence[int] = DEFAULT_KS, seed: int = DEFAULT_SEED) -> Scorer:
    """Correctness, overall line/branch coverage and cov@k of the generated tests.

    The metrics registered here cover ``DEFAULT_KS``. A task that passes other
    ``ks`` must also pass ``metrics=overall_metrics(ks)``, as ``testeval_overall``
    does, so the registered keys match the score's keys.
    """
    ks = list(ks)
    if not ks or any(k < 1 for k in ks):
        raise ValueError(f"ks must be positive integers, got {ks}")

    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata
        func_name = metadata["func_name"]
        raw_tests = state.store_as(GeneratedTests).raw_tests
        formatted = [format_test(raw, func_name) for raw in raw_tests]

        job = {
            "code": metadata["python_solution"],
            "func_name": func_name,
            "tests": formatted,
            "ks": ks,
            # Seed per program so results don't depend on sample order or concurrency.
            "seed": f"{seed}:{metadata['task_num']}",
            "coveragerc": COVERAGERC,
        }
        job_dir = f"/home/agent/testeval/{state.uuid}"
        env = sandbox()
        await env.write_file(f"{job_dir}/coverage_runner.py", runner_source())
        await env.write_file(f"{job_dir}/job.json", json.dumps(job))
        run = await env.exec(
            ["python", f"{job_dir}/coverage_runner.py", f"{job_dir}/job.json"],
            timeout=RUNNER_TIMEOUT_SECONDS,
        )
        if not run.success:
            raise RuntimeError(
                f"coverage runner failed (exit {run.returncode}): {run.stderr[-2000:]}"
            )
        try:
            result = json.loads(run.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"coverage runner printed invalid JSON: {run.stdout[-2000:]}") from e

        value = score_value(result, ks)
        failures = result["coverage_failures"]
        explanation = (
            f"{result['exec_correct']}/{result['total_cases']} tests executable, "
            f"line coverage {value['line_coverage']:.3f}, "
            f"branch coverage {value['branch_coverage']:.3f}"
        )
        if failures:
            explanation += f"; {len(failures)} coverage run(s) produced no report, counted as 0"
        return Score(
            value=value,
            explanation=explanation,
            metadata={
                "formatted_tests": formatted,
                "test_results": result["tests"],
                "coverage_failures": failures,
            },
        )

    return score


# One test, plus one pytest run for its coverage.
TARGET_RUNNER_TIMEOUT_SECONDS = 300
EXECUTED_OUTCOMES = ("ok", "assertion_error")


async def _run_target_job(
    state: TaskState, check_line: int, measure_errors: bool
) -> dict[str, Any]:
    return await _run_single_test_job(
        state,
        {
            "code": state.metadata["python_solution"],
            "check_line": check_line,
            "measure_errors": measure_errors,
            "coveragerc": COVERAGERC,
        },
    )


async def _run_path_job(state: TaskState) -> dict[str, Any]:
    metadata = state.metadata
    return await _run_single_test_job(
        state,
        {
            # The instrumented program logs each branch it enters to this file.
            "code": metadata["python_solution_instrumented"],
            "log_file": f"test_logs/{metadata['task_title']}.log",
            "target_path": metadata["target_path"],
        },
    )


async def _run_single_test_job(state: TaskState, fields: dict[str, Any]) -> dict[str, Any]:
    """Format the sample's one generated test and run it in the sandbox with ``fields``."""
    func_name = state.metadata["func_name"]
    raw_tests = state.store_as(GeneratedTests).raw_tests
    if len(raw_tests) != 1:
        raise RuntimeError(f"expected one generated test, found {len(raw_tests)}")
    formatted = format_test(raw_tests[0], func_name)
    job = {"func_name": func_name, "test": formatted, **fields}
    job_dir = f"/home/agent/testeval/{state.uuid}"
    env = sandbox()
    await env.write_file(f"{job_dir}/coverage_runner.py", runner_source())
    await env.write_file(f"{job_dir}/job.json", json.dumps(job))
    run = await env.exec(
        ["python", f"{job_dir}/coverage_runner.py", f"{job_dir}/job.json"],
        timeout=TARGET_RUNNER_TIMEOUT_SECONDS,
    )
    if not run.success:
        raise RuntimeError(f"coverage runner failed (exit {run.returncode}): {run.stderr[-2000:]}")
    try:
        result: dict[str, Any] = json.loads(run.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"coverage runner printed invalid JSON: {run.stdout[-2000:]}") from e
    result["formatted_test"] = formatted
    return result


def _target_explanation(result: dict[str, Any], what: str, covered: bool) -> str:
    if not result["syntax_ok"]:
        return f"the test does not compile, so {what} is not covered"
    detail = result["outcome"] + (f" ({result['error']})" if "error" in result else "")
    if not result["called"]:
        detail += ", function under test not called"
    explanation = f"test outcome: {detail}; {what} {'covered' if covered else 'not covered'}"
    if result["coverage_failure"]:
        explanation += "; the coverage run produced no report, counted as not covered"
    return explanation


def _target_metadata(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("formatted_test", "outcome", "error", "called", "line_hit", "coverage_failure")
    return {key: result[key] for key in keys if key in result}


def target_line_metrics() -> list[Metric | dict[str, list[Metric]]]:
    return [{key: [mean()] for key in ("syntax_correct", "exec_correct", "covered")}]


@scorer(metrics=target_line_metrics())
def target_line_coverage() -> Scorer:
    """Whether the generated test executes the target line, as eval_linecov.py checks.

    ``exec_correct`` counts a test that compiles, runs (a failing assertion
    counts) and calls the function under test; ``covered`` additionally needs
    the target line among the lines coverage records as executed.
    """

    async def score(state: TaskState, target: Target) -> Score:
        lineno: int = state.metadata["target_line"]
        result = await _run_target_job(state, check_line=lineno, measure_errors=False)
        executed = result["called"] and result["outcome"] in EXECUTED_OUTCOMES
        covered = executed and result["line_hit"] is True
        return Score(
            value={
                "syntax_correct": float(result["syntax_ok"]),
                "exec_correct": float(executed),
                "covered": float(covered),
            },
            explanation=_target_explanation(result, f"line {lineno}", covered),
            metadata=_target_metadata(result),
        )

    return score


def target_branch_metrics() -> list[Metric | dict[str, list[Metric]]]:
    by_difficulty = grouped(mean(), "branch_difficulty_label", all="samples", all_label="all")
    by_difficulty_reference = grouped(
        mean(), "branch_difficulty_label", all="samples", all_label="all"
    )
    return [
        {
            "syntax_correct": [mean()],
            "exec_correct": [mean()],
            "covered": [by_difficulty],
            "exec_correct_reference": [mean()],
            "covered_reference": [by_difficulty_reference],
        }
    ]


@scorer(metrics=target_branch_metrics())
def target_branch_coverage() -> Scorer:
    """Whether the generated test enters the target branch, as eval_branchcov.py checks.

    A branch counts as covered when the line after its first line (its first
    body line) is executed. ``covered`` is reported for all branches and for
    easy, medium and hard ones.

    The ``_reference`` keys reproduce eval_branchcov.py exactly. It tests
    ``res != False``, so a test that raises an exception other than
    AssertionError still counts as executable, and its partial coverage can
    still cover the branch; only a timeout does not. The plain keys use the
    paper's definition, as eval_linecov.py does: such a test is not executable.
    """

    async def score(state: TaskState, target: Target) -> Score:
        start: int = state.metadata["branch_start"]
        end: int = state.metadata["branch_end"]
        result = await _run_target_job(state, check_line=start + 1, measure_errors=True)
        executed = result["called"] and result["outcome"] in EXECUTED_OUTCOMES
        executed_reference = result["called"] and result["outcome"] in (
            *EXECUTED_OUTCOMES,
            "error",
        )
        hit = result["line_hit"] is True
        covered = executed and hit
        return Score(
            value={
                "syntax_correct": float(result["syntax_ok"]),
                "exec_correct": float(executed),
                "covered": float(covered),
                "exec_correct_reference": float(executed_reference),
                "covered_reference": float(executed_reference and hit),
            },
            explanation=_target_explanation(result, f"branch {start}-{end}", covered),
            metadata=_target_metadata(result),
        )

    return score


def target_path_metrics() -> list[Metric | dict[str, list[Metric]]]:
    keys = (
        "syntax_correct",
        "exec_correct",
        "path_exact_match",
        "path_similarity",
        "path_exact_match_reference",
        "path_similarity_reference",
    )
    return [{key: [mean()] for key in keys}]


@scorer(metrics=target_path_metrics())
def target_path_coverage() -> Scorer:
    """How closely the generated test follows the target execution path, as eval_pathcov.py scores it.

    The instrumented program logs every branch and loop it enters. For a test
    that compiles, runs (a failing assertion counts) and calls the function
    under test, ``path_similarity`` is the longest unbroken run of the target
    path found in the log, as a share of the target path's length, and
    ``path_exact_match`` is 1 when the whole target path appears unbroken in
    the log. Any other test scores 0 on both.

    For the 8 empty target paths the test must enter no branch or loop: it
    scores 1 on both if its log is empty. The ``_reference`` keys reproduce
    eval_pathcov.py exactly, which scores every test 0 on an empty target path
    (its ``match_path`` divides by zero and a bare ``except`` swallows it).
    """

    async def score(state: TaskState, target: Target) -> Score:
        result = await _run_path_job(state)
        executed = result["called"] and result["outcome"] in EXECUTED_OUTCOMES
        similarity = float(result["similarity"]) if executed else 0.0
        similarity_reference = float(result["similarity_reference"]) if executed else 0.0
        if not result["syntax_ok"]:
            explanation = "the test does not compile"
        else:
            detail = result["outcome"] + (f" ({result['error']})" if "error" in result else "")
            if not result["called"]:
                detail += ", function under test not called"
            explanation = f"test outcome: {detail}"
            if executed:
                explanation += (
                    f"; path similarity {similarity:.3f} over {result['path_length']} "
                    f"logged branch entries"
                )
        keys = ("formatted_test", "outcome", "error", "called", "path_length", "path_preview")
        return Score(
            value={
                "syntax_correct": float(result["syntax_ok"]),
                "exec_correct": float(executed),
                "path_exact_match": float(executed and similarity == 1.0),
                "path_similarity": similarity,
                "path_exact_match_reference": float(executed and similarity_reference == 1.0),
                "path_similarity_reference": similarity_reference,
            },
            explanation=explanation,
            metadata={key: result[key] for key in keys if key in result},
        )

    return score
