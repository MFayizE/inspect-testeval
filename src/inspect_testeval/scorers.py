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
