import math
import urllib.request
from collections.abc import Callable
from typing import cast

import pytest
from inspect_ai.scorer import Metric, SampleScore, Score
from pytest_inspect_evals.metrics import assert_agreeing_epochs_change_nothing, run_metrics

from inspect_testeval.dataset import DATASET_COMMIT
from inspect_testeval.scorers import (
    COVERAGERC,
    coverage_keys,
    execution_correctness,
    overall_coverage,
    overall_metrics,
    runner_source,
    score_value,
    syntax_correctness,
)


def _sample_score(**value: float) -> SampleScore:
    return SampleScore(score=Score(value=value))


def _call(metric: Metric, scores: list[SampleScore]) -> float:
    # Metric is typed as a union of call signatures; these metrics take SampleScores.
    return cast(Callable[[list[SampleScore]], float], metric)(scores)


def _result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "total_cases": 3,
        "syntax_correct": 2,
        "exec_correct": 1,
        "line_coverage": 0.5,
        "branch_coverage": 0.25,
        "cov_at_k": {
            "paper": {"1": {"line": 0.5, "branch": 0.25}},
            "reference": {"1": {"line": 0.5, "branch": 0.25}},
        },
        "tests": [],
    }
    result.update(overrides)
    return result


@pytest.mark.dataset_download
def test_coveragerc_matches_upstream() -> None:
    url = (
        "https://raw.githubusercontent.com/LLM4SoftwareTesting/TestEval/"
        f"{DATASET_COMMIT}/.coveragerc"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        assert response.read().decode("utf-8") == COVERAGERC


def test_runner_source_is_the_packaged_script() -> None:
    source = runner_source()
    assert "def evaluate(" in source
    assert 'if __name__ == "__main__":' in source


def test_coverage_keys_cover_both_cov_at_k_definitions() -> None:
    assert coverage_keys([1, 5]) == [
        "line_coverage",
        "branch_coverage",
        "line_cov@1",
        "branch_cov@1",
        "line_cov@1_reference",
        "branch_cov@1_reference",
        "line_cov@5",
        "branch_cov@5",
        "line_cov@5_reference",
        "branch_cov@5_reference",
    ]


def test_score_value_has_every_metric_key() -> None:
    value = score_value(_result(), [1])
    assert set(value) == {"total_cases", "syntax_correct", "exec_correct", *coverage_keys([1])}
    assert value["line_cov@1"] == 0.5
    assert value["branch_cov@1_reference"] == 0.25


def test_score_value_counts_unmeasured_coverage_as_zero() -> None:
    # Deliberate zero: eval_overall.py adds nothing for a program without
    # executable tests but still divides by the number of programs.
    value = score_value(
        _result(exec_correct=0, line_coverage=None, branch_coverage=None, cov_at_k=None), [1, 2]
    )
    for key in coverage_keys([1, 2]):
        assert value[key] == 0.0


def test_correctness_metrics_divide_summed_counts() -> None:
    scores = [
        _sample_score(total_cases=10, syntax_correct=10, exec_correct=9),
        _sample_score(total_cases=30, syntax_correct=15, exec_correct=3),
    ]
    # Micro-average over all tests, not the mean of per-program rates.
    assert _call(syntax_correctness(), scores) == pytest.approx(25 / 40)
    assert _call(execution_correctness(), scores) == pytest.approx(12 / 40)


def test_correctness_metrics_report_nan_without_tests() -> None:
    scores = [_sample_score(total_cases=0, syntax_correct=0, exec_correct=0)]
    assert math.isnan(_call(syntax_correctness(), scores))
    assert math.isnan(_call(execution_correctness(), scores))


def test_correctness_metric_rejects_non_dict_values() -> None:
    with pytest.raises(ValueError, match="dict score value"):
        _call(syntax_correctness(), [SampleScore(score=Score(value=1.0))])


def _value(total: float, syntax: float, executed: float, line: float) -> dict[str, float]:
    value = {"total_cases": total, "syntax_correct": syntax, "exec_correct": executed}
    value.update(dict.fromkeys(coverage_keys([1]), line))
    return value


def test_metrics_unchanged_by_agreeing_epochs() -> None:
    assert_agreeing_epochs_change_nothing(
        overall_metrics([1]),
        [_value(10, 10, 9, 0.8), _value(10, 4, 1, 0.0), _value(10, 0, 0, 0.0)],
    )


def test_metrics_after_mean_reduction_of_disagreeing_epochs() -> None:
    results = run_metrics(
        overall_metrics([1]),
        [
            [_value(10, 10, 10, 1.0), _value(10, 0, 0, 0.0)],  # program A, two epochs
            [_value(10, 10, 6, 0.5), _value(10, 10, 4, 0.5)],  # program B, two epochs
        ],
        reducer="mean",
    )
    # Every epoch asks for the same number of tests, so the ratio of reduced
    # counts equals the share over all 40 generated tests.
    assert results["syntax_correctness"] == pytest.approx(30 / 40)
    assert results["execution_correctness"] == pytest.approx(20 / 40)
    assert results["line_coverage/mean"] == pytest.approx((0.5 + 0.5) / 2)


@pytest.mark.parametrize("ks", [[], [0], [1, -2]])
def test_scorer_rejects_invalid_ks(ks: list[int]) -> None:
    with pytest.raises(ValueError, match="ks must be positive"):
        overall_coverage(ks=ks)
