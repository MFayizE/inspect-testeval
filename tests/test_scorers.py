import asyncio
import math
import urllib.request
from collections.abc import Callable, Coroutine
from typing import Any, cast

import pytest
from inspect_ai.model import ModelName
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target
from inspect_ai.solver import TaskState
from pytest_inspect_evals.metrics import assert_agreeing_epochs_change_nothing, run_metrics

from inspect_testeval import scorers
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
    target_branch_coverage,
    target_branch_metrics,
    target_line_coverage,
    target_line_metrics,
    target_path_coverage,
    target_path_metrics,
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


def _branch_value(executed: float, covered: float, reference: float) -> dict[str, float]:
    return {
        "syntax_correct": 1.0,
        "exec_correct": executed,
        "covered": covered,
        "exec_correct_reference": reference,
        "covered_reference": reference,
    }


BRANCH_METADATA = [
    {"branch_difficulty_label": "easy"},
    {"branch_difficulty_label": "easy"},
    {"branch_difficulty_label": "hard"},
]


def test_branch_metrics_report_coverage_by_difficulty() -> None:
    results = run_metrics(
        target_branch_metrics(),
        [
            [_branch_value(1.0, 1.0, 1.0)],
            [_branch_value(0.0, 0.0, 1.0)],  # raised after entering the branch
            [_branch_value(1.0, 0.0, 1.0)],
        ],
        metadata=BRANCH_METADATA,
    )
    covered = {k: v for k, v in results.items() if k.startswith("covered/")}
    reference = {k: v for k, v in results.items() if k.startswith("covered_reference/")}
    assert covered == {
        "covered/grouped_all": pytest.approx(1 / 3),
        "covered/grouped_easy": pytest.approx(1 / 2),
        "covered/grouped_hard": pytest.approx(0.0),
    }
    assert reference == {
        "covered_reference/grouped_all": pytest.approx(1.0),
        "covered_reference/grouped_easy": pytest.approx(1.0),
        "covered_reference/grouped_hard": pytest.approx(1.0),
    }
    assert results["exec_correct/mean"] == pytest.approx(2 / 3)


def test_target_metrics_unchanged_by_agreeing_epochs() -> None:
    assert_agreeing_epochs_change_nothing(
        target_branch_metrics(),
        [_branch_value(1.0, 1.0, 1.0), _branch_value(0.0, 0.0, 1.0), _branch_value(1.0, 0.0, 0.0)],
        metadata=BRANCH_METADATA,
    )
    line = {"syntax_correct": 1.0, "exec_correct": 1.0, "covered": 0.0}
    assert_agreeing_epochs_change_nothing(target_line_metrics(), [line, {**line, "covered": 1.0}])


def test_target_line_metrics_after_mean_reduction() -> None:
    results = run_metrics(
        target_line_metrics(),
        [
            [
                {"syntax_correct": 1.0, "exec_correct": 1.0, "covered": 1.0},
                {"syntax_correct": 1.0, "exec_correct": 1.0, "covered": 0.0},
            ],
            [{"syntax_correct": 0.0, "exec_correct": 0.0, "covered": 0.0}] * 2,
        ],
        reducer="mean",
    )
    assert results["covered/mean"] == pytest.approx(0.25)
    assert results["syntax_correct/mean"] == pytest.approx(0.5)


def _state(metadata: dict[str, object]) -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="s",
        epoch=1,
        input="prompt",
        messages=[],
        metadata={"func_name": "f", "python_solution": "", **metadata},
    )


def _runner_result(outcome: str, called: bool = True, line_hit: bool | None = True) -> dict:
    return {
        "syntax_ok": outcome != "syntax_error",
        "outcome": outcome,
        "called": called,
        "line_hit": line_hit,
        "coverage_failure": False,
        "formatted_test": "def test_f(): ...",
    }


def _score_with(
    monkeypatch: pytest.MonkeyPatch, scorer: Scorer, state: TaskState, result: dict
) -> tuple[dict[str, float], list[tuple[int, bool]]]:
    calls: list[tuple[int, bool]] = []

    async def fake_job(state: TaskState, check_line: int, measure_errors: bool) -> dict:
        calls.append((check_line, measure_errors))
        return result

    monkeypatch.setattr(scorers, "_run_target_job", fake_job)
    score = asyncio.run(scorer(state, Target("")))
    assert score is not None
    assert isinstance(score.value, dict)
    return cast(dict[str, float], score.value), calls


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_runner_result("ok"), (1.0, 1.0, 1.0)),
        (_runner_result("assertion_error"), (1.0, 1.0, 1.0)),
        (_runner_result("ok", line_hit=False), (1.0, 1.0, 0.0)),
        (_runner_result("ok", called=False, line_hit=None), (1.0, 0.0, 0.0)),
        (_runner_result("error", line_hit=None), (1.0, 0.0, 0.0)),
        (_runner_result("timeout", line_hit=None), (1.0, 0.0, 0.0)),
        (_runner_result("syntax_error", called=False, line_hit=None), (0.0, 0.0, 0.0)),
    ],
)
def test_target_line_coverage_scores_each_outcome(
    monkeypatch: pytest.MonkeyPatch, result: dict, expected: tuple[float, float, float]
) -> None:
    value, calls = _score_with(
        monkeypatch, target_line_coverage(), _state({"target_line": 7}), result
    )
    assert calls == [(7, False)]  # the target line itself; errors are not measured
    assert (value["syntax_correct"], value["exec_correct"], value["covered"]) == expected


@pytest.mark.parametrize(
    ("result", "paper", "reference"),
    [
        (_runner_result("ok"), (1.0, 1.0), (1.0, 1.0)),
        # Raised after entering the branch: only the reference scorer counts it.
        (_runner_result("error"), (0.0, 0.0), (1.0, 1.0)),
        (_runner_result("error", line_hit=False), (0.0, 0.0), (1.0, 0.0)),
        (_runner_result("timeout", line_hit=None), (0.0, 0.0), (0.0, 0.0)),
        (_runner_result("error", called=False, line_hit=None), (0.0, 0.0), (0.0, 0.0)),
    ],
)
def test_target_branch_coverage_scores_both_definitions(
    monkeypatch: pytest.MonkeyPatch,
    result: dict,
    paper: tuple[float, float],
    reference: tuple[float, float],
) -> None:
    state = _state({"branch_start": 10, "branch_end": 12})
    value, calls = _score_with(monkeypatch, target_branch_coverage(), state, result)
    assert calls == [(11, True)]  # the branch's first body line; errors are measured
    assert (value["exec_correct"], value["covered"]) == paper
    assert (value["exec_correct_reference"], value["covered_reference"]) == reference


def test_target_score_explains_a_coverage_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    result = {**_runner_result("ok", line_hit=None), "coverage_failure": True}
    state = _state({"target_line": 3})
    monkeypatch.setattr(scorers, "_run_target_job", _async_returning(result))
    score = asyncio.run(target_line_coverage()(state, Target("")))
    assert score is not None
    assert score.explanation is not None
    assert "line 3 not covered" in score.explanation
    assert "produced no report" in score.explanation


def _async_returning(result: dict) -> Callable[..., Coroutine[Any, Any, dict]]:
    async def fake_job(*args: object, **kwargs: object) -> dict:
        return result

    return fake_job


def _path_result(
    outcome: str, similarity: float | None, reference: float | None, called: bool = True
) -> dict:
    return {
        "syntax_ok": outcome != "syntax_error",
        "outcome": outcome,
        "called": called,
        "similarity": similarity,
        "similarity_reference": reference,
        "path_length": 3,
        "formatted_test": "def test_f(): ...",
    }


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        # syntax, exec, exact, similarity, exact_reference, similarity_reference
        (_path_result("ok", 1.0, 1.0), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        (_path_result("assertion_error", 0.5, 0.5), (1.0, 1.0, 0.0, 0.5, 0.0, 0.5)),
        # Empty target path followed: only this eval's definition credits it.
        (_path_result("ok", 1.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)),
        (_path_result("error", None, None), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        (_path_result("ok", None, None, called=False), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        (_path_result("syntax_error", None, None, called=False), (0.0,) + (0.0,) * 5),
    ],
)
def test_target_path_coverage_scores_each_outcome(
    monkeypatch: pytest.MonkeyPatch, result: dict, expected: tuple[float, ...]
) -> None:
    monkeypatch.setattr(scorers, "_run_path_job", _async_returning(result))
    score = asyncio.run(target_path_coverage()(_state({}), Target("")))
    assert score is not None
    assert isinstance(score.value, dict)
    keys = (
        "syntax_correct",
        "exec_correct",
        "path_exact_match",
        "path_similarity",
        "path_exact_match_reference",
        "path_similarity_reference",
    )
    assert tuple(score.value[k] for k in keys) == expected


def test_path_metrics_unchanged_by_agreeing_epochs() -> None:
    value = {
        "syntax_correct": 1.0,
        "exec_correct": 1.0,
        "path_exact_match": 0.0,
        "path_similarity": 0.5,
        "path_exact_match_reference": 0.0,
        "path_similarity_reference": 0.5,
    }
    assert_agreeing_epochs_change_nothing(
        target_path_metrics(), [value, {**value, "path_similarity": 1.0}]
    )
