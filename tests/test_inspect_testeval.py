from collections.abc import Callable

import pytest
from inspect_ai import Task, eval
from inspect_ai.model import ModelOutput, get_model
from pytest_inspect_evals.assertions import assert_eval_success

from inspect_testeval import testeval_branch, testeval_line, testeval_overall, testeval_path
from inspect_testeval.environment import SANDBOX_COMPOSE_FILE
from inspect_testeval.inspect_testeval import EVAL_VERSION
from inspect_testeval.scorers import coverage_keys

# The first program in upstream order (shuffle=False) is LeetCode 4.
FIRST_FUNC = "findMedianSortedArrays"
GOOD_TEST = (
    "def test_findMedianSortedArrays():\n"
    "    solution=Solution()\n"
    "    assert solution.findMedianSortedArrays([1, 3], [2]) == 2.0"
)


def test_version_comes_from_eval_yaml() -> None:
    assert EVAL_VERSION == "1-A"


@pytest.mark.dataset_download
def test_task_defaults_follow_the_paper() -> None:
    task = testeval_overall()
    assert len(task.dataset) == 210
    assert task.config.temperature == 0.0
    assert task.config.max_tokens == 256
    assert task.sandbox is not None
    assert task.sandbox.type == "docker"
    assert task.sandbox.config == str(SANDBOX_COMPOSE_FILE)


@pytest.mark.dataset_download
def test_task_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="num_tests"):
        testeval_overall(num_tests=0)
    with pytest.raises(ValueError, match="ks must be positive"):
        testeval_overall(ks=[0])


@pytest.mark.dataset_download
@pytest.mark.docker
@pytest.mark.slow(300)
def test_end_to_end_scores_good_tests() -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", GOOD_TEST)] * 2,
    )
    [log] = eval(
        tasks=testeval_overall(num_tests=2, ks=[1, 2], shuffle=False),
        model=model,
        limit=1,
        display="none",
    )
    assert_eval_success(log, expected_sample_count=1)
    assert log.samples is not None
    assert log.samples[0].metadata["func_name"] == FIRST_FUNC
    scores = log.samples[0].scores
    assert scores is not None
    value = scores["overall_coverage"].value
    assert isinstance(value, dict)
    assert value["total_cases"] == 2
    assert value["exec_correct"] == 2
    line_coverage = value["line_coverage"]
    assert isinstance(line_coverage, float)
    assert 0.0 < line_coverage <= 1.0
    assert set(coverage_keys([1, 2])) <= set(value)
    assert log.results is not None
    metric_names = {m for s in log.results.scores for m in s.metrics}
    assert {"syntax_correctness", "execution_correctness"} <= metric_names


@pytest.mark.dataset_download
@pytest.mark.docker
@pytest.mark.slow(300)
def test_end_to_end_with_default_mock_output() -> None:
    # mockllm's default reply is prose, so no test compiles and coverage is 0.
    [log] = eval(
        tasks=testeval_overall(num_tests=2, shuffle=False),
        model="mockllm/model",
        limit=1,
        display="none",
    )
    assert_eval_success(log, expected_sample_count=1)
    assert log.samples is not None
    scores = log.samples[0].scores
    assert scores is not None
    value = scores["overall_coverage"].value
    assert isinstance(value, dict)
    assert value["exec_correct"] == 0
    assert value["line_coverage"] == 0.0


@pytest.mark.dataset_download
@pytest.mark.parametrize(("task", "samples"), [(testeval_line, 1340), (testeval_branch, 983)])
def test_targeted_task_defaults_follow_the_paper(task: Callable[[], Task], samples: int) -> None:
    built = task()
    assert len(built.dataset) == samples
    assert built.config.temperature == 0.0
    assert built.config.max_tokens == 256
    assert built.sandbox is not None
    assert built.sandbox.config == str(SANDBOX_COMPOSE_FILE)


def _single_reply_eval(task: Task, reply: str) -> dict[str, object]:
    model = get_model(
        "mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", reply)]
    )
    [log] = eval(tasks=task, model=model, limit=1, display="none")
    assert_eval_success(log, expected_sample_count=1)
    assert log.samples is not None
    scores = log.samples[0].scores
    assert scores is not None
    [score] = scores.values()
    assert isinstance(score.value, dict)
    return {"id": log.samples[0].id, **score.value}


@pytest.mark.dataset_download
@pytest.mark.docker
@pytest.mark.slow(300)
def test_line_end_to_end_covers_the_target() -> None:
    # The first target in upstream order is line 16 of LeetCode 4, the
    # recursive call taken when nums1 is the longer array, as in GOOD_TEST.
    value = _single_reply_eval(testeval_line(shuffle=False), GOOD_TEST)
    assert value == {
        "id": "4_line_16",
        "syntax_correct": 1.0,
        "exec_correct": 1.0,
        "covered": 1.0,
    }


@pytest.mark.dataset_download
@pytest.mark.docker
@pytest.mark.slow(300)
def test_branch_end_to_end_separates_reference_definition() -> None:
    # Enters branch 15-16 of LeetCode 4, then raises ZeroDivisionError: the
    # reference scorer counts that as executed and covered, the paper does not.
    raises = (
        "def test_findMedianSortedArrays():\n"
        "    solution=Solution()\n"
        "    assert solution.findMedianSortedArrays([1, 3], [2]) == 1 / 0"
    )
    value = _single_reply_eval(testeval_branch(shuffle=False), raises)
    assert value == {
        "id": "4_branch_15_16",
        "syntax_correct": 1.0,
        "exec_correct": 0.0,
        "covered": 0.0,
        "exec_correct_reference": 1.0,
        "covered_reference": 1.0,
    }


@pytest.mark.dataset_download
def test_path_task_defaults_follow_the_paper() -> None:
    task = testeval_path()
    assert len(task.dataset) == 854
    assert task.config.temperature == 0.0
    assert task.config.max_tokens == 256


@pytest.mark.dataset_download
@pytest.mark.docker
@pytest.mark.slow(300)
def test_path_end_to_end_follows_the_target() -> None:
    # The first path in upstream order is path 0 of LeetCode 4: two loop
    # iterations ending in the odd-length branch. GOOD_TEST takes it, after
    # first swapping the arrays (an extra leading entry, still a full match).
    value = _single_reply_eval(testeval_path(shuffle=False), GOOD_TEST)
    assert value == {
        "id": "4_path_0",
        "syntax_correct": 1.0,
        "exec_correct": 1.0,
        "path_exact_match": 1.0,
        "path_similarity": 1.0,
        "path_exact_match_reference": 1.0,
        "path_similarity_reference": 1.0,
    }
