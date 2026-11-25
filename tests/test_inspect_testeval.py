import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model
from pytest_inspect_evals.assertions import assert_eval_success

from inspect_testeval import testeval_overall
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
