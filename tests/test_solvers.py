import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
    get_model,
)

from inspect_testeval.prompts import OVERALL_FOLLOW_UP, overall_prompt, system_message
from inspect_testeval.solvers import GeneratedTests, overall_tests, targeted_test

PROGRAM = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
METADATA = {"python_solution": PROGRAM, "description": "Add two numbers.", "func_name": "add"}


def _run(num_tests: int, replies: list[str]):
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", r) for r in replies],
    )
    task = Task(
        dataset=[Sample(id=1, input=PROGRAM, metadata=METADATA)],
        solver=overall_tests(num_tests=num_tests),
    )
    [log] = eval(task, model=model, display="none")
    assert log.status == "success", log.error
    assert log.samples is not None
    return log.samples[0]


def test_collects_one_reply_per_round_in_order() -> None:
    replies = ["def test_add():\n    pass", "second", "third"]
    sample = _run(3, replies)
    assert sample.store_as(GeneratedTests).raw_tests == replies


def test_conversation_matches_reference_multiround_shape() -> None:
    sample = _run(3, ["r1", "r2", "r3"])
    messages = sample.messages
    assert isinstance(messages[0], ChatMessageSystem)
    assert messages[0].text == system_message()
    assert isinstance(messages[1], ChatMessageUser)
    assert messages[1].text == overall_prompt(PROGRAM, "Add two numbers.", "add")
    # system, prompt, then reply / follow-up pairs; no follow-up after the last reply.
    roles = [m.role for m in messages[2:]]
    assert roles == ["assistant", "user", "assistant", "user", "assistant"]
    follow_ups = [m.text for m in messages[2:] if isinstance(m, ChatMessageUser)]
    assert follow_ups == [OVERALL_FOLLOW_UP, OVERALL_FOLLOW_UP]
    assert [m.text for m in messages if isinstance(m, ChatMessageAssistant)] == ["r1", "r2", "r3"]


def test_rejects_non_positive_num_tests() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        overall_tests(num_tests=0)


def test_targeted_test_asks_once_in_a_fresh_conversation() -> None:
    prompt = "Your test case must cover line 3."
    model = get_model(
        "mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", "reply")]
    )
    task = Task(
        dataset=[Sample(id="1_line_3", input=prompt, metadata=METADATA)],
        solver=targeted_test(),
    )
    [log] = eval(task, model=model, display="none")
    assert log.status == "success", log.error
    assert log.samples is not None
    sample = log.samples[0]
    assert [m.role for m in sample.messages] == ["system", "user", "assistant"]
    assert sample.messages[0].text == system_message()
    assert sample.messages[1].text == prompt
    assert sample.store_as(GeneratedTests).raw_tests == ["reply"]
