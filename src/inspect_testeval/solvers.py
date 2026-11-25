"""Solvers that ask the model for test cases, following the TestEval generation scripts."""

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import StoreModel
from pydantic import Field

from inspect_testeval.prompts import OVERALL_FOLLOW_UP, overall_prompt, system_message

# The paper queries every model for 20 rounds (N = 20, Section 3.2); the
# reference scripts default to 10.
DEFAULT_NUM_TESTS = 20


class GeneratedTests(StoreModel):
    """Raw model replies, one per requested test, in generation order."""

    raw_tests: list[str] = Field(default_factory=list)


@solver
def overall_tests(num_tests: int = DEFAULT_NUM_TESTS) -> Solver:
    """Collect ``num_tests`` tests in one conversation, as ``testgeneration_multiround`` does.

    The first user turn is the overall-coverage prompt. After each reply the
    model is asked for another, different test, so later tests see the earlier
    ones in context.
    """
    if num_tests < 1:
        raise ValueError(f"num_tests must be at least 1, got {num_tests}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata
        state.messages = [
            ChatMessageSystem(content=system_message()),
            ChatMessageUser(
                content=overall_prompt(
                    program=metadata["python_solution"],
                    description=metadata["description"],
                    func_name=metadata["func_name"],
                )
            ),
        ]
        store = state.store_as(GeneratedTests)
        for round_index in range(num_tests):
            state = await generate(state)
            # Assign rather than append so the store records the change.
            store.raw_tests = [*store.raw_tests, state.output.completion]
            if round_index < num_tests - 1:
                state.messages.append(ChatMessageUser(content=OVERALL_FOLLOW_UP))
        return state

    return solve
