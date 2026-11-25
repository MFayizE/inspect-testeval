"""TestEval: Benchmarking Large Language Models for Test Case Generation.

Wang et al., NAACL 2025 Findings. https://arxiv.org/abs/2406.04531
Reference implementation: https://github.com/LLM4SoftwareTesting/TestEval
"""

from collections.abc import Sequence

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_evals_utils.metadata import load_eval_version

from inspect_testeval.dataset import (
    DEFAULT_SHUFFLE_SEED,
    load_target_dataset,
    load_testeval_dataset,
)
from inspect_testeval.environment import SANDBOX
from inspect_testeval.scorers import (
    DEFAULT_KS,
    DEFAULT_SEED,
    overall_coverage,
    overall_metrics,
    target_branch_coverage,
    target_line_coverage,
    target_path_coverage,
)
from inspect_testeval.solvers import DEFAULT_NUM_TESTS, overall_tests, targeted_test

# The version comes from eval.yaml, the file beside this module.
EVAL_VERSION = load_eval_version(__file__)

# The paper decodes with temperature 0; the reference scripts cap replies at 256 tokens.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256
GENERATE_CONFIG = GenerateConfig(temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS)


@task
def testeval_overall(
    num_tests: int = DEFAULT_NUM_TESTS,
    ks: Sequence[int] = DEFAULT_KS,
    shuffle: bool | int = DEFAULT_SHUFFLE_SEED,
    cov_at_k_seed: int = DEFAULT_SEED,
) -> Task:
    """Overall coverage: generate ``num_tests`` different tests per program in one conversation.

    Reports syntax and execution correctness over all generated tests, the line
    and branch coverage of each program's executable tests, and cov@k for each
    k in ``ks`` (paper definition, plus ``_reference`` keys for the reference
    code's definition).

    Args:
        num_tests: Tests requested per program (the paper uses 20).
        ks: Group sizes for cov@k.
        shuffle: Dataset order: a seed, ``True`` for an unseeded shuffle, or ``False``.
        cov_at_k_seed: Seed for splitting each program's tests into cov@k groups.
    """
    return Task(
        dataset=load_testeval_dataset(shuffle=shuffle),
        solver=overall_tests(num_tests=num_tests),
        scorer=overall_coverage(ks=ks, seed=cov_at_k_seed),
        metrics=overall_metrics(ks),
        sandbox=SANDBOX,
        config=GENERATE_CONFIG,
        version=EVAL_VERSION,
    )


@task
def testeval_line(shuffle: bool | int = DEFAULT_SHUFFLE_SEED) -> Task:
    """Targeted line coverage: one test per target line, which must execute that line.

    1,340 samples, one per target line across the 210 programs. Reports syntax
    and execution correctness and the share of target lines covered.

    Args:
        shuffle: Sample order: a seed, ``True`` for an unseeded shuffle, or ``False``.
    """
    return Task(
        dataset=load_target_dataset("line", shuffle=shuffle),
        solver=targeted_test(),
        scorer=target_line_coverage(),
        sandbox=SANDBOX,
        config=GENERATE_CONFIG,
        version=EVAL_VERSION,
    )


@task
def testeval_branch(shuffle: bool | int = DEFAULT_SHUFFLE_SEED) -> Task:
    """Targeted branch coverage: one test per target branch, which must enter that branch.

    983 samples, one per target branch across the 210 programs. Reports syntax
    and execution correctness and the share of target branches covered, for
    all branches and by branch difficulty (easy, medium, hard). The
    ``_reference`` keys reproduce the reference scorer, which also counts a
    test that raised an exception as executable.

    Args:
        shuffle: Sample order: a seed, ``True`` for an unseeded shuffle, or ``False``.
    """
    return Task(
        dataset=load_target_dataset("branch", shuffle=shuffle),
        solver=targeted_test(),
        scorer=target_branch_coverage(),
        sandbox=SANDBOX,
        config=GENERATE_CONFIG,
        version=EVAL_VERSION,
    )


@task
def testeval_path(shuffle: bool | int = DEFAULT_SHUFFLE_SEED) -> Task:
    """Targeted path coverage: one test per target execution path, which it must follow.

    854 samples, one per sampled execution path across the 210 programs. The
    prompt gives the path as a sequence of branch conditions. Reports syntax
    and execution correctness, the share of paths followed exactly and the mean
    path similarity.

    Args:
        shuffle: Sample order: a seed, ``True`` for an unseeded shuffle, or ``False``.
    """
    return Task(
        dataset=load_target_dataset("path", shuffle=shuffle),
        solver=targeted_test(),
        scorer=target_path_coverage(),
        sandbox=SANDBOX,
        config=GENERATE_CONFIG,
        version=EVAL_VERSION,
    )
