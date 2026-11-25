"""TestEval: Benchmarking Large Language Models for Test Case Generation.

Wang et al., NAACL 2025 Findings. https://arxiv.org/abs/2406.04531
Reference implementation: https://github.com/LLM4SoftwareTesting/TestEval
"""

from collections.abc import Sequence

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_evals_utils.metadata import load_eval_version

from inspect_testeval.dataset import DEFAULT_SHUFFLE_SEED, load_testeval_dataset
from inspect_testeval.environment import SANDBOX
from inspect_testeval.scorers import DEFAULT_KS, DEFAULT_SEED, overall_coverage, overall_metrics
from inspect_testeval.solvers import DEFAULT_NUM_TESTS, overall_tests

# The version comes from eval.yaml, the file beside this module.
EVAL_VERSION = load_eval_version(__file__)

# The paper decodes with temperature 0; the reference scripts cap replies at 256 tokens.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256


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
        config=GenerateConfig(temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS),
        version=EVAL_VERSION,
    )
