"""Load the TestEval dataset (210 LeetCode programs) from the authors' repository.

The data comes from LLM4SoftwareTesting/TestEval (MIT licence), pinned to a
commit so results stay reproducible. `leetcode-py-all.jsonl` combines the three
files the original evaluation scripts read (`leetcode-py.jsonl`,
`leetcode-py-instrumented.jsonl` and `tgt_paths.jsonl`); its fields are
identical to theirs for every program.
"""

from typing import Any

from inspect_ai.dataset import Dataset, Sample
from inspect_evals_utils.datasets import load_json_dataset
from inspect_evals_utils.shuffle import shuffle_and_seed

DATASET_COMMIT = "6f67dee71de6ccda1767b8c219a93f4ae0a7ce90"
DATASET_URL = (
    "https://raw.githubusercontent.com/LLM4SoftwareTesting/TestEval/"
    f"{DATASET_COMMIT}/data/leetcode-py-all.jsonl"
)
EVAL_NAME = "inspect_testeval"

# Records are sorted by LeetCode problem number, which tracks release date, so an
# unshuffled `--limit N` run would only see the oldest problems. Shuffle with a
# fixed seed by default so every run with the same limit sees the same programs.
DEFAULT_SHUFFLE_SEED = 42

# Program-level difficulty in the data is 1/2/3 (the upstream README says 0-2).
PROGRAM_DIFFICULTY = {1: "easy", 2: "medium", 3: "hard"}

METADATA_FIELDS = (
    "task_num",
    "task_title",
    "difficulty",
    "func_name",
    "description",
    "python_solution",
    "python_solution_instrumented",
    "target_lines",
    "blocks",
    "sampled_paths",
    "sampled_condition_paths",
)


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert one TestEval program into a sample.

    Every task in this eval works on the same 210 programs; the task-specific
    prompts are built by the solvers from the metadata, so the sample input is
    the program under test.
    """
    missing = [field for field in METADATA_FIELDS if field not in record]
    if missing:
        raise ValueError(f"TestEval record {record.get('task_num')!r} is missing {missing}")
    if record["difficulty"] not in PROGRAM_DIFFICULTY:
        raise ValueError(
            f"TestEval record {record['task_num']!r} has unknown difficulty "
            f"{record['difficulty']!r}"
        )
    metadata = {field: record[field] for field in METADATA_FIELDS}
    metadata["difficulty_label"] = PROGRAM_DIFFICULTY[record["difficulty"]]
    return Sample(
        id=record["task_num"],
        input=record["python_solution"],
        metadata=metadata,
    )


def load_testeval_dataset(shuffle: bool | int = DEFAULT_SHUFFLE_SEED) -> Dataset:
    """Load the pinned TestEval programs as an Inspect dataset.

    Args:
        shuffle: A seed, ``True`` for an unseeded shuffle, or ``False`` to keep
            the upstream order (sorted by LeetCode problem number).
    """
    should_shuffle, seed = shuffle_and_seed(shuffle)
    return load_json_dataset(
        DATASET_URL,
        EVAL_NAME,
        sample_fields=record_to_sample,
        shuffle=should_shuffle,
        seed=seed,
    )
