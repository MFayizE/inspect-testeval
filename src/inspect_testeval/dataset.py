"""Load the TestEval dataset (210 LeetCode programs) from the authors' repository.

The data comes from LLM4SoftwareTesting/TestEval (MIT licence), pinned to a
commit so results stay reproducible. `leetcode-py-all.jsonl` combines the three
files the original evaluation scripts read (`leetcode-py.jsonl`,
`leetcode-py-instrumented.jsonl` and `tgt_paths.jsonl`): its fields equal
those of `leetcode-py.jsonl`, plus the instrumented program and the target
paths from the other two. `leetcode-py-instrumented.jsonl` has different
descriptions for 14 programs (it keeps exponents such as `10^9`), so the path
task reads its descriptions from that file, as the original path generator
does.
"""

from functools import partial
from typing import Any, Literal

from inspect_ai.dataset import Dataset, RecordToSample, Sample
from inspect_evals_utils.datasets import load_json_dataset
from inspect_evals_utils.shuffle import shuffle_and_seed

from inspect_testeval.prompts import branch_prompt, line_prompt, path_prompt

DATASET_COMMIT = "6f67dee71de6ccda1767b8c219a93f4ae0a7ce90"
DATASET_URL = (
    "https://raw.githubusercontent.com/LLM4SoftwareTesting/TestEval/"
    f"{DATASET_COMMIT}/data/leetcode-py-all.jsonl"
)
# Only the path task reads this file, for its program descriptions.
INSTRUMENTED_URL = (
    "https://raw.githubusercontent.com/LLM4SoftwareTesting/TestEval/"
    f"{DATASET_COMMIT}/data/leetcode-py-instrumented.jsonl"
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


# Branch-level difficulty in `blocks` is 0/1/2. eval_branchcov.py reports the
# targeted-branch result separately for each.
BRANCH_DIFFICULTY = {0: "easy", 1: "medium", 2: "hard"}

# The program fields a targeted sample keeps; its prompt already holds the rest.
TARGET_METADATA_FIELDS = (
    "task_num",
    "task_title",
    "difficulty",
    "func_name",
    "description",
    "python_solution",
)


def _validate(record: dict[str, Any]) -> None:
    missing = [field for field in METADATA_FIELDS if field not in record]
    if missing:
        raise ValueError(f"TestEval record {record.get('task_num')!r} is missing {missing}")
    if record["difficulty"] not in PROGRAM_DIFFICULTY:
        raise ValueError(
            f"TestEval record {record['task_num']!r} has unknown difficulty "
            f"{record['difficulty']!r}"
        )


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert one TestEval program into a sample.

    The overall task works on whole programs; its prompt is built by the
    solver from the metadata, so the sample input is the program under test.
    """
    _validate(record)
    metadata = {field: record[field] for field in METADATA_FIELDS}
    metadata["difficulty_label"] = PROGRAM_DIFFICULTY[record["difficulty"]]
    return Sample(
        id=record["task_num"],
        input=record["python_solution"],
        metadata=metadata,
    )


def _program_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = {field: record[field] for field in TARGET_METADATA_FIELDS}
    metadata["difficulty_label"] = PROGRAM_DIFFICULTY[record["difficulty"]]
    return metadata


def record_to_line_samples(record: dict[str, Any]) -> list[Sample]:
    """One sample per target line of a program, with the targeted-line prompt as input."""
    _validate(record)
    num_lines = len(record["python_solution"].split("\n"))
    samples = []
    for lineno in record["target_lines"]:
        if not 1 <= lineno <= num_lines:
            raise ValueError(
                f"TestEval record {record['task_num']!r} targets line {lineno}, "
                f"outside its {num_lines} lines"
            )
        samples.append(
            Sample(
                id=f"{record['task_num']}_line_{lineno}",
                input=line_prompt(
                    record["python_solution"],
                    record["description"],
                    record["func_name"],
                    lineno,
                ),
                metadata=_program_metadata(record) | {"target_line": lineno},
            )
        )
    return samples


def record_to_branch_samples(record: dict[str, Any]) -> list[Sample]:
    """One sample per target branch of a program, with the targeted-branch prompt as input."""
    _validate(record)
    num_lines = len(record["python_solution"].split("\n"))
    samples = []
    for block in record["blocks"]:
        start, end = block["start"], block["end"]
        if not 1 <= start < end <= num_lines:
            raise ValueError(
                f"TestEval record {record['task_num']!r} has branch {start}-{end}, "
                f"outside its {num_lines} lines"
            )
        if block["difficulty"] not in BRANCH_DIFFICULTY:
            raise ValueError(
                f"TestEval record {record['task_num']!r} branch {start}-{end} has unknown "
                f"difficulty {block['difficulty']!r}"
            )
        samples.append(
            Sample(
                id=f"{record['task_num']}_branch_{start}_{end}",
                input=branch_prompt(
                    record["python_solution"],
                    record["description"],
                    record["func_name"],
                    start,
                    end,
                ),
                metadata=_program_metadata(record)
                | {
                    "branch_start": start,
                    "branch_end": end,
                    "branch_type": block["type"],
                    "branch_difficulty": block["difficulty"],
                    "branch_difficulty_label": BRANCH_DIFFICULTY[block["difficulty"]],
                },
            )
        )
    return samples


def record_to_path_samples(record: dict[str, Any], descriptions: dict[int, str]) -> list[Sample]:
    """One sample per target execution path of a program, with the targeted-path prompt as input.

    ``descriptions`` maps each program to its description in
    leetcode-py-instrumented.jsonl, which generate_pathcov_*.py reads: for 14
    programs it keeps exponents such as ``10^9`` that the other file has lost
    (``109``).
    """
    _validate(record)
    paths, conditions = record["sampled_paths"], record["sampled_condition_paths"]
    if len(paths) != len(conditions):
        raise ValueError(
            f"TestEval record {record['task_num']!r} has {len(paths)} paths but "
            f"{len(conditions)} condition paths"
        )
    description = descriptions[record["task_num"]]
    samples = []
    for index, (path, condition_path) in enumerate(zip(paths, conditions, strict=True)):
        samples.append(
            Sample(
                id=f"{record['task_num']}_path_{index}",
                input=path_prompt(
                    record["python_solution"], description, record["func_name"], condition_path
                ),
                metadata=_program_metadata(record)
                | {
                    "description": description,
                    "task_title": record["task_title"],
                    "python_solution_instrumented": record["python_solution_instrumented"],
                    "path_index": index,
                    "target_path": path,
                    "condition_path": condition_path,
                },
            )
        )
    return samples


def _instrumented_descriptions() -> dict[int, str]:
    dataset = load_json_dataset(
        INSTRUMENTED_URL,
        EVAL_NAME,
        sample_fields=lambda record: Sample(id=record["task_num"], input=record["description"]),
    )
    descriptions: dict[int, str] = {}
    for sample in dataset:
        if not isinstance(sample.id, int) or not isinstance(sample.input, str):
            raise ValueError(f"unexpected record in {INSTRUMENTED_URL}: {sample.id!r}")
        descriptions[sample.id] = sample.input
    return descriptions


def _load(sample_fields: RecordToSample, shuffle: bool | int) -> Dataset:
    should_shuffle, seed = shuffle_and_seed(shuffle)
    return load_json_dataset(
        DATASET_URL,
        EVAL_NAME,
        sample_fields=sample_fields,
        shuffle=should_shuffle,
        seed=seed,
    )


def load_testeval_dataset(shuffle: bool | int = DEFAULT_SHUFFLE_SEED) -> Dataset:
    """Load the pinned TestEval programs as an Inspect dataset, one sample per program.

    Args:
        shuffle: A seed, ``True`` for an unseeded shuffle, or ``False`` to keep
            the upstream order (sorted by LeetCode problem number).
    """
    return _load(record_to_sample, shuffle)


def load_target_dataset(
    target: Literal["line", "branch", "path"], shuffle: bool | int = DEFAULT_SHUFFLE_SEED
) -> Dataset:
    """Load one sample per target line (1,340), branch (983) or execution path (854).

    Args:
        target: ``"line"``, ``"branch"`` or ``"path"``.
        shuffle: A seed, ``True`` for an unseeded shuffle, or ``False`` to keep
            the upstream order (programs by LeetCode number, targets in file order).
    """
    if target == "line":
        return _load(record_to_line_samples, shuffle)
    if target == "branch":
        return _load(record_to_branch_samples, shuffle)
    if target == "path":
        descriptions = _instrumented_descriptions()
        return _load(partial(record_to_path_samples, descriptions=descriptions), shuffle)
    raise ValueError(f"target must be 'line', 'branch' or 'path', got {target!r}")
