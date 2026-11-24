import json
import urllib.request
from collections import Counter
from typing import Any

import pytest

from inspect_testeval.dataset import (
    DATASET_COMMIT,
    DEFAULT_SHUFFLE_SEED,
    METADATA_FIELDS,
    load_testeval_dataset,
    record_to_sample,
)


def make_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_num": 1,
        "task_title": "Two Sum",
        "difficulty": 1,
        "func_name": "twoSum",
        "description": "Return indices of the two numbers that add up to target.",
        "python_solution": "class Solution:\n    def twoSum(self, nums, target):\n        return []\n",
        "python_solution_instrumented": "class Solution: ...\n",
        "target_lines": [3],
        "blocks": [{"type": "if", "start": 2, "end": 3, "difficulty": 0, "eg_cov_prob": 1.0}],
        "sampled_paths": [["BRANCH #1: Covered if branch at line 2-3\n"]],
        "sampled_condition_paths": [["Line 2: (x)"]],
    }
    record.update(overrides)
    return record


def test_record_to_sample_keeps_every_field() -> None:
    record = make_record()
    sample = record_to_sample(record)
    assert sample.id == 1
    assert sample.input == record["python_solution"]
    assert sample.metadata is not None
    for field in METADATA_FIELDS:
        assert sample.metadata[field] == record[field]
    assert sample.metadata["difficulty_label"] == "easy"


@pytest.mark.parametrize(("difficulty", "label"), [(1, "easy"), (2, "medium"), (3, "hard")])
def test_record_to_sample_maps_program_difficulty(difficulty: int, label: str) -> None:
    sample = record_to_sample(make_record(difficulty=difficulty))
    assert sample.metadata is not None
    assert sample.metadata["difficulty_label"] == label


def test_record_to_sample_rejects_unknown_difficulty() -> None:
    with pytest.raises(ValueError, match="unknown difficulty 0"):
        record_to_sample(make_record(difficulty=0))


def test_record_to_sample_rejects_missing_fields() -> None:
    record = make_record()
    del record["sampled_paths"]
    with pytest.raises(ValueError, match="sampled_paths"):
        record_to_sample(record)


def _read_upstream_jsonl(name: str) -> list[dict[str, Any]]:
    url = (
        "https://raw.githubusercontent.com/LLM4SoftwareTesting/TestEval/"
        f"{DATASET_COMMIT}/data/{name}"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        return [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line]


@pytest.mark.dataset_download
def test_dataset_matches_published_counts() -> None:
    samples = list(load_testeval_dataset())
    metadata = [s.metadata for s in samples if s.metadata is not None]
    assert len(samples) == len(metadata) == 210
    assert len({s.id for s in samples}) == 210
    # Totals from the TestEval README and paper.
    assert sum(len(m["target_lines"]) for m in metadata) == 1340
    assert sum(len(m["blocks"]) for m in metadata) == 983
    assert sum(len(m["sampled_paths"]) for m in metadata) == 854
    assert Counter(m["difficulty_label"] for m in metadata) == {
        "easy": 9,
        "medium": 100,
        "hard": 101,
    }
    for m in metadata:
        assert len(m["sampled_paths"]) == len(m["sampled_condition_paths"])
        # Instrumented programs log executed branches to this relative path.
        assert f"test_logs/{m['task_title']}.log" in m["python_solution_instrumented"]


def _sample_ids(shuffle: bool | int) -> list[int]:
    ids = [s.id for s in load_testeval_dataset(shuffle=shuffle)]
    assert all(isinstance(i, int) for i in ids)
    return [i for i in ids if isinstance(i, int)]


@pytest.mark.dataset_download
def test_default_shuffle_is_seeded_and_false_keeps_upstream_order() -> None:
    first = _sample_ids(shuffle=DEFAULT_SHUFFLE_SEED)
    second = _sample_ids(shuffle=DEFAULT_SHUFFLE_SEED)
    upstream = _sample_ids(shuffle=False)
    other_seed = _sample_ids(shuffle=7)
    assert first == second  # same default seed, same order
    assert [s.id for s in load_testeval_dataset()] == first  # no argument uses the default seed
    assert upstream == sorted(upstream)  # upstream file is sorted by problem number
    assert first != upstream
    assert other_seed not in (first, upstream)
    assert sorted(first) == sorted(upstream) == sorted(other_seed)


@pytest.mark.dataset_download
def test_combined_file_matches_files_used_by_original_scripts() -> None:
    combined = [s.metadata for s in load_testeval_dataset(shuffle=False)]
    plain = _read_upstream_jsonl("leetcode-py.jsonl")
    instrumented = _read_upstream_jsonl("leetcode-py-instrumented.jsonl")
    paths = _read_upstream_jsonl("tgt_paths.jsonl")
    assert len(combined) == len(plain) == len(instrumented) == len(paths) == 210
    for row, p, i, t in zip(combined, plain, instrumented, paths, strict=True):
        assert row is not None
        for field, value in p.items():
            assert row[field] == value, (row["task_num"], field)
        assert row["python_solution_instrumented"] == i["python_solution_instrumented"]
        assert row["sampled_paths"] == t["sampled_paths"]
        assert row["sampled_condition_paths"] == t["sampled_condition_paths"]
