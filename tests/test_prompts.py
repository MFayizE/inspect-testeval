import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from inspect_testeval.dataset import DATASET_COMMIT, load_target_dataset
from inspect_testeval.prompts import (
    BRANCH_TEMPLATE,
    LINE_TEMPLATE,
    OVERALL_FOLLOW_UP,
    OVERALL_TEMPLATE,
    PATH_TEMPLATE,
    SYSTEM_TEMPLATE,
    add_lineno,
    branch_prompt,
    line_prompt,
    overall_prompt,
    path_prompt,
    system_message,
)

UPSTREAM = f"https://raw.githubusercontent.com/LLM4SoftwareTesting/TestEval/{DATASET_COMMIT}"


def _upstream(path: str) -> str:
    with urllib.request.urlopen(f"{UPSTREAM}/{path}", timeout=60) as response:
        return response.read().decode("utf-8")


@pytest.mark.dataset_download
@pytest.mark.parametrize(
    ("path", "constant"),
    [("prompt/system.txt", SYSTEM_TEMPLATE), ("prompt/template_base.txt", OVERALL_TEMPLATE)],
)
def test_templates_match_upstream_files(path: str, constant: str) -> None:
    assert constant == _upstream(path)


@pytest.mark.dataset_download
@pytest.mark.parametrize(
    "path", ["generate_cov_openai.py", "generate_cov_hf.py", "generate_cov_gemini.py"]
)
def test_follow_up_matches_upstream_scripts(path: str) -> None:
    match = re.search(r'template_append="([^"]*)"', _upstream(path))
    assert match is not None
    assert match.group(1) == OVERALL_FOLLOW_UP


def test_system_message_fills_language() -> None:
    message = system_message()
    assert "{lang}" not in message
    assert message.startswith("You are a professional who writes python test methods.")
    assert message.endswith("\n")


def test_overall_prompt_fills_every_field() -> None:
    prompt = overall_prompt(
        program="class Solution:\n    def f(self): ...", description="Do f.", func_name="f"
    )
    assert "{" not in prompt.replace("{}", "")
    assert "the function 'f'" in prompt
    assert "class Solution:\n    def f(self): ...\n----" in prompt
    assert prompt.endswith("def test_f():\n    solution=Solution()\n")


def test_prompts_module_has_no_crlf() -> None:
    source = Path(__file__).parent.parent / "src" / "inspect_testeval" / "prompts.py"
    assert b"\r" not in source.read_bytes()


@pytest.mark.dataset_download
@pytest.mark.parametrize(
    ("path", "constant"),
    [("prompt/template_line.txt", LINE_TEMPLATE), ("prompt/template_branch.txt", BRANCH_TEMPLATE)],
)
def test_target_templates_match_upstream_files(path: str, constant: str) -> None:
    assert constant == _upstream(path)


def test_add_lineno_numbers_every_line_and_ends_each_with_newline() -> None:
    # data_utils.add_lineno: f"{i+1}. {line}\n" for every line of code.split("\n").
    assert add_lineno("a\n  b\n") == "1. a\n2.   b\n3. \n"
    assert add_lineno("x") == "1. x\n"


PROGRAM = (
    "class Solution:\n    def f(self, x):\n        if x:\n            return 1\n        return 0\n"
)


def test_line_prompt_shows_numbered_program_and_target() -> None:
    prompt = line_prompt(PROGRAM, "Do f.", "f", 4)
    assert "----\n1. class Solution:\n2.     def f(self, x):\n" in prompt
    # add_lineno's trailing newline leaves a blank line before the closing rule.
    assert "6. \n\n----" in prompt
    assert "Your test case must cover line 4:             return 1.\n" in prompt
    assert prompt.endswith("def test_f():\n    solution=Solution()\n")


def test_branch_prompt_quotes_the_numbered_branch_lines() -> None:
    prompt = branch_prompt(PROGRAM, "Do f.", "f", 3, 4)
    assert (
        "----\nYour test case must cover the branch \n'''\n"
        "3.         if x:\n4.             return 1\n'''\n"
        "Your test method should begin with:\n"
    ) in prompt


def _upstream_target_prompts(rows: list[dict[str, Any]]) -> list[str]:
    """Rebuild every targeted prompt the way generate_targetcov_openai.py does."""
    line_template = _upstream("prompt/template_line.txt")
    branch_template = _upstream("prompt/template_branch.txt")
    prompts = []
    for data in rows:
        code, desc, func_name = data["python_solution"], data["description"], data["func_name"]
        code_withlineno = "".join(f"{i + 1}. {line}\n" for i, line in enumerate(code.split("\n")))
        for lineno in data["target_lines"]:
            target_line_withlineno = f"{lineno}: {code.split(chr(10))[lineno - 1]}"
            prompts.append(
                line_template.format(
                    lang="python",
                    program=code_withlineno,
                    description=desc,
                    func_name=func_name,
                    lineno=target_line_withlineno,
                )
            )
        for branch in data["blocks"]:
            split_lines = code_withlineno.split("\n")
            target_lines = split_lines[branch["start"] - 1 : branch["end"]]
            branch_input = "\n'''\n" + "\n".join(target_lines) + "\n'''"
            prompts.append(
                branch_template.format(
                    lang="python",
                    program=code_withlineno,
                    description=desc,
                    func_name=func_name,
                    branch=branch_input,
                )
            )
    return prompts


@pytest.mark.dataset_download
def test_every_target_prompt_matches_the_reference_generator() -> None:
    rows = [json.loads(line) for line in _upstream("data/leetcode-py.jsonl").splitlines() if line]
    lines = list(load_target_dataset("line", shuffle=False))
    branches = list(load_target_dataset("branch", shuffle=False))
    ours: list[object] = []
    for data in rows:
        prefix = f"{data['task_num']}_"
        ours += [s.input for s in lines if str(s.id).startswith(prefix)]
        ours += [s.input for s in branches if str(s.id).startswith(prefix)]
    expected = _upstream_target_prompts(rows)
    assert len(expected) == 1340 + 983
    assert ours == expected


@pytest.mark.dataset_download
def test_path_template_matches_upstream_file() -> None:
    assert _upstream("prompt/template_path.txt") == PATH_TEMPLATE


def test_path_prompt_quotes_and_chains_conditions() -> None:
    prompt = path_prompt(PROGRAM, "Do f.", "f", ["Line 3: (x)", "Line 5: "])
    assert "Target execution path: 'Line 3: (x)' -> 'Line 5: '\n----\n" in prompt
    assert "must cover the following execution path in function f." in prompt
    assert prompt.endswith("def test_f():\n    solution=Solution()\n")


@pytest.mark.dataset_download
def test_every_path_prompt_matches_the_reference_generator() -> None:
    # generate_pathcov_openai.py reads programs and descriptions from
    # leetcode-py-instrumented.jsonl and paths from tgt_paths.jsonl.
    template = _upstream("prompt/template_path.txt")
    programs = [
        json.loads(line)
        for line in _upstream("data/leetcode-py-instrumented.jsonl").splitlines()
        if line
    ]
    paths = [json.loads(line) for line in _upstream("data/tgt_paths.jsonl").splitlines() if line]
    expected = []
    for data, path_data in zip(programs, paths, strict=True):
        code_withlineno = "".join(
            f"{i + 1}. {line}\n" for i, line in enumerate(data["python_solution"].split("\n"))
        )
        for condition_path in path_data["sampled_condition_paths"]:
            path = " -> ".join(f"'{condition}'" for condition in condition_path)
            expected.append(
                template.format(
                    func_name=data["func_name"],
                    description=data["description"],
                    program=code_withlineno,
                    path=path,
                )
            )
    ours = [s.input for s in load_target_dataset("path", shuffle=False)]
    assert len(expected) == 854
    assert ours == expected
