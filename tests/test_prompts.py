import re
import urllib.request
from pathlib import Path

import pytest

from inspect_testeval.dataset import DATASET_COMMIT
from inspect_testeval.prompts import (
    OVERALL_FOLLOW_UP,
    OVERALL_TEMPLATE,
    SYSTEM_TEMPLATE,
    overall_prompt,
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
