"""Clean a model's raw reply into one test function, as the reference implementation does.

Ported from format.py in https://github.com/LLM4SoftwareTesting/TestEval (MIT
licence, commit 6f67dee71de6ccda1767b8c219a93f4ae0a7ce90). The logic is kept as
upstream wrote it, quirks included, so formatted tests match the paper's
pipeline. Unreachable code after the early return in ``remove_extra`` is left out.
"""

import ast
import textwrap


def change_function_name(code: str, new_name: str) -> str:
    """Rename the first function definition found by ``ast.walk`` to ``new_name``.

    If the code does not parse, it is returned unchanged. When the first function
    already has the name, the code is still round-tripped through ``ast.unparse``.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name != new_name:
                    node.name = new_name
                break
        return ast.unparse(tree)
    except Exception:  # upstream catches everything and keeps the original text
        return code


def reformat_case_byrules(testcase: str, func_name: str) -> str:
    """Dedent, drop an incomplete last line, and rename the test function.

    A last line that does not compile on its own (usually cut off by the token
    limit) is removed. ``func_name`` is the test function's name, ``test_<name>``.
    """
    if testcase.startswith(" "):
        testcase = textwrap.dedent(testcase)
    lines = testcase.split("\n")
    last_line = textwrap.dedent(lines[-1])
    try:
        compile(last_line, "<string>", "exec")
    except Exception:  # upstream uses a bare except; any compile failure drops the line
        lines = lines[:-1]
    testcase = "\n".join(lines)
    return change_function_name(testcase, func_name)


def remove_extra(testcase: str, func_name: str) -> str:
    """Keep the lines from the first ``def test`` through the first call of the function under test.

    Text before the test function and anything after the first
    ``solution.<func_name>`` call (more test inputs, explanations) is dropped.
    """
    lines = testcase.split("\n")
    func_startline = 0
    for i, line in enumerate(lines):
        if line.find("def test") >= 0:
            func_startline = i
            break
    test_endline = len(lines)
    for i, line in enumerate(lines):
        if line.find(f"solution.{func_name}") >= 0:
            test_endline = i + 1
            break
    return "\n".join(lines[func_startline:test_endline])


def format_test(raw: str, func_name: str) -> str:
    """Format one raw model reply the way ``format.py --mode overall`` does."""
    extracted = remove_extra(raw, func_name)
    return reformat_case_byrules(extracted, f"test_{func_name}")
