"""Prompts from the TestEval reference implementation, copied verbatim.

Source: https://github.com/LLM4SoftwareTesting/TestEval (MIT licence), commit
6f67dee71de6ccda1767b8c219a93f4ae0a7ce90. The original scripts read these with
``open(path).read()``, so the trailing newline is part of each template.
"""

# prompt/system.txt, formatted with lang="python" by every generation script.
SYSTEM_TEMPLATE = (
    "You are a professional who writes {lang} test methods. You always respond with a {lang} "
    "function without any natural language descriptions. Especially, Your answer should not "
    "include any quotation marks or descriptions outside the function definition.\n"
)

# prompt/template_base.txt: the overall-coverage prompt.
OVERALL_TEMPLATE = (
    "Please write a test method for the function '{func_name}' given the following program "
    "under test and function description. Your answer should only contain one test input.\n"
    "\n"
    "Program under test:\n"
    "----\n"
    "{program}\n"
    "----\n"
    "\n"
    "Function description for '{func_name}':\n"
    "----\n"
    "{description}\n"
    "----\n"
    "\n"
    "Your test method should begin with:\n"
    "def test_{func_name}():\n"
    "    solution=Solution()\n"
)

# prompt/template_path.txt: cover one target execution path.
PATH_TEMPLATE = (
    "Please write a test method for the function '{func_name}' given the following program "
    "under test and function description. Your answer should only contain one test input.\n"
    "\n"
    "Program under test:\n"
    "----\n"
    "{program}\n"
    "----\n"
    "\n"
    "Function description for '{func_name}':\n"
    "----\n"
    "{description}\n"
    "----\n"
    "\n"
    "Your test case must cover the following execution path in function {func_name}. The path "
    "is a sequence of branch conditions. When executing your test case, each branch condition "
    "in the target execution path must be satisfied sequentially.\n"
    "Target execution path: {path}\n"
    "----\n"
    "Your test method should begin with:\n"
    "def test_{func_name}():\n"
    "    solution=Solution()\n"
)

# generate_cov_{openai,hf,gemini}.py: the user turn added before each further test.
OVERALL_FOLLOW_UP = (
    "Generate another test method for the function under test. Your answer must be different "
    "from previously-generated test cases, and should cover different statements and branches."
)

# prompt/template_line.txt: cover one target line.
LINE_TEMPLATE = (
    "Please write a test method for the function '{func_name}' given the following program "
    "under test and function description. Your answer should only contain one test input.\n"
    "\n"
    "Program under test:\n"
    "----\n"
    "{program}\n"
    "----\n"
    "\n"
    "Function description for '{func_name}':\n"
    "----\n"
    "{description}\n"
    "----\n"
    "\n"
    "Your test case must cover line {lineno}.\n"
    "Your test method should begin with:\n"
    "def test_{func_name}():\n"
    "    solution=Solution()\n"
)

# prompt/template_branch.txt: cover one target branch. Unlike the line
# template, it has no blank line before the target.
BRANCH_TEMPLATE = (
    "Please write a test method for the function '{func_name}' given the following program "
    "under test and function description. Your answer should only contain one test input.\n"
    "\n"
    "Program under test:\n"
    "----\n"
    "{program}\n"
    "----\n"
    "\n"
    "Function description for '{func_name}':\n"
    "----\n"
    "{description}\n"
    "----\n"
    "Your test case must cover the branch {branch}\n"
    "Your test method should begin with:\n"
    "def test_{func_name}():\n"
    "    solution=Solution()\n"
)


def system_message(lang: str = "python") -> str:
    return SYSTEM_TEMPLATE.format(lang=lang)


def overall_prompt(program: str, description: str, func_name: str) -> str:
    return OVERALL_TEMPLATE.format(program=program, description=description, func_name=func_name)


def add_lineno(code: str) -> str:
    """Number every line as ``"<n>. <line>"``, as data_utils.add_lineno does.

    Every line, including the last, ends with a newline.
    """
    return "".join(f"{i + 1}. {line}\n" for i, line in enumerate(code.split("\n")))


def line_prompt(program: str, description: str, func_name: str, lineno: int) -> str:
    """The targeted-line prompt, built as generate_targetcov_openai.py builds it.

    The program is shown with line numbers and the target as ``"<n>: <line>"``.
    """
    target_line = program.split("\n")[lineno - 1]
    target = f"{lineno}: {target_line}"
    return LINE_TEMPLATE.format(
        program=add_lineno(program), description=description, func_name=func_name, lineno=target
    )


def branch_prompt(program: str, description: str, func_name: str, start: int, end: int) -> str:
    """The targeted-branch prompt, built as generate_targetcov_openai.py builds it.

    The branch is lines ``start`` to ``end`` of the numbered program, between
    triple quotes on their own lines.
    """
    numbered = add_lineno(program)
    lines = numbered.split("\n")[start - 1 : end]
    branch = "\n'''\n" + "\n".join(lines) + "\n'''"
    return BRANCH_TEMPLATE.format(
        program=numbered, description=description, func_name=func_name, branch=branch
    )


def path_prompt(program: str, description: str, func_name: str, conditions: list[str]) -> str:
    """The targeted-path prompt, built as generate_pathcov_openai.py builds it.

    The path is its branch conditions, each in single quotes, joined by
    ``" -> "`` (prompt_utils.generate_path).
    """
    path = " -> ".join(f"'{condition}'" for condition in conditions)
    return PATH_TEMPLATE.format(
        program=add_lineno(program), description=description, func_name=func_name, path=path
    )
