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

# generate_cov_{openai,hf,gemini}.py: the user turn added before each further test.
OVERALL_FOLLOW_UP = (
    "Generate another test method for the function under test. Your answer must be different "
    "from previously-generated test cases, and should cover different statements and branches."
)


def system_message(lang: str = "python") -> str:
    return SYSTEM_TEMPLATE.format(lang=lang)


def overall_prompt(program: str, description: str, func_name: str) -> str:
    return OVERALL_TEMPLATE.format(program=program, description=description, func_name=func_name)
