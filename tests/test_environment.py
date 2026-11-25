import json
import re
from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox
from pytest_inspect_evals.sandbox import assert_sandbox_test_passed, create_sandbox_tool_task

from inspect_testeval.environment import SANDBOX, SANDBOX_COMPOSE_FILE, SANDBOX_DIR

LOCKED_VERSIONS = {
    "pytest": "9.2.0",
    "pytest-cov": "7.1.0",
    "coverage": "7.17.3",
    "sortedcontainers": "2.4.0",
}

PROGRAM = """\
class Solution:
    def sign(self, x: int) -> str:
        if x > 0:
            return "positive"
        return "non-positive"
"""

TEST = """\
from under_test import Solution


def test_sign():
    solution = Solution()
    assert solution.sign(5) == "positive"
"""


def test_sandbox_paths_are_absolute_and_present() -> None:
    assert SANDBOX_COMPOSE_FILE.is_absolute()
    assert ("docker", str(SANDBOX_COMPOSE_FILE)) == SANDBOX
    for name in ("compose.yaml", "Dockerfile", "requirements.in", "requirements.txt"):
        assert (SANDBOX_DIR / name).is_file(), name


def _locked(requirements: str) -> dict[str, str]:
    return dict(
        re.findall(r"^([a-z0-9][a-z0-9._-]*)==([^\s\\]+)", requirements, flags=re.MULTILINE)
    )


def test_requirements_lock_matches_top_level_pins() -> None:
    top_level = _locked((SANDBOX_DIR / "requirements.in").read_text(encoding="utf-8"))
    lock_text = (SANDBOX_DIR / "requirements.txt").read_text(encoding="utf-8")
    locked = _locked(lock_text)
    assert top_level == LOCKED_VERSIONS
    for name, version in top_level.items():
        assert locked[name] == version, name
    # Every locked package carries at least one hash (pip --require-hashes needs them).
    for block in re.split(r"\n(?=[a-z0-9])", lock_text.split("\n", 1)[1]):
        if "==" in block:
            assert "--hash=sha256:" in block, block.splitlines()[0]


def test_compose_runs_without_network_and_as_built_image() -> None:
    compose = SANDBOX_COMPOSE_FILE.read_text(encoding="utf-8")
    assert "network_mode: none" in compose
    assert "build:" in compose
    assert re.search(r"cap_drop:\s*\n\s*- ALL", compose)
    assert "no-new-privileges:true" in compose
    assert re.search(r"pids_limit: \d+", compose)
    dockerfile = (SANDBOX_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM \S+@sha256:[0-9a-f]{64}$", dockerfile, flags=re.MULTILINE)
    assert "--require-hashes" in dockerfile
    assert re.search(r"^USER agent$", dockerfile, flags=re.MULTILINE)


@solver
def _probe_sandbox() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        env = sandbox()

        user = await env.exec(["id", "-u"])
        assert user.success, user.stderr
        assert user.stdout.strip() != "0", "sandbox must not run as root"

        packages = sorted(LOCKED_VERSIONS)
        version_script = (
            "import importlib.metadata as m, json; "
            f"print(json.dumps({{p: m.version(p) for p in {packages!r}}}))"
        )
        versions = await env.exec(["python", "-c", version_script])
        assert versions.success, versions.stderr
        assert json.loads(versions.stdout) == LOCKED_VERSIONS

        network = await env.exec(
            [
                "python",
                "-c",
                "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)",
            ]
        )
        assert not network.success, "sandbox must have no network access"

        status = await env.exec(["cat", "/proc/self/status"])
        assert status.success, status.stderr
        capabilities = dict(
            line.split(":\t", 1) for line in status.stdout.splitlines() if line.startswith("Cap")
        )
        assert capabilities["CapEff"].strip() == "0000000000000000", capabilities
        assert capabilities["CapBnd"].strip() == "0000000000000000", capabilities
        assert "NoNewPrivs:\t1" in status.stdout

        await env.write_file("under_test.py", PROGRAM)
        await env.write_file("test_sign.py", TEST)
        run = await env.exec(
            [
                "python",
                "-m",
                "pytest",
                "--cov=under_test",
                "--cov-branch",
                "--cov-report=json:coverage.json",
                "test_sign.py",
            ],
            timeout=60,
        )
        assert run.success, run.stdout + run.stderr
        report = json.loads(await env.read_file("coverage.json"))
        file_report = report["files"]["under_test.py"]
        # Line 5 (the non-positive return) is never reached by the test.
        assert file_report["missing_lines"] == [5]
        assert report["totals"]["num_branches"] == 2
        assert report["totals"]["covered_branches"] == 1

        state.metadata["test_passed"] = True
        return state

    return solve


@pytest.mark.docker
@pytest.mark.slow(60)
def test_sandbox_runs_pytest_coverage() -> None:
    task = create_sandbox_tool_task(_probe_sandbox(), SANDBOX_COMPOSE_FILE)
    [log] = eval(task, model="mockllm/model")
    assert_sandbox_test_passed(log)


def test_sandbox_dir_is_inside_package() -> None:
    package_dir = Path(__file__).parent.parent / "src" / "inspect_testeval"
    assert SANDBOX_DIR.resolve() == (package_dir / "sandbox").resolve()
