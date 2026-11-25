# inspect-testeval

An [Inspect AI](https://inspect.aisi.org.uk/) port of [TestEval](https://arxiv.org/abs/2406.04531) (NAACL 2025 Findings), a benchmark for unit test generation: models write pytest tests for 210 LeetCode Python programs, scored by overall line and branch coverage and by covering a given target line, branch or execution path. Tests run in a Docker sandbox, and every task reproduces the authors' reference scorer exactly on real model outputs.

**Documentation:** [src/inspect_testeval/README.md](src/inspect_testeval/README.md) covers the tasks, dataset, scoring, deviations from the reference implementation and the parity check.

## Quick start

Docker must be running: generated tests run in a sandbox container.

```bash
uv sync
uv run inspect eval inspect_testeval/testeval_line --model openai/gpt-4o-mini --limit 20
uv run inspect view
```

The four tasks are `testeval_overall`, `testeval_line`, `testeval_branch` and `testeval_path`. Put `INSPECT_EVAL_MODEL` and your API keys in a `.env` file (it is git-ignored) to avoid passing `--model` each time. The dev dependencies include the `anthropic` and `openai` SDKs; add the package for any other provider with `uv add --dev <package>`.

## Checks and tests

```bash
make check   # ruff, basedpyright, inspect-evals-lint, generated README sections, pre-commit
make test    # pytest
```

Tests that need Docker, download the dataset or run long are marked `docker`, `dataset_download` and `slow`; slow tests run with `uv run pytest --runslow`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the other gates.

The parity check against the reference scripts (needs Docker) is described in [tools/parity/README.md](tools/parity/README.md).

## Layout

```text
src/inspect_testeval/            the eval: tasks, dataset, solvers, scorers, eval.yaml, README.md
src/inspect_testeval/sandbox/    Docker image the generated tests run in
tests/                           tests (pytest)
tools/parity/                    comparison with the unmodified reference scorers
agent_artefacts/trajectory_analysis/inspect_scout/   Scout scanners for the transcript audit
```

## Skills

The `inspect-evals` skills (create-eval, investigate-dataset, ensure-test-coverage, read-eval-logs, check-trajectories-workflow, eval-validity-review, eval-report-workflow) come from [Generality-Labs/agent-skills](https://github.com/Generality-Labs/agent-skills).

- Claude Code: open this folder and trust it. The plugin loads from `.claude/settings.json`; skills appear as `/inspect-evals:<name>`.
- Codex: `codex plugin marketplace add Generality-Labs/agent-skills` then `codex plugin add inspect-evals@generality`, once per machine.
- Other agents: `npx skills add Generality-Labs/agent-skills --agent <agent>`.

## Licence and attribution

MIT. The dataset, prompts and scoring logic come from [LLM4SoftwareTesting/TestEval](https://github.com/LLM4SoftwareTesting/TestEval) (MIT). The test fixtures and parity inputs use model outputs published in [HyunjoonCho/TestEval](https://github.com/HyunjoonCho/TestEval) (MIT).

## Keeping up with the template

This repository was created from [inspect-evals-template](https://github.com/Generality-Labs/inspect-evals-template). A weekly workflow runs `copier update` and opens a pull request when the template changed. `README.md`, `src/inspect_testeval/` and `tests/test_inspect_testeval.py` are never overwritten by an update; the other test files are not part of the template.
