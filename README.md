# inspect-testeval

TestEval (NAACL 2025): benchmarking LLMs on test case generation with overall, targeted line, branch and path coverage

<!-- Contributors: Automatically Generated -->
Contributed by [@MFayizE](https://github.com/MFayizE)
<!-- /Contributors: Automatically Generated -->

<!-- Usage: Automatically Generated -->
## Usage

First, install dependencies:

```bash
uv sync
```

Then run evaluations:

```bash
uv run inspect eval inspect_testeval/inspect_testeval --model openai/gpt-5-nano
```

You can also import tasks as Python objects:

```python
from inspect_ai import eval
from inspect_testeval import inspect_testeval
eval(inspect_testeval)
```

If your directory name doesn't match your package name but you have registered your eval task, you can also run:

```bash
uv run inspect eval inspect_testeval --model openai/gpt-5-nano
```

After running evaluations, view logs with:

```bash
uv run inspect view
```

If you don't want to specify `--model` each time, create a `.env` file:

```bash
INSPECT_EVAL_MODEL=anthropic/claude-opus-4-1-20250805
ANTHROPIC_API_KEY=<anthropic-api-key>
```
<!-- /Usage: Automatically Generated -->

<!-- Options: Automatically Generated -->
## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval inspect_testeval/inspect_testeval --limit 10
uv run inspect eval inspect_testeval/inspect_testeval --max-connections 10
uv run inspect eval inspect_testeval/inspect_testeval --temperature 0.5
```

See `uv run inspect eval --help` for all available options.
<!-- /Options: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `inspect_testeval`

No task parameters.
<!-- /Parameters: Automatically Generated -->

## Dataset

Describe the dataset here: where it comes from, how many samples it has, and how a record becomes a `Sample`.

## Scoring

Describe the scorer here: what counts as correct, and which metrics the task reports.

## Deviations from the original implementation

| What changed | Why | Expected effect on scores |
| --- | --- | --- |
| Nothing yet. | | |

## Architecture

Add a diagram of the task here: dataset, solver, scorer, and any sandbox or judge model.

## Getting started

```bash
uv sync
uv run pre-commit install
uv run inspect eval inspect_testeval/inspect_testeval --model anthropic/claude-sonnet-5-5
uv run inspect view
```

Put `INSPECT_EVAL_MODEL` and your API keys in a `.env` file (it is git-ignored) to avoid passing `--model` each time.

The dev dependencies include the `anthropic` and `openai` SDKs. Inspect needs one package per model provider, so add the package for any other provider with `uv add --dev <package>`; Inspect's error message names the package when it is missing.

## Checks and tests

```bash
make check   # ruff, basedpyright, inspect-evals-lint, generated README sections, pre-commit
make test    # pytest
```

Slow tests run with `uv run pytest --runslow`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the other gates and markers.

## Layout

```text
src/inspect_testeval/        the eval: task module, eval.yaml
tests/test_inspect_testeval.py   its tests
agent_artefacts/trajectory_analysis/inspect_scout/   Scout scanners for the transcript audit
```

## Skills

The `inspect-evals` skills (create-eval, investigate-dataset, ensure-test-coverage, read-eval-logs, check-trajectories-workflow, eval-validity-review, eval-report-workflow) come from [Generality-Labs/agent-skills](https://github.com/Generality-Labs/agent-skills).

- Claude Code: open this folder and trust it. The plugin loads from `.claude/settings.json`; skills appear as `/inspect-evals:<name>`.
- Codex: `codex plugin marketplace add Generality-Labs/agent-skills` then `codex plugin add inspect-evals@generality`, once per machine.
- Other agents: `npx skills add Generality-Labs/agent-skills --agent <agent>`.

## Keeping up with the template

A weekly workflow runs `copier update` and opens a pull request when the template changed.
`README.md`, `src/inspect_testeval/` and `tests/inspect_testeval/` are never overwritten by an update.
