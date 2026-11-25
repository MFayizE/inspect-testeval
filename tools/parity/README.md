# Parity check against the reference scorers

`parity.py` scores the same model-generated tests with this eval's coverage
runner and with the unmodified reference `eval_overall.py`, in the same Linux
image, and compares every metric. `parity_target.py` does the same for the
targeted tasks against the unmodified `eval_linecov.py` and `eval_branchcov.py`,
program by program, and `parity_path.py` for the path task against
`eval_pathcov.py`, test by test.

## Inputs

`fetch.py` downloads into `work/` (ignored by git):

- `eval_overall.py`, `eval_linecov.py`, `eval_branchcov.py`, `eval_pathcov.py`,
  `data_utils.py`, `.coveragerc` and `data/leetcode-py.jsonl`,
  `data/leetcode-py-instrumented.jsonl` and `data/tgt_paths.jsonl` from
  [LLM4SoftwareTesting/TestEval](https://github.com/LLM4SoftwareTesting/TestEval)
  at commit `6f67dee71de6ccda1767b8c219a93f4ae0a7ce90`, the commit the eval pins.
- `totalcov_`, `linecov_`, `branchcov_` and `pathcov_<model>_format.jsonl` for gpt-4o and
  Meta-Llama-3-8B-Instruct from
  [HyunjoonCho/TestEval](https://github.com/HyunjoonCho/TestEval) (MIT) at commit
  `0f64e7aa9bda39258d7ce7126c0b9d4860e18099`: real model replies after the
  reference formatter (210 programs x 10 tests; 1,340 target lines; 983 target
  branches; 854 target paths). That fork changes only an output filename in `format.py`, adds
  per-task logging to the evaluators and records token costs in the generator;
  the prompts and scoring logic are upstream's.

## Modes

- `identity`: neither side shuffles tests before forming cov@k groups, so
  every metric must match. The reference cov@k is compared with this eval's
  `*_reference` keys (the reference code's definition); the paper-definition
  keys are printed alongside.
- `seeded`: both shuffle with their own seeds, so cov@k for k > 1 differs by
  design. Informational only.

## Excluded program

`--exclude` defaults to task 952 (Largest Component Size by Common Factor).
One Meta-Llama-3-8B test calls it with `[1000000000, 1000000000]`, outside
LeetCode's input limits; the solution allocates an array that large. The
reference scorer runs every test inside its own single process, so the whole
run is killed (exit 137) unless the machine has enough memory. This eval runs
each test in a separate process: under the sandbox's 1 GB limit that test is
recorded as a timeout and the other nine are scored normally.

## Targeted tasks

The reference `eval_linecov.py` and `eval_branchcov.py` print only totals, so
`run_official_target.py` calls `eval_correctness` once per program and reads
the counts line and the `covered line N` / `covered branch S-E` lines it
prints. For every program the number of syntax-correct and executable tests
and the set of covered targets must equal the port's. Branches are compared
with the port's `_reference` definition, which, like `eval_branchcov.py`
(`res != False`), counts a test that raised an exception as executable; the
paper definition is printed alongside. Between calls the script resets only
what the reference code leaves behind (see its docstring); the scoring code is
not changed.

## Path task

`eval_pathcov.py` prints only totals and reads the target path by position,
so `run_official_path.py` calls `eval_correctness` once per test, with the
data files holding just that program and that path; the four rates it prints
are then that test's own results. Within a program the imported program
module stays cached between calls, as in a full reference run. Every test's
syntax and execution result, exact match and path similarity must equal the
port's `_reference` values. The port's own definition differs only on the 8
empty target paths, where `eval_pathcov.py` divides by zero inside a bare
`except` and scores 0.

## Run

From the repository root, with Docker running:

```bash
uv run python tools/parity/parity.py --modes identity
uv run python tools/parity/parity_target.py
uv run python tools/parity/parity_path.py
```

The reports are written to `work/parity_report.json`,
`work/parity_target_report.json` and `work/parity_path_report.json`.
