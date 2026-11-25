# Parity check against the reference scorer

`parity.py` scores the same model-generated tests with this eval's coverage
runner and with the unmodified reference `eval_overall.py`, in the same Linux
image, and compares every metric.

## Inputs

`fetch.py` downloads into `work/` (ignored by git):

- `eval_overall.py`, `data_utils.py` and `.coveragerc` from
  [LLM4SoftwareTesting/TestEval](https://github.com/LLM4SoftwareTesting/TestEval)
  at commit `6f67dee71de6ccda1767b8c219a93f4ae0a7ce90`, the commit the eval pins.
- `totalcov_<model>_format.jsonl` for gpt-4o and Meta-Llama-3-8B-Instruct from
  [HyunjoonCho/TestEval](https://github.com/HyunjoonCho/TestEval) (MIT) at commit
  `0f64e7aa9bda39258d7ce7126c0b9d4860e18099`: real model replies after the
  reference formatter, 210 programs x 10 tests each. That fork changes only an
  output filename in `format.py` and adds per-task logging to `eval_overall.py`.

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

## Run

From the repository root, with Docker running:

```bash
uv run python tools/parity/parity.py --modes identity
```

The report is written to `work/parity_report.json`.
