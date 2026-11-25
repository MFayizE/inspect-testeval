# Comparison with the paper

`compare_paper.py` reads Inspect logs of the TestEval tasks and prints, for
each, the paper's value, this eval's value and the difference in percentage
points.

```bash
uv run inspect eval inspect_testeval/testeval_line --model openai/gpt-4o-mini
uv run python tools/validation/compare_paper.py logs/ --out comparison.md
```

`paper_tables.json` holds the rows for GPT-3.5-turbo, GPT-4o, GPT-4o-mini and
Llama3-8b from Tables 1 to 4 of the paper (arXiv 2406.04531, HTML version).
They were transcribed from the page text and checked against it value by
value. Assertion correctness and the no-target baselines are left out, since
this eval does not compute them.

The paper's numbers come from the reference scripts, so cov@k, branch and path
results are compared with this eval's `_reference` keys, which reproduce those
scripts; this eval's own definitions are printed under each table. A run will
not match the paper exactly: model versions behind an API change over time,
and even at temperature 0 replies are not fully deterministic.
