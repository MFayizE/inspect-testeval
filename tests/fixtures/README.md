# Test fixtures

`format_cases.jsonl` holds real model replies for TestEval's overall-coverage
task and the test each one becomes after the reference `format.py`. They are a
sample of `predictions/totalcov_gpt-4o.jsonl`,
`predictions/totalcov_Meta-Llama-3-8B-Instruct.jsonl` and their `_format`
counterparts from https://github.com/HyunjoonCho/TestEval at commit
0f64e7aa9bda39258d7ce7126c0b9d4860e18099 (MIT licence), a fork of
LLM4SoftwareTesting/TestEval whose `format.py` logic matches upstream.
