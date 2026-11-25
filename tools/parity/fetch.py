"""Download the reference scorer and real model outputs used by the parity check.

Reference code: LLM4SoftwareTesting/TestEval at the commit the eval pins.
Model outputs: HyunjoonCho/TestEval (MIT), a fork whose format.py and
eval_overall.py scoring logic match upstream; its *_format.jsonl files are the
reference formatter's output for real replies from each model.
"""

import urllib.request
from pathlib import Path

UPSTREAM = "https://raw.githubusercontent.com/LLM4SoftwareTesting/TestEval/6f67dee71de6ccda1767b8c219a93f4ae0a7ce90"
FORK = "https://raw.githubusercontent.com/HyunjoonCho/TestEval/0f64e7aa9bda39258d7ce7126c0b9d4860e18099"
MODELS = ["gpt-4o", "Meta-Llama-3-8B-Instruct"]

WORK = Path(__file__).parent / "work"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    print(f"{path.relative_to(WORK)}  <-  {url}")


def main() -> None:
    for name in ("eval_overall.py", "data_utils.py", ".coveragerc"):
        download(f"{UPSTREAM}/{name}", WORK / "official" / name)
    for model in MODELS:
        name = f"totalcov_{model}_format.jsonl"
        download(f"{FORK}/predictions/{name}", WORK / "predictions" / name)


if __name__ == "__main__":
    main()
