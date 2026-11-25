"""Score a slice of formatted tests with the unmodified reference eval_overall.py.

Runs inside the parity container. The reference code works in its own
directory (it creates and deletes tmp_* folders there), so each run copies it
to a private directory first.

Usage: python run_official.py <official_dir> <predictions.jsonl> <start> <end> <seeded|identity>
"""

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> None:
    official_dir, predictions, start, end, mode = sys.argv[1:6]
    run_dir = Path(tempfile.mkdtemp(prefix="official_"))
    for name in ("eval_overall.py", "data_utils.py", ".coveragerc"):
        shutil.copy(Path(official_dir) / name, run_dir / name)
    os.chdir(run_dir)
    sys.path.insert(0, str(run_dir))

    import eval_overall  # the reference scorer, unmodified

    if mode == "identity":
        # Keep test order so cov@k groups are identical to the port's.
        eval_overall.random.shuffle = lambda x: None

    with open(predictions, encoding="utf-8") as fh:
        data = [json.loads(line) for line in fh][int(start) : int(end)]

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        eval_overall.check_correctness(data, ks=[1, 2, 5])
    text = stdout.getvalue()

    def number(pattern: str) -> float:
        match = re.search(pattern, text)
        if match is None:
            raise RuntimeError(f"pattern {pattern!r} not found in reference output:\n{text}")
        return float(match.group(1))

    result = {
        "n": len(data),
        "syntax_correct": number(r"Syntax Correctness: ([0-9.eE+-]+)"),
        "exec_correct": number(r"Executable Correctness: ([0-9.eE+-]+)"),
        "line_coverage": number(r"Average Line Coverage: ([0-9.eE+-]+)"),
        "branch_coverage": number(r"Average Branch Coverage: ([0-9.eE+-]+)"),
    }
    for k in (1, 2, 5):
        result[f"line_cov@{k}"] = number(rf"line coverage@{k} ([0-9.eE+-]+)")
        result[f"branch_cov@{k}"] = number(rf"branch coverage@{k} ([0-9.eE+-]+)")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
