"""Score a slice of path tests with the unmodified eval_pathcov.py, one test per call.

Runs inside the parity container. eval_pathcov.py prints only totals and reads
the target path by position (``ref_dataset[i]['sampled_paths'][j]``), so each
test is scored by its own ``eval_correctness`` call on a one-program,
one-test input, with ``data/leetcode-py-instrumented.jsonl`` and
``data/tgt_paths.jsonl`` holding just that program and that path. The four
rates it prints are then the test's own results. Nothing in the reference
code is changed.

Within a program the imported program module stays cached between calls, as
it does in a full reference run (every test of a program imports the same
``tmp_<i>_<difficulty>.under_test``). Between programs this script removes
the ``tmp_*`` folders and their modules, since every call names its folder
``tmp_0_<difficulty>``, and resets the working directory.

Usage: python run_official_path.py <official_dir> <predictions.jsonl> <start> <end>
"""

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

RATES = {
    "syntax": r"Syntax Correctness: ([0-9.eE+-]+)",
    "exec": r"Executable Correctness: ([0-9.eE+-]+)",
    "exact": r"path exact match accuracy: ([0-9.eE+-]+)",
    "similarity": r"path similarity score: ([0-9.eE+-]+)",
}


def by_task(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    return {record["task_num"]: record for record in records}


def main() -> None:
    official_dir, predictions, start, end = sys.argv[1:5]
    official = Path(official_dir)
    instrumented = by_task(official / "data" / "leetcode-py-instrumented.jsonl")
    paths = by_task(official / "data" / "tgt_paths.jsonl")
    run_dir = Path(tempfile.mkdtemp(prefix="official_"))
    for name in ("eval_pathcov.py", "data_utils.py"):
        shutil.copy(official / name, run_dir / name)
    data_dir = run_dir / "data"
    data_dir.mkdir()
    os.chdir(run_dir)
    sys.path.insert(0, str(run_dir))

    import eval_pathcov  # the reference scorer, unmodified

    with open(predictions, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh][int(start) : int(end)]

    results = []
    for row in rows:
        task_num = row["task_num"]
        os.chdir(run_dir)
        for path in run_dir.glob("tmp*"):
            shutil.rmtree(path)
        for name in [m for m in sys.modules if m.startswith("tmp_")]:
            del sys.modules[name]
        (data_dir / "leetcode-py-instrumented.jsonl").write_text(
            json.dumps(instrumented[task_num]) + "\n", encoding="utf-8"
        )
        target = paths[task_num]
        tests = []
        for j, test in enumerate(row["tests"]):
            os.chdir(run_dir)
            one_path = {
                **target,
                "sampled_paths": [target["sampled_paths"][j]],
                "sampled_condition_paths": [target["sampled_condition_paths"][j]],
            }
            (data_dir / "tgt_paths.jsonl").write_text(json.dumps(one_path) + "\n", encoding="utf-8")
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    eval_pathcov.eval_correctness([{**row, "tests": [test]}])
            except Exception:
                tests.append({"error": traceback.format_exc(limit=3)})
                continue
            text = stdout.getvalue()
            found = {key: re.findall(pattern, text) for key, pattern in RATES.items()}
            missing = [key for key, values in found.items() if not values]
            if missing:
                tests.append({"error": f"missing {missing} in output:\n{text[-2000:]}"})
                continue
            tests.append({key: float(values[-1]) for key, values in found.items()})
        results.append({"task_num": task_num, "tests": tests})
    print(json.dumps(results))


if __name__ == "__main__":
    main()
