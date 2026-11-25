"""Score a slice of path tests with this eval's coverage runner.

Runs inside the parity container and reports, per test, what
run_official_path.py reports for the reference scorer, plus this eval's own
definition for empty target paths.

Usage: python run_port_path.py <runner_dir> <official_dir> <predictions.jsonl> <start> <end>
"""

import json
import sys
import tempfile
from pathlib import Path

EXECUTED = ("ok", "assertion_error")


def by_task(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    return {record["task_num"]: record for record in records}


def main() -> None:
    runner_dir, official_dir, predictions, start, end = sys.argv[1:6]
    sys.path.insert(0, runner_dir)
    import coverage_runner

    instrumented = by_task(Path(official_dir) / "data" / "leetcode-py-instrumented.jsonl")
    paths = by_task(Path(official_dir) / "data" / "tgt_paths.jsonl")
    with open(predictions, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh][int(start) : int(end)]

    root = Path(tempfile.mkdtemp(prefix="port_"))
    results = []
    for i, row in enumerate(rows):
        program = instrumented[row["task_num"]]
        tests = []
        for j, test in enumerate(row["tests"]):
            job = {
                "code": program["python_solution_instrumented"],
                "func_name": row["func_name"],
                "test": test,
                "log_file": f"test_logs/{program['task_title']}.log",
                "target_path": paths[row["task_num"]]["sampled_paths"][j],
            }
            result = coverage_runner.evaluate_path(job, root / f"p{i}_{j}")
            executed = result["called"] and result["outcome"] in EXECUTED
            similarity = result["similarity"] if executed else 0.0
            reference = result["similarity_reference"] if executed else 0.0
            tests.append(
                {
                    "syntax": float(result["syntax_ok"]),
                    "exec": float(executed),
                    "exact": float(executed and reference == 1.0),
                    "similarity": reference,
                    "exact_paper": float(executed and similarity == 1.0),
                    "similarity_paper": similarity,
                }
            )
        results.append({"task_num": row["task_num"], "tests": tests})
    print(json.dumps(results))


if __name__ == "__main__":
    main()
