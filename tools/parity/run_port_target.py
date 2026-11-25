"""Score a slice of targeted tests with this eval's coverage runner.

Runs inside the parity container and reports, per program, what
run_official_target.py reports for the reference scorer, under both this
eval's definition and the reference definition (for lines they are the same).

Usage: python run_port_target.py <runner_dir> <predictions.jsonl> <start> <end> <line|branch>
"""

import json
import sys
import tempfile
from pathlib import Path

EXECUTED = ("ok", "assertion_error")


def main() -> None:
    runner_dir, predictions, start, end, target = sys.argv[1:6]
    sys.path.insert(0, runner_dir)
    import coverage_runner

    coveragerc = (Path(runner_dir) / ".coveragerc").read_text(encoding="utf-8")
    with open(predictions, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh][int(start) : int(end)]

    root = Path(tempfile.mkdtemp(prefix="port_"))
    results = []
    for i, row in enumerate(rows):
        if target == "line":
            # eval_linecov.py: {"<lineno>": test}; the target is that line.
            targets = [(int(lineno), int(lineno), test) for lineno, test in row["tests"].items()]
        else:
            # eval_branchcov.py: covered when the line after the branch's first line runs.
            targets = [
                (f"{t['start']}-{t['end']}", t["start"] + 1, t["test"]) for t in row["tests"]
            ]
        out = {
            "task_num": row["task_num"],
            "total": 0,
            "syntax": 0,
            "exec": 0,
            "exec_reference": 0,
            "covered": [],
            "covered_reference": [],
        }
        for j, (name, check_line, test) in enumerate(targets):
            job = {
                "code": row["code"],
                "func_name": row["func_name"],
                "test": test,
                "check_line": check_line,
                "measure_errors": target == "branch",
                "coveragerc": coveragerc,
            }
            result = coverage_runner.evaluate_target(job, root / f"p{i}_{j}")
            executed = result["called"] and result["outcome"] in EXECUTED
            reference_outcomes = (*EXECUTED, "error") if target == "branch" else EXECUTED
            executed_reference = result["called"] and result["outcome"] in reference_outcomes
            out["total"] += 1
            out["syntax"] += result["syntax_ok"]
            out["exec"] += executed
            out["exec_reference"] += executed_reference
            if result["line_hit"] is True:
                if executed:
                    out["covered"].append(name)
                if executed_reference:
                    out["covered_reference"].append(name)
        out["covered"].sort()
        out["covered_reference"].sort()
        results.append(out)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
