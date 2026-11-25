"""Score a slice of formatted tests with this eval's coverage runner.

Runs inside the parity container and aggregates exactly as eval_overall.py
does: correctness over all tests, coverage and cov@k averaged over all
programs with 0 for a program that has no executable test.

Usage: python run_port.py <runner_dir> <predictions.jsonl> <start> <end> <seeded|identity>
"""

import json
import random
import sys
import tempfile
from pathlib import Path

KS = [1, 2, 5]


class NoShuffle(random.Random):
    """Keep test order, as identity mode does for the reference scorer."""

    def shuffle(self, x):
        return None


def main() -> None:
    runner_dir, predictions, start, end, mode = sys.argv[1:6]
    sys.path.insert(0, runner_dir)
    import coverage_runner

    coveragerc = (Path(runner_dir) / ".coveragerc").read_text(encoding="utf-8")
    with open(predictions, encoding="utf-8") as fh:
        data = [json.loads(line) for line in fh][int(start) : int(end)]

    totals = {"total": 0, "syntax": 0, "exec": 0}
    sums: dict[str, float] = {}

    def add(key: str, value: float | None) -> None:
        sums[key] = sums.get(key, 0.0) + (value or 0.0)

    root = Path(tempfile.mkdtemp(prefix="port_"))
    for i, row in enumerate(data):
        job = {
            "code": row["code"],
            "func_name": row["func_name"],
            "tests": row["tests"],
            "ks": KS,
            "seed": f"42:{row['task_num']}",
            "coveragerc": coveragerc,
        }
        rng = NoShuffle() if mode == "identity" else None
        result = coverage_runner.evaluate(job, root / f"p{i}", rng=rng)
        totals["total"] += result["total_cases"]
        totals["syntax"] += result["syntax_correct"]
        totals["exec"] += result["exec_correct"]
        add("line_coverage", result["line_coverage"])
        add("branch_coverage", result["branch_coverage"])
        cov = result["cov_at_k"] or {}
        for k in KS:
            for definition, suffix in (("paper", ""), ("reference", "_reference")):
                entry = cov.get(definition, {}).get(str(k), {})
                add(f"line_cov@{k}{suffix}", entry.get("line"))
                add(f"branch_cov@{k}{suffix}", entry.get("branch"))

    n = len(data)
    out = {
        "n": n,
        "total_cases": totals["total"],
        "syntax_correct": totals["syntax"] / totals["total"],
        "exec_correct": totals["exec"] / totals["total"],
    }
    out.update({key: value / n for key, value in sums.items()})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
