"""Compare this eval's path scoring with the reference eval_pathcov.py, test by test.

Both score the same formatted tests (see fetch.py) inside the same Linux
image: eval_pathcov.py unmodified, one test per call (run_official_path.py),
and this eval's coverage runner (run_port_path.py). For every test the syntax
and execution results, the exact-match result and the path similarity must
match the port's reference definition. The port's own definition, which
differs only on the 8 empty target paths, is printed alongside.

Usage (from the repository root, Docker running):
    uv run python tools/parity/parity_path.py [--models ...] [--limit N]
"""

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from parity import HERE, WORK, build_image, prepare_port_dir

IMAGE = "inspect-testeval-parity:dev"
KEYS = ("syntax", "exec", "exact", "similarity")


def run_slice(args: list[str]) -> list[dict[str, Any]]:
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "-v", f"{HERE.resolve()}:/parity:ro",
        "-v", f"{WORK.resolve()}:/work:ro",
        IMAGE, "python", *args,
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{args} failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def compare(official: list[dict[str, Any]], port: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches, errors = [], []
    sums = {f"{side}_{key}": 0.0 for side in ("official", "port") for key in KEYS}
    sums |= {"port_exact_paper": 0.0, "port_similarity_paper": 0.0}
    compared = 0
    for o, p in zip(official, port, strict=True):
        assert o["task_num"] == p["task_num"]
        for j, (ot, pt) in enumerate(zip(o["tests"], p["tests"], strict=True)):
            if "error" in ot:
                errors.append({"task_num": o["task_num"], "test": j, "error": ot["error"]})
                continue
            compared += 1
            if any(not math.isclose(ot[k], pt[k], abs_tol=1e-12) for k in KEYS):
                mismatches.append(
                    {"task_num": o["task_num"], "test": j, "official": ot, "port": pt}
                )
            for k in KEYS:
                sums[f"official_{k}"] += ot[k]
                sums[f"port_{k}"] += pt[k]
            sums["port_exact_paper"] += pt["exact_paper"]
            sums["port_similarity_paper"] += pt["similarity_paper"]
    return {
        "tests_compared": compared,
        "mismatches": mismatches,
        "official_errors": errors,
        "rates": {k: v / (compared or 1) for k, v in sums.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "Meta-Llama-3-8B-Instruct"])
    parser.add_argument("--limit", type=int, default=210)
    parser.add_argument("--slice", type=int, default=10, help="programs per container")
    parser.add_argument("--jobs", type=int, default=6, help="containers at once")
    args = parser.parse_args()

    subprocess.run(["python", str(HERE / "fetch.py")], check=True)
    build_image()
    prepare_port_dir()

    report: dict[str, Any] = {}
    for model in args.models:
        name = f"pathcov_{model}_format.jsonl"
        rows = (WORK / "predictions" / name).read_text(encoding="utf-8").splitlines()
        count = min(len([r for r in rows if r]), args.limit)
        predictions = f"/work/predictions/{name}"
        slices = [(s, min(s + args.slice, count)) for s in range(0, count, args.slice)]
        jobs = [
            ("official", ["/parity/run_official_path.py", "/work/official", predictions, str(s), str(e)])
            for s, e in slices
        ] + [
            ("port", ["/parity/run_port_path.py", "/work/port", "/work/official", predictions, str(s), str(e)])
            for s, e in slices
        ]  # fmt: skip
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [(side, pool.submit(run_slice, cmd)) for side, cmd in jobs]
            results = [(side, future.result()) for side, future in futures]
        official = [r for side, part in results if side == "official" for r in part]
        port = [r for side, part in results if side == "port" for r in part]
        outcome = compare(official, port)
        report[model] = outcome
        r = outcome["rates"]
        print(
            f"path {model}: {outcome['tests_compared']} tests, "
            f"{len(outcome['mismatches'])} mismatched, "
            f"{len(outcome['official_errors'])} reference errors"
        )
        for side in ("official", "port"):
            print(
                f"  {side:8}: syntax={r[f'{side}_syntax']:.4f} exec={r[f'{side}_exec']:.4f} "
                f"exact={r[f'{side}_exact']:.4f} similarity={r[f'{side}_similarity']:.4f}"
            )
        print(
            f"  port, own definition: exact={r['port_exact_paper']:.4f} "
            f"similarity={r['port_similarity_paper']:.4f}"
        )
        for m in outcome["mismatches"][:10]:
            print(f"  MISMATCH {m}")
        for e in outcome["official_errors"][:5]:
            print(f"  REFERENCE ERROR task {e['task_num']} test {e['test']}: {e['error'][-300:]}")

    out = WORK / "parity_path_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
