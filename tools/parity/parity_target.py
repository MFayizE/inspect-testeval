"""Compare this eval's targeted line and branch scoring with the reference scripts.

Both score the same formatted tests (see fetch.py) inside the same Linux image,
program by program: eval_linecov.py / eval_branchcov.py unmodified on one side
(run_official_target.py), this eval's coverage runner on the other
(run_port_target.py). For every program the counts of syntax-correct and
executable tests and the set of covered targets must match. For branches the
reference definition (``_reference`` keys) is compared; the paper definition
is printed alongside.

Usage (from the repository root, Docker running):
    uv run python tools/parity/parity_target.py [--targets line branch] [--models ...]
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from parity import HERE, WORK, build_image, prepare_port_dir

IMAGE = "inspect-testeval-parity:dev"


def run_slice(
    script: str, arg_dir: str, predictions: str, start: int, end: int, target: str
) -> list[dict[str, Any]]:
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "-v", f"{HERE.resolve()}:/parity:ro",
        "-v", f"{WORK.resolve()}:/work:ro",
        IMAGE, "python", f"/parity/{script}", arg_dir, predictions, str(start), str(end), target,
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script} {start}:{end} failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def compare(
    target: str, official: list[dict[str, Any]], port: list[dict[str, Any]]
) -> dict[str, Any]:
    suffix = "_reference" if target == "branch" else ""
    mismatches, errors = [], []
    totals = {
        "targets": 0,
        "syntax": 0,
        "exec": 0,
        "covered": 0,
        "exec_paper": 0,
        "covered_paper": 0,
    }
    official_totals = {"syntax": 0, "exec": 0, "covered": 0}
    for o, p in zip(official, port, strict=True):
        assert o["task_num"] == p["task_num"]
        if "error" in o:
            errors.append({"task_num": o["task_num"], "error": o["error"]})
            continue
        ours = {
            "total": p["total"],
            "syntax": p["syntax"],
            "exec": p[f"exec{suffix}"],
            "covered": p[f"covered{suffix}"],
        }
        theirs = {k: o[k] for k in ours}
        if ours != theirs:
            mismatches.append({"task_num": o["task_num"], "official": theirs, "port": ours})
        totals["targets"] += p["total"]
        totals["syntax"] += p["syntax"]
        totals["exec"] += ours["exec"]
        totals["covered"] += len(ours["covered"])
        totals["exec_paper"] += p["exec"]
        totals["covered_paper"] += len(p["covered"])
        official_totals["syntax"] += o["syntax"]
        official_totals["exec"] += o["exec"]
        official_totals["covered"] += len(o["covered"])
    return {
        "programs_compared": len(official) - len(errors),
        "mismatches": mismatches,
        "official_errors": errors,
        "port_totals": totals,
        "official_totals": official_totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["line", "branch"])
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "Meta-Llama-3-8B-Instruct"])
    parser.add_argument("--limit", type=int, default=210)
    parser.add_argument("--slice", type=int, default=10, help="programs per container")
    parser.add_argument("--jobs", type=int, default=6, help="containers at once")
    args = parser.parse_args()

    subprocess.run(["python", str(HERE / "fetch.py")], check=True)
    build_image()
    prepare_port_dir()

    report: dict[str, Any] = {}
    for target in args.targets:
        for model in args.models:
            name = f"{target}cov_{model}_format.jsonl"
            rows = (WORK / "predictions" / name).read_text(encoding="utf-8").splitlines()
            count = min(len([r for r in rows if r]), args.limit)
            predictions = f"/work/predictions/{name}"
            slices = [(s, min(s + args.slice, count)) for s in range(0, count, args.slice)]
            jobs = [("run_official_target.py", "/work/official", s, e) for s, e in slices]
            jobs += [("run_port_target.py", "/work/port", s, e) for s, e in slices]
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = [
                    (script, pool.submit(run_slice, script, arg_dir, predictions, s, e, target))
                    for script, arg_dir, s, e in jobs
                ]
                results = [(script, future.result()) for script, future in futures]
            official = [
                r for script, part in results if script == "run_official_target.py" for r in part
            ]
            port = [r for script, part in results if script == "run_port_target.py" for r in part]
            outcome = compare(target, official, port)
            report[f"{target} / {model}"] = outcome
            t, o = outcome["port_totals"], outcome["official_totals"]
            n = t["targets"] or 1
            print(
                f"{target} {model}: {outcome['programs_compared']} programs, {t['targets']} targets, "
                f"{len(outcome['mismatches'])} mismatched programs, "
                f"{len(outcome['official_errors'])} reference errors"
            )
            print(
                f"  official: syntax={o['syntax'] / n:.4f} exec={o['exec'] / n:.4f} "
                f"covered={o['covered'] / n:.4f}"
            )
            print(
                f"  port:     syntax={t['syntax'] / n:.4f} exec={t['exec'] / n:.4f} "
                f"covered={t['covered'] / n:.4f}"
            )
            if target == "branch":
                print(
                    f"  port, paper definition: exec={t['exec_paper'] / n:.4f} "
                    f"covered={t['covered_paper'] / n:.4f}"
                )
            for m in outcome["mismatches"][:10]:
                print(f"  MISMATCH {m}")
            for e in outcome["official_errors"][:5]:
                print(f"  REFERENCE ERROR task {e['task_num']}: {e['error'].splitlines()[-1]}")

    out = WORK / "parity_target_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
