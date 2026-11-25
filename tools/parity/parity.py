"""Compare this eval's scorer with the reference eval_overall.py on real model outputs.

Both score the same formatted tests (see fetch.py) inside the same Linux
image. In ``identity`` mode neither shuffles tests before forming cov@k groups,
so every metric must match exactly; ``seeded`` mode is informational because
the two shuffle with different random streams.

Usage (from the repository root, Docker running):
    uv run python tools/parity/parity.py [--models gpt-4o ...] [--limit N] [--jobs 6]
"""

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from inspect_testeval.scorers import COVERAGERC

HERE = Path(__file__).parent
REPO = HERE.parent.parent
WORK = HERE / "work"
IMAGE = "inspect-testeval-parity:dev"
SHARED_KEYS = [
    "syntax_correct",
    "exec_correct",
    "line_coverage",
    "branch_coverage",
    *[f"{kind}_cov@{k}" for k in (1, 2, 5) for kind in ("line", "branch")],
]


def build_image() -> None:
    subprocess.run(
        ["docker", "build", "-q", "-t", IMAGE, "-f", str(HERE / "Dockerfile"), str(REPO)],
        check=True,
    )


def prepare_port_dir() -> Path:
    port_dir = WORK / "port"
    port_dir.mkdir(parents=True, exist_ok=True)
    source = REPO / "src" / "inspect_testeval" / "coverage_runner.py"
    (port_dir / "coverage_runner.py").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (port_dir / ".coveragerc").write_text(COVERAGERC, encoding="utf-8")
    return port_dir


def run_slice(script: str, arg_dir: str, predictions: str, start: int, end: int, mode: str) -> dict:
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "-v", f"{HERE.resolve()}:/parity:ro",
        "-v", f"{WORK.resolve()}:/work:ro",
        IMAGE, "python", f"/parity/{script}", arg_dir, predictions, str(start), str(end), mode,
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{script} {start}:{end} failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def combine(parts: list[dict]) -> dict:
    total = sum(p["n"] for p in parts)
    keys = [k for k in parts[0] if k not in ("n", "total_cases")]
    return {"n": total, **{k: sum(p[k] * p["n"] for p in parts) / total for k in keys}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "Meta-Llama-3-8B-Instruct"])
    parser.add_argument("--modes", nargs="+", default=["identity", "seeded"])
    parser.add_argument("--limit", type=int, default=210)
    parser.add_argument("--slice", type=int, default=15, help="programs per container")
    parser.add_argument("--jobs", type=int, default=6, help="containers at once")
    parser.add_argument(
        "--exclude",
        nargs="*",
        type=int,
        default=[952],
        help=(
            "task_num values left out of the comparison. Default 952: a Meta-Llama-3-8B "
            "test calls it with 10**9, the program allocates an array that large and the "
            "reference scorer, which runs every test inside its own single process, is "
            "killed (exit 137)."
        ),
    )
    args = parser.parse_args()

    subprocess.run(["python", str(HERE / "fetch.py")], check=True)
    build_image()
    prepare_port_dir()

    report: dict[str, dict] = {}
    for model in args.models:
        lines = (WORK / "predictions" / f"totalcov_{model}_format.jsonl").read_text(
            encoding="utf-8"
        )
        rows = [json.loads(line) for line in lines.splitlines() if line][: args.limit]
        excluded = [r["task_num"] for r in rows if r["task_num"] in args.exclude]
        rows = [r for r in rows if r["task_num"] not in args.exclude]
        # Both scorers read the same filtered file, so slice indices line up.
        filtered = WORK / "predictions" / f"compared_{model}.jsonl"
        filtered.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        predictions = f"/work/predictions/{filtered.name}"
        if excluded:
            print(f"{model}: excluded task_num {excluded}")
        assert len({len(r["tests"]) for r in rows}) == 1, "slices are weighted by program count"
        slices = [(s, min(s + args.slice, len(rows))) for s in range(0, len(rows), args.slice)]
        for mode in args.modes:
            jobs = [("run_official.py", "/work/official", s, e) for s, e in slices]
            jobs += [("run_port.py", "/work/port", s, e) for s, e in slices]
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = [
                    (script, pool.submit(run_slice, script, arg_dir, predictions, s, e, mode))
                    for script, arg_dir, s, e in jobs
                ]
                results = [(script, future.result()) for script, future in futures]
            official = combine([r for name, r in results if name == "run_official.py"])
            port = combine([r for name, r in results if name == "run_port.py"])
            rows_out = {}
            for key in SHARED_KEYS:
                # The reference code's cov@k is the port's "_reference" definition.
                port_key = f"{key}_reference" if "_cov@" in key else key
                diff = port[port_key] - official[key]
                rows_out[key] = {"official": official[key], "port": port[port_key], "diff": diff}
            rows_out["port_only"] = {
                k: v for k, v in port.items() if "_cov@" in k and not k.endswith("_reference")
            }
            report[f"{model} / {mode}"] = rows_out
            worst = max(abs(v["diff"]) for k, v in rows_out.items() if k != "port_only")
            print(f"{model} [{mode}] n={official['n']}: max |port - official| = {worst:.3g}")
            for key in SHARED_KEYS:
                v = rows_out[key]
                flag = "" if math.isclose(v["diff"], 0.0, abs_tol=1e-12) else "  <-- differs"
                print(f"  {key:16} official={v['official']:.6f} port={v['port']:.6f}{flag}")
            paper = rows_out["port_only"]
            print(
                "  paper definition (port only): "
                + ", ".join(f"{k}={v:.4f}" for k, v in paper.items())
            )

    out = WORK / "parity_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
