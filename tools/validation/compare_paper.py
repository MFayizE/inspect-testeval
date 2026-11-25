"""Compare Inspect logs of the TestEval tasks with the paper's Tables 1-4.

Reads the metrics from each log's header and prints one Markdown table per
task: the paper's value, this eval's value and the difference, in percentage
points. The paper's numbers were produced by the reference scripts, so cov@k,
branch and path results are compared with this eval's ``_reference`` keys;
the paper-definition values are listed below each table.

Usage:
    uv run python tools/validation/compare_paper.py LOG [LOG ...] [--paper-model GPT-4o-mini]

Each LOG is an .eval file or a directory of them. The paper row is chosen from
the log's model name unless ``--paper-model`` names one.
"""

import argparse
import json
from pathlib import Path

from inspect_ai.log import EvalLog, read_eval_log

HERE = Path(__file__).parent

# Paper column -> metric key in the log ("<score name>/<metric>").
MAPPING: dict[str, dict[str, str]] = {
    "overall": {
        "syntax": "overall_coverage/syntax_correctness",
        "execution": "overall_coverage/execution_correctness",
        "line": "line_coverage/mean",
        "branch": "branch_coverage/mean",
        **{
            f"{kind}_cov@{k}": f"{kind}_cov@{k}_reference/mean"
            for kind in ("line", "branch")
            for k in (1, 2, 5)
        },
    },
    "line": {
        "syntax": "syntax_correct/mean",
        "execution": "exec_correct/mean",
        "coverage": "covered/mean",
    },
    "branch": {
        "syntax": "syntax_correct/mean",
        "execution": "exec_correct_reference/mean",
        "total": "covered_reference/grouped_all",
        "easy": "covered_reference/grouped_easy",
        "medium": "covered_reference/grouped_medium",
        "hard": "covered_reference/grouped_hard",
    },
    "path": {
        "syntax": "syntax_correct/mean",
        "execution": "exec_correct/mean",
        "path_cov": "path_exact_match_reference/mean",
        "path_similarity": "path_similarity_reference/mean",
    },
}

# This eval's own definitions, shown for reference where they differ.
PAPER_DEFINITION: dict[str, list[str]] = {
    "overall": [f"{kind}_cov@{k}/mean" for kind in ("line", "branch") for k in (1, 2, 5)],
    "branch": ["exec_correct/mean", "covered/grouped_all"],
    "path": ["path_exact_match/mean", "path_similarity/mean"],
}

MODEL_NAMES = {
    "gpt-3.5-turbo": "GPT-3.5-turbo",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "meta-llama-3-8b": "Llama3-8b",
    "llama-3-8b": "Llama3-8b",
    "llama3-8b": "Llama3-8b",
}


def flat_metrics(log: EvalLog) -> dict[str, float]:
    if log.results is None:
        raise ValueError(f"{log.location}: no results (status {log.status})")
    return {
        f"{score.name}/{name}": metric.value
        for score in log.results.scores
        for name, metric in score.metrics.items()
    }


def task_kind(log: EvalLog) -> str:
    name = log.eval.task.rsplit("/", 1)[-1]
    kind = name.removeprefix("testeval_")
    if kind not in MAPPING:
        raise ValueError(f"{log.location}: not a TestEval task ({log.eval.task})")
    return kind


def paper_model(log: EvalLog, override: str | None, models: dict[str, object]) -> str:
    if override:
        return override
    name = log.eval.model.rsplit("/", 1)[-1].lower()
    for prefix, paper_name in sorted(MODEL_NAMES.items(), key=lambda kv: -len(kv[0])):
        if name.startswith(prefix):
            return paper_name
    raise ValueError(
        f"no paper row for model {log.eval.model}; pass --paper-model (one of {sorted(models)})"
    )


def report(log: EvalLog, paper: dict[str, dict[str, dict[str, float]]], model: str) -> str:
    kind = task_kind(log)
    metrics = flat_metrics(log)
    samples = log.results.completed_samples if log.results else 0
    lines = [
        f"### {log.eval.task} ({log.eval.model}, {samples} samples) vs paper {model}",
        "",
        "| Metric | Paper | This eval | Difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for column, key in MAPPING[kind].items():
        expected = paper[model][kind][column]
        actual = metrics.get(key)
        if actual is None:
            lines.append(f"| {column} | {expected:.2f} | missing (`{key}`) | |")
            continue
        ours = 100 * actual
        lines.append(f"| {column} | {expected:.2f} | {ours:.2f} | {ours - expected:+.2f} |")
    if kind in PAPER_DEFINITION:
        own = [f"`{k}` {100 * metrics[k]:.2f}" for k in PAPER_DEFINITION[kind] if k in metrics]
        lines += ["", "This eval's own definition: " + ", ".join(own)]
    return "\n".join(lines)


def log_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for path in map(Path, paths):
        files += sorted(path.glob("*.eval")) if path.is_dir() else [path]
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--paper-model")
    parser.add_argument("--out", type=Path, help="also write the tables to this Markdown file")
    args = parser.parse_args()

    tables = json.loads((HERE / "paper_tables.json").read_text(encoding="utf-8"))["models"]
    if args.paper_model and args.paper_model not in tables:
        parser.error(f"--paper-model must be one of {sorted(tables)}")
    sections = []
    for file in log_files(args.logs):
        log = read_eval_log(str(file), header_only=True)
        sections.append(report(log, tables, paper_model(log, args.paper_model, tables)))
    text = "\n\n".join(sections)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
