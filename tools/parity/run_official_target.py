"""Score a slice of targeted tests with the unmodified eval_linecov.py or eval_branchcov.py.

Runs inside the parity container. The reference scripts only print totals, so
each program is scored by its own ``eval_correctness`` call and the result is
read from what that call prints: the counts line (syntax-correct, executable,
total) and one ``covered line N`` or ``covered branch S-E`` line per covered
target. Nothing in the reference code is changed; between calls this script
only resets what the reference code leaves behind for the next program:

* the working directory (a crash can leave it inside a ``tmp_*`` folder),
* the ``tmp_*`` folders and their modules in ``sys.modules``: every call names
  its folder ``tmp_0_<difficulty>``, so without this a later program would
  import an earlier program's ``Solution``,
* ``data/leetcode-py.jsonl``, which eval_branchcov.py reads by position: it
  holds just this program's record.

After the per-target lines, a reference script can raise ZeroDivisionError
computing a rate for a category with no members in a single program (no hard
branch, no executable test). That error is caught; any other error is
recorded for the program.

Usage: python run_official_target.py <official_dir> <predictions.jsonl> <start> <end> <line|branch>
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

COUNTS = re.compile(r"^(\d+) (\d+) (\d+)$", re.MULTILINE)


def main() -> None:
    official_dir, predictions, start, end, target = sys.argv[1:6]
    official = Path(official_dir)
    run_dir = Path(tempfile.mkdtemp(prefix="official_"))
    script = f"eval_{target}cov.py"
    for name in (script, "data_utils.py", ".coveragerc"):
        shutil.copy(official / name, run_dir / name)
    (run_dir / "data").mkdir()
    reference = {}
    with open(official / "data" / "leetcode-py.jsonl", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            reference[record["task_num"]] = line
    os.chdir(run_dir)
    sys.path.insert(0, str(run_dir))

    module = __import__(script[:-3])  # the reference scorer, unmodified

    with open(predictions, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh][int(start) : int(end)]

    results = []
    for row in rows:
        os.chdir(run_dir)
        for path in run_dir.glob("tmp*"):
            shutil.rmtree(path)
        for name in [m for m in sys.modules if m.startswith("tmp_")]:
            del sys.modules[name]
        (run_dir / "data" / "leetcode-py.jsonl").write_text(
            reference[row["task_num"]], encoding="utf-8"
        )

        stdout = io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(stdout):
                module.eval_correctness([row])
        except ZeroDivisionError:
            pass  # a rate over an empty category, computed after every target was scored
        except Exception:
            error = traceback.format_exc(limit=3)
        text = stdout.getvalue()
        counts = COUNTS.findall(text)
        if error is None and not counts:
            error = f"no counts line in output:\n{text[-2000:]}"
        if error is not None:
            results.append({"task_num": row["task_num"], "error": error})
            continue
        syntax, executed, total = (int(x) for x in counts[-1])
        if target == "line":
            covered = sorted(
                int(n) for n in re.findall(r"^covered line (\d+)$", text, re.MULTILINE)
            )
        else:
            covered = sorted(
                f"{s}-{e}"
                for s, e in re.findall(r"^covered branch (\d+)-(\d+)$", text, re.MULTILINE)
            )
        results.append(
            {
                "task_num": row["task_num"],
                "total": total,
                "syntax": syntax,
                "exec": executed,
                "covered": covered,
            }
        )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
