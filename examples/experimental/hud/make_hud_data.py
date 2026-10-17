# [MODULE DOC]
# Command line tool that converts a HUD environment package's task list into a Miles training
# prompt file in JSON lines format. It loads the package's tasks module by file path (temporarily
# adding the package directory to the module search path), optionally keeps only tasks whose slugs
# are listed, optionally merges a JSON object of argument overrides over every selected task's
# arguments, and writes the whole selection repeated the requested number of times (the repeat loop
# is outermost, so tasks interleave). Each output record has a one-message user prompt holding the
# task slug, an empty label and a metadata mapping with the task row (environment name, task id,
# arguments, slug) under the HUD task key. Exits with a message if nothing is selected. Prints the
# selected slugs and the number of rows written.
"""Build a Miles prompt jsonl from a HUD v6 taskset.

A v6 task is a data row -- an env name, a template id, bound args -- minted by
calling a template in the env package's ``tasks.py``. This walks that module's
public ``tasks`` list and writes one Miles row per task (repeated ``--repeat``
times: GRPO needs groups, and a taskset like the browser starter's 2048 has a
handful of templates, not thousands of rows).

The ``prompt`` column is one chat message carrying the task slug: with a
chat-completions harness the model never sees this file -- its actual prompt
is the template's first yield, delivered inside the environment at run time.
Miles uses the column only to group samples of the same task (and its VLM data
path requires the chat-message shape, hence not a bare string).

    python -m examples.experimental.hud.make_hud_data \
        --env-dir /root/v6browser --task-ids 2048-score-5000 \
        --repeat 256 --output /root/hud2048_train.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_tasks(env_dir: str | Path):
    # [FUNCTION DOC] Imports the tasks module (tasks file inside the expanded, resolved directory)
    # under a fixed module name with the directory temporarily first on the module search path
    # (removed afterwards even on error), executing its code, and returns a copy of its public
    # tasks list. Raises if the file is missing or lacks the list. Executes arbitrary code from the
    # package.
    """The env package's minted task rows (its public ``tasks`` list)."""
    env_dir = Path(env_dir).expanduser().resolve()
    sys.path.insert(0, str(env_dir))
    try:
        spec = importlib.util.spec_from_file_location("hud_env_tasks", env_dir / "tasks.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return list(module.tasks)
    finally:
        sys.path.remove(str(env_dir))


def task_row(task) -> dict:
    # [FUNCTION DOC] Returns a plain mapping with the task's environment name, id, a copy of its
    # arguments (empty if none) and slug: the part needed to re-create the task at rollout time.
    """The part of a Task that travels to the rollout: enough to re-mint it."""
    return {"env": task.env, "id": task.id, "args": dict(task.args or {}), "slug": task.slug}


def main() -> None:
    # [FUNCTION DOC] Command line entry point. Options: required environment package directory;
    # optional list of task slugs (default all); optional JSON text of argument overrides; repeat
    # count (default 1); output path (default hud_train.jsonl under the root directory). Overrides
    # are merged shallowly over each task's arguments (override wins). Opens the output for writing
    # (truncating) and writes one JSON line per task per repeat, without ASCII escaping.
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", required=True, help="HUD env package (has tasks.py + Dockerfile.hud)")
    ap.add_argument("--task-ids", nargs="*", default=None, help="slugs to include; default all")
    ap.add_argument(
        "--args-json",
        default=None,
        help=(
            "JSON merged over each selected task's args, e.g. "
            "'{\"target_score\": 1024}' -- how a recipe retunes a template "
            "without editing the env package"
        ),
    )
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--output", default="/root/hud_train.jsonl")
    args = ap.parse_args()

    tasks = load_tasks(args.env_dir)
    if args.task_ids:
        tasks = [t for t in tasks if t.slug in set(args.task_ids)]
    if not tasks:
        raise SystemExit(f"no tasks selected from {args.env_dir}")
    print(f"{len(tasks)} task(s): {[t.slug for t in tasks]}")

    overrides = json.loads(args.args_json) if args.args_json else {}
    written = 0
    with open(args.output, "w") as f:
        for _ in range(args.repeat):
            for t in tasks:
                row = task_row(t)
                row["args"] = {**row["args"], **overrides}
                record = {
                    "prompt": [{"role": "user", "content": t.slug}],
                    "label": "",
                    "metadata": {"hud_task": row},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    print(f"wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
