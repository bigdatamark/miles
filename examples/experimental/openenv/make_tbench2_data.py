# [MODULE DOC] Command-line generator of the prompt-data JSON-lines file for Terminal-Bench-2
# training and eval. Each output row has a prompt of one system message (the agent contract prompt
# imported from the agent function module) and metadata with the task id. Options: tasks_dir
# (TB2 checkout; default /workspace/terminal-bench-2), output (default /root/tbench2_train.jsonl),
# n (first n discovered tasks; zero means all) and an explicit comma-separated task id list which
# takes precedence over discovery and over n. Every selected task must have a task.toml file in its
# directory, otherwise the program exits with an error listing the missing ones before writing.
"""Generate a prompt dataset for the OpenEnv Terminal-Bench-2 (tbench2) run.

The *tasks* are not hand-written: they are the
Terminal-Bench-2 suite shipped in the laude-institute/terminal-bench-2 repo. The
env serves the per-task instruction at reset(), so each prompt-data row only
needs the system prompt (how the agent should behave) plus the ``task_id`` in
metadata. ``openenv_agent_function`` drives the multi-turn loop and reads
metadata["task_id"].

task_ids are the top-level task directory names in the TB2 repo checkout
(e.g. "chess-best-move"); a valid task dir contains a ``task.toml``.

    # all 89 tasks
    python make_tbench2_data.py --tasks_dir /workspace/terminal-bench-2 \
        --output /root/tbench2_train.jsonl
    # a small smoke subset
    python make_tbench2_data.py --tasks_dir /workspace/terminal-bench-2 \
        --output /root/tbench2_smoke.jsonl --n 8
    # an explicit subset
    python make_tbench2_data.py --tasks_dir /workspace/terminal-bench-2 \
        --tasks chess-best-move,circuit-fibsqrt
"""

import json
from pathlib import Path

# The agent contract (one ```bash block per turn; TASK_COMPLETE to stop) is
# defined by the adapter, next to the code that parses it.
from openenv_agent_function import TB2_AGENT_SYSTEM_PROMPT
from tap import Tap


class Args(Tap):
    # [CLASS DOC] Command-line options: tasks_dir (path of the TB2 checkout), output (file path to
    # write), n (limit on discovered tasks, zero for all), tasks (optional comma-separated ids).
    tasks_dir: str = "/workspace/terminal-bench-2"  # TB2 repo checkout
    output: str = "/root/tbench2_train.jsonl"
    n: int = 0  # 0 = all discovered tasks
    tasks: str = ""  # optional comma-separated explicit task_ids


def _discover_task_ids(tasks_dir: Path) -> list[str]:
    # [FUNCTION DOC] HELPER: Lists the names of the immediate children of the directory that contain
    # a regular file named task.toml, sorted alphabetically. Raises if the directory is missing.
    return sorted(p.name for p in tasks_dir.iterdir() if (p / "task.toml").is_file())


def main() -> None:
    # [FUNCTION DOC] Script entry point: resolves the tasks dir (user home expanded), picks task ids
    # (explicit list trimmed with empties dropped, else discovered ids truncated to the first n when
    # n is positive), exits with an error if any lacks task.toml, writes one JSON line per task to
    # the output file (overwriting it), and prints how many tasks were written.
    args = Args().parse_args()
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()

    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_ids = _discover_task_ids(tasks_dir)
        if args.n > 0:
            task_ids = task_ids[: args.n]

    missing = [t for t in task_ids if not (tasks_dir / t / "task.toml").is_file()]
    if missing:
        raise SystemExit(f"task_ids without a task.toml in {tasks_dir}: {missing}")

    with open(args.output, "w") as f:
        for tid in task_ids:
            row = {
                "prompt": [{"role": "system", "content": TB2_AGENT_SYSTEM_PROMPT}],
                "metadata": {"task_id": tid},
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(task_ids)} TB2 tasks to {args.output}")


if __name__ == "__main__":
    main()
