# [MODULE DOC]
# Command line tool (argument parsing with the Tap library) converting a native Workplace
# Assistant task file in JSON lines format into Miles prompt data. Each output row keeps only what
# the policy may see: the input messages as the prompt, the task id as the label, and a metadata
# mapping with the task id, category, the reviewed policy-visible request parameters and the
# maximum turn count; the ground-truth verifier targets are required to exist in the source but are
# deliberately not copied. Prints one JSON object with task count, SHA-256 digests of the source and
# target files and the turn limit.
"""Convert native Workplace rows to Miles input without exposing gold actions."""

import hashlib
import json
from pathlib import Path

from tap import Tap


def convert(source: Path, target: Path, max_turns: int = 24) -> dict:
    # [FUNCTION DOC] Converts the file and returns a summary mapping (task count, source digest,
    # target digest, turn limit). Raises a value error if the turn limit is below one or the source
    # and target resolve to the same path; if there are no rows or task ids repeat; if a row's
    # request parameters contain keys other than input, tools, parallel tool calls and temperature;
    # or if a row lacks tools, input or ground truth (empty values). Blank lines in the source are
    # skipped. Creates the target's parent directories and writes all rows (one JSON line each,
    # without ASCII escaping) in one write after validating every row, so failures write nothing.
    # Missing required keys raise key errors. Side effects: reads and writes files.
    if max_turns < 1 or source.resolve() == target.resolve():
        raise ValueError("Use a positive turn limit and a target distinct from the native dataset")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty rows with unique task IDs")
    target.parent.mkdir(parents=True, exist_ok=True)
    converted = []
    for row in rows:
        params = row["responses_create_params"]
        if set(params) - {"input", "tools", "parallel_tool_calls", "temperature"}:
            raise ValueError("Unreviewed policy field in native row")
        if not params["tools"] or not params["input"] or not row["ground_truth"]:
            raise ValueError("Missing policy input, tools, or verifier target")
        output = {
            "prompt": params["input"],
            "label": str(row["id"]),
            "metadata": {
                "workplace_task_id": row["id"],
                "workplace_category": row["category"],
                "workplace_policy": params,
                "workplace_max_turns": max_turns,
            },
        }
        converted.append(json.dumps(output, ensure_ascii=False) + "\n")
    target.write_text("".join(converted))
    return {
        "tasks": len(rows),
        "native_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "miles_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "max_turns": max_turns,
    }


class Args(Tap):
    # [CLASS DOC]
    # Command line options: required source path, required target path, and the maximum number of
    # turns stored in each row's metadata (default 24).
    source: Path
    target: Path
    max_turns: int = 24


def main() -> None:
    # [FUNCTION DOC] Parses the command line, runs the conversion and prints its summary mapping
    # as a single JSON line.
    args = Args().parse_args()
    print(json.dumps(convert(args.source, args.target, args.max_turns)))


if __name__ == "__main__":
    main()
