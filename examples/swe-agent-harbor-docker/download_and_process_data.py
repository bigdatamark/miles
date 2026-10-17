#!/usr/bin/env python3
# [MODULE DOC] Command line tool that turns a task dataset into the JSON-lines training format
# of this repository: one record per task with a prompt string and a metadata mapping. Input is
# either a local JSON-lines file (an existing path ending in the jsonl extension) or a HuggingFace
# dataset name; a HuggingFace dataset is first exported to a temporary JSON-lines file (deleted
# afterwards even on failure) and then converted by the same function. The metadata is a copy of
# the whole original record plus the agent name and split labels, so downstream task-directory
# preparation can use any original field. The prompt is read from a configurable key, falling
# back (in order) to the problem statement, instruction and prompt keys when that is empty.
# Options: required input and output; split (default train, used for loading and recorded in
# metadata); optional record limit; agent name (default mini-swe-agent); prompt key (default
# problem_statement); append switch (otherwise the output file is overwritten). Requires the
# datasets library at import time.
"""Download and convert datasets to Miles format for SWE-agent training.

Supports any task type — SWE-bench, Terminal-Bench, custom datasets, etc.
Each record's metadata is enriched with ``agent_name`` so that
the Harbor agent server can route to the correct agent.
All original fields are preserved in metadata so that
prepare_harbor_tasks.py can infer the right task directory layout.

Usage examples:

    # SWE-bench from HuggingFace (defaults)
    python download_and_process_data.py \\
        --input SWE-Gym/SWE-Gym --output /root/swe_train.jsonl

    # Terminal-Bench from local JSONL
    python download_and_process_data.py \\
        --input /data/tb_tasks.jsonl --output /root/tb_train.jsonl \\
        --agent-name terminus-2 --prompt-key instruction

    # Custom dataset
    python download_and_process_data.py \\
        --input /data/my_tasks.jsonl --output /root/custom_train.jsonl \\
        --agent-name my-agent --prompt-key task_description

    # Merge multiple outputs into one mixed JSONL
    cat /root/swe_train.jsonl /root/tb_train.jsonl > /root/mixed.jsonl
"""

import argparse
import json
import tempfile
from pathlib import Path

from datasets import load_dataset

_PROMPT_KEY_FALLBACKS = ("problem_statement", "instruction", "prompt")


def convert_to_miles_format(
    input_path: str,
    output_path: str,
    *,
    limit: int | None = None,
    split: str = "train",
    agent_name: str = "mini-swe-agent",
    prompt_key: str = "problem_statement",
    append: bool = False,
) -> int:
    # [FUNCTION DOC] Stream records from the input JSON-lines file to the output file (opened for
    # append or overwrite) and return the number written. Reading stops once the limit (if not
    # none) is reached; a limit of zero therefore writes nothing. Each input line is parsed as a
    # JSON object (a blank or invalid line raises a decoding error). Output record: a prompt text
    # (value of the prompt key, else the first non-empty of the fallback keys, else empty string)
    # and a metadata mapping holding a shallow copy of the input record with the agent name and
    # split added (overriding same-named input fields). Prints a summary line.
    """Convert JSONL to Miles format.

    Returns the number of records written.
    """
    count = 0
    mode = "a" if append else "w"
    with open(input_path) as fin, open(output_path, mode) as fout:
        for line in fin:
            if limit is not None and count >= limit:
                break

            instance = json.loads(line)

            metadata = dict(instance)
            metadata["agent_name"] = agent_name
            metadata["split"] = split

            prompt = instance.get(prompt_key, "")
            if not prompt:
                for fallback in _PROMPT_KEY_FALLBACKS:
                    prompt = instance.get(fallback, "")
                    if prompt:
                        break

            miles_sample = {
                "prompt": prompt,
                "metadata": metadata,
            }

            fout.write(json.dumps(miles_sample) + "\n")
            count += 1

    print(f"Converted {count} samples: {input_path} -> {output_path}")
    return count


def main():
    # [FUNCTION DOC] Command line entry (options in the module doc). Local jsonl input is
    # converted directly. Otherwise loads the named HuggingFace dataset split, truncates it to the
    # limit when the limit is truthy (note: a limit of zero means no limit here, unlike in the
    # conversion function), writes it to a temporary JSON-lines file, converts that to the output
    # and removes the temporary file in a finally block. Prints progress and a final done line.
    parser = argparse.ArgumentParser(
        description="Download dataset and convert to Miles format",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="HuggingFace dataset path or local JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument(
        "--agent-name",
        type=str,
        default="mini-swe-agent",
        help="Harbor agent name injected into metadata " "(default: mini-swe-agent)",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="problem_statement",
        help="JSON key to use as prompt text " "(default: problem_statement)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    common_kwargs = dict(
        limit=args.limit,
        split=args.split,
        agent_name=args.agent_name,
        prompt_key=args.prompt_key,
        append=args.append,
    )

    if input_path.exists() and input_path.suffix == ".jsonl":
        print(f"Processing local file: {args.input}")
        convert_to_miles_format(args.input, args.output, **common_kwargs)
    else:
        print(f"Loading HuggingFace dataset: " f"{args.input} (split={args.split})")
        ds = load_dataset(args.input, split=args.split)

        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".jsonl",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name

            print(f"Downloading to temporary file: {tmp_path}")
            ds.to_json(tmp_path)

            print(f"Converting to Miles format: {args.output}")
            convert_to_miles_format(tmp_path, args.output, **common_kwargs)
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    print("Done.")


if __name__ == "__main__":
    main()
