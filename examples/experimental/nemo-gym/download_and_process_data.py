#!/usr/bin/env python3
# [MODULE DOC] Command-line tool that turns a SWE-style dataset (a Hugging Face dataset id or a
# local JSON-lines file) into the Miles prompt-data format used by the NeMo Gym example: one JSON
# line per sample with a prompt (the problem statement) and a metadata object (the full original
# instance plus the subset and split labels). Arguments: input (required), output (required),
# split (default train; only used for Hugging Face datasets but always recorded in metadata),
# limit (optional sample cap) and subset (default gym; "gym" selects SWE-Gym images, "verified"
# the official SWE-bench images). Prints progress to standard output.
"""Download and process data to Miles format."""

import argparse
import json
import tempfile
from pathlib import Path
from datasets import load_dataset


def convert_to_miles_format(
    input_path: str, output_path: str, limit: int = None, split: str = "train", subset: str = "gym"
):
    # [FUNCTION DOC] Streams the input JSON-lines file and writes the output JSON-lines file (created
    # or overwritten), one output line per non-skipped input line, then prints a count message.
    # Each output record has "prompt" (the instance's problem_statement, or an empty string when
    # missing) and "metadata" (a shallow copy of the whole instance with "subset" and "split" keys
    # set, overwriting any existing values). Stops once limit records are written. Returns nothing.
    # Raises on unreadable input or malformed JSON lines (a blank line also fails to parse).
    # KNOWN ISSUE: the limit check is a truthiness test, so a limit of zero means "no limit"; the
    # annotation for limit says integer but the default is none (should be optional).
    """Convert JSONL to Miles format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file in Miles format
        limit: Optional limit on number of samples
        split: Dataset split name (used in metadata)
        subset: NeMo Gym subset controlling per-task image selection and eval
            ("gym" -> SWE-Gym images, "verified" -> official SWE-bench images)
    """
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if limit and count >= limit:
                break

            instance = json.loads(line)

            # Add subset and split to metadata for Gym API
            metadata = dict(instance)
            metadata["subset"] = subset
            metadata["split"] = split

            miles_sample = {
                "prompt": instance.get("problem_statement", ""),
                "metadata": metadata,
            }

            fout.write(json.dumps(miles_sample) + "\n")
            count += 1

    print(f"Converted {count} samples: {input_path} -> {output_path}")


def main():
    # [FUNCTION DOC] Script entry point. Parses the arguments. If the input path exists locally and
    # ends with the jsonl suffix, converts it directly. Otherwise treats the input as a Hugging Face
    # dataset id: loads the requested split (network download), optionally truncates it to the
    # first limit rows, dumps it to a temporary JSON-lines file, converts that file to the output
    # (passing the limit again, which is harmless), and always deletes the temporary file in a
    # finally block. Prints "Done." at the end.
    parser = argparse.ArgumentParser(description="Download HuggingFace dataset and convert to Miles format")
    parser.add_argument("--input", type=str, required=True, help="HuggingFace dataset path or local JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (default: train, only for HF datasets)"
    )
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument(
        "--subset", type=str, default="gym", help='NeMo Gym subset: "gym" (SWE-Gym) or "verified" (SWE-bench Verified)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.exists() and input_path.suffix == ".jsonl":
        print(f"Processing local file: {args.input}")
        convert_to_miles_format(args.input, args.output, args.limit, args.split, args.subset)
    else:
        print(f"Loading HuggingFace dataset: {args.input} (split={args.split})")
        ds = load_dataset(args.input, split=args.split)

        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
                tmp_path = tmp.name

            print(f"Downloading to temporary file: {tmp_path}")
            ds.to_json(tmp_path)

            print(f"Converting to Miles format: {args.output}")
            convert_to_miles_format(tmp_path, args.output, args.limit, split=args.split, subset=args.subset)
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    print("Done.")


if __name__ == "__main__":
    main()
