# [MODULE DOC]
# Command line data preparation for the single-round formal-math (Lean 4 theorem proving)
# examples, built with typer. Two modes. Mode rl: downloads the FineLeanCorpus dataset, keeps rows
# whose Lean code contains exactly one theorem (optionally also a given difficulty, and optionally
# only questions with positive mean reward in saved rollout dumps), shuffles with a fixed seed,
# takes train plus validation rows, splits off the validation rows from the end, rewrites each
# statement into a chat prompt (the Lean theorem with its informal statement as a comment and its
# proof body replaced by a placeholder, wrapped in a proof-plan prompt template following
# DeepSeek-Prover-V2) and writes train and test JSON-lines files; also processes the miniF2F test
# set into a JSON-lines file of prompts. Mode sft: processes the Leanabell SFT dataset into a
# parquet file of user/assistant message lists. Every row gets a metadata mapping with a question
# id made of the dataset name and a source id or row index. Output goes to a directory named by
# the output name option, or a timestamp plus random number, under a base directory. Needs network
# access to the Hugging Face hub and (for the rollout-dump filter) the torch and polars packages.
import datetime
import pprint
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import polars as pl
import torch
import typer
from datasets import load_dataset

self_stem = Path(__file__).stem

# https://github.com/deepseek-ai/DeepSeek-Prover-V2
_PROMPT_TEMPLATE = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()


_NEEDLE_THEOREM = "theorem "


def process_flc(
    dir_output: Path,
    train_flc_select_num_rows: int,
    val_flc_select_num_rows: int,
    filter_difficulty: int | None,
    filter_solvable_by_rollout_dumps: str,
):
    # [FUNCTION DOC] Builds the RL train and validation prompt files from FineLeanCorpus. Steps:
    # load the train split; add the metadata column keyed by the corpus id with dataset name flc;
    # if a rollout-dump filter is given (a comma-separated list of path groups), compute the set of
    # question ids with positive mean reward; filter rows (using sixty-four processes) to those
    # whose Lean code has exactly one occurrence of the theorem keyword, whose difficulty equals the
    # requested one when set, and whose question id is in the solvable set when the filter value is
    # non-empty; shuffle with seed 42; keep the id, statement, Lean code and metadata columns; cut to
    # the train plus validation row count; split by taking the last validation-count rows as test
    # (no further shuffling); map each row into a prompt message list (see the nested helpers) with
    # the statement and code columns removed; write flc_train.jsonl and flc_test.jsonl into the
    # output directory. Prints progress and sample rows.
    # KNOWN ISSUE: the filter treats an empty string as not provided when filtering rows, but an
    # empty string still triggers the (failing) dump computation because only a missing value is
    # checked first.
    ds = load_dataset("m-a-p/FineLeanCorpus", split="train")
    print(f"Loaded dataset: {len(ds)=}")
    ds = _add_metadata_column(ds, dataset_name="flc", column_id="id")

    if (x := filter_solvable_by_rollout_dumps) is not None:
        interesting_question_ids = _SolvableByRolloutDumpFilter.compute_interesting_question_ids(x)

    def _filter_row(lean_code, difficulty, metadata):
        # [FUNCTION DOC] HELPER: Row predicate. True when the Lean code has exactly one theorem
        # keyword occurrence (multi-theorem data is dropped), and the difficulty filter is unset or
        # equal, and the rollout-dump filter is falsy or the row's question id is in the solvable set.
        return (
            # we remove multi-theorem data currently
            (lean_code.count(_NEEDLE_THEOREM) == 1)
            and ((filter_difficulty is None) or (difficulty == filter_difficulty))
            and ((not filter_solvable_by_rollout_dumps) or (metadata["question_id"] in interesting_question_ids))
        )

    ds = ds.filter(
        lambda batch: [
            _filter_row(lean_code, difficulty, metadata)
            for lean_code, difficulty, metadata in zip(
                batch["lean_code"], batch["difficulty"], batch["metadata"], strict=True
            )
        ],
        batched=True,
        num_proc=64,
    )
    print(f"Filtered dataset: {len(ds)=}")

    ds = ds.shuffle(seed=42)
    ds = ds.select_columns(["id", "statement", "lean_code", "metadata"])
    ds = ds.select(range(min(len(ds), train_flc_select_num_rows + val_flc_select_num_rows)))
    ds = ds.train_test_split(test_size=val_flc_select_num_rows, shuffle=False, seed=42)
    print(f"Split dataset: {len(ds['train'])=} {len(ds['test'])=}")

    def _process_prompt(statement, lean_code):
        # [FUNCTION DOC] HELPER: Builds one prompt message list. Asserts exactly one theorem keyword;
        # inserts a block comment containing the informal statement and a newline just before that
        # keyword; converts the proof tail to a by-sorry placeholder; formats it into the prompt
        # template; wraps it as a single user message. Returns the message list.
        assert lean_code.count(_NEEDLE_THEOREM) == 1, f"{lean_code=}"
        x = lean_code.replace(_NEEDLE_THEOREM, f"/- {statement} -/\n{_NEEDLE_THEOREM}")

        x = _convert_to_by_sorry(x)
        x = _PROMPT_TEMPLATE.format(x)
        x = _to_messages(x)
        return x

    def _process_batch(batch):
        # [FUNCTION DOC] HELPER: Batched map function; returns a mapping with a prompt column made
        # by pairing the batch's statement and Lean code columns (equal lengths enforced).
        return {
            "prompt": [
                _process_prompt(statement, lean_code)
                for statement, lean_code in zip(batch["statement"], batch["lean_code"], strict=True)
            ]
        }

    ds = ds.map(_process_batch, batched=True, num_proc=64, remove_columns=["statement", "lean_code"])
    _write_file(ds["train"], dir_output / "flc_train.jsonl")
    _write_file(ds["test"], dir_output / "flc_test.jsonl")


class _SolvableByRolloutDumpFilter:
    # [CLASS DOC] HELPER: Namespace of static helpers that scan saved rollout debug dumps and find
    # the question ids that the policy solved at least sometimes (mean reward above zero).
    @staticmethod
    def compute_interesting_question_ids(paths):
        # [FUNCTION DOC] HELPER: Takes a string of groups separated by commas (each group itself a
        # string that may again hold comma-separated dump files), processes each group in a pool of
        # up to sixty-four processes and returns the union of the resulting question id sets. Prints
        # the overall count and a few examples.
        paths = paths.split(",")
        interesting_question_ids = set()
        with ProcessPoolExecutor(max_workers=64) as executor:
            for partial_question_ids in executor.map(
                _SolvableByRolloutDumpFilter._compute_interesting_question_ids_one, paths
            ):
                interesting_question_ids |= partial_question_ids
        print(f"(overall) {len(interesting_question_ids)=} {list(interesting_question_ids)[:5]=}")
        return interesting_question_ids

    @staticmethod
    def _compute_interesting_question_ids_one(paths):
        # [FUNCTION DOC] HELPER: Worker function for one group; loads its samples table and returns
        # the set of solvable question ids from it.
        df_samples = _SolvableByRolloutDumpFilter._compute_df_samples(paths)
        return _SolvableByRolloutDumpFilter._compute_interesting_question_ids_from_df_samples(df_samples)

    @staticmethod
    def _compute_df_samples(paths: str):
        # [FUNCTION DOC] HELPER: Splits the group string on commas, loads each file with the torch
        # loader (each file is a mapping with a samples list of per-sample records), turns each into
        # a data frame and concatenates them diagonally with relaxed type matching (missing columns
        # filled with nulls). Prints the table and returns it.
        print(f"compute_df_samples {paths=}")
        paths = paths.split(",")
        df_samples = pl.concat([pl.DataFrame(torch.load(p)["samples"]) for p in paths], how="diagonal_relaxed")
        print(f"{df_samples=}")
        return df_samples

    @staticmethod
    def _compute_interesting_question_ids_from_df_samples(df_samples: pl.DataFrame):
        # [FUNCTION DOC] HELPER: From the samples table selects prompt, response, the reward value
        # field of each reward record and the question id field of each metadata record; averages the
        # reward value per question id; keeps question ids whose mean is strictly positive. Prints
        # the count and a few ids and returns them as a set.
        df = df_samples

        df = df.select(
            "prompt",
            "response",
            pl.col("reward").struct.field("reward_value"),
            pl.col("metadata").struct.field("question_id"),
        )
        df = df.group_by("question_id").agg(pl.col("reward_value").mean())

        interesting_question_ids = df.filter(pl.col("reward_value") > 0)["question_id"].sort().to_list()
        print(f"(partial) {len(interesting_question_ids)=} {interesting_question_ids[:5]=}")

        return set(interesting_question_ids)


_LEANABELL_ORIGINAL_PREFIX = """Complete the following Lean 4 code with explanatory comments preceding each line of code:

```lean4
"""


def process_leanabell(
    dir_output: Path,
):
    # [FUNCTION DOC] Builds the SFT file from the Leanabell prover training dataset. Loads the
    # train split, adds metadata keyed by row index with dataset name leanabell, shuffles with seed
    # 42, maps each row (using sixty-four processes) into a two-message conversation, removing the
    # raw prompt and output columns, and writes leanabell.parquet into the output directory. The
    # user message is the prompt template filled with the original question after removing the
    # dataset's fixed instruction prefix (asserted present) and converting the proof tail to a
    # by-sorry placeholder; the assistant message is the original output prefixed by an opened Lean
    # code fence.
    ds = load_dataset("stoney0062/Leanabell-Prover-Traindata-SFT", split="train")
    ds = _add_metadata_column(ds, dataset_name="leanabell")
    ds = ds.shuffle(seed=42)

    def _compute_messages(raw_prompt, raw_output):
        # [FUNCTION DOC] HELPER: Converts one raw prompt and output into the user/assistant message
        # list described by the enclosing function.
        question_lean = _ensure_remove_prefix(raw_prompt, _LEANABELL_ORIGINAL_PREFIX)
        question_lean = _convert_to_by_sorry(question_lean)

        return [
            {"role": "user", "content": _PROMPT_TEMPLATE.format(question_lean)},
            {"role": "assistant", "content": f"```lean\n{raw_output}"},
        ]

    def _process_batch(batch):
        # [FUNCTION DOC] HELPER: Batched map function returning a messages column built from the
        # batch's prompt and output columns (equal lengths enforced).
        return {
            "messages": [
                _compute_messages(prompt, output)
                for prompt, output in zip(batch["prompt"], batch["output"], strict=True)
            ]
        }

    ds = ds.map(_process_batch, batched=True, num_proc=64, remove_columns=["prompt", "output"])
    _write_file(ds, dir_output / "leanabell.parquet")


def _ensure_remove_prefix(s: str, prefix: str):
    # [FUNCTION DOC] HELPER: Asserts the string starts with the prefix and returns it with the
    # prefix removed.
    assert s.startswith(prefix), f"{prefix=} {s=}"
    return s.removeprefix(prefix)


def process_minif2f(
    dir_output: Path,
):
    # [FUNCTION DOC] Builds the evaluation prompt file from the miniF2F test set. Loads the train
    # split (the only split), adds metadata keyed by row index with dataset name minif2f, shuffles
    # with seed 42, drops the name and informal prefix columns, maps each formal statement into a
    # single-user-message prompt (proof tail replaced by a placeholder, wrapped in the template) in
    # a prompt column, and writes minif2f_test.jsonl into the output directory.
    ds = load_dataset("AI-MO/minif2f_test", split="train")
    ds = _add_metadata_column(ds, dataset_name="minif2f")
    ds = ds.shuffle(seed=42)
    ds = ds.remove_columns(["name", "informal_prefix"])

    def _process_prompt(x):
        # [FUNCTION DOC] HELPER: Turns a formal statement into a prompt message list (placeholder
        # conversion, template formatting, user message wrapping).
        x = _convert_to_by_sorry(x)
        x = _PROMPT_TEMPLATE.format(x)
        x = _to_messages(x)
        return x

    def _process_batch(batch):
        # [FUNCTION DOC] HELPER: Batched map function returning a prompt column built from the
        # batch's formal statement column.
        return {"prompt": [_process_prompt(x) for x in batch["formal_statement"]]}

    ds = ds.map(_process_batch, batched=True)
    _write_file(ds, dir_output / "minif2f_test.jsonl")


def _write_file(ds, path):
    # [FUNCTION DOC] HELPER: Writes the dataset to the path in the format given by its suffix
    # (JSON lines or parquet; any other suffix raises a not-implemented error), then prints the
    # path, the row count and the first three rows.
    # KNOWN ISSUE: indexing the first three rows raises an index error for datasets with fewer than
    # three rows (after the file was already written).
    match path.suffix:
        case ".jsonl":
            ds.to_json(path)
        case ".parquet":
            ds.to_parquet(path)
        case _:
            raise NotImplementedError(f"{path=} {path.suffix=}")

    print(f"Write to {path}, {len(ds)=}, example data:")
    pprint.pprint([ds[i] for i in range(3)])


def _convert_to_by_sorry(s: str):
    # [FUNCTION DOC] HELPER: Replaces the trailing proof of a Lean statement by a fixed placeholder.
    # Removes (asserting at least one match) every line-end occurrence of optional spaces, the
    # definition marker, optional by keyword and optional sorry keyword, then appends the definition
    # marker, by keyword, newline and an indented sorry.
    return _ensure_remove_pattern(s, r" *:=\s*(?:by\s*)?(?:sorry\s*)?$") + " := by\n  sorry"


def _ensure_remove_pattern(text: str, pattern: str):
    # [FUNCTION DOC] HELPER: Asserts the multi-line regular expression matches somewhere in the
    # text and returns the text with all matches deleted.
    assert re.search(pattern, text, flags=re.MULTILINE), f"{pattern=} {text=}"
    return re.sub(pattern, "", text, flags=re.MULTILINE)


def _to_messages(content):
    # [FUNCTION DOC] HELPER: Wraps text into a one-element chat message list with the user role.
    return [{"role": "user", "content": content}]


def _add_metadata_column(ds, dataset_name: str, column_id=None):
    # [FUNCTION DOC] HELPER: Adds a metadata column of mappings, each with a question id. With a
    # source id column name, the question id is the dataset name, two underscores and the row's
    # value in that column; otherwise it is the dataset name, two underscores, the letters idx and
    # the row index. Returns the extended dataset.
    if column_id is not None:
        col_metadata = [dict(question_id=f"{dataset_name}__{value}") for value in ds[column_id]]
    else:
        col_metadata = [dict(question_id=f"{dataset_name}__idx{i}") for i in range(len(ds))]
    return ds.add_column("metadata", col_metadata)


def main(
    mode: Annotated[str, typer.Option()] = "rl",
    dir_output_base: Annotated[str, typer.Option()] = "/root/datasets/formal_math_single_round/",
    output_name: Annotated[str, typer.Option()] = None,
    train_flc_select_num_rows: Annotated[int, typer.Option()] = 20000,
    val_flc_select_num_rows: Annotated[int, typer.Option()] = 100,
    filter_difficulty: Annotated[int | None, typer.Option()] = None,
    filter_solvable_by_rollout_dumps: Annotated[str | None, typer.Option()] = None,
):
    # [FUNCTION DOC] Typer command. Options: mode (rl default, or sft), base output directory
    # (default under the shared datasets directory), optional output sub-directory name (default a
    # timestamp plus random number), number of FineLeanCorpus train rows (default 20000) and
    # validation rows (default 100), optional difficulty filter and optional comma-separated rollout
    # dump filter. Creates the output directory (with parents), prints it, then runs the RL
    # processing (FineLeanCorpus then miniF2F) or the SFT processing; any other mode raises a
    # not-implemented error. The row-count, difficulty and dump-filter options are ignored in sft
    # mode.
    dir_output = Path(dir_output_base) / (
        output_name or f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 1000000)}"
    )
    dir_output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {dir_output}")

    if mode == "rl":
        process_flc(
            dir_output=dir_output,
            train_flc_select_num_rows=train_flc_select_num_rows,
            val_flc_select_num_rows=val_flc_select_num_rows,
            filter_difficulty=filter_difficulty,
            filter_solvable_by_rollout_dumps=filter_solvable_by_rollout_dumps,
        )
        process_minif2f(
            dir_output=dir_output,
        )
    elif mode == "sft":
        process_leanabell(
            dir_output=dir_output,
        )
    else:
        raise NotImplementedError(f"{mode=}")


if __name__ == "__main__":
    typer.run(main)
