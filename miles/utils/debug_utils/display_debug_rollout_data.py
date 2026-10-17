# [MODULE DOC] Command-line tool that prints dumped debug rollout data. Scans numbered dump
# files (a path template with a rollout id placeholder), loads each with the torch loader and
# prints performance metrics computed from the samples and/or a whitelisted subset of each
# sample's fields as one json line. Run as a module with the load-debug-rollout-data option.
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import torch
import typer

from miles.ray.rollout import compute_perf_metrics_from_samples
from miles.utils.types import Sample

_WHITELIST_KEYS = [
    "group_index",
    "index",
    "prompt",
    "response",
    "response_length",
    "label",
    "reward",
    "status",
    "metadata",
]


def main(
    # Deliberately make this name consistent with main training arguments
    load_debug_rollout_data: Annotated[str, typer.Option()],
    show_metrics: bool = True,
    show_samples: bool = True,
    category: list[str] = None,
):
    # [FUNCTION DOC] Entry point. Inputs: path template for dumps (named like the training
    # argument), flags to show metrics and/or samples (both default on), and the categories to
    # scan (default train and eval). For every existing dump found, prints a separator banner with
    # the rollout id and path, loads the file (a mapping with a samples list), and: if metrics are
    # on, rebuilds Sample objects, computes performance metrics using a fixed fake argument set
    # (grpo advantage estimator, no reward key, no reward category logging) and prints them; if
    # samples are on, prints each sample restricted to the whitelisted keys as a json line.
    # TODO in code: the metric configuration should be read from the dump rather than fixed.
    if category is None:
        category = ["train", "eval"]
    for rollout_id, path in _get_rollout_dump_paths(load_debug_rollout_data, category):
        print("-" * 80)
        print(f"{rollout_id=} {path=}")
        print("-" * 80)

        pack = torch.load(path)
        sample_dicts = pack["samples"]

        if show_metrics:
            # TODO read these configs from dumps
            args = SimpleNamespace(
                advantage_estimator="grpo",
                reward_key=None,
                log_reward_category=None,
            )
            sample_objects = [Sample.from_dict(s) for s in sample_dicts]
            metrics = compute_perf_metrics_from_samples(args, sample_objects)
            print("metrics", metrics)

        if show_samples:
            for sample in sample_dicts:
                print(json.dumps({k: v for k, v in sample.items() if k in _WHITELIST_KEYS}))


def _get_rollout_dump_paths(load_debug_rollout_data: str, categories: list[str]):
    # [FUNCTION DOC] HELPER: Generator probing rollout ids 0 to 999 in order and, for each, every
    # requested category in the given order. The path is the template formatted with the rollout
    # id preceded by a prefix (empty for train, "eval_" for eval). Yields (rollout id, path) for
    # paths that exist. An unknown category raises a key error. Dumps with id 1000 or above are
    # never found (hard limit).
    # may improve later
    for rollout_id in range(1000):
        for category in categories:
            prefix = {
                "train": "",
                "eval": "eval_",
            }[category]
            path = Path(load_debug_rollout_data.format(rollout_id=f"{prefix}{rollout_id}"))
            if path.exists():
                yield rollout_id, path


if __name__ == "__main__":
    """python -m miles.utils.debug_utils.display_debug_rollout_data --load-debug-rollout-data ..."""
    typer.run(main)
