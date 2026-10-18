# [MODULE DOC]
# HELPER: Shared execution and verification helpers for the multi-policy (solver and verifier)
# GSM8K end-to-end tests. The execute function launches a training run through the example
# launcher and then verifies the run's audit event logs: every trainer rank reported its
# environment with the per-policy argument overrides it was built with, every policy logged at
# least one held-out evaluation point, and (optionally) evaluation scores rose within bounds.
# The module-level mapping fixes how many override arguments each policy (solver 25, verifier
# 26) must have, so a shrinking verification fails loudly.
from pathlib import Path
from typing import NamedTuple

from examples.multi_policy.run_solver_verifier_gsm8k import (
    EVAL_DATASET_NAME,
    SOLVER_MODEL_ID,
    VERIFIER_MODEL_ID,
    ScriptArgs,
    build_train_args,
    compute_events_dir,
    compute_megatron_config,
    launch_train,
)

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import EnvReportEvent, MetricEvent
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


class EvalScoreBounds(NamedTuple):
    # [CLASS DOC] HELPER: Learning bounds for one policy's evaluation score series. Fields: the
    # highest allowed first score (to prove the run did not start already solved), the lowest
    # required peak score, and an optional minimum growth from first to peak score (not checked
    # when absent).
    initial_max: float
    peak_min: float
    min_growth: float | None = None


NUM_VERIFIED_ARGS_PER_POLICY = {SOLVER_MODEL_ID: 25, VERIFIER_MODEL_ID: 26}


def execute(
    args: ScriptArgs,
    *,
    wandb_args: str,
    eval_score_bounds: dict[str, EvalScoreBounds] | None = None,
) -> None:
    # [FUNCTION DOC] Run the multi-policy training and verify it. Steps: compute the audit events
    # directory and the Megatron configuration (list of trainers with model ids and overrides) from
    # the script arguments; build the training arguments (with the given tracking arguments) and
    # launch the training (side effect: a long GPU job); then assert that every trainer rank
    # trained with its policy's arguments (expected rank count = actor GPUs per policy), that each
    # trainer's model id reported at least one evaluation point on the evaluation dataset, and,
    # only if evaluation score bounds are given, that scores satisfy them. Raises assertion errors.
    events_dir = compute_events_dir(args)
    megatron_config = compute_megatron_config(args)

    launch_train(build_train_args(args, wandb_args=wandb_args, megatron_config=megatron_config), args)

    assert_ranks_trained_with_policy_args(
        events_dir, megatron_config=megatron_config, expected_num_ranks=args.actor_num_gpus_per_policy
    )
    assert_policies_reported_eval_points(
        events_dir,
        model_ids=[trainer["model_id"] for trainer in megatron_config["trainers"]],
        dataset_name=EVAL_DATASET_NAME,
    )
    if eval_score_bounds is not None:
        assert_policy_eval_scores_learned(events_dir, bounds=eval_score_bounds, dataset_name=EVAL_DATASET_NAME)


def assert_ranks_trained_with_policy_args(events_dir: Path, *, megatron_config: dict, expected_num_ranks: int) -> None:
    # [FUNCTION DOC] Verify from the audit event logs that each policy's trainer ranks were built
    # with the intended arguments. Reads all events under the directory and groups environment
    # report events whose source is a training process by model id. Asserts: (1) the set of model
    # ids reported equals the set of trainer model ids in the configuration (a policy with no
    # reports was never trained); (2) for every policy, the sorted distinct (cell index, rank
    # within cell) pairs number exactly the expected rank count, and equal the union of ranks seen
    # over all policies; (3) for every trainer, the number of its override arguments equals the
    # fixed expected count for its model id (the claimed verification size must not shrink);
    # (4) for every report of that policy, the reported trainer model id equals the policy's, all
    # override keys are present in the reported argument values, and their values equal the
    # overrides. Raises assertion errors with descriptive messages.
    reports_by_model_id: dict[str, list[EnvReportEvent]] = {}
    for event in read_events(events_dir):
        if isinstance(event, EnvReportEvent) and isinstance(event.source, TrainProcessIdentity):
            reports_by_model_id.setdefault(event.source.model_id, []).append(event)

    expected_model_ids = sorted(trainer["model_id"] for trainer in megatron_config["trainers"])
    assert sorted(reports_by_model_id) == expected_model_ids, (
        f"the env reports under {events_dir} come from trainer ranks of {sorted(reports_by_model_id)}, but this "
        f"run trains {expected_model_ids}; a policy whose ranks reported nothing was never actually trained"
    )

    ranks_by_model_id: dict[str, list[tuple[int, int]]] = {
        model_id: sorted({(event.source.cell_index, event.source.rank_within_cell) for event in reports})
        for model_id, reports in reports_by_model_id.items()
    }
    every_rank = sorted({rank for ranks in ranks_by_model_id.values() for rank in ranks})
    for model_id, ranks in ranks_by_model_id.items():
        assert len(ranks) == expected_num_ranks, (
            f"policy {model_id!r} is reported from ranks {ranks}, while this run trains each policy on "
            f"{expected_num_ranks} rank(s); a rank that reported nothing leaves the arguments it was built with "
            f"unverified"
        )
        assert ranks == every_rank, (
            f"policy {model_id!r} is reported from ranks {ranks}, while this run's trainer ranks are {every_rank}; "
            f"a rank that reported nothing leaves the arguments it was built with unverified"
        )

    for trainer in megatron_config["trainers"]:
        model_id = trainer["model_id"]
        expected = dict(trainer["overrides"])
        assert len(expected) == NUM_VERIFIED_ARGS_PER_POLICY[model_id], (
            f"policy {model_id!r} overrides {sorted(expected)}, but this test claims to verify "
            f"{NUM_VERIFIED_ARGS_PER_POLICY[model_id]} arguments; a verification that quietly shrank proves nothing"
        )
        for report in reports_by_model_id[model_id]:
            values = report.report.process.args.values
            assert values["trainer_model_id"] == model_id, (
                f"rank {report.source.to_name()} reports trainer_model_id {values['trainer_model_id']!r} while its "
                f"process identity says {model_id!r}"
            )
            actual = {key: values[key] for key in expected}
            assert len(actual) == len(expected), f"{sorted(set(expected) - set(actual))} never reached the report"
            assert actual == expected, (
                f"rank {report.source.to_name()} of policy {model_id!r} was built with {actual}, but its "
                f"--megatron-config overrides prescribe {expected}"
            )


def assert_policies_reported_eval_points(events_dir: Path, *, model_ids: list[str], dataset_name: str) -> None:
    # [FUNCTION DOC] For each model id, collect from the event logs under the directory the
    # values of the metric named "eval/<dataset>/<model id>" across all metric events (re-reading
    # the events for each model), and assert at least one point exists; otherwise fail saying held-
    # out evaluation never ran for that policy.
    for model_id in model_ids:
        eval_key = f"eval/{dataset_name}/{model_id}"
        points = [
            event.metrics[eval_key]
            for event in read_events(events_dir)
            if isinstance(event, MetricEvent) and eval_key in event.metrics
        ]
        assert points, (
            f"policy {model_id!r} logged no {eval_key} point under {events_dir}, but every run evaluates at least "
            f"once, so held-out eval never actually ran for this policy"
        )


def assert_policy_eval_scores_learned(
    events_dir: Path, *, bounds: dict[str, EvalScoreBounds], dataset_name: str
) -> None:
    # [FUNCTION DOC] For each model id in the bounds mapping, read its evaluation score series
    # (ordered by weight version) and assert: the series is non-empty; the first score is at most
    # the initial maximum; the peak (maximum) score is at least the peak minimum; and, if a
    # minimum growth is set, peak minus first is at least that growth. Raises assertion errors.
    for model_id, model_bounds in bounds.items():
        scores = _read_eval_score_series(events_dir, model_id=model_id, dataset_name=dataset_name)
        assert scores, f"policy {model_id!r}: no eval/{dataset_name}/{model_id} points under {events_dir}"

        first = scores[0]
        peak = max(scores)
        assert (
            first <= model_bounds.initial_max
        ), f"policy {model_id!r}: first eval score {first} > {model_bounds.initial_max} (starts already solved)"
        assert peak >= model_bounds.peak_min, f"policy {model_id!r}: peak eval score {peak} < {model_bounds.peak_min}"
        if model_bounds.min_growth is not None:
            assert peak - first >= model_bounds.min_growth, (
                f"policy {model_id!r}: eval growth {peak - first} < {model_bounds.min_growth} "
                f"(first {first}, peak {peak})"
            )


def _read_eval_score_series(events_dir: Path, *, model_id: str, dataset_name: str) -> list[float]:
    # [FUNCTION DOC] HELPER: Return the evaluation scores of one policy, in training order. From
    # every metric event under the directory that has the metric "eval/<dataset>/<model id>", pair
    # the maximum weight version metric for it (default 0 if absent) with the score, sort the
    # pairs (by version, then score) and return just the scores.
    eval_key = f"eval/{dataset_name}/{model_id}"
    version_key = f"{eval_key}/weight_version/max"
    points = [
        (event.metrics.get(version_key, 0), event.metrics[eval_key])
        for event in read_events(events_dir)
        if isinstance(event, MetricEvent) and eval_key in event.metrics
    ]
    return [score for _, score in sorted(points)]
