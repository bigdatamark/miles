# [MODULE DOC]
# Performance metric logging: turns the global Timer's recorded durations into tracker metrics,
# including TFLOPs, tokens per second, MFU and train wait ratios.
import logging
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy

from miles.utils.device_flops import local_peak_bf16_tflops
from miles.utils.metric_utils import compute_rollout_step, namespace_metrics
from miles.utils.timer import Timer
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


def log_perf_data_raw(
    rollout_id: int,
    args: Namespace,
    is_primary_rank: bool,
    compute_total_fwd_flops: Callable,
    extra_metrics: dict | None = None,
) -> None:
    # [FUNCTION DOC] Snapshot (deep copy) and reset the global Timer on every rank (so timers do
    # not accumulate), then, on the primary rank only: build metrics "perf/<name>_time" from the
    # timer; merge extra metrics; if the actor train time is present and a flops callable is given,
    # compute total forward flops from the timer's sequence lengths (passed as keyword), and add
    # log-probs and ref-log-probs TFLOPs (forward flops / time), and, when actor train time > 0,
    # actor train TFLOPs (3x forward flops / time), tokens per second (sum of sequence lengths /
    # time), plus peak TFLOPs (args override or detected device peak bf16) and MFU = achieved /
    # peak, when a peak is known. If both train wait and train times exist and their sum is
    # positive, adds step time and wait ratio. Logs an info line and sends namespaced metrics with
    # the rollout-based step to the tracker.
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}
    if extra_metrics:
        log_dict.update(extra_metrics)

    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]

            peak_tflops = getattr(args, "mfu_peak_tflops", None) or local_peak_bf16_tflops()
            if peak_tflops:
                log_dict["perf/mfu_peak_tflops"] = peak_tflops
                log_dict["perf/actor_train_mfu"] = log_dict["perf/actor_train_tflops"] / peak_tflops

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    log_dict, step_key = namespace_metrics(
        log_dict,
        trainer_model_id=args.trainer_model_id,
        step_name="rollout/step",
        step=compute_rollout_step(args, rollout_id),
    )
    tracking.log(args, log_dict, step_key=step_key)
