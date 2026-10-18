# [MODULE DOC] CPU tests for data logging in true-on-policy mode: the rollout log-prob dtype follows
# training precision (bf16 or fp16) only in true-on-policy mode, else fp32; the CI log-prob checker
# passes when trainer and rollout log-probs match in values and dtype; and sampling-mask transport
# fields (ids, offsets) are excluded from the metrics sent to the gather.
from argparse import Namespace
from types import SimpleNamespace

import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.metrics import log_utils


def test_true_on_policy_rollout_logprob_dtype_follows_training_precision():
    # [FUNCTION DOC] Test: the rollout log-prob dtype is bf16 for true-on-policy with bf16, fp16 for
    # true-on-policy with fp16, and fp32 when not true-on-policy (even with bf16).
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=True, fp16=False)) is torch.bfloat16
    )
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=False, fp16=True)) is torch.float16
    )
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=False, bf16=True, fp16=False)) is torch.float32
    )


def test_true_on_policy_log_checker_passes_when_values_and_dtype_match(monkeypatch):
    # [FUNCTION DOC] Test: with a fake single-rank, last-stage parallel state patched into the log
    # utilities and context-parallel module, and the gather function replaced by one that captures
    # the log dictionary, logging rollout data (one sample, bf16 log-probs equal to bf16 rollout
    # log-probs [-13.25, -13.5]) under CI-test args with the log-prob checker enabled and
    # true-on-policy mode does not raise, and the captured logged log-probs equal the logged
    # rollout log-probs.
    captured = {}
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(size=1),
        is_pp_last_stage=True,
    )

    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        log_utils,
        "gather_log_data",
        lambda metric_name, args, rollout_id, log_dict: captured.setdefault("log_dict", log_dict),
    )

    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "log_probs": [torch.tensor([-13.25, -13.5], dtype=torch.bfloat16)],
        "rollout_log_probs": [torch.tensor([-13.25, -13.5], dtype=torch.bfloat16)],
    }

    log_utils.log_rollout_data(
        1,
        Namespace(
            ci_test=True,
            ci_disable_logprobs_checker=False,
            trainer_model_id=None,
            true_on_policy_mode=True,
            qkv_format="thd",
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        rollout_data,
    )

    assert captured["log_dict"]["log_probs"] == captured["log_dict"]["rollout_log_probs"]


def test_sampling_mask_csr_payload_is_not_averaged_as_a_metric(monkeypatch):
    # [FUNCTION DOC] Test: rollout data containing sampling-mask ids and offsets (compressed sparse
    # row payload) logged under non-CI args: the dictionary captured by the patched gather contains
    # neither key.
    captured = {}
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(size=1),
        is_pp_last_stage=True,
    )
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        log_utils,
        "gather_log_data",
        lambda _metric_name, _args, _rollout_id, log_dict: captured.setdefault("log_dict", log_dict),
    )
    rollout_data = {
        "tokens": [torch.tensor([1, 2])],
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [torch.tensor([1], dtype=torch.int32)],
        "rollout_sampling_mask_ids": [[1, 7]],
        "rollout_sampling_mask_offsets": [[0, 2]],
    }

    log_utils.log_rollout_data(
        1,
        SimpleNamespace(
            ci_test=False,
            ci_disable_logprobs_checker=True,
            qkv_format="thd",
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        rollout_data,
    )

    assert "rollout_sampling_mask_ids" not in captured["log_dict"]
    assert "rollout_sampling_mask_offsets" not in captured["log_dict"]
