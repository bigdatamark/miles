# [MODULE DOC] CPU tests for the pipeline-stage guard in the advantage computation: the PPO
# estimator runs only on the last pipeline stage (decided from the parallel state, not from which
# tensors happen to be present), so intermediate stages return early without writing advantages or
# returns, and the last stage computes them per sample.
import torch

from miles.backends.training_utils.loss.objective import compute_advantages_and_returns

from .loss_test_utils import make_args, make_parallel_state


def _ppo_args(use_rollout_logprobs: bool):
    # [FUNCTION DOC] HELPER: Returns default test args with the ppo advantage estimator, the given
    # rollout-log-probs flag, zero KL coefficient and lambda 1.0.
    return make_args(
        advantage_estimator="ppo",
        use_rollout_logprobs=use_rollout_logprobs,
        kl_coef=0.0,
        lambd=1.0,
    )


def _rollout_data(log_probs_key: str | None, with_values: bool) -> dict:
    # [FUNCTION DOC] HELPER: Builds rollout data for one sample with response length 4 (total 6):
    # zero reference log-probs, reward 1.0, all-ones loss mask. Optionally adds a zero log-probs
    # entry under the given key (skipped when the key is none) and zero values.
    response_length = 4
    data = {
        "ref_log_probs": [torch.zeros(response_length)],
        "rewards": [1.0],
        "response_lengths": [response_length],
        "loss_masks": [torch.ones(response_length)],
        "total_lengths": [response_length + 2],
    }
    if log_probs_key is not None:
        data[log_probs_key] = [torch.zeros(response_length)]
    if with_values:
        data["values"] = [torch.zeros(response_length)]
    return data


def test_intermediate_pp_stage_returns_early_with_rollout_log_probs() -> None:
    # [FUNCTION DOC] Test: on a non-last pipeline stage with rollout log-probs and no values, the
    # advantage computation returns early; neither advantages nor returns appear in the data.
    """Under --use-rollout-logprobs an intermediate stage has log-probs but no
    values; deciding the stage from the tensors runs the PPO estimator there
    and dereferences `values`."""
    make_parallel_state(is_pp_last_stage=False)
    rollout_data = _rollout_data("rollout_log_probs", with_values=False)

    compute_advantages_and_returns(_ppo_args(use_rollout_logprobs=True), rollout_data)

    assert "advantages" not in rollout_data
    assert "returns" not in rollout_data


def test_intermediate_pp_stage_returns_early_without_any_log_probs() -> None:
    # [FUNCTION DOC] Test: on a non-last stage with no log-probs of any kind, the computation returns
    # early and writes no advantages.
    make_parallel_state(is_pp_last_stage=False)
    rollout_data = _rollout_data(None, with_values=False)

    compute_advantages_and_returns(_ppo_args(use_rollout_logprobs=False), rollout_data)

    assert "advantages" not in rollout_data


def test_last_pp_stage_computes_advantages_with_rollout_log_probs() -> None:
    # [FUNCTION DOC] Test: on the last stage with rollout log-probs and values, one advantages and
    # one returns tensor are produced and the advantages tensor has shape (response length,) = 4.
    make_parallel_state(is_pp_last_stage=True)
    rollout_data = _rollout_data("rollout_log_probs", with_values=True)

    compute_advantages_and_returns(_ppo_args(use_rollout_logprobs=True), rollout_data)

    assert len(rollout_data["advantages"]) == 1
    assert len(rollout_data["returns"]) == 1
    assert rollout_data["advantages"][0].shape == (4,)
