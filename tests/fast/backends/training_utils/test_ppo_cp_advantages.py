# [MODULE DOC] Two-process CPU (Gloo) tests that PPO advantage and return computation under
# context parallelism of 2 matches the single-rank baseline: terminal reward is added to the tail of
# the global response (even when the tail or rank 0 shard is empty), masked generalized advantage
# estimation with trainable-token gaps matches the baseline for two gamma/lambda settings (and the
# gamma 0 case matches hand-computed values), padded (bshd) layouts use the padded layout metadata
# for slicing and gathering, and with the actor forward-only pass skipped the synthesized zero KL
# gives identical grpo advantages and returns. Each worker initializes Gloo and always destroys the
# process group. Each test launches workers through the multi-process runner.
from argparse import Namespace

import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.data.context_parallel import all_gather_with_cp, slice_log_prob_with_cp
from miles.backends.training_utils.loss.hub.advantages import compute_advantages
from miles.backends.training_utils.loss.objective import compute_advantages_and_returns
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _parallel_state(rank: int = 0, world_size: int = 1) -> ParallelState:
    # [FUNCTION DOC] HELPER: Builds a parallel state with trivial groups except context parallel and
    # intra-dp-with-cp which use the given rank and world size (default rank 0, size 1); the world
    # process group is attached only when world size exceeds 1. Not installed globally.
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    cp_group = dist.group.WORLD if world_size > 1 else None
    return ParallelState(
        intra_dp=trivial_group,
        intra_dp_cp=GroupInfo(rank=rank, size=world_size, group=cp_group),
        cp=GroupInfo(rank=rank, size=world_size, group=cp_group),
        tp=trivial_group,
        pp=trivial_group,
        ep=trivial_group,
        etp=trivial_group,
        indep_dp=trivial_group,
    )


def _run_ppo_case(rank: int, total_length: int, response_length: int, expected_local_sizes: list[int]) -> None:
    # [FUNCTION DOC] HELPER: Per-rank body for the terminal-reward cases. Args: ppo, KL coefficient
    # 0.1, gamma 0, lambda 0, packed layout. KL is 1..response_length, values are zeros, reward 10.
    # Installs a two-rank state, slices KL and values to this rank (asserting the local KL size
    # equals the expected size for the rank), computes advantages and returns on the local shard,
    # gathers them across context-parallel ranks, then installs a one-rank state and computes the
    # baseline on full tensors. Asserts gathered advantages and returns equal -0.1 times KL with 10
    # added at the last element, and equal the baseline.
    args = Namespace(advantage_estimator="ppo", kl_coef=0.1, gamma=0.0, lambd=0.0, qkv_format="thd")
    full_kl = torch.arange(1, response_length + 1, dtype=torch.float32)
    full_values = torch.zeros(response_length)

    set_parallel_state(_parallel_state(rank=rank, world_size=2))
    local_kl = slice_log_prob_with_cp(full_kl, total_length, response_length)
    local_values = slice_log_prob_with_cp(full_values, total_length, response_length)
    assert local_kl.numel() == expected_local_sizes[rank]

    advantages, returns = compute_advantages(
        args=args,
        kl=[local_kl],
        rewards=[10.0],
        log_probs=[torch.empty_like(local_kl)],
        loss_masks=[torch.ones(response_length)],
        total_lengths=[total_length],
        response_lengths=[response_length],
        max_seq_lens=None,
        values=[local_values],
    )
    cp_advantages = all_gather_with_cp(advantages[0], total_length, response_length)
    cp_returns = all_gather_with_cp(returns[0], total_length, response_length)

    set_parallel_state(_parallel_state())
    baseline_advantages, baseline_returns = compute_advantages(
        args=args,
        kl=[full_kl.clone()],
        rewards=[10.0],
        log_probs=[torch.empty_like(full_kl)],
        loss_masks=[torch.ones(response_length)],
        total_lengths=[total_length],
        response_lengths=[response_length],
        max_seq_lens=None,
        values=[full_values],
    )

    expected = -0.1 * full_kl
    expected[-1] += 10.0
    torch.testing.assert_close(cp_advantages, expected)
    torch.testing.assert_close(cp_returns, expected)
    torch.testing.assert_close(cp_advantages, baseline_advantages[0])
    torch.testing.assert_close(cp_returns, baseline_returns[0])


def _run_ppo_masked_case(rank: int) -> None:
    # [FUNCTION DOC] HELPER: Per-rank body for the masked case: total length 7, response length 6,
    # loss mask [1,1,0,0,1,0], values [0.5,-0.3,0.7,0.1,-0.2,0.4], KL 1..6, reward 10. For (gamma,
    # lambda) in (0,0) and (0.9,0.8): computes advantages/returns on this rank's shard under a
    # two-rank state, gathers, and compares with a one-rank baseline on full tensors (both close);
    # asserts masked positions have exactly zero advantage and return. For gamma = lambda = 0,
    # additionally asserts hand-computed advantages: trainable positions 0, 1, 4 equal
    # -0.1 * KL - value, with the terminal reward 10 added at position 4 (the last trainable
    # token), zeros elsewhere.
    total_length, response_length = 7, 6
    loss_mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    for gamma, lambd in [(0.0, 0.0), (0.9, 0.8)]:
        args = Namespace(advantage_estimator="ppo", kl_coef=0.1, gamma=gamma, lambd=lambd, qkv_format="thd")
        full_kl = torch.arange(1, response_length + 1, dtype=torch.float32)
        full_values = torch.tensor([0.5, -0.3, 0.7, 0.1, -0.2, 0.4])

        set_parallel_state(_parallel_state(rank=rank, world_size=2))
        local_kl = slice_log_prob_with_cp(full_kl, total_length, response_length).clone()
        local_values = slice_log_prob_with_cp(full_values, total_length, response_length).clone()

        advantages, returns = compute_advantages(
            args=args,
            kl=[local_kl],
            rewards=[10.0],
            log_probs=[torch.empty_like(local_kl)],
            loss_masks=[loss_mask.clone()],
            total_lengths=[total_length],
            response_lengths=[response_length],
            values=[local_values],
        )
        cp_advantages = all_gather_with_cp(advantages[0], total_length, response_length)
        cp_returns = all_gather_with_cp(returns[0], total_length, response_length)

        set_parallel_state(_parallel_state())
        baseline_advantages, baseline_returns = compute_advantages(
            args=args,
            kl=[full_kl.clone()],
            rewards=[10.0],
            log_probs=[torch.empty_like(full_kl)],
            loss_masks=[loss_mask.clone()],
            total_lengths=[total_length],
            response_lengths=[response_length],
            values=[full_values.clone()],
        )

        torch.testing.assert_close(cp_advantages, baseline_advantages[0])
        torch.testing.assert_close(cp_returns, baseline_returns[0])
        assert torch.all(cp_advantages[loss_mask == 0] == 0)
        assert torch.all(cp_returns[loss_mask == 0] == 0)

        if gamma == 0.0 and lambd == 0.0:
            # Terminal reward lands on the last trainable token (index 4), and
            # with gamma = 0 each trainable advantage is reward - value.
            expected = torch.zeros(response_length)
            expected[0] = -0.1 * 1.0 - 0.5
            expected[1] = -0.1 * 2.0 - (-0.3)
            expected[4] = -0.1 * 5.0 + 10.0 - (-0.2)
            torch.testing.assert_close(cp_advantages, expected)


def _worker_tail_on_rank_one(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Worker; init Gloo, run the terminal-reward case with total 7, response
    # 6 (rank 0 holds 2 response entries, rank 1 holds 4, so the tail is on rank 1), destroy group.
    init_gloo(rank, world_size, port=port)
    try:
        _run_ppo_case(rank, total_length=7, response_length=6, expected_local_sizes=[2, 4])
    finally:
        dist.destroy_process_group()


def _worker_empty_rank_zero(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Worker; init Gloo, run the terminal-reward case with total 7, response
    # 2 (rank 0 holds 0 response entries, rank 1 holds 2), destroy group.
    init_gloo(rank, world_size, port=port)
    try:
        _run_ppo_case(rank, total_length=7, response_length=2, expected_local_sizes=[0, 2])
    finally:
        dist.destroy_process_group()


def _worker_bshd_layout_metadata(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Worker for the padded-layout case. Init Gloo; ppo args with padded
    # layout, KL coefficient 0.1, gamma/lambda 0. Two samples: total lengths 8 and 11, response
    # lengths 5 and 6, padded max length 12 each, rewards 10 and 20, log-probs 1..n, zero reference
    # log-probs and values. Slices log-probs per sample with padded-layout metadata (asserts local
    # sizes [1,1] for rank 0 and [4,5] for rank 1), runs the advantage-and-returns routine on
    # rollout data, gathers advantages and returns per sample with the same metadata, then runs the
    # routine on full tensors under a one-rank state and asserts each gathered advantage and return
    # is close to the baseline. Destroys the process group at the end.
    init_gloo(rank, world_size, port=port)
    try:
        args = Namespace(
            advantage_estimator="ppo",
            use_rollout_logprobs=False,
            skip_actor_forward_only=False,
            kl_coef=0.1,
            kl_loss_type="k1",
            gamma=0.0,
            lambd=0.0,
            qkv_format="bshd",
            use_opd=False,
            normalize_advantages=False,
        )
        total_lengths = [8, 11]
        response_lengths = [5, 6]
        max_seq_lens = [12, 12]
        rewards = [10.0, 20.0]
        full_log_probs = [
            torch.arange(1, response_length + 1, dtype=torch.float32) for response_length in response_lengths
        ]
        full_ref_log_probs = [torch.zeros_like(log_probs) for log_probs in full_log_probs]
        full_values = [torch.zeros_like(log_probs) for log_probs in full_log_probs]

        set_parallel_state(_parallel_state(rank=rank, world_size=world_size))
        local_log_probs = [
            slice_log_prob_with_cp(log_probs, total_length, response_length, "bshd", max_seq_len)
            for log_probs, total_length, response_length, max_seq_len in zip(
                full_log_probs, total_lengths, response_lengths, max_seq_lens, strict=True
            )
        ]
        local_ref_log_probs = [torch.zeros_like(log_probs) for log_probs in local_log_probs]
        local_values = [torch.zeros_like(log_probs) for log_probs in local_log_probs]
        expected_local_sizes = [[1, 1], [4, 5]][rank]
        assert [tensor.numel() for tensor in local_log_probs] == expected_local_sizes

        rollout_data = {
            "log_probs": local_log_probs,
            "ref_log_probs": local_ref_log_probs,
            "rewards": rewards,
            "values": local_values,
            "response_lengths": response_lengths,
            "loss_masks": [torch.ones(response_length) for response_length in response_lengths],
            "total_lengths": total_lengths,
            "max_seq_lens": max_seq_lens,
        }
        compute_advantages_and_returns(args, rollout_data)
        cp_advantages = [
            all_gather_with_cp(advantage, total_length, response_length, "bshd", max_seq_len)
            for advantage, total_length, response_length, max_seq_len in zip(
                rollout_data["advantages"], total_lengths, response_lengths, max_seq_lens, strict=True
            )
        ]
        cp_returns = [
            all_gather_with_cp(ret, total_length, response_length, "bshd", max_seq_len)
            for ret, total_length, response_length, max_seq_len in zip(
                rollout_data["returns"], total_lengths, response_lengths, max_seq_lens, strict=True
            )
        ]

        set_parallel_state(_parallel_state())
        baseline_data = {
            "log_probs": [tensor.clone() for tensor in full_log_probs],
            "ref_log_probs": [tensor.clone() for tensor in full_ref_log_probs],
            "rewards": rewards,
            "values": [tensor.clone() for tensor in full_values],
            "response_lengths": response_lengths,
            "loss_masks": [torch.ones(response_length) for response_length in response_lengths],
            "total_lengths": total_lengths,
            "max_seq_lens": max_seq_lens,
        }
        compute_advantages_and_returns(args, baseline_data)

        for cp_advantage, cp_return, baseline_advantage, baseline_return in zip(
            cp_advantages,
            cp_returns,
            baseline_data["advantages"],
            baseline_data["returns"],
            strict=True,
        ):
            torch.testing.assert_close(cp_advantage, baseline_advantage)
            torch.testing.assert_close(cp_return, baseline_return)
    finally:
        dist.destroy_process_group()


def _worker_masked_case(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Worker; init Gloo, run the masked case for this rank, destroy group.
    init_gloo(rank, world_size, port=port)
    try:
        _run_ppo_masked_case(rank)
    finally:
        dist.destroy_process_group()


def test_ppo_terminal_reward_is_added_to_global_response_tail() -> None:
    # [FUNCTION DOC] Test: runs the tail-on-rank-one worker across processes.
    run_multiprocess(_worker_tail_on_rank_one)


def test_ppo_terminal_reward_handles_empty_rank_zero_shard() -> None:
    # [FUNCTION DOC] Test: runs the empty-rank-zero worker across processes.
    run_multiprocess(_worker_empty_rank_zero)


def test_ppo_bshd_cp_uses_padded_layout_metadata() -> None:
    # [FUNCTION DOC] Test: runs the padded-layout worker across processes.
    run_multiprocess(_worker_bshd_layout_metadata)


def test_ppo_masked_gae_matches_single_rank_baseline() -> None:
    # [FUNCTION DOC] Test: runs the masked-case worker across processes.
    run_multiprocess(_worker_masked_case)


def _worker_reused_zero_kl(rank: int, world_size: int, port: int) -> None:
    # [FUNCTION DOC] HELPER: Worker for the skipped-forward case. Init Gloo; grpo args with the
    # skip-actor-forward-only flag on, KL coefficient 0, gamma 1, lambda 0.95. Rollout data has one
    # sample (total 7, response 6, reward 3.0, all-ones mask) and no log-probs. Under a two-rank
    # state, computes advantages and returns (the routine synthesizes zero KL), gathers them across
    # context-parallel ranks. Then under a one-rank state, builds baseline data with zero log-probs,
    # no stored advantages/returns, skip flag off, recomputes, and asserts gathered results are
    # close to the baseline. Destroys the process group at the end.
    init_gloo(rank, world_size, port=port)
    try:
        args = Namespace(
            advantage_estimator="grpo",
            use_rollout_logprobs=False,
            skip_actor_forward_only=True,
            kl_coef=0.0,
            kl_loss_type="k1",
            gamma=1.0,
            lambd=0.95,
            qkv_format="thd",
            use_opd=False,
            normalize_advantages=False,
        )
        total_length, response_length = 7, 6
        rollout_data = {
            "rewards": [3.0],
            "response_lengths": [response_length],
            "loss_masks": [torch.ones(response_length)],
            "total_lengths": [total_length],
        }

        set_parallel_state(_parallel_state(rank=rank, world_size=world_size))
        compute_advantages_and_returns(args, rollout_data)
        cp_advantages = all_gather_with_cp(rollout_data["advantages"][0], total_length, response_length)
        cp_returns = all_gather_with_cp(rollout_data["returns"][0], total_length, response_length)

        set_parallel_state(_parallel_state())
        baseline = {
            **rollout_data,
            "log_probs": [torch.zeros(response_length)],
        }
        baseline.pop("advantages")
        baseline.pop("returns")
        args.skip_actor_forward_only = False
        compute_advantages_and_returns(args, baseline)

        torch.testing.assert_close(cp_advantages, baseline["advantages"][0])
        torch.testing.assert_close(cp_returns, baseline["returns"][0])
    finally:
        dist.destroy_process_group()


def test_reused_zero_kl_matches_single_rank_baseline_with_context_parallelism() -> None:
    # [FUNCTION DOC] Test: runs the skipped-forward zero-KL worker across processes.
    run_multiprocess(_worker_reused_zero_kl)
