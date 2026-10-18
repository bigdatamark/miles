# [MODULE DOC] CPU tests for masked generalized advantage estimation in the batch advantage
# routine, in both chunked and non-chunked variants: a masked-gap sequence matches a serial
# reference over the compressed subsequence of trainable tokens, the carry across a mask gap
# decays by exactly one gamma*lambda factor regardless of gap length, the terminal reward goes to
# the last trainable token (not the last response token), fully masked samples yield zeros, an
# all-ones mask matches the classic padded GAE, and truncation bootstraps with zero value. An
# autouse fixture installs a trivial parallel state.
import pytest
import torch

from miles.backends.training_utils.loss.hub.math_utils import get_advantages_and_returns_batch, vanilla_gae
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


@pytest.fixture(autouse=True)
def _trivial_parallel_state() -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture; installs a global parallel state with all groups
    # trivial (rank 0, size 1, no group) before each test. Never restored afterwards.
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial_group,
            intra_dp_cp=trivial_group,
            cp=trivial_group,
            tp=trivial_group,
            pp=trivial_group,
            ep=trivial_group,
            etp=trivial_group,
            indep_dp=trivial_group,
        )
    )


def _reference_masked_gae(
    values: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    terminal_reward: float,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Reference implementation. Selects trainable positions (nonzero mask);
    # if none, returns zero advantages and returns. Takes values and per-token rewards at those
    # positions, adds the terminal reward to the last one, then runs the backward recursion: delta
    # = reward + gamma * next value (zero after the last) - value; running estimate = delta +
    # gamma * lambda * previous estimate. Writes the estimates back at the trainable positions
    # (zeros elsewhere); returns are estimates plus values at those positions. Returns (advantages,
    # returns), both shaped like the values.
    """Serial GAE over the compressed subsequence of trainable tokens."""
    idx = mask.nonzero(as_tuple=True)[0]
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    if idx.numel() == 0:
        return advantages, returns

    v = values[idx]
    r = rewards[idx].clone()
    r[-1] += terminal_reward

    K = v.numel()
    compressed_adv = torch.zeros_like(v)
    lastgaelam = 0.0
    for t in reversed(range(K)):
        next_value = v[t + 1] if t < K - 1 else 0.0
        delta = r[t] + gamma * next_value - v[t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        compressed_adv[t] = lastgaelam

    advantages[idx] = compressed_adv
    returns[idx] = compressed_adv + v
    return advantages, returns


def _compute(
    values: list[torch.Tensor],
    rewards: list[torch.Tensor],
    terminal_rewards: list[float],
    loss_masks: list[torch.Tensor],
    gamma: float,
    lambd: float,
    chunked: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Calls the batch advantage routine with response lengths taken from the
    # values, total lengths one more than that, packed layout, no padded max lengths, and the given
    # values, rewards, terminal rewards, masks, gamma, lambda and chunked flag. Returns its
    # (advantages list, returns list).
    response_lengths = [v.numel() for v in values]
    return get_advantages_and_returns_batch(
        total_lengths=[length + 1 for length in response_lengths],
        response_lengths=response_lengths,
        values_list=values,
        rewards_list=rewards,
        terminal_rewards=terminal_rewards,
        qkv_format="thd",
        max_seq_lens=None,
        loss_masks=loss_masks,
        gamma=gamma,
        lambd=lambd,
        chunked=chunked,
    )


@pytest.mark.parametrize("chunked", [False, True])
def test_masked_gap_matches_compressed_reference(chunked: bool) -> None:
    # [FUNCTION DOC] Test (parametrized over chunked): a length-10 sequence with mask
    # [1,1,0,0,0,1,1,0,1,0], seeded random values and rewards, terminal reward 3, gamma 0.9,
    # lambda 0.95. Results are close to the serial reference, and masked positions are exactly zero
    # in both advantages and returns.
    torch.manual_seed(0)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    values = torch.randn(10)
    rewards = torch.randn(10)
    terminal_reward = 3.0
    gamma, lambd = 0.9, 0.95

    advantages, returns = _compute(
        [values.clone()], [rewards.clone()], [terminal_reward], [mask.clone()], gamma, lambd, chunked
    )
    expected_adv, expected_ret = _reference_masked_gae(values, rewards, mask, terminal_reward, gamma, lambd)

    torch.testing.assert_close(advantages[0], expected_adv)
    torch.testing.assert_close(returns[0], expected_ret)
    assert torch.all(advantages[0][mask == 0] == 0)
    assert torch.all(returns[0][mask == 0] == 0)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("gap_len", [1, 5])
def test_gae_carry_crosses_mask_gap_undecayed(chunked: bool, gap_len: int) -> None:
    # [FUNCTION DOC] Test (parametrized over chunked and gap length 1 or 5): gamma 0.9, lambda 0.5,
    # zero values, a reward of 2 at the last position, trainable tokens only at the two ends. The
    # last advantage is 2; the first is gamma*lambda*2; all gap positions are exactly zero.
    # Two trainable tokens separated by a masked gap: the carry from the second
    # to the first must decay by exactly one factor of gamma * lambd, no matter
    # how long the gap is.
    gamma, lambd = 0.9, 0.5
    length = 2 + gap_len
    mask = torch.zeros(length)
    mask[0] = 1.0
    mask[-1] = 1.0
    values = torch.zeros(length)
    rewards = torch.zeros(length)
    rewards[-1] = 2.0

    advantages, _ = _compute([values], [rewards], [0.0], [mask], gamma, lambd, chunked)

    torch.testing.assert_close(advantages[0][-1], torch.tensor(2.0))
    torch.testing.assert_close(advantages[0][0], torch.tensor(gamma * lambd * 2.0))
    assert torch.all(advantages[0][1:-1] == 0)


@pytest.mark.parametrize("chunked", [False, True])
def test_terminal_reward_on_last_trainable_token(chunked: bool) -> None:
    # [FUNCTION DOC] Test (parametrized over chunked): mask [1,1,1,0,0] with gamma 0, lambda 0,
    # terminal reward 10, rewards [0.1,0.2,0.3,7,7] and values [0.5,-0.2,0.3,0.9,0.9]. Expected
    # advantages are reward minus value at trainable positions with the terminal reward added at
    # index 2: [0.1-0.5, 0.2+0.2, 0.3+10-0.3, 0, 0]; returns equal advantages plus masked values.
    # Masked tail: with gamma = 0 the terminal reward would be lost entirely if
    # it were injected at the last response token instead of the last trainable
    # token.
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    values = torch.tensor([0.5, -0.2, 0.3, 0.9, 0.9])
    rewards = torch.tensor([0.1, 0.2, 0.3, 7.0, 7.0])
    terminal_reward = 10.0

    advantages, returns = _compute([values], [rewards], [terminal_reward], [mask], 0.0, 0.0, chunked)

    expected = torch.tensor([0.1 - 0.5, 0.2 + 0.2, 0.3 + 10.0 - 0.3, 0.0, 0.0])
    torch.testing.assert_close(advantages[0], expected)
    torch.testing.assert_close(returns[0], expected + values * mask)


@pytest.mark.parametrize("chunked", [False, True])
def test_fully_masked_sample_yields_zeros(chunked: bool) -> None:
    # [FUNCTION DOC] Test (parametrized over chunked): two samples (lengths 4 and 6); the first is
    # fully masked and must give exactly zero advantages and returns; the second (all ones) matches
    # the serial reference with terminal reward 2.0, gamma 0.9, lambda 0.95.
    torch.manual_seed(1)
    values = [torch.randn(4), torch.randn(6)]
    rewards = [torch.randn(4), torch.randn(6)]
    masks = [torch.zeros(4), torch.ones(6)]

    advantages, returns = _compute(
        [v.clone() for v in values], [r.clone() for r in rewards], [5.0, 2.0], masks, 0.9, 0.95, chunked
    )

    assert torch.all(advantages[0] == 0)
    assert torch.all(returns[0] == 0)

    expected_adv, expected_ret = _reference_masked_gae(values[1], rewards[1], masks[1], 2.0, 0.9, 0.95)
    torch.testing.assert_close(advantages[1], expected_adv)
    torch.testing.assert_close(returns[1], expected_ret)


@pytest.mark.parametrize("chunked", [False, True])
def test_all_ones_mask_preserves_unmasked_behavior(chunked: bool) -> None:
    # [FUNCTION DOC] Test (parametrized over chunked): two samples of lengths 5 and 3 with all-ones
    # masks, terminal rewards 4 and -1, gamma 0.99, lambda 0.9. Compares with the classic padded
    # GAE routine run on zero-padded values and rewards (terminal reward added at the last real
    # position of each sample), restricted to each sample's real length.
    torch.manual_seed(2)
    lengths = [5, 3]
    values = [torch.randn(length) for length in lengths]
    rewards = [torch.randn(length) for length in lengths]
    terminal_rewards = [4.0, -1.0]
    gamma, lambd = 0.99, 0.9

    advantages, returns = _compute(
        [v.clone() for v in values],
        [r.clone() for r in rewards],
        terminal_rewards,
        [torch.ones(length) for length in lengths],
        gamma,
        lambd,
        chunked,
    )

    max_len = max(lengths)
    padded_values = torch.zeros(2, max_len)
    padded_rewards = torch.zeros(2, max_len)
    for i, length in enumerate(lengths):
        padded_values[i, :length] = values[i]
        padded_rewards[i, :length] = rewards[i]
        padded_rewards[i, length - 1] += terminal_rewards[i]
    expected_adv, expected_ret = vanilla_gae(padded_rewards, padded_values, gamma, lambd)

    for i, length in enumerate(lengths):
        torch.testing.assert_close(advantages[i], expected_adv[i, :length])
        torch.testing.assert_close(returns[i], expected_ret[i, :length])


@pytest.mark.parametrize("chunked", [False, True])
def test_truncation_uses_zero_bootstrap(chunked: bool) -> None:
    # [FUNCTION DOC] Test (parametrized over chunked): gamma = lambda = 1 with all-ones mask over
    # four tokens and terminal reward 1.5: advantages equal the reverse cumulative sum of rewards
    # plus the terminal reward minus the value at each position.
    # Truncated rollouts are treated like terminated ones: with gamma = lambd
    # = 1 the GAE telescopes to sum(rewards) + terminal - V_t, i.e. the value
    # after the last trainable token is bootstrapped as exactly zero.
    values = torch.tensor([0.4, -0.1, 0.25, 0.6])
    rewards = torch.tensor([0.1, -0.2, 0.3, 0.05])
    terminal_reward = 1.5

    advantages, _ = _compute(
        [values.clone()], [rewards.clone()], [terminal_reward], [torch.ones(4)], 1.0, 1.0, chunked
    )

    reward_tail_sums = torch.flip(torch.cumsum(torch.flip(rewards, dims=[0]), dim=0), dims=[0])
    expected = reward_tail_sums + terminal_reward - values
    torch.testing.assert_close(advantages[0], expected)
