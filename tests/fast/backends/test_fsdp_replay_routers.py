# [MODULE DOC] CPU tests for the FSDP routing-replay router installers for Qwen3 MoE and GLM4 MoE
# lite. Uses small fake routers that mirror the upstream transformers forward logic, wires replay
# onto them through the shared test fixture, and drives the global routing replay manager through
# its stages (fallthrough, record, replay forward, replay backward). Verifies fallthrough equals the
# stock forward, replay returns recorded expert indices, forward and backward cursors are
# independent, replayed weights stay differentiable and are recomputed from the replayed experts.
# Registered as a CPU test in the first CPU stage.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.fast.fixtures.replay_fixtures import reset_routing_replay_manager, wire_replay

from miles.backends.fsdp_utils.models.replay_routers import (
    install_glm4_moe_lite_router_replay,
    install_qwen3_router_replay,
)
from miles.utils.replay_base import routing_replay_manager


@pytest.fixture(autouse=True)
def _reset_manager():
    # [FUNCTION DOC] HELPER: Autouse fixture; resets the global routing replay manager to an enabled
    # clean state before each test and to a disabled state afterwards.
    reset_routing_replay_manager(enabled=True)
    yield
    reset_routing_replay_manager(enabled=False)


class _Qwen3MoeTopKRouter(nn.Module):
    # [CLASS DOC] HELPER: Fake top-k router with a random expert-by-hidden weight, copying the
    # upstream Qwen3 MoE router forward: linear logits, fp32 softmax, top-k selection and optional
    # renormalization of the selected probabilities.
    """Mirrors transformers 5.12.1 Qwen3MoeTopKRouter.forward."""

    def __init__(self, num_experts=8, hidden_dim=4, top_k=2, norm_topk_prob=True):
        # [FUNCTION DOC] HELPER: Stores the sizes and the normalization flag and creates a random
        # normal weight of shape experts by hidden.
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim))

    def forward(self, hidden_states):
        # [FUNCTION DOC] HELPER: Flattens input to rows of hidden size, computes logits, fp32
        # softmax probabilities, top-k values and indices, divides values by their row sum if the
        # normalization flag is set, casts values to the logits dtype. Returns (logits, top values,
        # top indices).
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


def _wire_qwen3(router):
    # [FUNCTION DOC] HELPER: Installs the Qwen3 router replay on the router via the shared wiring
    # fixture and returns what that returns (the replay object).
    return wire_replay(router, install_qwen3_router_replay)


def test_fallthrough_matches_stock_forward():
    # [FUNCTION DOC] Test: with a seeded router, outputs before wiring equal outputs after wiring
    # under the fallthrough stage, element-wise (all three outputs close).
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter()
    x = torch.randn(6, 4)
    expected = router(x)

    _wire_qwen3(router)
    routing_replay_manager.stage = "fallthrough"
    got = router(x)

    for a, b in zip(expected, got, strict=True):
        assert torch.allclose(a, b)


def test_replay_forward_returns_the_recorded_indices():
    # [FUNCTION DOC] Test: records indices from one input, clears forward cursors, switches to
    # replay forward and runs on a different random input; the returned indices equal the recorded
    # ones and the weights have the same shape as the indices.
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter()
    _wire_qwen3(router)
    x = torch.randn(6, 4)

    routing_replay_manager.stage = "record"
    _, _, recorded = router(x)

    routing_replay_manager.clear_all_forward()
    routing_replay_manager.stage = "replay_forward"
    _, weights, replayed = router(torch.randn(6, 4))

    assert torch.equal(replayed.cpu(), recorded.cpu())
    assert weights.shape == recorded.shape


def test_forward_and_backward_cursors_are_independent():
    # [FUNCTION DOC] Test: records two calls, clears the forward cursor, then one replay-forward
    # call and one replay-backward call each return the first recorded indices, showing each stage
    # has its own cursor.
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter()
    _wire_qwen3(router)

    routing_replay_manager.stage = "record"
    _, _, first = router(torch.randn(3, 4))
    router(torch.randn(3, 4))
    routing_replay_manager.clear_all_forward()

    routing_replay_manager.stage = "replay_forward"
    _, _, fwd0 = router(torch.randn(3, 4))
    routing_replay_manager.stage = "replay_backward"
    _, _, bwd0 = router(torch.randn(3, 4))

    assert torch.equal(fwd0.cpu(), first.cpu())
    assert torch.equal(bwd0.cpu(), first.cpu())


def test_weights_stay_differentiable_under_replay():
    # [FUNCTION DOC] Test: after recording, a replay-forward call's weights summed and
    # backpropagated give the router weight a finite non-empty gradient.
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter()
    _wire_qwen3(router)

    routing_replay_manager.stage = "record"
    router(torch.randn(5, 4))
    routing_replay_manager.clear_all_forward()

    routing_replay_manager.stage = "replay_forward"
    _, weights, _ = router(torch.randn(5, 4))
    weights.sum().backward()

    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()


def test_qwen3_5_variant_without_norm_topk_prob_attribute():
    # [FUNCTION DOC] Test: with the normalization attribute deleted from the router (as in the
    # Qwen3.5 variant), the recorded-stage weights per row still sum to one within 1e-5, i.e. the
    # installer treats a missing attribute as normalization on.
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter()
    del router.norm_topk_prob
    _wire_qwen3(router)

    routing_replay_manager.stage = "record"
    _, weights, _ = router(torch.randn(4, 4))

    assert torch.allclose(weights.float().sum(dim=-1), torch.ones(4), atol=1e-5)


def test_replayed_weights_are_the_probs_at_the_replayed_experts():
    # [FUNCTION DOC] Test: with normalization off, after record then replay-forward, the returned
    # weights equal the fp32 softmax of the returned logits gathered at the replayed indices, within
    # 1e-6.
    torch.manual_seed(0)
    router = _Qwen3MoeTopKRouter(norm_topk_prob=False)
    _wire_qwen3(router)

    routing_replay_manager.stage = "record"
    router(torch.randn(4, 4))
    routing_replay_manager.clear_all_forward()

    x = torch.randn(4, 4)
    routing_replay_manager.stage = "replay_forward"
    logits, weights, idx = router(x)

    expected = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1).gather(1, idx)
    assert torch.allclose(weights.float(), expected, atol=1e-6)


class _Gate(nn.Module):
    # [CLASS DOC] HELPER: Fake gate holding only a zero bias buffer, one entry per routed expert.
    def __init__(self, n_routed_experts):
        # [FUNCTION DOC] HELPER: Registers the zero score-correction bias buffer.
        super().__init__()
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))


class _Glm4MoeLiteMoE(nn.Module):
    # [CLASS DOC] HELPER: Fake GLM4 MoE lite block exposing only the routing function copied from
    # upstream: sigmoid scores, grouped top-k selection, gathered weights, optional normalization,
    # then scaling.
    """Mirrors transformers 5.12.1 Glm4MoeLiteMoE.route_tokens_to_experts."""

    def __init__(self, n_routed_experts=8, n_group=2, topk_group=1, top_k=2):
        # [FUNCTION DOC] HELPER: Creates the gate and stores expert/group/top-k sizes, with
        # normalization on and routed scaling factor 1.5.
        super().__init__()
        self.gate = _Gate(n_routed_experts)
        self.n_routed_experts = n_routed_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.5
        self.top_k = top_k

    def route_tokens_to_experts(self, router_logits):
        # [FUNCTION DOC] HELPER: Applies sigmoid to the logits; adds the correction bias for choice
        # only; scores each group by the sum of its two best entries; keeps the best groups (count
        # topk_group); masks other groups to negative infinity; picks the top-k experts (unsorted);
        # gathers the sigmoid scores there; if normalizing divides by their sum plus a tiny epsilon;
        # multiplies by the routed scaling factor. Returns (expert indices, weights).
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


def _wire_glm(block):
    # [FUNCTION DOC] HELPER: Installs the GLM4 MoE lite router replay on the block via the shared
    # wiring fixture and returns the replay object.
    return wire_replay(block, install_glm4_moe_lite_router_replay)


def test_fallthrough_matches_stock_routing():
    # [FUNCTION DOC] Test: indices and weights from the fake block before wiring equal those after
    # wiring under the fallthrough stage (exact for indices, close for weights).
    torch.manual_seed(0)
    block = _Glm4MoeLiteMoE()
    logits = torch.randn(5, 8)
    exp_idx, exp_w = block.route_tokens_to_experts(logits)

    _wire_glm(block)
    routing_replay_manager.stage = "fallthrough"
    got_idx, got_w = block.route_tokens_to_experts(logits)

    assert torch.equal(exp_idx, got_idx)
    assert torch.allclose(exp_w, got_w)


def test_replay_forward_overrides_expert_selection():
    # [FUNCTION DOC] Test: records indices from one random logit input, then under replay forward
    # with another input the returned indices equal the recorded ones, weights have matching shape
    # and are all finite.
    torch.manual_seed(0)
    block = _Glm4MoeLiteMoE()
    _wire_glm(block)

    routing_replay_manager.stage = "record"
    recorded, _ = block.route_tokens_to_experts(torch.randn(5, 8))

    routing_replay_manager.clear_all_forward()
    routing_replay_manager.stage = "replay_forward"
    replayed, weights = block.route_tokens_to_experts(torch.randn(5, 8))

    assert torch.equal(replayed.cpu(), recorded.cpu())
    assert weights.shape == recorded.shape
    assert torch.isfinite(weights).all()


def test_replayed_weights_gather_from_the_sigmoid_logits():
    # [FUNCTION DOC] Test: with normalization off, replayed weights equal the sigmoid of the current
    # logits gathered at the replayed indices times the routed scaling factor, within 1e-6.
    torch.manual_seed(0)
    block = _Glm4MoeLiteMoE()
    block.norm_topk_prob = False
    _wire_glm(block)

    routing_replay_manager.stage = "record"
    block.route_tokens_to_experts(torch.randn(5, 8))
    routing_replay_manager.clear_all_forward()

    logits = torch.randn(5, 8)
    routing_replay_manager.stage = "replay_forward"
    idx, weights = block.route_tokens_to_experts(logits)

    expected = logits.sigmoid().gather(1, idx) * block.routed_scaling_factor
    assert torch.allclose(weights, expected, atol=1e-6)


def test_replay_can_select_experts_outside_the_recomputed_groups():
    # [FUNCTION DOC] Test: pushes a forced index tensor (3 rows, experts spanning both groups) onto
    # the replay's recorded list directly and sets replay forward. Routing returns exactly the forced
    # indices (group selection is bypassed), finite weights equal to the sigmoid gathered at them
    # times the scaling factor (normalization off).
    torch.manual_seed(0)
    block = _Glm4MoeLiteMoE()
    block.norm_topk_prob = False
    replay = _wire_glm(block)

    forced = torch.tensor([[0, 7], [7, 0], [1, 6]])
    replay.recorded.append(forced)
    routing_replay_manager.stage = "replay_forward"

    logits = torch.randn(3, 8)
    idx, weights = block.route_tokens_to_experts(logits)

    assert torch.equal(idx, forced)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights, logits.sigmoid().gather(1, forced) * block.routed_scaling_factor, atol=1e-6)


def test_gradients_flow_to_the_router_logits_under_replay():
    # [FUNCTION DOC] Test: after recording, routing under replay forward with logits requiring grad,
    # summing the weights and backpropagating yields a finite non-empty gradient on the logits.
    torch.manual_seed(0)
    block = _Glm4MoeLiteMoE()
    _wire_glm(block)

    routing_replay_manager.stage = "record"
    block.route_tokens_to_experts(torch.randn(4, 8))
    routing_replay_manager.clear_all_forward()

    logits = torch.randn(4, 8, requires_grad=True)
    routing_replay_manager.stage = "replay_forward"
    _, weights = block.route_tokens_to_experts(logits)
    weights.sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
