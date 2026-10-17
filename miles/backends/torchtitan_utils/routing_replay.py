# [MODULE DOC] Routing replay (recording and replaying which experts each token was routed to) for
# torchtitan MoE models. Installation swaps each token-choice top-k router's forward for a version
# that picks the top-k expert ids through the global replay manager (so rollout routing can be
# replayed in training), registers every router as a stream indexed by its decoder-layer number, and
# wraps each model part's forward so that the recompute forward in backward uses the forward replay
# stage. Also provides helpers to keep pipeline-schedule metadata probing from consuming replay
# data, and a guard that verifies each replay stream advanced exactly once per microbatch.
# Module-level state: a marker attribute name and an "initializing" record used by the probe bypass.
import contextlib
import functools
import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from miles.backends.training_utils.replay.routing_replay import FALLTHROUGH, REPLAY_BACKWARD, REPLAY_FORWARD, stage
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: The local shard of a distributed tensor, or the tensor itself otherwise.
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _like(local: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Wraps a local tensor to match the reference tensor's distribution. If
    # the reference is a plain tensor, returns the local tensor. If it is a distributed tensor,
    # raises a runtime error when any placement shards its last dimension (the expert axis, since
    # replayed expert ids cannot live in a shard of that axis); otherwise returns a distributed
    # tensor built from the local one with the reference's mesh and placements.
    if not isinstance(reference, DTensor):
        return local
    expert_dim = reference.ndim - 1
    for placement in reference.placements:
        if placement.is_shard() and placement.dim in (expert_dim, -1):
            raise RuntimeError(
                f"router scores are sharded over experts ({reference.placements}); replayed "
                "expert ids have no shard of that axis to live in"
            )
    return DTensor.from_local(local, reference.device_mesh, reference.placements)


def _token_router_forward(self, x_BLD: torch.Tensor, expert_bias_E: torch.Tensor | None = None):
    # [FUNCTION DOC] Replacement forward bound to a router object. Input: hidden states of shape
    # batch by length by dimension and an optional per-expert bias. Steps: computes gate scores in
    # float32 autocast; applies sigmoid or softmax (over experts) according to the router's score
    # function, raising a not-implemented error for others; adds the expert bias for the choice
    # scores if given; applies node-limited group routing to the choice scores if expert groups are
    # configured; selects the top-k expert ids through the replay-aware top-k function on the local
    # choice scores flattened to tokens by experts (ids reshaped back to batch by length by k and
    # wrapped to match the distribution); gathers the unbiased scores at those ids; if the debug
    # flag to force load balance is set, replaces ids and scores from the debug routing; if route
    # normalization is set, divides scores by their sum over k (plus a tiny constant); multiplies
    # by the route scale. Returns (top-k scores, top-k expert ids, full scores).
    with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
        scores_BLE = self.gate(x_BLD)

    if self.score_func == "sigmoid":
        scores_BLE = torch.sigmoid(scores_BLE)
    elif self.score_func == "softmax":
        scores_BLE = F.softmax(scores_BLE, dim=-1)
    else:
        raise NotImplementedError(f"Unknown score function {self.score_func}")

    scores_for_choice_BLE = scores_BLE if expert_bias_E is None else scores_BLE + expert_bias_E
    if self.num_expert_groups is not None:
        scores_for_choice_BLE = self._get_node_limited_routing_scores(scores_for_choice_BLE)

    local_choice = _local(scores_for_choice_BLE)
    b, seq_len, _ = local_choice.shape
    topk_expert_ids_BLK = _like(
        self._miles_replay_topk(local_choice.reshape(b * seq_len, -1), self.top_k).reshape(b, seq_len, self.top_k),
        scores_for_choice_BLE,
    )

    topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)

    if self._debug_force_load_balance:
        topk_expert_ids_BLK, topk_scores_BLK = self._debug_force_load_balance_routing(scores_BLE)

    if self.route_norm:
        denominator = topk_scores_BLK.sum(dim=-1, keepdim=True) + 1e-20
        topk_scores_BLK = topk_scores_BLK / denominator
    topk_scores_BLK = topk_scores_BLK * self.route_scale

    return topk_scores_BLK, topk_expert_ids_BLK, scores_BLE


_INSTALLED_ATTR = "_miles_replay_installed"

_initializing: dict | None = None


def install(model_parts: list[nn.Module]) -> int:
    # [FUNCTION DOC] Installs routing replay on the model parts and returns the number of routers
    # registered (zero, doing nothing, when replay is disabled in the global manager). Finds every
    # token-choice top-k router; the layer index is the first all-digit segment of the router's
    # module path (value error if none). Raises a value error if replay is enabled but no router
    # exists (MoE models only). Wraps each part's forward and marks the part as installed. For
    # routers in layer order: gives it a replay-aware top-k function from the manager (taking an
    # unsorted top-k index selection as the base, without probabilities), swaps its forward for the
    # replacement, and registers it with the manager as a routing replay module with the layer index
    # as its stream. Logs an info line with the count and index range.
    if not routing_replay_manager.enabled:
        return 0

    from torchtitan.models.common.moe import TokenChoiceTopKRouter

    routers: list[tuple[int, nn.Module]] = []
    for part in model_parts:
        for name, module in part.named_modules():
            if not isinstance(module, TokenChoiceTopKRouter):
                continue
            layer_key = next((p for p in name.split(".") if p.isdigit()), None)
            if layer_key is None:
                raise ValueError(f"cannot derive a decoder-layer index from router path {name!r}")
            routers.append((int(layer_key), module))

    if not routers:
        raise ValueError(
            "routing replay is enabled but this model has no torchtitan TokenChoiceTopKRouter; "
            "R3 applies to MoE models only"
        )

    for part in model_parts:
        _bracket_real_forward(part)
        setattr(part, _INSTALLED_ATTR, True)

    for layer_idx, router in sorted(routers, key=lambda pair: pair[0]):
        router._miles_replay_topk = routing_replay_manager.get_topk_fn(
            lambda scores, k: torch.topk(scores, k, dim=-1, sorted=False)[1], return_probs=False
        )
        router.forward = types.MethodType(_token_router_forward, router)
        routing_replay_manager.register_to_module(router, "routing_replay", stream_idx=layer_idx)

    indices = sorted(idx for idx, _ in routers)
    logger.info(
        f"[titan routing_replay] registered {len(routers)} MoE layers " f"(global indices {indices[0]}..{indices[-1]})"
    )
    return len(routers)


def _is_installed(model_parts: list[nn.Module]) -> bool:
    # [FUNCTION DOC] HELPER: True when replay is enabled in the manager and every part carries the
    # installed marker.
    return routing_replay_manager.enabled and all(getattr(part, _INSTALLED_ATTR, False) for part in model_parts)


def bypass_schedule_initialization(model_parts: list[nn.Module]) -> None:
    # [FUNCTION DOC] Called before a pipeline schedule's first forward, which runs a shape-probing
    # pass that must not consume replay data. Does nothing unless replay is installed on all parts.
    # Otherwise records, in the module-level state, the ids of the parts not yet probed and the
    # manager's current stage, and sets the manager's stage to pass-through (fall through). The
    # state is cleared by the wrapped forward after every part has been probed, or by the
    # consumption guard.
    global _initializing
    if not _is_installed(model_parts):
        return
    _initializing = {
        "unprobed": {id(part) for part in model_parts},
        "stage": routing_replay_manager.stage,
    }
    routing_replay_manager.stage = FALLTHROUGH


def _end_initialization() -> None:
    # [FUNCTION DOC] HELPER: If a probe bypass is active, restores the manager's saved stage and
    # clears the state; otherwise does nothing.
    global _initializing
    if _initializing is None:
        return
    routing_replay_manager.stage = _initializing["stage"]
    _initializing = None


@contextlib.contextmanager
def consumption_guard(model_parts: list[nn.Module], expected: int):
    # [FUNCTION DOC] Context manager around one pass of the given number of microbatches. A no-op
    # if replay is not installed. Otherwise records each replay stream's forward and backward
    # counters at entry; at exit (always ending any probe bypass first) checks every stream:
    # the forward counter must have advanced exactly the expected number of times, and the backward
    # (recompute) counter must have advanced zero or exactly the expected number of times; a
    # violation raises a runtime error naming the stream. The checks run only when the body
    # completes without error.
    if not _is_installed(model_parts):
        yield
        return
    before = {id(replay): (replay.forward_index, replay.backward_index) for replay in routing_replay_manager.replays}
    try:
        yield
    finally:
        _end_initialization()
    for replay in routing_replay_manager.replays:
        forward_before, backward_before = before[id(replay)]
        advance = replay.forward_index - forward_before
        if advance != expected:
            raise RuntimeError(
                f"routing replay stream {replay.stream_idx} advanced {advance} times over a pass "
                f"of {expected} microbatches; the queues no longer line up with the microbatches"
            )
        recompute = replay.backward_index - backward_before
        if recompute not in (0, expected):
            raise RuntimeError(
                f"routing replay stream {replay.stream_idx} recomputed {recompute} times over a "
                f"pass of {expected} microbatches; the recompute pass is replaying the wrong "
                "microbatches"
            )


def _bracket_real_forward(part: nn.Module) -> None:
    # [FUNCTION DOC] HELPER: Replaces the part's forward with a wrapper (keeping the original's
    # metadata). The wrapper: while a probe bypass is active, lets the first call of each unprobed
    # part run unchanged (marking it probed) and ends the bypass on any later call; then, if the
    # manager is in the backward-replay stage (activation recomputation during backward), runs the
    # original forward under the forward-replay stage so the recompute sees the same routing;
    # otherwise runs it unchanged. Mutates the part.
    inner = part.forward

    @functools.wraps(inner)
    def forward(*args, **kwargs):
        # [FUNCTION DOC] HELPER: The wrapper described by the enclosing function; passes all
        # arguments to the original forward and returns its result.
        if _initializing is not None:
            if id(part) in _initializing["unprobed"]:
                _initializing["unprobed"].discard(id(part))
                return inner(*args, **kwargs)
            _end_initialization()
        if routing_replay_manager.stage == REPLAY_BACKWARD:
            with stage(REPLAY_FORWARD):
                return inner(*args, **kwargs)
        return inner(*args, **kwargs)

    part.forward = forward
