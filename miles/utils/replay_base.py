# [MODULE DOC] Record-and-replay machinery for top-k selection results (routing replay "R3" for MoE
# experts and indexer replay for sparse-attention token positions). A manager wraps a module's
# top-k function so that during the rollout-logprob/forward pass the selected indices are recorded
# into per-module Replay buffers (pinned CPU memory), and during later forward/backward passes the
# recorded indices are returned instead of recomputing them. Two module-level singleton managers
# (routing and indexer) are exposed, plus a list of both.
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get_rank():
    # [FUNCTION DOC] HELPER: Returns the distributed rank of this process, or 0 when the
    # distributed process group has not been initialized. Used only for log/assert messages.
    return dist.get_rank() if dist.is_initialized() else 0


class Replay:
    # [CLASS DOC] A FIFO buffer of recorded top-k index tensors for one module (optionally tagged
    # with a stream index). Holds two independent read cursors: a forward cursor and a backward
    # cursor, both walking the same recorded list from the start. Recorded tensors live in pinned
    # CPU memory and are moved to the current CUDA device when popped.
    def __init__(self, stream_idx: int | None = None):
        # [FUNCTION DOC] Stores the optional stream index and starts with both cursors at zero and
        # an empty recorded list.
        self.stream_idx = stream_idx
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: list[torch.Tensor] = []

    def record(self, top_indices: torch.Tensor):
        # [FUNCTION DOC] Copies the given index tensor into a freshly allocated pinned CPU buffer
        # of the same shape and dtype and appends it to the recorded list. Synchronous with respect
        # to the copy call issued here; no value is returned.
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self) -> torch.Tensor:
        # [FUNCTION DOC] Returns the next recorded tensor for the forward cursor, moved to the
        # current CUDA device, and advances the forward cursor. Raises an index error (with the
        # cursor, list length and the shapes of all recorded entries in the message) when the
        # cursor is already past the end of the recorded list.
        if self.forward_index >= len(self.top_indices_list):
            shapes = [
                t.shape if isinstance(t, torch.Tensor) else f"non-tensor({type(t)})" for t in self.top_indices_list
            ]
            raise IndexError(
                f"pop_forward out of range: forward_index={self.forward_index}, "
                f"len(top_indices_list)={len(self.top_indices_list)}, shapes={shapes}"
            )
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self) -> torch.Tensor:
        # [FUNCTION DOC] Same as the forward pop but using the backward cursor: returns the
        # recorded tensor at the backward cursor on the current CUDA device and advances that
        # cursor. Unlike the forward pop it has no friendly bounds check, so running past the end
        # raises the plain list index error.
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def clear(self):
        # [FUNCTION DOC] Resets both cursors to zero and drops all recorded tensors.
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        # [FUNCTION DOC] Rewinds only the forward cursor to zero, keeping recorded tensors and the
        # backward cursor intact (so the same recording can be replayed forward again).
        self.forward_index = 0


class BaseReplayManager:
    # [CLASS DOC] Base class for a replay manager: owns a list of Replay buffers (one per
    # registered module), a "current" replay selected by forward pre-hooks, an enabled flag and a
    # stage string that selects behavior of the wrapped top-k function. Stages: "fallthrough"
    # (default, call the original function), "record", "replay_forward", "replay_backward"; any
    # unknown stage also falls through. Subclasses must define the class attributes name,
    # filename, data_key, if_sp_region, enable_check_replay_result and
    # replay_check_max_mismatch_fraction (the base class does not define the last four).
    # Class attributes defined here: name and filename (empty strings, subclass identifiers),
    # replay_check_min_overlap_ratio (per-token minimum overlap fraction for the check; 0.0 means
    # a token only mismatches when it has zero overlap) and replay_indices_are_token_positions
    # (true when indices are token/KV positions, so per-sample 0-based rollout indices need to be
    # rebased onto the packed training sequence; consumed by other modules).
    name: str = ""
    filename: str = ""
    replay_check_min_overlap_ratio = 0.0  # 0.0 = mismatch only on zero overlap
    # True when replayed indices are token/KV positions (indexer): rebase the
    # per-sample 0-based rollout indices onto the packed training sequence.
    replay_indices_are_token_positions = False

    def __init__(self):
        # [FUNCTION DOC] Initializes an empty replay list, no current replay, disabled state,
        # the "fallthrough" stage, and an unset register_replay_list_func slot (a hook slot that
        # is never read in this module; presumably assigned and used by external code).
        self.replays: list[Replay] = []
        self.current: Replay | None = None
        self.enabled = False
        self.stage = "fallthrough"
        self.register_replay_list_func = None

    def create_replay(self, stream_idx: int | None = None) -> Replay:
        # [FUNCTION DOC] Creates a new Replay with the given stream index, appends it to the
        # manager's replay list and returns it.
        replay = Replay(stream_idx=stream_idx)
        self.replays.append(replay)
        return replay

    def set_current(self, replay: Replay):
        # [FUNCTION DOC] Marks the given replay as the one that the wrapped top-k function uses.
        self.current = replay

    def get_current(self) -> Replay | None:
        # [FUNCTION DOC] Returns the currently selected replay, or nothing if none selected yet.
        return self.current

    def clear_all(self):
        # [FUNCTION DOC] Calls the full clear on every managed replay (cursors and recordings).
        for replay in self.replays:
            replay.clear()

    def clear_all_forward(self):
        # [FUNCTION DOC] Rewinds the forward cursor of every managed replay.
        for replay in self.replays:
            replay.clear_forward()

    def get_topk_fn(self, old_topk_fn, return_probs):
        # [FUNCTION DOC] Builds and returns a replacement for a top-k function. The original
        # function takes (scores, topk, extra args) and returns either only the index tensor or,
        # when return_probs is true, a pair of (probabilities, indices). The returned wrapper
        # behaves as follows depending on the manager state read at call time: if the manager is
        # not enabled or the stage is "fallthrough" (or unknown), it just calls the original.
        # Stage "record": calls the original, records the indices (the second element of the
        # result when return_probs is true) into the current replay and returns the original
        # result unchanged. Stage "replay_forward" / "replay_backward": pops the next recorded
        # indices from the current replay (forward or backward cursor) and builds the result from
        # them without calling the original (except for the optional check below). The current
        # replay must have been set (by a registered pre-forward hook) for record/replay stages.
        # Replay result construction: asserts the recorded indices have the same number of rows
        # as the scores and the same width as the requested topk; if the manager's
        # enable_check_replay_result flag is set, runs the mismatch check against the original
        # function; rows whose indices are all -1 (padding tokens) are replaced by the sequence
        # 0..topk-1 taken modulo the score width, to avoid invalid reads; then returns either
        # (scores gathered at those indices along dim 1, indices) when return_probs is true, or
        # only the indices. Note the inner function reads the check flag from self while the
        # wrapper reads state from the captured manager; these are the same object.
        manager = self

        def _get_replay_result(top_indices, scores, topk, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Turns recorded indices into the top-k function's result;
            # see the enclosing function for the algorithm. Raises assertion errors when the row
            # count differs from the scores or the width differs from topk (the rank is included
            # in the first message). May log and raise via the replay check.
            assert (
                top_indices.shape[0] == scores.shape[0]
            ), f"rank {_get_rank()}: replay n_tokens {top_indices.shape[0]} does not match scores n_tokens {scores.shape[0]}"

            assert (
                top_indices.shape[1] == topk
            ), f"replay topk does not match expected topk, replay topk {top_indices.shape[1]}, topk {topk}"

            if self.enable_check_replay_result:
                self.check_replay_result(old_topk_fn, scores, topk, top_indices, *args, **kwargs)

            # fill padding tokens with arange to avoid invalid reading
            all_invalid = (top_indices == -1).all(dim=-1)
            if all_invalid.any():
                ar = (
                    torch.arange(top_indices.shape[1], device=top_indices.device, dtype=top_indices.dtype)
                    % scores.shape[1]
                )
                top_indices = torch.where(all_invalid.unsqueeze(-1), ar, top_indices)

            if return_probs:
                return scores.gather(1, top_indices), top_indices
            else:
                return top_indices

        def new_topk_fn(scores, topk, *args, **kwargs):
            # [FUNCTION DOC] The wrapper returned to the caller; dispatches on the manager's
            # enabled flag and stage as described in the enclosing function. Side effects: in
            # the record stage appends to the current replay; in replay stages advances a cursor.
            if not manager.enabled:
                return old_topk_fn(scores, topk, *args, **kwargs)

            stage = manager.stage
            replay = manager.get_current()

            if stage == "fallthrough":
                return old_topk_fn(scores, topk, *args, **kwargs)

            elif stage == "record":
                result = old_topk_fn(scores, topk, *args, **kwargs)
                if return_probs:
                    probs, top_indices = result
                else:
                    top_indices = result
                replay.record(top_indices)
                return result

            elif stage == "replay_forward":
                return _get_replay_result(replay.pop_forward(), scores, topk, *args, **kwargs)

            elif stage == "replay_backward":
                return _get_replay_result(replay.pop_backward(), scores, topk, *args, **kwargs)

            else:
                return old_topk_fn(scores, topk, *args, **kwargs)

        return new_topk_fn

    def register_to_module(self, module, attr_name: str, stream_idx: int | None = None):
        # [FUNCTION DOC] Attaches a fresh replay to a torch module. Does nothing when the manager
        # is disabled at call time. Otherwise creates a replay (with the optional stream index),
        # stores it on the module under the given attribute name, and registers a forward
        # pre-hook on the module that makes this replay the manager's current replay before each
        # forward call. Side effects: mutates the module and the manager's replay list.
        if not self.enabled:
            return
        replay = self.create_replay(stream_idx=stream_idx)
        setattr(module, attr_name, replay)
        manager = self

        def pre_forward_hook(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Forward pre-hook; ignores its arguments and sets the
            # captured replay as the manager's current replay.
            manager.set_current(replay)

        module.register_forward_pre_hook(pre_forward_hook)

    def check_replay_result(self, old_topk_fn, scores, topk, top_indices, *args, **kwargs):
        # [FUNCTION DOC] CI/debug checker comparing replayed indices with a fresh recomputation.
        # Calls the original top-k function on the scores (taking the index part if it returns a
        # pair) and flattens both index sets to a two-dimensional tokens-by-topk layout. For
        # each token computes how many valid (non -1) replayed indices also appear among the
        # valid recomputed indices, using a boolean membership matrix over index values
        # (sized by the largest index seen plus one, with padding routed to an extra sentinel
        # column) instead of an all-pairs comparison. A token with no valid replayed indices is
        # padding and never counts as mismatched. The required overlap is 1 when the class's
        # min-overlap ratio is exactly zero, otherwise the ceiling of valid count times the
        # ratio. A token is mismatched when its overlap is below the requirement. The allowed
        # mismatch count is the fraction (read from the environment variable
        # MILES_TEST_R3_THRESHOLD as a float if set, else the class's max mismatch fraction)
        # times the number of tokens. Logs an info line with the counts (rank and stage
        # included); returns early if no mismatches. Otherwise logs up to 10 warnings, each
        # showing the mismatching token and up to three neighbours on each side with original
        # and replayed indices. Raises an assertion error when the mismatch count strictly
        # exceeds the allowed count. Requires the subclass to define the max mismatch fraction.
        # Cost: runs the original top-k, a device-to-host sync for several scalars.
        """
        CI checker for R3. Only enable when enable_check_replay_result=True.
        Per token, measure the overlap between the training engine's recomputed
        topk and the replayed topk (ignoring -1 padding). A token is mismatched
        when overlap < replay_check_min_overlap_ratio of its valid picks (ratio 0.0 =
        only zero overlap). Raise when the mismatched fraction exceeds
        replay_check_max_mismatch_fraction.
        """
        orig_top_indices = old_topk_fn(scores, topk, *args, **kwargs)
        if isinstance(orig_top_indices, tuple):
            _, orig_top_indices = orig_top_indices

        orig_flat = orig_top_indices.view(-1, orig_top_indices.shape[-1])  # [n_tokens, topk]
        replay_flat = top_indices.view(-1, top_indices.shape[-1])
        valid_orig = orig_flat != -1
        valid_replay = replay_flat != -1

        # token-wise set overlap via a membership mask, avoiding the O(n*topk^2)
        # all-pairs tensor; -1 padding is routed to a sentinel column
        n_kv = int(torch.maximum(orig_flat.max(), replay_flat.max()).clamp_min(0)) + 1
        membership = orig_flat.new_zeros((orig_flat.shape[0], n_kv + 1), dtype=torch.bool)
        membership.scatter_(1, torch.where(valid_orig, orig_flat, n_kv).long(), True)
        replay_idx = torch.where(valid_replay, replay_flat, n_kv).long()
        hit = membership.gather(1, replay_idx) & valid_replay  # [n_tokens, topk]

        overlap = hit.sum(dim=1)
        valid_count = valid_replay.sum(dim=1)
        is_padding = valid_count == 0
        if self.replay_check_min_overlap_ratio == 0.0:
            required = torch.ones_like(overlap)  # legacy: mismatch only on zero overlap
        else:
            required = (valid_count * self.replay_check_min_overlap_ratio).ceil().to(overlap.dtype)
        is_mismatch = (overlap < required) & ~is_padding

        mismatch_count = is_mismatch.sum().item()
        threshold = float(os.environ.get("MILES_TEST_R3_THRESHOLD", self.replay_check_max_mismatch_fraction))
        mismatch_threshold = threshold * orig_flat.shape[0]
        logger.info(
            f"Replay check (rank {_get_rank()}, stage {self.stage}): "
            f"mismatch {mismatch_count}/{orig_flat.shape[0]} tokens, threshold {mismatch_threshold:.0f}"
        )
        if mismatch_count == 0:
            return

        mismatch_indices = is_mismatch.nonzero(as_tuple=False).squeeze(1)
        for idx in mismatch_indices[:10]:
            i = idx.item()
            lines = []
            for j in range(max(0, i - 3), min(len(orig_flat), i + 4)):
                marker = " <<<" if j == i else ""
                lines.append(f"  token {j}: orig={orig_flat[j].tolist()}, replay={replay_flat[j].tolist()}{marker}")
            logger.warning(
                f"Replay check (rank {_get_rank()}, stage {self.stage}): "
                f"token {i} overlap {overlap[i].item()}/{valid_count[i].item()}, topk={topk}\n" + "\n".join(lines)
            )

        if mismatch_count > mismatch_threshold:
            raise AssertionError(f"R3 mismatch tokens ({mismatch_count}) > threshold ({mismatch_threshold:.0f})")


class RoutingReplayManager(BaseReplayManager):
    # [CLASS DOC] Manager for MoE expert-routing replay (R3). Identifier "routing", persisted
    # file name "routing_replay.pt", sample data key "rollout_routed_experts", flagged as living
    # in a sequence-parallel region, with the debug check off by default and a max allowed
    # mismatch fraction of 1 percent of tokens (min-overlap ratio inherited: zero-overlap only).
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    if_sp_region = True
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2


class IndexerReplayManager(BaseReplayManager):
    # [CLASS DOC] Manager for sparse-attention indexer top-k replay. Identifier "indexer",
    # file name "indexer_replay.pt", sample data key "rollout_indexer_topk", not in a
    # sequence-parallel region, debug check off by default, max mismatch fraction 1 percent,
    # requires at least 80 percent overlap of valid picks per token, and its indices are token
    # positions (so they must be rebased onto the packed training sequence).
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    if_sp_region = False
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2
    replay_check_min_overlap_ratio = 0.8
    replay_indices_are_token_positions = True


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]
