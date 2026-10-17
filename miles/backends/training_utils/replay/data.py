# [MODULE DOC]
# Prepares rollout replay tensors (per-sample routing indices of shape [tokens - 1, streams, top-k],
# padded with -1) for consumption by replay objects inside the model. It replays the training data
# iterator in the same micro-batch order as the forward passes, pads each sample by one trailing
# position, slices/pads according to the sequence layout (padded "bshd" or packed "thd"), context
# parallelism and sequence parallelism, and hands each micro-batch's tensor to a registration
# function that maps streams to modules.
from typing import Protocol

import torch

from miles.backends.training_utils.data.context_parallel import slice_with_cp
from miles.backends.training_utils.parallel import get_parallel_state


class RegisterReplayListFunc(Protocol):
    # [CLASS DOC] Structural type of the registration callbacks: called with the list of replay
    # objects, the local replay tensor for one micro-batch and extra keyword arguments (the models);
    # returns nothing and records data into the replay objects.
    def __call__(self, replay_list: list, replay_data: torch.Tensor, **kwargs) -> None:
        # [FUNCTION DOC] Protocol method; no implementation (body is an ellipsis).
        ...


def register_replay_list_sequential(replay_list, replay_data, **_kwargs):
    # [FUNCTION DOC] Default registration. For each replay object in order, picks stream index =
    # its own stream index when set, else its position in the list; raises an assertion error if
    # that index is outside the number of streams (second dimension) of the replay tensor; then
    # tells the replay object to record the corresponding column slice (all tokens, that stream).
    # Extra keyword arguments are ignored.
    """Map replay streams to registered modules.

    Each replay records `replay_data[:, replay.stream_idx]` if `stream_idx` is
    set (used for sparse layer layouts under PP/VPP, where the global replay
    tensor contains more streams than this rank registered). Otherwise falls
    back to 1:1 enumeration order.
    """
    for replay_idx, replay in enumerate(replay_list):
        stream_idx = replay.stream_idx if replay.stream_idx is not None else replay_idx
        if not 0 <= stream_idx < replay_data.shape[1]:
            raise AssertionError(
                f"replay stream_idx {stream_idx} out of range " f"(replay_data has {replay_data.shape[1]} streams)"
            )
        replay.record(replay_data[:, stream_idx])


def fill_replay_data(
    *,
    args,
    models,
    data_iterator,
    num_microbatches,
    rollout_data,
    data_key: str,
    replay_list: list,
    register_replay_list_func: RegisterReplayListFunc,
    if_sp_region=True,
    indices_are_token_positions=False,
):
    # [FUNCTION DOC] Keyword-only. Raises a value error if the data key is missing from the rollout
    # data. Resets all data iterators, then for each micro-batch in total (sum of the per-iterator
    # micro-batch counts) pulls the next batch (replay tensors, tokens, padded lengths) from the
    # first iterator and asserts each replay tensor has exactly one fewer token than its token
    # tensor. Each sample is padded with one trailing -1 position. Layout handling: padded layout
    # (bshd): with all-gather CP and CP > 1, pad each sample to the batch's padded length (which
    # must be divisible by CP size) and take this CP rank's contiguous block; otherwise slice with
    # the CP slicing helper; stack samples and flatten batch and sequence into one token dimension.
    # Packed layout (thd): pad multiple = TP size * the data pad multiplier; with all-gather CP and
    # CP > 1, concatenate samples, pad to a multiple of CP size * pad multiple and take this rank's
    # chunk; otherwise slice each sample with CP, optionally (when indices are token positions) add
    # the running token offset to every non -1 entry so indices refer to packed positions,
    # concatenate and pad to the pad multiple. If sequence parallelism is on (and the flag for an
    # SP region is true), asserts the length is divisible by TP size and keeps this TP rank's
    # contiguous part. Finally calls the registration function with the replay list, the tensor and
    # the models. After all micro-batches deletes the data key from the rollout data (mutation) and
    # resets the iterators again. Padding values are -1. Returns nothing.
    """Load rollout replay tensors into module replay queues.

    `rollout_data[data_key]` contains one tensor per sample with shape
    `[num_tokens - 1, num_streams, topk]`. This function replays the training
    data iterator to process those tensors in the same microbatch order as
    log-prob and train forwards, pads/slices them to match the local CP/SP
    token layout, and then delegates stream-to-module mapping to
    `register_replay_list_func`.
    """
    if data_key not in rollout_data:
        raise ValueError(f"{data_key} is required in rollout_data for replay.")

    for iterator in data_iterator:
        iterator.reset()

    parallel_state = get_parallel_state()
    tp_rank = parallel_state.tp.rank
    tp_size = parallel_state.tp.size
    qkv_format = args.qkv_format

    def pad_func(data, pad):
        # [FUNCTION DOC] HELPER: Append "pad" rows of -1 (same dtype and device) to a tensor of shape
        # [tokens, layers, top-k] along the token dimension; returns the new tensor.
        _, num_layers, topk = data.shape
        pad_tensor = torch.full(
            (pad, num_layers, topk),
            fill_value=-1,
            device=data.device,
            dtype=data.dtype,
        )
        return torch.cat([data, pad_tensor], dim=0)

    for _ in range(sum(num_microbatches)):
        batch = data_iterator[0].get_next([data_key, "tokens", "max_seq_lens"])
        replay_data = batch[data_key]
        tokens = batch["tokens"]
        assert len(replay_data) == len(tokens)
        for a, b in zip(replay_data, tokens, strict=False):
            assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

        # Pad replay data to align with the token batch's final token. The padded token is masked from loss.
        # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
        replay_data = [pad_func(r, 1) for r in replay_data]
        # TODO: maybe extract a common process function for here and get_batch?

        cp_size = parallel_state.cp.size
        cp_rank = parallel_state.cp.rank
        if qkv_format == "bshd":
            max_seqlen = batch["max_seq_lens"][0]
            if args.allgather_cp and cp_size > 1:
                assert max_seqlen % cp_size == 0, f"max_seqlen {max_seqlen} must be divisible by cp_size {cp_size}"
                local_len = max_seqlen // cp_size
                start = cp_rank * local_len
                replay_data = [pad_func(r, max_seqlen - r.size(0))[start : start + local_len] for r in replay_data]
            else:
                replay_data = [slice_with_cp(r, pad_func, qkv_format, max_seqlen) for r in replay_data]
            replay_data = torch.stack(replay_data, dim=0)
            batch_size, seqlen, num_layers, topk = replay_data.shape
            replay_data = replay_data.reshape(batch_size * seqlen, num_layers, topk)
        else:
            pad_size = parallel_state.tp.size * args.data_pad_size_multiplier
            if args.allgather_cp and cp_size > 1:
                replay_data = torch.cat(replay_data, dim=0)
                global_pad_size = cp_size * pad_size
                pad = (global_pad_size - replay_data.size(0) % global_pad_size) % global_pad_size
                if pad != 0:
                    replay_data = pad_func(replay_data, pad)
                replay_data = replay_data.chunk(cp_size, dim=0)[cp_rank]
            else:
                replay_data = [slice_with_cp(r, pad_func, qkv_format) for r in replay_data]
                if indices_are_token_positions:
                    # map indices to thd format
                    offset = 0
                    for i, r in enumerate(replay_data):
                        replay_data[i] = torch.where(r != -1, r + offset, r)
                        offset += r.shape[0]
                replay_data = torch.cat(replay_data, dim=0)
                pad = (pad_size - replay_data.size(0) % pad_size) % pad_size
                if pad != 0:
                    replay_data = pad_func(replay_data, pad)

        # sequence_parallel is Megatron-only; FSDP has tp_size == 1 so the slice is a no-op there.
        if getattr(args, "sequence_parallel", False) and if_sp_region:
            seqlen = replay_data.size(0)
            assert seqlen % tp_size == 0
            start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
            replay_data = replay_data[start:end]

        register_replay_list_func(replay_list, replay_data, models=models)

    del rollout_data[data_key]

    for iterator in data_iterator:
        iterator.reset()
