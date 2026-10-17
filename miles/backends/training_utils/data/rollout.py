# [MODULE DOC] Turns rollout data into per-micro-batch training inputs. Three stages: (1)
# get_rollout_data fetches this data-parallel rank's share of the rollout from the object store and
# moves token-like fields to the GPU, computing padded max sequence lengths for the bshd format and
# context-parallel-slicing the full-response log-prob style fields; (2) get_data_iterator decides the
# micro-batch schedule (fixed-size contiguous micro-batches, or dynamic token-budget micro-batches
# balanced by sequence length) and wraps the data in iterators, one per virtual pipeline stage;
# (3) get_batch pulls one micro-batch from an iterator and builds the model inputs: packed (thd) or
# padded (bshd) tokens sliced for context parallelism, cumulative sequence lengths, input-aligned
# loss masks, optional position and witness ids, multi-LoRA per-adapter token counts, and
# concatenated multimodal tensors. "thd" packs all samples of a micro-batch into one token row;
# "bshd" stacks padded rows. Samples are Python lists of per-sample tensors keyed by field name.
import logging
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from miles.backends.training_utils.data.context_parallel import slice_log_prob_with_cp, slice_with_cp
from miles.backends.training_utils.data.multimodal import expand_multimodal_rollout_data_in_place
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.audit_utils.witness.allocator import WitnessInfo
from miles.utils.data import get_minimum_num_micro_batch_size, process_rollout_data
from miles.utils.ft_utils.process_group_utils import GeneralPGUtil
from miles.utils.object_store import ObjectStoreGetResult
from miles.utils.ray_utils import Box
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


def _rollout_logprob_dtype(args: Namespace) -> torch.dtype:
    # [FUNCTION DOC] HELPER: Element type for rollout log-probs. In true-on-policy mode returns
    # bfloat16 if the bf16 flag is set, else float16 if the fp16 flag is set (flags default false);
    # in every other case float32.
    if getattr(args, "true_on_policy_mode", False):
        if getattr(args, "bf16", False):
            return torch.bfloat16
        if getattr(args, "fp16", False):
            return torch.float16
    return torch.float32


def get_rollout_data(
    args: Namespace,
    rollout_data_ref: Box,
    witness_info: WitnessInfo | None = None,
) -> tuple[RolloutBatch, ObjectStoreGetResult]:
    # [FUNCTION DOC] Fetches and prepares this rank's rollout data. Inputs: arguments, a boxed
    # reference to the rollout data in the object store, optional witness info. Returns (rollout data
    # dictionary of per-sample lists, object store get result). Requires a CUDA device.
    # Steps: fetch this effective-data-parallel rank's slice of the rollout (CPU, through Ray; first
    # and last pipeline stages both receive it); convert tokens to long tensors and loss masks to int
    # tensors on the current GPU; convert rollout mask sums (if present) to one float32 GPU tensor;
    # with witness enabled, pop the per-sequence witness ids and expand each to a long tensor with
    # one id per token (lengths must match, strict); move every multimodal input dictionary's
    # tensors to the GPU (none entries kept). For the bshd format: max sequence length = the largest
    # total length rounded up to a multiple of the pad size, where pad size = tensor-parallel size
    # times the data pad multiplier, raised to at least (largest attention compress ratio, doubled
    # when CP is on, times CP size) if compress ratios are configured; stores that value once per
    # sample under max sequence lengths. Then, for each of rollout log-probs, teacher log-probs and
    # on-policy-distillation reverse KL that are present, slices each sample's full-response values
    # into the local CP layout and converts to a GPU tensor (element type from the dtype helper for
    # rollout log-probs, float32 for the others). Routed-expert and indexer top-k arrays are
    # converted from NumPy to tensors (CPU, sharing memory).
    # KNOWN ISSUE: the sample, length lists are zipped without a strict length check in the log-prob
    # conversion; with compress ratios the final pad size is not guaranteed to remain a multiple of
    # the tensor-parallel padding.
    parallel_state = get_parallel_state()
    # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
    # Both first pp stage and the last pp stage will receive the data.
    rollout_data, store_get_result = process_rollout_data(
        args,
        rollout_data_ref,
        parallel_state.effective_dp.rank,
        parallel_state.effective_dp.size,
        witness_info=witness_info,
    )
    # move tokens to GPU in advance
    rollout_data["tokens"] = [
        torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
    ]
    rollout_data["loss_masks"] = [
        torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
    ]
    if "rollout_mask_sums" in rollout_data:
        rollout_data["rollout_mask_sums"] = torch.tensor(
            rollout_data["rollout_mask_sums"], dtype=torch.float32, device=torch.cuda.current_device()
        )
    if args.enable_witness:
        seq_witness_ids = rollout_data.pop("seq_witness_ids")
        rollout_data["witness_ids"] = [
            torch.full((len(t),), fill_value=sid, dtype=torch.long, device=torch.cuda.current_device())
            for t, sid in zip(rollout_data["tokens"], seq_witness_ids, strict=True)
        ]

    if "multimodal_train_inputs" in rollout_data:
        # Move multimodal training tensors to GPU in advance
        rollout_data["multimodal_train_inputs"] = [
            (
                {key: tensor.to(device=torch.cuda.current_device()) for key, tensor in mm_dict.items()}
                if mm_dict is not None
                else None
            )
            for mm_dict in rollout_data["multimodal_train_inputs"]
        ]

    if args.qkv_format == "bshd":
        # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
        max_seq_len = max(rollout_data["total_lengths"])

        # pad to reduce memory fragmentation and maybe make the computation faster
        pad_size = parallel_state.tp.size * args.data_pad_size_multiplier
        max_compress_ratio = max(args.compress_ratios) if args.compress_ratios else 0
        if max_compress_ratio:
            local_seqlen_multiple = max_compress_ratio * (2 if parallel_state.cp.size > 1 else 1)
            pad_size = max(pad_size, local_seqlen_multiple * parallel_state.cp.size)
        max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

        rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

    # Full-response SGLang OPD fields share rollout CP slicing but retain float32 precision.
    for key in ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl"):
        if key in rollout_data:
            dtype = _rollout_logprob_dtype(args) if key == "rollout_log_probs" else torch.float32
            rollout_data[key] = [
                torch.as_tensor(
                    slice_log_prob_with_cp(
                        value,
                        total_length,
                        response_length,
                        args.qkv_format,
                        rollout_data["max_seq_lens"][i] if args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=dtype,
                )
                for i, (value, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
    if "rollout_routed_experts" in rollout_data:
        rollout_data["rollout_routed_experts"] = [torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]]
    if "rollout_indexer_topk" in rollout_data:
        rollout_data["rollout_indexer_topk"] = [torch.from_numpy(r) for r in rollout_data["rollout_indexer_topk"]]
    return rollout_data, store_get_result


def _prepare_loss_masks(
    loss_masks: Sequence[torch.Tensor],
    total_lengths: Sequence[int],
    response_lengths: Sequence[int],
    *,
    qkv_format: str,
    max_seqlen: int,
    pad: int,
    allgather_cp: bool,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Builds the loss mask aligned one-to-one with the input tokens of a
    # micro-batch. For each sample (lists must have equal length, strict): left-pads the response
    # mask with zeros by the prompt length (so entry i is one iff token i is a supervised response
    # token), then, unless allgather CP is used, slices it into the local zigzag CP layout with pad
    # value zero (bshd needs the max sequence length). bshd: with allgather CP, each mask is
    # right-padded to the max sequence length and cut to this rank's contiguous window
    # (max length divided by CP size); then the rows are stacked into a batch-by-length tensor.
    # thd: concatenates all masks, right-pads by the given pad count with zeros, with allgather CP
    # keeps this rank's contiguous 1/CP chunk, and adds a leading batch dimension of one. No label
    # shift is applied here.
    """Expand response masks to input-token alignment, then pack/slice them like ``tokens``.

    Contract: ``mask[i]`` is 1 iff ``tokens[i]`` is a supervised response token.
    Whoever derives labels owns the prediction shift. Megatron does so when
    ``labels=None``: it rolls ``input_ids`` and this mask together to build the
    MTP labels, and it does not read the mask at all when MTP is off. The
    alignment is fixed before CP slicing because a local roll in Miles would
    have no halo exchange across shard or packed-sequence boundaries.
    """
    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size
    cp_rank = parallel_state.cp.rank
    aligned_masks = []
    for loss_mask, total_length, response_length in zip(loss_masks, total_lengths, response_lengths, strict=True):
        prompt_length = total_length - response_length
        loss_mask = F.pad(loss_mask, (prompt_length, 0), value=0)
        if not allgather_cp:
            loss_mask = slice_with_cp(loss_mask, 0, qkv_format, max_seqlen)
        aligned_masks.append(loss_mask)

    if qkv_format == "bshd":
        if allgather_cp:
            local_len = max_seqlen // cp_size
            start = cp_rank * local_len
            aligned_masks = [
                F.pad(mask, (0, max_seqlen - mask.size(0)), value=0)[start : start + local_len]
                for mask in aligned_masks
            ]
        return torch.stack(aligned_masks)

    loss_mask = F.pad(torch.cat(aligned_masks), (0, pad), value=0)
    if allgather_cp:
        loss_mask = loss_mask.chunk(cp_size, dim=0)[cp_rank]
    return loss_mask.unsqueeze(0)


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    qkv_format: str = "thd",
    get_position_ids: bool = False,
    allgather_cp: bool = False,
) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
    # [FUNCTION DOC] Builds one micro-batch of model inputs from the iterator and returns it as a
    # dictionary. Arguments: the data iterator; the keys to fetch (must include tokens; the
    # adapter slots key is added automatically); pad multiplier (default 128); qkv format
    # (default thd; anything else except bshd raises a value error); whether to also build position
    # ids (default false); whether the allgather flavor of context parallelism is used (default
    # false, unsupported together with multi-LoRA adapter slots).
    # Batch contents: the fetched fields (absent keys become none); the dynamic global batch size and
    # the loss function and its config copied from the iterator's rollout data when present; the
    # original per-sample token list under unconcatenated tokens (before CP slicing); the final
    # tokens as a batch of rows; input loss masks of the same shape as the tokens; and format-
    # specific fields. A no-op multimodal expansion is run first as a safety net.
    # bshd: the max sequence length is the first sample's entry of max sequence lengths (asserted to
    # cover every sample); allgather CP pads each sample to it and cuts this rank's contiguous
    # window (requires divisibility by CP size), otherwise each sample is zigzag-sliced; the rows are
    # stacked.
    # thd with allgather CP: sequences are concatenated first and the global stream is padded to a
    # multiple of (CP size times tensor-parallel size times pad multiplier), the padding counted as
    # one extra segment; this rank takes its contiguous chunk; cumulative sequence lengths describe
    # the global stream. thd otherwise: each sample is zigzag-sliced, concatenated, padded to a
    # multiple of tensor-parallel size times pad multiplier (padding as an extra segment), and the
    # cumulative lengths (device tensor and host tuple) are scaled by CP size to express the
    # original lengths. Then stores cumulative sequence lengths, their host copy and the maximum
    # segment length, and gives the tokens a leading batch dimension of one. Pad token id is zero.
    # Multi-LoRA: when adapter slots are given they must be sorted (assertion error otherwise); per
    # adapter token counts are the sums of post-slicing sample lengths per slot, with any padding
    # tokens added to the last sample's slot, stored as adapter token counts (int32 on GPU).
    # Optional extras: position ids (arange per sample) and witness ids go through the same slicing
    # and padding as tokens (not supported with allgather CP). Multimodal inputs are merged across
    # samples: per key, tensors are concatenated along the first dimension (samples with none
    # skipped), keys ending in positions are first shifted by the sample's start offset within the
    # concatenated unsliced tokens, and a second mapping records each key's per-sample item counts.
    # KNOWN ISSUE: the docstring omits several parameters and returned keys; the sample token
    # lengths used for multi-LoRA counts are not defined on the thd allgather path (guarded by the
    # assertion that rejects that combination).
    """
    Generate a CP-ready micro-batch with packed sequence parameters.

    Steps:
    - Fetch raw fields via iterator.
    - Save original token tensors under "unconcat_tokens".
    - Slice tokens into two batches for Context Parallelism (CP), concatenate, and pad to a configurable multiple.
    - Build cu_seqlens and `PackedSeqParams` with T-H-D layout (T: sequence length, H: attention heads, D: head dimension).

    Args:
        data_iterator: Iterator providing micro-batch data.
        keys: List of keys to fetch from the iterator.
        pad_multiplier: Multiplier for padding size calculation (default: 128).

    Returns a dict including:
    - "tokens": torch.LongTensor of shape [1, T_padded] on the current CUDA device
    - "unconcat_tokens": list[torch.LongTensor] for the micro-batch before CP slicing/concat
    - "packed_seq_params": PackedSeqParams with T-H-D settings (cu_seqlens on CUDA, dtype=int)
    - "input_loss_masks": same shape as "tokens"; 1 where the token is a supervised response token.
      Consumers that derive labels (Megatron with labels=None) shift it themselves.
    Plus any other requested keys forwarded from the iterator.
    """

    parallel_state = get_parallel_state()

    assert "tokens" in keys
    # get_batch consumes adapter_slots itself (per-adapter token counts below);
    # fetch it here so callers don't have to know. None for non-multi-LoRA runs.
    if "adapter_slots" not in keys:
        keys = [*keys, "adapter_slots"]
    batch = data_iterator.get_next(keys)

    if "dynamic_global_batch_size" in data_iterator.rollout_data:
        batch["dynamic_global_batch_size"] = data_iterator.rollout_data["dynamic_global_batch_size"]

    for unit_key in ("loss_fn", "loss_fn_config"):
        if unit_key in data_iterator.rollout_data:
            batch[unit_key] = data_iterator.rollout_data[unit_key]

    # No-op safety net if batches reach get_batch without rollout-level preprocessing.
    expand_multimodal_rollout_data_in_place(batch, qkv_format=qkv_format)

    tokens = batch["tokens"]
    # use 0 as the pad token id should be fine?
    pad_token_id = 0
    pad_size = parallel_state.tp.size * pad_multiplier
    pad = 0

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    cp_size = parallel_state.cp.size

    if qkv_format == "bshd":
        max_seqlen = batch["max_seq_lens"][0]
        assert max([t.size(0) for t in tokens]) <= max_seqlen

        if allgather_cp:
            assert batch.get("adapter_slots") is None, "allgather CP is currently not supported with multi-LoRA: "
            assert max_seqlen % cp_size == 0, f"max_seqlen {max_seqlen} not divisible by cp_size {cp_size}"
            local_len = max_seqlen // cp_size
            start = parallel_state.cp.rank * local_len
            tokens = [
                F.pad(t, (0, max_seqlen - t.size(0)), value=pad_token_id)[start : start + local_len] for t in tokens
            ]
        else:
            tokens = [slice_with_cp(t, pad_token_id, qkv_format, max_seqlen) for t in tokens]
        sample_token_lengths = [t.size(0) for t in tokens]
        tokens = torch.stack(tokens)

    elif qkv_format == "thd":
        cp_rank = parallel_state.cp.rank

        if allgather_cp:
            assert batch.get("adapter_slots") is None, "allgather CP is currently not supported with multi-LoRA: "
            # DSA mode: concatenate all sequences first, then slice once with CP.
            # We also pad the *global* concatenated stream to make per-rank batches equal.
            cu_seqlens_list: list[int] = [0]
            for t in tokens:
                cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

            tokens = torch.cat(tokens, dim=0)

            # Pad global stream so (1) divisible by cp_size (equal batches),
            # (2) divisible by pad_size (reduce fragmentation).
            global_pad_size = cp_size * pad_size
            pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

            cu_seqlens_host = tuple(cu_seqlens_list)
            cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=torch.cuda.current_device())
            tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
        else:
            tokens = [slice_with_cp(t, pad_token_id, qkv_format) for t in tokens]
            sample_token_lengths = [t.size(0) for t in tokens]

            cu_seqlens = [0]
            for t in tokens:
                cu_seqlens.append(cu_seqlens[-1] + t.size(0))

            tokens = torch.cat(tokens)

            # Always pad to reduce memory fragmentation and maybe make the computation faster
            pad = (pad_size - tokens.size(0) % pad_size) % pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens.append(cu_seqlens[-1] + pad)

            # thd requires the cu_seqlens to be of the origin length
            cu_seqlens_host = tuple(boundary * cp_size for boundary in cu_seqlens)
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        tokens = tokens.unsqueeze(0)

        batch["cu_seqlens"] = cu_seqlens
        # the same boundaries on the host, for consumers that must not sync the device to read them
        batch["cu_seqlens_host"] = cu_seqlens_host
        batch["max_seqlen"] = max_seqlen
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    # Multi-LoRA: compute per-adapter token counts from post-CP per-sample lengths.
    # NOTE: allgather CP is currently not supported
    adapter_slots = batch.get("adapter_slots")
    if adapter_slots is not None:
        assert all(
            adapter_slots[i] <= adapter_slots[i + 1] for i in range(len(adapter_slots) - 1)
        ), f"adapter_slots not sorted in micro-batch: {adapter_slots}"
        n_adapters = data_iterator.rollout_data["n_adapters"]
        total_tokens = tokens.numel()
        counts = torch.zeros(n_adapters, dtype=torch.int32, device=torch.cuda.current_device())
        for slot, length in zip(adapter_slots, sample_token_lengths, strict=True):
            counts[slot] += length
        counts[adapter_slots[-1]] += total_tokens - counts.sum().item()
        batch["adapter_token_counts"] = counts

    batch["tokens"] = tokens

    def _compute_transform_like_token_ids(ids_list: list):
        # [FUNCTION DOC] HELPER: Applies to a per-sample list of id tensors the same layout as the
        # tokens, with pad value zero: asserts allgather CP is off; bshd slices each sample and
        # stacks rows; thd slices, concatenates, right-pads by the same pad count as the tokens, and
        # adds a leading batch dimension. Any other format raises a not-implemented error.
        assert not allgather_cp, "allgather CP is not supported for FSDP"
        if qkv_format == "bshd":
            ids = [slice_with_cp(p, 0, qkv_format, max_seqlen) for p in ids_list]
            ids = torch.stack(ids)
        elif qkv_format == "thd":
            ids = [slice_with_cp(p, 0, qkv_format) for p in ids_list]
            ids = torch.cat(ids)
            if pad != 0:
                ids = F.pad(ids, (0, pad), value=0)
            ids = ids.unsqueeze(0)
        else:
            raise NotImplementedError
        return ids

    if get_position_ids:
        position_ids_list = []
        for t in batch["unconcat_tokens"]:
            seq_len = t.size(0)
            pos_ids = torch.arange(seq_len, device=t.device, dtype=torch.long)
            position_ids_list.append(pos_ids)

        batch["position_ids"] = _compute_transform_like_token_ids(position_ids_list)

    if (witness_ids := batch.get("witness_ids")) is not None:
        batch["witness_ids"] = _compute_transform_like_token_ids(witness_ids)

    input_loss_masks = _prepare_loss_masks(
        batch["loss_masks"],
        batch["total_lengths"],
        batch["response_lengths"],
        qkv_format=qkv_format,
        max_seqlen=max_seqlen,
        pad=pad,
        allgather_cp=allgather_cp,
    )
    assert (
        input_loss_masks.shape == tokens.shape
    ), f"input_loss_masks.shape: {input_loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["input_loss_masks"] = input_loss_masks

    # Process multimodal training tensors if present
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        sample_offsets = [0]
        for t in batch["unconcat_tokens"]:
            sample_offsets.append(sample_offsets[-1] + t.size(0))
        multimodal_data = {}  # key -> concatenated tensor
        multimodal_num_items = {}  # key -> list of item counts per sequence
        for i, mm_input_dict in enumerate(multimodal_train_inputs):
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    if key.endswith("_positions"):
                        mm_tensor = mm_tensor + sample_offsets[i]
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                        multimodal_num_items[key] = [mm_tensor.size(0)]
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
                        multimodal_num_items[key].append(mm_tensor.size(0))
        batch["multimodal_train_inputs"] = multimodal_data
        batch["multimodal_num_items"] = multimodal_num_items

    return batch


class DataIterator:
    # [CLASS DOC] Cursor over a rollout dictionary of per-sample lists. State: the rollout data, an
    # optional fixed micro-batch size, an optional list of index lists (one per micro-batch), and a
    # position counter. Exactly one of size or index lists must be set for fetching to work; the
    # counter advances by the size (fixed mode) or by one (indexed mode) per fetch.
    """Micro-batch iterator over rollout dicts.

    Supports either fixed contiguous micro-batches or an explicit per-step
    index schedule (for dynamic batch sizing / sequence-length balancing).
    """

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_size: int | None = None,
        micro_batch_indices: list[list[int]] | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the arguments and sets the position to zero. Asserts that the micro
        # batch size and the index lists are not both given (both may be none).
        """Initialize an iterator over `rollout_data`.

        Args:
            rollout_data: Dict of per-sample fields for the local step.
            micro_batch_size: Fixed contiguous slice size when not using dynamic scheduling.
            micro_batch_indices: Explicit indices per micro-batch when using dynamic balancing.
                Must be mutually exclusive with `micro_batch_size`.
        """
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        assert micro_batch_size is None or micro_batch_indices is None
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        # [FUNCTION DOC] Returns a dictionary from each requested key to the list of that field's
        # values for the next micro-batch, or none for keys absent from the rollout data. Indexed
        # mode: picks the entries listed in the current index list. Fixed mode: takes the contiguous
        # slice of micro-batch size starting at the position (assertion error with details if it
        # runs past the end). The position then advances (by one in indexed mode, by the size in
        # fixed mode) even if no key was found. Raises a type error if neither mode is configured.
        """Return the next micro-batch for the requested keys.

        - If `micro_batch_indices` is provided, selects rows according to the current
          index list for each requested key.
        - Otherwise, slices a contiguous adapter batch of size `micro_batch_size` starting
          at the current offset.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                if self.micro_batch_indices is not None:
                    indices = self.micro_batch_indices[self.offset]
                    batch[key] = [vals[i] for i in indices]
                else:
                    assert self.offset + self.micro_batch_size <= len(
                        vals
                    ), f"offset: {self.offset}, micro_batch_size: {self.micro_batch_size}, len(vals): {len(vals)}"
                    batch[key] = vals[self.offset : self.offset + self.micro_batch_size]

        if self.micro_batch_indices is not None:
            self.offset += 1
        else:
            self.offset += self.micro_batch_size
        return batch

    def reset(self) -> "DataIterator":
        # [FUNCTION DOC] Sets the position back to zero and returns the same iterator.
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def get_num_rollouts(args: Namespace, rollout_data: RolloutBatch, num_steps: int) -> list[int]:
    # [FUNCTION DOC] Returns the list of rollout counts, one per training step. If the rollout data
    # carries an explicit list under the number-of-rollouts key, returns it as is; otherwise returns
    # a list of length num_steps filled with the dynamic global batch size from the rollout data, or
    # the configured global batch size when absent.
    """Per-step rollout counts (total across DP); one entry per training step."""
    if "num_rollouts" in rollout_data:
        return rollout_data["num_rollouts"]
    return [rollout_data.get("dynamic_global_batch_size", args.global_batch_size)] * num_steps


def get_data_iterator(
    args: Namespace,
    model: torch.nn.Module | Sequence[torch.nn.Module],
    rollout_data: RolloutBatch,
) -> tuple[list[DataIterator], list[int]]:
    # [FUNCTION DOC] Returns (list of data iterators, list of micro-batch counts per step). First
    # runs the in-place multimodal expansion on the rollout data. The model argument is unused.
    # Precomputed schedule: if the rollout data already has micro-batch indices, asserts the
    # dynamic-global-batch-size flag agrees with the presence of a dynamic size, builds one indexed
    # iterator per virtual pipeline stage and returns the stored micro-batch counts.
    # Otherwise: local global batch size = global batch size (dynamic if present, with an info log if
    # it differs from configured) divided by effective data-parallel size; steps = local samples
    # divided by that (remainder dropped). Fixed mode: for multi-LoRA data (adapter slots present),
    # the local batch must divide by the micro batch size (value error otherwise); each step has
    # local batch / micro batch size micro-batches; iterators slice contiguously.
    # Dynamic mode (requires a max-tokens-per-GPU setting): per step, the minimum micro-batch count
    # for the step's sample lengths under a budget of max tokens times CP size is computed; counts
    # are all-reduced with MAX over the data-parallel group so ranks agree; with a virtual pipeline,
    # counts are rounded down to a multiple of the microbatch group size per virtual stage but at
    # least one; then for each step the samples are partitioned into that many micro-batches
    # balanced by length (sizes need not be equal), indices are made global by adding the step
    # start, and for multi-LoRA data each micro-batch is sorted by adapter slot. Asserts every local
    # sample is covered. Iterators use the explicit index lists (one per virtual stage).
    # KNOWN ISSUE: the lower clamp of one can leave the count not divisible by the group size when
    # the group size exceeds one and the count is smaller than the group size.
    """
    Create iterators and a micro-batch schedule for a rollout step.

    - If `use_dynamic_batch_size` is False, splits into fixed-size contiguous
      micro-batches of `micro_batch_size`.
    - If True, computes the number of micro-batches per local step based on
      `max_tokens_per_gpu` and per-sample lengths, all-reduces to a DP-wide
      maximum, optionally enforces divisibility for Virtual Pipeline Parallelism (VPP), and builds a balanced
      index schedule to equalize token counts across micro-batches.

    Returns `(data_iterators, num_microbatches)` where:
    - `data_iterators`: list of `DataIterator`, one per VPP stage (size 1 if VPP disabled)
    - `num_microbatches`: list[int], one per local step in the rollout (length = steps)
    """
    expand_multimodal_rollout_data_in_place(rollout_data, qkv_format=args.qkv_format)

    parallel_state = get_parallel_state()

    if "micro_batch_indices" in rollout_data:
        assert args.use_dynamic_global_batch_size == ("dynamic_global_batch_size" in rollout_data)
        micro_batch_indices = rollout_data["micro_batch_indices"]
        data_iterator = [
            DataIterator(rollout_data, micro_batch_indices=micro_batch_indices) for _ in range(parallel_state.vpp_size)
        ]
        return data_iterator, rollout_data["num_microbatches"]

    dp_size = parallel_state.effective_dp.size
    dp_group = parallel_state.effective_dp.group
    vpp_size = parallel_state.vpp_size
    microbatch_group_size_per_vp_stage = parallel_state.microbatch_group_size_per_vp_stage

    cp_size = parallel_state.cp.size

    num_local_samples = len(rollout_data["total_lengths"])
    assert args.use_dynamic_global_batch_size == ("dynamic_global_batch_size" in rollout_data)
    global_batch_size = rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
    num_local_gbs = global_batch_size // dp_size
    num_steps_per_rollout = num_local_samples // num_local_gbs

    if global_batch_size != args.global_batch_size:
        logger.info(
            f"Using dynamic global_batch_size={global_batch_size} (original={args.global_batch_size}), "
            f"num_local_samples={num_local_samples}, num_steps_per_rollout={num_steps_per_rollout}"
        )

    def _generate_data_iterator(rollout_data, micro_batch_size, micro_batch_indices=None):
        # [FUNCTION DOC] HELPER: Creates one new iterator per virtual pipeline stage over the same
        # rollout data with the given size or index lists (all iterators share the lists).
        data_iterator = []
        for _ in range(vpp_size):
            data_iterator.append(DataIterator(rollout_data, micro_batch_size, micro_batch_indices))
        return data_iterator

    if not args.use_dynamic_batch_size:
        if "adapter_slots" in rollout_data and num_local_gbs % args.micro_batch_size != 0:
            raise ValueError(
                "A multi-LoRA local batch must be divisible by --micro-batch-size; "
                f"got local_batch_size={num_local_gbs}, micro_batch_size={args.micro_batch_size}. "
                "Use --use-dynamic-batch-size or choose compatible adapter batch shapes."
            )
        num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
        data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
    else:
        assert args.max_tokens_per_gpu is not None
        # calculate the number of mirobatches for each step
        samples = rollout_data["total_lengths"]
        assert len(samples) == num_local_samples
        num_microbatches = []
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            num_microbatches.append(
                get_minimum_num_micro_batch_size(samples[start:end], args.max_tokens_per_gpu * cp_size)
            )

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        GeneralPGUtil.create(dp_group).all_reduce(num_microbatches, dp_group, op=dist.ReduceOp.MAX)

        if vpp_size > 1:
            # vpp requies the number of microbatches to be divisible by vpp_size
            num_microbatches = torch.clamp(
                num_microbatches // microbatch_group_size_per_vp_stage * microbatch_group_size_per_vp_stage,
                min=1,
            )

        num_microbatches = num_microbatches.tolist()

        # balance the each micro batch
        samples = rollout_data["total_lengths"]
        # balance the number of mirobatches across steps
        micro_batch_indices = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            samples = rollout_data["total_lengths"][start:end]
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
                # Multi-LoRA: microbatches must be contiguous-by-slot for the
                # grouped GEMM's per-adapter token-count math.
                if "adapter_slots" in rollout_data:
                    partitions[j].sort(key=lambda index: rollout_data["adapter_slots"][index])
            micro_batch_indices.extend(partitions)

        assert len(set(sum(micro_batch_indices, []))) == num_local_samples

        data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)

    return (
        data_iterator,
        num_microbatches,
    )
