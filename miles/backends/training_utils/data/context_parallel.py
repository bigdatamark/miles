# [MODULE DOC] Context-parallel (CP) helpers for per-sample sequences. Layout ("zigzag"): each
# sample's sequence is cut into 2*cp_size equal chunks and rank r holds chunk r and chunk
# 2*cp_size-1-r, which balances causal-attention work. Because a logit at position i predicts token
# i+1, response log-probs for a sample live at logit positions from (prompt_length - 1) up to
# (total_length - 1), exclusive, so offsets for logits are the token offsets minus one. Functions
# compute those offsets, slice masks/tokens/log-probs into the local layout, gather local pieces
# back into full response tensors with a differentiable sum-reduction over zero-padded pieces, and
# redistribute between the allgather-CP layout (contiguous chunks) and the zigzag layout.
# "thd" means all samples are packed into one token dimension; "bshd" means batch rows padded to a
# maximum sequence length.
import logging
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)


def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    cp_rank: int | None = None,
    cp_size: int | None = None,
):
    # [FUNCTION DOC] Computes which parts of one sample a CP rank owns. Inputs: total length (prompt
    # plus response), response length, qkv format (default thd), the padded max sequence length
    # (required, else assertion error, for bshd), and optional explicit CP rank and size (both must
    # be given to override; otherwise both come from the current parallel state). Asserts CP size
    # is greater than one. Chunk size = ceiling of (total length, or max sequence length for bshd)
    # divided by twice the CP size. The rank's two chunks are [rank*chunk, (rank+1)*chunk) and
    # [(2*cp-rank-1)*chunk, (2*cp-rank)*chunk). The logit range of each chunk is the chunk
    # intersected with [prompt_length - 1, total_length - 1); the matching token range is the logit
    # range shifted by one. A chunk with an empty intersection gets the empty range (0, 0) for both
    # logits and tokens (an empty slice keeps gradient flow). Returns (chunk size, pair of chunk
    # ranges, pair of logit ranges, pair of token ranges); all offsets are measured from the start
    # of the prompt (sample start).
    """
    All offsets start from the begining of the prompt.

    ``cp_rank`` / ``cp_size`` default to this process's parallel state; pass them
    explicitly to compute another rank's offsets outside the process group.
    """
    if cp_rank is None or cp_size is None:
        parallel_state = get_parallel_state()
        cp_rank = parallel_state.cp.rank
        cp_size = parallel_state.cp.size
    assert cp_size > 1

    prompt_length = total_length - response_length
    if qkv_format == "thd":
        chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    else:
        assert max_seq_len is not None, "max_seq_len must be provided for qkv_format=bshd"
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # the offset of 2 chunks
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)

    # the offset of 2 logits, note that the logits need a "-1".
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))

    # when the sequence is empty, make an empty slice to continue the gradient flow.
    if logits_0[0] < logits_0[1]:
        token_0 = (logits_0[0] + 1, logits_0[1] + 1)
    else:
        logits_0 = (0, 0)
        token_0 = (0, 0)

    if logits_1[0] < logits_1[1]:
        token_1 = (logits_1[0] + 1, logits_1[1] + 1)
    else:
        logits_1 = (0, 0)
        token_1 = (0, 0)

    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)


def _slice_loss_mask_for_local_cp(
    total_length: int,
    response_length: int,
    loss_mask: torch.Tensor,
    qkv_format: str,
    max_seq_len: int | None,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Takes the response-length loss mask of one sample and returns the part
    # that corresponds to this rank's two token ranges (offsets converted from sample-start to
    # response-start by subtracting the prompt length), concatenated in order. Requires CP size
    # greater than one (through the offset helper).
    """Slice a per-sample response loss mask into this CP rank's local zigzag layout."""
    prompt_length = total_length - response_length
    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_seq_len
    )
    mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
    mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
    return torch.cat([mask_0, mask_1], dim=0)


def slice_loss_masks_for_local_cp(
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> list[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Thin alias that calls the local response loss mask function with the
    # arguments reordered (it takes the masks after the lengths). Returns its result.
    """Backward-compatible wrapper for local CP response mask slicing."""
    return get_local_response_loss_masks(total_lengths, response_lengths, loss_masks, qkv_format, max_seq_lens)


def get_sum_of_sample_mean(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
    *,
    denominators: list[torch.Tensor] | torch.Tensor | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    # [FUNCTION DOC] Returns a reduction function over a concatenated per-token tensor (first
    # dimension is all samples' response tokens back to back). By default denominators are each
    # sample's full loss-mask sum (may also be given as a list or tensor, keyword-only, one per
    # sample). Without CP the tensor is split by response lengths and the masks are the given masks;
    # with CP greater than one, each mask is first sliced to this rank's local part and the tensor is
    # split by those local lengths (so denominators must still be the full-sample counts). The
    # returned function is either (default) the sum over samples of masked sum divided by the
    # denominator clamped to at least one, or, when the per-token-loss flag is true, the plain masked
    # sum over all tokens (denominators unused). The qkv format and max sequence lengths are only
    # used for CP slicing.
    """Calculate correct sample mean for CP; ``denominators`` overrides each
    sample's own ``loss_mask.sum()`` (e.g. pass ``rollout_mask_sums`` for
    per-rollout means)."""
    if denominators is None:
        denominators = [m.sum() for m in loss_masks]

    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size
    if cp_size == 1:

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Without CP. Splits the tensor by response lengths and returns
            # the sum over samples of the masked sum divided by the sample's denominator (at least
            # one). Raises a value error if the lists differ in length.
            return sum(
                [
                    (x_i * loss_mask_i).sum() / torch.clamp_min(denominator, 1)
                    for x_i, loss_mask_i, denominator in zip(
                        x.split(response_lengths, dim=0), loss_masks, denominators, strict=True
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Without CP. Sum of masked values over all samples.
            return sum(
                [
                    (x_i * loss_mask_i).sum()
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=True)
                ]
            )

    else:
        cp_chunk_lengths = []
        chunked_loss_masks = []
        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=True)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunked_loss_masks.append(
                _slice_loss_mask_for_local_cp(total_length, response_length, loss_mask, qkv_format, max_seq_len)
            )
            cp_chunk_lengths.append(chunked_loss_masks[i].size(0))

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: With CP. Splits the local tensor by the local chunk lengths and
            # returns the sum over samples of the masked local sum divided by the sample's full
            # denominator (at least one); partial results from CP ranks add up to the full mean.
            return sum(
                [
                    (x_i * chunked_loss_mask).sum() / torch.clamp_min(denominator, 1)
                    for x_i, chunked_loss_mask, denominator in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, denominators, strict=True
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: With CP. Sum of masked local values over all samples.
            return sum(
                [
                    (x_i * chunked_loss_mask).sum()
                    for x_i, chunked_loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, strict=True
                    )
                ]
            )

    return sum_of_sample_mean if not calculate_per_token_loss else sum_of_token


def get_local_response_loss_masks(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Without CP (CP size one) returns the given masks unchanged (same list). With CP
    # returns a new list in which each mask is sliced to this rank's local two token ranges (the
    # per-sample max sequence length is used only for bshd).
    """Return response loss masks aligned with this rank's local log-probs."""
    parallel_state = get_parallel_state()
    if parallel_state.cp.size == 1:
        return loss_masks

    local_masks = []
    for i, (total_length, response_length, loss_mask) in enumerate(
        zip(total_lengths, response_lengths, loss_masks, strict=True)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
        local_masks.append(
            _slice_loss_mask_for_local_cp(total_length, response_length, loss_mask, qkv_format, max_seq_len)
        )

    return local_masks


def all_gather_with_cp(
    tensor: torch.Tensor,
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Rebuilds one sample's full response tensor from this rank's local per-logit
    # values. Without CP returns the input. With CP: the local tensor (first dimension is the sum of
    # this rank's two logit ranges) is split into the first-chunk and second-chunk parts (asserting
    # the second part has the expected size); a full tensor of the response length is made by
    # placing the non-empty parts at their positions (position = logit offset minus
    # (prompt_length - 1)) between zero tensors that require gradients (four cases: both empty, only
    # first, only second, both present); asserts the result has the response length; then sums it
    # over the CP group with a differentiable all-reduce, so each position receives the value from
    # the one rank that owns it. Returns the full tensor (trailing dimensions preserved).
    """
    Gather tensors across all ranks in the context parallel group.
    The first dimension of the output tensor will be the `response_length`.
    """
    parallel_state = get_parallel_state()
    cp_group = parallel_state.cp.group
    cp_size = parallel_state.cp.size

    if cp_size == 1:
        return tensor

    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_seq_len
    )

    prompt_length = total_length - response_length

    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    assert chunk_1.shape[0] == logits_offset[1][1] - logits_offset[1][0]

    def zero(len: int) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: A zero tensor with the given first-dimension length, the trailing
        # shape, element type and device of the input tensor, requiring gradients.
        return torch.zeros(
            [len] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=True,
        )

    # logprob should be within the range of [prompt_length - 1, total_length - 1]
    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        # all empty
        full_tensor = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        # only first chunk
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        full_tensor = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        # only second chunk
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_1, right], dim=0)
    else:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)

    assert full_tensor.shape[0] == response_length, f"Expected {response_length}, got {full_tensor.shape}"
    full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
    return full_tensor


def slice_with_cp(
    tokens: torch.Tensor,
    pad_value: tuple[int, float, Callable],
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    parallel_state: object | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Pads and slices one sample's token-like tensor (first dimension is the
    # sequence) into this rank's local zigzag layout. Inputs: the tensor; a pad value that is either
    # a number or a callable taking (tensor, pad length) and returning the padded tensor; qkv format
    # (default thd); the padded max sequence length (required for bshd, assertion error otherwise);
    # and an optional parallel state (default: the current one) supplying CP rank and size.
    # CP size one: bshd pads the end of the first dimension up to the max sequence length; thd
    # returns the input untouched. CP greater than one: chunk size = ceiling of (length, or max
    # sequence length for bshd) over twice the CP size; the tensor is padded at the end to twice the
    # CP size times the chunk size; then the rank's two chunks (chunk r and chunk 2*cp-1-r) are
    # concatenated. A numeric pad value pads only the first dimension.
    # KNOWN ISSUE: the annotation of the pad value claims a tuple of three alternatives but the
    # code uses a single number or a callable; for bshd a sequence longer than the max sequence
    # length gives a negative pad.
    """
    Slice tokens into the local zigzag CP layout.
    """
    if parallel_state is None:
        parallel_state = get_parallel_state()
    cp_rank = parallel_state.cp.rank
    cp_size = parallel_state.cp.size

    if qkv_format == "bshd":
        assert max_seq_len is not None

    def pad_tokens(tokens, pad):
        # [FUNCTION DOC] HELPER: Appends the given number of pad entries to the first dimension: via
        # the callable pad value if one was given, else with a constant fill of the numeric pad value.
        if isinstance(pad_value, Callable):
            pad_func = pad_value
            tokens = pad_func(tokens, pad)
        else:
            # pad on the first dimension
            pad_tuple = (0, 0) * (tokens.dim() - 1) + (0, pad)
            tokens = F.pad(tokens, pad_tuple, value=pad_value)
        return tokens

    if cp_size == 1:
        if qkv_format == "bshd":
            pad = max_seq_len - tokens.size(0)
            tokens = pad_tokens(tokens, pad)
        return tokens

    token_len = len(tokens)
    if qkv_format == "thd":
        chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    else:
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # pad
    pad = 2 * cp_size * chunk_size - token_len
    tokens = pad_tokens(tokens, pad)

    # get 2 chunk for thd cp
    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])


def natural_to_zigzag_slice(tensor: torch.Tensor, dim: int, cp_size: int, cp_rank: int) -> torch.Tensor:
    # [FUNCTION DOC] Takes the two chunks (index rank and index 2*cp-1-rank, out of 2*cp equal
    # parts) of the tensor along the given dimension and concatenates them along that dimension.
    # If the dimension size is not divisible by twice the CP size, prints a warning to standard
    # output (not the logger) and returns the tensor unchanged. No padding is done.
    """Slice a full-length tensor into the zigzag ring-attention CP layout.

    Rank ``cp_rank`` owns chunks ``[cp_rank, 2*cp_size - 1 - cp_rank]`` from the
    ``2*cp_size`` equal-sized partitions along ``dim``. This is the inverse of
    an all-gather over the zigzag CP layout (hence "natural → zigzag").

    Unlike :func:`slice_with_cp`, this helper does not pad — it expects the
    input to already be divisible by ``2 * cp_size`` along ``dim``. If not, it
    prints a warning and returns the tensor unchanged.
    """
    total = tensor.shape[dim]
    num_chunks = 2 * cp_size
    if total % num_chunks != 0:
        print(f"Warning: dim {dim} size {total} not divisible by 2*cp_size={num_chunks}")
        return tensor

    chunk_size = total // num_chunks
    chunk_indices = [cp_rank, 2 * cp_size - 1 - cp_rank]

    slices = [tensor.narrow(dim, idx * chunk_size, chunk_size) for idx in chunk_indices]
    return torch.cat(slices, dim=dim)


def allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits: torch.Tensor,
    args,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> None:
    # [FUNCTION DOC] In-place conversion of each named list of per-sample response tensors from the
    # allgather-CP layout (each rank holds one contiguous slice of the packed sequence; local length
    # taken from the logits' second dimension, chunk start = rank times local length) to the zigzag
    # layout. For every key: for every sample, finds the global range of its response logits (sample
    # start is zero for bshd, the running sum of total lengths for thd; range = start + prompt - 1
    # to start + total - 1); if this rank's slice overlaps it, pads the sample's local values with
    # zeros before and after to the full response length at the proper offsets, otherwise creates a
    # zero tensor requiring gradients and attached to the value's graph (adds zero times the value's
    # sum); asserts the length; concatenates all samples and sums over the CP group with a
    # differentiable all-reduce (one collective per key); then splits per sample and slices each
    # into the zigzag layout. Replaces each list in the mapping. Returns nothing.
    # KNOWN ISSUE: samples and lengths are zipped without strict checking, so mismatched list
    # lengths silently truncate.
    """Redistribute response tensors from allgather-CP layout to zigzag ring-attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits: Model output used only to determine the local sequence length
            (``logits.size(1)``). For ``bshd``, each batch row has its own
            sequence window.
        args: Configuration (needs ``qkv_format``).
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample.
    """
    parallel_state = get_parallel_state()
    cp_group = parallel_state.cp.group
    cp_rank = parallel_state.cp.rank

    logits_local_len = logits.size(1)  # logits shape: [1, T_local, ...]
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        # Reconstruct full response tensors with each rank's contiguous contribution
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            sample_start = 0 if args.qkv_format == "bshd" else seq_start
            logit_global_start = sample_start + prompt_length - 1
            logit_global_end = sample_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            if e <= s:
                # This rank has no response logprobs for this sample
                full_resp = (
                    torch.zeros(
                        (response_length, *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                        requires_grad=True,
                    )
                    + value.sum() * 0
                )
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                full_resp = F.pad(value, (0, 0) * (value.ndim - 1) + (resp_start, response_length - resp_end))

            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        # Single differentiable all-reduce to gather full response from all CP ranks
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(
            zip(all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            new_values.append(
                slice_log_prob_with_cp(full_resp, total_length, response_length, args.qkv_format, max_seq_len)
            )

        res[key] = new_values


def slice_log_prob_with_cp(
    log_prob: list[float] | torch.Tensor,
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_token_len: int | None = None,
) -> list[float] | torch.Tensor:
    # [FUNCTION DOC] Takes one sample's full response-length values (list or tensor, asserted to
    # have exactly the response length) and returns this rank's local portion. CP size one returns
    # the input. Otherwise slices the two logit ranges of this rank (converted to response-relative
    # positions by subtracting prompt_length - 1) and joins them: list concatenation for lists,
    # tensor concatenation along the first dimension for tensors. The max token length (default
    # none) is only needed for bshd.
    assert len(log_prob) == response_length

    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size

    if cp_size == 1:
        return log_prob

    prompt_length = total_length - response_length
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length, response_length, qkv_format, max_token_len
    )

    chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1) : logits_offset[0][1] - (prompt_length - 1)]
    chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1) : logits_offset[1][1] - (prompt_length - 1)]

    if isinstance(log_prob, list):
        return chunk_1 + chunk_2
    else:
        return torch.cat([chunk_1, chunk_2], dim=0)


def assemble_log_prob_from_cp(
    chunks: dict[int, torch.Tensor],
    total_length: int,
    response_length: int,
    cp_size: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Builds one sample's full response vector from a mapping of CP rank to that
    # rank's local slice. Asserts CP size is greater than one and that every rank from zero to CP
    # size minus one is present. Allocates a zero vector of the response length on the CPU with the
    # element type of the first slice, then for each rank computes its logit ranges with the same
    # offset helper (explicit rank and size) and copies consecutive parts of that rank's slice into
    # the matching response positions, skipping empty ranges; asserts each slice is fully consumed.
    # Returns the assembled CPU tensor (any device of inputs is not preserved).
    """Inverse of `slice_log_prob_with_cp`: per-rank slices back to one response.

    `chunks` maps cp_rank to that rank's slice; every rank must be present.
    Offsets come from the same helper the forward split uses.
    """
    assert cp_size > 1, "no reassembly needed at cp_size=1"
    missing = sorted(set(range(cp_size)) - set(chunks))
    assert not missing, f"cp ranks {missing} missing; cannot reassemble a partial group"

    prompt_length = total_length - response_length
    out = torch.zeros(response_length, dtype=next(iter(chunks.values())).dtype)
    for cp_rank, chunk in chunks.items():
        _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
            total_length, response_length, qkv_format, max_seq_len, cp_rank=cp_rank, cp_size=cp_size
        )
        taken = 0
        for lo, hi in logits_offset:
            start, stop = lo - (prompt_length - 1), hi - (prompt_length - 1)
            width = stop - start
            if width <= 0:
                continue
            out[start:stop] = chunk[taken : taken + width]
            taken += width
        assert taken == len(chunk), f"cp rank {cp_rank}: consumed {taken} of {len(chunk)} values"
    return out
