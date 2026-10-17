# [MODULE DOC] HELPER: runtime monkey-patches (applied by calling the install function once) that
# make Megatron-Bridge's Qwen3-VL model correct for packed (THD, a single row of concatenated
# sequences) batches, with and without context parallelism (CP). Three independent patches:
# (1) the text rotary embedding forward tolerates a packed-sequence-params keyword; (2) the model
# forward is wrapped so multimodal rotary positions (3 axes) are computed per packed segment and fed
# to the bridge through a patched rope-index function, and the bridge's internal re-sharding
# helper is turned into an identity when Miles has already sharded the row for CP; (3) the vision
# embedding all-gather autograd function is wrapped so it accepts its CP group as a keyword.
# All patches are idempotent (guarded by marker attributes) and silently skip when the bridge
# modules cannot be imported. Per-call state is passed through a thread-local holder holding the
# pending positions and the CP-local context, always cleared after the model forward.
# CP layout: ranks hold zigzag chunks (rank r owns chunks r and 2*cp-1-r of 2*cp equal partitions of
# every segment); the cumulative lengths of the packed params are the FULL padded boundaries.
"""miles-side fix for Qwen3-VL THD packed mRoPE positions (no Megatron-Bridge edit).

Bridge's Qwen3VLModel.forward resets position_ids=None and recomputes via the
module-level get_rope_index over the whole [1, total] packed row, so MRoPE positions
don't restart per packed segment (wrong for multimodal). We hijack that call: stash
correct per-segment positions and have a patched get_rope_index return them.
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

_PATCHED = "_miles_qwen3_vl_thd_mrope_patched"
_tls = threading.local()


def install_qwen3_vl_packed_mrope_patch() -> None:
    # [FUNCTION DOC] Public entry point: applies the three patches in order (rotary signature,
    # model forward plus rope index, vision-embedding all-gather keyword). Safe to call repeatedly.
    # Mutates classes and module attributes of the Megatron-Bridge package in process.
    _patch_rotary_signature()
    _patch_model_forward_and_rope_index()
    _patch_allgather_vision_embeddings_kwarg()


def _patch_allgather_vision_embeddings_kwarg() -> None:
    # [FUNCTION DOC] HELPER: replace the bridge model module's vision-embedding all-gather symbol
    # with a shim class whose apply accepts the CP group as a keyword (torch autograd functions
    # reject keyword arguments) and forwards it positionally to the original. Does nothing if the
    # bridge module cannot be imported, the symbol is absent, or it is already the shim.
    """megatron-bridge 0.5.0 calls AllGatherVisionEmbeddings.apply(..., cp_group=...) in the
    Qwen3-VL vision_dp_when_cp path, but torch.autograd.Function.apply rejects keyword args
    (TypeError: apply() takes no keyword arguments). Replace the symbol with a shim whose
    .apply accepts cp_group as a kwarg and forwards it positionally.
    """
    try:
        model_mod = importlib.import_module("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model")
    except ImportError:
        return
    orig = getattr(model_mod, "AllGatherVisionEmbeddings", None)
    if orig is None or getattr(orig, "_miles_kwarg_shim", False):
        return

    class _AllGatherVisionEmbeddingsKwargShim:
        # [CLASS DOC] HELPER: stand-in for the original autograd function class; carries a marker
        # attribute so the patch is not applied twice, and exposes only a static apply.
        _miles_kwarg_shim = True

        @staticmethod
        def apply(input, seqlens_on_cp_ranks, cp_group=None):
            # [FUNCTION DOC] HELPER: call the original function's apply with the input, the
            # per-CP-rank sequence lengths and the CP group (default none) as positional arguments;
            # returns its result.
            return orig.apply(input, seqlens_on_cp_ranks, cp_group)

    model_mod.AllGatherVisionEmbeddings = _AllGatherVisionEmbeddingsKwargShim


def _patch_rotary_signature() -> None:
    # [FUNCTION DOC] HELPER: for the dense and MoE Qwen3-VL text rotary embedding classes in the
    # bridge's text model module, wrap their forward so an extra packed-sequence-params keyword is
    # accepted and dropped before calling the original. Skips classes that are absent or already
    # marked patched (marker stored on the class itself); returns silently if import fails.
    # Let the rotary embedding forward tolerate the packed_seq_params kwarg.
    try:
        text_model = importlib.import_module("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model")
    except ImportError:
        return
    for name in ("Qwen3VLTextRotaryEmbedding", "Qwen3VLMoETextRotaryEmbedding"):
        cls = getattr(text_model, name, None)
        if cls is None or cls.__dict__.get(_PATCHED, False):
            continue
        _orig = cls.forward

        def _make(orig):
            # [FUNCTION DOC] HELPER: factory binding the original forward (needed so each loop
            # iteration captures its own original); returns the wrapper function.
            def _fwd(self, *args, packed_seq_params=None, **kwargs):
                # [FUNCTION DOC] HELPER: wrapper forward that discards the packed-sequence-params
                # keyword and delegates everything else to the original forward.
                return orig(self, *args, **kwargs)

            return _fwd

        cls.forward = _make(_orig)
        setattr(cls, _PATCHED, True)


def _patch_model_forward_and_rope_index() -> None:
    # [FUNCTION DOC] HELPER: the main patch. Returns silently if the bridge model module cannot be
    # imported or is already marked patched. Steps: (1) replace the module-level rope-index
    # function with a wrapper that returns the thread-local pending positions when set, else
    # delegates to the original; (2) install the identity wrapper for the packed-sequence
    # preprocessing function; (3) log a warning if the bridge model class lacks native CP-local
    # vision-embed selection (CP runs with vision tokens would misplace embeddings); (4) unless
    # the model class is missing or already patched, replace its forward with a wrapper that parses
    # the packed batch, builds per-segment positions and the CP-local context, stores them in the
    # thread-local, calls the original forward, and always clears the thread-local afterwards;
    # (5) mark the class and module as patched.
    try:
        model_mod = importlib.import_module("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model")
    except ImportError:
        return
    if getattr(model_mod, _PATCHED, False):
        return

    orig_get_rope_index = model_mod.get_rope_index

    def patched_get_rope_index(*args, **kwargs):
        # [FUNCTION DOC] HELPER: if positions were stashed for this forward, return them as the
        # pair (positions, None) without computing; otherwise call the original with all arguments.
        pending = getattr(_tls, "packed_positions", None)
        if pending is not None:
            return pending, None
        return orig_get_rope_index(*args, **kwargs)

    model_mod.get_rope_index = patched_get_rope_index

    # Under CP, miles pre-shards the THD row (slice_with_cp), but the bridge forward re-shards
    # internally via preprocess_packed_seqs (it expects the FULL input). When miles already
    # sharded, make that internal call an identity that returns miles' packed_seq_params (so CP
    # attention still sees the full cu_seqlens) instead of re-splitting the already-local data.
    _patch_preprocess_packed_seqs_identity(model_mod)

    Qwen3VLModel = getattr(model_mod, "Qwen3VLModel", None)
    # The bridge selects CP-local vision embeds natively; warn when running an old one.
    if Qwen3VLModel is not None and not hasattr(Qwen3VLModel, "_cp_local_vision_embed_indices"):
        logger.warning(
            "megatron-bridge Qwen3VLModel lacks native CP-local vision-embed selection; "
            "CP runs with vision tokens will mis-place vision embeddings. "
            "Apply the matching Megatron-Bridge patch (radixark/Megatron-Bridge PR #9)."
        )

    if Qwen3VLModel is None or Qwen3VLModel.__dict__.get(_PATCHED, False):
        setattr(model_mod, _PATCHED, True)
        return

    orig_forward = Qwen3VLModel.forward

    def patched_forward(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: wrapper around the model forward (all arguments passed through).
        # Parses the packed THD batch (none for non-packed), computes per-segment multimodal
        # positions (none means use the bridge's dense path), prepares the CP-local context, stores
        # whichever are not none in the thread-local, runs the original forward and returns its
        # result; the stashed positions and context are reset to none in a finally block.
        parsed = _parse_packed_thd(args, kwargs)
        packed = _build_packed_positions(self, parsed, kwargs, orig_get_rope_index)
        ctx = _prepare_cp_local_context(parsed)
        if packed is not None:
            _tls.packed_positions = packed
        if ctx is not None:
            _tls.cp_local = ctx
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            _tls.packed_positions = None
            _tls.cp_local = None

    Qwen3VLModel.forward = patched_forward
    setattr(Qwen3VLModel, _PATCHED, True)
    setattr(model_mod, _PATCHED, True)


def _patch_preprocess_packed_seqs_identity(model_mod) -> None:
    # [FUNCTION DOC] HELPER: wrap the bridge model module's packed-sequence preprocessing function
    # (which splits the full input for CP internally) so that, while a CP-local context is stashed,
    # it returns the already-local input ids together with Miles' packed-sequence params instead of
    # re-sharding; otherwise it calls the original. No-op when the function is absent or already
    # wrapped (marker attribute on the wrapper).
    orig = getattr(model_mod, "preprocess_packed_seqs", None)
    if orig is None or getattr(orig, "_miles_identity_wrapped", False):
        return

    def wrapped(input_ids, attention_mask, *args, **kwargs):
        # [FUNCTION DOC] HELPER: identity-with-params when a CP-local context is stashed (returns
        # the input ids unchanged and the stashed packed-sequence params); otherwise delegates to
        # the original with all arguments. The attention mask is ignored in the identity case.
        ctx = getattr(_tls, "cp_local", None)
        if ctx is not None:
            # Already CP-local: skip re-sharding and hand back miles' full-cu packed_seq_params.
            return input_ids, ctx["psp"]
        return orig(input_ids, attention_mask, *args, **kwargs)

    wrapped._miles_identity_wrapped = True
    model_mod.preprocess_packed_seqs = wrapped


class _PackedTHD(NamedTuple):
    # [CLASS DOC] HELPER: immutable parsed view of a single-row packed THD batch. Fields: the
    # packed-sequence params object; the flat 1-D token ids; the query cumulative sequence lengths
    # as a host list and as the original tensor; the local row length (token count on this rank);
    # CP size and CP rank.
    """Parsed view of a THD single-row packed batch (see ``_parse_packed_thd``)."""

    psp: object
    flat: torch.Tensor
    cu: list  # cu_seqlens_q as a host-side list (one GPU->host copy, shared by both paths)
    cu_t: torch.Tensor
    local_len: int
    cp_size: int
    cp_rank: int


def _parse_packed_thd(args, kwargs):
    # [FUNCTION DOC] HELPER: extract the packed-batch view from the model forward's arguments.
    # Input ids come from the keyword or else the first positional argument. Returns none unless
    # the packed-sequence params exist with the "thd" format, the ids are 2-D with batch size 1,
    # and the query cumulative lengths exist with at least two entries. Otherwise flattens the ids,
    # reads CP size / rank, copies the cumulative lengths to the host (one device sync) and returns
    # the packed view.
    """Extract the preamble shared by the mRoPE-position and CP-context paths, doing the one
    GPU->host copy of cu_seqlens_q a single time. Returns None when this is not a THD packed
    ``[1, T]`` batch, in which case both callers run the dense / unchanged path."""
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    psp = kwargs.get("packed_seq_params")
    if psp is None or getattr(psp, "qkv_format", None) != "thd":
        return None
    if input_ids is None or input_ids.dim() != 2 or input_ids.shape[0] != 1:
        return None
    cu_t = getattr(psp, "cu_seqlens_q", None)
    if cu_t is None or cu_t.numel() < 2:
        return None
    flat = input_ids.reshape(-1)
    cp_size, cp_rank = _cp_size_rank()
    return _PackedTHD(psp, flat, cu_t.detach().cpu().tolist(), cu_t, flat.numel(), cp_size, cp_rank)


def _prepare_cp_local_context(parsed):
    # [FUNCTION DOC] HELPER: decide whether the row is already CP-sharded. Returns a mapping with
    # the packed-sequence params (key psp) only when a packed view exists, CP size > 1, the first
    # cumulative length is zero and the last equals CP size times the local row length; otherwise
    # none (the bridge then runs unchanged).
    """When CP has already pre-sharded this THD row, capture miles' packed_seq_params so the
    preprocess_packed_seqs identity wrapper can hand them back unchanged (the bridge does
    CP-local vision-embed selection natively). Returns None for the non-CP / full-input cases
    (bridge runs unchanged)."""
    if parsed is None or parsed.cp_size <= 1 or parsed.cu[0] != 0:
        return None
    if parsed.cu[-1] != parsed.cp_size * parsed.local_len:
        return None
    return {"psp": parsed.psp}


def _cp_size_rank():
    # [FUNCTION DOC] HELPER: return (context-parallel world size, rank) from Megatron's parallel
    # state; on any error (e.g. parallel state not initialised) returns (1, 0).
    """Context-parallel (size, rank); (1, 0) when CP is unavailable."""
    try:
        from megatron.core import parallel_state as _ps

        return _ps.get_context_parallel_world_size(), _ps.get_context_parallel_rank()
    except Exception:
        return 1, 0


def _natural_to_zigzag_slice(t, cp_size, cp_rank, dim):
    # [FUNCTION DOC] HELPER: take this rank's two chunks (indices rank and 2*cp-1-rank, in that
    # order) out of the 2*cp equal chunks of t along the given dim, concatenated along that dim.
    # Chunk size is the integer quotient of the length by 2*cp; no divisibility check, so a
    # non-divisible length would silently drop trailing elements. Returns a new tensor.
    """Slice a full-length tensor into this rank's zigzag (load-balanced ring-attn) CP chunks.

    Mirrors miles.backends.training_utils.data.context_parallel.slice_with_cp / natural_to_zigzag_slice:
    rank r owns chunks [r, 2*cp_size-1-r] of the 2*cp_size equal partitions along ``dim``.
    """
    total = t.shape[dim]
    num_chunks = 2 * cp_size
    chunk = total // num_chunks
    idxs = [cp_rank, 2 * cp_size - 1 - cp_rank]
    return torch.cat([t.narrow(dim, i * chunk, chunk) for i in idxs], dim=dim)


def _cp_allgather_unzigzag(flat, cu, cp_size):
    # [FUNCTION DOC] HELPER: collective. All-gathers every CP rank's local flat row over the CP
    # group (same length on every rank assumed) and reassembles the full natural-order row via the
    # reassembly helper. Returns the full row, or none if some segment length is not divisible by
    # 2*cp. Must be called by all CP ranks.
    """Reconstruct the full THD packed row from this rank's zigzag chunks.

    Under CP, ``flat`` holds only chunks [cp_rank, 2*cp-1-cp_rank] of every segment, and
    ``cu`` (== psp.cu_seqlens_q) gives the FULL padded per-segment boundaries. All-gather the
    per-rank rows over the CP group and de-interleave each segment back to natural order.
    Returns None (caller falls back to dense) if a segment is not divisible by 2*cp.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state as _ps

    group = _ps.get_context_parallel_group()
    gathered = [torch.empty_like(flat) for _ in range(cp_size)]
    dist.all_gather(gathered, flat.contiguous(), group=group)
    return _reassemble_full_row(gathered, cu, cp_size)


def _reassemble_full_row(gathered, cu, cp_size):
    # [FUNCTION DOC] HELPER: pure function (no collectives). Allocates a zero row of the full
    # length (last cumulative length) with the dtype and device of the gathered rows. For each
    # non-empty segment (length from the cumulative list), returns none if the length is not a
    # multiple of 2*cp; else with chunk size length/(2*cp) and the segment's local offset (segment
    # start divided by cp), copies for every rank r its first local chunk to natural chunk r and
    # its second local chunk to natural chunk 2*cp-1-r. Returns the full row.
    """De-interleave per-rank zigzag rows back into the full natural-order packed row.

    ``gathered[r]`` is rank r's local row; ``cu`` (full padded per-segment boundaries, i.e.
    miles' cu_seqlens * cp) locates each segment. For segment ``i`` of full length ``L`` (a
    multiple of 2*cp), rank r contributed chunk r and chunk 2*cp-1-r, each of size L/(2*cp),
    at local offset cu[i]//cp. Pure (no collectives) so it is unit-testable. Returns None if a
    segment is not divisible by 2*cp (caller falls back to the dense path).
    """
    full = torch.zeros(cu[-1], dtype=gathered[0].dtype, device=gathered[0].device)
    for i in range(len(cu) - 1):
        seg_full = cu[i + 1] - cu[i]
        if seg_full <= 0:
            continue
        if seg_full % (2 * cp_size) != 0:
            return None
        c = seg_full // (2 * cp_size)
        local_off = cu[i] // cp_size  # this segment's offset within a per-rank (local) row
        for r in range(cp_size):
            mir = 2 * cp_size - 1 - r
            full[cu[i] + r * c : cu[i] + (r + 1) * c] = gathered[r][local_off : local_off + c]
            full[cu[i] + mir * c : cu[i] + (mir + 1) * c] = gathered[r][local_off + c : local_off + 2 * c]
    return full


def _segment_positions(model, flat, cu, cu_t, kwargs, orig_get_rope_index):
    # [FUNCTION DOC] HELPER: compute multimodal rotary positions for every non-empty segment of a
    # full (unsharded) packed row. Inputs: model (provides spatial merge size and the image, video
    # and vision-start token ids), flat ids, host and tensor cumulative lengths, the forward's
    # keyword arguments (image and video grid tables, each a count by 3 table), and the original
    # rope-index function. Returns a list of position tensors of shape 3 by 1 by segment length.
    # Media counting: finds every vision-start token that is not the last token, looks at the token
    # after it, assigns it to a segment by bucketing against the cumulative lengths, and counts
    # image and video starts per segment (one device-to-host copy). Segments with no media get a
    # plain 0..L-1 range expanded over the 3 axes (a non-contiguous view); others call the original
    # rope-index function with the segment's ids (batch of one), the next image / video grid rows
    # (offsets advance by each segment's counts), and no attention mask, and trim the result to the
    # segment length.
    """Per-segment MRoPE positions for a full (unsharded) packed row `flat` with boundaries `cu`.

    Returns a list of [3, 1, seg_len] tensors (one per non-empty segment), text segments get
    a linear 0..L range, media segments call get_rope_index with the matching grid slice.
    """
    image_grid_thw = kwargs.get("image_grid_thw")
    video_grid_thw = kwargs.get("video_grid_thw")
    merge = model.config.spatial_merge_size
    img_id, vid_id, vstart = model.image_token_id, model.video_token_id, model.vision_start_token_id

    # Vectorized media count per segment (one GPU->host copy total, no per-segment .item()).
    num_segments = len(cu) - 1
    img_counts = [0] * num_segments
    vid_counts = [0] * num_segments
    starts = torch.nonzero(flat == vstart, as_tuple=False).flatten()
    starts = starts[starts + 1 < flat.numel()]
    if starts.numel() > 0:
        toks = flat[starts + 1]
        seg_idx = torch.bucketize(starts, cu_t, right=True) - 1  # [start,end) -> segment index
        img_counts = torch.bincount(seg_idx[toks == img_id], minlength=num_segments).cpu().tolist()
        vid_counts = torch.bincount(seg_idx[toks == vid_id], minlength=num_segments).cpu().tolist()

    img_off = vid_off = 0
    segments = []
    for i, (start, end) in enumerate(zip(cu[:-1], cu[1:], strict=False)):
        if end <= start:
            continue
        seg = flat[start:end]
        ic, vc = img_counts[i], vid_counts[i]
        if ic == 0 and vc == 0:
            pos = torch.arange(seg.numel(), dtype=seg.dtype, device=seg.device).view(1, 1, -1).expand(3, 1, -1)
        else:
            pos, _ = orig_get_rope_index(
                merge,
                img_id,
                vid_id,
                vstart,
                seg.unsqueeze(0),
                image_grid_thw=_slice(image_grid_thw, img_off, ic),
                video_grid_thw=_slice(video_grid_thw, vid_off, vc),
                attention_mask=None,
            )
            pos = pos[:, :, : seg.numel()]
        img_off += ic
        vid_off += vc
        segments.append(pos)
    return segments


def _build_packed_positions(model, parsed, kwargs, orig_get_rope_index):
    # [FUNCTION DOC] HELPER: produce the positions tensor (3 by 1 by local row length, contiguous)
    # to feed the bridge, or none to let the bridge use its dense path. Returns none when there is
    # no packed view, or the first cumulative length is nonzero. Three cases: (a) the last
    # cumulative length equals the local length (non-CP or single chunk): positions per segment,
    # concatenated along the token axis (none if no non-empty segment); (b) CP size > 1 and the last
    # cumulative length equals CP size times the local length: all-gather the full row (collective;
    # none, with a debug log, if a segment is not divisible by 2*cp), compute full per-segment
    # positions, re-slice each segment into this rank's zigzag chunks along the token axis and
    # concatenate; (c) any other layout: debug log and none.
    """Per-segment MRoPE positions for a THD packed batch; None to run the dense path."""
    if parsed is None or parsed.cu[0] != 0:
        return None
    flat, cu, cu_t = parsed.flat, parsed.cu, parsed.cu_t
    local_len, cp_size, cp_rank = parsed.local_len, parsed.cp_size, parsed.cp_rank

    # Non-CP (or single chunk): cu_seqlens_q already describes this row exactly.
    if cu[-1] == local_len:
        segments = _segment_positions(model, flat, cu, cu_t, kwargs, orig_get_rope_index)
        return torch.cat(segments, dim=2).contiguous() if segments else None

    # CP + THD packing: cu_seqlens_q gives the FULL padded per-segment boundaries (miles
    # builds cu_seqlens * cp_size), while this row holds only this rank's zigzag chunks
    # (full_len / cp). Reconstruct the full row across the CP group, build full per-segment
    # MRoPE positions, then re-slice each segment into this rank's zigzag layout so the
    # positions line up with the tokens that slice_with_cp produced.
    if cp_size > 1 and cu[-1] == cp_size * local_len:
        full_flat = _cp_allgather_unzigzag(flat, cu, cp_size)
        if full_flat is None:
            logger.debug("qwen3_vl packed mRoPE: CP segment not divisible by 2*cp; dense path")
            return None
        segments = _segment_positions(model, full_flat, cu, cu_t, kwargs, orig_get_rope_index)
        if not segments:
            return None
        local_segments = [_natural_to_zigzag_slice(p, cp_size, cp_rank, dim=2) for p in segments]
        return torch.cat(local_segments, dim=2).contiguous()

    # Unrecognized layout -> let the dense get_rope_index run.
    logger.debug(
        "qwen3_vl packed mRoPE: cu_seqlens_q (%d) vs local len (%d), cp=%d; dense path",
        cu[-1],
        local_len,
        cp_size,
    )
    return None


def _slice(grid, offset, count):
    # [FUNCTION DOC] HELPER: rows offset .. offset+count of a grid table, or none if the table is
    # none or the count is zero.
    return None if grid is None or count == 0 else grid[offset : offset + count]
