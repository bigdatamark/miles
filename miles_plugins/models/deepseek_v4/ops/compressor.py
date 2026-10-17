# [MODULE DOC] KV compressor for DeepSeek-V4 attention. Pools every group of compress_ratio consecutive
# tokens into one compressed key/value entry using a learned softmax gate over the group (plus a
# learned per-position bias), then applies RMS norm, rotary embedding on the trailing rope dims, and
# optional rotation and fp8 quantization-aware-training simulation. Supports ratio 4 (with overlapping
# groups, doubled width) and ratio 128 (no overlap). Provides a regular batch-major path (with
# contiguous context-parallel support) and a packed variable-length (THD) path.
import einops
import torch
import torch.nn as nn
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn import Linear

from miles_plugins.models.deepseek_v4.ops.cp_utils import all_gather_cp, get_freqs_cis_for_cp
from miles_plugins.models.deepseek_v4.ops.norm import RMSNorm
from miles_plugins.models.deepseek_v4.ops.precision_aligned_ops import linear_bf16_fp32
from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat
from miles_plugins.models.deepseek_v4.ops.rope import apply_rotary_emb, wrapped_precompute_freqs_cis
from miles_plugins.models.deepseek_v4.ops.thd_utils import ThdLayout, batch_of_row, compressed_cu_seqlens
from miles_plugins.models.deepseek_v4.ops.utils import rotate_activation


def _overlap_transform(tensor: torch.Tensor, *, compress_ratio: int, head_dim: int, value=0) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Builds the overlapped group tensor for the batch-major layout. Input is
    # batch, group count, ratio, 2*head_dim (the doubled width of the overlap mode). Output has the
    # ratio axis doubled and width head_dim, filled with the given value by default. The second
    # half of the channels of each group goes into the last ratio slots of the same group; the first
    # half of the channels of the previous group goes into the first ratio slots of the current group.
    # The first group has no predecessor, so its first slots keep the fill value. Returns a new tensor;
    # the input is not modified.
    """Overlap-transform for compress_ratio=4: for each token group of size ``ratio``,
    split into (first_half, second_half) halves along ``head_dim`` and re-arrange
    them across a doubled ratio axis (`2 * ratio`), shifting the first half by one
    group so that adjacent groups overlap by ``ratio`` positions.
    """
    b, s, _, _ = tensor.size()
    new_tensor = tensor.new_full((b, s, 2 * compress_ratio, head_dim), value)
    new_tensor[:, :, compress_ratio:] = tensor[:, :, :, head_dim:]
    new_tensor[:, 1:, :compress_ratio] = tensor[:, :-1, :, :head_dim]
    return new_tensor


def _overlap_transform_thd(
    tensor: torch.Tensor, *, compress_ratio: int, head_dim: int, is_first: torch.Tensor, value=0
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Packed-stream variant of the overlap transform. Groups are flat along
    # dim 0 and the batch axis comes after the ratio axis. The current group's second channel half
    # fills the last ratio slots; the previous flat group's first channel half (a circular shift by one
    # along the group axis) fills the first ratio slots, and rows flagged as first-of-segment are reset
    # to the fill value so no data leaks across segment boundaries. Returns a new tensor.
    """Overlap-transform for packed THD groups.

    Same rearrangement as :func:`_overlap_transform`, but groups are flat along dim 0
    and a segment's first group has no predecessor to pull from.

    Args:
        tensor: ``[total_comp, ratio, batch, 2 * head_dim]``
        is_first: ``[total_comp]`` bool, True for the first group of each segment.
    """
    total_comp, ratio, bsz, _ = tensor.size()
    new_tensor = tensor.new_full((total_comp, 2 * ratio, bsz, head_dim), value)
    new_tensor[:, ratio:] = tensor[:, :, :, head_dim:]
    prev = torch.roll(tensor[:, :, :, :head_dim], shifts=1, dims=0)
    prev[is_first] = value
    new_tensor[:, :ratio] = prev
    return new_tensor


class DeepSeekV4Compressor(nn.Module):
    # [CLASS DOC] Learned gated-pooling compressor. Owns a float32 position bias of shape ratio by
    # coff*head_dim (coff is 2 when overlapping, i.e. ratio 4, else 1), two bias-free bfloat16 linear
    # projections (key/value and gate) from hidden size to coff*head_dim, and an RMS norm over head_dim.
    # Optional context-parallel group makes the batch-major path gather groups across ranks.
    def __init__(
        self,
        config: TransformerConfig,
        head_dim: int,
        compress_ratio: int,
        rotate: bool,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ):
        # [FUNCTION DOC] Reads hidden size, rope head dim (qk_pos_emb_head_dim), and norm epsilon from
        # the config and builds the submodules. head_dim must be 128 or 512, rope head dim must be 64,
        # compress_ratio must be 4 or 128, epsilon must be 1e-6, and the compress rotary base must be
        # 160000; each is enforced with an assertion. Overlap mode is on only for ratio 4. fp8 QAT
        # simulation is enabled when the config has fp8 set. cp size and rank come from the group
        # (1 and 0 when absent). The position bias is marked to stay in float32 under Megatron's
        # mixed-precision casting. The bias is created uninitialized (empty); initialization is
        # expected to come from checkpoint loading.
        super().__init__()

        dim = config.hidden_size
        rope_head_dim = config.qk_pos_emb_head_dim
        norm_eps = config.layernorm_epsilon

        assert head_dim in {128, 512}
        assert rope_head_dim == 64
        assert compress_ratio in {4, 128}
        assert norm_eps == 1e-6

        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = head_dim - rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap
        self.use_fp8_qat = config.fp8 is not None

        self.cp_group = cp_group
        self.cp_size = cp_group.size() if cp_group is not None else 1
        self.cp_rank = cp_group.rank() if cp_group is not None else 0

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.linear_wkv = Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.linear_wgate = Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.norm = RMSNorm(self.head_dim, norm_eps)

        mark_keep_in_fp32(self.ape)

        base = config.csa_compress_rotary_base
        assert rope_head_dim == 64
        assert base == 160000

    def overlap_transform_raw(self, tensor: torch.Tensor, value=0):
        # [FUNCTION DOC] Thin wrapper applying the batch-major overlap transform with this module's
        # ratio and head_dim; no context-parallel communication. Returns the transformed tensor.
        """Raw overlap transform without CP handling."""
        return _overlap_transform(tensor, compress_ratio=self.compress_ratio, head_dim=self.head_dim, value=value)

    def overlap_transform_with_cp(self, tensor: torch.Tensor, value=0) -> torch.Tensor:
        # [FUNCTION DOC] Overlap transform that is correct across contiguous context-parallel ranks.
        # With a single rank it just applies the raw transform. Otherwise it all-gathers the local
        # groups along the group axis (so the previous-group dependency at the rank boundary is
        # satisfied), applies the raw transform on the full group axis, and slices back this rank's
        # contiguous block of groups. Fill value is 0 for key/value and negative infinity for scores.
        # Communication: one differentiable all-gather over the cp group.
        # KNOWN ISSUE: the docstring's output shape says the last dim is coff*d and ratio unchanged,
        # but after the transform the ratio axis is doubled and the last dim is head_dim. Also the
        # gather helper is wrapped in a single-entry cache keyed on its tensor arguments, which is
        # suspicious for tensors that change every call.
        """
        Overlap transform with CP support.

        Args:
            tensor: [bsz, G_local, ratio, coff*d]
            value: Fill value for overlap transform (0 for kv, -inf for score)

        Returns:
            [bsz, G_local, ratio, coff*d]
        """
        if self.cp_size == 1:
            return self.overlap_transform_raw(tensor, value)

        tensor = all_gather_cp(tensor, dim=1, cp_group=self.cp_group)

        tensor = self.overlap_transform_raw(tensor, value)

        G_local = tensor.shape[1] // self.cp_size
        start = self.cp_rank * G_local
        return tensor[:, start : start + G_local, :, :]

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Batch-major compression. Input is batch, local sequence length, hidden dim;
        # local length must be a multiple of the ratio (and of twice the ratio when cp size > 1) and
        # at least the ratio. Output is batch, local length / ratio, head_dim. Steps: project to
        # key/value and gate scores with a bf16-in float32-out linear; split the sequence axis into
        # groups of ratio; add the position bias to scores; in overlap mode apply the cp-aware
        # overlap transform (fill 0 for values, negative infinity for scores); softmax the scores over
        # the within-group axis and take the weighted sum of values; cast back to the input dtype and
        # RMS-normalize; apply rotary embedding in place to the last rope_head_dim channels using
        # frequencies computed for the global length, sliced for this rank and strided by the ratio
        # (position of each group is its first token); finally if rotate is set, apply the activation
        # rotation and (with fp8 QAT) simulate fp8 quantization with block 128; otherwise with fp8 QAT
        # clone and simulate fp8 with block 64 on only the non-rope channels. Asserts that parameter
        # dtypes are float32 bias and bf16 weights.
        assert self.ape.dtype == torch.float32
        assert self.linear_wkv.weight.dtype == torch.bfloat16
        assert self.linear_wgate.weight.dtype == torch.bfloat16

        bsz, seqlen_local, _ = x.size()
        ratio, overlap, _ = self.compress_ratio, self.overlap, self.head_dim
        dtype = x.dtype

        assert (seqlen_local >= ratio) and (seqlen_local % ratio == 0), f"{seqlen_local=} {ratio=}"
        if self.cp_size > 1:
            assert seqlen_local % (ratio * 2) == 0

        kv = linear_bf16_fp32(x, self.linear_wkv.weight)
        score = linear_bf16_fp32(x, self.linear_wgate.weight)

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if overlap:
            kv = self.overlap_transform_with_cp(kv, 0)
            score = self.overlap_transform_with_cp(score, float("-inf"))

        score_softmax = score.softmax(dim=2)
        kv = (kv * score_softmax).sum(dim=2)

        kv = self.norm(kv.to(dtype))

        freqs_cis = wrapped_precompute_freqs_cis(
            self.config,
            self.rope_head_dim,
            self.config.csa_compress_rotary_base,
            False,
            seqlen_local * self.cp_size,
            x.device,
        )
        freqs_cis = get_freqs_cis_for_cp(freqs_cis, seqlen_local, self.cp_size, self.cp_group, stride=ratio)

        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)

        if self.rotate:
            kv = rotate_activation(kv)
            if self.use_fp8_qat:
                kv = fp8_simulate_qat(kv, 128)
        else:
            if self.use_fp8_qat:
                kv = kv.clone()
                kv[..., : self.nope_head_dim] = fp8_simulate_qat(kv[..., : self.nope_head_dim], 64)

        return kv

    def _forward_thd(self, x: torch.Tensor, thd_layout: ThdLayout):
        # [FUNCTION DOC] HELPER: Packed-stream compression; same math as the batch-major path
        # (projection, bias, optional overlap, gated softmax pooling, norm, rotary, optional rotation
        # and fp8 QAT simulation) but group boundaries follow per-segment cumulative lengths. Two input
        # modes. Unpacked-groups mode: the number of compressed rows per segment is derived from the
        # cumulative sequence lengths; returns None if the total is zero; each group's token rows are
        # gathered by index (segment start plus group index times ratio plus offset in group). Pre-grouped
        # mode (layout carries group ids): rows are already contiguous per group so the projections are
        # just reshaped; raises a value error if the layout lacks max_seqlen. Group ids give each
        # group's index within its segment and drive the overlap first-group mask; padding groups carry
        # a negative id which is clamped to zero for the rotary position lookup. Rotary frequencies are
        # computed up to max_seqlen (pre-grouped) or the total token count, and indexed at group index
        # times ratio. Here softmax runs over dim 1 (the ratio axis) since the layout is group, ratio,
        # batch, width.
        """Compress a THD-packed stream, grouping tokens within each segment.

        The trailing ``seqlen % compress_ratio`` tokens of a segment get no compressed
        entry and rely on the sliding window instead. This matches inference, where a
        decode token sitting in an incomplete buffer has no compressed entry either, and
        avoids the train/inference mismatch that padding to compress_ratio would introduce.

        Args:
            x: [total, batch, dim] packed SBHD layout, or [c_cap * ratio, batch, dim] already
                grouped when the layout carries compressed_group_ids.
            thd_layout: packed-stream layout.
        Returns:
            k [total_comp, batch, head_dim], or None when no segment reaches compress_ratio.
        """
        cu_seqlens = thd_layout.cu_seqlens
        compressed_group_ids = thd_layout.compressed_group_ids
        max_seqlen = thd_layout.max_seqlen

        assert self.ape.dtype == torch.float32
        assert self.linear_wkv.weight.dtype == torch.bfloat16
        assert self.linear_wgate.weight.dtype == torch.bfloat16

        total_tokens = x.size(0)
        ratio, overlap = self.compress_ratio, self.overlap
        dtype = x.dtype

        pre_grouped = compressed_group_ids is not None
        if pre_grouped:
            if max_seqlen is None:
                raise ValueError(
                    "Pre-grouped compressor input needs max_seqlen: its group ids address "
                    "positions past the compacted row count."
                )
            local_pos = compressed_group_ids
            total_comp = local_pos.size(0)
        else:
            cu_seqlens_compressed = compressed_cu_seqlens(cu_seqlens, ratio)
            total_comp = int(cu_seqlens_compressed[-1])
            if total_comp == 0:
                return None

        kv = linear_bf16_fp32(x, self.linear_wkv.weight)
        score = linear_bf16_fp32(x, self.linear_wgate.weight)

        if pre_grouped:
            # Compaction already laid rows out as [g * ratio, (g + 1) * ratio); no gather.
            kv = kv.unflatten(0, (total_comp, ratio))
            score = score.unflatten(0, (total_comp, ratio)) + self.ape.view(1, ratio, 1, -1)
        else:
            batch_ids = batch_of_row(cu_seqlens_compressed, total_comp)
            local_pos = torch.arange(total_comp, device=x.device) - cu_seqlens_compressed[batch_ids]
            gather_idx = (cu_seqlens[batch_ids] + local_pos * ratio).unsqueeze(1) + torch.arange(
                ratio, device=x.device
            )
            kv = kv[gather_idx]
            score = score[gather_idx] + self.ape.view(1, ratio, 1, -1)

        if overlap:
            is_first = local_pos == 0
            kv = _overlap_transform_thd(kv, compress_ratio=ratio, head_dim=self.head_dim, is_first=is_first, value=0)
            score = _overlap_transform_thd(
                score,
                compress_ratio=ratio,
                head_dim=self.head_dim,
                is_first=is_first,
                value=float("-inf"),
            )

        score_softmax = score.softmax(dim=1)
        kv = (kv * score_softmax).sum(dim=1)

        kv = self.norm(kv.to(dtype))

        freqs_cis = wrapped_precompute_freqs_cis(
            self.config,
            self.rope_head_dim,
            self.config.csa_compress_rotary_base,
            False,
            max_seqlen if pre_grouped else total_tokens,
            x.device,
        )
        # Capacity padding carries -1; clamp keeps the gather in range and those rows are
        # dropped by seq_to_rank_row anyway.
        rope_positions = local_pos.clamp(min=0) * ratio if pre_grouped else local_pos * ratio
        apply_rotary_emb(
            kv.transpose(0, 1)[..., -self.rope_head_dim :],
            freqs_cis.index_select(0, rope_positions),
        )

        if self.rotate:
            kv = rotate_activation(kv)
            if self.use_fp8_qat:
                kv = fp8_simulate_qat(kv, 128)
        else:
            if self.use_fp8_qat:
                kv = kv.clone()
                kv[..., : self.nope_head_dim] = fp8_simulate_qat(kv[..., : self.nope_head_dim], 64)

        return kv

    def forward(self, x: torch.Tensor, thd_layout: ThdLayout | None = None):
        # [FUNCTION DOC] Dispatches: with a packed layout, delegates to the packed path; otherwise
        # transposes sequence-major input to batch-major, runs the batch-major path, and transposes the
        # result back to compressed-sequence-major.
        """
        Args:
            x: [seqlen, batch, dim] SBHD layout (Megatron standard); [total, batch, dim]
                when thd_layout is given.
            thd_layout: packed-stream layout, or None when unpacked.
        Returns:
            k: [seqlen // compress_ratio, batch, head_dim] SBHD layout; [total_comp, batch,
                head_dim], or None when no segment reaches compress_ratio, for THD packing.
        """
        if thd_layout is not None:
            return self._forward_thd(x, thd_layout)
        x_bshd = einops.rearrange(x, "s b d -> b s d")
        k_bshd = self.forward_raw(x_bshd)
        k = einops.rearrange(k_bshd, "b sc d -> sc b d")
        return k
