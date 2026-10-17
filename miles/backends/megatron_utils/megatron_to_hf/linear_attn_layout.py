# [MODULE DOC]
# Layout helpers for linear-attention layers (gated delta net, KDA): a frozen record of head
# counts and dimensions with derived sizes, a reader of that record from the HF config (with a
# cached by-checkpoint variant), and row-permutation functions between the HF flat layout
# [all queries, all keys, all values] and Megatron's group-major layout (one block per key head
# containing that head's query, key and the value heads that share it), so tensor-parallel chunking
# of the output dimension lands on whole head groups. All tensor functions return contiguous copies.
"""Weight layout of a linear-attention layer (GDN, KDA): head counts and dims, and the Megatron row
layout of the GDN q/k/v projection and short conv.

HF stores them flat, ``[Q_all, K_all, V_all]``. Megatron shards the output dimension in contiguous
chunks, so the Megatron copy is *group-major*: one block per key head ``g`` holding ``[q_g, k_g, v_g]``
with ``R = num_v_heads // num_k_heads`` value heads in ``v_g``. A TP chunk is then exactly the heads
that rank owns, for any TP size, and the projection output feeds the conv without a copy.
"""

from dataclasses import dataclass
from functools import cache

import torch

from miles.utils.hf_utils.config import load_hf_config


@dataclass(frozen=True)
class LinearAttnHeads:
    # [CLASS DOC]
    # Immutable record: number of key heads, number of value heads, per-head key dimension and
    # per-head value dimension. Query heads equal key heads. The derived properties give the sizes
    # used when splitting the fused projection.
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int

    @property
    def v_per_k(self) -> int:
        # [FUNCTION DOC] Value heads per key head (integer division); asserts the value head count is
        # a multiple of the key head count.
        assert self.num_v_heads % self.num_k_heads == 0
        return self.num_v_heads // self.num_k_heads

    @property
    def key_dim(self) -> int:
        # [FUNCTION DOC] Total key width: key heads times key head dimension (same for queries).
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        # [FUNCTION DOC] Total value width: value heads times value head dimension.
        return self.num_v_heads * self.head_v_dim

    @property
    def qkv_dim(self) -> int:
        # [FUNCTION DOC] Row count of the fused query/key/value projection: twice the key width plus
        # the value width.
        return 2 * self.key_dim + self.value_dim

    @property
    def group_qkv_dim(self) -> int:
        # [FUNCTION DOC] Rows of one group-major block: query head dim + key head dim + value heads
        # per key head times value head dim.
        return 2 * self.head_k_dim + self.v_per_k * self.head_v_dim

    @property
    def group_value_dim(self) -> int:
        # [FUNCTION DOC] Value rows within one group block: value heads per key head times value
        # head dim.
        return self.v_per_k * self.head_v_dim

    def local(self, tp_size: int) -> "LinearAttnHeads":
        # [FUNCTION DOC] Per-rank head record for a tensor-parallel size: key heads and value heads
        # both divided by the size (key heads must divide evenly, asserted; value heads divisibility
        # is implied only when value heads are a multiple of key heads); dims unchanged.
        assert self.num_k_heads % tp_size == 0, f"{self.num_k_heads} key heads do not split across TP={tp_size}"
        return LinearAttnHeads(
            self.num_k_heads // tp_size, self.num_v_heads // tp_size, self.head_k_dim, self.head_v_dim
        )


def gdn_heads(hf_config) -> LinearAttnHeads:
    # [FUNCTION DOC] Reads the linear-attention key heads, value heads, key head dim and value head
    # dim from the config (or from its nested text config when one exists) and returns the record.
    # Attribute errors if the config lacks these fields.
    """Head layout of a Qwen3.5 / Qwen3-Next GDN layer; VLM configs nest it under ``text_config``."""
    text_config = getattr(hf_config, "text_config", hf_config)
    return LinearAttnHeads(
        num_k_heads=text_config.linear_num_key_heads,
        num_v_heads=text_config.linear_num_value_heads,
        head_k_dim=text_config.linear_key_head_dim,
        head_v_dim=text_config.linear_value_head_dim,
    )


def group_major(blocks: list[torch.Tensor], heads: LinearAttnHeads) -> torch.Tensor:
    # [FUNCTION DOC] Takes tensors whose first axis is key heads times per-group rows (head-major),
    # reshapes each to (key heads, rows per group, remaining dims), concatenates them along the
    # per-group axis in the given order, and flattens back to rows. Result: for each key head g,
    # its rows from block 0, then block 1, then block 2... Returns a contiguous tensor.
    """Interleave head-major blocks (dim 0 is ``num_k_heads * rows_per_group``) group by group."""
    rest = blocks[0].shape[1:]
    grouped = [block.reshape(heads.num_k_heads, -1, *rest) for block in blocks]
    return torch.cat(grouped, dim=1).reshape(-1, *rest).contiguous()


def split_group_major(weight: torch.Tensor, rows_per_group: list[int], heads: LinearAttnHeads) -> list[torch.Tensor]:
    # [FUNCTION DOC] Reshapes the weight to (key heads, rows per group, remaining dims), splits the
    # per-group axis into consecutive chunks of the given sizes, and returns each chunk flattened
    # back to head-major rows (contiguous). The sizes must sum to the per-group row count.
    """Inverse of :func:`group_major`."""
    rest = weight.shape[1:]
    parts = weight.reshape(heads.num_k_heads, -1, *rest).split(rows_per_group, dim=1)
    return [part.reshape(-1, *rest).contiguous() for part in parts]


def _qkv_rows(heads: LinearAttnHeads) -> list[int]:
    # [FUNCTION DOC] HELPER: Per-group row sizes of the fused projection: query head dim, key head
    # dim, and group value rows.
    return [heads.head_k_dim, heads.head_k_dim, heads.group_value_dim]


def qkv_flat_to_group_major(weight: torch.Tensor, heads: LinearAttnHeads) -> torch.Tensor:
    # [FUNCTION DOC] Asserts the row count equals the fused width, splits the rows into all-queries
    # (key width), all-keys (key width) and all-values (value width) and interleaves them group by
    # group. Returns the group-major tensor of the same shape.
    """``[Q_all, K_all, V_all]`` (HF) -> group-major ``[q, k, v]``."""
    assert weight.shape[0] == heads.qkv_dim, (weight.shape, heads)
    return group_major(list(weight.split([heads.key_dim, heads.key_dim, heads.value_dim])), heads)


def qkv_group_major_to_flat(weight: torch.Tensor, heads: LinearAttnHeads) -> torch.Tensor:
    # [FUNCTION DOC] Inverse of the flat-to-group-major function: asserts the fused row count, splits
    # each group into query, key, value rows and concatenates all queries, then all keys, then all
    # values. Returns a tensor of the same shape in the HF flat layout.
    assert weight.shape[0] == heads.qkv_dim, (weight.shape, heads)
    return torch.cat(split_group_major(weight, _qkv_rows(heads), heads))


@cache
def gdn_heads_of(hf_checkpoint: str) -> LinearAttnHeads:
    # [FUNCTION DOC] Memoized per checkpoint path: loads the HF config from the path (disk read) and
    # returns its linear-attention head record.
    return gdn_heads(load_hf_config(hf_checkpoint))
