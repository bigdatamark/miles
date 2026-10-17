# [PACKAGE DOC] DSA (indexed sparse) attention: re-exports the public API. From the indexer module:
# causal range helpers (plain and compressed), indexer logits (two layouts), indexer top-k scores
# and the lightning indexer; from the sparse attention module the sparse attention function; from
# the topk module the factory returning the top-k selection function. The tilelang subpackage
# holds the underlying kernels. The export list contains exactly these names.
from miles.kernels.attention.dsa.indexer import (
    causal_ranges,
    causal_ranges_compressed,
    indexer_logits,
    indexer_logits_sbhd,
    indexer_topk_scores,
    lighting_indexer,
)
from miles.kernels.attention.dsa.sparse_attention import sparse_attention
from miles.kernels.attention.dsa.topk import get_dsa_topk_fn

__all__ = [
    "causal_ranges",
    "causal_ranges_compressed",
    "get_dsa_topk_fn",
    "indexer_logits",
    "indexer_logits_sbhd",
    "indexer_topk_scores",
    "lighting_indexer",
    "sparse_attention",
]
