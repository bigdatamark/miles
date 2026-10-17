# [MODULE DOC] Per-layer embeddings (PLE) for Qwen3.8-Next. Pipeline: token ids are turned into
# sliding n-gram windows (end-of-sequence tokens stop context from crossing document boundaries),
# hashed with per-layer multipliers and modulo per-head vocabulary sizes into row ids of a large
# flat embedding table; the table is frozen, kept in pinned host memory, split into fixed shards
# that are assigned whole to tensor-parallel ranks, and read straight from the HuggingFace
# safetensors files (it is deliberately not a parameter, buffer, or checkpointed state). The ids
# reach the layer through a thread-local side channel published by model forward hooks, because
# input ids never reach transformer layers. The PLE module then projects the looked-up embeddings
# and applies a fused gate + dilated short convolution to produce an increment added to the
# hyper-connection residual streams. The hashing functions have no Megatron dependency.
"""Qwen3.8-Next PLE: n-gram hashing, token-id side channel, frozen table, layer.

Hashing is Megatron-free (unit-tested vs sglang). The side channel has no silent
default: PLE with nothing published raises. The 102GB table is host-resident,
TP-row-sharded, and deliberately NOT a checkpointed parameter.
"""

import json
import logging
import struct
import threading

import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles.kernels.embedding.ple_gate_conv import ple_gate_conv
from miles.kernels.embedding.ple_gather import gather_ple_rows

logger = logging.getLogger(__name__)


def shift_right_ignore_eos(tokens: Tensor, n: int, eos_token_id: int) -> Tensor:
    # [FUNCTION DOC] Shift a batch of token rows (batch, length) to the right by n along the length
    # axis, so each position sees the token n steps earlier, but never across an end-of-sequence
    # boundary: positions whose in-segment offset (distance from the start of the segment that
    # follows the previous EOS token; an EOS token belongs to the segment it ends) is less than n,
    # or whose source index is negative, receive the EOS id instead. Returns the same tensor object
    # when n is zero; otherwise a new tensor of the same shape and dtype.
    # Steps: mark EOS positions, running maximum for last EOS index (exclusive of the current
    # position), compute offsets, gather from clamped source indexes, select with validity mask.
    """Shift right by ``n`` without letting context cross an EOS boundary."""
    if n == 0:
        return tokens
    batch_size, seq_len = tokens.shape
    idx = torch.arange(seq_len, device=tokens.device, dtype=torch.long)

    eos_pos = torch.where(tokens == eos_token_id, idx, torch.full_like(idx, -1))
    prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
    prev_eos = torch.cat([eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]], dim=1)
    pos_in_segment = idx.unsqueeze(0) - (prev_eos + 1)

    src_idx = idx - n
    gathered = tokens.gather(dim=1, index=torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1))
    valid = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
    return torch.where(valid, gathered, tokens.new_full((), eos_token_id))


def ngram_hash_ids(
    contexts: Tensor,
    layer_multipliers: Tensor,
    head_vocab_sizes: Tensor,
    head_offsets: Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> Tensor:
    # [FUNCTION DOC] Hash n-gram windows into row ids of the flat PLE table. Inputs: contexts
    # (tokens, window), one sliding window per token whose last column is that token; the layer's
    # per-position multipliers (one per window slot); per-head vocabulary sizes and per-head row
    # offsets (each of length (ngram_size - 1) * heads_per_ngram, grouped by n-gram order 2 up to
    # ngram_size, heads_per_ngram heads each); ngram size; heads per n-gram; EOS id.
    # Algorithm: build window copies shifted right by 0..ngram_size-1 slots (EOS-aware). For each
    # order n from 2 to ngram_size: multiply the first n shifted copies by their multipliers and
    # combine them with bitwise exclusive-or (64-bit integer, wrapping multiplication), take the last
    # column, reduce modulo each head's vocabulary size for the heads of that order, and add the
    # head's row offset. Concatenate all orders along the head axis.
    # Returns an integer tensor (tokens, total heads). Note the mix is computed over the whole
    # window although only the last column is used.
    """Row ids into the flat PLE table, one per hash head."""
    shifted = [contexts]
    for shift in range(1, ngram_size):
        shifted.append(shift_right_ignore_eos(contexts, shift, eos_token_id))

    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mix = shifted[0] * layer_multipliers[0]
        for pos in range(1, ngram):
            mix = torch.bitwise_xor(mix, shifted[pos] * layer_multipliers[pos])
        ids = torch.remainder(mix[:, -1:].unsqueeze(-1), head_vocab_sizes[start:end].view(1, 1, -1))
        blocks.append((ids + head_offsets[start:end].view(1, 1, -1))[:, 0])
    return torch.cat(blocks, dim=-1)


def build_ngram_contexts(tokens: Tensor, ngram_size: int, eos_token_id: int) -> Tensor:
    # [FUNCTION DOC] Build sliding windows for one sequence of token ids (1-D; asserted). The
    # sequence is left-padded with ngram_size minus one EOS ids and unfolded with window size
    # ngram_size and stride one, so row t holds the tokens t-ngram_size+1 .. t (EOS where before the
    # start). Returns a view of shape (tokens, ngram_size).
    """``[T]`` token ids -> ``[T, ngram_size]`` sliding windows, one row per token."""
    assert tokens.dim() == 1, f"expected a 1-D token sequence, got {tuple(tokens.shape)}"
    pad = tokens.new_full((ngram_size - 1,), eos_token_id)
    return torch.cat([pad, tokens]).unfold(0, ngram_size, 1)


def build_ngram_contexts_packed(
    tokens: Tensor, cu_seqlens: Tensor | None, ngram_size: int, eos_token_id: int
) -> Tensor:
    # [FUNCTION DOC] Sliding n-gram windows for a flat packed batch. Without cumulative lengths the
    # whole token array is treated as one sequence. With them (copied to host, a device sync), each
    # sequence is windowed independently (so windows are EOS-padded at every sequence start), empty
    # sequences are skipped, and results are concatenated along the token axis. Returns
    # (tokens, ngram_size). If every sequence is empty the concatenation of nothing raises.
    """``build_ngram_contexts`` for a packed (THD) batch."""
    if cu_seqlens is None:
        return build_ngram_contexts(tokens, ngram_size, eos_token_id)
    bounds = cu_seqlens.tolist()
    parts = [
        build_ngram_contexts(tokens[lo:hi], ngram_size, eos_token_id)
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
        if hi > lo
    ]
    return torch.cat(parts, dim=0)


# Thread-local holder of the published PLE batch (pair of n-gram ids and cumulative lengths).
_state = threading.local()


def current_ple_batch():
    # [FUNCTION DOC] Return the published (n-gram ids, cumulative sequence lengths) pair for the
    # current thread. Raises a runtime error if nothing has been published (deliberately no silent
    # default, since skipping PLE would silently change logits).
    batch = getattr(_state, "batch", None)
    if batch is None:
        raise RuntimeError(
            "PLE ran with no n-gram ids published; publish_ple_batch(ngram_ids, "
            "cu_seqlens) must run before the forward (model_provider hooks do this). "
            "Skipping PLE silently would change the logits without failing."
        )
    return batch


def publish_ple_batch(ngram_ids: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> None:
    # [FUNCTION DOC] Store the n-gram ids and optional cumulative sequence lengths in the
    # thread-local side channel, replacing any previous value. Called by the model's forward
    # pre-hook. Returns nothing.
    """Pre-hook/post-hook publication (no scope survives between paired hooks)."""
    _state.batch = (ngram_ids, cu_seqlens)


def clear_ple_batch() -> None:
    # [FUNCTION DOC] Reset the thread-local published batch to nothing (called by the model's
    # forward post-hook).
    _state.batch = None


# Process-wide cache from checkpoint directory to its parsed tensor-name-to-file mapping.
_WEIGHT_MAP_CACHE: dict[str, dict] = {}


def _weight_map(hf_checkpoint: str) -> dict:
    # [FUNCTION DOC] HELPER: return the mapping from tensor name to safetensors file name for a
    # checkpoint directory, read from its safetensors index JSON file (weight map section) on first
    # use and cached for the process. Raises file-not-found if the index file is missing (e.g. a
    # single-file checkpoint).
    """The checkpoint's name->file map, parsed once per path per process."""
    if hf_checkpoint not in _WEIGHT_MAP_CACHE:
        with open(f"{hf_checkpoint}/model.safetensors.index.json") as f:
            _WEIGHT_MAP_CACHE[hf_checkpoint] = json.load(f)["weight_map"]
    return _WEIGHT_MAP_CACHE[hf_checkpoint]


def _safetensors_slice(path: str, name: str, header_cache: dict) -> tuple[int, int, list[int]]:
    # [FUNCTION DOC] HELPER: locate one tensor inside a safetensors file. Reads the file header
    # (an 8-byte little-endian length followed by a JSON header) once per path, caching the parsed
    # header and the header size in the caller-supplied cache mapping (mutated). Returns absolute
    # start byte, end byte (exclusive) of the tensor data within the file, and the tensor shape.
    # Raises a key error if the tensor name is not in the header. dtype is not checked.
    """Byte range and shape of one tensor, from the safetensors header."""
    if path not in header_cache:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header_cache[path] = (json.loads(f.read(n)), 8 + n)
    header, base = header_cache[path]
    meta = header[name]
    start, end = meta["data_offsets"]
    return base + start, base + end, meta["shape"]


class Qwen38NextFrozenNGramEmbedding(MegatronModule):
    # [CLASS DOC] The layer's n-gram embedding table. Not a parameter or buffer (so never
    # checkpointed, never moved to the GPU by module moves): a pinned CPU bfloat16 tensor of this
    # rank's rows (shape: rows owned by the rank by embedding dim), lazily filled from the
    # HuggingFace safetensors on first forward (or explicitly via load_from_hf). The table is split
    # into the config's fixed number of shards of equal height; each TP rank owns a contiguous run
    # of whole shards. Three non-persistent integer buffers (layer multipliers, per-head vocabulary
    # sizes, per-head offsets) parameterise the hash and are read from the checkpoint at
    # construction when its path is configured. Embedding dim is the total PLE embed dim divided by
    # the head count ((ngram size - 1) times heads per n-gram).
    """Frozen host-resident n-gram table, row-sharded over TP."""

    def __init__(self, config: TransformerConfig, layer_number: int, tp_group=None):
        # [FUNCTION DOC] Read ngram size, heads per n-gram, embed dim, shard count and EOS id
        # (default 0 when the config lacks it) from the Qwen3.8-Next config fields; the HF layer
        # index is layer_number minus one. TP size and rank come from tp_group (1 and 0 if none).
        # Raises a value error if the shard count is not divisible by TP size. This rank owns
        # shards_per_rank consecutive shard ids starting at rank times that. Rows per shard: the
        # config's rows-per-shard field, else derived by reading the shape of this layer's shard 0
        # weight in the HF checkpoint; value error if neither the field nor the checkpoint path is
        # configured. Computes the global row range of this rank, allocates the pinned CPU bf16
        # table uninitialised (requires CUDA for pinning), marks it not loaded, registers the three
        # zero-initialised int64 buffers (non-persistent), and loads them from the checkpoint if
        # its path is configured.
        super().__init__(config)
        self.layer_number = layer_number
        self.hf_layer_index = layer_number - 1
        heads = (config.qwen3_8_next_ngram_size - 1) * config.qwen3_8_next_heads_per_ngram
        self.embedding_dim = config.qwen3_8_next_ple_embed_dim // heads
        self.num_shards = config.qwen3_8_next_split_ngram_parts
        self.ngram_size = config.qwen3_8_next_ngram_size
        self._heads_per_ngram = config.qwen3_8_next_heads_per_ngram
        self.eos_token_id = getattr(config, "qwen3_8_next_eos_token_id", 0)
        self.tp_group = tp_group

        tp_size = tp_group.size() if tp_group is not None else 1
        tp_rank = tp_group.rank() if tp_group is not None else 0
        if self.num_shards % tp_size:
            raise ValueError(
                f"split_ngram_parts={self.num_shards} must be divisible by the "
                f"tensor-parallel size {tp_size}: shards are assigned whole so that "
                "changing TP only changes which of the fixed HF shards a rank reads."
            )
        self.shards_per_rank = self.num_shards // tp_size
        self.shard_ids = list(range(tp_rank * self.shards_per_rank, (tp_rank + 1) * self.shards_per_rank))

        self.rows_per_shard = getattr(config, "qwen3_8_next_ngram_rows_per_shard", None)
        if self.rows_per_shard is None:
            hf = getattr(config, "qwen3_8_next_hf_checkpoint", None)
            if hf is None:
                raise ValueError(
                    "PLE shard height unknown: neither qwen3_8_next_ngram_rows_per_shard "
                    "nor qwen3_8_next_hf_checkpoint is set on the config; deriving it "
                    "from ngram_vocab_size_base drifts 12 rows/shard."
                )
            name = f"model.language_model.layers.{layer_number - 1}.ple.ple_embedding.ngram_embedding.shard_0.weight"
            index = _weight_map(hf)
            _, _, shape = _safetensors_slice(f"{hf}/{index[name]}", name, {})
            self.rows_per_shard = int(shape[0])
        self.row_start = self.shard_ids[0] * self.rows_per_shard
        self.row_end = (self.shard_ids[-1] + 1) * self.rows_per_shard

        self.table = torch.empty(
            (self.row_end - self.row_start, self.embedding_dim),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        self._loaded = False
        self._hf_checkpoint = getattr(config, "qwen3_8_next_hf_checkpoint", None)

        self.register_buffer("layer_multipliers", torch.zeros(self.ngram_size, dtype=torch.long), persistent=False)
        self.register_buffer("ngram_heads_vocab_sizes", torch.zeros(heads, dtype=torch.long), persistent=False)
        self.register_buffer("ngram_heads_offsets", torch.zeros(heads, dtype=torch.long), persistent=False)

        if self._hf_checkpoint is not None:
            self.load_metadata_from_hf(self._hf_checkpoint)

    def load_metadata_from_hf(self, hf_checkpoint: str) -> None:
        # [FUNCTION DOC] Fill the three hash buffers from the checkpoint tensors stored under this
        # layer's PLE embedding prefix: for each, find its file and byte range from the safetensors
        # header, read the raw bytes, interpret as 64-bit integers, reshape per the stored shape and
        # copy into the buffer in place. Raises key error if a tensor is missing from the index.
        """Load the three integer tensors that parameterise the hash."""
        index = _weight_map(hf_checkpoint)
        prefix = f"model.language_model.layers.{self.hf_layer_index}.ple.ple_embedding"
        cache: dict = {}
        for buf_name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
            name = f"{prefix}.{buf_name}"
            path = f"{hf_checkpoint}/{index[name]}"
            start, end, shape = _safetensors_slice(path, name, cache)
            with open(path, "rb") as f:
                f.seek(start)
                raw = f.read(end - start)
            vals = torch.frombuffer(bytearray(raw), dtype=torch.int64).clone()
            getattr(self, buf_name).copy_(vals.reshape(shape))

    def load_from_hf(self, hf_checkpoint: str) -> None:
        # [FUNCTION DOC] Fill the host table from the checkpoint. For each shard owned by this rank
        # (in order), locates the shard weight tensor in its file, asserts that the destination slice
        # of the table starting at that shard's local row offset, with as many rows as the file
        # tensor, has exactly the tensor's shape, and reads the raw bytes directly into the table's
        # memory (no dtype conversion: the file is assumed to hold bfloat16). Marks the table loaded.
        # Side effects: heavy disk reads (tens of GB across ranks).
        """Fill the table from the HF safetensors."""
        index = _weight_map(hf_checkpoint)
        prefix = f"model.language_model.layers.{self.hf_layer_index}.ple.ple_embedding"
        cache: dict = {}
        for i, shard_id in enumerate(self.shard_ids):
            name = f"{prefix}.ngram_embedding.shard_{shard_id}.weight"
            path = f"{hf_checkpoint}/{index[name]}"
            start, end, shape = _safetensors_slice(path, name, cache)
            rows = i * self.rows_per_shard
            dst = self.table[rows : rows + shape[0]]
            assert tuple(dst.shape) == tuple(shape), f"{name}: {tuple(dst.shape)} vs {shape}"
            with open(path, "rb") as f:
                f.seek(start)
                mv = memoryview(dst.view(torch.uint8).reshape(-1).numpy())  # type: ignore[arg-type]
                f.readinto(mv)

        self._loaded = True

    def compute_ngram_ids(self, contexts: Tensor) -> Tensor:
        # [FUNCTION DOC] Hash sliding windows (tokens, ngram size) into row ids (tokens, heads) using
        # this layer's buffers, ngram size, heads per n-gram and EOS id. Pure; see the hash function.
        """``[T, ngram_size]`` sliding windows -> ``[T, n_heads]`` row ids."""
        return ngram_hash_ids(
            contexts,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            self.ngram_size,
            self._heads_per_ngram,
            self.eos_token_id,
        )

    def forward(self, ids: Tensor) -> Tensor:
        # [FUNCTION DOC] Look up embeddings for row ids (tokens, heads), int64. Lazily loads the
        # table from the HF checkpoint on first use (runtime error if no checkpoint path is
        # configured, since a zero table would silently change logits). Asserts that the table is
        # still a pinned host tensor. Gathers the rows through a custom kernel restricted to this
        # rank's row range [row_start, row_end), flattens the head and embedding dims into one
        # (tokens, heads times dim) bfloat16 tensor, and sums it in place across the TP group when
        # TP size > 1 (each rank contributes only rows it owns). Returns that tensor.
        """``[T, n_heads]`` int64 -> ``[T, n_heads * embedding_dim]`` bf16."""
        if not self._loaded:
            if self._hf_checkpoint is None:
                raise RuntimeError(
                    "PLE table never loaded and config.qwen3_8_next_hf_checkpoint "
                    "is unset; a zero table changes logits without failing."
                )
            self.load_from_hf(self._hf_checkpoint)
        assert self.table.device.type == "cpu", (
            f"PLE table must stay on the host, found {self.table.device}. Something "
            "moved it -- most likely by registering it as a buffer again."
        )
        assert self.table.is_pinned(), "PLE table lost its pinning"
        rows = gather_ple_rows(self.table, ids, self.row_start, self.row_end)
        out = rows.flatten(start_dim=-2)
        if self.tp_group is not None and self.tp_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.tp_group)
        return out


class Qwen38NextPLE(MegatronModule):
    # [CLASS DOC] Per-layer-embedding block. Owns the frozen n-gram embedding, two replicated
    # (duplicated parallel mode, no bias) Transformer Engine linears projecting the embedding to the
    # key (width streams times hidden) and value (width hidden), three norm weights (key, query,
    # conv; wide, zero-initialised) and a depthwise convolution weight (wide by 1 by kernel size,
    # zero-initialised, so weights must come from a checkpoint). The convolution dilation equals the
    # n-gram size (matching the rollout engine; the checkpoint has no separate field). The four
    # direct parameters are tagged sequence-parallel per config.
    """PLE increment for the hyper-connection state."""

    def __init__(self, config: TransformerConfig, layer_number: int, tp_group=None):
        # [FUNCTION DOC] Reads stream count, hidden size, epsilon, PLE embed dim and conv kernel size
        # from the config, builds the embedding with the given TP group, the two projections, the
        # norm and conv parameters (in the config's parameter dtype), and sets the dilation.
        super().__init__(config)
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon
        self.embed_dim = config.qwen3_8_next_ple_embed_dim
        wide = self.n * self.hidden_size

        self.ple_embedding = Qwen38NextFrozenNGramEmbedding(config, layer_number=layer_number, tp_group=tp_group)

        self.key_proj = TELinear(
            self.embed_dim,
            wide,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.value_proj = TELinear(
            self.embed_dim,
            self.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        dtype = config.params_dtype
        self.norm_key = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.norm_query = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.norm_conv = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))

        kernel = config.qwen3_8_next_ple_conv_kernel_size
        # sglang dilates the short conv by the n-gram size; the checkpoint has no separate field
        self.conv_dilation = config.qwen3_8_next_ngram_size
        self.conv1d_weight = torch.nn.Parameter(torch.zeros(wide, 1, kernel, dtype=dtype))

        for p in (self.norm_key, self.norm_query, self.norm_conv, self.conv1d_weight):
            p.sequence_parallel = config.sequence_parallel

    def forward(self, hc_state: Tensor, ngram_ids: Tensor, cu_seqlens: Tensor | None = None) -> Tensor:
        # [FUNCTION DOC] Compute the PLE increment for flattened tokens. Inputs: hc_state (tokens,
        # streams times hidden), ngram_ids (tokens, heads), optional cumulative sequence lengths
        # (so the convolution does not cross sequences). Raises runtime errors if either input is not
        # 2-D, if token counts differ, or if the state width is not streams times hidden. Steps: look
        # up embeddings; project to key and value (biases discarded); call the fused gate-and-conv
        # kernel with state, key, value, the three norm weights, conv weight, stream count, epsilon,
        # dilation and cumulative lengths. Returns the increment (tokens, streams times hidden).
        """``hc_state`` ``[T, n*C]``, ``ngram_ids`` ``[T, n_heads]`` -> increment ``[T, n*C]``."""
        if hc_state.dim() != 2 or ngram_ids.dim() != 2:
            raise RuntimeError(
                "PLE takes a flat token axis: expected hc_state [T, n*C] and ngram_ids "
                f"[T, heads], got {tuple(hc_state.shape)} and {tuple(ngram_ids.shape)}"
            )
        if hc_state.shape[0] != ngram_ids.shape[0]:
            raise RuntimeError(f"PLE token counts differ: state {hc_state.shape[0]} vs ids {ngram_ids.shape[0]}")

        embeddings = self.ple_embedding(ngram_ids)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)

        if hc_state.shape[-1] != self.n * self.hidden_size:
            raise RuntimeError(
                "PLE hidden size does not match the hyper-connection layout: expected "
                f"{self.n * self.hidden_size}, got {hc_state.shape[-1]}"
            )

        return ple_gate_conv(
            hc_state,
            key,
            value,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            self.conv1d_weight,
            self.n,
            self.norm_eps,
            self.conv_dilation,
            cu_seqlens,
        )
