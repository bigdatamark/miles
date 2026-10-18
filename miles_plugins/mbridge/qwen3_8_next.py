# [MODULE DOC]
# mbridge bridge for Qwen3.8-Next, derived from the Qwen3.5 bridge. Adds hyper-connection, indexer and
# per-layer embedding (PLE) parameter name mappings, vision tower name mappings, removal of the
# inherited block and final layer norm entries (the checkpoint folds them into the hyper-connection
# norms), a device-side weight slicing cache for loading, a memoized state dict during load, and
# config attributes for the hyper-connections, PLE n-gram tables and the indexer.
"""Bridge for Qwen3.8-Next (HF ``Qwen4ExpForConditionalGeneration``).

Qwen3.5 plus two hyper-connections per layer, a model-level HC mixer, one PLE layer
and a QSA indexer on the full-attention layers. The checkpoint has no block or final
layernorms (each ``hc_norm`` is the pre-block norm), so the inherited entries are popped.
"""

import torch
from mbridge.core import register_model

from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge


@register_model(["qwen3.8_next", "qwen3_8_next", "qwen4_exp"])
class Qwen38NextBridge(Qwen3_5Bridge):
    # [CLASS DOC]
    # Registered under the names qwen3.8_next, qwen3_8_next and qwen4_exp. Class attributes: the
    # group-major tuple is emptied (this model still uses the replicated linear attention layout);
    # the direct mapping is the parent's without the final norm plus the model-level hyper-connection
    # mixer weights; the attention mapping is the parent's without the fused input norm entries plus
    # the indexer entries; the MLP mapping is the parent's without the fused norm entries; the other
    # mapping covers per-layer hyper-connections and PLE parameters (mcore's self attention hyper
    # connection slot is the attn hyper connection in Hugging Face); the vision direct and per-block
    # mappings map the vision tower; and a two-entry limit on the device slice cache.
    """Weight mapping + Megatron config for Qwen3.8-Next."""

    # Qwen3.8-Next still runs the replicated GDN, which holds q/k/v and the conv in HF's own
    # layout rather than the head-sharded layer's group-major rows
    _GDN_GROUP_MAJOR = ()

    _DIRECT_MAPPING = Qwen3_5Bridge._DIRECT_MAPPING.copy()

    # no final norm in the checkpoint: the final mixer's hc_norm is the final norm
    _DIRECT_MAPPING.pop("decoder.final_layernorm.weight", None)

    _DIRECT_MAPPING.update(
        {
            "decoder.hc_head_contraction.hc_norm_weight": "model.language_model.hyper_connection_mixer.hc_norm.weight",
            "decoder.hc_head_contraction.input_mix_weight_down": "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            "decoder.hc_head_contraction.input_mix_weight_up": "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        }
    )

    _ATTENTION_MAPPING = Qwen3_5Bridge._ATTENTION_MAPPING.copy()

    # the spec drops the fused pre-attention norm, so nothing fills layer_norm_weight
    _ATTENTION_MAPPING.pop("self_attention.linear_qkv.layer_norm_weight", None)
    _ATTENTION_MAPPING.pop("self_attention.input_layernorm.weight", None)

    _ATTENTION_MAPPING.update(
        {
            "self_attention.indexer.index_qk_proj.weight": [
                "model.language_model.layers.{layer_number}.self_attn.indexer.index_qk_proj.weight"
            ],
            # plain nn.Parameters on the mcore side; HF wraps each in a submodule
            "self_attention.indexer.q_layernorm": [
                "model.language_model.layers.{layer_number}.self_attn.indexer.q_layernorm.weight"
            ],
            "self_attention.indexer.k_layernorm": [
                "model.language_model.layers.{layer_number}.self_attn.indexer.k_layernorm.weight"
            ],
        }
    )

    _MLP_MAPPING = Qwen3_5Bridge._MLP_MAPPING.copy()

    _MLP_MAPPING.pop("mlp.linear_fc1.layer_norm_weight", None)
    _MLP_MAPPING.pop("pre_mlp_layernorm", None)

    # mcore's self_attention_hyper_connection slot is attn_hyper_connection in HF
    _OTHER_MAPPING = {
        "self_attention_hyper_connection.hc_norm_weight": [
            "model.language_model.layers.{layer_number}.attn_hyper_connection.hc_norm.weight"
        ],
        "self_attention_hyper_connection.input_mix_weight_down": [
            "model.language_model.layers.{layer_number}.attn_hyper_connection.input_mix_weight_down.weight"
        ],
        "self_attention_hyper_connection.input_mix_weight_up": [
            "model.language_model.layers.{layer_number}.attn_hyper_connection.input_mix_weight_up.weight"
        ],
        "self_attention_hyper_connection.block_inject_weight": [
            "model.language_model.layers.{layer_number}.attn_hyper_connection.block_inject_weight.weight"
        ],
        "mlp_hyper_connection.hc_norm_weight": [
            "model.language_model.layers.{layer_number}.mlp_hyper_connection.hc_norm.weight"
        ],
        "mlp_hyper_connection.input_mix_weight_down": [
            "model.language_model.layers.{layer_number}.mlp_hyper_connection.input_mix_weight_down.weight"
        ],
        "mlp_hyper_connection.input_mix_weight_up": [
            "model.language_model.layers.{layer_number}.mlp_hyper_connection.input_mix_weight_up.weight"
        ],
        "mlp_hyper_connection.block_inject_weight": [
            "model.language_model.layers.{layer_number}.mlp_hyper_connection.block_inject_weight.weight"
        ],
        "self_attention_hyper_connection.ple.key_proj.weight": [
            "model.language_model.layers.{layer_number}.ple.key_proj.weight"
        ],
        "self_attention_hyper_connection.ple.value_proj.weight": [
            "model.language_model.layers.{layer_number}.ple.value_proj.weight"
        ],
        "self_attention_hyper_connection.ple.conv1d_weight": [
            "model.language_model.layers.{layer_number}.ple.conv1d.weight"
        ],
        "self_attention_hyper_connection.ple.norm_conv": [
            "model.language_model.layers.{layer_number}.ple.norm_conv.weight"
        ],
        "self_attention_hyper_connection.ple.norm_key": [
            "model.language_model.layers.{layer_number}.ple.norm_key.weight"
        ],
        "self_attention_hyper_connection.ple.norm_query": [
            "model.language_model.layers.{layer_number}.ple.norm_query.weight"
        ],
        # the n-gram table and its hash metadata are read from the HF safetensors on
        # first use, so they never enter the torch_dist checkpoint
    }

    # vision tower: resolved only for parameters the model creates, so text-only runs skip it
    _VISION_DIRECT_MAPPING = {
        "vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
        "vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
        "vision_model.pos_embed.weight": "model.visual.pos_embed.weight",
        "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
        "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
        "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
        "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
        "vision_model.merger.norm.weight": "model.visual.merger.norm.weight",
        "vision_model.merger.norm.bias": "model.visual.merger.norm.bias",
    }

    _VISION_LAYER_MAPPING = {
        "attn.qkv.weight": ["model.visual.blocks.{layer_number}.attn.qkv.weight"],
        "attn.qkv.bias": ["model.visual.blocks.{layer_number}.attn.qkv.bias"],
        "attn.proj.weight": ["model.visual.blocks.{layer_number}.attn.proj.weight"],
        "attn.proj.bias": ["model.visual.blocks.{layer_number}.attn.proj.bias"],
        "mlp.linear_fc1.weight": ["model.visual.blocks.{layer_number}.mlp.linear_fc1.weight"],
        "mlp.linear_fc1.bias": ["model.visual.blocks.{layer_number}.mlp.linear_fc1.bias"],
        "mlp.linear_fc2.weight": ["model.visual.blocks.{layer_number}.mlp.linear_fc2.weight"],
        "mlp.linear_fc2.bias": ["model.visual.blocks.{layer_number}.mlp.linear_fc2.bias"],
        "norm1.weight": ["model.visual.blocks.{layer_number}.norm1.weight"],
        "norm1.bias": ["model.visual.blocks.{layer_number}.norm1.bias"],
        "norm2.weight": ["model.visual.blocks.{layer_number}.norm2.weight"],
        "norm2.bias": ["model.visual.blocks.{layer_number}.norm2.bias"],
    }

    def _weight_name_mapping_vision(self, mcore_weights_name: str) -> list[str]:
        # [FUNCTION DOC] Hugging Face names for a vision tower parameter. Direct mapping hits return a
        # one-element list. Names under the vision block prefix are split into the block number and the
        # remaining tail; if the tail is in the per-block mapping, the block-number placeholder is
        # filled. Otherwise raises not-implemented.
        """``vision_model.*`` -> ``model.visual.*``."""
        if mcore_weights_name in self._VISION_DIRECT_MAPPING:
            return [self._VISION_DIRECT_MAPPING[mcore_weights_name]]

        prefix = "vision_model.blocks."
        if mcore_weights_name.startswith(prefix):
            rest = mcore_weights_name[len(prefix) :]
            layer_number, _, tail = rest.partition(".")
            if tail in self._VISION_LAYER_MAPPING:
                return [t.format(layer_number=int(layer_number)) for t in self._VISION_LAYER_MAPPING[tail]]

        raise NotImplementedError(f"Unsupported vision parameter name: {mcore_weights_name}")

    def _ple_layer_ids(self) -> list[int]:
        # [FUNCTION DOC] HELPER: The sorted unique zero-based decoder layer indices that carry a per-layer
        # embedding, from the text config's one-based id list (empty if absent or empty).
        """0-based decoder layer indices carrying PLE; ``ple_layer_ids`` in the HF config is 1-based."""
        text_config = self._get_text_config()
        return sorted({int(i) - 1 for i in getattr(text_config, "ple_layer_ids", None) or []})

    def _ngram_rows_per_shard(self) -> int | None:
        # [FUNCTION DOC] HELPER: Row count of the first n-gram embedding shard of the first PLE layer,
        # read from the checkpoint tensor shape through the safetensor reader. Returns none when there
        # are no PLE layers, when the lookup fails for any reason, or when the shape is empty.
        """Height of one n-gram shard, read from the checkpoint; None when the index is unavailable."""
        layer_ids = self._ple_layer_ids()
        if not layer_ids:
            return None
        key = f"model.language_model.layers.{layer_ids[0]}.ple.ple_embedding.ngram_embedding.shard_0.weight"
        try:
            shape = self.safetensor_io.get_tensor_shape(key)
        except Exception:
            return None
        return int(shape[0]) if shape else None

    def _weight_name_mapping_other(self, mcore_weights_name: str) -> list[str]:
        # [FUNCTION DOC] Hugging Face names for hyper-connection, indexer-adjacent and PLE parameters.
        # Strips a leading decoder prefix (a bare transformer block has none); if the name then starts
        # with the layers prefix, extracts the layer number and the remainder. A remainder found in the
        # other mapping returns the formatted names, but raises not-implemented if no layer number was
        # present. Anything else raises not-implemented.
        """HC / indexer / PLE names."""
        layer_number = None
        name = mcore_weights_name
        # a bare TransformerBlock has no "decoder." prefix; the remaining keys are unambiguous
        if name.startswith("decoder."):
            name = name[len("decoder.") :]
        if name.startswith("layers."):
            parts = name.split(".")
            layer_number = int(parts[1])
            name = ".".join(parts[2:])

        if name in self._OTHER_MAPPING:
            if layer_number is None:
                raise NotImplementedError(f"{mcore_weights_name} needs a layer index")
            return [t.format(layer_number=layer_number) for t in self._OTHER_MAPPING[name]]

        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        # [FUNCTION DOC] Hugging Face names for a Megatron parameter: vision tower names use the vision
        # mapping; otherwise the parent mapping, falling back to the other mapping when the parent
        # raises not-implemented.
        if mcore_weights_name.startswith("vision_model."):
            return self._weight_name_mapping_vision(mcore_weights_name)
        try:
            return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        except NotImplementedError:
            return self._weight_name_mapping_other(mcore_weights_name)

    # every per-expert mcore param slices the same fused HF tensor, so one upload serves
    # a layer's experts; two entries because the loop alternates gate_up and down
    _GPU_CACHE_SIZE = 2

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]):
        # [FUNCTION DOC]
        # Without a gpu, defer to the parent. With one, first move every Hugging Face tensor to the
        # current gpu through a small least-recently-inserted cache keyed by memory address, shape and
        # dtype (at most two entries; the oldest is evicted), so repeated reads of the same fused expert
        # tensor upload once; then run the parent conversion on the device tensors. Creates the cache
        # on the instance lazily.
        # KNOWN ISSUE: the cache is keyed by data pointer, which can be reused by a different tensor
        # after the original is freed; the cache evicts by insertion order and does not refresh hits.
        """Slice on the device, reusing one upload across a layer's experts."""
        if not torch.cuda.is_available():
            return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

        cache = getattr(self, "_gpu_weight_cache", None)
        if cache is None:
            cache = self._gpu_weight_cache = {}

        moved = []
        for w in hf_weights:
            # data_ptr is stable for repeated reads of one tensor from the same mmap
            key = (w.data_ptr(), tuple(w.shape), w.dtype)
            hit = cache.get(key)
            if hit is None:
                if len(cache) >= self._GPU_CACHE_SIZE:
                    cache.pop(next(iter(cache)))
                hit = w.to(torch.cuda.current_device(), non_blocking=False)
                cache[key] = hit
            moved.append(hit)
        return super()._weight_to_mcore_format(mcore_weights_name, moved)

    def load_weights(self, model, *args, **kwargs):
        # [FUNCTION DOC]
        # Load weights into the model (a single model chunk or a list or tuple of chunks) through the
        # parent. Before loading, each chunk's state dict is computed once and the chunk's state dict
        # method is replaced by one returning that cached result (avoiding quadratic recomputation).
        # Afterwards, always: restore each chunk's original method (by deleting the instance override,
        # or re-assigning the saved method if deletion is impossible), clear the device weight cache
        # and empty the gpu allocator cache if a gpu is available. Returns the parent's result.
        """Memoise ``state_dict()`` for the load, then drop the device cache.

        mbridge calls ``state_dict()`` once per parameter, which is quadratic with
        per-expert params. Caching is safe because the load only writes into existing
        tensors in place.
        """
        chunks = model if isinstance(model, (list, tuple)) else [model]
        originals = []
        for chunk in chunks:
            cached = chunk.state_dict()
            originals.append((chunk, chunk.state_dict))
            chunk.state_dict = lambda *a, _c=cached, **k: _c
        try:
            return super().load_weights(model, *args, **kwargs)
        finally:
            for chunk, orig in originals:
                try:
                    del chunk.state_dict
                except AttributeError:
                    chunk.state_dict = orig
            self._gpu_weight_cache = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build_config(self):
        # [FUNCTION DOC] Build the parent config, then attach attributes read from the text config with
        # defaults when absent: hyper-connections enabled with residual streams (4) and low-rank size
        # (320); the zero-based PLE layer ids, embedding dim (2560), n-gram size (3), heads per n-gram
        # (8), n-gram vocabulary base (20000000), split parts (128), PLE convolution kernel size (4),
        # end-of-sequence token id (0, where the n-gram hash resets) and the rows per n-gram shard read
        # from the checkpoint (none if unavailable); indexer budget (2048), compress ratio (4), heads
        # (4), head dim (128) and key-value heads (1); and a flag saying blocks have no layer norms.
        # Returns the config.
        text_config = self._get_text_config()
        config = super()._build_config()

        config.enable_hyper_connections = True
        config.num_residual_streams = getattr(text_config, "hc_count", 4)
        config.qwen3_8_next_hc_lowrank = getattr(text_config, "hc_lowrank", 320)

        config.qwen3_8_next_ple_layer_ids = self._ple_layer_ids()
        config.qwen3_8_next_ple_embed_dim = getattr(text_config, "ple_embed_dim", 2560)
        config.qwen3_8_next_ngram_size = getattr(text_config, "ngram_size", 3)
        config.qwen3_8_next_heads_per_ngram = getattr(text_config, "heads_per_ngram", 8)
        config.qwen3_8_next_ngram_vocab_size_base = getattr(text_config, "ngram_vocab_size_base", 20000000)
        config.qwen3_8_next_split_ngram_parts = getattr(text_config, "split_ngram_parts", 128)
        config.qwen3_8_next_ple_conv_kernel_size = getattr(text_config, "ple_conv_kernel_size", 4)
        # the n-gram hash resets at EOS so n-grams never straddle a document
        config.qwen3_8_next_eos_token_id = getattr(text_config, "eos_token_id", 0)
        # the checkpoint pads the last shard, so the height is not a ceil over the config
        config.qwen3_8_next_ngram_rows_per_shard = self._ngram_rows_per_shard()

        config.qwen3_8_next_indexer_budget = getattr(text_config, "indexer_budget", 2048)
        config.qwen3_8_next_indexer_compress_ratio = getattr(text_config, "indexer_compress_ratio", 4)
        config.qwen3_8_next_indexer_n_heads = getattr(text_config, "indexer_n_heads", 4)
        config.qwen3_8_next_indexer_head_dim = getattr(text_config, "indexer_head_dim", 128)
        config.qwen3_8_next_indexer_kv_heads = getattr(text_config, "indexer_kv_heads", 1)

        config.qwen3_8_next_no_block_layernorms = True

        return config
