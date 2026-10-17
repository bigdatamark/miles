# [MODULE DOC]
# Monkey-patches for Hugging Face NemotronH models so packed training rows (several documents
# concatenated into one row) do not leak state between documents. Three classes get wrapped
# forwards: the Mamba2 mixer (injects per-token document indices into the conv and scan kernels),
# the attention mixer (switches to variable-length flash attention using cumulative sequence
# lengths) and the causal LM (computes the packed-document context from position ids and stashes
# it on every mixer/attention module before delegating). Wrapping is marked on the wrapper to keep
# it idempotent. Also provides a post-load fixup that re-reads two parameter kinds from the
# checkpoint safetensors files. Entry points: the weight reload function and the apply-patch
# function; the rest are helpers.
"""Reset NemotronH Mamba2 (conv+SSM) and attention state at packed-document boundaries (FSDP packing).

NemotronH is a Mamba2/attention hybrid; both are stateful and bleed across documents packed into one
forward row, which dominates its train/rollout logprob gap. We derive per-doc boundaries from
position_ids and feed seq_idx to the mixer's un-fused conv/scan kernels + run attention as varlen
flash-attn with cu_seqlens, so each doc stays isolated. Boundaries are stashed from the CausalLM
forward (position_ids don't reach the mixers otherwise). No-op when not packing.

The post-load fixup re-asserts checkpoint weights that transformers' Mamba ``_init_weights`` clobbers.
"""

import functools
import glob
import json
import logging
import os
import sys

import torch

from ..adaptations.packing.boundaries import packed_seq_context

logger = logging.getLogger(__name__)

_CLOBBERED_PARAM_SUFFIXES = (".mixer.dt_bias", ".mixer.out_proj.weight")

# Set per attention-mixer instance by plugins/hf_kernels/binders.py; read per forward.
HUB_VARLEN_ATTR = "_hub_flash_attn_varlen_func"


def reload_nemotron_h_clobbered_weights(model, ckpt_path, tol=1e-3) -> int:
    # [FUNCTION DOC]
    # Re-applies checkpoint values for parameters whose names end with the mixer dt bias or mixer
    # output projection weight (the ones the library's weight-init clobbers after loading).
    # Steps: list the safetensors files in the checkpoint directory (return 0 if none, or if the
    # safetensors package cannot be imported); if an index json exists use its weight map to find
    # the one shard per parameter, otherwise scan all files. For every non-meta-device matching
    # parameter, read the tensor from the first file containing it (unreadable files are skipped);
    # if the shape matches, cast to the parameter's device and dtype and, when the max absolute
    # difference exceeds the tolerance, copy it in place (without gradient tracking).
    # Returns the number of parameters overwritten and logs it when nonzero. Inputs: model, the
    # checkpoint directory path, tolerance (default 1e-3). Reads disk; mutates parameters.
    # Parameters not found in any file are silently left untouched.
    """Restore NemotronH mixer parameters overwritten after checkpoint loading."""
    try:
        from safetensors import safe_open
    except Exception:  # pragma: no cover
        return 0
    files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
    if not files:
        return 0
    index = os.path.join(ckpt_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shard_of = json.load(f)["weight_map"]
    else:
        shard_of = {}

    reloaded = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.endswith(_CLOBBERED_PARAM_SUFFIXES) or param.device.type == "meta":
                continue
            shards = [os.path.join(ckpt_path, shard_of[name])] if name in shard_of else files
            for f in shards:
                try:
                    with safe_open(f, framework="pt") as sf:
                        if name not in sf.keys():
                            continue
                        disk = sf.get_tensor(name)
                except Exception:
                    continue
                if disk.shape == param.shape:
                    disk = disk.to(device=param.device, dtype=param.dtype)
                    if (param.detach() - disk).abs().max().item() > tol:
                        param.copy_(disk)
                        reloaded += 1
                break
    if reloaded:
        logger.info(
            "[fsdp post_load] restored %d NemotronH mixer parameter(s) overwritten by _init_weights",
            reloaded,
        )
    return reloaded


def _inject_seq_idx(fn, seq_idx):
    # [FUNCTION DOC] HELPER: Returns a wrapper around a kernel function that always passes the given
    # per-token document index tensor as the keyword argument seq_idx (overriding any caller value),
    # preserving the wrapped function's metadata.
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Sets the seq_idx keyword to the captured value and calls the
        # original kernel with all other arguments unchanged; returns its result.
        kwargs["seq_idx"] = seq_idx
        return fn(*args, **kwargs)

    return wrapped


def _patch_mixer_forward(mixer_cls):
    # [FUNCTION DOC] HELPER: Replaces the Mamba2 mixer class's fused-kernel forward method with a
    # packing-aware wrapper (in place, class-level). Does nothing if already patched.
    orig = mixer_cls.cuda_kernels_forward
    if getattr(orig, "_nemotron_packing", False):
        return

    @functools.wraps(orig)
    def cuda_kernels_forward(self, hidden_states, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper forward. Falls back to the original when the instance has no
        # stashed document index (not packing) or a cache object is passed (first positional or
        # keyword cache_params; i.e. generation). Otherwise, in the mixer's defining module it
        # temporarily replaces the causal conv function and the chunked scan function (those that
        # exist) with versions injecting the document index, forces the module out of training mode
        # so the original takes its un-fused branch (the fused kernel cannot take the index
        # correctly), calls the original, and in a finally block restores the training flag and the
        # module-level functions. Returns the original's output. Not thread-safe (global swap).
        seq_idx = getattr(self, "_packing_seq_idx", None)
        cache_params = args[0] if args else kwargs.get("cache_params")
        if seq_idx is None or cache_params is not None:
            return orig(self, hidden_states, *args, **kwargs)
        mod = sys.modules[mixer_cls.__module__]
        saved = {}
        for n in ("causal_conv1d_fn", "mamba_chunk_scan_combined"):
            fn = getattr(mod, n, None)
            if fn is not None:
                saved[n] = fn
                setattr(mod, n, _inject_seq_idx(fn, seq_idx))
        was_training = self.training
        self.training = False  # force the un-fused branch (the fused kernel can't take seq_idx correctly)
        try:
            return orig(self, hidden_states, *args, **kwargs)
        finally:
            self.training = was_training
            for n, fn in saved.items():
                setattr(mod, n, fn)

    cuda_kernels_forward._nemotron_packing = True
    mixer_cls.cuda_kernels_forward = cuda_kernels_forward


def _patch_attn_forward(attn_cls):
    # [FUNCTION DOC] HELPER: Replaces the attention mixer class's forward with a packing-aware
    # variable-length version (class-level, in place; idempotent). Resolves the native flash-attn
    # varlen function at patch time (none if the package is missing).
    orig = attn_cls.forward
    if getattr(orig, "_nemotron_packing", False):
        return

    try:
        from flash_attn import flash_attn_varlen_func as native_varlen_func
    except Exception:  # pragma: no cover
        native_varlen_func = None

    @functools.wraps(orig)
    def forward(self, hidden_states, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper attention forward. Delegates to the original when there are
        # no stashed cumulative sequence lengths, when a past-key-values (or cache_params) keyword
        # is non-empty, or when no varlen function is available (an instance-bound hub kernel is
        # preferred over the native flash-attn one; without either, attention stays dense across
        # documents). Otherwise: takes hidden states of shape (batch, tokens, hidden), projects
        # query/key/value with the module's projections, flattens batch and tokens into one token
        # axis and heads of size head_dim (keys/values keep their own head count; GQA handled by the
        # kernel), runs causal varlen attention with the stashed cumulative lengths and max length,
        # and returns the output projection of the reshaped result paired with None for attention
        # weights.
        cu = getattr(self, "_packing_cu_seqlens", None)
        cache = kwargs.get("past_key_values", kwargs.get("cache_params"))
        # A hub kernel bound on the instance wins over the flash-attn wheel; without either,
        # the packed-document reset below cannot run and attention goes dense across documents.
        varlen_func = getattr(self, HUB_VARLEN_ATTR, None) or native_varlen_func
        if cu is None or cache is not None or varlen_func is None:
            return orig(self, hidden_states, *args, **kwargs)
        # Packed rows arrive as (1, total_tokens, hidden); flash varlen consumes flat
        # (tokens, heads, head_dim) and handles GQA natively, so K/V stay at kv-head count.
        b, q, _ = hidden_states.size()
        qf = self.q_proj(hidden_states).view(b * q, -1, self.head_dim)
        kf = self.k_proj(hidden_states).view(b * q, -1, self.head_dim)
        vf = self.v_proj(hidden_states).view(b * q, -1, self.head_dim)
        ml = self._packing_max_seqlen
        o = varlen_func(qf, kf, vf, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=ml, max_seqlen_k=ml, causal=True)
        return self.o_proj(o.reshape(b, q, -1)), None

    forward._nemotron_packing = True
    attn_cls.forward = forward


def _patch_causallm_forward(causallm_cls, mixer_cls, attn_cls):
    # [FUNCTION DOC] HELPER: Replaces the causal LM class's forward (class-level, idempotent) with a
    # wrapper that, before each call, derives the packed-document context from the position_ids
    # argument and stashes it on submodules, then calls the original. attn_cls may be none.
    orig = causallm_cls.forward
    if getattr(orig, "_nemotron_packing", False):
        return
    import inspect

    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Finds position_ids by binding the arguments to the original
        # signature (falling back to the keyword argument if binding fails). Builds the packed
        # context; with none (unpacked) the three stashed values are cleared to none. Walks all
        # submodules: every Mamba2 mixer gets the per-token document index, every attention mixer
        # gets the cumulative sequence lengths and the maximum sequence length. Mutates module
        # attributes on each call, then returns the original forward's output.
        try:
            position_ids = sig.bind(self, *args, **kwargs).arguments.get("position_ids")
        except TypeError:
            position_ids = kwargs.get("position_ids")
        ctx = packed_seq_context(position_ids)
        cu = ctx.cu_seqlens if ctx is not None else None
        si = ctx.seq_idx if ctx is not None else None
        ml = ctx.max_seqlen if ctx is not None else None
        for mod in self.modules():
            if isinstance(mod, mixer_cls):
                mod._packing_seq_idx = si
            elif attn_cls is not None and isinstance(mod, attn_cls):
                mod._packing_cu_seqlens = cu
                mod._packing_max_seqlen = ml
        return orig(self, *args, **kwargs)

    forward._nemotron_packing = True
    causallm_cls.forward = forward


def apply_nemotron_h_sglang_match_patch(model):
    # [FUNCTION DOC]
    # Entry point. Scans the model's modules to find the first class whose name ends with the
    # Mamba2 mixer suffix and has a fused-kernel forward, and the class of the mixer inside the
    # first block whose block_type is attention. Returns False without changes if no Mamba2 mixer is
    # found (not a NemotronH model). Otherwise patches the mixer, the attention class (if found) and
    # the model's own class, logs it, and returns True. Mutates classes globally (all instances).
    # The classes are patched even for unpacked use; the wrappers pass straight through then.
    """Reset NemotronH Mamba2 conv+SSM state AND attention at packed-document boundaries. Idempotent;
    no-op for non-NemotronH models and for single-document (unpacked) forwards."""
    mixer_cls = attn_cls = None
    for mod in model.modules():
        cn = type(mod).__name__
        if mixer_cls is None and cn.endswith("Mamba2Mixer") and hasattr(type(mod), "cuda_kernels_forward"):
            mixer_cls = type(mod)
        if attn_cls is None and getattr(mod, "block_type", None) == "attention" and hasattr(mod, "mixer"):
            attn_cls = type(mod.mixer)
    if mixer_cls is None:
        return False
    _patch_mixer_forward(mixer_cls)
    if attn_cls is not None:
        _patch_attn_forward(attn_cls)
    _patch_causallm_forward(type(model), mixer_cls, attn_cls)
    logger.info(
        "[fsdp] NemotronH packed-doc reset applied (mamba seq_idx + attention varlen cu_seqlens); " "mixer=%s attn=%s",
        mixer_cls.__module__,
        attn_cls.__name__ if attn_cls else None,
    )
    return True
