# [MODULE DOC]
# Monkey-patches for Hugging Face GatedDeltaNet hybrid architectures (qwen3_5, qwen3_5_moe,
# qwen3_next) so packed rows do not carry linear-attention recurrence or causal-conv state across
# document boundaries. Two class-level wrappers cooperate: the decoder layer's forward computes the
# packed-document context from position_ids and stashes cumulative sequence lengths and per-token
# document indices on each GatedDeltaNet submodule; the GatedDeltaNet forward then, if anything was
# stashed, temporarily rebinds its kernel attributes so they receive those values as defaults.
# Wrapped forwards carry a marker attribute so patching is idempotent. Entry point is the apply
# function; everything else is a helper.
"""Reset GatedDeltaNet recurrence + conv state at packed-document boundaries (FSDP packing).

GatedDeltaNet's linear-attn recurrence and causal_conv1d run over the whole packed row and bleed
across documents, inflating the train/rollout logprob gap. The decoder layer derives cu_seqlens/seq_idx
from position_ids and stashes them on its GDN submodule, which injects them into both kernels so each
document resets. Runs inside the gradient-checkpointed layer, so boundaries recompute identically on
backward. No-op outside THD packing.
"""

import functools
import logging

from ..adaptations.packing.boundaries import packed_seq_context

logger = logging.getLogger(__name__)


def _inject_kwarg(fn, key, value):
    # [FUNCTION DOC] HELPER: Returns a wrapper around a kernel callable that fills in the named
    # keyword argument with the given value whenever the caller did not pass it or passed none; an
    # explicitly provided non-none value wins.
    """Wrap a kernel callable to default a kwarg (cu_seqlens / seq_idx) when unset."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Applies the default keyword if missing/none, then calls the
        # original kernel and returns its result.
        if kwargs.get(key) is None:
            kwargs[key] = value
        return fn(*args, **kwargs)

    return wrapped


def _patch_gdn_forward(gdn_cls):
    # [FUNCTION DOC] HELPER: Replaces the GatedDeltaNet class's forward (class-level, in place,
    # idempotent) with a wrapper that injects document boundaries into its kernels. The injection
    # table maps the chunked and recurrent gated delta rule kernels to the cumulative-sequence-length
    # keyword and the causal conv kernel to the sequence-index keyword.
    orig = gdn_cls.forward
    if getattr(orig, "_gdn_packing", False):
        return

    # rebind the kernel instance-attrs for the duration of the forward to inject per-doc boundaries
    _INJECT = (
        ("chunk_gated_delta_rule", "cu_seqlens"),
        ("recurrent_gated_delta_rule", "cu_seqlens"),
        ("causal_conv1d_fn", "seq_idx"),
    )

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper forward. If neither the stashed cumulative lengths nor the
        # stashed sequence index exist, calls the original untouched. Otherwise, for each kernel
        # attribute on the instance that exists and whose matching stashed value is not none, saves
        # it and rebinds the instance attribute to a version defaulting the keyword to the stashed
        # value; calls the original; in a finally block restores every saved attribute. Returns the
        # original's output. Mutates instance attributes temporarily.
        cu = getattr(self, "_gdn_cu_seqlens", None)
        si = getattr(self, "_gdn_seq_idx", None)
        if cu is None and si is None:
            return orig(self, *args, **kwargs)
        saved = {}
        for attr, key in _INJECT:
            value = cu if key == "cu_seqlens" else si
            fn = getattr(self, attr, None)
            if fn is not None and value is not None:
                saved[attr] = fn
                setattr(self, attr, _inject_kwarg(fn, key, value))
        try:
            return orig(self, *args, **kwargs)
        finally:
            for attr, fn in saved.items():
                setattr(self, attr, fn)

    forward._gdn_packing = True
    gdn_cls.forward = forward


def _patch_decoder_forward(dl_cls, gdn_cls):
    # [FUNCTION DOC] HELPER: Replaces the decoder layer class's forward (class-level, idempotent)
    # with a wrapper that publishes packed-document boundaries to its GatedDeltaNet submodules.
    orig = dl_cls.forward
    if getattr(orig, "_gdn_packing", False):
        return

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper forward. Builds the packed-document context from the
        # position_ids keyword argument only (a position_ids passed positionally is not seen).
        # For every GatedDeltaNet submodule sets the stashed cumulative lengths and sequence index
        # (or none when there is no packing context), then calls and returns the original forward.
        ctx = packed_seq_context(kwargs.get("position_ids"))
        for module in self.modules():
            if isinstance(module, gdn_cls):
                module._gdn_cu_seqlens = ctx.cu_seqlens if ctx is not None else None
                module._gdn_seq_idx = ctx.seq_idx if ctx is not None else None
        return orig(self, *args, **kwargs)

    forward._gdn_packing = True
    dl_cls.forward = forward


def _find_class(mod, suffix):
    # [FUNCTION DOC] HELPER: Returns the first attribute of the module (in sorted attribute-name
    # order) whose name ends with the given suffix, or none if there is none.
    for name in dir(mod):
        if name.endswith(suffix):
            return getattr(mod, name)
    return None


def apply_gateddeltanet_packing_patch():
    # [FUNCTION DOC]
    # Entry point. For each of the three architecture module names, tries to import its modeling
    # module from transformers (skipping on any import failure), finds the class whose name ends
    # with GatedDeltaNet and the one ending with DecoderLayer (skipping the module if either is
    # missing), and patches both. Logs once if anything was patched. Returns True if at least one
    # architecture was patched, else False. Mutates the library classes globally.
    """Patch every GatedDeltaNet hybrid arch present (idempotent). Returns True if anything was patched."""
    patched = False
    for mod_name in ("qwen3_5", "qwen3_5_moe", "qwen3_next"):
        try:
            mod = __import__(f"transformers.models.{mod_name}.modeling_{mod_name}", fromlist=["x"])
        except Exception:
            continue
        gdn_cls = _find_class(mod, "GatedDeltaNet")
        dl_cls = _find_class(mod, "DecoderLayer")
        if gdn_cls is None or dl_cls is None:
            continue
        _patch_gdn_forward(gdn_cls)
        _patch_decoder_forward(dl_cls, gdn_cls)
        patched = True

    if patched:
        logger.info(
            "[fsdp] GatedDeltaNet packing fix applied: cu_seqlens/seq_idx reset the "
            "linear-attn recurrence and causal-conv state per packed document"
        )
    return patched
