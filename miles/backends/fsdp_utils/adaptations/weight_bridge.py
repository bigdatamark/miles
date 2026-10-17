# [MODULE DOC] Registry of per-model-type transforms applied to parameters while syncing weights
# from FSDP training to the rollout engine, for models whose training names or shapes differ from
# what the engine loader expects. A transform has a selector (name and parameter to boolean) and an
# expander (name, full tensor and model to a stream of name and tensor pairs). Lookup is by model
# type, tried in registration order; no match means pass through unchanged. Also contains the
# shared batched-experts selector/expander used by several architectures, which depends on a
# transformers version providing the weight conversion revert helper (imported at module load).
"""WeightBridge: the registered train->rollout parameter-name/shape contract for the FSDP backend.

HF/FSDP training and the SGLang rollout loader don't always agree on param names/shapes (e.g.
transformers>=5.6 stores qwen3_moe experts as one batched tensor; SGLang wants per-expert names). Each
disagreeing model_type registers a ParamTransform (a ``matches`` selector + an ``expand`` that rewrites
the materialized tensor) instead of editing the sync loop -- the FSDP analogue of Megatron's
megatron_to_hf. ``expand`` never touches DTensor/device state, so transforms are CPU-unit-testable.
"""

from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch
from transformers.core_model_loading import revert_weight_conversion


class ParamTransform(NamedTuple):
    # [CLASS DOC] Immutable pair. matches: function of parameter name and parameter object
    # returning whether this transform handles it. expand: function of name, the full
    # (materialized, unsharded) tensor and the model, returning an iterable of output name and
    # tensor pairs to send instead of the original.
    matches: Callable[[str, object], bool]
    expand: Callable[[str, torch.Tensor, torch.nn.Module], Iterable[tuple[str, torch.Tensor]]]


# model_type -> registered transforms, tried in registration order
_REGISTRY: dict[str, list[ParamTransform]] = {}


def register_param_transform(model_type: str, matches: Callable, expand: Callable) -> None:
    # [FUNCTION DOC] Add a transform for a model type: creates the type's list if needed and
    # appends a transform built from the selector and expander (earlier ones are tried first).
    _REGISTRY.setdefault(model_type, []).append(ParamTransform(matches, expand))


def get_param_transform(name: str, param, model_type: str):
    # [FUNCTION DOC] Find the expander for a parameter: scans the transforms registered for the
    # model type in order and returns the expand function of the first whose selector accepts
    # the name and parameter; returns none (pass through) if the type is unknown or nothing
    # matches.
    """Return the ``expand`` fn for the transform matching this param, or None (passthrough)."""
    for transform in _REGISTRY.get(model_type, ()):
        if transform.matches(name, param):
            return transform.expand
    return None


# batched-expert archs (qwen3_moe, glm4_moe_lite, ...): unfuse into the per-expert names SGLang expects.
def _batched_experts_matches(name: str, param) -> bool:
    # [FUNCTION DOC] HELPER: Selector for fused expert weights: true when the parameter has a
    # dimension count method returning 3 (objects without one count as 0) and the name ends with
    # the experts gate-and-up projection name or the experts down projection name.
    return getattr(param, "dim", lambda: 0)() == 3 and (
        name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj")
    )


def _hf_unfuse_experts_expand(name: str, full: torch.Tensor, model) -> Iterable[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Generator that yields output name and tensor pairs. Passes the single
    # entry of name to full tensor through the transformers revert-weight-conversion routine for
    # the model, which returns the mapping that saving the model would write (per-expert names),
    # and yields each pair with the tensor made contiguous in memory.
    """Unfuse a batched experts tensor through transformers' own save path (``revert_weight_conversion``),
    yielding exactly what ``save_pretrained`` would write -- the on-disk dialect SGLang's loader expects.
    The revert ops slice views, so contiguity is re-asserted before streaming."""
    for out_name, tensor in revert_weight_conversion(model, {name: full}).items():
        yield out_name, tensor.contiguous()
