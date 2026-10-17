# [MODULE DOC] Enumerates the named parameters (and expert-bias buffers) of a list of Megatron
# model chunks, either with local names or with globally consistent names, used by weight
# update/export code. Global names fold the pipeline layer offset and expert-parallel expert
# offset into the layer and expert indices so the same weight has the same name on every rank.
# The names use fixed string patterns for decoder layers, multi-token-prediction layers and
# grouped expert weights; parameters flagged as witness parameters are always skipped.
"""Enumerate a megatron model's named parameters and buffers.

Global naming folds virtual-PP and EP offsets into layer indices
(decoder.layers, mtp.layers, expert_bias) so names are comparable across
ranks; witness params are skipped.
"""

import inspect
import re
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix
from miles.backends.training_utils.parallel import get_parallel_state


def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC] Returns a lazy iterator of (name, tensor) pairs over parameters and expert-bias
    # buffers of all model chunks. When convert-to-global-name is true (default) names come from
    # the global enumeration, otherwise from the vanilla one. When translate-gpu-to-cpu is true
    # (default false), each tensor is replaced by its zero-copy CPU backup if the GPU
    # memory-saver registry holds one, else left unchanged. No eager work is done.
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model)
    else:
        ans = _named_params_and_buffers_vanilla(model)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans


def _maybe_get_cpu_backup(x: torch.Tensor):
    # [FUNCTION DOC] HELPER: Looks up a zero-copy CPU backup of the tensor in the torch memory
    # saver (imported lazily so the dependency is optional); returns the backup when one exists,
    # otherwise the original tensor.
    from torch_memory_saver import torch_memory_saver

    if (cpu_tensor := torch_memory_saver.get_cpu_backup(x, zero_copy=True)) is not None:
        return cpu_tensor

    return x


def _named_params_and_buffers_vanilla(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Yields every parameter (skipping witness parameters) and then every
    # buffer whose name contains the expert-bias marker, for each chunk in order. Names are
    # prefixed with the chunk index as a virtual-stage label, followed by the original name with
    # its wrapper prefix stripped. No layer or expert index adjustments are made.
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            # [FUNCTION DOC] HELPER: Builds the full name from the (default-bound) chunk index and
            # the stripped parameter name.
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            if getattr(param, "_is_witness_param", False):
                continue
            yield _compute_fqn(name), param

        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            yield _compute_fqn(name), buffer


def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Yields (global name, tensor) for all non-witness parameters and
    # expert-bias buffers of every chunk. Name normalization: names not starting with the double
    # wrapper prefix get one wrapper prefix prepended. Decoder-layer names (optionally under a
    # language-model segment) get their layer index shifted by the pipeline layer offset of that
    # chunk (computed from the chunk's config, passing the chunk index as the virtual stage only
    # when the offset function accepts it); the language-model segment is dropped from the output
    # name. Within those, grouped-expert weights (suffix is a weight plus integer) get their
    # expert index shifted by expert-parallel rank times experts divided by expert-parallel size.
    # Multi-token-prediction layer names keep their index unchanged but have their expert weight
    # indices shifted the same way. Other names pass through unchanged. Buffers with the
    # expert-bias marker are handled like decoder parameters (layer index shifted, no expert shift).
    # KNOWN ISSUE: the expert offset is only defined when the number of experts is set; a
    # model with grouped expert weights but a zero/unset expert count would hit an undefined
    # variable. Expert-bias buffers inside mtp layers are passed through without adjustments.
    """
    Yield (global_name, param/buffer) with consistent names across PP/EP. Adjusts indices for
    virtual PP + EP offsets. Handles decoder.layers, mtp.layers (Multi-Token Prediction), expert_bias.
    """
    ep_size = get_parallel_state().ep.size
    ep_rank = get_parallel_state().ep.rank
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for name, param in model_module.named_parameters():
            if getattr(param, "_is_witness_param", False):
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # MTP layer indices start from 0
                layer_idx, rest = match.groups()
                expert_pattern = r"(transformer_layer|mtp_model_layer)\.mlp\.experts\.(.+)\.weight(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                inner, rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield (
                    f"module.module.mtp.layers.{layer_idx}.{inner}.mlp.experts.{rest}.weight{expert_idx}",
                    param,
                )
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}", param
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer
