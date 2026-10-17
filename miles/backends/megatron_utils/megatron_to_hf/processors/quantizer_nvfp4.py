# [MODULE DOC]
# NVFP4 quantizer for converted Hugging Face tensors. Only routed-expert linears of the main
# language decoder are quantized (attention, dense and shared MLPs, MTP layers and modality towers
# stay in high precision), subject to an ignore/exclude policy from the quantization config
# (literal module names with dotted-prefix semantics, or "re:"-prefixed regular expressions).
# Each quantized weight yields three tensors: packed FP4 data, per-block scales, and a global
# second-level scale. Gate and up projections of an expert must be quantized together so they share
# one global scale. Weights need a last dimension divisible by the NVFP4 group size (16).
import re
from functools import lru_cache

import torch

from miles.utils.nvfp4 import NVFP4_GROUP_SIZE, nvfp4_quantize_1d, nvfp4_quantize_1d_pair

GATED_PAIR_SUFFIXES = {
    ".gate_proj.weight": "gate",
    ".up_proj.weight": "up",
    ".w1.weight": "gate",
    ".w3.weight": "up",
}


def _get_ignore_rules(quantization_config) -> list[str]:
    # [FUNCTION DOC] HELPER: Collects the config's "ignore" rules (none or a single string accepted
    # and normalized to a list) followed by the "exclude_modules" rules not already present, as one
    # list preserving order.
    ignore_rules = quantization_config.get("ignore", []) or []
    if isinstance(ignore_rules, str):
        ignore_rules = [ignore_rules]
    exclude_rules = quantization_config.get("exclude_modules", []) or []
    if isinstance(exclude_rules, str):
        exclude_rules = [exclude_rules]
    return list(ignore_rules) + [rule for rule in exclude_rules if rule not in ignore_rules]


@lru_cache(maxsize=16)
def _literal_ignore_rules(ignore_rules: tuple[str, ...]) -> frozenset[str] | None:
    # [FUNCTION DOC] HELPER: Memoized (16 entries) by the tuple of rules. Returns none if any rule
    # is a regular-expression rule (prefix "re:"), forcing the ordered slow path; otherwise the
    # frozen set of rules for fast prefix lookup.
    # Preserve the original ordered matching (including short-circuit behavior)
    # for regex policies. Snapshot keys also invalidate after config mutations.
    if any(rule.startswith("re:") for rule in ignore_rules):
        return None
    return frozenset(ignore_rules)


def _is_ignored(name: str, ignore_rules: list[str], literal_rules: frozenset[str] | None = None) -> bool:
    # [FUNCTION DOC] HELPER: True if the module/parameter name is covered by a rule. Fast path (all
    # rules literal): checks the name, then repeatedly drops the last dotted component and
    # rechecks, until a rule matches (true) or no dot remains (false). Slow path: in order, a "re:"
    # rule matches by anchored-at-start regular expression on the name; a literal rule matches when
    # the name equals it or begins with it followed by a dot.
    if literal_rules is not None:
        while True:
            if name in literal_rules:
                return True
            name, separator, _ = name.rpartition(".")
            if not separator:
                return False
    for rule in ignore_rules:
        if rule.startswith("re:"):
            if re.match(rule[3:], name):
                return True
            continue
        if name == rule or name.startswith(f"{rule}."):
            return True
    return False


def quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config):
    # [FUNCTION DOC]
    # Asserts the config is an NVFP4 one (algo NVFP4 or method nvfp4). Raises not-implemented if the
    # args request FP4 parameter storage or FP4 parameter gather. Returns the input unchanged if a
    # name in args' extra high-precision layer list is a substring of the Megatron name, or the
    # Megatron name is not a main language-model decoder layer (the pattern tolerates repeated
    # "module." prefixes and an optional "language_model." segment; MTP and vision layers fall out),
    # or the layer is in the bf16 head/tail range (first-last-layers-bf16 flag with start and end
    # counts). For routed expert linears (linear_fc1/linear_fc2) delegates to the MoE quantization
    # with the ignore rules; everything else is returned unchanged. The args object may be none.
    assert quantization_config is not None
    assert quantization_config.get("quant_algo") == "NVFP4" or quantization_config.get("quant_method") == "nvfp4"
    if args is not None and bool(getattr(args, "fp4_param", False) or getattr(args, "fp4_param_gather", False)):
        raise NotImplementedError("fp4-param-gather is unsupported for Miles NVFP4 checkpoint export.")

    if getattr(args, "extra_high_precision_layers_megatron", False):
        for layer_name in getattr(args, "extra_high_precision_layers_megatron", ()):
            if layer_name in megatron_name:
                return converted_named_params

    ignore_rules = _get_ignore_rules(quantization_config)

    # Only the main language decoder is eligible; MTP and modality towers stay in high precision.
    decoder_layers_pattern = r"(?:module\.)*(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)
    if not match:
        return converted_named_params
    layer_idx, rest = match.groups()

    # Skip quantization for BF16 tail of main decoder layers.
    if getattr(args, "first_last_layers_bf16", False):
        num_layers = int(args.num_layers)
        num_layers_at_start_in_bf16 = int(getattr(args, "num_layers_at_start_in_bf16", 0))
        num_layers_at_end_in_bf16 = int(getattr(args, "num_layers_at_end_in_bf16", 0))
        head_end_idx = num_layers_at_start_in_bf16
        tail_start_idx = num_layers - num_layers_at_end_in_bf16
        if int(layer_idx) < head_end_idx or int(layer_idx) >= tail_start_idx:
            return converted_named_params

    # routed experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, _ = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            return _quantize_moe_params(converted_named_params, ignore_rules)

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _quantize_moe_params(converted_named_params, ignore_rules):
    # [FUNCTION DOC] HELPER:
    # Pass 1: for each pair whose name ends with a gate or up suffix (gate_proj/up_proj, or w1/w3)
    # and should be quantized, records it under its base name and role; raises value error on a
    # duplicate role for a base. Then for each base, requires both gate and up (value error naming
    # the roles found otherwise) and quantizes the pair jointly so they share one global scale,
    # remembering outputs by name. Pass 2 builds the result in input order: pairs that should not be
    # quantized pass through; quantized ones emit (name, packed data), (name with ".weight" replaced by
    # ".weight_scale", block scales) and (".weight_scale_2", global scale), using the joint result
    # for gate/up and the individual quantizer otherwise (e.g. down projections).
    # Build/hash the policy once per conversion batch, not once per weight check.
    literal_rules = _literal_ignore_rules(tuple(ignore_rules))
    gated_candidates = {}
    for converted_name, param in converted_named_params:
        base, role = _split_gated_pair_name(converted_name)
        if base is None or role is None:
            continue
        if _should_quantize_param(converted_name, param, ignore_rules, literal_rules):
            roles = gated_candidates.setdefault(base, {})
            if role in roles:
                raise ValueError(
                    f"NVFP4 requires a single complete gate/up pair per conversion batch; "
                    f"found duplicate {role} tensor for {base}."
                )
            roles[role] = (converted_name, param)

    paired_outputs = {}
    for base, roles in gated_candidates.items():
        if set(roles) != {"gate", "up"}:
            present = ", ".join(sorted(roles))
            raise ValueError(
                f"NVFP4 requires gate/up tensors to be quantized together so they can share "
                f"one global amax; found only {{{present}}} for {base}."
            )
        gate_name, gate_weight = roles["gate"]
        up_name, up_weight = roles["up"]
        gate_output, up_output = nvfp4_quantize_1d_pair(gate_weight, up_weight)
        paired_outputs[gate_name] = gate_output
        paired_outputs[up_name] = up_output

    quantize_named_params = []
    for converted_name, param in converted_named_params:
        if not _should_quantize_param(converted_name, param, ignore_rules, literal_rules):
            quantize_named_params.append((converted_name, param))
            continue
        if converted_name in paired_outputs:
            qweight, block_scale, weight_scale_2 = paired_outputs[converted_name]
        else:
            qweight, block_scale, weight_scale_2 = quantize_nvfp4(param)
        quantize_named_params.append((converted_name, qweight))
        quantize_named_params.append((converted_name.replace(".weight", ".weight_scale"), block_scale))
        quantize_named_params.append((converted_name.replace(".weight", ".weight_scale_2"), weight_scale_2))

    return quantize_named_params


def _should_quantize_param(name, weight, ignore_rules, literal_rules=None):
    # [FUNCTION DOC] HELPER: False if the name is ignored by the rules, does not end with ".weight",
    # the dtype is not half/bfloat16/float32, or it has fewer than 2 dimensions. Otherwise raises a
    # value error if the last dimension is not divisible by the NVFP4 group size, else true.
    if ignore_rules and _is_ignored(name, ignore_rules, literal_rules):
        return False
    if not name.endswith(".weight"):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {weight.shape[-1]} must be divisible by {NVFP4_GROUP_SIZE} for NVFP4 ({name}).")
    return True


def _split_gated_pair_name(name: str):
    # [FUNCTION DOC] HELPER: If the name ends with one of the gate/up suffixes, returns (name without
    # the suffix, "gate" or "up"); otherwise (none, none). The first matching suffix in table order
    # wins.
    for suffix, role in GATED_PAIR_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def _quantize_nvfp4_1d(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Makes the 2-D weight contiguous, raises value error if the column
    # count is not divisible by the group size, and delegates to the project's 1-D NVFP4 quantizer.
    # Returns (packed data, block scales, global scale) as described below.
    """
    NVFP4 1D quantization (tile shape = 1x16).

    Returns:
      qweight: uint8 packed fp4, shape (M, K // 2)
      block_scale: float8_e4m3fn, shape (M, K // 16)
      global_scale: float32 scalar tensor
    """
    weight = weight.contiguous()
    _, n = weight.shape
    if n % NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"NVFP4 requires K divisible by {NVFP4_GROUP_SIZE}, got {n}.")

    return nvfp4_quantize_1d(weight)


def quantize_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Quantizes a 2-D weight directly; for a 3-D weight (stack of matrices, e.g.
    # grouped experts) quantizes each slice along axis 0 independently and stacks the data, block
    # scales and global scales along a new axis 0 (one global scale per slice). Raises value error
    # for other ranks. Returns (packed data, block scales, global scale(s)).
    if weight.dim() == 2:
        return _quantize_nvfp4_1d(weight)
    if weight.dim() == 3:
        qweights = []
        block_scales = []
        global_scales = []
        for idx in range(weight.shape[0]):
            qweight, block_scale, global_scale = _quantize_nvfp4_1d(weight[idx])
            qweights.append(qweight)
            block_scales.append(block_scale)
            global_scales.append(global_scale)
        return (
            torch.stack(qweights, dim=0),
            torch.stack(block_scales, dim=0),
            torch.stack(global_scales, dim=0),
        )
    raise ValueError(f"Unsupported weight rank {weight.dim()} for NVFP4 quantization.")
