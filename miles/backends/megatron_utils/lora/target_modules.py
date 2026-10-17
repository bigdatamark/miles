# [MODULE DOC]
# Translation between Hugging Face LoRA target names (e.g. projection names) and the Megatron
# module names that adapters are attached to, driven by the Megatron-Bridge weight mapping
# registry. Megatron fuses several HF projections into one module (query/key/value into one
# linear, gate/up into one), so selecting targets must respect fusion: a fused module is adapted
# only if all its HF parameters are selected, except in "canonical" LoRA mode which splits fused
# modules into separate per-projection adapters. Two public functions: resolve HF targets to
# Megatron adapter module names for a built model, and normalize arbitrary (legacy Megatron or
# HF) selectors into HF module names before the trainer exists.
import re

from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora.utils import matches_lora_target

_CANONICAL_PROJECTIONS = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
}


def _canonical_adapter_module(module, checkpoint_parameter):
    # [FUNCTION DOC] HELPER: Maps a fused Megatron module name plus one HF checkpoint parameter name
    # to the split-adapter module name for canonical LoRA. Takes the parameter's last path component
    # without a trailing weight suffix, asserts it is one of the known projections (q, k, v, gate,
    # up; assertion error otherwise), and returns the module's parent path joined with the matching
    # canonical name (linear_q, linear_k, linear_v, linear_fc1_gate, linear_fc1_up).
    leaf = checkpoint_parameter.removesuffix(".weight").rsplit(".", 1)[-1]
    assert leaf in _CANONICAL_PROJECTIONS, f"CanonicalLoRA has no split adapter for {checkpoint_parameter!r}"
    return f"{module.rsplit('.', 1)[0]}.{_CANONICAL_PROJECTIONS[leaf]}"


def _checkpoint_parameters(mapping):
    # [FUNCTION DOC] HELPER: The set of HF checkpoint parameter names of a weight mapping: a
    # singleton set if the mapping's HF parameter is a single string, otherwise the set of values
    # of its name dictionary (fused mappings).
    return {mapping.hf_param} if isinstance(mapping.hf_param, str) else set(mapping.hf_param.values())


def _select_checkpoint_parameters(checkpoint_parameters, targets, *, hf_mapping, megatron_module, exclude_modules):
    # [FUNCTION DOC] HELPER: Returns the subset of checkpoint parameter names that should receive
    # adapters. Converts each checkpoint name to its HF model parameter name via the HF mapping.
    # A parameter is selected if (the HF model has no parameter-name inventory, or the name is in it;
    # this drops auxiliary layers like MTP absent from the HF model) and its name without the weight
    # suffix matches any target. Then drops any selected parameter whose HF module name or the
    # Megatron module name matches any exclude pattern.
    hf_parameters = {name: hf_mapping.model_parameter(name) for name in checkpoint_parameters}
    # Bridge can expose auxiliary layers, such as MTP, absent from the HF model.
    selected = {
        checkpoint_parameter
        for checkpoint_parameter, hf_parameter in hf_parameters.items()
        if (not hf_mapping.parameter_names or hf_parameter in hf_mapping.parameter_names)
        and any(matches_lora_target(hf_parameter.removesuffix(".weight"), target) for target in targets)
    }
    return {
        checkpoint_parameter
        for checkpoint_parameter in selected
        if not any(
            matches_lora_target(hf_parameters[checkpoint_parameter].removesuffix(".weight"), target)
            or matches_lora_target(megatron_module, target)
            for target in exclude_modules
        )
    }


def _resolve_adapter_targets(megatron_module, checkpoint_parameters, selected, *, canonical):
    # [FUNCTION DOC] HELPER: Decides which Megatron module name(s) carry adapters for one matched
    # Megatron module. Canonical mode on a non-expert fused QKV or FC1 module with more than one
    # checkpoint parameter: returns the de-duplicated split-adapter module names of the selected
    # parameters (a subset is allowed). Otherwise asserts all of the module's checkpoint parameters
    # were selected (assertion error: fused module needs all HF targets). Canonical mode on an
    # expert FC1 module: returns two names, the module name with a gate suffix and with an up suffix
    # (split aliases required even for fused expert adapters). Everything else: a one-element list
    # with the module name.
    if (
        canonical
        and len(checkpoint_parameters) > 1
        and ".experts." not in megatron_module
        and megatron_module.rsplit(".", 1)[-1] in ("linear_qkv", "linear_fc1")
    ):
        return list(dict.fromkeys(_canonical_adapter_module(megatron_module, name) for name in selected))
    assert (
        selected == checkpoint_parameters
    ), f"LoRA on fused module {megatron_module!r} requires all HF targets with this adapter"
    if canonical and ".experts." in megatron_module and megatron_module.endswith(".linear_fc1"):
        # CanonicalLoRA requires split aliases even for fused expert adapters.
        return [f"{megatron_module}_gate", f"{megatron_module}_up"]
    return [megatron_module]


def resolve_megatron_lora_targets(targets, mappings, *, parameter_names, hf_mapping, canonical, exclude_modules=()):
    # [FUNCTION DOC]
    # Returns an ordered, de-duplicated list of Megatron module names to attach adapters to. For
    # each weight mapping from the bridge registry whose Megatron parameter ends in a weight (or a
    # starred weight pattern for per-layer/per-expert templates): skip other mappings; turn the
    # parameter pattern into a regular expression where each star matches any text; for every
    # not-yet-matched real Megatron parameter name (sorted; so each name is consumed by the first
    # mapping that matches it) that fully matches, resolve the mapping with the captured groups into
    # HF checkpoint parameter names and select those matching targets and not excluded. If no
    # instance of this mapping selects anything, skip it. Otherwise asserts every instance selected
    # something (assertion error: cannot select a subset across layers/experts), resolves the adapter
    # module names per instance, asserts all instances yield the identical set of adapter names
    # (assertion error otherwise), and accumulates them. Finally asserts the result is non-empty
    # (assertion error: no Megatron modules). Inputs: targets (HF-style patterns), mappings,
    # the set of all global parameter names, an HF mapping object, a canonical flag, and optional
    # exclude patterns (default none).
    adapter_targets = {}
    matched_megatron_parameters = set()
    for mapping in mappings:
        megatron_module, weight = mapping.megatron_param.rsplit(".", 1)
        if weight not in ("weight", "weight*"):
            continue
        regex = re.compile(re.escape(mapping.megatron_param).replace(r"\*", "(.*)"))
        selections = []
        for megatron_parameter in sorted(parameter_names - matched_megatron_parameters):
            match = regex.fullmatch(megatron_parameter)
            if match is None:
                continue
            matched_megatron_parameters.add(megatron_parameter)
            checkpoint_parameters = _checkpoint_parameters(mapping.resolve(match.groups()))
            selected = _select_checkpoint_parameters(
                checkpoint_parameters,
                targets,
                hf_mapping=hf_mapping,
                megatron_module=megatron_module,
                exclude_modules=exclude_modules,
            )
            selections.append((checkpoint_parameters, selected))
        if not any(selected for _, selected in selections):
            continue
        # Template injection must select the same projections for every layer/expert it matches.
        assert all(
            selected for _, selected in selections
        ), f"LoRA cannot select a subset of parameters in {mapping.megatron_param!r}"
        adapter_selections = set()
        for checkpoint_parameters, selected in selections:
            selected_adapters = _resolve_adapter_targets(
                megatron_module, checkpoint_parameters, selected, canonical=canonical
            )
            adapter_selections.add(frozenset(selected_adapters))
            adapter_targets.update(dict.fromkeys(selected_adapters))
        assert (
            len(adapter_selections) == 1
        ), f"LoRA cannot select different projections across parameters in {mapping.megatron_param!r}"
    assert adapter_targets, "LoRA targets have no Megatron modules"
    return list(adapter_targets)


def normalize_lora_targets_to_hf(hf_checkpoint, target_modules, *, canonical, exclude_modules, explicit_targets=()):
    # [FUNCTION DOC]
    # Converts user target selectors (which may be legacy Megatron names or HF names) into a sorted
    # list of HF module names, using Bridge loaded from the HF checkpoint (remote code trusted; the
    # import is local because only legacy selectors need Bridge before the trainer exists). For
    # each weight mapping (as in resolve), for each HF checkpoint parameter of the mapping: derive
    # the HF module name; skip it if the HF model has a parameter inventory and the module matches no
    # name in it; for each target, a match is: the HF module or the Megatron module name matches the
    # target, or (canonical mode, fused QKV/FC1 non-expert module) the split-adapter module name
    # matches. On a match record the target as matched; compute whether any exclude pattern matches
    # the HF or Megatron module; asserts an explicitly requested target is not excluded (assertion
    # error naming the overlap); if not excluded add the HF module to the result. After all
    # mappings, asserts every target matched at least one module (error lists unmatched targets) and
    # that the result is not empty (all excluded). Network/disk access via Bridge.
    # Only legacy Megatron selectors need Bridge before trainer creation.
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)
    hf_mapping = HfWeightMapping.from_config(bridge.hf_pretrained.config)
    model_bridge = bridge._model_bridge
    model_bridge.hf_pretrained = bridge.hf_pretrained
    selected_hf_modules, matched_targets = set(), set()
    for mapping in model_bridge.mapping_registry().get_all_mappings():
        megatron_module, weight = mapping.megatron_param.rsplit(".", 1)
        if weight not in ("weight", "weight*"):
            continue
        for checkpoint_parameter in _checkpoint_parameters(mapping):
            hf_module = hf_mapping.model_parameter(checkpoint_parameter).removesuffix(".weight")
            if hf_mapping.parameter_names and not any(
                matches_lora_target(name.removesuffix(".weight"), hf_module) for name in hf_mapping.parameter_names
            ):
                continue
            for target in target_modules:
                matches = matches_lora_target(hf_module, target) or matches_lora_target(megatron_module, target)
                if (
                    canonical
                    and megatron_module.rsplit(".", 1)[-1] in ("linear_qkv", "linear_fc1")
                    and ".experts." not in megatron_module
                ):
                    matches |= matches_lora_target(
                        _canonical_adapter_module(megatron_module, checkpoint_parameter), target
                    )
                if matches:
                    matched_targets.add(target)
                    excluded = any(
                        matches_lora_target(hf_module, exclude) or matches_lora_target(megatron_module, exclude)
                        for exclude in exclude_modules
                    )
                    assert not (
                        excluded and target in explicit_targets
                    ), f"Explicit LoRA target {target!r} overlaps --exclude-modules at {hf_module!r}"
                    if not excluded:
                        selected_hf_modules.add(hf_module)
    assert (
        set(target_modules) <= matched_targets
    ), f"LoRA targets have no Bridge mapping: {sorted(set(target_modules) - matched_targets)}"
    assert selected_hf_modules, "No LoRA targets remain after applying --exclude-modules"
    return sorted(selected_hf_modules)
