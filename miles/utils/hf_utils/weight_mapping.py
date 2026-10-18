# [MODULE DOC]
# Maps checkpoint (huggingface) weight names to the names of the corresponding model parameters,
# using transformers' conversion rules (renamings and converters) and a meta-device instantiation of
# the model to learn parameter names.
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming
from transformers.models.auto.auto_factory import _get_model_class


@dataclass(frozen=True)
class HfWeightMapping:
    # [CLASS DOC] Immutable mapping data: the set of two- and three-dimensional parameter names of
    # the model, and the tuple of conversion rules.
    parameter_names: frozenset[str]
    conversions: tuple = ()

    @classmethod
    def from_config(cls, config):
        # [FUNCTION DOC] Builds the mapping from a model config. Tries the image-text model auto class
        # then the causal lm auto class; for the first whose model mapping contains the config's type:
        # resolves the model class' config class (rebuilding the config from a dict if the type
        # differs), instantiates the model on the meta device with eager attention and a forked
        # random state (no weights allocated), collects names of parameters with 2 or 3 dimensions
        # (duplicates kept), and fetches conversion rules without legacy ones. If no auto class
        # matches, returns an empty mapping (names kept as in the checkpoint).
        # VLMs may also register a CausalLM compatibility class that drops the vision/text namespace.
        for auto_model in (AutoModelForImageTextToText, AutoModelForCausalLM):
            if type(config) in auto_model._model_mapping:
                # HF's lazy mapping also matches remote-code configs by class name.
                config_class = _get_model_class(config, auto_model._model_mapping).config_class
                if type(config) is not config_class:
                    config = config_class.from_dict(config.to_dict())
                # Only structure is needed; never allocate or load base weights.
                with torch.random.fork_rng(devices=[]), torch.device("meta"):
                    model = auto_model.from_config(config, attn_implementation="eager")
                parameter_names = frozenset(
                    name for name, param in model.named_parameters(remove_duplicate=False) if param.ndim in (2, 3)
                )
                return cls(parameter_names, tuple(get_model_conversion_mapping(model, add_legacy=False)))
        # Custom HF implementations without native conversion rules retain their checkpoint namespace.
        return cls(frozenset())

    def model_parameter(self, checkpoint_name):
        # [FUNCTION DOC] Returns the model parameter name for a checkpoint weight name. If the name
        # (optionally without a weight suffix) is already a parameter name, returns it (suffix
        # stripped if that form matched). Otherwise applies all renaming rules in order to the name,
        # then the first converter whose source pattern matches yields its target (asserting the
        # converter has exactly one target). If nothing matches returns the (renamed) name.
        if checkpoint_name.removesuffix(".weight") in self.parameter_names:
            checkpoint_name = checkpoint_name.removesuffix(".weight")
        if checkpoint_name in self.parameter_names:
            return checkpoint_name
        for conversion in self.conversions:
            if isinstance(conversion, WeightRenaming):
                checkpoint_name, _ = conversion.rename_source_key(checkpoint_name)
        for conversion in self.conversions:
            if isinstance(conversion, WeightConverter):
                target, source_pattern = conversion.rename_source_key(checkpoint_name)
                if source_pattern is not None:
                    assert (
                        len(conversion.target_patterns) == 1
                    ), f"HF target binding does not support one-to-many conversion of {checkpoint_name!r}"
                    return target
        return checkpoint_name
