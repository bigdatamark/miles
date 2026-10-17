# [MODULE DOC] Per-model-family true-on-policy capabilities. Defines the profile record (supported
# train/rollout parallel layouts, Megatron model type names per model name, backend support
# flags, bound contract), the one registered profile for dense Qwen3 models, and lookup helpers
# keyed by model name. Layout names used are "dp", "tp", "pp" and "ulysses_cp".

from __future__ import annotations

from dataclasses import dataclass

from .contracts import QWEN3_DENSE_TRUE_ON_POLICY_V1, LogprobContract, ModelFamily, TrueOnPolicyContract

ParallelLayout = str


@dataclass(frozen=True)
class TrueOnPolicyModelProfile:
    # [CLASS DOC] Immutable capability record for one model family: family name; the model names
    # it covers; a mapping from model name to Megatron model type string; the supported training
    # and rollout layout name tuples; the bound contract; and flags for Megatron and FSDP support
    # (both default true).
    """Model-specific true-on-policy capabilities and launch defaults."""

    family: ModelFamily
    model_names: tuple[str, ...]
    megatron_model_types: dict[str, str]
    supported_train_layouts: tuple[ParallelLayout, ...]
    supported_rollout_layouts: tuple[ParallelLayout, ...]
    contract: TrueOnPolicyContract
    supports_megatron: bool = True
    supports_fsdp: bool = True

    @property
    def required_kernel_contracts(self):
        # [FUNCTION DOC] HELPER: Pass-through to the bound contract's required kernel contracts.
        return self.contract.required_kernel_contracts

    @property
    def logprob_contract(self) -> LogprobContract:
        # [FUNCTION DOC] HELPER: Pass-through to the bound contract's logprob contract.
        return self.contract.logprob_contract

    @property
    def sglang_attention_backend(self) -> str:
        # [FUNCTION DOC] HELPER: Pass-through to the bound contract's SGLang attention backend.
        return self.contract.sglang_attention_backend

    @property
    def fsdp_attention_implementation(self) -> str:
        # [FUNCTION DOC] HELPER: Pass-through to the bound contract's FSDP attention
        # implementation.
        return self.contract.fsdp_attention_implementation

    @property
    def disable_megatron_sequence_parallel(self) -> bool:
        # [FUNCTION DOC] HELPER: Pass-through to the bound contract's flag for disabling Megatron
        # sequence parallelism.
        return self.contract.disable_megatron_sequence_parallel

    @property
    def supports_ulysses_cp(self) -> bool:
        # [FUNCTION DOC] Returns true when the Ulysses context-parallel layout name is among the
        # supported training layouts.
        return "ulysses_cp" in self.supported_train_layouts

    @property
    def supports_tp_invariant(self) -> bool:
        # [FUNCTION DOC] Returns true when the tensor-parallel layout name is among either the
        # supported training layouts or the supported rollout layouts.
        return "tp" in self.supported_train_layouts or "tp" in self.supported_rollout_layouts

    def megatron_model_type_for(self, model_name: str) -> str:
        # [FUNCTION DOC] Maps a model name to its Megatron model type string using this profile's
        # mapping. Raises a value error (chained from the missing-key error) that names the
        # profile family and lists the sorted supported names when the model name is absent.
        try:
            return self.megatron_model_types[model_name]
        except KeyError as exc:
            supported = ", ".join(sorted(self.megatron_model_types))
            raise ValueError(
                f"{model_name!r} does not have a Megatron model type in profile "
                f"{self.family!r}; supported names: {supported}"
            ) from exc


QWEN3_DENSE_PROFILE = TrueOnPolicyModelProfile(
    family="qwen3_dense",
    model_names=(
        "Qwen3-0.6B",
        "Qwen3-4B",
        "Qwen3-4B-Base",
        "Qwen3-4B-Instruct-2507",
    ),
    megatron_model_types={
        "Qwen3-0.6B": "qwen3-0.6B",
        "Qwen3-4B": "qwen3-4B",
        "Qwen3-4B-Base": "qwen3-4B",
        "Qwen3-4B-Instruct-2507": "qwen3-4B-Instruct-2507",
    },
    supported_train_layouts=("dp", "tp", "pp", "ulysses_cp"),
    supported_rollout_layouts=("dp", "tp"),
    contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
)


_MODEL_PROFILES = (QWEN3_DENSE_PROFILE,)
_PROFILE_BY_MODEL_NAME = {model_name: profile for profile in _MODEL_PROFILES for model_name in profile.model_names}


def get_true_on_policy_model_profile(model_name: str) -> TrueOnPolicyModelProfile:
    # [FUNCTION DOC] Finds the profile whose model names include the given exact model name.
    # Returns it. Raises a value error listing the sorted supported model names (chained from the
    # missing-key error) when none matches.
    try:
        return _PROFILE_BY_MODEL_NAME[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILE_BY_MODEL_NAME))
        raise ValueError(
            f"true-on-policy does not have a model profile for {model_name!r}. " f"Supported models: {supported}"
        ) from exc


def get_megatron_model_type(model_name: str) -> str:
    # [FUNCTION DOC] Convenience lookup: resolves the profile for the model name, then returns that
    # profile's Megatron model type. Raises a value error from either lookup when unsupported.
    return get_true_on_policy_model_profile(model_name).megatron_model_type_for(model_name)
