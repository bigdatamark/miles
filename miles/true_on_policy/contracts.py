# [MODULE DOC] Wraps the declarative contract schemas in a behavior-bearing contract object and
# keeps the registry of known contracts by name. The contract object exposes schema fields as
# read-only properties and derives the kernel/runtime switch values from the training backend and
# the SGLang rollout target. Only the dense Qwen3 v1 contract is registered.

from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
    KernelContract,
    LogprobContract,
    ModelFamily,
    TrueOnPolicyContractName,
    TrueOnPolicyContractSchema,
)


@dataclass(frozen=True)
class TrueOnPolicyContract:
    # [CLASS DOC] Immutable wrapper around one contract schema. All properties below are plain
    # read-only pass-throughs to the same-named schema field; the only added behavior is the
    # derivation of kernel policy switches.
    """Internal parity contract selected by Miles and implemented by each backend."""

    schema: TrueOnPolicyContractSchema

    @property
    def name(self) -> TrueOnPolicyContractName:
        # [FUNCTION DOC] HELPER: Returns the schema's contract name.
        return self.schema.name

    @property
    def model_family(self) -> ModelFamily:
        # [FUNCTION DOC] HELPER: Returns the schema's target model family.
        return self.schema.model_family

    @property
    def required_kernel_contracts(self) -> tuple[KernelContract, ...]:
        # [FUNCTION DOC] HELPER: Returns the schema's tuple of required kernel contracts.
        return self.schema.required_kernel_contracts

    @property
    def logprob_contract(self) -> LogprobContract:
        # [FUNCTION DOC] HELPER: Returns the schema's log-probability contract.
        return self.schema.logprob_contract

    @property
    def sglang_attention_backend(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the schema's SGLang attention backend name.
        return self.schema.sglang_attention_backend

    @property
    def fsdp_attention_implementation(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the schema's FSDP attention implementation name.
        return self.schema.fsdp_attention_implementation

    @property
    def disable_megatron_sequence_parallel(self) -> bool:
        # [FUNCTION DOC] HELPER: Returns the schema's flag saying Megatron sequence parallelism
        # must be disabled.
        return self.schema.disable_megatron_sequence_parallel

    def kernel_policy_kwargs_for(
        self,
        *,
        train_backend: str,
        sglang_target: str,
    ) -> dict[str, object]:
        # [FUNCTION DOC] Builds the keyword values for the kernel policy record (used as keyword
        # arguments by the config layer). Inputs: train backend name and SGLang target name, both
        # keyword-only. Returns a mapping of nine switch names to values: deterministic inference
        # and deterministic training are always true; the SGLang attention backend comes from this
        # contract; the Megatron-uses-SGLang-backend, rope-fusion-disable, bias-swiglu-fusion-
        # disable and batch-invariant-mode switches are true exactly when the backend is
        # "megatron"; the tp-invariant row linear and deterministic tp all-reduce switches are
        # true exactly when the target is "fsdp_tp". Unknown names are not validated (they just
        # yield false). No side effects.
        uses_megatron = train_backend == "megatron"
        uses_tp_invariant_rollout = sglang_target == "fsdp_tp"
        return {
            "deterministic_inference": True,
            "deterministic_training": True,
            "sglang_attention_backend": self.sglang_attention_backend,
            "megatron_uses_sglang_backend": uses_megatron,
            "disable_rope_fusion": uses_megatron,
            "disable_bias_swiglu_fusion": uses_megatron,
            "batch_invariant_mode": uses_megatron,
            "tp_invariant_row_linear": uses_tp_invariant_rollout,
            "deterministic_tp_allreduce": uses_tp_invariant_rollout,
        }


QWEN3_DENSE_TRUE_ON_POLICY_V1 = TrueOnPolicyContract(
    schema=QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
)


_CONTRACT_BY_NAME = {
    QWEN3_DENSE_TRUE_ON_POLICY_V1.name: QWEN3_DENSE_TRUE_ON_POLICY_V1,
}


def get_true_on_policy_contract(name: str) -> TrueOnPolicyContract:
    # [FUNCTION DOC] Looks up a registered contract by name. Returns the contract object. Raises a
    # value error listing the sorted supported names (chained from the missing-key error) when the
    # name is unknown.
    try:
        return _CONTRACT_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONTRACT_BY_NAME))
        raise ValueError(f"Unsupported true-on-policy contract {name!r}. Supported contracts: {supported}") from exc
