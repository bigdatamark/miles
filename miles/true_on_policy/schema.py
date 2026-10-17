# [MODULE DOC] Declarative, dependency-free description of true-on-policy contracts: the literal
# type aliases naming contracts, model families, kernel contracts and logprob contracts, the frozen
# schema record, and the single concrete schema instance for dense Qwen3. Higher layers wrap this
# record; nothing here has behavior or side effects.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TrueOnPolicyContractName = Literal["qwen3_dense_true_on_policy_v1"]
ModelFamily = Literal["qwen3_dense", "qwen3_moe", "qwen3_next"]
KernelContract = Literal["qwen3_dense_sglang_math"]
LogprobContract = Literal["sglang_prefill"]


@dataclass(frozen=True)
class TrueOnPolicyContractSchema:
    # [CLASS DOC] Immutable record identifying a parity contract shared across repos (Miles,
    # SGLang, Megatron). Fields: contract name; the model family it targets; the tuple of kernel
    # contracts the backends must implement; the log-probability contract; the SGLang attention
    # backend name; the Hugging Face attention implementation name used under FSDP; and whether
    # Megatron sequence parallelism must be turned off.
    """Declarative cross-repo identity for a true-on-policy parity contract."""

    name: TrueOnPolicyContractName
    model_family: ModelFamily
    required_kernel_contracts: tuple[KernelContract, ...]
    logprob_contract: LogprobContract
    sglang_attention_backend: str
    fsdp_attention_implementation: str
    disable_megatron_sequence_parallel: bool


QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="qwen3_dense_true_on_policy_v1",
    model_family="qwen3_dense",
    required_kernel_contracts=("qwen3_dense_sglang_math",),
    logprob_contract="sglang_prefill",
    sglang_attention_backend="fa3",
    fsdp_attention_implementation="flash_attention_3",
    disable_megatron_sequence_parallel=True,
)
