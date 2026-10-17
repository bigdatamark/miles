# [MODULE DOC] Turns the public true-on-policy switch plus parsed run arguments into a validated
# configuration and an immutable launch plan. The plan carries the extra command-line flag lists
# for SGLang, Megatron, FSDP and Miles itself, plus environment variables, and a derived kernel
# policy. Also provides a script-time hook that adjusts parsed arguments before command assembly.
# Layout vocabulary: the SGLang target is "fsdp_tp" when any tensor parallelism is involved
# (training or rollout engine spanning more than one GPU), otherwise "fsdp".

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Literal

from .contracts import TrueOnPolicyContract, get_true_on_policy_contract
from .model_profiles import TrueOnPolicyModelProfile, get_true_on_policy_model_profile


OnPolicyTarget = Literal["fsdp", "fsdp_tp"]
TrainBackend = Literal["fsdp", "megatron"]


@dataclass(frozen=True)
class TrueOnPolicyArgList:
    # [CLASS DOC] Immutable ordered tuple of command-line tokens (flags and their values) that is
    # only joined into a shell string at the point of launching.
    """Structured command-line args that stringify only at launch boundaries."""

    values: tuple[str, ...] = ()

    def as_cli_string(self) -> str:
        # [FUNCTION DOC] Renders the tokens as one shell-safe string: each token is shell-quoted,
        # tokens are joined by single spaces, and a trailing space is appended so lists can be
        # concatenated directly. Returns an empty string (no trailing space) when there are no
        # tokens.
        if not self.values:
            return ""
        return " ".join(shlex.quote(value) for value in self.values) + " "

    def contains(self, flag: str) -> bool:
        # [FUNCTION DOC] Returns true if the exact token is present anywhere in the tuple (an exact
        # match on any token, flag or value).
        return flag in self.values


@dataclass(frozen=True)
class TrueOnPolicyParallelLayout:
    # [CLASS DOC] Immutable sizes of the training tensor/context/pipeline parallel groups and the
    # number of GPUs per rollout engine. Each "uses_*" property is true when the matching size is
    # strictly greater than one.
    """Training and rollout topology relevant to true-on-policy parity."""

    train_tensor_parallel_size: int
    train_context_parallel_size: int
    train_pipeline_parallel_size: int
    rollout_num_gpus_per_engine: int

    @property
    def uses_train_tp(self) -> bool:
        # [FUNCTION DOC] True when the training tensor parallel size exceeds one.
        return self.train_tensor_parallel_size > 1

    @property
    def uses_ulysses_cp(self) -> bool:
        # [FUNCTION DOC] True when the training context parallel size exceeds one (context
        # parallelism is treated as Ulysses-style).
        return self.train_context_parallel_size > 1

    @property
    def uses_train_pp(self) -> bool:
        # [FUNCTION DOC] True when the training pipeline parallel size exceeds one.
        return self.train_pipeline_parallel_size > 1

    @property
    def uses_rollout_tp(self) -> bool:
        # [FUNCTION DOC] True when a rollout engine spans more than one GPU.
        return self.rollout_num_gpus_per_engine > 1


@dataclass(frozen=True)
class TrueOnPolicyKernelPolicy:
    # [CLASS DOC] Immutable bundle of kernel/runtime switches (fields are filled from the
    # contract's kernel policy mapping) plus the contract itself. Provides builders for the
    # SGLang flags, the Megatron flags and the environment variables implied by the switches.
    # KNOWN ISSUE: the fields deterministic_training, tp_invariant_row_linear and
    # deterministic_tp_allreduce are never read by any builder in this module.
    """Kernel/runtime switches required to keep SGLang and Megatron aligned."""

    contract: TrueOnPolicyContract
    deterministic_inference: bool
    deterministic_training: bool
    sglang_attention_backend: str
    megatron_uses_sglang_backend: bool
    disable_rope_fusion: bool
    disable_bias_swiglu_fusion: bool
    batch_invariant_mode: bool
    tp_invariant_row_linear: bool
    deterministic_tp_allreduce: bool

    def build_sglang_args(self) -> TrueOnPolicyArgList:
        # [FUNCTION DOC] Builds the SGLang-side flags: the contract-name flag with the contract
        # name, then the attention-backend flag with the backend. When deterministic inference is
        # on, the enable-deterministic-inference flag is inserted at the very front. Pure.
        values = [
            "--sglang-true-on-policy-contract",
            self.contract.name,
            "--sglang-attention-backend",
            self.sglang_attention_backend,
        ]
        if self.deterministic_inference:
            values.insert(0, "--sglang-enable-deterministic-inference")
        return TrueOnPolicyArgList(tuple(values))

    def build_megatron_args(self) -> TrueOnPolicyArgList:
        # [FUNCTION DOC] Builds the Megatron-side flags in this order, each conditional on its
        # switch: when Megatron uses the SGLang backend, the true-on-policy-contract flag with the
        # contract name, the transformer-impl flag set to "local", and the use-cpu-initialization
        # flag; then batch-invariant-mode; then no-bias-swiglu-fusion if that fusion is disabled;
        # then no-rope-fusion if rope fusion is disabled. Pure.
        values: list[str] = []
        if self.megatron_uses_sglang_backend:
            values.extend(
                [
                    "--true-on-policy-contract",
                    self.contract.name,
                    "--transformer-impl",
                    "local",
                    "--use-cpu-initialization",
                ]
            )
        if self.batch_invariant_mode:
            values.append("--batch-invariant-mode")
        if self.disable_bias_swiglu_fusion:
            values.append("--no-bias-swiglu-fusion")
        if self.disable_rope_fusion:
            values.append("--no-rope-fusion")
        return TrueOnPolicyArgList(tuple(values))

    def build_env_vars(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns the environment variables to set for the run: NCCL_ALGO (the
        # current process environment's value if set, otherwise "Ring"), NVTE_ALLOW_NONDETERMINISTIC_ALGO
        # set to "0", and CUBLAS_WORKSPACE_CONFIG set to ":4096:8". Reads the environment but does
        # not modify it. Ignores every policy field (always the same set).
        return {
            "NCCL_ALGO": os.environ.get("NCCL_ALGO", "Ring"),
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }


@dataclass(frozen=True)
class TrueOnPolicyLaunchPlan:
    # [CLASS DOC] Immutable result of planning a run. When not enabled, every other field keeps its
    # default (absent objects, empty argument lists, empty environment mapping). When enabled all
    # fields are populated by the config's launch plan builder. Carries four separate argument
    # lists (SGLang, Megatron, FSDP, Miles) and the environment variable mapping.
    """Derived cross-repo launch contract for one true-on-policy run."""

    enabled: bool
    model_profile: TrueOnPolicyModelProfile | None = None
    contract: TrueOnPolicyContract | None = None
    train_backend: TrainBackend | None = None
    sglang_target: OnPolicyTarget | None = None
    parallel_layout: TrueOnPolicyParallelLayout | None = None
    kernel_policy: TrueOnPolicyKernelPolicy | None = None
    sglang_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    megatron_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    fsdp_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    miles_args: TrueOnPolicyArgList = field(default_factory=TrueOnPolicyArgList)
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def train_args(self) -> str:
        # [FUNCTION DOC] Returns one command-line string concatenating, in order, the SGLang,
        # Megatron, FSDP and Miles argument lists, each rendered with a trailing space (empty
        # lists contribute nothing). Pure.
        return (
            self.sglang_args.as_cli_string()
            + self.megatron_args.as_cli_string()
            + self.fsdp_args.as_cli_string()
            + self.miles_args.as_cli_string()
        )


@dataclass(frozen=True)
class TrueOnPolicyConfig:
    # [CLASS DOC] Immutable validated-on-demand configuration for an enabled run: model profile,
    # train backend ("fsdp" or "megatron"), the three training parallel sizes, rollout GPUs per
    # engine and an optional contract name that overrides the profile's contract. Offers derived
    # properties, validation, kernel policy building and launch plan building.
    # KNOWN ISSUE: the enabled field is only consulted by validate; the launch plan builder always
    # reports enabled even if this field is false.
    """Typed contract derived from the single public true-on-policy switch."""

    enabled: bool
    model_profile: TrueOnPolicyModelProfile
    train_backend: TrainBackend
    tensor_model_parallel_size: int
    context_parallel_size: int
    pipeline_model_parallel_size: int
    rollout_num_gpus_per_engine: int
    contract_override: str | None = None

    @property
    def parallel_layout(self) -> TrueOnPolicyParallelLayout:
        # [FUNCTION DOC] Builds a parallel layout record from the four size fields of this
        # config (a new object on every access).
        return TrueOnPolicyParallelLayout(
            train_tensor_parallel_size=self.tensor_model_parallel_size,
            train_context_parallel_size=self.context_parallel_size,
            train_pipeline_parallel_size=self.pipeline_model_parallel_size,
            rollout_num_gpus_per_engine=self.rollout_num_gpus_per_engine,
        )

    @property
    def requires_tp_invariant_rollout(self) -> bool:
        # [FUNCTION DOC] True when either training uses tensor parallelism or the rollout engine
        # spans more than one GPU.
        layout = self.parallel_layout
        return layout.uses_train_tp or layout.uses_rollout_tp

    @property
    def sglang_target(self) -> OnPolicyTarget:
        # [FUNCTION DOC] Returns "fsdp_tp" when a tensor-parallel invariant rollout is required,
        # otherwise "fsdp". The name is used for both train backends.
        return "fsdp_tp" if self.requires_tp_invariant_rollout else "fsdp"

    @property
    def contract(self) -> TrueOnPolicyContract:
        # [FUNCTION DOC] Returns the contract named by the override when one is set (looked up in
        # the registry; an unknown name raises a value error), otherwise the model profile's
        # contract.
        if self.contract_override is not None:
            return get_true_on_policy_contract(self.contract_override)
        return self.model_profile.contract

    def validate(self) -> None:
        # [FUNCTION DOC] Checks the run against the profile's capabilities; does nothing when not
        # enabled. Raises a value error, checked in this order, when: the contract's model family
        # differs from the profile family; the backend is Megatron but the profile lacks Megatron
        # support; the backend is FSDP but the profile lacks FSDP support; context parallelism is
        # used but Ulysses CP is unsupported; pipeline parallelism is used but "pp" is not among
        # the supported train layouts; the SGLang target is "fsdp_tp" but TP invariance is
        # unsupported. Returns nothing.
        if not self.enabled:
            return
        layout = self.parallel_layout
        if self.contract.model_family != self.model_profile.family:
            raise ValueError(
                f"Contract {self.contract.name!r} is for {self.contract.model_family}, "
                f"but model profile is {self.model_profile.family}"
            )
        if self.train_backend == "megatron" and not self.model_profile.supports_megatron:
            raise ValueError(f"{self.model_profile.family} does not support Megatron true-on-policy")
        if self.train_backend == "fsdp" and not self.model_profile.supports_fsdp:
            raise ValueError(f"{self.model_profile.family} does not support FSDP true-on-policy")
        if layout.uses_ulysses_cp and not self.model_profile.supports_ulysses_cp:
            raise ValueError(f"{self.model_profile.family} does not support Ulysses CP true-on-policy")
        if layout.uses_train_pp and "pp" not in self.model_profile.supported_train_layouts:
            raise ValueError(f"{self.model_profile.family} does not support PP true-on-policy")
        if self.sglang_target == "fsdp_tp" and not self.model_profile.supports_tp_invariant:
            raise ValueError(f"{self.model_profile.family} does not support TP-invariant true-on-policy")

    def build_kernel_policy(self) -> TrueOnPolicyKernelPolicy:
        # [FUNCTION DOC] Creates the kernel policy from the contract and the contract's derived
        # switch values for this config's train backend and SGLang target. Does not validate.
        return TrueOnPolicyKernelPolicy(
            contract=self.contract,
            **self.contract.kernel_policy_kwargs_for(
                train_backend=self.train_backend,
                sglang_target=self.sglang_target,
            ),
        )

    def build_launch_plan(self) -> TrueOnPolicyLaunchPlan:
        # [FUNCTION DOC] Validates, then assembles an enabled launch plan. Miles flags are always
        # deterministic-mode, true-on-policy-mode and recompute-logprobs-via-prefill. For the
        # Megatron backend the Megatron list comes from the kernel policy and the FSDP list is
        # empty; for FSDP the Megatron list is empty and the FSDP list is the attn-implementation
        # flag with the contract's FSDP attention implementation. Any other backend raises a
        # not-implemented error. SGLang flags and environment variables come from the kernel
        # policy. Propagates validation value errors.
        self.validate()
        kernel_policy = self.build_kernel_policy()
        miles_args = TrueOnPolicyArgList(
            (
                "--deterministic-mode",
                "--true-on-policy-mode",
                "--recompute-logprobs-via-prefill",
            )
        )

        if self.train_backend == "megatron":
            megatron_args = kernel_policy.build_megatron_args()
            fsdp_args = TrueOnPolicyArgList()
        elif self.train_backend == "fsdp":
            megatron_args = TrueOnPolicyArgList()
            fsdp_args = TrueOnPolicyArgList(("--attn-implementation", self.contract.fsdp_attention_implementation))
        else:
            raise NotImplementedError(f"Unsupported true-on-policy train backend: {self.train_backend}")

        return TrueOnPolicyLaunchPlan(
            enabled=True,
            model_profile=self.model_profile,
            contract=self.contract,
            train_backend=self.train_backend,
            sglang_target=self.sglang_target,
            parallel_layout=self.parallel_layout,
            kernel_policy=kernel_policy,
            sglang_args=kernel_policy.build_sglang_args(),
            megatron_args=megatron_args,
            fsdp_args=fsdp_args,
            miles_args=miles_args,
            env_vars=kernel_policy.build_env_vars(),
        )


def _get_required_int(args: Any, name: str) -> int:
    # [FUNCTION DOC] HELPER: Reads the named attribute from the parsed arguments object and
    # returns it converted to an integer. Raises a value error if the attribute is None; a missing
    # attribute raises the usual attribute error.
    value = getattr(args, name)
    if value is None:
        raise ValueError(f"{name} must be initialized before deriving true-on-policy config")
    return int(value)


def build_true_on_policy_config(args: Any) -> TrueOnPolicyConfig | None:
    # [FUNCTION DOC] Builds a config from parsed run arguments. Returns None when the true-on-policy
    # attribute is missing or false. Otherwise looks up the model profile by the model name
    # attribute (value error if unsupported), and reads the train backend, the three parallel
    # sizes, rollout GPUs per engine (each must be non-None, else value error) and an optional
    # contract override attribute (missing means none). Does not validate the config.
    if not getattr(args, "true_on_policy", False):
        return None

    profile = get_true_on_policy_model_profile(args.model_name)
    return TrueOnPolicyConfig(
        enabled=True,
        model_profile=profile,
        train_backend=args.train_backend,
        tensor_model_parallel_size=_get_required_int(args, "tensor_model_parallel_size"),
        context_parallel_size=_get_required_int(args, "context_parallel_size"),
        pipeline_model_parallel_size=_get_required_int(args, "pipeline_model_parallel_size"),
        rollout_num_gpus_per_engine=_get_required_int(args, "rollout_num_gpus_per_engine"),
        contract_override=getattr(args, "true_on_policy_contract", None),
    )


def build_true_on_policy_launch_plan(args: Any) -> TrueOnPolicyLaunchPlan:
    # [FUNCTION DOC] Public entry point: returns a disabled (all defaults) plan when the feature
    # is off, otherwise the validated enabled plan from the config. Propagates value errors from
    # config building and validation. No side effects.
    config = build_true_on_policy_config(args)
    if config is None:
        return TrueOnPolicyLaunchPlan(enabled=False)
    return config.build_launch_plan()


def apply_true_on_policy_script_defaults(args: Any) -> None:
    # [FUNCTION DOC] Mutates the parsed arguments object in place. Does nothing when the feature
    # is off. Otherwise validates the config (value error on mismatch) and, for the Megatron
    # backend when the profile says sequence parallelism must be disabled, sets the
    # use-sequence-parallel attribute to false (overriding any user setting). Returns nothing.
    # KNOWN ISSUE: the sequence parallel decision is read from the profile; a contract override
    # with a different setting is ignored.
    """Apply derived defaults that must be visible before command assembly."""
    config = build_true_on_policy_config(args)
    if config is None:
        return

    config.validate()
    if args.train_backend == "megatron" and config.model_profile.disable_megatron_sequence_parallel:
        args.use_sequence_parallel = False
