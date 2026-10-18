# [MODULE DOC] Tests for the true-on-policy configuration layer (miles.true_on_policy): model
# profiles, contracts, script-default application, and the launch plan built from a parsed
# argument namespace. Arguments are simulated with a simple attribute namespace built by a helper
# (default: Qwen3-4B, Megatron backend, tensor parallel 2, context parallel 4). The tests assert
# on the plan's sglang target, kernel policy, parallel layout, argument groups (sglang, Megatron,
# FSDP, Miles), the rendered train-argument string and the environment variables. All CPU only.
from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    apply_true_on_policy_script_defaults,
    build_true_on_policy_launch_plan,
    get_megatron_model_type,
    get_true_on_policy_contract,
    get_true_on_policy_model_profile,
)


def _args(**overrides):
    # [FUNCTION DOC] HELPER: Build a fake parsed-arguments object. Starts from defaults (true
    # on-policy on, model Qwen3-4B, Megatron backend, tensor parallel 2, context parallel 4,
    # pipeline parallel 1, one rollout GPU per engine, no sglang target and no contract override,
    # sequence parallel on), updates them with the keyword overrides (new keys allowed) and returns
    # a simple attribute namespace.
    values = {
        "true_on_policy": True,
        "model_name": "Qwen3-4B",
        "train_backend": "megatron",
        "tensor_model_parallel_size": 2,
        "context_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "rollout_num_gpus_per_engine": 1,
        "sglang_rl_on_policy_target": None,
        "true_on_policy_contract": None,
        "use_sequence_parallel": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen3_dense_profile_resolves_model_names():
    # [FUNCTION DOC] Looks up the profile for Qwen3-4B and the contract by name. Asserts the profile
    # family is qwen3 dense, its contract is the very same object as the module's Qwen3 dense
    # contract constant and as the contract looked up by name, the contract schema carries the
    # contract name and model family, supported train layouts are data, tensor, pipeline and
    # Ulysses context parallel, rollout layouts are data and tensor, the required kernel contract
    # is the Qwen3 dense sglang math one, the logprob contract is sglang prefill, Ulysses context
    # parallel and tensor-parallel invariance are supported, and the Megatron model type for
    # Qwen3-4B and Qwen3-4B-Instruct-2507 maps to the lower-case-b names.
    profile = get_true_on_policy_model_profile("Qwen3-4B")
    contract = get_true_on_policy_contract("qwen3_dense_true_on_policy_v1")

    assert profile.family == "qwen3_dense"
    assert profile.contract is QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert profile.contract is contract
    assert contract.schema.name == "qwen3_dense_true_on_policy_v1"
    assert contract.schema.model_family == "qwen3_dense"
    assert profile.supported_train_layouts == ("dp", "tp", "pp", "ulysses_cp")
    assert profile.supported_rollout_layouts == ("dp", "tp")
    assert profile.required_kernel_contracts == ("qwen3_dense_sglang_math",)
    assert profile.logprob_contract == "sglang_prefill"
    assert profile.supports_ulysses_cp
    assert profile.supports_tp_invariant
    assert get_megatron_model_type("Qwen3-4B") == "qwen3-4B"
    assert get_megatron_model_type("Qwen3-4B-Instruct-2507") == "qwen3-4B-Instruct-2507"


def test_unknown_true_on_policy_model_fails_early():
    # [FUNCTION DOC] Asserts that looking up a profile for an unknown model name raises a value
    # error whose message says the model does not have a model profile.
    with pytest.raises(ValueError, match="does not have a model profile"):
        get_true_on_policy_model_profile("unknown-model")


@pytest.mark.parametrize(
    ("tp_size", "rollout_tp_size", "expected_target"),
    [
        (1, 1, "fsdp"),
        (2, 1, "fsdp_tp"),
        (1, 2, "fsdp_tp"),
    ],
)
def test_true_on_policy_target_is_derived_from_train_and_rollout_tp(
    tp_size: int,
    rollout_tp_size: int,
    expected_target: str,
):
    # [FUNCTION DOC] Parametrized over (train tp, rollout tp, expected target): (1,1) gives the
    # plain fsdp target, while (2,1) and (1,2) give the fsdp with tensor parallel target. Builds args
    # with context parallel 1, applies the script defaults and builds the plan. Asserts the args'
    # own sglang target stays unset, the plan's target equals the expected one, the plan contract is
    # the Qwen3 dense one, the sglang argument group is exactly deterministic inference, the true
    # on-policy contract name, and the fa3 attention backend, and the legacy target flag is absent
    # from the rendered train arguments.
    args = _args(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=rollout_tp_size,
    )

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.sglang_rl_on_policy_target is None
    assert plan.sglang_target == expected_target
    assert plan.contract is QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert plan.sglang_args.values == (
        "--sglang-enable-deterministic-inference",
        "--sglang-true-on-policy-contract",
        "qwen3_dense_true_on_policy_v1",
        "--sglang-attention-backend",
        "fa3",
    )
    assert "--sglang-rl-on-policy-target" not in plan.train_args


def test_legacy_sglang_target_override_does_not_change_contract_policy():
    # [FUNCTION DOC] Sets the legacy sglang target on the args to plain fsdp while tensor parallel
    # is 2 on training and 1 on rollout. After applying defaults and building the plan, asserts the
    # args keep their override, the plan's target is still derived as fsdp with tensor parallel
    # (override ignored), the kernel policy exists with tensor-parallel invariant row linear on,
    # the legacy flag is not rendered into train arguments, and no row-linear invariance
    # environment variable is exported.
    args = _args(
        tensor_model_parallel_size=2,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=1,
        sglang_rl_on_policy_target="fsdp",
    )

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.sglang_rl_on_policy_target == "fsdp"
    assert plan.sglang_target == "fsdp_tp"
    assert plan.kernel_policy is not None
    assert plan.kernel_policy.tp_invariant_row_linear
    assert "--sglang-rl-on-policy-target" not in plan.train_args
    assert "ROW_LINEAR_ENABLE_INV" not in plan.env_vars


def test_contract_object_owns_miles_kernel_policy_values():
    # [FUNCTION DOC] Builds a plan for Megatron with tensor parallel 2, context parallel 1 and one
    # rollout GPU per engine (no defaults applied). Asserts the kernel policy exists, references the
    # Qwen3 dense contract, uses the fa3 sglang attention backend, makes Megatron use the sglang
    # backend, disables rope fusion and bias-swiglu fusion, and turns on batch-invariant mode,
    # tensor-parallel invariant row linear and the deterministic tensor-parallel all-reduce.
    args = _args(
        train_backend="megatron",
        tensor_model_parallel_size=2,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=1,
    )

    plan = build_true_on_policy_launch_plan(args)

    assert plan.kernel_policy is not None
    assert plan.kernel_policy.contract is QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert plan.kernel_policy.sglang_attention_backend == "fa3"
    assert plan.kernel_policy.megatron_uses_sglang_backend
    assert plan.kernel_policy.disable_rope_fusion
    assert plan.kernel_policy.disable_bias_swiglu_fusion
    assert plan.kernel_policy.batch_invariant_mode
    assert plan.kernel_policy.tp_invariant_row_linear
    assert plan.kernel_policy.deterministic_tp_allreduce


def test_megatron_true_on_policy_disables_sequence_parallel_and_enables_backend_flags():
    # [FUNCTION DOC] With default Megatron args (sequence parallel on), applies defaults and builds
    # the plan. Asserts sequence parallel is switched off on the args; the rendered train arguments
    # do not contain the use-sglang flag but do contain the true on-policy contract flag and the
    # sglang contract flag with the Qwen3 dense name, the recompute-logprobs-via-prefill flag,
    # batch-invariant mode and no rope fusion; and neither the row-linear invariance nor the
    # deterministic Megatron all-reduce environment variable is set.
    args = _args(train_backend="megatron", use_sequence_parallel=True)

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is False
    assert "--use-sglang" not in plan.train_args
    assert "--true-on-policy-contract qwen3_dense_true_on_policy_v1" in plan.train_args
    assert "--sglang-true-on-policy-contract qwen3_dense_true_on_policy_v1" in plan.train_args
    assert "--recompute-logprobs-via-prefill" in plan.train_args
    assert "--batch-invariant-mode" in plan.train_args
    assert "--no-rope-fusion" in plan.train_args
    assert "ROW_LINEAR_ENABLE_INV" not in plan.env_vars
    assert "MEGATRON_USE_DETERMINISTIC_ALLREDUCE" not in plan.env_vars


def test_megatron_tp2_cp4_normal_topology_has_complete_true_on_policy_contract(monkeypatch):
    # [FUNCTION DOC] Removes the NCCL algorithm environment variable first so the plan's value is
    # not shadowed. Uses Megatron with tensor parallel 2, context parallel 4, pipeline parallel 1,
    # 8 rollout GPUs per engine and sequence parallel on; applies defaults and builds the plan.
    # Asserts: sequence parallel off and the sglang target left unset on the args; the parallel
    # layout records train tp 2, cp 4, rollout 8 GPUs per engine and flags train tp, Ulysses cp and
    # rollout tp as used; the kernel policy has tp-invariant row linear and deterministic tp
    # all-reduce; and the exact ordered argument groups: sglang (deterministic inference, contract
    # name, fa3 backend), Megatron (contract name, local transformer implementation, CPU
    # initialization, batch-invariant mode, no bias-swiglu fusion, no rope fusion), Miles
    # (deterministic mode, true on-policy mode, recompute logprobs via prefill); and the exact
    # environment variables: NCCL algorithm Ring, nondeterministic algorithms disallowed for the
    # transformer library, and the cuBLAS workspace config of :4096:8.
    monkeypatch.delenv("NCCL_ALGO", raising=False)

    args = _args(
        train_backend="megatron",
        tensor_model_parallel_size=2,
        context_parallel_size=4,
        pipeline_model_parallel_size=1,
        rollout_num_gpus_per_engine=8,
        use_sequence_parallel=True,
    )

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is False
    assert args.sglang_rl_on_policy_target is None
    assert plan.parallel_layout is not None
    assert plan.parallel_layout.train_tensor_parallel_size == 2
    assert plan.parallel_layout.train_context_parallel_size == 4
    assert plan.parallel_layout.rollout_num_gpus_per_engine == 8
    assert plan.parallel_layout.uses_train_tp
    assert plan.parallel_layout.uses_ulysses_cp
    assert plan.parallel_layout.uses_rollout_tp
    assert plan.kernel_policy is not None
    assert plan.kernel_policy.tp_invariant_row_linear
    assert plan.kernel_policy.deterministic_tp_allreduce
    assert plan.sglang_args.values == (
        "--sglang-enable-deterministic-inference",
        "--sglang-true-on-policy-contract",
        "qwen3_dense_true_on_policy_v1",
        "--sglang-attention-backend",
        "fa3",
    )
    assert plan.megatron_args.values == (
        "--true-on-policy-contract",
        "qwen3_dense_true_on_policy_v1",
        "--transformer-impl",
        "local",
        "--use-cpu-initialization",
        "--batch-invariant-mode",
        "--no-bias-swiglu-fusion",
        "--no-rope-fusion",
    )
    assert plan.miles_args.values == (
        "--deterministic-mode",
        "--true-on-policy-mode",
        "--recompute-logprobs-via-prefill",
    )
    assert plan.env_vars == {
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }


def test_true_on_policy_contract_override_is_validated():
    # [FUNCTION DOC] Sets the contract override on the args to an unknown name and asserts building
    # the plan raises a value error saying the true-on-policy contract is unsupported.
    args = _args(true_on_policy_contract="unknown_contract")

    with pytest.raises(ValueError, match="Unsupported true-on-policy contract"):
        build_true_on_policy_launch_plan(args)


def test_fsdp_true_on_policy_uses_fsdp_attention_without_megatron_backend_flags():
    # [FUNCTION DOC] Uses the FSDP backend with parallelism sizes of 1. After defaults and plan
    # building, asserts sequence parallel stays on, the target is plain fsdp, the FSDP argument
    # group is the attention-implementation flag set to flash attention 3, the Megatron group is
    # empty, the rendered train arguments include the FSDP attention flag and not the use-sglang
    # flag, and the row-linear invariance variable is not exported.
    args = _args(
        train_backend="fsdp",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        rollout_num_gpus_per_engine=1,
    )

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is True
    assert plan.sglang_target == "fsdp"
    assert plan.fsdp_args.values == ("--attn-implementation", "flash_attention_3")
    assert plan.megatron_args.values == ()
    assert "--attn-implementation flash_attention_3" in plan.train_args
    assert "--use-sglang" not in plan.train_args
    assert "ROW_LINEAR_ENABLE_INV" not in plan.env_vars


def test_fsdp_e2e_uses_current_true_on_policy_contract(monkeypatch):
    # [FUNCTION DOC] Imports the FSDP true on-policy end-to-end test module lazily (so a failing
    # import only affects this test), replaces its default wandb arguments helper with one returning
    # an empty string, and replaces the base backend's train execution with a function capturing
    # its keyword arguments. Runs the e2e module's execute entry point (nothing real is launched)
    # and asserts the captured train arguments contain the sglang true on-policy contract flag with
    # the Qwen3 dense name, and contain neither recompute-logprobs-via-prefill nor the legacy
    # sglang target flag.
    from tests.e2e.fsdp import test_qwen3_4B_fsdp_true_on_policy as fsdp_e2e

    from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend

    captured = {}
    monkeypatch.setattr(fsdp_e2e.command_utils, "get_default_wandb_args", lambda *args, **kwargs: "")
    monkeypatch.setattr(BaseCommandBackend, "execute_train", lambda _self, **kwargs: captured.update(kwargs))

    fsdp_e2e.execute()

    train_args = captured["train_args"]
    assert "--sglang-true-on-policy-contract qwen3_dense_true_on_policy_v1" in train_args
    assert "--recompute-logprobs-via-prefill" not in train_args
    assert "--sglang-rl-on-policy-target" not in train_args


def test_off_policy_builds_empty_launch_plan_and_does_not_mutate_args():
    # [FUNCTION DOC] With the true on-policy switch off and sequence parallel on, applies defaults
    # and builds the plan. Asserts args are untouched (sequence parallel still on, sglang target
    # still unset), the plan is not enabled, its train arguments are the empty string, and its
    # environment variables are an empty mapping.
    args = _args(true_on_policy=False, use_sequence_parallel=True)

    apply_true_on_policy_script_defaults(args)
    plan = build_true_on_policy_launch_plan(args)

    assert args.use_sequence_parallel is True
    assert args.sglang_rl_on_policy_target is None
    assert not plan.enabled
    assert plan.train_args == ""
    assert plan.env_vars == {}
