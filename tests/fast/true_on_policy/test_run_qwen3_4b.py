# [MODULE DOC] Tests that the Qwen3-4B launch script (scripts/run_qwen3_4b) expands its true
# on-policy switch into the full set of training arguments and environment variables. Each test
# patches the command backend's train execution to capture the keyword arguments (train argument
# string and extra environment variables) instead of launching anything, and patches the default
# wandb-arguments helper to return an empty string. CPU only; no processes are started.
from __future__ import annotations

from scripts import run_qwen3_4b

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend


def test_qwen3_script_true_on_policy_single_knob_expands_to_megatron_contract(monkeypatch):
    # [FUNCTION DOC] Builds script arguments for H100, Qwen3-4B, true on-policy on, eval and KL loss
    # off, and first asserts the legacy sglang target is unset and sequence parallel is already off
    # on the constructed arguments. Runs the script's train execution and asserts the captured train
    # arguments contain true on-policy mode, deterministic sglang inference, the sglang and
    # Megatron contract flags with the Qwen3 dense name, the fa3 backend, recompute logprobs via
    # prefill, the load path of the Qwen3-4B torch-dist checkpoint, the save path under the shared
    # data directory for the run id, batch-invariant mode and no rope fusion; and do not contain
    # the legacy target flag, use-sglang, or sequence parallel. The row-linear invariance and
    # deterministic Megatron all-reduce variables must be absent from the extra environment.
    captured = {}

    def fake_execute_train(_self, **kwargs):
        # [FUNCTION DOC] HELPER: Stand-in for the backend train execution; stores its keyword
        # arguments into the enclosing test's capture mapping and returns nothing.
        captured.update(kwargs)

    monkeypatch.setattr(BaseCommandBackend, "execute_train", fake_execute_train)
    monkeypatch.setattr(command_utils, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_qwen3_4b.ScriptArgs(
        hardware="H100",
        run_id="unit-test",
        model_name="Qwen3-4B",
        true_on_policy=True,
        enable_eval=False,
        use_kl_loss=False,
    )

    assert args.sglang_rl_on_policy_target is None
    assert args.use_sequence_parallel is False

    run_qwen3_4b._execute_train(args)

    train_args = captured["train_args"]
    env_vars = captured["extra_env_vars"]

    assert "--true-on-policy-mode" in train_args
    assert "--sglang-enable-deterministic-inference" in train_args
    assert "--sglang-true-on-policy-contract qwen3_dense_true_on_policy_v1" in train_args
    assert "--sglang-rl-on-policy-target" not in train_args
    assert "--sglang-attention-backend fa3" in train_args
    assert "--true-on-policy-contract qwen3_dense_true_on_policy_v1" in train_args
    assert "--recompute-logprobs-via-prefill" in train_args
    assert "--load /root/models/Qwen3-4B_torch_dist" in train_args
    assert "--save /root/shared_data/unit-test/checkpoints" in train_args
    assert "--use-sglang" not in train_args
    assert "--batch-invariant-mode" in train_args
    assert "--no-rope-fusion" in train_args
    assert "--sequence-parallel" not in train_args
    assert "ROW_LINEAR_ENABLE_INV" not in env_vars
    assert "MEGATRON_USE_DETERMINISTIC_ALLREDUCE" not in env_vars


def test_qwen3_script_true_on_policy_tp2_cp4_normal_topology_contract(monkeypatch):
    # [FUNCTION DOC] Like the single-knob test but in normal mode with the script arguments then
    # overridden to tensor parallel 2, context parallel 4, pipeline parallel 1, context-parallel
    # communication type a2a and 8 rollout GPUs per engine (set after construction). Asserts the
    # sglang target is unset and sequence parallel off before execution. After running the train
    # execution, asserts the arguments carry those sizes, the a2a type, the load and save paths for
    # the run id, the sglang contract flag, fa3, recompute via prefill, batch-invariant mode, no
    # bias-swiglu fusion and no rope fusion, and lack the legacy target, use-sglang and sequence
    # parallel flags. Asserts the environment has the NCCL algorithm Ring, nondeterministic
    # algorithms disabled for the transformer library, cuBLAS workspace config :4096:8, and neither
    # the row-linear invariance nor the deterministic Megatron all-reduce variable.
    captured = {}

    def fake_execute_train(_self, **kwargs):
        # [FUNCTION DOC] HELPER: Stand-in for the backend train execution; stores its keyword
        # arguments into the enclosing test's capture mapping and returns nothing.
        captured.update(kwargs)

    monkeypatch.setattr(BaseCommandBackend, "execute_train", fake_execute_train)
    monkeypatch.setattr(command_utils, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_qwen3_4b.ScriptArgs(
        hardware="H100",
        run_id="unit-test-tp2-cp4",
        model_name="Qwen3-4B",
        mode="normal",
        true_on_policy=True,
        enable_eval=False,
        use_kl_loss=False,
    )
    args.tensor_model_parallel_size = 2
    args.context_parallel_size = 4
    args.pipeline_model_parallel_size = 1
    args.cp_comm_type = "a2a"
    args.rollout_num_gpus_per_engine = 8

    assert args.sglang_rl_on_policy_target is None
    assert args.use_sequence_parallel is False

    run_qwen3_4b._execute_train(args)

    train_args = captured["train_args"]
    env_vars = captured["extra_env_vars"]

    assert "--tensor-model-parallel-size 2" in train_args
    assert "--context-parallel-size 4" in train_args
    assert "--cp-comm-type a2a" in train_args
    assert "--rollout-num-gpus-per-engine 8" in train_args
    assert "--load /root/models/Qwen3-4B_torch_dist" in train_args
    assert "--save /root/shared_data/unit-test-tp2-cp4/checkpoints" in train_args
    assert "--sglang-true-on-policy-contract qwen3_dense_true_on_policy_v1" in train_args
    assert "--sglang-rl-on-policy-target" not in train_args
    assert "--sglang-attention-backend fa3" in train_args
    assert "--recompute-logprobs-via-prefill" in train_args
    assert "--use-sglang" not in train_args
    assert "--batch-invariant-mode" in train_args
    assert "--no-bias-swiglu-fusion" in train_args
    assert "--no-rope-fusion" in train_args
    assert "--sequence-parallel" not in train_args
    assert env_vars["NCCL_ALGO"] == "Ring"
    assert env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert env_vars["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "ROW_LINEAR_ENABLE_INV" not in env_vars
    assert "MEGATRON_USE_DETERMINISTIC_ALLREDUCE" not in env_vars


def test_qwen3_script_off_policy_does_not_emit_true_on_policy_contract(monkeypatch):
    # [FUNCTION DOC] Builds script arguments with true on-policy off and runs the train execution.
    # Asserts the captured train arguments contain none of: true on-policy mode, the sglang
    # contract flag, the legacy target flag, the Megatron contract flag, recompute-logprobs-via-
    # prefill, use-sglang; and the extra environment lacks the row-linear invariance variable.
    captured = {}

    def fake_execute_train(_self, **kwargs):
        # [FUNCTION DOC] HELPER: Stand-in for the backend train execution; stores its keyword
        # arguments into the enclosing test's capture mapping and returns nothing.
        captured.update(kwargs)

    monkeypatch.setattr(BaseCommandBackend, "execute_train", fake_execute_train)
    monkeypatch.setattr(command_utils, "get_default_wandb_args", lambda *args, **kwargs: "")

    args = run_qwen3_4b.ScriptArgs(
        hardware="H100",
        run_id="unit-test",
        model_name="Qwen3-4B",
        true_on_policy=False,
        enable_eval=False,
        use_kl_loss=False,
    )

    run_qwen3_4b._execute_train(args)

    train_args = captured["train_args"]
    env_vars = captured["extra_env_vars"]

    assert "--true-on-policy-mode" not in train_args
    assert "--sglang-true-on-policy-contract" not in train_args
    assert "--sglang-rl-on-policy-target" not in train_args
    assert "--true-on-policy-contract" not in train_args
    assert "--recompute-logprobs-via-prefill" not in train_args
    assert "--use-sglang" not in train_args
    assert "ROW_LINEAR_ENABLE_INV" not in env_vars
