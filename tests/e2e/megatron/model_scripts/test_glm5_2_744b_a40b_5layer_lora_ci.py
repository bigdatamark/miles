# [MODULE DOC] CI smoke test of the GLM-5.2 LoRA launcher script on a 5-layer slice (sparse
# attention cross-layer path), on one node of 4 GPUs (CUDA: 4-GPU H200 stage, estimated 1200
# seconds, labels megatron, model-scripts and lora, hopper and blackwell). The registration is
# currently disabled because the pinned Megatron-Bridge lacks a property the sparse-attention
# variant reads, pending a bridge bump. Run as a script: strips all proxy environment variables
# (both cases), prepares once, then runs each combination in sequence: "megatron attention
# backend + shared outer expert LoRAs + virtual experts" and "tilelang + per-expert + no virtual
# experts" (the GLM-5.1 sibling runs the complementary pair). Before each combination ray and
# SGLang processes are killed and the script sleeps 10 seconds; the run is 2 rollouts with
# wandb off. Every combination must complete. Common extra flags enable CI test mode, disable the
# logprobs checker and skip two engine-side stacked parameter groups in the weight-update check.

import os

from scripts.run_glm5_2_744b_a40b_lora import ScriptArgs, _prepare_download, _train
from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils


# Smoke test for scripts/run_glm5_2_744b_a40b_lora.py on the 5-layer toy (DSA cross-layer
# path, full rollout -> train -> save loop). Runs one diagonal of the MoE-expert LoRA
# matrix — {megatron + shared-outer + virtual-experts, tilelang + per-expert +
# no-virtual-experts} — and every combination must pass. The GLM-5.1 6-layer sibling runs
# the complementary diagonal, so nightly still covers all four {dsa backend} x {lora layout}
# cells across the pair. Functionality, not accuracy; 4 GPUs (TP=EP=4).


# TODO: need to bump megatron-bridge
register_cuda_ci(
    est_time=1200,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "model-scripts", "lora"],
    hardware=["hopper", "blackwell"],
    disabled="Megatron's dsa variant now builds AbsorbedMLASelfAttention, which reads "
    "linear_kv_up_proj.weight directly; the pinned Megatron-Bridge's LoRALinear has no such "
    "property. Upstream Bridge added an effective-weight property — re-enable after bumping it.",
)

# skip the engine-side stacked params a frozen-base LoRA run cannot re-ship
# (they keep their correct checkpoint values; everything else is verified)
_BASE_EXTRA = (
    "--ci-test " "--ci-disable-logprobs-checker " "--check-weight-update-skip-list fused_qkv_a_proj_with_mqa indexer. "
)

# (name, dsa_attention_backend, experts_shared_outer_loras, virtual_experts_serving)
_CONFIGS = [
    ("megatron + shared-outer + virtual-experts", "megatron", True, True),
    ("tilelang + per-expert + no-virtual-experts", "tilelang", False, False),
]


def _args(dsa: str, shared_outer: bool, virtual_experts: bool) -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds launcher arguments from the environment for one combination:
    # model GLM-5.2_5layer, 1 node of 4 GPUs, 2 rollouts, wandb off, the given sparse attention
    # backend, the given shared-outer LoRA flag, and extra flags equal to the common set plus, if
    # virtual experts are off, a flag turning off virtual-expert LoRA serving in SGLang.
    return ScriptArgs.from_env(
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=2,
        enable_wandb=False,
        dsa_attention_backend=dsa,
        experts_shared_outer_loras=shared_outer,
        extra_args=_BASE_EXTRA + ("" if virtual_experts else "--no-sglang-lora-use-virtual-experts "),
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates the output directory through the arguments' backend and runs the
    # launcher's download step.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.output_dir}")
    _prepare_download(args)


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Launches training through the launcher's train step.
    _train(args)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(_args(*_CONFIGS[0][1:]))
    for name, dsa, shared_outer, virtual_experts in _CONFIGS:
        print(f"[glm5.2-lora-ci] ===== combo: {name} =====", flush=True)
        # fresh ray/sglang between combos
        command_utils.default_config().create_backend().exec_command_cpu(
            "ray stop --force || true; pkill -9 sglang || true; sleep 10"
        )
        execute(_args(dsa, shared_outer, virtual_experts))
        print(f"[glm5.2-lora-ci] ===== combo PASSED: {name} =====", flush=True)
