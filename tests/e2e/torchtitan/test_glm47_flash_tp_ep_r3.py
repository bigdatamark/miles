# [MODULE DOC] Single-case end-to-end test of the torchtitan backend with GLM-4.7-Flash (a
# 30B-A3B mixture-of-experts model whose flavor is registered on the miles side, reusing the
# DeepSeek-V3 blocks) on 8 colocated GPUs: tensor parallel 2, expert parallel 4, rollout routing
# replay on, sequence length 16384, max response 8192, SGLang engines of 4 GPUs each (the model's
# 20 attention heads do not divide by 8), memory fraction 0.5, triton attention and MoE runner so
# Hopper and Blackwell behave alike (the flashinfer prefill kernel for Blackwell lacks the padded
# query head size). Covers the Hugging Face load through the adapter, parallelism, replay and the
# weight stream back into SGLang. Registered for CUDA 8-GPU H200 (est. 2400 s, Hopper only), labels
# torchtitan and replay. Run as a script: prepare then execute the case.
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=2400, suite="stage-c-8-gpu-h200", labels=["torchtitan", "replay"], hardware=["hopper"])

# The first model torchtitan itself does not ship: the flavor lives under
# miles/backends/torchtitan_utils/models/ and reuses torchtitan's DeepSeek-V3
# blocks (MLA attention, sigmoid router with the correction bias, shared
# expert), so this case covers the miles-side registry end to end -- HF load
# through the DeepSeek adapter, tensor and expert parallelism, routing replay, and the
# weight stream back into SGLang's own glm4_moe_lite implementation. The engines
# take four GPUs each because the model's 20 attention heads do not divide by eight. SGLang runs
# attention on triton so the case is the same on Hopper and Blackwell: flashinfer's
# SM100 prefill kernel has no head_dim 256 instantiation, which is what MLA's
# 192 + 64 query head pads to.
CASE = CaseConfig(
    model_repo="zai-org/GLM-4.7-Flash",
    titan_model_name="glm4_moe_lite",
    titan_model_flavor="30B-A3B",
    num_gpus=8,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=4,
    seq_len=16384,
    max_response_len=8192,
    use_r3=True,
    mem_fraction_static=0.5,
    extra_args="--sglang-attention-backend triton --sglang-moe-runner-backend triton ",
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
