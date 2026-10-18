# [MODULE DOC] Single-case end-to-end test of rollout routing replay (R3) with DeepEP and FP8
# rollout for Qwen3-30B-A3B on 4 GPUs, colocated: Megatron context 2, pipeline 1, tensor 2, expert
# 4; one 4-GPU SGLang engine with expert parallel 4. Registered for CUDA 4-GPU H200 (est. 1800 s)
# but disabled; no CI metric gates are registered. Run as a script: strips proxy variables, then
# prepare and execute.
# KNOWN ISSUE: the registration is disabled because of a mismatch between FP8 rollout and BF16
# training, and a fixme comment above it marks it as unfixed; the disabled reason cites no
# issue or PR. Unlike sibling cases it registers no metric gates, and unlike the BF16 control it
# does not lower the max running requests, which would put it at the DeepEP low-latency cap.
import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

# FIXME: fix here
register_cuda_ci(
    est_time=1800,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "replay"],
    hardware=["hopper", "blackwell"],
    disabled="Failed due to mismatch between fp8 rollout and bf16 training.",
)

CASE = CaseConfig(
    use_deepep=True,
    use_fp8_rollout=True,
    use_int4_rollout=False,
    use_bridge=False,
    use_r3=True,
    num_gpus_per_node=4,
    cp_size=2,
    pp_size=1,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=4,
    sglang_ep_size=4,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
