# [MODULE DOC] Single-case end-to-end test of tensor plus pipeline parallelism with the torchtitan
# backend and Qwen3-8B on 4 colocated GPUs (tensor 2, pipeline 2), sequence length 4096, max
# response 2048, with a reference model and KL loss enabled. It checks the two conventions the
# backend must satisfy: tensor parallelism shards the vocabulary and the loss reduces the softmax
# over the tensor group, so logits must arrive as the local shard (gathering them would shift every
# log prob by minus the natural log of the tensor size); pipeline parallelism infers stage buffers
# once, so every micro-batch is padded to the sequence length with consecutive positions rather
# than zeros. The untied dense model is used because the output head cannot be tied to the
# embedding across stages, and it is not truncated because the weight check compares every layer.
# Registered for CUDA 4-GPU H200 (est. 900 s, Hopper only), label torchtitan. Run as a script:
# prepare then execute.
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=900, suite="stage-c-4-gpu-h200", labels=["torchtitan"], hardware=["hopper"])

# Tensor and pipeline parallelism together, which is where the two dialects the
# backend has to speak both bite. Tensor parallelism shards the vocabulary, and
# miles' loss reduces the softmax over the tp group, so the logits have to reach
# it as the local shard -- gathering them shifts every log prob by -ln(tp).
# Pipeline parallelism infers its stage buffers once, so every microbatch of the
# run is padded to --seq-length, and the pad carries consecutive positions
# rather than zeros (thousands of one-token documents break linear attention).
# The model is the untied dense one: torchtitan cannot tie lm_head to the
# embedding across stages, so the tied 0.6B is refused under PP. It is not
# truncated, because the engine loads the whole checkpoint and the weight check
# compares every layer.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-8B",
    titan_model_name="qwen3",
    titan_model_flavor="8B",
    num_gpus=4,
    tp_size=2,
    pp_size=2,
    seq_len=4096,
    max_response_len=2048,
    with_ref=True,
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
