# [MODULE DOC] Single-case end-to-end test of pushing weights out of a pipelined trainer with the
# torchtitan backend and Qwen3-8B, on 8 GPUs non-colocated: 4 training GPUs (pipeline parallel 2,
# so each stage holds only its layers and the export is rank-partial) and 4 rollout GPUs (one
# engine), broadcast transfer mode, sequence length 8192, max response 4096, 2 rollouts, memory
# fraction 0.6. It guards both halves of the export: completing the whole model on every rank is
# unaffordable, and skipping completion within a stage once corrupted expert weights. The untied
# dense model is used because torchtitan cannot tie the output head to the embedding across
# pipeline stages. Registered for CUDA 8-GPU H200 (est. 1200 s, Hopper only), labels torchtitan
# and weight-update. Run as a script: prepare then execute.
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1200, suite="stage-c-8-gpu-h200", labels=["torchtitan", "weight-update"], hardware=["hopper"]
)

# Pushing weights out of a pipelined trainer, which is where the stream stops
# being something every rank can complete on its own. Each stage holds only its
# own layers, so the export is rank-partial twice over -- expert parallelism
# splits a stage's experts as well -- and the transfer protocol chooses how much
# of that gets reassembled before it ships. Completing the whole model on every
# rank is what a pipelined 30B could not afford, and skipping the completion
# inside a stage is what silently corrupted expert weights, so this case exists
# to keep both halves honest. The model is the untied dense one: torchtitan
# cannot tie lm_head to the embedding across stages, so a tied checkpoint is
# refused under PP.
CASE = CaseConfig(
    model_repo="Qwen/Qwen3-8B",
    titan_model_name="qwen3",
    titan_model_flavor="8B",
    num_gpus=4,
    pp_size=2,
    seq_len=8192,
    max_response_len=4096,
    colocate=False,
    rollout_num_gpus=4,
    transfer_mode="broadcast",
    num_rollout=2,
    mem_fraction_static=0.6,
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
