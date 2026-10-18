# [MODULE DOC]
# HELPER: Derived quantities of megatron parallelism arguments that do not need megatron itself.
"""
Utils for megatron arguments, but not related to megatron core logic
"""


def compute_megatron_world_size_except_dp(args) -> int:
    # [FUNCTION DOC] HELPER: Number of ranks in one model replica: tensor parallel size times
    # pipeline parallel size times context parallel size (expert parallelism is not included).
    return args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
