# [MODULE DOC]
# Process-level initialization of the Megatron backend: sets Megatron's global args, initializes
# model parallelism, builds the independent-DP group and the global parallel state, seeds all
# random number generators (python, numpy, torch, Megatron's tracked CUDA states), builds the
# tokenizer, sets up the (unused but required) microbatch calculator, configures deterministic
# mode, tensor-parallel communication overlap and an optional user init hook. Also provides a
# capture/restore object for random generator states and a rank predicate for logging.
import dataclasses
import logging
import os
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.config import set_experimental_flag
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.tensor_parallel.random import _get_all_rng_states, _set_all_rng_states
from megatron.training.global_vars import _build_tokenizer, set_args

from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.hf_utils.config import register_hf_config_aliases

from .ft.indep_dp import create_indep_dp_group
from .parallel import create_megatron_parallel_state

logger = logging.getLogger(__name__)


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    # [FUNCTION DOC] HELPER: Seeds python, numpy, torch and Megatron's model-parallel CUDA RNG
    # tracker. Effective seed = base seed + 100 times the pipeline-parallel rank (so stages differ),
    # plus, when data-parallel random init is on, 10 times the effective data-parallel rank. The
    # last three flags are forwarded to the Megatron seeding call (TE tracker, inference tracker,
    # CUDA-graph-capable RNG). Global side effects on all RNG states.
    """Set random seed for reproducability."""
    # Ensure that different pipeline MP stages get different seeds.
    seed = seed_ + (100 * get_parallel_state().pp.rank)
    # Ensure different data parallel ranks get different seeds
    if data_parallel_random_init:
        seed = seed + (10 * get_parallel_state().effective_dp.rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)


def set_random_seed_from_args(args) -> None:
    # [FUNCTION DOC] Logs the seed on rank zero and seeds all generators from args.seed using the
    # data-parallel-random-init, TE-RNG-tracker and inference-RNG-tracker flags. The CUDA-graph RNG
    # option is left at its default (off).
    if args.rank == 0:
        logger.info(f"> setting random seeds to {args.seed} ...")
    _set_random_seed(
        args.seed,
        args.data_parallel_random_init,
        args.te_rng_tracker,
        args.inference_rng_tracker,
    )


@dataclasses.dataclass(frozen=True)
class RandomState:
    # [CLASS DOC]
    # Immutable snapshot of random generator states: the python generator state, the numpy global
    # generator state, and Megatron's tuple of tracked CUDA RNG states. (The torch CPU generator is
    # not captured.)
    python: tuple[Any, ...]
    numpy: dict[str, Any] | tuple[Any, ...]
    megatron: tuple[Any, ...]

    @classmethod
    def capture(cls) -> "RandomState":
        # [FUNCTION DOC] Reads the current python, numpy and Megatron CUDA RNG states and returns
        # them as a new snapshot object. Does not modify any generator.
        return cls(python=random.getstate(), numpy=np.random.get_state(), megatron=_get_all_rng_states())

    def restore(self) -> None:
        # [FUNCTION DOC] Sets the python, numpy and Megatron CUDA generators back to the captured
        # states (global side effects).
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        _set_all_rng_states(*self.megatron)


def _initialize_distributed(args, get_embedding_ranks=None, get_position_embedding_ranks=None):
    # [FUNCTION DOC] HELPER: Calls Megatron's model-parallel initialization with the tensor,
    # pipeline, virtual pipeline, context (flat and hierarchical), expert and expert-tensor
    # parallel sizes, the distributed optimizer instance count, timeout minutes, NCCL communicator
    # config path and pipeline communication backend from args. The rank ordering is
    # tensor-context-expert-data-pipeline unless the TP-PP-DP mapping flag is set, in which case
    # pipeline precedes data. Optional embedding-rank callbacks are forwarded; gloo groups are
    # created per the use-gloo flag. Assumes torch distributed is already initialized by the caller.
    """Initialize torch.distributed and core model parallel."""
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
        context_parallel_size=args.context_parallel_size,
        hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
        expert_model_parallel_size=args.expert_model_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        order="tp-cp-ep-dp-pp" if not args.use_tp_pp_dp_mapping else "tp-cp-ep-pp-dp",
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        create_gloo_process_groups=args.use_gloo_process_groups,
    )


def init(
    args,
    indep_dp_store_addr: str | None = None,
    indep_dp_info: IndepDPInfo | None = None,
):
    # [FUNCTION DOC]
    # Main initialization entry. Steps in order: default the independent-DP info to the trivial
    # (single-cell) one; set Megatron's global args; enable Megatron experimental features if
    # requested; initialize model parallelism; create the independent-DP group from the store
    # address and info with this process's global rank and world size; build and install the global
    # parallel state; if independent DP is on assert the data-parallel size is 1; seed RNGs;
    # register HF config aliases; build the tokenizer; initialize the Megatron microbatch
    # calculator (needed only to satisfy Megatron validation); in deterministic mode enable
    # deterministic cuDNN and torch algorithms (strict, errors not warnings), disable cuDNN
    # benchmark, and set torch's fill-uninitialized-memory to follow the MILES_FILL_UNINIT
    # environment variable being "1" (default off); with the deterministic-collective debug flag
    # assert overlapped grad reduce is off; if TP comm overlap is on, initialize Megatron's
    # tensor-parallel communicators; finally, if a custom init path is set, load that function and
    # call it with args. Returns nothing; extensive global side effects.
    if indep_dp_info is None:
        indep_dp_info = IndepDPInfo.create_trivial()

    set_args(args)
    if args.enable_experimental:
        logger.info("Enable megatron experimental")
        set_experimental_flag(True)

    # Pytorch distributed.
    _initialize_distributed(args)

    indep_dp = create_indep_dp_group(
        store_addr=indep_dp_store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=dist.get_rank(),
        megatron_world_size=dist.get_world_size(),
    )

    set_parallel_state(create_megatron_parallel_state(indep_dp=indep_dp))

    # sanity check
    if getattr(args, "indep_dp", False):
        assert args.data_parallel_size == 1

    # Random seeds for reproducibility.
    set_random_seed_from_args(args)
    register_hf_config_aliases()
    _build_tokenizer(args)
    # We won't use this. initialize to pass some validation in megatron.
    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
        args.decrease_batch_size_if_needed,
    )

    if args.deterministic_mode:
        if args.rank == 0:
            logger.info("> running in deterministic mode")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        # deterministic mode otherwise fills every torch.empty, which only catches reads of
        # uninitialized memory and costs ~5% of forward kernel time
        torch.utils.deterministic.fill_uninitialized_memory = os.environ.get("MILES_FILL_UNINIT", "0") == "1"

    if args.debug_deterministic_collective:
        assert not args.overlap_grad_reduce, "deterministic collectives require synchronous grad sync"

    if args.tp_comm_overlap:
        from megatron.training.initialize import _initialize_tp_communicators

        _initialize_tp_communicators()

    if getattr(args, "custom_megatron_init_path", None):
        from miles.utils.function_registry import load_function

        custom_init = load_function(args.custom_megatron_init_path)
        custom_init(args)


# TODO shall we use a simpler method to determine which rank to init wandb?
def is_first_replica_megatron_main_rank():
    # [FUNCTION DOC] True on exactly one rank per model replica group: the rank with zero effective
    # data-and-context-parallel rank, zero tensor-parallel rank and the last pipeline stage. Used to
    # decide which rank logs/tracks. Reads the global parallel state.
    return (
        get_parallel_state().effective_dp_cp.rank == 0
        and get_parallel_state().tp.rank == 0
        and get_parallel_state().pp.rank == get_parallel_state().pp.size - 1
    )
