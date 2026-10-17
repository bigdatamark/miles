# [MODULE DOC] Builds the torchtitan trainer configuration from miles' parsed arguments: resolves
# the model specification from a miles-provided or torchtitan-provided model package, maps
# parallelism, batch, optimizer, learning-rate, loss, data loader, checkpoint and logging settings,
# and decides whether to resume from the latest saved step or to load the initial Hugging Face
# weights. Side effects: may create a temporary directory for dumps, reads the Hugging Face config
# and the checkpoint directory listing.
import importlib
import logging
import os
import tempfile
from argparse import Namespace

from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.trainer import Trainer

from miles.backends.torchtitan_utils.components import EmptyDataLoader, TiedCheckpointManager
from miles.backends.torchtitan_utils.loss import RLLossAdapter
from miles.backends.torchtitan_utils.parallel import parallel_dims_from_config
from miles.utils.hf_utils.config import load_hf_config

logger = logging.getLogger(__name__)


_MODEL_PACKAGE_ROOTS = ("miles.backends.torchtitan_utils.models", "torchtitan.models")


def _model_package(name: str):
    # [FUNCTION DOC] HELPER: Imports and returns the model package with the given name, looking
    # under the miles models root first and then the torchtitan models root. A module-not-found
    # error is swallowed only when it is for the exact candidate module; a missing nested dependency
    # is re-raised. If neither exists, raises a value error listing both candidate paths.
    for root in _MODEL_PACKAGE_ROOTS:
        module_name = f"{root}.{name}"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name != module_name:
                raise
    raise ValueError(
        f"--titan-model-name {name!r} is neither a miles model package nor a torchtitan one "
        f"(looked for {', '.join(f'{root}.{name}' for root in _MODEL_PACKAGE_ROOTS)})"
    )


def resolve_model_spec(args: Namespace):
    # [FUNCTION DOC] Finds the model package named in the arguments and calls its model registry
    # with the model flavor and the flex attention backend, returning the model specification.
    # Raises a value error if the package exposes no registry function.
    module = _model_package(args.titan_model_name)
    registry = getattr(module, "model_registry", None)
    if registry is None:
        raise ValueError(f"{module.__name__} exposes no model_registry(); cannot build a ModelSpec")
    return registry(args.titan_model_flavor, attn_backend="flex")


def _latest_step_dir(load_root: str, dump_subdir: str) -> str | None:
    # [FUNCTION DOC] HELPER: Looks in the checkpoint folder under the load root, the torchtitan
    # directory and the dump subdirectory. Returns none if the folder does not exist or holds no
    # entries starting with the step prefix; otherwise returns the path of the entry with the largest
    # step number.
    # KNOWN ISSUE: an entry whose name starts with the prefix but has a non-integer suffix raises a
    # value error.
    folder = os.path.join(load_root, "torchtitan", dump_subdir, "checkpoint")
    if not os.path.isdir(folder):
        return None
    steps = [int(name.removeprefix("step-")) for name in os.listdir(folder) if name.startswith("step-")]
    return os.path.join(folder, f"step-{max(steps)}") if steps else None


def build_trainer_config(args: Namespace, *, hf_assets_path: str, lr_total_steps: int, dump_subdir: str):
    # [FUNCTION DOC] Returns a populated torchtitan trainer config. Keyword inputs: the Hugging Face
    # assets path, total learning-rate schedule steps, and the dump subdirectory name.
    # Validation (value errors): the optimizer must be adam; a checkpoint with tied embeddings is
    # rejected under pipeline parallelism greater than one (when tied otherwise, enables weight
    # tying in the model spec).
    # Settings: model spec from the arguments; assets path; dump folder = save path (or a new
    # temporary directory) / torchtitan / subdirectory; parallel degrees (data replicate, tensor,
    # pipeline, context, expert) from arguments, pipeline microbatch size 1, partial-dtensor SPMD
    # backend; sequence length; under pipeline parallelism the global batch size must be a multiple
    # of data-parallel size times micro batch size (value error); local batch size =
    # global batch size / dp size / micro batch size, at least one; global batch size recomputed as
    # local batch size times dp size; steps = max(lr total steps, 1); gradient clip norm; CUDA
    # graphs disabled; one AdamW parameter group matching all parameters with learning rate, betas,
    # epsilon and weight decay from arguments; warmup steps from warmup iterations; for the constant
    # decay style the minimum learning-rate factor is 1.0, otherwise the decay type is set and the
    # factor is min lr divided by lr; RL loss adapter, empty data loader, tied-aware checkpoint
    # manager; full activation checkpointing only if gradient checkpointing is on; seed;
    # checkpointing enabled with full (not model-only) last save; resume: if a load path is given
    # and a latest step directory exists there, loads that path with optimizer state; otherwise
    # loads only model weights from the Hugging Face format; TensorBoard, W&B logging and the
    # validator are turned off.
    # KNOWN ISSUE: the minimum learning-rate factor divides by lr, which fails if lr is zero.
    if args.optimizer != "adam":
        raise ValueError(f"torchtitan backend supports --optimizer adam, got {args.optimizer!r}")

    ties_embeddings = load_hf_config(hf_assets_path).tie_word_embeddings
    if ties_embeddings and args.pipeline_model_parallel_size > 1:
        raise ValueError(
            "the checkpoint ties lm_head to the embedding, which torchtitan cannot do across pipeline "
            "stages: it would train a separate lm_head that the HF export then has no tensor to ship "
            "into. Use --pipeline-model-parallel-size 1 or an untied checkpoint."
        )

    config = Trainer.Config()
    config.model_spec = resolve_model_spec(args)
    if ties_embeddings:
        config.model_spec.model.enable_weight_tying = True
        logger.info("Checkpoint ties lm_head to the embedding; excluding lm_head.weight from the HF export")

    config.hf_assets_path = hf_assets_path
    config.dump_folder = os.path.join(
        args.save or tempfile.mkdtemp(prefix="miles-torchtitan-"), "torchtitan", dump_subdir
    )

    config.parallelism.data_parallel_replicate_degree = args.dp_replicate_size
    config.parallelism.tensor_parallel_degree = args.tensor_model_parallel_size
    config.parallelism.pipeline_parallel_degree = args.pipeline_model_parallel_size
    config.parallelism.context_parallel_degree = args.context_parallel_size
    config.parallelism.expert_parallel_degree = args.expert_model_parallel_size
    config.parallelism.pipeline_parallel_microbatch_size = 1
    config.parallelism.spmd_backend = "partial_dtensor"
    parallel_dims = parallel_dims_from_config(config.parallelism)
    dp_size = parallel_dims.dp_replicate * parallel_dims.dp_shard

    config.training.seq_len = args.seq_length
    if parallel_dims.pp_enabled and args.global_batch_size % (dp_size * args.micro_batch_size):
        raise ValueError(
            f"--global-batch-size {args.global_batch_size} must be a multiple of dp * micro_batch_size "
            f"({dp_size} * {args.micro_batch_size}) under pipeline parallelism"
        )
    config.training.local_batch_size = max(args.global_batch_size // dp_size // args.micro_batch_size, 1)
    config.training.global_batch_size = config.training.local_batch_size * dp_size
    config.training.steps = max(lr_total_steps, 1)
    config.training.max_norm = args.clip_grad
    config.training.disable_cuda_graphs = True

    config.optimizer.param_groups = [
        ParamGroupConfig(
            pattern=r".*",
            optimizer_name="AdamW",
            optimizer_kwargs={
                "lr": args.lr,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
            },
        )
    ]

    config.lr_scheduler.warmup_steps = args.lr_warmup_iters
    if args.lr_decay_style == "constant":
        config.lr_scheduler.min_lr_factor = 1.0
    else:
        config.lr_scheduler.decay_type = args.lr_decay_style
        config.lr_scheduler.min_lr_factor = args.min_lr / args.lr

    config.loss = RLLossAdapter.Config()
    config.dataloader = EmptyDataLoader.Config()
    config.checkpoint = TiedCheckpointManager.Config()
    config.activation_checkpoint = FullAC.Config() if args.gradient_checkpointing else None
    config.debug.seed = args.seed

    config.checkpoint.enable = True
    config.checkpoint.last_save_model_only = False
    resume_from = _latest_step_dir(args.load, dump_subdir) if args.load else None
    if resume_from is None:
        config.checkpoint.initial_load_model_only = True
        config.checkpoint.initial_load_in_hf = True
    else:
        config.checkpoint.initial_load_path = resume_from
        config.checkpoint.initial_load_model_only = False
        config.checkpoint.initial_load_in_hf = False

    config.metrics.enable_tensorboard = False
    config.metrics.enable_wandb = False
    config.validator.enable = False
    return config
