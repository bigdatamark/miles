# [MODULE DOC] Entry script that serves the Tinker-compatible HTTP gateway on top of a miles
# multi-LoRA training and rollout deployment. When run as a script it parses the canonical miles
# arguments plus Tinker-specific ones (entry name serve, with a Tinker preprocessing step), forces
# dynamic global batch size and delayed data-parallel splitting (each command ships one work unit
# whose size is the batch size), and runs the serve coroutine under a disposer that cleans up
# registered resources. Needs GPUs, a Ray cluster, a frozen HF base checkpoint and rollout routers.
import asyncio
import logging
from contextlib import suppress

import uvicorn

from miles.backends.megatron_utils.megatron_config import compute_trainer_args
from miles.ray.placement_group import create_trainer_handles
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.specs.inference import compute_router_providers, create_inference_controller_handle
from miles.ray.specs.train import ACTOR_ROLE, compute_trainer_configs
from miles.ray.wiring import get_backend_capability
from miles.tinker.arguments import add_tinker_arguments, configure_tinker_args
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import GatewayConfig
from miles.tinker.runtime import MilesBackend
from miles.tinker.server.app import build_app
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.http_utils import init_http_client
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


async def serve(args, *, disposer: Disposer):
    # [FUNCTION DOC] Brings up the inference side, one trainer and the Tinker gateway, then runs
    # the gateway until either its dispatcher or the HTTP server ends. Inputs: parsed args and a
    # disposer that receives every created resource for cleanup. Raises assertion errors if multi
    # LoRA is not enabled (adapter count must be above zero), if the load path differs from the HF
    # checkpoint (trainers and engines must share the same frozen base), if no checkpoint root can
    # be derived (explicit Tinker checkpoint root, else a tinker subfolder of the save path), or
    # if the trainer token budget is not positive.
    # Steps: (1) load the HF text config; the per-datum token limit is the model's maximum
    # position embeddings, further capped when max tokens per GPU is set: the per-GPU limit is
    # rounded down to a multiple of tensor-parallel size times the data pad multiplier, made even
    # when context parallelism is above 1 (each rank holds two zigzag chunks), then multiplied by
    # the context-parallel size. (2) Initialize the orchestration script (worker manager, result
    # unused) and the HTTP client. (3) Determine the backend capability, resolve router addresses
    # from providers, create the inference controller, initialize it and register it for disposal.
    # (4) Compute trainer configs, pick the single actor-role config (unpacking fails unless
    # exactly one exists), create trainer handles, initialize the actor trainer with its computed
    # args and register it. (5) Build the gateway config: base model (Tinker base model argument or
    # the HF checkpoint), adapter slots, checkpoint root, vocabulary size, per-datum token limit,
    # LoRA alpha and max rank, and flags for whether attention, MLP and unembedding groups are
    # trained according to the LoRA group list. (6) Compute data-parallel size as actor world size
    # (nodes x GPUs per node) divided by tensor x pipeline x context parallel sizes, build the
    # service around a backend using the trainer, router URL (from router ip and port) and that
    # size, and an ASGI server on the configured host and port with info logging. (7) Run the
    # service loop and the server as concurrent tasks, wait for the first to finish and re-raise its
    # error if any; finally cancel both and await them, ignoring cancellation. A crash of the
    # dispatcher thus takes the HTTP server down instead of leaving a healthy-looking server whose
    # training futures never resolve.
    assert args.multi_lora, "serve_tinker requires --multi-lora-n-adapters > 0"
    assert args.load == args.hf_checkpoint, "Tinker trainers and engines must load the same frozen HF base"
    checkpoint_root = args.tinker_checkpoint_root or (args.save and f"{args.save}/tinker")
    assert checkpoint_root, "set --tinker-checkpoint-root (or --save to derive <save>/tinker)"
    hf_config = load_hf_config(args.hf_checkpoint).get_text_config()
    max_tokens_per_datum = hf_config.max_position_embeddings
    if args.max_tokens_per_gpu is not None:
        # The trainer pads each packed microbatch to this multiple.
        pad_size = args.tensor_model_parallel_size * args.data_pad_size_multiplier
        local_token_limit = args.max_tokens_per_gpu // pad_size * pad_size
        if args.context_parallel_size > 1:
            # each rank holds two equal zigzag chunks per datum
            local_token_limit -= local_token_limit % 2
        trainer_token_limit = local_token_limit * args.context_parallel_size
        max_tokens_per_datum = min(max_tokens_per_datum, trainer_token_limit)
    assert max_tokens_per_datum > 0, "trainer token budget must fit at least one padding block"
    _worker_manager = init_orchestration_script(args, disposer=disposer)
    init_http_client(args)

    capability = get_backend_capability(args)
    await resolve_router_addrs(args, router_providers=compute_router_providers(args, capability=capability))
    inference_controller = create_inference_controller_handle(capability=capability)
    await inference_controller.init()
    disposer.add(inference_controller)

    trainer_configs = compute_trainer_configs(args)
    [actor_config] = [config for config in trainer_configs if config.role == ACTOR_ROLE]
    trainer = create_trainer_handles(args, trainer_configs=trainer_configs)[actor_config.trainer_id]
    await trainer.init(compute_trainer_args(args, actor_config))
    disposer.add(trainer)

    config = GatewayConfig(
        base_model=args.tinker_base_model or args.hf_checkpoint,
        n_slots=args.multi_lora_n_adapters,
        checkpoint_root=checkpoint_root,
        vocab_size=hf_config.vocab_size,
        max_tokens_per_datum=max_tokens_per_datum,
        lora_alpha=args.lora_alpha,
        max_lora_rank=args.lora_rank,
        trains_attn="attn" in args.tinker_lora_groups,
        trains_mlp="mlp" in args.tinker_lora_groups,
        trains_unembed="unembed" in args.tinker_lora_groups,
    )
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    dp_size = actor_world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    service = TinkerService(MilesBackend(trainer, router_url, dp_size=dp_size), config)

    server = uvicorn.Server(
        uvicorn.Config(
            build_app(service), host=args.tinker_server_host, port=args.tinker_server_port, log_level="info"
        )
    )
    logger.info(f"tinker gateway serving {config.base_model} on :{args.tinker_server_port}")
    # supervise both: a crashed dispatcher must take the HTTP server down with it,
    # not keep answering /healthz while every training future pends forever
    service_task = asyncio.create_task(service.run())
    server_task = asyncio.create_task(server.serve())
    try:
        done, _ = await asyncio.wait({service_task, server_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in (service_task, server_task):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


if __name__ == "__main__":
    args = parse_args(add_tinker_arguments, entry="serve", preprocess_args=configure_tinker_args)
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
    asyncio.run(with_disposer(serve, args))
