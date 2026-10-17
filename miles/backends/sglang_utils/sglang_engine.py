# [MODULE DOC] Builds the shell command line that launches one SGLang inference engine process (one
# node of an engine) and computes the server-argument dictionary behind it. Argument precedence,
# lowest to highest: SGLang defaults, then the user's "--sglang-" options (applied last, and only
# for names not already set), then values miles derives from its own arguments (model path,
# parallel sizes, memory saver, routing replay, LoRA, disaggregation mode, and so on), then the
# per-group overrides which win over everything. Keys unknown to the installed SGLang are dropped
# with a log message. Also contains small URL helpers. Reads one environment variable that switches
# weight loading to dummy.
import functools
import ipaddress
import logging
import os
import shlex

from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.server_args_utils import server_args_to_argv
from miles.utils.lora.utils import (
    LORA_ADAPTER_NAME,
    engine_loads_adapter_from_disk,
    is_multi_lora_enabled,
    lora_base_cpu_backup_enabled,
    lora_rollout_enabled,
)
from miles.utils.workers.argv_utils import _record_field_names

logger = logging.getLogger(__name__)


def format_v6_uri(addr: str | None) -> str | None:
    # [FUNCTION DOC] Wraps an IPv6 literal address in square brackets for use inside a URL. None,
    # empty strings, already-bracketed text, IPv4 addresses and non-address text (host names) are
    # returned unchanged.
    if not addr or addr.startswith("["):
        return addr
    try:
        if ipaddress.ip_address(addr).version == 6:
            return f"[{addr}]"
    except ValueError:
        pass
    return addr


def build_server_url(host: str, port: int) -> str:
    # [FUNCTION DOC] Returns the plain-HTTP base URL for a host (IPv6 literals bracketed) and port.
    return f"http://{format_v6_uri(host)}:{port}"


def compute_engine_launch_cmd(
    args,
    *,
    interpreter_prefix: list[str],
    node_rank: int,
    worker_type: str,
    base_gpu_id: int,
    sglang_overrides: dict,
    num_gpus_per_engine: int,
    dist_init_addr: str,
    nccl_port: int,
    host: str,
    port: int,
    disaggregation_bootstrap_port: int | None,
    engine_info_bootstrap_port: int,
    gated_launch_port: int,
    random_seed: int,
) -> str:
    # [FUNCTION DOC] Returns the shell-quoted command that starts one engine node. All inputs after
    # the parsed arguments are keyword-only: the interpreter command prefix (list of tokens), node
    # rank, worker type (regular, prefill or decode), first GPU id, per-group overrides, GPUs per
    # engine, distributed init address, NCCL port, host, port, the disaggregation bootstrap port
    # (required for prefill workers), the engine-info bootstrap port, the gated launch port and the
    # random seed. First asserts the installed SGLang supports the gated launch option. Builds the
    # server-argument dictionary, removes square brackets from the host (IPv6 literals are given
    # bare on the command line), renders the dictionary to arguments with the round-trip-checked
    # renderer, and joins the prefix, the module-run flag with the launch-server module, and those
    # arguments with proper shell quoting.
    _assert_launch_gate_served()

    server_args_dict = _compute_server_args(
        args,
        node_rank=node_rank,
        dist_init_addr=dist_init_addr,
        nccl_port=nccl_port,
        host=host,
        port=port,
        worker_type=worker_type,
        disaggregation_bootstrap_port=disaggregation_bootstrap_port,
        base_gpu_id=base_gpu_id,
        engine_info_bootstrap_port=engine_info_bootstrap_port,
        sglang_overrides=sglang_overrides,
        num_gpus_per_engine=num_gpus_per_engine,
        gated_launch_port=gated_launch_port,
        random_seed=random_seed,
    )

    launch_args = {**server_args_dict, "host": server_args_dict["host"].strip("[]")}
    return shlex.join([*interpreter_prefix, "-m", "sglang.launch_server", *server_args_to_argv(launch_args)])


def _compute_server_args(
    args,
    *,
    node_rank: int,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None,
    base_gpu_id: int,
    engine_info_bootstrap_port: int | None,
    sglang_overrides: dict | None,
    num_gpus_per_engine: int | None,
    gated_launch_port: int,
    random_seed: int,
):
    # [FUNCTION DOC] HELPER: Builds the server-argument dictionary for one engine node (parameters
    # as described for the command builder; worker type defaults to regular; overrides and GPUs per
    # engine may be none). Steps: GPUs per engine falls back to the rollout GPUs-per-engine
    # argument; node count is GPUs per engine divided by GPUs per node, at least one. Base entries:
    # checkpoint path, trust remote code, seed, memory saver iff rollout offload is on, host, port,
    # NCCL port, node count and rank, init address, GPU id step one and base id, gated launch port,
    # tensor-parallel size = GPUs per engine, data/pipeline/expert parallel sizes from the user
    # options, skip warmup always, draft-weights CPU backup always, metrics always. An environment
    # variable equal to "1" selects dummy weight loading. Prefill workers: disaggregation mode
    # prefill, round-robin load balancing, and the bootstrap port (assertion error if none).
    # Decode workers: mode decode and prefill round-robin balancing. Routing replay and indexer
    # replay arguments turn on returning routed experts / indexer top-k; fp16 sets the dtype;
    # engine-info bootstrap port is added if given.
    # LoRA: multi-LoRA enables LoRA with batch size = number of adapters, rank = max(configured
    # rank or zero, 1), and targets (a single "all" entry for the all-linear setting, else the
    # configured list). Single-adapter rollout LoRA does the same with batch size one, plus: if the
    # engine loads the adapter from disk, a startup path entry pairing the adapter name with the
    # path; if a path exists but is pushed by the trainer, logs that startup loading is skipped;
    # else logs random initial weights; and when the base CPU backup is enabled, turns on weights
    # CPU backup (the trainer then skips base weight sync).
    # Then applies the group overrides (they win over everything above). Then fills in the user's
    # SGLang options: for each field name of the installed server arguments (decode workers skip the
    # hierarchical cache option) a user option with that name is copied only if the key is not
    # already set. Keys in the dictionary that are not server-argument fields are logged and
    # removed (compatibility with older SGLang). With multi-LoRA: asserts dummy loading is off and
    # defaults the maximum loaded adapters to twice the batch size. Finally defaults the device to
    # "cuda". Returns the dictionary.
    # KNOWN ISSUE: the unknown-key message is logged at info level but is worded as a warning; and
    # a decode worker's hierarchical cache key is never marked as used, so an override of it would
    # be removed as unsupported.
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    kwargs = {
        "model_path": args.hf_checkpoint,
        "trust_remote_code": True,
        "random_seed": random_seed,
        # memory
        "enable_memory_saver": args.offload_rollout,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base_gpu_id,
        "gated_launch_port": gated_launch_port,
        # parallel
        "tp_size": _gpus_per_engine,
        "dp_size": args.sglang_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
        # always serve /metrics so Prometheus scrapers can read engine stats.
        "enable_metrics": True,
    }

    if os.environ.get("MILES_SGLANG_DUMMY_LOAD") == "1":
        kwargs["load_format"] = "dummy"

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs.setdefault("load_balance_method", "round_robin")
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.use_rollout_indexer_replay:
        kwargs["enable_return_indexer_topk"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    if engine_info_bootstrap_port is not None:
        kwargs["engine_info_bootstrap_port"] = engine_info_bootstrap_port

    if is_multi_lora_enabled(args):
        kwargs["enable_lora"] = True
        kwargs["max_loras_per_batch"] = args.multi_lora_n_adapters
        kwargs["max_lora_rank"] = max(getattr(args, "lora_rank", 0), 1)
        kwargs["lora_target_modules"] = (
            ["all"] if args.lora_adapter_targets == "all-linear" else args.lora_adapter_targets
        )
    elif lora_rollout_enabled(args):
        kwargs["enable_lora"] = True
        kwargs["max_loras_per_batch"] = 1
        kwargs["max_lora_rank"] = max(getattr(args, "lora_rank", 0), 1)
        kwargs["lora_target_modules"] = (
            ["all"] if args.lora_adapter_targets == "all-linear" else args.lora_adapter_targets
        )

        if engine_loads_adapter_from_disk(args):
            kwargs["lora_paths"] = [f"{LORA_ADAPTER_NAME}={args.lora_adapter_path}"]
        elif args.lora_adapter_path is not None:
            logger.info("Skipping startup lora_paths: the trainer pushes the adapter in the first weight sync")
        else:
            logger.info("No pre-trained LoRA adapter_path provided, will use random initial weights")

        if lora_base_cpu_backup_enabled(args):
            # Host-RAM mirror of the base weights so they survive
            # torch_memory_saver.pause() across rollout/training swaps without
            # needing to be re-shipped from the trainer. The trainer mirrors
            # this by skipping the base weight sync entirely (see
            # UpdateWeightFromTensor.update_weights).
            kwargs["enable_weights_cpu_backup"] = True
            logger.info(
                "LoRA + colocate: enabling SGLang enable_weights_cpu_backup=True; "
                "the trainer will skip per-step base weight sync."
            )

    # Last, so a per-group override wins over every args-derived default above.
    if sglang_overrides:
        kwargs.update(sglang_overrides)

    unused_keys = set(kwargs.keys())
    for name in _record_field_names(ServerArgs):
        if worker_type == "decode" and name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{name}") and name not in kwargs:
            kwargs[name] = getattr(args, f"sglang_{name}")
        unused_keys.discard(name)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    if is_multi_lora_enabled(args):
        assert kwargs.get("load_format") != "dummy", "Tinker engines must load the frozen base from disk"
        if kwargs.get("max_loaded_loras") is None:
            # use --sglang-max-loaded-loras to override
            # TODO: dynamic allocation
            kwargs["max_loaded_loras"] = 2 * kwargs["max_loras_per_batch"]

    if kwargs.get("device") is None:
        kwargs["device"] = "cuda"

    return kwargs


@functools.cache
def _assert_launch_gate_served() -> None:
    # [FUNCTION DOC] HELPER: Cached (checked once per process) assertion that the installed SGLang
    # server arguments include the gated launch port field; raises an assertion error telling the
    # user to upgrade SGLang otherwise. Returns nothing.
    assert "gated_launch_port" in _record_field_names(ServerArgs), (
        "this sglang has no --gated-launch-port, and miles launches every inference engine through "
        "that gate; upgrade sglang to one that serves it"
    )
