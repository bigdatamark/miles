# [MODULE DOC] Glue between miles and sglang's tensor dumper. SGLang side: builds the environment
# variables for engine processes and, once engines have registered with the router, configures
# the dumper on each over HTTP. Megatron side: DumperMegatronUtil configures the dumper per
# phase and rollout, optionally registers hooks, wraps the forward step to advance the dumper per
# micro-batch, and at the end dumps weights and gradients. Phases: inference, forward-only and
# forward-backward; each has key-value override options on args, with a global enable flag.
# Dump files go under the dumper directory option; only effective data-parallel rank 0 writes
# files; stale directories are removed by global rank 0 of cell 0.
from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
import shutil
from argparse import Namespace
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from sglang.srt.debug_utils.dumper import DumperConfig, _get_rank, dumper

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.ft_utils.process_group_utils import GeneralPGUtil
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.tracking_utils.structured_log import log_structured

logger = logging.getLogger(__name__)

_WORKER_REGISTRATION_TIMEOUT_SECONDS = 120.0


class DumperPhase(enum.Enum):
    # [CLASS DOC] Phases that can be dumped; the string values (inference, fwd_only, fwd_bwd) are
    # used in option names (dumper_<value>) and directory names.
    INFERENCE = "inference"
    FWD_ONLY = "fwd_only"
    FWD_BWD = "fwd_bwd"


# ------------------------------- SGLang -------------------------------------


def get_sglang_env(args: Namespace) -> dict[str, str]:
    # [FUNCTION DOC] Returns an empty mapping unless the inference phase is enabled. Otherwise a
    # mapping with the dumper server port set to "reuse", plus the non-intrusive mode (as text)
    # when the inference overrides set it, plus the source patcher config: the dedicated
    # inference option wins, else the override value (as text) if set. These must be present at
    # engine start because hooks are registered during model loading.
    if not _is_phase_enabled(args, DumperPhase.INFERENCE):
        return {}

    env: dict[str, str] = {"DUMPER_SERVER_PORT": "reuse"}
    overrides = _get_phase_override_configs(args, DumperPhase.INFERENCE)

    # SGLang registers non-intrusive hooks while loading the model. Configs that
    # affect hook registration must be present in the actor environment; the
    # later HTTP configure call only controls active dumping/output location.
    if non_intrusive_mode := overrides.get("non_intrusive_mode"):
        env["DUMPER_NON_INTRUSIVE_MODE"] = str(non_intrusive_mode)

    if source_patcher_config := args.dumper_source_patcher_config_inference:
        env["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config
    elif source_patcher_config := overrides.get("source_patcher_config"):
        env["DUMPER_SOURCE_PATCHER_CONFIG"] = str(source_patcher_config)

    return env


async def configure_sglang(args: Namespace) -> None:
    # [FUNCTION DOC] Does nothing unless the inference phase is enabled. Otherwise waits until the
    # router lists enough engines, deletes the "engines" dump directory (cell 0 rank 0 only),
    # then concurrently posts to each engine's dumper configure endpoint a body with enable true,
    # the dump directory, experiment name "engines/engine_<index>" and the inference overrides
    # (which can override the earlier keys). Engine index follows the URL list order. Logs the
    # engine count. Network and filesystem side effects; request errors propagate.
    if not _is_phase_enabled(args, DumperPhase.INFERENCE):
        return

    from miles.utils.http_utils import post

    worker_urls = await _wait_registered_worker_urls(args)
    overrides = _get_phase_override_configs(args, DumperPhase.INFERENCE)

    engines_dir: Path = _get_dir(args) / "engines"
    _cleanup_dump_dir(engines_dir, indep_dp_rank=0)

    coros = []
    for i, url in enumerate(worker_urls):
        body = {
            "enable": True,
            "dir": str(_get_dir(args)),
            "exp_name": f"engines/engine_{i}",
            **overrides,
        }
        coros.append(post(f"{url}/dumper/configure", body))

    await asyncio.gather(*coros)
    logger.info("Configured dumper on %d SGLang engines", len(worker_urls))


async def _wait_registered_worker_urls(args: Namespace) -> list[str]:
    # [FUNCTION DOC] HELPER: Expected count = number of server cells of the first model in the
    # resolved sglang config. Repeatedly fetches worker urls from the router (retrying on
    # assertion failure, up to 120 seconds) until at least that many are reported; returns the
    # list (may be longer than expected). The rollout module is imported lazily.
    from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls

    expected_worker_count = resolve_sglang_config(args).models[0].num_server_cells

    async def _attempt(_remaining_seconds: float) -> list[str]:
        # [FUNCTION DOC] HELPER: One retry attempt: fetch urls and assert at least the expected
        # number is present (assertion message shows found/expected); returns them.
        worker_urls = await get_worker_urls(args)
        assert (
            len(worker_urls) >= expected_worker_count
        ), f"router reports {len(worker_urls)}/{expected_worker_count} inference engines to configure the dumper on"
        return worker_urls

    return await retry_until_deadline(
        _attempt,
        total_seconds=_WORKER_REGISTRATION_TIMEOUT_SECONDS,
        retry_on=AssertionError,
        log_fields=dict(op="dumper_wait_workers"),
    )


# ------------------------------- Megatron -------------------------------------


class DumperMegatronUtil:
    # [CLASS DOC] Per-phase, per-rollout dumper controller for Megatron training. Attributes:
    # phase, rollout id, the phase's override options, and enabled (whether the phase's overrides
    # enable dumping). Usage: construct before the step, wrap the forward step function, call
    # finalize after the step.
    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        phase: DumperPhase,
        *,
        rollout_id: int,
        store_prefix: str = "",
    ) -> None:
        # [FUNCTION DOC] Stores phase and rollout id, computes the override options, and
        # configures the dumper (see the configure helper; may delete directories and hit
        # barriers). If enabled, registers non-intrusive dumping hooks on the single model chunk
        # (assertion if there is more than one chunk). store_prefix (default empty) is added to the
        # experiment name.
        self.phase = phase
        self.rollout_id = rollout_id
        self.overrides = _get_phase_override_configs(args, phase)
        self.enabled = self._configure(
            args, phase=phase, rollout_id=rollout_id, store_prefix=store_prefix, overrides=self.overrides
        )
        if self.enabled:
            dumper.register_non_intrusive_dumper(self._extract_model(model))

    def wrap_forward_step(self, forward_step_func: Callable) -> Callable:
        # [FUNCTION DOC] Returns the function unchanged when disabled; otherwise a wrapper that
        # advances the dumper step before every call except the first.
        if not self.enabled:
            return forward_step_func

        return _wrap_forward_step_with_stepping(forward_step_func)

    def finalize(self, model: Sequence[torch.nn.Module]) -> None:
        # [FUNCTION DOC] Does nothing when disabled. Otherwise, for the forward-backward phase
        # with the model-grad override set, logs gradient coverage and builds a getter that
        # returns fully gathered gradients. Then resets the dumper, reconfigures it without
        # cleaning previous output (so activations from this rollout survive), dumps the model
        # weights (and gradients via the getter), advances one step, and disables the dumper.
        # Weights and gradients therefore land at step 0 of the reset dumper.
        if not self.enabled:
            return

        extracted_model = self._extract_model(model)
        get_grad: Callable[[torch.nn.Parameter], torch.Tensor | None] | None = None
        if self.phase is DumperPhase.FWD_BWD and self.overrides.get("enable_model_grad"):
            _log_model_grad_coverage(extracted_model)
            get_grad = _build_full_grad_getter(extracted_model)

        # Weights/grads are a once-per-rollout end-state, so pin them to step 0 instead of
        # the running per-microbatch step. _configure already cleaned the scoped paths;
        # disable lazy cleanup after reset to preserve activations from this rollout.
        dumper.reset()
        dumper.configure(cleanup_previous=False)
        dumper.dump_model(extracted_model, get_grad=get_grad)
        dumper.step()
        dumper.configure(enable=False)

    @staticmethod
    def _extract_model(model: Sequence[torch.nn.Module]) -> torch.nn.Module:
        # [FUNCTION DOC] HELPER: Static. Asserts exactly one model chunk (virtual pipeline
        # parallelism unsupported) and returns it.
        assert (
            len(model) == 1
        ), f"Dumper does not yet support virtual pipeline parallelism (got {len(model)} model chunks)"
        return model[0]

    @staticmethod
    def _configure(
        args: Namespace,
        *,
        phase: DumperPhase,
        rollout_id: int,
        store_prefix: str = "",
        overrides: dict[str, Any] | None = None,
    ) -> bool:
        # [FUNCTION DOC] HELPER: Static. Uses the given overrides or computes them; returns false
        # if they do not enable dumping. Otherwise builds a config: directory from args,
        # experiment name "<phase>/<store_prefix>rollout_<id>", console output off by default,
        # then the overrides on top. On ranks whose effective data-parallel rank is not 0, file
        # and console output are forced off (they still join collectives). Validates by building
        # the dumper config object, resets the dumper, deletes the whole phase directory when the
        # rollout id is 0 and the experiment directory always (both only by global rank 0 of
        # cell 0), synchronizes ranks and cells, and applies the config. Returns true.
        if overrides is None:
            overrides = _get_phase_override_configs(args, phase)
        if not overrides.get("enable"):
            return False

        exp_name = f"{phase.value}/{store_prefix}rollout_{rollout_id}"
        merged = {
            "dir": str(_get_dir(args)),
            "exp_name": exp_name,
            "enable_output_console": False,
            **overrides,
        }

        # Only write dump files on effective DP rank 0 (covers both intra-DP
        # and indep-DP). Other DP ranks still participate in dumper collectives
        # (barrier, broadcast, allgather) but don't produce output files.
        # TODO: optimize — non-DP-rank-0 ranks currently run full dumper logic
        # (forward hooks, model iteration) without producing output.
        if get_parallel_state().effective_dp.rank != 0:
            merged["enable_output_file"] = False
            merged["enable_output_console"] = False

        full_config = DumperConfig(**merged)
        dumper.reset()
        # Wipe the whole phase dir only at run start (rollout 0). Gating on a
        # per-process latch instead would make a respawned process re-wipe the
        # phase dir mid-run, deleting dumps already written by surviving cells.
        indep_dp_rank = get_parallel_state().indep_dp.rank
        if rollout_id == 0:
            _cleanup_dump_dir(Path(merged["dir"]) / phase.value, indep_dp_rank=indep_dp_rank)
        _cleanup_dump_dir(Path(merged["dir"]) / merged["exp_name"], indep_dp_rank=indep_dp_rank)
        _barrier_after_dump_dir_cleanup()
        dumper.configure(**dataclasses.asdict(full_config))
        return True


def _build_full_grad_getter(
    model_chunk: torch.nn.Module,
) -> Callable[[torch.nn.Parameter], torch.Tensor | None]:
    # [FUNCTION DOC] HELPER: Collective over the optimizer-instance groups. For every gradient
    # bucket group (regular and expert) that uses the distributed optimizer, and each of its
    # buckets: if the optimizer instance size is above 1, allocates a buffer like the bucket's
    # gradient data, takes this rank's owned equal shard, and all-gathers shards into the buffer
    # (reading, not mutating, the bucket); otherwise uses the bucket data directly. Maps every
    # parameter in the bucket to its slice of the flattened buffer, reshaped to the parameter
    # shape. Returns a getter: the gathered gradient if the parameter was mapped, else the
    # parameter's grad, else its main grad, else none. Gathered views reference the temporary
    # buffers, kept alive by the getter.
    """Build get_grad(param): all-gather distributed-optimizer grad shards into a
    fresh buffer (grad_data is read, not mutated) and return per-param views."""
    grad_map: dict[torch.nn.Parameter, torch.Tensor] = {}
    # Bucket iteration copied from indep_dp.allreduce_grads_and_losses_across_replicas,
    # which cross-cell all-reduces these same bucket.grad_data buffers.
    bucket_groups = list(getattr(model_chunk, "bucket_groups", [])) + list(
        getattr(model_chunk, "expert_parallel_bucket_groups", [])
    )
    for bucket_group in bucket_groups:
        if not bucket_group.ddp_config.use_distributed_optimizer:
            continue
        # Same group/size/rank Megatron's grad reduce-scatter uses
        # (Megatron-LM param_and_grad_buffer.py _ParamAndGradBucketGroup.start_grad_sync).
        group = bucket_group.intra_distributed_optimizer_instance_group
        instance_size = bucket_group.intra_distributed_optimizer_instance_size
        instance_rank = bucket_group.intra_distributed_optimizer_instance_rank
        for bucket in bucket_group.buckets:
            grad_data = bucket.grad_data
            if instance_size > 1:
                full = torch.empty_like(grad_data)
                # shard slicing copied from Megatron shard_buffer(); local_shard is
                # this rank's owned (reduce-scattered) slice.
                shard_numel = grad_data.numel() // instance_size
                local_shard = grad_data[instance_rank * shard_numel : (instance_rank + 1) * shard_numel]
                # all-gather copied from Megatron start_param_sync (it does this on
                # bucket.param_data); here on grad, into a fresh buffer (grad_data read-only).
                dist.all_gather_into_tensor(full, local_shard.contiguous(), group=group)
            else:
                full = grad_data
            flat = full.view(-1)
            # per-param slice copied from Megatron's own bucket.param_data.view(-1)
            # [start:end].view(shape), using the bucket-local bucket.param_to_index.
            for param, (start, end) in bucket.param_to_index.items():
                grad_map[param] = flat[start:end].view(param.shape)

    def get_grad(param: torch.nn.Parameter) -> torch.Tensor | None:
        # [FUNCTION DOC] HELPER: Looks the parameter up in the gathered map; falls back to its grad
        # attribute, then its main-grad attribute, then none.
        reduced = grad_map.get(param)
        if reduced is not None:
            return reduced
        # fallback copied from sglang dumper's original grad read (.grad else main_grad).
        return param.grad if param.grad is not None else getattr(param, "main_grad", None)

    return get_grad


def _log_model_grad_coverage(model: torch.nn.Module) -> None:
    # [FUNCTION DOC] HELPER: Over trainable parameters, counts those with a grad (grad attribute
    # or main grad) and collects names of those without. Logs at info level the rank (distributed
    # rank if initialized, else the dumper's rank), counts, and the first 20 missing names.
    missing: list[str] = []
    with_grad = 0
    total = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        total += 1
        grad = param.grad if param.grad is not None else getattr(param, "main_grad", None)
        if grad is None:
            missing.append(name)
        else:
            with_grad += 1

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else _get_rank()
    logger.info(
        "Dumper fwd_bwd model grad coverage rank=%s with_grad=%d total=%d missing=%d missing_names=%s",
        rank,
        with_grad,
        total,
        len(missing),
        missing[:20],
    )


def _wrap_forward_step_with_stepping(forward_step_func: Callable) -> Callable:
    # [FUNCTION DOC] HELPER: Returns a wrapper that, on every call after the first, advances the
    # dumper by one step before delegating with the same arguments and returning the result (so
    # each micro-batch gets its own step index).
    is_first_call = True

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: The wrapper described above.
        nonlocal is_first_call
        if not is_first_call:
            dumper.step()
        is_first_call = False
        return forward_step_func(*args, **kwargs)

    return _wrapped


# ------------------------------- Common -------------------------------------


def _cleanup_dump_dir(dump_dir: Path, *, indep_dp_rank: int) -> None:
    # [FUNCTION DOC] HELPER: Recursively deletes the directory, only when the dumper rank is 0,
    # the independent data-parallel (cell) rank is 0 and the directory exists. An operating
    # system error during deletion is logged as a warning and swallowed.
    # Only cell 0's rank 0 deletes — avoids race when multiple cells' rank 0
    # all see _get_rank()==0 and try to rmtree the same directory.
    # Best-effort: stale handles from a peer that crashed (NFS .nfsXXXX stubs)
    # can make rmtree fail with "Directory not empty"; we don't want that to
    # propagate up and mark the (healthy) cell as errored.
    if (_get_rank() == 0) and (indep_dp_rank == 0) and dump_dir.is_dir():
        try:
            shutil.rmtree(dump_dir)
        except OSError:
            logger.warning("dump dir cleanup failed; continuing", exc_info=True)


def _barrier_after_dump_dir_cleanup() -> None:
    # [FUNCTION DOC] HELPER: Barrier on the default group if distributed is initialized. Then, if
    # there is a cross-cell group, logs a structured start event, runs a barrier on that group and
    # logs a success end event. If that barrier raises (a dead peer aborted the group), logs an
    # error event marked degraded and continues without raising.
    if dist.is_initialized():
        dist.barrier()

    indep_dp = get_parallel_state().indep_dp
    if indep_dp.group is not None:
        log_structured(
            logger.info,
            tag="ft",
            op="cross_cell",
            phase="start",
            kind="dump_barrier",
            cell_rank=indep_dp.rank,
            members=indep_dp.size,
            **indep_dp.debug_info,
        )
        try:
            GeneralPGUtil.create(indep_dp.group).barrier(indep_dp.group)
            log_structured(
                logger.info,
                tag="ft",
                op="cross_cell",
                phase="end",
                kind="dump_barrier",
                cell_rank=indep_dp.rank,
                members=indep_dp.size,
                **indep_dp.debug_info,
                success=True,
            )
        except Exception:
            # A dead peer aborts the cross-cell PG and releases this barrier with an
            # error. Proceed: the peer cannot dump anyway, and the gradient allreduce
            # later in the step turns the abort into DISCARDED_SHOULD_RETRY.
            log_structured(
                logger.error,
                tag="ft",
                op="cross_cell",
                phase="end",
                kind="dump_barrier",
                cell_rank=indep_dp.rank,
                members=indep_dp.size,
                **indep_dp.debug_info,
                success=False,
                degraded=True,
                exc_info=True,
            )


def _get_phase_override_configs(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Reads the phase's key-value option (attribute dumper_<phase>) and
    # converts it with the dumper config's parser into a mapping; returns it with the global
    # dumper enable flag first, so an "enable" key in the phase options overrides the global flag.
    raw = getattr(args, f"dumper_{phase.value}")
    return {"enable": args.dumper_enable, **DumperConfig._kv_pairs_to_dict(raw)}


def _is_phase_enabled(args: Namespace, phase: DumperPhase) -> bool:
    # [FUNCTION DOC] HELPER: The enable value from the phase's overrides (global flag unless the
    # phase options set it), false if absent.
    return _get_phase_override_configs(args, phase).get("enable", False)


def _get_dir(args: Namespace) -> Path:
    # [FUNCTION DOC] HELPER: The dumper directory option as a path.
    return Path(args.dumper_dir)
