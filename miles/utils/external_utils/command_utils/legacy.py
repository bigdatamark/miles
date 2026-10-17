# [MODULE DOC]
# Legacy (v1) function-style facade of command_utils. Re-exports commonly used helpers and exposes
# module-level functions (execute train, run commands on cpu/gpu/all nodes, convert checkpoints,
# rsync, start ray workers, download datasets, cast fp8 to bf16, check nvlink) that each create a
# ray command backend on the fly and delegate to it. Only the ray cluster backend is supported;
# anything else is rejected with an assertion error pointing to the newer backend API.
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig as _CurrentExecuteTrainConfig
from miles.utils.external_utils.command_utils.base_backend import resolve_hardware
from miles.utils.external_utils.command_utils.common import (
    GENERATION_HARDWARE,
    MOONCAKE_MASTER_LOG_PATH,
    MOONCAKE_MASTER_METRICS_PORT,
    MOONCAKE_MASTER_PORT,
    NUM_GPUS_OF_HARDWARE,
    create_run_id,
    encode_pseudo_file,
    get_bool_env_var,
    get_default_wandb_args,
    get_env_enable_infinite_run,
    get_mooncake_object_store_args,
    repo_base_dir,
    rsync_cmd,
)
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.external_utils.command_utils.ray_backend.command import start_mooncake_master
from miles.utils.typer_utils import dataclass_cli
from miles.utils.workers.types import ClusterBackend

__all__ = [
    "ExecuteTrainConfig",
    "GENERATION_HARDWARE",
    "MOONCAKE_MASTER_LOG_PATH",
    "MOONCAKE_MASTER_METRICS_PORT",
    "MOONCAKE_MASTER_PORT",
    "NUM_GPUS_OF_HARDWARE",
    "check_has_nvlink",
    "convert_checkpoint",
    "create_run_id",
    "dataclass_cli",
    "encode_pseudo_file",
    "exec_command_cpu",
    "exec_command_gpu",
    "exec_command_multi_node",
    "execute_train",
    "fp8_cast_bf16",
    "get_bool_env_var",
    "get_default_wandb_args",
    "get_env_enable_infinite_run",
    "get_mooncake_object_store_args",
    "hf_download_dataset",
    "repo_base_dir",
    "resolve_hardware",
    "rsync_simple",
    "ssh_start_ray_workers",
    "start_mooncake_master",
]


@dataclass
class ExecuteTrainConfig:
    # [CLASS DOC] Legacy train configuration: cuda core dump flag (default off), node count
    # (default from the scheduler job node count environment variable, else 1), extra environment
    # variable text (default empty), output directory (default a shared data directory) and a
    # keyword-only external mooncake flag. Converted field by field to the current config type.
    cuda_core_dump: bool = False
    num_nodes: int = field(default_factory=lambda: int(os.environ.get("SLURM_JOB_NUM_NODES", "1")))
    extra_env_vars: str = ""
    output_dir: str = "/root/shared_data"
    external_mooncake: bool = field(default=False, kw_only=True)


def execute_train(
    train_args: str,
    num_gpus_per_node: int,
    megatron_model_type: str | None,
    train_script: str = "train.py",
    before_ray_job_submit: Callable[[], None] | None = None,
    extra_env_vars: dict[str, str] | None = None,
    config: ExecuteTrainConfig | None = None,
    megatron_path: str = "/root/Megatron-LM",
    job_lifetime: Literal["independent", "launcher"] = "independent",
) -> None:
    # [FUNCTION DOC] Runs a training job through the ray backend. Uses a default config if none is
    # given, converts it to the current config type, creates the backend (asserting ray), and
    # forwards all arguments: train argument text, gpus per node, megatron model type, script name,
    # optional hook run before the ray job is submitted, extra environment mapping, megatron path
    # and job lifetime (independent of, or tied to, the launcher). Side effects: processes, ray.
    if config is None:
        config = ExecuteTrainConfig()
    current_config = _to_current_config(config)
    _create_ray_backend(current_config).execute_train(
        config=current_config,
        train_args=train_args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        train_script=train_script,
        before_ray_job_submit=before_ray_job_submit,
        extra_env_vars=extra_env_vars,
        megatron_path=megatron_path,
        job_lifetime=job_lifetime,
    )


def exec_command_cpu(cmd: str, capture_output: bool = False) -> str | None:
    # [FUNCTION DOC] Runs a shell command on the local machine via the ray backend; returns its
    # captured output when requested, else none.
    return _create_ray_backend().exec_command_cpu(cmd, capture_output=capture_output)


def exec_command_gpu(cmd: str, capture_output: bool = False) -> str | None:
    # [FUNCTION DOC] Runs a shell command in a gpu context via the ray backend; returns captured
    # output when requested, else none.
    return _create_ray_backend().exec_command_gpu(cmd, capture_output=capture_output)


def exec_command_multi_node(cmd: str, capture_output: bool = False, num_nodes: int | None = None) -> list[str | None]:
    # [FUNCTION DOC] Runs a shell command on each of the first num_nodes nodes (all when none) via
    # the ray backend; returns a list of per-node outputs (none entries when not captured).
    return _create_ray_backend().exec_command_multi_node(cmd, capture_output=capture_output, num_nodes=num_nodes)


def convert_checkpoint(
    model_name: str,
    megatron_model_type: str | None,
    num_gpus_per_node: int,
    multinode: bool = False,
    num_nodes: int | None = None,
    extra_args: str = "",
    dir_dst: str = "/root",
    hf_checkpoint: str | None = None,
    megatron_path: str = "/root/Megatron-LM",
) -> None:
    # [FUNCTION DOC] Converts a checkpoint via the ray backend; forwards all arguments unchanged
    # (model name, megatron model type, gpus per node, multinode flag, node count, extra argument
    # text, destination directory, optional hf checkpoint, megatron path).
    _create_ray_backend().convert_checkpoint(
        model_name=model_name,
        megatron_model_type=megatron_model_type,
        num_gpus_per_node=num_gpus_per_node,
        multinode=multinode,
        num_nodes=num_nodes,
        extra_args=extra_args,
        dir_dst=dir_dst,
        hf_checkpoint=hf_checkpoint,
        megatron_path=megatron_path,
    )


def rsync_simple(path_src: str, path_dst: str, num_nodes: int | None = None) -> None:
    # [FUNCTION DOC] Runs an rsync command (built by the common helper) on the nodes via the
    # multi-node executor to sync source to destination.
    _create_ray_backend().exec_command_multi_node(rsync_cmd(path_src=path_src, path_dst=path_dst), num_nodes=num_nodes)


def ssh_start_ray_workers(
    master_addr: str,
    num_gpus_per_node: int,
    hostfile: str = "/root/mpi_rack_hostfile",
    head_host: str | None = None,
) -> None:
    # [FUNCTION DOC] Starts ray workers over ssh via the ray backend, using the hostfile (default
    # path given), master address, gpus per node and optional head host.
    _create_ray_backend().ssh_start_ray_workers(
        master_addr=master_addr,
        num_gpus_per_node=num_gpus_per_node,
        hostfile=hostfile,
        head_host=head_host,
    )


def hf_download_dataset(full_name: str, data_dir: str = "/root/datasets") -> None:
    # [FUNCTION DOC] Downloads a dataset by its full hub name into the data directory via the ray
    # backend.
    _create_ray_backend().hf_download_dataset(full_name=full_name, data_dir=data_dir)


def fp8_cast_bf16(path_src: str, path_dst: str) -> None:
    # [FUNCTION DOC] Casts an fp8 checkpoint at the source path to bf16 at the destination path via
    # the ray backend.
    _create_ray_backend().fp8_cast_bf16(path_src=path_src, path_dst=path_dst)


def check_has_nvlink() -> bool:
    # [FUNCTION DOC] Returns whether the machine has nvlink connections, using the backend's private
    # probe (runs a gpu topology query).
    return _create_ray_backend()._check_has_nvlink()


def _to_current_config(config: ExecuteTrainConfig) -> _CurrentExecuteTrainConfig:
    # [FUNCTION DOC] HELPER: Copies the five legacy config fields to the current config type; the
    # current type's other fields keep their defaults.
    return _CurrentExecuteTrainConfig(
        cuda_core_dump=config.cuda_core_dump,
        external_mooncake=config.external_mooncake,
        num_nodes=config.num_nodes,
        extra_env_vars=config.extra_env_vars,
        output_dir=config.output_dir,
    )


def _create_ray_backend(config: _CurrentExecuteTrainConfig | None = None) -> RayCommandBackend:
    # [FUNCTION DOC] HELPER: Creates the backend from the config (a default current config if none).
    # Asserts the configured cluster backend is ray (the legacy API cannot express other launches)
    # and that the created backend is a ray command backend.
    if config is None:
        config = _CurrentExecuteTrainConfig()
    assert config.cluster_backend is ClusterBackend.RAY, (
        f"This is the v1 command_utils API, which was written before miles could launch onto anything but a ray "
        f"cluster and therefore has no way to express the namespace, release and manifests a "
        f"{config.cluster_backend.value} launch is made of, yet the config it was handed asks for "
        f"{config.cluster_backend.value}; drop this module and drive the launch through the v2 backend "
        f"API instead, i.e. build the backend with config.create_backend() and call execute_train on it"
    )
    backend = config.create_backend()
    assert isinstance(backend, RayCommandBackend)
    return backend
