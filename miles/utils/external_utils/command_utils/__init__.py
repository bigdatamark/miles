# [PACKAGE DOC] Command utilities used by launch scripts and tests to run training and helper
# commands on a chosen cluster backend (Ray or Kubernetes through helm). This package re-exports
# the public surface: the backend configuration and request types, config loading from the
# environment, hardware detection and tables, run-id and environment-variable helpers, wandb and
# mooncake argument builders, pseudo-file encoding, the rsync command builder and the command-line
# dataclass decorator. Submodules: base_backend (abstract backend and shared launch checks), common
# (stateless helpers), helm_backend (Kubernetes implementation).
from miles.utils.external_utils.command_utils.base_backend import (
    CommandUtilConfig,
    ExecuteTrainConfig,
    default_config,
    exclusive_path_lock,
    resolve_extra_env_vars,
    resolve_hardware,
)
from miles.utils.external_utils.command_utils.common import (
    GENERATION_HARDWARE,
    NUM_GPUS_OF_HARDWARE,
    create_run_id,
    detect_hardware,
    encode_pseudo_file,
    get_bool_env_var,
    get_default_wandb_args,
    get_env_enable_infinite_run,
    get_mooncake_object_store_args,
    repo_base_dir,
    rsync_cmd,
)
from miles.utils.typer_utils import dataclass_cli

__all__ = [
    "CommandUtilConfig",
    "ExecuteTrainConfig",
    "GENERATION_HARDWARE",
    "NUM_GPUS_OF_HARDWARE",
    "create_run_id",
    "dataclass_cli",
    "default_config",
    "detect_hardware",
    "encode_pseudo_file",
    "exclusive_path_lock",
    "get_bool_env_var",
    "get_default_wandb_args",
    "get_env_enable_infinite_run",
    "get_mooncake_object_store_args",
    "repo_base_dir",
    "resolve_extra_env_vars",
    "resolve_hardware",
    "rsync_cmd",
]
