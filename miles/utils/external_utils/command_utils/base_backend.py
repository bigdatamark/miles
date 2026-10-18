# [MODULE DOC] Backend-independent core of the command utilities. Defines the launch configuration
# dataclasses (a base config selecting the cluster backend, and a training config with run
# identity, deploy component, hot restart and environment options), the validated training request
# model, the abstract command backend (training launch with argument checks, checkpoint
# conversion with a version-stamped cache, ray worker startup over ssh, dataset download, fp8 to
# bf16 casting, command execution entry points that always delegate to protected inner methods),
# and helpers for file-lock serialization, hardware resolution and conflict checks between launch
# config and train arguments. A conversion version constant invalidates cached checkpoint
# conversions when weight layouts change.
from __future__ import annotations

import fcntl
import logging
import os
import shlex
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, get_args

from miles.utils.external_utils.command_utils.common import (
    ArgvManipulator,
    _parse_extra_env_vars,
    _pythonpath_with_sources,
    create_run_id,
    detect_hardware,
    repo_base_dir,
    run_shell_command,
)
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.typer_utils import dataclass_from_env
from miles.utils.workers.types import ClusterBackend, DeployComponent, HotRestartComponent, parse_hot_restart

logger = logging.getLogger(__name__)


CONVERSION_VERSION_FILE = "miles_conversion_version.txt"
# Bump when a conversion this checkout writes would not load on a checkout holding another value:
# the skip path trusts a cache only when the stamp equals the running checkout's value, so every
# weight layout change bumps it. Values 1 and 2 were written briefly by an earlier numbering and
# are never trusted. 3: the layouts before the head-sharded linear-attention layer.
# 4: Qwen3.5 / Qwen3-Next GDN weights in group-major rows. 5: Kimi-K3 KDA under linear_attn.*.
CONVERSION_VERSION = "5"


@dataclass
class CommandUtilConfig:
    # [CLASS DOC] Base launch configuration. Fields: cluster_backend (default Ray), namespace
    # (Kubernetes namespace, empty by default), helm_values (tuple of values file paths, empty),
    # ci_run flag (default false).
    cluster_backend: ClusterBackend = ClusterBackend.RAY
    namespace: str = ""
    helm_values: tuple[str, ...] = ()
    ci_run: bool = False

    @classmethod
    def from_env(cls, **kwargs: Any) -> CommandUtilConfig:
        # [FUNCTION DOC] Class method: builds the config from environment variables via the shared
        # loader, with the keyword arguments as explicit overrides.
        return dataclass_from_env(cls, overrides=kwargs)

    def create_backend(self) -> BaseCommandBackend:
        # [FUNCTION DOC] Creates the backend for the configured cluster backend, importing its
        # module lazily: Kubernetes gives the helm-based backend, Ray gives the Ray backend, each
        # constructed with this config.
        # KNOWN ISSUE: the match has no default branch, so any other cluster backend value would
        # return none silently.
        match self.cluster_backend:
            case ClusterBackend.KUBERNETES:
                from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend

                return KubernetesCommandBackend(self)
            case ClusterBackend.RAY:
                from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend

                return RayCommandBackend(self)


# This class can be extended by concrete scripts
@dataclass
class ExecuteTrainConfig(CommandUtilConfig):
    # [CLASS DOC] Training launch configuration (extendable by concrete scripts). Adds: cuda_core_dump
    # (default off), num_nodes (default from the SLURM_JOB_NUM_NODES environment variable, else
    # 1), extra_env_vars (a string, json or space-separated key=value pairs; empty), output_dir
    # (default /root/shared_data), deploy_component (default all), optional deploy_instance_id,
    # hot_restart (string spec, empty), run_id (default generated), optional run_uuid,
    # skip_upgrade_check and external_mooncake flags (default off).
    cuda_core_dump: bool = False
    num_nodes: int = field(default_factory=lambda: int(os.environ.get("SLURM_JOB_NUM_NODES", "1")))
    extra_env_vars: str = ""
    output_dir: str = "/root/shared_data"
    deploy_component: DeployComponent = DeployComponent.ALL
    deploy_instance_id: str | None = None
    hot_restart: str = ""
    run_id: str = field(default_factory=create_run_id)
    run_uuid: str | None = None
    skip_upgrade_check: bool = False
    external_mooncake: bool = False

    @property
    def parsed_hot_restart(self) -> list[HotRestartComponent]:
        # [FUNCTION DOC] Property: the hot restart spec string parsed into a list of components.
        return parse_hot_restart(self.hot_restart)


def default_config(config_class: type = ExecuteTrainConfig) -> ExecuteTrainConfig:
    # [FUNCTION DOC] Builds an instance of the given config class (default the training config)
    # from environment variables with the shared loader.
    return dataclass_from_env(config_class)


class ExecuteTrainRequest(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable, strictly validated request passed to a backend's train launch.
    # Fields: the final train argument string, GPUs per node, Megatron model type (none for
    # non-Megatron runs), absolute train script path, flag for the non-Megatron (FSDP) backend,
    # extra environment variables, Megatron path, optional callback to run before a Ray job is
    # submitted, job lifetime (independent by default, or tied to the launcher), a mapping from
    # role to preparation command, and a list of extra manifests.
    train_args: str
    num_gpus_per_node: int
    megatron_model_type: str | None
    train_script: str
    train_backend_fsdp: bool
    extra_env_vars: dict[str, str]
    megatron_path: str
    before_ray_job_submit: Callable[[], None] | None
    job_lifetime: Literal["independent", "launcher"] = "independent"
    prepare_cmd: dict[str, str]
    extra_manifests: list[str]


CLUSTER_BACKEND_FLAG = "--cluster-backend"
_DEPLOY_COMPONENT_FLAG = "--deploy-component"
_RUN_UUID_FLAG = "--run-uuid"

TRAINER_ROLE = "trainer"
_PREPARE_CMD_ROLES = frozenset({TRAINER_ROLE})


_DELEGATING_METHODS = ("execute_train", "exec_command_cpu", "exec_command_gpu", "exec_command_multi_node")


class BaseCommandBackend(ABC):
    # [CLASS DOC] Abstract command backend. Public methods (execute_train, exec_command_cpu,
    # exec_command_gpu, exec_command_multi_node) are defined only here and delegate to protected
    # "_inner" hooks that subclasses implement, so that one place can intercept every command; a
    # subclass defining one of those public names fails at class creation. Other helpers
    # (checkpoint conversion, ray workers over ssh, dataset download, fp8 cast) are built from the
    # command methods. Holds the launch config.
    def __init__(self, config: CommandUtilConfig) -> None:
        # [FUNCTION DOC] Asserts the config is a command config, configures the process logger
        # under the name "launcher" (side effect on logging), and stores the config.
        from miles.utils.logging_utils import configure_logger_raw

        assert isinstance(config, CommandUtilConfig), "config must be a CommandUtilConfig"
        configure_logger_raw("launcher")
        self.config = config

    def __init_subclass__(cls, **kwargs: object) -> None:
        # [FUNCTION DOC] Class-creation hook: asserts the new subclass does not define, in its own
        # namespace, any of the four delegating public methods (message tells it to implement the
        # inner variant). Inherited definitions from intermediate subclasses are not checked.
        super().__init_subclass__(**kwargs)
        for name in _DELEGATING_METHODS:
            assert name not in vars(cls), (
                f"{cls.__name__} defines its own {name}, but every command any backend runs has to pass through "
                f"the one BaseCommandBackend defines, so that stubbing that single method out cannot be bypassed "
                f"by a backend nobody remembered to name; implement _{name}_inner instead"
            )

    def execute_train(
        self,
        train_args: str,
        num_gpus_per_node: int,
        megatron_model_type: str | None,
        train_script: str = "train.py",
        before_ray_job_submit: Callable[[], None] | None = None,
        extra_env_vars: dict[str, str] | None = None,
        megatron_path: str = "/root/Megatron-LM",
        job_lifetime: Literal["independent", "launcher"] = "independent",
        prepare_cmd: dict[str, str] | None = None,
        extra_manifests: list[str] | None = None,
        config: ExecuteTrainConfig | None = None,
    ) -> None:
        # [FUNCTION DOC] Validates and completes a training launch, then hands a request to the
        # backend hook. Inputs: train argument string, GPUs per node, Megatron model type, script
        # (default train.py, made absolute against the repository root if relative), optional
        # pre-submit callback, extra env vars (default empty), Megatron path (default
        # /root/Megatron-LM), job lifetime (independent or launcher), preparation commands per role
        # (only the trainer role is allowed), extra manifests, and an optional config (default the
        # backend's own, which must then be a training config). Checks (assertions unless noted):
        # config's cluster backend equals the backend's; job lifetime valid; raises a value error if
        # the removed MILES_ROUTER_EXTERNAL_HOST variable appears in the process environment or the
        # resolved extra env vars; hot restart only on Kubernetes; Megatron model type given exactly
        # when the train backend flag (last occurrence, default Megatron) is Megatron; train args do
        # not name a different cluster backend, deploy component, or (when set) run uuid. Then
        # appends the deploy component flag, the deploy instance id flag if set, and the run uuid
        # flag if set, and calls the inner hook with a request whose FSDP flag is the opposite of the
        # Megatron check.
        if config is None:
            assert isinstance(
                self.config, ExecuteTrainConfig
            ), "execute_train requires an ExecuteTrainConfig, either as config or as the backend's config"
            config = self.config

        assert config.cluster_backend is self.config.cluster_backend, (
            f"This backend was built to talk to {self.config.cluster_backend.value}, but the launch it is handed "
            f"describes a run on {config.cluster_backend.value}, so everything this launch installs would be named "
            f"for one cluster and installed onto the other; build the backend from the config of the launch"
        )

        assert job_lifetime in ("independent", "launcher")
        extra_env_vars = extra_env_vars if extra_env_vars is not None else {}
        # nothing reads this variable any more, so an old export would be ignored without a word
        if "MILES_ROUTER_EXTERNAL_HOST" in {**os.environ, **resolve_extra_env_vars(extra_env_vars, config)}:
            raise ValueError(
                "MILES_ROUTER_EXTERNAL_HOST is no longer read. Pass --session-server-external-host for one host that "
                "reaches every session server, or set MILES_NODE_EXTERNAL_IP on each node to its own address."
            )
        assert not (
            config.parsed_hot_restart and config.cluster_backend is not ClusterBackend.KUBERNETES
        ), "--hot-restart is only supported on the kubernetes backend"

        prepare_cmd = prepare_cmd if prepare_cmd is not None else {}
        assert set(prepare_cmd) <= _PREPARE_CMD_ROLES, (
            f"prepare_cmd names the roles {sorted(set(prepare_cmd) - _PREPARE_CMD_ROLES)}, but a backend only "
            f"knows how to run a preparation command for {sorted(_PREPARE_CMD_ROLES)}"
        )
        if not os.path.isabs(train_script):
            train_script = f"{repo_base_dir}/{train_script}"

        train_argv = shlex.split(train_args)
        train_backend_megatron = ArgvManipulator.get_effective(train_argv, "--train-backend") in (None, "megatron")
        assert train_backend_megatron == (megatron_model_type is not None), (
            f"megatron_model_type must be set for a Megatron run and omitted otherwise "
            f"(megatron={train_backend_megatron}, megatron_model_type={megatron_model_type!r})"
        )
        _assert_train_args_name_no_other_backend(train_argv, cluster_backend=config.cluster_backend.value)
        _assert_train_args_name_no_other_deploy_component(train_argv, deploy_component=config.deploy_component.value)
        train_args = f"{train_args} {_DEPLOY_COMPONENT_FLAG} {config.deploy_component.value}"
        if config.deploy_instance_id is not None:
            train_args = f"{train_args} --deploy-instance-id {config.deploy_instance_id}"
        if config.run_uuid is not None:
            _assert_train_args_name_no_other_run_uuid(train_argv, run_uuid=config.run_uuid)
            train_args = f"{train_args} {_RUN_UUID_FLAG} {config.run_uuid}"

        self._execute_train_inner(
            request=ExecuteTrainRequest(
                train_args=train_args,
                num_gpus_per_node=num_gpus_per_node,
                megatron_model_type=megatron_model_type,
                train_script=train_script,
                train_backend_fsdp=not train_backend_megatron,
                extra_env_vars=extra_env_vars if extra_env_vars is not None else {},
                megatron_path=megatron_path,
                before_ray_job_submit=before_ray_job_submit,
                job_lifetime=job_lifetime,
                prepare_cmd=prepare_cmd,
                extra_manifests=extra_manifests if extra_manifests is not None else [],
            ),
            config=config,
        )

    def convert_checkpoint(
        self,
        model_name,
        megatron_model_type,
        num_gpus_per_node: int,
        multinode: bool = False,
        num_nodes: int | None = None,
        extra_args: str = "",
        dir_dst: str = "/root",
        hf_checkpoint: str | None = None,
        megatron_path: str = "/root/Megatron-LM",
    ):
        # [FUNCTION DOC] Converts a HuggingFace checkpoint to Megatron torch-dist format, cached.
        # Destination = dir_dst (default /root) + "/<model_name>_torch_dist"; source defaults to
        # /root/models/<model_name>. Under an exclusive file lock on the destination: skips if the
        # destination's latest-iteration tracker says "release" and its version stamp equals the
        # current conversion version; logs a redo if the tracker says release but the stamp is
        # missing or different. Otherwise runs the conversion tool under torchrun with the model
        # arguments, source, destination and extra args, through the GPU command method, or, for
        # multinode, the multi-node method with placeholders for master address, node count and node
        # rank (master port 23456), using the given node count. Python path includes the repository,
        # the Megatron path and the existing path. Afterwards writes the version stamp (creating
        # directories).
        hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"

        # TODO shall we make it in host-mapped folder and thus can cache it to speedup CI
        path_dst = f"{dir_dst}/{model_name}_torch_dist"
        with exclusive_path_lock(path_dst):
            tracker = Path(path_dst) / "latest_checkpointed_iteration.txt"
            stamp = Path(path_dst) / CONVERSION_VERSION_FILE
            cached = tracker.exists() and tracker.read_text().strip() == "release"
            if cached and stamp.exists() and stamp.read_text().strip() == CONVERSION_VERSION:
                logger.info(f"convert_checkpoint skip {path_dst} since tracker is 'release'")
                return
            if cached:
                logger.info(f"convert_checkpoint redo {path_dst}: converted before layout {CONVERSION_VERSION}")

            multinode_args = ""
            if multinode:
                multinode_args = (
                    "--master-addr {{master_addr}} "
                    "--master-port 23456 "
                    "--nnodes={{nnodes}} "
                    "--node-rank {{node_rank}} "
                )

            if multinode:
                fn = partial(self.exec_command_multi_node, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
            else:
                fn = partial(self.exec_command_gpu, num_gpus_per_node=num_gpus_per_node)
            pythonpath = shlex.quote(_pythonpath_with_sources(megatron_path))
            fn(
                f"PYTHONPATH={pythonpath} "
                f"torchrun "
                f"--nproc-per-node {num_gpus_per_node} "
                f"{multinode_args}"
                f"{repo_base_dir}/tools/convert_hf_to_torch_dist.py "
                f"{shell_safe_model_args(megatron_model_type)} "
                f"--hf-checkpoint {hf_checkpoint} "
                f"--save {path_dst} "
                f"{extra_args}"
            )
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text(CONVERSION_VERSION)

    def ssh_start_ray_workers(
        self,
        master_addr: str,
        num_gpus_per_node: int,
        hostfile: str = "/root/mpi_rack_hostfile",
        head_host: str | None = None,
    ):
        # [FUNCTION DOC] Runs one local CPU shell command that loops over the first column of the
        # hostfile (default /root/mpi_rack_hostfile), skips the head host (default the master
        # address), and for each other host starts, in the background over ssh as root, a remote
        # command that force-kills sglang and miles processes, stops ray, and starts a ray worker
        # joining master_address:6379 with the given GPU count and the host's own address; waits for
        # all to finish. Failures of individual ssh commands are not detected.
        """Join every host in an MPI-style hostfile to the ray cluster over ssh, in parallel.

        Ray itself cannot bring up the workers: the head is already running locally and the
        workers have no agent yet. Pass this as `execute_train(before_ray_job_submit=...)` so
        the cluster is complete before the job is submitted.
        """
        head_host = head_host or master_addr
        self.exec_command_cpu(
            f"for worker_ip in $(awk '{{print $1}}' {hostfile}); do "
            f'if [ "$worker_ip" = {shlex.quote(head_host)} ]; then continue; fi; '
            'echo "Starting Ray worker on $worker_ip"; '
            'ssh root@"$worker_ip" '
            '"pkill -9 sglang ; ray stop --force ; pkill -9 miles ; '
            f"ray start --address={master_addr}:6379 --num-gpus {num_gpus_per_node} "
            '--node-ip-address $worker_ip --disable-usage-stats" & '
            "done; wait"
        )

    def hf_download_dataset(self, full_name: str, data_dir: str = "/root/datasets"):
        # [FUNCTION DOC] Downloads a dataset repository named "owner/name" with the hf command line
        # tool into data_dir/name (default data_dir /root/datasets), via the CPU command method. A
        # name without exactly one slash raises a value error from unpacking.
        _, partial_name = full_name.split("/")
        self.exec_command_cpu(f"hf download --repo-type dataset {full_name} --local-dir {data_dir}/{partial_name}")

    def fp8_cast_bf16(self, path_src, path_dst):
        # [FUNCTION DOC] Casts an fp8 HuggingFace checkpoint to bf16, cached. Under an exclusive
        # lock on the destination, skips if the destination already has the safetensors index file;
        # otherwise runs the repository's cast tool with the source and destination via the GPU
        # command method.
        with exclusive_path_lock(path_dst):
            sentinel = Path(path_dst) / "model.safetensors.index.json"
            if sentinel.exists():
                logger.info(f"fp8_cast_bf16 skip {path_dst} since {sentinel} exists")
                return

            self.exec_command_gpu(
                f"python {repo_base_dir}/tools/fp8_cast_bf16.py "
                f"--input-fp8-hf-path {path_src} "
                f"--output-bf16-hf-path {path_dst} "
            )

    def api_server_host(self, config: ExecuteTrainConfig) -> str:
        # [FUNCTION DOC] Host name where the run's api server can be reached; the default is
        # "localhost" and ignores the config. Backends override.
        return "localhost"

    @abstractmethod
    # [FUNCTION DOC] Abstract hook (comment placed above because the body is on the signature line):
    # performs the actual training launch for a validated request and its config; returns nothing.
    def _execute_train_inner(self, *, request: ExecuteTrainRequest, config: ExecuteTrainConfig) -> None: ...

    def exec_command_cpu(self, cmd: str, capture_output: bool = False) -> str | None:
        # [FUNCTION DOC] Runs a shell command that needs no GPU; delegates to the inner hook and
        # returns its result (captured output text if requested, else none).
        return self._exec_command_cpu_inner(cmd, capture_output=capture_output)

    def exec_command_gpu(
        self, cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None
    ) -> str | None:
        # [FUNCTION DOC] Runs a shell command on one node with GPUs; delegates to the inner hook
        # with the capture flag and optional GPUs per node; returns the captured output or none.
        return self._exec_command_gpu_inner(cmd, capture_output=capture_output, num_gpus_per_node=num_gpus_per_node)

    def exec_command_multi_node(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        # [FUNCTION DOC] Runs a shell command on several nodes; delegates to the inner hook with
        # the capture flag, node count and GPUs per node; returns one entry per node (captured
        # output or none).
        return self._exec_command_multi_node_inner(
            cmd, capture_output=capture_output, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node
        )

    def _exec_command_cpu_inner(self, cmd: str, capture_output: bool = False) -> str | None:
        # [FUNCTION DOC] HELPER: Default implementation: runs the command locally through the
        # shared shell runner (bash, failing on non-zero exit) and returns its captured stdout if
        # requested.
        return run_shell_command(cmd, capture_output=capture_output)

    @abstractmethod
    # [FUNCTION DOC] HELPER: Abstract hook (comment placed above because the body is on the
    # signature line): runs a command on a GPU node; returns captured output if requested.
    def _exec_command_gpu_inner(
        self, cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None
    ) -> str | None: ...

    @abstractmethod
    # [FUNCTION DOC] HELPER: Abstract hook (comment placed above because the body is on the
    # signature line): runs a command on several nodes; returns a list with one captured output
    # (or none) per node.
    def _exec_command_multi_node_inner(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]: ...


def resolve_extra_env_vars(extra_env_vars: dict[str, str], config: ExecuteTrainConfig) -> dict[str, str]:
    # [FUNCTION DOC] Merges the explicit mapping with the variables parsed from the config's
    # extra-env string; the config's entries win on conflict. Returns a new mapping.
    return {
        **extra_env_vars,
        **_parse_extra_env_vars(config.extra_env_vars),
    }


@contextmanager
def exclusive_path_lock(path_dst: str):
    # [FUNCTION DOC] Context manager holding an exclusive advisory file lock on a sibling file
    # named after the destination with ".lock" appended (parent directories are created, the file
    # is created/truncated). Blocks until the lock is available; released when the file closes at
    # exit. Works between processes on the same host (or shared file system with lock support).
    """Serialize prepare steps racing on a cache dir shared between runners on one host."""
    path = Path(path_dst)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def resolve_hardware(config: ExecuteTrainConfig) -> str:
    # [FUNCTION DOC] Reads the config's hardware field: "auto" runs the hardware detector and
    # prints the result to standard error; anything else is used as given. Asserts the result is
    # one of the allowed values declared in the type annotation of that field. Returns it.
    # KNOWN ISSUE: the training config class itself has no hardware field; this only works for
    # subclasses that declare one (otherwise attribute or key errors).
    """`auto` asks the node the launcher runs on; anything explicit overrides it."""
    if config.hardware == "auto":
        hardware = detect_hardware()
        print(f"detected --hardware {hardware}", file=sys.stderr, flush=True)
    else:
        hardware = config.hardware
    supported = get_args(config.__dataclass_fields__["hardware"].type)
    assert hardware in supported, f"{type(config).__name__} has no verified profile for {hardware}"
    return hardware


def _assert_train_args_name_no_other_deploy_component(train_argv: list[str], *, deploy_component: str) -> None:
    # [FUNCTION DOC] HELPER: Asserts that every deploy-component flag value in the train argv
    # equals the launch's component (repeating the same value is fine).
    conflicting = sorted(set(ArgvManipulator.get(train_argv, _DEPLOY_COMPONENT_FLAG)) - {deploy_component})
    assert not conflicting, (
        f"This launch deploys the {deploy_component} part of the run, and everything it installs is named after "
        f"that, so its pods cannot be told {_DEPLOY_COMPONENT_FLAG} {conflicting}; set "
        f"ExecuteTrainConfig.deploy_component instead of naming it in the train arguments"
    )


def _assert_train_args_name_no_other_run_uuid(train_argv: list[str], *, run_uuid: str) -> None:
    # [FUNCTION DOC] HELPER: Asserts that every run-uuid flag value in the train argv equals the
    # launch's run uuid.
    conflicting = sorted(set(ArgvManipulator.get(train_argv, _RUN_UUID_FLAG)) - {run_uuid})
    assert (
        not conflicting
    ), f"the train arguments already say {_RUN_UUID_FLAG} {conflicting}, but this launch drives run {run_uuid}"


def _declared_cluster_backends(train_argv: list[str]) -> list[str]:
    # [FUNCTION DOC] HELPER: All values given for the cluster-backend flag in the argv, in order.
    return ArgvManipulator.get(train_argv, CLUSTER_BACKEND_FLAG)


def _assert_train_args_name_no_other_backend(train_argv: list[str], *, cluster_backend: str) -> None:
    # [FUNCTION DOC] HELPER: Asserts that every cluster-backend flag value in the train argv equals
    # the backend the launch targets.
    conflicting = sorted(set(_declared_cluster_backends(train_argv)) - {cluster_backend})
    assert not conflicting, (
        f"This run is launched onto {cluster_backend}, so its pods have to be told {CLUSTER_BACKEND_FLAG} "
        f"{cluster_backend}, but the train args already say {conflicting}; drop that flag or launch with the "
        f"backend it names"
    )
