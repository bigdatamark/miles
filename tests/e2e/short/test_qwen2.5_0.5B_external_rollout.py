# [MODULE DOC] End-to-end test of training against external (not launched by the trainer) SGLang
# rollout engines, for Qwen2.5-0.5B-Instruct: 2 training GPUs (bridge mode, a Megatron checkpoint
# is still converted in prepare) and 2 external engines of 1 GPU each, 4 GPUs in total. The cluster
# backend decides how the engines exist. On Ray (single host) the engines are launched as
# background subprocesses by a shell command run before training on the trainer node, pinned to
# the last two visible devices, listening on loopback ports 32001 and 32002 and waited on until
# their server-info endpoint responds. On Kubernetes a headless service plus a stateful set of 2
# engine pods is rendered from the run's infra values and installed alongside the run, addressed
# by stable pod DNS names, and the Mooncake object store arguments are added. Rollout: GSM8K, math
# reward, 3 rollouts of 8 prompts times 4 samples with oversampling 16 and a non-zero reward std
# filter, global batch 32. Registered for CUDA 4-GPU H200 (est. 300 s) and ROCm 4-GPU MI350
# nightly (est. 700 s). Module constants: model name/type, GPU counts, Ray engine host and ports,
# and Kubernetes object, port, container and health-path names. Run as a script: prepare, strip
# proxy variables, execute.
import json
import os
import shlex
import textwrap
from dataclasses import dataclass
from typing import Any

import yaml
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import chart_dir, repo_base_dir
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    InfraValues,
    VolumeEntry,
    VolumeMount,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import InfraInfo
from miles.utils.workers.types import ClusterBackend

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-h200", labels=["short"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=700, suite="nightly-stage-c-4-gpu-mi350", labels=["short"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_TRAIN_GPUS = 2
NUM_ENGINES = 2
GPUS_PER_ENGINE = 1

RAY_ENGINE_HOST = "127.0.0.1"
RAY_ENGINE_PORTS = [32001, 32002]

ENGINE_OBJECT_NAME = "external-sglang"
ENGINE_PORT = 30000
ENGINE_CONTAINER_NAME = "sglang"
ENGINE_HEALTH_PATH = "/health_generate"


# ===== Entry points =====


def compute_train_and_engine_devices(visible_devices: str | None) -> tuple[list[str], list[str]]:
    # [FUNCTION DOC] Splits the visible-device list (a comma-separated string, possibly empty or
    # none) into training devices and engine devices. If empty, uses the device indices 0 through
    # 3 as text. Raises an assertion error if the count is not 4 (2 training plus 2 engines).
    # Returns the first 2 as training devices and the remaining 2 as engine devices, in order.
    # Entries are not trimmed of whitespace.
    devices = (visible_devices or "").split(",") if visible_devices else []
    if not devices:
        devices = [str(i) for i in range(NUM_TRAIN_GPUS + NUM_ENGINES)]
    assert len(devices) == NUM_TRAIN_GPUS + NUM_ENGINES, (
        f"this test trains on {NUM_TRAIN_GPUS} gpus and pins one engine to each of {NUM_ENGINES} more, "
        f"but the runner offered {devices}"
    )
    return devices[:NUM_TRAIN_GPUS], devices[NUM_TRAIN_GPUS:]


def prepare():
    # [FUNCTION DOC] Creates model/dataset directories, downloads Qwen2.5-0.5B-Instruct and GSM8K,
    # and converts the model to a Megatron distributed checkpoint using the training GPU count (2).
    # Side effects: shell commands, network, filesystem, GPU.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_TRAIN_GPUS)


def execute():
    # [FUNCTION DOC] Resolves the external engines for the active cluster backend from the
    # backend's config, builds training arguments with the engine addresses and the
    # backend-specific object store arguments, and launches training on the 2 training GPUs,
    # passing the per-role pre-run commands and extra manifests from the engines description.
    # Returns nothing.
    U = command_utils.default_config().create_backend()
    engines = _external_engines(U.config)

    U.execute_train(
        train_args=_train_args(engines.addrs, object_store_args=_object_store_args(U.config)),
        num_gpus_per_node=NUM_TRAIN_GPUS,
        megatron_model_type=MODEL_TYPE,
        prepare_cmd=engines.prepare_cmd,
        extra_manifests=engines.extra_manifests,
    )


@dataclass(frozen=True)
class _ExternalEngines:
    # [CLASS DOC] HELPER: Immutable description of how the external engines exist for a backend:
    # their addresses as host-and-port strings, a mapping from role name to a shell command to run
    # before training on that role (empty for Kubernetes), and a list of extra manifest texts to
    # install alongside the run (empty for Ray).
    addrs: list[str]
    prepare_cmd: dict[str, str]
    extra_manifests: list[str]


def _external_engines(config: command_utils.ExecuteTrainConfig) -> _ExternalEngines:
    # [FUNCTION DOC] HELPER: Dispatches on the config's cluster backend: Ray gives engines started
    # beside the trainer, Kubernetes gives engines installed beside the run.
    # KNOWN ISSUE: there is no default branch, so any other backend value makes the function
    # return nothing instead of raising.
    match config.cluster_backend:
        case ClusterBackend.RAY:
            return _engines_started_beside_trainer()
        case ClusterBackend.KUBERNETES:
            return _engines_installed_beside_run(config)


def _object_store_args(config: command_utils.ExecuteTrainConfig) -> str:
    # [FUNCTION DOC] HELPER: Returns an empty string for the Ray backend and the Mooncake object
    # store argument text for the Kubernetes backend.
    # KNOWN ISSUE: no default branch, so an unknown backend returns nothing instead of a string.
    match config.cluster_backend:
        case ClusterBackend.RAY:
            return ""
        case ClusterBackend.KUBERNETES:
            return command_utils.get_mooncake_object_store_args()


# ===== Training arguments =====


def _train_args(engine_addrs: list[str], *, object_store_args: str) -> str:
    # [FUNCTION DOC] HELPER: Builds the training argument string. Checkpoint: Hugging Face
    # directory plus the converted Megatron directory as reference load. Rollout: GSM8K parquet,
    # math reward, 3 rollouts of 8 prompts times 4 samples, 1024 tokens, temperature 0.8,
    # oversampling batch 16 with the non-zero-reward-std dynamic filter, global batch 32. All
    # parallel sizes 1 with sequence parallel, dynamic batching at 9216 tokens per GPU. GRPO with
    # KL loss at zero coefficient, Adam at 1e-6. External engines: the space-joined engine
    # addresses, total rollout GPUs (engines times GPUs per engine) and GPUs per engine. CI test
    # flag; no colocation flag; 2 actor GPUs; bridge mode. The object store args are inserted after
    # the checkpoint args. Returns the concatenated string.
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 16 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 32 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    external_args = (
        "--rollout-external-engine-addrs "
        + " ".join(engine_addrs)
        + f" --rollout-num-gpus {NUM_ENGINES * GPUS_PER_ENGINE} "
        + f"--rollout-num-gpus-per-engine {GPUS_PER_ENGINE} "
    )

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
        "--megatron-to-hf-mode bridge "
    )

    return (
        f"{ckpt_args} "
        f"{object_store_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{external_args} "
        f"{ci_args} "
        f"{misc_args} "
    )


# ===== Engines the run talks to but does not launch =====


def _engines_started_beside_trainer() -> _ExternalEngines:
    # [FUNCTION DOC] HELPER: Ray-backend engines. Reads the visible-device environment variable
    # (CUDA; on ROCm, as selected by the hardware-platform environment variable, the HIP variable
    # takes precedence when set). Splits devices into training and engine devices (the training
    # list is unused here; it may raise an assertion error on a wrong device count). Returns engine
    # addresses on loopback with the two fixed ports, a pre-run command for the "trainer" role that
    # launches the engines on the engine devices, and no extra manifests.
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if os.environ.get("MILES_HARDWARE_PLATFORM") == "rocm":
        visible_devices = os.environ.get("HIP_VISIBLE_DEVICES") or visible_devices
    _train_devices, engine_devices = compute_train_and_engine_devices(visible_devices)
    return _ExternalEngines(
        addrs=[f"{RAY_ENGINE_HOST}:{port}" for port in RAY_ENGINE_PORTS],
        prepare_cmd={"trainer": _ray_engines_launch_cmd(engine_devices)},
        extra_manifests=[],
    )


def _engines_installed_beside_run(config: command_utils.ExecuteTrainConfig) -> _ExternalEngines:
    # [FUNCTION DOC] HELPER: Kubernetes-backend engines. The object name is the base name plus the
    # config's run id. Addresses are the stable per-pod DNS names (object name, pod index, then the
    # service name) with the engine port, one per engine. No pre-run commands. One extra manifest
    # rendered from the infra values loaded from the config's helm value files.
    object_name = f"{ENGINE_OBJECT_NAME}-{config.run_id}"
    return _ExternalEngines(
        addrs=[f"{object_name}-{index}.{object_name}:{ENGINE_PORT}" for index in range(NUM_ENGINES)],
        prepare_cmd={},
        extra_manifests=[_engine_manifests(_infra_values(config.helm_values), object_name=object_name)],
    )


def _ray_engines_launch_cmd(engine_devices: list[str]) -> str:
    # [FUNCTION DOC] HELPER: Builds a shell command that starts one engine per device and waits for
    # them. Launch part: for each device and fixed port (paired strictly, so a length mismatch
    # raises a value error) a backgrounded subshell sets the visible-devices variable (HIP on ROCm,
    # else CUDA) to the shell-quoted device, starts the engine detached in a new session, and
    # redirects stdout and stderr to a log file named from the port under the temp directory. Wait
    # part: up to 120 rounds of 5 seconds, each running a silent fail-on-error HTTP request to every
    # engine's server-info endpoint chained with logical-and; on success it stops waiting; if never
    # successful it prints a failure message and the last 100 lines of the logs to standard error
    # and exits with status 1. Returns the launch part, a space, and the wait part.
    device_variable = (
        "HIP_VISIBLE_DEVICES" if os.environ.get("MILES_HARDWARE_PLATFORM") == "rocm" else "CUDA_VISIBLE_DEVICES"
    )
    launches = " ".join(
        f"({device_variable}={shlex.quote(device)} setsid {shlex.join(_engine_argv(port))} "
        f"> /tmp/miles_external_engine_{port}.log 2>&1 &);"
        for device, port in zip(engine_devices, RAY_ENGINE_PORTS, strict=True)
    )
    probes = " && ".join(
        f"curl -sf http://{RAY_ENGINE_HOST}:{port}/server_info >/dev/null" for port in RAY_ENGINE_PORTS
    )
    wait = (
        f"ok=0; for _ in $(seq 1 120); do {probes} && ok=1 && break; sleep 5; done; "
        f"[ \"$ok\" -eq 1 ] || {{ echo 'external sglang engines failed to start' >&2; "
        f"tail -n 100 /tmp/miles_external_engine_*.log >&2; exit 1; }}"
    )
    return f"{launches} {wait}"


def _engine_argv(port: int) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the argument vector that launches an SGLang server: the
    # python3 interpreter running the SGLang server module with the local model path, host
    # 0.0.0.0, the given port, tensor parallel size equal to GPUs per engine, static memory
    # fraction 0.7, and remote code trusted. Used for both Ray subprocesses and the Kubernetes
    # container command.
    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        f"/root/models/{MODEL_NAME}",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp",
        str(GPUS_PER_ENGINE),
        "--mem-fraction-static",
        "0.7",
        "--trust-remote-code",
    ]


# ===== Kubernetes manifests for those engines =====


def _engine_manifests(infra: InfraValues, *, object_name: str) -> str:
    # [FUNCTION DOC] HELPER: Renders the Kubernetes YAML text (two documents) for the engines.
    # Document one: a headless service named by the object name selecting pods labeled with it and
    # exposing the engine port. Document two: a stateful set with one replica per engine, parallel
    # pod management, bound to that service. Pod spec fields taken from the infra values and
    # emitted only when non-empty: image pull secrets, node selector, tolerations and affinity
    # (from scheduling, when present) and volumes. The single container uses the infra image
    # repository and tag, runs the engine argument vector (as a JSON list, which is valid YAML) on
    # the engine port, requests GPUs equal to GPUs per engine through the NVIDIA resource key, and
    # has an HTTP readiness probe on the health path every 10 seconds with 60 allowed failures.
    # Further container fields emitted when non-empty: image pull policy, environment variables
    # sorted by name, and volume mounts flattened across volumes. Returns the text.
    # KNOWN ISSUE: the GPU resource key is hard-coded to the NVIDIA name, so the manifests cannot
    # request AMD GPUs even though a ROCm registration exists.
    scheduling = infra.scheduling

    pod_fields: dict[str, Any] = {
        "imagePullSecrets": [dict(name=secret) for secret in infra.image.pull_secrets or []],
        "nodeSelector": scheduling.node_selector if scheduling is not None else None,
        "tolerations": scheduling.tolerations if scheduling is not None else None,
        "affinity": scheduling.affinity if scheduling is not None else None,
        "volumes": [_volume(volume) for volume in infra.volumes],
    }
    container_fields: dict[str, Any] = {
        "imagePullPolicy": infra.image.pull_policy,
        "env": [dict(name=name, value=value) for name, value in sorted((infra.env or {}).items())],
        "volumeMounts": [_volume_mount(volume, mount) for volume in infra.volumes for mount in volume.mounts],
    }

    return f"""\
apiVersion: v1
kind: Service
metadata:
  name: {object_name}
spec:
  clusterIP: "None"
  selector:
    app: {object_name}
  ports:
    - name: http
      port: {ENGINE_PORT}
      targetPort: {ENGINE_PORT}
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {object_name}
spec:
  replicas: {NUM_ENGINES}
  serviceName: {object_name}
  podManagementPolicy: Parallel
  selector:
    matchLabels:
      app: {object_name}
  template:
    metadata:
      labels:
        app: {object_name}
    spec:
{_yaml_block(pod_fields, indent=6)}\
      containers:
        - name: {ENGINE_CONTAINER_NAME}
          image: {infra.image.repository}:{infra.image.tag}
          command: {json.dumps(_engine_argv(ENGINE_PORT))}
          ports:
            - name: http
              containerPort: {ENGINE_PORT}
          resources:
            limits:
              nvidia.com/gpu: {GPUS_PER_ENGINE}
          readinessProbe:
            httpGet:
              path: {ENGINE_HEALTH_PATH}
              port: {ENGINE_PORT}
            periodSeconds: 10
            failureThreshold: 60
{_yaml_block(container_fields, indent=10)}"""


def _yaml_block(fields: dict[str, Any], *, indent: int) -> str:
    # [FUNCTION DOC] HELPER: Drops entries with falsy values; if none remain returns an empty
    # string. Otherwise dumps the remaining mapping to YAML preserving key order and indents every
    # line by the given number of spaces (the result ends with a newline).
    present = {key: value for key, value in fields.items() if value}
    if not present:
        return ""

    return textwrap.indent(yaml.safe_dump(present, sort_keys=False), " " * indent)


def _volume(volume: VolumeEntry) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Converts a volume entry to a pod volume mapping: a host path volume
    # (path, with type defaulting to "Directory"), else a persistent volume claim reference, else
    # an empty-directory volume from its values form. Raises an assertion error if none of the
    # three is set.
    if (host_path := volume.host_path) is not None:
        return dict(name=volume.name, hostPath=dict(path=host_path.path, type=host_path.type or "Directory"))
    if (claim := volume.persistent_volume_claim) is not None:
        return dict(name=volume.name, persistentVolumeClaim=dict(claimName=claim.claim_name))

    assert volume.empty_dir is not None
    return dict(name=volume.name, emptyDir=volume.empty_dir.as_values())


def _volume_mount(volume: VolumeEntry, mount: VolumeMount) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds a container volume-mount mapping from the volume's name and the
    # mount's path, adding the sub path when set and the read-only flag only when it is true.
    fields = dict(name=volume.name, mountPath=mount.mount_path)
    if mount.sub_path is not None:
        fields["subPath"] = mount.sub_path
    if mount.read_only:
        fields["readOnly"] = mount.read_only
    return fields


def _infra_values(helm_values: tuple[str, ...]) -> InfraValues:
    # [FUNCTION DOC] HELPER: Loads the infra values for the repository's chart directory merged
    # with the given helm values files (given as a tuple, passed on as a list) through the infra
    # info loader, and returns them.
    return InfraInfo.load(chart_dir(repo_base_dir=repo_base_dir), list(helm_values))


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
