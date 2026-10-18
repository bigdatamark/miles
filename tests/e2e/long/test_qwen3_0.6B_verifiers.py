# [MODULE DOC] End-to-end test of the experimental "verifiers" rollout integration with
# Qwen3-0.6B on 2 colocated GPUs, running a single tiny rollout of the code-golf environment
# (CUDA: 2-GPU H200 stage, estimated 300 seconds, short label, hopper and blackwell; ROCm:
# nightly 2-GPU MI350 suite, estimated 900 seconds). Paths come from environment variables with
# defaults: model directory, Megatron path, and a run directory holding a dedicated virtual
# environment for the verifiers package and the environment checkout. prepare builds that
# environment and installs the environment package, execute trains for one rollout and then
# verifies the dumped rollout data, and verify holds the assertions. Run as a script: prepare,
# strip all proxy environment variables (both cases), execute.

import math
import os
import shutil
import site
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=300, suite="stage-c-2-gpu-h200", labels=["short"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=900, suite="nightly-stage-c-2-gpu-mi350", labels=["short"])

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 2
MODEL_DIR = Path(os.environ.get("MILES_E2E_MODEL_DIR", "/root/models"))
MEGATRON_PATH = Path(os.environ.get("MILES_E2E_MEGATRON_PATH", "/root/Megatron-LM"))
RUN_DIR = Path(os.environ.get("MILES_E2E_RUN_DIR", "/tmp/miles-verifiers-e2e"))
VERIFIERS_DIR = Path("/tmp/verifiers-v0.2.0")
ADAPTER_DIR = Path(command_utils.repo_base_dir) / "examples" / "experimental" / "verifiers"
VERIFIERS_VENV = RUN_DIR / "verifiers-venv"
VERIFIERS_SITE_PACKAGES = (
    VERIFIERS_VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
)
CODE_GOLF_DIR = RUN_DIR / "environments" / "code_golf_v1"


def prepare():
    # [FUNCTION DOC] Sets up everything the run needs through the default backend. Steps: create
    # the model and run directories; download Qwen3-0.6B; create a fresh virtual environment (with
    # seed packages and system site packages, using the current interpreter) and write a path file
    # into its site-packages exposing the launcher's site-packages; install the verifiers adapter
    # requirements into that environment; install the pinned prime command-line tool (version
    # 0.6.19) with its sandboxes dependency capped below 0.3 (0.3.0 dropped a class the pinned
    # prime imports); shallow-clone the verifiers repository at tag v0.2.0 unless the clone
    # directory already exists; copy its code-golf environment folder into the run directory
    # (merging into an existing one); install that environment with the prime tool from the run
    # directory; convert the checkpoint to Megatron format on 2 GPUs. Side effects: network,
    # subprocesses, filesystem writes.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {RUN_DIR}")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.exec_command_cpu(f"uv venv --clear --seed --python {sys.executable} --system-site-packages {VERIFIERS_VENV}")
    (VERIFIERS_SITE_PACKAGES / "launcher-site.pth").write_text("\n".join(site.getsitepackages()) + "\n")
    U.exec_command_cpu(
        f"{VERIFIERS_VENV}/bin/python -m pip install "
        f"-r {command_utils.repo_base_dir}/examples/experimental/verifiers/requirements.txt"
    )
    # prime pins no upper bound on prime-sandboxes, and 0.3.0 dropped the
    # CommandRequest the pinned prime imports, so the tool env has to cap it.
    U.exec_command_cpu("uv tool install 'prime==0.6.19' --with 'prime-sandboxes<0.3'")
    if not VERIFIERS_DIR.exists():
        U.exec_command_cpu(
            f"git clone --depth 1 --branch v0.2.0 "
            f"https://github.com/PrimeIntellect-ai/verifiers.git {VERIFIERS_DIR}"
        )
    shutil.copytree(
        VERIFIERS_DIR / "environments" / "code_golf_v1",
        CODE_GOLF_DIR,
        dirs_exist_ok=True,
    )
    U.exec_command_cpu(f'cd {RUN_DIR} && "$(uv tool dir --bin)/prime" --plain env install code-golf-v1')
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst=str(MODEL_DIR),
        hf_checkpoint=str(MODEL_DIR / MODEL_NAME),
        megatron_path=str(MEGATRON_PATH),
    )


def execute():
    # [FUNCTION DOC] Writes a verifiers config file into the run directory (task set code-golf-v1,
    # 300 second rollout timeout), builds the training arguments, runs training, and then calls
    # verify on the dump directory. Arguments: the verifiers rollout function replaces the default
    # rollout; no global dataset; 1 rollout of 3 prompts times 4 samples (oversampling batch 3);
    # response length 512, context length 2048, temperature 0.8; global batch 12 with data
    # balancing; GRPO with clip 0.2/0.28; Adam learning rate 1e-6; no gradient accumulation
    # fusion; one GPU per rollout engine with static memory fraction 0.6; parallelism all 1;
    # dynamic batch size 4096 tokens per GPU; flash attention; one actor node of 2 GPUs,
    # colocated; detailed dumps into the run's dump directory. The run's environment sets the
    # legacy rollout v1 switch, info-level verifiers logging, a Python path combining the
    # verifiers environment, the environment folder, Megatron, the adapter and the repository,
    # and the config path.
    U = command_utils.default_config().create_backend()
    config_path = RUN_DIR / "code-golf.toml"
    config_path.write_text('[taskset]\nid = "code-golf-v1"\n\n[timeout]\nrollout = 300\n')
    dump_dir = RUN_DIR / "dump"

    train_args = " ".join(
        [
            f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}",
            "--sglang-tokenizer-path Qwen/Qwen3-0.6B",
            f"--ref-load {MODEL_DIR}/{MODEL_NAME}_torch_dist",
            "--rollout-function-path verifiers_rollout.generate_rollout",
            "--disable-rollout-global-dataset",
            "--num-rollout 1",
            "--rollout-batch-size 3",
            "--n-samples-per-prompt 4",
            "--over-sampling-batch-size 3",
            "--rollout-max-response-len 512",
            "--rollout-max-context-len 2048",
            "--rollout-temperature 0.8",
            "--global-batch-size 12",
            "--balance-data",
            "--advantage-estimator grpo",
            "--entropy-coef 0.0",
            "--eps-clip 0.2",
            "--eps-clip-high 0.28",
            "--optimizer adam",
            "--lr 1e-6",
            "--lr-decay-style constant",
            "--weight-decay 0.1",
            "--adam-beta1 0.9",
            "--adam-beta2 0.98",
            "--no-gradient-accumulation-fusion",
            "--rollout-num-gpus-per-engine 1",
            "--sglang-mem-fraction-static 0.6",
            "--sglang-enable-metrics",
            "--tensor-model-parallel-size 1",
            "--pipeline-model-parallel-size 1",
            "--context-parallel-size 1",
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu 4096",
            "--attention-backend flash",
            "--actor-num-nodes 1",
            f"--actor-num-gpus-per-node {NUM_GPUS}",
            "--colocate",
            f"--dump-details {dump_dir}",
            command_utils.get_default_wandb_args(__file__),
        ]
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            "MILES_USE_LEGACY_ROLLOUT_V1": "1",
            "VF_LOG_LEVEL": "INFO",
            "PYTHONPATH": (
                f"{VERIFIERS_SITE_PACKAGES}:{CODE_GOLF_DIR}:{MEGATRON_PATH}:{ADAPTER_DIR}:{command_utils.repo_base_dir}"
            ),
            "VERIFIERS_CONFIG": str(config_path),
        },
        megatron_path=str(MEGATRON_PATH),
    )
    verify(dump_dir)


def verify(dump_dir: Path):
    # [FUNCTION DOC] Loads the dumped samples of rollout 0 and asserts: exactly 12 samples, in 3
    # groups (indices 0-2) of 4 each; every status is completed or truncated and at least one is
    # completed; all rewards are finite; for each sample the rollout log-prob count, loss mask
    # length and response length are equal and all log-probs are finite; each sample's verifiers
    # metadata has exactly the rewards correct, fastest and most_concise, a passed metric of 0 or
    # 1, a finite latency metric, and no error entry; at least one sample passed; and within each
    # group the "fastest" rewards sum to about 0.5.
    samples = torch.load(dump_dir / "rollout_data" / "0.pt", weights_only=False)["samples"]

    assert len(samples) == 12
    assert Counter(sample["group_index"] for sample in samples) == {0: 4, 1: 4, 2: 4}
    assert {sample["status"] for sample in samples} <= {"completed", "truncated"}
    assert any(sample["status"] == "completed" for sample in samples)
    assert all(math.isfinite(sample["reward"]) for sample in samples)
    assert all(
        len(sample["rollout_log_probs"]) == len(sample["loss_mask"]) == sample["response_length"] for sample in samples
    )
    assert all(math.isfinite(value) for sample in samples for value in sample["rollout_log_probs"])

    for sample in samples:
        metadata = sample["metadata"]["verifiers"]
        assert set(metadata["rewards"]) == {"correct", "fastest", "most_concise"}
        assert metadata["metrics"]["passed"] in {0.0, 1.0}
        assert math.isfinite(metadata["metrics"]["latency"])
        assert "error" not in metadata

    assert any(sample["metadata"]["verifiers"]["metrics"]["passed"] for sample in samples)
    for group_index in range(3):
        group = [sample for sample in samples if sample["group_index"] == group_index]
        assert sum(sample["metadata"]["verifiers"]["rewards"]["fastest"] for sample in group) == pytest.approx(0.5)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
