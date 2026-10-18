# [MODULE DOC] End-to-end precision test: checks that training on a fixed set of rollout samples
# gives equivalent results whether sequences are fed in batch-sequence-head-dim layout (one
# sequence per micro-batch) or in packed token-head-dim layout (all samples packed in one
# micro-batch), for a 4-layer FP8 DeepSeek-V4-Flash slice on 4 H200 GPUs through the DeepSeek V4
# launcher script (its argument class, download/convert steps, train step and precision config are
# imported). Flow: prepare the checkpoint once under a file lock; run a rollout-only job that
# dumps 16 samples (8 prompts times 2 samples) using a custom deterministic reward, then validate
# the dump; run two training jobs (bshd and thd) that each replay that dumped rollout and save the
# gradient norm and dumped train data; finally compare per-token log probabilities and the
# loss, PPO KL and grad norm metrics between the two layouts within tolerances. Registered for
# CUDA 4-GPU H200 (est. 1100 s, Hopper only, labels precision and long). Module constants: model
# name, GPU count, prompt and sample counts, rollout identifiers, the import path of the custom
# reward, and log-prob tolerance thresholds (relative 0.2, absolute 0.9, mean absolute 0.1, 99th
# percentile absolute 0.9, max absolute 1.5). Run as a script: calls main.
import fcntl
import math
import os
import tempfile
from argparse import Namespace
from pathlib import Path

import torch
from scripts.run_deepseek_v4 import (
    _DSV4_TE_PRECISION_CONFIG,
    ScriptArgs,
    _prepare_download,
    _prepare_single,
    _prepare_spmd,
    _train,
)
from tests.ci.ci_register import register_cuda_ci

from miles.utils.test_utils.comparisons.metrics import compare_metrics
from miles.utils.types import Sample

register_cuda_ci(
    est_time=1100,
    suite="stage-c-4-gpu-h200",
    labels=["precision", "long"],
    hardware=["hopper"],
)

_MODEL_NAME = "DeepSeek-V4-Flash-FP8-4layer"
_NUM_GPUS = 4
_NUM_PROMPTS = 8
_SAMPLES_PER_PROMPT = 2
_NUM_SAMPLES = _NUM_PROMPTS * _SAMPLES_PER_PROMPT
_ROLLOUT_RUN_ID = "rollout"
_ROLLOUT_ID = "0"
_CUSTOM_RM_PATH = "tests.e2e.precision.test_dsv4_bshd_thd_rollout_parity.deterministic_index_reward"

_LOG_PROB_RTOL = 0.2
_LOG_PROB_ATOL = 0.9
_LOG_PROB_MEAN_ABS = 0.1
_LOG_PROB_P99_ABS = 0.9
_LOG_PROB_MAX_ABS = 1.5


async def deterministic_index_reward(
    args: Namespace,
    samples: Sample | list[Sample],
    **_kwargs,
) -> float | list[float]:
    # [FUNCTION DOC] Custom asynchronous reward function plugged into the rollout by import path.
    # Gives a reward that depends only on the sample's global index, making rewards deterministic
    # and distinct within a group: the index modulo samples per prompt, as a float. Accepts either
    # one sample (returns one float) or a list of samples (returns a list of floats in order).
    # Extra keyword arguments are ignored. Raises an assertion error if a sample has no index.
    def score(sample: Sample) -> float:
        # [FUNCTION DOC] HELPER: Asserts the sample has an index and returns that index modulo the
        # samples-per-prompt count from the enclosing arguments, as a float.
        assert sample.index is not None
        return float(sample.index % args.n_samples_per_prompt)

    if isinstance(samples, list):
        return [score(sample) for sample in samples]
    return score(samples)


def _prepare_args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds the launcher argument object used only for the preparation
    # phase: run id "prepare", the 4-layer model, the in-repo implementation, the GSM8K task, eval
    # off, one node with 4 GPUs, H200 hardware, saving skipped and fault tolerance off.
    return ScriptArgs(
        run_id="prepare",
        model_name=_MODEL_NAME,
        dsv4_impl="miles",
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=_NUM_GPUS,
        hardware="H200",
        skip_saving=True,
        use_fault_tolerance=False,
    )


def _common_extra_args(*, te_precision_config_path: Path, micro_batch_size: int) -> str:
    # [FUNCTION DOC] HELPER: Returns the extra command-line text shared by the rollout and train
    # jobs: CI test mode with quantization error allowed in the weight check and the logprob
    # checker disabled; experiment tracking disabled; one rollout of 8 prompts times 2 samples
    # with global batch equal to the sample count (16) and the given micro-batch size; 256 max
    # response tokens at temperature 0.7; rollout seed 42, global seed 1234; decode CUDA-graph
    # max batch 4; and the path of the transformer-engine precision config file.
    return (
        "--ci-test "
        "--check-weight-update-allow-quant-error "
        "--ci-disable-logprobs-checker "
        "--wandb-mode disabled "
        "--num-rollout 1 "
        f"--rollout-batch-size {_NUM_PROMPTS} "
        f"--n-samples-per-prompt {_SAMPLES_PER_PROMPT} "
        f"--global-batch-size {_NUM_SAMPLES} "
        f"--micro-batch-size {micro_batch_size} "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.7 "
        "--rollout-seed 42 "
        "--sglang-cuda-graph-max-bs-decode 4 "
        "--seed 1234 "
        f"--te-precision-config-file {te_precision_config_path} "
    )


def _run_args(
    *,
    run_id: str,
    debug_root: Path,
    hf_checkpoint: str,
    extra_args: str,
    debug_train: bool,
) -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds the launcher argument object for a rollout or train job with
    # the given run id, debug data root directory, Hugging Face checkpoint path and extra argument
    # text. Fixed settings: 4-layer model, in-repo implementation, GSM8K, eval off, 1 node with 4
    # GPUs, H200, saving skipped, fault tolerance off, detail dumping on. When debug_train is true
    # the job is told to replay the earlier rollout run id and rollout id from the debug root;
    # otherwise those two replay settings are unset.
    return ScriptArgs(
        run_id=run_id,
        model_name=_MODEL_NAME,
        dsv4_impl="miles",
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=_NUM_GPUS,
        hardware="H200",
        hf_checkpoint=hf_checkpoint,
        skip_saving=True,
        use_fault_tolerance=False,
        dump_details=True,
        debug_data_root=str(debug_root),
        debug_train_run_id=_ROLLOUT_RUN_ID if debug_train else None,
        debug_train_rollout_id=_ROLLOUT_ID if debug_train else None,
        extra_args=extra_args,
    )


def _prepare(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Prepares the model once per host. Creates the model directory if
    # needed, opens a per-model lock file inside it in append mode, takes an exclusive file lock
    # (blocking, so concurrent runs serialize), and while holding it runs the launcher's download,
    # single-process conversion and SPMD conversion steps in that order; the lock is released when
    # the file closes. Returns the path of the converted model under the local model directory.
    # Side effects: filesystem writes, network download, GPU conversion work.
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir / f".{args.model_name}.ci-prepare.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        _prepare_download(args)
        _prepare_single(args)
        _prepare_spmd(args)

    return f"{args.model_local_dir}/{args.model_name}"


def _assert_rollout_artifact(path: Path) -> None:
    # [FUNCTION DOC] HELPER: Loads the dumped rollout file onto the CPU (full unpickling allowed)
    # and validates it with assertions. Rollout id must be 0. Samples sorted by index must have
    # indices 0 through 15 and group indices that repeat each prompt number twice in order. For
    # each consecutive pair (one prompt): same group, rewards exactly 0.0 then 1.0 (from the
    # deterministic reward), response lengths in 1 through 256, and the two responses (the last
    # response-length tokens) must differ. Raises assertion errors on any violation.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["rollout_id"] == 0
    samples = sorted(payload["samples"], key=lambda sample: sample["index"])
    assert [sample["index"] for sample in samples] == list(range(_NUM_SAMPLES))
    assert [sample["group_index"] for sample in samples] == [
        group_index for group_index in range(_NUM_PROMPTS) for _ in range(_SAMPLES_PER_PROMPT)
    ]

    for start in range(0, _NUM_SAMPLES, _SAMPLES_PER_PROMPT):
        pair = samples[start : start + _SAMPLES_PER_PROMPT]
        assert pair[0]["group_index"] == pair[1]["group_index"]
        assert [sample["reward"] for sample in pair] == [0.0, 1.0]
        assert all(0 < sample["response_length"] <= 256 for sample in pair)
        responses = [tuple(sample["tokens"][-sample["response_length"] :]) for sample in pair]
        assert responses[0] != responses[1]


def _run_rollout(*, debug_root: Path, hf_checkpoint: str, te_precision_config_path: Path) -> None:
    # [FUNCTION DOC] HELPER: Runs the rollout-only job and checks its dump. Builds arguments with
    # run id "rollout", the common extra arguments with micro-batch size equal to the sample count,
    # plus the debug rollout-only flag, SGLang server concurrency 1 (for determinism), and the
    # custom reward import path. Calls the launcher's train step, then asserts on the rollout data
    # file at the debug root under the rollout run, dump details, rollout data, named by rollout id.
    args = _run_args(
        run_id=_ROLLOUT_RUN_ID,
        debug_root=debug_root,
        hf_checkpoint=hf_checkpoint,
        debug_train=False,
        extra_args=(
            _common_extra_args(
                te_precision_config_path=te_precision_config_path,
                micro_batch_size=_NUM_SAMPLES,
            )
            + "--debug-rollout-only "
            + "--sglang-server-concurrency 1 "
            + f"--custom-rm-path {_CUSTOM_RM_PATH} "
        ),
    )
    _train(args)
    rollout_path = debug_root / _ROLLOUT_RUN_ID / "dump_details" / "rollout_data" / f"{_ROLLOUT_ID}.pt"
    _assert_rollout_artifact(rollout_path)


def _run_train(
    *,
    qkv_format: str,
    debug_root: Path,
    hf_checkpoint: str,
    te_precision_config_path: Path,
) -> Path:
    # [FUNCTION DOC] HELPER: Runs one training job replaying the dumped rollout in the given
    # query-key-value layout (bshd or thd) and returns the directory holding its dumped details.
    # Micro-batch size is 1 for bshd (one sequence per micro-batch, the reference) and the full
    # sample count for thd (everything packed). Passes the layout flag and a path for saving the
    # gradient norm under the debug root. After the job, loads the saved gradient norm as a float
    # and asserts it is finite and positive. Returns the dump-details directory of that run id.
    grad_norm_path = debug_root / f"{qkv_format}_grad_norm.pt"
    # BSHD uses one sequence per microbatch as the reference; THD packs all samples.
    micro_batch_size = 1 if qkv_format == "bshd" else _NUM_SAMPLES
    args = _run_args(
        run_id=qkv_format,
        debug_root=debug_root,
        hf_checkpoint=hf_checkpoint,
        debug_train=True,
        extra_args=(
            _common_extra_args(
                te_precision_config_path=te_precision_config_path,
                micro_batch_size=micro_batch_size,
            )
            + f"--qkv-format {qkv_format} "
            + f"--ci-save-grad-norm {grad_norm_path} "
        ),
    )
    _train(args)
    grad_norm = float(torch.load(grad_norm_path, map_location="cpu", weights_only=False))
    assert math.isfinite(grad_norm) and grad_norm > 0
    return debug_root / qkv_format / "dump_details"


def _load_train_data_shard(directory: Path) -> dict:
    # [FUNCTION DOC] HELPER: Loads the single dumped train-data file for rollout id 0 from the
    # directory (files are written per data-parallel/context-parallel shard; this run has one of
    # each). Asserts exactly one matching file, rollout id 0, and sample indices equal to 0
    # through 15 in order. Returns the rollout data mapping from the payload.
    # Train data is dumped once per (dp, cp) shard; this run has dp=1, cp=1.
    dump_files = sorted(directory.glob(f"{_ROLLOUT_ID}_*.pt"))
    assert len(dump_files) == 1, [path.name for path in dump_files]

    payload = torch.load(dump_files[0], map_location="cpu", weights_only=False)
    assert payload["rollout_id"] == 0
    rollout_data = payload["rollout_data"]
    assert rollout_data["sample_indices"] == list(range(_NUM_SAMPLES))
    return rollout_data


def _compare_train_log_probs(bshd_dir: Path, thd_dir: Path) -> None:
    # [FUNCTION DOC] HELPER: Compares per-token training log probabilities between the two layouts
    # using the dumped train data in the two directories. For every sample asserts equal response
    # lengths and exactly equal tokens and loss masks, then asserts each side's log probs are
    # one-dimensional, floating point, of length equal to the response length, and finite. All
    # samples are concatenated as double precision, absolute differences computed, and the mean,
    # 99th percentile and max printed. Asserts closeness within relative 0.2 and absolute 0.9,
    # mean absolute difference at most 0.1, 99th percentile at most 0.9 and max at most 1.5.
    bshd = _load_train_data_shard(bshd_dir)
    thd = _load_train_data_shard(thd_dir)
    bshd_log_probs: list[torch.Tensor] = []
    thd_log_probs: list[torch.Tensor] = []

    for sample_index in range(_NUM_SAMPLES):
        assert bshd["response_lengths"][sample_index] == thd["response_lengths"][sample_index]
        response_length = bshd["response_lengths"][sample_index]
        for field in ("tokens", "loss_masks"):
            torch.testing.assert_close(
                bshd[field][sample_index],
                thd[field][sample_index],
                rtol=0,
                atol=0,
            )

        baseline = bshd["log_probs"][sample_index]
        target = thd["log_probs"][sample_index]
        for log_probs in (baseline, target):
            assert log_probs.ndim == 1
            assert log_probs.is_floating_point()
            assert log_probs.numel() == response_length
            assert torch.isfinite(log_probs).all()
        bshd_log_probs.append(baseline)
        thd_log_probs.append(target)

    baseline = torch.cat(bshd_log_probs).double()
    target = torch.cat(thd_log_probs).double()
    diffs = (baseline - target).abs()
    mean_abs = float(diffs.mean())
    p99_abs = float(torch.quantile(diffs, 0.99))
    max_abs = float(diffs.max())
    print(
        f"Train log-prob comparison: count={diffs.numel()}, "
        f"mean_abs={mean_abs:.8e}, p99_abs={p99_abs:.8e}, max_abs={max_abs:.8e}"
    )

    torch.testing.assert_close(baseline, target, rtol=_LOG_PROB_RTOL, atol=_LOG_PROB_ATOL)
    assert mean_abs <= _LOG_PROB_MEAN_ABS
    assert p99_abs <= _LOG_PROB_P99_ABS
    assert max_abs <= _LOG_PROB_MAX_ABS


def _compare(bshd_dump: Path, thd_dump: Path) -> None:
    # [FUNCTION DOC] HELPER: Runs the full comparison between the two dump directories: first the
    # per-token log-prob comparison on their train-data subdirectories, then the shared metric
    # comparison helper (baseline is the bshd run) with relative tolerance 0.1 and absolute 0.03
    # over metrics whose names start with train loss, train PPO KL, or train grad norm, excluding
    # none. Raises assertion errors from either step.
    _compare_train_log_probs(
        bshd_dump / "train_data",
        thd_dump / "train_data",
    )
    compare_metrics(
        baseline_dir=str(bshd_dump),
        target_dir=str(thd_dump),
        rtol=0.1,
        atol=0.03,
        key_prefixes=["train/loss", "train/ppo_kl", "train/grad_norm"],
        exclude_keys=[],
    )


def main() -> None:
    # [FUNCTION DOC] Orchestrates the whole test. Prepares the model (under the file lock), then
    # removes proxy variables and the experiment-tracking API key from the environment. Creates a
    # temporary debug directory under the system temp directory (never cleaned up) and writes the
    # imported precision config text plus a newline to a YAML file there. Runs the rollout-only
    # job, then the bshd training job, then the thd training job, and finally compares the two
    # dumps. Side effects: environment mutation, filesystem writes, subprocess/GPU jobs.
    prepare_args = _prepare_args()
    hf_checkpoint = _prepare(prepare_args)
    for env_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "WANDB_API_KEY"):
        os.environ.pop(env_var, None)

    debug_root = Path(tempfile.mkdtemp(prefix="miles-dsv4-bshd-thd-", dir="/tmp"))
    te_precision_config_path = debug_root / "te_precision.yaml"
    te_precision_config_path.write_text(f"{_DSV4_TE_PRECISION_CONFIG}\n", encoding="utf-8")
    _run_rollout(
        debug_root=debug_root,
        hf_checkpoint=hf_checkpoint,
        te_precision_config_path=te_precision_config_path,
    )
    bshd_dump = _run_train(
        qkv_format="bshd",
        debug_root=debug_root,
        hf_checkpoint=hf_checkpoint,
        te_precision_config_path=te_precision_config_path,
    )
    thd_dump = _run_train(
        qkv_format="thd",
        debug_root=debug_root,
        hf_checkpoint=hf_checkpoint,
        te_precision_config_path=te_precision_config_path,
    )
    _compare(bshd_dump, thd_dump)


if __name__ == "__main__":
    main()
