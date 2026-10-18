# [MODULE DOC] HELPER: Shared harness for the torchtitan training-backend end-to-end tests in this
# directory (one file per case, each differing only in topology and model). Provides a validated
# case-configuration dataclass, a prepare function (downloads), a pure function building the whole
# command line for a case, and an execute function that strips proxy variables and launches
# training through the legacy command utilities module (used as a module alias). Parallel flags use
# Megatron's names; the backend copies them into the torchtitan parallelism config.
# KNOWN ISSUE: the existing docstring says context parallelism is told to be 1 to the shared
# training loop, but the command line passes the case's context-parallel size and the
# divisibility check includes it.
"""Shared case harness for the torchtitan backend's end-to-end tests.

One file per case, all of them differing only in a topology and a model, so the
command line is built once here. The parallelism flags are the ones Megatron
takes; the backend copies them into torchtitan's ``Parallelism`` config.

Context parallelism is deliberately not a field of the topology check below in
the way the others are: it is internal to the trainer, which shards the sequence
and gathers the logits back before the loss sees them, so the shared training
loop is told cp=1 either way.
"""

import os
from dataclasses import dataclass

import miles.utils.external_utils.command_utils.legacy as U


@dataclass
class CaseConfig:
    # [CLASS DOC] HELPER: Mutable data holder for one case. Required: Hugging Face model repository,
    # torchtitan model name and flavor, total GPUs for training, sequence length, max response
    # length. Optional with defaults: tensor, pipeline, context and expert parallel sizes (1);
    # rollout routing replay (off); a reference model with KL loss (off); colocation (on); rollout
    # GPU count and GPUs per engine (unset); fully-async (off); SGLang static memory fraction 0.7;
    # weight transfer mode (unset); number of rollouts (2); extra argument text. Exposes the
    # derived engine size and local model directory as properties.
    """One end-to-end configuration.

    Topology values are passed explicitly rather than inferred, so a case file
    reads as the configuration it is testing.
    """

    model_repo: str
    titan_model_name: str
    titan_model_flavor: str
    num_gpus: int
    seq_len: int
    max_response_len: int
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    use_r3: bool = False
    with_ref: bool = False
    colocate: bool = True
    rollout_num_gpus: int | None = None
    rollout_num_gpus_per_engine: int | None = None
    fully_async: bool = False
    mem_fraction_static: float = 0.7
    transfer_mode: str | None = None
    num_rollout: int = 2
    extra_args: str = ""

    def __post_init__(self):
        # [FUNCTION DOC] HELPER: Validation only; raises a value error if: fully-async is combined
        # with colocation; not colocated and no rollout GPU count; total GPUs is not divisible by
        # tensor times pipeline times context size; the sequence length does not exceed the max
        # response length (a prompt needs room, because torchtitan sizes its rotary tables from the
        # sequence length); or the rollout pool (total GPUs if colocated, else the rollout GPU
        # count) is not divisible by the engine size.
        if self.fully_async and self.colocate:
            raise ValueError(
                "fully_async cannot colocate: generation continues while training runs, "
                "so the two need separate GPUs"
            )
        if not self.colocate and self.rollout_num_gpus is None:
            raise ValueError("rollout_num_gpus must be set when colocate is False")
        if self.num_gpus % (self.tp_size * self.pp_size * self.cp_size) != 0:
            raise ValueError(
                "num_gpus must be divisible by tp * pp * cp: "
                f"{self.num_gpus=} {self.tp_size=} {self.pp_size=} {self.cp_size=}"
            )
        if self.seq_len <= self.max_response_len:
            raise ValueError(
                "seq_len must leave room for a prompt ahead of max_response_len: torchtitan "
                f"sizes its rotary tables from it ({self.seq_len=} {self.max_response_len=})"
            )
        rollout_pool = self.num_gpus if self.colocate else self.rollout_num_gpus
        if rollout_pool % self.engine_size != 0:
            raise ValueError(f"rollout pool {rollout_pool} is not divisible by the engine size {self.engine_size}")

    @property
    def engine_size(self) -> int:
        # [FUNCTION DOC] HELPER: Property. GPUs per rollout engine: the explicit value if set,
        # otherwise the whole rollout pool (all GPUs when colocated, else the rollout GPU count),
        # i.e. one engine spanning the pool.
        if self.rollout_num_gpus_per_engine is not None:
            return self.rollout_num_gpus_per_engine
        return self.num_gpus if self.colocate else self.rollout_num_gpus

    @property
    def model_dir(self) -> str:
        # [FUNCTION DOC] HELPER: Property. Local model path: the models root plus the last path
        # component of the repository name.
        return f"/root/models/{self.model_repo.split('/')[-1]}"


def prepare(case: CaseConfig) -> None:
    # [FUNCTION DOC] Creates model/dataset directories, downloads the case's model repository into
    # its local model directory and downloads the dapo-math-17k dataset. No checkpoint conversion
    # (torchtitan loads Hugging Face weights). Side effects: shell commands, network, filesystem.
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download {case.model_repo} --local-dir {case.model_dir}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")


def build_train_args(case: CaseConfig, *, wandb_file: str) -> str:
    # [FUNCTION DOC] Builds the full space-separated argument string; reads the infinite-run
    # environment switch and calls the wandb-args helper keyed on the given file. Sections:
    # rollout (dapo-math data, deepscaler reward, 3000 rollouts if the infinite-run switch is on
    # else the case's count, 8 prompts times 4 samples, case's max response length, temperature 1,
    # global batch 32, balanced; for fully-async adds the fully-async flag and in-place pause mode
    # because the default retract mode can deadlock cache flushing); GRPO with zero-coefficient KL
    # and entropy terms and clip range 0.2 to 0.28, plus rollout routing replay if enabled; Adam at
    # 1e-6; SGLang (engine size, decode log interval 1000, case's memory fraction); torchtitan
    # (backend selection, model name and flavor, sequence length, the four parallel sizes, micro
    # batch size 1, gradient checkpointing, 512 MiB weight update buffer); CI test with the KL
    # checker disabled; misc (HF checkpoint dir, one actor node with the case's GPUs, colocate or
    # explicit rollout GPUs, reference load plus KL loss if a reference is wanted, the transfer mode
    # if set, and for disk-delta a published directory and a local checkpoint directory under a
    # fixed root). The case's extra args end the string. Returns it.
    """The whole command line for `case`.

    Separate from `execute` so a CPU-only test can read the string back without
    standing up a job.
    """
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {3000 if U.get_env_enable_infinite_run() else case.num_rollout} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        f"--rollout-max-response-len {case.max_response_len} "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
    )
    if case.fully_async:
        # retract, the default, can deadlock flush_cache under load in
        # fully_async, which is why the shipped recipes pin in_place.
        rollout_args += "--fully-async --pause-generation-mode in_place "

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    if case.use_r3:
        grpo_args += "--use-rollout-routing-replay "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {case.engine_size} "
        "--sglang-decode-log-interval 1000 "
        f"--sglang-mem-fraction-static {case.mem_fraction_static} "
    )

    titan_args = (
        "--train-backend torchtitan "
        f"--titan-model-name {case.titan_model_name} "
        f"--titan-model-flavor {case.titan_model_flavor} "
        f"--seq-length {case.seq_len} "
        f"--tensor-model-parallel-size {case.tp_size} "
        f"--pipeline-model-parallel-size {case.pp_size} "
        f"--context-parallel-size {case.cp_size} "
        f"--expert-model-parallel-size {case.ep_size} "
        "--micro-batch-size 1 "
        "--gradient-checkpointing "
        f"--update-weight-buffer-size {512 * 1024 * 1024} "
    )

    misc_args = (
        f"--hf-checkpoint {case.model_dir} " "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {case.num_gpus} "
    )
    misc_args += "--colocate " if case.colocate else f"--rollout-num-gpus {case.rollout_num_gpus} "
    if case.with_ref:
        misc_args += f"--ref-load {case.model_dir} --use-kl-loss "
    if case.transfer_mode is not None:
        misc_args += f"--update-weight-transfer-mode {case.transfer_mode} "
    if case.transfer_mode == "disk-delta":
        # The delta stream is published to a directory both sides can see, and
        # each rollout host patches its own copy of the checkpoint in place.
        misc_args += (
            "--update-weight-disk-dir /root/titan_delta/published "
            "--update-weight-local-checkpoint-dir /root/titan_delta/local "
        )

    ci_args = "--ci-test --ci-disable-kl-checker "

    return (
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(wandb_file)} "
        f"{titan_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{case.extra_args} "
    )


def execute(case: CaseConfig, *, wandb_file: str) -> None:
    # [FUNCTION DOC] Removes proxy variables from the process environment, then launches training
    # with the built arguments, no Megatron model type, GPUs equal to the case's GPUs plus the
    # rollout GPUs when not colocated, and the async train script when fully-async else the
    # standard one. Returns nothing.
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    U.execute_train(
        train_args=build_train_args(case, wandb_file=wandb_file),
        num_gpus_per_node=case.num_gpus + (0 if case.colocate else case.rollout_num_gpus),
        megatron_model_type=None,
        train_script="train_async.py" if case.fully_async else "train.py",
    )
