# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

# [MODULE DOC] Registry of FT test modes (topologies). Defines model constants (a 5-layer
# Qwen3-30B-A3B MoE slice, the full Qwen3-30B-A3B, and the small dense Qwen3-0.6B, each with its
# Hugging Face repo and Megatron model type), the Hugging Face repo of recorded debug rollout
# data, the frozen mode description, the table of named modes, and the naming scheme that
# derives a mode name from its fields. Modes are 1-node (8 GPU) fake-rollout or real-rollout
# variants, a colocated rollout-kill variant, and one 6-node (48 GPU) disaggregated variant.

import shlex
from dataclasses import dataclass

import typer

MODEL_NAME: str = "Qwen3-30B-A3B-5layer"
MODEL_HF_REPO: str = f"fzyzcjy/{MODEL_NAME}"
MODEL_TYPE: str = "qwen3-30B-A3B-5layer"
DEBUG_ROLLOUT_DATA_HF_REPO: str = "fzyzcjy/miles-test-rollout-Qwen3-30B-A3B-5layer"

FULL_MODEL_NAME: str = "Qwen3-30B-A3B"
FULL_MODEL_HF_REPO: str = f"Qwen/{FULL_MODEL_NAME}"
FULL_MODEL_TYPE: str = "qwen3-30B-A3B"

# Small real dense model for with_failure's real_rollout mode (see README mode table).
DENSE_MODEL_NAME: str = "Qwen3-0.6B"
DENSE_MODEL_HF_REPO: str = f"Qwen/{DENSE_MODEL_NAME}"
DENSE_MODEL_TYPE: str = "qwen3-0.6B"


@dataclass(frozen=True)
class FTTestMode:
    # [CLASS DOC] Immutable description of one FT test topology. Fields: model name, Hugging Face
    # repo, Megatron model type, number of trainer cells (data-parallel replicas), parallelism
    # argument text, training nodes (default 1), training GPUs per node (default 8), number of
    # rollout engines (default 0, meaning replay of recorded rollout data instead of real
    # engines), GPUs per rollout engine (default 0), whether rollout colocates on the training
    # GPUs (default false), FT components enabled (default only train), and number of training
    # steps (default 10).
    model_name: str
    model_hf_repo: str
    megatron_model_type: str
    num_cells: int
    parallel_args: str
    train_num_nodes: int = 1
    train_gpus_per_node: int = 8
    rollout_num_engines: int = 0
    rollout_gpus_per_engine: int = 0
    colocate: bool = False
    ft_components: tuple[str, ...] = ("train",)
    num_steps: int = 10

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validates the mode: raises an assertion error if rollout FT is enabled
        # without real rollout engines, or if a colocated mode needs more rollout GPUs than the
        # node's training GPUs.
        assert "rollout" not in self.ft_components or self.has_real_rollout, (
            f"Mode declares ft components {self.ft_components} but has no real rollout engines, so a rollout "
            f"injection would have nothing to crash and the soak would silently prove nothing about rollout ft"
        )
        assert not self.colocate or self.total_rollout_gpus <= self.train_gpus_per_node, (
            f"Colocated mode oversubscribes its node: {self.total_rollout_gpus} rollout gpus "
            f"do not fit in {self.train_gpus_per_node} train gpus"
        )

    @property
    def has_real_rollout(self) -> bool:
        # [FUNCTION DOC] True when at least one rollout engine is configured.
        return self.rollout_num_engines > 0

    @property
    def total_rollout_gpus(self) -> int:
        # [FUNCTION DOC] Number of rollout engines times GPUs per engine.
        return self.rollout_num_engines * self.rollout_gpus_per_engine

    @property
    def total_node_gpus(self) -> int:
        # [FUNCTION DOC] GPUs the cluster needs per node: the training GPUs per node when colocated,
        # otherwise training GPUs per node plus all rollout GPUs.
        # KNOWN ISSUE: adds total (not per-node) rollout GPUs to a per-node figure, which is only
        # right for single-node training topologies.
        if self.colocate:
            return self.train_gpus_per_node
        return self.train_gpus_per_node + self.total_rollout_gpus


KILL_SEGMENT_OF_FT_COMPONENTS: dict[tuple[str, ...], str] = {
    ("train",): "kill_train",
    ("rollout",): "kill_rollout",
    ("train", "rollout"): "kill_train_rollout",
}

MODEL_SEGMENT_OF_MODEL_NAME: dict[str, str | None] = {
    DENSE_MODEL_NAME: None,
    MODEL_NAME: "moe_5layer",
    FULL_MODEL_NAME: "moe_full",
}

PARALLELISM_SEGMENT_OF_FLAG: dict[str, str] = {
    "--context-parallel-size": "cp",
    "--tensor-model-parallel-size": "tp",
    "--pipeline-model-parallel-size": "pp",
    "--expert-model-parallel-size": "ep",
    "--expert-tensor-parallel-size": "etp",
}


def compute_mode_name(mode: FTTestMode) -> str:
    # [FUNCTION DOC] Derives the canonical mode name used as the key in the mode table and in CI
    # file names. Joins with double underscores: the kill segment for the FT components (train,
    # rollout, or both; raises a key error for other combinations), the parallelism segment, a
    # fake-rollout marker when there are no real engines, the model segment (none for the dense
    # model, moe_5layer or moe_full for the MoE models; key error for other models), and a
    # colocate marker when colocated.
    segments: list[str] = [KILL_SEGMENT_OF_FT_COMPONENTS[mode.ft_components], _compute_parallelism_segment(mode)]

    if not mode.has_real_rollout:
        segments.append("fake_rollout")
    if (model_segment := MODEL_SEGMENT_OF_MODEL_NAME[mode.model_name]) is not None:
        segments.append(model_segment)
    if mode.colocate:
        segments.append("colocate")

    return "__".join(segments)


def _compute_parallelism_segment(mode: FTTestMode) -> str:
    # [FUNCTION DOC] HELPER: Builds the parallelism part of the name: "dp" plus the cell count,
    # followed (in fixed order: context, tensor, pipeline, expert, expert-tensor) by the short
    # name and size of each parallel-size flag found in the mode's argument text, joined by
    # underscores. Flags are read by splitting the argument text shell-style and pairing each
    # token with the next one.
    tokens: list[str] = shlex.split(mode.parallel_args)
    size_of_flag: dict[str, str] = dict(zip(tokens, tokens[1:], strict=False))

    parts: list[str] = [f"dp{mode.num_cells}"]
    parts += [
        f"{short}{size_of_flag[flag]}" for flag, short in PARALLELISM_SEGMENT_OF_FLAG.items() if flag in size_of_flag
    ]

    return "_".join(parts)


MODES: dict[str, FTTestMode] = {
    # --- 1-node (8 GPUs) variants ---
    "kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer": FTTestMode(
        model_name=MODEL_NAME,
        model_hf_repo=MODEL_HF_REPO,
        megatron_model_type=MODEL_TYPE,
        num_cells=2,
        parallel_args=(
            "--tensor-model-parallel-size 2 "
            "--context-parallel-size 2 "
            "--expert-model-parallel-size 2 "
            "--sequence-parallel"
        ),
    ),
    "kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer": FTTestMode(
        model_name=MODEL_NAME,
        model_hf_repo=MODEL_HF_REPO,
        megatron_model_type=MODEL_TYPE,
        num_cells=2,
        parallel_args=(
            "--pipeline-model-parallel-size 2 "
            "--context-parallel-size 2 "
            "--decoder-first-pipeline-num-layers 3 "
            "--decoder-last-pipeline-num-layers 2"
        ),
    ),
    "kill_train__dp4_cp2__fake_rollout__moe_5layer": FTTestMode(
        model_name=MODEL_NAME,
        model_hf_repo=MODEL_HF_REPO,
        megatron_model_type=MODEL_TYPE,
        num_cells=4,
        parallel_args="--context-parallel-size 2",
    ),
    "kill_train__dp2_cp2__moe_5layer": FTTestMode(
        model_name=MODEL_NAME,
        model_hf_repo=MODEL_HF_REPO,
        megatron_model_type=MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        parallel_args="--context-parallel-size 2",
    ),
    # Same topology as kill_train__dp2_cp2__moe_5layer but a small real dense model (see README).
    "kill_train__dp2_cp2": FTTestMode(
        model_name=DENSE_MODEL_NAME,
        model_hf_repo=DENSE_MODEL_HF_REPO,
        megatron_model_type=DENSE_MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        parallel_args="--context-parallel-size 2",
    ),
    # Same topology again, with ft on both kinds so one run crashes trainer cells and engines.
    "kill_train_rollout__dp2_cp2": FTTestMode(
        model_name=DENSE_MODEL_NAME,
        model_hf_repo=DENSE_MODEL_HF_REPO,
        megatron_model_type=DENSE_MODEL_TYPE,
        num_cells=2,
        train_gpus_per_node=4,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        ft_components=("train", "rollout"),
        parallel_args="--context-parallel-size 2",
    ),
    # --- 1-node (8 GPUs) colocated: engines share the trainer's gpus ---
    "kill_rollout__dp4__colocate": FTTestMode(
        model_name=DENSE_MODEL_NAME,
        model_hf_repo=DENSE_MODEL_HF_REPO,
        megatron_model_type=DENSE_MODEL_TYPE,
        num_cells=4,
        train_gpus_per_node=4,
        rollout_num_engines=4,
        rollout_gpus_per_engine=1,
        colocate=True,
        ft_components=("rollout",),
        parallel_args="",
    ),
    # --- 6-node (48 GPUs) disaggregated: 4 train nodes + 2 rollout nodes ---
    "kill_train__dp4_cp2_tp2_pp2_ep2_etp2__moe_full": FTTestMode(
        model_name=FULL_MODEL_NAME,
        model_hf_repo=FULL_MODEL_HF_REPO,
        megatron_model_type=FULL_MODEL_TYPE,
        num_cells=4,
        train_num_nodes=4,
        train_gpus_per_node=8,
        rollout_num_engines=2,
        rollout_gpus_per_engine=8,
        parallel_args=(
            "--tensor-model-parallel-size 2 "
            "--context-parallel-size 2 "
            "--pipeline-model-parallel-size 2 "
            "--expert-model-parallel-size 2 "
            "--expert-tensor-parallel-size 2 "
            "--sequence-parallel"
        ),
    ),
}


def resolve_mode(mode: str | None) -> FTTestMode:
    # [FUNCTION DOC] Looks up a mode by name in the mode table. Raises a typer bad-parameter error
    # (listing valid names) when the name is none or unknown.
    if mode is None:
        raise typer.BadParameter(
            f"this suite takes the topology it runs from --mode, and none was passed; pass one of "
            f"{list(MODES.keys())}"
        )
    if mode not in MODES:
        raise typer.BadParameter(f"Unknown mode {mode!r}, valid: {list(MODES.keys())}")
    return MODES[mode]
