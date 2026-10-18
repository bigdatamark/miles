# [MODULE DOC]
# Tests of the deepseek v4 launcher script (its train entrypoint) using the python launch harness: the
# launcher is imported by path with a frozen environment and a command recorder, and the last recorded
# command (the training submission) is inspected. Checks parallelism splits between tensor and context
# parallel, fixed context-parallel recipes, missing recipes, and the rollout engine profile per
# hardware. Constants are override mappings for one node with four layers, 8 nodes of 4 gpus, 8 nodes
# of 8 gpus and 32 nodes of 8 gpus, all on H200.
import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


_SINGLE_NODE_4LAYER = {
    "hardware": "H200",
    "model_name": "DeepSeek-V4-Flash-FP8-4layer",
    "num_nodes": 1,
    "num_gpus_per_node": 4,
}
_EIGHT_NODES = {"hardware": "H200", "num_nodes": 8, "num_gpus_per_node": 4}
_EIGHT_NODES_OF_8 = {"hardware": "H200", "num_nodes": 8, "num_gpus_per_node": 8}
_THIRTY_TWO_NODES_OF_8 = {"hardware": "H200", "num_nodes": 32, "num_gpus_per_node": 8}


def _train_command(monkeypatch, tmp_path, overrides):
    # [FUNCTION DOC] HELPER: Freezes the environment, starts the command recorder, imports the deepseek
    # v4 launcher, calls its train entrypoint with the overrides (sandbox is the temp dir), and returns
    # the last recorded command string. Exceptions from the launcher propagate.
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    call_entrypoint(module, "train", overrides, sandbox=tmp_path)
    return recording.commands[-1]


@pytest.mark.parametrize(
    ("cp_size", "expected"),
    [
        (None, ["--tensor-model-parallel-size 4", "--sequence-parallel", "--context-parallel-size 1"]),
        (2, ["--tensor-model-parallel-size 2", "--sequence-parallel", "--context-parallel-size 2", "--allgather-cp"]),
        (4, ["--tensor-model-parallel-size 1", "--context-parallel-size 4", "--allgather-cp"]),
    ],
)
def test_single_node_miles_impl_splits_gpus_between_tp_and_cp(monkeypatch, tmp_path, cp_size, expected):
    # [FUNCTION DOC] Parametrized over context-parallel size none, 2 and 4. Runs the single-node
    # four-layer recipe with the miles implementation (and the cp size if given) and asserts each
    # expected flag substring is in the command: tensor parallel size is 4 divided by cp size, sequence
    # parallel appears only when tensor parallel exceeds 1, and the all-gather context parallel flag
    # appears exactly when tensor parallel is below 4.
    """DSV4 CP must use the all-gather split; TP takes the GPUs CP leaves, with SP only above TP1."""
    overrides = _SINGLE_NODE_4LAYER | {"dsv4_impl": "miles"} | ({"cp_size": cp_size} if cp_size else {})
    command = _train_command(monkeypatch, tmp_path, overrides)

    for flag in expected:
        assert flag in command
    tp_size = 4 // (cp_size or 1)
    assert ("--allgather-cp" in command) == (tp_size < 4)
    assert ("--sequence-parallel" in command) == (tp_size > 1)


# Every recipe that pins its CP size, with that size.
_FIXED_CP_RECIPES = [
    (_SINGLE_NODE_4LAYER | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES | {"dsv4_impl": "miles"}, 2),
    (_EIGHT_NODES_OF_8 | {"dsv4_impl": "megatron"}, 1),
    (_EIGHT_NODES_OF_8 | {"dsv4_impl": "miles"}, 1),
    (_THIRTY_TWO_NODES_OF_8 | {"dsv4_impl": "miles"}, 1),
]


@pytest.mark.parametrize(("recipe", "recipe_cp_size"), _FIXED_CP_RECIPES)
def test_recipes_with_a_fixed_cp_size_accept_it_and_reject_another(monkeypatch, tmp_path, recipe, recipe_cp_size):
    # [FUNCTION DOC] Parametrized over the fixed-cp recipes. Asserts passing the recipe's own cp size
    # yields a command with that context parallel size, and passing double that size raises a
    # not-implemented error whose message says it is untested here.
    command = _train_command(monkeypatch, tmp_path, recipe | {"cp_size": recipe_cp_size})
    assert f"--context-parallel-size {recipe_cp_size}" in command

    with pytest.raises(NotImplementedError, match="is untested here"):
        _train_command(monkeypatch, tmp_path, recipe | {"cp_size": recipe_cp_size * 2})


def test_a_node_count_without_a_recipe_reports_the_missing_recipe(monkeypatch, tmp_path):
    # [FUNCTION DOC] With 2 nodes of 8 gpus and cp size 2 (no preset), expects a not-implemented error
    # saying no pre-set parallel config exists.
    with pytest.raises(NotImplementedError, match="No pre-set parallel config"):
        _train_command(
            monkeypatch, tmp_path, {"hardware": "H200", "num_nodes": 2, "num_gpus_per_node": 8, "cp_size": 2}
        )


@pytest.mark.parametrize(
    ("overrides", "expected_size"),
    [
        ({"hardware": "H200", "num_nodes": 8, "num_gpus_per_node": 4}, 4),
        ({"hardware": "GB300", "num_nodes": 8, "num_gpus_per_node": 4}, 8),
        (
            {
                "hardware": "H200",
                "model_name": "DeepSeek-V4-Flash-FP8-4layer",
                "num_nodes": 1,
                "num_gpus_per_node": 4,
            },
            4,
        ),
        (
            {
                "hardware": "GB300",
                "model_name": "DeepSeek-V4-Flash-FP8-4layer",
                "num_nodes": 1,
                "num_gpus_per_node": 4,
            },
            4,
        ),
    ],
)
def test_the_rollout_profile_follows_the_hardware(monkeypatch, tmp_path, overrides, expected_size):
    # [FUNCTION DOC] Parametrized over four configs (H200 8x4 gives 4, GB300 8x4 gives 8, and the
    # four-layer single node on H200 or GB300 gives 4). Runs train and asserts the last command has the
    # expected value for the gpus per engine flag and the sglang tensor and expert parallel size flags.
    # The flags are matched as substrings, so e.g. a value of 4 would also match 40.
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")

    call_entrypoint(module, "train", overrides, sandbox=tmp_path)

    train_command = recording.commands[-1]
    assert f"--rollout-num-gpus-per-engine {expected_size}" in train_command
    assert f"--sglang-tp-size {expected_size}" in train_command
    assert f"--sglang-ep-size {expected_size}" in train_command
