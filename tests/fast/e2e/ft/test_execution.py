# [MODULE DOC]
# Tests for the argument builders in the ft execution module: the common training argument string
# (event-data dump flag, colocate flag, debug-train-only and rollout gpu flags depending on the
# mode) and the fault-tolerance argument string, using the registered named test modes.
import dataclasses
from pathlib import Path

from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args
from tests.e2e.ft.conftest_ft.modes import MODES


class TestGetCommonTrainArgs:
    # [CLASS DOC] Tests for the common training argument string builder.
    def test_fault_tolerance_runs_enable_the_events_the_checksums_are_written_into(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Builds args for the colocated rollout-kill mode and asserts they contain
        # the save-debug-event-data flag with the dump dir followed by a slash.
        """Inference-engine weight checksums are only recorded while the event logger is initialised."""
        args = get_common_train_args(MODES["kill_rollout__dp4__colocate"], dump_dir=str(tmp_path))

        assert f"--save-debug-event-data {tmp_path}/" in args

    def test_a_colocated_real_rollout_mode_emits_the_colocate_flag(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Asserts the colocated mode's args contain the colocate flag followed by a space.
        """A colocated mode must tell the trainer to share its gpus with the rollout engines."""
        args = get_common_train_args(MODES["kill_rollout__dp4__colocate"], dump_dir=str(tmp_path))

        assert "--colocate " in args

    def test_a_disaggregated_real_rollout_mode_omits_the_colocate_flag(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] For the trainer-kill mode, asserts the args contain the rollout gpu count
        # flag and no colocate flag.
        """Rollout engines on their own gpus must not be colocated with the trainer."""
        args = get_common_train_args(MODES["kill_train__dp2_cp2__moe_5layer"], dump_dir=str(tmp_path))

        assert "--rollout-num-gpus" in args
        assert "--colocate" not in args

    def test_a_debug_rollout_mode_omits_the_colocate_flag_even_when_the_mode_is_colocated(
        self, tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Copies the colocated mode with zero rollout engines and only the train ft
        # component; asserts the copy is still marked colocated, the args contain the debug
        # train-only flag, and no colocate flag is emitted.
        """Without real rollout engines there is nothing to colocate, whatever the mode declares."""
        mode = dataclasses.replace(
            MODES["kill_rollout__dp4__colocate"], rollout_num_engines=0, ft_components=("train",)
        )

        args = get_common_train_args(mode, dump_dir=str(tmp_path))

        assert mode.colocate is True
        assert "--debug-train-only" in args
        assert "--colocate" not in args


class TestGetFtArgs:
    # [CLASS DOC] Tests for the fault-tolerance argument string builder.
    def test_a_rollout_only_ft_mode_propagates_the_rollout_component_and_api_server_port(self) -> None:
        # [FUNCTION DOC] Asserts the exact string for the rollout-kill mode: use-fault-tolerance,
        # ft-components rollout, api-server-port 0, with a trailing space.
        """Rollout-only fault tolerance must not silently enable trainer fault tolerance."""
        args = get_ft_args(MODES["kill_rollout__dp4__colocate"])

        assert args == "--use-fault-tolerance --ft-components rollout --api-server-port 0 "

    def test_a_trainer_mode_propagates_the_train_component(self) -> None:
        # [FUNCTION DOC] Asserts the exact string for the trainer-kill mode, with ft-components train.
        """The trainer-fault-tolerance modes keep sending the train component."""
        args = get_ft_args(MODES["kill_train__dp2_cp2__moe_5layer"])

        assert args == "--use-fault-tolerance --ft-components train --api-server-port 0 "
