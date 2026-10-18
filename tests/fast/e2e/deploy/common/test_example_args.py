# [MODULE DOC]
# Tests for the deploy example argument helpers: the check that an example's training arguments
# build the same parallelism flags the test mode declares (matching values, differing values,
# absent flags, and value-less flags), and that the deterministic test argument builder also
# enables the debug event data output that checksums are written into.
import pytest
from tests.e2e.deploy.conftest_deploy.common.example_args import (
    assert_example_parallelism_matches,
    build_deterministic_test_args,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode


def _mode(parallel_args: str) -> FTTestMode:
    # [FUNCTION DOC] HELPER: Builds a fault-tolerance test mode for model Qwen3-0.6B (hub repo
    # Qwen/Qwen3-0.6B, megatron type qwen3-0.6B) with 2 cells, 4 train gpus per node and the given
    # parallelism argument string.
    return FTTestMode(
        model_name="Qwen3-0.6B",
        model_hf_repo="Qwen/Qwen3-0.6B",
        megatron_model_type="qwen3-0.6B",
        num_cells=2,
        train_gpus_per_node=4,
        parallel_args=parallel_args,
    )


class TestAssertTheExampleBuildsTheParallelismOf:
    # [CLASS DOC] Tests of the parallelism-matching assertion between a mode and an example's
    # training argument string.
    def test_an_example_building_the_parallelism_the_mode_declares_passes(self):
        # [FUNCTION DOC] A mode declaring context parallel size 2 and training arguments with the
        # same flag and value passes without raising.
        """The happy path has to stay reachable, or the refusals below prove nothing."""
        assert_example_parallelism_matches(
            _mode("--context-parallel-size 2"), train_args="--lr 1e-6 --context-parallel-size 2 "
        )

    def test_an_example_building_another_degree_than_the_mode_declares_is_caught(self):
        # [FUNCTION DOC] Mode says context parallel size 2 but the example builds 4: raises an
        # assertion error matching "builds --context-parallel-size as".
        """A mode describing a topology nobody launched is a comparison against a different run."""
        with pytest.raises(AssertionError, match="builds --context-parallel-size as"):
            assert_example_parallelism_matches(
                _mode("--context-parallel-size 2"), train_args="--lr 1e-6 --context-parallel-size 4 "
            )

    def test_an_example_that_never_builds_the_flag_at_all_is_caught(self):
        # [FUNCTION DOC] Example arguments lack the flag the mode declares: raises an assertion
        # error matching "builds no --context-parallel-size at all".
        """The mode's whole point is the parallelism it names, and an example ignoring it runs another shape."""
        with pytest.raises(AssertionError, match="builds no --context-parallel-size at all"):
            assert_example_parallelism_matches(_mode("--context-parallel-size 2"), train_args="--lr 1e-6 ")

    def test_a_flag_the_mode_declares_without_a_value_is_checked_for_being_built_at_all(self):
        # [FUNCTION DOC] A value-less flag (sequence parallel) declared by the mode and present in
        # the example arguments (in any position) passes; only presence is checked for it.
        """--sequence-parallel takes no value, so the flag being there is the whole of what it declares."""
        assert_example_parallelism_matches(
            _mode("--sequence-parallel --context-parallel-size 2"),
            train_args="--sequence-parallel --lr 1e-6 --context-parallel-size 2 ",
        )

    def test_a_flag_the_mode_declares_without_a_value_and_the_example_omits_is_caught(self):
        # [FUNCTION DOC] The mode declares the value-less sequence parallel flag but the example
        # omits it: raises an assertion error matching "builds no --sequence-parallel at all".
        """A run built without it parallelises differently, however little the flag itself says."""
        with pytest.raises(AssertionError, match="builds no --sequence-parallel at all"):
            assert_example_parallelism_matches(
                _mode("--sequence-parallel --context-parallel-size 2"),
                train_args="--lr 1e-6 --context-parallel-size 2 ",
            )


class TestBuildDeterministicTestArgs:
    # [CLASS DOC] Tests of the deterministic-run argument builder.
    def test_the_events_the_checksums_are_written_into_are_also_enabled(self):
        # [FUNCTION DOC] Building arguments for a dump directory "/dumps/run" with the dumper off
        # still includes the debug-event-data save flag pointing at that directory with a trailing
        # slash (the checksum emitter does nothing unless the event logger is initialised by it).
        """The checksum emitter is a no-op unless the event logger is initialised from --save-debug-event-data."""
        args = build_deterministic_test_args(
            mode=_mode("--context-parallel-size 2"), dump_dir="/dumps/run", enable_dumper=False
        )

        assert "--save-debug-event-data /dumps/run/" in args
