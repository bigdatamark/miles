# [MODULE DOC]
# Tests for the dumper utilities module: wrapping a forward step so the dumper advances its step
# counter, the environment settings handed to the inference engine, the Megatron dumper helper
# (per-rollout directory layout, rollout-zero parent wipe, finalize behaviour including pinning
# model dumps to step zero and dumping distributed-optimizer bucket gradients), the post-cleanup
# cross-cell barrier and its structured log records, best-effort dump directory cleanup gated on
# ranks, and configuring all registered inference engines over HTTP. Parallel state, the process
# group module, the HTTP post and the router worker listing are mocked where needed.
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from miles.utils import dumper_utils
from miles.utils.dumper_utils import DumperMegatronUtil, DumperPhase


class TestWrapForwardStepWithStepping:
    # [CLASS DOC] Tests of the wrapper that adds dumper stepping around a forward-step callable.

    @pytest.fixture()
    def setup(self):
        # [FUNCTION DOC] HELPER: Fixture returning a triple of an inner mock forward step (which
        # returns a fixed output/loss-function pair), the wrapped version of it, and a mock dumper
        # object to be patched in by tests.
        inner = MagicMock(return_value=("output", "loss_fn"))
        wrapped = dumper_utils._wrap_forward_step_with_stepping(inner)
        mock_dumper = MagicMock()
        return inner, wrapped, mock_dumper

    @pytest.mark.parametrize(("n_calls", "expected_steps"), [(1, 0), (2, 1), (5, 4)])
    def test_step_called_n_minus_1_times(self, setup, n_calls: int, expected_steps: int) -> None:
        # [FUNCTION DOC] Parametrized over call counts 1, 2 and 5. Call the wrapped step that many
        # times with the dumper patched; assert the dumper step method ran one fewer time (zero,
        # one, four), i.e. the first call does not advance the step.
        _inner, wrapped, mock_dumper = setup
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            for _ in range(n_calls):
                wrapped("iter", "model")
        assert mock_dumper.step.call_count == expected_steps

    def test_passes_args_and_returns_result(self, setup) -> None:
        # [FUNCTION DOC] Call the wrapped step once with positional and keyword arguments; assert
        # the inner step received them unchanged exactly once and its return value is returned.
        inner, wrapped, mock_dumper = setup
        with patch("miles.utils.dumper_utils.dumper", mock_dumper):
            result = wrapped("my_iter", "my_model", extra=True)
        inner.assert_called_once_with("my_iter", "my_model", extra=True)
        assert result == ("output", "loss_fn")


def test_sglang_env_includes_startup_dumper_settings() -> None:
    # [FUNCTION DOC] With the inference phase enabled via override strings and a source patcher
    # config path, assert the engine environment mapping holds exactly: server port set to reuse,
    # the non-intrusive mode value, and the patcher config path.
    args = SimpleNamespace(
        dumper_enable=False,
        dumper_inference=["enable=true", "non_intrusive_mode=all"],
        dumper_source_patcher_config_inference="/tmp/patcher.yaml",
    )

    env = dumper_utils.get_sglang_env(args)

    assert env == {
        "DUMPER_SERVER_PORT": "reuse",
        "DUMPER_NON_INTRUSIVE_MODE": "all",
        "DUMPER_SOURCE_PATCHER_CONFIG": "/tmp/patcher.yaml",
    }


def test_sglang_env_disabled_when_inference_phase_disabled() -> None:
    # [FUNCTION DOC] With global enable off and the inference overrides not enabling the phase,
    # assert the engine environment mapping is empty.
    args = SimpleNamespace(
        dumper_enable=False,
        dumper_inference=["non_intrusive_mode=all"],
        dumper_source_patcher_config_inference=None,
    )

    assert dumper_utils.get_sglang_env(args) == {}


def _make_args(dump_dir: Path, *, enable: bool = True) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Build a minimal arguments namespace with the enable flag (default
    # true), the dump directory as text, and empty override lists for the forward-backward and
    # forward-only phases.
    return SimpleNamespace(
        dumper_enable=enable,
        dumper_dir=str(dump_dir),
        dumper_fwd_bwd=[],
        dumper_fwd_only=[],
    )


class TestDumperMegatronUtilConfigure:
    # [CLASS DOC] Tests of the Megatron dumper helper's configure step: per-rollout experiment
    # naming and when the phase parent directory gets wiped.
    """Per-rollout dump directory layout and the rollout-0 parent-wipe gate."""

    @pytest.fixture()
    def parallel_state(self):
        # [FUNCTION DOC] HELPER: Fixture patching the parallel-state getter in the module under
        # test to return a state whose effective data-parallel rank and independent data-parallel
        # rank are zero (no group); yields that state for the test duration.
        """Single-rank, DP-rank-0 parallel state so cleanup and output run locally."""
        state = SimpleNamespace(
            effective_dp=SimpleNamespace(rank=0),
            indep_dp=SimpleNamespace(rank=0, group=None),
        )
        with patch("miles.utils.dumper_utils.get_parallel_state", return_value=state):
            yield state

    def _configure(self, args: SimpleNamespace, *, phase: DumperPhase, rollout_id: int) -> bool:
        # [FUNCTION DOC] HELPER: Run the helper class's configure step for the given phase and
        # rollout with the dumper and the process-group module mocked (group reported as not
        # initialized). Returns a pair of the enabled result and the mock dumper.
        # KNOWN ISSUE: the return annotation says a single boolean but a tuple of the enabled flag
        # and the mock dumper is returned.
        with (
            patch("miles.utils.dumper_utils.dumper") as mock_dumper,
            patch("miles.utils.dumper_utils.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            enabled = DumperMegatronUtil._configure(args, phase=phase, rollout_id=rollout_id)
            return enabled, mock_dumper

    def test_disabled_phase_returns_false(self, tmp_path: Path, parallel_state) -> None:
        # [FUNCTION DOC] Configure with the enable flag off; assert the result is exactly false and
        # the dumper's configure method was never called.
        """A phase whose override has enable=false short-circuits without configuring the dumper."""
        args = _make_args(tmp_path, enable=False)
        enabled, mock_dumper = self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=0)
        assert enabled is False
        mock_dumper.configure.assert_not_called()

    def test_exp_name_includes_phase_and_rollout_id(self, tmp_path: Path, parallel_state) -> None:
        # [FUNCTION DOC] Configure the forward-backward phase at rollout three; assert it returns
        # true and the dumper is configured with experiment name "fwd_bwd/rollout_3".
        """exp_name is '{phase}/rollout_{rollout_id}' so each rollout dumps to its own subdirectory."""
        args = _make_args(tmp_path)
        enabled, mock_dumper = self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=3)
        assert enabled is True
        config_kwargs = mock_dumper.configure.call_args.kwargs
        assert config_kwargs["exp_name"] == "fwd_bwd/rollout_3"

    def test_rollout_zero_configure_wipes_phase_parent_dir(self, tmp_path: Path, parallel_state) -> None:
        # [FUNCTION DOC] Pre-create a stale rollout directory with a file under the phase parent,
        # configure at rollout zero, and assert the whole phase parent directory is gone.
        """Configuring at rollout 0 removes the whole phase parent dir (stale rollouts from a prior run)."""
        args = _make_args(tmp_path)
        phase_parent = tmp_path / "fwd_bwd"
        stale_rollout = phase_parent / "rollout_9"
        stale_rollout.mkdir(parents=True)
        (stale_rollout / "stale.pt").write_text("stale")

        self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=0)

        assert not phase_parent.exists()

    def test_second_configure_does_not_wipe_parent_only_own_subdir(self, tmp_path: Path, parallel_state) -> None:
        # [FUNCTION DOC] Configure rollout zero, then create rollout-zero data and stale
        # rollout-one data on disk and configure rollout one. Assert rollout zero's file survives
        # and the stale rollout-one directory is removed (only its own subdirectory is cleaned).
        """The second rollout keeps sibling rollout dirs and only cleans/recreates its own subdir."""
        args = _make_args(tmp_path)
        self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=0)

        phase_parent = tmp_path / "fwd_bwd"
        rollout_0 = phase_parent / "rollout_0"
        rollout_0.mkdir(parents=True)
        (rollout_0 / "keep.pt").write_text("rollout_0 data")
        rollout_1 = phase_parent / "rollout_1"
        rollout_1.mkdir(parents=True)
        (rollout_1 / "stale.pt").write_text("old rollout_1 data")

        self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=1)

        assert (rollout_0 / "keep.pt").exists()
        assert not rollout_1.exists()

    def test_respawned_process_at_nonzero_rollout_preserves_phase_parent_dir(
        self, tmp_path: Path, parallel_state
    ) -> None:
        # [FUNCTION DOC] With rollout-zero data on disk, configure at rollout two (as after a
        # respawn) and assert the rollout-zero file still exists, i.e. no parent wipe.
        """A fresh process configuring mid-run (rollout > 0, e.g. after a respawn) keeps earlier rollout dumps."""
        args = _make_args(tmp_path)
        phase_parent = tmp_path / "fwd_bwd"
        rollout_0 = phase_parent / "rollout_0"
        rollout_0.mkdir(parents=True)
        (rollout_0 / "keep.pt").write_text("rollout_0 data")

        self._configure(args, phase=DumperPhase.FWD_BWD, rollout_id=2)

        assert (rollout_0 / "keep.pt").exists()

    def test_rollout_zero_wipe_only_touches_own_phase_dir(self, tmp_path: Path, parallel_state) -> None:
        # [FUNCTION DOC] With forward-backward data on disk, configure the forward-only phase at
        # rollout zero and assert the forward-backward file is untouched.
        """A rollout-0 wipe of one phase leaves the other phase's dumps intact."""
        args = _make_args(tmp_path)
        other_phase_file = tmp_path / "fwd_bwd" / "rollout_0" / "keep.pt"
        other_phase_file.parent.mkdir(parents=True)
        other_phase_file.write_text("fwd_bwd data")

        self._configure(args, phase=DumperPhase.FWD_ONLY, rollout_id=0)

        assert other_phase_file.exists()


@pytest.mark.parametrize("cleanup_previous", [False, True])
def test_finalize_preserves_activations_and_pins_model_dumps_to_step_zero(
    tmp_path: Path, cleanup_previous: bool
) -> None:
    # [FUNCTION DOC] Parametrized by the cleanup-previous option. Build the helper for a tiny
    # linear model with a gradient, enabling model value and grad dumps, at rollout three (parallel
    # state mocked to rank zero, no process group). Dump one activation tensor and assert one
    # activation file exists and the dumper config carries the cleanup flag; advance the dumper
    # step and finalize. Assert the dumper then reports step one, is disabled and has cleanup
    # turned off (reset in a finally step). Then assert three dump files exist (activation plus
    # two model dumps: value and grad) and the two model dumps are tagged and recorded with step
    # zero, so activations are preserved and model dumps are pinned to step zero.
    args = _make_args(tmp_path)
    args.dumper_fwd_bwd = [
        "enable_model_value=true",
        "enable_model_grad=true",
        f"cleanup_previous={str(cleanup_previous).lower()}",
    ]
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    state = SimpleNamespace(
        effective_dp=SimpleNamespace(rank=0),
        indep_dp=SimpleNamespace(rank=0, group=None),
    )

    with (
        patch("miles.utils.dumper_utils.get_parallel_state", return_value=state),
        patch("miles.utils.dumper_utils.dist") as mock_dist,
    ):
        mock_dist.is_initialized.return_value = False
        util = DumperMegatronUtil(args, [model], DumperPhase.FWD_BWD, rollout_id=3)
        dumper_utils.dumper.dump("activation", torch.ones(1))
        dump_dir = tmp_path / "fwd_bwd" / "rollout_3"
        assert len(list(dump_dir.glob("*___name=activation___*.pt"))) == 1
        assert dumper_utils.dumper.get_state()["config"]["cleanup_previous"] is cleanup_previous
        dumper_utils.dumper.step()
        try:
            util.finalize([model])
            dumper_state = dumper_utils.dumper.get_state()
            assert dumper_state["step"] == 1
            assert dumper_state["config"]["enable"] is False
            assert dumper_state["config"]["cleanup_previous"] is False
        finally:
            dumper_utils.dumper.reset()
            dumper_utils.dumper.configure(enable=False)

    dump_files = sorted(dump_dir.glob("*.pt"))
    assert len(dump_files) == 3
    model_dump_files = [
        dump_file
        for dump_file in dump_files
        if torch.load(dump_file, weights_only=False)["meta"]["name"] != "activation"
    ]
    assert len(model_dump_files) == 2
    for dump_file in model_dump_files:
        step_tags = [tag for tag in dump_file.stem.split("___") if tag.startswith("step=")]
        assert step_tags == ["step=0"]
        assert torch.load(dump_file, weights_only=False)["meta"]["step"] == 0


def _make_distributed_optimizer_bucket_group(param: torch.nn.Parameter, grad_data: torch.Tensor) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Build a fake gradient bucket group for a distributed optimizer: one
    # bucket whose flat gradient buffer is the given tensor and which maps the given parameter to
    # the whole buffer range; the group reports the distributed optimizer as in use, a single
    # instance of size one with rank zero and no process group.
    bucket = SimpleNamespace(grad_data=grad_data, param_to_index={param: (0, grad_data.numel())})
    return SimpleNamespace(
        ddp_config=SimpleNamespace(use_distributed_optimizer=True),
        intra_distributed_optimizer_instance_group=None,
        intra_distributed_optimizer_instance_size=1,
        intra_distributed_optimizer_instance_rank=0,
        buckets=[bucket],
    )


class TestDumperMegatronUtilFinalize:
    # [CLASS DOC] Tests of the Megatron dumper helper's finalize step.
    def test_finalize_dumps_distributed_optimizer_gradients_without_param_grad(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A linear model with no gradient on its parameter but a fake bucket group
        # holding a gradient buffer filled with 3.5. Enable model grad dumps, finalize, then
        # reset the dumper. Assert exactly one dump file exists, named for the weight gradient,
        # whose value equals a one-by-two tensor of 3.5 (taken from the bucket buffer).
        """A distributed-optimizer param carries its gradient only in the bucket buffer, and that must be dumped."""
        args = _make_args(tmp_path)
        args.dumper_fwd_bwd = ["enable_model_grad=true"]
        model = torch.nn.Linear(2, 1, bias=False)
        model.bucket_groups = [_make_distributed_optimizer_bucket_group(model.weight, torch.full((2,), 3.5))]
        state = SimpleNamespace(
            effective_dp=SimpleNamespace(rank=0),
            indep_dp=SimpleNamespace(rank=0, group=None),
        )

        with (
            patch("miles.utils.dumper_utils.get_parallel_state", return_value=state),
            patch("miles.utils.dumper_utils.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            util = DumperMegatronUtil(args, [model], DumperPhase.FWD_BWD, rollout_id=1)
            assert model.weight.grad is None
            try:
                util.finalize([model])
            finally:
                dumper_utils.dumper.reset()
                dumper_utils.dumper.configure(enable=False)

        dump_files = sorted((tmp_path / "fwd_bwd" / "rollout_1").glob("*.pt"))
        assert len(dump_files) == 1
        dumped = torch.load(dump_files[0], weights_only=False)
        assert dumped["meta"]["name"] == "grad__param__weight"
        assert torch.equal(dumped["value"], torch.full((1, 2), 3.5))


class TestBarrierAfterDumpDirCleanup:
    # [CLASS DOC] Tests of the barrier run after dump-directory cleanup across independent
    # data-parallel cells, focusing on its structured log records and tolerance of aborted groups.
    @staticmethod
    def _run(group: MagicMock) -> None:
        # [FUNCTION DOC] HELPER: Invoke the barrier function with a mocked parallel state (cell
        # rank one of two, the given mock group, quorum debug info of seven) and the process-group
        # module reported as uninitialized.
        state = SimpleNamespace(indep_dp=SimpleNamespace(rank=1, size=2, group=group, debug_info={"quorum": 7}))
        with (
            patch("miles.utils.dumper_utils.get_parallel_state", return_value=state),
            patch("miles.utils.dumper_utils.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            dumper_utils._barrier_after_dump_dir_cleanup()

    def test_successful_cross_cell_barrier_records_keep_the_ft_tag(self, caplog) -> None:
        # [FUNCTION DOC] Run the barrier with a healthy group, capturing info logs from the module
        # logger. Assert every message starts with the "ft " tag, the first is the cross-cell
        # start record for the dump barrier, the second is the matching end record, and the end
        # record contains the quorum of seven and success true.
        """The start and success records stay discoverable by the external ft structured-log tag."""
        with caplog.at_level(logging.INFO, logger="miles.utils.dumper_utils"):
            self._run(MagicMock())

        messages = [record.getMessage() for record in caplog.records if record.name == "miles.utils.dumper_utils"]
        assert all(message.startswith("ft ") for message in messages)
        assert "op=cross_cell phase=start kind=dump_barrier" in messages[0]
        assert "op=cross_cell phase=end kind=dump_barrier" in messages[1]
        assert "quorum=7 success=true" in messages[1]

    def test_degraded_cross_cell_barrier_record_keeps_the_ft_tag(self, caplog) -> None:
        # [FUNCTION DOC] Make the group's barrier raise a runtime error (communicator aborted).
        # Assert all log messages still start with the "ft " tag and the end record reports success
        # false and degraded true.
        """The degraded end record stays discoverable by the external ft structured-log tag."""
        group = MagicMock()
        group.barrier.side_effect = RuntimeError("NCCL communicator was aborted on rank 1")

        with caplog.at_level(logging.INFO, logger="miles.utils.dumper_utils"):
            self._run(group)

        messages = [record.getMessage() for record in caplog.records if record.name == "miles.utils.dumper_utils"]
        assert all(message.startswith("ft ") for message in messages)
        assert "op=cross_cell phase=end kind=dump_barrier" in messages[1]
        assert "success=false degraded=true" in messages[1]

    def test_cross_cell_barrier_abort_does_not_raise(self) -> None:
        # [FUNCTION DOC] With the group's barrier raising a runtime error and quorum zero, call the
        # barrier function directly; it must not raise, and the group barrier is called exactly
        # once.
        """A peer death aborts the cross-cell PG mid-barrier; the survivor continues instead of erroring."""
        group = MagicMock()
        group.barrier.side_effect = RuntimeError("NCCL communicator was aborted on rank 1")
        state = SimpleNamespace(indep_dp=SimpleNamespace(rank=1, size=2, group=group, debug_info={"quorum": 0}))

        with (
            patch("miles.utils.dumper_utils.get_parallel_state", return_value=state),
            patch("miles.utils.dumper_utils.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            dumper_utils._barrier_after_dump_dir_cleanup()

        group.barrier.assert_called_once()


class TestCleanupDumpDir:
    # [CLASS DOC] Tests of the dump-directory cleanup function with the global rank getter mocked.
    """Best-effort dump-dir wipe gated on global rank 0 + indep-DP (cell) rank 0."""

    def test_rank0_cell0_existing_dir_is_removed(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With global rank zero and cell rank zero, cleanup of an existing directory
        # containing a file removes the directory entirely.
        """Global rank 0 of cell 0 deletes the dump dir when it exists."""
        dump_dir = tmp_path / "fwd_bwd"
        dump_dir.mkdir()
        (dump_dir / "stale.pt").write_text("stale")

        with patch("miles.utils.dumper_utils._get_rank", return_value=0):
            dumper_utils._cleanup_dump_dir(dump_dir, indep_dp_rank=0)

        assert not dump_dir.exists()

    def test_rmtree_oserror_is_swallowed_and_warns(self, tmp_path: Path, caplog) -> None:
        # [FUNCTION DOC] Make directory removal raise an OS error; assert cleanup does not
        # propagate it, a warning containing "dump dir cleanup failed" is logged, and the
        # directory still exists.
        """A failing rmtree (e.g. NFS stale handle) is logged as a warning and does not propagate."""
        dump_dir = tmp_path / "fwd_bwd"
        dump_dir.mkdir()

        with (
            patch("miles.utils.dumper_utils._get_rank", return_value=0),
            patch("miles.utils.dumper_utils.shutil.rmtree", side_effect=OSError("Directory not empty")),
            caplog.at_level(logging.WARNING, logger="miles.utils.dumper_utils"),
        ):
            dumper_utils._cleanup_dump_dir(dump_dir, indep_dp_rank=0)

        assert any("dump dir cleanup failed" in record.message for record in caplog.records)
        assert dump_dir.exists()

    def test_nonzero_rank_does_not_delete(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With global rank one and cell rank zero, assert directory removal is never
        # invoked and the directory remains.
        """A non-zero global rank never calls rmtree, so the dir survives for rank 0 to own."""
        dump_dir = tmp_path / "fwd_bwd"
        dump_dir.mkdir()

        with (
            patch("miles.utils.dumper_utils._get_rank", return_value=1),
            patch("miles.utils.dumper_utils.shutil.rmtree") as mock_rmtree,
        ):
            dumper_utils._cleanup_dump_dir(dump_dir, indep_dp_rank=0)

        mock_rmtree.assert_not_called()
        assert dump_dir.exists()

    def test_nonzero_indep_dp_rank_does_not_delete(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With global rank zero but cell rank one, assert directory removal is never
        # invoked and the directory remains.
        """Only cell 0 (indep_dp_rank 0) deletes; other cells leave the shared dir intact."""
        dump_dir = tmp_path / "fwd_bwd"
        dump_dir.mkdir()

        with (
            patch("miles.utils.dumper_utils._get_rank", return_value=0),
            patch("miles.utils.dumper_utils.shutil.rmtree") as mock_rmtree,
        ):
            dumper_utils._cleanup_dump_dir(dump_dir, indep_dp_rank=1)

        mock_rmtree.assert_not_called()
        assert dump_dir.exists()

    def test_missing_dir_is_noop(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Cleanup of a nonexistent directory as rank zero, cell zero: assert no
        # removal call is made and the directory still does not exist.
        """When the dump dir does not exist, cleanup is a silent no-op (no rmtree, no error)."""
        dump_dir = tmp_path / "does_not_exist"

        with (
            patch("miles.utils.dumper_utils._get_rank", return_value=0),
            patch("miles.utils.dumper_utils.shutil.rmtree") as mock_rmtree,
        ):
            dumper_utils._cleanup_dump_dir(dump_dir, indep_dp_rank=0)

        mock_rmtree.assert_not_called()
        assert not dump_dir.exists()


class TestConfigureSglang:
    # [CLASS DOC] Async tests of configuring every inference engine registered with the router.
    # The class attribute resolved_config is a fake resolved serving config with one model that
    # expects two server cells (so two engines). The router worker listing, config resolution,
    # HTTP post and directory cleanup are patched per test; the posted URLs are collected.
    resolved_config = SimpleNamespace(models=[SimpleNamespace(num_server_cells=2)])

    @staticmethod
    def _make_args(tmp_path: Path) -> SimpleNamespace:
        # [FUNCTION DOC] HELPER: Build an arguments namespace with the dumper enabled, no
        # inference overrides, and the given path as the dump directory.
        return SimpleNamespace(
            dumper_enable=True,
            dumper_inference=[],
            dumper_dir=str(tmp_path),
        )

    @pytest.mark.asyncio
    async def test_every_registered_engine_is_configured(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With two engines registered, run the sglang configure step and assert one
        # configure request is posted to each engine, in order, at its dumper configure path.
        """An engine that never receives the configure call silently produces no dumps."""
        posted: list[str] = []

        async def _post(url, body):
            # [FUNCTION DOC] HELPER: Fake async HTTP post that records the URL and ignores the body.
            posted.append(url)

        with (
            patch(
                "miles.rollout.inference_rollout.inference_rollout_train.get_worker_urls",
                new=AsyncMock(return_value=["http://a:1", "http://b:2"]),
            ),
            patch("miles.utils.dumper_utils.resolve_sglang_config", return_value=self.resolved_config),
            patch("miles.utils.http_utils.post", new=_post),
            patch("miles.utils.dumper_utils._cleanup_dump_dir"),
        ):
            await dumper_utils.configure_sglang(self._make_args(tmp_path))

        assert posted == ["http://a:1/dumper/configure", "http://b:2/dumper/configure"]

    @pytest.mark.asyncio
    async def test_surplus_registered_engines_are_accepted_and_configured(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With three engines registered while two are expected, assert the worker
        # listing is awaited exactly once and all three engines receive a configure request.
        """More registered engines than expected is a complete roster, and every one of them is configured."""
        posted: list[str] = []

        async def _post(url, body):
            # [FUNCTION DOC] HELPER: Fake async HTTP post that records the URL and ignores the body.
            posted.append(url)

        get_worker_urls = AsyncMock(return_value=["http://a:1", "http://b:2", "http://c:3"])

        with (
            patch(
                "miles.rollout.inference_rollout.inference_rollout_train.get_worker_urls",
                new=get_worker_urls,
            ),
            patch("miles.utils.dumper_utils.resolve_sglang_config", return_value=self.resolved_config),
            patch("miles.utils.http_utils.post", new=_post),
            patch("miles.utils.dumper_utils._cleanup_dump_dir"),
        ):
            await dumper_utils.configure_sglang(self._make_args(tmp_path))

        assert get_worker_urls.await_count == 1
        assert posted == [
            "http://a:1/dumper/configure",
            "http://b:2/dumper/configure",
            "http://c:3/dumper/configure",
        ]

    @pytest.mark.asyncio
    async def test_configuring_engines_does_not_enter_a_process_group_barrier(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With the process-group module reported as initialized, configure two
        # engines and assert two requests were posted and no process-group barrier was entered.
        """A cell recovering on its own must configure its engines without waiting for the other cells."""
        posted: list[str] = []

        async def _post(url, body):
            # [FUNCTION DOC] HELPER: Fake async HTTP post that records the URL and ignores the body.
            posted.append(url)

        with (
            patch(
                "miles.rollout.inference_rollout.inference_rollout_train.get_worker_urls",
                new=AsyncMock(return_value=["http://a:1", "http://b:2"]),
            ),
            patch("miles.utils.dumper_utils.resolve_sglang_config", return_value=self.resolved_config),
            patch("miles.utils.http_utils.post", new=_post),
            patch("miles.utils.dumper_utils._cleanup_dump_dir"),
            patch("miles.utils.dumper_utils.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = True
            await dumper_utils.configure_sglang(self._make_args(tmp_path))

        assert len(posted) == 2
        mock_dist.barrier.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_incomplete_router_roster_is_waited_out(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] The worker listing returns an empty roster, then one engine, then two.
        # Assert the listing is awaited three times (polling until complete) and exactly the two
        # final engines are configured.
        """Configuration waits until every expected engine has registered."""
        posted: list[str] = []

        async def _post(url, body):
            # [FUNCTION DOC] HELPER: Fake async HTTP post that records the URL and ignores the body.
            posted.append(url)

        get_worker_urls = AsyncMock(side_effect=[[], ["http://a:1"], ["http://a:1", "http://b:2"]])

        with (
            patch(
                "miles.rollout.inference_rollout.inference_rollout_train.get_worker_urls",
                new=get_worker_urls,
            ),
            patch("miles.utils.dumper_utils.resolve_sglang_config", return_value=self.resolved_config),
            patch("miles.utils.http_utils.post", new=_post),
            patch("miles.utils.dumper_utils._cleanup_dump_dir"),
        ):
            await dumper_utils.configure_sglang(self._make_args(tmp_path))

        assert get_worker_urls.await_count == 3
        assert posted == ["http://a:1/dumper/configure", "http://b:2/dumper/configure"]

    @pytest.mark.asyncio
    async def test_a_roster_that_never_fills_fails_loudly(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] The worker listing always returns one engine while two are expected, with
        # the registration timeout patched to 0.05 s. Assert an assertion error whose message
        # reports "1/2 inference engines".
        """A partial router roster eventually reports the missing inference engine."""
        with (
            patch(
                "miles.rollout.inference_rollout.inference_rollout_train.get_worker_urls",
                new=AsyncMock(return_value=["http://a:1"]),
            ),
            patch("miles.utils.dumper_utils.resolve_sglang_config", return_value=self.resolved_config),
            patch("miles.utils.dumper_utils._cleanup_dump_dir"),
            patch("miles.utils.dumper_utils._WORKER_REGISTRATION_TIMEOUT_SECONDS", 0.05),
            pytest.raises(AssertionError, match="1/2 inference engines"),
        ):
            await dumper_utils.configure_sglang(self._make_args(tmp_path))
