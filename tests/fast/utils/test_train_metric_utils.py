# [MODULE DOC] Unit tests for the performance metric logging of the training backend: throughput
# in teraflops per second and model flops utilization (MFU, throughput over the device peak) for
# the actor train timer. Tracking's log function is replaced by a recorder, the global timer
# singleton is reset around each test, and the device peak is patched per call. Constants: the
# sequence lengths given to the timer and the fake forward teraflops value (60.0) returned by the
# flops model; train flops are three times the forward flops.
from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.backends.training_utils.metrics import perf
from miles.backends.training_utils.metrics.perf import log_perf_data_raw
from miles.utils.misc import SingletonMeta
from miles.utils.timer import Timer

SEQ_LENS = [1024, 2048]
FWD_TFLOPS = 60.0


@pytest.fixture
def timer():
    # [FUNCTION DOC] HELPER: Fixture yielding a fresh timer. Removes any cached timer singleton
    # instance before and after the test, creates a new timer and sets its sequence lengths to the
    # module constant.
    SingletonMeta._instances.pop(Timer, None)
    instance = Timer()
    instance.seq_lens = SEQ_LENS
    yield instance
    SingletonMeta._instances.pop(Timer, None)


@pytest.fixture
def logged(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture that replaces the tracking log function in the perf module
    # with a recorder (reverted at teardown) and returns the list to which every logged payload
    # (the metrics mapping, ignoring other arguments) is appended.
    calls = []
    monkeypatch.setattr(perf.tracking, "log", lambda args, payload, **kw: calls.append(payload))
    return calls


def make_args(**overrides):
    # [FUNCTION DOC] HELPER: Returns a fake argument namespace with defaults (train-step axis off,
    # no MFU peak override, no trainer model id), updated with the keyword overrides.
    defaults = dict(wandb_always_use_train_step=False, mfu_peak_tflops=None, trainer_model_id=None)
    return SimpleNamespace(**{**defaults, **overrides})


def run(timer, args, *, times: dict[str, float], peak: float | None):
    # [FUNCTION DOC] HELPER: Sets the timer's recorded durations to a copy of the given mapping,
    # patches the local device peak function to return the given peak (none means unknown) within
    # a scoped patch context, and calls the perf logger for rollout 0 as primary rank with a flops
    # model that always returns the module forward teraflops value. Logged payloads land in the
    # recorder fixture.
    timer.timers = dict(times)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(perf, "local_peak_bf16_tflops", lambda: peak)
        log_perf_data_raw(
            rollout_id=0,
            args=args,
            is_primary_rank=True,
            compute_total_fwd_flops=lambda seq_lens: FWD_TFLOPS,
        )


def test_mfu_is_train_tflops_over_the_device_peak(timer, logged):
    # [FUNCTION DOC] With a 2 second actor train time and peak 989, asserts exactly one payload
    # is logged with train teraflops equal to three times the forward value over the time (90),
    # MFU equal to that over the peak, and the peak itself logged.
    run(timer, make_args(), times={"actor_train": 2.0}, peak=989.0)
    [payload] = logged
    assert payload["perf/actor_train_tflops"] == pytest.approx(3 * FWD_TFLOPS / 2.0)
    assert payload["perf/actor_train_mfu"] == pytest.approx(90.0 / 989.0)
    assert payload["perf/mfu_peak_tflops"] == 989.0


def test_denominator_is_logged_so_the_ratio_is_auditable(timer, logged):
    # [FUNCTION DOC] Asserts the logged train teraflops equals the logged MFU times the logged
    # peak, so the ratio can be audited from the payload alone.
    run(timer, make_args(), times={"actor_train": 2.0}, peak=989.0)
    [payload] = logged
    assert payload["perf/actor_train_tflops"] == pytest.approx(
        payload["perf/actor_train_mfu"] * payload["perf/mfu_peak_tflops"]
    )


def test_unknown_peak_omits_mfu_instead_of_defaulting(timer, logged):
    # [FUNCTION DOC] With an unknown device peak, asserts the throughput metric is still logged
    # but neither the MFU nor the peak keys appear.
    run(timer, make_args(), times={"actor_train": 2.0}, peak=None)
    [payload] = logged
    assert "perf/actor_train_tflops" in payload  # the throughput metric still lands
    assert "perf/actor_train_mfu" not in payload
    assert "perf/mfu_peak_tflops" not in payload


def test_no_mfu_without_a_flops_model(timer, logged):
    # [FUNCTION DOC] Calls the perf logger directly with no flops model (none) and a known peak;
    # asserts neither MFU nor train teraflops are logged.
    timer.timers = {"actor_train": 2.0}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(perf, "local_peak_bf16_tflops", lambda: 989.0)
        log_perf_data_raw(rollout_id=0, args=make_args(), is_primary_rank=True, compute_total_fwd_flops=None)
    [payload] = logged
    assert "perf/actor_train_mfu" not in payload
    assert "perf/actor_train_tflops" not in payload


def test_zero_train_time_emits_neither_throughput_nor_mfu(timer, logged):
    # [FUNCTION DOC] With a zero actor train time (avoiding division by zero), asserts neither
    # train teraflops nor MFU is logged.
    run(timer, make_args(), times={"actor_train": 0.0}, peak=989.0)
    [payload] = logged
    assert "perf/actor_train_tflops" not in payload
    assert "perf/actor_train_mfu" not in payload


def test_non_primary_rank_logs_nothing(timer, logged):
    # [FUNCTION DOC] Calls the perf logger with the primary-rank flag false (and a flops model
    # accepting only keyword arguments, which must not be called) and asserts nothing was logged.
    timer.timers = {"actor_train": 2.0}
    log_perf_data_raw(rollout_id=0, args=make_args(), is_primary_rank=False, compute_total_fwd_flops=lambda **_: 1.0)
    assert logged == []


class TestLogPerfData:
    # [CLASS DOC] Tests of how the perf logger names its metrics and step axis per policy.
    def test_the_perf_curves_follow_the_policy_step_axis(self, timer, monkeypatch):
        # [FUNCTION DOC] Replaces the tracking log function with a recorder of (payload, step key)
        # pairs, sets a 2 second actor train time, and logs rollout 3 for trainer model alpha
        # without a flops model. Asserts exactly one call whose step key is alpha's rollout step
        # key, and whose payload holds only the prefixed train time (2.0) and the prefixed
        # rollout step (3).
        """Every perf point of a policy must land on that policy's own step axis, not on a shared one."""
        calls: list[tuple[dict, str]] = []
        monkeypatch.setattr(perf.tracking, "log", lambda _args, payload, step_key: calls.append((payload, step_key)))
        timer.timers = {"actor_train": 2.0}

        log_perf_data_raw(
            rollout_id=3,
            args=make_args(trainer_model_id="alpha"),
            is_primary_rank=True,
            compute_total_fwd_flops=None,
        )

        [(payload, step_key)] = calls
        assert step_key == "alpha/rollout/step"
        assert payload == {"alpha/perf/actor_train_time": 2.0, "alpha/rollout/step": 3}
