# [MODULE DOC]
# Defines a dummy ray actor class standing in for a trainer worker in cell and controller tests.
# It needs no gpu and almost no cpu. It exposes the methods a trainer worker is driven through,
# records every call as (method name, args, kwargs), can be told to raise an injected error for
# chosen methods, returns configurable values for train (single value or per-attempt queue) and
# update weights, serves a configurable train parallel config, and implements a simple heartbeat
# whose staleness and failure can be controlled by tests.
"""Lightweight Ray actor for unit testing TrainerCell/TrainerController without GPU or real training.

Records all method calls so tests can verify what was dispatched.
"""

import os
from typing import Any

import ray

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.utils.ft_utils.heartbeat_utils import HeartbeatStatus, SimpleHeartbeat


@ray.remote(num_gpus=0, num_cpus=0)
class DummyTrainActor:
    # [CLASS DOC] Ray remote class (no gpu, no cpu reservation) emulating a trainer worker. All
    # driven methods record their invocation first (which may raise an injected failure) and then
    # return a canned value. State lives in the actor process, so tests configure and read it
    # through remote calls.

    def __init__(self):
        # [FUNCTION DOC] Initializes the call log (empty), the set of method names that fail
        # (empty), the train return value (a normal step outcome), the per-attempt train return
        # queue (empty), the update weights return value (none), the train parallel config (empty
        # mapping), a heartbeat that has been bumped once, and the heartbeat failure flag (off).
        self._calls: list[tuple[str, tuple, dict]] = []
        self._fail_methods: set[str] = set()
        self._train_return_value: Any = TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
        self._train_return_values_per_attempt: list[Any] = []
        self._update_weights_return_value: Any = None
        self._train_parallel_config: dict = {}
        self._heartbeat = SimpleHeartbeat()
        self._heartbeat.bump()
        self._heartbeat_fail: bool = False

    def set_fail_methods(self, methods: list[str]) -> None:
        # [FUNCTION DOC] Replaces the set of method names whose recorded invocation raises an
        # injected runtime error. Pass an empty list to clear.
        self._fail_methods = set(methods)

    def set_train_return_value(self, value: Any) -> None:
        # [FUNCTION DOC] Sets the value returned by every train call once the per-attempt queue is empty.
        self._train_return_value = value

    def set_train_return_values_per_attempt(self, values: list[Any]) -> None:
        # [FUNCTION DOC] Stores a copy of the given list as a queue: each train call pops and
        # returns the first element until the queue is empty, after which the plain train return
        # value is used.
        self._train_return_values_per_attempt = list(values)

    def set_train_parallel_config(self, config: dict) -> None:
        # [FUNCTION DOC] Sets the mapping returned by the train parallel config getter.
        self._train_parallel_config = config

    def _record(self, method: str, args: tuple, kwargs: dict) -> None:
        # [FUNCTION DOC] HELPER: Appends (method name, args, kwargs) to the call log, then raises a
        # runtime error whose message names the method if that method is in the failing set. The
        # call is logged before the failure is raised.
        self._calls.append((method, args, kwargs))
        if method in self._fail_methods:
            raise RuntimeError(f"Injected failure in {method}")

    def get_calls(self) -> list[tuple[str, tuple, dict]]:
        # [FUNCTION DOC] Returns a copy of the call log: one (method name, args, kwargs) tuple per
        # recorded invocation, in call order.
        return list(self._calls)

    def init(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Records an init call (may raise if configured to fail); returns none.
        self._record("init", args, kwargs)

    def configure_master_addr_and_port(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Records a master address configuration call (may raise if configured); returns none.
        self._record("configure_master_addr_and_port", args, kwargs)

    def load_state(self, *args: Any, **kwargs: Any) -> int:
        # [FUNCTION DOC] Records a state load call (may raise if configured) and returns the fixed
        # restored rollout value 7.
        self._record("load_state", args, kwargs)
        return 7

    def reconfigure_indep_dp(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Records an independent dp reconfiguration call (may raise if configured); returns none.
        self._record("reconfigure_indep_dp", args, kwargs)

    def send_ckpt(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Records a checkpoint send call (may raise if configured); returns none.
        self._record("send_ckpt", args, kwargs)

    def train(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Records a train call (may raise if configured). Returns the next element
        # of the per-attempt queue (removing it) if the queue is non-empty, otherwise the plain
        # train return value.
        self._record("train", args, kwargs)
        if self._train_return_values_per_attempt:
            return self._train_return_values_per_attempt.pop(0)
        return self._train_return_value

    def get_train_parallel_config(self) -> dict:
        # [FUNCTION DOC] Records the call (may raise if configured) and returns the configured mapping.
        self._record("get_train_parallel_config", (), {})
        return self._train_parallel_config

    def wake_up(self) -> None:
        # [FUNCTION DOC] Records a wake up call (may raise if configured); returns none.
        self._record("wake_up", (), {})

    def sleep(self) -> None:
        # [FUNCTION DOC] Records a sleep call (may raise if configured); returns none.
        self._record("sleep", (), {})

    def clear_memory(self) -> None:
        # [FUNCTION DOC] Records a memory clearing call (may raise if configured); returns none.
        self._record("clear_memory", (), {})

    def offload_grad_buffer(self) -> None:
        # [FUNCTION DOC] Records a gradient buffer offload call (may raise if configured); returns none.
        self._record("offload_grad_buffer", (), {})

    def save_model(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Records a model save call (may raise if configured); returns none.
        self._record("save_model", args, kwargs)

    def export_hf(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Records a hugging face export call (may raise if configured); returns none.
        self._record("export_hf", args, kwargs)

    def set_update_weights_return_value(self, value: Any) -> None:
        # [FUNCTION DOC] Sets the value returned by every update weights call (initially none).
        self._update_weights_return_value = value

    def update_weights(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Records an update weights call (may raise if configured) and returns the
        # configured value.
        self._record("update_weights", args, kwargs)
        return self._update_weights_return_value

    def kill_self(self) -> None:
        # [FUNCTION DOC] Records a self kill call (may raise if configured), then terminates the
        # actor's process immediately with exit status 1 (no cleanup), so any later remote call
        # fails with an actor death error. Never returns normally.
        self._record("kill_self", (), {})
        os._exit(1)

    def set_heartbeat_fail(self, fail: bool) -> None:
        # [FUNCTION DOC] Turns on or off the heartbeat failure flag; when on, the heartbeat status
        # getter raises an injected runtime error.
        self._heartbeat_fail = fail

    def set_last_active_timestamp(self, ts: float) -> None:
        # [FUNCTION DOC] Replaces the heartbeat status with one carrying the given last-active
        # timestamp and the previous bump count (reaches into a private attribute of the
        # heartbeat object).
        self._heartbeat._status = HeartbeatStatus(
            last_active_timestamp=ts,
            bump_count=self._heartbeat._status.bump_count,
        )

    def get_heartbeat_status(self) -> HeartbeatStatus:
        # [FUNCTION DOC] Raises an injected runtime error if the heartbeat failure flag is on;
        # otherwise returns the current heartbeat status. The call itself is not recorded in the call log.
        if self._heartbeat_fail:
            raise RuntimeError("Injected heartbeat failure")
        return self._heartbeat.status()
