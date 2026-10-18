# [MODULE DOC] Tests for the multi-policy data buffer of the fully asynchronous rollout path. Part 1
# checks that per-policy metric requests are forwarded only to the custom buffer composed for that
# policy, that return values pass through, and that an unknown policy is refused. Part 2 uses the
# real default multi buffer with two policies and checks that collecting one policy's metrics does
# not reset another policy's raw reward window. CPU only; the megatron config is a test fixture
# encoding.
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout import fully_async_data_buffer
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultMultiDataBuffer,
)


class _RecordingBuffer(DataBuffer):
    # [CLASS DOC] HELPER: Custom buffer subclass that records which policy ids it is asked metrics
    # about. Put and get are unsupported.
    """A custom DataBuffer of the kind --custom-async-data-buffer-path-per-model names."""

    def __init__(self, input: DataBufferConstructorInput) -> None:
        # [FUNCTION DOC] HELPER: Keeps the constructor input and an empty list of policy ids asked
        # about. Does not call the base constructor.
        self.input = input
        self.metrics_asked_about: list[str | None] = []

    async def put(self, input: DataBufferInput) -> None:
        # [FUNCTION DOC] HELPER: Not supported; always raises a not-implemented error.
        raise NotImplementedError

    async def get(self, **context) -> DataBufferInput:
        # [FUNCTION DOC] HELPER: Not supported; always raises a not-implemented error.
        raise NotImplementedError

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        # [FUNCTION DOC] HELPER: Appends the policy id (possibly none) to the recorded list and
        # returns a mapping with one key "asked" whose value is the number of calls so far, as a
        # float.
        self.metrics_asked_about.append(trainer_model_id)
        return {"asked": float(len(self.metrics_asked_about))}


def _multi_buffer(monkeypatch: pytest.MonkeyPatch, *, model_ids: list[str]) -> DefaultMultiDataBuffer:
    # [FUNCTION DOC] HELPER: Patches the buffer module so the resolved megatron config reports the
    # given model ids and any custom buffer path loads the recording buffer class. Builds args whose
    # per-model custom buffer paths map every model id to a dummy path, and returns a default multi
    # buffer built from them (with a no-op handler). The patches are undone by the monkeypatch
    # fixture.
    monkeypatch.setattr(
        fully_async_data_buffer, "resolve_megatron_config", lambda args: SimpleNamespace(model_ids=model_ids)
    )
    monkeypatch.setattr(fully_async_data_buffer, "load_function", lambda path: _RecordingBuffer)
    args = Namespace(custom_async_data_buffer_path_per_model=[f"{one}=recording.Buffer" for one in model_ids])
    return DefaultMultiDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=lambda samples: None))


def _composed(multi: DefaultMultiDataBuffer, model_id: str) -> _RecordingBuffer:
    # [FUNCTION DOC] HELPER: Returns the inner buffer the multi buffer composed for the model id,
    # read from its private mapping of inner buffers. Raises a key error for an unknown id.
    return multi._inners[model_id]


class TestTheMetricsOfOnePolicy:
    # [CLASS DOC] Tests of per-policy metric forwarding in the multi buffer, using recording inner
    # buffers for the policies "solver" and "verifier".
    def test_the_policy_the_drain_asked_about_reaches_the_buffer_it_composes(self, monkeypatch):
        # [FUNCTION DOC] Requesting metrics for "solver" calls the solver inner buffer with the id
        # "solver" (not none).
        """A buffer that selects by policy saw None and could attribute its metrics to the wrong one."""
        multi = _multi_buffer(monkeypatch, model_ids=["solver", "verifier"])

        multi.get_metrics("solver")

        assert _composed(multi, "solver").metrics_asked_about == ["solver"]

    def test_a_policy_is_never_asked_about_another_one(self, monkeypatch):
        # [FUNCTION DOC] After requesting metrics for "solver" then "verifier", each inner buffer
        # saw only its own id once.
        """Each policy keeps its own window counters, and a drain resets the ones it reads."""
        multi = _multi_buffer(monkeypatch, model_ids=["solver", "verifier"])

        multi.get_metrics("solver")
        multi.get_metrics("verifier")

        assert _composed(multi, "solver").metrics_asked_about == ["solver"]
        assert _composed(multi, "verifier").metrics_asked_about == ["verifier"]

    def test_the_metrics_returned_are_the_ones_that_policy_buffer_reported(self, monkeypatch):
        # [FUNCTION DOC] Two successive metric requests for "solver" return the inner buffer's
        # mappings unchanged: asked 1.0 then asked 2.0.
        """Forwarding the policy may not cost the caller the numbers it came for."""
        multi = _multi_buffer(monkeypatch, model_ids=["solver", "verifier"])

        assert multi.get_metrics("solver") == {"asked": 1.0}
        assert multi.get_metrics("solver") == {"asked": 2.0}

    def test_a_policy_this_run_does_not_train_is_refused(self, monkeypatch):
        # [FUNCTION DOC] With only "solver" configured, requesting metrics for "stranger" fails an
        # assertion whose message says the run trains no such policy.
        """The composed buffers are one per policy of the run, so any other name selects nothing."""
        multi = _multi_buffer(monkeypatch, model_ids=["solver"])

        with pytest.raises(AssertionError, match="trains no policy of this run"):
            multi.get_metrics("stranger")


from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.utils.types import Sample


def _make_args() -> Namespace:
    # [FUNCTION DOC] HELPER: Builds args for a real default multi buffer: capacity factor 1.0, no
    # custom buffer paths, no dynamic sampling filter, no staleness limit, a megatron config encoding
    # the two policies "solver" and "verifier", no reward key, and rollout batch size 1.
    return Namespace(
        async_data_buffer_capacity_factor=1.0,
        custom_async_data_buffer_path_per_model=None,
        dynamic_sampling_filter_path=None,
        max_weight_staleness=None,
        megatron_config=encode_megatron_config("solver", "verifier"),
        reward_key=None,
        rollout_batch_size=1,
    )


def _make_sample(*, index: int, reward: float, trainer_model_id: str) -> Sample:
    # [FUNCTION DOC] HELPER: Builds a completed sample with the given index and reward, fixed prompt
    # and response text, and tags it with the trainer model id (policy). Returns it.
    sample = Sample(
        index=index,
        prompt="prompt",
        response="response",
        reward=reward,
        status=Sample.Status.COMPLETED,
    )
    sample.trainer_model_id = trainer_model_id
    return sample


def _ignore_group(group: list[Sample]) -> None:
    # [FUNCTION DOC] HELPER: No-op group handler passed to the buffer constructor input.
    pass


class TestPerPolicyMetrics:
    # [CLASS DOC] Tests of per-policy metric windows on the real default multi buffer.
    async def test_collecting_one_policy_metrics_does_not_reset_another_policy_window(self) -> None:
        # [FUNCTION DOC] Puts one group holding a solver sample (reward 1.0) and a verifier sample
        # (reward 3.0). Collecting solver metrics reports the unfiltered raw reward 1.0; collecting
        # verifier metrics still reports 3.0 (not reset by the solver collection); and a second
        # verifier collection no longer contains that metric key (its window was reset by the first
        # read).
        """Collecting one policy preserves another policy's resettable raw-reward window."""
        buffer = DefaultMultiDataBuffer(DataBufferConstructorInput(args=_make_args(), unused_handler_fn=_ignore_group))
        group = [
            _make_sample(index=1, reward=1.0, trainer_model_id="solver"),
            _make_sample(index=2, reward=3.0, trainer_model_id="verifier"),
        ]

        await buffer.put(DataBufferInput(prompt_group=group, group=group))

        assert buffer.get_metrics("solver")["rollout/raw_reward_unfiltered"] == 1.0
        assert buffer.get_metrics("verifier")["rollout/raw_reward_unfiltered"] == 3.0
        assert "rollout/raw_reward_unfiltered" not in buffer.get_metrics("verifier")
