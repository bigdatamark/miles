# [MODULE DOC]
# Tests for the Megatron model module's training and inference step functions using hand-written
# fakes (model chunk, data iterator, pipeline-parallel utility, forward/backward engine, parallel
# state). Verifies that a single training step emits its structured log line with the expected
# tag and fields, and that the forward-only pass does not pass the sampling mask to callbacks
# that do not replay sampling support. No GPU or distributed setup is required.
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

import pytest


class FakeModelChunk:
    # [CLASS DOC] HELPER: Fake model chunk that only counts how many times its gradient buffer was
    # zeroed.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts the zero-gradient-buffer call counter at 0.
        self.zero_grad_buffer_count = 0

    def zero_grad_buffer(self) -> None:
        # [FUNCTION DOC] HELPER: Increments the call counter; returns nothing.
        self.zero_grad_buffer_count += 1


class FakeDataIterator:
    # [CLASS DOC] HELPER: Fake data iterator holding an empty list of batches (list of mappings).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the empty batch list.
        self.batches: list[dict[str, Any]] = []


class FakeMpu:
    # [CLASS DOC] HELPER: Fake replacement for the Megatron parallel-utility module, answering
    # only whether this rank is on the last pipeline stage.
    def __init__(self, *, is_last_pipeline_stage: bool = False) -> None:
        # [FUNCTION DOC] HELPER: Stores the fixed last-stage answer (default false).
        self.is_last_pipeline_stage = is_last_pipeline_stage

    def is_pipeline_last_stage(self, ignore_virtual: bool = False) -> bool:
        # [FUNCTION DOC] HELPER: Returns the stored answer; the ignore-virtual flag is accepted
        # for signature compatibility and ignored.
        return self.is_last_pipeline_stage


class FakeForwardBackwardEngine:
    # [CLASS DOC] HELPER: Callable standing in for Megatron's forward-backward function; records
    # the keyword arguments of every call and returns an empty list of per-microbatch results.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the empty call record list.
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Appends the received keyword arguments to the record and returns
        # an empty list.
        self.calls.append(kwargs)
        return []


@dataclass
class FakeParallelGroup:
    # [CLASS DOC] HELPER: Fake parallel group with a size and rank, both defaulting to 1 and 0.
    size: int = 1
    rank: int = 0


@dataclass
class FakeParallelState:
    # [CLASS DOC] HELPER: Fake parallel state exposing an independent data-parallel group and an
    # effective data-parallel group, each a default single-member fake group.
    indep_dp: FakeParallelGroup = field(default_factory=FakeParallelGroup)
    effective_dp: FakeParallelGroup = field(default_factory=FakeParallelGroup)


@dataclass
class TrainOneStepEnv:
    # [CLASS DOC] HELPER: Bundle of everything a training-step test needs: the arguments
    # namespace, the list of fake model chunks, the list of fake data iterators, the fake
    # parallel state and the fake forward-backward engine.
    args: Namespace
    model: list[FakeModelChunk]
    data_iterator: list[FakeDataIterator]
    parallel_state: FakeParallelState
    forward_backward_engine: FakeForwardBackwardEngine


def make_train_one_step_args(**overrides: Any) -> Namespace:
    # [FUNCTION DOC] HELPER: Returns an arguments namespace for the training step: optimizer
    # enabled, multi-LoRA off, no custom hooks, dumper off, sequence length 8, micro batch size 1,
    # NaN check on, CI test off, MTP training off, max response length 512, sampling-support
    # replay off, witness off, local weight checksum off. Keyword overrides replace defaults.
    defaults: dict[str, Any] = dict(
        debug_disable_optimizer=False,
        multi_lora=False,
        custom_megatron_before_train_step_hook_path=None,
        dumper_enable=False,
        dumper_fwd_bwd=[],
        seq_length=8,
        decoder_seq_length=8,
        micro_batch_size=1,
        check_for_nan_in_loss_and_grad=True,
        ci_test=False,
        enable_mtp_training=False,
        rollout_max_response_len=512,
        use_sampling_support_replay=False,
        enable_witness=False,
        save_local_weight_checksum=False,
    )
    return Namespace(**{**defaults, **overrides})


@pytest.fixture
def train_one_step_env(monkeypatch) -> TrainOneStepEnv:
    # [FUNCTION DOC] HELPER: Pytest fixture. Builds a training-step environment with default
    # arguments, one fake model chunk, one fake data iterator, a default fake parallel state and
    # a fresh fake engine; patches the model module's arguments getter, parallel-state getter,
    # forward-backward getter and parallel-utility module to return those fakes (undone by
    # monkeypatch). Returns the environment.
    from miles.backends.megatron_utils import model as model_module

    env = TrainOneStepEnv(
        args=make_train_one_step_args(),
        model=[FakeModelChunk()],
        data_iterator=[FakeDataIterator()],
        parallel_state=FakeParallelState(),
        forward_backward_engine=FakeForwardBackwardEngine(),
    )

    monkeypatch.setattr(model_module, "get_args", lambda: env.args)
    monkeypatch.setattr(model_module, "get_parallel_state", lambda: env.parallel_state)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: env.forward_backward_engine)
    monkeypatch.setattr(model_module, "mpu", FakeMpu())

    return env


class TestTrainOneStepStructuredLog:
    # [CLASS DOC] Tests for the structured log line emitted by the single training step function.
    def test_train_one_step_emits_the_train_tag_in_its_structured_event(
        self, train_one_step_env: TrainOneStepEnv, caplog
    ):
        # [FUNCTION DOC] Runs one training step (rollout 7, step 3, attempt 2, one microbatch, no
        # optimizer or scheduler, no witness) against the fakes with info-level capture on the
        # model module's logger. Asserts the captured messages contain the exact line
        # "train op=train_step rollout=7 step=3 attempt=2 outcome=NORMAL valid_step=true".
        """Log consumers key train-step events off the train tag, so the caller must emit that tag with its fields."""
        from miles.backends.megatron_utils.model import train_one_step

        with caplog.at_level(logging.INFO, logger="miles.backends.megatron_utils.model"):
            train_one_step(
                args=train_one_step_env.args,
                rollout_id=7,
                step_id=3,
                data_iterator=train_one_step_env.data_iterator,
                model=train_one_step_env.model,
                optimizer=None,
                opt_param_scheduler=None,
                num_microbatches=1,
                num_rollouts=1,
                witness_info=None,
                attempt=2,
            )

        assert "train op=train_step rollout=7 step=3 attempt=2 outcome=NORMAL valid_step=true" in caplog.messages


def test_forward_only_omits_sampling_mask_for_callbacks_that_do_not_replay_sampling_support(monkeypatch):
    # [FUNCTION DOC] Patches the model module with a dumper mock (whose forward-step wrapper is
    # the identity), a model config mock, a batch getter returning an empty-ish batch of
    # tokens/masks/lengths, a packed-sequence-params getter returning none, a fake forward-backward
    # function and a fake parallel utility. The fake forward-backward calls the forward-step
    # function for the first iterator and model and feeds its output to the returned callback.
    # Runs the forward-only pass with a callback that records its keyword arguments, and asserts
    # the recorded arguments do not include the rollout sampling mask.
    from miles.backends.megatron_utils import model as model_module

    def forward_backward(**kwargs):
        # [FUNCTION DOC] HELPER: Fake forward-backward function. Calls the supplied forward-step
        # function with the first data iterator and first model; the result is an output and a
        # callback; returns a one-element list holding the callback applied to the output.
        output, callback = kwargs["forward_step_func"](kwargs["data_iterator"][0], kwargs["model"][0])
        return [callback(output)]

    callback_kwargs = {}

    def critic_callback(_output, **kwargs):
        # [FUNCTION DOC] HELPER: Callback that stores its keyword arguments into the enclosing
        # test's dictionary (ignoring the output) and returns an empty mapping.
        callback_kwargs.update(kwargs)
        return {}

    args = Namespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        enable_witness=False,
        use_rollout_entropy=False,
        custom_megatron_before_log_prob_hook_path=None,
        seq_length=8,
        micro_batch_size=1,
    )
    batch = {
        "tokens": None,
        "input_loss_masks": None,
        "multimodal_train_inputs": None,
        "unconcat_tokens": [],
        "total_lengths": [],
        "response_lengths": [],
        "max_seq_lens": None,
    }
    dumper = Mock()
    dumper.wrap_forward_step.side_effect = lambda fn: fn

    monkeypatch.setattr(model_module, "DumperMegatronUtil", lambda *_args, **_kwargs: dumper)
    monkeypatch.setattr(model_module, "get_model_config", lambda _model: Mock(timers=None))
    monkeypatch.setattr(model_module, "get_batch", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(model_module, "get_packed_seq_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward)
    monkeypatch.setattr(model_module, "mpu", FakeMpu())

    model_module.forward_only(
        critic_callback,
        args,
        [Mock()],
        [Mock()],
        [1],
        rollout_id=0,
    )

    assert "rollout_sampling_mask" not in callback_kwargs
