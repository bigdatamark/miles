# [MODULE DOC]
# Tests for the local weight checksum module of the Megatron backend, which hashes each rank's
# model parameters, buffers and optimizer state into a checksum record and logs it through the
# audit event logger. Model chunks and optimizers are mocks; real tensors are hashed. Covers key
# naming, determinism, hash sensitivity, recursive tensor-to-hash transformation, fail-fast
# assertions, and the enabled/disabled dump path with a real event logger in a temporary directory.
"""Tests for local_weight_checksum module."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from miles.backends.megatron_utils.local_weight_checksum import (
    _compute_weight_checksum_state,
    _transform_tensor_to_hash,
    dump_local_weight_checksums,
)
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import TrainEngineLocalWeightChecksumEvent
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


def _make_mock_model_chunk(
    params: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor] | None = None
) -> MagicMock:
    # [FUNCTION DOC] HELPER: Builds a mock model chunk. Takes a mapping of name to tensor for
    # parameters and an optional one for buffers. Side effect: sets each parameter tensor's
    # main-parameter attribute to the tensor itself (mimicking a mixed-precision optimizer's fp32
    # copy). The mock's named-parameters and named-buffers calls return lists sorted by name.
    # Returns the mock.
    """Create a mock DDP model chunk with given named parameters and buffers."""
    chunk = MagicMock()

    param_list = sorted(params.items(), key=lambda x: x[0])
    for _name, tensor in param_list:
        tensor.main_param = tensor

    chunk.named_parameters.return_value = param_list
    chunk.named_buffers.return_value = sorted((buffers or {}).items(), key=lambda x: x[0])
    return chunk


def _make_mock_optimizer_with_state_dict(
    params: list[torch.Tensor],
    states: dict[int, dict[str, torch.Tensor]] | None = None,
) -> MagicMock:
    # [FUNCTION DOC] HELPER: Builds a mock chained optimizer with exactly one sub-optimizer whose
    # inner optimizer mimics an Adam optimizer. Takes the list of parameter tensors and an optional
    # mapping from parameter index to its state mapping (default per parameter: a step counter
    # tensor of 1). The inner optimizer has one param group (learning rate 0.001) with all the
    # params, a state keyed by tensor, and a state-dict call returning state keyed by index plus a
    # param group listing indices. Returns the mock chained optimizer.
    """Create a mock optimizer that has a torch.optim.Optimizer-like inner optimizer."""
    inner = MagicMock(spec=torch.optim.Adam)
    inner.param_groups = [{"params": params, "lr": 0.001}]

    inner_state: dict[torch.Tensor, dict] = {}
    sd_state: dict[int, dict] = {}
    for i, p in enumerate(params):
        s = (states or {}).get(i, {"step": torch.tensor(1)})
        inner_state[p] = s
        sd_state[i] = s
    inner.state = inner_state
    inner.state_dict.return_value = {
        "state": sd_state,
        "param_groups": [{"params": list(range(len(params))), "lr": 0.001}],
    }

    sub_opt = MagicMock(spec=["optimizer"])
    sub_opt.optimizer = inner

    optimizer = MagicMock(spec=["chained_optimizers"])
    optimizer.chained_optimizers = [sub_opt]
    return optimizer


class TestComputeWeightChecksums:
    # [CLASS DOC] Tests for computing the checksum state from mock model chunks and optimizers.
    def test_param_hashes_keys_match_expected_names(self) -> None:
        # [FUNCTION DOC] Two parameters in one chunk. Asserts the parameter-hash keys are the
        # parameter names prefixed with the pipeline chunk index, "pp0.".
        params = {
            "module.layers.0.weight": torch.randn(4, 4),
            "module.layers.1.weight": torch.randn(4, 4),
        }
        model = [_make_mock_model_chunk(params=params)]
        optimizer = _make_mock_optimizer_with_state_dict(params=list(params.values()))

        entry = _compute_weight_checksum_state(model=model, optimizer=optimizer)

        assert set(entry.param_hashes.keys()) == {
            "pp0.module.layers.0.weight",
            "pp0.module.layers.1.weight",
        }

    def test_hash_determinism_same_params_same_hash(self) -> None:
        # [FUNCTION DOC] Computes the checksum state twice from the same model and optimizer.
        # Asserts the parameter hashes are identical.
        tensor = torch.tensor([1.0, 2.0, 3.0])
        params = {"weight": tensor}
        model = [_make_mock_model_chunk(params=params)]
        optimizer = _make_mock_optimizer_with_state_dict(params=[tensor])

        entry1 = _compute_weight_checksum_state(model=model, optimizer=optimizer)
        entry2 = _compute_weight_checksum_state(model=model, optimizer=optimizer)

        assert entry1.param_hashes == entry2.param_hashes

    def test_param_value_change_changes_hash(self) -> None:
        # [FUNCTION DOC] Two models whose single weight differs in one element. Asserts the
        # resulting hashes for the same parameter name differ.
        tensor_a = torch.tensor([1.0, 2.0, 3.0])
        tensor_b = torch.tensor([1.0, 2.0, 4.0])
        model_a = [_make_mock_model_chunk(params={"weight": tensor_a})]
        model_b = [_make_mock_model_chunk(params={"weight": tensor_b})]
        opt_a = _make_mock_optimizer_with_state_dict(params=[tensor_a])
        opt_b = _make_mock_optimizer_with_state_dict(params=[tensor_b])

        entry_a = _compute_weight_checksum_state(model=model_a, optimizer=opt_a)
        entry_b = _compute_weight_checksum_state(model=model_b, optimizer=opt_b)

        assert entry_a.param_hashes["pp0.weight"] != entry_b.param_hashes["pp0.weight"]

    def test_buffer_hashing_produces_values(self) -> None:
        # [FUNCTION DOC] A chunk with one parameter and two buffers. Asserts both buffers appear
        # in the buffer hashes under the pp0 prefix and that a hash is 64 characters long (a hex
        # digest of 256 bits).
        params = {"weight": torch.randn(4, 4)}
        buffers = {"running_mean": torch.randn(4), "running_var": torch.randn(4)}
        model = [_make_mock_model_chunk(params=params, buffers=buffers)]
        optimizer = _make_mock_optimizer_with_state_dict(params=list(params.values()))

        entry = _compute_weight_checksum_state(model=model, optimizer=optimizer)

        assert "pp0.running_mean" in entry.buffer_hashes
        assert "pp0.running_var" in entry.buffer_hashes
        assert len(entry.buffer_hashes["pp0.running_mean"]) == 64

    def test_optimizer_state_dict_captured(self) -> None:
        # [FUNCTION DOC] Optimizer state for parameter 0 has an exponential-average tensor and a
        # step tensor. Asserts exactly one optimizer hash record, whose first param name is
        # "pp0.weight" and whose state-dict entry for the exponential average was converted to a
        # 64-character hash string.
        weight = torch.randn(4, 4)
        exp_avg = torch.randn(4, 4)

        model = [_make_mock_model_chunk(params={"weight": weight})]
        optimizer = _make_mock_optimizer_with_state_dict(
            params=[weight],
            states={0: {"exp_avg": exp_avg, "step": torch.tensor(5)}},
        )

        entry = _compute_weight_checksum_state(model=model, optimizer=optimizer)

        assert len(entry.optimizer_hashes) == 1
        info = entry.optimizer_hashes[0]
        assert info.param_names[0] == "pp0.weight"
        assert isinstance(info.state_dict["state"][0]["exp_avg"], str)
        assert len(info.state_dict["state"][0]["exp_avg"]) == 64


class TestTransformTensorToHash:
    # [CLASS DOC] Tests for the recursive tensor-to-hash transformation (plus one multi-chunk
    # checksum test).
    # KNOWN ISSUE: test_multiple_pp_chunks_produce_distinct_prefixes exercises the checksum-state
    # computation, not the transformation, so it sits in a class it does not belong to.
    def test_replaces_tensor_with_hash(self) -> None:
        # [FUNCTION DOC] Asserts a tensor becomes a 64-character string.
        t = torch.tensor([1.0, 2.0])
        result = _transform_tensor_to_hash(t)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_preserves_non_tensor_values(self) -> None:
        # [FUNCTION DOC] Asserts an integer and a string pass through unchanged.
        assert _transform_tensor_to_hash(42) == 42
        assert _transform_tensor_to_hash("hello") == "hello"

    def test_recurses_into_dict(self) -> None:
        # [FUNCTION DOC] A mapping with a tensor value and an integer value. Asserts the tensor
        # becomes a string and the integer is kept.
        result = _transform_tensor_to_hash({"a": torch.tensor([1.0]), "b": 2})
        assert isinstance(result["a"], str)
        assert result["b"] == 2

    def test_recurses_into_list(self) -> None:
        # [FUNCTION DOC] A list with a tensor then an integer. Asserts the tensor becomes a string
        # and the integer is kept.
        result = _transform_tensor_to_hash([torch.tensor([1.0]), 3])
        assert isinstance(result[0], str)
        assert result[1] == 3

    def test_multiple_pp_chunks_produce_distinct_prefixes(self) -> None:
        # [FUNCTION DOC] Two model chunks with different parameters and one optimizer covering all
        # params. Asserts the first chunk's parameter is keyed with the pp0 prefix and the second
        # chunk's with pp1.
        params_0 = {"embed.weight": torch.randn(4, 4)}
        params_1 = {"head.weight": torch.randn(4, 4)}
        model = [
            _make_mock_model_chunk(params=params_0),
            _make_mock_model_chunk(params=params_1),
        ]
        all_params = list(params_0.values()) + list(params_1.values())
        optimizer = _make_mock_optimizer_with_state_dict(params=all_params)

        entry = _compute_weight_checksum_state(model=model, optimizer=optimizer)

        assert "pp0.embed.weight" in entry.param_hashes
        assert "pp1.head.weight" in entry.param_hashes

    def test_transform_tensor_to_hash_preserves_tuple_type(self) -> None:
        # [FUNCTION DOC] A tuple of a tensor and an integer. Asserts the result is still a tuple,
        # with the tensor turned into a string and the integer kept.
        result = _transform_tensor_to_hash((torch.tensor([1.0]), 42))
        assert isinstance(result, tuple)
        assert isinstance(result[0], str)
        assert result[1] == 42


class TestFailFastAssertions:
    # [CLASS DOC] Tests that misconfigured inputs trigger descriptive assertion failures.
    def test_assert_event_logger_initialized_when_enabled(self) -> None:
        # [FUNCTION DOC] Temporarily clears the module-level event logger global (restored in a
        # finally block). With checksum saving enabled, asserts the dump function raises an
        # assertion error saying the event logger is not initialized.
        import miles.utils.audit_utils.event_logger.logger as mod

        original = mod._event_logger
        mod._event_logger = None
        try:
            args = Namespace(save_local_weight_checksum=True)
            model = [_make_mock_model_chunk(params={"w": torch.randn(2, 2)})]
            optimizer = _make_mock_optimizer_with_state_dict(params=[torch.randn(2, 2)])

            with pytest.raises(AssertionError, match="EventLogger is not initialized"):
                dump_local_weight_checksums(args=args, model=model, optimizer=optimizer)
        finally:
            mod._event_logger = original

    def test_assert_empty_params_fails(self) -> None:
        # [FUNCTION DOC] A chunk with no parameters. Asserts the state computation raises an
        # assertion error saying no parameters were found.
        model = [_make_mock_model_chunk(params={})]
        optimizer = MagicMock()
        optimizer.chained_optimizers = []

        with pytest.raises(AssertionError, match="No parameters found"):
            _compute_weight_checksum_state(model=model, optimizer=optimizer)

    def test_assert_no_sub_optimizers_fails(self) -> None:
        # [FUNCTION DOC] A valid chunk but an optimizer whose chained list is empty. Asserts an
        # assertion error saying no sub-optimizers were found.
        model = [_make_mock_model_chunk(params={"w": torch.randn(2, 2)})]
        optimizer = MagicMock()
        optimizer.chained_optimizers = []

        with pytest.raises(AssertionError, match="No sub-optimizers found"):
            _compute_weight_checksum_state(model=model, optimizer=optimizer)

    def test_assert_param_without_main_param_fails(self) -> None:
        # [FUNCTION DOC] A chunk whose parameter lacks a main-parameter attribute. Asserts the
        # name-by-tensor-id builder raises an assertion error mentioning main_param is none.
        from miles.backends.megatron_utils.local_weight_checksum import _build_name_by_tensor_id

        chunk = MagicMock()
        param = torch.randn(2, 2)
        chunk.named_parameters.return_value = [("weight", param)]

        with pytest.raises(AssertionError, match="main_param is None"):
            _build_name_by_tensor_id([chunk])

    def test_assert_unmapped_fp32_param_fails(self) -> None:
        # [FUNCTION DOC] An inner optimizer holding a parameter absent from an empty name mapping.
        # Asserts the optimizer param-name builder raises an assertion error saying the parameter
        # was not found in the model name mapping.
        from miles.backends.megatron_utils.local_weight_checksum import _build_param_names_for_optimizer

        unmapped_param = torch.randn(2, 2)
        inner = MagicMock(spec=torch.optim.Adam)
        inner.param_groups = [{"params": [unmapped_param]}]

        with pytest.raises(AssertionError, match="not found in model name mapping"):
            _build_param_names_for_optimizer(inner, name_by_tensor_id={})


class TestDumpLocalWeightChecksums:
    # [CLASS DOC] Tests for the public dump function's enabled and disabled behavior.
    def test_does_nothing_when_disabled(self) -> None:
        # [FUNCTION DOC] With the save flag off and a degenerate model and optimizer that would
        # fail if inspected, calls the dump function. Passes if it returns without raising.
        args = Namespace(save_local_weight_checksum=False)
        model = [_make_mock_model_chunk(params={})]
        optimizer = MagicMock()
        optimizer.chained_optimizers = []

        dump_local_weight_checksums(args=args, model=model, optimizer=optimizer)

    def test_dumps_when_enabled(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Installs a real event logger (writing under the temporary directory, with
        # a trainer identity of cell index 2, rank 7) as the global logger; resets the global to
        # none afterwards. Dumps checksums inside a logging context with rollout id 0 and attempt
        # 0, closes the logger, reads events back, and asserts exactly one local-weight-checksum
        # event exists carrying the source cell index 2 and rank within cell 7.
        source = TrainProcessIdentity(component="actor", cell_index=2, rank_within_cell=7)
        event_logger = EventLogger(log_dir=tmp_path, source=source)
        set_event_logger(event_logger)
        try:
            weight = torch.randn(2, 2)
            args = Namespace(save_local_weight_checksum=True)
            model = [_make_mock_model_chunk(params={"w": weight})]
            optimizer = _make_mock_optimizer_with_state_dict(params=[weight])

            with event_logger.with_context({"rollout_id": 0, "attempt": 0}):
                dump_local_weight_checksums(args=args, model=model, optimizer=optimizer)

            event_logger.close()

            events = read_events(tmp_path)
            checksum_events = [e for e in events if isinstance(e, TrainEngineLocalWeightChecksumEvent)]
            assert len(checksum_events) == 1
            assert checksum_events[0].source.cell_index == 2
            assert checksum_events[0].source.rank_within_cell == 7
        finally:
            set_event_logger(None)
