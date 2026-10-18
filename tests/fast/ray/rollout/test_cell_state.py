# [MODULE DOC]
# Tests of the rollout cell state payload models (uninitialized, initializing, pending weights,
# serving, disposed) and the cell address info model. Verifies that every model rejects unknown
# fields and that every model is frozen. A shared address info value and a table of (name, model,
# constructor arguments) per state drive the parametrized cases.
from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    StateDisposed,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)

_ADDR_INFO = CellAddrInfo(
    server_url="http://10.0.0.1:30000",
    bootstrap_port=None,
    gate_url="http://10.0.0.1:13000",
)

_STATE_FACTORIES = [
    ("StateUninitialized", StateUninitialized, {}),
    ("StateInitializing", StateInitializing, {"addr_info": _ADDR_INFO, "start_time": 1000.0}),
    ("StatePendingWeights", StatePendingWeights, {"addr_info": _ADDR_INFO}),
    ("StateServing", StateServing, {"addr_info": _ADDR_INFO}),
    ("StateDisposed", StateDisposed, {}),
]


class TestCellStateModelsRejectUnknownFields:
    # [CLASS DOC] Checks that the state models and the address info model forbid extra fields, so a
    # misspelled field raises a validation error naming that field instead of being dropped.
    """A typo'd payload field must fail loudly instead of being silently dropped."""

    @pytest.mark.parametrize(
        "model, kwargs",
        [(model, kwargs) for _name, model, kwargs in _STATE_FACTORIES],
        ids=[name for name, _model, _kwargs in _STATE_FACTORIES],
    )
    def test_an_unknown_field_is_rejected(self, model, kwargs) -> None:
        # [FUNCTION DOC] Parametrized over all five state models with their table constructor
        # arguments. Constructing the model with an additional unexpected address-info field named
        # stale_addr_info must raise a validation error whose message mentions that field name.
        """Every state payload forbids extra fields."""
        with pytest.raises(ValidationError, match="stale_addr_info"):
            model(**kwargs, stale_addr_info=_ADDR_INFO)

    def test_cell_addr_info_rejects_an_unknown_field(self) -> None:
        # [FUNCTION DOC] Builds the address info model with valid server url, bootstrap port and gate
        # url plus an extra bootstrap host field; expects a validation error mentioning that field.
        """The address payload forbids extra fields too."""
        with pytest.raises(ValidationError, match="bootstrap_host"):
            CellAddrInfo(
                server_url="http://10.0.0.1:30000",
                bootstrap_port=None,
                gate_url="http://10.0.0.1:13000",
                bootstrap_host="10.0.0.1",
            )


class TestCellStateModelsAreFrozen:
    # [CLASS DOC] Checks the frozen-model contract: the state machine swaps whole state objects, so
    # in-place attribute assignment on any state or on the address info must be impossible.
    """The state machine replaces the whole state object; in-place edits must be impossible."""

    @pytest.mark.parametrize(
        "model, kwargs",
        [(model, kwargs) for _name, model, kwargs in _STATE_FACTORIES],
        ids=[name for name, _model, _kwargs in _STATE_FACTORIES],
    )
    def test_every_state_payload_is_frozen(self, model, kwargs) -> None:
        # [FUNCTION DOC] Parametrized over all five state models. Builds an instance and asserts its
        # model configuration has the frozen flag set (this also covers states without fields). For
        # states that carry an address info, additionally assigns a new address info to the instance
        # and expects a validation error.
        """Every state payload retains the frozen-model contract, including fieldless states."""
        state = model(**kwargs)

        assert state.model_config["frozen"] is True

        if "addr_info" in kwargs:
            with pytest.raises(ValidationError):
                state.addr_info = _ADDR_INFO

    def test_cell_addr_info_cannot_be_reassigned_in_place(self) -> None:
        # [FUNCTION DOC] Assigns a different server url to the shared module-level address info and
        # expects a validation error, so the shared object stays safe to hand out by reference.
        """A shared address payload must stay safe to hand out by reference."""
        with pytest.raises(ValidationError):
            _ADDR_INFO.server_url = "http://10.0.0.2:30000"
