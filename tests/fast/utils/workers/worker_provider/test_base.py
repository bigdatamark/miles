# [MODULE DOC] Tests for the base worker provider contract using a recording subclass: the default
# expected cell count is none; the cell info wire model round-trips through dump and validate and
# rejects undeclared fields (extra-forbidden validation error); get_handle derives the cell id
# from the worker name, returns the handle of exactly the named worker and raises an assertion
# error (mentioning worker_name=) for an unknown worker; workers without a worker class get no
# handle.
from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo


class _Handle(BaseWorkerHandle):
    # [CLASS DOC] HELPER: Minimal concrete handle that remembers the worker name; readiness wait is
    # a no-op and the death probe always reports alive.
    def __init__(self, worker_name: str) -> None:
        self.worker_name = worker_name

    async def wait_ready(self, *, timeout: float) -> None: ...

    async def probe_is_dead(self) -> bool:
        # [FUNCTION DOC] HELPER: Always reports the worker alive.
        return False


class _RecordingProvider(BaseWorkerProvider):
    # [CLASS DOC] HELPER: Provider test double holding one fixed list of worker infos (one per
    # given name, generation 0, no addresses or gpus, with the given worker class) and recording the
    # cell id lists it is asked about.
    def __init__(self, *worker_names: str, worker_class: str | None = "pkg.Worker") -> None:
        self.requested_cell_ids: list[list[str]] = []
        self._infos = [
            WorkerInfo(name=name, generation=0, self_addrs={}, gpu_ids=[], worker_class=worker_class)
            for name in worker_names
        ]

    async def get_addrs(self, worker_name: str):
        # [FUNCTION DOC] HELPER: Not supported; always raises a not-implemented error.
        raise NotImplementedError

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # [FUNCTION DOC] HELPER: Records the requested cell ids and returns one group holding all
        # configured infos, regardless of the ids.
        self.requested_cell_ids.append(cell_ids)
        return [self._infos]

    async def watch_cells(self, reconcile):
        # [FUNCTION DOC] HELPER: Not supported; always raises a not-implemented error.
        raise NotImplementedError

    def _build_handle_of_worker_info(self, info: WorkerInfo) -> BaseWorkerHandle | None:
        # [FUNCTION DOC] HELPER: Returns a handle named after the worker when the info has a worker
        # class, otherwise none.
        return _Handle(info.name) if info.worker_class is not None else None


class TestExpectedNumCells:
    # [CLASS DOC] Tests of the default expected cell count.
    def test_the_default_provider_has_no_expected_cell_count(self) -> None:
        # [FUNCTION DOC] The expected cell count for a group is none for the base implementation.
        """A provider without fleet sizing logic leaves the expected cell count unspecified."""
        provider = _RecordingProvider()

        assert provider.expected_num_cells(group_id="inference") is None


class TestCellInfoWireModel:
    # [CLASS DOC] Tests of the cell info model's serialization.
    def test_cell_info_round_trips_through_its_strict_wire_model(self) -> None:
        # [FUNCTION DOC] Builds a cell info with two worker names and mixed-type metadata (integer,
        # boolean, list); asserts the dump equals the expected plain mapping, validating the dump
        # gives an equal object whose metadata keeps its types, and that an extra undeclared field
        # raises a validation error mentioning extra_forbidden.
        """Cell descriptions preserve every typed wire field and reject undeclared fields."""
        cell_info = CellInfo(
            cell_id="trainer-engine-actor-00003",
            pool_id="trainer-engine-actor",
            alive=True,
            worker_names=[
                "trainer-engine-actor-00003-00000",
                "trainer-engine-actor-00003-00001",
            ],
            workers_hash="workers-sha256",
            meta={"attempt": 2, "draining": False, "labels": ["trainer", "primary"]},
        )

        dumped = cell_info.model_dump()
        restored = CellInfo.model_validate(dumped)

        assert dumped == {
            "cell_id": "trainer-engine-actor-00003",
            "pool_id": "trainer-engine-actor",
            "alive": True,
            "worker_names": [
                "trainer-engine-actor-00003-00000",
                "trainer-engine-actor-00003-00001",
            ],
            "workers_hash": "workers-sha256",
            "meta": {"attempt": 2, "draining": False, "labels": ["trainer", "primary"]},
        }
        assert restored == cell_info
        assert restored.meta["attempt"] == 2
        assert isinstance(restored.meta["attempt"], int)
        assert restored.meta["draining"] is False

        with pytest.raises(ValidationError, match="extra_forbidden"):
            CellInfo.model_validate({**dumped, "undeclared": "field"})


class TestGetHandle:
    # [CLASS DOC] Tests of handle lookup by worker name.
    def test_the_cell_is_derived_from_the_worker_name(self):
        # [FUNCTION DOC] Asking for worker "inference-controller-00000-00000" makes the provider
        # query exactly the cell "inference-controller-00000".
        """A worker name carries its pool and cell, so the provider must not be asked for anything else."""
        provider = _RecordingProvider("inference-controller-00000-00000")

        provider.get_handle("inference-controller-00000-00000")

        assert provider.requested_cell_ids == [["inference-controller-00000"]]

    def test_the_handle_of_the_named_worker_is_returned(self):
        # [FUNCTION DOC] With two workers in the cell, asking for the second returns the handle
        # whose worker name is the second.
        """A cell holds several workers, so the one whose name matches must come back, not merely the first."""
        provider = _RecordingProvider("trainer-engine-actor-00000-00000", "trainer-engine-actor-00000-00001")

        handle = provider.get_handle("trainer-engine-actor-00000-00001")

        assert handle.worker_name == "trainer-engine-actor-00000-00001"

    def test_a_worker_the_cell_does_not_hold_is_rejected(self):
        # [FUNCTION DOC] Asking for a worker not in the cell raises an assertion error mentioning
        # worker_name=.
        """Answering with a handle to some other worker would silently drive the wrong process."""
        provider = _RecordingProvider("trainer-engine-actor-00000-00000")

        with pytest.raises(AssertionError, match="worker_name="):
            provider.get_handle("trainer-engine-actor-00000-00001")


class TestTheHandlesOfACell:
    # [CLASS DOC] Tests of building handles for a list of worker infos.
    def test_a_worker_that_is_only_launched_has_no_handle(self):
        # [FUNCTION DOC] Infos without a worker class produce an empty mapping of handles.
        """An sglang engine is started by the run and called over its own http api, never as a worker."""
        provider = _RecordingProvider("inference-engine-00000-00000", worker_class=None)

        assert provider.get_handles_of_worker_infos(provider._infos) == {}
