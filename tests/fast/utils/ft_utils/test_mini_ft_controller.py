# [MODULE DOC] Tests for the mini fault tolerance controller: the healing poll suspends an
# unhealthy cell without an explicit resume when cells auto resume, and the start function, with
# the http client and thread classes replaced by fakes (a fake client serves one unhealthy cell
# list then cancels on the second poll; the fake thread runs its target synchronously and
# swallows the cancellation), sends the patch requests each cluster backend owns (ray: suspend
# then resume; kubernetes: suspend only).
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from miles.utils.ft_utils import mini_ft_controller
from miles.utils.ft_utils.mini_ft_controller import CellHealthStatus, _CellSnapshot, _MiniFTController


class TestControllerHealing:
    # [CLASS DOC] Test of the controller's poll and heal step.
    async def test_an_auto_resuming_cell_is_suspended_without_an_explicit_resume(self) -> None:
        # [FUNCTION DOC] Asynchronous test. A controller with mocked callbacks, a cell list of one
        # unhealthy cell, zero delays, auto resume enabled and a fixed clock runs one poll and
        # heal. Asserts suspend was awaited exactly once with the cell name and resume was never
        # awaited.
        """An auto-resuming cell completes healing without an explicit resume request."""
        get_cells = AsyncMock(return_value=[_CellSnapshot(name="cell-0", status=CellHealthStatus.UNHEALTHY)])
        suspend_cell = AsyncMock()
        resume_cell = AsyncMock()
        controller = _MiniFTController(
            get_cells=get_cells,
            suspend_cell=suspend_cell,
            resume_cell=resume_cell,
            poll_interval=0.0,
            resume_delay=0.0,
            cells_auto_resume=True,
            clock=lambda: 10.0,
        )

        await controller._poll_and_heal()

        suspend_cell.assert_awaited_once_with("cell-0")
        resume_cell.assert_not_awaited()


class _Response:
    # [CLASS DOC] HELPER: Fake http response holding an optional json payload.
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        # [FUNCTION DOC] HELPER: Stores the payload (may be none).
        self._payload = payload

    def json(self) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Returns the payload; asserts one was given.
        assert self._payload is not None
        return self._payload

    def raise_for_status(self) -> None:
        # [FUNCTION DOC] HELPER: Never raises; always reports success.
        return None


class _HttpClientFake:
    # [CLASS DOC] HELPER: Fake asynchronous http client. The class attribute lists every created
    # instance (tests reset it). Each instance records the suspend values it was patched with.
    instances: list[_HttpClientFake] = []

    def __init__(self, **_: Any) -> None:
        # [FUNCTION DOC] HELPER: Ignores construction arguments, starts with no patches and a zero
        # poll count, and registers itself in the class level instance list.
        self.patches: list[bool] = []
        self._poll_count = 0
        self.instances.append(self)

    async def get(self, _: str) -> _Response:
        # [FUNCTION DOC] HELPER: Counts polls. The first poll returns a cell list with one running
        # cell named cell-0 whose healthy condition is false and whose suspend flag is false; any
        # later poll raises the asynchronous cancellation error to end the controller loop.
        self._poll_count += 1
        if self._poll_count > 1:
            raise asyncio.CancelledError
        return _Response(
            {
                "apiVersion": "miles.io/v1",
                "kind": "CellList",
                "items": [
                    {
                        "apiVersion": "miles.io/v1",
                        "kind": "Cell",
                        "metadata": {"name": "cell-0", "labels": {}},
                        "spec": {"suspend": False},
                        "status": {
                            "phase": "Running",
                            "conditions": [{"type": "Healthy", "status": "False"}],
                            "workers_hash": "hash-0",
                        },
                    }
                ],
            }
        )

    async def patch(self, _: str, *, content: str, headers: dict[str, str]) -> _Response:
        # [FUNCTION DOC] HELPER: Asserts the content type header is json, records the suspend value
        # from the json body's spec, and returns an empty response.
        assert headers == {"Content-Type": "application/json"}
        self.patches.append(json.loads(content)["spec"]["suspend"])
        return _Response()

    async def aclose(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        return None


class _ThreadFake:
    # [CLASS DOC] HELPER: Fake thread that runs its target inline when started.
    def __init__(self, *, target: Any, daemon: bool) -> None:
        # [FUNCTION DOC] HELPER: Remembers the target; the daemon flag is ignored.
        self._target = target

    def start(self) -> None:
        # [FUNCTION DOC] HELPER: Runs the target synchronously and swallows the asynchronous
        # cancellation error that the fake http client raises to end the loop.
        try:
            self._target()
        except asyncio.CancelledError:
            pass


class TestMaybeStartMiniFtController:
    # [CLASS DOC] Test of the controller start function.
    @pytest.mark.parametrize(
        ("cluster_backend", "expected_patches"),
        [("ray", [True, False]), ("kubernetes", [True])],
    )
    def test_each_cluster_backend_emits_the_expected_healing_requests(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cluster_backend: str,
        expected_patches: list[bool],
    ) -> None:
        # [FUNCTION DOC] Parametrized over ray (expects suspend true then resume false patches) and
        # kubernetes (expects suspend true only). Resets the fake client's instance list, swaps in
        # the fake http client and thread, and starts the controller with args enabling it on port
        # 8080 with zero poll interval and resume delay. Asserts exactly one client was created and
        # its recorded patch values equal the expected list.
        """Each cluster backend emits only the cell healing requests it owns."""
        _HttpClientFake.instances = []
        monkeypatch.setattr(mini_ft_controller.httpx, "AsyncClient", _HttpClientFake)
        monkeypatch.setattr(mini_ft_controller.threading, "Thread", _ThreadFake)

        mini_ft_controller.maybe_start_mini_ft_controller(
            argparse.Namespace(
                mini_ft_controller_enable=True,
                api_server_port=8080,
                mini_ft_controller_poll_interval=0.0,
                mini_ft_controller_resume_delay=0.0,
                cluster_backend=cluster_backend,
            )
        )

        assert len(_HttpClientFake.instances) == 1
        assert _HttpClientFake.instances[0].patches == expected_patches
