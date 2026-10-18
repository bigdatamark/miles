# [MODULE DOC] CPU test for the abort routine of the SGLang rollout module. Network calls (listing
# workers, posting abort requests) and the agent abort hook are replaced with in-test fakes; no real
# router, GPU or network is used.
import asyncio
import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout import sglang_rollout


class TestAbort:
    # [CLASS DOC] Tests of the rollout abort routine.
    async def test_abort_survives_one_unresponsive_worker_and_finishes_cleanup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] Sets up args with partial rollout and the miles router enabled, a fake
        # generate state holding one pending task (a group that finishes only after the agent abort
        # hook ran, returning one partial sample), a fake worker listing with a healthy and an
        # unresponsive worker, a fake post that records URLs and raises a connection error for the
        # unresponsive one, and a fake agent abort hook that records its args and signals. Calls the
        # abort routine for rollout 23 and asserts: abort requests were posted to both workers in
        # listing order; the hook was called once with the args; the result is the one aborted group
        # of the partial sample; the sample's metadata records start rollout id 23; and a warning
        # naming the failed worker and its error was logged by the module's logger.
        """One failed worker abort still runs the agent hook and drains partial samples."""
        args = Namespace(
            partial_rollout=True,
            sglang_router_ip="router",
            sglang_router_port=30000,
            use_miles_router=True,
        )
        sample = SimpleNamespace(response="partial", metadata={})
        state = SimpleNamespace(args=args, aborted=False)
        hook_calls: list[Namespace] = []
        posted_urls: list[str] = []
        hook_finished = asyncio.Event()

        async def fake_get(url: str) -> dict[str, list[str]]:
            # [FUNCTION DOC] HELPER: Ignores the URL and returns a mapping with the two worker URLs.
            return {"urls": ["http://healthy", "http://unresponsive"]}

        async def fake_post(url: str, payload: dict[str, bool]) -> None:
            # [FUNCTION DOC] HELPER: Records the URL (payload ignored); raises a connection error with
            # the message "worker cannot answer" when the URL contains "unresponsive".
            posted_urls.append(url)
            if "unresponsive" in url:
                raise ConnectionError("worker cannot answer")

        async def fake_agent_abort_hook(hook_args: Namespace) -> None:
            # [FUNCTION DOC] HELPER: Records the args and sets the hook-finished event.
            hook_calls.append(hook_args)
            hook_finished.set()

        async def finish_group() -> list[SimpleNamespace]:
            # [FUNCTION DOC] HELPER: Pending group task body: waits until the abort hook has run, then
            # returns a list holding the single partial sample.
            await hook_finished.wait()
            return [sample]

        state.pendings = {asyncio.create_task(finish_group())}
        await asyncio.sleep(0)
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda state_args: state)
        monkeypatch.setattr(sglang_rollout, "get", fake_get)
        monkeypatch.setattr(sglang_rollout, "post", fake_post)
        monkeypatch.setattr(sglang_rollout, "call_agent_abort_hook", fake_agent_abort_hook)

        with caplog.at_level(logging.WARNING, logger=sglang_rollout.__name__):
            aborted_groups = await sglang_rollout.abort(args, rollout_id=23)

        assert posted_urls == ["http://healthy/abort_request", "http://unresponsive/abort_request"]
        assert hook_calls == [args]
        assert aborted_groups == [[sample]]
        assert sample.metadata["start_rollout_id"] == 23
        assert "Failed to abort worker at http://unresponsive: worker cannot answer" in caplog.text
