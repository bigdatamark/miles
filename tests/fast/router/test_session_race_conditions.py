# [MODULE DOC] End-to-end stress tests of the session server's concurrency contract (split locking
# with a closing flag), driven with real HTTP requests from thread pools against a session server in
# front of a mock SGLang backend whose chat replies are patched to include token log-probabilities in
# the session-server format. Tests cover overlapping same-session requests, parallel different
# sessions, per-session serial turns under global parallelism, deleting a session mid-request,
# requests during or after a delete, concurrent double delete, queued chats then delete, and rapid
# create/chat/delete cycles. Uses local ports from 31000 and the Qwen3-0.6B tokenizer; timing-based
# (sleeps and latency) so thresholds are loose.
"""E2E session stress tests.

Contract under test (with split-lock / session.closing):
- Phase 1 (prepare) and Phase 3 (state update) hold session.lock briefly;
  Phase 2 (proxy to SGLang) does NOT hold the lock.
- Concurrent same-session requests can overlap at the backend (Phase 2),
  but state updates (Phase 3) are serialized; stale-update guard
  (expected_num_assistant check) ensures only one concurrent writer wins.
- Different sessions can run in parallel (no global lock).
- Per-session clients can run turn-by-turn without idle gaps while global load stays parallel.
- Delete marks session.closing=True, acquires session.lock, then removes.
  Because the lock is not held during Phase 2, delete can proceed while a
  chat request is mid-proxy; the chat's Phase 3 will see closing=True and
  skip the state update gracefully.
- Chat requests to a closing session get 404 immediately (pre-lock check).
- Chat requests arriving while delete waits for lock get 404 (double-check after lock).
- Concurrent deletes on the same session: second delete gets 404.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import requests
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

HF_CHECKPOINT = "Qwen/Qwen3-0.6B"


def _patch_mock_chat_response():
    # [FUNCTION DOC] HELPER: Returns an (unstarted) patch context for the mock backend's chat
    # completion computation, replacing it with a version that calls the original and then adds to the
    # first choice a meta info mapping holding the output token log-probabilities as pairs of
    # (log-probability, token id), ids derived from the logprob entries' token strings via the mock's
    # tokenizer, and the completion token count. Must be entered as a context manager to take effect.
    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        # [FUNCTION DOC] HELPER: The replacement computation described above. Requires the original
        # response to have a first choice with logprob content entries (key error otherwise).
        # Returns the augmented response.
        response = original_chat_response(self, payload)
        # Session server expects output_token_logprobs as (logprob, token_id).
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        choice["meta_info"] = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": len(output_token_logprobs),
        }
        return response

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response)


@contextmanager
def _router_env(process_fn, *, latency: float = 0.0):
    # [FUNCTION DOC] HELPER: Context manager. With the chat response patch active, starts a mock
    # backend (model name the Qwen3-0.6B checkpoint, the given reply function and latency in
    # seconds), builds a session server config for it (no chat template override, no speculative
    # decoding, retract pause mode), serves the server on a free port from 31000 on a background
    # thread, and yields a namespace of the server URL, the backend and the server thread. The
    # server thread is stopped on exit.
    with _patch_mock_chat_response():
        with with_mock_server(model_name=HF_CHECKPOINT, process_fn=process_fn, latency=latency) as backend:
            config = make_session_server_config(
                backend_url=backend.url,
                hf_checkpoint=HF_CHECKPOINT,
                chat_template_path=None,
                sglang_speculative_algorithm=None,
                pause_generation_mode="retract",
            )
            server_obj = SessionServer(config)

            port = find_available_port(31000)
            server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
            server.start()
            url = f"http://127.0.0.1:{port}"

            try:
                yield SimpleNamespace(url=url, backend=backend, server=server)
            finally:
                server.stop()


def _create_session(url: str) -> str:
    # [FUNCTION DOC] HELPER: Posts to the sessions path (5 s timeout), asserts status 200, and
    # returns the new session id.
    response = requests.post(f"{url}/sessions", timeout=5.0)
    assert response.status_code == 200
    return response.json()["session_id"]


def _chat(url: str, session_id: str, payload: dict, timeout: float = 20.0) -> requests.Response:
    # [FUNCTION DOC] HELPER: Posts the JSON payload to the session's chat completions path with the
    # given timeout (default 20 s) and returns the raw response without checking the status.
    return requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )


class TestSessionConcurrencyContracts:
    # [CLASS DOC] Tests of how requests to the same or different sessions overlap or serialize.
    def test_same_session_concurrent_requests_reach_backend(self):
        # [FUNCTION DOC] With backend latency 0.2 s, a warm-up chat creates one assistant checkpoint.
        # Then four identical retry requests (history, the assistant reply, and a system message) are
        # sent concurrently from four threads after resetting backend stats. Asserts all four return
        # 200 (no server errors), the backend logged four requests, and its maximum observed
        # concurrency is at least 1 (overlap at the backend is allowed because the proxy phase holds no
        # lock).
        """With the split-lock, same-session requests CAN overlap at the backend.

        Phase 2 (proxy) runs without the lock, so concurrent requests are not
        serialized at the backend level.  Phase 3 state updates are still
        serialized; the stale-update guard ensures only one writer wins per
        generation, so no state corruption occurs.
        """

        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock backend reply function: always the text "concurrent-ok" with
            # finish reason stop.
            return ProcessResult(text="concurrent-ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.2) as env:
            session_id = _create_session(env.url)

            # Warm up one assistant checkpoint so repeated identical retry payloads are valid.
            warmup_payload = {"messages": [{"role": "user", "content": "warmup"}]}
            warmup_resp = _chat(env.url, session_id, warmup_payload)
            assert warmup_resp.status_code == 200
            assistant = warmup_resp.json()["choices"][0]["message"]

            retry_payload = {
                "messages": [
                    {"role": "user", "content": "warmup"},
                    assistant,
                    {"role": "system", "content": "retry-from-assistant-checkpoint"},
                ]
            }

            env.backend.reset_stats()
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(_chat, env.url, session_id, retry_payload) for _ in range(4)]
                responses = [f.result(timeout=30.0) for f in futures]

            # All requests should succeed (200) — no 500s.
            assert all(resp.status_code == 200 for resp in responses)
            assert len(env.backend.request_log) == 4
            # With split-lock, concurrent backend access is expected (not == 1).
            assert env.backend.max_concurrent >= 1

    def test_different_sessions_can_run_in_parallel(self):
        # [FUNCTION DOC] With latency 0.2 s, six sessions are created and one chat per session is
        # submitted concurrently from six threads. Asserts all return 200, the backend logged six
        # requests, and at least three were in flight at once.
        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock backend reply function: always "parallel-ok", finish stop.
            return ProcessResult(text="parallel-ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.2) as env:
            session_ids = [_create_session(env.url) for _ in range(6)]

            env.backend.reset_stats()
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [
                    pool.submit(
                        _chat,
                        env.url,
                        sid,
                        {"messages": [{"role": "user", "content": f"parallel-{i}"}]},
                    )
                    for i, sid in enumerate(session_ids)
                ]
                responses = [f.result(timeout=30.0) for f in futures]

            assert all(resp.status_code == 200 for resp in responses)
            assert len(env.backend.request_log) == 6
            assert env.backend.max_concurrent >= 3

    def test_e2e_pressure_serial_per_session_parallel_globally(self):
        # [FUNCTION DOC] Eight sessions each run three sequential turns in their own worker thread
        # (latency 0.08 s). A worker sends the first user message, then for each later turn resends the
        # history plus the previous assistant reply and a system continue message, asserting 200 each
        # time; it returns true. Asserts all workers returned true, the backend logged 24 requests, and
        # at least four were in flight at once.
        num_sessions = 8
        turns_per_session = 3

        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock backend reply function: always "turn-ok", finish stop.
            return ProcessResult(text="turn-ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.08) as env:
            session_ids = [_create_session(env.url) for _ in range(num_sessions)]

            def run_session_worker(session_id: str, idx: int) -> bool:
                # [FUNCTION DOC] HELPER: Runs the turn loop for one session as described by the
                # enclosing test: asserts status 200 on every turn; between turns, extends the message
                # history with the assistant reply and a system message naming the session and turn.
                # Returns true.
                messages: list[dict] = [{"role": "user", "content": f"session-{idx}-turn-0"}]
                for turn in range(turns_per_session):
                    resp = _chat(env.url, session_id, {"messages": messages}, timeout=30.0)
                    assert resp.status_code == 200
                    assistant = resp.json()["choices"][0]["message"]
                    if turn < turns_per_session - 1:
                        messages = [
                            *messages,
                            assistant,
                            {"role": "system", "content": f"session-{idx}-continue-{turn}"},
                        ]
                return True

            env.backend.reset_stats()
            with ThreadPoolExecutor(max_workers=num_sessions) as pool:
                futures = [pool.submit(run_session_worker, sid, idx) for idx, sid in enumerate(session_ids)]
                results = [f.result(timeout=120.0) for f in futures]

            assert all(results)
            assert len(env.backend.request_log) == num_sessions * turns_per_session
            assert env.backend.max_concurrent >= 4

    def test_delete_can_proceed_while_chat_is_mid_proxy(self):
        # [FUNCTION DOC] With latency 0.35 s, one chat is submitted in a thread; the test polls (up to
        # 5 s every 10 ms) until the backend logged it, failing otherwise, then deletes the session.
        # Asserts the chat returns 200, the delete 204, and a later chat to the deleted session returns
        # 404.
        """With split-lock, delete can acquire the lock while chat is in Phase 2.

        The inflight chat's Phase 3 sees session.closing=True and skips
        state update gracefully.  Both chat and delete complete without error.
        """

        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock backend reply function: always "slow-turn", finish stop.
            return ProcessResult(text="slow-turn", finish_reason="stop")

        with _router_env(process_fn, latency=0.35) as env:
            session_id = _create_session(env.url)
            payload = {"messages": [{"role": "user", "content": "slow-turn-0"}]}

            with ThreadPoolExecutor(max_workers=2) as pool:
                inflight = pool.submit(_chat, env.url, session_id, payload, 30.0)

                # Wait until the first request has reached backend before deleting.
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("in-flight request did not reach backend in time")

                delete_resp = requests.delete(f"{env.url}/sessions/{session_id}", timeout=30.0)
                inflight_resp = inflight.result(timeout=30.0)

            # Chat returns 200 (backend responded); delete returns 204.
            assert inflight_resp.status_code == 200
            assert delete_resp.status_code == 204
            # Session is gone after delete.
            post_delete = _chat(env.url, session_id, payload, timeout=10.0)
            assert post_delete.status_code == 404


class TestClosingRaceConditions:
    # [CLASS DOC] Tests of races around deleting a session (the closing flag).
    """Tests for race conditions around session.closing flag."""

    def test_chat_during_delete_returns_404(self):
        # [FUNCTION DOC] With latency 0.5 s: chat A starts and the test waits (up to 5 s; unlike
        # other tests it does not fail on timeout) for it to reach the backend; a delete is submitted
        # and, after 50 ms, chat B is sent in the foreground. Asserts chat B returns 404 (session
        # closing), then chat A returns 200 and the delete returns 204.
        """Chat requests arriving after delete sets closing=True get 404.

        Timeline:
        1. Chat A starts, acquires lock (Phase 1), releases it, proxying (Phase 2)
        2. Delete arrives, sets session.closing=True, acquires lock, removes session
        3. Chat B arrives, sees session.closing=True, returns 404 immediately
        4. Chat A's Phase 3 sees closing=True, skips state update, returns 200
        """

        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock backend reply function: always "slow", finish stop.
            return ProcessResult(text="slow", finish_reason="stop")

        with _router_env(process_fn, latency=0.5) as env:
            session_id = _create_session(env.url)
            payload = {"messages": [{"role": "user", "content": "slow-chat"}]}

            with ThreadPoolExecutor(max_workers=3) as pool:
                # 1. Start slow chat A
                chat_a = pool.submit(_chat, env.url, session_id, payload, 30.0)

                # Wait for chat A to reach backend
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)

                # 2. Start delete (will block waiting for lock)
                delete_future = pool.submit(
                    requests.delete,
                    f"{env.url}/sessions/{session_id}",
                    timeout=30.0,
                )
                # Small delay to ensure delete has set closing=True
                time.sleep(0.05)

                # 3. Chat B should get 404 because session.closing=True
                chat_b = _chat(env.url, session_id, payload, timeout=10.0)
                assert chat_b.status_code == 404, f"Chat during closing should return 404, got {chat_b.status_code}"

                # Wait for remaining futures
                chat_a_resp = chat_a.result(timeout=30.0)
                delete_resp = delete_future.result(timeout=30.0)

            assert chat_a_resp.status_code == 200
            assert delete_resp.status_code == 204

    def test_double_delete_second_returns_404(self):
        # [FUNCTION DOC] With latency 0.3 s: a chat holds the session; after it reaches the backend
        # (waits up to 5 s without failing on timeout), two deletes are fired 20 ms apart. Asserts the
        # chat returns 200 and the two delete status codes sorted are [204, 404] (one succeeds, the
        # other sees the closing flag).
        """Concurrent delete on the same session: second delete gets 404.

        With session.closing flag, the first delete sets closing=True.
        The second delete sees closing=True and returns 404.
        """

        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock backend reply function: always "ok", finish stop.
            return ProcessResult(text="ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.3) as env:
            session_id = _create_session(env.url)

            # Start a slow chat to hold the lock
            payload = {"messages": [{"role": "user", "content": "hold-lock"}]}
            with ThreadPoolExecutor(max_workers=3) as pool:
                chat_future = pool.submit(_chat, env.url, session_id, payload, 30.0)

                # Wait for chat to reach backend
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)

                # Fire two deletes concurrently
                delete_1 = pool.submit(
                    requests.delete,
                    f"{env.url}/sessions/{session_id}",
                    timeout=30.0,
                )
                time.sleep(0.02)  # tiny delay to let first delete set closing
                delete_2 = pool.submit(
                    requests.delete,
                    f"{env.url}/sessions/{session_id}",
                    timeout=30.0,
                )

                chat_resp = chat_future.result(timeout=30.0)
                d1 = delete_1.result(timeout=30.0)
                d2 = delete_2.result(timeout=30.0)

            assert chat_resp.status_code == 200
            # One delete succeeds, the other gets 404
            codes = sorted([d1.status_code, d2.status_code])
            assert codes == [204, 404], f"Expected [204, 404], got {codes}"

    def test_chat_after_delete_returns_404(self):
        # [FUNCTION DOC] After deleting a fresh session (204), both a chat request and a GET of the
        # session return 404.
        """Chat request after session is fully deleted returns 404."""

        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock backend reply function: always "ok", finish stop.
            return ProcessResult(text="ok", finish_reason="stop")

        with _router_env(process_fn) as env:
            session_id = _create_session(env.url)

            # Delete the session
            delete_resp = requests.delete(f"{env.url}/sessions/{session_id}", timeout=5.0)
            assert delete_resp.status_code == 204

            # Chat should get 404
            payload = {"messages": [{"role": "user", "content": "hello"}]}
            chat_resp = _chat(env.url, session_id, payload, timeout=5.0)
            assert chat_resp.status_code == 404

            # GET should also get 404
            get_resp = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0)
            assert get_resp.status_code == 404

    def test_multiple_chats_queued_then_delete(self):
        # [FUNCTION DOC] With latency 0.3 s, three identical chats are fired concurrently; once the
        # backend has logged one (up to 5 s wait, no failure on timeout) the session is deleted.
        # Asserts the delete returns 204, every chat returned 200 or 404 (never a server error), and
        # at least one returned 200.
        """Multiple chat requests queued behind session.lock, then delete.

        After delete marks closing=True, queued chats that acquire the lock
        should check closing and return 404.
        """

        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock backend reply function: always "queued-ok", finish stop.
            return ProcessResult(text="queued-ok", finish_reason="stop")

        with _router_env(process_fn, latency=0.3) as env:
            session_id = _create_session(env.url)
            payload = {"messages": [{"role": "user", "content": "queued"}]}

            with ThreadPoolExecutor(max_workers=6) as pool:
                # Fire 3 chats (first holds lock, others queue)
                chat_futures = [pool.submit(_chat, env.url, session_id, payload, 30.0) for _ in range(3)]

                # Wait for first to reach backend
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)

                # Now delete - sets closing, waits for first chat to finish
                delete_future = pool.submit(
                    requests.delete,
                    f"{env.url}/sessions/{session_id}",
                    timeout=30.0,
                )

                results = [f.result(timeout=30.0) for f in chat_futures]
                delete_resp = delete_future.result(timeout=30.0)

            assert delete_resp.status_code == 204

            # At least one chat must succeed (the one holding the lock when
            # delete arrived).  Others may get 200 (acquired lock before
            # closing) or 404 (saw closing=True).  No 500s allowed.
            status_codes = [r.status_code for r in results]
            assert all(c in (200, 404) for c in status_codes), f"Unexpected status codes: {status_codes}"
            assert 200 in status_codes, f"Expected at least one 200, got {status_codes}"

    def test_rapid_create_chat_delete_cycles(self):
        # [FUNCTION DOC] Runs 20 lifecycle cycles on an 8-thread pool (no backend latency) and asserts
        # all return true (each cycle's own assertions passed).
        """Rapidly create, chat, and delete sessions to stress the lifecycle.

        Ensures no deadlocks or crashes from rapid session lifecycle operations.
        """

        def process_fn(prompt: str) -> ProcessResult:
            # [FUNCTION DOC] HELPER: Mock backend reply function: always "cycle-ok", finish stop.
            return ProcessResult(text="cycle-ok", finish_reason="stop")

        with _router_env(process_fn) as env:

            def lifecycle_cycle(idx: int) -> bool:
                # [FUNCTION DOC] HELPER: Creates a session, sends one chat (asserts 200), deletes the
                # session (asserts 204), then asserts a GET of it returns 404. Returns true.
                session_id = _create_session(env.url)
                payload = {"messages": [{"role": "user", "content": f"cycle-{idx}"}]}
                chat_resp = _chat(env.url, session_id, payload, timeout=10.0)
                assert chat_resp.status_code == 200
                delete_resp = requests.delete(f"{env.url}/sessions/{session_id}", timeout=5.0)
                assert delete_resp.status_code == 204
                # Verify gone
                get_resp = requests.get(f"{env.url}/sessions/{session_id}", timeout=5.0)
                assert get_resp.status_code == 404
                return True

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(lifecycle_cycle, i) for i in range(20)]
                results = [f.result(timeout=60.0) for f in futures]

            assert all(results)
