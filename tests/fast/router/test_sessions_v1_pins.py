# [MODULE DOC] HTTP-level behavior pins for the v1 (linear trajectory) session server's
# retry/rollback surface, using the router_env fixture shared from the sessions test module
# (re-bound here because the shared fixture may not be visible in some CI shard combinations).
# Covers pure-drop and divergent retries, the exact 400 text for a too-deep rollback, first-turn
# retries, failed first turns, resending exact history, invalid retry arguments, disallowed
# append roles, sample collection after rollback and few-shot prompts.
"""HTTP-level pins for the v1 (linear trajectory) retry/rollback surface.

v1 is the default session server; these pins lock its production behavior
byte-exactly — every 400 text asserted including interpolated numbers — so
the opt-in v2 tree server (--use-session-server v2) can never silently
change the default path. Additive file: the v1 implementation and its
original test modules stay untouched.
"""

from unittest.mock import patch

import pytest
import requests
from fastapi.responses import JSONResponse

from tests.fast.router import test_sessions
from tests.fast.router.test_sessions import _create_session, _post_chat

from miles.utils.test_utils.mock_sglang_server import MockSGLangServer


# registered here too: in some CI shard combinations the conftest's copy is not visible to this module
router_env = test_sessions.router_env


class TestRollbackPins:
    # [CLASS DOC] Pins for the v1 rollback dispatch. Class constants: U1 is a user question, T1 a
    # tool result for call id t0, T1_DIFF a different tool result for the same call id.
    """Byte-exact pins for the rollback dispatch surface (retry semantics)."""

    U1 = {"role": "user", "content": "What is 1+2?"}
    T1 = {"role": "tool", "content": "tool-result-1", "tool_call_id": "t0"}
    T1_DIFF = {"role": "tool", "content": "tool-result-DIFFERENT", "tool_call_id": "t0"}

    def _turn(self, url: str, session_id: str, messages: list) -> dict:
        # [FUNCTION DOC] HELPER: Posts the messages as a chat request, asserts HTTP 200, and
        # returns the first choice's assistant message.
        resp = _post_chat(url, session_id, {"messages": messages})
        assert resp.status_code == 200
        return resp.json()["choices"][0]["message"]

    def _get(self, url: str, session_id: str) -> dict:
        # [FUNCTION DOC] HELPER: Fetches the session over HTTP (5 second timeout) and returns its
        # JSON body.
        return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()

    def _two_turn_session(self, env) -> tuple[str, dict, dict]:
        # [FUNCTION DOC] HELPER: Creates a session and runs two turns (user; then user, first
        # assistant, tool result), asserting two stored records. Returns the session id and the
        # two assistant messages.
        """Stored history after this: [U1, a1, T1, a2] with 2 records."""
        session_id = _create_session(env.url)
        a1 = self._turn(env.url, session_id, [self.U1])
        a2 = self._turn(env.url, session_id, [self.U1, a1, self.T1])
        assert len(self._get(env.url, session_id)["records"]) == 2
        return session_id, a1, a2

    def test_pure_drop_retry_rolls_back_and_regenerates(self, router_env):
        # [FUNCTION DOC] Re-sending the history without the last assistant returns 200 and
        # regenerates: still two records, the last one ending with the tool result.
        session_id, a1, _ = self._two_turn_session(router_env)

        retry = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1]})

        assert retry.status_code == 200
        records = self._get(router_env.url, session_id)["records"]
        assert len(records) == 2
        assert records[-1]["request"]["messages"][-1] == self.T1

    def test_divergent_retry_rolls_back_and_continues(self, router_env):
        # [FUNCTION DOC] Replacing the tool result returns 200; still two records, the last one
        # ending with the different tool result.
        session_id, a1, _ = self._two_turn_session(router_env)

        retry = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1_DIFF]})

        assert retry.status_code == 200
        records = self._get(router_env.url, session_id)["records"]
        assert len(records) == 2
        assert records[-1]["request"]["messages"][-1] == self.T1_DIFF

    def test_deep_rollback_400_byte_exact_and_state_unchanged(self, router_env):
        # [FUNCTION DOC] After three turns (three records), a retry diverging two generations back
        # returns 400 with the exact message that the discard count 2 exceeds the max rollback
        # steps 1, with stored 6 and request 3 message counts. Records and accumulated token ids
        # are unchanged, and a later pure extension still returns 200.
        session_id, a1, a2 = self._two_turn_session(router_env)
        t2 = {"role": "tool", "content": "tool-result-2", "tool_call_id": "t1"}
        a3 = self._turn(router_env.url, session_id, [self.U1, a1, self.T1, a2, t2])
        before = self._get(router_env.url, session_id)
        assert len(before["records"]) == 3

        resp = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1_DIFF]})

        assert resp.status_code == 400
        assert resp.json()["error"] == (
            "rollback failed: discard_count=2 exceeds max_assistant_rollback_steps=1 "
            "(stored has 6 messages, request has 3 messages)"
        )
        after = self._get(router_env.url, session_id)
        assert after["records"] == before["records"]
        assert after["metadata"]["accumulated_token_ids"] == before["metadata"]["accumulated_token_ids"]

        t3 = {"role": "tool", "content": "tool-result-3", "tool_call_id": "t2"}
        extend = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1, a2, t2, a3, t3]})
        assert extend.status_code == 200

    def test_first_turn_retry_rolls_back_to_empty_and_regenerates(self, router_env):
        # [FUNCTION DOC] Re-sending only the first user message returns 200 and leaves a single
        # record with that request; the regenerated assistant can then be extended with a tool
        # result (200).
        session_id = _create_session(router_env.url)
        self._turn(router_env.url, session_id, [self.U1])

        resp = _post_chat(router_env.url, session_id, {"messages": [self.U1]})

        assert resp.status_code == 200
        regenerated = resp.json()["choices"][0]["message"]
        after = self._get(router_env.url, session_id)
        assert len(after["records"]) == 1
        assert after["records"][0]["request"]["messages"] == [self.U1]

        extend = _post_chat(router_env.url, session_id, {"messages": [self.U1, regenerated, self.T1]})
        assert extend.status_code == 200

    def test_failed_first_turn_then_different_first_request_accepted(self, router_env):
        # [FUNCTION DOC] With the backend patched to reject with 400, the first request fails and
        # the session has no records; a completely different first request is then accepted (200)
        # and stored as the only record.
        """A failed first turn records nothing, so a completely different first
        request is a fresh first turn and must be accepted — the empty-session
        accept-anything semantics that later dispatch rewrites must preserve."""
        session_id = _create_session(router_env.url)

        async def reject(self, request, compute_fn):
            # [FUNCTION DOC] HELPER: Async stand-in for the backend request handler; ignores its
            # inputs and returns a 400 JSON response with the error "context too long".
            return JSONResponse(content={"error": "context too long"}, status_code=400)

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=reject):
            failed = _post_chat(router_env.url, session_id, {"messages": [self.U1]})
        assert failed.status_code == 400
        assert self._get(router_env.url, session_id)["records"] == []

        other_first = {"role": "user", "content": "a completely different opening"}
        resp = _post_chat(router_env.url, session_id, {"messages": [other_first]})
        assert resp.status_code == 200
        records = self._get(router_env.url, session_id)["records"]
        assert len(records) == 1
        assert records[0]["request"]["messages"] == [other_first]

    def test_degenerate_extension_resend_exact_history(self, router_env):
        # [FUNCTION DOC] Re-sending exactly the stored history (user plus first assistant) returns
        # 200 and is treated as an extension, giving two records.
        session_id = _create_session(router_env.url)
        a1 = self._turn(router_env.url, session_id, [self.U1])

        resend = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1]})

        assert resend.status_code == 200
        assert len(self._get(router_env.url, session_id)["records"]) == 2

    @pytest.mark.parametrize(
        "invalid_args",
        [{"input_ids": [999]}, {"chat_template_kwargs": []}, {"lora_path": "wrong-adapter"}],
    )
    def test_invalid_retry_args_400_preserves_history(self, router_env, invalid_args):
        # [FUNCTION DOC] Parametrized over invalid extra arguments (explicit input ids, a
        # non-mapping chat template kwargs, a wrong lora path). A divergent retry carrying them
        # returns 400, leaves the session unchanged, sends nothing to the backend, and a later
        # valid extension still succeeds with three records.
        session_id, a1, a2 = self._two_turn_session(router_env)
        before = self._get(router_env.url, session_id)
        backend_requests = len(router_env.backend.request_log)

        resp = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1_DIFF], **invalid_args})

        assert resp.status_code == 400
        assert self._get(router_env.url, session_id) == before
        assert len(router_env.backend.request_log) == backend_requests
        t2 = {"role": "tool", "content": "next", "tool_call_id": "t1"}
        self._turn(router_env.url, session_id, [self.U1, a1, self.T1, a2, t2])
        assert len(self._get(router_env.url, session_id)["records"]) == 3

    def test_disallowed_append_role_400_preserves_history(self, router_env):
        # [FUNCTION DOC] Appending a developer-role message returns 400 with an error ending in the
        # statement that the TITO fixed template does not support appending this role; the session
        # is unchanged and the backend gets no new request.
        session_id, a1, _ = self._two_turn_session(router_env)
        before = self._get(router_env.url, session_id)
        backend_requests = len(router_env.backend.request_log)

        resp = _post_chat(
            router_env.url, session_id, {"messages": [self.U1, a1, {"role": "developer", "content": "another"}]}
        )

        assert resp.status_code == 400
        error = resp.json()["error"]
        assert error.endswith("; the selected TITO fixed template does not support appending this role")
        assert self._get(router_env.url, session_id) == before
        assert len(router_env.backend.request_log) == backend_requests

    def test_collect_samples_after_rollback_single_sample(self, router_env):
        # [FUNCTION DOC] With the backend patched to drop replay payloads from the meta info, runs
        # the two-turn session, performs a pure-drop retry (200), requests the session's samples
        # and decodes the reply. Asserts a 200 response, no empty reason, exactly one sample with
        # positive response length and a loss mask as long as the response.
        from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
        from miles.utils.types import Sample

        # The class fixture plants fake R3 replay payloads that only the
        # records path tolerates; assembly would try to decode them. This pin
        # is about rollback x assembly, so run it with clean meta_info.
        fixture_response = MockSGLangServer._compute_chat_completions_response

        def clean_meta_response(mock_self, payload: dict) -> dict:
            # [FUNCTION DOC] HELPER: Computes the normal response and removes the routed-experts
            # and indexer-top-k entries from the first choice's meta info; returns the response.
            response = fixture_response(mock_self, payload)
            meta = response["choices"][0]["meta_info"]
            meta.pop("routed_experts", None)
            meta.pop("indexer_topk", None)
            return response

        with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=clean_meta_response):
            session_id, a1, _ = self._two_turn_session(router_env)
            retry = _post_chat(router_env.url, session_id, {"messages": [self.U1, a1, self.T1]})
            assert retry.status_code == 200

            resp = requests.post(f"{router_env.url}/sessions/{session_id}/samples", json={}, timeout=10.0)

        assert resp.status_code == 200
        reply = decode_samples_and_merge_input_sample(resp.content, Sample())
        assert reply.empty_reason is None
        assert len(reply.samples) == 1
        [sample] = reply.samples
        assert sample.response_length > 0
        assert len(sample.loss_mask) == sample.response_length

    def test_few_shot_first_request_divergent_retry_uses_generated_checkpoint(self, router_env):
        # [FUNCTION DOC] Runs a two-turn session starting from a few-shot prompt (user, assistant,
        # user), then a divergent retry with a different tool result; asserts 200, still two
        # records, and the last record ends with the different tool result.
        """Client-carried assistants stay prompt history, so only the first
        backend-generated response anchors the divergent retry."""
        few_shot = [
            {"role": "user", "content": "Q-few-shot"},
            {"role": "assistant", "content": "A-few-shot"},
            {"role": "user", "content": "Q-real"},
        ]
        session_id = _create_session(router_env.url)
        a1 = self._turn(router_env.url, session_id, few_shot)
        self._turn(router_env.url, session_id, [*few_shot, a1, self.T1])
        assert len(self._get(router_env.url, session_id)["records"]) == 2

        retry = _post_chat(router_env.url, session_id, {"messages": [*few_shot, a1, self.T1_DIFF]})

        assert retry.status_code == 200
        records = self._get(router_env.url, session_id)["records"]
        assert len(records) == 2
        assert records[-1]["request"]["messages"][-1] == self.T1_DIFF
