# [MODULE DOC]
# Offline tests (skipped unless the hud package is importable) for the HUD example's sglang
# compatibility http transport shim. The shim wraps an inner transport: for chat completion
# requests it rewrites HUD's vLLM-style token flags into sglang's (drops the return-token-ids and
# prompt-token-ids fields, sets return-prompt-token-ids and return-meta-info) and synthesizes
# per-choice token ids from the sglang output-token log probability meta info in the response;
# other endpoints pass through. The inner transport is a canned fake that records the body it saw.
"""Offline tests: no GPU, no network, no Daytona, no live model.

sglang_compat.py's translation: the request/response rewrite between
HUD's vLLM-flavoured token flags and stock sglang.
The HUD side of the seam is faked at exactly the boundary the real code
touches.
"""

from __future__ import annotations

import asyncio
import json

import pytest

hud = pytest.importorskip("hud", reason="pip install hud -- the recipe's one extra dependency")

from examples.experimental.hud.sglang_compat import _SglangTokenIds


def _run(coro):
    # [FUNCTION DOC] HELPER: Runs the coroutine to completion on the current event loop and returns its result.
    return asyncio.get_event_loop().run_until_complete(coro)


class _CannedTransport:
    # [CLASS DOC] HELPER: Fake async http transport returning a fixed json payload with status 200
    # and remembering the decoded json body of the last request (empty mapping if the body is empty).
    """Fake inner transport: captures the outgoing request, returns a canned
    sglang-shaped response."""

    def __init__(self, payload):
        # [FUNCTION DOC] HELPER: Stores the payload; the seen body starts as none.
        self.payload = payload
        self.seen_body = None

    async def handle_async_request(self, request):
        # [FUNCTION DOC] HELPER: Decodes the request body as json (empty mapping if empty), stores
        # it, and returns a 200 response tied to the request with the canned payload as json.
        import httpx

        self.seen_body = json.loads(request.content or b"{}")
        return httpx.Response(200, json=self.payload, request=request)


def _roundtrip(request_body, response_payload):
    # [FUNCTION DOC] HELPER: Builds the shim with a canned inner transport, sends a post to the
    # chat completions url with the json-encoded request body, and returns a pair: the body the
    # inner transport saw, and the json-decoded response body returned by the shim.
    import httpx

    shim = _SglangTokenIds()
    canned = _CannedTransport(response_payload)
    shim._inner = canned
    request = httpx.Request("POST", "http://x/v1/chat/completions", content=json.dumps(request_body).encode())
    response = _run(shim.handle_async_request(request))
    return canned.seen_body, json.loads(response.content)


def test_shim_rewrites_flags_and_synthesizes_token_ids():
    # [FUNCTION DOC] Request with return-token-ids and prompt-token-ids, response with two output
    # token log probability triples. Asserts the sent body lacks the two vLLM fields, has the
    # return-prompt-token-ids and return-meta-info flags true, and the response choice gained token
    # ids [5, 6] (the second element of each triple).
    sent, got = _roundtrip(
        {"model": "m", "return_token_ids": True, "prompt_token_ids": [1, 2]},
        {"choices": [{"meta_info": {"output_token_logprobs": [[-0.1, 5, "a"], [-0.2, 6, "b"]]}}]},
    )
    assert "return_token_ids" not in sent and "prompt_token_ids" not in sent
    assert sent["return_prompt_token_ids"] is True and sent["return_meta_info"] is True
    assert got["choices"][0]["token_ids"] == [5, 6]


def test_shim_leaves_other_endpoints_alone():
    # [FUNCTION DOC] A get request to the models endpoint goes through; asserts the inner
    # transport saw an empty body mapping (nothing rewritten).
    import httpx

    shim = _SglangTokenIds()
    canned = _CannedTransport({"data": []})
    shim._inner = canned
    request = httpx.Request("GET", "http://x/v1/models")
    _run(shim.handle_async_request(request))
    assert canned.seen_body == {}
