# [MODULE DOC]
# Adapter that lets an unmodified sglang server satisfy the token-id contract of HUD's chat agent.
# An HTTP transport wraps the standard asynchronous transport. For chat-completions POST requests it
# rewrites the JSON body: if the HUD return-token-ids flag was set, it is removed and replaced by
# sglang's flags asking for prompt token ids and meta information (output token log probabilities),
# and any prompt-token-ids continuation field is dropped. For successful responses it parses the
# JSON and, for each choice lacking token ids, synthesises them from the second element of each
# (log probability, token id, text) triple of the output log probabilities. Content length and
# encoding headers are fixed up on rewritten messages. Other requests pass through untouched.
# A factory builds an OpenAI async client using this transport with a 600 second timeout.
"""An AsyncOpenAI client that lets stock sglang serve HUD's token-id contract.

HUD's ``OpenAIChatAgent`` (with ``extra_body={"return_token_ids": True}``)
expects each choice to carry ``token_ids`` (output) and ``prompt_token_ids``.
sglang has both, under different names and shapes:

  request:  return_prompt_token_ids=True  -> choice.prompt_token_ids
            return_meta_info=True         -> choice.meta_info.output_token_logprobs
                                             as (logprob, token_id, text) triples
  (an input ``prompt_token_ids`` continuation field is not accepted; sglang
   re-renders the chat template per turn. Verified: the rendered prompt of
   turn k+1 starts with prompt+output of turn k, so exact stitching survives.)

This transport rewrites the request flags on the way out and synthesizes
``choice.token_ids`` on the way back. No sglang changes, no HUD changes.
"""

from __future__ import annotations

import json

import httpx
from openai import AsyncOpenAI


class _SglangTokenIds(httpx.AsyncBaseTransport):
    # [CLASS DOC] HELPER: Asynchronous HTTP transport that translates between HUD's and sglang's
    # token-id request and response conventions, delegating the actual network I/O to a plain
    # transport it owns. Assumes non-streaming chat completion bodies.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the inner standard asynchronous transport.
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC] HELPER: Sends the request through the inner transport with rewriting for
        # chat completion paths only. Request side: decode the JSON body (empty body treated as an
        # empty object), pop the return-token-ids flag and, if truthy, set the prompt-token-ids and
        # meta-info flags; always drop the prompt-token-ids field; rebuild the request with the
        # re-encoded body and without the old content-length header. Response side, only for status
        # 200 chat responses: read the whole body, parse JSON, add a token-ids list to each choice
        # that has output log probability triples and no token ids, re-encode, drop content-length,
        # content-encoding and transfer-encoding headers, and rebuild the response. Returns the
        # (possibly rewritten) response. A non-JSON 200 body raises a JSON error.
        is_chat = request.url.path.endswith("/chat/completions")
        if is_chat:
            body = json.loads(request.content or b"{}")
            if body.pop("return_token_ids", None):
                body["return_prompt_token_ids"] = True
                body["return_meta_info"] = True
            # HUD's KV-continuation field; sglang re-renders the template instead.
            body.pop("prompt_token_ids", None)
            content = json.dumps(body).encode()
            headers = {k: v for k, v in request.headers.items() if k.lower() != "content-length"}
            request = httpx.Request(request.method, request.url, headers=headers, content=content)

        response = await self._inner.handle_async_request(request)

        if is_chat and response.status_code == 200:
            data = json.loads(await response.aread())
            for choice in data.get("choices", []):
                triples = (choice.get("meta_info") or {}).get("output_token_logprobs") or []
                if triples and "token_ids" not in choice:
                    choice["token_ids"] = [t[1] for t in triples]
            content = json.dumps(data).encode()
            headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
            }
            response = httpx.Response(response.status_code, headers=headers, content=content, request=request)
        return response


def sglang_client(base_url: str, api_key: str = "EMPTY") -> AsyncOpenAI:
    # [FUNCTION DOC] Creates an asynchronous OpenAI client for the base URL with the given API key
    # (default a placeholder) using the token-id adapter transport and a 600 second timeout.
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.AsyncClient(transport=_SglangTokenIds(), timeout=600),
    )
