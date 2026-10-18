# [MODULE DOC] HTTP-level tests of the tinker gateway web application using an in-process ASGI
# client over a service running on a fake backend (the shared service fixture). Covers the full
# training conversation (create session, create model, get info, forward/backward, optimizer
# step, sampler save, asynchronous sampling) with futures polled via the retrieve-future route;
# error status mapping (400 bad input, 403 foreign tenant, 410 unknown future, 400 without an API
# key); failed futures reporting their category; capabilities and telemetry shapes; bearer
# authorization; the weights-info probe; and long polling of pending futures. Authentication in
# these tests is by an API key header whose value identifies the tenant.
"""HTTP responses preserve the SDK conversation and retry/error contract."""

import asyncio

import httpx
import pytest
from tests.fast.tinker.harness import ADAM

from miles.tinker.server.app import build_app


@pytest.fixture
async def client(service):
    # [FUNCTION DOC] HELPER: Async generator fixture. Builds the web application for the service,
    # wraps it in an in-process ASGI transport and an async HTTP client with base url
    # http://gateway, attaches the service to the client as an attribute for tests to reach
    # internals, and yields the client; the client closes on teardown.
    transport = httpx.ASGITransport(app=build_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as http:
        http.service = service
        yield http


def _headers(tenant: str = "tenant-a") -> dict:
    # [FUNCTION DOC] HELPER: Returns request headers carrying the API key header set to the tenant
    # name (default tenant-a).
    return {"X-API-Key": tenant}


async def _poll(client, request_id: str, tenant: str = "tenant-a") -> dict:
    # [FUNCTION DOC] HELPER: Async; posts to the retrieve-future route up to 400 times, 5
    # milliseconds apart, until the response body's type is not try-again, and returns that body.
    # Raises an assertion error if it never settles.
    for _ in range(400):
        response = await client.post(
            "/api/v1/retrieve_future", json={"request_id": request_id}, headers=_headers(tenant)
        )
        body = response.json()
        if body.get("type") != "try_again":
            return body
        await asyncio.sleep(0.005)
    raise AssertionError(f"{request_id} never settled")


async def _model_body(client, tenant: str = "tenant-a", **extra) -> dict:
    # [FUNCTION DOC] HELPER: Async; creates a session over HTTP for the tenant and returns a
    # create-model request body with base model "base", that session id, model sequence id 1 and
    # the extra fields.
    session = (await client.post("/api/v1/create_session", json={}, headers=_headers(tenant))).json()
    return {"base_model": "base", "session_id": session["session_id"], "model_seq_id": 1, **extra}


def _fb_body(model_id: str, seq_id: int) -> dict:
    # [FUNCTION DOC] HELPER: Builds a forward/backward request body for the model and sequence id
    # with one datum: three encoded-text input tokens, target tokens 2, 3, 4, unit weights, and
    # the cross-entropy loss.
    return {
        "model_id": model_id,
        "seq_id": seq_id,
        "forward_backward_input": {
            "data": [
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "loss_fn_inputs": {"target_tokens": [2, 3, 4], "weights": [1.0, 1.0, 1.0]},
                }
            ],
            "loss_fn": "cross_entropy",
        },
    }


async def test_the_training_conversation(client):
    # [FUNCTION DOC] Walks the SDK conversation over HTTP: create session (id starts with
    # session-), create model and poll until its type is create-model, get info (model name base),
    # forward/backward whose result has three logprobs, optimizer step with a gradient norm
    # metric, sampler save returning a path, then an asynchronous sample for 2 samples whose
    # immediate answer lists two sample sequence ids and whose settled result has two sequences.
    session = await client.post("/api/v1/create_session", json={}, headers=_headers())
    assert session.json()["session_id"].startswith("session-")

    body = {"base_model": "base", "session_id": session.json()["session_id"], "model_seq_id": 1}
    created = (await client.post("/api/v1/create_model", json=body, headers=_headers())).json()
    assert (await _poll(client, created["request_id"]))["type"] == "create_model"
    model_id = created["model_id"]

    info = (await client.post("/api/v1/get_info", json={"model_id": model_id}, headers=_headers())).json()
    assert info["model_data"]["model_name"] == "base"

    fb = (await client.post("/api/v1/forward_backward", json=_fb_body(model_id, 1), headers=_headers())).json()
    fb_result = await _poll(client, fb["request_id"])
    assert len(fb_result["loss_fn_outputs"][0]["logprobs"]["data"]) == 3

    optim = (
        await client.post(
            "/api/v1/optim_step",
            json={"model_id": model_id, "seq_id": 2, "adam_params": dict(ADAM)},
            headers=_headers(),
        )
    ).json()
    optim_result = await _poll(client, optim["request_id"])
    assert "grad_norm" in optim_result["metrics"]

    sampler = (
        await client.post(
            "/api/v1/save_weights_for_sampler", json={"model_id": model_id, "seq_id": 3}, headers=_headers()
        )
    ).json()
    sampler_path = (await _poll(client, sampler["request_id"]))["path"]

    sample = (
        await client.post(
            "/api/v1/asample",
            json={
                "model_path": sampler_path,
                "num_samples": 2,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
                "sampling_params": {"max_tokens": 4},
            },
            headers=_headers(),
        )
    ).json()
    assert len(sample["sample_sequence_ids"]) == 2, "asample must answer an UntypedAPIFuture"
    assert len((await _poll(client, sample["request_id"]))["sequences"]) == 2


async def test_bad_input_answers_400(client):
    # [FUNCTION DOC] A forward/backward body for a missing model whose target tokens (2) do not
    # match the input length (3) answers HTTP 400.
    body = _fb_body("model-missing", 1)
    body["forward_backward_input"]["data"][0]["loss_fn_inputs"]["target_tokens"] = [9, 9]
    response = await client.post("/api/v1/forward_backward", json=body, headers=_headers())
    assert response.status_code == 400


async def test_a_foreign_future_answers_403(client):
    # [FUNCTION DOC] Retrieving a future created by tenant a as tenant b answers HTTP 403.
    created = (
        await client.post("/api/v1/create_model", json=await _model_body(client), headers=_headers("tenant-a"))
    ).json()
    response = await client.post(
        "/api/v1/retrieve_future", json={"request_id": created["request_id"]}, headers=_headers("tenant-b")
    )
    assert response.status_code == 403


async def test_an_unknown_future_answers_410(client):
    # [FUNCTION DOC] Retrieving an unknown request id answers HTTP 410.
    response = await client.post("/api/v1/retrieve_future", json={"request_id": "req-gone"}, headers=_headers())
    assert response.status_code == 410


async def test_a_failed_future_reports_the_category(client):
    # [FUNCTION DOC] With the backend's next call failing with boom, creating a model settles as a
    # failure whose body has category server and an error containing boom.
    body = await _model_body(client)
    client.service.backend.fail_next = {"error": "boom"}
    created = (await client.post("/api/v1/create_model", json=body, headers=_headers())).json()
    body = await _poll(client, created["request_id"])
    assert (body["category"], "boom" in body["error"]) == ("server", True)


async def test_capabilities_and_telemetry_shapes(client):
    # [FUNCTION DOC] The capabilities route (no key needed) lists the served base model first;
    # the telemetry route accepts an empty body and answers status accepted.
    capabilities = (await client.get("/api/v1/get_server_capabilities")).json()
    assert capabilities["supported_models"][0]["model_name"] == "base"
    assert (await client.post("/api/v1/telemetry", json={})).json() == {"status": "accepted"}


async def test_a_request_without_an_api_key_is_rejected(client):
    # [FUNCTION DOC] Creating a session without credentials answers HTTP 400 with an error that
    # mentions the API key header.
    response = await client.post("/api/v1/create_session", json={})
    assert response.status_code == 400
    assert "X-API-Key" in response.json()["error"]


async def test_a_bearer_authorization_still_authenticates(client):
    # [FUNCTION DOC] Creating a session with a bearer authorization header (no API key header)
    # answers HTTP 200.
    response = await client.post("/api/v1/create_session", json={}, headers={"Authorization": "Bearer tenant-b"})
    assert response.status_code == 200


async def test_weights_info_answers_the_sdk_resume_probe(client):
    # [FUNCTION DOC] Creates a rank-8 LoRA model, saves weights named ck without overwrite, and
    # asks the weights-info route for the returned path: the answer reports base model "base",
    # is-LoRA true and rank 8.
    created = (
        await client.post(
            "/api/v1/create_model",
            json=await _model_body(client, lora_config={"rank": 8}),
            headers=_headers(),
        )
    ).json()
    await _poll(client, created["request_id"])
    saved = (
        await client.post(
            "/api/v1/save_weights",
            json={"model_id": created["model_id"], "seq_id": 1, "path": "ck", "overwrite": False},
            headers=_headers(),
        )
    ).json()
    path = (await _poll(client, saved["request_id"]))["path"]
    info = (await client.post("/api/v1/weights_info", json={"tinker_path": path}, headers=_headers())).json()
    assert (info["base_model"], info["is_lora"], info["lora_rank"]) == ("base", True, 8)


async def test_retrieve_long_polls_until_settlement(client):
    # [FUNCTION DOC] Creates a pending future directly in the service's future store and starts a
    # retrieve request as a task; after 50 ms the request is still open (no try-again answer).
    # Resolving the future with an optimizer-step result then lets the request finish (2 second
    # timeout) with a body of that type and metrics.
    future = client.service.futures.create("model", "tenant-a")
    poll = asyncio.create_task(
        client.post("/api/v1/retrieve_future", json={"request_id": future.request_id}, headers=_headers())
    )
    await asyncio.sleep(0.05)
    assert not poll.done(), "a pending future must hold the poll open instead of answering try_again"
    client.service.futures.resolve(future.request_id, {"op": "optim_step", "metrics": {"grad_norm": 1.0}})
    response = await asyncio.wait_for(poll, timeout=2)
    assert response.json() == {"type": "optim_step", "metrics": {"grad_norm": 1.0}}
