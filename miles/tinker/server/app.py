# [MODULE DOC]
# HTTP application of the Tinker gateway: a web app exposing the Tinker SDK's API (sessions and
# heartbeats, model creation and info, forward-backward and optimizer commands, checkpoint
# save/load, sampler sessions and sampling, long-poll future retrieval and cancellation, health,
# config and capability stubs). It only translates: request bodies are validated/decoded by the
# encoding and protobuf codec modules, work is delegated to the service object, and results are
# rendered as JSON or protobuf. The tenant identity is the API key itself (no verification).
# Module constants: the long-poll duration for future retrieval (30 seconds) and a mapping from
# command route path to operation name for simple JSON commands.
"""Tinker SDK endpoints with JSON/protobuf translation and tenant authentication."""

import asyncio
from contextlib import suppress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from miles.tinker.core.future import FAILED, PENDING
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import OwnershipError, UserInputError
from miles.tinker.server.encoding import (
    decode_command,
    decode_sample_request,
    render_result,
    validate_create_model,
    validate_create_sampling_session,
)
from miles.tinker.server.proto_codec import (
    PROTO_CONTENT_TYPE,
    PROTO_ENCODERS,
    decode_forward_backward_request,
    maybe_decompress,
)

RETRIEVE_LONG_POLL_S = 30.0

COMMAND_ROUTES = {
    "/api/v1/optim_step": "optim_step",
    "/api/v1/save_weights": "save_state",
    "/api/v1/load_weights": "load_state",
    "/api/v1/save_weights_for_sampler": "save_weights_for_sampler",
}


def _tenant(request: Request) -> str:
    # [FUNCTION DOC] HELPER: The tenant string: the x-api-key header, else the authorization header
    # with a leading "Bearer " removed and whitespace trimmed. Raises a user input error (HTTP 400)
    # if neither yields a non-empty value. The key is used as-is and not authenticated.
    """Require X-API-Key or a bearer token to establish the tenant identity."""
    key = (
        request.headers.get("x-api-key")
        or (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
    )
    if not key:
        raise UserInputError("missing API key: send it in the X-API-Key header")
    return key


def build_app(service: TinkerService) -> FastAPI:
    # [FUNCTION DOC] Create the web app and register all routes as closures over the service;
    # returns the app. Error mapping: user input errors reply HTTP 400 and ownership errors 403,
    # both as a JSON mapping with an error message. The routes are documented one by one below.
    app = FastAPI()

    @app.exception_handler(UserInputError)
    async def _user_error(request: Request, error: UserInputError):
        # [FUNCTION DOC] HELPER: Exception handler: HTTP 400 with the error message.
        return JSONResponse(status_code=400, content={"error": str(error)})

    @app.exception_handler(OwnershipError)
    async def _ownership_error(request: Request, error: OwnershipError):
        # [FUNCTION DOC] HELPER: Exception handler: HTTP 403 with the error message.
        return JSONResponse(status_code=403, content={"error": str(error)})

    @app.get("/api/v1/healthz")
    async def healthz():
        # [FUNCTION DOC] Liveness probe: always replies with status ok (no check of the service).
        return {"status": "ok"}

    @app.post("/api/v1/client/config")
    @app.get("/api/v1/client/config")
    async def client_config():
        # [FUNCTION DOC] Stub for the SDK client config call (GET or POST): replies an empty object.
        return {}

    @app.post("/api/v1/client/dynamic_config")
    @app.get("/api/v1/client/dynamic_config")
    async def client_dynamic_config():
        # [FUNCTION DOC] Stub for the SDK dynamic config call (GET or POST): replies an empty object.
        return {}

    @app.post("/api/v1/telemetry")
    async def telemetry():
        # [FUNCTION DOC] Telemetry sink: discards the body and replies status accepted.
        return {"status": "accepted"}

    @app.post("/api/v1/create_session")
    async def create_session(request: Request):
        # [FUNCTION DOC] Create a client session for the caller's tenant; replies the type name and
        # the new session id. Body ignored.
        session_id = service.create_session(_tenant(request))
        return {"type": "create_session", "session_id": session_id}

    @app.post("/api/v1/session_heartbeat")
    async def session_heartbeat(request: Request):
        # [FUNCTION DOC] Renew the session lease named in the JSON body. Replies HTTP 410 with an
        # error if the session is unknown, expired or owned by another tenant; else a heartbeat
        # acknowledgement.
        tenant = _tenant(request)
        payload = await request.json()
        if not service.heartbeat(tenant, payload["session_id"]):
            return JSONResponse(status_code=410, content={"error": "unknown or expired session"})
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/get_server_capabilities")
    @app.get("/api/v1/get_server_capabilities")
    async def get_server_capabilities():
        # [FUNCTION DOC] Replies the single served base model, marked trainable and sampleable.
        return {"supported_models": [{"model_name": service.config.base_model, "trainable": True, "sampleable": True}]}

    @app.post("/api/v1/create_model")
    async def create_model(request: Request):
        # [FUNCTION DOC] Validate the JSON body against the SDK's create model request model (user
        # error 400 on failure), then ask the service to admit it. Replies the request future id and
        # the new model id; the client polls the future for completion.
        payload = await request.json()
        validate_create_model(payload)
        request_id, model_id = service.create_model(_tenant(request), payload)
        return {"request_id": request_id, "model_id": model_id}

    @app.post("/api/v1/get_info")
    async def get_info(request: Request):
        # [FUNCTION DOC] Look up the model in the body for the caller's tenant (user error if
        # unknown, 403 if another tenant's) and reply its id, base model name (as model name), a
        # LoRA flag that is always true, the LoRA rank, and a model data block with the name.
        payload = await request.json()
        record = service.get_model(_tenant(request), payload["model_id"])
        return {
            "type": "get_info",
            "model_id": record.model_id,
            "model_name": record.base_model,
            "is_lora": True,
            "lora_rank": record.lora_rank,
            "model_data": {"model_name": record.base_model},
        }

    @app.post("/api/v1/forward_backward")
    async def forward_backward(request: Request):
        # [FUNCTION DOC] Submit a forward-backward or forward-only command. A protobuf content
        # type body (decompressed first if zstd-encoded) is decoded by the protobuf codec;
        # otherwise the JSON body is decoded as a forward-backward command. Content errors do not
        # raise here: they become a validation error carried by the command and fail its future
        # later (the model id and sequence id must still be decodable, else a user error). Replies
        # the request future id and the model id.
        if PROTO_CONTENT_TYPE in request.headers.get("content-type", ""):
            body = maybe_decompress(await request.body(), request.headers.get("content-encoding"))
            op, payload = decode_forward_backward_request(body)
        else:
            op, payload = decode_command("forward_backward", await request.json())
        request_id = service.submit(_tenant(request), op, payload)
        return {"request_id": request_id, "model_id": payload["model_id"]}

    for route, route_op in COMMAND_ROUTES.items():

        def _command_handler(route_op: str):
            # [FUNCTION DOC] HELPER: Factory binding the operation name for one command route (a
            # factory avoids the late-binding problem of the loop variable). Returns the handler.
            async def command(request: Request):
                # [FUNCTION DOC] HELPER: Route handler for optimizer step, save weights, load
                # weights and save sampler weights: decode the JSON body for the bound operation
                # (content errors become the command's validation error), submit it to the service,
                # and reply the request future id and model id.
                op, payload = decode_command(route_op, await request.json())
                request_id = service.submit(_tenant(request), op, payload)
                return {"request_id": request_id, "model_id": payload["model_id"]}

            return command

        app.post(route)(_command_handler(route_op))

    @app.post("/api/v1/weights_info")
    async def weights_info(request: Request):
        # [FUNCTION DOC] Reply checkpoint information for the tinker path in the body (base model,
        # rank, layout flags) via the service; path or ownership problems become 400 or 403.
        payload = await request.json()
        return service.weights_info(_tenant(request), payload["tinker_path"])

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(request: Request):
        # [FUNCTION DOC]
        # Long-poll for a request's outcome. Unknown or expired id replies HTTP 410. If the future is
        # pending, wait up to the long-poll time for it to settle (the client retries immediately
        # without backoff, so waiting avoids busy polling). Still pending replies a try-again
        # marker with queue state active. A failed future replies HTTP 200 with its error text and
        # category. A done future is rendered as protobuf if an encoder exists for its operation and
        # the accept header allows protobuf, otherwise as the JSON result rendering.
        payload = await request.json()
        future = service.retrieve_future(_tenant(request), payload["request_id"])
        if future is None:
            return JSONResponse(status_code=410, content={"error": "unknown or expired request"})
        if future.state == PENDING:
            # long-poll: the SDK retries try_again with no backoff, so answer on
            # settlement instead of turning every pending future into a busy-poll
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(future.settled.wait(), timeout=RETRIEVE_LONG_POLL_S)
        if future.state == PENDING:
            return {"type": "try_again", "queue_state": "active"}
        if future.state == FAILED:
            return {"error": future.error, "category": future.error_category}
        encoder = PROTO_ENCODERS.get(future.result["op"])
        if encoder is not None and PROTO_CONTENT_TYPE in request.headers.get("accept", ""):
            return Response(content=encoder(future.result), media_type=PROTO_CONTENT_TYPE)
        return render_result(future.result)

    @app.post("/api/v1/cancel_future")
    async def cancel_future(request: Request):
        # [FUNCTION DOC] Cancel the sampling task of the request id in the body (ignored for
        # unknown ids and training commands) and reply status ok. Another tenant's id gives 403.
        payload = await request.json()
        service.cancel(_tenant(request), payload["request_id"])
        return {"status": "ok"}

    @app.post("/api/v1/create_sampling_session")
    async def create_sampling_session(request: Request):
        # [FUNCTION DOC] Validate the body against the SDK's request model (400 on failure), create
        # the sampling session via the service, and reply the type name and its id.
        payload = await request.json()
        validate_create_sampling_session(payload)
        sampling_session_id = service.create_sampling_session(_tenant(request), payload)
        return {"type": "create_sampling_session", "sampling_session_id": sampling_session_id}

    @app.get("/api/v1/samplers/{sampling_session_id}")
    async def get_sampler(sampling_session_id: str, request: Request):
        # [FUNCTION DOC] Route GET with the sampling session id in the path: reply the sampler
        # description (id, base model, model path) for the caller's tenant.
        return service.get_sampler(_tenant(request), sampling_session_id)

    @app.post("/api/v1/asample")
    async def asample(request: Request):
        # [FUNCTION DOC] Asynchronous sampling. Decodes and validates the JSON body into the
        # internal sampling payload (invalid content raises a 400 here, unlike training commands),
        # starts the sampling through the service, and replies the request future id and the list
        # of sample sequence ids.
        payload = decode_sample_request(await request.json())
        request_id, sequence_ids = service.submit_sample(_tenant(request), payload)
        return {"request_id": request_id, "sample_sequence_ids": sequence_ids}

    return app
