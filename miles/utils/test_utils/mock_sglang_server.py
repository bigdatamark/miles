# [MODULE DOC] HELPER: Mock of the inference server's HTTP API for tests, served by FastAPI under
# a background uvicorn thread. Endpoints: the native generate endpoint, the OpenAI-style chat
# completions endpoint, a health check and an abort endpoint (both always return status ok). Reply
# text comes from a pluggable process function that maps the rendered prompt string to a result
# (text, finish reason, cached token count, extra metadata); tokenization uses a real tokenizer
# loaded by model name; logprobs are synthetic (the i-th output token gets log probability
# minus i divided by 128). Tracks the request payloads and the maximum number of concurrent
# in-flight requests. Also provides a default process function that answers simple "1 plus N"
# arithmetic prompts, and a context manager that starts and stops a server.
import asyncio
import re
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from miles.utils.http_utils import find_available_port
from miles.utils.misc import NodeProbeMixin
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@dataclass(frozen=True)
class ProcessResultMetaInfo:
    # [CLASS DOC] HELPER: Immutable optional extra metadata fields that a canned result may add to
    # the response metadata: weight version, list of weight version records, routed experts
    # payload, and three speculative-decoding counters (correct drafts, proposed drafts, verify
    # count). All default to absent.
    weight_version: str | None = None
    weight_versions: list[dict[str, str | int]] | None = None
    routed_experts: str | None = None
    spec_num_correct_drafts: int | None = None
    spec_num_proposed_drafts: int | None = None
    spec_verify_ct: int | None = None

    def to_dict(self) -> dict:
        # [FUNCTION DOC] HELPER: Returns the fields as a mapping, omitting those that are none.
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ProcessResult:
    # [CLASS DOC] HELPER: Immutable canned reply from the process function: the reply text, the
    # finish reason (default "stop"), the number of cached prompt tokens to report (default 0),
    # and extra metadata fields (default all absent).
    text: str
    finish_reason: str = "stop"
    cached_tokens: int = 0
    meta_info: ProcessResultMetaInfo = ProcessResultMetaInfo()


ProcessFn = Callable[[str], ProcessResult]


class MockSGLangServer:
    # [CLASS DOC] HELPER: The mock server object. Lifecycle: construct (loads tokenizer, builds the
    # app and routes), start (runs uvicorn in a background thread bound on all interfaces), stop.
    # Public state: tokenizer, process function, host and port, simulated latency, the list of
    # received request payloads (both endpoints, in arrival order), and the peak concurrency.
    def __init__(
        self,
        model_name: str,
        process_fn: ProcessFn,
        host: str | None,
        port: int,
        latency: float = 0.0,
        chat_template_path: str | None = None,
    ):
        # [FUNCTION DOC] HELPER: Loads the tokenizer for the model name (optionally overriding the
        # chat template from a file path, trusting remote code), stores the process function,
        # resolves the advertised host (given, else the node IP) and port (given and truthy, else
        # the first available port from 30000 upward; note a port of zero or none both trigger the
        # search), stores the artificial per-request latency in seconds, creates the FastAPI app
        # and an empty request log and concurrency counter, and registers the routes. Does not
        # start listening.
        self.tokenizer = load_tokenizer(model_name, chat_template_path=chat_template_path, trust_remote_code=True)
        self.process_fn = process_fn
        self.host = host or NodeProbeMixin._get_node_ip()
        self.port = port or find_available_port(30000)
        self.latency = latency

        self.app = FastAPI()
        self._server: UvicornThreadServer | None = None

        self.request_log: list[dict] = []
        self._concurrency = Counter()

        self._setup_routes()

    @property
    def max_concurrent(self) -> int:
        # [FUNCTION DOC] HELPER: Peak number of simultaneously in-flight generate-like requests
        # since construction or the last reset.
        return self._concurrency.max_value

    def reset_stats(self):
        # [FUNCTION DOC] HELPER: Clears the request log and resets the concurrency counter (current
        # and peak) to zero.
        self.request_log.clear()
        self._concurrency.reset()

    def start(self):
        # [FUNCTION DOC] HELPER: Creates a uvicorn thread server for the app on the advertised
        # port, binding to all interfaces, and starts it (see that class for readiness behavior).
        self._server = UvicornThreadServer(self.app, host=self.host, port=self.port, bind_host="0.0.0.0")
        self._server.start()

    def stop(self):
        # [FUNCTION DOC] HELPER: Stops the background server if it was started; otherwise does
        # nothing.
        if self._server is not None:
            self._server.stop()

    @property
    def url(self) -> str:
        # [FUNCTION DOC] HELPER: Base http URL using the advertised host and port.
        return f"http://{self.host}:{self.port}"

    def _setup_routes(self):
        # [FUNCTION DOC] HELPER: Registers four routes on the app: POST generate and POST chat
        # completions (both through the shared request handler with different response
        # computers), GET health and POST abort request (both return a JSON object whose status is
        # ok and ignore their input; abort does nothing).
        @self.app.post("/generate")
        async def generate(request: Request):
            # [FUNCTION DOC] HELPER: Route handler for the native generate endpoint.
            return await self._handle_generate_like_request(request, self._compute_generate_response)

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            # [FUNCTION DOC] HELPER: Route handler for the chat completions endpoint.
            return await self._handle_generate_like_request(request, self._compute_chat_completions_response)

        @self.app.get("/health")
        async def health():
            # [FUNCTION DOC] HELPER: Route handler returning the ok status object.
            return JSONResponse(content={"status": "ok"})

        @self.app.post("/abort_request")
        async def abort_request(_request: Request):
            # [FUNCTION DOC] HELPER: Route handler that accepts and ignores an abort request,
            # returning the ok status object.
            return JSONResponse(content={"status": "ok"})

    async def _handle_generate_like_request(self, request: Request, compute_fn: Callable[[dict], dict]):
        # [FUNCTION DOC] HELPER: Shared handler: parses the JSON body, appends it to the request
        # log, then inside the concurrency tracker optionally sleeps the configured latency
        # (asynchronously) and calls the synchronous response computer on the payload; returns its
        # result as a JSON response with status 200. Exceptions from the computer propagate and
        # become server errors.
        payload = await request.json()
        self.request_log.append(payload)
        with self._concurrency.track():
            if self.latency > 0:
                await asyncio.sleep(self.latency)
            response = compute_fn(payload)
        return JSONResponse(content=response)

    def _compute_generate_response(self, payload: dict) -> dict:
        # [FUNCTION DOC] HELPER: Builds the native generate response. Asserts that the payload's
        # return-logprob flag (default true) is true. Decodes the input token ids (default empty)
        # to a prompt string keeping special tokens, calls the process function, and encodes the
        # reply text without special tokens to get output ids. Metadata: finish reason as an object
        # with a type (plus the completion token count under a length key when the reason is
        # "length"), prompt token count, cached token count from the result, completion token
        # count, output token logprobs as pairs of (synthetic logprob, token id) with the i-th
        # logprob equal to minus i over 128, and the optional extra metadata fields. Returns a
        # mapping with the reply text and that metadata. Sampling parameters in the payload are
        # ignored.
        assert payload.get("return_logprob", True) is True, "MockSGLangServer requires return_logprob=True"
        input_ids = payload.get("input_ids", [])

        prompt_str = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        process_result = self.process_fn(prompt_str)
        output_ids = self.tokenizer.encode(process_result.text, add_special_tokens=False)

        prompt_tokens = len(input_ids)
        completion_tokens = len(output_ids)

        finish_reason_dict = {"type": process_result.finish_reason}
        if process_result.finish_reason == "length":
            finish_reason_dict["length"] = completion_tokens

        output_token_logprobs = [(-1 / 128 * i, token_id) for i, token_id in enumerate(output_ids)]

        meta_info = {
            "finish_reason": finish_reason_dict,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": process_result.cached_tokens,
            "completion_tokens": completion_tokens,
            "output_token_logprobs": output_token_logprobs,
            **process_result.meta_info.to_dict(),
        }

        return {"text": process_result.text, "meta_info": meta_info}

    def _compute_chat_completions_response(self, payload: dict) -> dict:
        # [FUNCTION DOC] HELPER: Builds a chat completion response. Renders the messages (default
        # empty) with the tokenizer's chat template as text with a generation prompt and the
        # payload's tools. If the payload asks to return prompt token ids, uses the payload's input
        # ids when given, else encodes the rendered prompt without special tokens. Calls the process
        # function on the rendered prompt and encodes its text to output ids; per-token logprob
        # entries carry the token string, id and synthetic logprob (minus i over 128). Tool
        # parsing: only when tools are given and the finish reason is "stop", parses the reply
        # text with the qwen2.5-style function call parser; if tool calls are found, the content
        # becomes the parser's remaining normal text, the finish reason becomes "tool_calls", and
        # each call becomes an entry with id "call" plus a five-digit zero-padded position, the
        # tool's index in the tools list, type function, name and arguments (an empty object string
        # when there are none). Otherwise the content is the reply text (when tools are given and
        # the reason is "stop" but no calls parse, the content is the parser's remaining text).
        # Builds one choice (index 0) with assistant message (content and tool calls, none when
        # absent), logprobs content, finish reason, and a metadata object with output token
        # logprobs pairs, completion count and extra metadata fields; plus prompt token ids if
        # requested. The usage prompt count is the payload input ids length when given, else the
        # encoded rendered prompt length. Returns an OpenAI-shaped completion with a random
        # eight-hex-digit id, object type, creation time in seconds, the fixed model name
        # "mock-model", the choice and usage totals. Request sampling parameters are ignored.
        messages = payload.get("messages", [])
        tools = payload.get("tools")

        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )

        prompt_ids = None
        if payload.get("return_prompt_token_ids"):
            input_ids = payload.get("input_ids")
            if input_ids is not None:
                prompt_ids = list(input_ids)
            else:
                prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

        process_result = self.process_fn(prompt_str)
        output_ids = self.tokenizer.encode(process_result.text, add_special_tokens=False)

        logprobs_content = [
            {
                "token": self.tokenizer.convert_ids_to_tokens(tid),
                "token_id": tid,
                "logprob": -1 / 128 * i,
            }
            for i, tid in enumerate(output_ids)
        ]

        finish_reason = process_result.finish_reason
        tool_calls = None
        if tools and finish_reason == "stop":
            parser = FunctionCallParser(
                tools=TypeAdapter(list[Tool]).validate_python(tools),
                tool_call_parser="qwen25",
            )
            message_content, parsed_calls = parser.parse_non_stream(process_result.text)
            if parsed_calls:
                finish_reason = "tool_calls"
                # Mirror SGLang's non-streaming wire shape: ToolCall always
                # serializes ``index`` (the tool's position in the tools list,
                # via ToolCallItem.tool_index; -1 when unknown).
                tool_calls = [
                    {
                        "id": f"call{i:05d}",
                        "index": call.tool_index,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.parameters or "{}"},
                    }
                    for i, call in enumerate(parsed_calls)
                ]
        else:
            message_content = process_result.text

        output_token_logprobs = [(-1 / 128 * i, tid) for i, tid in enumerate(output_ids)]

        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": message_content,
                "tool_calls": tool_calls,
            },
            "logprobs": {"content": logprobs_content},
            "finish_reason": finish_reason,
            "meta_info": {
                "output_token_logprobs": output_token_logprobs,
                "completion_tokens": len(output_ids),
                **process_result.meta_info.to_dict(),
            },
        }
        if prompt_ids is not None:
            choice["prompt_token_ids"] = prompt_ids

        # Real SGLang chat responses always carry ``usage``; clients that
        # validate the full ChatCompletionResponse schema require it.
        if payload.get("input_ids") is not None:
            prompt_token_count = len(payload["input_ids"])
        else:
            prompt_token_count = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": len(output_ids),
                "total_tokens": prompt_token_count + len(output_ids),
            },
        }


class Counter:
    # [CLASS DOC] HELPER: Tracks the current and peak number of simultaneously active scopes.
    # Not thread-safe; intended for single event-loop use.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with current and peak both zero.
        self._current = 0
        self._max = 0

    @property
    def max_value(self) -> int:
        # [FUNCTION DOC] HELPER: The peak concurrent count seen so far.
        return self._max

    def reset(self):
        # [FUNCTION DOC] HELPER: Resets both current and peak counts to zero (do not call while
        # scopes are active, since their exits would drive the current count negative).
        self._current = 0
        self._max = 0

    @contextmanager
    def track(self):
        # [FUNCTION DOC] HELPER: Context manager: on entry increments the current count and
        # updates the peak; on exit (also on error) decrements the current count.
        self._current += 1
        self._max = max(self._max, self._current)
        try:
            yield
        finally:
            self._current -= 1


def default_process_fn(prompt: str) -> ProcessResult:
    # [FUNCTION DOC] HELPER: Default reply function. Searches the prompt for the phrase "What is
    # 1+N?" (N one or more digits; the first match wins). If found, replies with the sum 1+N
    # wrapped in a LaTeX boxed expression with finish reason "stop"; otherwise replies "I don't
    # understand." with finish reason "stop".
    match = re.search(r"What is 1\+(\d+)\?", prompt)
    if match:
        num = int(match.group(1))
        ans = 1 + num
        return ProcessResult(text=f"\\boxed{{{ans}}}", finish_reason="stop")
    return ProcessResult(text="I don't understand.", finish_reason="stop")


@contextmanager
def with_mock_server(
    model_name: str = "Qwen/Qwen3-0.6B",
    process_fn: ProcessFn = default_process_fn,
    host: str | None = None,
    port: int | None = None,
    latency: float = 0.0,
):
    # [FUNCTION DOC] HELPER: Context manager that builds a mock server with the given model name
    # (default a small Qwen3 model), process function (default the arithmetic one), optional
    # host and port (port none means auto-pick), and latency; starts it, yields the server, and
    # always stops it on exit (even on error). If construction raises, nothing is yielded.
    server = MockSGLangServer(
        model_name=model_name,
        process_fn=process_fn,
        host=host,
        port=port,
        latency=latency,
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()
