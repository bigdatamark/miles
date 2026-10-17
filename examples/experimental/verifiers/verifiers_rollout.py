# [MODULE DOC] Rollout adapter that lets Miles train on Verifiers V1 environments. Verifiers runs
# the grouped episodes and computes rewards; this module supplies (1) a transport that makes the
# Verifiers/Renderers training client talk to Miles' SGLang router (token-id generate requests
# with per-token log probabilities, Miles' prompt/response/context limits and stop settings),
# (2) a train client subclass that turns unsupported request kinds into clear provider errors and
# keeps local checkpoints usable by restoring a registered renderer identity, (3) conversion of
# Verifiers traces/branches into Miles samples (tokens, loss mask from the sampled-token mask,
# log probs, status, metadata), (4) the rollout function class used by the refactored rollout
# manager (training batches with over-sampling, dynamic filtering and optional Miles reward
# functions, and evaluation over the whole taskset), and (5) a legacy function entry point wrapping
# a persistent instance of that class. Verifiers settings come from the TOML file named by the
# environment variable VERIFIERS_CONFIG. Requires Python 3.11+ and supported verifiers (at least
# 0.2.0, below 0.2.1) and renderers (at least 0.1.8, below 0.1.11) versions.
"""Verifiers V1 rollout adapter: Verifiers owns grouped episode execution and
reward computation, Miles owns everything around it.

Wire it up with ``--rollout-function-path verifiers_rollout.VerifiersRolloutFn``
(or ``.generate_rollout`` under ``MILES_USE_LEGACY_ROLLOUT_V1=1``) plus
``--disable-rollout-global-dataset``, and point ``VERIFIERS_CONFIG`` at a
Verifiers ``EnvConfig`` TOML file. ``run.py`` in this directory does all of
that; see README.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import uuid
from argparse import Namespace
from collections import OrderedDict
from collections.abc import Iterable
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.rollout.filter_hub.common_filters import apply_preput_filters
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.utils.lora.utils import LORA_ADAPTER_NAME, is_lora_enabled
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_MIN_VERIFIERS_VERSION = Version("0.2.0")
_MAX_VERIFIERS_VERSION = Version("0.2.1")
_MIN_RENDERERS_VERSION = Version("0.1.8")
_MAX_RENDERERS_VERSION = Version("0.1.11")
_UNSUPPORTED_ERROR_PREFIX = "Miles' Verifiers adapter does not support"
_CONFIG_ENV_VAR = "VERIFIERS_CONFIG"


def _load_config_data(path: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Reads a Verifiers config file and returns its parsed contents as a
    # mapping. Raises a value error if the file suffix (case-insensitive) is not toml; raises the
    # optional-dependency runtime error on Python older than 3.11 (the TOML parser is imported
    # lazily); raises a value error if the root is not a mapping (cannot happen with the parser).
    # Reads the file from disk.
    config_path = Path(path)
    if config_path.suffix.lower() != ".toml":
        raise ValueError("--verifiers-config must point to a Verifiers TOML config.")
    if sys.version_info < (3, 11):
        raise _optional_dependency_error()

    import tomllib

    data = tomllib.loads(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the root.")
    return data


def _optional_dependency_error() -> RuntimeError:
    # [FUNCTION DOC] HELPER: Builds (does not raise) the runtime error telling the user that
    # Verifiers rollouts need Python 3.11 or newer and the optional extra of the package.
    return RuntimeError(
        "Verifiers rollouts require Python 3.11+ and the optional dependencies. "
        "Install Miles with `pip install -e '.[verifiers]'`."
    )


def _installed_version(package: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the installed version text of a distribution; if it is not
    # installed raises the optional-dependency runtime error (chained to the lookup error).
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError as error:
        raise _optional_dependency_error() from error


def _check_version(
    package: str,
    raw_version: str,
    minimum: Version,
    maximum: Version | None = None,
    *,
    maximum_reason: str | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Checks a version text against an inclusive minimum and an exclusive
    # optional maximum. Raises a runtime error if it cannot be parsed, is below the minimum, or (when
    # a maximum is given) is at or above it; the upper-bound message appends the optional reason.
    try:
        installed = Version(raw_version)
    except InvalidVersion as error:
        raise RuntimeError(f"Could not parse installed {package} version {raw_version!r}.") from error
    if installed < minimum:
        raise RuntimeError(f"Verifiers rollouts require {package}>={minimum}; found {installed}.")
    if maximum is not None and installed >= maximum:
        message = f"Verifiers rollouts require {package}>={minimum},<{maximum}; found {installed}."
        if maximum_reason is not None:
            message += f" {maximum_reason}"
        raise RuntimeError(message)


def _import_verifiers():
    # [FUNCTION DOC] HELPER: Verifies the environment and lazily imports the Verifiers V1 runtime.
    # Raises the optional-dependency error on Python older than 3.11 or if an import fails; raises a
    # runtime error if the verifiers version is outside [0.2.0, 0.2.1) (upper bound explained by an
    # OpenAI dependency conflict with SGLang) or the renderers version is outside [0.1.8, 0.1.11)
    # (upper bound explained by removed renderer pool APIs). Returns a simple namespace exposing the
    # environment config class, environment class, decorated-function discovery helper, model
    # context class, overlong-prompt error, provider error, sampling config class and train client
    # class.
    if sys.version_info < (3, 11):
        raise _optional_dependency_error()
    _check_version(
        "verifiers",
        _installed_version("verifiers"),
        _MIN_VERIFIERS_VERSION,
        _MAX_VERIFIERS_VERSION,
        maximum_reason="Verifiers 0.2.1 requires OpenAI>=2.9, while SGLang 0.5.15 pins OpenAI==2.6.1.",
    )
    _check_version(
        "renderers",
        _installed_version("renderers"),
        _MIN_RENDERERS_VERSION,
        _MAX_RENDERERS_VERSION,
        maximum_reason="Renderers 0.1.11 removes the renderer pool APIs used by Miles and Verifiers 0.2.0.",
    )
    try:
        from verifiers.v1 import EnvConfig, Environment, ModelContext, SamplingConfig
        from verifiers.v1.clients.train import TrainClient
        from verifiers.v1.decorators import discover_decorated
        from verifiers.v1.errors import OverlongPromptError, ProviderError
    except ImportError as error:
        raise _optional_dependency_error() from error
    return SimpleNamespace(
        EnvConfig=EnvConfig,
        Environment=Environment,
        discover_decorated=discover_decorated,
        ModelContext=ModelContext,
        OverlongPromptError=OverlongPromptError,
        ProviderError=ProviderError,
        SamplingConfig=SamplingConfig,
        TrainClient=TrainClient,
    )


def _renderer_identity(checkpoint: str) -> str | None:
    # [FUNCTION DOC] HELPER: Finds the registered renderer model id for a checkpoint name or path,
    # or none. Returns the input itself if it is a key of the renderer registry. Otherwise collects
    # candidates: for each path component that starts with "models--" (hub cache layout) the
    # organization/name form obtained by dropping the prefix and turning double hyphens into
    # slashes; plus every registered id whose last segment equals the path's final component.
    # Keeps the candidates that are registered ids (sorted, deduplicated); none -> none; if all kept
    # ids use the same renderer returns the first in sorted order, else none (ambiguous).
    from renderers.base import MODEL_RENDERER_MAP

    if checkpoint in MODEL_RENDERER_MAP:
        return checkpoint

    path = Path(checkpoint)
    candidates = []
    for part in path.parts:
        if part.startswith("models--"):
            candidates.append(part.removeprefix("models--").replace("--", "/"))
    candidates.extend(model_id for model_id in MODEL_RENDERER_MAP if model_id.rsplit("/", 1)[-1] == path.name)
    matches = sorted(set(candidates) & MODEL_RENDERER_MAP.keys())
    if not matches:
        return None
    renderer_names = {MODEL_RENDERER_MAP[model_id] for model_id in matches}
    return matches[0] if len(renderer_names) == 1 else None


def _train_client(
    runtime,
    args: Namespace,
    model: str,
    pool_size: int,
    *,
    router_args: Namespace | None = None,
):
    # [FUNCTION DOC] HELPER: Builds the training client. Inputs: the lazily imported runtime
    # namespace, rollout args, the model name or path, renderer pool size, and optionally the args
    # object used for the router address (so a copied args object can follow the live one). The
    # tokenizer source is the sglang tokenizer path option if set, else the model; the renderer
    # identity is looked up for the model, else for the tokenizer source. Defines a subclass of the
    # runtime train client that: converts not-implemented errors and tool-unsupported value errors
    # raised during response generation into provider errors with status 400 and the adapter's
    # unsupported prefix (other value errors re-raised); rejects streaming relay and auxiliary
    # dialect relay with such provider errors; and, when an identity was found, creates the renderer
    # pool lazily once (loading the tokenizer from the renderer model name or requested model,
    # overriding its name with the registered identity, creating renderers with the client config and
    # chat template kwargs of the FIRST request) and reuses it afterwards; without an identity it
    # uses the parent behavior. Returns an instance wired to the Miles SGLang transport.
    tokenizer_source = getattr(args, "sglang_tokenizer_path", None) or model
    identity = _renderer_identity(model) or _renderer_identity(tokenizer_source)

    # TrainClient uses one path for both tokenizer loading and renderer lookup.
    # Miles checkpoints are commonly local snapshots, so keep local tokenizer
    # files while restoring the canonical identity used by Renderers' registry.
    class TrainClient(runtime.TrainClient):
        # [CLASS DOC] HELPER: Local subclass of the runtime train client; see the enclosing
        # function for the overridden behavior.
        @staticmethod
        def _unsupported_request(kind: str):
            # [FUNCTION DOC] HELPER: Builds (does not raise) a provider error with status 400 whose
            # message is the unsupported prefix plus the kind of request.
            return runtime.ProviderError(
                f"{_UNSUPPORTED_ERROR_PREFIX} {kind}.",
                status_code=400,
            )

        async def get_response(self, *args, **kwargs):
            # [FUNCTION DOC] Calls the parent method and returns its response. A not-implemented
            # error becomes a provider error (status 400, unsupported prefix and the original
            # text); a value error whose text contains "does not support tools" becomes a provider
            # error advising a renderer-registered model identity; other value errors propagate.
            try:
                return await super().get_response(*args, **kwargs)
            except NotImplementedError as error:
                raise runtime.ProviderError(
                    f"{_UNSUPPORTED_ERROR_PREFIX} this request: {error}",
                    status_code=400,
                ) from error
            except ValueError as error:
                if "does not support tools" not in str(error):
                    raise
                raise runtime.ProviderError(
                    f"{_UNSUPPORTED_ERROR_PREFIX} tools with this renderer: {error} "
                    "Use a Renderers-registered model identity in "
                    "--hf-checkpoint or --sglang-tokenizer-path.",
                    status_code=400,
                ) from error

        async def relay(self, *args, **kwargs):
            # [FUNCTION DOC] Always raises the provider error for unsupported streaming requests.
            raise self._unsupported_request("streaming requests")

        async def relay_aux(self, *args, **kwargs):
            # [FUNCTION DOC] Always raises the provider error for unsupported auxiliary dialect
            # routes.
            raise self._unsupported_request("auxiliary dialect routes")

        def _renderer_pool(self, requested_model, *, chat_template_kwargs=None):
            # [FUNCTION DOC] HELPER: Returns the renderer pool. With no registered identity,
            # delegates to the parent. Otherwise creates the pool on first use (see the enclosing
            # function) and returns the cached pool afterwards, ignoring later template kwargs.
            if identity is None:
                return super()._renderer_pool(
                    requested_model,
                    chat_template_kwargs=chat_template_kwargs,
                )
            if self._pool is None:
                from renderers import RendererPool, create_renderer
                from renderers.base import load_tokenizer

                source = self.renderer_model_name or requested_model

                def factory():
                    # [FUNCTION DOC] HELPER: Pool factory: loads the tokenizer from the source,
                    # sets its name to the registered identity, and creates a renderer from it with
                    # the client config and chat template kwargs; returns the renderer.
                    tokenizer = load_tokenizer(source)
                    tokenizer.name_or_path = identity
                    return create_renderer(
                        tokenizer,
                        self.config,
                        chat_template_kwargs=chat_template_kwargs,
                    )

                self._pool = RendererPool(factory, size=self.pool_size)
            return self._pool

    return TrainClient(
        MilesSGLangTransport(args, router_args=router_args),
        pool_size=pool_size,
        renderer_model_name=tokenizer_source,
    )


def _generate_url(args: Namespace, endpoint: str = "/generate") -> str:
    # [FUNCTION DOC] HELPER: Builds the router url "http://<ip>:<port><endpoint>". The address is the
    # "default" entry of the model-routers mapping (a pair of ip and port) when that mapping exists
    # and has that entry; otherwise the single router ip and port attributes.
    routers = getattr(args, "sglang_model_routers", None)
    if routers and "default" in routers:
        ip, port = routers["default"]
    else:
        ip, port = args.sglang_router_ip, args.sglang_router_port
    return f"http://{ip}:{port}{endpoint}"


async def _sglang_worker_urls(args: Namespace) -> list[str]:
    # [FUNCTION DOC] HELPER: Lists the SGLang worker urls behind the router. Unless the Miles router
    # is in use, first asks the router's workers endpoint and returns the "url" of each listed worker;
    # any failure there is logged at debug level and the Miles router's list-workers endpoint is used
    # instead, returning its "urls" list. Network side effects; errors from the fallback propagate.
    from miles.utils.http_utils import get

    router_url = _generate_url(args).removesuffix("/generate")
    if not getattr(args, "use_miles_router", False):
        try:
            response = await get(f"{router_url}/workers")
            return [worker["url"] for worker in response["workers"]]
        except Exception:
            logger.debug("SGLang /workers lookup failed; trying Miles /list_workers.", exc_info=True)
    response = await get(f"{router_url}/list_workers")
    return list(response["urls"])


def _finish_reason(output: dict[str, Any]) -> str:
    # [FUNCTION DOC] HELPER: Extracts the finish reason from an engine output's meta info (a mapping
    # with a type entry, or a plain text). Raises a runtime error for "abort". Returns "stop",
    # "length" or "content_filter" unchanged and maps anything else (including missing) to "stop".
    finish_reason = (output.get("meta_info") or {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type")
    if finish_reason == "abort":
        raise RuntimeError("SGLang aborted the Verifiers generation request.")
    return finish_reason if finish_reason in {"stop", "length", "content_filter"} else "stop"


class MilesSGLangTransport:
    # [CLASS DOC] HTTP-client-shaped transport used by the Renderers/Verifiers train client. It
    # accepts requests in the token-id generate format, applies Miles' limits and settings, sends
    # them to the SGLang router's generate endpoint and returns responses in the format the client
    # expects (completion token ids with log probabilities and a finish reason). Keeps a bounded
    # least-recently-used record of session ids to check the initial prompt length once per session.
    """Translate Renderers' /inference/v1/generate wire format to Miles' SGLang endpoint."""

    def __init__(self, args: Namespace, *, router_args: Namespace | None = None):
        # [FUNCTION DOC] Stores the rollout args and the args used to find the router (defaults to
        # the same object), an empty ordered record of seen session ids and its size cap of 10000.
        self.args = args
        self.router_args = router_args or args
        self._seen_sessions: OrderedDict[str, None] = OrderedDict()
        self._session_cache_size = 10_000

    @property
    def base_url(self) -> str:
        # [FUNCTION DOC] Property: the router base url ending in "/v1", computed on every access
        # from the router args (the address is only known after SGLang starts).
        # The refactored RolloutManager constructs rollout functions before it
        # starts SGLang and fills in the router address.
        return f"{_generate_url(self.router_args, '').rstrip('/')}/v1"

    async def get(self, _path: str, **_kwargs) -> dict[str, list[Any]]:
        # [FUNCTION DOC] Answers every GET probe with an empty model list (so no engine context cap
        # is reported); Miles' own limits are enforced when a request is posted.
        # Renderers probes this endpoint for an engine context cap (max_model_len).
        # Miles owns separate prompt and response limits, so the transport
        # enforces them at POST time.
        return {"data": []}

    def _sampling_params(self, raw: dict[str, Any], prompt_len: int) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Converts the client's sampling mapping to SGLang's. Copies it, drops
        # the log-probs entry, renames max tokens to max new tokens, min tokens to min new tokens and
        # seed to sampling seed. Miles settings win over the request: the default stop strings from
        # args are added only if the request has none; the stop token ids become the request's
        # (if any) followed by the args' ones, deduplicated keeping order; special-token skipping
        # comes from args; stop trimming is disabled, spaces between special tokens are off and the
        # sample count is 1. Token budget: the requested max new tokens (default the response
        # limit) is cut to the context limit minus the prompt length when a context limit is set,
        # then to the response limit. If the budget is not positive, raises the overlong-prompt error
        # (imported lazily). Returns the new mapping.
        values = dict(raw)
        values.pop("logprobs", None)
        for source, target in (
            ("max_tokens", "max_new_tokens"),
            ("min_tokens", "min_new_tokens"),
            ("seed", "sampling_seed"),
        ):
            if source in values:
                values[target] = values.pop(source)

        if self.args.rollout_stop is not None:
            values.setdefault("stop", self.args.rollout_stop)
        if self.args.rollout_stop_token_ids is not None:
            renderer_stops = list(values.get("stop_token_ids") or [])
            values["stop_token_ids"] = list(dict.fromkeys([*renderer_stops, *self.args.rollout_stop_token_ids]))
        values["skip_special_tokens"] = self.args.rollout_skip_special_tokens
        values["no_stop_trim"] = True
        values["spaces_between_special_tokens"] = False
        values["n"] = 1

        context_limit = self.args.rollout_max_context_len
        response_limit = self.args.rollout_max_response_len
        requested = int(values.get("max_new_tokens", response_limit))
        if context_limit is not None:
            requested = min(requested, context_limit - prompt_len)
        values["max_new_tokens"] = min(requested, response_limit)
        if values["max_new_tokens"] <= 0:
            runtime = _import_verifiers()
            raise runtime.OverlongPromptError(
                f"prompt has {prompt_len} tokens, rollout_max_context_len={context_limit}"
            )
        return values

    async def post(self, endpoint: str, *, body: dict[str, Any], options=None, **_kwargs) -> httpx.Response:
        # [FUNCTION DOC] Handles one generate request and returns a successful HTTP response object
        # whose JSON body has a request id (the engine's, else "vf-" plus a random hex), and one
        # choice with the completion token ids, a log-prob content list (one item per token with the
        # text "token_id:<id>" and the log prob) and a finish reason. Steps: raises a not-implemented
        # error if the body has multimodal features. Reads prompt token ids from the body and the
        # session id from the "X-Session-ID" request header (options). For a session id: if already
        # seen, marks it recently used; otherwise, if a maximum prompt length is configured and the
        # prompt is longer, raises the overlong-prompt error; records it and evicts the oldest ids
        # beyond the cap. Builds the engine payload: input ids, converted sampling params, log probs
        # enabled, the LoRA adapter name when LoRA is enabled, and priority and cache salt (as extra
        # key) when present. With the consistent-hashing or manual router policy and a session id,
        # sends the routing key header equal to the session id. Posts to the router generate url.
        # Collects token ids and log probs from the output's per-token log-prob list; raises a runtime
        # error if the count differs from the reported completion token count (default: the list
        # length). Raises a runtime error on an abort finish reason.
        if body.get("features") is not None:
            raise NotImplementedError("Miles Verifiers rollouts do not yet support multimodal renderer features.")

        prompt_ids = list(body["token_ids"])
        headers = dict((options or {}).get("headers") or {})
        session_id = headers.get("X-Session-ID")
        if session_id:
            if session_id in self._seen_sessions:
                self._seen_sessions.move_to_end(session_id)
            else:
                max_prompt_len = getattr(self.args, "rollout_max_prompt_len", None)
                if max_prompt_len is not None and len(prompt_ids) > max_prompt_len:
                    runtime = _import_verifiers()
                    raise runtime.OverlongPromptError(
                        f"initial prompt has {len(prompt_ids)} tokens, rollout_max_prompt_len={max_prompt_len}"
                    )
                self._seen_sessions[session_id] = None
                if len(self._seen_sessions) > self._session_cache_size:
                    self._seen_sessions.popitem(last=False)

        payload: dict[str, Any] = {
            "input_ids": prompt_ids,
            "sampling_params": self._sampling_params(body.get("sampling_params") or {}, len(prompt_ids)),
            "return_logprob": True,
        }
        if is_lora_enabled(self.args):
            payload["lora_path"] = LORA_ADAPTER_NAME
        if body.get("priority") is not None:
            payload["priority"] = body["priority"]
        if body.get("cache_salt") is not None:
            payload["extra_key"] = body["cache_salt"]

        request_headers = None
        if getattr(self.args, "sglang_router_policy", None) in ("consistent_hashing", "manual") and session_id:
            request_headers = {"X-SMG-Routing-Key": session_id}

        from miles.utils.http_utils import post

        output = await post(_generate_url(self.router_args), payload, headers=request_headers)
        meta_info = dict(output.get("meta_info") or {})
        token_logprobs = list(meta_info.get("output_token_logprobs") or [])
        completion_ids = [int(item[1]) for item in token_logprobs]
        completion_logprobs = [float(item[0]) for item in token_logprobs]
        expected = int(meta_info.get("completion_tokens", len(completion_ids)))
        if len(completion_ids) != expected:
            raise RuntimeError(
                "SGLang generate response has mismatched completion token metadata: "
                f"{len(completion_ids)} != {expected}"
            )

        response_body = {
            "request_id": output.get("request_id") or f"vf-{uuid.uuid4().hex}",
            "choices": [
                {
                    "token_ids": completion_ids,
                    "logprobs": {
                        "content": [
                            {"token": f"token_id:{token_id}", "logprob": logprob}
                            for token_id, logprob in zip(completion_ids, completion_logprobs, strict=True)
                        ]
                    },
                    "finish_reason": _finish_reason(output),
                }
            ],
        }
        request = httpx.Request("POST", endpoint)
        return httpx.Response(200, json=response_body, request=request)

    async def close(self) -> None:
        # [FUNCTION DOC] No-op; there is nothing to release.
        return None


def _sample_status(trace) -> Sample.Status:
    # [FUNCTION DOC] HELPER: Maps a trace to a sample status: failed if the trace has an error, else
    # truncated if it was truncated, else completed.
    if trace.has_error:
        return Sample.Status.FAILED
    if trace.is_truncated:
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _serialize_prompt(prompt):
    # [FUNCTION DOC] HELPER: Makes a task prompt storable. A list of messages becomes a list in
    # which message objects with a dump method are converted to JSON-compatible mappings without
    # null entries and other items are kept; anything else is returned as is, or an empty string if
    # falsy.
    if isinstance(prompt, list):
        return [
            message.model_dump(mode="json", exclude_none=True) if hasattr(message, "model_dump") else message
            for message in prompt
        ]
    return prompt or ""


def _validate_group_reward_sample_counts(args: Namespace, tasks, discover_decorated) -> None:
    # [FUNCTION DOC] HELPER: Does nothing unless some task defines group-level reward functions
    # (found by the discovery helper). Otherwise raises a value error if training is enabled (the
    # rollout count is not zero or missing) and samples per prompt is below 2, or if an eval
    # interval is set (not none) and samples per eval prompt is below 2, since group rewards need
    # at least two rollouts to rank.
    if not any(discover_decorated(task, "group_reward") for task in tasks):
        return
    if getattr(args, "num_rollout", None) != 0 and args.n_samples_per_prompt < 2:
        raise ValueError("Verifiers tasks with @group_reward require --n-samples-per-prompt >= 2.")
    if getattr(args, "eval_interval", None) is not None and args.n_samples_per_eval_prompt < 2:
        raise ValueError("Verifiers tasks with @group_reward require --n-samples-per-eval-prompt >= 2.")


def _raise_for_unsupported_trace_errors(traces) -> None:
    # [FUNCTION DOC] HELPER: Raises a runtime error with the message of the first trace whose error
    # message starts with the adapter's unsupported-feature prefix (so unsupported requests fail the
    # run instead of being silently dropped). Returns nothing otherwise.
    for trace in traces:
        if trace.error is not None and trace.error.message.startswith(_UNSUPPORTED_ERROR_PREFIX):
            raise RuntimeError(trace.error.message)


def _branch_to_sample(args: Namespace, trace, branch, *, group_index: int, index: int) -> Sample:
    # [FUNCTION DOC] HELPER: Builds one validated Miles sample from a trace branch. Reads token ids,
    # the sampled-token mask and log probs of the branch (raises a value error if lengths differ).
    # The response starts at the first sampled token (or at the end if none): response length is
    # tokens minus that position; the loss mask is the sampled mask from there as integers (so
    # environment tokens inside the response get zero) and rollout log probs are sliced the same way;
    # everything before is the prompt. Reward: the trace reward, or, when a reward key is configured,
    # a mapping of all named rewards plus "reward". Label: the task data's label, else its answer.
    # Metadata: a "verifiers" mapping with branch index, task index, named rewards, metrics, stop
    # condition and, if present, the serialized error. Other fields: group and sample indexes, the
    # serialized task prompt, the trace's last reply as response, status from the helper, and the
    # trace id as routing key. Calls the sample's own validation before returning.
    tokens = list(branch.token_ids)
    sampled_mask = list(branch.sampled_mask)
    logprobs = list(branch.logprobs)
    if len(tokens) != len(sampled_mask) or len(tokens) != len(logprobs):
        raise ValueError(
            f"Trace {trace.id} token metadata mismatch: "
            f"tokens={len(tokens)}, mask={len(sampled_mask)}, logprobs={len(logprobs)}"
        )
    first_sampled = sampled_mask.index(True) if True in sampled_mask else len(tokens)
    response_length = len(tokens) - first_sampled
    reward = trace.reward if args.reward_key is None else {**trace.rewards, "reward": trace.reward}
    task_data = trace.task.data
    label = getattr(task_data, "label", None)
    if label is None:
        label = getattr(task_data, "answer", None)

    metadata = {
        "verifiers": {
            "branch_index": branch.index,
            "task_index": getattr(task_data, "idx", None),
            "rewards": dict(trace.rewards),
            "metrics": dict(trace.metrics),
            "stop_condition": trace.stop_condition,
        }
    }
    if trace.error is not None:
        metadata["verifiers"]["error"] = trace.error.model_dump(mode="json", exclude_none=True)

    sample = Sample(
        group_index=group_index,
        index=index,
        prompt=_serialize_prompt(getattr(task_data, "prompt", "")),
        tokens=tokens,
        response=trace.last_reply,
        response_length=response_length,
        label=label,
        reward=reward,
        loss_mask=[int(value) for value in sampled_mask[first_sampled:]],
        rollout_log_probs=logprobs[first_sampled:],
        status=_sample_status(trace),
        metadata=metadata,
        routing_key=trace.id,
    )
    sample.validate()
    return sample


def trace_to_samples(args: Namespace, trace, *, group_index: int, index_start: int) -> list[Sample]:
    # [FUNCTION DOC] Converts a trace to a list of samples. A trace without branches is skipped with
    # a warning (including its error) and gives an empty list. More than one branch raises a
    # not-implemented error (trace groups cannot be preserved yet). Otherwise returns a one-item list
    # holding the sample for the single branch, using index_start as its sample index.
    if not trace.branches:
        error = trace.error.model_dump(mode="json", exclude_none=True) if trace.error is not None else None
        logger.warning(
            "Verifiers trace %s has no graph branches; omitting it from training (error=%s).",
            trace.id,
            error,
        )
        return []
    if len(trace.branches) != 1:
        raise NotImplementedError(
            "Miles cannot yet preserve Verifiers trace groups when a rollout produces "
            f"multiple graph branches (trace {trace.id} produced {len(trace.branches)})."
        )
    return [
        _branch_to_sample(
            args,
            trace,
            trace.branches[0],
            group_index=group_index,
            index=index_start,
        )
    ]


def trace_to_sample(args: Namespace, trace, *, group_index: int, index: int) -> Sample:
    # [FUNCTION DOC] Converts a trace to exactly one sample (via the list version); raises a value
    # error if the number of samples is not one (for example a trace without branches).
    samples = trace_to_samples(args, trace, group_index=group_index, index_start=index)
    if len(samples) != 1:
        raise ValueError(f"Verifiers trace {trace.id} produced {len(samples)} branches, expected one.")
    return samples[0]


def _trace_metrics(traces) -> dict[str, float]:
    # [FUNCTION DOC] HELPER: Returns the means over the traces of: reward, error indicator, truncated
    # indicator and turn count, under the keys "verifiers/reward_mean", "verifiers/error_rate",
    # "verifiers/truncated_rate" and "verifiers/num_turns_mean". Empty input gives an empty mapping.
    if not traces:
        return {}
    return {
        "verifiers/reward_mean": sum(trace.reward for trace in traces) / len(traces),
        "verifiers/error_rate": sum(trace.has_error for trace in traces) / len(traces),
        "verifiers/truncated_rate": sum(trace.is_truncated for trace in traces) / len(traces),
        "verifiers/num_turns_mean": sum(trace.num_turns for trace in traces) / len(traces),
    }


def _trace_eval_reward(trace, reward_key: str | None):
    # [FUNCTION DOC] HELPER: Evaluation reward of a trace. No reward key: the trace reward. With a
    # key: looks it up in the named rewards plus "reward"; for an errored trace a missing key gives
    # none, otherwise a missing key raises a key error.
    if reward_key is None:
        return trace.reward
    rewards = {**trace.rewards, "reward": trace.reward}
    if trace.has_error:
        return rewards.get(reward_key)
    return rewards[reward_key]


def _flatten_samples(values: Iterable[Any]) -> list[Sample]:
    # [FUNCTION DOC] HELPER: Recursively flattens arbitrarily nested lists of samples into one flat
    # list, preserving order.
    flattened = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(_flatten_samples(value))
        else:
            flattened.append(value)
    return flattened


def _make_eval_args(args: Namespace) -> Namespace:
    # [FUNCTION DOC] HELPER: Returns a copy of the args with evaluation overrides applied. For each
    # of temperature, top-p, top-k, max response length and max context length: if the eval setting
    # is set (not none) it replaces the rollout setting in the copy. Unconditionally (even when
    # none) the rollout max prompt length becomes the eval max prompt length and the rollout min new
    # tokens becomes the eval min new tokens; the reward key becomes the eval reward key if set,
    # else the rollout reward key. The original args are not modified.
    eval_args = Namespace(**vars(args))
    for eval_name, rollout_name in (
        ("eval_temperature", "rollout_temperature"),
        ("eval_top_p", "rollout_top_p"),
        ("eval_top_k", "rollout_top_k"),
        ("eval_max_response_len", "rollout_max_response_len"),
        ("eval_max_context_len", "rollout_max_context_len"),
    ):
        if (value := getattr(args, eval_name, None)) is not None:
            setattr(eval_args, rollout_name, value)
    eval_args.rollout_max_prompt_len = args.eval_max_prompt_len
    eval_args.rollout_min_new_tokens = args.eval_min_new_tokens
    eval_args.reward_key = args.eval_reward_key or args.reward_key
    return eval_args


def _config_path() -> str:
    # [FUNCTION DOC] HELPER: Returns the value of the VERIFIERS_CONFIG environment variable; raises a
    # value error explaining how to set it if it is unset or empty.
    path = os.environ.get(_CONFIG_ENV_VAR)
    if not path:
        raise ValueError(
            f"Verifiers rollouts need {_CONFIG_ENV_VAR} set to a Verifiers EnvConfig TOML file. "
            "Launch through run.py in this directory, or forward the variable yourself "
            "(Ray runtime_env for a hand-rolled command)."
        )
    return path


def _validate_args(args: Namespace) -> None:
    # [FUNCTION DOC] HELPER: Raises a value error for each unsupported option combination, checked
    # in this order: the global prompt dataset still enabled (must be disabled because Verifiers
    # supplies the tasks); partial rollout; multimodal keys; a custom chat template path (renderers
    # cannot take one); and any of on-policy-distillation, rollout routing replay or rollout indexer
    # replay (the transport does not carry the extra token metadata; the error lists all enabled
    # flags). Returns nothing when all are fine.
    """Reject the Miles options this adapter cannot honor.

    run.py never builds these combinations; this catches a hand-rolled command
    before an episode runs and produces silently wrong training data.
    """
    if getattr(args, "rollout_global_dataset", False):
        raise ValueError(
            "Verifiers rollouts replace Miles prompt data with the configured taskset; "
            "pass --disable-rollout-global-dataset."
        )
    if args.partial_rollout:
        raise ValueError(
            "--partial-rollout is not supported for Verifiers because an episode "
            "cannot be resumed from partially executed environment state."
        )
    if args.multimodal_keys is not None:
        raise ValueError(
            "--multimodal-keys is not supported by the Verifiers transport, which "
            "handles text-only renderer inputs."
        )
    if args.chat_template_path is not None:
        raise ValueError(
            "--chat-template-path is not supported because renderers does not accept a custom "
            "Jinja template. Use the checkpoint's template and --apply-chat-template-kwargs."
        )
    unsupported = [
        flag
        for enabled, flag in (
            (args.use_opd, "--use-opd"),
            (args.use_rollout_routing_replay, "--use-rollout-routing-replay"),
            (getattr(args, "use_rollout_indexer_replay", False), "--use-rollout-indexer-replay"),
        )
        if enabled
    ]
    if unsupported:
        raise ValueError(
            f"{', '.join(unsupported)} is not supported by the Verifiers SGLang transport, "
            "which does not preserve its additional token metadata."
        )


class VerifiersRolloutFn(BaseRolloutFn):
    # [CLASS DOC] Miles rollout function backed by a Verifiers V1 environment. One instance holds the
    # environment, the training and evaluation model contexts and clients, the loaded and optionally
    # shuffled task list, the dynamic sampling filter, and running counters for the next training
    # task index (set from the first rollout id), group index and sample index. Calling it with an
    # input produces either a training batch or an evaluation result.
    def __init__(self, input: RolloutFnConstructorInput):
        # [FUNCTION DOC] Builds the adapter from the constructor input (args and data source).
        # Steps: lazily import the Verifiers runtime; validate the args; load and validate the
        # environment config from the file named by the config environment variable (a legacy-style
        # config or the codex harness raises a value error); create the environment; the model is
        # the HF checkpoint; build sampling configs for training and for evaluation (from a copy of
        # the args with eval overrides); engines = rollout GPUs divided by GPUs per engine and the
        # maximum concurrency = per-server concurrency times engines; renderer pool size = that
        # concurrency clamped to between 1 and 16; create the training client and an evaluation
        # client whose router address is read from the live training args; build the two model
        # contexts; load the dynamic sampling filter function; load the tasks from the taskset,
        # shuffled with the rollout seed if rollout shuffle is on (zero tasks raises a value error);
        # validate group-reward sample counts; initialize the counters (next training task index
        # unset).
        super().__init__(input)
        runtime = _import_verifiers()
        self.args = input.args
        self.data_source = input.data_source
        _validate_args(self.args)
        self.config = runtime.EnvConfig.model_validate(_load_config_data(_config_path()))
        if self.config.is_legacy:
            raise ValueError("Miles' Verifiers integration supports V1 environment configs only.")
        if self.config.harness.id == "codex":
            raise ValueError(
                "Miles' Verifiers adapter does not support the Codex harness because it uses the Responses dialect."
            )

        self.env = runtime.Environment(self.config)
        self.model = self.args.hf_checkpoint
        self.sampling = self._sampling_config(runtime.SamplingConfig, self.args)
        self.eval_args = _make_eval_args(self.args)
        self.eval_sampling = self._sampling_config(runtime.SamplingConfig, self.eval_args)

        engine_count = self.args.rollout_num_gpus // self.args.rollout_num_gpus_per_engine
        self.max_concurrent = self.args.sglang_server_concurrency * engine_count
        pool_size = max(1, min(self.max_concurrent, 16))
        self.client = _train_client(runtime, self.args, self.model, pool_size)
        self.eval_client = _train_client(
            runtime,
            self.eval_args,
            self.model,
            pool_size,
            router_args=self.args,
        )
        self.ctx = runtime.ModelContext(client=self.client, model=self.model, sampling=self.sampling)
        self.eval_ctx = runtime.ModelContext(client=self.eval_client, model=self.model, sampling=self.eval_sampling)

        from miles.utils.function_registry import load_function

        self.dynamic_filter = load_function(self.args.dynamic_sampling_filter_path)
        self._tasks = list(self.env.taskset.load())
        if self.args.rollout_shuffle:
            random.Random(self.args.rollout_seed).shuffle(self._tasks)
        if not self._tasks:
            raise ValueError("Verifiers taskset selected zero tasks.")
        _validate_group_reward_sample_counts(self.args, self._tasks, runtime.discover_decorated)
        self._next_train_task_idx = None
        self._next_group_index = 0
        self._next_sample_index = 0

    @staticmethod
    def _sampling_config(SamplingConfig, args: Namespace):
        # [FUNCTION DOC] HELPER: Builds a validated Verifiers sampling config from args: temperature,
        # top-p and max tokens (the rollout response length); top-k if not none; min tokens if the
        # min-new-tokens setting exists and is not none; and, if chat template kwargs are given, an
        # extra body entry carrying them. Returns the validated config object.
        data: dict[str, Any] = {
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
            "max_tokens": args.rollout_max_response_len,
        }
        if args.rollout_top_k is not None:
            data["top_k"] = args.rollout_top_k
        if (min_tokens := getattr(args, "rollout_min_new_tokens", None)) is not None:
            data["min_tokens"] = min_tokens
        if args.apply_chat_template_kwargs:
            data["extra_body"] = {"chat_template_kwargs": args.apply_chat_template_kwargs}
        return SamplingConfig.model_validate(data)

    def _task(self, index: int):
        # [FUNCTION DOC] HELPER: Returns the task at the position modulo the task count (the task
        # list is cycled endlessly).
        return self._tasks[index % len(self._tasks)]

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        # [FUNCTION DOC] Rollout entry point: runs evaluation if the input says so, else produces a
        # training batch; returns the corresponding output object.
        return await (self._call_eval(input) if input.evaluation else self._call_train(input))

    async def _run_task_group(self, task, n: int, semaphore: asyncio.Semaphore, seed_base: int, ctx=None):
        # [FUNCTION DOC] HELPER: Runs n rollouts of one task as an episode group and returns the list
        # of traces. Uses the given model context (default the training one). If deterministic
        # inference is enabled in args, each rollout gets its own context copy whose sampling seed is
        # seed_base plus the rollout's offset. The episode is run under the shared concurrency
        # semaphore.
        runtime = _import_verifiers()
        ctx = ctx or self.ctx
        episode = self.env.episode(task, ctx, n=n)
        if getattr(self.args, "sglang_enable_deterministic_inference", False):
            for offset, rollout in enumerate(episode.rollouts):
                sampling = ctx.sampling.model_copy(update={"sampling_seed": seed_base + offset})
                rollout.ctx = runtime.ModelContext(client=ctx.client, model=self.model, sampling=sampling)
        return await episode.run(semaphore)

    def _convert_group(self, traces, *, group_index: int, preserve_empty: bool = False):
        # [FUNCTION DOC] HELPER: Converts a group of traces into a list of samples, one per trace,
        # assigning consecutive global sample indexes (the counter advances by the number of samples
        # produced, even if the group is later discarded). A trace that yields no sample marks the
        # group incomplete; with preserve_empty a none placeholder keeps its position. Returns the
        # list when complete or when preserve_empty is set; otherwise an empty list.
        group = []
        complete = True
        for trace in traces:
            converted = trace_to_samples(
                self.args,
                trace,
                group_index=group_index,
                index_start=self._next_sample_index,
            )
            self._next_sample_index += len(converted)
            if not converted:
                complete = False
                if preserve_empty:
                    group.append(None)
                continue
            group.append(converted[0])
        return group if complete or preserve_empty else []

    async def _apply_miles_rewards(self, group) -> None:
        # [FUNCTION DOC] HELPER: Optionally replaces the Verifiers rewards with Miles reward model
        # rewards, in place on the samples. Flattens the group; with group-level reward modeling
        # computes all rewards in one batched call; else if a custom reward path or a reward type is
        # configured computes one reward per sample concurrently; else leaves the Verifiers rewards.
        # Raises a value error if the number of rewards does not match the number of samples.
        from miles.rollout.rm_hub import async_rm, batched_async_rm

        samples = _flatten_samples(group)
        if self.args.group_rm:
            rewards = await batched_async_rm(self.args, samples)
        elif self.args.custom_rm_path is not None or self.args.rm_type:
            rewards = await asyncio.gather(*(async_rm(self.args, sample) for sample in samples))
        else:
            return
        if rewards is None or len(rewards) != len(samples):
            raise ValueError("Miles reward model returned an unexpected number of rewards.")
        for sample, reward in zip(samples, rewards, strict=True):
            sample.reward = reward

    async def _postprocess_train_samples(self, data, all_data) -> None:
        # [FUNCTION DOC] HELPER: Post-processes a finished training batch in place. data is the kept
        # groups, all_data every converted group. If configured, calls the sample filter function
        # with the args and kept groups, and the all-samples process function with the args, all
        # groups and the data source. Then recomputes the rollout log probabilities of the kept
        # (flattened) samples with a prefill pass against the router generate endpoint, using the
        # rollout temperature, top-p, top-k and response length as sampling params (network side
        # effect; the sampled values from the transport are replaced).
        from miles.utils.function_registry import load_function

        if function := load_function(self.args.rollout_sample_filter_path):
            function(self.args, data)
        if function := load_function(self.args.rollout_all_samples_process_path):
            function(self.args, all_data, self.data_source)
        await recompute_samples_rollout_logprobs_via_prefill(
            self.args,
            _flatten_samples(data),
            url=_generate_url(self.args),
            sampling_params={
                "temperature": self.args.rollout_temperature,
                "top_p": self.args.rollout_top_p,
                "top_k": self.args.rollout_top_k,
                "max_new_tokens": self.args.rollout_max_response_len,
            },
        )

    async def _cancel_pending(self, futures: Iterable[asyncio.Task]) -> None:
        # [FUNCTION DOC] HELPER: Cancels every unfinished task. If any were unfinished, asks every
        # SGLang worker to abort ALL of its in-flight requests (the failure to look up or post is
        # logged, not raised; individual post errors are ignored). Finally waits for all given tasks,
        # swallowing their exceptions.
        pending = [future for future in futures if not future.done()]
        for future in pending:
            future.cancel()
        if pending:
            try:
                from miles.utils.http_utils import post

                urls = await _sglang_worker_urls(self.args)
                await asyncio.gather(
                    *(post(f"{url}/abort_request", {"abort_all": True}) for url in urls),
                    return_exceptions=True,
                )
            except Exception:
                logger.exception("Failed to abort pending Verifiers requests.")
        await asyncio.gather(*futures, return_exceptions=True)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        # [FUNCTION DOC] Produces one training batch of rollout_batch_size groups. Configures the
        # SGLang dumper first. On the first call sets the next task index to rollout id times the
        # batch size (later calls continue from where the counter stopped). Within the environment's
        # serving context, loops until enough groups are kept: while kept groups plus pending tasks
        # are below the target, starts over-sampling-batch-size new group tasks (task index counter
        # advances; seed = rollout seed + task index times samples per prompt); waits for the first
        # task to finish; for each finished task: an exception is logged and counted as a dropped
        # "episode_error"; otherwise the traces are added to the metric pool, unsupported-feature
        # errors raise, any trace error drops the group as "trace_error"; the group is converted (the
        # group counter advances even if dropped), a group with the wrong number of samples is
        # dropped as "empty_trace"; Miles rewards are applied; the group is recorded among all
        # groups and in the pre-filter metrics; the dynamic filter decides keep or drop (with the
        # filter's reason); a kept group is added while below the target. Always cancels leftover
        # pending tasks afterwards. Then sorts kept and all groups by the index of their first
        # sample, post-processes the batch, and returns the kept groups with filter metrics merged
        # with the trace metrics over all traces.
        # KNOWN ISSUE: each top-up starts a whole over-sampling batch even when fewer groups are
        # missing, so more groups than needed may be started (extra ones are cancelled at the end).
        from miles.utils import dumper_utils

        await dumper_utils.configure_sglang(self.args)
        target = self.args.rollout_batch_size
        if self._next_train_task_idx is None:
            self._next_train_task_idx = input.rollout_id * target

        groups = []
        all_groups = []
        all_traces = []
        metrics = MetricGatherer()

        semaphore = asyncio.Semaphore(self.max_concurrent)
        pending: set[asyncio.Task] = set()
        async with self.env.serving():
            try:
                while len(groups) < target:
                    while len(groups) + len(pending) < target:
                        for _ in range(self.args.over_sampling_batch_size):
                            task_index = self._next_train_task_idx
                            self._next_train_task_idx += 1
                            seed = self.args.rollout_seed + task_index * self.args.n_samples_per_prompt
                            pending.add(
                                asyncio.create_task(
                                    self._run_task_group(
                                        self._task(task_index),
                                        self.args.n_samples_per_prompt,
                                        semaphore,
                                        seed,
                                    )
                                )
                            )
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for future in done:
                        try:
                            traces = future.result()
                        except Exception:
                            logger.exception("Verifiers episode failed; resampling.")
                            metrics.on_dynamic_filter_drop(reason="episode_error")
                            continue
                        all_traces.extend(traces)
                        _raise_for_unsupported_trace_errors(traces)
                        if any(trace.has_error for trace in traces):
                            metrics.on_dynamic_filter_drop(reason="trace_error")
                            continue
                        group = self._convert_group(traces, group_index=self._next_group_index)
                        self._next_group_index += 1
                        if len(group) != self.args.n_samples_per_prompt:
                            metrics.on_dynamic_filter_drop(reason="empty_trace")
                            continue
                        await self._apply_miles_rewards(group)
                        all_groups.append(group)
                        metrics.on_group_before_dynamic_filter(self.args, _flatten_samples(group))
                        filter_output = apply_preput_filters(
                            self.args,
                            self.dynamic_filter,
                            _flatten_samples(group),
                        )
                        if not filter_output.keep:
                            metrics.on_dynamic_filter_drop(reason=filter_output.reason)
                            continue
                        if len(groups) < target:
                            groups.append(group)
            finally:
                await self._cancel_pending(pending)

        groups.sort(key=lambda group: _flatten_samples(group)[0].index)
        all_groups.sort(key=lambda group: _flatten_samples(group)[0].index)
        await self._postprocess_train_samples(groups, all_groups)
        output_metrics = metrics.collect()
        output_metrics.update(_trace_metrics(all_traces))
        return RolloutFnTrainOutput(samples=groups, metrics=output_metrics)

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        # [FUNCTION DOC] Evaluates on the whole taskset. Asserts group-level reward modeling is off.
        # Configures the SGLang dumper, then within the environment's serving context runs one group
        # per task (samples-per-eval-prompt rollouts, seed base = rollout seed + task position times
        # that count, evaluation context) concurrently and gathers all; always cancels leftovers.
        # Then per group: collects traces, raises on unsupported-feature errors, converts the group
        # keeping empty placeholders, applies Miles rewards to the converted samples, and collects
        # the samples. Per trace the evaluation reward is: when Miles rewards are in use and the
        # sample exists, the mean of its reward values under the eval args; when Miles rewards are in
        # use but the sample is missing, the trace reward; otherwise the Verifiers evaluation reward
        # for the configured eval (or rollout) reward key. Also records each trace's truncated flag.
        # Returns an output whose data maps the environment id (or the name "verifiers") to rewards,
        # truncated flags and samples, with the trace metrics.
        assert not self.args.group_rm, "Group RM is not supported for eval rollout"

        from miles.utils import dumper_utils

        await dumper_utils.configure_sglang(self.args)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        async with self.env.serving():
            futures = [
                asyncio.create_task(
                    self._run_task_group(
                        task,
                        self.args.n_samples_per_eval_prompt,
                        semaphore,
                        self.args.rollout_seed + index * self.args.n_samples_per_eval_prompt,
                        self.eval_ctx,
                    )
                )
                for index, task in enumerate(self._tasks)
            ]
            try:
                trace_groups = await asyncio.gather(*futures)
            finally:
                await self._cancel_pending(futures)

        samples = []
        rewards = []
        truncated = []
        all_traces = []
        reward_key = self.args.eval_reward_key or self.args.reward_key
        use_miles_rewards = bool(self.args.custom_rm_path is not None or self.args.rm_type)
        for group_index, traces in enumerate(trace_groups):
            all_traces.extend(traces)
            _raise_for_unsupported_trace_errors(traces)
            group = self._convert_group(traces, group_index=group_index, preserve_empty=True)
            trainable = [value for value in group if value is not None]
            await self._apply_miles_rewards(trainable)
            samples.extend(_flatten_samples(trainable))
            for trace, value in zip(traces, group, strict=True):
                if use_miles_rewards and value is not None:
                    sample_rewards = [sample.get_reward_value(self.eval_args) for sample in _flatten_samples([value])]
                    reward = sum(sample_rewards) / len(sample_rewards)
                else:
                    reward = trace.reward if use_miles_rewards else _trace_eval_reward(trace, reward_key)
                rewards.append(reward)
                truncated.append(trace.is_truncated)
        return RolloutFnEvalOutput(
            data={
                self.config.env_id
                or "verifiers": {
                    "rewards": rewards,
                    "truncated": truncated,
                    "samples": samples,
                }
            },
            metrics=_trace_metrics(all_traces),
        )


_LEGACY_INSTANCES: dict[tuple[int, int, bool], VerifiersRolloutFn] = {}


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    # [FUNCTION DOC] Legacy function-style rollout entry point. Keeps one adapter instance per
    # combination of args object identity, data source identity and evaluation flag in a module-level
    # cache (created on first use), builds the matching train or eval input from the rollout id, runs
    # the adapter's coroutine to completion with the shared async runner and returns its output.
    """Legacy Miles entrypoint backed by one persistent rollout adapter."""
    from miles.utils.async_utils import run

    # The refactored rollout manager constructs separate train and eval adapters.
    # Preserve that lifecycle under the legacy function interface as well.
    key = (id(args), id(data_source), evaluation)
    adapter = _LEGACY_INSTANCES.get(key)
    if adapter is None:
        adapter = VerifiersRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
        _LEGACY_INSTANCES[key] = adapter
    input = RolloutFnEvalInput(rollout_id) if evaluation else RolloutFnTrainInput(rollout_id)
    return run(adapter(input))
