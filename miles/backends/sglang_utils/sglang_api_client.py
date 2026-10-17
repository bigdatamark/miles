# [MODULE DOC] Asynchronous HTTP client for one SGLang server (and standalone health helpers). All
# calls go through a shared HTTP client provider. The SGLangApiClient class wraps the server's
# endpoints for health, weights update (from tensors, disk, or a distributed group), LoRA adapter
# management, memory release/resume, cache flush, generation pause/resume, request abort, weight
# update sessions, version query and profiling. Failed responses raise the HTTP library's status
# error unless stated otherwise.
import asyncio
import dataclasses
import logging

import httpx

from miles.utils.http_utils import GeneralHttpClientProvider

logger = logging.getLogger(__name__)


def _compute_headers(api_key: str | None) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Builds request headers: a JSON content type with UTF-8 charset and a
    # bearer authorization header built from the key.
    # KNOWN ISSUE: when the key is none the header literally contains the text "None" as the token
    # rather than being omitted.
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }


async def probe_server_healthy(server_url: str, api_key: str | None, timeout: float = 5.0) -> bool:
    # [FUNCTION DOC] One-shot health probe: sends a GET to the server's generation health endpoint
    # with the timeout (default 5 seconds) and returns true only for status 200. Any HTTP-library
    # error or operating-system error (connection failures, timeouts) yields false; other statuses
    # also yield false without raising.
    try:
        response = await GeneralHttpClientProvider.client().get(
            f"{server_url}/health_generate",
            headers=_compute_headers(api_key),
            timeout=timeout,
        )
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


async def wait_server_healthy(server_url, api_key):
    # [FUNCTION DOC] Blocks until the server is ready for offload; returns nothing. Phase one: poll
    # the generation health endpoint every 2 seconds until it returns 200 (HTTP-library errors are
    # swallowed and retried). Phase two: poll the flush-cache endpoint every 2 seconds until it
    # returns 200, so the work queue is empty. There is no overall timeout, so it can wait forever.
    headers = _compute_headers(api_key)

    http_client = GeneralHttpClientProvider.client()
    while True:
        try:
            response = await http_client.get(f"{server_url}/health_generate", headers=headers)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            pass

        await asyncio.sleep(2)

    # use flush_cache to make sure the working queue is empty, so that we can do offload
    while True:
        try:
            response = await http_client.get(f"{server_url}/flush_cache", headers=headers)
            if response.status_code == 200:
                break

        except httpx.HTTPError:
            pass

        await asyncio.sleep(2)


@dataclasses.dataclass(frozen=True)
class SGLangApiClient:
    # [CLASS DOC] Immutable client for one SGLang server, holding its base URL and optional API key.
    # Methods are coroutines. Most POST endpoints go through one request helper that sends a JSON
    # body and returns the decoded JSON reply (or none for an empty body); a few use direct GET or
    # POST calls and return decoded JSON or the raw response.
    server_url: str
    api_key: str | None = None

    @property
    def _headers(self) -> dict[str, str]:
        # [FUNCTION DOC] HELPER: Request headers computed from the API key.
        return _compute_headers(self.api_key)

    async def _make_request(self, endpoint: str, payload: dict | None = None, *, timeout: float | None = None):
        # [FUNCTION DOC] HELPER: POSTs a JSON body (empty object if the payload is none or empty) to
        # the server URL joined with the endpoint path, with the headers and, only if given, a
        # per-request timeout (otherwise the client's default). A failed status raises the HTTP
        # status error, with the response text attached as an exception note when supported.
        # Returns the decoded JSON body, or none if the body is empty.
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        url = f"{self.server_url}/{endpoint}"
        bound: dict[str, float] = {} if timeout is None else dict(timeout=timeout)
        response = await GeneralHttpClientProvider.client().post(
            url, json=payload or {}, headers=self._headers, **bound
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if hasattr(e, "add_note"):
                e.add_note(f"{response.text=}")
            raise
        if not response.content:
            return None
        return response.json()

    async def health_generate(self, timeout: float = 5.0) -> bool:
        # [FUNCTION DOC] GETs the generation health endpoint with the timeout (default 5 seconds).
        # Returns true on success; any non-success status or transport failure raises.
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            httpx.HTTPError: If the request fails for any reason, including timeout.
        """
        response = await GeneralHttpClientProvider.client().get(
            f"{self.server_url}/health_generate",
            headers=self._headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    async def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
        selector: str = "all",
    ):
        # [FUNCTION DOC] Asks the server to load weights from serialized named-tensor handles
        # (the tensor data stays on GPU; only metadata travels over HTTP). The payload carries the
        # serialized tensors, optional load format, a flush-cache flag (default false), the target
        # selector (default all) and the weight version only if given. Returns the decoded reply.
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
            "selector": selector,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return await self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    async def get_remote_instance_transfer_engine_info(self, rank: int):
        # [FUNCTION DOC] GETs the transfer-engine information endpoint for the given rank with a
        # 5 second timeout and returns the value under the remote-instance transfer engine info
        # key of the reply (key error if absent).
        # TODO: will be changed to `remote_instance_transfer_engine_info` when the sglang side is ready.
        response = await GeneralHttpClientProvider.client().get(
            f"{self.server_url}/get_remote_instance_transfer_engine_info",
            params={"rank": rank},
            headers=self._headers,
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()["remote_instance_transfer_engine_info"]

    async def get_parallelism_info(self, rank: int):
        # [FUNCTION DOC] GETs the parallelism config endpoint for the given rank (5 second timeout)
        # and returns the decoded JSON.
        response = await GeneralHttpClientProvider.client().get(
            f"{self.server_url}/parallelism_config",
            params={"rank": rank},
            headers=self._headers,
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()

    async def get_server_info(self):
        # [FUNCTION DOC] GETs the server info endpoint (5 second timeout) and returns decoded JSON.
        response = await GeneralHttpClientProvider.client().get(
            f"{self.server_url}/server_info",
            headers=self._headers,
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()

    async def load_lora_adapter_from_tensors(
        self,
        lora_name: str,
        config_dict: dict,
        serialized_tensors: str | None = None,
        serialized_named_tensors: list | None = None,
        load_format: str | None = None,
        pinned: bool = False,
        added_tokens_config: dict | None = None,
        upsert: bool = False,
        expected_checksums: dict | None = None,
    ):
        # [FUNCTION DOC] Uploads a LoRA adapter by value. Raises a value error unless exactly one of
        # the whole-adapter serialized tensors or the per-tensor-parallel-rank serialized list is
        # given. The payload has the adapter name, config dictionary and pinned flag, the chosen
        # tensor field, and only-if-set extras: upsert (true only), load format, added-tokens config
        # and expected checksums. Returns the decoded reply.
        """Load a LoRA adapter from either transport (exactly one of the two).

        ``serialized_named_tensors[tp_rank]`` is bytes for that TP rank; ``serialized_tensors``
        is the whole adapter. With ``upsert``, the already-loaded ``lora_name`` is overwritten
        in place (no unload/register).
        """
        if (serialized_tensors is None) == (serialized_named_tensors is None):
            raise ValueError("pass exactly one of serialized_tensors / serialized_named_tensors")
        payload = {
            "lora_name": lora_name,
            "config_dict": config_dict,
            "pinned": pinned,
        }
        if serialized_tensors is not None:
            payload["serialized_tensors"] = serialized_tensors
        else:
            payload["serialized_named_tensors"] = serialized_named_tensors
        if upsert:
            payload["upsert"] = True
        if load_format is not None:
            payload["load_format"] = load_format
        if added_tokens_config is not None:
            payload["added_tokens_config"] = added_tokens_config
        if expected_checksums is not None:
            payload["expected_checksums"] = expected_checksums

        return await self._make_request(
            "load_lora_adapter_from_tensors",
            payload,
        )

    async def load_lora_adapter_from_distributed(
        self,
        lora_name: str,
        config_dict: dict,
        names: list,
        dtypes: list,
        shapes: list,
        group_name: str,
        pinned: bool = False,
        added_tokens_config: dict | None = None,
        upsert: bool = False,
    ):
        # [FUNCTION DOC] Registers a LoRA adapter whose weights will arrive over a distributed
        # group. The payload has the name, config, tensor names, element-type names (the "torch."
        # prefix of each type's text removed), shapes, group name, pinned flag, and the upsert flag
        # (always sent, unlike the tensor variant), plus the added-tokens config only if given.
        # Returns the decoded reply.
        """Load a LoRA adapter: only metadata is sent; weights arrive via NCCL broadcast over ``group_name``.
        With ``upsert``, the already-loaded ``lora_name`` is overwritten in place (no unload/register)."""
        payload = {
            "lora_name": lora_name,
            "config_dict": config_dict,
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "pinned": pinned,
            "upsert": upsert,
        }
        if added_tokens_config is not None:
            payload["added_tokens_config"] = added_tokens_config

        return await self._make_request(
            "load_lora_adapter_from_distributed",
            payload,
        )

    async def flush_cache(self):
        # [FUNCTION DOC] Retries a GET of the flush-cache endpoint up to 60 times, one second apart,
        # until a 200 reply; any non-200 reply or any exception (logged at info level) is recorded
        # as the last message and retried. Raises a timeout error carrying the last message if all
        # attempts fail. Returns nothing.
        """Flush the cache of the server."""
        last_message = None
        for _ in range(60):
            try:
                response = await GeneralHttpClientProvider.client().get(
                    f"{self.server_url}/flush_cache", headers=self._headers
                )
                if response.status_code == 200:
                    break
                last_message = response.text
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                last_message = str(e)
            await asyncio.sleep(1)
        else:
            raise TimeoutError(f"Timeout while flushing cache: {last_message}")

    async def get_weight_version(self):
        # [FUNCTION DOC] Tries the model-info endpoint, then the legacy weight-version endpoint,
        # returning the weight version field of the first 200 reply. If neither returns 200, raises
        # the status error of the last reply.
        # new sglang change api from /get_weight_version to /model_info
        for endpoint in ("/model_info", "/get_weight_version"):
            response = await GeneralHttpClientProvider.client().get(
                f"{self.server_url}{endpoint}", headers=self._headers
            )
            if response.status_code == 200:
                return response.json()["weight_version"]
        response.raise_for_status()

    async def register_lora_adapter(self, lora_name: str, config_dict: dict, pinned: bool = False):
        # [FUNCTION DOC] Sends the adapter name, config dictionary and pinned flag (default false) to
        # the register endpoint and returns the decoded reply.
        """Create-or-refresh a LoRA adapter's identity and config (weights zeroed)."""
        return await self._make_request(
            "register_lora_adapter",
            {"lora_name": lora_name, "config_dict": config_dict, "pinned": pinned},
        )

    async def release_memory_occupation(self, tags: list[str] = None):
        # [FUNCTION DOC] First flushes the server cache (with its retry and timeout behavior), then
        # asks the server to release the memory groups named by the tags (none means the server's
        # default) and returns the decoded reply. The tags annotation is not marked optional.
        """Release memory occupation. Available tags: weights, kv_cache."""
        await self.flush_cache()
        return await self._make_request(
            "release_memory_occupation",
            {"tags": tags},
        )

    async def resume_memory_occupation(self, tags: list[str] = None):
        # [FUNCTION DOC] Asks the server to resume the memory groups named by the tags (none means
        # the server's default); returns the decoded reply. Does not flush the cache.
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        return await self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        # [FUNCTION DOC] Calls the weights-checker endpoint with the action text, an allow
        # quantization error flag (default false), the selector (default all), and, if a skip list
        # is given, sends it under the server's field name for the skipped tensor list. Returns the
        # decoded reply.
        payload = {"action": action, "allow_quant_error": allow_quant_error, "selector": selector}
        if skip_list is not None:
            # sglang's CheckWeightsReqInput names this field `skip_tensor_list`.
            payload["skip_tensor_list"] = skip_list
        return await self._make_request("weights_checker", payload)

    async def pull_weights(self, target_version: int, local_checkpoint_dir: str, source_dir: str):
        # [FUNCTION DOC] Posts the target version, the host-local checkpoint directory and the source
        # directory to the pull-weights endpoint and returns the decoded reply (the pull is carried
        # out by the server; the weights are reloaded separately).
        """Have the engine sync every host it spans to target_version: each host pulls the
        published weights (a full checkpoint copied as-is, or deltas verified per-tensor and
        applied onto the local checkpoint) into its local checkpoint dir. The engine reloads
        it afterwards via update_weights_from_disk."""
        return await self._make_request(
            "pull_weights",
            {
                "local_checkpoint_dir": local_checkpoint_dir,
                "source_dir": source_dir,
                "target_version": target_version,
            },
        )

    async def update_weights_from_disk(
        self, model_path: str, load_format: str | None = None, weight_version: str | None = None
    ):
        # [FUNCTION DOC] Sends the model path, plus the load format and weight version only when
        # given, to the update-from-disk endpoint; returns the decoded reply.
        """Reload weights from *model_path* without restarting the engine.

        Used for non-updatable (frozen) models that overlap with megatron (after offload,
        weights are restored from disk instead of CPU cache), and by disk-delta weight sync
        to reload the patched host-local checkpoint.
        """
        payload = {"model_path": model_path}
        if load_format is not None:
            payload["load_format"] = load_format
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return await self._make_request("update_weights_from_disk", payload)

    async def init_weights_update_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend
    ):
        # [FUNCTION DOC] Asks the server to join a weight-update process group: sends master
        # address and port, the rank offset of this server's ranks, world size, group name and
        # communication backend. Returns the decoded reply.
        return await self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    async def destroy_weights_update_group(self, group_name):
        # [FUNCTION DOC] Asks the server to destroy the named weight-update group. Returns the
        # decoded reply, or none if any HTTP-library error occurs (swallowed, e.g. a freshly created
        # engine that has no such group).
        try:
            return await self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except httpx.HTTPError:
            # catch the case there the engine is just created and does not have the group.
            pass

    async def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
        selector: str = "all",
        load_format: str | None = None,
    ):
        # [FUNCTION DOC] Tells the server to receive weights over a distributed group: sends tensor
        # names, element-type names (the "torch." prefix removed), shapes, group name, flush-cache
        # flag (default false) and selector (default all); weight version and load format are sent
        # only if given. Returns the decoded reply.
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
            "selector": selector,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        if load_format is not None:
            payload["load_format"] = load_format
        return await self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    async def pause_generation(self, mode: str = "retract"):
        # [FUNCTION DOC] POSTs the pause mode (default retract) to the pause endpoint; raises on a
        # failed status and returns the raw response object.
        response = await GeneralHttpClientProvider.client().post(
            f"{self.server_url}/pause_generation",
            json={"mode": mode},
            headers=self._headers,
        )
        response.raise_for_status()
        return response

    async def continue_generation(self):
        # [FUNCTION DOC] POSTs an empty body to the continue endpoint; raises on a failed status and
        # returns the raw response object.
        response = await GeneralHttpClientProvider.client().post(
            f"{self.server_url}/continue_generation", json={}, headers=self._headers
        )
        response.raise_for_status()
        return response

    async def abort_all_requests(self, timeout: float | None = None):
        # [FUNCTION DOC] Sends an abort request with the abort-all flag set, with an optional
        # per-request timeout, and returns the decoded reply.
        return await self._make_request("abort_request", {"abort_all": True}, timeout=timeout)

    async def begin_weight_update(self, selector: str = "all", sync_base: bool = True):
        # [FUNCTION DOC] Sends the selector (default all) and the sync-base flag (default true) to
        # the begin endpoint and returns the decoded reply.
        """Open a weight-update session on the engine. sync_base=False declares an
        adapter-only session (no quant unpack; base tensors rejected)."""
        return await self._make_request("begin_weight_update", {"selector": selector, "sync_base": sync_base})

    async def end_weight_update(self, expected_lora_checksums=None):
        # [FUNCTION DOC] Sends the optional expected LoRA checksums to the end endpoint and returns
        # the decoded reply.
        """Close the weight-update session: re-finalize base weights (sync_base
        sessions) and apply the streamed LoRA stash."""
        return await self._make_request("end_weight_update", {"expected_lora_checksums": expected_lora_checksums})

    async def update_weight_version(self, weight_version: str, abort_all_requests: bool = False):
        # [FUNCTION DOC] Sets the server's weight version label (sent as the new version) with an
        # abort-all-requests flag (default false); returns the decoded reply.
        return await self._make_request(
            "update_weight_version",
            {"new_version": weight_version, "abort_all_requests": abort_all_requests},
        )

    async def start_profile(
        self,
        # The output directory
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        # [FUNCTION DOC] Starts profiling on the server by posting the output directory, start step,
        # number of steps (when set, profiling stops by itself after that many steps), activity
        # names, profile-by-stage flag (default false), with-stack and record-shapes options (all
        # sent as given, including none). Raises on a failed status; returns the raw response.
        response = await GeneralHttpClientProvider.client().post(
            f"{self.server_url}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
            headers=self._headers,
        )
        response.raise_for_status()
        return response

    async def stop_profile(self):
        # [FUNCTION DOC] POSTs an empty body to the stop-profile endpoint; raises on a failed
        # status and returns the raw response.
        response = await GeneralHttpClientProvider.client().post(
            f"{self.server_url}/stop_profile", json={}, headers=self._headers
        )
        response.raise_for_status()
        return response
