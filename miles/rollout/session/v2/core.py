# [MODULE DOC]
# The v2 (tree-of-trajectories) session core. Subclasses the v1 core, inheriting the transport
# shell (health, create, delete, raw proxy) and overriding the session semantics: metadata is
# derived from the latest committed generation, get-session returns the records along its path,
# samples are built per leaf and passed through the user-configurable picker and postprocessor
# hooks, and chat completion attaches each request under the right tree node and commits the result.
import json
import logging
import time

from starlette.responses import Response

from miles.rollout.session.core import (
    JSON_MEDIA_TYPE,
    ProxyRequest,
    SessionCore,
    _chat_client_response,
    _render_json,
    _samples_response,
    extract_completion,
    proxy_result_to_response,
)
from miles.rollout.session.errors import SessionNotFoundError, TokenizationError
from miles.rollout.session.request_args import filter_turn_args, parse_chat_request
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, ROLLOUT_SAMPLING_MASK_FIELDS, encode_samples
from miles.rollout.session.types import GetSessionResponse, SessionRecord
from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY, build_session_rollout_metrics
from miles.rollout.session.v2.session_state import (
    SessionRegistryV2,
    commit_generation,
    prepare_token_ids_and_request_args,
)
from miles.rollout.session.v2.utils import build_leaf_material, tree_metadata
from miles.utils.function_registry import load_function

logger = logging.getLogger(__name__)


class SessionCoreV2(SessionCore):
    # [CLASS DOC]
    # Tree-serving session core. Extra attributes beyond the base: the sample picker hook (takes
    # the list of candidate leaf samples and the session metadata, returns a subset without
    # duplicates) and the sample postprocessor hook (takes the picked samples and metadata,
    # returns the final samples).
    """``SessionCore`` with tree serving: overrides the session-semantics
    methods (positioning/commit, metadata, samples op), inherits the
    transport shell (health, create/delete, raw proxy)."""

    def __init__(
        self, backend, registry: SessionRegistryV2, config, session_server_instance_id=None, *, use_addition_r3=False
    ):
        # [FUNCTION DOC] Initialize the base core, then load the picker and postprocessor
        # functions from the import paths in the config through the function registry, requiring
        # them to be resolvable synchronously (registry is process-local, so only import paths
        # work in production). A failed load propagates.
        super().__init__(backend, registry, config, session_server_instance_id, use_addition_r3=use_addition_r3)
        # Import-path only in production: function_registry is process-local.
        self.sample_picker = load_function(config.session_sample_picker_path, sync_required=True)
        self.sample_postprocessor = load_function(config.session_sample_postprocessor_path, sync_required=True)

    def _session_metadata(self, session_id: str, session) -> dict:
        # [FUNCTION DOC] HELPER: Build the metadata mapping like the base class, but based on the
        # latest committed generation: optional tito session mismatch (a tokenization error is
        # logged and ignored), accumulated token ids of the latest generation (empty list if
        # none), max trim tokens, a tree description block, and the filtered turn args of the latest
        # generation (empty mapping if none).
        """Mirrors ``core.SessionCore._session_metadata``: token ids come from
        the latest committed generation's path, plus the ``tree`` block."""
        metadata: dict = {}
        try:
            mismatch = self.registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        latest = session.latest()
        metadata["accumulated_token_ids"] = latest.token_ids if latest is not None else []
        metadata["max_trim_tokens"] = self.registry.tito_tokenizer.max_trim_tokens
        metadata["tree"] = tree_metadata(session)
        metadata["turn_args"] = filter_turn_args(latest.turn_args) if latest is not None else {}
        return metadata

    async def get_session(self, session_id: str) -> Response:
        # [FUNCTION DOC] Reply HTTP 200 JSON with the session id, the records of the nodes on the
        # latest committed generation's path (empty list if none) and the metadata.
        """Mirrors ``core.SessionCore.get_session``, serving the records along the
        latest committed generation's path."""
        session = self.registry.get_session(session_id)
        metadata = self._session_metadata(session_id, session)
        latest = session.latest()
        records = [node.record for node in latest.path_nodes()] if latest is not None else []
        payload = GetSessionResponse(session_id=session_id, records=records, metadata=metadata)
        return Response(
            content=_render_json(payload.model_dump(mode="json")), status_code=200, media_type=JSON_MEDIA_TYPE
        )

    async def collect_samples(
        self, session_id: str, *, max_seq_len: int | None, agent_metadata: dict | None = None
    ) -> Response:
        # [FUNCTION DOC]
        # Build samples for the session and reply with the encoded payload. Wire fields: the v2
        # computed fields, plus the sampling mask field when the session has sampling-support
        # replay. If agent metadata is given it is stored in the metadata under agent. An empty
        # tree replies with empty reason "no_records". Candidate samples (one per leaf) are built
        # from the tree with the config, registry, session id, optional max sequence length and
        # additional-replay flag; assertion/value errors become HTTP 422 plain text. No candidates
        # replies empty reason "all_truncated". Hooks: the picker is called with candidates and
        # metadata and must return a duplicate-free subset of the same sample objects (checked by
        # identity; otherwise error); then the postprocessor is called with the picked samples and
        # metadata. Any exception in the hook stage becomes HTTP 422 plain text naming both hook
        # paths. An empty final list replies empty reason "all_truncated". If a speculative
        # algorithm is configured, rollout metrics for the session are put in the metadata only
        # now (after hooks, so hooks cannot alter them). Runs synchronously w.r.t. the event loop
        # (no awaits) so it cannot interleave with chat commits.
        """Samples op: assemble one raw sample per leaf, then run the
        pick/post-process hook pipeline and encode the result.

        Synchronous on the server loop (no await), so the session read cannot
        interleave with chat commits. Deterministic assembly/hook failures map
        to 422; unknown exceptions propagate.
        """
        session = self.registry.get_session(session_id)
        metadata = self._session_metadata(session_id, session)
        fields = COMPUTED_FIELDS_V2
        if session.sampling_support_replay:
            fields += ROLLOUT_SAMPLING_MASK_FIELDS
        if agent_metadata is not None:
            metadata["agent"] = agent_metadata
        if not session.tree.nodes:
            return _samples_response(encode_samples([], metadata, empty_reason="no_records", fields=fields))

        try:
            material = build_leaf_material(
                self.config,
                session,
                self.registry,
                session_id=session_id,
                max_seq_len=max_seq_len,
                use_addition_r3=self.use_addition_r3,
            )
        except (AssertionError, ValueError) as exc:
            return Response(content=str(exc).encode(), status_code=422, media_type="text/plain")
        if not material:
            return _samples_response(encode_samples([], metadata, empty_reason="all_truncated", fields=fields))

        # Hook lane: a policy bug is a deterministic 422 carrying the hook's
        # identity, never a masked 500 (server death stays loud).
        try:
            picked = self.sample_picker(material, metadata)
            picked_ids = [id(sample) for sample in picked]
            allowed = {id(sample) for sample in material}
            if any(sample_id not in allowed for sample_id in picked_ids) or len(picked_ids) != len(set(picked_ids)):
                raise ValueError(
                    "pick hook must return a subset of its input samples without duplicates (pure selection)"
                )
            samples = self.sample_postprocessor(picked, metadata)
        except Exception as exc:
            body = (
                f"session sample hook failed (picker={self.config.session_sample_picker_path}, "
                f"postprocessor={self.config.session_sample_postprocessor_path}): {exc}"
            )
            return Response(content=body.encode(), status_code=422, media_type="text/plain")
        if not samples:
            return _samples_response(encode_samples([], metadata, empty_reason="all_truncated", fields=fields))
        # Hooks may inspect or mutate session metadata, so publish the
        # authoritative server-owned value only at the wire boundary.
        if self.config.sglang_speculative_algorithm is not None:
            metadata[SESSION_ROLLOUT_METRICS_KEY] = build_session_rollout_metrics(session_id, session.tree.nodes)
        return _samples_response(encode_samples(samples, metadata, fields=fields))

    async def chat_completions(
        self, session_id: str, *, method: str, query: str, headers: dict, body: bytes
    ) -> Response:
        # [FUNCTION DOC]
        # Tree version of chat completion. Raises not found if the session is missing or closing.
        # Phase 1 (lock): parse the body; prepare the request, which also determines the tree node
        # to attach under (the parent, or none for a new root); set the additional-replay start
        # offset relative to the parent's token ids (empty if no parent); serialize the body.
        # Phase 2 (no lock): proxy to the backend chat path with the routing key header set to the
        # session id; a non-200 reply passes through unrecorded.
        # Phase 3: validate the response, post-process the assistant message; under the lock again,
        # if the session closed meanwhile just reply, else build a record and commit a new
        # generation node under the parent with the request messages, assistant message, prompt and
        # completion ids, trim limit, record, response id (empty if missing), finish reason (empty
        # if missing) and the request as turn args. Unlike the linear core there is no concurrent
        # state-change check. The reply is built from the stored response as in the base class.
        """Proxy a chat completion through the backend with TITO token tracking.

        Flow: prepare pretokenized input_ids (lock held briefly) → proxy to
        backend (NO lock) → validate response → update trajectory checkpoint and
        append record (lock held briefly). The lock is NOT held during the long
        inference call so DELETE/other ops are not blocked if the agent disconnects.
        """
        request_timestamp = time.time()
        session = self.registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

        # --- Phase 1: prepare request (lock held briefly) ---
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            client_args = parse_chat_request(body)
            prepared, attach_parent = prepare_token_ids_and_request_args(
                session,
                client_args,
                config=self.config,
                tito_tokenizer=self.registry.tito_tokenizer,
                message_matcher=self.registry.message_matcher,
            )
            request_body, tito_tokenizer = prepared.body, self.registry.tito_tokenizer
            response_intent = prepared.response_intent
            request_messages = request_body.get("messages", [])
            prompt_token_ids = request_body["input_ids"]
            logger.debug("Using TITO input_ids: %d tokens", len(prompt_token_ids))

            checkpoint_token_ids = attach_parent.token_ids if attach_parent is not None else []
            self._maybe_request_addition_r3(request_body, checkpoint_token_ids, prompt_token_ids)

            proxy_body = json.dumps(request_body).encode()
        # --- lock released ---

        # --- Phase 2: proxy to backend (NO lock held) ---
        headers = {**headers, "X-SMG-Routing-Key": session_id}
        result = await self.backend.do_proxy(
            ProxyRequest(method=method, query=query), "v1/chat/completions", body=proxy_body, headers=headers
        )

        # Non-200 (e.g. 400 context too long) passes through unrecorded so the
        # agent can retry or handle the error.
        if result["status_code"] != 200:
            return proxy_result_to_response(result)

        response, choice, assistant_message, completion_token_ids = extract_completion(result)
        assistant_message = tito_tokenizer.postprocess_completion(
            choice=choice,
            assistant_message=assistant_message,
            completion_token_ids=completion_token_ids,
        )

        # --- Phase 3: update state (lock held briefly) ---
        async with session.lock:
            if session.closing:
                logger.debug("Session %s closed during proxy, skipping state update", session_id)
                return _chat_client_response(result, response, response_intent)

            record = SessionRecord(
                timestamp=time.time(),
                request_timestamp=request_timestamp,
                method=method,
                path="/v1/chat/completions",
                status_code=result["status_code"],
                request=request_body,
                response=response,
            )
            commit_generation(
                session,
                parent=attach_parent,
                request_messages=request_messages,
                assistant_message=assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                max_trim_tokens=tito_tokenizer.max_trim_tokens,
                record=record,
                response_id=response.get("id", ""),
                finish_reason=choice.get("finish_reason") or "",
                turn_args=request_body,
            )
        # --- lock released ---

        return _chat_client_response(result, response, response_intent)
