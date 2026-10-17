# [MODULE DOC] HELPER: Client-side tracer for sessions on the OpenAI-compatible session server. A
# session is created on one randomly chosen server instance (which then owns it for its lifetime); the
# agent talks to the session's URL; afterwards the tracer fetches the training samples the server
# assembled, always deleting the session. Request timeout for session calls: 120 seconds.
"""
Utilities for the OpenAI endpoint
"""

import asyncio
import logging
import random
from argparse import Namespace

from miles.rollout.generate_utils.sampling_mask import should_return_sampling_mask
from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    ROLLOUT_SAMPLING_MASK_FIELDS,
    SamplesReply,
    decode_samples_and_merge_input_sample,
)
from miles.rollout.session.types import CreateSessionRequest
from miles.utils.http_utils import post, post_bytes_no_retry
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_REQUEST_TIMEOUT = 120


class OpenAIEndpointTracer:
    # [CLASS DOC] HELPER: Handle of one live session: the owning instance's router URL, the session id,
    # the derived session URLs (driver-side and agent-side), the instance id (may be none) and the
    # allowlist of sample fields expected on the wire (must match the server's encoding). Create via the
    # static factory.

    def __init__(
        self,
        router_url: str,
        session_id: str,
        session_server_instance_id: str | None = None,
        samples_wire_fields: tuple[str, ...] = COMPUTED_FIELDS,
        agent_router_url: str | None = None,
    ):
        # [FUNCTION DOC] HELPER: Store the fields. The driver-side session URL is router URL + "/sessions/"
        # + session id; the agent-side one uses the agent router URL when given (the agent may run
        # outside the cluster) else the same router URL. No network calls.
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        # The agent may run outside the cluster; the driver's own calls stay on base_url.
        self.agent_base_url = f"{agent_router_url or router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id
        # The samples-wire allowlist must match the server's encode: v1 default,
        # extended under --use-session-server v2 (create() selects from args;
        # direct constructions keep v1).
        self.samples_wire_fields = samples_wire_fields

    @property
    def session_server_id(self) -> str:
        # [FUNCTION DOC] HELPER: Property: the router URL with a leading "http://" removed (host:port as
        # recorded in sample metadata); an https URL would keep its scheme.
        """``ip:port`` of the instance owning this session, as recorded in sample metadata."""
        return self.router_url.removeprefix("http://")

    @staticmethod
    async def create(args: Namespace, *, evaluation: bool = False, sampling_params: dict | None = None):
        # [FUNCTION DOC] HELPER: Create a session and return a tracer for it. Raises runtime error if no
        # session server instances are configured in args. Picks one instance uniformly at random (the
        # only routing decision; the session stays there). Filters the sampling parameters to the fields
        # known by the session creation request model, validates the body together with the evaluation
        # flag, and posts it (without none-valued fields) to the instance's sessions endpoint. Wire
        # fields: version-2 allowlist if the session server flag says v2, else the version-1 one, plus
        # the sampling mask fields when a sampling mask is to be returned. Returns a tracer with the
        # session id from the response, the instance id, the wire fields, and the instance's external
        # URL for the agent. Network I/O; validation errors from the request model propagate.
        instances = getattr(args, "session_server_instances", None)
        if not instances:
            raise RuntimeError(
                "session_server_instances is not set. Pass --use-session-server to start the session server."
            )
        # The only routing decision in the system: pick the owning instance once
        # per session; every later touch of the session reuses this URL.
        instance = random.choice(instances)
        session_url = instance.url
        # Drop engine-only sampling fields before validating the session creation body.
        session_params = {
            key: value for key, value in (sampling_params or {}).items() if key in CreateSessionRequest.model_fields
        }
        body = CreateSessionRequest.model_validate({**session_params, "evaluation": evaluation})
        use_v2 = getattr(args, "use_session_server", None) == "v2"
        samples_wire_fields = COMPUTED_FIELDS_V2 if use_v2 else COMPUTED_FIELDS
        if should_return_sampling_mask(args, sampling_params, evaluation=evaluation):
            samples_wire_fields += ROLLOUT_SAMPLING_MASK_FIELDS
        response = await post(f"{session_url}/sessions", body.model_dump(exclude_none=True), action="post")
        session_id = response["session_id"]
        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=instance.instance_id,
            samples_wire_fields=samples_wire_fields,
            agent_router_url=instance.external_url,
        )

    async def collect_samples(
        self, input_sample: Sample, *, max_seq_len: int | None, agent_metadata: dict | None = None
    ) -> SamplesReply:
        # [FUNCTION DOC] HELPER: Fetch the training samples the server assembled for this session and
        # merge them with the input sample. Posts a request body with the maximum sequence length (and the
        # agent metadata when not none) to the session's samples endpoint WITHOUT retries, with a 120 s
        # timeout; timeouts and transport errors propagate to the caller after cleanup. Always (in a
        # finally step) deletes the session with a 120 s limit; deletion failures are only logged as
        # warnings. Then decodes the response bytes in a worker thread using the wire field allowlist
        # and merges with the input sample. Returns the decoded reply.
        """Fetch server-assembled training samples for this session."""
        body: dict = {"max_seq_len": max_seq_len}
        if agent_metadata is not None:
            body["metadata"] = agent_metadata
        try:
            # Timeouts and transport errors propagate after cleanup, for `generate` to handle.
            payload = await post_bytes_no_retry(
                f"{self.base_url}/samples",
                body,
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        finally:
            try:
                await asyncio.wait_for(
                    post(self.base_url, {}, action="delete"),
                    timeout=_SESSION_REQUEST_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Failed to delete session {self.session_id} after collecting samples: {e}")

        return await asyncio.to_thread(
            decode_samples_and_merge_input_sample,
            payload,
            input_sample,
            fields=self.samples_wire_fields,
        )
