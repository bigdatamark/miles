# [MODULE DOC] Agent function plug-in for the agentic tool-call generate function: instead of
# running an agent loop itself, it sends each task to an external Harbor agent server over HTTP,
# pointing the remote agent at this run's model session endpoint, and returns the server's
# environment results (reward, exit status, evaluation report, agent metrics) as a plain mapping.
# Also provides an abort hook that tells the agent server to cancel in-flight trials.
# Environment variables: agent server URL (with an older alias; default local port 11000 for
# run, no default for abort), agent model name (older alias, default "model"), per-trial timeout
# in seconds (default 7200) and an optional admin secret for flushing. One shared HTTP client
# per process, with TCP keepalive and no request timeout (the caller enforces timeouts).
"""
Custom agent function for ``agentic_tool_call.generate``.

Dispatches to a Harbor-based agent server and returns env metadata
as a plain dict. The generate layer merges this into sample.metadata so
downstream reward models (--custom-rm-path) can extract reward, eval
reports, etc.

Task-type agnostic — the server + Harbor task directory handle all
differentiation (environment, grading harness, agent selection).
"""

import asyncio
import logging
import os
import socket
from typing import Any
from urllib.parse import urlsplit

import httpx
from miles.rollout.agentic.session import openai_session_url
from miles.utils.http_utils import post

logger = logging.getLogger(__name__)

# Backstop for an unreachable agent server; its own --agent-timeout should fire first.
_DEFAULT_AGENT_TRIAL_TIMEOUT_S = 7200

_agent_server_client: httpx.AsyncClient | None = None


def _agent_trial_timeout_s() -> int:
    # [FUNCTION DOC] HELPER: Read the trial timeout environment variable as an integer number of
    # seconds, defaulting to 7200; a non-integer value raises a value error.
    """Per-trial ceiling for the agent-server call, overridable via AGENT_TRIAL_TIMEOUT."""
    return int(os.environ.get("AGENT_TRIAL_TIMEOUT", _DEFAULT_AGENT_TRIAL_TIMEOUT_S))


def _get_agent_server_client() -> httpx.AsyncClient:
    # [FUNCTION DOC] HELPER: Lazily create (once per process, stored in a module global) and
    # return the shared async HTTP client. Transport sockets enable TCP keepalive with idle time 60
    # seconds, probe interval 30 seconds and 5 probes; when the platform lacks the named option
    # constants, Linux option numbers are used as fallbacks. Connection pool: at most 64
    # connections and 32 kept alive. No request timeout.
    # KNOWN ISSUE: the fallback option numbers are Linux-specific and would be wrong on other
    # operating systems that lack the named constants.
    """Return a client whose long-running requests survive idle network paths."""
    global _agent_server_client
    if _agent_server_client is None:
        socket_options = [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", 4), 60),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL", 5), 30),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT", 6), 5),
        ]
        transport = httpx.AsyncHTTPTransport(socket_options=socket_options)
        _agent_server_client = httpx.AsyncClient(
            transport=transport,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            timeout=None,
        )
    return _agent_server_client


async def _post_agent_server(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Post the payload as JSON with the shared client, raise on an HTTP
    # error status, and return the parsed JSON body.
    client = _get_agent_server_client()
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    # [FUNCTION DOC] Run one task through the agent server. Inputs: base_url of the run's model
    # endpoint for this session; prompt (unused); optional request kwargs (sampling parameters
    # forwarded as is); optional sample metadata; extra keyword arguments ignored. Request body:
    # all metadata fields, then the OpenAI-style session URL derived from base_url, the model name
    # prefixed with the provider label, and the sampling parameters; the max sequence length from
    # metadata converted to an integer when present; the session server id, when present in
    # metadata, REPLACED by the host and port of the session URL (metadata holds the cluster
    # address but the agent needs the address the base URL names); the session server instance id
    # when present. Posts to the run endpoint of the agent server under the trial timeout. Returns
    # a mapping with reward (default 0.0), exit status (default empty text), evaluation report
    # (default empty mapping) and agent metrics (default empty mapping). On timeout, cancellation
    # or any other error it logs and returns none instead of raising.
    # KNOWN ISSUE: swallowing task cancellation and returning none hides cancellation from the
    # caller (done on purpose for sibling failures, but it breaks normal cancellation semantics).
    """Run a single task instance via the Harbor agent server."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    agent_server_url = os.getenv(
        "AGENT_SERVER_URL",
        os.getenv("SWE_AGENT_URL", "http://localhost:11000"),
    )
    model_name = os.getenv(
        "AGENT_MODEL_NAME",
        os.getenv("SWE_AGENT_MODEL_NAME", "model"),
    )

    session_url = openai_session_url(base_url)

    request: dict[str, Any] = {
        **metadata,
        "base_url": session_url,
        "model": f"openai/{model_name}",
        "sampling_params": request_kwargs,
    }

    max_seq_len = metadata.get("max_seq_len")
    if max_seq_len is not None:
        request["max_seq_len"] = int(max_seq_len)

    if metadata.get("session_server_id") is not None:
        # The id in metadata is the cluster address; the agent server needs the address base_url names.
        request["session_server_id"] = urlsplit(session_url).netloc

    session_server_instance_id = metadata.get("session_server_instance_id")
    if session_server_instance_id is not None:
        request["session_server_instance_id"] = session_server_instance_id

    trial_timeout_s = _agent_trial_timeout_s()
    try:
        response = await asyncio.wait_for(
            _post_agent_server(f"{agent_server_url}/run", request),
            timeout=trial_timeout_s,
        )
    except asyncio.TimeoutError:
        logger.error(f"Agent server call timed out after {trial_timeout_s}s")
        return None
    except asyncio.CancelledError:
        logger.warning("Agent server call cancelled (sibling task failure?)")
        return None
    except Exception as e:
        logger.error(f"Agent server call failed: {e}")
        return None

    return {
        "reward": response.get("reward", 0.0),
        "exit_status": response.get("exit_status", ""),
        "eval_report": response.get("eval_report", {}),
        "agent_metrics": response.get("agent_metrics", {}),
    }


async def abort(args) -> None:
    # [FUNCTION DOC] Abort hook called when oversampling aborts generation. Collects the set of
    # session server instance ids from the plural instances list on args (non-empty ids only) and
    # the singular id attribute (kept for backward compatibility). Returns doing nothing if the
    # agent server URL (environment, with alias, no default) or the ids are missing. Otherwise, for
    # each id, posts a flush request with that id to the agent server's flush endpoint (URL
    # trailing slash stripped; up to 3 retries; bearer authorization header when the admin secret
    # environment variable is set), logging the result; failures are logged as warnings and do not
    # stop the remaining ids.
    """Teardown hook for oversampling abort (called by sglang_rollout.abort).

    When Miles has enough samples and aborts SGLang, the in-flight Harbor trials
    keep looping and hitting SGLang until they hit their own max_seq_len/timeout.
    Flush the agent server so it cancels those ``/run`` tasks and releases their
    containers. No-op unless AGENT_SERVER_URL and session_server_instance_id are
    available.
    """
    agent_server_url = os.getenv("AGENT_SERVER_URL", os.getenv("SWE_AGENT_URL"))

    instances = getattr(args, "session_server_instances", None) or []
    instance_ids = {instance.instance_id for instance in instances if instance.instance_id}
    singular = getattr(args, "session_server_instance_id", None)  # back-compat / child path
    if singular:
        instance_ids.add(singular)

    if not agent_server_url or not instance_ids:
        return

    headers = None
    admin_secret = os.getenv("HARBOR_ADMIN_SECRET")
    if admin_secret:
        headers = {"Authorization": f"Bearer {admin_secret}"}

    for instance_id in instance_ids:
        try:
            result = await post(
                f"{agent_server_url.rstrip('/')}/flush",
                {"session_server_instance_id": instance_id},
                max_retries=3,
                headers=headers,
            )
            logger.info(f"Flushed agent server {agent_server_url}: {result}")
        except Exception as e:
            logger.warning(f"Failed to flush agent server {agent_server_url}: {e}")
