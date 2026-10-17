# [MODULE DOC] Reference checkpoint-eval backend that evaluates trainer snapshots on a standalone
# sglang server outside the Ray-managed rollout fleet.
# The module defines one class, a concrete checkpoint eval function. It either spawns a local
# sglang server as a subprocess on explicitly chosen GPUs, or attaches to an already running
# server given by URL. For every eval point it waits (once) for the server to be healthy, pins
# the server to the exported HF snapshot by asking it to reload weights from disk and verifying
# the reported weight version, then runs the standard eval datasets against it using the
# training args retargeted at that server. Configuration comes from four environment variables
# read in the constructor: the attach URL, the comma separated GPU list used to launch, the
# launch port (default 31000) and extra raw server flags (split shell-style). The URL takes
# precedence; if neither URL nor GPUs is set the constructor raises a value error.
"""Self-contained external sglang eval backend — a reference ``CheckpointEvalFn``.

``__init__`` prepares the backend: launch our own sglang server on spare GPUs, or
attach to one already running anywhere. Each eval then pins the server to the
snapshot in ``checkpoint_dir`` and runs the standard miles eval datasets. Because
the fn runs inside the training job, all eval config (datasets, sampling, rm,
lora, ...) comes from the real training args — nothing is hand-copied.

Config (env vars, set them in the run launcher):

    MILES_EXTERNAL_EVAL_URL          attach to a running server, e.g. http://host:31000
    MILES_EXTERNAL_EVAL_GPUS         launch our own, e.g. "6,7" (tp = #gpus); the
                                     server runs on this node; ignored when URL is set
    MILES_EXTERNAL_EVAL_PORT         launch-mode port (default 31000)
    MILES_EXTERNAL_EVAL_SERVER_ARGS  extra raw sglang flags for launch mode, e.g.
                                     "--attention-backend fa3 --mem-fraction-static 0.9"

See ``run_qwen3_5_4b_fully_async_eval.py`` for a full launcher. A non-sglang black
box implements the same contract without any of this module: ``evaluate_checkpoint``
submits ``checkpoint_dir`` to the external service and maps its response into
``RolloutFnEvalOutput(data=...)``; raise ``EvalSkip(reason)`` for an attributable skip.
"""

import asyncio
import os
import shlex
import subprocess
import sys

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnEvalOutput
from miles.rollout.checkpoint_eval import CheckpointEvalFn, retarget_args
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.utils.http_utils import get, post, wait_http_ok


class ExternalSglangEvalFn(CheckpointEvalFn):
    # [CLASS DOC] Checkpoint eval function backed by one standalone sglang server (one engine).
    # State: the child process handle (None in attach mode), the server base URL, the training
    # args retargeted to the server, a lazily created generation state, a cache dictionary
    # passed to the eval dataset runner across calls, and a flag recording that the health wait
    # has completed. Lifecycle: constructed once, evaluate_checkpoint called per eval point,
    # dispose called at shutdown.
    """Launch (or attach to) a standalone sglang server and eval snapshots on it."""

    def __init__(self, input: RolloutFnConstructorInput):
        # [FUNCTION DOC] Resolve the backend and prepare (but do not wait for) the server.
        # Reads the environment variables described in the module doc. If no URL is set: requires
        # the GPU list (else raises a value error), takes the port (default 31000), uses the
        # number of listed GPUs as the tensor parallel size, and spawns the sglang launch module
        # as a subprocess with the training run's HF checkpoint as the model path, bound to
        # localhost, static memory fraction 0.8, trust-remote-code, plus any extra flags. The
        # child's visible GPUs are forced to the list because the manager process under Ray sees
        # no GPUs. The URL then becomes localhost with that port. The URL is split at the last
        # colon into host and port (an http prefix is stripped; the port must be an integer or
        # the conversion fails), a normalized base URL is stored, and the eval args are produced
        # by retargeting the training args to that host/port with engine and GPU counts both
        # equal to the GPU count (1 in attach mode). The health wait is deliberately deferred so
        # server loading does not block training start.
        # Side effect: may start a subprocess.
        super().__init__(input)
        args = input.args
        url = os.environ.get("MILES_EXTERNAL_EVAL_URL")
        gpus = os.environ.get("MILES_EXTERNAL_EVAL_GPUS")
        self._proc: subprocess.Popen | None = None
        num_gpus = 1
        if url is None:
            if gpus is None:
                raise ValueError(
                    "ExternalSglangEvalFn needs a backend: set MILES_EXTERNAL_EVAL_URL (attach) "
                    "or MILES_EXTERNAL_EVAL_GPUS (launch our own, e.g. '6,7')"
                )
            port = int(os.environ.get("MILES_EXTERNAL_EVAL_PORT", "31000"))
            num_gpus = len(gpus.split(","))
            cmd = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                args.hf_checkpoint,
                "--tp",
                str(num_gpus),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--mem-fraction-static",
                "0.8",
                "--trust-remote-code",
                *shlex.split(os.environ.get("MILES_EXTERNAL_EVAL_SERVER_ARGS", "")),
            ]
            # Explicit GPU pinning: under Ray the manager process has no visible GPUs.
            self._proc = subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus})
            url = f"http://127.0.0.1:{port}"

        ip, _, srv_port = url.removeprefix("http://").rpartition(":")
        self._url = f"http://{ip}:{srv_port}"
        # Sizes the client-side concurrency semaphore (one engine behind the URL).
        self._eval_args = retarget_args(args, ip, int(srv_port), num_gpus, num_gpus)
        self._state: GenerateState | None = None
        self._cache: dict = {}
        self._ready = False  # health-wait deferred: server load must not block training start

    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        # [FUNCTION DOC] Evaluate one snapshot on the server and return the eval results.
        # Inputs: checkpoint_dir is the exported HF snapshot directory; input carries the weight
        # version string. Steps: on first call poll the server's generation health endpoint until
        # OK (timeout 1800 seconds, errors from the wait propagate); pin the server to the
        # snapshot (raises a runtime error if pinning fails); lazily create the generation state
        # from the retargeted eval args; run all configured eval datasets with the shared cache.
        # Returns an eval output whose data is the dataset runner's result.
        # Side effects: network calls to the server, server weights replaced.
        if not self._ready:
            await wait_http_ok(f"{self._url}/health_generate", timeout=1800.0)
            self._ready = True
        await self._pin(checkpoint_dir, input.weight_version)
        if self._state is None:
            self._state = GenerateState(self._eval_args)
        return RolloutFnEvalOutput(data=await run_eval_datasets(self._state, self._cache))

    async def _pin(self, checkpoint_dir: str, weight_version: str, *, retries: int = 2) -> None:
        # [FUNCTION DOC] HELPER: Load the snapshot into the server and verify the weight version.
        # Makes up to retries attempts (default 2, i.e. one retry). Each attempt posts a
        # reload-from-disk request carrying the model path and weight version, then reads the
        # version back from the server's model info; the attempt is bounded by a 600 second
        # timeout and any exception (including timeout) just moves to the next attempt. Returns
        # as soon as the reported version, compared as a string, equals the expected one. After
        # all attempts fail raises a runtime error naming the checkpoint and expected version.
        # KNOWN ISSUE: the docstring says "retry once", which matches the default of 2 attempts,
        # but exceptions are swallowed silently so the real failure cause is lost.
        """Pin discipline: after loading, read the version back and compare — never
        trust a silent load; retry once, then fail the point loudly."""

        async def load_and_read():
            # [FUNCTION DOC] HELPER: One load attempt. Posts the reload request to the server,
            # then fetches model info and returns its weight version field (None if absent).
            await post(
                f"{self._url}/update_weights_from_disk",
                {"model_path": checkpoint_dir, "weight_version": weight_version},
            )
            return (await get(f"{self._url}/model_info")).get("weight_version")

        for _attempt in range(retries):
            try:
                loaded = await asyncio.wait_for(load_and_read(), timeout=600.0)
            except Exception:
                continue
            if str(loaded) == weight_version:
                return
        raise RuntimeError(f"weight_version pin failed for {checkpoint_dir} (expected {weight_version})")

    async def dispose(self) -> None:
        # [FUNCTION DOC] Shut down the backend: if a server subprocess was launched, send it a
        # terminate signal (no wait for exit, no kill escalation). No-op in attach mode.
        if self._proc is not None:
            self._proc.terminate()
