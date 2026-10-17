# [MODULE DOC]
# Submits a ray job and follows it. Two lifetimes: "independent" shells out to the ray job submit
# command (job keeps running if the launcher dies; the command streams logs and waits), and
# "launcher" uses the job submission client, tails logs, and stops the job when the launcher
# finishes or is interrupted (terminate/interrupt signals are intercepted and turned into a clean
# stop and an exit status of 128 plus the signal number).
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import uuid
from contextlib import suppress
from typing import Literal

from ray.job_submission import JobStatus, JobSubmissionClient

from miles.utils.external_utils.command_utils.common import run_shell_command as exec_command_cpu


def run_ray_job(
    *, entrypoint: str, runtime_env: dict, job_lifetime: Literal["independent", "launcher"] = "independent"
) -> None:
    # [FUNCTION DOC] Runs an entrypoint command as a ray job and blocks until it finishes. Independent
    # lifetime: runs a shell command that disables proxying for loopback, forces unbuffered output,
    # and invokes ray job submit with the dashboard address (loopback port 8265 unless the ray
    # address environment variable is set, in which case the address option is omitted), the runtime
    # environment passed as quoted json, and the entrypoint. Launcher lifetime: uses the address from
    # the environment (default same loopback dashboard) and the launcher-owned job path. Any other
    # lifetime raises a value error. The entrypoint is interpolated into a shell command unquoted in
    # the independent case.
    """Submit and follow a job; launcher-owned jobs stop when the launcher exits."""
    if job_lifetime == "independent":
        exec_command_cpu(
            "export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && "
            f"""ray job submit {'' if 'RAY_ADDRESS' in os.environ else '--address="http://127.0.0.1:8265" '}"""
            f"--runtime-env-json={shlex.quote(json.dumps(runtime_env))} -- {entrypoint}"
        )
    elif job_lifetime == "launcher":
        _run_launcher_owned_job(
            address=os.environ.get("RAY_ADDRESS", "http://127.0.0.1:8265"),
            entrypoint=entrypoint,
            runtime_env=runtime_env,
        )
    else:
        raise ValueError(f"Unknown job lifetime: {job_lifetime}")


def _run_launcher_owned_job(*, address: str, entrypoint: str, runtime_env: dict) -> None:
    # [FUNCTION DOC] HELPER: Adds unbuffered output to the runtime environment's variables (without
    # mutating the argument). Temporarily extends the lower-case no-proxy environment variable with
    # the existing value (falling back to the upper-case one) plus loopback and localhost so the
    # dashboard is reached directly; runs the async submit-and-stop routine with a fresh unique
    # submission id; then restores the variable (deleting it if it was unset before).
    runtime_env = {**runtime_env, "env_vars": {**runtime_env.get("env_vars", {}), "PYTHONUNBUFFERED": "1"}}
    previous_no_proxy = os.environ.get("no_proxy")
    os.environ["no_proxy"] = ",".join(
        filter(None, (previous_no_proxy or os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"))
    )
    try:
        asyncio.run(
            _run_and_stop_job(
                address=address,
                submission_id=f"miles-{uuid.uuid4().hex}",
                entrypoint=entrypoint,
                runtime_env=runtime_env,
            )
        )
    finally:
        if previous_no_proxy is None:
            del os.environ["no_proxy"]
        else:
            os.environ["no_proxy"] = previous_no_proxy


async def _run_and_stop_job(address: str, submission_id: str, entrypoint: str, runtime_env: dict) -> None:
    # [FUNCTION DOC] HELPER: Installs handlers for interrupt and terminate that record the signal
    # number and set an async stop event. Submits the job, prints a submission message, follows it
    # until it ends or a stop is requested; in all cases (finally) runs the ray command line job stop
    # (waits for a terminal status; check enabled, 45 second timeout, so a failure or timeout
    # raises). Afterwards: if a signal was received raises a system exit with 128 plus the signal;
    # else fetches the status and raises a runtime error including the job message unless it
    # succeeded. Previous signal handlers are always restored.
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()
    stop_signal = None

    def request_stop(signum, frame):
        # [FUNCTION DOC] HELPER: Signal handler; remembers the signal and thread-safely sets the
        # stop event on the event loop.
        nonlocal stop_signal
        stop_signal = signum
        loop.call_soon_threadsafe(stop_requested.set)

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        client = JobSubmissionClient(address)
        try:
            client.submit_job(submission_id=submission_id, entrypoint=entrypoint, runtime_env=runtime_env)
            print(f"Ray job {submission_id} submitted to {address}", flush=True)
            await _follow_job(client, submission_id, stop_requested)
        finally:
            # Unlike the stop HTTP response, the CLI waits for a terminal job status.
            subprocess.run(
                [sys.executable, "-m", "ray.scripts.scripts", "job", "stop", "--address", address, submission_id],
                check=True,
                timeout=45,
            )
        if stop_signal is not None:
            raise SystemExit(128 + stop_signal)
        status = client.get_job_status(submission_id)
        if status != JobStatus.SUCCEEDED:
            raise RuntimeError(
                f"Ray job {submission_id} ended with {status}: {client.get_job_info(submission_id).message}"
            )
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


async def _follow_job(client: JobSubmissionClient, submission_id: str, stop_requested: asyncio.Event) -> None:
    # [FUNCTION DOC] HELPER: Runs a log-tailing task and a stop-waiting task concurrently, returns
    # when the first completes (re-raising its exception if any), and always cancels both tasks and
    # awaits them, swallowing cancellation.
    async def print_logs():
        # [FUNCTION DOC] HELPER: Streams the job's log chunks to standard output unbuffered, with no
        # added newlines, until the stream ends.
        async for chunk in client.tail_job_logs(submission_id):
            print(chunk, end="", flush=True)

    logs_task = asyncio.create_task(print_logs())
    stop_task = asyncio.create_task(stop_requested.wait())
    try:
        done, _ = await asyncio.wait((logs_task, stop_task), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in (logs_task, stop_task):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
