# [MODULE DOC] HELPER: Shared setup for the Tinker GPU acceptance tests. Downloads the base model
# (Qwen3-4B-Instruct-2507) and runs the example gateway server script as a background subprocess
# in its own session, exposing a context manager that yields the gateway's base url on the
# loopback address (fixed port 10613) once its health endpoint answers (startup timeout 1200
# seconds), and tears it down on exit. The serve command uses tensor parallel 2, expert
# parallel 1, LoRA rank 8 and alpha 16 and passes the base model name as an extra argument.

"""Shared gateway setup for Tinker GPU acceptance tests."""

import os
import signal
import subprocess
import time
import urllib.request
from contextlib import contextmanager, suppress

from miles.utils.external_utils import command_utils
from miles.utils.http_utils import is_port_available

MODEL_NAME = "Qwen3-4B-Instruct-2507"
BASE_MODEL = f"Qwen/{MODEL_NAME}"
GATEWAY_PORT = 10613
SERVE_TIMEOUT_S = 1200


def prepare_gateway():
    # [FUNCTION DOC] HELPER: Creates the root-level models directory and downloads the base model
    # from Hugging Face into it, through the default config's backend.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models")
    U.exec_command_cpu(f"hf download {BASE_MODEL} --local-dir /root/models/{MODEL_NAME}")


def _wait_for_gateway(server: subprocess.Popen) -> None:
    # [FUNCTION DOC] HELPER: Polls the gateway health endpoint (2 second request timeout, 5 second
    # sleep between failed attempts, up to 1200 seconds) and returns when it answers. Raises a
    # runtime error immediately if the server process has exited (with its exit code) and a
    # timeout error if the deadline passes. Only connection-level errors are retried.
    deadline = time.time() + SERVE_TIMEOUT_S
    url = f"http://127.0.0.1:{GATEWAY_PORT}/api/v1/healthz"
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"gateway exited during startup with code {server.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"gateway not serving after {SERVE_TIMEOUT_S}s")


@contextmanager
def running_gateway():
    # [FUNCTION DOC] HELPER: Context manager yielding the gateway base url. Refuses (runtime error)
    # if something already listens on the fixed port. Starts the serve command through a shell with
    # exec, in a new session, then waits until healthy. On exit (also on error in the body): sends
    # a terminate signal (ignoring a missing process), waits up to 180 seconds, and raises a
    # runtime error if the exit code is neither 0 nor the code of death by the terminate signal.
    # If any of that fails (including the wait timing out), force-kills the whole process group,
    # waits 30 seconds, forcibly stops ray (raising if that fails), and re-raises the original
    # error, because a stuck launcher cannot finish its own ray cleanup. If startup itself
    # fails, the same teardown runs.
    if not is_port_available(GATEWAY_PORT):
        raise RuntimeError(f"port {GATEWAY_PORT} already has a listener; refusing to reuse a gateway not started here")
    serve_cmd = (
        "python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve "
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        "--model-type qwen3-4B-Instruct-2507 --tp 2 --ep 1 --lora-rank 8 --lora-alpha 16 "
        f'--extra-args "--tinker-base-model {BASE_MODEL}"'
    )
    server = subprocess.Popen(["bash", "-c", f"exec {serve_cmd}"], start_new_session=True)
    try:
        _wait_for_gateway(server)
        yield f"http://127.0.0.1:{GATEWAY_PORT}"
    finally:
        try:
            with suppress(ProcessLookupError):
                server.terminate()
            returncode = server.wait(timeout=180)
            if returncode not in (0, 128 + signal.SIGTERM):
                raise RuntimeError(f"gateway launcher failed during shutdown with code {returncode}")
        except BaseException:
            # A stuck launcher cannot finish its own Ray cleanup.
            with suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=30)
            subprocess.run(["ray", "stop", "--force"], check=True, timeout=120)
            raise
