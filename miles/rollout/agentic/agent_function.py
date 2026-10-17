# [MODULE DOC] Run a user-supplied asynchronous agent function either inline on the rollout event loop
# or in a fresh child Python process per call ("subproc" mode). Child protocol: the parent starts
# the interpreter with a bootstrap one-liner, passing its own pid as the only argument, and writes
# to the child's stdin two consecutive pickles: first the parent's module search path (so the child
# can import the same code before importing Miles), then a tuple of (function, keyword arguments,
# cancel grace seconds, leftover cleanup seconds). The child writes ONE pickled outcome to the
# original stdout and then closes it: either ("ok", value) or ("error", exception, traceback text);
# an empty stream means the child died without an outcome. Parent death kills the child (Linux
# parent-death signal). Cancelling the parent call sends SIGTERM, which cancels the agent and gives
# it a grace period (120 s) to clean up, then a further 600 s for leftover threads once the call is
# over; SIGKILL is the parent-side backstop after the sum of both. Environment of the child carries
# the rollout limits scope so cross-process limits are shared.
"""Call a custom agent function on the rollout loop or in a child process of its own.

Every sample's agent function runs concurrently in the rollout process. On one
event loop, a synchronous step in one episode (a large response to parse, a slow
file write) stalls every other episode, long enough under many episodes for their
sandbox SDK requests to time out. ``subproc`` gives each call a fresh child
process, so a slow step, a crash or leaked state stays inside its own episode.
It is a plain subprocess rather than a Ray task because
``--cluster-backend kubernetes`` runs the rollout without Ray.
"""

import asyncio
import contextlib
import ctypes
import logging
import os
import pickle
import signal
import sys
import threading
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, BinaryIO, NoReturn

from miles.rollout.agentic import rollout_limits
from miles.utils.logging_utils import configure_logger_raw

logger = logging.getLogger(__name__)

AGENT_FUNCTION_MODES = ("subproc", "inline")

# A cancelled agent gets this long to run its own cleanup (closing its sandbox) before its process exits.
_CANCEL_GRACE_S = 120.0
# Once the call is over, cleanup the agent left running (a thread closing a sandbox) gets this long.
_LEFTOVER_CLEANUP_S = 600.0

# the child takes the rollout's import path before it imports anything from miles
_CHILD_BOOTSTRAP = (
    "import pickle, sys; sys.path[:] = pickle.load(sys.stdin.buffer); "
    "from miles.rollout.agentic.agent_function import _child_main; _child_main(int(sys.argv[1]))"
)

_PR_SET_PDEATHSIG = 1


class _RemoteTraceback(Exception):
    # [CLASS DOC] HELPER: Exception that carries the child process's formatted traceback text; chained
    # as the cause of the re-raised child error so the remote stack is visible (same technique as the
    # standard process pool). Its string form is exactly the traceback text.
    """The agent process's traceback, chained under the error it raised, as concurrent.futures does."""

    def __init__(self, tb: str):
        # [FUNCTION DOC] HELPER: Store the traceback text and pass it as the exception message.
        super().__init__(tb)
        self.tb = tb

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: The traceback text unchanged.
        return self.tb


async def call_agent_function(fn: Callable[..., Awaitable[Any]], *, mode: str, **kwargs: Any) -> Any:
    # [FUNCTION DOC] Run the agent function with the keyword arguments and return its result. Inline
    # mode (checked first): simply awaits it on the current loop. Any other mode value is treated as
    # subproc (not validated here; the allowed names are in the module constant). Subproc mode:
    # serializes the call BEFORE spawning (so unserializable arguments fail in the caller), starts a
    # child interpreter with piped stdin and stdout (stderr inherited) and an environment that adds
    # the rollout limits scope of this process, exchanges data with it, and interprets the outcome.
    # If the awaiting task is cancelled, the child is asked to terminate (see terminate helper) and
    # the cancellation propagates. If the child produced no output, waits for it to exit and raises a
    # runtime error with its exit code. Otherwise returns the child's value, or re-raises the child's
    # exception with the remote traceback chained. KNOWN ISSUE: errors other than cancellation raised
    # while exchanging data leave the child running (it only dies with its parent).
    """Await ``fn(**kwargs)`` on this loop (``inline``) or in a child process of its own (``subproc``)."""
    if mode == "inline":
        return await fn(**kwargs)
    # pickled before the process starts, so an unpicklable argument fails here
    call = pickle.dumps(sys.path) + pickle.dumps((fn, kwargs, _CANCEL_GRACE_S, _LEFTOVER_CLEANUP_S))
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _CHILD_BOOTSTRAP,
        str(os.getpid()),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        env={**os.environ, rollout_limits.SCOPE_ENV_VAR: rollout_limits.current_scope()},
    )
    try:
        output = await _exchange(proc, call)
    except asyncio.CancelledError:
        _terminate(proc)
        raise
    if not output:
        returncode = await proc.wait()
        raise RuntimeError(f"The agent process exited with code {returncode} before returning a result")
    return _unpack_outcome(output)


async def _exchange(proc: asyncio.subprocess.Process, call: bytes) -> bytes:
    # [FUNCTION DOC] HELPER: Write the whole call payload to the child's stdin and always close stdin
    # afterwards. A broken pipe or connection reset while writing (child already dead) is ignored.
    # Then reads the child's stdout until end of stream and returns all bytes (possibly empty). The
    # stream ends when the child closes its outcome file, which happens before leftover cleanup.
    """Send the call and read the outcome, which arrives before the process finishes its leftover cleanup."""
    try:
        proc.stdin.write(call)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass  # the agent process died before reading its call; its exit code says why
    finally:
        proc.stdin.close()
    return await proc.stdout.read()


def _terminate(proc: asyncio.subprocess.Process) -> None:
    # [FUNCTION DOC] HELPER: Ask the child to stop: send SIGTERM now (the child then cancels the agent
    # and enforces its own deadlines) and schedule, on the running event loop, a SIGKILL after the
    # cancel grace plus leftover cleanup seconds in case the child cannot run Python code (for example
    # stuck holding the interpreter lock). The delayed kill only fires if the process has not exited.
    _send_signal(proc, signal.SIGTERM)  # the agent process cancels the agent and enforces both deadlines itself
    # a backstop for an agent process that cannot run Python to enforce them, such as one stuck holding the GIL
    asyncio.get_running_loop().call_later(_CANCEL_GRACE_S + _LEFTOVER_CLEANUP_S, _send_signal, proc, signal.SIGKILL)


def _send_signal(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    # [FUNCTION DOC] HELPER: Send the signal only while the process has no return code yet; a process
    # that vanished in between is ignored.
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(sig)


def _unpack_outcome(output: bytes) -> Any:
    # [FUNCTION DOC] HELPER: Deserialize the child's outcome. A success outcome returns its value. An
    # error outcome raises the child's exception object, chained from a remote-traceback exception whose
    # text is the child traceback wrapped between triple-quote lines. Deserializing runs untrusted-code
    # risk only with the own child process (same trust domain).
    outcome = pickle.loads(output)
    if outcome[0] == "ok":
        return outcome[1]
    _, error, child_traceback = outcome
    raise error from _RemoteTraceback(f'\n"""\n{child_traceback}"""')


def _child_main(parent_pid: int) -> None:
    # [FUNCTION DOC] HELPER: Entry point of the child process (called from the bootstrap one-liner with
    # the parent pid). Steps: arm parent-death kill; keep the ORIGINAL stdout as the binary outcome
    # channel (duplicate descriptor 1) and redirect descriptor 1 onto stderr so agent prints cannot
    # corrupt the outcome stream; configure logging with the child's pid; read the call tuple from
    # stdin (a failure to read/deserialize is reported as an error outcome and the process exits with
    # code 0); remember the threads existing before the call; run the agent on a fresh event loop
    # (see run-and-report helper); on any failure of the runner itself print the traceback and exit
    # with code 1; after the loop ends, wait for every non-daemon thread started by the agent (the
    # exit timer armed by the runner bounds this wait); exit with code 0 without interpreter
    # shutdown. Never returns.
    _die_with_parent(parent_pid)
    # The outcome goes to the original stdout; what the agent prints goes to stderr instead.
    outcome_file = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    configure_logger_raw(f"agent_function pid={os.getpid()}")
    try:
        fn, kwargs, cancel_grace_s, leftover_cleanup_s = pickle.load(sys.stdin.buffer)
    except Exception as error:
        _send_outcome(outcome_file, ("error", error, traceback.format_exc()))
        _exit(0)
    threads_before = set(threading.enumerate())
    try:
        asyncio.run(_run_and_report(fn, kwargs, outcome_file, cancel_grace_s, leftover_cleanup_s))
    except BaseException:
        traceback.print_exc()
        _exit(1)
    # the deadline armed once the call was over bounds this wait
    for thread in set(threading.enumerate()) - threads_before:
        if not thread.daemon:
            thread.join()
    _exit(0)


def _die_with_parent(parent_pid: int) -> None:
    # [FUNCTION DOC] HELPER: Linux only (no-op elsewhere). Asks the kernel (via the process control
    # system call) to deliver SIGKILL to this process when its parent thread exits; raises an OS error
    # with the errno if the call fails. Then guards the race where the parent already died before the
    # arming: if the current parent pid differs from the expected one, exits immediately with code 1.
    """Have the kernel kill this process when the rollout process dies, so no agent outlives its rollout."""
    if not sys.platform.startswith("linux"):
        return
    if ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent_pid:
        os._exit(1)  # the rollout process died before the signal was armed


async def _run_and_report(
    fn: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
    outcome_file: BinaryIO,
    cancel_grace_s: float,
    leftover_cleanup_s: float,
) -> None:
    # [FUNCTION DOC] HELPER: Run the agent and report its outcome (child side). Installs a SIGTERM
    # handler that, on the first signal, cancels the current task (so the agent can run its cleanup)
    # and arms a daemon timer that hard-exits the process (code 1) after the cancel grace seconds.
    # Awaits the agent function with the keyword arguments. Outcome: success value; cancellation gives
    # no outcome (the parent was cancelled and reads nothing); an ordinary exception gives an error
    # outcome with the formatted traceback; a base-level exception (exit, interrupt) is wrapped as a
    # runtime error naming it, so it ends only this episode. Afterwards cancels the grace timers,
    # removes the loop handler and makes SIGTERM ignored, writes the outcome and closes the outcome
    # file (the parent stops waiting), and arms a daemon timer that hard-exits (code 1) after the
    # leftover cleanup seconds, bounding any remaining executor threads.
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    cancel_deadlines: list[threading.Timer] = []

    def cancel_on_sigterm() -> None:
        # [FUNCTION DOC] HELPER: Signal callback: only on the first SIGTERM, cancel the agent task and
        # start the grace deadline timer; later signals are ignored.
        # the caller sends SIGTERM when it is cancelled: cancelling the agent runs its cleanup
        if not cancel_deadlines:
            task.cancel()
            cancel_deadlines.append(_exit_after(cancel_grace_s))

    loop.add_signal_handler(signal.SIGTERM, cancel_on_sigterm)
    try:
        outcome = ("ok", await fn(**kwargs))
    except asyncio.CancelledError:
        outcome = None  # the caller was cancelled and reads no outcome
    except Exception as error:
        outcome = ("error", error, traceback.format_exc())
    except BaseException as error:
        # an agent that exits or is interrupted ends only its own episode, not the rollout
        outcome = ("error", RuntimeError(f"The agent function raised {error!r}"), traceback.format_exc())
    for deadline in cancel_deadlines:
        deadline.cancel()
    # once the call is over, its leftover cleanup runs to its own deadline whatever the caller does
    loop.remove_signal_handler(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # sent before the loop shuts down, which waits for the agent's leftover executor threads
    _send_outcome(outcome_file, outcome)
    _exit_after(leftover_cleanup_s)


def _exit_after(seconds: float) -> threading.Timer:
    # [FUNCTION DOC] HELPER: Start and return a daemon timer thread that, after the given seconds,
    # terminates the process with exit code 1 (flushing output first). Cancelable by the caller.
    timer = threading.Timer(seconds, _exit, args=(1,))
    timer.daemon = True
    timer.start()
    return timer


def _send_outcome(outcome_file: BinaryIO, outcome: tuple | None) -> None:
    # [FUNCTION DOC] HELPER: Write the serialized outcome (skipped when none, i.e. cancelled) to the
    # outcome channel, then close it, which signals the parent that the result is complete.
    if outcome is not None:
        outcome_file.write(_dump_outcome(outcome))
    outcome_file.close()  # the caller stops waiting here


def _exit(code: int) -> NoReturn:
    # [FUNCTION DOC] HELPER: Flush stdout and stderr and terminate immediately with the code, without
    # interpreter shutdown (which would wait indefinitely for running threads).
    sys.stdout.flush()
    sys.stderr.flush()
    # skips interpreter shutdown, which would wait without limit for a thread still running
    os._exit(code)


def _dump_outcome(outcome: tuple) -> bytes:
    # [FUNCTION DOC] HELPER: Serialize an outcome so the parent can always read it. A success outcome
    # that cannot be serialized is replaced by an error outcome (type error describing the failure)
    # with the current traceback. An error outcome is serialized and then trial-deserialized; if it
    # fails either way (some exceptions serialize but cannot be rebuilt), it is replaced by a runtime
    # error whose message is the qualified error class name and text, keeping the original traceback
    # text. Returns bytes.
    """Pickle the outcome, replacing what the caller could not rebuild."""
    if outcome[0] == "ok":
        try:
            return pickle.dumps(outcome)
        except Exception as e:
            error = TypeError(f"The agent function returned a value that cannot be pickled: {e}")
            outcome = ("error", error, traceback.format_exc())
    _, error, child_traceback = outcome
    try:
        data = pickle.dumps(outcome)
        # some errors pickle but cannot be rebuilt, such as one whose constructor takes keyword-only arguments
        pickle.loads(data)
        return data
    except Exception:
        return pickle.dumps(("error", RuntimeError(f"{type(error).__qualname__}: {error}"), child_traceback))
