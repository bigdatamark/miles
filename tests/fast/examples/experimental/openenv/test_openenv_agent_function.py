# [MODULE DOC]
# Offline unit tests for the shared-server leg of the openenv agent function module (imported by
# bare name): its per-episode loop function takes a policy client, model name, initial messages,
# request kwargs and a task description, and returns a pair of reward and a metrics mapping (turns,
# tool calls, end reason). Fakes defined here (and reused by the sandbox test module): an
# environment recording every step action and answering evaluate with reward 1.0 plus a harness
# marker, an action record, observation and result records, completion builders, and policies
# (emit a bash command then a completion token; whitespace only; length-truncated). The tests
# replace the module's environment loader and environment context helper so the loop runs against
# the fake. Covered: command pass-through, trial-directory purge, scoring through evaluate, length
# cut-off ending the episode, no-command and maximum-turns end reasons, rejecting a reward from a
# server lacking the harness marker, and truncated turns not being executed.
"""Offline unit tests for the openenv tbench2 adapter (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers the shared-server leg of the agent loop (this module's run_episode):
its exec form, scoring path, and cleanup. The sandbox leg's dispatch and
sandbox-create machinery live in test_openenv_sandbox_common.py; the fakes
below are shared with it.
"""

import asyncio
import types

import openenv_agent_function as oaf


def run_async(coro):
    # [FUNCTION DOC] HELPER: Runs the coroutine to completion in a fresh event loop and returns its result.
    return asyncio.run(coro)


# --- fakes ---------------------------------------------------------------


class _FakeObs:
    # [CLASS DOC] HELPER: Fake observation: a plain attribute bag whose attributes are the given keyword arguments.
    def __init__(self, **kw):
        # [FUNCTION DOC] HELPER: Copies the keyword arguments into the instance attributes.
        self.__dict__.update(kw)


class _FakeResult:
    # [CLASS DOC] HELPER: Fake step result holding an observation (output, instruction, info
    # mapping defaulting to empty) and, only when a reward is given, a reward attribute.
    def __init__(self, output="", reward=None, instruction="", info=None):
        # [FUNCTION DOC] HELPER: Builds the observation; sets the reward attribute only if the reward is not none.
        self.observation = _FakeObs(output=output, instruction=instruction, info=info or {})
        if reward is not None:
            self.reward = reward


class _FakeEnv:
    # [CLASS DOC] HELPER: Fake async environment client usable as an async context manager. Every
    # step action is appended to its own list, and the latest instance's list is also published in
    # a class-level attribute so tests can inspect it. Reset returns an instruction; step answers
    # an evaluate action with reward 1.0 and info marking tests passed with a harness path, and any
    # other action with output ok.
    """Records every step() action; answers `evaluate` like a contract-carrying
    server (reward plus the canonical-harness marker)."""

    last_actions: list = []

    def __init__(self, base_url="", message_timeout_s=0):
        # [FUNCTION DOC] HELPER: Creates the action list and publishes it as the class-level last actions (arguments ignored).
        self.actions = []
        _FakeEnv.last_actions = self.actions

    async def __aenter__(self):
        # [FUNCTION DOC] HELPER: Returns the instance itself.
        return self

    async def __aexit__(self, *a):
        # [FUNCTION DOC] HELPER: Returns false so exceptions propagate.
        return False

    async def reset(self, task_id=None):
        # [FUNCTION DOC] HELPER: Returns a result whose instruction is a fixed string (task id ignored).
        return _FakeResult(instruction="do the thing")

    async def step(self, action):
        # [FUNCTION DOC] HELPER: Records the action. For an evaluate action returns reward 1.0 with
        # info tests-passed true and a harness script path; otherwise returns output ok.
        self.actions.append(action)
        if action.action_type == "evaluate":
            return _FakeResult(reward=1.0, info={"tests_passed": True, "harness": "tests/test.sh"})
        return _FakeResult(output="ok")


class _FakeAction:
    # [CLASS DOC] HELPER: Fake action record with an action type and an optional command.
    def __init__(self, action_type, command=None):
        # [FUNCTION DOC] HELPER: Stores the action type and command.
        self.action_type = action_type
        self.command = command


def _fake_completion(text, finish_reason="stop"):
    # [FUNCTION DOC] HELPER: Builds a fake chat completion with one choice whose message has the
    # given content and a dump method returning an assistant role message (extra none-filter
    # flag accepted), with the given finish reason (default stop).
    msg = types.SimpleNamespace(
        content=text, model_dump=lambda exclude_none=True: {"role": "assistant", "content": text}
    )
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish_reason)])


class _FakePolicy:
    # [CLASS DOC] HELPER: Fake policy client exposing a chat completions create method. It counts
    # calls; the first call returns a message with a bash code block that echoes hi and the later
    # calls return the task-complete token.
    """Turn 1: emit a bash command. Turn 2: TASK_COMPLETE."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts the call counter at zero and wires the chat completions create method.
        self.n = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        # [FUNCTION DOC] HELPER: Increments the counter and returns the command completion on the first call, the completion token afterwards.
        self.n += 1
        text = "```bash\necho hi\n```" if self.n == 1 else "TASK_COMPLETE"
        return _fake_completion(text)


class _NoCommandPolicy(_FakePolicy):
    # [CLASS DOC] HELPER: Policy variant that always returns a whitespace-only message.
    """Emits only whitespace, so there is no command to execute."""

    async def _create(self, **kw):
        # [FUNCTION DOC] HELPER: Increments the counter and returns a completion with whitespace content.
        self.n += 1
        return _fake_completion(" \n")


_CLASSES = {"env": _FakeEnv, "action": _FakeAction}


# --- episode dispatch ------------------------------------------------------


def test_shared_leg_dispatch(monkeypatch):
    # [FUNCTION DOC] Replaces the module's class loader with the fake classes and its environment
    # context helper with one that calls the body on a fresh fake environment. Runs one episode
    # with the standard policy. Asserts reward 1.0; the first exec command is exactly echo hi; some
    # exec command mentions the trial runs directory (purge); an evaluate action was sent; and the
    # metrics show 2 turns, end reason task complete, 1 tool call.
    """The shared-server run_episode: exec commands pass through unmodified
    (the server resolves the workdir), scoring via the standard `evaluate`
    action — and the trial-dir purge (post_episode) runs, since the shared
    server outlives the episode."""
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        # [FUNCTION DOC] HELPER: Stand-in for the environment context helper: instantiates the
        # environment class with no arguments and awaits the body on it, returning its result.
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    reward, metrics = run_async(
        oaf.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]

    assert reward == 1.0
    assert execs[0].command == "echo hi"
    assert any("/tmp/tbench2_env_runs" in (a.command or "") for a in execs), "trial-dir purge missing"
    assert any(a.action_type == "evaluate" for a in actions)
    assert metrics["turns"] == 2
    assert metrics["end_reason"] == "task_complete"
    assert metrics["tool_calls"] == 1


class _TruncatingPolicy(_FakePolicy):
    # [CLASS DOC] HELPER: Policy variant whose completions are marked as cut off by the length limit.
    """Emits a command the model never finished writing."""

    async def _create(self, **kw):
        # [FUNCTION DOC] HELPER: Gets the parent completion and sets its first choice's finish reason to length.
        completion = await super()._create(**kw)
        completion.choices[0].finish_reason = "length"
        return completion


def test_length_capped_turn_ends_the_episode(monkeypatch):
    # [FUNCTION DOC] Runs an episode with the truncating policy (same patching as the dispatch
    # test). Asserts 1 turn, end reason length, 0 tool calls, and that no exec action ran the
    # echo hi command.
    """A turn cut off by the token cap must not be executed: the command is
    truncated, so running it would send an arbitrary prefix to the sandbox."""
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        # [FUNCTION DOC] HELPER: Instantiates the environment class and awaits the body on it.
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    _, metrics = run_async(
        oaf.run_episode(_TruncatingPolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )

    assert metrics["turns"] == 1
    assert metrics["end_reason"] == "length"
    assert metrics["tool_calls"] == 0
    execs = [a for a in _FakeEnv.last_actions if a.action_type == "exec"]
    assert not [a for a in execs if "echo hi" in (a.command or "")], "ran a command the model never finished"


def test_no_command_reports_end_reason(monkeypatch):
    # [FUNCTION DOC] Runs an episode with the whitespace-only policy; asserts 1 turn, end reason
    # no command, 0 tool calls.
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        # [FUNCTION DOC] HELPER: Instantiates the environment class and awaits the body on it.
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    _, metrics = run_async(
        oaf.run_episode(_NoCommandPolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )

    assert metrics["turns"] == 1
    assert metrics["end_reason"] == "no_command"
    assert metrics["tool_calls"] == 0


def test_max_turns_reports_end_reason(monkeypatch):
    # [FUNCTION DOC] Sets the maximum turns environment variable to 1 and runs an episode with the
    # standard policy; asserts the policy was called once, 1 turn, end reason max turns, 1 tool call.
    monkeypatch.setenv("OPENENV_MAX_TURNS", "1")
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        # [FUNCTION DOC] HELPER: Instantiates the environment class and awaits the body on it.
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    policy = _FakePolicy()
    _, metrics = run_async(oaf.run_episode(policy, "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"}))

    assert policy.n == 1
    assert metrics["turns"] == 1
    assert metrics["end_reason"] == "max_turns"
    assert metrics["tool_calls"] == 1


def test_old_server_reward_is_not_trusted(monkeypatch):
    # [FUNCTION DOC] Uses an environment subclass whose evaluate step reports reward 1.0 with info
    # lacking the harness marker (only tests passed and exit code). Runs an episode; asserts the
    # returned reward is none while the metrics still show 2 turns (the episode ran, only scoring
    # was rejected).
    """A server without the canonical contract (e.g. an out-of-date install)
    answers `evaluate` with a plausible-looking reward but no harness marker
    (its info is {tests_passed, exit_code} from bare pytest). That reward must
    be dropped, not ingested: source preflight is impossible against a remote
    server."""

    class _OldServerEnv(_FakeEnv):
        # [CLASS DOC] HELPER: Fake environment emulating an old server whose evaluate answer has no harness marker.
        async def step(self, action):
            # [FUNCTION DOC] HELPER: For an evaluate action, records it and returns reward 1.0 with
            # info tests-passed true and exit code 0 (no harness marker); otherwise defers to the parent.
            if action.action_type == "evaluate":
                self.actions.append(action)
                return _FakeResult(reward=1.0, info={"tests_passed": True, "exit_code": 0})
            return await super().step(action)

    monkeypatch.setattr(oaf, "load_tbench2", lambda: {"env": _OldServerEnv, "action": _CLASSES["action"]})

    async def spying_with_env(env_cls, env_url, body):
        # [FUNCTION DOC] HELPER: Instantiates the environment class and awaits the body on it.
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    reward, metrics = run_async(
        oaf.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    assert reward is None
    assert metrics["turns"] == 2  # the episode itself completed; only scoring was rejected


class _TruncatedPolicy:
    # [CLASS DOC] HELPER: Fake policy client (not derived from the standard fake) whose every
    # completion is a bash command cut off mid-line and marked as length-capped.
    """Every turn returns a command cut off by the per-turn cap."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts the call counter at zero and wires the chat completions create method.
        self.n = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        # [FUNCTION DOC] HELPER: Increments the counter and returns a completion with a truncated
        # bash command and finish reason length.
        self.n += 1
        text = "```bash\nmake -j && ./run_all_the"
        msg = types.SimpleNamespace(
            content=text, model_dump=lambda exclude_none=True: {"role": "assistant", "content": text}
        )
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="length")])


def test_truncated_turn_ends_the_episode(monkeypatch):
    # [FUNCTION DOC] Runs an episode with a policy whose every turn is truncated. Asserts the
    # policy was called once, every exec action is the trial-directory purge (the cut-off command
    # never ran), an evaluate action was still sent, reward 1.0, 1 turn and end reason length.
    """A finish_reason="length" turn closes the trainable sample — collection
    keeps nothing past it, so the loop stops there. The cut-off command is not
    executed; scoring still runs."""
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        # [FUNCTION DOC] HELPER: Instantiates the environment class and awaits the body on it.
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    policy = _TruncatedPolicy()
    reward, metrics = run_async(
        oaf.run_episode(policy, "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    assert policy.n == 1, "the loop must stop at the truncated turn"
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]
    assert all("/tmp/tbench2_env_runs" in (a.command or "") for a in execs), execs
    assert any(a.action_type == "evaluate" for a in actions), "scoring still runs"
    assert reward == 1.0 and metrics["turns"] == 1
    assert metrics["end_reason"] == "length"
