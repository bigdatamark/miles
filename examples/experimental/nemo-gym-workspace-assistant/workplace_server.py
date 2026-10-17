# [MODULE DOC] FastAPI server exposing NeMo Gym's Workplace Assistant tools and verifier over HTTP,
# one isolated episode (session) per rollout. Endpoints: GET /health (counters), POST
# /sessions/{task_id} (create an episode, returns a session id), POST /sessions/{session_id}/tool
# (execute one tool call), POST /sessions/{session_id}/verify (grade and release), DELETE
# /sessions/{session_id} (release). Task data (including gold actions) is loaded from a JSON-lines
# dataset file and never sent to the policy. Sessions live in memory in a single worker process and
# expire after a time-to-live since their last tool call. Run as a script with a dataset path,
# host (default 0.0.0.0) and port (default 8211).
"""Thin HTTP transport for NeMo Gym's unmodified Workplace tools and verifier.

Gold actions stay in the server's task catalog; policy requests contain task IDs.
Run with a pinned Gym checkout on PYTHONPATH and this example's requirements.
"""

import json
import logging
import math
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from resources_servers.workplace_assistant.utils import execute_actions_and_reset_state, get_tools, is_correct
from tap import Tap

logger = logging.getLogger(__name__)
DOMAINS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]
TABLES = {
    "calendar": "_calendar_events",
    "email": "_emails",
    "analytics": "_plots_data",
    "project_management": "_project_tasks",
    "customer_relationship_manager": "_crm_data",
}


def json_safe(value: Any) -> Any:
    # [FUNCTION DOC] HELPER: Recursively makes a value JSON-serializable: non-finite floats (NaN or
    # infinities) become null; dicts and lists/tuples are rebuilt with converted items (tuples
    # become lists); anything else is returned unchanged. Pure; returns a new structure.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def tables(env: Mapping[str, Any]) -> dict:
    # [FUNCTION DOC] HELPER: Snapshots the state of a Workplace environment for comparison. For each
    # domain it takes the backing data table of that domain's container (attribute names given by
    # the table mapping), converts it from a pandas frame to a polars frame with NaN turned into
    # null, and then to a list of row dicts. Returns a mapping from domain name to rows.
    return {
        domain: pl.from_pandas(getattr(env["containers"][domain], attr), nan_to_null=True).to_dicts()
        for domain, attr in TABLES.items()
    }


class ToolCall(BaseModel):
    # [CLASS DOC] Request body of one tool call: the tool name and its arguments as a JSON-encoded
    # string (the shape of a chat-completions function call). Unknown fields are rejected.
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: str


@dataclass
class Episode:
    # [CLASS DOC] State of one live episode. Fields: task, the dataset row (with tool declarations
    # and ground truth); env, a freshly built tool environment for all five domains; actions, the
    # ordered list of tool calls issued so far (including calls whose execution failed); updated,
    # a monotonic timestamp of the last tool activity used for expiry; lock, a mutex serializing
    # operations on this episode.
    task: dict
    env: dict = field(default_factory=lambda: get_tools(DOMAINS))
    actions: list = field(default_factory=list)
    updated: float = field(default_factory=time.monotonic)
    lock: Any = field(default_factory=threading.Lock)


class Workplace:
    # [CLASS DOC] In-memory episode manager and grader. Holds the task catalog keyed by task id, the
    # active sessions keyed by random hex session id, a lock protecting the session map and the
    # counters, the session time-to-live in seconds, and counters of graded and solved episodes.
    def __init__(self, dataset: Path, ttl: float = 10800) -> None:
        # [FUNCTION DOC] Loads the dataset file (JSON lines; blank lines skipped) into the task
        # catalog keyed by each row's id field. Raises a value error if the file is empty or ids
        # are not unique. Initializes empty sessions, counters at zero, and ttl (default three hours).
        # Reads the file from disk.
        rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
        self.tasks = {row["id"]: row for row in rows}
        if len(self.tasks) != len(rows) or not rows:
            raise ValueError("Task IDs must be unique and dataset nonempty")
        self.sessions: dict[str, Episode] = {}
        self.lock = threading.Lock()
        self.ttl = ttl
        self.graded = 0
        self.solved = 0

    def create(self, task_id: int) -> str:
        # [FUNCTION DOC] Starts an episode for a task id and returns a new random hex session id.
        # Raises an HTTP 404 error for an unknown task id. Builds a fresh episode (new tool
        # environment) and, under the manager lock, first drops every session not updated within the
        # ttl, then registers the new one.
        if task_id not in self.tasks:
            raise HTTPException(404, "Unknown task ID")
        episode = Episode(task=self.tasks[task_id])
        session_id = uuid.uuid4().hex
        with self.lock:
            cutoff = time.monotonic() - self.ttl
            self.sessions = {k: v for k, v in self.sessions.items() if v.updated > cutoff}
            self.sessions[session_id] = episode
        return session_id

    def get(self, session_id: str) -> Episode:
        # [FUNCTION DOC] HELPER: Looks up a live episode by session id under the manager lock.
        # Raises an HTTP 404 error ("unknown or expired") when absent. Does not refresh expiry.
        with self.lock:
            episode = self.sessions.get(session_id)
        if episode is None:
            raise HTTPException(404, "Unknown or expired episode")
        return episode

    def delete(self, session_id: str) -> None:
        # [FUNCTION DOC] Removes a session if present (idempotent, no error when missing), under
        # the manager lock.
        with self.lock:
            self.sessions.pop(session_id, None)

    def tool(self, session_id: str, call: ToolCall) -> dict:
        # [FUNCTION DOC] Executes one tool call in the session and returns a dict with a single
        # "output" entry. Raises HTTP 404 for an unknown session. Under the episode lock: refreshes
        # the expiry timestamp; if the tool name is not among the tools declared by the task,
        # returns an error-text output WITHOUT recording the call. Otherwise records the call in the
        # action list first (so failed calls are still replayed by the verifier), parses the
        # arguments string as JSON (must be an object, else treated as failure), drops arguments
        # whose value is null (matching the native resource server), invokes the environment
        # function by name with the remaining arguments as keywords, and returns the result made
        # JSON-safe. Any exception is logged with traceback and turned into a generic error-text
        # output instead of failing the request.
        episode = self.get(session_id)
        with episode.lock:
            episode.updated = time.monotonic()
            allowed = {tool["name"] for tool in episode.task["responses_create_params"]["tools"]}
            if call.name not in allowed:
                return {"output": "Error executing tool: tool is not declared for this task"}
            episode.actions.append(call.model_dump())
            try:
                args = json.loads(call.arguments)
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                # Exactly the native resource-server rule: omit JSON null kwargs.
                args = {key: value for key, value in args.items() if value is not None}
                result = episode.env["functions"][call.name](**args)
                return {"output": json_safe(result)}
            except Exception:
                logger.exception("Workplace tool execution failed: %s", call.name)
                return {"output": "Error executing tool: invalid arguments or operation failed"}

    def verify(self, session_id: str) -> dict:
        # [FUNCTION DOC] Grades an episode and always releases the session afterwards (even on
        # error). Raises HTTP 404 for an unknown session. Under the episode lock: replays the
        # recorded actions on a fresh environment with the native replay helper and compares the
        # resulting tables with the live environment's tables; a difference raises a runtime error
        # (an infrastructure fault, not a zero reward). Then computes reward as a float from the
        # native correctness check of the recorded actions against the task's ground truth. Under the
        # manager lock it increments the graded counter and adds the reward (as an integer) to the
        # solved counter. Returns a dict with reward, a valid flag, a state-replay-consistent flag,
        # the task id and the number of recorded actions.
        episode = self.get(session_id)
        try:
            with episode.lock:
                # Also check the live tool state against native replay. A mismatch is
                # an infrastructure error, never a zero-reward policy sample.
                replay = execute_actions_and_reset_state(episode.actions)
                if tables(replay) != tables(episode.env):
                    raise RuntimeError("Live tools and native verifier replay disagree")
                reward = float(is_correct(episode.actions, episode.task["ground_truth"], None))
            with self.lock:
                self.graded += 1
                self.solved += int(reward)
            return {
                "reward": reward,
                "valid": True,
                "state_replay_consistent": True,
                "task_id": episode.task["id"],
                "tool_calls": len(episode.actions),
            }
        finally:
            self.delete(session_id)


def create_app(dataset: Path) -> FastAPI:
    # [FUNCTION DOC] Builds the web application around a new episode manager loaded from the
    # dataset path (also stored on the app state under the name workplace) and registers the
    # routes defined by the nested handlers below. Returns the app. Reads the dataset file.
    service = Workplace(dataset)
    app = FastAPI()
    app.state.workplace = service

    @app.get("/health")
    def health() -> dict:
        # [FUNCTION DOC] GET /health. Returns counts: number of tasks, active sessions, graded
        # episodes and solved episodes, read under the manager lock.
        with service.lock:
            return {
                "tasks": len(service.tasks),
                "active": len(service.sessions),
                "graded": service.graded,
                "solved": service.solved,
            }

    @app.post("/sessions/{task_id}")
    def create(task_id: int) -> dict:
        # [FUNCTION DOC] POST /sessions/{task_id}. Creates an episode and returns its session id;
        # 404 for an unknown task. (Shares a path shape with the other session routes but only
        # this one takes an integer task id.)
        return {"session_id": service.create(task_id)}

    @app.post("/sessions/{session_id}/tool")
    def tool(session_id: str, body: ToolCall) -> dict:
        # [FUNCTION DOC] POST /sessions/{session_id}/tool. Executes the tool call in the body in
        # the session and returns the output dict of the manager's tool method.
        return service.tool(session_id, body)

    @app.post("/sessions/{session_id}/verify")
    def verify(session_id: str) -> dict:
        # [FUNCTION DOC] POST /sessions/{session_id}/verify. Grades and releases the session; returns
        # the manager's verification result.
        return service.verify(session_id)

    @app.delete("/sessions/{session_id}")
    def delete(session_id: str) -> dict:
        # [FUNCTION DOC] DELETE /sessions/{session_id}. Releases the session if it exists and
        # always returns a deleted-true flag.
        service.delete(session_id)
        return {"deleted": True}

    return app


class Args(Tap):
    # [CLASS DOC] Command-line arguments: dataset (required path to the JSON-lines task file),
    # host (default 0.0.0.0, all interfaces) and port (default 8211).
    dataset: Path
    host: str = "0.0.0.0"
    port: int = 8211


def main() -> None:
    # [FUNCTION DOC] Script entry point: parses the command-line arguments, builds the app and
    # serves it with uvicorn on the given host and port using exactly one worker process (required
    # because sessions are held in process memory). Blocks until the server stops.
    args = Args().parse_args()
    uvicorn.run(create_app(args.dataset), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
