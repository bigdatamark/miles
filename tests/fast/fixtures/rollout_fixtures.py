# [MODULE DOC]
# Fixtures for testing the rollout function end to end on cpu: a mock sglang server, a real miles
# router in a background thread that registers the mock as its worker, an optional session server (for
# agentic custom-agent runs), and a buffered rollout data source over a temporary jsonl prompt file.
# Provides a frozen config dataclass (extra command line tokens, dataset rows, mock latency), a frozen
# environment dataclass, and the parametrized pytest fixture that yields the environment. Constant:
# default dataset rows, one arithmetic question with label 8.
"""
Fixtures to test rollout-function
"""

import json
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from miles.rollout.data_source import DataSource, RolloutDataSourceWithBuffer
from miles.rollout.session.config import compute_session_server_config
from miles.rollout.session.server import SessionServer
from miles.rollout.session.types import SessionServerInstance
from miles.router.config import compute_miles_router_config
from miles.router.router import MilesRouter
from miles.utils.arguments import parse_args
from miles.utils.http_utils import find_available_port, init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@dataclass(frozen=True)
class RolloutEnvConfig:
    # [CLASS DOC] HELPER: Immutable parameters for the rollout environment fixture (passed as the
    # indirect parameter): extra command line tokens appended to the base ones (default none), dataset
    # rows as mappings (none or empty means the module default rows), and mock server latency in
    # seconds (default 0).
    extra_argv: list[str] | None = None
    data_rows: list[dict] | None = None
    latency: float = 0.0


@dataclass(frozen=True)
class RolloutEnv:
    # [CLASS DOC] HELPER: Immutable environment yielded by the fixture: parsed args, the buffered data
    # source, and the mock sglang server.
    args: Namespace
    data_source: DataSource
    mock_server: MockSGLangServer


def _build_args(*, data_path: str, router_port: int, extra_argv: list[str] | None = None) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds the parsed arguments. Base command line: fsdp backend, ci test mode,
    # one rollout batch, one sample per prompt, one rollout, one gpu and one gpu per engine, the Qwen3
    # 0.6B checkpoint, the data path as prompt data with input and label keys, math reward type, the same
    # path as an eval dataset named toy, the miles router enabled, router ip 127.0.0.1 and the given
    # port, response length 16; extra tokens are appended last. Parses with the real parser under a
    # patched process argv, then initializes the global http client (side effect). Returns the args.
    argv = [
        "pytest",
        "--train-backend",
        "fsdp",
        "--ci-test",
        "--rollout-batch-size",
        "1",
        "--n-samples-per-prompt",
        "1",
        "--num-rollout",
        "1",
        "--rollout-num-gpus",
        "1",
        "--rollout-num-gpus-per-engine",
        "1",
        "--hf-checkpoint",
        "Qwen/Qwen3-0.6B",
        "--prompt-data",
        data_path,
        "--input-key",
        "input",
        "--label-key",
        "label",
        "--rm-type",
        "math",
        "--eval-prompt-data",
        "toy",
        data_path,
        "--use-miles-router",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        str(router_port),
        "--rollout-max-response-len",
        "16",
    ] + (extra_argv or [])
    with patch("sys.argv", argv):
        args = parse_args()
    init_http_client(args)
    return args


@contextmanager
def _with_miles_router(args: Namespace) -> Iterator[UvicornThreadServer]:
    # [FUNCTION DOC] HELPER: Context manager. Computes the router config from the args' router ip and
    # port with one engine, builds a quiet router, serves its web app in a background thread, yields the
    # server, and stops the thread on exit even after an error.
    config = compute_miles_router_config(args, host=args.sglang_router_ip, port=args.sglang_router_port, num_engines=1)
    router = MilesRouter(config, verbose=False)
    server = UvicornThreadServer(router.app, host=config.host, port=config.port)
    try:
        server.start()
        yield server
    finally:
        server.stop()


def _write_jsonl(path: str, rows: list[dict]) -> None:
    # [FUNCTION DOC] HELPER: Writes one json object per line (non-ascii characters kept as is) to the
    # path as utf-8 text, overwriting any existing file.
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


DEFAULT_DATA_ROWS = [{"input": "What is 1+7?", "label": "8"}]


@contextmanager
def _with_session_server(args: Namespace, backend_url: str) -> Iterator[UvicornThreadServer]:
    # [FUNCTION DOC] HELPER: Context manager. Picks a free port from 31000, computes the session server
    # config (localhost, no instance id, the backend url), serves the session app in a background
    # thread, then publishes a single session server instance record (address only) on the args (side
    # effect) and yields the server; stops the thread on exit even after an error.
    """Start a SessionServer for agentic variants that need TITO session tracking."""
    port = find_available_port(31000)
    config = compute_session_server_config(
        args,
        host="127.0.0.1",
        port=port,
        instance_id=None,
        backend_url=backend_url,
    )
    session_server = SessionServer(config)
    server = UvicornThreadServer(session_server.app, host="127.0.0.1", port=port)
    try:
        server.start()
        args.session_server_instances = [SessionServerInstance(addr=f"127.0.0.1:{port}")]
        yield server
    finally:
        server.stop()


def _needs_session_server(extra_argv: list[str] | None) -> bool:
    # [FUNCTION DOC] HELPER: Returns true when extra tokens exist and contain the custom agent function
    # path flag.
    return extra_argv is not None and "--custom-agent-function-path" in extra_argv


@pytest.fixture
def rollout_env(tmp_path, request) -> RolloutEnv:
    # [FUNCTION DOC] Pytest fixture (generator) yielding a rollout environment. The indirect parameter
    # must be a rollout config (else an assertion error). Steps: write the rows (default rows if none
    # given) to a jsonl file in the temp dir; clear all singleton instances; start the mock server with
    # the configured latency; pick a free router port from 20000; build the args; start the miles router
    # and register the mock server with it through an add-worker http post (5 second timeout, http errors
    # raised); if extra tokens request a custom agent function, also run a session server in front of the
    # router; build the buffered data source and yield the environment. Teardown stops servers and clears
    # singletons again.
    config = request.param
    assert isinstance(config, RolloutEnvConfig)

    data_rows = config.data_rows or DEFAULT_DATA_ROWS

    data_path = str(tmp_path / "data.jsonl")
    _write_jsonl(data_path, data_rows)

    SingletonMeta.clear_all_instances()

    with with_mock_server(latency=config.latency) as mock_server:
        router_port = find_available_port(20000)
        args = _build_args(data_path=data_path, router_port=router_port, extra_argv=config.extra_argv)
        with _with_miles_router(args) as router_server:
            r = requests.post(
                f"{router_server.url}/add_worker",
                params={"url": mock_server.url},
                timeout=5.0,
            )
            r.raise_for_status()

            if _needs_session_server(config.extra_argv):
                with _with_session_server(args, router_server.url):
                    data_source = RolloutDataSourceWithBuffer(args)
                    yield RolloutEnv(args=args, data_source=data_source, mock_server=mock_server)
            else:
                data_source = RolloutDataSourceWithBuffer(args)
                yield RolloutEnv(args=args, data_source=data_source, mock_server=mock_server)

    SingletonMeta.clear_all_instances()
