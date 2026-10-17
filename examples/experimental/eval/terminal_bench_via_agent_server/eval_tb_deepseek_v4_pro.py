# [MODULE DOC]
# Standalone asynchronous client script that evaluates an LLM on the Terminal-Bench 2.0 task set
# through an already running agent server. Reads the list of task names for a named dataset from a
# registry JSON file, then sends one run request per (task, trial) pair with bounded concurrency
# (a semaphore), each carrying the LLM endpoint, model, API key, agent name, sampling parameters
# and sequence length cap. The API key is read from the environment variable named by an option
# (default for the DeepSeek key) and never stored. Each finished trial is appended as one JSON line
# to a results file (truncated at start) and printed as a progress line; at the end all rows
# (without the nested agent metrics and eval report) are written as a parquet table and a summary of
# completed and passing (reward exactly 1.0) trials is printed. No GPU, training or Ray is used.
"""Evaluate DeepSeek-V4-Pro on Terminal-Bench 2.0 via a running ``miles_agent_server``.

This is a pure client: no training, no Ray, no Megatron. It just POSTs
``/run`` requests concurrently to an already-running agent server and
aggregates the per-trial results.

The agent server is expected to be the one shipped in harbor's
``harbor-miles-v0.20.0`` branch (see that repo's README for setup), running
with ``$HARBOR_TASKS_DIR`` populated with the 89 Terminal-Bench 2.0 tasks.
The server is dataset-agnostic; this script just resolves the 89 task names
from ``registry.json`` and dispatches one ``/run`` request per
(task, trial) pair.

Usage::

    export DEEPSEEK_API_KEY=<your-key>
    python eval_tb_deepseek_v4_pro.py \\
        --server-url http://localhost:8080 \\
        --registry-path /path/to/harbor/registry.json \\
        --n-trials-per-task 4 \\
        --max-concurrent 16

Outputs (one row per (task, trial)):

- ``--output-jsonl`` streamed as trials complete
- ``--output-parquet`` written once at the end (via polars)
"""

import argparse
import asyncio
import json
import os
from collections.abc import Awaitable
from pathlib import Path
from time import monotonic
from typing import Any

import httpx
import polars as pl


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC] HELPER: Defines and parses the command line. Options and defaults: server URL
    # (localhost port 8080), registry path (registry.json), dataset name (terminal-bench), trials
    # per task (4), max concurrent requests (16), streamed JSONL path, parquet path, model
    # identifier (openai/deepseek-v4-pro), LLM base URL (DeepSeek API v1), name of the API key
    # environment variable (DEEPSEEK_API_KEY), agent name (mini-swe-agent), max sequence length
    # (65536), temperature (0.7), max output tokens (8192) and per-request client timeout seconds
    # (3600). Help output shows defaults. Returns the namespace.
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-V4-Pro on Terminal-Bench 2.0 via miles_agent_server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8080",
        help="Base URL of a running miles_agent_server.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("registry.json"),
        help="Path to harbor's registry.json (used to read the 89 TB2 task names).",
    )
    parser.add_argument(
        "--dataset-name",
        default="terminal-bench",
        help="Registry dataset name. 'terminal-bench' is TB 2.0 (89 tasks).",
    )
    parser.add_argument(
        "--n-trials-per-task",
        type=int,
        default=4,
        help="Replicates per task.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="Cap on in-flight /run requests.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("tb_deepseek_v4_pro_results.jsonl"),
        help="Streamed per-trial results.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("tb_deepseek_v4_pro_results.parquet"),
        help="Aggregated results, written after all trials complete.",
    )
    parser.add_argument(
        "--model",
        default="openai/deepseek-v4-pro",
        help="LiteLLM-style model identifier sent in the /run payload.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com/v1",
        help="LLM endpoint base URL sent in the /run payload (OpenAI-compatible).",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help=(
            "Name of the environment variable holding the LLM API key. "
            "The key is read only from os.environ; never hard-code it."
        ),
    )
    parser.add_argument(
        "--agent-name",
        default="mini-swe-agent",
        help=(
            "Agent to dispatch. Any installed Harbor agent works "
            "(mini-swe-agent, terminus-2, claude-code, codex, ...)."
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=65536,
        help="Max sequence length enforced by the agent server's poll_steps wrapper.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="LLM max output tokens per response.",
    )
    parser.add_argument(
        "--per-request-timeout-sec",
        type=int,
        default=3600,
        help="Client-side cap per /run request, seconds.",
    )
    return parser.parse_args()


def load_task_names(registry_path: Path, dataset_name: str) -> list[str]:
    # [FUNCTION DOC] Reads the registry file (a JSON list of dataset objects) and returns the names
    # of the tasks of the first dataset whose name equals the requested one; only task entries that
    # are objects with a name are kept, in file order. Raises a value error listing the available
    # (sorted) dataset names if none matches. File and JSON errors propagate.
    """Pull the list of task names for ``dataset_name`` from ``registry_path``."""
    data: list[dict[str, Any]] = json.loads(registry_path.read_text())
    for dataset in data:
        if dataset.get("name") == dataset_name:
            return [task["name"] for task in (dataset.get("tasks") or []) if isinstance(task, dict) and "name" in task]
    available: list[str] = sorted(d["name"] for d in data if isinstance(d.get("name"), str))
    raise ValueError(f"Dataset {dataset_name!r} not found in {registry_path}. " f"Available datasets: {available}")


async def run_one_trial(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    api_key: str,
    instance_id: str,
    trial_idx: int,
) -> dict[str, Any]:
    # [FUNCTION DOC] Sends one run request and returns a flat result row; never raises for request
    # failures. The body carries the LLM base URL, model, API key, task instance id, agent name,
    # sampling parameters (temperature and max tokens) and max sequence length, posted as JSON to
    # the run path of the server URL with the per-request timeout (connect timeout thirty seconds).
    # A response body that is not JSON, or not a JSON object, is replaced by a mapping holding its
    # first 4000 characters of text. Success row: instance id, trial index, HTTP status code,
    # elapsed seconds (monotonic clock), reward, exit status, agent metrics and eval report (the
    # latter two default to empty mappings). Any exception while posting yields a row with status
    # code -1, elapsed time, no reward, exit status "ClientError" and an error string of the
    # exception type and message. Note the HTTP status is not checked, so an error status still
    # produces a normal row whose reward is absent.
    """Submit one /run request and shape the response into a flat record."""
    body: dict[str, Any] = {
        "base_url": args.base_url,
        "model": args.model,
        "api_key": api_key,
        "instance_id": instance_id,
        "agent_name": args.agent_name,
        "sampling_params": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "max_seq_len": args.max_seq_len,
    }
    timeout = httpx.Timeout(args.per_request_timeout_sec, connect=30.0)
    url = f"{args.server_url.rstrip('/')}/run"
    t0 = monotonic()
    try:
        response = await client.post(url, json=body, timeout=timeout)
        elapsed = monotonic() - t0
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                payload = {"_raw": str(payload)[:4000]}
        except Exception:
            payload = {"_raw": response.text[:4000]}
        return {
            "instance_id": instance_id,
            "trial_idx": trial_idx,
            "status_code": response.status_code,
            "elapsed_sec": elapsed,
            "reward": payload.get("reward"),
            "exit_status": payload.get("exit_status"),
            "agent_metrics": payload.get("agent_metrics", {}),
            "eval_report": payload.get("eval_report", {}),
        }
    except Exception as exc:
        return {
            "instance_id": instance_id,
            "trial_idx": trial_idx,
            "status_code": -1,
            "elapsed_sec": monotonic() - t0,
            "reward": None,
            "exit_status": "ClientError",
            "error": f"{type(exc).__name__}: {exc}",
        }


async def main(args: argparse.Namespace) -> None:
    # [FUNCTION DOC] Orchestrates the evaluation. Exits with a message if the API key environment
    # variable is unset or empty, or if there are zero trials. Prints a plan line, creates output
    # parent directories and deletes any existing JSONL file. Creates one task per (task name,
    # trial index) pair (all at once, gated by a semaphore of the concurrency cap) and, as each
    # finishes, appends its row as a JSON line (non-serialisable values stringified) and prints a
    # progress line with counts, truncated instance id, trial, status code, elapsed seconds, reward
    # and exit status. Afterwards writes the parquet file from the rows minus nested metrics and
    # report columns, and prints counts of rows, trials with a reward and trials with reward exactly
    # one (pass rate over the scored trials, using at least one as denominator).
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Environment variable {args.api_key_env!r} is not set. " f"Export your DeepSeek API key first."
        )

    task_names = load_task_names(args.registry_path, args.dataset_name)
    total = len(task_names) * args.n_trials_per_task
    if total == 0:
        raise SystemExit(
            f"No trials to run. Dataset {args.dataset_name!r} has "
            f"{len(task_names)} tasks and n_trials_per_task is "
            f"{args.n_trials_per_task}."
        )
    print(
        f"{args.dataset_name}: {len(task_names)} tasks "
        f"x {args.n_trials_per_task} trials = {total} runs; "
        f"max_concurrent={args.max_concurrent}; server={args.server_url}"
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.unlink(missing_ok=True)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.max_concurrent)

    async def bounded(client: httpx.AsyncClient, iid: str, idx: int) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Runs one trial while holding the concurrency semaphore and returns
        # its result row.
        async with sem:
            return await run_one_trial(client, args, api_key, iid, idx)

    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        tasks: list[Awaitable[dict[str, Any]]] = [
            asyncio.create_task(bounded(client, iid, trial_idx))
            for iid in task_names
            for trial_idx in range(args.n_trials_per_task)
        ]
        for done_count, fut in enumerate(asyncio.as_completed(tasks), start=1):
            row = await fut
            results.append(row)
            with args.output_jsonl.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(
                f"[{done_count:>4d}/{total}] "
                f"{row['instance_id'][:40]:<40s} trial={row['trial_idx']} "
                f"code={row['status_code']:>4d} "
                f"elapsed={row['elapsed_sec']:>7.1f}s "
                f"reward={row.get('reward')} "
                f"status={row.get('exit_status')}",
                flush=True,
            )

    flat_rows = [{k: v for k, v in row.items() if k not in {"agent_metrics", "eval_report"}} for row in results]
    pl.from_dicts(flat_rows).write_parquet(args.output_parquet)

    completed = [r for r in results if r.get("reward") is not None]
    n_pass = sum(1 for r in completed if r["reward"] == 1.0)
    print(
        f"\nWrote {len(results)} rows to {args.output_parquet} "
        f"(streamed jsonl: {args.output_jsonl}).\n"
        f"Completed (verifier-scored): {len(completed)}/{len(results)}.\n"
        f"Pass rate (reward==1.0): {n_pass}/{len(completed)} "
        f"= {n_pass / max(len(completed), 1):.1%}."
    )


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
