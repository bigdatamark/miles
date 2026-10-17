# [MODULE DOC] Command-line driver for evaluating an external OpenAI-compatible policy API on
# Terminal-Bench-2 tasks, without Miles training. Rows come from either a JSON-lines data file
# (the output of the data generator in this directory) or a comma-separated list of task ids
# (each given the agent system prompt as its only prompt message). Each row is run through an
# episode function chosen at start: if the environment variable OPENENV_TB2_TASKS_DIR is set, the
# per-episode sandbox backend resolved from OPENENV_SANDBOX_BACKEND supplies it (via the common
# sandbox module); otherwise the shared-env-server episode function from the agent function module
# is used (server url from OPENENV_ENV_URL or its default). Episodes run with bounded concurrency
# under a per-episode wall-clock cap (OPENENV_MAX_ROLLOUT_TIME_SECONDS, default 3600) and failures
# are recorded as errored rather than scored zero. Prints per-task lines and a pass summary, and
# can write per-task results as JSON lines. Policy settings come from the environment variables
# DEEPSEEK_API_KEY or POLICY_API_KEY (required), POLICY_BASE_URL (default the DeepSeek api root)
# and POLICY_MODEL (default deepseek-v4-flash); only the sampling temperature is forwarded.
"""Standalone Terminal-Bench-2 eval: an OpenAI-compatible *API* as the policy,
per-episode sandboxes as the env. No GPU, no miles training pipeline.

This reuses the exact agent-env loop miles runs during training
(``openenv_agent_function.multi_turn``: reset -> {policy emits a shell command
-> exec -> feed output back} -> canonical tests/test.sh -> binary reward) but
swaps miles' session-server policy for a plain API client. The machine running
this only orchestrates; the policy runs in the cloud (e.g. DeepSeek) and each
episode runs in its own per-provider sandbox (the task's OFFICIAL image +
env server layer — recipe in the sibling ``tb2_sandbox_recipe`` module,
materialized per provider), created
before the episode and deleted after.

Why a separate script and not ``run-openenv-tbench2.py``: those launchers always
bring up Megatron+sglang via Ray (the policy must be miles' own engine so the
session server can record on-policy tokens for the backward pass). Pure eval
needs none of that. The ``run()`` entry also hardcodes ``api_key="EMPTY"``
(self-hosted engines don't check it), so we call each module's ``run_episode``
with our own authenticated client instead.

Env vars:
  DEEPSEEK_API_KEY / POLICY_API_KEY   policy API key (required)
  POLICY_BASE_URL   OpenAI-compatible root (default https://api.deepseek.com)
  POLICY_MODEL      default deepseek-v4-flash
  OPENENV_SANDBOX_BACKEND   per-episode sandbox mode: agentenv (an alias for
                    e2b) / daytona / e2b / modal. Left unset, OPENENV_ENV_URL
                    (one shared env server) is used instead.
  OPENENV_TB2_TASKS_DIR     required with the above: the TB2 checkout to build
                    task images from. Each backend then resolves its OWN
                    credential exactly as it does under training -- from the
                    environment, else its key file -- so nothing is passed
                    through here; see the recipe README for each provider's.
  OPENENV_MAX_TURNS, OPENENV_MAX_ROLLOUT_TIME_SECONDS, ...   as in the adapter.

Usage:
  DEEPSEEK_API_KEY=sk-... \
  OPENENV_SANDBOX_BACKEND=e2b OPENENV_TB2_TASKS_DIR=/path/to/terminal-bench-2 \
  python eval_tbench2_via_api.py --tasks chess-best-move --concurrency 2
  # or from make_tbench2_data.py output:
  python eval_tbench2_via_api.py --data /root/tbench2_train.jsonl
"""

import argparse
import asyncio
import json
import os
import sys

from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openenv_agent_function as oaf  # noqa: E402
import openenv_sandbox_common as sandbox_common  # noqa: E402


def _load_rows(args: argparse.Namespace) -> list[dict]:
    # [FUNCTION DOC] HELPER: Builds the list of work rows. With a tasks string: splits on commas,
    # trims, drops empties, and creates one row per id holding a single system-role prompt message
    # (the shared agent system prompt) and metadata with the task id. Otherwise reads the data file
    # as JSON lines, skipping blank lines, and returns the parsed rows unchanged.
    """--tasks builds rows with the adapter's own agent-contract prompt, so it
    reproduces exactly what make_tbench2_data.py emits for training."""
    if args.tasks:
        ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        return [
            {"prompt": [{"role": "system", "content": oaf.TB2_AGENT_SYSTEM_PROMPT}], "metadata": {"task_id": t}}
            for t in ids
        ]
    with open(args.data) as f:
        return [json.loads(line) for line in f if line.strip()]


async def _eval_one(
    run_episode, policy: AsyncOpenAI, model: str, row: dict, request_kwargs: dict
) -> tuple[str, float | None, dict]:
    # [FUNCTION DOC] HELPER: Runs one row's episode and never raises. Returns a triple of task id
    # (question mark when missing), reward (none when errored, timed out or without a verdict) and
    # a metrics dict. The episode function is called with the policy client, model name, the row's
    # prompt messages (default empty), the request kwargs and the row's metadata, under a timeout
    # read from OPENENV_MAX_ROLLOUT_TIME_SECONDS (default 3600 seconds). On timeout the metrics hold
    # an error string naming the limit; on any other exception, an error string with the exception
    # class name and message.
    task_id = row.get("metadata", {}).get("task_id", "?")
    cap = float(os.getenv("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "3600"))
    try:
        # Per-episode wall-clock cap: run_episode deliberately leaves timeout
        # and failure semantics to the caller (run() maps a timeout to reward 0
        # for training; a sweep must count it as errored instead). Bounds a
        # task that loops on slow generations so one straggler can't stall the
        # whole sweep.
        reward, metrics = await asyncio.wait_for(
            run_episode(
                policy,
                model,
                row.get("prompt", []),
                request_kwargs,
                row.get("metadata", {}),
            ),
            timeout=cap,
        )
        return task_id, reward, metrics
    except asyncio.TimeoutError:
        return task_id, None, {"error": f"timeout>{cap:.0f}s"}
    except Exception as e:  # noqa: BLE001 - one task failing must not sink the sweep
        return task_id, None, {"error": f"{type(e).__name__}: {e}"}


async def main() -> None:
    # [FUNCTION DOC] Script entry point. Options: exactly one of data file or tasks list
    # (required), concurrency (default 4), sampling temperature (default 0.8) and an optional
    # output path. Exits with a message if no api key is set. Chooses the episode function (sandbox
    # backend when the tasks-dir variable is non-empty, else the shared server; an invalid backend
    # name exits with the error text), prints a header line, creates the policy client (3600 s
    # timeout, no automatic retries), and runs all rows concurrently limited by a semaphore,
    # printing one line per finished task (reward or ERR, task id, error text or turn count).
    # Then prints the solved count over scored tasks (reward at least 1.0) with the errored count,
    # the sorted solved ids, and if requested writes one JSON line per task with task id, reward
    # and metrics in input order. The policy client is never explicitly closed.
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="jsonl from make_tbench2_data.py (one row per task)")
    src.add_argument("--tasks", help="comma-separated task_ids (system prompt inlined)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default="", help="optional path to write per-task jsonl results")
    args = ap.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("POLICY_API_KEY")
    if not api_key:
        sys.exit("set DEEPSEEK_API_KEY (or POLICY_API_KEY)")
    base_url = os.getenv("POLICY_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("POLICY_MODEL", "deepseek-v4-flash")
    rows = _load_rows(args)
    tasks_dir = os.getenv("OPENENV_TB2_TASKS_DIR", "").strip()
    if tasks_dir:
        try:
            backend = sandbox_common.resolve_backend(os.getenv("OPENENV_SANDBOX_BACKEND"))
            run_episode = sandbox_common.load_backend(backend).run_episode
        except ValueError as e:
            sys.exit(str(e))
        env_desc = f"{backend} sandboxes (tasks_dir={tasks_dir})"
    else:
        run_episode = oaf.run_episode
        env_desc = os.getenv("OPENENV_ENV_URL", oaf._DEFAULT_ENV_URL)
    print(f"policy={model} @ {base_url} | env={env_desc} | " f"{len(rows)} tasks | concurrency={args.concurrency}")

    policy = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=3600.0, max_retries=0)
    request_kwargs = {"temperature": args.temperature}
    sem = asyncio.Semaphore(args.concurrency)

    async def _run(row: dict) -> tuple[str, float | None, dict]:
        # [FUNCTION DOC] HELPER: Evaluates one row while holding the concurrency semaphore, prints
        # its result line, and returns the triple of task id, reward and metrics.
        async with sem:
            tid, reward, metrics = await _eval_one(run_episode, policy, model, row, request_kwargs)
            tag = "ERR" if reward is None else f"{reward:.0f}"
            extra = metrics.get("error") or f"turns={metrics.get('turns')}"
            print(f"  [{tag}] {tid:40s} {extra}", flush=True)
            return tid, reward, metrics

    results = await asyncio.gather(*(_run(r) for r in rows))

    scored = [(t, r) for t, r, _ in results if r is not None]
    solved = sum(1 for _, r in scored if r >= 1.0)
    errs = sum(1 for _, r, _ in results if r is None)
    print(f"\n=== pass {solved}/{len(scored)} scored ({errs} errored) ===")
    if scored:
        print("solved:", sorted(t for t, r in scored if r >= 1.0))
    if args.out:
        with open(args.out, "w") as f:
            for t, r, m in results:
                f.write(json.dumps({"task_id": t, "reward": r, "metrics": m}) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
