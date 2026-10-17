# [MODULE DOC] Synthetic rollout function for stress testing the fully async pipeline: it
# fabricates samples from random token ids, sends multi-turn requests to the SGLang router, and
# assigns random rewards, with no dataset, reward model or data buffer involved.
# Architecture: a process-wide background worker (one thread running its own event loop)
# continuously keeps a bounded number of sample groups in flight and pushes completed groups
# into a bounded thread-safe queue (max 1000). The synchronous rollout entry point runs a
# coroutine that drains that queue until the requested number of groups is collected. Tunables
# are module constants: prompt length range 50-2800 tokens, per-turn new-token range 512-2048,
# filler ratio 3 filler tokens per generated token, maximum context (environment variable,
# default 60000), per-GPU concurrency (environment variable, default 64), vocabulary size
# 151643, request timeout (environment variable, default 30 minutes).
"""Fully-async rollout using random tokens and random rewards.

Exercises the entire async rollout <-> trainer loop for agentic-like toy data. 
Each Sample drives a multi-turn loop:

  1. start with a random ``input_ids`` sequence,
  2. POST it to the SGLang router's ``/generate`` endpoint
     (``ignore_eos=True``, ``max_new_tokens`` capped per turn),
  3. append the engine's response + random filler tokens to ``input_ids``
     and send the extended sequence as the next turn,
  4. stop when accumulated context exceeds ``MAX_CONTEXT_TOKENS``,
  5. assign a uniformly random reward.

Use this to smoke-test the async pipeline (rollout worker, weight sync,
trainer queue) and to stress-test SGLang under realistic multi-turn
long-context load. 

Wire it up via::

    --rollout-function-path random_async_rollout.generate_rollout_random_async
    --disable-rollout-global-dataset

The ``data_buffer`` argument is accepted for signature compatibility but
ignored; we generate ``Sample`` objects from scratch.
"""

import asyncio
import atexit
import logging
import os
import queue
import random
import threading
import time

import numpy as np
import pybase64
from random_async_sglang_metrics import SGLangMetricsReporter, record_agent_request

from miles.rollout.data_source import DataSource
from miles.utils.async_utils import run
from miles.utils.http_utils import post as http_post
from miles.utils.types import Sample, WeightVersionsPerCall

logger = logging.getLogger(__name__)

# Random tokens never trigger EOS, so ``ignore_eos=True`` plus a hard
# ``max_new_tokens`` cap (drawn from this range) is what terminates each turn.
PROMPT_TOKEN_RANGE = (50, 2800)
MAX_TOKENS_RANGE = (512, 2048)
FILLER_RATIO = 3.0
# Cap on accumulated context per Sample. The multi-turn loop appends each
# turn's response + filler to the next request's input_ids until this is
# exceeded, then the Sample is finalised.
MAX_CONTEXT_TOKENS = int(os.environ.get("RANDOM_ASYNC_MAX_CONTEXT_TOKENS", "60000"))
# Concurrent in-flight SGLang requests target ``CONCURRENCY_PER_GPU`` per
# prefill GPU; each Sample drives one request at a time within its multi-turn
# loop, so this translates to a max-concurrent-Samples bound.
CONCURRENCY_PER_GPU = int(os.environ.get("RANDOM_ASYNC_CONCURRENCY_PER_GPU", "64"))
# Qwen3.5-35B vocab size; works for any model with vocab >= this.
VOCAB_SIZE = 151643
SGLANG_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("RANDOM_ASYNC_SGLANG_REQUEST_TIMEOUT_SECONDS", str(30 * 60)))


def _rand_ids(n: int) -> list[int]:
    # [FUNCTION DOC] HELPER: Return a list of n token ids drawn independently and uniformly from
    # zero up to the vocabulary size (exclusive), using the global random generator.
    return [random.randrange(VOCAB_SIZE) for _ in range(n)]


def _decode_routed_experts(args, encoded: str, token_count: int, start_len: int = 0) -> np.ndarray:
    # [FUNCTION DOC] HELPER: Decode the engine's base64 routed-experts payload (ASCII text of
    # packed 32-bit signed integers) into a three dimensional array of shape rows by number of
    # layers by router top-k (both from args). rows is token_count minus one minus start_len,
    # because the last token has no routing record and the first start_len rows were already
    # received in earlier turns. Raises a value error if rows would be negative; a payload of the
    # wrong size makes the reshape fail.
    row_count = token_count - 1 - start_len
    if row_count < 0:
        raise ValueError(f"routed_experts_start_len={start_len} exceeds token_count - 1 ({token_count - 1})")
    return np.frombuffer(pybase64.b64decode(encoded.encode("ascii")), dtype=np.int32).reshape(
        row_count,
        args.num_layers,
        args.moe_router_topk,
    )


async def _generate_one_random_sample(args, sample: Sample) -> Sample:
    # [FUNCTION DOC] HELPER: Fill one empty sample in place with a random multi-turn trajectory
    # and return it. Setup: a random prompt (length uniform in the prompt range); the router
    # generate URL from args; a sticky data-parallel rank equal to the sample index modulo
    # GPUs per engine; routing replay is on if the args say so.
    # Each turn: stop if the remaining context budget is not positive; choose new tokens as a
    # random value in the range capped by the remaining budget; post the full current token list
    # with temperature 0.8, ignoring end-of-sequence, requesting log probabilities, the sticky
    # rank for routing and for the disaggregated prefill rank, plus (replay only) a request for
    # routed experts starting after rows already received; a per-sample routing key header keeps
    # the sample on one backend. The post uses one retry and an overall timeout (raises a timeout
    # error with sample identifiers). Validates the response: raises a runtime error if there are
    # no output log probabilities, if replay is on and routed experts are absent, or if more
    # tokens came back than the remaining budget. Replay chunks are decoded and appended.
    # Records request metrics (output, prompt, cached tokens, ideal cacheable prefix, duration).
    # Appends filler: random ids numbering 3 times the output count (dropped entirely under
    # replay when output plus filler would reach the remaining budget, since every final token
    # needs routed experts), with log probability 0 and loss mask 0 for filler and the real log
    # probabilities and mask 1 for generated tokens; the segment is truncated to the remaining
    # budget; the ideal cacheable prefix becomes context length plus retained generated tokens;
    # prefix cache info and a weight version record (cut at that prefix) are added to the sample.
    # The loop ends when the context reaches the budget.
    # Finalization: tokens, response length, empty response text, reward uniform in minus one to
    # one, loss mask, rollout log probabilities; under replay requires at least one chunk and
    # that the concatenated rows equal token count minus one, else raises a runtime error;
    # status set to completed; prints a one-line summary.
    # KNOWN ISSUE: remaining_tokens is recomputed after the output arrives but the current
    # context has not changed since it was first computed, so the recomputation is a no-op.
    """Multi-turn rollout against SGLang with random prompts, fillers, and reward.

    Send a request, take the engine's response, append response + random filler
    to the running ``input_ids``, send the extended sequence as the next turn —
    until the accumulated context exceeds ``MAX_CONTEXT_TOKENS``.
    """
    prompt_ids = _rand_ids(random.randint(*PROMPT_TOKEN_RANGE))
    current_ids = list(prompt_ids)
    accumulated_response: list[int] = []
    accumulated_log_probs: list[float] = []
    accumulated_loss_mask: list[int] = []
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    start_time = time.time()
    turns = 0
    perfect_cacheable_prefix_len = 0
    sticky_dp_rank = sample.index % args.rollout_num_gpus_per_engine
    use_routing_replay = getattr(args, "use_rollout_routing_replay", False)
    routed_experts_chunks: list[np.ndarray] = []
    routed_experts_start_len = 0

    while True:
        turns += 1
        remaining_tokens = MAX_CONTEXT_TOKENS - len(current_ids)
        if remaining_tokens <= 0:
            break
        max_new_tokens = min(random.randint(*MAX_TOKENS_RANGE), remaining_tokens)
        payload = {
            "input_ids": current_ids,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.8,
                "ignore_eos": True,
            },
            "return_logprob": True,
            # Keep each sample on one prefill DP rank, and tell decode exactly
            # which prefill DP owns the KV instead of relying on room queries.
            "routed_dp_rank": sticky_dp_rank,
            "disagg_prefill_dp_rank": sticky_dp_rank,
        }
        if use_routing_replay:
            payload["return_routed_experts"] = True
            payload["routed_experts_start_len"] = routed_experts_start_len
        headers = {"x-smg-routing-key": f"random-async-sample-{sample.index}"}
        request_start = time.monotonic()
        try:
            data = await asyncio.wait_for(
                http_post(url, payload, max_retries=1, headers=headers),
                timeout=SGLANG_REQUEST_TIMEOUT_SECONDS,
            )
        except TimeoutError as e:
            raise TimeoutError(
                "SGLang /generate timed out after "
                f"{SGLANG_REQUEST_TIMEOUT_SECONDS}s "
                f"(sample_index={sample.index}, group_index={sample.group_index}, "
                f"input_tokens={len(current_ids)}, max_new_tokens={max_new_tokens})"
            ) from e
        request_duration = time.monotonic() - request_start
        meta = data["meta_info"]
        lp = meta["output_token_logprobs"]
        if not lp:
            raise RuntimeError(f"SGLang returned no output_token_logprobs: {data}")
        out_tokens = [item[1] for item in lp]
        out_log_probs = [item[0] for item in lp]
        if use_routing_replay and "routed_experts" not in meta:
            raise RuntimeError(
                "SGLang response missing required routed_experts "
                f"(sample_index={sample.index}, group_index={sample.group_index}, turns={turns}, "
                f"input_tokens={len(current_ids)}, output_tokens={len(out_tokens)}, "
                f"meta_keys={sorted(meta.keys())}, finish_reason={meta['finish_reason']})"
            )
        if len(out_tokens) > remaining_tokens:
            raise RuntimeError(
                "SGLang returned more tokens than requested "
                f"(sample_index={sample.index}, output_tokens={len(out_tokens)}, "
                f"remaining_tokens={remaining_tokens})"
            )
        if use_routing_replay:
            routed_experts_chunk = _decode_routed_experts(
                args,
                meta["routed_experts"],
                len(current_ids) + len(out_tokens),
                routed_experts_start_len,
            )
            routed_experts_chunks.append(routed_experts_chunk)
            routed_experts_start_len += routed_experts_chunk.shape[0]
        cached_tokens = meta["cached_tokens"]
        prompt_tokens = meta["prompt_tokens"]
        record_agent_request(
            output_tokens=len(out_tokens),
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            perfect_cacheable_tokens=perfect_cacheable_prefix_len,
            request_time=request_duration,
        )

        filler = _rand_ids(int(len(out_tokens) * FILLER_RATIO))
        if use_routing_replay and len(out_tokens) + len(filler) >= remaining_tokens:
            # R3 replay requires routed experts for every final token. Filler is
            # covered by the next turn's prompt, so do not let filler end a sample.
            filler = []
        segment = out_tokens + filler
        segment_log_probs = out_log_probs + [0.0] * len(filler)
        segment_loss_mask = [1] * len(out_tokens) + [0] * len(filler)
        remaining_tokens = MAX_CONTEXT_TOKENS - len(current_ids)
        if len(segment) > remaining_tokens:
            segment = segment[:remaining_tokens]
            segment_log_probs = segment_log_probs[:remaining_tokens]
            segment_loss_mask = segment_loss_mask[:remaining_tokens]
        retained_generated_tokens = min(len(out_tokens), len(segment))
        perfect_cacheable_prefix_len = len(current_ids) + retained_generated_tokens
        accumulated_response.extend(segment)
        accumulated_log_probs.extend(segment_log_probs)
        accumulated_loss_mask.extend(segment_loss_mask)
        current_ids.extend(segment)
        sample.prefix_cache_info.add(meta)

        sample.weight_versions.append(
            WeightVersionsPerCall.from_meta_info(meta, output_end=perfect_cacheable_prefix_len)
        )

        if len(current_ids) >= MAX_CONTEXT_TOKENS:
            break

    sample.tokens = current_ids
    sample.response_length = len(accumulated_response)
    sample.response = ""
    sample.reward = random.uniform(-1.0, 1.0)
    sample.loss_mask = accumulated_loss_mask
    sample.rollout_log_probs = accumulated_log_probs
    if use_routing_replay:
        if not routed_experts_chunks:
            raise RuntimeError(f"Routing replay enabled but no routed experts were returned for sample {sample.index}")
        routed_experts = np.concatenate(routed_experts_chunks, axis=0)
        if routed_experts.shape[0] != len(current_ids) - 1:
            raise RuntimeError(
                "Routing replay metadata length does not match final sample tokens "
                f"(sample_index={sample.index}, routed_experts={routed_experts.shape[0]}, "
                f"tokens_minus_one={len(current_ids) - 1})"
            )
        sample.rollout_routed_experts = routed_experts
    sample.status = Sample.Status.COMPLETED
    print(
        f"Random rollout sample finished: group={sample.group_index}, "
        f"sample={sample.index}, turns={turns}, tokens={len(current_ids)}, "
        f"response_tokens={sample.response_length}, duration={time.time() - start_time:.2f}s",
        flush=True,
    )
    return sample


async def _generate_random_group(args, group: list[Sample]) -> list[Sample]:
    # [FUNCTION DOC] HELPER: Generate all samples of a group concurrently (one task each), return
    # them in the original order, and print a summary with group index, sample count, total token
    # count and duration. An exception in any sample propagates.
    start_time = time.time()
    result = list(await asyncio.gather(*[_generate_one_random_sample(args, s) for s in group]))
    total_tokens = sum(len(sample.tokens) for sample in result)
    print(
        f"Random rollout group finished: group={group[0].group_index}, "
        f"samples={len(result)}, total_tokens={total_tokens}, duration={time.time() - start_time:.2f}s",
        flush=True,
    )
    return result


_global_worker: "AsyncRandomRolloutWorker | None" = None
_worker_lock = threading.Lock()


def get_global_worker(args) -> "AsyncRandomRolloutWorker":
    # [FUNCTION DOC] Return the process-wide worker, creating and starting it under a lock if it
    # does not exist or its thread has died (a replacement worker starts with a fresh queue; the
    # old one is not stopped). Side effects: sets the module global, starts a thread.
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global random-async rollout worker...")
            _global_worker = AsyncRandomRolloutWorker(args)
            _global_worker.start()
        return _global_worker


def stop_global_worker() -> None:
    # [FUNCTION DOC] HELPER: Under the lock, stop the process-wide worker if any and clear the
    # global. Registered to run at interpreter exit.
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(stop_global_worker)


class AsyncRandomRolloutWorker:
    # [CLASS DOC] Owns the background thread, its event loop and the output queue. State: the run
    # args, a running flag, a bounded output queue of pairs of group id and sample list, the
    # thread handle, an optional metrics reporter, the first fatal failure, the current number of
    # in-flight groups, and running counters for sample and group indices. Groups are fabricated
    # locally; each contains the configured samples per prompt.
    """Background asyncio loop that fills an output queue with random sample groups.

    Mirrors the worker inside ``miles.rollout.fully_async_rollout.FullyAsyncRolloutFn`` but
    skips the data buffer and reward model entirely.
    """

    def __init__(self, args):
        # [FUNCTION DOC] Store args and initialize empty state (queue capacity 1000, indices 0).
        # If the args enable sglang metrics, create a metrics reporter pointed at the router
        # address, using GPUs per engine as both the prefill and decode GPU counts. The reporter
        # and thread are not started here.
        self.args = args
        self.running = True
        self.output_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.metrics_reporter: SGLangMetricsReporter | None = None
        self.failure: BaseException | None = None
        self.active_count = 0
        self._sample_index = 0
        self._group_index = 0

        if args.sglang_enable_metrics:
            router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
            self.metrics_reporter = SGLangMetricsReporter(
                router_url=router_url,
                prefill_num_gpus=args.rollout_num_gpus_per_engine,
                decode_num_gpus=args.rollout_num_gpus_per_engine,
            )

    def _make_random_group(self) -> list[Sample]:
        # [FUNCTION DOC] HELPER: Create a group of empty samples, one per sample-per-prompt, all
        # sharing the next group index and receiving consecutive globally increasing sample
        # indices; advances both counters. Returns the list.
        group: list[Sample] = []
        for _ in range(self.args.n_samples_per_prompt):
            s = Sample()
            s.group_index = self._group_index
            s.index = self._sample_index
            self._sample_index += 1
            group.append(s)
        self._group_index += 1
        return group

    async def continuous_worker_loop(self):
        # [FUNCTION DOC] The worker's main coroutine. The in-flight group bound is the per-GPU
        # concurrency times GPUs per engine divided (integer) by samples per prompt, at least 1.
        # Every iteration (about once per second): harvest finished group tasks, putting a pair of
        # group id and result into the queue (a blocking put when full; a failed task re-raises
        # its exception here, ending the loop); then, if below the bound, launch exactly one new
        # group task; update the active count; sleep 1 second. On exit (stop flag, error or
        # cancel) all remaining tasks are cancelled and awaited, and the active count is updated.
        # KNOWN ISSUE: the launch step is written as a while loop but ends with an unconditional
        # exit, so at most one group is started per second, which can make ramp-up to the bound
        # slow for large bounds.
        print("Continuous random-async rollout worker started")
        active: dict[asyncio.Task, int] = {}
        max_concurrent_groups = max(
            1,
            (CONCURRENCY_PER_GPU * self.args.rollout_num_gpus_per_engine) // self.args.n_samples_per_prompt,
        )
        gid_counter = 0

        try:
            while self.running:
                done = [task for task in active if task.done()]
                for task in done:
                    gid = active.pop(task)
                    result = task.result()
                    self.output_queue.put((gid, result))

                while len(active) < max_concurrent_groups and self.running:
                    group = self._make_random_group()
                    gid = gid_counter
                    gid_counter += 1
                    active[asyncio.create_task(_generate_random_group(self.args, group))] = gid
                    break

                self.active_count = len(active)
                await asyncio.sleep(1)
        finally:
            self.active_count = len(active)
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            print("Continuous random-async rollout worker stopped")

    def worker_thread_func(self):
        # [FUNCTION DOC] Thread body: run the worker loop on a new event loop. On any exception,
        # record it as the failure, clear the running flag, log it and re-raise (ending the thread).
        try:
            asyncio.run(self.continuous_worker_loop())
        except Exception as e:
            self.failure = e
            self.running = False
            logger.exception("Random rollout worker crashed")
            raise

    def start(self):
        # [FUNCTION DOC] Start the metrics reporter (if any) and a daemon worker thread, unless a
        # live thread already exists. Prints a message.
        if self.worker_thread is None or not self.worker_thread.is_alive():
            if self.metrics_reporter is not None:
                self.metrics_reporter.start()
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started random-async worker thread")

    def stop(self):
        # [FUNCTION DOC] Clear the running flag, stop the metrics reporter, and wait up to 5
        # seconds for the thread to finish. Prints a message.
        self.running = False
        if self.metrics_reporter is not None:
            self.metrics_reporter.stop()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Stopped random-async worker thread")

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        # [FUNCTION DOC] Non-blocking drain: remove and return every pair of group id and sample
        # list currently in the queue, in arrival order (possibly empty).
        out: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                out.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return out

    def get_queue_size(self) -> int:
        # [FUNCTION DOC] HELPER: Approximate number of completed groups waiting in the queue.
        return self.output_queue.qsize()

    def get_active_count(self) -> int:
        # [FUNCTION DOC] HELPER: Number of groups in flight as of the last loop iteration.
        return self.active_count

    def check_failures(self) -> None:
        # [FUNCTION DOC] Re-raise the worker's recorded failure if any, else the metrics reporter's
        # failure if any; otherwise do nothing.
        if self.failure is not None:
            raise self.failure
        if self.metrics_reporter is not None and self.metrics_reporter.failure is not None:
            raise self.metrics_reporter.failure


async def _generate_rollout_async(args, rollout_id: int) -> list[list[Sample]]:
    # [FUNCTION DOC] HELPER: Collect the rollout batch size number of groups from the global
    # worker and return them sorted by the index of each group's first sample. Polls every 10
    # milliseconds: re-raises worker failures, raises a runtime error if the worker thread died
    # silently, appends drained groups, and every 30 seconds without progress prints a status line.
    # Prints start and completion lines.
    # KNOWN ISSUE: the drain takes everything from the queue, but groups beyond the target are
    # discarded (the loop stops appending without putting them back), so finished work is lost.
    worker = get_global_worker(args)
    target = args.rollout_batch_size

    print(
        f"Random rollout {rollout_id}: collecting {target} groups, "
        f"queue_depth={worker.get_queue_size()}, active={worker.get_active_count()}"
    )

    data: list[list[Sample]] = []
    start_time = time.time()
    last_progress = start_time
    progress_warn_interval = 30.0

    while len(data) < target:
        worker.check_failures()
        if worker.worker_thread is not None and not worker.worker_thread.is_alive():
            raise RuntimeError("Random rollout worker exited without reporting a failure")

        for _, group in worker.get_completed_groups():
            if len(data) >= target:
                break
            data.append(group)
            last_progress = time.time()

        if time.time() - last_progress > progress_warn_interval:
            print(
                f"Random rollout {rollout_id}: no progress for {progress_warn_interval}s, "
                f"queue_depth={worker.get_queue_size()}, active={worker.get_active_count()}, "
                f"collected={len(data)}/{target}"
            )
            last_progress = time.time()

        await asyncio.sleep(0.01)

    duration = time.time() - start_time
    print(f"Random rollout {rollout_id} done in {duration:.2f}s")
    data.sort(key=lambda g: g[0].index)
    return data


def generate_rollout_random_async(args, rollout_id, data_buffer: DataSource, evaluation: bool = False):
    # [FUNCTION DOC] Synchronous rollout entry point. Raises a value error in evaluation mode.
    # Discards the data buffer, runs the async collection to completion on the shared event loop
    # runner and returns the list of sample groups.
    """Entry point referenced by ``--rollout-function-path``.

    ``data_buffer`` is accepted for signature compatibility and ignored; the
    rollout fabricates its own random ``Sample`` groups.
    """
    if evaluation:
        raise ValueError("random_async_rollout does not support evaluation mode.")
    del data_buffer  # unused
    return run(_generate_rollout_async(args, rollout_id))
