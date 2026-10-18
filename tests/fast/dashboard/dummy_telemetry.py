# [MODULE DOC] HELPER: Test fixture generator for fake live-collection telemetry written through
# the real metric store into the dashboard folder of a dump directory. Writes run metadata, two
# topology snapshots (engine B moves ports at the restart instant), per-second GPU samples (a
# 30 second idle period before step zero, then per-step utilization shaped by rollout, training and
# weight update phases, with one 100 percent spike), phase events per step (a rollout on a manager
# node and, for each GPU rank, wait / train / weight update phases with small per-rank skew), one
# metrics record per step, engine samples every two seconds (running requests, throughput, token
# and first-token counters), and trajectory lifecycle events per sample. Module constants define the
# timeline (base time, node addresses, step length 100 s, rollout 40 s, drain times, restart offset,
# spike location). Timestamps are synthetic seconds starting at one million.
"""Generate DUMMY live-collection telemetry through miles' REAL store write path.

Companion of ``dummy_dump.py`` (which fabricates the ``.pt`` dump side): only
the *content* here is fake — every record goes through the actual
``MetricStore`` append/flush pipeline into ``{dump_details}/dashboard/``, so
schema or query changes break the timeline tests instead of drifting
silently.

This file doubles as the OUTPUT CONTRACT for the live collector (PR-10 in
the implementation plan): the collector must produce files that
``MetricStore.load`` plus the timeline queries interpret exactly like this
fixture.

The fabricated story deliberately contains the shapes the efficiency view is
designed to surface:

1. colocate cadence per 100 s step: rollout [T, T+40) then per-rank
   ``train_wait`` / ``actor_train`` / ``update_weights`` with cross-rank skew;
2. a rollout long tail: engine A stays busy until T+38 while engine B drains
   at T+25 (util drops, ``num_running_reqs`` hits 0 — a bubble);
3. a fault-tolerance engine restart MID-ROLLOUT of step 1: engine B moves
   from port 15004 to 15006, so topology windows split and the rollout-lane
   expansion must clip intervals at the boundary;
4. one util spike (100%) at a known instant for downsampling assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from miles.dashboard.store import (
    EngineInfo,
    EngineSample,
    GpuSample,
    Meta,
    MetricsRecord,
    MetricStore,
    PhaseEvent,
    Role,
    TopologySnapshot,
    TrajectoryEvent,
    TrajectoryEventKind,
)

BASE_TS = 1_000_000.0
GPU_NODE = "10.0.0.1"
MANAGER_NODE = "10.0.0.2"  # rollout manager runs on a different (GPU-less) node

ENGINE_A = "http://10.0.0.1:15000"
ENGINE_B_OLD = "http://10.0.0.1:15004"
ENGINE_B_NEW = "http://10.0.0.1:15006"

STEP_SECONDS = 100
INIT_SECONDS = 30  # gap between collector start (meta.start_ts) and step 0
ROLLOUT_SECONDS = 40
DRAIN_A = 38  # engine A busy almost to the end of the rollout window
DRAIN_B = 25  # engine B drains early -> bubble on its lanes
RESTART_OFFSET = 20  # engine B restarts at step-1 rollout start + 20s (mid-rollout)

SPIKE = dict(step=0, offset=50, gpu=0, util=100)


@dataclass
class DummyTelemetryTruth:
    # [CLASS DOC] HELPER: Ground truth of the generated telemetry: step count, GPU count, the
    # timestamp of the engine B restart, and the timestamp of the utilization spike, plus helpers
    # giving the start time of a step and its rollout interval.
    """Ground truth returned by :func:`dump_dummy_telemetry` for test assertions."""

    steps: int
    gpus: int
    restart_ts: float
    spike_ts: float

    def step_start(self, step: int) -> float:
        # [FUNCTION DOC] HELPER: Returns the base time plus the init gap plus step times the step
        # length (seconds).
        return BASE_TS + INIT_SECONDS + step * STEP_SECONDS

    def rollout_interval(self, step: int) -> tuple[float, float]:
        # [FUNCTION DOC] HELPER: Returns the pair (step start, step start plus rollout seconds).
        return self.step_start(step), self.step_start(step) + ROLLOUT_SECONDS


SAMPLES_PER_STEP = 8  # dummy_dump's default num_prompts * n_samples_per_prompt


def _trajectory_events(store: MetricStore, truth: DummyTelemetryTruth, samples_per_step: int) -> None:
    # [FUNCTION DOC] HELPER: Appends trajectory events to the store for every step and every sample
    # position. Global sample index = step times samples per step plus position; agentic when the
    # index is divisible by three; group index = position integer-divided by two (fixed, not tied to
    # samples per prompt); weight version = index mod 3 (agentic second turn uses one more). Start
    # time is staggered across the rollout window (rollout length minus 16 s, divided by the sample
    # count, times the position). Events per sample: attempt start at start; generation start at +1
    # (turn 1) and end at +6; if agentic, tool start at +6 and end at +9 (detail "1 calls"), then
    # generation start at +9 and end at +14 on turn 2 with the bumped version; attempt end at +15
    # (agentic) or +7 with detail "completed".
    """Lifecycle events for each step's batch, aligned with dummy_dump's
    global sample indices (step * n + i) and its agentic pattern (every third
    sample: two turns, one tool call, mixed weight versions)."""
    for step in range(truth.steps):
        t0, t1 = truth.rollout_interval(step)
        for i in range(samples_per_step):
            index = step * samples_per_step + i
            agentic = index % 3 == 0
            # stagger submits so every lifecycle fits inside the rollout window
            start = t0 + i * (t1 - t0 - 16.0) / samples_per_step
            version = str(index % 3)

            def emit(kind, ts, turn=-1, version=version, detail="", index=index, group=i // 2):
                # [FUNCTION DOC] HELPER: Appends one trajectory event of the given kind and time to
                # the store. Turn defaults to -1 (not turn-specific); version, detail, sample index
                # and group index default to the loop's current values (bound at definition).
                store.append(
                    TrajectoryEvent(
                        ts=ts,
                        kind=kind,
                        sample_index=index,
                        group_index=group,
                        turn=turn,
                        weight_version=version,
                        detail=detail,
                    )
                )

            emit(TrajectoryEventKind.ATTEMPT_START, start)
            emit(TrajectoryEventKind.GEN_START, start + 1.0, turn=1)
            emit(TrajectoryEventKind.GEN_END, start + 6.0, turn=1)
            if agentic:
                emit(TrajectoryEventKind.TOOL_START, start + 6.0, turn=1, detail="1 calls")
                emit(TrajectoryEventKind.TOOL_END, start + 9.0, turn=1, detail="1 calls")
                bumped = str(index % 3 + 1)
                emit(TrajectoryEventKind.GEN_START, start + 9.0, turn=2, version=bumped)
                emit(TrajectoryEventKind.GEN_END, start + 14.0, turn=2, version=bumped)
            emit(TrajectoryEventKind.ATTEMPT_END, start + (15.0 if agentic else 7.0), detail="completed")


def dump_dummy_telemetry(
    dump_dir: Path, *, steps: int = 3, gpus: int = 4, samples_per_step: int = SAMPLES_PER_STEP
) -> DummyTelemetryTruth:
    # [FUNCTION DOC] HELPER: Writes the whole fake telemetry story into a metric store rooted at the
    # dashboard folder of the dump directory and returns the truth record (restart time = base +
    # init gap + one step + restart offset; spike time from the spike step and offset). Inputs
    # (keyword-only): steps (3), GPU count (4), samples per step (default 8, matching the dump
    # generator's default batch of 4 prompts times 2). Writes: metadata (run name "dummy-telemetry",
    # start time = base, colocate argument); two topology snapshots (before: engines A on GPUs 0-1
    # and B at its old address on GPUs 2-3; at the restart time: B at its new address); 30 seconds
    # of idle GPU samples before step zero (20 percent util); per step a rollout phase on the
    # manager node, per rank wait (1 + 0.2 times rank seconds), train (until 80 + 0.5 times rank
    # after step start) and weight update (4 + 0.3 times rank seconds) phases, a metrics record near
    # the end of the step (step time, wait ratio, reward mean, MFU and peak teraflops values that
    # shift with the step), per-second GPU samples (during rollout 90 percent until the GPU's
    # engine drains, 5 afterwards, drain being 38 s for GPUs 0-1 and 25 s for others; 95 percent
    # until second 80; then 30; with the configured spike overriding one value), and every two
    # seconds engine samples for A and for B (old or new address by time): running requests (16
    # until drain or rollout end, else 0), generation throughput (50 times running), cumulative
    # generation tokens (100 per second of offset), and first-token latency sum (0.4 per second) and
    # count (2 per second). Finally appends trajectory events and flushes the store.
    truth = DummyTelemetryTruth(
        steps=steps,
        gpus=gpus,
        restart_ts=BASE_TS + INIT_SECONDS + STEP_SECONDS + RESTART_OFFSET,
        spike_ts=BASE_TS + INIT_SECONDS + SPIKE["step"] * STEP_SECONDS + SPIKE["offset"],
    )
    store = MetricStore(dump_dir / "dashboard")
    store.write_meta(Meta(run_name="dummy-telemetry", start_ts=BASE_TS, args={"colocate": True}))

    def engine(addr: str, engine_rank: int, gpu_ids: list[int]) -> EngineInfo:
        # [FUNCTION DOC] HELPER: Builds a regular engine description at the address and rank, with
        # GPUs on the GPU node for the given ids and unknown GPU unique identifiers.
        return EngineInfo(
            addr=addr,
            worker_type="regular",
            engine_rank=engine_rank,
            gpus=[[GPU_NODE, g] for g in gpu_ids],
            gpu_uuids=[None] * len(gpu_ids),
        )

    store.append(TopologySnapshot(ts=BASE_TS, engines=[engine(ENGINE_A, 0, [0, 1]), engine(ENGINE_B_OLD, 1, [2, 3])]))
    store.append(
        TopologySnapshot(ts=truth.restart_ts, engines=[engine(ENGINE_A, 0, [0, 1]), engine(ENGINE_B_NEW, 1, [2, 3])])
    )

    def engine_b_addr(ts: float) -> str:
        # [FUNCTION DOC] HELPER: Returns engine B's old address for times before the restart time and
        # its new address from then on.
        return ENGINE_B_OLD if ts < truth.restart_ts else ENGINE_B_NEW

    for offset in range(INIT_SECONDS):
        for gpu in range(gpus):
            store.append(GpuSample(ts=BASE_TS + offset, node=GPU_NODE, gpu=gpu, util=20, mem_mb=30_000, power_w=300))

    for step in range(steps):
        t = truth.step_start(step)
        rollout_end = t + ROLLOUT_SECONDS
        store.append(
            PhaseEvent(
                name="rollout", t0=t, t1=rollout_end, node=MANAGER_NODE, gpus=[], rank=-1, role=Role.ROLLOUT_MANAGER
            )
        )
        for rank in range(gpus):
            wait_end = rollout_end + 1 + 0.2 * rank
            train_end = t + 80 + 0.5 * rank
            store.append(
                PhaseEvent(
                    name="train_wait",
                    t0=rollout_end,
                    t1=wait_end,
                    node=GPU_NODE,
                    gpus=[rank],
                    rank=rank,
                    role=Role.TRAIN,
                )
            )
            store.append(
                PhaseEvent(
                    name="actor_train",
                    t0=wait_end,
                    t1=train_end,
                    node=GPU_NODE,
                    gpus=[rank],
                    rank=rank,
                    role=Role.TRAIN,
                )
            )
            store.append(
                PhaseEvent(
                    name="update_weights",
                    t0=train_end,
                    t1=train_end + 4 + 0.3 * rank,
                    node=GPU_NODE,
                    gpus=[rank],
                    rank=rank,
                    role=Role.TRAIN,
                )
            )
        store.append(
            MetricsRecord(
                ts=t + STEP_SECONDS - 5,
                step_key="rollout/step",
                step=step,
                metrics={
                    "perf/step_time": 95.0 - step,
                    "perf/wait_time_ratio": 0.4 - 0.1 * step,
                    "rollout/rewards_mean": 0.4 + 0.05 * step,
                    "perf/actor_train_mfu": 0.38 - 0.02 * step,
                    "perf/mfu_peak_tflops": 989.0,
                },
            )
        )

        for offset in range(STEP_SECONDS):
            ts = t + offset
            for gpu in range(gpus):
                drain = DRAIN_A if gpu < 2 else DRAIN_B
                if offset < ROLLOUT_SECONDS:
                    util = 90 if offset < drain else 5
                elif offset < 80:
                    util = 95
                else:
                    util = 30
                if step == SPIKE["step"] and gpu == SPIKE["gpu"] and offset == SPIKE["offset"]:
                    util = SPIKE["util"]
                store.append(
                    GpuSample(ts=ts, node=GPU_NODE, gpu=gpu, util=util, mem_mb=60_000 + 100 * gpu, power_w=600)
                )

        for offset in range(0, STEP_SECONDS, 2):
            ts = t + offset
            for addr, drain in ((ENGINE_A, DRAIN_A), (engine_b_addr(ts), DRAIN_B)):
                running = 16.0 if offset < min(drain, ROLLOUT_SECONDS) else 0.0
                store.append(
                    EngineSample(ts=ts, addr=addr, metric="sglang_num_running_reqs", labels={}, value=running)
                )
                store.append(
                    EngineSample(ts=ts, addr=addr, metric="sglang_gen_throughput", labels={}, value=running * 50.0)
                )
                store.append(
                    EngineSample(
                        ts=ts, addr=addr, metric="sglang_generation_tokens_total", labels={}, value=offset * 100.0
                    )
                )
                store.append(
                    EngineSample(
                        ts=ts,
                        addr=addr,
                        metric="sglang_time_to_first_token_seconds_sum",
                        labels={},
                        value=offset * 0.4,
                    )
                )
                store.append(
                    EngineSample(
                        ts=ts,
                        addr=addr,
                        metric="sglang_time_to_first_token_seconds_count",
                        labels={},
                        value=offset * 2.0,
                    )
                )

    _trajectory_events(store, truth, samples_per_step)
    store.flush()
    return truth
