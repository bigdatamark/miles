# [MODULE DOC] Rollout executor: the worker that runs rollout generation (training and evaluation),
# logs metrics, optionally dumps/loads/injects debug rollout data, and converts samples into
# training data packs. Supports two rollout function generations (legacy v1 function-style, and
# the newer constructor-based objects with save/load/dispose) selected by an environment switch.
# Evaluation can be a plain call to the eval rollout function, or snapshot (checkpoint) based using
# either a dedicated eval fleet or an external backend. Importing this module lowers the log level
# of the HTTP client libraries (httpx, httpcore) to warning.
import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.ray.rollout.eval_fleet import EvalFleetInfo, RolloutExecutorEvalFleet
from miles.ray.rollout.metrics import log_eval_rollout_data, log_eval_skip, log_rollout_data
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    convert_samples_to_train_data,
    split_train_data_by_dp,
)
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    call_rollout_fn,
)
from miles.rollout.checkpoint_eval import CheckpointEvalFn, EvalSkip
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils import object_store
from miles.utils.async_utils import maybe_await
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import event_logger_context
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.data import RolloutDataPack
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.function_registry import load_function
from miles.utils.hf_utils.config import is_complete_hf_export
from miles.utils.http_utils import init_http_client
from miles.utils.init_once import InitOnce, init_once
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.timer import timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.weight_version import assert_samples_weight_version_sane, assert_weight_version_is_published
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class RolloutExecutor:
    # [CLASS DOC] Stateful executor. Construction is cheap; the heavy setup is the one-time init.
    # Per-trainer-model state is kept in mappings keyed by trainer model id (none for a single
    # policy): the latest published weight version, the count of rollouts since the last weight
    # publish, and the trainer's parallel config (set by the orchestrator). Also holds the last
    # rollout id, an eval lock (serializes checkpoint evals), an optional eval fleet proxy, the
    # data source, the train and eval rollout functions, optional custom reward post-process and
    # sample-conversion functions, and an optional metric checker. Methods are called remotely.
    """The class to run rollout and convert rollout data to training data."""

    def __init__(
        self,
        *,
        args,
        router_providers: Sequence[BaseWorkerProvider],
        session_server_provider: BaseWorkerProvider | None,
        inference_controller_provider: BaseWorkerProvider,
    ):
        # [FUNCTION DOC] Create the init-once guard, configure process logging with the rollout
        # executor identity, and store args, the three providers and empty per-model state
        # mappings. No network or data source work happens here.
        self._init_once = InitOnce(type(self).__name__)

        configure_logger(args, source=SimpleProcessIdentity(component="rollout_executor"))

        self.args = args
        # set by the training actor after each weight update, keyed by trainer model id (None for one policy)
        self._weight_versions_of_model_id: dict[str | None, int] = {}
        self._rollouts_since_publish_of_model_id: dict[str | None, int] = defaultdict(int)
        self._train_parallel_configs_of_model_id: dict[str | None, dict[str, Any]] = {}
        self._router_providers = router_providers
        self._session_server_provider = session_server_provider
        self._inference_controller_provider = inference_controller_provider

    @init_once
    async def init(self) -> None:
        # [FUNCTION DOC] One-time setup (repeat calls are no-ops). Steps: unless train-only without
        # eval GPUs, resolve router addresses (recorded in args) and wait for session servers.
        # Initialize the tracking backend as a non-primary logger pointed at the router's http
        # address, the object store (not contributing a segment), and the shared http client.
        # Instantiate the data source from the configured class path with args. Rollout functions:
        # in the new style, if debug rollout data is being loaded both functions are none; else
        # construct the train rollout function from its path with a constructor input (args, data
        # source) and reuse the same instance for eval if the eval path equals the train path
        # (so stateful functions are not built twice), else construct the eval one separately. In
        # legacy style load both as plain functions. Load optional custom reward post-process and
        # sample conversion functions by path. Log the imports when a train function exists. Reset
        # last rollout id to minus one, create the eval lock, no eval fleet, and an optional metric
        # checker from args. KNOWN ISSUE: in legacy style with debug data loading, the rollout
        # functions are still loaded (only the new style skips them).
        args = self.args
        if not args.debug_train_only or args.eval_num_gpus > 0:
            await resolve_router_addrs(args, router_providers=self._router_providers)
            await wait_session_server_ready(args, provider=self._session_server_provider)

        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        object_store.init_instance(args, contribute_segment=False)

        init_http_client(args)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.use_legacy_rollout_v1 = use_legacy_rollout_v1()
        if not self.use_legacy_rollout_v1:
            if self.args.load_debug_rollout_data is not None:
                self.generate_rollout = None
                self.eval_generate_rollout = None
            else:
                input = RolloutFnConstructorInput(args=args, data_source=self.data_source)
                self.generate_rollout = load_rollout_function(input, self.args.rollout_function_path)
                if self.args.eval_function_path == self.args.rollout_function_path:
                    # Reuse the instance so train and eval share one state (and stateful
                    # rollout fns like FullyAsyncRolloutFn are not constructed twice).
                    self.eval_generate_rollout = self.generate_rollout
                else:
                    self.eval_generate_rollout = load_rollout_function(input, self.args.eval_function_path)
        else:
            self.generate_rollout = load_function(self.args.rollout_function_path)
            self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if (x := self.args.custom_reward_post_process_path) is not None:
            self.custom_reward_post_process_func = load_function(x)
        self.custom_convert_samples_to_train_data_func = None
        if (x := self.args.custom_convert_samples_to_train_data_path) is not None:
            self.custom_convert_samples_to_train_data_func = load_function(x)
        if self.generate_rollout is not None:
            logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
            logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        self.rollout_id = -1
        self._eval_lock = asyncio.Lock()
        self._eval_fleet: RolloutExecutorEvalFleet | None = None

        self._metric_checker = MetricChecker.maybe_create(args)

    async def get_init_state(self) -> str:
        # [FUNCTION DOC] Return the string value of the init-once guard's state (used by the
        # orchestrator to see whether init has not started, is running, or finished).
        return self._init_once.state.value

    # -------------------------- lifecycle -----------------------------

    async def dispose(self) -> None:
        # [FUNCTION DOC] Release resources: in the new style, dispose the train rollout function if it
        # exists (awaiting if asynchronous); close the data source if it has a close method; run the
        # event analysis configured in args; dispose the metric checker if any; and dispose the eval
        # rollout function if it is a checkpoint-eval function. Note: if the eval function is the
        # same object as the train function and is a checkpoint-eval function, it is disposed twice.
        if not self.use_legacy_rollout_v1 and self.generate_rollout is not None:
            await maybe_await(self.generate_rollout.dispose())
        if (close := getattr(self.data_source, "close", None)) is not None:
            close()
        event_analyzer.run_analysis_from_args(self.args)
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        if isinstance(self.eval_generate_rollout, CheckpointEvalFn):
            await maybe_await(self.eval_generate_rollout.dispose())

    # -------------------------- data generation -----------------------------

    @event_logger_context(lambda _self, rollout_id, trainer_model_id=None: dict(rollout_id=rollout_id))
    async def get(self, rollout_id: int, trainer_model_id: str | None = None) -> RolloutDataPack:
        # [FUNCTION DOC] Produce training data for one rollout step (runs inside an event-log context
        # tagged with the rollout id). Steps: record the start time and last rollout id; increment
        # the rollouts-since-publish counter for the trainer model and assert the weight version has
        # been published recently enough (see weight version helper); report data source buffer
        # length to the dashboard if the source exposes it; generate (or load) the rollout data
        # inside a timer named "rollout" (prefixed with the model id when given); optionally save
        # debug data; log rollout metrics (including elapsed wall time since start); convert samples
        # to a train data mapping with the custom hooks; read its sample indices; then either put
        # the whole mapping into the object store (when splitting by data-parallel rank is delayed)
        # or split it per data-parallel rank using the trainer's parallel config (key error if the
        # config was never set). Returns a rollout data pack with sample indices and a data
        # reference. Side effects: mutates counters, last rollout id, object store, tracking logs.
        start_time = time.time()
        self.rollout_id = rollout_id
        self._rollouts_since_publish_of_model_id[trainer_model_id] += 1
        assert_weight_version_is_published(
            self.args, rollouts_since_publish=self._rollouts_since_publish_of_model_id[trainer_model_id]
        )
        if (get_buffer_length := getattr(self.data_source, "get_buffer_length", None)) is not None:
            dashboard_hooks.report_data_buffer(get_buffer_length())
        with timer("rollout" if trainer_model_id is None else f"{trainer_model_id}/rollout"):
            data, metadata, metrics = await self._get_rollout_data(
                rollout_id=rollout_id, trainer_model_id=trainer_model_id
            )
        save_debug_rollout_data(
            self.args,
            data,
            rollout_id=rollout_id,
            evaluation=False,
            metadata=metadata,
            trainer_model_id=trainer_model_id,
        )
        log_rollout_data(
            rollout_id, self.args, data, metrics, time.time() - start_time, trainer_model_id=trainer_model_id
        )
        data = convert_samples_to_train_data(
            self.args,
            data,
            metadata=metadata,
            custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,
            custom_reward_post_process_func=self.custom_reward_post_process_func,
        )
        sample_indices = data.get("sample_indices")
        if self.args.delay_split_train_data_by_dp:
            data_ref = object_store.get_instance().put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        else:
            data_ref = split_train_data_by_dp(
                self.args, data, self._train_parallel_configs_of_model_id[trainer_model_id]
            )
        return RolloutDataPack(sample_indices=sample_indices, data_ref=data_ref)

    async def eval(
        self,
        rollout_id: int,
        hf_dir: str | None = None,
        export_time_seconds: float | None = None,
        require_marker: bool = True,
    ) -> None:
        # [FUNCTION DOC] Run one evaluation. Does nothing in debug train-only mode unless snapshot
        # evaluation is enabled. In snapshot mode delegates to the checkpoint evaluation (with the
        # snapshot directory, export time and marker requirement). Otherwise runs the eval rollout
        # function in a worker thread inside a timer named eval_rollout (new style with an eval
        # input, or legacy style with evaluation flag set), saves debug data, logs eval metrics
        # (including the function's extra metrics) and, if a metric checker exists, hands it the
        # logged metrics. The snapshot arguments are ignored on the non-snapshot path. Returns nothing.
        if self.args.debug_train_only and not self.args.eval_uses_snapshots:
            return

        if self.args.eval_uses_snapshots:
            return await self._eval_checkpoint(rollout_id, hf_dir, export_time_seconds, require_marker)

        with timer("eval_rollout"):
            if not self.use_legacy_rollout_v1:
                result = await asyncio.to_thread(
                    call_rollout_function, self.eval_generate_rollout, RolloutFnEvalInput(rollout_id=rollout_id)
                )
            else:
                result = await asyncio.to_thread(
                    call_rollout_fn,
                    self.eval_generate_rollout,
                    self.args,
                    rollout_id,
                    self.data_source,
                    evaluation=True,
                )
        data = result.data
        save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=True)
        metrics = log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    async def _eval_checkpoint(
        self, rollout_id: int, hf_dir: str | None, export_time_seconds: float | None, require_marker: bool
    ):
        # [FUNCTION DOC] HELPER: Evaluate an HF snapshot directory (must not be none: assertion error).
        # Under the eval lock: if a completion marker is required and the export is missing or
        # incomplete, warn and report a skip with reason "ckpt_missing". Otherwise use the rollout id
        # as the weight version text, pin the snapshot onto the eval fleet if one exists (yielding a
        # generation state; none without a fleet, i.e. external backend) and run the eval rollout
        # function in a thread with a checkpoint-eval input. An eval-skip exception from pin or
        # eval is converted to a skip report with its reason. On success saves debug data and logs
        # metrics: the function's metrics plus lag in steps (last rollout id minus this rollout id,
        # floored at zero), duration in seconds, and export time if provided; hands the logged
        # metrics to the metric checker if present. The skip report may itself raise in CI mode.
        """Evaluate a snapshot through the checkpoint eval fn (fleet or external
        backend) and log at ``rollout_id``. Every failure degrades to a skipped
        point; the lock serializes pins against a single backend."""
        assert hf_dir is not None, "checkpoint eval requires an HF snapshot dir"
        start_time = time.time()
        async with self._eval_lock:
            if require_marker and not is_complete_hf_export(hf_dir):
                logger.warning(f"Eval snapshot {hf_dir} missing or incomplete, skipping eval {rollout_id}")
                return self.report_eval_skip(rollout_id, "ckpt_missing")

            version = str(rollout_id)
            try:
                state = await self._eval_fleet.pin(hf_dir, version) if self._eval_fleet else None
                eval_input = RolloutFnEvalInput(
                    rollout_id=rollout_id, weight_version=version, hf_dir=hf_dir, generate_state=state
                )
                result = await asyncio.to_thread(call_rollout_function, self.eval_generate_rollout, eval_input)
            except EvalSkip as e:
                return self.report_eval_skip(rollout_id, e.reason)

            data = result.data
            save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=True)
            extra_metrics = dict(result.metrics or {})
            extra_metrics["eval/lag_steps"] = max(self.rollout_id - rollout_id, 0)
            extra_metrics["eval/duration_seconds"] = time.time() - start_time
            if export_time_seconds is not None:
                extra_metrics["eval/export_time_seconds"] = export_time_seconds
            metrics = log_eval_rollout_data(rollout_id, self.args, data, extra_metrics)
            if self._metric_checker is not None:
                self._metric_checker.on_eval(metrics)

    def report_eval_skip(self, rollout_id: int, reason: str) -> None:
        # [FUNCTION DOC] Log a skipped eval point (warning and tracking counter). In CI test mode
        # additionally raises a runtime error so any skipped point fails the run.
        log_eval_skip(rollout_id, self.args, reason)
        if self.args.ci_test:
            raise RuntimeError(f"CI eval {rollout_id} skipped: {reason}")

    async def _get_rollout_data(self, rollout_id, trainer_model_id: str | None = None):
        # [FUNCTION DOC] HELPER: Obtain (samples, metadata, metrics) for a training rollout. If
        # debug rollout data loading is configured, loads it from disk (metrics none). Otherwise runs
        # the train rollout function in a worker thread (new style with a train input carrying the
        # rollout id, the model's latest weight version and the trainer model id; legacy style with
        # evaluation off), takes its metrics and samples, post-processes them (flatten and trim,
        # using the trainer's parallel config) and asserts sample weight versions are sane. If CI
        # data injection applies to this rollout, replaces the generated data and metadata with the
        # recorded ones after asserting they match the generated ones, and clears metrics.
        if self.args.load_debug_rollout_data is not None:
            data, metadata = load_debug_rollout_data(self.args, rollout_id=rollout_id)
            metrics = None
        else:
            if not self.use_legacy_rollout_v1:
                data = await asyncio.to_thread(
                    call_rollout_function,
                    self.generate_rollout,
                    RolloutFnTrainInput(
                        rollout_id=rollout_id,
                        weight_version=self._weight_versions_of_model_id.get(trainer_model_id),
                        trainer_model_id=trainer_model_id,
                    ),
                )
            else:
                data = await asyncio.to_thread(
                    call_rollout_fn, self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False
                )
            metrics = data.metrics
            data = data.samples
            data, metadata = postprocess_rollout_data(
                self.args, data, train_parallel_config=self._train_parallel_configs_of_model_id[trainer_model_id]
            )
            assert_samples_weight_version_sane(self.args, samples=data)
            if RolloutDataInjectionUtil.should_inject(self.args, rollout_id):
                generated_data = data
                data, metadata = RolloutDataInjectionUtil.load(self.args, rollout_id=rollout_id)
                RolloutDataInjectionUtil.assert_matches_generated(
                    self.args, generated=generated_data, injected=data, rollout_id=rollout_id
                )
                metrics = None

        return data, metadata, metrics

    # -------------------------- checkpointing -----------------------------

    # TODO the train and eval rollout functions will become one object, so one save/load is enough here
    def save(self, rollout_id: int) -> None:
        # [FUNCTION DOC] Persist rollout-side state for a rollout id: the data source; in the new style
        # the train rollout function (if any) and the eval rollout function (if any and a different
        # object); then take an event logger checkpoint snapshot. Legacy style saves only the data
        # source.
        self.data_source.save(rollout_id)
        if not self.use_legacy_rollout_v1:
            if self.generate_rollout is not None:
                self.generate_rollout.save(rollout_id)
            if (eval_fn := self.eval_generate_rollout) is not None and eval_fn is not self.generate_rollout:
                eval_fn.save(rollout_id)
        event_logger_checkpoint.snapshot(self.args, rollout_id)

    def load(self, rollout_id: int | None = None) -> None:
        # [FUNCTION DOC] Restore rollout-side state for a rollout id (none means latest/default as
        # defined by each component): the data source, and in the new style the train rollout
        # function and a distinct eval rollout function when they exist. Mirrors save.
        self.data_source.load(rollout_id)
        if not self.use_legacy_rollout_v1:
            if self.generate_rollout is not None:
                self.generate_rollout.load(rollout_id)
            if (eval_fn := self.eval_generate_rollout) is not None and eval_fn is not self.generate_rollout:
                eval_fn.load(rollout_id)

    # -------------------------- misc APIs -----------------------------

    def get_num_rollout_per_epoch(self) -> int:
        # [FUNCTION DOC] Number of rollout steps in one pass over the dataset: dataset length divided
        # (integer division) by the rollout batch size. Asserts a global dataset is used.
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def set_weight_version(self, weight_version: int, trainer_model_id: str | None = None) -> None:
        # [FUNCTION DOC] Record the newly published weight version for a trainer model and reset its
        # rollouts-since-publish counter to zero. If the version is lower than the previous one, raises
        # assertion error unless independent data parallel fault tolerance is on, in which case it
        # only warns.
        # warning instead of assert when use indep_dp ft
        previous = self._weight_versions_of_model_id.get(trainer_model_id)
        if previous is not None and weight_version < previous:
            message = f"Engine weight version went backwards: {previous} -> {weight_version}"
            assert self.args.indep_dp, message
            logger.warning(message)
        self._weight_versions_of_model_id[trainer_model_id] = weight_version
        self._rollouts_since_publish_of_model_id[trainer_model_id] = 0

    def set_train_parallel_config(self, config: dict[str, Any], trainer_model_id: str | None = None) -> None:
        # [FUNCTION DOC] Store the trainer's parallel config mapping (e.g. data parallel size) for the
        # trainer model; later used to split and trim data.
        self._train_parallel_configs_of_model_id[trainer_model_id] = config

    async def set_eval_fleet_info(self, eval_fleet_info: EvalFleetInfo | None) -> None:
        # [FUNCTION DOC] Install or remove the eval fleet proxy: none clears it; otherwise builds a
        # new executor-side fleet from args, the fleet description and the inference controller
        # provider (which also builds a fresh generation state targeting the fleet's router).
        if eval_fleet_info is None:
            self._eval_fleet = None
            return

        self._eval_fleet = RolloutExecutorEvalFleet(
            self.args, info=eval_fleet_info, inference_controller_provider=self._inference_controller_provider
        )
