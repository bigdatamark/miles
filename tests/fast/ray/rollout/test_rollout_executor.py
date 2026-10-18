# [MODULE DOC]
# Unit tests for the rollout executor: disposal of train and eval rollout objects that dispose
# synchronously, and installing or clearing the eval fleet info, which switches checkpoint
# evaluation between pinned fleet routing and the default routing. Executors are built without
# their constructor and given only the attributes the method under test reads; the controller,
# its provider and the eval function are small fakes defined here.
import asyncio
from argparse import Namespace

import pytest

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput
from miles.rollout.inference_rollout import inference_rollout_common
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.workers.worker_spec import HostAndPort


class FakeInferenceController:
    # [CLASS DOC] HELPER: Fake inference controller that records eval-fleet pin requests as pairs
    # of checkpoint directory and weight version.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the empty list of recorded pins.
        self.pins: list[tuple[str, str]] = []

    async def pin_eval_fleet(self, *, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        # [FUNCTION DOC] HELPER: Records the checkpoint directory and weight version and returns a
        # pin result with no skip reason (pin succeeded).
        self.pins.append((checkpoint_dir, weight_version))
        return EvalFleetPin(skip_reason=None)


class FakeInferenceControllerProvider:
    # [CLASS DOC] HELPER: Fake provider that hands out one fixed fake controller for any worker name.
    def __init__(self, controller: FakeInferenceController) -> None:
        # [FUNCTION DOC] HELPER: Stores the controller to hand out.
        self.controller = controller

    def get_handle(self, worker_name: str) -> FakeInferenceController:
        # [FUNCTION DOC] HELPER: Ignores the worker name and returns the stored controller.
        return self.controller


class FakeEvalFunction:
    # [CLASS DOC] HELPER: Fake eval rollout function that records each eval input it is called
    # with and returns an eval output with empty data.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the empty list of recorded inputs.
        self.inputs: list[RolloutFnEvalInput] = []

    def __call__(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        # [FUNCTION DOC] HELPER: Appends the input to the recorded list and returns an eval output
        # with an empty data mapping.
        self.inputs.append(input)
        return RolloutFnEvalOutput(data={})


class _SynchronousDisposable:
    # [CLASS DOC] HELPER: Fake object whose dispose method is synchronous (returns nothing, not an
    # awaitable) and appends its name to a shared list; also usable as a replacement class for the
    # checkpoint eval function type.
    def __init__(self, disposed: list[str], name: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the shared list of disposed names and this object's name.
        self._disposed = disposed
        self._name = name

    def dispose(self) -> None:
        # [FUNCTION DOC] HELPER: Appends this object's name to the shared disposed list.
        self._disposed.append(self._name)


class TestDispose:
    # [CLASS DOC] Tests the executor's dispose method.
    async def test_synchronous_train_and_eval_rollout_disposers_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Builds an executor (no legacy rollout v1) whose train and eval rollout
        # objects have synchronous dispose methods, with no metric checker. Patches the checkpoint
        # eval function type check to accept the fake and the event analysis to a no-op. After
        # dispose, asserts the train object was disposed first, then the eval object.
        """Out-of-tree rollout hooks may tear down synchronously without returning an awaitable."""
        disposed: list[str] = []
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.use_legacy_rollout_v1 = False
        executor.generate_rollout = _SynchronousDisposable(disposed, "train")
        executor.eval_generate_rollout = _SynchronousDisposable(disposed, "eval")
        executor.data_source = object()
        executor.args = Namespace()
        executor._metric_checker = None
        monkeypatch.setattr(rollout_executor_module, "CheckpointEvalFn", _SynchronousDisposable)
        monkeypatch.setattr(rollout_executor_module.event_analyzer, "run_analysis_from_args", lambda _args: None)

        await executor.dispose()

        assert disposed == ["train", "eval"]


class TestSetEvalFleetInfo:
    # [CLASS DOC] Tests the executor's set-eval-fleet-info method and its effect on checkpoint
    # evaluation.
    async def test_setting_and_clearing_eval_fleet_info_changes_checkpoint_evaluation_routing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Stubs tokenizer and processor loading. Builds an executor with a minimal
        # arguments namespace, the fake controller provider, no eval fleet, an eval lock, the fake
        # eval function, rollout id 9 and no metric checker. Installs a fleet info (router
        # 10.0.0.2:31000, 2 GPUs, 1 GPU per engine), evaluates checkpoint 5, clears the fleet info
        # and evaluates checkpoint 6. Asserts: the controller was asked to pin only checkpoint 5
        # with weight version "5"; the first eval input has a generate state whose arguments carry
        # the fleet's router host and port, GPU count and GPUs per engine; the second eval input
        # has no generate state (unpinned default routing).
        """Installing a fleet pins checkpoint evals, while clearing it restores unpinned routing."""
        monkeypatch.setattr(inference_rollout_common, "load_tokenizer", lambda *args, **kwargs: object())
        monkeypatch.setattr(inference_rollout_common, "load_processor", lambda *args, **kwargs: object())
        controller = FakeInferenceController()
        provider = FakeInferenceControllerProvider(controller)
        eval_function = FakeEvalFunction()
        executor = RolloutExecutor.__new__(RolloutExecutor)
        executor.args = Namespace(
            chat_template_path=None,
            custom_eval_rollout_log_function_path=None,
            custom_generate_function_path=None,
            global_batch_size=1,
            hf_checkpoint="unused",
            log_passrate=False,
            n_samples_per_prompt=1,
            rollout_batch_size=1,
            rollout_max_response_len=16,
            rollout_num_gpus=1,
            rollout_num_gpus_per_engine=1,
            rollout_skip_special_tokens=True,
            rollout_stop=None,
            rollout_stop_token_ids=None,
            rollout_temperature=1.0,
            rollout_top_k=-1,
            rollout_top_p=1.0,
            save_debug_rollout_data=None,
            sglang_server_concurrency=2,
            wandb_always_use_train_step=False,
        )
        executor._inference_controller_provider = provider
        executor._eval_fleet = None
        executor._eval_lock = asyncio.Lock()
        executor.eval_generate_rollout = eval_function
        executor.rollout_id = 9
        executor._metric_checker = None
        info = EvalFleetInfo(
            router=HostAndPort(host="10.0.0.2", port=31000),
            num_gpus=2,
            num_gpus_per_engine=1,
        )

        await executor.set_eval_fleet_info(info)
        await executor._eval_checkpoint(
            rollout_id=5,
            hf_dir="/snap/step_5",
            export_time_seconds=None,
            require_marker=False,
        )
        await executor.set_eval_fleet_info(None)
        await executor._eval_checkpoint(
            rollout_id=6,
            hf_dir="/snap/step_6",
            export_time_seconds=None,
            require_marker=False,
        )

        assert controller.pins == [("/snap/step_5", "5")]
        first, second = eval_function.inputs
        assert isinstance(first.generate_state, GenerateState)
        assert first.generate_state.args.sglang_router_ip == info.router.host
        assert first.generate_state.args.sglang_router_port == info.router.port
        assert first.generate_state.args.rollout_num_gpus == info.num_gpus
        assert first.generate_state.args.rollout_num_gpus_per_engine == info.num_gpus_per_engine
        assert second.generate_state is None
