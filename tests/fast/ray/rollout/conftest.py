# [MODULE DOC]
# Shared pytest fixtures and builders for the rollout tests: a full argument namespace factory covering
# every field read by the ray rollout package (layered over the real parser defaults and the sglang
# router defaults), sample and grouped-sample factories, sglang config yaml rendering, tracked server
# cell disposal, a ray actor leak check, autouse isolation of the object store singleton, an autouse
# leak check for launched subprocesses, a dedent helper, and a mock engine factory for the address
# allocator.
from __future__ import annotations

import textwrap
from argparse import ArgumentParser, Namespace
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
import ray
from sglang_router.launch_router import RouterArgs

from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.utils import object_store
from miles.utils.types import Sample


def make_args(**overrides: Any) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds the args namespace. Layers, lowest to highest precedence: the real
    # parser defaults; an explicit defaults mapping (8 rollout gpus, 1 gpu per engine, no eval gpus,
    # small batch sizes, ray backends, megatron train backend, router and session server off, no
    # external rollout, no fault tolerance, debug and ci options off or empty, fake checkpoint path,
    # default rollout, eval and data source dotted paths, and so on); the sglang router cli defaults
    # (parsed with a prefix and no host or port); and the keyword overrides. The explicit mapping is
    # updated by the router defaults and then the overrides before being merged over the parser
    # defaults, so those two win over the parser. If starts-inference-engines is not provided it is set
    # to true unless debug train only is on with no eval gpus. When debug train only is on the rollout
    # gpu count is forced to 0, even over an explicit override.
    """Args namespace covering every field touched by ``miles/ray/rollout/``.
    Adding a new field is fine; deleting one likely breaks tests."""
    parser: ArgumentParser = ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    router_defaults: dict[str, Any] = vars(parser.parse_args([]))
    defaults: dict[str, Any] = dict(
        # rollout core
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        eval_uses_snapshots=False,
        num_gpus_per_node=8,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        n_samples_per_eval_prompt=4,
        rollout_max_response_len=512,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        use_sampling_support_replay=False,
        over_sampling_batch_size=None,
        rollout_global_dataset=False,
        num_rollout=1,
        update_weights_interval=1,
        check_weight_update_equal=False,
        check_weight_update_skip_list=None,
        # batch / training
        global_batch_size=8,
        use_dynamic_global_batch_size=False,
        wandb_always_use_train_step=False,
        disable_rollout_trim_samples=False,
        balance_data=False,
        delay_split_train_data_by_dp=False,
        # object store
        object_store_backend="ray",
        worker_comm_backend="ray",
        mooncake_store_init_kwargs=None,
        mooncake_replica_num=1,
        # advantage / reward
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        reward_key=None,
        log_reward_category=None,
        log_passrate=False,
        pin_rollout_manager_to_head=False,
        cluster_backend="ray",
        # placement / colocation
        debug_train_only=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        indep_dp=False,
        train_backend="megatron",
        kl_coef=0,
        use_kl_loss=False,
        use_opd=False,
        opd_type="megatron",
        train_env_vars={},
        dumper_source_patcher_config_train=None,
        offload_train=False,
        offload_train_target="cpu",
        offload_train_disk_dir="/tmp/offload",
        offload_train_disk_chunk_mb=64,
        critic_num_nodes=0,
        critic_num_gpus_per_node=0,
        use_critic=False,
        megatron_config=None,
        critic_train_only=False,
        # sglang router
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_router_policy=None,
        sglang_router_request_timeout_secs=600,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        sglang_api_key=None,
        multi_lora_n_adapters=0,
        target_modules=None,
        sglang_speculative_algorithm=None,
        sglang_config=None,
        sglang_model_routers=None,
        prefill_num_servers=None,
        # routers / session server
        use_miles_dashboard=False,
        use_miles_router=False,
        miles_router_timeout=None,
        miles_router_max_connections=None,
        miles_router_health_check_failure_threshold=3,
        sglang_server_concurrency=512,
        use_session_server=False,
        use_rollout_routing_replay=False,
        session_server_ip=None,
        session_server_external_host=None,
        session_server_port=None,
        session_server_workers=1,
        run_uuid="0123456789abcdef",
        # deployment
        deploy_component="all",
        deploy_instance_id=None,
        init_expected_num_cells=None,
        trainer_controller_addrs=None,
        inference_controller_addr=None,
        # external rollout
        rollout_external=False,
        rollout_external_engine_addrs=None,
        rollout_external_router_pd=False,
        custom_inference_engine_provider_path="miles.ray.specs.inference.backend_inference_engine_provider",
        # offload / fault tolerance
        offload_rollout=False,
        use_fault_tolerance=False,
        ft_components=[],
        rollout_health_check_interval=30.0,
        rollout_health_check_timeout=30.0,
        rollout_health_check_first_wait=0.0,
        rollout_health_check_failure_threshold=1,
        # engine launch command
        seed=42,
        fp16=False,
        use_rollout_indexer_replay=False,
        env_report=None,
        env_report_interval_seconds=3600.0,
        # checkpoint / data source
        hf_checkpoint="/fake/model",
        lora_rank=0,
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        eval_function_path="miles.rollout.sglang_rollout.eval_generate_rollout",
        data_source_path="miles.data.dummy.DummyDataSource",
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        custom_rollout_log_function_path=None,
        custom_eval_rollout_log_function_path=None,
        # debug data
        save_debug_rollout_data=None,
        save_debug_trajectory_data=None,
        load_debug_rollout_data=None,
        load_debug_rollout_data_subsample=None,
        ci_inject_rollout_data_path=None,
        ci_inject_rollout_data_start_rollout_id=None,
        ci_inject_rollout_data_min_match_ratio=0.9,
        # event checkpointing (event_logger.restore/snapshot in RolloutExecutor)
        save_debug_event_data=None,
        load=None,
        save=None,
        # CI
        ci_test=False,
        ci_tito_special_token_count_threshold=0.0,
        # dumper (sglang debug dumper integration)
        dumper_enable=False,
        dumper_inference=False,
    )
    defaults.update(router_defaults)
    defaults.update(overrides)
    defaults.setdefault("starts_inference_engines", not defaults["debug_train_only"] or defaults["eval_num_gpus"] > 0)
    if defaults["debug_train_only"]:
        defaults["rollout_num_gpus"] = 0
    return Namespace(**{**parser_defaults(), **defaults})


def make_sample(
    *,
    group_index: int = 0,
    index: int = 0,
    response_length: int = 4,
    reward: float | dict | None = 1.0,
    status: Sample.Status = Sample.Status.COMPLETED,
    **overrides: Any,
) -> Sample:
    # [FUNCTION DOC] HELPER: Builds a sample with prompt text "prompt", response "response", label
    # "label", tokens equal to the integers 0 to response length minus 1, the given group index, index,
    # response length, reward (default 1.0, may be a mapping or none) and status (default completed).
    # Any extra keyword arguments are then set as attributes on the sample (unvalidated).
    """Build a Sample with sensible defaults. Token list defaults to a length
    matching ``response_length`` so loss_mask/effective_response_length checks pass."""
    s = Sample(
        group_index=group_index,
        index=index,
        prompt="prompt",
        tokens=list(range(response_length)),
        response="response",
        response_length=response_length,
        label="label",
        reward=reward,
        status=status,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_samples_grouped(
    n_groups: int,
    group_size: int,
    *,
    rewards: list[float] | None = None,
    response_length: int = 4,
) -> list[Sample]:
    # [FUNCTION DOC] HELPER: Builds groups times group size samples, ordered group by group; sample
    # global index is group times size plus position; group index is the group. Reward is taken from the
    # rewards list by global index when given (its length must equal the total, else an assertion
    # error), otherwise it is the position within the group as a float (0.0, 1.0, ...). Each sample is
    # built by the single sample factory with the given response length.
    """Construct ``n_groups * group_size`` samples laid out group-by-group.

    If ``rewards`` is given, must have length n_groups*group_size."""
    total = n_groups * group_size
    if rewards is not None:
        assert len(rewards) == total, f"rewards must have length {total}, got {len(rewards)}"
    samples: list[Sample] = []
    for g in range(n_groups):
        for k in range(group_size):
            i = g * group_size + k
            r = rewards[i] if rewards is not None else float(k)
            samples.append(
                make_sample(
                    group_index=g,
                    index=i,
                    reward=r,
                    response_length=response_length,
                )
            )
    return samples


def make_sglang_config_yaml(
    *,
    name: str = "default",
    server_groups: list[dict] | None = None,
    update_weights: bool | None = None,
    model_path: str | None = None,
) -> str:
    # [FUNCTION DOC] HELPER: Renders yaml text with a single sglang model entry named as given (default
    # "default"), optional model path, optional update-weights flag rendered as lowercase true or false,
    # and server groups (default: one regular group of 8 gpus with 1 gpu per engine). Each group
    # writes its worker type and gpu count, and the gpus per engine only if present in the group
    # mapping. Returns the lines joined with newlines and a trailing newline. Missing worker type or
    # gpu count keys raise a key error.
    """Render a small SglangConfig YAML for from_file_arg() round-trip tests."""
    server_groups = server_groups or [{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
    lines = ["sglang:", f"  - name: {name}"]
    if model_path is not None:
        lines.append(f"    model_path: {model_path}")
    if update_weights is not None:
        lines.append(f"    update_weights: {str(update_weights).lower()}")
    lines.append("    server_groups:")
    for g in server_groups:
        lines.append(f"      - worker_type: {g['worker_type']}")
        lines.append(f"        num_gpus: {g['num_gpus']}")
        if "num_gpus_per_engine" in g:
            lines.append(f"        num_gpus_per_engine: {g['num_gpus_per_engine']}")
    return "\n".join(lines) + "\n"


def make_args_with_sglang_config(tmp_path, *, server_groups: list[dict] | None = None, **overrides: Any) -> Namespace:
    # [FUNCTION DOC] HELPER: Writes the rendered sglang config (with the given server groups and default
    # name) to a yaml file in the directory, and returns args built with that file path as the sglang
    # config plus the overrides. Side effect: writes the file.
    """Args namespace pointed at a freshly written sglang config file."""
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(make_sglang_config_yaml(server_groups=server_groups))
    return make_args(sglang_config=str(config_path), **overrides)


# --------------------------- server cell fixtures ---------------------------

_tracked_server_cells: list[Any] = []


def track_server_cell(cell: Any) -> Any:
    # [FUNCTION DOC] HELPER: Appends the cell to the module-level tracking list and returns it
    # unchanged. The disposal fixture awaits disposal of every tracked cell after the test.
    """Register a cell for teardown. ``ServerCell.__del__`` asserts that every cell was disposed."""
    _tracked_server_cells.append(cell)
    return cell


@pytest.fixture
async def dispose_tracked_server_cells() -> AsyncIterator[None]:
    # [FUNCTION DOC] HELPER: Async fixture. Clears the tracking list, yields, then awaits disposal of
    # each tracked cell in tracking order and clears the list. If a disposal raises, the remaining
    # cells are not disposed and the list is not cleared.
    """Dispose every cell registered through ``track_server_cell`` during the test."""
    _tracked_server_cells.clear()
    yield
    for cell in _tracked_server_cells:
        await cell.dispose()
    _tracked_server_cells.clear()


# --------------------------- ray fixtures ---------------------------


@pytest.fixture
def ray_actor_baseline(ray_local_mode):
    # [FUNCTION DOC] HELPER: Fixture depending on a local-mode ray fixture defined elsewhere. Counts
    # named ray actors before and after the test and asserts the count did not grow (message reports
    # both counts). Does not use the module-level ray import, importing it again locally.
    """Snapshot live ray actor count before / after a test; asserts no leak."""
    import ray

    def _count():
        # [FUNCTION DOC] HELPER: Returns the number of truthy names among the named actors listed by
        # ray, or 0 if listing raises any exception.
        try:
            return len([a for a in ray.util.list_named_actors() if a])
        except Exception:
            return 0

    before = _count()
    yield
    after = _count()
    assert after <= before, f"Ray actor leaked: before={before} after={after}"


@pytest.fixture(autouse=True)
def _autouse_reset_object_store(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture. Sets the object store module's process-wide instance
    # variable to none for the test (restored afterwards by the monkeypatch).
    """Isolate the process-wide object store singleton between tests."""
    monkeypatch.setattr(object_store, "_INSTANCE", None)


@pytest.fixture(autouse=True)
def _autouse_subprocess_leak_check(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture. Wraps the process utils bound-subprocess launcher so each
    # launched process is recorded, and records the pids of active multiprocessing children before the
    # test. After the test, computes leaked multiprocessing children (new pids) and launched processes
    # that are still running. If any leaked: terminates leaked multiprocessing children (terminate then
    # join up to 2 seconds, errors ignored; note it iterates all active children and filters by pid),
    # terminates the process tree of each leaked launched process, then raises an assertion error naming
    # the leaked pids. Does nothing when nothing leaked. Note: if the test body fails, the code after
    # the yield still runs under pytest fixture semantics.
    """Catch leaked router / session-server children (multiprocessing and Popen)."""
    import multiprocessing

    from miles.utils.workers import process_utils

    launched: list = []
    real_launch = process_utils.launch_bound_subprocess

    def _recording_launch(argv, *, envs):
        # [FUNCTION DOC] HELPER: Launches through the real launcher with the same argv and envs,
        # appends the process to the recorded list, and returns it.
        process = real_launch(argv, envs=envs)
        launched.append(process)
        return process

    monkeypatch.setattr(process_utils, "launch_bound_subprocess", _recording_launch)

    before = {p.pid for p in multiprocessing.active_children()}
    yield
    leaked_mp = {p.pid for p in multiprocessing.active_children()} - before
    leaked_popen = [p for p in launched if p.poll() is None]
    if leaked_mp or leaked_popen:
        # Tear down leaked children to avoid cascading test failures.
        for p in multiprocessing.active_children():
            if p.pid in leaked_mp:
                try:
                    p.terminate()
                    p.join(timeout=2)
                except Exception:
                    pass
        for p in leaked_popen:
            process_utils.terminate_process_tree(p)
        raise AssertionError(
            f"Subprocess leaked from previous test: mp={leaked_mp} popen={[p.pid for p in leaked_popen]}"
        )


def dedent(s: str) -> str:
    # [FUNCTION DOC] HELPER: Removes common leading whitespace from all lines and then strips leading
    # newlines (only at the start) from the result.
    return textwrap.dedent(s).lstrip("\n")


def fake_engine(host: str = "10.0.0.1", port_seed: int = 30000) -> MagicMock:
    # [FUNCTION DOC] HELPER: Returns a mock engine whose class marker is set to the ray actor handle
    # class (so type checks pass) and whose port cursor starts at the seed. Its free-port-block remote
    # call is an async function taking a start port (default 15000) and count (default 1): it picks the
    # larger of the cursor and the start port, advances the cursor past the block (port plus count) and
    # returns the port. Its node-ip remote call is an async function returning the host.
    """MagicMock that mimics the engine ``CommandActor`` enough for ``addr_allocator``.

    Mocks ``_get_free_port_block.remote(start_port, count)`` with a
    deterministic ``max(seq, start_port)`` counter so allocator tests can
    predict and assert on port assignment. ``_get_node_ip.remote()`` is the
    node-ip probe, which the cell awaits, so it returns an awaitable just like
    a real ``ObjectRef``. It also passes ``isinstance(x, ray.actor.ActorHandle)``
    so it can be handed to ``mark_allocated_uninitialized``."""
    e = MagicMock()
    e._spec_class = ray.actor.ActorHandle
    e._port_cursor = port_seed

    async def _alloc(start_port: int = 15000, count: int = 1):
        # [FUNCTION DOC] HELPER: Allocates a port block as described by the enclosing fake engine.
        port = max(e._port_cursor, start_port)
        e._port_cursor = port + count
        return port

    async def _probe():
        # [FUNCTION DOC] HELPER: Returns the fake engine's host address.
        return host

    e._get_free_port_block.remote.side_effect = _alloc
    e._get_node_ip.remote.side_effect = _probe
    return e
