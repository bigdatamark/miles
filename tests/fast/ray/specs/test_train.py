# [MODULE DOC]
# Unit tests for the trainer and trainer controller worker specs. Covers the set of specs per run
# (actor, optional critic, multi-policy via a megatron config), scheduling shape (cells, workers,
# pods, fractional gpu and cpu reservation, placement group slot offsets), constructor kwargs,
# ray concurrency groups, environment variables (user variables, fp8 scaling default, torch memory
# saver offload preload), port declarations, controller specs and their rendering into helm values,
# trainer configs, external controller address parsing and provider selection. CPU only; heavy
# modules such as the memory saver library are replaced by fakes.
import asyncio
import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.fixtures.megatron_config_fixtures import write_megatron_config, write_megatron_config_trainers
from tests.fast.ray.rollout.conftest import make_args_with_sglang_config

from miles.backends.megatron_utils.megatron_config import compute_trainer_args
from miles.ray.placement_group import _get_placement_group_layout
from miles.ray.specs import train as train_specs
from miles.ray.specs.train import (
    TRAINER_CONCURRENCY_GROUPS,
    TRAINER_CONTROLLER_WORKER_CLASS,
    _compute_trainer_controller_provider,
    compute_trainer_configs,
    compute_trainer_controller_pool_id,
    compute_trainer_ids,
    compute_trainer_pool_id,
    external_trainer_controller_addrs,
    specs_trainer,
    specs_trainer_controller,
    trainer_controller_cell_id,
    trainer_controller_worker_name,
)
from miles.ray.train_actor import TrainRayActor
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import SECTION_OF_CATEGORY, LaunchPlan
from miles.utils.workers.rpc.common.metadata import _find_rpc_config, declared_concurrency_groups
from miles.utils.workers.types import DeployComponent, PlatformAccess
from miles.utils.workers.worker_spec import WorkerCtorContext


def _make_args(**overrides) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds a trainer test argument object. Starts from a fixed set of
    # trainer related fields (1 node of 4 gpus for actor and critic, no critic, megatron backend,
    # no offload, grpo, default learning rate and so on), layers them over the parser defaults
    # (explicit fields win), then applies the given overrides as attributes. Returns the object.
    args = SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        critic_num_nodes=1,
        critic_num_gpus_per_node=4,
        use_critic=False,
        indep_dp=False,
        train_backend="megatron",
        multi_lora=False,
        use_fault_tolerance=False,
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
        megatron_config=None,
        trainer_model_id=None,
        advantage_estimator="grpo",
        lr=1e-6,
        optimizer="adam",
        use_distributed_optimizer=True,
        debug_disable_optimizer=False,
        save=None,
        load=None,
        megatron_to_hf_mode="core",
        ref_load=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        ref_ckpt_step=None,
        ckpt_step=None,
        start_rollout_id=None,
        lr_warmup_iters=None,
        eps_clip=0.2,
        disable_param_buffers_cpu_backup=False,
        critic_load=None,
        critic_save=None,
        critic_lr=None,
        critic_lr_warmup_iters=None,
    )
    args = SimpleNamespace(**{**parser_defaults(), **vars(args)})
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _make_context(**overrides) -> WorkerCtorContext:
    # [FUNCTION DOC] HELPER: Builds a worker constructor context for cell 0, worker 0 with gpu id
    # 0 and a fake backend capability; keyword overrides replace any of these fields.
    kwargs = dict(cell_index=0, worker_in_cell_index=0, gpu_ids=[0], capability=FakeBackendCapability())
    kwargs.update(overrides)
    return WorkerCtorContext(**kwargs)


def _install_fake_torch_memory_saver(monkeypatch, get_binary_path: MagicMock) -> MagicMock:
    # [FUNCTION DOC] HELPER: Installs fake torch memory saver modules (a package and its utils
    # module exposing the given function as the binary path resolver) into the module table, and
    # replaces the path byte reader so every library file appears to contain the disk backup
    # feature marker. Returns the given resolver mock. Side effects: patches the module table and
    # the path byte reader through the monkeypatch fixture (undone at test end).
    package = ModuleType("torch_memory_saver")
    package.__path__ = []
    utils = ModuleType("torch_memory_saver.utils")
    utils.get_binary_path_from_package = get_binary_path
    monkeypatch.setitem(sys.modules, "torch_memory_saver", package)
    monkeypatch.setitem(sys.modules, "torch_memory_saver.utils", utils)
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"TMS_INIT_ENABLE_DISK_BACKUP")
    return get_binary_path


def _unavailable_fp8_probe() -> str:
    # [FUNCTION DOC] HELPER: Stand-in for the fp8 hardware default probe that always raises a
    # runtime error, simulating hardware where the default cannot be computed.
    raise RuntimeError("the fp8 hardware default is unavailable here")


class TestSpecSet:
    # [CLASS DOC] Tests which trainer specs are produced and how critic arguments are derived.
    def test_only_the_actor_is_declared_without_a_critic(self):
        # [FUNCTION DOC] Without a critic, the only trainer spec is the actor pool.
        """Most runs have no critic, so no idle critic workers may be scheduled."""
        specs = specs_trainer(_make_args())

        assert [spec.name for spec in specs] == [compute_trainer_pool_id("actor")]

    def test_the_critic_gets_its_own_spec(self):
        # [FUNCTION DOC] With a critic, two specs are produced in order: actor pool, critic pool.
        """Actor and critic are separate worker sets even though they share GPUs."""
        specs = specs_trainer(_make_args(use_critic=True))

        assert [spec.name for spec in specs] == [
            compute_trainer_pool_id("actor"),
            compute_trainer_pool_id("critic"),
        ]

    def test_the_critic_args_are_neutralized(self):
        # [FUNCTION DOC] With kl coefficient 0.1, kl loss and on-policy distillation enabled, the
        # critic spec's constructor kwargs carry args with kl coefficient 0 and distillation off.
        """A critic must not apply the actor's KL or on-policy distillation settings."""
        specs = specs_trainer(_make_args(use_critic=True, kl_coef=0.1, use_kl_loss=True, use_opd=True))

        critic_args = specs[1].ctor_kwargs(_make_context())["args"]
        assert (critic_args.kl_coef, critic_args.use_opd) == (0, False)


class TestScheduling:
    # [CLASS DOC] Tests scheduling attributes of trainer specs: placement group names and slot
    # offsets, cells and workers per cell, pods per cell, and fractional resource reservations. Where
    # independent data parallelism splits the world, the megatron world size excluding data
    # parallelism is replaced by a stand-in.
    def test_actor_and_critic_share_one_placement_group(self):
        # [FUNCTION DOC] Both specs use the placement group named actor.
        """Shared actor/critic PPO puts both roles on the same GPUs."""
        specs = specs_trainer(_make_args(use_critic=True))

        assert {spec.scheduling.pg_name for spec in specs} == {"actor"}

    def test_one_worker_per_gpu_without_independent_dp(self):
        # [FUNCTION DOC] With 8 gpus per node on 1 node and no independent dp, there is 1 cell of 8
        # workers.
        """The trainer world is one rank per GPU in a single cell."""
        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=8))

        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 8)

    def test_independent_dp_splits_the_world_into_cells(self, monkeypatch):
        # [FUNCTION DOC] With a stand-in world size excluding dp of 2 and 8 gpus, independent dp
        # gives 4 cells of 2 workers.
        """Each independent-DP replica becomes one cell the manager can restart alone."""
        monkeypatch.setattr("miles.ray.specs.train.compute_megatron_world_size_except_dp", lambda _args: 2)

        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=8, indep_dp=True))

        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (4, 2)

    def test_independent_dp_critic_cells_use_the_critic_gpu_shape(self, monkeypatch):
        # [FUNCTION DOC] The actor has 3 nodes of 8 gpus and the critic 2 nodes of 4 gpus (8 gpus).
        # With the stand-in world size of 2, the critic spec has 4 cells of 2 workers, i.e. it is
        # split by the critic's own gpu count rather than the actor's.
        """A critic sized differently from the actor must be split by its own GPU count."""
        monkeypatch.setattr("miles.ray.specs.train.compute_megatron_world_size_except_dp", lambda _args: 2)

        _actor_spec, critic_spec = specs_trainer(
            _make_args(
                use_critic=True,
                indep_dp=True,
                actor_num_nodes=3,
                actor_num_gpus_per_node=8,
                critic_num_nodes=2,
                critic_num_gpus_per_node=4,
            )
        )

        assert (critic_spec.scheduling.num_cells, critic_spec.scheduling.num_workers_per_cell) == (4, 2)

    def test_a_nondivisible_independent_dp_trainer_layout_is_rejected(self, monkeypatch):
        # [FUNCTION DOC] 5 gpus cannot be divided into cells of 2; building the specs raises an
        # assertion error saying the count must be divisible.
        """A GPU count that cannot be split into equal cells must fail loudly instead of dropping ranks."""
        monkeypatch.setattr("miles.ray.specs.train.compute_megatron_world_size_except_dp", lambda _args: 2)

        with pytest.raises(AssertionError, match="must be divisible"):
            specs_trainer(_make_args(indep_dp=True, actor_num_nodes=1, actor_num_gpus_per_node=5))

    def test_a_cell_spanning_several_nodes_is_deployed_as_one_pod_per_node(self):
        # [FUNCTION DOC] With 2 nodes of 8 gpus and no independent dp, a cell has 2 pods and 8
        # workers per pod.
        """A trainer rank owns one gpu, so a two-node cell has to be two pods of a node's worth."""
        (spec,) = specs_trainer(_make_args(actor_num_nodes=2, actor_num_gpus_per_node=8))

        assert (spec.scheduling.pods_per_cell(), spec.scheduling.workers_per_pod()) == (2, 8)

    def test_each_role_packs_its_pods_with_its_own_per_node_count(self):
        # [FUNCTION DOC] Actor: 1 node of 8 gpus gives 1 pod of 8 workers. Critic: 2 nodes of 2 gpus
        # gives 2 pods of 2 workers each, i.e. the critic is not packed by the actor's node size.
        """A critic on smaller nodes must not inherit the actor's packing, or every name and port shifts."""
        args = _make_args(
            use_critic=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=2,
            critic_num_gpus_per_node=2,
        )

        actor_spec, critic_spec = specs_trainer(args)

        assert (actor_spec.scheduling.pods_per_cell(), actor_spec.scheduling.workers_per_pod()) == (1, 8)
        assert (critic_spec.scheduling.pods_per_cell(), critic_spec.scheduling.workers_per_pod()) == (2, 2)

    def test_a_cell_smaller_than_a_node_fits_into_one_pod(self, monkeypatch):
        # [FUNCTION DOC] With the stand-in world size of 2 on 8 gpus, each cell has 2 workers, 1 pod
        # per cell and 2 workers per pod.
        """An independent-DP cell of two ranks must not claim a pod bigger than the cell itself."""
        monkeypatch.setattr("miles.ray.specs.train.compute_megatron_world_size_except_dp", lambda _args: 2)

        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=8, indep_dp=True))

        assert spec.scheduling.num_workers_per_cell == 2
        assert (spec.scheduling.pods_per_cell(), spec.scheduling.workers_per_pod()) == (1, 2)

    def test_a_worker_reserves_a_fraction_of_its_gpu(self):
        # [FUNCTION DOC] With default args, a worker requests 0.4 of a gpu and occupies 1 gpu slot.
        """The rollout engine shares the same GPU slot, so the trainer must not claim it whole."""
        (spec,) = specs_trainer(_make_args())

        assert spec.scheduling.num_gpus_per_worker == 0.4
        assert spec.scheduling.num_gpu_slots_per_worker == 1

    def test_a_trainer_worker_reserves_matching_fractional_cpu_and_gpu_resources(self):
        # [FUNCTION DOC] A worker requests 0.4 cpus, and that equals its gpu fraction.
        """Claiming a whole CPU per worker would let Ray refuse to co-schedule the rollout engine."""
        (spec,) = specs_trainer(_make_args())

        assert spec.scheduling.num_cpus_per_worker == 0.4
        assert spec.scheduling.num_cpus_per_worker == spec.scheduling.num_gpus_per_worker

    def test_a_policy_parallelism_override_reshapes_only_its_own_cells(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] The world size stand-in returns the tensor parallel size of the args it is
        # given. Two policies share 8 gpus; policy a overrides tensor parallel size to 2 while b keeps
        # 1. Expected cell counts are 4 for a and 8 for b, since each trainer sizes by its own args.
        """The cell count is computed from that trainer's own args, so an override must reshape only its pool."""
        monkeypatch.setattr(
            "miles.ray.specs.train.compute_megatron_world_size_except_dp",
            lambda args: args.tensor_model_parallel_size,
        )
        args = _make_args(
            indep_dp=True,
            actor_num_gpus_per_node=8,
            tensor_model_parallel_size=1,
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "a", "overrides": {"tensor_model_parallel_size": 2}}, {"model_id": "b"}]
            ),
        )

        spec_a, spec_b = specs_trainer(args)

        assert (spec_a.scheduling.num_cells, spec_b.scheduling.num_cells) == (4, 8)

    def test_a_critic_is_sized_by_the_critic_node_counts(self):
        # [FUNCTION DOC] Actor with 8 gpus per node, critic with 1 node of 2 gpus: the critic spec
        # has 2 workers per cell and 2 gpus per node.
        """The critic reads critic_num_*, so sizing it by the actor's counts would reserve the wrong pool."""
        _actor_spec, critic_spec = specs_trainer(
            _make_args(use_critic=True, actor_num_gpus_per_node=8, critic_num_nodes=1, critic_num_gpus_per_node=2)
        )

        assert critic_spec.scheduling.num_workers_per_cell == 2
        assert critic_spec.scheduling.num_gpus_per_node == 2


class TestConstructorArguments:
    # [CLASS DOC] Tests the keyword arguments and worker class that a trainer spec supplies.
    def test_each_worker_learns_its_own_rank(self):
        # [FUNCTION DOC] For a 2-worker cell, constructor kwargs built for worker indices 0 and 1
        # carry ranks 0 and 1.
        """Ranks come from the spec now that no worker asks rank 0 for them."""
        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=2))

        ranks = [spec.ctor_kwargs(_make_context(worker_in_cell_index=i))["rank"] for i in range(2)]
        assert ranks == [0, 1]

    def test_the_world_size_is_the_cell_size(self):
        # [FUNCTION DOC] For a 4-worker cell, the constructor kwarg world size is 4.
        """A rank joins the process group of its own cell, not of the whole job."""
        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=4))

        assert spec.ctor_kwargs(_make_context())["world_size"] == 4

    def test_no_quorum_store_address_is_baked_into_the_spec(self, monkeypatch):
        # [FUNCTION DOC] With independent dp, the constructor kwargs contain no independent dp store
        # address entry (the store is minted later, not in the spec).
        """Every pod recomputes the spec, so an address minted here would give each pod its own quorum."""
        monkeypatch.setattr("miles.ray.specs.train.compute_megatron_world_size_except_dp", lambda _args: 2)

        (spec,) = specs_trainer(_make_args(actor_num_gpus_per_node=8, indep_dp=True))

        assert "indep_dp_store_addr" not in spec.ctor_kwargs(_make_context())

    @pytest.mark.parametrize(
        "backend,multi_lora,actor_class",
        [
            ("megatron", False, "miles.backends.megatron_utils.actor.MegatronTrainRayActor"),
            ("megatron", True, "miles.backends.megatron_utils.lora.actor.MultiLoRATrainRayActor"),
            ("fsdp", False, "miles.backends.fsdp_utils.actor.FSDPTrainRayActor"),
        ],
        ids=["megatron", "multi-lora", "fsdp"],
    )
    def test_the_backend_selects_the_worker_class(self, backend, multi_lora, actor_class):
        # [FUNCTION DOC] Parametrized over (megatron, plain), (megatron, multi lora) and (fsdp).
        # The actor spec's worker class path must equal the expected one for the combination, and
        # the critic spec's must equal the table entry for the backend alone.
        actor_spec, critic_spec = specs_trainer(
            _make_args(train_backend=backend, multi_lora=multi_lora, use_critic=True)
        )

        assert actor_spec.worker_class == actor_class
        assert critic_spec.worker_class == train_specs._TRAINER_ACTOR_CLASSES[backend]


class TestConcurrencyGroups:
    # [CLASS DOC] Tests the ray concurrency groups of trainer specs and their consistency with the
    # rpc annotations on the train actor class.
    def test_the_heartbeat_rpc_is_always_isolated(self):
        # [FUNCTION DOC] With fault tolerance on, the spec's concurrency groups are exactly
        # heartbeat_status, default, fault_injector and kill_self, each with a limit of 1.
        # KNOWN ISSUE: the test name says "always", but the groups exist only when fault tolerance
        # is enabled (the next test shows they are none otherwise).
        """A heartbeat queued behind a train step reads as a dead cell."""
        (spec,) = specs_trainer(_make_args(use_fault_tolerance=True))

        assert spec.concurrency_groups == {"heartbeat_status": 1, "default": 1, "fault_injector": 1, "kill_self": 1}

    def test_a_run_without_fault_tolerance_gets_a_plain_actor(self):
        # [FUNCTION DOC] Without fault tolerance, the spec's concurrency groups are none.
        """A threaded trainer actor runs NCCL setup off the main thread and deadlocked a non-FT run."""
        (spec,) = specs_trainer(_make_args())

        assert spec.concurrency_groups is None

    def test_the_actor_is_not_annotated_statically(self):
        # [FUNCTION DOC] For every method of the train actor class that declares a concurrency
        # group, the ray level concurrency group attribute must be absent. At least one such method
        # must exist so the check is not vacuous.
        """A static @ray.method(concurrency_group=...) makes Ray reject the plain non-FT actor."""
        annotations: list[str | None] = [
            getattr(getattr(TrainRayActor, name), "__ray_concurrency_group__", None)
            for name in declared_concurrency_groups(TrainRayActor)
        ]

        assert annotations and annotations == [None] * len(annotations)

    def test_both_trainer_roles_follow_the_same_gate(self):
        # [FUNCTION DOC] With a critic, both specs have concurrency groups when fault tolerance is
        # on and neither has them when it is off.
        """A critic threaded while its actor is not would deadlock exactly the run the gate protects."""
        fault_tolerant_specs = specs_trainer(_make_args(use_critic=True, use_fault_tolerance=True))
        plain_specs = specs_trainer(_make_args(use_critic=True))

        assert [spec.concurrency_groups is None for spec in fault_tolerant_specs] == [False, False]
        assert [spec.concurrency_groups is None for spec in plain_specs] == [True, True]

    def test_the_isolated_methods_are_annotated_on_the_actor(self):
        # [FUNCTION DOC] The declared groups of the heartbeat status getter, the fault injector and
        # the self kill method on the train actor class are heartbeat_status, fault_injector and
        # kill_self respectively.
        """Dropping an @rpc concurrency group would silently queue that call behind a train step."""
        declared = declared_concurrency_groups(TrainRayActor)

        assert {name: declared.get(name) for name in ("get_heartbeat_status", "inject_fault", "kill_self")} == {
            "get_heartbeat_status": "heartbeat_status",
            "inject_fault": "fault_injector",
            "kill_self": "kill_self",
        }

    def test_every_annotated_group_is_declared(self):
        # [FUNCTION DOC] The set of groups named by the actor's annotations is non-empty and a
        # subset of the trainer concurrency group declarations.
        """Ray rejects an actor whose method names a concurrency group the class never declares."""
        annotated_groups: set[str] = set(declared_concurrency_groups(TrainRayActor).values())

        assert annotated_groups
        assert annotated_groups <= set(TRAINER_CONCURRENCY_GROUPS)

    def test_the_same_methods_are_isolated_under_rpc_communication(self):
        # [FUNCTION DOC] Reading the rpc configuration of the three methods (heartbeat status,
        # fault injection, self kill) from the actor class gives the same three concurrency groups
        # as the ray level annotations.
        """The rpc server has its own group per method, and an unannotated one queues behind a train step."""
        groups = {
            name: _find_rpc_config(getattr(TrainRayActor, name)).concurrency_group
            for name in ("get_heartbeat_status", "inject_fault", "kill_self")
        }

        assert groups == {
            "get_heartbeat_status": "heartbeat_status",
            "inject_fault": "fault_injector",
            "kill_self": "kill_self",
        }


class TestEnvironmentVariables:
    # [CLASS DOC] Tests the per-worker environment variable mapping of a trainer spec: user
    # variables, the fp8 block scaling default captured at spec build time, and disk offload
    # backup variables.
    def test_user_env_vars_are_forwarded(self):
        # [FUNCTION DOC] A user supplied training variable appears with its value in the rendered
        # environment.
        """--train-env-vars must reach the worker process."""
        (spec,) = specs_trainer(_make_args(train_env_vars={"MY_VAR": "1"}))

        assert spec.env_var(_make_context())["MY_VAR"] == "1"

    def test_user_train_env_vars_override_framework_defaults(self, monkeypatch):
        # [FUNCTION DOC] The process environment sets NCCL_CUMEM_ENABLE=0, NVSHMEM_DISABLE_NCCL=1 and
        # NVTE_FP8_BLOCK_SCALING_FP32_SCALES=0, while the user variables give the opposite values
        # (1, 0, 1). The rendered environment must carry the user values.
        """A user who overrides a framework default must win, otherwise the flag is unusable."""
        monkeypatch.setenv("NCCL_CUMEM_ENABLE", "0")
        monkeypatch.setenv("NVSHMEM_DISABLE_NCCL", "1")
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")

        (spec,) = specs_trainer(
            _make_args(
                train_env_vars={
                    "NCCL_CUMEM_ENABLE": "1",
                    "NVSHMEM_DISABLE_NCCL": "0",
                    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
                }
            )
        )
        env_vars = spec.env_var(_make_context())

        assert (
            env_vars["NCCL_CUMEM_ENABLE"],
            env_vars["NVSHMEM_DISABLE_NCCL"],
            env_vars["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"],
        ) == ("1", "0", "1")

    def test_the_fp8_scaling_default_is_captured_when_the_spec_is_built(self, monkeypatch):
        # [FUNCTION DOC] With no process pin, the hardware default probe returns "0" while the spec
        # is built and "1" afterwards; the rendered fp8 scaling variable is "0", the build-time value.
        """The environment is rendered later inside the gpu-less worker manager, which would decide the wrong default."""
        monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "0")
        (spec,) = specs_trainer(_make_args())
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "1")

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "0"

    def test_an_explicit_fp8_scaling_env_is_still_forwarded(self, monkeypatch):
        # [FUNCTION DOC] A pinned process value "0" for the fp8 scaling variable appears unchanged
        # in the rendered environment.
        """An operator who pinned the value in the launcher environment must still reach the trainer."""
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")

        (spec,) = specs_trainer(_make_args())

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "0"

    def test_a_later_fp8_scaling_env_change_does_not_reach_the_trainer(self, monkeypatch):
        # [FUNCTION DOC] The spec is built with the variable unset and a probe default "0"; the
        # process environment then sets it to "1". The rendered value stays "0", i.e. the environment
        # read at render time is not consulted.
        """The environment is rendered in another process, so a value looked up at render time is the wrong one."""
        monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "0")

        (spec,) = specs_trainer(_make_args())
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "0"

    def test_an_explicit_fp8_scaling_env_is_forwarded_without_probing_the_hardware(self, monkeypatch):
        # [FUNCTION DOC] With the variable pinned to "1" and a probe that raises, building the spec
        # succeeds (the probe is not called) and the rendered value is "1".
        """A pinned value must reach the trainer even where the hardware default cannot be computed at all."""
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", _unavailable_fp8_probe)

        (spec,) = specs_trainer(_make_args())

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "1"

    def test_an_empty_fp8_scaling_env_is_forwarded_verbatim(self, monkeypatch):
        # [FUNCTION DOC] A pinned empty string wins over a probe default of "1": the rendered value
        # is the empty string.
        """An empty pin disables fp32 scales, so substituting the hardware default would silently re-enable them."""
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "")
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "1")

        (spec,) = specs_trainer(_make_args())

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == ""

    def test_train_env_vars_override_the_captured_fp8_scaling_default(self, monkeypatch):
        # [FUNCTION DOC] With the variable unset, a probe default "0" and a user variable "1", the
        # rendered value is "1": the user entry is not shadowed by the captured default.
        """The captured default is written into the same dict and must not shadow a --train-env-vars entry."""
        monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "0")

        (spec,) = specs_trainer(_make_args(train_env_vars={"NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1"}))

        assert spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "1"

    def test_the_critic_spec_captures_the_fp8_scaling_default_as_well(self, monkeypatch):
        # [FUNCTION DOC] Building actor and critic specs with the probe returning "0" and then
        # changing it to "1": both specs render "0".
        """The critic trainer imports transformer engine like the actor, so its spec needs the same captured value."""
        monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "0")

        specs = specs_trainer(_make_args(use_critic=True))
        monkeypatch.setattr(train_specs, "default_fp8_block_scaling_fp32_scales", lambda: "1")

        assert [spec.env_var(_make_context())["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] for spec in specs] == ["0", "0"]

    def test_disk_offload_forwards_backend_flags_and_nondefault_chunk_size(self, monkeypatch):
        # [FUNCTION DOC] With the fake memory saver installed and disk offload targeted with chunk
        # size 128 MB, the rendered environment has the cpu backup flag "0", the disk backup flag
        # "1" and the disk chunk size "128".
        """The disk backend must be switched on in place of the cpu one and use the requested chunk size."""
        _install_fake_torch_memory_saver(monkeypatch, MagicMock(return_value=Path("/opt/tms.so")))
        args = _make_args(offload_train=True, offload_train_target="disk", offload_train_disk_chunk_mb=128)

        (spec,) = specs_trainer(args)
        env_vars = spec.env_var(_make_context())

        assert (
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"],
            env_vars["TMS_INIT_ENABLE_DISK_BACKUP"],
            env_vars["TMS_DISK_BACKUP_CHUNK_MB"],
        ) == ("0", "1", "128")

    def test_disk_offload_gets_a_directory_per_worker(self, monkeypatch):
        # [FUNCTION DOC] For workers 0 and 1 of cell 1, the disk backup directory is the configured
        # base (/tmp/offload) plus a name made of zero padded cell and rank indices, giving
        # cell00001_rank00000 and cell00001_rank00001.
        """Two ranks sharing one directory would overwrite each other's offloaded weights."""
        _install_fake_torch_memory_saver(monkeypatch, MagicMock(return_value=Path("/opt/tms.so")))
        args = _make_args(offload_train=True, offload_train_target="disk")

        (spec,) = specs_trainer(args)

        directories = [
            spec.env_var(_make_context(cell_index=1, worker_in_cell_index=i))["TMS_DISK_BACKUP_DIR"] for i in range(2)
        ]
        assert directories == ["/tmp/offload/cell00001_rank00000", "/tmp/offload/cell00001_rank00001"]

    def test_a_library_without_the_disk_backend_is_rejected(self, monkeypatch):
        # [FUNCTION DOC] When the fake library file contents lack the disk backup marker, rendering
        # the environment for disk offload raises an assertion error saying there is no disk backend.
        """Launching disk offload against a library that cannot write to disk would
        silently lose the offloaded weights."""
        _install_fake_torch_memory_saver(monkeypatch, MagicMock(return_value=Path("/opt/tms.so")))
        monkeypatch.setattr(Path, "read_bytes", lambda self: b"built without the disk backend")

        (spec,) = specs_trainer(_make_args(offload_train=True, offload_train_target="disk"))

        with pytest.raises(AssertionError, match="has no disk backend"):
            spec.env_var(_make_context())

    def test_no_disk_directory_without_disk_offload(self):
        # [FUNCTION DOC] With offload off, the rendered environment has no disk backup directory.
        """The cpu backup path must not be told to write to disk."""
        (spec,) = specs_trainer(_make_args(offload_train=False))

        assert "TMS_DISK_BACKUP_DIR" not in spec.env_var(_make_context())


class TestTorchMemorySaverPreload:
    # [CLASS DOC] Tests the preload environment (preload library path and enable flags) used for
    # cpu offload of megatron trainers, and its absence for fsdp.
    def test_the_preload_library_is_resolved_from_the_package(self, monkeypatch):
        # [FUNCTION DOC] With cpu offload and a fake package returning a known library path, the
        # resolver is called exactly once with the hook-mode preload library name; the environment
        # has LD_PRELOAD equal to that path, the memory saver enable flag "1" and the cpu backup flag "1".
        """The hook must be preloaded from the installed package, not a hardcoded path."""
        expected_path = Path("/opt/torch_memory_saver_hook_mode_preload_cu13.abi3.so")
        get_binary_path = _install_fake_torch_memory_saver(monkeypatch, MagicMock(return_value=expected_path))

        (spec,) = specs_trainer(_make_args(offload_train=True, offload_train_target="cpu"))
        env_vars = spec.env_var(_make_context())

        get_binary_path.assert_called_once_with("torch_memory_saver_hook_mode_preload")
        assert env_vars["LD_PRELOAD"] == str(expected_path)
        assert env_vars["TMS_INIT_ENABLE"] == "1"
        assert env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"

    def test_fsdp_offload_does_not_enable_the_megatron_preload(self, monkeypatch):
        # [FUNCTION DOC] The module import hook is replaced so that importing anything from the
        # memory saver package raises an assertion error. With the fsdp backend and cpu offload,
        # building the spec and rendering the environment must succeed without LD_PRELOAD or the
        # memory saver enable flag.
        """fsdp has its own offload implementation, so preloading the hook only breaks its allocator."""
        original_import = builtins.__import__

        def reject_torch_memory_saver_import(
            name: str,
            globals_: dict[str, object] | None = None,
            locals_: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> ModuleType:
            # [FUNCTION DOC] HELPER: Replacement import hook. Raises an assertion error when the
            # top level package of the requested name is the memory saver; otherwise delegates to
            # the original import function with the same arguments and returns its module.
            if name.partition(".")[0] == "torch_memory_saver":
                raise AssertionError("FSDP offload must not import torch_memory_saver")
            return original_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", reject_torch_memory_saver_import)

        (spec,) = specs_trainer(_make_args(train_backend="fsdp", offload_train=True, offload_train_target="cpu"))
        env_vars = spec.env_var(_make_context())

        assert "LD_PRELOAD" not in env_vars
        assert "TMS_INIT_ENABLE" not in env_vars

    def test_a_missing_preload_library_is_not_swallowed(self, monkeypatch):
        # [FUNCTION DOC] If the library path resolver raises a runtime error, rendering the
        # environment propagates that error (it is not caught and skipped).
        """Silently launching without the hook would make offload corrupt weights."""
        _install_fake_torch_memory_saver(monkeypatch, MagicMock(side_effect=RuntimeError("missing preload library")))

        (spec,) = specs_trainer(_make_args(offload_train=True, offload_train_target="cpu"))

        with pytest.raises(RuntimeError, match="missing preload library"):
            spec.env_var(_make_context())


class TestPorts:
    # [CLASS DOC] Tests the port declarations of the trainer spec.
    def test_the_master_port_is_shared_across_the_cell(self):
        # [FUNCTION DOC] Exactly one port is named master; it is master mode and dynamically
        # allocatable.
        """All ranks of a cell rendezvous on one address, so it is a master port."""
        (spec,) = specs_trainer(_make_args())

        (master,) = [port for port in spec.port_infos if port.name == "master"]
        assert master.mode == "master"
        assert master.allow_dynamic is True


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_the_pool_name_encodes_the_role(role):
    # [FUNCTION DOC] Parametrized over actor and critic: the trainer pool id for a role is
    # trainer-engine- followed by the role name.
    """Spec names identify trainer cells apart from inference cells."""
    assert compute_trainer_pool_id(role) == f"trainer-engine-{role}"


class _FakeStaticProvider:
    # [CLASS DOC] HELPER: Fake static worker provider that records the worker names for which a
    # handle was requested and returns itself as the handle.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty list of requested handle names.
        self.handles: list[str] = []

    def get_handle(self, worker_name: str) -> object:
        # [FUNCTION DOC] HELPER: Appends the worker name to the record and returns the provider.
        self.handles.append(worker_name)
        return self


def _controller_layout() -> LaunchPlan:
    # [FUNCTION DOC] HELPER: Returns a fixed launch plan for rendering controller specs into helm
    # values (worker argv names the actor node count).
    return LaunchPlan(
        run_id="260101-000000-000",
        release="miles-run-260101",
        namespace="rl",
        state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
        orchestrator_command=["python", "/repo/train.py"],
        worker_argv=["--actor-num-nodes", "1"],
    )


def _controller_context(capability: FakeBackendCapability) -> WorkerCtorContext:
    # [FUNCTION DOC] HELPER: Builds a gpu-less constructor context for cell 0, worker 0 with the
    # given fake capability.
    return WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)


def _controller_providers() -> FakeBackendCapability:
    # [FUNCTION DOC] HELPER: Builds a fake capability with a plain object as cells provider, the
    # fake static provider, and a plain object as cell operations.
    return FakeBackendCapability(
        cells_provider=object(), static_provider=_FakeStaticProvider(), cell_operations=object()
    )


class TestSpecTrainerController:
    # [CLASS DOC] Tests the trainer controller specs: constructor kwargs, scheduling, worker class,
    # platform access, names, helm rendering and provider requests through a fake capability.
    def test_the_constructor_receives_its_deployment_identity_and_cell_operations(self) -> None:
        # [FUNCTION DOC] With run uuid, trainer deploy component and instance id policy-b in the
        # args, the first controller spec's constructor kwargs hold a deployment identity with those
        # three values (the component as its string value), and the cell operations object of the
        # capability itself.
        """The controller receives the exact deployment identity and backend operations for its trainer release."""
        operations = object()
        capability = FakeBackendCapability(cells_provider=object(), cell_operations=operations)
        args = _make_args(
            run_uuid="run-uuid",
            deploy_component=DeployComponent.TRAINER,
            deploy_instance_id="policy-b",
        )

        kwargs = specs_trainer_controller(args)[0].ctor_kwargs(_controller_context(capability))

        identity = kwargs["deployment_identity"]
        assert (identity.run_uuid, identity.deploy_component, identity.deploy_instance_id) == (
            "run-uuid",
            DeployComponent.TRAINER.value,
            "policy-b",
        )
        assert kwargs["cell_operations"] is operations

    def test_one_controller_per_trainer_role(self):
        # [FUNCTION DOC] Without a critic the first controller spec is named trainer-controller-actor;
        # with a critic the second one is named trainer-controller-critic.
        """Each controller owns exactly one trainer pool, so a critic run needs a second one."""
        assert specs_trainer_controller(_make_args())[0].name == "trainer-controller-actor"
        assert specs_trainer_controller(_make_args(use_critic=True))[1].name == "trainer-controller-critic"

    def test_it_is_a_gpuless_worker_on_both_backends(self):
        # [FUNCTION DOC] The controller spec has 1 cell of 1 worker and zero gpus and zero gpu
        # slots per worker.
        """A gpu request would reserve a whole trainer slot for a process that only sends rpcs."""
        spec = specs_trainer_controller(_make_args())[0]

        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 1)
        assert spec.scheduling.num_gpus_per_worker == 0
        assert spec.scheduling.num_gpu_slots_per_worker == 0

    def test_the_worker_class_is_the_controller_itself(self):
        # [FUNCTION DOC] The controller spec's worker class equals the trainer controller worker
        # class constant.
        """The spec names the class a pod or actor constructs, so it must be the real implementation."""
        spec = specs_trainer_controller(_make_args())[0]

        assert spec.worker_class == TRAINER_CONTROLLER_WORKER_CLASS

    def test_it_keeps_the_platform_delete_capability_used_to_suspend_cells(self):
        # [FUNCTION DOC] The controller spec's platform access is the read-and-delete level.
        """Fault injection deletes trainer pods, so moving readers off the orchestrator account must not break it."""
        spec = specs_trainer_controller(_make_args())[0]

        assert spec.platform_access is PlatformAccess.READ_DELETE

    def test_the_worker_and_cell_names_are_stable(self):
        # [FUNCTION DOC] For the actor role, the controller worker name is
        # trainer-controller-actor-00000-00000 and the cell id is trainer-controller-actor-00000.
        """The driver looks the controller up by name, so these names are part of the release's contract."""
        assert trainer_controller_worker_name("actor") == "trainer-controller-actor-00000-00000"
        assert trainer_controller_cell_id("actor") == "trainer-controller-actor-00000"

    def test_it_renders_into_static_workers_with_its_rpc_port(self):
        # [FUNCTION DOC] Renders the actor controller spec into helm values. Exactly one static
        # worker entry results, in the static workers section; it is named trainer-controller-actor,
        # exposes an rpc port 8000, passes its pool id on the command line, uses the controller
        # worker class and requests no resources.
        """The release has to contain the controller pod, or the address book would point at nothing."""
        spec = specs_trainer_controller(_make_args())[0]

        values = build_values([spec], _controller_layout()).as_values()

        (entry,) = values["run"]["staticWorkers"]
        assert SECTION_OF_CATEGORY[spec.category] == "staticWorkers"
        assert entry["name"] == "trainer-controller-actor"
        assert entry["ports"] == [{"name": "rpc", "port": 8000}]
        assert "--pool-id" in entry["command"]
        assert entry["command"][entry["command"].index("--pool-id") + 1] == "trainer-controller-actor"
        assert spec.worker_class == TRAINER_CONTROLLER_WORKER_CLASS
        assert "resources" not in entry

    def test_it_asks_for_a_provider_over_its_own_trainer_pool(self):
        # [FUNCTION DOC] With a critic, building constructor kwargs for both controllers makes the
        # capability receive provider requests for the actor pool and then the critic pool only,
        # and each kwargs mapping holds the capability's cells provider as its cell provider.
        """A controller that watched both pools would try to heal the other role's cells."""
        capability = _controller_providers()

        args = _make_args(use_critic=True)
        specs = specs_trainer_controller(args)
        kwargs = [spec.ctor_kwargs(_controller_context(capability)) for spec in specs]

        assert capability.requested_pool_ids == [["trainer-engine-actor"], ["trainer-engine-critic"]]
        assert [entry["cell_provider"] for entry in kwargs] == [capability.cells_provider] * 2

    def test_no_controller_is_built_with_a_handle_on_the_inference_controller(self):
        # [FUNCTION DOC] Building kwargs for both controllers requests no static provider and puts
        # no inference controller entry in any kwargs mapping.
        """The orchestration script drives the update window, so a trainer never reaches the engines itself."""
        capability = _controller_providers()

        args = _make_args(use_critic=True)
        kwargs = [spec.ctor_kwargs(_controller_context(capability)) for spec in specs_trainer_controller(args)]

        assert capability.requested_static_pool_ids == []
        assert all("inference_controller" not in entry for entry in kwargs)

    def test_the_run_shape_flags_are_resolved_by_the_spec(self):
        # [FUNCTION DOC] With kl coefficient 0.1 and on-policy distillation of the megatron type, the
        # actor controller kwargs carry trainer id actor, with-reference true and with-distillation
        # teacher true.
        """These are functions of args, so the worker can answer them from the argv it parses itself."""
        capability = _controller_providers()

        spec = specs_trainer_controller(_make_args(kl_coef=0.1, use_opd=True, opd_type="megatron"))[0]
        kwargs = spec.ctor_kwargs(_controller_context(capability))

        assert (kwargs["trainer_id"], kwargs["with_ref"], kwargs["with_opd_teacher"]) == ("actor", True, True)

    def test_a_policy_that_switches_off_its_kl_loss_gets_no_reference_cells(self, tmp_path):
        # [FUNCTION DOC] Global kl loss is on, but the megatron config makes policy a override it to
        # off. Policy a's controller kwargs have with-reference false, policy b's have it true.
        """with_ref is read off that trainer's own args, so one policy may need reference cells while another does not."""
        args = _make_args(
            use_kl_loss=True,
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "a", "overrides": {"use_kl_loss": False}}, {"model_id": "b"}]
            ),
        )

        spec_a, spec_b = specs_trainer_controller(args)

        assert spec_a.ctor_kwargs(_controller_context(_controller_providers()))["with_ref"] is False
        assert spec_b.ctor_kwargs(_controller_context(_controller_providers()))["with_ref"] is True

    def test_the_critic_controller_gets_no_reference_or_teacher_cells(self):
        # [FUNCTION DOC] With kl and distillation on, the critic controller kwargs have both
        # with-reference and with-distillation teacher false.
        """A critic controller must not hand its cells the actor's KL and OPD settings."""
        spec = specs_trainer_controller(_make_args(use_critic=True, kl_coef=0.1, use_kl_loss=True, use_opd=True))[1]
        critic_kwargs = spec.ctor_kwargs(_controller_context(_controller_providers()))

        assert (critic_kwargs["with_ref"], critic_kwargs["with_opd_teacher"]) == (False, False)

    def test_no_args_are_frozen_into_the_controller_at_spec_time(self):
        # [FUNCTION DOC] The actor controller kwargs contain no args entry.
        """The spec is built before the driver finishes deriving args, so a captured copy would be stale."""
        actor_kwargs = specs_trainer_controller(_make_args())[0].ctor_kwargs(
            _controller_context(_controller_providers())
        )

        assert "args" not in actor_kwargs

    def test_the_controller_pool_name_encodes_the_role(self):
        # [FUNCTION DOC] The controller pool id for the critic role is trainer-controller-critic.
        """The two controllers of a critic run must not collide in the address book."""
        assert compute_trainer_controller_pool_id("critic") == "trainer-controller-critic"


class TestTrainerConfigs:
    # [CLASS DOC] Tests the per-trainer configurations (trainer id, model id, role) computed from
    # args and an optional megatron config file, and how they shape specs, slot offsets, controllers
    # and worker arguments for single and multi policy runs.
    def test_a_single_policy_run_names_its_trainers_actor_and_critic(self):
        # [FUNCTION DOC] With a critic and no megatron config, the configs are ordered actor then
        # critic, with trainer ids actor and critic, no model ids and roles actor and critic.
        """Every existing pool name, worker name and checkpoint path is written against these two ids."""
        configs = compute_trainer_configs(_make_args(use_critic=True))

        assert [config.trainer_id for config in configs] == ["actor", "critic"]
        assert [config.model_id for config in configs] == [None, None]
        assert [config.role for config in configs] == ["actor", "critic"]

    def test_every_policy_of_a_multi_policy_run_gets_a_trainer_of_its_own(self, tmp_path):
        # [FUNCTION DOC] A megatron config file with models alpha and beta gives trainer ids
        # alpha-actor and beta-actor, model ids alpha and beta, and role actor for both.
        """One TrainerController per trainer id is what keeps two policies from sharing cells."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        configs = compute_trainer_configs(args)

        assert [config.trainer_id for config in configs] == ["alpha-actor", "beta-actor"]
        assert [config.model_id for config in configs] == ["alpha", "beta"]
        assert [config.role for config in configs] == ["actor", "actor"]

    def test_the_pool_ids_of_two_policies_do_not_collide(self, tmp_path):
        # [FUNCTION DOC] Two policies produce specs named trainer-engine-alpha-actor and
        # trainer-engine-beta-actor.
        """Pool ids become Kubernetes names, so a collision would put both policies in one pool."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        specs = specs_trainer(args)

        assert [spec.name for spec in specs] == ["trainer-engine-alpha-actor", "trainer-engine-beta-actor"]

    def test_each_policy_gets_its_own_slice_of_the_placement_group(self, tmp_path):
        # [FUNCTION DOC] With 4 gpus per trainer, two policies get placement group slot offsets 0 and 4.
        """Two policies sharing the same slots would be scheduled onto the same gpus."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        specs = specs_trainer(args)

        assert [spec.scheduling.pg_slot_offset for spec in specs] == [0, 4]

    def test_the_slice_of_each_policy_follows_the_trainer_size(self, tmp_path):
        # [FUNCTION DOC] With 2 nodes of 4 gpus (8 per trainer) and three policies, the slot offsets
        # are 0, 8 and 16 (stride equals one trainer's gpu count).
        """The stride is one trainer's gpu count; a hardcoded one overlaps as soon as a policy grows."""
        args = _make_args(
            megatron_config=write_megatron_config(tmp_path, "alpha", "beta", "gamma"),
            actor_num_nodes=2,
            actor_num_gpus_per_node=4,
        )

        specs = specs_trainer(args)

        assert [spec.scheduling.pg_slot_offset for spec in specs] == [0, 8, 16]

    def test_a_critic_shares_the_first_slice_with_its_actor(self):
        # [FUNCTION DOC] Without a megatron config but with a critic, actor and critic specs both
        # have slot offset 0 and are named trainer-engine-actor and trainer-engine-critic.
        """Actor and critic sit in one placement group on purpose; changing that must be deliberate."""
        specs = specs_trainer(_make_args(use_critic=True))

        assert [(spec.name, spec.scheduling.pg_slot_offset) for spec in specs] == [
            ("trainer-engine-actor", 0),
            ("trainer-engine-critic", 0),
        ]

    def test_every_slice_fits_inside_the_placement_group(self, tmp_path):
        # [FUNCTION DOC] Computes the trainer gpu count of the placement group layout for a
        # deployment with three policies (rollout 8 gpus, no critic, all-in-one deploy component).
        # For each trainer spec, the slot offset plus cells times gpus per cell must not exceed
        # that trainer gpu count.
        """A slot past the end of the group is a pending bundle nobody ever schedules."""
        megatron_config = write_megatron_config(tmp_path, "alpha", "beta", "gamma")
        args = _make_args(megatron_config=megatron_config, actor_num_nodes=2, actor_num_gpus_per_node=4)
        _, actor_num_gpus = _get_placement_group_layout(
            SimpleNamespace(
                actor_num_nodes=2,
                actor_num_gpus_per_node=4,
                rollout_num_gpus=8,
                eval_num_gpus=0,
                debug_train_only=False,
                debug_rollout_only=False,
                rollout_external=False,
                colocate=False,
                use_critic=False,
                megatron_config=megatron_config,
                deploy_component="all",
            )
        )

        specs = specs_trainer(args)

        for spec in specs:
            reserved = spec.scheduling.num_cells * spec.scheduling.gpus_per_cell()
            assert spec.scheduling.pg_slot_offset + reserved <= actor_num_gpus

    def test_each_policy_gets_a_controller_of_its_own(self, tmp_path):
        # [FUNCTION DOC] Two policies give two controller specs whose kwargs have trainer ids
        # alpha-actor and beta-actor and role actor.
        """A policy whose controller is another policy's would train the wrong ranks."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        specs = specs_trainer_controller(args)
        kwargs = [spec.ctor_kwargs(_controller_context(_controller_providers())) for spec in specs]

        assert [entry["trainer_id"] for entry in kwargs] == ["alpha-actor", "beta-actor"]
        assert [entry["role"] for entry in kwargs] == ["actor", "actor"]

    def test_a_worker_is_told_which_policy_it_serves_through_its_args(self, tmp_path):
        # [FUNCTION DOC] Worker kwargs of the two policy specs carry args whose trainer model id is
        # alpha and beta respectively, and role actor.
        """The worker namespaces its logs and metrics by this id; a shared one merges the two runs."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        kwargs = [spec.ctor_kwargs(_make_context()) for spec in specs_trainer(args)]

        assert [entry["args"].trainer_model_id for entry in kwargs] == ["alpha", "beta"]
        assert [entry["role"] for entry in kwargs] == ["actor", "actor"]

    def test_a_worker_of_every_policy_still_gets_the_plain_actor_role(self, tmp_path):
        # [FUNCTION DOC] For a run without a megatron config, the single spec's worker kwargs have
        # no trainer model id and role actor. For the two-policy config, the spec names are
        # trainer-engine-alpha-actor and trainer-engine-beta-actor.
        # KNOWN ISSUE: despite the name, the role of the policy workers is never asserted; only the
        # single-policy worker kwargs are checked for the plain actor role.
        """The worker layer branches on the role literal, so a policy specific role would break training."""
        args = _make_args(megatron_config=write_megatron_config(tmp_path, "alpha", "beta"))

        [kwargs] = [spec.ctor_kwargs(_make_context()) for spec in specs_trainer(_make_args())]

        assert (kwargs["args"].trainer_model_id, kwargs["role"]) == (None, "actor")
        assert [spec.name for spec in specs_trainer(args)] == [
            "trainer-engine-alpha-actor",
            "trainer-engine-beta-actor",
        ]

    def test_a_policy_overlays_its_own_megatron_args_onto_its_worker(self, tmp_path):
        # [FUNCTION DOC] Policy alpha overrides the learning rate to 5e-7 in the megatron config and
        # beta overrides nothing; the worker args have learning rate 5e-7 and the default 1e-6.
        """A per-policy --lr that never reaches the worker would train both policies identically."""
        args = _make_args(
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "alpha", "overrides": {"lr": 5e-7}}, {"model_id": "beta"}]
            ),
        )

        kwargs = [spec.ctor_kwargs(_make_context()) for spec in specs_trainer(args)]

        assert [entry["args"].lr for entry in kwargs] == [5e-7, 1e-6]

    def test_the_critic_of_a_configured_policy_is_a_trainer_of_that_policy(self, tmp_path):
        # [FUNCTION DOC] With a critic and a single configured policy alpha, the configs are
        # (alpha-actor, actor, alpha) and (alpha-critic, critic, alpha).
        """A critic left without a model id would be built from raw args, ignoring the policy overlay."""
        args = _make_args(
            use_critic=True,
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "alpha", "overrides": {"lr": 5e-7}}]
            ),
        )

        configs = compute_trainer_configs(args)

        assert [(config.trainer_id, config.role, config.model_id) for config in configs] == [
            ("alpha-actor", "actor", "alpha"),
            ("alpha-critic", "critic", "alpha"),
        ]

    def test_the_critic_args_stack_the_policy_overlay_and_the_critic_neutralization(self, tmp_path):
        # [FUNCTION DOC] The policy overrides eps clip to 0.3 and learning rate to 5e-7; the global
        # args have kl 0.1, distillation on and critic learning rate 2e-6. The critic trainer's
        # computed args have eps clip 0.3 (overlay), kl 0 and distillation off (neutralization)
        # and learning rate 2e-6 (critic value wins over the overlay).
        """The two transforms are not exclusive: the critic of a policy needs that policy's overlay too."""
        args = _make_args(
            use_critic=True,
            kl_coef=0.1,
            use_opd=True,
            critic_lr=2e-6,
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "alpha", "overrides": {"eps_clip": 0.3, "lr": 5e-7}}]
            ),
        )

        critic_args = compute_trainer_args(args, compute_trainer_configs(args)[1])

        assert (critic_args.eps_clip, critic_args.kl_coef, critic_args.use_opd) == (0.3, 0, False)
        assert critic_args.lr == 2e-6

    def test_the_critic_pool_of_a_configured_policy_is_named_after_its_trainer_id(self, tmp_path):
        # [FUNCTION DOC] With a critic and policy alpha, trainer specs are named
        # trainer-engine-alpha-actor and trainer-engine-alpha-critic, and controller specs
        # trainer-controller-alpha-actor and trainer-controller-alpha-critic.
        """The critic pool must not collide with the policy's own pool nor with a plain 'critic' pool."""
        args = _make_args(use_critic=True, megatron_config=write_megatron_config(tmp_path, "alpha"))

        assert [spec.name for spec in specs_trainer(args)] == [
            "trainer-engine-alpha-actor",
            "trainer-engine-alpha-critic",
        ]
        assert [spec.name for spec in specs_trainer_controller(args)] == [
            "trainer-controller-alpha-actor",
            "trainer-controller-alpha-critic",
        ]


_TRAINER_IDS = ["actor", "critic"]


def _addressed_args(tmp_path, **overrides):
    # [FUNCTION DOC] HELPER: Builds test arguments from an sglang config written into the
    # temporary directory (one regular group, 8 gpus, 4 per engine, 8 rollout gpus) plus overrides.
    return make_args_with_sglang_config(
        tmp_path,
        server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 4}],
        rollout_num_gpus=8,
        **overrides,
    )


class TestStaticTrainerControllerAddrs:
    # [CLASS DOC] Tests parsing and validation of the external trainer controller address list
    # (entries of the form trainer id equals host and port).
    def test_a_run_without_the_flag_names_no_controller(self, tmp_path):
        # [FUNCTION DOC] Without the address flag, the result is none.
        """An all-in-one run finds its own controller, so nothing may be invented for it."""
        assert external_trainer_controller_addrs(_addressed_args(tmp_path), trainer_ids=_TRAINER_IDS) is None

    def test_the_one_trainer_of_a_run_is_named_by_its_own_entry(self, tmp_path):
        # [FUNCTION DOC] One entry actor=10.0.0.1:8000 for the trainer id actor gives a mapping
        # whose actor entry has host 10.0.0.1 and port 8000.
        """The trainer id is what tells the entry apart from the next run's, so even one trainer writes it."""
        args = _addressed_args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000"])

        addrs = external_trainer_controller_addrs(args, trainer_ids=["actor"])

        assert (addrs["actor"].host, addrs["actor"].port) == ("10.0.0.1", 8000)

    def test_each_trainer_is_addressed_separately(self, tmp_path):
        # [FUNCTION DOC] Entries for actor and critic give separate host and port pairs
        # (10.0.0.1, 8000) and (10.0.0.2, 9000) per trainer id.
        """A critic is its own controller in its own pod, and calling the actor's would train the wrong model."""
        args = _addressed_args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000", "critic=10.0.0.2:9000"])

        addrs = external_trainer_controller_addrs(args, trainer_ids=_TRAINER_IDS)

        assert [(addrs[trainer_id].host, addrs[trainer_id].port) for trainer_id in _TRAINER_IDS] == [
            ("10.0.0.1", 8000),
            ("10.0.0.2", 9000),
        ]

    def test_refuses_an_entry_that_names_no_trainer(self, tmp_path):
        # [FUNCTION DOC] A bare entry 10.0.0.1:8000 without a trainer id raises an assertion error
        # mentioning the expected host-and-port format.
        """A bare address belongs to whichever trainer the reader guesses, and a run may drive several."""
        args = _addressed_args(tmp_path, trainer_controller_addrs=["10.0.0.1:8000"])

        with pytest.raises(AssertionError, match="host:port"):
            external_trainer_controller_addrs(args, trainer_ids=["actor"])

    def test_refuses_a_trainer_id_that_is_not_one_of_the_run_s(self, tmp_path):
        # [FUNCTION DOC] An entry with a misspelled trainer id (actro) while the run drives actor
        # raises an assertion error saying each trainer must be named exactly once.
        """A typo would otherwise leave the trainer it meant to name silently unaddressed."""
        args = _addressed_args(tmp_path, trainer_controller_addrs=["actro=10.0.0.1:8000"])

        with pytest.raises(AssertionError, match="exactly once"):
            external_trainer_controller_addrs(args, trainer_ids=["actor"])

    def test_refuses_a_run_whose_second_trainer_was_left_unaddressed(self, tmp_path):
        # [FUNCTION DOC] Only actor is addressed while the run drives actor and critic: assertion
        # error about naming each trainer exactly once.
        """A trainer named by nothing would be reached at the first one's controller and train the wrong model."""
        args = _addressed_args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000"])

        with pytest.raises(AssertionError, match="exactly once"):
            external_trainer_controller_addrs(args, trainer_ids=_TRAINER_IDS)

    def test_refuses_two_controllers_for_one_trainer_id(self, tmp_path):
        # [FUNCTION DOC] Two entries for the same trainer id actor raise the same exactly-once
        # assertion error.
        """One trainer id is one trainer, and silently using one of the two would drop the other."""
        args = _addressed_args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000", "actor=10.0.0.2:8000"])

        with pytest.raises(AssertionError, match="exactly once"):
            external_trainer_controller_addrs(args, trainer_ids=["actor"])

    def test_refuses_an_address_written_as_a_url(self, tmp_path):
        # [FUNCTION DOC] An address given as an http url instead of host and port raises an
        # assertion error mentioning the host-and-port format.
        """The flag takes host:port, and a pasted url would otherwise be read as a host named 'http'."""
        args = _addressed_args(tmp_path, trainer_controller_addrs=["actor=http://10.0.0.1:8000"])

        with pytest.raises(AssertionError, match="host:port"):
            external_trainer_controller_addrs(args, trainer_ids=["actor"])


class TestTrainerIds:
    # [CLASS DOC] Tests the list of trainer ids a run drives.
    def test_a_single_policy_run_drives_one_trainer(self, tmp_path):
        # [FUNCTION DOC] Without a critic, the trainer ids are just actor.
        """The flag keys on trainer ids, so the ids a run drives are exactly the entries it takes."""
        assert compute_trainer_ids(_addressed_args(tmp_path)) == ["actor"]

    def test_a_critic_run_drives_a_trainer_per_trainer_id(self, tmp_path):
        # [FUNCTION DOC] With a critic, the trainer ids are actor then critic.
        """A critic is deployed as its own trainer and so takes its own entry of the flag."""
        assert compute_trainer_ids(_addressed_args(tmp_path, use_critic=True)) == ["actor", "critic"]


class TestProviderSelection:
    # [CLASS DOC] Tests which provider is used to reach a trainer controller: a static provider
    # from a given address versus the backend capability's own.
    def test_a_given_trainer_controller_address_is_used_instead_of_the_backend_s(self, tmp_path):
        # [FUNCTION DOC] With address actor=10.0.0.1:8000, the returned provider answers the
        # address lookup for the controller worker name with an rpc endpoint at http://10.0.0.1:8000,
        # and the capability was not asked for any static pool provider.
        """The trainer lives in another deployment, whose names this one's backend cannot resolve."""
        args = _addressed_args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000"])
        capability = FakeBackendCapability(static_provider=object())

        provider = _compute_trainer_controller_provider(args, capability=capability, trainer_id="actor")

        addrs = asyncio.run(provider.get_addrs("trainer-controller-actor-00000-00000"))
        assert addrs["rpc"].addr == "http://10.0.0.1:8000"
        assert capability.requested_static_pool_ids == []

    def test_an_all_in_one_run_still_asks_its_own_backend_for_the_trainer_controller(self, tmp_path):
        # [FUNCTION DOC] Without addresses, the returned provider is the capability's static
        # provider and the requested static pool id list is just trainer-controller-actor.
        """Nothing addresses a ray actor statically, so the all-in-one path must be untouched."""
        capability = FakeBackendCapability(static_provider=object())

        provider = _compute_trainer_controller_provider(
            _addressed_args(tmp_path), capability=capability, trainer_id="actor"
        )

        assert provider is capability.static_provider
        assert capability.requested_static_pool_ids == ["trainer-controller-actor"]
