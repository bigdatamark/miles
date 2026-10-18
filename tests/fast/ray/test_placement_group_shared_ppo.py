# [MODULE DOC]
# Unit tests for the placement group layout calculation and the training model creation of the
# placement group module. The layout function returns a pair: the total number of gpu bundles to
# reserve, and the offset at which the rollout (inference) slice begins. The tests cover shared
# actor/critic bundles, colocation, train-only and rollout-only debug modes, external rollout,
# multi-policy runs, flag precedence and split deployments (trainer, inference, primary). The
# second half tests creating training models with fake trainer controller handles: critic and
# actor consistency, argument neutralization and remapping for the critic, init ordering, parallel
# config propagation, waits for externally addressed controllers, and run identity validation.
from argparse import Namespace

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.ray import placement_group as placement_group_module
from miles.ray.placement_group import _assert_external_trainer_in_run, _get_placement_group_layout
from miles.utils.workers.types import DeploymentIdentity

_RUN_UUID = "0" * 16


def _layout_args(**overrides):
    # [FUNCTION DOC] HELPER: Builds a plain namespace of layout inputs: actor 1 node of 2 gpus,
    # critic 1 node of 3 gpus (critic enabled), 4 rollout gpus, no eval gpus, all debug,
    # external and colocate flags off, no megatron config, the all-in-one deploy component and no
    # instance id; the given overrides replace values. Returns the namespace.
    values = {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 2,
        "rollout_num_gpus": 4,
        "eval_num_gpus": 0,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "rollout_external": False,
        "colocate": False,
        "use_critic": True,
        "megatron_config": None,
        "critic_load": None,
        "critic_save": None,
        "critic_lr": None,
        "critic_lr_warmup_iters": None,
        "deploy_component": "all",
        "deploy_instance_id": None,
        "critic_num_nodes": 1,
        "critic_num_gpus_per_node": 3,
    }
    values.update(overrides)
    return Namespace(**values)


class TestTheLayoutOfOneTrainerDeployment:
    # [CLASS DOC] Tests the layout of a split trainer deployment versus a whole run.
    def test_a_trainer_deployment_reserves_the_gpus_of_the_trainer_its_arguments_describe(self):
        # [FUNCTION DOC] For the trainer deploy component with instance a-actor, no critic and an
        # encoded one-policy megatron config, the layout is (2, 2): one trainer's gpus, with the
        # rollout slice starting at the end.
        """Its arguments carry one model and never learn of the others, so they alone size the release."""
        args = _layout_args(
            deploy_component="trainer",
            deploy_instance_id="a-actor",
            use_critic=False,
            megatron_config=encode_megatron_config("a"),
        )

        assert _get_placement_group_layout(args) == (2, 2)

    def test_a_whole_run_still_reserves_every_trainer_it_drives(self):
        # [FUNCTION DOC] For an unsplit run with a two-policy config and no critic, the layout is
        # (8, 4): two trainers (4 gpus) plus 4 rollout gpus, rollout offset at 4.
        """An unsplit run carries the whole multi-model config, and its layout must not move."""
        args = _layout_args(use_critic=False, megatron_config=encode_megatron_config("a", "b"))

        assert _get_placement_group_layout(args) == (8, 4)


@pytest.mark.parametrize(
    ("colocate", "expected"),
    [
        (False, (6, 2)),
        (True, (4, 0)),
    ],
)
def test_shared_ppo_counts_actor_bundles_once(colocate, expected):
    # [FUNCTION DOC] With actor and critic enabled (they share the actor bundles), the layout is
    # (6, 2) without colocation (actor 2 plus rollout 4, rollout offset 2) and (4, 0) with
    # colocation (rollout shares the trainer gpus, offset 0).
    assert _get_placement_group_layout(_layout_args(colocate=colocate)) == expected


def test_debug_train_only_counts_actor_bundles_once_and_leaves_the_rollout_entry_empty():
    # [FUNCTION DOC] With train-only debug (critic on), the layout is (2, 2): only the actor
    # bundles, with the rollout offset at the end so the rollout slice is empty.
    """No engine is deployed, so every bundle belongs to the trainer and the rollout slice starts past the end."""
    assert _get_placement_group_layout(_layout_args(debug_train_only=True)) == (2, 2)


def test_debug_train_only_never_reads_the_rollout_size_it_was_not_given():
    # [FUNCTION DOC] With train-only debug and the rollout gpu count unset (none), the layout is
    # still (2, 2): the rollout size is not read in this mode.
    """A train-only run is launched without --rollout-num-gpus, so naming the rollout contribution
    before returning makes every one of them die on None at startup."""
    assert _get_placement_group_layout(_layout_args(debug_train_only=True, rollout_num_gpus=None)) == (2, 2)


def test_debug_rollout_only_bundles_the_eval_engines_too():
    # [FUNCTION DOC] With rollout-only debug and 3 eval gpus, the layout is (7, 0): 4 rollout plus
    # 3 eval gpus, no trainer bundles.
    """--eval-num-gpus buys engines this run launches, and leaving them out of the group strands them."""
    assert _get_placement_group_layout(_layout_args(debug_rollout_only=True, eval_num_gpus=3)) == (7, 0)


def test_colocate_bundles_the_eval_engines_too():
    # [FUNCTION DOC] With colocation and 3 eval gpus, the layout is (7, 0): the shared 4 gpus plus
    # the 3 extra eval gpus, with rollout offset 0.
    """Colocated engines share the trainer's gpus, but the eval engines are extra ones nobody placed."""
    assert _get_placement_group_layout(_layout_args(colocate=True, eval_num_gpus=3)) == (7, 0)


def test_debug_train_only_reserves_dedicated_eval_bundles_after_actor():
    # [FUNCTION DOC] With train-only debug and 4 eval gpus, the layout is (6, 2): actor 2 plus 4
    # eval gpus placed after the actor bundles.
    assert _get_placement_group_layout(_layout_args(debug_train_only=True, eval_num_gpus=4)) == (6, 2)


def test_external_rollout_only_reserves_no_local_bundles():
    # [FUNCTION DOC] With rollout-only debug and external rollout, the layout is (0, 0).
    assert _get_placement_group_layout(_layout_args(debug_rollout_only=True, rollout_external=True)) == (0, 0)


def test_external_rollout_reserves_gpus_for_the_trainer_only():
    # [FUNCTION DOC] With external rollout, the layout is (2, 2): only the trainer's gpus.
    """External engines run outside ray, so only the trainer's gpus may be bundled."""
    assert _get_placement_group_layout(_layout_args(rollout_external=True)) == (2, 2)


def _multi_policy_layout_args(num_policies: int, **overrides):
    # [FUNCTION DOC] HELPER: Builds layout arguments with an encoded megatron config naming the
    # given number of policies (single letter ids a, b, c and so on), the critic disabled, and any
    # extra overrides.
    model_ids = [chr(ord("a") + i) for i in range(num_policies)]
    return _layout_args(megatron_config=encode_megatron_config(*model_ids), use_critic=False, **overrides)


class TestPlacementGroupLayout:
    # [CLASS DOC] Tests how the number of policies scales the layout.
    def test_every_policy_reserves_a_trainer_slice_of_its_own(self):
        # [FUNCTION DOC] With 1, 2 and 3 policies (2 trainer gpus each, 4 rollout gpus) the layouts
        # are (6, 2), (8, 4) and (10, 6).
        """Sizing the group for one policy lands every policy on the same gpus."""
        assert _get_placement_group_layout(_multi_policy_layout_args(1)) == (6, 2)
        assert _get_placement_group_layout(_multi_policy_layout_args(2)) == (8, 4)
        assert _get_placement_group_layout(_multi_policy_layout_args(3)) == (10, 6)

    def test_the_rollout_side_of_the_layout_is_not_multiplied(self):
        # [FUNCTION DOC] With 3 policies, total minus rollout offset is 4, the rollout gpu count.
        """Policies train apart but share one inference fleet, whose size --rollout-num-gpus alone decides."""
        num_gpus, rollout_offset = _get_placement_group_layout(_multi_policy_layout_args(3))

        assert num_gpus - rollout_offset == 4

    def test_the_policy_count_multiplies_the_debug_and_external_layouts_too(self):
        # [FUNCTION DOC] With 3 policies, both train-only debug and external rollout give (6, 6):
        # three trainers of 2 gpus and no rollout slice.
        """The flags that zero the rollout side must not also drop the policies out of the trainer side."""
        assert _get_placement_group_layout(_multi_policy_layout_args(3, debug_train_only=True)) == (6, 6)
        assert _get_placement_group_layout(_multi_policy_layout_args(3, rollout_external=True)) == (6, 6)

    def test_only_the_actors_are_counted_as_policies(self):
        # [FUNCTION DOC] With one policy and the critic enabled, the layout is (6, 2): the critic
        # does not add a trainer slice.
        """An actor and its critic share the actor placement group, so a critic must not widen it."""
        args = _layout_args(megatron_config=encode_megatron_config("a"), use_critic=True)

        assert _get_placement_group_layout(args) == (6, 2)


class TestWhichFlagWinsWhenTwoAreSet:
    # [CLASS DOC] Tests precedence between the debug, external and colocate flags in the layout.
    """The layout is one if-chain, and reordering it changes only which engines land outside the group."""

    def test_training_only_outranks_every_rollout_flag(self):
        # [FUNCTION DOC] With train-only, external, rollout-only and colocate all set, the layout
        # is (2, 2): train-only wins.
        """It is the one flag that says there are no engines at all, so nothing below it can add any."""
        args = _layout_args(debug_train_only=True, rollout_external=True, debug_rollout_only=True, colocate=True)

        assert _get_placement_group_layout(args) == (2, 2)

    def test_an_external_rollout_outranks_colocation(self):
        # [FUNCTION DOC] With external rollout and colocation, the layout is (2, 2): external wins.
        """Colocation shares gpus with engines this run does not own, so it has nothing left to share."""
        args = _layout_args(rollout_external=True, colocate=True)

        assert _get_placement_group_layout(args) == (2, 2)

    def test_an_external_rollout_that_is_also_rollout_only_reserves_nothing(self):
        # [FUNCTION DOC] With external, rollout-only and colocate all set, the layout is (0, 0).
        """No trainer of ours and no engine of ours leaves an empty group, not a trainer-sized one."""
        args = _layout_args(rollout_external=True, debug_rollout_only=True, colocate=True)

        assert _get_placement_group_layout(args) == (0, 0)

    def test_a_rollout_only_run_outranks_colocation(self):
        # [FUNCTION DOC] With rollout-only, colocation and 3 eval gpus, the layout is (7, 0): the
        # sum of rollout and eval gpus rather than a maximum with the trainer size.
        """There is no trainer to colocate with, so taking the max would size the group off a dead operand."""
        args = _layout_args(debug_rollout_only=True, colocate=True, eval_num_gpus=3)

        assert _get_placement_group_layout(args) == (7, 0)


class TestTheLayoutOfASplitDeployment:
    # [CLASS DOC] Tests the layout for each deploy component of a split run.
    def test_a_trainer_deployment_bundles_the_trainer_gpus_and_leaves_the_rollout_entry_empty(self):
        # [FUNCTION DOC] The trainer deploy component gives (2, 2).
        """This release installs no engine, so a rollout slice over its bundles would hand them out twice."""
        assert _get_placement_group_layout(_layout_args(deploy_component="trainer")) == (2, 2)

    def test_an_inference_deployment_counts_the_eval_engines_too(self):
        # [FUNCTION DOC] The inference deploy component with 3 eval gpus gives (7, 0).
        """--eval-num-gpus buys engines that live in this release, so its group has to hold them."""
        assert _get_placement_group_layout(_layout_args(deploy_component="inference", eval_num_gpus=3)) == (7, 0)

    @pytest.mark.parametrize("overrides", [{"debug_train_only": True}, {"rollout_external": True}])
    def test_an_inference_deployment_without_local_engines_bundles_nothing(self, overrides):
        # [FUNCTION DOC] Parametrized over train-only debug and external rollout: the inference
        # deploy component gives (0, 0).
        """Neither flag leaves an engine inside ray, and an empty group is what asks the cluster for nothing."""
        assert _get_placement_group_layout(_layout_args(deploy_component="inference", **overrides)) == (0, 0)

    def test_a_primary_deployment_bundles_nothing(self):
        # [FUNCTION DOC] The primary deploy component gives (0, 0).
        """It holds no engine and no rank of its own, so every gpu it reserved would sit idle."""
        assert _get_placement_group_layout(_layout_args(deploy_component="primary")) == (0, 0)

    @pytest.mark.parametrize("deploy_instance_id", [None, "extra"])
    def test_an_inference_deployment_bundles_the_engine_gpus_alone(self, deploy_instance_id):
        # [FUNCTION DOC] Parametrized over no instance id and the instance id extra: the inference
        # deploy component gives (4, 0), i.e. the rollout gpus only.
        """An engine release carries the same engines the primary would, and never a trainer rank."""
        args = _layout_args(deploy_component="inference", deploy_instance_id=deploy_instance_id)

        assert _get_placement_group_layout(args) == (4, 0)

    def test_a_trainer_deployment_of_a_rollout_only_run_bundles_nothing(self):
        # [FUNCTION DOC] The trainer deploy component with rollout-only debug gives (0, 0).
        """--debug-rollout-only trains nothing, so this release has no rank to place."""
        assert _get_placement_group_layout(_layout_args(deploy_component="trainer", debug_rollout_only=True)) == (0, 0)


class _RecordingRolloutExecutor:
    # [CLASS DOC] HELPER: Fake rollout executor recording the train parallel config, the trainer
    # model id it came with, and the rollout id passed to load.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with no recorded config and no loaded rollout id.
        self.train_parallel_config = None
        self.loaded_rollout_id = None

    async def set_train_parallel_config(self, config, trainer_model_id=None):
        # [FUNCTION DOC] HELPER: Records the config and the optional trainer model id.
        self.train_parallel_config = config
        self.train_parallel_config_model_id = trainer_model_id

    async def load(self, rollout_id=None):
        # [FUNCTION DOC] HELPER: Records the rollout id.
        self.loaded_rollout_id = rollout_id


def _patch_train_controller_handles(monkeypatch, *, restored: dict[str, list[int]] | None = None) -> list:
    # [FUNCTION DOC] HELPER: Replaces the backend capability lookup (with a fake capability) and
    # the trainer controller handle factory (with one creating a recording fake handle per
    # trainer id). Returns the list collecting the created fake handles in creation order. The
    # fake handles share one ordered call list; their init stores the args and returns the
    # restored rollout list configured for their trainer id (default [0]); the parallel config is
    # a data parallel size of 2 for the actor and 99 otherwise; they report not initialized; and
    # their deployment identity is the trainer deploy component with the trainer id as instance
    # id and trainer id, under the module run uuid.
    handles = []
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(placement_group_module, "get_backend_capability", lambda args: FakeBackendCapability())

    class _Handle:
        # [CLASS DOC] HELPER: Recording fake trainer controller handle (see the enclosing helper).
        def __init__(self, trainer_id):
            # [FUNCTION DOC] HELPER: Stores the trainer id, a placeholder for the init args and the
            # shared call list, and registers itself in the creation-ordered handle list.
            self.trainer_id = trainer_id
            self.inited_with = None
            self.calls = calls
            handles.append(self)

        async def init(self, args):
            # [FUNCTION DOC] HELPER: Records an init call, stores the args, and returns the
            # restored rollout list for this trainer id, defaulting to [0].
            calls.append((self.trainer_id, "init"))
            self.inited_with = args
            return (restored or {}).get(self.trainer_id, [0])

        async def get_train_parallel_config(self):
            # [FUNCTION DOC] HELPER: Records the call and returns a config mapping whose data
            # parallel size is 2 for the actor trainer and 99 for any other trainer.
            calls.append((self.trainer_id, "get_train_parallel_config"))
            return {"dp_size": 2 if self.trainer_id == "actor" else 99}

        async def is_initialized(self) -> bool:
            # [FUNCTION DOC] HELPER: Always reports not initialized (no call recorded).
            return False

        async def get_deployment_identity(self) -> DeploymentIdentity:
            # [FUNCTION DOC] HELPER: Returns an identity with the module run uuid, the trainer
            # deploy component, and the trainer id as both instance id and trainer id.
            return DeploymentIdentity(
                run_uuid=_RUN_UUID,
                deploy_component="trainer",
                deploy_instance_id=self.trainer_id,
                trainer_id=self.trainer_id,
            )

    monkeypatch.setattr(
        placement_group_module,
        "create_trainer_controller_handle",
        lambda args, *, capability, trainer_id: _Handle(trainer_id),
    )
    return handles


def _training_models_args(**overrides):
    # [FUNCTION DOC] HELPER: Builds a namespace of training model inputs: actor and critic
    # of 1 node x 2 gpus, critic enabled, kl coefficient 0.1 and distillation of the megatron
    # type on, param buffer cpu backup disabled, actor load/save /ckpt/run, learning rate 1e-6 with
    # 10 warmup iterations, critic load /ckpt/critic, save /ckpt/run_critic, learning rate 2e-6 and
    # 3 warmup iterations, no external controller addresses, and the module run uuid; overrides
    # replace entries.
    values = {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 2,
        "critic_num_nodes": 1,
        "critic_num_gpus_per_node": 2,
        "use_critic": True,
        "kl_coef": 0.1,
        "use_kl_loss": False,
        "use_opd": True,
        "opd_type": "megatron",
        "disable_param_buffers_cpu_backup": True,
        "start_rollout_id": None,
        "rollout_global_dataset": False,
        "megatron_config": None,
        "trainer_model_id": None,
        "load": "/ckpt/run",
        "save": "/ckpt/run",
        "lr": 1e-6,
        "lr_warmup_iters": 10,
        "critic_load": "/ckpt/critic",
        "critic_save": "/ckpt/run_critic",
        "critic_lr": 2e-6,
        "critic_lr_warmup_iters": 3,
        "trainer_controller_addrs": None,
        "run_uuid": _RUN_UUID,
    }
    values.update(overrides)
    return Namespace(**values)


async def test_an_actor_and_a_critic_that_restored_to_different_rollouts_are_refused(monkeypatch):
    # [FUNCTION DOC] Actor restores rollout 5 and critic 4: creating training models raises an
    # assertion error.
    """The two checkpoint trees are written one after the other, so a crash between them lands exactly here."""
    _patch_train_controller_handles(monkeypatch, restored={"actor": [5], "critic": [4]})

    with pytest.raises(AssertionError):
        await placement_group_module.create_training_models(
            _training_models_args(), rollout_executor=_RecordingRolloutExecutor()
        )


async def test_an_explicit_start_rollout_id_does_not_hide_a_mismatch(monkeypatch):
    # [FUNCTION DOC] Same mismatch (5 versus 4) with an explicit start rollout id 9 still raises an
    # assertion error.
    """The override says which rollout to run next, not that the two checkpoint trees agree."""
    _patch_train_controller_handles(monkeypatch, restored={"actor": [5], "critic": [4]})

    with pytest.raises(AssertionError):
        await placement_group_module.create_training_models(
            _training_models_args(start_rollout_id=9), rollout_executor=_RecordingRolloutExecutor()
        )


async def test_an_actor_and_a_critic_that_agree_set_the_start_rollout_id(monkeypatch):
    # [FUNCTION DOC] Both restore rollout 5: args receive start rollout id 5 and the executor is
    # told to load rollout 4 (the one before the start).
    """A resume takes its position from the checkpoints, and the executor replays from the rollout before it."""
    _patch_train_controller_handles(monkeypatch, restored={"actor": [5], "critic": [5]})
    args = _training_models_args()
    rollout_executor = _RecordingRolloutExecutor()

    await placement_group_module.create_training_models(args, rollout_executor=rollout_executor)

    assert args.start_rollout_id == 5
    assert rollout_executor.loaded_rollout_id == 4


async def test_a_run_without_a_critic_takes_the_actor_position(monkeypatch):
    # [FUNCTION DOC] With no critic and the actor restoring 5, args receive start rollout id 5.
    """With one trainer there is nothing to agree with, and the actor's own position must still be taken."""
    _patch_train_controller_handles(monkeypatch, restored={"actor": [5]})
    args = _training_models_args(use_critic=False)

    await placement_group_module.create_training_models(args, rollout_executor=_RecordingRolloutExecutor())

    assert args.start_rollout_id == 5


async def test_a_critic_run_inits_one_controller_per_role(monkeypatch):
    # [FUNCTION DOC] With a critic, two handles are created for trainer ids actor then critic,
    # and both received init args.
    """Each role is its own worker, and both have to be inited before anybody calls them."""
    handles = _patch_train_controller_handles(monkeypatch)

    await placement_group_module.create_training_models(
        _training_models_args(),
        rollout_executor=_RecordingRolloutExecutor(),
    )

    assert [handle.trainer_id for handle in handles] == ["actor", "critic"]
    assert all(handle.inited_with is not None for handle in handles)


async def test_the_critic_controller_is_inited_with_neutralized_args(monkeypatch):
    # [FUNCTION DOC] The actor is inited with kl 0.1, distillation on and cpu backup disabled
    # kept; the critic is inited with kl 0, distillation off and cpu backup not disabled; and the
    # caller's own args are left unchanged.
    """A critic controller must not hand its cells the actor's KL and OPD settings."""
    handles = _patch_train_controller_handles(monkeypatch)
    args = _training_models_args()

    await placement_group_module.create_training_models(args, rollout_executor=_RecordingRolloutExecutor())

    actor_args, critic_args = (handle.inited_with for handle in handles)
    assert (actor_args.kl_coef, actor_args.use_opd, actor_args.disable_param_buffers_cpu_backup) == (0.1, True, True)
    assert (critic_args.kl_coef, critic_args.use_opd, critic_args.disable_param_buffers_cpu_backup) == (
        0,
        False,
        False,
    )
    assert (args.kl_coef, args.use_opd, args.disable_param_buffers_cpu_backup) == (0.1, True, True)


async def test_the_critic_controller_is_inited_with_the_critic_checkpoint_and_schedule(monkeypatch):
    # [FUNCTION DOC] The critic's init args carry the critic load and save paths, learning rate
    # and warmup iterations in the standard fields, while the caller's own args keep the actor values.
    """The worker no longer swaps critic_* onto the standard fields, so the args must arrive remapped."""
    handles = _patch_train_controller_handles(monkeypatch)
    args = _training_models_args()

    await placement_group_module.create_training_models(args, rollout_executor=_RecordingRolloutExecutor())

    _actor_args, critic_args = (handle.inited_with for handle in handles)
    assert (critic_args.load, critic_args.save, critic_args.lr, critic_args.lr_warmup_iters) == (
        "/ckpt/critic",
        "/ckpt/run_critic",
        2e-6,
        3,
    )
    assert (args.load, args.save, args.lr, args.lr_warmup_iters) == ("/ckpt/run", "/ckpt/run", 1e-6, 10)


async def test_the_controllers_are_inited_before_the_driver_calls_them(monkeypatch):
    # [FUNCTION DOC] The shared call list equals: actor init, critic init, then the actor parallel
    # config fetch (only the actor is asked for the config).
    """init() is what hands a controller its args, so any earlier call reaches a controller without them."""
    handles = _patch_train_controller_handles(monkeypatch)

    await placement_group_module.create_training_models(
        _training_models_args(),
        rollout_executor=_RecordingRolloutExecutor(),
    )

    assert handles[0].calls == [("actor", "init"), ("critic", "init"), ("actor", "get_train_parallel_config")]


async def test_a_run_without_a_critic_starts_only_the_actor_controller(monkeypatch):
    # [FUNCTION DOC] With the critic disabled, only one handle (actor) is created.
    """A critic controller nobody asked for would sit waiting for cells that are never scheduled."""
    handles = _patch_train_controller_handles(monkeypatch)

    await placement_group_module.create_training_models(
        _training_models_args(use_critic=False),
        rollout_executor=_RecordingRolloutExecutor(),
    )

    assert [handle.trainer_id for handle in handles] == ["actor"]


async def test_train_parallel_config_travels_from_trainer_to_rollout_executor(monkeypatch):
    # [FUNCTION DOC] Without a critic, the executor ends up with the actor's config (data parallel
    # size 2) and loaded rollout -1 (restored 0 minus one).
    """The driver reads the parallel config off the trainer and writes it into the executor."""
    _patch_train_controller_handles(monkeypatch)
    rollout_executor = _RecordingRolloutExecutor()

    await placement_group_module.create_training_models(
        _training_models_args(use_critic=False),
        rollout_executor=rollout_executor,
    )

    assert rollout_executor.train_parallel_config == {"dp_size": 2}
    assert rollout_executor.loaded_rollout_id == -1


async def test_train_parallel_config_comes_from_the_actor_not_the_critic(monkeypatch):
    # [FUNCTION DOC] With a critic (whose fake config would have data parallel size 99), the
    # executor still gets the actor's config (data parallel size 2).
    """With a critic present, the config still comes from the actor group."""
    _patch_train_controller_handles(monkeypatch)
    rollout_executor = _RecordingRolloutExecutor()

    await placement_group_module.create_training_models(
        _training_models_args(use_critic=True),
        rollout_executor=rollout_executor,
    )

    assert rollout_executor.train_parallel_config == {"dp_size": 2}


class TestTheRunWaitsForEveryTrainerItReachesByAddress:
    # [CLASS DOC] Tests that the driver waits for externally addressed trainer controllers before
    # initializing any of them, and waits for none when it deploys its own.
    @staticmethod
    def _patched(monkeypatch) -> list[list[tuple[str, int]]]:
        # [FUNCTION DOC] HELPER: Installs the fake trainer handles and replaces the address
        # readiness wait with a fake that records each call as a list of (host, port) pairs.
        # Returns the list collecting those recorded calls.
        dialled: list[list[tuple[str, int]]] = []
        _patch_train_controller_handles(monkeypatch)

        async def _dial(addrs) -> None:
            # [FUNCTION DOC] HELPER: Records the host and port pairs of the given addresses.
            dialled.append([(addr.host, addr.port) for addr in addrs])

        monkeypatch.setattr(placement_group_module, "wait_static_addrs_ready", _dial)
        return dialled

    async def test_it_dials_every_addressed_controller_before_any_of_them_is_inited(self, monkeypatch):
        # [FUNCTION DOC] With external addresses for actor and critic, exactly one wait call occurs,
        # covering both addresses in order.
        # KNOWN ISSUE: despite the name, the test does not assert that the wait happens before any
        # init; it only checks the recorded wait call.
        """A deployment installed a moment earlier is not listening yet, and init would simply fail against it."""
        dialled = self._patched(monkeypatch)
        args = _training_models_args(
            megatron_config=None, trainer_controller_addrs=["actor=10.0.0.1:8000", "critic=10.0.0.2:9000"]
        )

        await placement_group_module.create_training_models(args, rollout_executor=_RecordingRolloutExecutor())

        assert dialled == [[("10.0.0.1", 8000), ("10.0.0.2", 9000)]]

    async def test_a_run_that_deploys_its_own_trainer_waits_for_nobody(self, monkeypatch):
        # [FUNCTION DOC] Without external addresses, no wait call is recorded.
        """It installs the trainer itself, so there is no other deployment whose readiness it could dial."""
        dialled = self._patched(monkeypatch)

        await placement_group_module.create_training_models(
            _training_models_args(), rollout_executor=_RecordingRolloutExecutor()
        )

        assert dialled == []


class _IdentifyingHandle:
    # [CLASS DOC] HELPER: Fake trainer handle with a configurable deployment identity, recording
    # identity queries and init calls into a shared list.
    def __init__(self, *, trainer_id: str, run_uuid: str, deploy_component: str, calls: list[tuple[str, str]]) -> None:
        # [FUNCTION DOC] HELPER: Stores the trainer id, builds the identity from the run uuid and
        # deploy component (trainer id doubles as instance id) and keeps the shared call list.
        self.trainer_id = trainer_id
        self.identity = DeploymentIdentity(
            run_uuid=run_uuid,
            deploy_component=deploy_component,
            deploy_instance_id=trainer_id,
            trainer_id=trainer_id,
        )
        self.calls = calls

    async def get_deployment_identity(self) -> DeploymentIdentity:
        # [FUNCTION DOC] HELPER: Records an identity query and returns the configured identity.
        self.calls.append((self.trainer_id, "get_deployment_identity"))
        return self.identity

    async def init(self, args) -> list[int]:
        # [FUNCTION DOC] HELPER: Records an init call and returns the restored list [0].
        self.calls.append((self.trainer_id, "init"))
        return [0]


def _split_run_args(**overrides):
    # [FUNCTION DOC] HELPER: Training model arguments for a split run with external controller
    # addresses for actor (10.0.0.1:8000) and critic (10.0.0.2:8000) and no megatron config.
    return _training_models_args(
        megatron_config=None,
        trainer_controller_addrs=["actor=10.0.0.1:8000", "critic=10.0.0.2:8000"],
        **overrides,
    )


def _patch_identifying_handles(monkeypatch, *, identities: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    # [FUNCTION DOC] HELPER: Replaces the capability lookup, the address readiness wait (does
    # nothing) and the trainer handle factory (builds an identifying handle per trainer id from a
    # mapping of trainer id to (run uuid, deploy component)). Returns the shared list of
    # (trainer id, call name) records the handles append to.
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(placement_group_module, "get_backend_capability", lambda args: FakeBackendCapability())

    async def _dial(addrs) -> None:
        # [FUNCTION DOC] HELPER: Address readiness wait that does nothing.
        return None

    monkeypatch.setattr(placement_group_module, "wait_static_addrs_ready", _dial)
    monkeypatch.setattr(
        placement_group_module,
        "create_trainer_controller_handle",
        lambda args, *, capability, trainer_id: _IdentifyingHandle(
            trainer_id=trainer_id,
            run_uuid=identities[trainer_id][0],
            deploy_component=identities[trainer_id][1],
            calls=calls,
        ),
    )
    return calls


class TestEveryAddressedTrainerIsCheckedBeforeAnyInitRuns:
    # [CLASS DOC] Tests that identity validation of all external trainers happens before any init.
    async def test_a_second_controller_of_another_run_is_caught_before_the_first_is_inited(self, monkeypatch):
        # [FUNCTION DOC] The critic controller belongs to another run: creation raises an
        # assertion error saying it drives a different run, and the recorded calls are exactly the
        # two identity queries (no init).
        """init runs once per deployment, so an init before the check leaves a trainer nothing can re-init."""
        calls = _patch_identifying_handles(
            monkeypatch, identities={"actor": ("0" * 16, "trainer"), "critic": ("f" * 16, "trainer")}
        )

        with pytest.raises(AssertionError, match="drives run"):
            await placement_group_module.create_training_models(
                _split_run_args(), rollout_executor=_RecordingRolloutExecutor()
            )

        assert sorted(calls) == [("actor", "get_deployment_identity"), ("critic", "get_deployment_identity")]

    async def test_a_deployment_that_carries_an_orchestration_script_of_its_own_is_refused(self, monkeypatch):
        # [FUNCTION DOC] The actor controller belongs to an all-in-one deployment (it has its own
        # driver script): creation raises an assertion error saying only the trainer may be
        # deployed, and the actor is never inited.
        """Its own script drives that trainer too, so both runs would train one model from two rollout streams."""
        calls = _patch_identifying_handles(
            monkeypatch, identities={"actor": ("0" * 16, "all"), "critic": ("0" * 16, "trainer")}
        )

        with pytest.raises(AssertionError, match="nothing but the trainer"):
            await placement_group_module.create_training_models(
                _split_run_args(), rollout_executor=_RecordingRolloutExecutor()
            )

        assert ("actor", "init") not in calls


class TestTheAddressesNameOneRun:
    # [CLASS DOC] Tests the in-run validation of an external trainer's deployment identity against
    # the arguments of the driving launch.
    @staticmethod
    def _identity(
        *,
        run_uuid: str,
        deploy_component: str = "trainer",
        deploy_instance_id: str | None = None,
        trainer_id: str | None = None,
    ) -> DeploymentIdentity:
        # [FUNCTION DOC] HELPER: Builds a deployment identity from the given run uuid, deploy
        # component (default trainer), instance id and trainer id (both default none).
        return DeploymentIdentity(
            run_uuid=run_uuid,
            deploy_component=deploy_component,
            deploy_instance_id=deploy_instance_id,
            trainer_id=trainer_id,
        )

    def test_a_deployment_of_this_run_is_accepted(self):
        # [FUNCTION DOC] An identity with the same run uuid and the trainer component passes with
        # no error.
        """Every deployment of one run carries the same run uuid, so the usual case must pass silently."""
        args = _training_models_args(run_uuid="0123456789abcdef")

        _assert_external_trainer_in_run(self._identity(run_uuid=args.run_uuid), args=args)

    def test_a_deployment_of_another_run_stops_the_launch(self):
        # [FUNCTION DOC] A different run uuid raises an assertion error saying it drives a
        # different run.
        """Pointing at last run's release trains weights this run never updates, and looks like bad rewards."""
        args = _training_models_args(run_uuid="0123456789abcdef")

        with pytest.raises(AssertionError, match="drives run"):
            _assert_external_trainer_in_run(self._identity(run_uuid="ffffffffffffffff"), args=args)

    def test_an_unsplit_release_of_this_run_stops_the_launch(self):
        # [FUNCTION DOC] The same run uuid but the all-in-one component raises an assertion error
        # saying nothing but the trainer may be deployed.
        """It carries an orchestration script of its own, which drives the very trainer this launch would drive."""
        args = _training_models_args(run_uuid="0123456789abcdef")

        with pytest.raises(AssertionError, match="nothing but the trainer"):
            _assert_external_trainer_in_run(self._identity(run_uuid=args.run_uuid, deploy_component="all"), args=args)

    def test_a_deployment_reached_as_another_trainer_stops_the_launch(self):
        # [FUNCTION DOC] An identity with instance id critic reached under trainer id actor raises
        # an assertion error saying addresses are keyed by trainer id.
        """Its ranks would be driven through the workflow of a trainer they do not belong to."""
        args = _training_models_args(run_uuid="0123456789abcdef")

        with pytest.raises(AssertionError, match="are keyed by trainer id"):
            _assert_external_trainer_in_run(
                self._identity(run_uuid=args.run_uuid, deploy_instance_id="critic"), args=args, trainer_id="actor"
            )

    def test_a_deployment_that_was_never_named_is_accepted_as_any_trainer(self):
        # [FUNCTION DOC] An identity with no instance id and trainer id actor, checked as trainer
        # actor, passes with no error.
        """It was launched without --deploy-instance-id, so there is no name here to check the workflow against."""
        args = _training_models_args(run_uuid="0123456789abcdef")

        _assert_external_trainer_in_run(
            self._identity(run_uuid=args.run_uuid, trainer_id="actor"), args=args, trainer_id="actor"
        )
