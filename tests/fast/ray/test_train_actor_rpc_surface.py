# [MODULE DOC]
# Tests that the train actor class (and the concrete megatron and fsdp backends) expose the rpc
# surface a driver needs: every driven method is collected as an rpc method, internal methods are
# not exposed, concurrency groups isolate the heartbeat, the kill and the fault injector from the
# train step while the train step shares a group with init, request and result serialization of
# the train and init calls (object store references, step outputs, pickled args) round trips, and
# backend overrides accept every parameter the base declares. Backend modules are imported with
# an importorskip guard, so tests skip where their dependencies are missing.
from __future__ import annotations

import inspect
from argparse import Namespace

import pytest
from pydantic import ValidationError

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray.specs.train import TRAINER_CONCURRENCY_GROUPS
from miles.ray.train_actor import TrainRayActor
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs, declared_concurrency_groups

DRIVEN_METHODS = (
    "init",
    "train",
    "sleep",
    "wake_up",
    "clear_memory",
    "save_model",
    "export_hf",
    "update_weights",
    "get_train_parallel_config",
    "get_heartbeat_status",
    "inject_fault",
    "kill_self",
    "configure_master_addr_and_port",
    "propose_master_addr_and_port",
)


MEGATRON_ONLY_DRIVEN_METHODS = frozenset({"reconfigure_indep_dp", "send_ckpt"})


class TestTheTrainerSurfaceIsCallableOverRpc:
    # [CLASS DOC] Tests which methods of the base train actor are exposed over rpc and how they are
    # grouped for concurrency.
    def test_the_whole_surface_is_accepted(self):
        # [FUNCTION DOC] Collecting rpc method specs from the base actor succeeds (an unannotated
        # public method would make this fail) and contains every driven method name.
        """Under --worker-comm-backend rpc an unannotated public method makes the pool unreachable."""
        specs = collect_rpc_method_specs(TrainRayActor)

        assert set(DRIVEN_METHODS) <= set(specs)

    def test_no_internal_method_is_exposed(self):
        # [FUNCTION DOC] None of the internal method names (train actor, train critic, log prob
        # computation, model class getter) is among the collected rpc specs.
        """A method that never crosses the wire is a method whose types nobody has to keep honest."""
        specs = collect_rpc_method_specs(TrainRayActor)

        assert not {"train_actor", "train_critic", "compute_log_prob", "get_model_cls"} & set(specs)

    def test_the_heartbeat_never_queues_behind_a_train_step(self):
        # [FUNCTION DOC] The heartbeat status method and the train method have different
        # concurrency groups.
        """A heartbeat answered late reads as a dead cell and costs the run its trainer."""
        specs = collect_rpc_method_specs(TrainRayActor)

        assert specs["get_heartbeat_status"].concurrency_group != specs["train"].concurrency_group

    def test_every_group_a_method_names_is_a_group_the_spec_declares(self):
        # [FUNCTION DOC] The set of concurrency groups declared by the actor's methods is a subset
        # of the trainer concurrency group declarations.
        """Ray refuses to build an actor whose method names a group its class never declared, at launch time."""
        named_by_methods = set(declared_concurrency_groups(TrainRayActor).values())

        assert named_by_methods <= set(TRAINER_CONCURRENCY_GROUPS)

    def test_the_train_step_is_serialized_with_the_rest_of_the_work(self):
        # [FUNCTION DOC] The train method and the init method share the same concurrency group.
        """One gpu worker runs one thing at a time; two concurrent steps would corrupt the model."""
        specs = collect_rpc_method_specs(TrainRayActor)

        assert specs["train"].concurrency_group == specs["init"].concurrency_group


class TestWhatATrainStepSendsAndReturns:
    # [CLASS DOC] Tests the serialization round trip of the train call's request and result,
    # using the serializer of the collected train rpc spec.
    def test_the_rollout_data_reference_crosses_as_a_store_reference(self):
        # [FUNCTION DOC] Encoding then decoding a query with rollout id 1 and a mooncake store
        # object reference yields an equal reference object (a typed model, not a plain mapping).
        """The reference has to arrive as the model the object store redeems, not as a plain mapping."""
        spec = collect_rpc_method_specs(TrainRayActor)["train"]
        ref = _MooncakeStoreObjectRef(payload={"key": "miles-object-store/7"})

        query = spec.serializer.encode_query(dict(rollout_id=1, rollout_data_ref=ref))
        decoded = spec.serializer.decode_query(query)

        assert decoded["rollout_data_ref"] == ref

    def test_a_sharded_rollout_reference_crosses_as_a_list(self):
        # [FUNCTION DOC] A query whose rollout data reference is a list of two store references
        # round trips to an equal list.
        """With --delay-split-train-data-by-dp off, each dp rank is handed its own shard."""
        spec = collect_rpc_method_specs(TrainRayActor)["train"]
        refs = [_MooncakeStoreObjectRef(payload={"key": f"miles-object-store/{index}"}) for index in range(2)]

        decoded = spec.serializer.decode_query(spec.serializer.encode_query(dict(rollout_id=1, rollout_data_ref=refs)))

        assert decoded["rollout_data_ref"] == refs

    def test_the_train_output_returns_with_its_values_reference(self):
        # [FUNCTION DOC] A normal step output carrying a store reference as values round trips
        # through result encoding and decoding to an equal output.
        """The critic ships its values by reference, and the actor step reads them back from it."""
        spec = collect_rpc_method_specs(TrainRayActor)["train"]
        output = TrainStepOutput(
            outcome=TrainStepOutcome.NORMAL, values=_MooncakeStoreObjectRef(payload={"key": "miles-object-store/9"})
        )

        restored = spec.serializer.decode_result(spec.serializer.encode_result(output))

        assert restored == output

    def test_the_critic_output_can_be_handed_back_as_external_data(self):
        # [FUNCTION DOC] A query with rollout id 1, a data reference and a step output as the
        # external data argument round trips with an equal external data value.
        """The actor step is called with what the critic step returned, one payload per worker."""
        spec = collect_rpc_method_specs(TrainRayActor)["train"]
        output = TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=_MooncakeStoreObjectRef(payload={"key": "k"}))

        decoded = spec.serializer.decode_query(
            spec.serializer.encode_query(
                dict(
                    rollout_id=1, rollout_data_ref=_MooncakeStoreObjectRef(payload={"key": "d"}), external_data=output
                )
            )
        )

        assert decoded["external_data"] == output


def _init_query(**overrides: object) -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Builds the keyword mapping of an init call: an empty namespace as
    # args, role actor, the trivial independent dp info and no store address; the given overrides
    # replace entries (merged over the defaults).
    return {
        "args": Namespace(),
        "role": "actor",
        "indep_dp_info": IndepDPInfo.create_trivial(),
        "indep_dp_store_addr": None,
    } | overrides


class TestTheArgsThatBuildATrainer:
    # [CLASS DOC] Tests serialization of the init call arguments using the init rpc spec.
    def test_they_cross_through_the_pickled_hatch_unchanged(self):
        # [FUNCTION DOC] An args namespace with several fields (including a nested mapping)
        # round trips as a namespace whose variables equal the original ones.
        """Megatron reads hundreds of fields off args, and a lossy round trip changes the run."""
        spec = collect_rpc_method_specs(TrainRayActor)["init"]
        args = Namespace(num_rollout=7, hf_checkpoint="/models/qwen", train_env_vars={"A": "1"})

        decoded = spec.serializer.decode_query(spec.serializer.encode_query(_init_query(args=args)))

        assert isinstance(decoded["args"], Namespace) and vars(decoded["args"]) == vars(args)

    def test_the_other_arguments_of_the_same_call_stay_typed(self):
        # [FUNCTION DOC] Encoding a query whose role is a non-string object raises a validation
        # error mentioning the role field.
        """The hatch is per parameter, so role is still refused when it is not a string."""
        spec = collect_rpc_method_specs(TrainRayActor)["init"]

        with pytest.raises(ValidationError, match="role"):
            spec.serializer.encode_query(_init_query(role=object()))

    def test_the_independent_dp_layout_a_trainer_is_restarted_into_crosses_too(self):
        # [FUNCTION DOC] The trivial independent dp info in an init query round trips to an equal
        # value.
        """Healing rebuilds a cell with the quorum it rejoins, and the base signature is what says so."""
        spec = collect_rpc_method_specs(TrainRayActor)["init"]

        decoded = spec.serializer.decode_query(spec.serializer.encode_query(_init_query()))

        assert decoded["indep_dp_info"] == IndepDPInfo.create_trivial()


CONCRETE_BACKENDS = [
    ("miles.backends.megatron_utils.actor", "MegatronTrainRayActor"),
    ("miles.backends.fsdp_utils.actor", "FSDPTrainRayActor"),
]


def _parameter_names(method: object) -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of parameter names of a callable's signature,
    # excluding the instance parameter.
    return set(inspect.signature(method).parameters) - {"self"}


class TestTheConcreteBackends:
    # [CLASS DOC] Tests the concrete megatron and fsdp actor classes against the base surface.
    @pytest.mark.parametrize("module_path, class_name", CONCRETE_BACKENDS, ids=[name for _, name in CONCRETE_BACKENDS])
    @pytest.mark.parametrize("method_name", ["init", "train"])
    def test_a_backend_accepts_every_parameter_the_base_declares(
        self, module_path: str, class_name: str, method_name: str
    ):
        # [FUNCTION DOC] Parametrized over both backends and the init and train methods. Imports
        # the backend module (skipping the test if unavailable) and requires the backend method's
        # parameter names to be a superset of the base method's.
        """A client builds its query from the declared surface, so a backend missing a parameter is a TypeError."""
        actor_module = pytest.importorskip(module_path)

        declared = _parameter_names(getattr(TrainRayActor, method_name))

        assert declared <= _parameter_names(getattr(getattr(actor_module, class_name), method_name))

    def test_the_megatron_actor_is_accepted(self):
        # [FUNCTION DOC] Skips if the megatron actor module cannot be imported; otherwise the
        # megatron actor's collected rpc specs contain every driven method.
        """The pool that matters most is the one a driver must be able to reach over rpc."""
        actor_module = pytest.importorskip("miles.backends.megatron_utils.actor")

        assert set(DRIVEN_METHODS) <= set(collect_rpc_method_specs(actor_module.MegatronTrainRayActor))

    def test_the_methods_only_the_megatron_actor_answers_are_accepted_too(self):
        # [FUNCTION DOC] Skips if the megatron actor module cannot be imported; otherwise the
        # megatron actor's rpc specs also contain the megatron-only driven methods (independent dp
        # reconfiguration and checkpoint sending).
        """These are driven by name from the controller, so privatising one only fails in ft or multi-lora."""
        actor_module = pytest.importorskip("miles.backends.megatron_utils.actor")

        specs = collect_rpc_method_specs(actor_module.MegatronTrainRayActor)

        assert MEGATRON_ONLY_DRIVEN_METHODS <= set(specs)

    def test_the_fsdp_actor_is_accepted(self):
        # [FUNCTION DOC] Skips if the fsdp actor module cannot be imported; otherwise the fsdp
        # actor's collected rpc specs contain every driven method.
        """The second backend shares the driver's call sites, so it shares the requirement."""
        actor_module = pytest.importorskip("miles.backends.fsdp_utils.actor")

        assert set(DRIVEN_METHODS) <= set(collect_rpc_method_specs(actor_module.FSDPTrainRayActor))
