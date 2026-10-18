# [MODULE DOC]
# Unit tests for the worker spec entrypoint: computing specs from a supplied argument vector
# (parsed in isolation, process argv restored), the full inventory of specs for a run (rollout
# executor, inference controller, routers, session server, engines per server group, trainer
# controllers and engines, in that order), the debug train-only run (no inference engines or
# routers, an empty session server), and filtering by deploy component (trainer, primary,
# inference, all) including the registration reporter that only inference deployments carry.
# Configs are written to temporary YAML files and test arguments come from the shared conftest.
from __future__ import annotations

import sys
from argparse import Namespace
from collections import Counter

import pytest
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config
from tests.fast.ray.rollout.conftest import make_args, make_args_with_sglang_config, make_sglang_config_yaml

from miles.ray.specs import entrypoint
from miles.ray.specs.entrypoint import compute_specs, compute_specs_from_argv
from miles.ray.specs.inference import INFERENCE_REGISTRATION_REPORTER_POOL_ID, compute_router_providers
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_provider.kubernetes.helm.env import NAMESPACE_ENV_VAR, RELEASE_ENV_VAR
from miles.utils.workers.worker_spec import BaseWorkerSpec, WorkerCtorContext


def _unusable_capability():
    # [FUNCTION DOC] HELPER: Returns a backend capability stand-in whose static worker provider
    # factory always raises an assertion error naming the pool id, to prove a train-only run asks
    # for no provider.
    class _Unusable:
        # [CLASS DOC] HELPER: Capability stand-in that refuses to hand out worker providers.
        def static_worker_provider(self, *, pool_id):
            # [FUNCTION DOC] HELPER: Always raises an assertion error naming the pool id.
            raise AssertionError(f"a train-only run asked for a worker provider for {pool_id!r}")

    return _Unusable()


class TestComputeSpecsFromArgv:
    # [CLASS DOC] Tests computing specs from an explicit argument vector.
    def test_the_supplied_argv_builds_only_its_selected_deployment_and_restores_process_argv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Sets the process argv to an ambient value and replaces the entrypoint's
        # argument parser with a fake that records the argv it sees and builds test arguments whose
        # deploy component is the value following the deploy-component flag. Computing specs from
        # the argv with the trainer component yields exactly trainer-controller-actor and
        # trainer-engine-actor, all with the trainer component; the parser saw the program name
        # followed by the supplied arguments; and afterwards the process argv is the identical
        # ambient list object.
        """The supplied trainer argv is parsed in isolation and leaves the caller's process arguments intact."""
        ambient_argv = ["ambient-program", "--not-a-miles-option"]
        monkeypatch.setattr(sys, "argv", ambient_argv)
        parsed_argv: list[str] | None = None

        def parse_args() -> Namespace:
            # [FUNCTION DOC] HELPER: Records a copy of the current process argv and returns test
            # arguments with the deploy component taken from the value after the flag in argv.
            nonlocal parsed_argv
            parsed_argv = sys.argv.copy()
            component = sys.argv[sys.argv.index("--deploy-component") + 1]
            return make_args(deploy_component=component)

        monkeypatch.setattr(entrypoint, "parse_args", parse_args)

        specs = compute_specs_from_argv(["--deploy-component", "trainer"])

        assert [spec.name for spec in specs] == ["trainer-controller-actor", "trainer-engine-actor"]
        assert {spec.deploy_component for spec in specs} == {DeployComponent.TRAINER}
        assert parsed_argv == [ambient_argv[0], "--deploy-component", "trainer"]
        assert sys.argv is ambient_argv


class TestComputeSpecs:
    # [CLASS DOC] Tests the spec inventory computed for a whole run.
    def test_launches_the_controller_then_routers_then_the_session_server_then_every_engine(self, tmp_path):
        # [FUNCTION DOC] A config with a regular group (4 GPUs), a placeholder group and a decode
        # group, 16 rollout GPUs and the session server disabled. The spec names in order are:
        # rollout-executor, inference-controller, inference-router-0, session-server,
        # inference-engine-all-0-0, inference-engine-all-0-2 (the placeholder group has no engine
        # spec), trainer-controller-actor, trainer-engine-actor.
        """The manager's whole inventory comes from here, so every component must be listed exactly once."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2},
                    {"worker_type": "placeholder", "num_gpus": 4, "num_gpus_per_engine": 4},
                    {"worker_type": "decode", "num_gpus": 8, "num_gpus_per_engine": 4},
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=16, use_session_server=False)

        specs = compute_specs(args)

        assert [spec.name for spec in specs] == [
            "rollout-executor",
            "inference-controller",
            "inference-router-0",
            "session-server",
            "inference-engine-all-0-0",
            "inference-engine-all-0-2",
            "trainer-controller-actor",
            "trainer-engine-actor",
        ]

    def test_a_train_only_debug_run_specs_no_inference_at_all(self):
        # [FUNCTION DOC] With debug train-only, no rollout GPU count and the session server enabled,
        # computing specs succeeds; no spec name starts with inference-router or inference-engine;
        # the session server spec has zero cells; and computing router providers with the refusing
        # capability returns an empty list.
        """--debug-train-only leaves --rollout-num-gpus unset, so anything that sizes an engine
        fleet from it cannot even be built, let alone launched."""
        args = make_args(rollout_num_gpus=None, debug_train_only=True, use_session_server=True)

        specs = {spec.name: spec for spec in compute_specs(args)}

        assert not [name for name in specs if name.startswith(("inference-router", "inference-engine"))]
        assert specs["session-server"].scheduling.num_cells == 0
        # the controller resolves the same config again inside its own actor, where a failure
        # is a dead actor rather than a readable error at launch
        assert compute_router_providers(args, capability=_unusable_capability()) == []

    def test_a_disabled_session_server_is_listed_with_no_cells(self, tmp_path):
        # [FUNCTION DOC] One regular group of 4 GPUs at 2 per engine with the session server
        # disabled: the session server spec exists with zero cells, and the engine spec
        # inference-engine-all-0-0 has two cells.
        """Disabling the session server must not remove it from the inventory, only empty it."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4, use_session_server=False)

        specs = {spec.name: spec for spec in compute_specs(args)}

        assert specs["session-server"].scheduling.num_cells == 0
        assert specs["inference-engine-all-0-0"].scheduling.num_cells == 2

    def test_debug_train_only_lists_no_inference_engine(self, tmp_path):
        # [FUNCTION DOC] Colocated debug train-only run with an 8 GPU config: no spec name starts
        # with inference-engine.
        """--debug-train-only must instantiate no sglang engine, since its bundles are the trainer's own gpus."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=8,
            use_session_server=False,
            colocate=True,
            debug_train_only=True,
        )

        specs = compute_specs(args)

        assert [spec.name for spec in specs if spec.name.startswith("inference-engine")] == []

    def test_debug_train_only_lists_no_router(self, tmp_path):
        # [FUNCTION DOC] Using the shared debug train-only helper, no spec name starts with
        # inference-router.
        """A router has no engine to route to here, so paying a worker for it is pure waste."""
        specs = _debug_train_only_specs(tmp_path)

        assert [spec.name for spec in specs if spec.name.startswith("inference-router")] == []

    def test_debug_train_only_still_lists_the_trainer_and_an_empty_session_server(self, tmp_path):
        # [FUNCTION DOC] Using the shared debug train-only helper (session server v1 with 2
        # workers), the spec names in order are exactly rollout-executor, inference-controller,
        # session-server, trainer-controller-actor, trainer-engine-actor, and the session server
        # has zero cells.
        """Dropping the router must not drop the training side, and the session server survives with no cells."""
        specs = {spec.name: spec for spec in _debug_train_only_specs(tmp_path)}

        assert list(specs) == [
            "rollout-executor",
            "inference-controller",
            "session-server",
            "trainer-controller-actor",
            "trainer-engine-actor",
        ]
        assert specs["session-server"].scheduling.num_cells == 0


def _debug_train_only_specs(tmp_path) -> list[BaseWorkerSpec]:
    # [FUNCTION DOC] HELPER: Writes a sglang config with one regular group (8 GPUs, 1 per engine)
    # into the temporary directory, builds colocated debug train-only arguments (8 rollout GPUs,
    # session server v1 with 2 workers) and returns the computed spec list.
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}])
    )
    args = make_args(
        sglang_config=str(config_path),
        rollout_num_gpus=8,
        use_session_server="v1",
        session_server_workers=2,
        colocate=True,
        debug_train_only=True,
    )
    return compute_specs(args)


class TestDeployComponentFiltering:
    # [CLASS DOC] Tests that the deploy component argument selects which specs a release carries:
    # trainer (trainer controllers and engines), primary (everything the other components do not
    # claim, including routers and the inference controller), inference (engines plus a single
    # registration reporter) and all.
    @staticmethod
    def _args(tmp_path, **overrides):
        # [FUNCTION DOC] HELPER: Builds arguments with a config of one regular group (4 GPUs, 2 per
        # engine), 4 rollout GPUs, the session server on, and a critic on one node of 2 GPUs; any
        # keyword overrides (for example the deploy component) replace those values.
        return make_args_with_sglang_config(
            tmp_path,
            server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}],
            **{
                "rollout_num_gpus": 4,
                "use_session_server": True,
                "use_critic": True,
                "critic_num_nodes": 1,
                "critic_num_gpus_per_node": 2,
                **overrides,
            },
        )

    def test_a_trainer_deployment_holds_the_trainer_controllers_and_their_ranks_only(self, tmp_path):
        # [FUNCTION DOC] With the trainer component and a critic, the spec names are exactly the
        # controllers and then engines of the actor and the critic, in that order.
        """A trainer release that also installed engines would double the run's gpu bill."""
        specs = compute_specs(self._args(tmp_path, deploy_component="trainer"))

        assert [spec.name for spec in specs] == [
            "trainer-controller-actor",
            "trainer-controller-critic",
            "trainer-engine-actor",
            "trainer-engine-critic",
        ]

    def test_the_primary_deployment_holds_everything_the_other_two_do_not(self, tmp_path):
        # [FUNCTION DOC] With the primary component, no spec name starts with trainer- or
        # inference-engine, while inference-controller and inference-router-0 are present.
        """primary is defined by subtraction, so anything unclaimed has to land here rather than nowhere."""
        names = [spec.name for spec in compute_specs(self._args(tmp_path, deploy_component="primary"))]

        assert not [name for name in names if name.startswith("trainer-")]
        assert not [name for name in names if name.startswith("inference-engine")]
        assert "inference-controller" in names
        assert "inference-router-0" in names

    def test_every_worker_of_a_trainer_deployment_can_build_its_constructor_arguments(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] Sets the release and namespace environment variables, computes the trainer
        # component specs, builds the helm backend capability from them and a constructor context
        # (cell 0, worker 0, no GPUs). For every trainer-controller spec it computes the constructor
        # keyword arguments and asserts exactly the actor and critic controllers exist and none of
        # their keyword arguments names an inference controller.
        # Side effect: sets two environment variables (restored by the monkeypatch fixture).
        """A spec that asks this release for a pool it does not install aborts the pod before it ever serves."""
        monkeypatch.setenv(RELEASE_ENV_VAR, "miles-run-260813-trainer")
        monkeypatch.setenv(NAMESPACE_ENV_VAR, "rl")
        specs = compute_specs(self._args(tmp_path, deploy_component="trainer"))
        capability = compute_helm_backend_capability(specs=specs)
        context = WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

        kwargs_by_name = {
            spec.name: spec.ctor_kwargs(context) for spec in specs if spec.name.startswith("trainer-controller-")
        }

        assert sorted(kwargs_by_name) == ["trainer-controller-actor", "trainer-controller-critic"]
        assert all("inference_controller" not in kwargs for kwargs in kwargs_by_name.values())

    def test_a_subset_carries_exactly_the_workers_of_its_own_component(self, tmp_path):
        # [FUNCTION DOC] Counts the specs of the all component per deploy component. For each of
        # primary, trainer and inference, the specs of that component (ignoring the registration
        # reporter) all carry that component and their number equals the count in the all listing.
        """A worker two subsets carry is installed twice, and one no subset carries is never installed at all."""
        whole = Counter(spec.deploy_component for spec in compute_specs(self._args(tmp_path, deploy_component="all")))

        for component in (DeployComponent.PRIMARY, DeployComponent.TRAINER, DeployComponent.INFERENCE):
            carried = [
                spec
                for spec in compute_specs(self._args(tmp_path, deploy_component=component.value))
                if spec.name != INFERENCE_REGISTRATION_REPORTER_POOL_ID
            ]

            assert {spec.deploy_component for spec in carried} == {component}
            assert len(carried) == whole[component]

    def test_a_named_trainer_deployment_holds_the_one_trainer_its_arguments_describe(self, tmp_path):
        # [FUNCTION DOC] With the trainer component, deploy instance id a-actor, no critic and an
        # encoded megatron config for a, the spec names are exactly trainer-controller-a-actor and
        # trainer-engine-a-actor.
        """Its arguments give one model's configuration, so the instance names the release and selects nothing."""
        args = self._args(
            tmp_path,
            deploy_component="trainer",
            deploy_instance_id="a-actor",
            use_critic=False,
            megatron_config=encode_megatron_config("a"),
        )

        specs = compute_specs(args)

        assert [spec.name for spec in specs] == ["trainer-controller-a-actor", "trainer-engine-a-actor"]

    def test_an_inference_deployment_holds_the_engines_and_the_one_reporter(self, tmp_path):
        # [FUNCTION DOC] With the inference component and a controller address, the spec names are
        # exactly inference-registration-reporter and inference-engine-inference-0-0.
        """An engine release carries no controller and no router; it only announces the engines it launches."""
        specs = compute_specs(
            self._args(
                tmp_path,
                deploy_component="inference",
                inference_controller_addr="controller:8000",
            )
        )

        assert [spec.name for spec in specs] == ["inference-registration-reporter", "inference-engine-inference-0-0"]

    @pytest.mark.parametrize("component", ["all", "primary", "trainer"])
    def test_only_an_inference_deployment_carries_a_reporter(self, tmp_path, component):
        # [FUNCTION DOC] Parametrized over all, primary and trainer: none of the specs is the
        # inference registration reporter.
        """A reporter beside the controller it reports into would register a deployment into itself."""
        specs = compute_specs(self._args(tmp_path, deploy_component=component))

        assert "inference-registration-reporter" not in [spec.name for spec in specs]
