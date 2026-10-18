# [MODULE DOC] Tests for the helm values builder (build_values), which turns worker specs plus a
# launch plan into the run section of the chart values. Covers: single top-level key, run id,
# orchestrator command and restart stamp handling (stamp only for stamped components), which
# section each spec lands in (static workers, inference engines, trainer engines), disabled
# (zero cell) specs, replica and size computation from cells and nodes, per cell static worker
# hostnames, port renaming, trainer rank supervision flags, environment handling, served worker
# bootstrap flags, object names, releases without the orchestration script, determinism across
# relaunches, decided static addresses, and agreement between the rendered entries and the
# scheduling math the worker provider relies on (ranks per pod, pods per cell).
import pytest
import yaml
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import (
    LAYOUT,
    engine,
    router,
    session_client,
    session_server,
    trainer,
)
from tests.fast.utils.workers.worker_provider.kubernetes.run_specs import _RELEASE, make_engine_spec, make_trainer_spec

from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.values import builder
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.worker_spec import BaseWorkerSpec

_STAMP = "2026-08-12T09:00:00+00:00"


def _stamped(*components: str) -> LaunchPlan:
    # [FUNCTION DOC] HELPER: Returns a copy of the shared sample launch plan with the restart
    # stamp set to a fixed timestamp and the stamped components set to the given component names.
    return LAYOUT.model_copy(update=dict(restart_at=_STAMP, stamped_components=frozenset(components)))


class TestBuildValues:
    # [CLASS DOC] General tests of the values builder output. All use the shared sample launch
    # plan (release r, namespace rl) unless noted, and read the run section of the values.
    def test_keeps_everything_it_computes_under_one_key(self):
        # [FUNCTION DOC] With no specs, the top level keys of the values are exactly the run key.
        """The infra values own every other top-level key, and a run must not be able to reach them."""
        assert list(build_values([], LAYOUT).as_values()) == ["run"]

    def test_records_the_run_id_the_run_directory_is_named_after(self):
        # [FUNCTION DOC] The run section's id equals the plan's run id.
        """The wrapper writes its exit file under this id, and the launcher polls the same path."""
        assert build_values([], LAYOUT).as_values()["run"]["id"] == "260101-000000-000"

    def test_hands_the_orchestrator_the_command_that_starts_the_run(self):
        # [FUNCTION DOC] The orchestrator entry equals a mapping with only the plan's command.
        """Nothing else launches training, so an empty orchestrator command is a run that never starts."""
        built = build_values([], LAYOUT).as_values()

        assert built["run"]["orchestrator"] == {"command": ["python", "train.py"]}

    def test_hands_the_orchestrator_the_restart_stamp_the_plan_carries(self):
        # [FUNCTION DOC] When the orchestrator component is stamped, the orchestrator entry has a
        # restart-at field equal to the stamp.
        """This is the only path from the plan to the annotation whose change rolls the orchestrator pod."""
        built = build_values([], _stamped(naming.ORCHESTRATOR_COMPONENT)).as_values()

        assert built["run"]["orchestrator"]["restartAt"] == _STAMP

    def test_an_orchestrator_the_launch_does_not_stamp_is_given_no_stamp(self):
        # [FUNCTION DOC] When only another component is stamped, the orchestrator entry has no
        # restart-at field.
        """A launch that carries a stamp for another component must not roll the orchestrator pod as well."""
        built = build_values([], _stamped("rollout-executor")).as_values()

        assert "restartAt" not in built["run"]["orchestrator"]

    def test_an_orchestrator_of_a_launch_that_restarts_nothing_is_given_no_stamp(self):
        # [FUNCTION DOC] With the plain plan (no stamp), the orchestrator entry has no restart-at
        # field.
        """A stamp on an ordinary launch would roll the pods of every run that relaunches."""
        built = build_values([], LAYOUT).as_values()

        assert "restartAt" not in built["run"]["orchestrator"]

    def test_files_every_spec_under_its_own_section(self):
        # [FUNCTION DOC] A router, engine and trainer land respectively in the static workers,
        # inference engines and trainer engines sections, each as a single entry with the spec name.
        """Each section renders a different workload kind, so a misfiled spec deploys the wrong shape."""
        built = build_values([router(), engine(), trainer()], LAYOUT).as_values()["run"]

        assert [entry["name"] for entry in built["staticWorkers"]] == ["inference-router-0"]
        assert [entry["name"] for entry in built["inferenceEngines"]] == ["inference-engine-0-0"]
        assert [entry["name"] for entry in built["trainerEngines"]] == ["trainer-engine-actor"]

    def test_deploys_nothing_for_a_spec_a_run_has_turned_off(self):
        # [FUNCTION DOC] A session server with zero cells next to a router: the static workers
        # contain only the router.
        """A session server the run disabled asks for no cells, and a pod for it would serve nobody."""
        built = build_values([router(), session_server(num_cells=0)], LAYOUT).as_values()["run"]

        assert [entry["name"] for entry in built["staticWorkers"]] == ["inference-router-0"]

    def test_leaves_the_addresses_of_a_disabled_spec_out_of_its_readers_commands(self):
        # [FUNCTION DOC] With only a zero cell session server, the static workers list is empty.
        # KNOWN ISSUE: the name promises that a disabled spec's addresses are left out of reader
        # commands, but no reader spec is built, so only the empty static workers list is checked.
        """A url of a pod nobody deploys would send every request of the run into a black hole."""
        built = build_values([session_server(num_cells=0)], LAYOUT).as_values()["run"]

        assert built["staticWorkers"] == []

    def test_gives_a_single_cell_pool_the_replica_count_it_asks_for(self):
        # [FUNCTION DOC] A one cell engine and a one cell trainer each get replicas equal to 1 (an
        # explicit value, not omitted).
        """An omitted count reads as an unmanaged field, and elastic then refuses to scale the pool_id at all."""
        built = build_values([engine(num_cells=1, gpus_per_engine=8), trainer(num_cells=1)], LAYOUT).as_values()

        assert built["run"]["inferenceEngines"][0]["replicas"] == 1
        assert built["run"]["trainerEngines"][0]["replicas"] == 1

    def test_gives_a_static_worker_one_instance_per_cell(self):
        # [FUNCTION DOC] A session server with two cells has replicas equal to 2.
        """Two session servers are two addresses, and one statefulset pod could answer only one of them."""
        built = build_values([session_server(num_cells=2)], LAYOUT).as_values()["run"]

        assert built["staticWorkers"][0]["replicas"] == 2

    def test_addresses_each_static_cell_by_its_own_hostname(self):
        # [FUNCTION DOC] A session client after a two cell session server: the client's command
        # ends with a comma separated list of one http url per server cell, each built from the
        # pod hostname (release, miles run, pool name, ordinal), the headless service name and
        # port 8000.
        """The rollout executor resolves a list of session server urls, which pod zero cannot all be."""
        built = build_values([session_server(num_cells=2), session_client()], LAYOUT).as_values()["run"]
        command = built["staticWorkers"][1]["command"]

        assert command[-1] == (
            "http://r-miles-run-session-server-0.r-miles-run-session-server:8000,"
            "http://r-miles-run-session-server-1.r-miles-run-session-server:8000"
        )

    def test_sizes_an_engine_group_by_its_nodes(self):
        # [FUNCTION DOC] The default engine (2 cells, 32 GPUs per engine) renders replicas 2 and
        # group size 4 (one pod per 8-GPU node), and per-pod resource limits of 8 GPUs.
        """A 32-gpu engine spans four 8-gpu nodes, which is four pods in one group."""
        built = build_values([engine()], LAYOUT).as_values()["run"]["inferenceEngines"][0]

        assert (built["replicas"], built["size"]) == (2, 4)
        assert built["resources"] == {"limits": {"nvidia.com/gpu": 8}}

    def test_binds_a_pod_own_listener_on_every_interface(self):
        # [FUNCTION DOC] The router command's host argument is rewritten to the all interfaces
        # address (0.0.0.0) instead of a computed host.
        """A pod has its own network namespace, so it need not guess which address it will get."""
        worker = build_values([router()], LAYOUT).as_values()["run"]["staticWorkers"][0]

        assert worker["command"][worker["command"].index("--host") + 1] == "0.0.0.0"

    def test_supervises_one_trainer_rank_per_gpu_in_the_pod(self):
        # [FUNCTION DOC] The default trainer (2 cells of 16 workers, 8 GPUs per node): the command
        # asks the supervisor for 8 subprocesses, with replicas 2 and size 2.
        """A trainer worker is a rank, and a pod holding eight gpus has to run eight of them."""
        built = build_values([trainer()], LAYOUT).as_values()["run"]["trainerEngines"][0]

        assert built["command"][built["command"].index("--num-subprocesses") + 1] == "8"
        assert (built["replicas"], built["size"]) == (2, 2)

    def test_passes_the_run_argv_to_every_served_worker(self):
        # [FUNCTION DOC] A served trainer's command ends with the plan's worker argv words.
        """A pod rebuilds the run's arguments itself rather than receiving a pickled namespace."""
        built = build_values([trainer()], LAYOUT).as_values()["run"]["trainerEngines"][0]

        assert built["command"][-2:] == ["--foo", "bar"]

    def test_serves_a_single_rank_trainer_without_a_supervisor(self):
        # [FUNCTION DOC] A trainer with one cell of one worker has no subprocess count flag.
        """One rank needs no supervisor, and adding one would bury its logs a level deeper."""
        built = build_values([trainer(num_cells=1, gpus_per_cell=1)], LAYOUT).as_values()["run"]

        assert "--num-subprocesses" not in built["trainerEngines"][0]["command"]

    def test_renames_ports_kubernetes_would_reject(self):
        # [FUNCTION DOC] Engine port names are rewritten: underscores become hyphens and names are
        # truncated to fifteen characters, giving primary, dist-init and engine-info-boo.
        """A port name may not hold an underscore and stops at fifteen characters."""
        built = build_values([engine()], LAYOUT).as_values()["run"]["inferenceEngines"][0]

        assert [port["name"] for port in built["ports"]] == ["primary", "dist-init", "engine-info-boo"]

    def test_a_served_pool_carries_no_environment_of_its_own(self):
        # [FUNCTION DOC] A served trainer entry has no env field, and its command carries the
        # specs flag (the pod rebuilds its env from the specs).
        """A trainer worker's env depends on which worker it is, which no values entry can know."""
        built = build_values([trainer()], LAYOUT).as_values()["run"]["trainerEngines"][0]

        assert "env" not in built
        assert "--specs" in built["command"]

    def test_refuses_a_command_pool_whose_environment_depends_on_its_rank(self):
        # [FUNCTION DOC] An engine whose env function reads the cell index: building raises an
        # assertion error saying it builds its environment out of the cell and worker.
        """Such a value would be rendered once for the whole pool and then be wrong for every cell but one."""
        spec = engine().model_copy(update={"env_var": lambda ctx: {"HOME_OF": str(ctx.cell_index)}})

        with pytest.raises(AssertionError, match="builds its environment out of the cell and worker"):
            build_values([spec], LAYOUT).as_values()

    def test_keeps_the_environment_a_command_pool_states_outright(self):
        # [FUNCTION DOC] The default engine's constant env var mapping appears as the entry's env.
        """An engine's env is the same for every cell, so the chart may carry it."""
        built = build_values([engine()], LAYOUT).as_values()["run"]["inferenceEngines"][0]

        assert built["env"] == {"NVSHMEM_DISABLE_NCCL": "1"}

    def test_omits_an_empty_environment(self):
        # [FUNCTION DOC] A router with empty env has no env field in its entry.
        """An empty env block in the values would render an empty env list into the pod."""
        assert "env" not in build_values([router()], LAYOUT).as_values()["run"]["staticWorkers"][0]

    def test_rejects_a_cell_that_is_not_a_whole_number_of_nodes(self):
        # [FUNCTION DOC] An engine spec edited to 3 workers per cell with 5 GPU slots per worker
        # (15 GPUs, not a multiple of 8): building raises an assertion error saying whole number.
        """Such a cell would leave its trailing ranks with no pod to run in."""
        spec = engine().model_copy(
            update={
                "scheduling": engine().scheduling.model_copy(
                    update={"num_workers_per_cell": 3, "num_gpu_slots_per_worker": 5}
                )
            }
        )

        with pytest.raises(AssertionError, match="whole number"):
            build_values([spec], LAYOUT).as_values()


class TestServedWorkerBootstrap:
    # [CLASS DOC] Tests of the command line the builder renders for served (trainer) pools, using
    # a one cell, one worker trainer unless noted.
    def test_names_the_spec_the_pod_has_to_rebuild(self):
        # [FUNCTION DOC] The word after the pool id flag in the trainer command is the trainer
        # pool name.
        """A served pod constructs its worker from the spec, which its command is the only record of."""
        command = build_values([trainer(num_cells=1, gpus_per_cell=1)], LAYOUT).as_values()["run"]["trainerEngines"][
            0
        ]["command"]

        assert command[command.index("--pool-id") + 1] == "trainer-engine-actor"

    def test_names_the_spec_table_the_pod_rebuilds_the_run_from(self):
        # [FUNCTION DOC] The word after the specs flag is the dotted path of the compute specs
        # from argv function in the specs entrypoint module.
        """The pod constructs its own worker, so it needs the same table the launcher rendered from."""
        command = build_values([trainer(num_cells=1, gpus_per_cell=1)], LAYOUT).as_values()["run"]["trainerEngines"][
            0
        ]["command"]

        assert command[command.index("--specs") + 1] == "miles.ray.specs.entrypoint.compute_specs_from_argv"

    def test_supervises_a_shared_pod_into_the_ranks_its_spec_packs_there(self):
        # [FUNCTION DOC] A one cell trainer with 8 workers: the subprocess count flag value is 8.
        """The ranks of one pod run the same command, and only the supervisor knows how many to start."""
        command = build_values([trainer(num_cells=1, gpus_per_cell=8)], LAYOUT).as_values()["run"]["trainerEngines"][
            0
        ]["command"]

        assert command[command.index("--num-subprocesses") + 1] == "8"

    def test_keeps_the_entrypoint_flags_ahead_of_the_run_argv(self):
        # [FUNCTION DOC] In the command, the specs flag comes before the double dash separator
        # that precedes the run's own argv.
        """Everything after the separator belongs to the run, so a flag placed there would never be read."""
        command = build_values([trainer(num_cells=1, gpus_per_cell=1)], LAYOUT).as_values()["run"]["trainerEngines"][
            0
        ]["command"]

        assert command.index("--specs") < command.index("--")


class TestObjectNames:
    # [CLASS DOC] Tests of the object names the builder computes for the chart (the chart derives
    # none itself).
    def test_every_pool_entry_carries_the_object_name_the_chart_must_render(self):
        # [FUNCTION DOC] Each router, engine and trainer entry has an object name made of the
        # release r, the miles run infix and the pool name.
        """The chart computes no names, so an entry without one renders an object called nothing."""
        built = build_values([router(), engine(), trainer()], LAYOUT).as_values()["run"]

        assert built["staticWorkers"][0]["objectName"] == "r-miles-run-inference-router-0"
        assert built["inferenceEngines"][0]["objectName"] == "r-miles-run-inference-engine-0-0"
        assert built["trainerEngines"][0]["objectName"] == "r-miles-run-trainer-engine-actor"

    def test_the_fixed_components_are_named_whether_or_not_they_are_enabled(self):
        # [FUNCTION DOC] With no specs, the object names mapping equals seven fixed entries
        # (orchestrator, platform read, platform read delete, mooncake master, colocate pairing,
        # uninstall, uninstall manifest), each prefixed with the release and miles run infix.
        """A schema-required field the launcher only sometimes writes would refuse half the runs."""
        assert build_values([], LAYOUT).as_values()["run"]["objectNames"] == {
            "orchestrator": "r-miles-run-orchestrator",
            "platformRead": "r-miles-run-platform-read",
            "platformReadDelete": "r-miles-run-platform-read-delete",
            "mooncakeMaster": "r-miles-run-mooncake-master",
            "colocatePairing": "r-miles-run-colocate-pairing",
            "uninstall": "r-miles-run-uninstall",
            "uninstallManifest": "r-miles-run-uninstall-manifest",
        }

    def test_leaves_the_self_uninstall_section_to_the_chart(self):
        # [FUNCTION DOC] With the plain plan, the run section has no auto uninstall key.
        """The chart default arms it and a user values file may turn it off; the launcher has no say."""
        assert "autoUninstall" not in build_values([], LAYOUT).as_values()["run"]


class TestADeploymentWithoutTheOrchestrationScript:
    # [CLASS DOC] Tests of a release (such as a trainer only release) whose plan has an empty
    # state file, empty orchestrator command and no mooncake plan.
    @staticmethod
    def _plan() -> LaunchPlan:
        # [FUNCTION DOC] HELPER: Returns a copy of the sample plan with state file emptied,
        # orchestrator command emptied and mooncake plan removed.
        return LAYOUT.model_copy(update=dict(state_file="", orchestrator_command=[], mooncake_plan=None))

    def test_renders_no_orchestrator_command_so_the_chart_installs_no_orchestrator(self):
        # [FUNCTION DOC] The orchestrator entry equals a mapping with an empty command list.
        """A trainer release starts no training, so an orchestrator pod in it would run a second run."""
        built = build_values([trainer()], self._plan()).as_values()["run"]

        assert built["orchestrator"] == {"command": []}

    def test_names_no_exit_file_because_this_release_reaches_no_verdict(self):
        # [FUNCTION DOC] The run section has no state file key.
        """Only the release carrying the orchestration script has a training outcome to publish."""
        built = build_values([trainer()], self._plan()).as_values()["run"]

        assert "stateFile" not in built

    def test_disarms_the_self_uninstall_job(self):
        # [FUNCTION DOC] The auto uninstall entry equals a mapping with enabled set to false.
        """Nothing here ever finishes, so an armed uninstall could only fire on a healthy deployment."""
        built = build_values([trainer()], self._plan()).as_values()["run"]

        assert built["autoUninstall"] == {"enabled": False}

    def test_refuses_an_exit_file_without_the_script_that_writes_it(self):
        # [FUNCTION DOC] A plan with a state file but an empty orchestrator command makes building
        # raise an assertion error mentioning the orchestration script (raised on build itself,
        # before rendering to values).
        """A watched file nobody writes makes the launcher wait for a verdict that never comes."""
        with pytest.raises(AssertionError, match="orchestration script"):
            build_values([trainer()], LAYOUT.model_copy(update=dict(orchestrator_command=[])))

    def test_refuses_the_script_without_the_exit_file_it_publishes(self):
        # [FUNCTION DOC] The reverse: a command but an empty state file also raises an assertion
        # error mentioning the orchestration script.
        """The launcher learns the run's outcome by reading that file, so the pair is inseparable."""
        with pytest.raises(AssertionError, match="orchestration script"):
            build_values([trainer()], LAYOUT.model_copy(update=dict(state_file="")))


class TestRelaunching:
    # [CLASS DOC] Determinism test of the builder.
    def test_relaunching_the_same_run_writes_byte_identical_values(self):
        # [FUNCTION DOC] Building values twice from the same specs and plan and dumping each to
        # yaml with sorted keys gives identical text.
        """helm upgrade replaces an object in place only while its name is unchanged."""
        specs = [router(), engine(), trainer()]

        first = yaml.safe_dump(build_values(specs, LAYOUT).as_values(), sort_keys=True)
        second = yaml.safe_dump(build_values(specs, LAYOUT).as_values(), sort_keys=True)

        assert first == second


class TestDecidedAddresses:
    # [CLASS DOC] Tests of addresses the launcher decides ahead of time for static pools.
    def test_a_command_reading_another_pool_gets_the_final_host_and_port(self):
        # [FUNCTION DOC] A client reading a one cell session server pool gets the final server url
        # (pod hostname, headless service, port 8000) as the last element of its command.
        """Nothing expands a placeholder inside a container command, so the address is baked in."""
        built = build_values([session_server(num_cells=1), session_client()], LAYOUT).as_values()["run"]

        assert built["staticWorkers"][1]["command"][-1] == (
            "http://r-miles-run-session-server-0.r-miles-run-session-server:8000"
        )

    def test_only_a_statically_addressed_pool_is_in_the_address_book(self):
        # [FUNCTION DOC] The builder's private address computation for release r over a router,
        # engine and trainer returns keys equal to only the router name (sorted).
        """A pool's pods are named by its workload controller, so the launcher decides no address for it."""
        specs = [router(), engine(), trainer()]

        assert sorted(builder._compute_addresses(specs, "r")) == ["inference-router-0"]


class TestRanksPerPod:
    # [CLASS DOC] Consistency tests between ranks launched per pod in the rendered command and
    # the scheduling spec's workers per pod.
    def test_the_supervisor_starts_the_ranks_the_provider_will_look_for(self):
        # [FUNCTION DOC] For a trainer spec with 24 workers per cell and 8 GPUs per node, the
        # launched workers per pod equals the spec's workers per pod.
        """The provider fans a pod out into the ranks its own command started, so the two cannot drift."""
        spec = make_trainer_spec(num_workers_per_cell=24, num_gpus_per_node=8)

        assert _launched_workers_per_pod(spec) == spec.scheduling.workers_per_pod()

    def test_an_engine_pod_runs_one_command_and_therefore_holds_one_rank(self):
        # [FUNCTION DOC] For the default engine test spec, both the launched workers per pod and
        # the spec's workers per pod equal 1.
        """An engine pod is a single server process spanning its gpus, whatever its spec counts as a worker."""
        spec = make_engine_spec()

        assert _launched_workers_per_pod(spec) == 1
        assert spec.scheduling.workers_per_pod() == 1


class TestPodsPerCell:
    # [CLASS DOC] Consistency tests between the rendered group size and the scheduling spec's
    # pods per cell.
    def test_the_chart_deploys_the_pods_the_provider_expects_to_observe(self):
        # [FUNCTION DOC] For a trainer spec with 24 workers per cell and 8 GPUs per node, the
        # rendered pods per cell equals the spec's pods per cell.
        """A cell rendered into fewer pods than the provider counts would never look complete."""
        spec = make_trainer_spec(num_workers_per_cell=24, num_gpus_per_node=8)

        assert _rendered_pods_per_cell(spec) == spec.scheduling.pods_per_cell()

    def test_a_cell_of_one_pod_is_rendered_without_a_size(self):
        # [FUNCTION DOC] For the default engine test spec, the rendered pods per cell (size
        # defaulting to 1 when absent) equals the spec's pods per cell, which is 1.
        """The chart's default is a single pod, and both sides have to read that absence the same way."""
        spec = make_engine_spec()

        assert _rendered_pods_per_cell(spec) == spec.scheduling.pods_per_cell() == 1


def _launched_workers_per_pod(spec: BaseWorkerSpec) -> int:
    # [FUNCTION DOC] HELPER: Renders the spec's entry and returns the integer after its
    # subprocess count flag in the command, or 1 when the command has no such flag.
    command = _rendered_entry(spec)["command"]
    if "--num-subprocesses" not in command:
        return 1
    return int(command[command.index("--num-subprocesses") + 1])


def _rendered_pods_per_cell(spec: BaseWorkerSpec) -> int:
    # [FUNCTION DOC] HELPER: Returns the size field of the spec's rendered entry, defaulting to 1
    # when the entry has no size.
    return _rendered_entry(spec).get("size", 1)


def _rendered_entry(spec: BaseWorkerSpec) -> dict:
    # [FUNCTION DOC] HELPER: Builds values for the single spec with a local launch plan (frozen
    # run id, state file path, the imported release constant, namespace rl, the train script as
    # orchestrator command, empty worker argv) and returns the first entry, searched in the
    # trainer engines, inference engines and static workers sections in that order, whose pool id
    # equals the spec name. Raises a stop iteration error if none matches.
    values = build_values(
        [spec],
        LaunchPlan(
            run_id="260101-000000-000",
            state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
            release=_RELEASE,
            namespace="rl",
            orchestrator_command=["python", "train.py"],
            worker_argv=[],
        ),
    ).as_values()
    return next(
        entry
        for section in ("trainerEngines", "inferenceEngines", "staticWorkers")
        for entry in values["run"][section]
        if entry["poolId"] == spec.name
    )
