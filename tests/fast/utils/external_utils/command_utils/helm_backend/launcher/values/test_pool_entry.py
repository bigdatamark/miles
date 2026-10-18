# [MODULE DOC] Tests for building a single pool entry of the helm values: refusal of specs with
# no launch mechanism or no cells, the leader address placeholder for master mode ports, base GPU
# sentinel rendering inside pool environments, validation of port names, the prepare command
# wrapper (only trainers get a bash wrapper, quoting, whole-node requirement), the restart stamp
# (only pools the launch replaces and whose template renders an annotation), and the service
# account chosen from a pool's platform access level. Module constants define a restarted plan
# stamped for the rollout executor pool and a plan with a prepare command for the trainer.
import pydantic
import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import (
    LAYOUT,
    engine,
    router,
    session_server,
    trainer,
)

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.utils.external_utils.command_utils.helm_backend.launcher.values import placeholders, pool_entry
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import PortEntry
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.types import PlatformAccess
from miles.utils.workers.worker_spec import BaseWorkerSpec, SchedulingSpec

STAMP = "2026-08-12T09:00:00+00:00"

RESTARTED_LAYOUT = LAYOUT.model_copy(
    update=dict(restart_at=STAMP, stamped_components=frozenset({ROLLOUT_EXECUTOR_POOL_ID}))
)

PREPARE_CMD = "mkdir -p /scratch/dataset && rsync -a /cluster-storage/dataset/ /scratch/dataset"

PREPARE_LAYOUT = LaunchPlan(
    run_id="260101-000000-000",
    state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
    release="r",
    namespace="rl",
    orchestrator_command=["python", "train.py"],
    worker_argv=["--foo", "bar"],
    prepare_cmd={"trainer": PREPARE_CMD},
)


def _prepared(section: str, spec: BaseWorkerSpec) -> list[str]:
    # [FUNCTION DOC] HELPER: Builds values for the single spec with the plan that carries a
    # prepare command for the trainer, and returns the command of the first entry in the named
    # section of the run values.
    return build_values([spec], PREPARE_LAYOUT).as_values()["run"][section][0]["command"]


class TestBuildEntry:
    # [CLASS DOC] Tests of the pool entry builder function.
    def test_refuses_a_spec_with_no_launch_mechanism(self):
        # [FUNCTION DOC] A base spec that is neither a command spec nor a serve spec makes
        # building an entry raise an assertion error saying it is neither launched by a command
        # nor served over rpc.
        """A pool without a command or RPC server cannot launch a worker."""
        spec = BaseWorkerSpec(
            name="unsupported-worker",
            port_infos=[],
            env_var=lambda context: {},
            scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        )

        with pytest.raises(AssertionError, match="neither launched by a command nor served over rpc"):
            pool_entry.build_entry(spec, plan=LAYOUT, addresses={})

    def test_refuses_to_build_an_entry_for_a_spec_that_wants_no_cell(self):
        # [FUNCTION DOC] A zero cell session server makes building an entry raise an assertion
        # error mentioning cells.
        """Every entry renders at least one pod, so a zero-cell spec reaching conversion is a silent deploy."""
        with pytest.raises(AssertionError, match="cells"):
            pool_entry.build_entry(session_server(num_cells=0), plan=LAYOUT, addresses={})

    def test_points_a_master_port_at_the_group_leader(self):
        # [FUNCTION DOC] The default engine's dist init address argument equals the leader
        # address placeholder plus port 9000.
        """A rank cannot know its leader's address until it is scheduled, but kubelet does."""
        command = build_values([engine()], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert command[command.index("--dist-init-addr") + 1] == f"{placeholders.LEADER_ADDRESS_PLACEHOLDER}:9000"

    def test_renders_a_sub_node_gpu_sentinel_in_the_pool_environment(self) -> None:
        # [FUNCTION DOC] A 4-GPU engine sharing a colocated trainer node with an env var derived
        # from the first GPU id: the entry's env value is the base GPU id placeholder, not a number.
        """An environment value derived from the assigned base GPU must remain a kubelet variable."""
        spec = engine(num_cells=2, gpus_per_engine=4).model_copy(
            update={"env_var": lambda context: {"BASE_GPU": str(context.gpu_ids[0])}}
        )
        layout = LAYOUT.model_copy(update={"colocate": True})

        entry = build_values([spec, trainer(num_cells=1, gpus_per_cell=8)], layout).as_values()["run"][
            "inferenceEngines"
        ][0]

        assert entry["env"]["BASE_GPU"] == placeholders._BASE_GPU_ID_PLACEHOLDER


class TestPortEntry:
    # [CLASS DOC] Tests of port entry name validation.
    @pytest.mark.parametrize("name", ["9000", "bad_name", "trailing-"])
    def test_refuses_a_name_that_is_not_an_iana_service_name(self, name: str):
        # [FUNCTION DOC] Parametrized over an all-digit name, a name with an underscore and a name
        # ending in a hyphen: constructing a port entry raises a validation error about the string
        # not matching the pattern.
        """The chart writes these straight into containerPort names, and the api server rejects the rest."""
        with pytest.raises(pydantic.ValidationError, match="String should match pattern"):
            PortEntry(name=name, port=9000)

    def test_accepts_the_names_the_launcher_computes(self):
        # [FUNCTION DOC] A fifteen character hyphenated name (as computed by the launcher) is
        # accepted and kept unchanged.
        """A name compute_port_name produced must never be turned away by the model that carries it."""
        assert PortEntry(name="engine-info-boo", port=9000).name == "engine-info-boo"


class TestPrepareCmd:
    # [CLASS DOC] Tests of the prepare command (a node-local setup shell command) wrapping.
    def test_a_trainer_runs_the_command_before_it_starts(self):
        # [FUNCTION DOC] The default trainer's command starts with bash and the dash c option, and
        # the script string begins with the prepare command, a logical and, and the exec keyword.
        """Every trainer rank reads the dataset every step, and the shared filesystem cannot serve that."""
        command = _prepared("trainerEngines", trainer())

        assert command[:2] == ["bash", "-c"]
        assert command[2].startswith(f"{PREPARE_CMD} && exec ")

    def test_an_engine_is_launched_without_the_command(self):
        # [FUNCTION DOC] The engine's command has no bash wrapper and does not mention the
        # prepared scratch path.
        """An sglang engine loads weights the trainer sends it, so the copy would move gigabytes nothing reads."""
        command = _prepared("inferenceEngines", engine())

        assert command[:2] != ["bash", "-c"]
        assert "/scratch/dataset" not in " ".join(command)

    def test_a_static_worker_is_launched_without_the_command(self):
        # [FUNCTION DOC] Same assertions for the router in the static workers section.
        """A router holds no data of its own, and the copy would only delay the run's first request."""
        command = _prepared("staticWorkers", router())

        assert command[:2] != ["bash", "-c"]
        assert "/scratch/dataset" not in " ".join(command)

    def test_a_trainer_keeps_the_command_it_would_have_run(self):
        # [FUNCTION DOC] The wrapped script string ends with the run's worker argv words joined by
        # a space.
        """The preparation prefixes the launch; a rewritten command would start the wrong worker."""
        command = _prepared("trainerEngines", trainer())

        assert command[2].endswith(" ".join(["--foo", "bar"]))

    def test_quotes_the_command_it_wraps(self):
        # [FUNCTION DOC] With a worker argv containing a json argument with spaces, the wrapped
        # script ends with that argument shell quoted in single quotes.
        """An argument carrying json or spaces would otherwise be re-split by the shell that runs the pair."""
        spec = trainer(num_cells=1, gpus_per_cell=8)
        layout = PREPARE_LAYOUT.model_copy(update={"worker_argv": ["--kwargs", '{"a": 1}']})

        command = build_values([spec], layout).as_values()["run"]["trainerEngines"][0]["command"]

        assert command[2].endswith("""--kwargs \'{"a": 1}\'""")

    def test_refuses_a_trainer_whose_pods_can_share_a_node(self):
        # [FUNCTION DOC] A trainer with 4 GPUs per cell (half a node) and a prepare command makes
        # building raise an assertion error saying pods can land on one node.
        """Two pods of one node would run the copy against the same node-local path at the same time."""
        with pytest.raises(AssertionError, match="can land on one node"):
            build_values([trainer(num_cells=1, gpus_per_cell=4)], PREPARE_LAYOUT).as_values()

    def test_accepts_a_trainer_that_takes_whole_nodes(self):
        # [FUNCTION DOC] A trainer with 8 GPUs per cell is accepted and gets the bash wrapper.
        """A pod holding every gpu of its node is the only pod there, so the copy cannot race itself."""
        command = _prepared("trainerEngines", trainer(num_cells=1, gpus_per_cell=8))

        assert command[:2] == ["bash", "-c"]

    def test_a_run_that_prepares_nothing_leaves_every_command_alone(self):
        # [FUNCTION DOC] With the plain plan (no prepare command), the trainer command has no bash
        # wrapper.
        """Most runs read from the shared filesystem directly, and a bash wrapper would hide their exit codes."""
        command = build_values([trainer()], LAYOUT).as_values()["run"]["trainerEngines"][0]["command"]

        assert command[:2] != ["bash", "-c"]


class TestTheRestartStamp:
    # [CLASS DOC] Tests of which pool entries receive the plan's restart stamp.
    @staticmethod
    def _entry(spec: BaseWorkerSpec, *, plan: LaunchPlan) -> dict:
        # [FUNCTION DOC] HELPER: Builds values for the single spec with the plan and returns the
        # first entry of the static workers section.
        return build_values([spec], plan).as_values()["run"]["staticWorkers"][0]

    @staticmethod
    def _executor() -> BaseWorkerSpec:
        # [FUNCTION DOC] HELPER: Returns the router spec renamed to the rollout executor pool id.
        return router().model_copy(update={"name": ROLLOUT_EXECUTOR_POOL_ID})

    def test_a_pool_the_launch_replaces_carries_the_stamp(self):
        # [FUNCTION DOC] The executor entry under the stamped plan has a restart-at field equal to
        # the stamp.
        """This is the only path from the plan to the annotation whose change rolls the executor pod."""
        assert self._entry(self._executor(), plan=RESTARTED_LAYOUT)["restartAt"] == STAMP

    def test_a_pool_the_launch_does_not_replace_carries_none(self):
        # [FUNCTION DOC] The router (not stamped) under the same plan has no restart-at field.
        """Stamping any other pool would roll pods this launch promises to keep alive."""
        assert "restartAt" not in self._entry(router(), plan=RESTARTED_LAYOUT)

    def test_an_ordinary_launch_stamps_no_pool_at_all(self):
        # [FUNCTION DOC] The executor under the unstamped plan has no restart-at field.
        """A stamp appearing without a restart would roll the executor of every run that relaunches."""
        assert "restartAt" not in self._entry(self._executor(), plan=LAYOUT)

    def test_stamping_a_pool_whose_template_renders_no_annotation_is_refused(self):
        # [FUNCTION DOC] A plan stamping the engine pool name makes building raise an assertion
        # error saying the template renders a restart stamp (it does not).
        """The engine template carries no annotation, so a stamp there rolls nothing while the launch believes it did."""
        plan = LAYOUT.model_copy(update=dict(restart_at=STAMP, stamped_components=frozenset({"inference-engine-0-0"})))

        with pytest.raises(AssertionError, match="renders a restart stamp"):
            build_values([engine()], plan).as_values()

    def test_stamping_a_trainer_pool_is_refused(self):
        # [FUNCTION DOC] A plan stamping the trainer pool name raises the same assertion error.
        """A trainer pool a hot restart promises to keep alive must never be handed a stamp at all."""
        plan = LAYOUT.model_copy(update=dict(restart_at=STAMP, stamped_components=frozenset({"trainer-engine-actor"})))

        with pytest.raises(AssertionError, match="renders a restart stamp"):
            build_values([trainer()], plan).as_values()


class TestTheAccountAPoolRunsUnder:
    # [CLASS DOC] Tests of the service account name set on a pool entry from its platform access.
    def test_a_pool_that_only_reads_the_platform_gets_the_account_that_may_read_it(self):
        # [FUNCTION DOC] A session server with read access gets a service account name made of
        # the release, the miles run infix and platform read.
        """Only these workers reconcile against pods, and the namespace default cannot list one."""
        spec = session_server(num_cells=1).model_copy(update={"platform_access": PlatformAccess.READ})

        entry = build_values([spec], LAYOUT).as_values()["run"]["staticWorkers"][0]

        assert entry["serviceAccountName"] == "r-miles-run-platform-read"

    def test_a_pool_that_also_deletes_platform_pods_gets_the_account_that_may_delete_them(self):
        # [FUNCTION DOC] With read and delete access, the account name ends in platform read
        # delete instead.
        """A trainer controller suspends cells by deleting pods and must retain that existing capability."""
        spec = session_server(num_cells=1).model_copy(update={"platform_access": PlatformAccess.READ_DELETE})

        entry = build_values([spec], LAYOUT).as_values()["run"]["staticWorkers"][0]

        assert entry["serviceAccountName"] == "r-miles-run-platform-read-delete"

    def test_every_other_pool_stays_on_the_namespace_default(self):
        # [FUNCTION DOC] A session server with default access has no service account name field.
        """An engine talks to no api server, and an account it never needs is one it could misuse."""
        entry = build_values([session_server(num_cells=1)], LAYOUT).as_values()["run"]["staticWorkers"][0]

        assert "serviceAccountName" not in entry

    def test_refuses_a_pool_whose_template_renders_no_account_at_all(self):
        # [FUNCTION DOC] An engine given read access (its chart template ignores the account)
        # makes building raise an assertion error saying the template renders a service account.
        """The engine template ignores the key, so the pod would run on the default and 403 far from here."""
        spec = engine(num_cells=1, gpus_per_engine=8).model_copy(update={"platform_access": PlatformAccess.READ})

        with pytest.raises(AssertionError, match="renders a service account"):
            build_values([spec], LAYOUT).as_values()
