# [MODULE DOC] Tests for the miscellaneous values types of the helm launcher: the mapping from
# worker pool category to chart section, strict validation of the launch plan model (unknown
# fields rejected), resolution of the shared runs root directory from the chart defaults plus
# user infra values files (variable substitution of the namespace, rejection of unknown
# variables, trailing slash trimming, null section handling, later files overriding earlier ones,
# deep merge of sections), loading and validating infra values (unknown sections, volumes with
# zero or two sources, chart defaults), the Mooncake plan read from parsed args (backend choice,
# port parsing and its errors, missing kwargs), rewriting the Mooncake master address in an argv
# to the in-cluster service host, and (only when helm is installed) agreement of the service name
# with the actual chart render.
import json
from argparse import Namespace

import pytest
import yaml
from pydantic import ValidationError
from tests.fast.charts.utils import NAMESPACE, REPO_ROOT, RUN_RELEASE_NAME, objects_of_kind, render_run, requires_helm
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import engine, router, trainer

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    SECTION_OF_CATEGORY,
    InfraInfo,
    LaunchPlan,
    MooncakeInfo,
)

RUN_CHART_DIR = REPO_ROOT / "charts" / "miles-run"

MOONCAKE_INIT_KWARGS = {"master_server_address": "127.0.0.1:50051", "local_hostname": "localhost"}


class TestSectionOf:
    # [CLASS DOC] Tests of the category to chart section lookup table.
    def test_sends_a_pool_that_declares_nothing_to_the_static_workers(self):
        # [FUNCTION DOC] A router's category maps to the static workers section.
        """A router is never healed per cell, so a pool_id would only add indirection."""
        assert SECTION_OF_CATEGORY[router().category] == "staticWorkers"

    def test_keeps_a_single_cell_engine_a_pool(self):
        # [FUNCTION DOC] A one cell, 8-GPU engine's category maps to the inference engines section.
        """The provider recognises cells by LeaderWorkerSet labels, which a plain workload would not carry."""
        assert SECTION_OF_CATEGORY[engine(num_cells=1, gpus_per_engine=8).category] == "inferenceEngines"

    def test_sends_a_pool_that_declares_itself_an_engine_to_the_engines(self):
        # [FUNCTION DOC] The default engine's category maps to the inference engines section.
        """An engine group is restarted as a unit and so needs a pool_id."""
        assert SECTION_OF_CATEGORY[engine().category] == "inferenceEngines"

    def test_sends_a_pool_that_declares_itself_a_trainer_to_the_trainer_engines(self):
        # [FUNCTION DOC] The trainer's category maps to the trainer engines section.
        """Trainers are served over rpc rather than launched as a command, and heal per dp group."""
        assert SECTION_OF_CATEGORY[trainer().category] == "trainerEngines"


class TestLaunchPlan:
    # [CLASS DOC] Tests of launch plan model validation.
    def test_rejects_a_field_the_chart_would_never_read(self):
        # [FUNCTION DOC] Constructing a launch plan with an extra misspelled field (node local
        # root with a typo) raises a validation error, i.e. unknown fields are forbidden.
        """A misspelled plan field would otherwise be dropped, and the run would launch mis-shaped."""
        with pytest.raises(ValidationError):
            LaunchPlan(
                run_id="260101-000000-000",
                state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
                release="r",
                namespace="rl",
                orchestrator_command=[],
                worker_argv=[],
                node_local_rooot="/scratch",
            )


def _resolved(tmp_path, *files: dict, namespace: str = "myns") -> str:
    # [FUNCTION DOC] HELPER: Writes each given mapping as a yaml file named infra-N in tmp_path
    # (N the position), loads infra info from the run chart directory with those files in order,
    # and returns the shared root directory resolved for the namespace (default myns). Side
    # effect: file writes. Errors from loading or resolving propagate.
    paths = []
    for index, values in enumerate(files):
        path = tmp_path / f"infra-{index}.yaml"
        path.write_text(yaml.safe_dump(values))
        paths.append(str(path))
    return InfraInfo.shared_root(InfraInfo.load(RUN_CHART_DIR, paths), namespace=namespace)


class TestSharedRootOf:
    # [CLASS DOC] Tests of the shared runs root resolution; they read the real chart defaults.
    def test_falls_back_to_the_chart_defaults_when_no_file_says_otherwise(self, tmp_path):
        # [FUNCTION DOC] With an empty values file, the root is the chart default cluster storage
        # path with the namespace substituted and a miles data directory.
        """The chart's own values.yaml is the single source of these defaults; Python must not carry a copy."""
        assert _resolved(tmp_path, {}) == "/cluster-storage/myns/miles_data"

    def test_resolves_the_namespace_the_render_resolves(self, tmp_path):
        # [FUNCTION DOC] A runs root containing the namespace variable is resolved with the given
        # namespace substituted in place.
        """The launcher and the render must name one directory: a literal ${NAMESPACE} here would be another."""
        values = {"infra": {"paths": {"runsRoot": "/mnt/x/${NAMESPACE}/teamdata"}}}

        assert _resolved(tmp_path, values, namespace="tom-other") == "/mnt/x/tom-other/teamdata"

    def test_refuses_a_path_that_names_an_unknown_variable(self, tmp_path):
        # [FUNCTION DOC] A runs root with a misspelled variable raises a value error whose message
        # contains both the whole path and the unknown variable.
        """A misspelt variable left in place would give every namespace one literal directory to share."""
        values = {"infra": {"paths": {"runsRoot": "/mnt/x/${NAMESPCE}/teamdata"}}}

        with pytest.raises(ValueError) as error:
            _resolved(tmp_path, values)

        assert "/mnt/x/${NAMESPCE}/teamdata" in str(error.value)
        assert "${NAMESPCE}" in str(error.value)

    def test_is_the_container_path_the_values_file_names_and_nothing_derived(self, tmp_path):
        # [FUNCTION DOC] A plain runs root path is returned exactly as given.
        """The runs directory is one path on one mount; deriving it from a volume is what tied it to one disk."""
        values = {"infra": {"paths": {"runsRoot": "/mnt/x/teamdata"}}}

        assert _resolved(tmp_path, values) == "/mnt/x/teamdata"

    def test_a_trailing_slash_does_not_change_the_directory(self, tmp_path):
        # [FUNCTION DOC] A runs root with a trailing slash is returned without it.
        """The launcher joins the run id onto this, and a doubled separator would name a different file."""
        values = {"infra": {"paths": {"runsRoot": "/mnt/x/"}}}

        assert _resolved(tmp_path, values) == "/mnt/x"

    def test_a_nulled_section_leaves_the_launcher_with_nothing_to_resolve(self, tmp_path):
        # [FUNCTION DOC] A values file that sets the paths section to null raises an assertion
        # error mentioning the runs root key (null deletes the chart default).
        """helm deletes a key a values file nulls, so a launcher that re-defaulted it would pick another path."""
        values = {"infra": {"paths": None}}

        with pytest.raises(AssertionError, match="runsRoot"):
            _resolved(tmp_path, values)

    def test_the_last_file_that_names_a_value_wins(self, tmp_path):
        # [FUNCTION DOC] Two files naming different runs roots: the second file's value is used.
        """helm applies --values files in order, and the launcher must resolve the same run directory it renders."""
        first = {"infra": {"paths": {"runsRoot": "/mnt/a"}}}
        second = {"infra": {"paths": {"runsRoot": "/mnt/b"}}}

        assert _resolved(tmp_path, first, second) == "/mnt/b"

    def test_a_file_that_sets_one_key_keeps_the_rest_of_the_section(self, tmp_path):
        # [FUNCTION DOC] A file that sets only the runs root still leaves the chart's volumes: the
        # loaded volume names equal the single cluster storage volume (deep, not shallow, merge).
        """A shallow merge would drop the chart's volumes and leave the run with nothing mounted at all."""
        values = {"infra": {"paths": {"runsRoot": "/mnt/x"}}}
        loaded = InfraInfo.load(RUN_CHART_DIR, [_written(tmp_path, values)])

        assert [volume.name for volume in loaded.volumes] == ["cluster-storage"]


class TestLoadInfraValues:
    # [CLASS DOC] Tests of infra values loading and validation against the real chart.
    def test_rejects_a_section_the_charts_do_not_define(self, tmp_path):
        # [FUNCTION DOC] A misspelled volumes key raises a value error mentioning that key.
        """helm would reject the same file at install time, and failing here says so before anything runs."""
        values = {"infra": {"volumez": [{"name": "v"}]}}

        with pytest.raises(ValueError, match="volumez"):
            InfraInfo.load(RUN_CHART_DIR, [_written(tmp_path, values)])

    def test_rejects_a_volume_that_names_two_sources(self, tmp_path):
        # [FUNCTION DOC] A volume with both a host path and a persistent volume claim raises a
        # value error saying exactly one source is required.
        """The launcher reads the same file helm does, so it must refuse the shapes helm's schema refuses."""
        volume = {
            "name": "v",
            "hostPath": {"path": "/s"},
            "persistentVolumeClaim": {"claimName": "c"},
            "mounts": [{"mountPath": "/mnt"}],
        }

        with pytest.raises(ValueError, match="exactly one"):
            InfraInfo.load(RUN_CHART_DIR, [_written(tmp_path, {"infra": {"volumes": [volume]}})])

    def test_rejects_a_volume_that_names_no_source_at_all(self, tmp_path):
        # [FUNCTION DOC] A volume with only a name and mounts raises the same exactly one source
        # value error.
        """A sourceless volume is a mount kubernetes cannot satisfy, so its pods only ever stay pending."""
        values = {"infra": {"volumes": [{"name": "v", "mounts": [{"mountPath": "/mnt"}]}]}}

        with pytest.raises(ValueError, match="exactly one"):
            InfraInfo.load(RUN_CHART_DIR, [_written(tmp_path, values)])

    def test_reads_the_defaults_the_chart_ships(self, tmp_path):
        # [FUNCTION DOC] Loading with no user files yields the image repository of the default
        # miles image and the single default mount path of cluster storage. The tmp_path
        # fixture is requested but unused.
        """Every launch merges onto these, so a chart whose defaults stopped validating breaks every run."""
        loaded = InfraInfo.load(RUN_CHART_DIR, [])

        assert loaded.image.repository == "radixark/miles"
        assert [mount.mount_path for volume in loaded.volumes for mount in volume.mounts] == ["/cluster-storage"]


def _written(tmp_path, values: dict) -> str:
    # [FUNCTION DOC] HELPER: Writes the mapping as yaml to a file named infra in tmp_path and
    # returns its path as text. Side effect: file write.
    path = tmp_path / "infra.yaml"
    path.write_text(yaml.safe_dump(values))
    return str(path)


def _mooncake_args(*, object_store_backend: str = "mooncake", **overrides: object) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a parsed-args namespace with the given object store backend
    # (default mooncake) and store init kwargs equal to the default local master address and host
    # name overlaid with any keyword overrides.
    return Namespace(
        object_store_backend=object_store_backend,
        mooncake_store_init_kwargs={**MOONCAKE_INIT_KWARGS, **overrides},
    )


def _mooncake_argv(**overrides: object) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns an argv list: python, train.py, the object store backend flag
    # with value mooncake, the init kwargs flag followed by the json text of the default kwargs
    # overlaid with overrides, then a learning rate flag with value 1e-6.
    kwargs = {**MOONCAKE_INIT_KWARGS, **overrides}
    return [
        "python",
        "train.py",
        "--object-store-backend",
        "mooncake",
        "--mooncake-store-init-kwargs",
        json.dumps(kwargs),
        "--lr",
        "1e-6",
    ]


def _rewritten_kwargs(argv: list[str]) -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Finds the init kwargs flag in the argv and returns the json decoded
    # mapping from the argument that follows. Raises if the flag is absent.
    return json.loads(argv[argv.index("--mooncake-store-init-kwargs") + 1])


class TestMooncakePlanOfArgs:
    # [CLASS DOC] Tests of deriving a Mooncake plan from parsed args.
    def test_reads_the_backend_and_the_port_the_run_configured(self):
        # [FUNCTION DOC] Default mooncake args yield a plan (not none) with port 50051.
        """The chart publishes this port, so reading it wrong points every client at a closed socket."""
        plan = MooncakeInfo.plan_of_args(_mooncake_args())

        assert plan is not None
        assert plan.port == 50051

    def test_ignores_a_run_that_names_another_object_store(self):
        # [FUNCTION DOC] An object store backend other than mooncake yields no plan (none).
        """A run on the default backend must not gain a master StatefulSet it never talks to."""
        assert MooncakeInfo.plan_of_args(_mooncake_args(object_store_backend="ray")) is None

    def test_refuses_an_address_that_carries_no_port(self):
        # [FUNCTION DOC] A master address without a port raises an assertion error saying it
        # carries no port.
        """A bare host would render a Service on port zero, which nothing can dial."""
        with pytest.raises(AssertionError, match="carries no port"):
            MooncakeInfo.plan_of_args(_mooncake_args(master_server_address="127.0.0.1"))

    def test_refuses_an_address_with_a_non_numeric_port(self) -> None:
        # [FUNCTION DOC] A master address whose port is not a number raises a value error.
        """A non-numeric port cannot configure either the Service or the Mooncake master process."""
        with pytest.raises(ValueError):
            MooncakeInfo.plan_of_args(_mooncake_args(master_server_address="host:not-a-port"))

    def test_refuses_a_mooncake_run_that_configured_no_kwargs(self):
        # [FUNCTION DOC] Mooncake backend with the init kwargs unset (none) raises an assertion
        # error saying the kwargs are missing.
        """There is no address to rewrite then, and a pod's loopback master would hang every client."""
        args = Namespace(object_store_backend="mooncake", mooncake_store_init_kwargs=None)

        with pytest.raises(AssertionError, match="is missing"):
            MooncakeInfo.plan_of_args(args)


class TestMooncakeWithClusterMaster:
    # [CLASS DOC] Tests of rewriting an argv so the Mooncake master address points at the
    # in-cluster service host.
    def test_points_the_master_address_at_the_in_cluster_service(self):
        # [FUNCTION DOC] The rewritten kwargs' master address is the given host plus the plan's
        # port 50051.
        """The launcher's own loopback address means nothing inside a pod, which would hang every client."""
        rewritten = MooncakeInfo.with_cluster_master(
            _mooncake_argv(),
            plan=MooncakeInfo.plan_of_args(_mooncake_args()),
            host="mooncake.myns.svc.cluster.local",
        )

        assert _rewritten_kwargs(rewritten)["master_server_address"] == "mooncake.myns.svc.cluster.local:50051"

    def test_a_run_that_never_named_the_flag_is_given_it(self):
        # [FUNCTION DOC] An argv without the init kwargs flag: the rewrite appends the flag with
        # kwargs holding the cluster master address, and keeps the original argv as the prefix of
        # the result.
        """The backend moves a run onto this store without being asked, so the flag is missing exactly
        when the run did not choose mooncake. Every pod parses this argv on its own, so omitting it
        leaves each of them looking for a master on its own loopback."""
        plan = MooncakeInfo.plan_of_args(_mooncake_args())
        argv = ["python", "train.py", "--num-rollout", "1"]

        rewritten = MooncakeInfo.with_cluster_master(argv, plan=plan, host="mooncake.myns.svc.cluster.local")

        assert _rewritten_kwargs(rewritten)["master_server_address"] == "mooncake.myns.svc.cluster.local:50051"
        assert rewritten[: len(argv)] == argv

    def test_keeps_the_port_the_run_configured(self):
        # [FUNCTION DOC] With a configured address on port 60000, the rewrite replaces only the
        # host and keeps port 60000.
        """The Service publishes the port the values carry, so rewriting the host must not move the port."""
        plan = MooncakeInfo.plan_of_args(_mooncake_args(master_server_address="1.2.3.4:60000"))

        rewritten = MooncakeInfo.with_cluster_master(
            _mooncake_argv(master_server_address="1.2.3.4:60000"), plan=plan, host="host"
        )

        assert _rewritten_kwargs(rewritten)["master_server_address"] == "host:60000"

    def test_keeps_every_other_init_kwarg(self):
        # [FUNCTION DOC] The local host name kwarg survives the rewrite unchanged.
        """The kwargs are rewritten as a whole, and a dropped one changes how the store is built."""
        rewritten = MooncakeInfo.with_cluster_master(
            _mooncake_argv(), plan=MooncakeInfo.plan_of_args(_mooncake_args()), host="host"
        )

        assert _rewritten_kwargs(rewritten)["local_hostname"] == "localhost"

    def test_leaves_the_rest_of_the_argv_untouched(self):
        # [FUNCTION DOC] The first five elements and the last two elements (learning rate flag
        # and value) of the argv are identical after the rewrite.
        """Only the address is cluster-specific; every other argument is the experiment itself."""
        rewritten = MooncakeInfo.with_cluster_master(
            _mooncake_argv(), plan=MooncakeInfo.plan_of_args(_mooncake_args()), host="host"
        )

        assert rewritten[:5] == _mooncake_argv()[:5]
        assert rewritten[-2:] == ["--lr", "1e-6"]

    def test_passes_a_non_mooncake_run_through_unchanged(self):
        # [FUNCTION DOC] With no plan (none), the rewrite returns an argv equal to the input.
        """A run that never asked for mooncake has no address to rewrite, and no kwargs to invent."""
        argv = ["python", "train.py", "--lr", "1e-6"]

        assert MooncakeInfo.with_cluster_master(argv, plan=None, host="host") == argv


@requires_helm
class TestMooncakeServiceNameCoupling:
    # [CLASS DOC] Skipped unless the helm binary is available. Verifies against a real chart
    # render that the Python side agrees with the chart's naming.
    def test_the_host_it_builds_is_the_service_the_chart_renders(self):
        # [FUNCTION DOC] Renders the run chart with mooncake enabled, selects Service objects whose
        # name ends with the mooncake master suffix, asserts exactly one exists, that the
        # launcher's master object name for the release equals that service name, and that the
        # service host equals the name plus namespace plus the svc suffix.
        """The pods dial this name; a chart rename would leave the launcher pointing at nothing."""
        objects = render_run("--set", "run.mooncake.enabled=true")
        services = [
            obj for obj in objects_of_kind(objects, "Service") if obj["metadata"]["name"].endswith("mooncake-master")
        ]

        assert len(services) == 1
        assert MooncakeInfo.master_object_name(RUN_RELEASE_NAME) == services[0]["metadata"]["name"]
        assert MooncakeInfo.master_service_host(RUN_RELEASE_NAME, NAMESPACE) == (
            f"{services[0]['metadata']['name']}.{NAMESPACE}.svc"
        )
