# [MODULE DOC]
# Tests of the run chart's automatic self-uninstall feature, rendered with helm (most tests need
# it): the uninstall job manifest stored in a config map (delayed helm uninstall, ignore-not-found,
# no inherited anti-affinity, dedicated service account, no release-selector labels, name and
# deadlines, tolerations, not a helm-owned Job), the orchestrator wiring (manifest argument and
# mount, role right to create that one job, nothing rendered when disabled), and the uninstaller
# service account naming contract shared with the launcher code.
from functools import cache
from typing import Any

import yaml
from tests.fast.charts.utils import (
    CHART_DIR,
    NAMESPACE,
    RUN_CHART_DIR,
    RUN_ORCHESTRATOR_NAME,
    RUN_PLATFORM_READ_DELETE_NAME,
    RUN_RELEASE_NAME,
    RUN_UNINSTALL_JOB_NAME,
    RUN_UNINSTALL_MANIFEST_NAME,
    UNINSTALLER_SERVICE_ACCOUNT,
    named_object,
    objects_of_kind,
    render_run,
    requires_helm,
    sole_container_of,
)

from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL

MANIFEST_PATH = "/etc/miles-uninstall/uninstall-job.yaml"

ENABLE_AUTO_UNINSTALL = ("--set", "run.autoUninstall.enabled=true")
DISABLE_AUTO_UNINSTALL = ("--set", "run.autoUninstall.enabled=false")


def _enabled_objects(*args: str) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Renders the run chart with auto uninstall enabled plus extra
    # arguments. With no extra arguments returns a fresh list copied from a cached default render
    # (so the helm call happens once per process).
    if args:
        return render_run(*ENABLE_AUTO_UNINSTALL, *args)
    return list(_default_enabled_objects())


@cache
def _default_enabled_objects() -> tuple[dict[str, Any], ...]:
    # [FUNCTION DOC] HELPER: Cached render of the run chart with auto uninstall enabled, as an
    # immutable tuple of objects.
    return tuple(render_run(*ENABLE_AUTO_UNINSTALL))


def _rendered_job(*args: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Renders with auto uninstall enabled plus extra arguments, finds the
    # uninstall manifest config map, and returns its stored job YAML parsed into a mapping.
    config_map = named_object(_enabled_objects(*args), "ConfigMap", RUN_UNINSTALL_MANIFEST_NAME)
    return yaml.safe_load(config_map["data"]["uninstall-job.yaml"])


@requires_helm
class TestUninstallManifest:
    # [CLASS DOC] Tests of the rendered uninstall job manifest (needs helm).
    def test_leaves_the_logs_readable_for_two_minutes_before_uninstalling_the_release(self):
        # [FUNCTION DOC] The first container's command is a shell invocation that sleeps 120
        # seconds and then runs helm uninstall of the run release in namespace myns with
        # ignore-not-found.
        """The wrapper creates the job the moment the verdict is in, so the delay has to live in the job."""
        container = _rendered_job()["spec"]["template"]["spec"]["containers"][0]

        assert container["command"] == [
            "sh",
            "-c",
            f"sleep 120 && helm uninstall {RUN_RELEASE_NAME} --namespace myns --ignore-not-found",
        ]

    def test_treats_an_already_gone_release_as_a_successful_uninstall(self):
        # [FUNCTION DOC] The final command string contains the ignore-not-found flag.
        """Split runs uninstall their own release, so this job routinely finds nothing and must not fail."""
        command = _rendered_job()["spec"]["template"]["spec"]["containers"][0]["command"][-1]

        assert "--ignore-not-found" in command

    def test_does_not_inherit_the_anti_affinity_that_spread_the_run(self):
        # [FUNCTION DOC] Sets scheduling affinity with a required pod anti-affinity (hostname
        # topology) and a node affinity; the job's pod affinity must keep only the node affinity.
        """Required anti-affinity can hold down every eligible node, leaving this pod nowhere to be scheduled
        and the release and its gpus in place, while node placement still has to be obeyed."""
        affinity = (
            "--set-json",
            'infra.scheduling.affinity={"podAntiAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":'
            '[{"topologyKey":"kubernetes.io/hostname"}]},"nodeAffinity":{"k":"v"}}',
        )

        spec = _rendered_job(*affinity)["spec"]["template"]["spec"]

        assert spec["affinity"] == {"nodeAffinity": {"k": "v"}}

    def test_runs_as_the_account_that_outlives_the_release(self):
        # [FUNCTION DOC] The job pod's service account name equals the uninstaller account constant.
        """helm deletes the release's own rolebinding halfway through, which would 403 the rest of the deletions."""
        assert _rendered_job()["spec"]["template"]["spec"]["serviceAccountName"] == UNINSTALLER_SERVICE_ACCOUNT

    def test_carries_no_label_the_release_selector_matches(self):
        # [FUNCTION DOC] Neither the job nor its pod template labels contain the instance label, and
        # the job labels equal exactly one label naming the uninstalled release.
        """The launcher follows and deletes objects by that selector, and would take the escape job with them."""
        job = _rendered_job()

        assert INSTANCE_LABEL not in job["metadata"]["labels"]
        assert INSTANCE_LABEL not in job["spec"]["template"]["metadata"]["labels"]
        assert job["metadata"]["labels"] == {"miles.radixark.io/uninstall-of": RUN_RELEASE_NAME}

    def test_names_the_job_what_the_launcher_deletes_on_a_relaunch(self):
        # [FUNCTION DOC] The job's name and namespace equal the expected uninstall job name and the
        # test namespace.
        """Defusing a pending uninstall needs the name python computed, not one the chart invented."""
        metadata = _rendered_job()["metadata"]

        assert (metadata["name"], metadata["namespace"]) == (RUN_UNINSTALL_JOB_NAME, NAMESPACE)

    def test_gives_up_and_collects_itself_instead_of_waiting_forever(self):
        # [FUNCTION DOC] The job's active deadline is 1020 seconds, time-to-live after finish is
        # 3600 seconds, and backoff limit is 2.
        """A job that never got its node must not hold a slot, and its corpse must not outlive the namespace."""
        spec = _rendered_job()["spec"]

        assert (spec["activeDeadlineSeconds"], spec["ttlSecondsAfterFinished"], spec["backoffLimit"]) == (
            1020,
            3600,
            2,
        )

    def test_schedules_where_every_other_pod_of_the_run_may_run(self):
        # [FUNCTION DOC] A configured toleration with key gpu appears unchanged in the job pod.
        """A cluster that needs a toleration needs it here too, or the release is never uninstalled."""
        job = _rendered_job("--set", "infra.scheduling.tolerations[0].key=gpu")

        assert job["spec"]["template"]["spec"]["tolerations"] == [{"key": "gpu"}]

    def test_stays_out_of_the_release_it_uninstalls(self):
        # [FUNCTION DOC] The rendered release contains no Job object (the job lives only inside the
        # config map).
        """A job helm owns would be deleted by the very uninstall it is running."""
        assert objects_of_kind(_enabled_objects(), "Job") == []


@requires_helm
class TestOrchestratorTrigger:
    # [CLASS DOC] Tests of how the orchestrator is wired to create the uninstall job (needs helm).
    def test_hands_the_wrapper_the_manifest_it_creates_the_job_from(self):
        # [FUNCTION DOC] In the orchestrator container's command, the two items before the double
        # dash separator are the manifest option and the manifest path, and the container mounts the
        # uninstall manifest volume read-only at the manifest directory.
        """The wrapper is the first authority on a finished run, and it can only apply a file it was told about."""
        container = sole_container_of(_enabled_objects(), "StatefulSet", RUN_ORCHESTRATOR_NAME)
        separator = container["command"].index("--")

        assert container["command"][separator - 2 : separator] == ["--uninstall-manifest", MANIFEST_PATH]
        assert {"name": "uninstall-manifest", "mountPath": "/etc/miles-uninstall", "readOnly": True} in container[
            "volumeMounts"
        ]

    def test_lets_the_orchestrator_create_that_one_job(self):
        # [FUNCTION DOC] The platform read-delete role contains a rule allowing only the create verb
        # on jobs in the batch API group.
        """The wrapper creates it as the run's own account, which is otherwise allowed pods and nothing else."""
        role = named_object(_enabled_objects(), "Role", RUN_PLATFORM_READ_DELETE_NAME)

        assert {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]} in role["rules"]

    def test_grants_no_job_rights_to_a_run_that_never_creates_one(self):
        # [FUNCTION DOC] With auto uninstall disabled, every rule of the read-delete role targets
        # only the core (empty name) API group.
        """A run that cannot uninstall itself has no business creating workloads of any kind."""
        role = named_object(render_run(*DISABLE_AUTO_UNINSTALL), "Role", RUN_PLATFORM_READ_DELETE_NAME)

        assert [rule["apiGroups"] for rule in role["rules"]] == [[""]]

    def test_says_nothing_about_uninstalling_when_the_launcher_found_no_uninstaller(self):
        # [FUNCTION DOC] With auto uninstall disabled, the orchestrator command lacks the manifest
        # option and no config map is rendered at all.
        """A namespace without that account cannot half-uninstall a release, so nothing about it is rendered."""
        objects = render_run(*DISABLE_AUTO_UNINSTALL)

        assert (
            "--uninstall-manifest" not in sole_container_of(objects, "StatefulSet", RUN_ORCHESTRATOR_NAME)["command"]
        )
        assert objects_of_kind(objects, "ConfigMap") == []


class TestUninstallerAccountName:
    # [CLASS DOC] Tests pinning the uninstaller service account name and default.
    def test_a_bare_helm_install_cleans_up_after_itself_too(self):
        # [FUNCTION DOC] The run chart's default values have auto uninstall enabled (static check,
        # no helm).
        """The launcher always writes the flag, so this default is what a hand-installed release gets."""
        run_default = yaml.safe_load((RUN_CHART_DIR / "values.yaml").read_text())["run"]["autoUninstall"]

        assert run_default["enabled"] is True

    @requires_helm
    def test_neither_chart_lets_values_move_that_name(self):
        # [FUNCTION DOC] The run chart's auto-uninstall defaults have no service account setting, the
        # workbench chart's defaults have no uninstaller setting, and the rendered job runs as the
        # constant account name. Needs helm.
        """The workbench creates that account and a run's job runs as it, so a values-only drift would 403 it."""
        assert (
            "serviceAccount" not in yaml.safe_load((RUN_CHART_DIR / "values.yaml").read_text())["run"]["autoUninstall"]
        )
        assert "uninstaller" not in yaml.safe_load((CHART_DIR / "values.yaml").read_text())
        assert _rendered_job()["spec"]["template"]["spec"]["serviceAccountName"] == UNINSTALLER_SERVICE_ACCOUNT

    def test_the_tests_and_the_code_agree_on_that_name(self):
        # [FUNCTION DOC] The test utils' uninstaller account constant equals the launcher naming
        # module's constant.
        """Every assertion here is written against the constant the launcher and the charts actually use."""
        assert UNINSTALLER_SERVICE_ACCOUNT == naming.UNINSTALLER_SERVICE_ACCOUNT
