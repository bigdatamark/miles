# [MODULE DOC]
# Tests (helm rendering of the run chart, needs helm) of the orchestrator workload: the command is
# wrapped by the exit-file wrapper module with the state file and uninstall manifest arguments,
# arguments with spaces stay single words, exactly one replica, service links disabled, shared
# storage and manifest mounts, a headless service in the namespace, schema errors for an invalid
# run id and for a missing state file or runs root, no Job in a normal install, the identity
# variables, and the pod rights and service account bound to the orchestrator.
from tests.fast.charts.utils import (
    NAMESPACE,
    RUN_RELEASE_NAME,
    RUN_STATE_FILE,
    named_object,
    objects_of_kind,
    pod_spec_of,
    render_run,
    render_run_error,
    requires_helm,
    schema_error_mentions,
    sole_container_of,
)

ORCHESTRATOR = "myrun-miles-run-orchestrator"
ORCHESTRATOR_ACCOUNT = "myrun-miles-run-platform-read-delete"
WRAPPER_MODULE = "miles.utils.external_utils.command_utils.helm_backend.orchestrator.wrapper"


@requires_helm
class TestOrchestrator:
    # [CLASS DOC] Tests of the orchestrator stateful set and service.
    def test_wraps_the_training_command_so_the_exit_code_reaches_shared_storage(self):
        # [FUNCTION DOC] The container command equals: python, module option, the wrapper module,
        # state file option and path, uninstall manifest option and path, a double dash, then the
        # supplied training command words.
        """The launcher polls the exit file, so the script must run under the wrapper rather than directly."""
        objects = render_run("--set-json", 'run.orchestrator.command=["python","train.py","--foo"]')

        command = sole_container_of(objects, "StatefulSet", ORCHESTRATOR)["command"]

        assert command == [
            "python",
            "-m",
            WRAPPER_MODULE,
            "--state-file",
            RUN_STATE_FILE,
            "--uninstall-manifest",
            "/etc/miles-uninstall/uninstall-job.yaml",
            "--",
            "python",
            "train.py",
            "--foo",
        ]

    def test_keeps_arguments_with_spaces_as_single_words(self):
        # [FUNCTION DOC] A final command argument holding a JSON object with a space is the last
        # command word, unchanged.
        """A model arg like a json blob must not split into several argv entries on its way through the chart."""
        objects = render_run("--set-json", 'run.orchestrator.command=["python","train.py","--kwargs","{\\"a\\": 1}"]')

        assert sole_container_of(objects, "StatefulSet", ORCHESTRATOR)["command"][-1] == '{"a": 1}'

    def test_runs_exactly_one_orchestrator(self):
        # [FUNCTION DOC] The orchestrator stateful set has one replica.
        """Two orchestration scripts would drive the same run twice."""
        statefulset = named_object(render_run(), "StatefulSet", ORCHESTRATOR)

        assert statefulset["spec"]["replicas"] == 1

    def test_disables_the_service_link_environment(self):
        # [FUNCTION DOC] The pod's enable-service-links flag is false.
        """Kubernetes would otherwise inject <SERVICE>_PORT vars, and a prometheus service breaks arg parsing."""
        assert pod_spec_of(render_run(), "StatefulSet", ORCHESTRATOR)["enableServiceLinks"] is False

    def test_mounts_shared_storage_where_the_run_directory_lives(self):
        # [FUNCTION DOC] The container's mount paths are exactly the storage root and the uninstall
        # manifest directory.
        """The wrapper writes the exit file to that path, so the mount has to be there."""
        mounts = sole_container_of(render_run(), "StatefulSet", ORCHESTRATOR)["volumeMounts"]

        assert [mount["mountPath"] for mount in mounts] == ["/cluster-storage", "/etc/miles-uninstall"]

    def test_serves_the_orchestrator_under_a_headless_service(self):
        # [FUNCTION DOC] The orchestrator service has cluster IP None and is in the test namespace.
        """Workers address each other by stable dns, which a headless service is what provides."""
        service = named_object(render_run(), "Service", ORCHESTRATOR)

        assert service["spec"]["clusterIP"] == "None"
        assert service["metadata"]["namespace"] == NAMESPACE

    def test_rejects_a_run_id_that_cannot_be_a_kubernetes_name(self):
        # [FUNCTION DOC] A run id with uppercase and underscores fails rendering and the error
        # mentions the run id path.
        """The release derives object names from it, so an invalid id must fail before anything is created."""
        assert schema_error_mentions(render_run_error("--set", "run.id=Not_A_Name"), path=("run", "id"))

    def test_refuses_an_orchestrated_run_that_names_no_state_file(self):
        # [FUNCTION DOC] A null state file fails rendering with an error mentioning it.
        """The launcher polls that path to learn the run finished; without it the launcher waits forever."""
        assert "run.stateFile" in render_run_error("--set", "run.stateFile=null")

    def test_refuses_a_run_with_nowhere_to_write_its_outcome(self):
        # [FUNCTION DOC] A null runs root fails rendering with an error mentioning it.
        """The launcher learns the outcome by reading the exit file, which needs a directory on a volume."""
        assert "runsRoot" in render_run_error("--set", "infra.paths.runsRoot=null")

    def test_ships_no_job_in_a_normal_install(self):
        # [FUNCTION DOC] A default run contains no Job.
        """Command jobs are applied on their own; installing one with a run would rerun it on every upgrade."""
        assert objects_of_kind(render_run(), "Job") == []


@requires_helm
class TestOrchestratorIdentity:
    # [CLASS DOC] Tests of the orchestrator's identity variables and platform rights.
    def test_tells_the_orchestrator_which_release_and_namespace_hold_its_workers(self):
        # [FUNCTION DOC] The container environment contains the namespace and release variables
        # with the test namespace and run release name (checked by merging the expected pairs into the
        # actual mapping and requiring it to be unchanged).
        """The orchestrator selects worker pods by release label, and it cannot recompute that name."""
        env = sole_container_of(render_run(), "StatefulSet", ORCHESTRATOR)["env"]

        assert {entry["name"]: entry["value"] for entry in env} | {
            "MILES_K8S_NAMESPACE": NAMESPACE,
            "MILES_K8S_RELEASE": RUN_RELEASE_NAME,
        } == {entry["name"]: entry["value"] for entry in env}

    def test_grants_the_orchestrator_the_pod_rights_observation_and_healing_need(self):
        # [FUNCTION DOC] The read-delete role has exactly two rules: pods with get, list, watch and
        # delete in the core group, and job creation in the batch group. The role binding references
        # that role and has exactly one service account subject of the same name in the namespace.
        """Healing a cell means deleting its pods, and observing one means watching them."""
        objects = render_run()
        role = named_object(objects, "Role", ORCHESTRATOR_ACCOUNT)
        binding = named_object(objects, "RoleBinding", ORCHESTRATOR_ACCOUNT)

        assert role["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "delete"]},
            {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create"]},
        ]
        assert binding["roleRef"]["name"] == ORCHESTRATOR_ACCOUNT
        assert binding["subjects"] == [dict(kind="ServiceAccount", name=ORCHESTRATOR_ACCOUNT, namespace=NAMESPACE)]

    def test_runs_the_orchestrator_under_the_account_that_binding_names(self):
        # [FUNCTION DOC] The orchestrator pod's service account is the bound account, and that
        # service account object exists in the test namespace.
        """A role bound to an account nobody runs as grants nothing at all."""
        assert pod_spec_of(render_run(), "StatefulSet", ORCHESTRATOR)["serviceAccountName"] == ORCHESTRATOR_ACCOUNT
        assert named_object(render_run(), "ServiceAccount", ORCHESTRATOR_ACCOUNT)["metadata"]["namespace"] == NAMESPACE
