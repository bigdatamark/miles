# [MODULE DOC]
# Tests (helm rendering of the run chart, needs helm) of how the run chart renders infra storage,
# paths and environment onto the orchestrator and other pods: checkout mounts over image copies and
# the code path variable, read-only mounts, validation that the runs root is on a writable mounted
# volume (including path spelling variants, nesting of mounts, and rejection of memory-only
# empty-dir volumes), the state file argument, cluster environment propagation, rejection of
# platform-owned variables (code path, launch record) in values, a working directory for every
# container, the launch record variable, and namespace variable substitution in host paths, mount
# paths, sub paths and the runs root, with rejection of unknown variables. Module data: sample
# orchestrator name, expected code path, expected orchestrator identity variables, a volume
# arguments tuple with all three repo mounts, and a whole-topology arguments tuple with one static
# worker, trainer engine and inference engine.
import json
from typing import Any

from tests.fast.charts.utils import (
    NAMESPACE,
    host_path_volume,
    pod_spec_of,
    render_run,
    render_run_error,
    requires_helm,
    sole_container_of,
    volumes_args,
    with_object_names,
)

ORCHESTRATOR = "myrun-miles-run-orchestrator"
CODE_PYTHONPATH = "/root/miles:/root/Megatron-LM:/sgl-workspace/sglang/python"
ORCHESTRATOR_IDENTITY = {
    "MILES_K8S_NAMESPACE": "myns",
    "MILES_K8S_RELEASE": "myrun",
    "PYTHONPATH": CODE_PYTHONPATH,
}

ALL_REPOS = volumes_args(
    host_path_volume(
        mounts=[
            {"mountPath": "/cluster-storage"},
            {"mountPath": "/root/miles", "subPath": "myuser/miles"},
            {"mountPath": "/root/Megatron-LM", "subPath": "myuser/Megatron-LM"},
            {"mountPath": "/sgl-workspace/sglang", "subPath": "myuser/sglang"},
        ]
    )
)


def orchestrator_container(*args: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Renders the run chart with extra arguments and returns the
    # orchestrator stateful set's sole container.
    return sole_container_of(render_run(*args), "StatefulSet", ORCHESTRATOR)


def environment(container: dict[str, Any]) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Returns a mapping from environment variable name to literal value for
    # the container (empty if no env). Entries without a literal value raise a key error.
    return {entry["name"]: entry["value"] for entry in container.get("env", [])}


@requires_helm
class TestCheckoutMounts:
    # [CLASS DOC] Tests of checkout mounts on the orchestrator.
    def test_a_configured_repo_mounts_over_the_copy_baked_into_the_image(self):
        # [FUNCTION DOC] With the storage volume mounting the root plus three repos by sub path, the
        # orchestrator's mounts are exactly: root, miles, Megatron-LM, sglang (in order), followed by
        # the read-only uninstall manifest mount.
        """Editing code on shared storage is only picked up if the mount lands on the path the image imports from."""
        mounts = orchestrator_container(*ALL_REPOS)["volumeMounts"]

        assert mounts == [
            {"name": "cluster-storage", "mountPath": "/cluster-storage"},
            {"name": "cluster-storage", "mountPath": "/root/miles", "subPath": "myuser/miles"},
            {"name": "cluster-storage", "mountPath": "/root/Megatron-LM", "subPath": "myuser/Megatron-LM"},
            {"name": "cluster-storage", "mountPath": "/sgl-workspace/sglang", "subPath": "myuser/sglang"},
            {"name": "uninstall-manifest", "mountPath": "/etc/miles-uninstall", "readOnly": True},
        ]

    def test_only_the_repos_that_were_named_are_overridden(self):
        # [FUNCTION DOC] With only the Megatron-LM repo mounted, the mount paths are the root, that
        # repo and the uninstall manifest directory.
        """Overriding one repo must not shadow the other two with an empty directory."""
        container = orchestrator_container(
            *volumes_args(
                host_path_volume(
                    mounts=[
                        {"mountPath": "/cluster-storage"},
                        {"mountPath": "/root/Megatron-LM", "subPath": "myuser/Megatron-LM"},
                    ]
                )
            )
        )

        assert [mount["mountPath"] for mount in container["volumeMounts"]] == [
            "/cluster-storage",
            "/root/Megatron-LM",
            "/etc/miles-uninstall",
        ]

    def test_the_defaults_override_no_repo_at_all(self):
        # [FUNCTION DOC] By default the mount paths are the storage root and the uninstall manifest
        # directory, and the environment equals exactly the orchestrator identity variables
        # (namespace, release and code path).
        """The image is self-contained, so a run that mounts no checkout of its own gets no extra environment."""
        container = orchestrator_container()

        assert [mount["mountPath"] for mount in container["volumeMounts"]] == [
            "/cluster-storage",
            "/etc/miles-uninstall",
        ]
        assert environment(container) == ORCHESTRATOR_IDENTITY


@requires_helm
class TestReadOnlyMounts:
    # [CLASS DOC] Tests of read-only mount flags.
    def test_a_mount_marked_read_only_reaches_the_container_that_way(self):
        # [FUNCTION DOC] A read-only models mount from a second host-path volume is rendered with
        # the read-only flag true.
        """A shared model cache is everyone's, and a run that can write it can corrupt every other run."""
        container = orchestrator_container(
            *volumes_args(
                host_path_volume(mounts=[{"mountPath": "/cluster-storage"}]),
                {
                    "name": "models",
                    "hostPath": {"path": "/gpfs/models", "type": "Directory"},
                    "mounts": [{"mountPath": "/models", "readOnly": True}],
                },
            )
        )

        assert {"name": "models", "mountPath": "/models", "readOnly": True} in container["volumeMounts"]

    def test_an_ordinary_mount_says_nothing_about_being_read_only(self):
        # [FUNCTION DOC] An ordinary mount is rendered with only name and mount path.
        """readOnly: false is the kubernetes default, and spelling it out only makes every diff noisier."""
        container = orchestrator_container(*volumes_args(host_path_volume()))

        assert {"name": "cluster-storage", "mountPath": "/cluster-storage"} in container["volumeMounts"]


@requires_helm
class TestRunsRootIsOnAMountedVolume:
    # [CLASS DOC] Tests that the runs root must lie under a writable mounted volume.
    def test_a_runs_root_under_no_mount_at_all_is_refused(self):
        # [FUNCTION DOC] A runs root outside every mount makes rendering fail with an error naming
        # that path.
        """A run would write its state file into the container's own filesystem, where the launcher never looks."""
        error = render_run_error(*volumes_args(host_path_volume()), "--set", "infra.paths.runsRoot=/elsewhere/data")

        assert "/elsewhere/data" in error

    def test_a_runs_root_on_a_read_only_mount_is_refused(self):
        # [FUNCTION DOC] With the default runs root under a read-only mount, rendering fails with an
        # error mentioning read-only.
        """The volume is there, but every run writes its state, values and exit file under this directory."""
        error = render_run_error(
            *volumes_args(host_path_volume(mounts=[{"mountPath": "/cluster-storage", "readOnly": True}]))
        )

        assert "read-only" in error

    def test_a_runs_root_that_is_a_mount_path_itself_is_accepted(self):
        # [FUNCTION DOC] A runs root equal to a mount path renders successfully (the container has an
        # image).
        """A cluster that dedicates a whole volume to miles must not be forced into a subdirectory."""
        container = orchestrator_container(
            *volumes_args(host_path_volume()), "--set", "infra.paths.runsRoot=/cluster-storage"
        )

        assert container["image"]

    def test_a_run_with_no_runs_root_at_all_is_refused(self):
        # [FUNCTION DOC] Setting the runs root to null makes rendering fail with an error mentioning
        # the runs root.
        """The launcher polls the exit file under it, so a run without one can only ever look like a hang."""
        assert "runsRoot" in render_run_error("--set", "infra.paths.runsRoot=null")


@requires_helm
class TestRunsRootContainmentIgnoresHowAPathIsSpelled:
    # [CLASS DOC] Tests that containment checks normalize path spelling.
    def test_a_mount_written_with_a_trailing_slash_still_covers_the_runs_root(self):
        # [FUNCTION DOC] A mount path with a trailing slash still covers a runs root beneath it
        # (renders successfully).
        """The mount is the same directory however it is typed, and the refusal would be unexplainable."""
        container = orchestrator_container(
            *volumes_args(host_path_volume(mounts=[{"mountPath": "/cluster-storage/"}])),
            "--set",
            "infra.paths.runsRoot=/cluster-storage/miles_data",
        )

        assert container["image"]

    def test_a_mount_at_the_filesystem_root_covers_every_runs_root(self):
        # [FUNCTION DOC] A mount at the filesystem root covers any runs root (renders successfully).
        """A volume mounted at / holds every path, and trimming its slash must not leave an empty prefix."""
        container = orchestrator_container(
            *volumes_args(host_path_volume(mounts=[{"mountPath": "/"}])),
            "--set",
            "infra.paths.runsRoot=/anywhere/miles_data",
        )

        assert container["image"]

    def test_redundant_separators_on_either_side_name_the_same_directory(self):
        # [FUNCTION DOC] A mount path with a dot segment and a runs root with a doubled slash name
        # the same directory tree, so rendering succeeds.
        """A path pasted together from two configured halves carries them, and names the mount all the same."""
        container = orchestrator_container(
            *volumes_args(host_path_volume(mounts=[{"mountPath": "/cluster-storage/./data"}])),
            "--set",
            "infra.paths.runsRoot=/cluster-storage//data/miles_data",
        )

        assert container["image"]

    def test_a_runs_root_beside_a_mount_is_not_taken_for_one_inside_it(self):
        # [FUNCTION DOC] A runs root in a sibling directory sharing a name prefix with the mount path
        # is refused, with an error naming the runs root.
        """/cluster-storage-2 starts with the mount path as text, and only the separator tells them apart."""
        error = render_run_error(
            *volumes_args(host_path_volume(mounts=[{"mountPath": "/cluster-storage/"}])),
            "--set",
            "infra.paths.runsRoot=/cluster-storage-2/miles_data",
        )

        assert "/cluster-storage-2/miles_data" in error

    def test_a_runs_root_under_an_empty_dir_mount_is_refused_however_the_mount_is_spelled(self):
        # [FUNCTION DOC] A runs root under an ephemeral empty-dir volume (mount path with trailing
        # slash) is refused with an error mentioning empty-dir.
        """Every pod would get a directory of its own, so no run could ever report a verdict."""
        error = render_run_error(
            *volumes_args({"name": "scratch", "emptyDir": {}, "mounts": [{"mountPath": "/scratch/"}]}),
            "--set",
            "infra.paths.runsRoot=/scratch/miles_data",
        )

        assert "emptyDir" in error


@requires_helm
class TestTheMountThatProvidesTheRunsRootDecides:
    # [CLASS DOC] Tests that the innermost mount covering the runs root decides validity.
    def test_a_read_only_mount_nested_under_a_writable_one_is_refused(self):
        # [FUNCTION DOC] A read-only mount at the runs root nested under a writable parent mount is
        # refused with a read-only error.
        """Kubernetes hands the path to the innermost mount, so the writable one above it never sees a write."""
        error = render_run_error(
            *volumes_args(
                host_path_volume(
                    name="data",
                    path="/data",
                    mounts=[{"mountPath": "/data"}, {"mountPath": "/data/runs", "readOnly": True}],
                )
            ),
            "--set",
            "infra.paths.runsRoot=/data/runs",
        )

        assert "read-only" in error

    def test_a_writable_mount_nested_under_a_read_only_one_is_accepted(self):
        # [FUNCTION DOC] A writable mount at the runs root nested under a read-only parent renders
        # successfully.
        """The same rule the other way round: the run writes into the inner mount, and that one is writable."""
        container = orchestrator_container(
            *volumes_args(
                host_path_volume(
                    name="data",
                    path="/data",
                    mounts=[{"mountPath": "/data", "readOnly": True}, {"mountPath": "/data/runs"}],
                )
            ),
            "--set",
            "infra.paths.runsRoot=/data/runs",
        )

        assert container["image"]

    def test_a_runs_root_provided_by_one_writable_mount_is_accepted(self):
        # [FUNCTION DOC] A runs root below a single writable mount renders successfully.
        """The ordinary cluster, where a single shared volume holds every run's directory."""
        container = orchestrator_container(
            *volumes_args(host_path_volume()), "--set", "infra.paths.runsRoot=/cluster-storage/miles_data"
        )

        assert container["image"]

    def test_an_empty_dir_nested_under_a_shared_mount_is_refused(self):
        # [FUNCTION DOC] An empty-dir mount at the runs root nested under a shared host-path mount is
        # refused with an empty-dir error.
        """The shared volume above it is never written: each pod gets the emptyDir's own copy of the directory."""
        error = render_run_error(
            *volumes_args(
                host_path_volume(name="data", path="/data", mounts=[{"mountPath": "/data"}]),
                {"name": "scratch", "emptyDir": {}, "mounts": [{"mountPath": "/data/runs"}]},
            ),
            "--set",
            "infra.paths.runsRoot=/data/runs",
        )

        assert "emptyDir" in error


@requires_helm
class TestRunDirectory:
    # [CLASS DOC] Test of the state file argument.
    def test_the_orchestrator_watches_the_state_file_the_launcher_named(self):
        # [FUNCTION DOC] The supplied state file path appears in the orchestrator's command.
        """The launcher polls the path it injected, so a chart that derives its own is a run that can only hang."""
        state_file = "/mnt/x/teamdata/miles-runs/myrun/state/orchestrator.state"

        container = orchestrator_container("--set", f"run.stateFile={state_file}")

        assert state_file in container["command"]


@requires_helm
class TestClusterEnvironment:
    # [CLASS DOC] Tests of cluster environment propagation.
    def test_the_cluster_environment_reaches_the_orchestrator_pod(self):
        # [FUNCTION DOC] A proxy and a mirror endpoint set via infra env appear in the orchestrator
        # environment next to the identity variables, and nothing else.
        """The orchestrator downloads datasets and reaches the api server, so it needs the cluster's proxy too."""
        container = orchestrator_container(
            "--set", "infra.env.HTTP_PROXY=http://proxy:7890", "--set", "infra.env.HF_ENDPOINT=https://mirror"
        )

        assert environment(container) == ORCHESTRATOR_IDENTITY | {
            "HTTP_PROXY": "http://proxy:7890",
            "HF_ENDPOINT": "https://mirror",
        }

    def test_the_cluster_environment_reaches_a_pod_that_mounts_its_own_checkouts(self):
        # [FUNCTION DOC] With all repo mounts and a proxy variable, the environment equals identity
        # variables plus the proxy.
        """Mounting code must not cost a pod the cluster's proxy, nor the proxy cost it the mounts."""
        container = orchestrator_container(*ALL_REPOS, "--set", "infra.env.HTTP_PROXY=http://proxy:7890")

        assert environment(container) == ORCHESTRATOR_IDENTITY | {"HTTP_PROXY": "http://proxy:7890"}


@requires_helm
class TestPythonPathIsNotAnEnvironmentVariable:
    # [CLASS DOC] Tests that the code path variable is platform-owned.
    def test_the_platform_puts_every_checkout_before_the_stale_editable_install(self):
        # [FUNCTION DOC] Across all containers of the whole topology (at least one), each has exactly
        # one code path variable and its value equals the expected path list.
        """Mounted checkouts must expose packages added after the image's editable metadata was built."""
        containers = containers_of(render_run(*WHOLE_TOPOLOGY))

        assert containers
        assert all(
            [entry["value"] for entry in container["env"] if entry["name"] == "PYTHONPATH"] == [CODE_PYTHONPATH]
            for container in containers
        )

    def test_the_schema_refuses_a_pythonpath_in_the_cluster_environment(self):
        # [FUNCTION DOC] Setting the code path variable in infra env fails rendering with an error
        # naming it.
        """The platform owns checkout precedence, so a hand-set PYTHONPATH cannot shadow it."""
        error = render_run_error("--set", "infra.env.PYTHONPATH=/somewhere")

        assert "PYTHONPATH" in error

    def test_the_schema_refuses_a_pythonpath_in_the_run_environment(self):
        # [FUNCTION DOC] Same as above for the run env section.
        """Same shadowing, one level down: a per-run value would reach the pods just as an infra one does."""
        error = render_run_error("--set", "run.env.PYTHONPATH=/somewhere")

        assert "PYTHONPATH" in error

    def test_an_ordinary_cluster_variable_is_still_accepted(self):
        # [FUNCTION DOC] An unreserved network interface variable set via infra env appears in the
        # orchestrator environment.
        """Only the variables the platform owns are reserved; refusing the rest would make infra.env useless."""
        objects = render_run("--set", "infra.env.NCCL_SOCKET_IFNAME=bond0")

        env = sole_container_of(objects, "StatefulSet", ORCHESTRATOR)["env"]
        assert {"name": "NCCL_SOCKET_IFNAME", "value": "bond0"} in env


# Module data for the whole-topology tests: one static worker, one trainer engine and one
# inference engine entry, and WHOLE_TOPOLOGY as the helm arguments setting all three.
STATIC_WORKERS = [{"name": "rollout-executor", "command": ["python", "-m", "miles.utils.workers.serving.serve"]}]
TRAINER_ENGINES = [
    {"name": "trainer-engine-actor", "command": ["python", "-m", "miles.utils.workers.process_supervisor"]}
]
INFERENCE_ENGINES = [{"name": "inference-engine-0-0", "command": ["python", "-m", "sglang.launch_server"]}]

WHOLE_TOPOLOGY = (
    "--set-json",
    f"run.staticWorkers={json.dumps(with_object_names(STATIC_WORKERS))}",
    "--set-json",
    f"run.trainerEngines={json.dumps(with_object_names(TRAINER_ENGINES))}",
    "--set-json",
    f"run.inferenceEngines={json.dumps(with_object_names(INFERENCE_ENGINES))}",
)


def containers_of(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Returns all containers found anywhere inside the objects (pod specs at
    # any nesting depth), flattened in order.
    return [container for obj in objects for container in _containers_in(obj)]


def _containers_in(node: Any) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Recursive search of a nested mapping and list structure. For a mapping,
    # collects its own containers list (if any), then recurses into every other value. For a list,
    # concatenates the results of its items. Anything else yields none.
    if isinstance(node, dict):
        found = list(node.get("containers") or [])
        for key, value in node.items():
            if key != "containers":
                found.extend(_containers_in(value))
        return found
    if isinstance(node, list):
        return [container for item in node for container in _containers_in(item)]
    return []


@requires_helm
class TestCheckoutWorkingDirectory:
    # [CLASS DOC] Test of container working directories.
    def test_every_container_runs_from_the_checkout_the_image_imports_from(self):
        # [FUNCTION DOC] In the whole topology the container names are exactly orchestrator, worker,
        # engine and trainer, and every container's working directory is /root/miles.
        """A container loading a custom function or a relative config path only resolves it from the checkout root."""
        containers = containers_of(render_run(*WHOLE_TOPOLOGY))

        assert {container["name"] for container in containers} == {"orchestrator", "worker", "engine", "trainer"}
        assert {container.get("workingDir") for container in containers} == {"/root/miles"}


@requires_helm
class TestTheLaunchRecordIsNotAnEnvironmentVariable:
    # [CLASS DOC] Tests of the launch record environment variable (platform-owned).
    def test_every_container_is_told_what_launched_this_run(self):
        # [FUNCTION DOC] Setting the launch record value makes the orchestrator's script env report
        # variable equal that path. KNOWN ISSUE: only the orchestrator container is checked despite
        # the name saying every container.
        """The record is how --env-report reaches each process's env report and the wandb config."""
        container = orchestrator_container("--set-string", "run.launchRecord=/shared/launches/launch-1.json")

        assert environment(container)["MILES_SCRIPT_ENV_REPORT"] == "/shared/launches/launch-1.json"

    def test_a_run_launched_without_a_record_carries_no_empty_one(self):
        # [FUNCTION DOC] Without a launch record the orchestrator has no such variable.
        """An empty record would look like a launcher that recorded nothing, rather than one that never ran."""
        assert "MILES_SCRIPT_ENV_REPORT" not in environment(orchestrator_container())

    def test_the_schema_refuses_a_record_in_the_cluster_environment(self):
        # [FUNCTION DOC] Setting the variable in infra env fails rendering with an error naming it.
        """infra.env outranks run.env, so a hand-set record would silently replace the real one."""
        assert "MILES_SCRIPT_ENV_REPORT" in render_run_error(
            "--set", "infra.env.MILES_SCRIPT_ENV_REPORT=/hijacked.json"
        )

    def test_the_schema_refuses_a_record_in_a_pool_environment(self):
        # [FUNCTION DOC] Setting the variable in a static worker's own env fails rendering with an
        # error naming it.
        """A per-pool variable outranks everything, so those pods would report a different launch."""
        assert "MILES_SCRIPT_ENV_REPORT" in render_run_error(
            "--set-json",
            """run.staticWorkers=[{"name":"router","objectName":"myrun-router","command":["sleep"],"""
            """"env":{"MILES_SCRIPT_ENV_REPORT":"hijacked"}}]""",
        )


@requires_helm
class TestNamespaceInterpolation:
    # [CLASS DOC] Tests of the namespace placeholder substitution in infra paths, and rejection of
    # other placeholders.
    def test_a_host_path_names_the_namespace_it_is_rendered_for(self):
        # [FUNCTION DOC] A host path containing the namespace placeholder renders with the actual
        # namespace in the orchestrator's volume.
        """Two agents in two namespaces want two directories on the host, and one values file has to give both."""
        objects = render_run(*volumes_args(host_path_volume(path="/data/${NAMESPACE}")))

        volumes = pod_spec_of(objects, "StatefulSet", ORCHESTRATOR)["volumes"]
        assert {"name": "cluster-storage", "hostPath": {"path": f"/data/{NAMESPACE}", "type": "Directory"}} in volumes

    def test_a_mount_path_names_the_namespace(self):
        # [FUNCTION DOC] A mount path containing the placeholder renders with the actual namespace.
        """A node-local scratch disk is one path on every node, so only the namespace keeps two runs apart."""
        container = orchestrator_container(
            *volumes_args(
                host_path_volume(mounts=[{"mountPath": "/cluster-storage"}, {"mountPath": "/scratch/${NAMESPACE}"}])
            )
        )

        assert {"name": "cluster-storage", "mountPath": f"/scratch/{NAMESPACE}"} in container["volumeMounts"]

    def test_a_sub_path_names_the_namespace(self):
        # [FUNCTION DOC] A sub path containing the placeholder renders with the actual namespace.
        """This is how five agents get five checkouts of the same repo out of one shared volume."""
        container = orchestrator_container(
            *volumes_args(
                host_path_volume(
                    mounts=[
                        {"mountPath": "/cluster-storage"},
                        {"mountPath": "/root/miles", "subPath": "repos/${NAMESPACE}/miles"},
                    ]
                )
            )
        )

        assert {
            "name": "cluster-storage",
            "mountPath": "/root/miles",
            "subPath": f"repos/{NAMESPACE}/miles",
        } in container["volumeMounts"]

    def test_a_path_that_names_no_variable_is_left_exactly_as_written(self):
        # [FUNCTION DOC] A plain host path leaves the default mount unchanged.
        """Several releases sharing one directory is a first-class choice, not an escape hatch."""
        container = orchestrator_container(*volumes_args(host_path_volume(path="/cluster-storage")))

        assert {"name": "cluster-storage", "mountPath": "/cluster-storage"} in container["volumeMounts"]

    def test_the_default_runs_root_is_a_directory_this_namespace_has_to_itself(self):
        # [FUNCTION DOC] With a mount that does not cover the default runs root, the error names the
        # default root "/cluster-storage/<namespace>/miles_data", i.e. the default includes the
        # namespace.
        """Isolation has to be what you get without asking: a default without it makes it one more thing to remember."""
        error = render_run_error(*volumes_args(host_path_volume(mounts=[{"mountPath": "/elsewhere"}])))

        assert f"/cluster-storage/{NAMESPACE}/miles_data" in error

    def test_the_mount_check_compares_resolved_paths(self):
        # [FUNCTION DOC] A mount at "/cluster-storage/<namespace>" covers the default runs root, so
        # rendering succeeds.
        """Comparing an unresolved runs root against a resolved mount path would refuse one that is in fact mounted."""
        container = orchestrator_container(
            *volumes_args(host_path_volume(mounts=[{"mountPath": f"/cluster-storage/{NAMESPACE}"}]))
        )

        assert container["image"]

    def test_an_unknown_variable_in_a_host_path_is_refused(self):
        # [FUNCTION DOC] An unknown placeholder in a host path fails with an error naming the exact
        # value path of the volume's host path and the placeholder.
        """Left in place it would name one literal directory for every namespace, which is the collision itself."""
        error = render_run_error(*volumes_args(host_path_volume(path="/data/${RELEASE}")))

        assert "infra.volumes[cluster-storage].hostPath.path" in error
        assert "${RELEASE}" in error

    def test_an_unknown_variable_in_a_mount_path_is_refused(self):
        # [FUNCTION DOC] Same for a mount path: the error names the volume, mount index and field.
        """The refusal has to say which mount of which volume, because a values file has many of both."""
        error = render_run_error(*volumes_args(host_path_volume(mounts=[{"mountPath": "/cluster-storage/${USER}"}])))

        assert "infra.volumes[cluster-storage].mounts[0].mountPath" in error
        assert "${USER}" in error

    def test_an_unknown_variable_in_a_sub_path_is_refused(self):
        # [FUNCTION DOC] Same for a sub path (a misspelled namespace placeholder in the second mount).
        """A misspelt ${NAMESPACE} is the likely typo, and it is the one that silently shares a directory."""
        error = render_run_error(
            *volumes_args(
                host_path_volume(
                    mounts=[
                        {"mountPath": "/cluster-storage"},
                        {"mountPath": "/root/miles", "subPath": "repos/${NAMESPCE}/miles"},
                    ]
                )
            )
        )

        assert "infra.volumes[cluster-storage].mounts[1].subPath" in error
        assert "${NAMESPCE}" in error

    def test_an_unknown_variable_in_the_runs_root_is_refused(self):
        # [FUNCTION DOC] Same for the runs root value.
        """Every run writes its state and exit file here, so a directory shared by accident is a run lost."""
        error = render_run_error("--set", "infra.paths.runsRoot=/cluster-storage/${NAMESPCE}/data")

        assert "infra.paths.runsRoot" in error
        assert "${NAMESPCE}" in error

    def test_an_unknown_variable_is_refused_by_a_render_that_deploys_nothing(self):
        # [FUNCTION DOC] Even with an empty orchestrator command (nothing deployed), an unknown
        # placeholder in the runs root is refused with an error naming the value and placeholder.
        """Every other template of this chart renders for some topology only, so the check needs a home without one."""
        error = render_run_error(
            "--set-json",
            "run.orchestrator.command=[]",
            "--set",
            "infra.paths.runsRoot=/cluster-storage/${NAMESPCE}/data",
        )

        assert "infra.paths.runsRoot" in error
        assert "${NAMESPCE}" in error
