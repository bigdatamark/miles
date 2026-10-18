# [MODULE DOC]
# Tests (helm rendering of the run chart, needs helm) of the trainer engines value: each entry
# renders one leader-worker-set (a group per data-parallel replica) with replicas and size
# (defaulting to one each), recreate-group restart policy, a container named "trainer", the
# supervisor command passed through, cell-index and pod-index variables projected from the
# group-index and worker-index labels, and nothing rendered when the list is empty. ACTOR_AND_CRITIC
# is the sample: an actor (two replicas of two pods, env, port, eight GPUs) and a minimal critic.
import json

from tests.fast.charts.utils import objects_of_kind, render_run, requires_helm, with_object_names

ACTOR_AND_CRITIC = [
    {
        "name": "trainer-engine-actor",
        "replicas": 2,
        "size": 2,
        "command": ["python", "-m", "miles.utils.workers.process_supervisor", "--rank-base", "$(LWS_WORKER_INDEX)"],
        "env": {"NCCL_CUMEM_ENABLE": "0"},
        "ports": [{"name": "master", "port": 9000}],
        "resources": {"limits": {"nvidia.com/gpu": 8}},
    },
    {"name": "trainer-engine-critic", "command": ["python", "-m", "miles.utils.workers.process_supervisor"]},
]


def _render(trainers=ACTOR_AND_CRITIC):
    # [FUNCTION DOC] HELPER: Renders the run chart with the trainer entries (given object names)
    # and returns only the leader-worker-set objects.
    return objects_of_kind(
        render_run("--set-json", f"run.trainerEngines={json.dumps(with_object_names(trainers))}"), "LeaderWorkerSet"
    )


@requires_helm
class TestTrainers:
    # [CLASS DOC] Tests of the rendered trainer leader-worker-sets.
    def test_a_group_is_a_dp_group_so_healing_one_leaves_the_others_running(self):
        # [FUNCTION DOC] The actor set has two replicas, group size two and the recreate-group
        # restart policy.
        """Miles restarts training a dp group at a time, which is exactly a LeaderWorkerSet group."""
        actor = _render()[0]

        assert actor["spec"]["replicas"] == 2
        assert actor["spec"]["leaderWorkerTemplate"]["size"] == 2
        assert actor["spec"]["leaderWorkerTemplate"]["restartPolicy"] == "RecreateGroupOnPodRestart"

    def test_gives_each_role_its_own_pool(self):
        # [FUNCTION DOC] Two sets render, named by actor then critic object names.
        """An actor and a critic size their groups differently, so one pool_id cannot hold both."""
        assert [obj["metadata"]["name"] for obj in _render()] == [
            "myrun-miles-run-trainer-engine-actor",
            "myrun-miles-run-trainer-engine-critic",
        ]

    def test_names_the_container_after_its_role(self):
        # [FUNCTION DOC] The actor pod's containers are exactly one named trainer.
        """kubectl logs picks a container by name, and "worker" would not say which pool_id it belongs to."""
        pod = _render()[0]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]

        assert [container["name"] for container in pod["containers"]] == ["trainer"]

    def test_passes_the_supervisor_command_through_unchanged(self):
        # [FUNCTION DOC] The actor container's command equals the supplied command.
        """A pod holds one node's ranks, and the launcher already computed how to supervise them."""
        container = _render()[0]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]

        assert container["command"] == ACTOR_AND_CRITIC[0]["command"]

    def test_defaults_a_single_group_of_one_pod(self):
        # [FUNCTION DOC] The critic (no replicas or size given) renders with one replica and size one.
        """A critic on one node must not have to spell out replicas and size."""
        critic = _render()[1]["spec"]

        assert (critic["replicas"], critic["leaderWorkerTemplate"]["size"]) == (1, 1)

    def test_renders_nothing_when_training_runs_elsewhere(self):
        # [FUNCTION DOC] An empty trainer list renders no leader-worker-sets.
        """A run that only serves inference installs no trainer."""
        assert _render([]) == []

    def test_projects_the_labels_the_pod_joins_into_its_cell_id(self):
        # [FUNCTION DOC] The cell index variable is a field reference to the group-index label of the
        # leader-worker-set (exactly one such entry in the actor container).
        """The pod and the driver must read the same label, or they number the cell differently."""
        container = _render()[0]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]

        (entry,) = [item for item in container["env"] if item["name"] == "MILES_CELL_INDEX"]

        assert (
            entry["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.labels['leaderworkerset.sigs.k8s.io/group-index']"
        )

    def test_projects_the_label_that_says_which_pod_of_its_cell_this_is(self):
        # [FUNCTION DOC] The pod index variable is a field reference to the worker-index label.
        """A cell spread over pods numbers its workers by pod, so a pod that read zero would shadow the leader."""
        container = _render()[0]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]

        (entry,) = [item for item in container["env"] if item["name"] == "MILES_POD_INDEX"]

        assert (
            entry["valueFrom"]["fieldRef"]["fieldPath"]
            == "metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']"
        )
