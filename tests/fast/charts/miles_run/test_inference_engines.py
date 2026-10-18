# [MODULE DOC]
# Tests (helm rendering of the run chart, needs helm) of the inference engines value: each entry
# renders one leader-worker-set with replicas and size (defaulting to one each), a recreate-group
# restart policy, a single shared worker template with the command passed through untouched, the
# launcher-computed GPU limit, service links disabled, one set per entry named from the object
# name, none when the inference pool is external, and per-entry sizing. ONE_GROUP is the sample
# entry: two replicas of four pods, a distributed launch command with worker-index placeholders, an
# env var, a port and an eight GPU limit.
import json

from tests.fast.charts.utils import (
    objects_of_kind,
    render_run,
    requires_helm,
    single_object_of_kind,
    with_object_names,
)

ONE_GROUP = [
    {
        "name": "inference-engine-0-0",
        "replicas": 2,
        "size": 4,
        "command": [
            "python",
            "-m",
            "sglang.launch_server",
            "--node-rank",
            "$(LWS_WORKER_INDEX)",
            "--dist-init-addr",
            "$(LWS_LEADER_ADDRESS):9000",
        ],
        "env": {"NVSHMEM_DISABLE_NCCL": "1"},
        "ports": [{"name": "primary", "port": 8000}],
        "resources": {"limits": {"nvidia.com/gpu": 8}},
    }
]


def _render(engines=ONE_GROUP, *args):
    # [FUNCTION DOC] HELPER: Renders the run chart with the given inference engine entries (default
    # the sample group, given object names derived from their names) and extra helm arguments;
    # returns the parsed objects.
    return render_run("--set-json", f"run.inferenceEngines={json.dumps(with_object_names(engines))}", *args)


def _leader_worker_set(engines=ONE_GROUP):
    # [FUNCTION DOC] HELPER: Renders with the given engines and returns the single leader-worker-set
    # (asserts exactly one).
    return single_object_of_kind(_render(engines), "LeaderWorkerSet")


@requires_helm
class TestInferenceEngines:
    # [CLASS DOC] Tests of the rendered leader-worker-sets.
    def test_a_group_is_one_engine_and_a_pod_is_one_of_its_nodes(self):
        # [FUNCTION DOC] The sample entry renders with two replicas and a group size of four.
        """A multi-node engine must be scheduled and restarted as a unit, which is what a group gives."""
        spec = _leader_worker_set()["spec"]

        assert spec["replicas"] == 2
        assert spec["leaderWorkerTemplate"]["size"] == 4

    def test_restarts_the_whole_engine_when_one_rank_dies(self):
        # [FUNCTION DOC] The restart policy is "RecreateGroupOnPodRestart".
        """A surviving rank of a dead engine would keep serving requests it can no longer answer."""
        assert _leader_worker_set()["spec"]["leaderWorkerTemplate"]["restartPolicy"] == "RecreateGroupOnPodRestart"

    def test_serves_every_rank_from_one_template(self):
        # [FUNCTION DOC] There is no separate leader template, and the worker template's command
        # contains the worker-index placeholder.
        """Ranks differ only by index, and kubelet expands $(LWS_WORKER_INDEX) per pod."""
        template = _leader_worker_set()["spec"]["leaderWorkerTemplate"]

        assert "leaderTemplate" not in template
        assert "$(LWS_WORKER_INDEX)" in template["workerTemplate"]["spec"]["containers"][0]["command"]

    def test_passes_the_command_through_unchanged(self):
        # [FUNCTION DOC] The worker container's command equals the supplied command exactly.
        """The launcher computed the sglang argv; the chart must not reinterpret its placeholders."""
        container = _leader_worker_set()["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]

        assert container["command"] == ONE_GROUP[0]["command"]

    def test_requests_the_gpus_the_launcher_computed(self):
        # [FUNCTION DOC] The container resources equal exactly a GPU limit of eight.
        """An engine pod owning a whole node must hold every gpu on it."""
        container = _leader_worker_set()["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]

        assert container["resources"] == {"limits": {"nvidia.com/gpu": 8}}

    def test_disables_the_service_link_environment(self):
        # [FUNCTION DOC] The worker pod's enable-service-links flag is false.
        """Engine pods import miles too, so the injected <SERVICE>_PORT vars would break arg parsing."""
        pod = _leader_worker_set()["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]

        assert pod["enableServiceLinks"] is False

    def test_defaults_a_single_pod_engine(self):
        # [FUNCTION DOC] An entry with only a name and command renders with one replica and size one.
        """A one-node engine is the common case and must not have to spell out replicas and size."""
        spec = _leader_worker_set([{"name": "engine", "command": ["python"]}])["spec"]

        assert (spec["replicas"], spec["leaderWorkerTemplate"]["size"]) == (1, 1)

    def test_renders_nothing_for_an_external_inference_pool(self):
        # [FUNCTION DOC] A default run (no engine entries) renders no leader-worker-set.
        """External rollout brings its own engines, and the run must install without any."""
        assert objects_of_kind(render_run(), "LeaderWorkerSet") == []

    def test_gives_every_group_of_every_model_its_own_pool(self):
        # [FUNCTION DOC] Three entries render three leader-worker-sets named "myrun-miles-run-" plus
        # each entry's name, in the given order.
        """Multi-model and prefill/decode runs differ only in how many groups the specs name."""
        groups = [
            {"name": "inference-engine-0-0", "command": ["python", "-m", "a"]},
            {"name": "inference-engine-1-0", "command": ["python", "-m", "b"]},
            {"name": "inference-engine-1-1", "command": ["python", "-m", "b", "--disaggregation-mode", "decode"]},
        ]

        rendered = objects_of_kind(_render(groups), "LeaderWorkerSet")

        assert [obj["metadata"]["name"] for obj in rendered] == [
            "myrun-miles-run-inference-engine-0-0",
            "myrun-miles-run-inference-engine-1-0",
            "myrun-miles-run-inference-engine-1-1",
        ]

    def test_sizes_each_group_on_its_own(self):
        # [FUNCTION DOC] A prefill entry (one replica of four pods) and a decode entry (eight
        # replicas of one pod) render with exactly those replica and size pairs, keyed by set name.
        """A prefill engine and a decode engine of one model rarely span the same number of nodes."""
        groups = [
            {"name": "prefill", "replicas": 1, "size": 4, "command": ["python"]},
            {"name": "decode", "replicas": 8, "size": 1, "command": ["python"]},
        ]

        rendered = objects_of_kind(_render(groups), "LeaderWorkerSet")
        sizes = {
            obj["metadata"]["name"]: (obj["spec"]["replicas"], obj["spec"]["leaderWorkerTemplate"]["size"])
            for obj in rendered
        }

        assert sizes == {"myrun-miles-run-prefill": (1, 4), "myrun-miles-run-decode": (8, 1)}
