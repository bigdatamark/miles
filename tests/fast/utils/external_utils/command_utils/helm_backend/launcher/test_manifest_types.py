# [MODULE DOC] Tests for the manifest parsing types of the helm launcher: parsing rendered multi
# document yaml into a manifest of typed objects (pod workload objects that model replicas and
# pod containers versus general objects carried through untouched), object identity (api group,
# kind, namespace, name; an unset namespace means the release namespace), duplicate detection,
# reading a flag value or the state file from a stateful set container command (both separate and
# equals sign spellings, last one wins, no prefix matches), the hot restart stamp annotation, lookup
# by object key, and tolerance of non string or null annotations. Documents are built as mappings
# and serialized to yaml with sorted keys; the release namespace is rl.
from typing import Any

import pytest
import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    RESTART_AT_ANNOTATION,
    STATEFUL_SET_KIND,
    GeneralManifestObject,
    Manifest,
    ManifestObjectKey,
    PodWorkloadObject,
)

ORCHESTRATOR = "myrun-miles-run-orchestrator"
NAMESPACE = "rl"


def _parse(rendered: str) -> Manifest:
    # [FUNCTION DOC] HELPER: Parses rendered yaml text into a manifest for the test namespace.
    return Manifest.parse(rendered, namespace=NAMESPACE)


def _rendered(*documents: dict) -> str:
    # [FUNCTION DOC] HELPER: Serializes each given mapping to yaml with sorted keys and joins them
    # as separate documents, with a document separator before the first and between each. Returns
    # the text.
    return "---\n" + "---\n".join(yaml.safe_dump(document, sort_keys=True) for document in documents)


def _stateful_set(
    *,
    name: str = ORCHESTRATOR,
    command: list[str] | None = None,
    annotations: dict[str, Any] | None = None,
    namespace: str | None = None,
) -> dict:
    # [FUNCTION DOC] HELPER: Builds a stateful set mapping (apps group) with one replica and one
    # container named orchestrator with a dev image. Keyword only inputs: name (default the
    # orchestrator name), command (added to the container only when given), annotations (put under
    # the pod template metadata only when given, even if empty) and namespace (put in metadata only
    # when given). Returns the mapping.
    container = {"name": "orchestrator", "image": "miles:dev"}
    if command is not None:
        container["command"] = command
    template: dict[str, Any] = {"spec": {"containers": [container]}}
    if annotations is not None:
        template["metadata"] = {"annotations": annotations}
    metadata: dict[str, Any] = {"name": name}
    if namespace is not None:
        metadata["namespace"] = namespace
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": metadata,
        "spec": {"replicas": 1, "template": template},
    }


class TestParse:
    # [CLASS DOC] Basic parsing behavior of the manifest.
    def test_identifies_an_object_the_way_the_api_server_does(self):
        # [FUNCTION DOC] A parsed stateful set is indexed by an identity tuple of api group, kind,
        # namespace (defaulting to the release namespace) and name.
        """Kind and name alone name several distinct objects, and a run may install more than one of them."""
        manifest = _parse(_rendered(_stateful_set()))

        assert list(manifest.by_identity) == [("apps", "StatefulSet", NAMESPACE, ORCHESTRATOR)]

    def test_skips_the_empty_documents_helm_leaves_behind(self):
        # [FUNCTION DOC] Text with extra empty documents before a real one parses to exactly one
        # object.
        """A template whose guard is off renders to nothing, and yaml reads that as a None document."""
        manifest = _parse("---\n---\n" + _rendered(_stateful_set()))

        assert len(manifest.objects) == 1

    def test_reads_the_replica_count_a_resize_moves(self):
        # [FUNCTION DOC] A stateful set parses to a pod workload object with replicas 1; a config
        # map (here without an api version, so the core group) parses to a general object.
        """The upgrade check compares this number, and a kind that carries no pod must not answer for it at all."""
        manifest = _parse(_rendered(_stateful_set(), {"kind": "ConfigMap", "metadata": {"name": "values"}}))

        stateful_set = manifest.by_identity[("apps", "StatefulSet", NAMESPACE, ORCHESTRATOR)]
        config_map = manifest.by_identity[("", "ConfigMap", NAMESPACE, "values")]

        assert isinstance(stateful_set, PodWorkloadObject) and stateful_set.replicas == 1
        assert isinstance(config_map, GeneralManifestObject)


class TestIdentity:
    # [CLASS DOC] Tests of what makes two objects the same object, and of duplicate refusal.
    def test_an_object_of_another_api_group_is_not_the_same_object(self):
        # [FUNCTION DOC] Services of equal kind and name in a custom group and the core group give
        # two index entries.
        """A crd may share a kind and a name with a built-in, and folding them hides one of the two."""
        mine = {"apiVersion": "example.com/v1", "kind": "Service", "metadata": {"name": "engine"}}
        builtin = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine"}}

        assert len(_parse(_rendered(mine, builtin)).by_identity) == 2

    def test_an_object_of_another_namespace_is_not_the_same_object(self):
        # [FUNCTION DOC] Same kind and name, one with no namespace and one naming another
        # namespace: two index entries.
        """An object may name any namespace, and two that do are two objects the cluster keeps apart."""
        here = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine"}}
        elsewhere = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine", "namespace": "other"}}

        assert len(_parse(_rendered(here, elsewhere)).by_identity) == 2

    def test_naming_this_release_own_namespace_is_the_same_object(self):
        # [FUNCTION DOC] An object with no namespace and the same object spelling out the release
        # namespace produce identical index keys.
        """Spelling out the namespace the object lands in anyway must not read as a second object."""
        implicit = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine"}}
        explicit = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine", "namespace": NAMESPACE}}

        assert _parse(_rendered(implicit)).by_identity.keys() == _parse(_rendered(explicit)).by_identity.keys()

    def test_two_objects_of_one_identity_are_refused(self):
        # [FUNCTION DOC] Parsing the same service twice and then reading the index raises an
        # assertion error whose message contains the kind, namespace and name path.
        """Keeping only the last of them would let a relaunch change the other one without being noticed."""
        service = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine"}}

        with pytest.raises(AssertionError, match="Service/rl/engine"):
            assert _parse(_rendered(service, service)).by_identity

    def test_two_served_versions_of_one_object_are_refused(self):
        # [FUNCTION DOC] Two documents of one group, kind and name but different versions count as
        # the same identity: reading the index raises an assertion error naming group, kind,
        # namespace and name.
        """A crd answers on every version it serves, and the two spellings are one object to apply."""
        served = {"apiVersion": "example.com/v1", "kind": "Engine", "metadata": {"name": "engine"}}
        aliased = {**served, "apiVersion": "example.com/v1beta1"}

        with pytest.raises(AssertionError, match="example.com/Engine/rl/engine"):
            assert _parse(_rendered(served, aliased)).by_identity


class TestFlagValue:
    # [CLASS DOC] Tests of reading a command line flag value from a named stateful set and
    # container.
    def test_reads_what_the_installed_release_was_told(self):
        # [FUNCTION DOC] Reading the state file flag returns the argument following it.
        """The pod command line is the only place a launch can read back what the one before it decided."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert (
            manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator") == "/runs/a.state"
        )

    def test_reads_only_the_object_it_was_asked_about(self):
        # [FUNCTION DOC] Asking for a stateful set name that does not exist returns none.
        """Every pod of a run is launched by the same image, so a flag found elsewhere means something else."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert manifest.flag_value("--state-file", stateful_set="another-run", container="orchestrator") is None

    def test_reads_only_the_container_it_was_asked_about(self):
        # [FUNCTION DOC] Asking for a container name that does not exist in the stateful set
        # returns none.
        """One pod can carry a sidecar, and its command line answers for the sidecar only."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="worker") is None

    def test_refuses_a_flag_the_command_leaves_unanswered(self):
        # [FUNCTION DOC] A command ending with the flag and no value makes the read raise an
        # assertion error mentioning that the flag takes a value.
        """Reading past the end would crash with an index, which says nothing about which release is malformed."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file"])))

        with pytest.raises(AssertionError, match="takes a value"):
            manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator")

    def test_refuses_two_containers_with_the_requested_name(self) -> None:
        # [FUNCTION DOC] A stateful set with two containers of the same name makes the read raise
        # an assertion error saying it declares 2 containers with that name.
        """Duplicate container names make the requested command line ambiguous."""
        stateful_set = _stateful_set(command=["python", "--state-file", "/runs/a.state"])
        stateful_set["spec"]["template"]["spec"]["containers"].append(
            {"name": "orchestrator", "command": ["python", "--state-file", "/runs/b.state"]}
        )
        manifest = _parse(_rendered(stateful_set))

        with pytest.raises(AssertionError, match="declares 2 containers named 'orchestrator'"):
            manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator")

    def test_refuses_two_stateful_sets_with_the_requested_name(self) -> None:
        # [FUNCTION DOC] Two same-named stateful sets in different namespaces: the read does not
        # raise and returns the value from the one in the release namespace.
        # KNOWN ISSUE: the test name says it refuses two stateful sets, but the body asserts the
        # read succeeds with the release namespace one's value (no refusal is expected).
        """A same-named StatefulSet in another namespace must not make the release ambiguous."""
        here = _stateful_set(command=["python", "--state-file", "/runs/a.state"])
        here["metadata"]["namespace"] = NAMESPACE
        elsewhere = _stateful_set(command=["python", "--state-file", "/runs/b.state"])
        elsewhere["metadata"]["namespace"] = "other"
        manifest = _parse(_rendered(here, elsewhere))

        assert (
            manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator") == "/runs/a.state"
        )


class TestStateFile:
    # [CLASS DOC] Tests of the state file lookup built on the flag value read.
    def test_finds_the_file_the_installed_orchestrator_already_writes(self):
        # [FUNCTION DOC] The state file lookup returns a path equal to the flag's value.
        """Re-attaching means waiting on the verdict of the launch that is running, not opening a second one."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert str(manifest.state_file(stateful_set=ORCHESTRATOR, container="orchestrator")) == "/runs/a.state"

    def test_names_nothing_when_no_container_of_that_name_carries_the_flag(self):
        # [FUNCTION DOC] A command without the flag makes the lookup return none.
        """A release installed without an orchestrator has no verdict to inherit."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "-m", "something"])))

        assert manifest.state_file(stateful_set=ORCHESTRATOR, container="orchestrator") is None


class TestKindsItDoesNotModel:
    # [CLASS DOC] Tests that kinds without a dedicated model are kept verbatim as general objects.
    def test_carries_every_kind_this_chart_renders_through_untouched(self):
        # [FUNCTION DOC] Four documents (service, leader-worker-set, role with rules, batch job)
        # are parsed; the list of each object's original body mapping equals the input documents,
        # in order.
        """Only replicas and a pod template are read; the rest of a spec still has to reach the diff verbatim."""
        documents = [
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": "engine"},
                "spec": {"clusterIP": "None", "ports": [{"port": 30000, "name": "primary"}]},
            },
            {
                "apiVersion": "leaderworkerset.x-k8s.io/v1",
                "kind": "LeaderWorkerSet",
                "metadata": {"name": "engine"},
                "spec": {
                    "replicas": 2,
                    "leaderWorkerTemplate": {"size": 2, "workerTemplate": {"spec": {"containers": []}}},
                },
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": {"name": "pairing"},
                "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}],
            },
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": "uninstall"},
                "spec": {
                    "completions": 1,
                    "template": {"spec": {"containers": [{"name": "helm", "image": "miles:dev"}]}},
                },
            },
        ]

        manifest = _parse(_rendered(*documents))

        assert [described.body for described in manifest.objects] == documents

    def test_reads_no_replicas_off_a_kind_that_has_none(self):
        # [FUNCTION DOC] A service document yields an empty pod workloads list.
        """A Service and a Role never scale, and inventing a count for them would read as a resize."""
        manifest = _parse(
            _rendered({"kind": "Service", "metadata": {"name": "engine"}, "spec": {"clusterIP": "None"}})
        )

        assert manifest.pod_workloads == []

    def test_finds_no_container_in_a_workload_it_does_not_model(self):
        # [FUNCTION DOC] A leader-worker-set with a container named orchestrator carrying the
        # state file flag: the state file lookup by that name as a stateful set returns none.
        """The state file flag means the orchestrator's container, and every other pod runs the same image."""
        manifest = _parse(
            _rendered(
                {
                    "apiVersion": "leaderworkerset.x-k8s.io/v1",
                    "kind": "LeaderWorkerSet",
                    "metadata": {"name": "engine"},
                    "spec": {
                        "leaderWorkerTemplate": {
                            "workerTemplate": {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "orchestrator",
                                            "command": ["python", "--state-file", "/runs/a.state"],
                                        }
                                    ]
                                }
                            }
                        }
                    },
                }
            )
        )

        assert manifest.state_file(stateful_set="engine", container="orchestrator") is None

    def test_keeps_a_custom_resource_whose_spec_is_shaped_like_no_pod(self):
        # [FUNCTION DOC] A custom monitoring rule resource parses without error to an object that
        # is not a pod workload, whose spec and body equal the input.
        """An extra manifest is any object the user hands the chart, and validating it as a workload rejects it."""
        document = {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "PrometheusRule",
            "metadata": {"name": "alerts"},
            "spec": {"groups": [{"name": "run", "rules": [{"alert": "Down", "expr": "up == 0"}]}]},
        }

        manifest = _parse(_rendered(document))

        described = manifest.objects[0]

        assert not isinstance(described, PodWorkloadObject)
        assert described.spec == document["spec"]
        assert described.body == document

    def test_decodes_a_workload_kind_as_the_object_that_carries_a_pod(self):
        # [FUNCTION DOC] A stateful set parses to a pod workload object with replicas 1 and one
        # container named orchestrator.
        """Only the variant that models a pod may answer for replicas and containers, so the kind has to pick it."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "train.py"])))

        described = manifest.objects[0]

        assert isinstance(described, PodWorkloadObject)
        assert described.replicas == 1
        assert [found.name for found in described.containers] == ["orchestrator"]


_STAMP = "2026-08-12T09:00:00+00:00"
_STAMP_WITH_MICROSECONDS = "2026-08-12T09:00:00.123456+00:00"


class TestTheRestartStamp:
    # [CLASS DOC] Tests of reading the restart-at annotation of a pod template by object name.
    def test_a_manifest_that_was_never_hot_restarted_carries_none(self):
        # [FUNCTION DOC] Without the annotation, the stamp read returns none.
        """Inventing a stamp would roll the pods of every run on its first ordinary relaunch."""
        manifest = _parse(_rendered(_stateful_set(name="orchestrator")))

        assert manifest.restart_at(object_name="orchestrator") is None

    def test_each_object_is_asked_for_its_own_stamp(self):
        # [FUNCTION DOC] Two stateful sets, only one stamped: each name returns its own value
        # (the stamp or none).
        """A pool that never got the stamp must not be rendered with it and turned into a refused diff."""
        manifest = _parse(
            _rendered(
                _stateful_set(name="orchestrator", annotations={RESTART_AT_ANNOTATION: _STAMP}),
                _stateful_set(name="rollout-executor"),
            )
        )

        assert manifest.restart_at(object_name="orchestrator") == _STAMP
        assert manifest.restart_at(object_name="rollout-executor") is None

    def test_the_stamp_is_read_off_the_stateful_set_and_not_off_a_service_of_the_same_name(self):
        # [FUNCTION DOC] A service listed before a stamped stateful set, same name: the stamp read
        # still returns the stateful set's stamp.
        """The chart renders the headless Service first, and reading that one always answers None."""
        manifest = _parse(
            _rendered(
                {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "orchestrator"}},
                _stateful_set(name="orchestrator", annotations={RESTART_AT_ANNOTATION: _STAMP}),
            )
        )

        assert manifest.restart_at(object_name="orchestrator") == _STAMP

    @pytest.mark.parametrize("stamp", [_STAMP, _STAMP_WITH_MICROSECONDS])
    def test_a_stamp_is_read_back_exactly_as_it_was_written(self, stamp: str):
        # [FUNCTION DOC] Parametrized over a timestamp without and with microseconds: the stamp
        # text is returned unchanged (same spelling).
        """A stamp that comes back in another spelling is a pod template diff the gate refuses forever."""
        manifest = _parse(_rendered(_stateful_set(name="orchestrator", annotations={RESTART_AT_ANNOTATION: stamp})))

        assert manifest.restart_at(object_name="orchestrator") == stamp

    def test_a_stamp_that_is_no_timestamp_is_carried_as_it_stands(self):
        # [FUNCTION DOC] A stamp that is not a timestamp (a plain word) is returned verbatim.
        """This launch never has to read a stamp, only to render back the one it found."""
        manifest = _parse(_rendered(_stateful_set(name="orchestrator", annotations={RESTART_AT_ANNOTATION: "soon"})))

        assert manifest.restart_at(object_name="orchestrator") == "soon"


class TestLookingAnObjectUpByItsKey:
    # [CLASS DOC] Tests of the object lookup by a key made of kind and name.
    def test_a_name_shared_by_two_kinds_answers_the_kind_that_was_asked_for(self):
        # [FUNCTION DOC] A service and a stateful set of equal name: looking up by the stateful
        # set kind returns an object of that kind.
        """The chart renders a Service beside every StatefulSet, and they share the object's name."""
        manifest = _parse(
            _rendered(
                {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "orchestrator"}},
                _stateful_set(name="orchestrator", command=["python", "train.py"]),
            )
        )

        found = manifest.object_keyed(key=ManifestObjectKey(kind=STATEFUL_SET_KIND, name="orchestrator"))

        assert found is not None and found.kind == STATEFUL_SET_KIND

    def test_a_key_no_object_carries_answers_nothing(self):
        # [FUNCTION DOC] Looking up a name that is absent returns none, without error.
        """A release that carries no such object is the ordinary answer, not a failure."""
        manifest = _parse(_rendered(_stateful_set(name="orchestrator")))

        assert manifest.object_keyed(key=ManifestObjectKey(kind=STATEFUL_SET_KIND, name="engine")) is None

    def test_a_key_two_objects_share_stops_the_launch(self):
        # [FUNCTION DOC] Two stateful sets of the same name in different namespaces: the lookup
        # raises an assertion error mentioning objects keyed.
        """Answering for whichever rendered first would gate the launch on an object nobody named."""
        manifest = _parse(
            _rendered(
                _stateful_set(name="orchestrator", namespace="rl"),
                _stateful_set(name="orchestrator", namespace="other"),
            )
        )

        with pytest.raises(AssertionError, match="objects keyed"):
            manifest.object_keyed(key=ManifestObjectKey(kind=STATEFUL_SET_KIND, name="orchestrator"))


class TestAnnotationsThatAreNotStrings:
    # [CLASS DOC] Tests tolerance of non string annotation values.
    def test_an_object_carrying_a_non_string_annotation_still_parses(self):
        # [FUNCTION DOC] An annotation with an integer value does not break parsing, and the
        # stamp read returns none.
        """A user's own annotation must not make the launcher fail to read the release it is upgrading."""
        manifest = _parse(_rendered(_stateful_set(name="orchestrator", annotations={"replicas": 3})))

        assert manifest.restart_at(object_name="orchestrator") is None


class TestWhichStatefulSetAFlagIsReadOff:
    # [CLASS DOC] Tests of which stateful set a flag read matches (release namespace and core
    # apps group only).
    def test_a_stateful_set_of_the_same_name_in_another_namespace_does_not_join_the_match(self):
        # [FUNCTION DOC] Same name in the release namespace and another namespace: the read
        # returns the release namespace one's flag value and does not raise.
        """Two of them tripped the at-most-one assert, which aborted every relaunch of the run."""
        manifest = _parse(
            _rendered(
                _stateful_set(command=["python", "--state-file", "/runs/a.state"]),
                _stateful_set(command=["python", "--state-file", "/runs/b.state"], namespace="other"),
            )
        )

        assert (
            manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator") == "/runs/a.state"
        )

    def test_a_stateful_set_of_another_api_group_answers_for_nothing(self):
        # [FUNCTION DOC] A stateful set kind in a custom group is ignored: the flag read returns
        # none.
        """A crd may share the kind and the name, and its command line says nothing about this release."""
        foreign = _stateful_set(command=["python", "--state-file", "/runs/foreign.state"])
        foreign["apiVersion"] = "example.com/v1"

        manifest = _parse(_rendered(foreign))

        assert manifest.flag_value("--state-file", stateful_set=ORCHESTRATOR, container="orchestrator") is None

    def test_the_stateful_set_this_release_installed_is_still_the_one_read(self):
        # [FUNCTION DOC] An ordinary apps-group stateful set is still found: the state file lookup
        # returns its flag value.
        """Narrowing the match must not stop the ordinary release from being found at all."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert str(manifest.state_file(stateful_set=ORCHESTRATOR, container="orchestrator")) == "/runs/a.state"


class TestHowAFlagMayBeSpelled:
    # [CLASS DOC] Tests of the accepted spellings of a flag in a command: separate value argument
    # or joined by an equals sign.
    def test_a_flag_joined_to_its_value_by_an_equals_sign_is_read(self):
        # [FUNCTION DOC] The equals sign spelling of the run uuid flag yields its value.
        """The uuid is passed this way, and reading it as absent made the relaunch mint a new one."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--run-uuid=abc123"])))

        assert manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator") == "abc123"

    def test_a_flag_separated_from_its_value_is_still_read(self):
        # [FUNCTION DOC] The separate argument spelling yields its value.
        """Both spellings reach the container the same way, and the launcher has to read either."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--run-uuid", "abc123"])))

        assert manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator") == "abc123"

    def test_the_last_of_two_spellings_of_one_flag_wins(self):
        # [FUNCTION DOC] When the flag appears twice (separate then equals sign), the last
        # occurrence's value is returned.
        """argparse reads the last occurrence, so anything else answers with a value the run never used."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--run-uuid", "first", "--run-uuid=second"])))

        assert manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator") == "second"

    def test_a_flag_the_command_never_names_is_still_absent(self):
        # [FUNCTION DOC] Reading a flag not present in the command returns none.
        """A release installed without the flag has nothing for the relaunch to inherit."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator") is None

    def test_a_longer_flag_that_starts_with_the_one_asked_for_is_not_read(self):
        # [FUNCTION DOC] A different flag that merely has the asked flag as a prefix, in equals
        # spelling, is not matched: returns none.
        """--run-uuid-file is another argument, and answering with its value would resume the wrong run."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--run-uuid-file=/runs/uuid"])))

        assert manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator") is None

    def test_a_command_ending_with_the_bare_flag_still_stops_the_launch(self):
        # [FUNCTION DOC] A command whose last argument is the bare flag raises an assertion error
        # mentioning that the flag takes a value.
        """Reading past the end says nothing about which release is malformed."""
        manifest = _parse(_rendered(_stateful_set(command=["python", "--run-uuid"])))

        with pytest.raises(AssertionError, match="takes a value"):
            manifest.flag_value("--run-uuid", stateful_set=ORCHESTRATOR, container="orchestrator")


def _stateful_set_with_null_annotations(name: str = ORCHESTRATOR) -> dict:
    # [FUNCTION DOC] HELPER: Returns a stateful set mapping (default orchestrator name) whose pod
    # template metadata has an annotations field explicitly set to null.
    document = _stateful_set(name=name)
    document["spec"]["template"]["metadata"] = {"annotations": None}
    return document


class TestAPodTemplateWhoseAnnotationsAreNull:
    # [CLASS DOC] Tests that a null annotations block is normalized to an empty mapping.
    def test_a_release_carrying_one_still_parses(self):
        # [FUNCTION DOC] Parsing succeeds and the single object's metadata name is the
        # orchestrator name.
        """An extra manifest written this way used to take the whole relaunch down in Manifest.parse."""
        manifest = _parse(_rendered(_stateful_set_with_null_annotations()))

        assert [described.metadata.name for described in manifest.objects] == [ORCHESTRATOR]

    def test_it_reads_as_an_object_that_carries_no_restart_stamp(self):
        # [FUNCTION DOC] The stamp read on such an object returns none.
        """An empty annotations block and a null one say the same thing: this object was never hot restarted."""
        manifest = _parse(_rendered(_stateful_set_with_null_annotations()))

        assert manifest.restart_at(object_name=ORCHESTRATOR) is None

    def test_the_null_block_reaches_the_diff_as_the_empty_block_it_was_read_as(self):
        # [FUNCTION DOC] The parsed object's body shows the pod template metadata annotations as
        # an empty mapping instead of null.
        """What the launch compares is the parsed object, so the normalization has to be visible there too."""
        manifest = _parse(_rendered(_stateful_set_with_null_annotations()))

        assert manifest.objects[0].body["spec"]["template"]["metadata"] == {"annotations": {}}

    def test_an_object_that_does_carry_annotations_still_reads_them(self):
        # [FUNCTION DOC] A real stamp annotation is still returned by the stamp read.
        """Normalizing the null must not swallow the stamp a hot restart writes into the same block."""
        manifest = _parse(_rendered(_stateful_set(annotations={RESTART_AT_ANNOTATION: _STAMP})))

        assert manifest.restart_at(object_name=ORCHESTRATOR) == _STAMP
