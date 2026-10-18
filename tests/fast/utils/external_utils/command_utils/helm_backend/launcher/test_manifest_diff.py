# [MODULE DOC] Tests for the manifest diff used by the helm launcher's upgrade gate, which
# compares the rendered manifests of an installed release (before) with a proposed one (after).
# Verifies: which changes are allowed (only replica growth on pools of the supported
# leader-worker-set api group) and which are refused (image, command, added or removed objects,
# data changes outside config maps, same-name objects in other namespaces or api groups), the
# structured change records (allowed_by reason, path, identity), the rebuild query, the optional
# allowlist of object keys a hot restart may rebuild freely, and the assertion raised when an
# allowlist key matches more than one object. Objects are built as plain mappings, serialized to
# yaml, and parsed into manifests in namespace rl.
import copy
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher import manifest_diff
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    STATEFUL_SET_KIND,
    Manifest,
    ManifestObjectKey,
)

NAMESPACE = "rl"


def _leader_worker_set(replicas: int = 2) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a mapping describing a supported-group leader-worker-set
    # object named for the engine pool, with the given replica count (default 2) and one worker
    # container (name worker, a dev image, command running the engine module).
    return {
        "apiVersion": "leaderworkerset.x-k8s.io/v1",
        "kind": "LeaderWorkerSet",
        "metadata": {"name": "myrun-miles-run-engine"},
        "spec": {
            "replicas": replicas,
            "leaderWorkerTemplate": {
                "workerTemplate": {
                    "spec": {
                        "containers": [{"name": "worker", "image": "miles:dev", "command": ["python", "-m", "engine"]}]
                    }
                }
            },
        },
    }


def _config_map() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a mapping describing a config map named for the run values,
    # holding one data entry for a values file with the text of an empty run section.
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "myrun-miles-run-values"},
        "data": {"values.yaml": "run: {}\n"},
    }


def _stateful_set() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a mapping describing the orchestrator stateful set (apps
    # group) with one replica and one container named orchestrator using a dev image.
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": "myrun-miles-run-orchestrator"},
        "spec": {
            "replicas": 1,
            "template": {"spec": {"containers": [{"name": "orchestrator", "image": "miles:dev"}]}},
        },
    }


def _manifest(objects: list[dict[str, Any]]) -> Manifest:
    # [FUNCTION DOC] HELPER: Serializes each object mapping to yaml with sorted keys, joins them
    # as separate yaml documents (leading and between separators), and parses the text into a
    # manifest for the test namespace. Returns the manifest.
    return Manifest.parse(
        "---\n" + "---\n".join(yaml.safe_dump(document, sort_keys=True) for document in objects), namespace=NAMESPACE
    )


def _objects() -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Returns the baseline release as a list in fixed order: the
    # leader-worker-set (index 0), the config map (index 1), the stateful set (index 2). Tests
    # index into this order.
    return [_leader_worker_set(), _config_map(), _stateful_set()]


def _manifest_after(mutate: Callable[[list[dict[str, Any]]], None]) -> Manifest:
    # [FUNCTION DOC] HELPER: Deep copies the baseline objects, lets the given callback mutate the
    # copy in place, and returns the manifest built from the mutated list.
    objects = copy.deepcopy(_objects())
    mutate(objects)
    return _manifest(objects)


def _worker_container(objects: list[dict[str, Any]]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns (not copies) the first worker container mapping of the
    # leader-worker-set at index 0 of the given list, so callers can mutate it in place.
    return objects[0]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]


class TestManifestScaling:
    # [CLASS DOC] Tests of the allowed scaling change and of the summaries for allowed and
    # disallowed changes.
    def test_allows_a_pool_that_only_grew(self):
        # [FUNCTION DOC] Raising the engine pool replicas from 2 to 6 must leave the diff allowed.
        """Chart templates render the live run's own manifests, and only the cell count may move under it."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects[0]["spec"].update(replicas=6))
        )

        assert diff.is_allowed

    def test_reports_the_replica_change_it_will_apply(self):
        # [FUNCTION DOC] After growing replicas 2 to 6, the allowed change lines are exactly one
        # string: group, kind, namespace and name of the object, then the replica change text.
        """An accepted upgrade still restarts nothing but adds pods, and the user must see how many."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects[0]["spec"].update(replicas=6))
        )

        assert diff.allowed_changed == [
            "leaderworkerset.x-k8s.io/LeaderWorkerSet/rl/myrun-miles-run-engine: replicas 2 -> 6"
        ]

    def test_refuses_a_configmap_whose_data_changed(self):
        # [FUNCTION DOC] Changing the config map's values entry yields one disallowed change line
        # naming the object and the data path of that entry.
        """The only configmap this chart renders holds the job that uninstalls the run, so its content is the run's."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(lambda objects: objects[1]["data"].update({"values.yaml": "run: {id: x}\n"})),
        )

        assert diff.disallowed_changed == ["ConfigMap/rl/myrun-miles-run-values: data.values.yaml"]

    def test_says_so_when_the_rendered_manifests_are_identical(self):
        # [FUNCTION DOC] Diffing identical manifests: the allowed changes summary text must contain
        # the phrase saying there is nothing to change.
        """Relaunching the same run id is how users check on a run, and it must read as a no-op."""
        diff = manifest_diff.diff_manifests(before=_manifest(_objects()), after=_manifest(_objects()))

        assert "nothing to change" in diff.summarize_allowed_changes()


class TestManifestRefusals:
    # [CLASS DOC] Tests of changes that must make the diff not allowed, and of the exact
    # disallowed change strings (object identity followed by the dotted field path, with list
    # positions in brackets).
    def test_refuses_a_changed_container_image(self):
        # [FUNCTION DOC] Changing the worker image: diff not allowed; the single disallowed line
        # points at the image field of the first container.
        """A new image restarts every pod of a live run, which is the restart this check exists to stop."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(lambda objects: _worker_container(objects).update(image="miles:other")),
        )

        assert not diff.is_allowed
        assert diff.disallowed_changed == [
            "leaderworkerset.x-k8s.io/LeaderWorkerSet/rl/myrun-miles-run-engine: "
            "spec.leaderWorkerTemplate.workerTemplate.spec.containers.[0].image"
        ]

    def test_refuses_a_changed_container_command(self):
        # [FUNCTION DOC] Changing the last element of the worker command: not allowed; the
        # disallowed line points at the command list element with index 2.
        """A template-only edit to the launch command would relaunch the pool_id as a different experiment."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(
                lambda objects: _worker_container(objects).update(command=["python", "-m", "other"])
            ),
        )

        assert not diff.is_allowed
        assert diff.disallowed_changed == [
            "leaderworkerset.x-k8s.io/LeaderWorkerSet/rl/myrun-miles-run-engine: "
            "spec.leaderWorkerTemplate.workerTemplate.spec.containers.[0].command.[2]"
        ]

    def test_refuses_an_object_the_upgrade_would_add(self):
        # [FUNCTION DOC] Adding a second leader-worker-set named second: not allowed, and the
        # additions list holds exactly that object's identity string.
        """A workload appearing mid-run was never part of the run being upgraded."""
        second = _leader_worker_set() | {"metadata": {"name": "second"}}
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects.append(second))
        )

        assert not diff.is_allowed
        assert [str(identity) for identity in diff.additions] == ["leaderworkerset.x-k8s.io/LeaderWorkerSet/rl/second"]

    def test_refuses_an_object_the_upgrade_would_delete(self):
        # [FUNCTION DOC] Removing the last baseline object (the orchestrator stateful set): not
        # allowed, and the removals list holds exactly its identity string.
        """Upgrading would remove the orchestrator, and with it the run it is driving."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects.pop())
        )

        assert not diff.is_allowed
        assert [str(identity) for identity in diff.removals] == ["apps/StatefulSet/rl/myrun-miles-run-orchestrator"]

    def test_refuses_replicas_moving_on_a_kind_that_does_not_scale(self):
        # [FUNCTION DOC] Changing the stateful set's replicas to 2: not allowed, reported as a
        # plain disallowed field change at the replicas path (not as a scaling line).
        """Only a LeaderWorkerSet adds cells without touching the pods it already has."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects[2]["spec"].update(replicas=2))
        )

        assert not diff.is_allowed
        assert diff.disallowed_changed == ["apps/StatefulSet/rl/myrun-miles-run-orchestrator: spec.replicas"]

    def test_refuses_a_data_block_moving_on_a_kind_that_is_not_a_configmap(self):
        # [FUNCTION DOC] A secret present in both manifests with a different encoded token value:
        # not allowed, with a disallowed line naming the secret and the token data path.
        """A Secret's data is mounted into running pods, so rewriting it is not a free change."""
        secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "creds"}, "data": {"token": "YQ=="}}
        before = _manifest([*_objects(), secret])
        after = _manifest([*_objects(), secret | {"data": {"token": "Yg=="}}])

        diff = manifest_diff.diff_manifests(before=before, after=after)

        assert not diff.is_allowed
        assert diff.disallowed_changed == ["Secret/rl/creds: data.token"]

    def test_refuses_a_change_to_an_object_another_namespace_shares_a_name_with(self):
        # [FUNCTION DOC] Two services with the same name, one in the release namespace and one in
        # another namespace; only the other one's cluster ip changes. Asserts not allowed and the
        # disallowed line names the other namespace, proving the two are not merged.
        """A release may hold both, and folding them would let this edit through as no change at all."""
        here = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "engine"}, "spec": {"clusterIP": "None"}}
        elsewhere = {**here, "metadata": {"name": "engine", "namespace": "other"}}
        before = _manifest([*_objects(), here, elsewhere])
        after = _manifest([*_objects(), here, {**elsewhere, "spec": {"clusterIP": "10.0.0.1"}}])

        diff = manifest_diff.diff_manifests(before=before, after=after)

        assert not diff.is_allowed
        assert diff.disallowed_changed == ["Service/other/engine: spec.clusterIP"]

    def test_refuses_a_change_to_an_object_another_api_group_shares_a_name_with(self):
        # [FUNCTION DOC] A built in service and a custom-group service of the same kind and name;
        # only the custom one's cluster ip changes. Asserts not allowed and the disallowed line
        # carries the custom api group prefix, proving the api group is part of the identity.
        """A crd of the same kind and name is a second object, and only the api group tells the two apart."""
        builtin = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "engine"},
            "spec": {"clusterIP": "None"},
        }
        crd = {**builtin, "apiVersion": "example.com/v1"}
        before = _manifest([*_objects(), builtin, crd])
        after = _manifest([*_objects(), builtin, {**crd, "spec": {"clusterIP": "10.0.0.1"}}])

        diff = manifest_diff.diff_manifests(before=before, after=after)

        assert not diff.is_allowed
        assert diff.disallowed_changed == ["example.com/Service/rl/engine: spec.clusterIP"]

    def test_names_the_field_it_refused(self):
        # [FUNCTION DOC] After an image change, the diff's full description text contains the path
        # of the image field of the first container.
        """A refusal the user cannot locate just makes them reach for --skip-upgrade-check."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(lambda objects: _worker_container(objects).update(image="miles:other")),
        )

        assert "containers.[0].image" in diff.describe()

    def test_keeps_the_replica_change_alongside_the_refusal(self):
        # [FUNCTION DOC] One mutation both grows replicas to 6 and changes the worker image.
        # Asserts the diff is not allowed yet the allowed changes list still reports the replica
        # change, so a refusal does not hide the scaling part.
        """The user asked for both, and reporting only the refusal hides half of what they asked for."""

        def grow_and_repoint(objects: list[dict[str, Any]]) -> None:
            # [FUNCTION DOC] HELPER: Mutates the list in place: sets the engine pool replicas to 6
            # and changes the worker container image to another value.
            objects[0]["spec"].update(replicas=6)
            _worker_container(objects).update(image="miles:other")

        diff = manifest_diff.diff_manifests(before=_manifest(_objects()), after=_manifest_after(grow_and_repoint))

        assert not diff.is_allowed
        assert diff.allowed_changed == [
            "leaderworkerset.x-k8s.io/LeaderWorkerSet/rl/myrun-miles-run-engine: replicas 2 -> 6"
        ]


class TestTheStructureBehindTheRenderedViews:
    # [CLASS DOC] Tests of the structured change records behind the rendered text views.
    def test_a_scaling_change_says_why_it_is_allowed_and_which_object_it_touched(self):
        # [FUNCTION DOC] For a replica growth, asserts there is exactly one change record whose
        # allowed_by reason is scaling, whose path is the spec then replicas pair, and whose
        # identity key has the leader-worker-set kind and the engine pool name.
        """The observation side reads these fields, so a rendered line is not enough to answer it."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects[0]["spec"].update(replicas=6))
        )

        [change] = diff.changes
        assert (change.allowed_by, change.path) == ("scaling", ("spec", "replicas"))
        assert change.identity.key == ManifestObjectKey(kind="LeaderWorkerSet", name="myrun-miles-run-engine")

    def test_a_refused_change_carries_no_reason_to_allow_it(self):
        # [FUNCTION DOC] For an image change, asserts exactly one change record and that its
        # allowed_by reason is unset (none).
        """`allowed_by` is the whole answer to whether a change stops the launch."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(lambda objects: _worker_container(objects).update(image="miles:other")),
        )

        [change] = diff.changes
        assert change.allowed_by is None


class TestReplicasThatOnlyOneSideHas:
    # [CLASS DOC] Tests that a replica count present on only one side is not treated as scaling.
    def test_a_pool_that_lost_its_replica_count_is_refused(self):
        # [FUNCTION DOC] Removing the replicas field from the proposed pool: not allowed and no
        # allowed change lines are reported.
        """A template that stops rendering replicas is a chart change, not a run being scaled."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=_manifest_after(lambda objects: objects[0]["spec"].pop("replicas"))
        )

        assert not diff.is_allowed
        assert diff.allowed_changed == []

    def test_a_pool_that_gained_a_replica_count_is_refused(self):
        # [FUNCTION DOC] Mirror case: the installed pool lacks replicas and the proposed one has
        # them; not allowed and no allowed change lines.
        """The same holds the other way round, and reading it as scaling would apply a template change silently."""
        before = copy.deepcopy(_objects())
        before[0]["spec"].pop("replicas")

        diff = manifest_diff.diff_manifests(before=_manifest(before), after=_manifest(_objects()))

        assert not diff.is_allowed
        assert diff.allowed_changed == []


_ORCHESTRATOR_KEY = ManifestObjectKey(kind=STATEFUL_SET_KIND, name="myrun-miles-run-orchestrator")


class TestTheObjectsAHotRestartRebuilds:
    # [CLASS DOC] Tests of the allow-diff object keys option: objects whose key is listed (the
    # orchestrator stateful set key) may change in any field.
    def test_a_rebuilt_object_may_change_in_any_field(self):
        # [FUNCTION DOC] Changing the orchestrator image while its key is allowlisted: diff is
        # allowed, and the allowed change lines contain the field path of that image change.
        """Changing the orchestration script's arguments is the whole point of a hot restart."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(
                lambda objects: objects[2]["spec"]["template"]["spec"]["containers"][0].update(image="miles:other")
            ),
            allow_diff_object_keys=frozenset({_ORCHESTRATOR_KEY}),
        )

        assert diff.is_allowed
        assert diff.allowed_changed == [
            "apps/StatefulSet/rl/myrun-miles-run-orchestrator: spec.template.spec.containers.[0].image"
        ]

    def test_a_rebuilt_object_says_the_whitelist_is_what_allowed_it(self):
        # [FUNCTION DOC] Same change as before; the single change record has allowed_by equal to
        # the whitelist reason.
        """The observation side asks why a change was allowed, and only scaling means the object stayed up."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(
                lambda objects: objects[2]["spec"]["template"]["spec"]["containers"][0].update(image="miles:other")
            ),
            allow_diff_object_keys=frozenset({_ORCHESTRATOR_KEY}),
        )

        [change] = diff.changes
        assert change.allowed_by == "whitelist"

    def test_every_other_object_is_still_refused(self):
        # [FUNCTION DOC] With only the orchestrator allowlisted, changing the engine image is
        # still not allowed and appears as the sole disallowed change.
        """A hot restart that also changed the trainer has to stop the launch."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(lambda objects: _worker_container(objects).update(image="miles:other")),
            allow_diff_object_keys=frozenset({_ORCHESTRATOR_KEY}),
        )

        assert not diff.is_allowed
        assert diff.disallowed_changed == [
            "leaderworkerset.x-k8s.io/LeaderWorkerSet/rl/myrun-miles-run-engine: "
            "spec.leaderWorkerTemplate.workerTemplate.spec.containers.[0].image"
        ]

    def test_an_ordinary_relaunch_exempts_nothing(self):
        # [FUNCTION DOC] Without the allowlist option, the orchestrator image change is not
        # allowed (the default is strict).
        """The default is the strict gate every run that is not being hot restarted keeps."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(
                lambda objects: objects[2]["spec"]["template"]["spec"]["containers"][0].update(image="miles:other")
            ),
        )

        assert not diff.is_allowed


class TestWhetherADiffRebuildsOneObject:
    # [CLASS DOC] Tests of the rebuilds query, which says whether the diff replaces the object
    # with a given key (any change other than pure scaling, or presence only in the proposal).
    def test_an_untouched_object_is_not_rebuilt(self):
        # [FUNCTION DOC] Identical manifests: the orchestrator key is not reported as rebuilt.
        """An ordinary relaunch renders the orchestrator it already installed, and nothing may be read into that."""
        diff = manifest_diff.diff_manifests(before=_manifest(_objects()), after=_manifest(_objects()))

        assert not diff.rebuilds(key=_ORCHESTRATOR_KEY)

    def test_a_changed_object_is_rebuilt(self):
        # [FUNCTION DOC] Orchestrator image changed (allowlisted): the orchestrator key is
        # reported as rebuilt.
        """This is what tells the launcher the orchestrator pod really rolls, whatever the reason."""
        after = _manifest_after(
            lambda objects: objects[2]["spec"]["template"]["spec"]["containers"][0].update(image="miles:other")
        )
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()), after=after, allow_diff_object_keys=frozenset({_ORCHESTRATOR_KEY})
        )

        assert diff.rebuilds(key=_ORCHESTRATOR_KEY)

    def test_a_change_to_another_object_is_not_read_as_this_one_rebuilding(self):
        # [FUNCTION DOC] Only the engine image changed (no allowlist); the orchestrator key is not
        # reported as rebuilt.
        """Repointing the engines must not make the launcher throw the orchestrator's state file away."""
        after = _manifest_after(lambda objects: _worker_container(objects).update(image="miles:other"))
        diff = manifest_diff.diff_manifests(before=_manifest(_objects()), after=after)

        assert not diff.rebuilds(key=_ORCHESTRATOR_KEY)

    def test_scaling_alone_is_not_a_rebuild(self):
        # [FUNCTION DOC] Only the engine replicas grew; the engine pool key is not reported as
        # rebuilt.
        """A LeaderWorkerSet gaining cells keeps the pods it has, so nothing about it was rebuilt."""
        after = _manifest_after(lambda objects: objects[0]["spec"].update(replicas=6))
        diff = manifest_diff.diff_manifests(before=_manifest(_objects()), after=after)

        assert not diff.rebuilds(key=ManifestObjectKey(kind="LeaderWorkerSet", name="myrun-miles-run-engine"))

    def test_an_object_that_only_the_proposal_has_is_rebuilt(self):
        # [FUNCTION DOC] The installed release lacks the orchestrator while the proposal has it;
        # the orchestrator key is reported as rebuilt.
        """A release that gains its orchestration script needs the same treatment as a first install."""
        before = _manifest([_leader_worker_set(), _config_map()])
        diff = manifest_diff.diff_manifests(before=before, after=_manifest(_objects()))

        assert diff.rebuilds(key=_ORCHESTRATOR_KEY)

    def test_an_object_neither_manifest_has_is_not_rebuilt(self):
        # [FUNCTION DOC] Both manifests lack the orchestrator; its key is not reported as rebuilt.
        """A trainer release carries no orchestrator at all, and nothing about it ever changes."""
        objects = [_leader_worker_set(), _config_map()]
        diff = manifest_diff.diff_manifests(before=_manifest(objects), after=_manifest(objects))

        assert not diff.rebuilds(key=_ORCHESTRATOR_KEY)


def _foreign_leader_worker_set(replicas: int = 2, api_version: str = "example.com/v1") -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a mapping for a leader-worker-set kind object named for a
    # third party, with the given replicas (default 2) and api version (default a foreign group,
    # version one). Its spec holds only replicas.
    return {
        "apiVersion": api_version,
        "kind": "LeaderWorkerSet",
        "metadata": {"name": "someone-elses-leaderworkerset"},
        "spec": {"replicas": replicas},
    }


_FOREIGN_KEY = ManifestObjectKey(kind="LeaderWorkerSet", name="someone-elses-leaderworkerset")


class TestScalingIsLimitedToTheSupportedApi:
    # [CLASS DOC] Tests that the scaling exemption applies only to the supported api group of the
    # leader-worker-set kind (any served version), not to same-kind objects of other groups.
    def test_refuses_a_replica_change_on_a_leaderworkerset_of_another_api(self):
        # [FUNCTION DOC] Foreign-group object replicas 2 to 6: not allowed, with a plain disallowed
        # field line at the replicas path.
        """The exemption is written for the api this launcher drives, and nothing else answers for it."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set()]),
            after=_manifest([_foreign_leader_worker_set(replicas=6)]),
        )

        assert not diff.is_allowed
        assert diff.disallowed_changed == [
            "example.com/LeaderWorkerSet/rl/someone-elses-leaderworkerset: spec.replicas"
        ]

    def test_reports_no_scaling_for_a_leaderworkerset_of_another_api(self):
        # [FUNCTION DOC] Same change; the allowed change lines list is empty.
        """Reporting it as scaling is what let the change through the restart gate without a word."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set()]),
            after=_manifest([_foreign_leader_worker_set(replicas=6)]),
        )

        assert diff.allowed_changed == []

    def test_counts_a_foreign_replica_change_as_a_rebuild(self):
        # [FUNCTION DOC] Same change; the rebuilds query for the foreign object's key is true.
        """Only the supported api is known to add cells without restarting what is already running."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set()]),
            after=_manifest([_foreign_leader_worker_set(replicas=6)]),
        )

        assert diff.rebuilds(key=_FOREIGN_KEY)

    def test_scales_another_served_version_of_the_supported_api_group(self):
        # [FUNCTION DOC] Replicas 2 to 6 on an object in the supported group but an alpha version:
        # the diff is allowed, so the exemption matches by group, not by exact version.
        """A set sized some other way carries no spec.replicas, so the group is precise enough on its own."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set(api_version="leaderworkerset.x-k8s.io/v1alpha1")]),
            after=_manifest([_foreign_leader_worker_set(replicas=6, api_version="leaderworkerset.x-k8s.io/v1alpha1")]),
        )

        assert diff.is_allowed

    def test_still_scales_the_supported_api(self):
        # [FUNCTION DOC] Replicas 2 to 6 on an object of the supported group and version one: the
        # diff is allowed and the allowed change line reports the replica change with the group.
        """The whole point of the exemption is that this run's own engine pool may grow."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set(api_version="leaderworkerset.x-k8s.io/v1")]),
            after=_manifest([_foreign_leader_worker_set(replicas=6, api_version="leaderworkerset.x-k8s.io/v1")]),
        )

        assert diff.is_allowed
        assert diff.allowed_changed == [
            "leaderworkerset.x-k8s.io/LeaderWorkerSet/rl/someone-elses-leaderworkerset: replicas 2 -> 6"
        ]

    def test_a_whitelisted_foreign_object_is_still_allowed_to_change(self):
        # [FUNCTION DOC] Foreign-group replicas change with the object's key allowlisted: the diff
        # is allowed (the allowlist ignores api group).
        """A hot restart names the objects it may rebuild, and that escape hatch is api-agnostic."""
        diff = manifest_diff.diff_manifests(
            before=_manifest([_foreign_leader_worker_set()]),
            after=_manifest([_foreign_leader_worker_set(replicas=6)]),
            allow_diff_object_keys=frozenset({_FOREIGN_KEY}),
        )

        assert diff.is_allowed


def _stateful_set_in(namespace: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a deep copy of the orchestrator stateful set mapping with its
    # metadata namespace set to the given value.
    document = copy.deepcopy(_stateful_set())
    document["metadata"]["namespace"] = namespace
    return document


class TestAWhitelistKeyTwoObjectsShare:
    # [CLASS DOC] Tests of the assertion raised when an allowlisted key (kind and name only)
    # resolves to more than one object in either manifest.
    def test_an_ambiguous_key_in_the_installed_release_stops_the_launch(self):
        # [FUNCTION DOC] Installed manifest holds two orchestrator stateful sets in different
        # namespaces and the key is allowlisted: diffing raises an assertion error mentioning
        # objects keyed.
        """The key waves a change through, and two objects behind it wave through one nobody named."""
        ambiguous = _manifest([*_objects(), _stateful_set_in("other")])

        with pytest.raises(AssertionError, match="objects keyed"):
            manifest_diff.diff_manifests(
                before=ambiguous, after=ambiguous, allow_diff_object_keys=frozenset({_ORCHESTRATOR_KEY})
            )

    def test_an_ambiguous_key_in_the_proposed_release_stops_the_launch(self):
        # [FUNCTION DOC] Same, but only the proposed manifest holds the duplicate; also raises an
        # assertion error.
        """The proposal is where the second object first appears, and it is whitelisted before it is diffed."""
        with pytest.raises(AssertionError, match="objects keyed"):
            manifest_diff.diff_manifests(
                before=_manifest(_objects()),
                after=_manifest([*_objects(), _stateful_set_in("other")]),
                allow_diff_object_keys=frozenset({_ORCHESTRATOR_KEY}),
            )

    def test_an_ambiguity_no_whitelist_key_names_is_not_this_check_business(self):
        # [FUNCTION DOC] Duplicate keyed objects in both manifests but no allowlist option: no
        # error, and identical manifests diff as allowed.
        """Only the keys a hot restart exempts are resolved; every other object is compared by identity anyway."""
        both = _manifest([*_objects(), _stateful_set_in("other")])

        assert manifest_diff.diff_manifests(before=both, after=both).is_allowed

    def test_a_whitelisted_key_no_object_carries_is_accepted(self):
        # [FUNCTION DOC] Allowlisting a stateful set key that matches no object in either manifest
        # is accepted: identical manifests diff as allowed.
        """A trainer release carries no orchestrator, and exempting one it never installs changes nothing."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest(_objects()),
            allow_diff_object_keys=frozenset({ManifestObjectKey(kind=STATEFUL_SET_KIND, name="absent")}),
        )

        assert diff.is_allowed

    def test_an_unambiguous_whitelisted_key_still_exempts_its_object(self):
        # [FUNCTION DOC] A single orchestrator with a changed image and its key allowlisted: no
        # error and the diff is allowed.
        """The check runs before every diff, so it must not refuse the releases a hot restart is meant for."""
        diff = manifest_diff.diff_manifests(
            before=_manifest(_objects()),
            after=_manifest_after(
                lambda objects: objects[2]["spec"]["template"]["spec"]["containers"][0].update(image="miles:other")
            ),
            allow_diff_object_keys=frozenset({_ORCHESTRATOR_KEY}),
        )

        assert diff.is_allowed
