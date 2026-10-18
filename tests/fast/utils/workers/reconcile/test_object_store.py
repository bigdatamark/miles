# [MODULE DOC]
# Unit tests for the keyed object store used by the reconcile loop. The store applies source
# events (upsert, delete, replace) and returns the set of parent keys affected by each, maps
# objects to parents through a key-mapping function (here the pod's cell label), and answers
# membership queries. Covers incremental events, whole-world replaces, queries and rejection of
# unknown event types.
from __future__ import annotations

import pytest
from tests.fast.utils.workers.reconcile.utils import make_pod, pod_cell

from miles.utils.workers.reconcile.object_store import ObjectStore
from miles.utils.workers.reconcile.source_event import DeleteEvent, ReplaceEvent, UpsertEvent


def make_store() -> ObjectStore:
    # [FUNCTION DOC] HELPER: Creates an empty store whose parent mapping is the pod's cell label.
    return ObjectStore(key_map=pod_cell)


class TestIncrementalEvents:
    # [CLASS DOC] Single upsert and delete events and the parents they affect.
    def test_an_upsert_stores_the_object_and_reports_its_parent(self):
        # [FUNCTION DOC] An upsert of a pod in cell-a returns {cell-a}, makes the key a member,
        # and lists the pod under cell-a.
        """A plain upsert lands in the store and wakes exactly its cell."""
        store = make_store()
        affected = store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))

        assert affected == {"cell-a"}
        assert "pod-0" in store
        assert [pod.metadata.name for pod in store.get_by_parent("cell-a")] == ["pod-0"]

    def test_a_reparenting_upsert_reports_both_parents(self):
        # [FUNCTION DOC] Moving a pod from cell-a to cell-b returns both cells and empties
        # cell-a.
        """Moving an object between cells affects the old and the new one."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        affected = store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-b")))

        assert affected == {"cell-a", "cell-b"}
        assert store.get_by_parent("cell-a") == []

    def test_a_delete_reports_the_stored_parent(self):
        # [FUNCTION DOC] Deleting a stored pod with no tombstone returns its stored cell and
        # removes the key.
        """Deleting a known object affects the cell it belonged to."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        affected = store.handle_event(DeleteEvent(key="pod-0", last_obj=None))

        assert affected == {"cell-a"}
        assert "pod-0" not in store

    def test_a_delete_of_an_unknown_object_uses_the_tombstone(self):
        # [FUNCTION DOC] Deleting an unknown key returns the cell of the tombstone object.
        """An unknown delete is attributed through last_obj."""
        store = make_store()
        affected = store.handle_event(DeleteEvent(key="pod-0", last_obj=make_pod("pod-0", cell="cell-a")))

        assert affected == {"cell-a"}

    def test_a_delete_whose_tombstone_disagrees_uses_the_stored_parent(self):
        # [FUNCTION DOC] The tombstone names cell-b but the stored pod is in cell-a; the result
        # is {cell-a} and the key is removed.
        """A relabel we never saw must not crash the delete nor wake the wrong cell."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        affected = store.handle_event(DeleteEvent(key="pod-0", last_obj=make_pod("pod-0", cell="cell-b")))

        assert affected == {"cell-a"}
        assert "pod-0" not in store

    def test_an_unmappable_upsert_is_dropped_and_removes_any_stored_object(self):
        # [FUNCTION DOC] Upserting a stored pod with its label removed returns the old cell and
        # removes the key.
        """A key_map failure turns the upsert into a departure, not a stale entry."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        affected = store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell=None)))

        assert affected == {"cell-a"}
        assert "pod-0" not in store

    def test_an_unmappable_upsert_of_an_unknown_object_affects_nobody(self):
        # [FUNCTION DOC] Upserting a new pod without a label returns an empty set and stores
        # nothing.
        """Nothing was stored under a parent, so there is no departure to report either."""
        store = make_store()

        affected = store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell=None)))

        assert affected == set()
        assert "pod-0" not in store


class TestReplace:
    # [CLASS DOC] Whole-world replace events.
    def test_replace_swaps_the_whole_store_and_reports_both_sides(self):
        # [FUNCTION DOC] Replacing a store holding a cell-a pod with a single cell-b pod returns
        # both cells, drops the old pod and lists the new one under cell-b.
        """A replace applies atomically and names the parents it added to and removed from."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-old", obj=make_pod("pod-old", cell="cell-a")))

        pod_new = make_pod("pod-new", cell="cell-b")
        affected = store.handle_event(ReplaceEvent(objects={"pod-new": pod_new}))

        assert affected == {"cell-a", "cell-b"}
        assert "pod-old" not in store
        assert [pod.metadata.name for pod in store.get_by_parent("cell-b")] == ["pod-new"]

    def test_replace_synthesizes_deletions_for_objects_that_vanished(self):
        # [FUNCTION DOC] Two pods in cell-a, then a replace holding one of them; the result is
        # {cell-a}, the missing pod is gone and only the survivor remains.
        """Objects missing from a relist must be removed, or ghost members persist forever."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        store.handle_event(UpsertEvent(key="pod-1", obj=make_pod("pod-1", cell="cell-a")))

        pod_0 = make_pod("pod-0", cell="cell-a")
        affected = store.handle_event(ReplaceEvent(objects={"pod-0": pod_0}))

        assert affected == {"cell-a"}
        assert "pod-1" not in store
        assert [pod.metadata.name for pod in store.get_by_parent("cell-a")] == ["pod-0"]

    def test_an_empty_replace_clears_the_store(self):
        # [FUNCTION DOC] An empty replace returns the previously occupied cell and leaves no
        # parent keys.
        """A relist that returns nothing means the pool is gone, not that nothing changed."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))

        affected = store.handle_event(ReplaceEvent(objects={}))

        assert affected == {"cell-a"}
        assert store.parent_keys() == set()

    def test_a_relist_refreshes_a_survivor_whose_parent_did_not_change(self):
        # [FUNCTION DOC] Replacing a pod with a newer version in the same cell returns that cell
        # and the stored copy has the new version.
        """A survivor still in the listing must be replaced by the newly listed copy, not left stale."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-0", obj=make_pod("pod-0", resource_version="1")))

        affected = store.handle_event(ReplaceEvent(objects={"pod-0": make_pod("pod-0", resource_version="2")}))

        assert affected == {"cell-a"}
        assert [pod.metadata.resource_version for pod in store.get_by_parent("cell-a")] == ["2"]

    def test_an_unmappable_object_in_a_replace_is_dropped(self):
        # [FUNCTION DOC] A replace with one unlabeled pod and one labeled pod returns {cell-a},
        # drops the unlabeled one and keeps the other.
        """One bad object must not stall the rest of the relist."""
        store = make_store()

        affected = store.handle_event(
            ReplaceEvent(objects={"pod-0": make_pod("pod-0", cell=None), "pod-1": make_pod("pod-1", cell="cell-a")})
        )

        assert affected == {"cell-a"}
        assert "pod-0" not in store
        assert "pod-1" in store


class TestQueries:
    # [CLASS DOC] Read queries on the store.
    def test_get_by_parent_returns_members_sorted_by_key(self):
        # [FUNCTION DOC] Pods inserted as b then a are listed as a then b.
        """Membership listing is deterministic."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-b", obj=make_pod("pod-b", cell="cell-a")))
        store.handle_event(UpsertEvent(key="pod-a", obj=make_pod("pod-a", cell="cell-a")))

        assert [pod.metadata.name for pod in store.get_by_parent("cell-a")] == ["pod-a", "pod-b"]

    def test_parent_keys_lists_every_cell_that_still_has_members(self):
        # [FUNCTION DOC] After inserting pods in two cells and deleting the only one of cell-b,
        # the parent keys are {cell-a}.
        """parent_keys is what a resync re-drives."""
        store = make_store()
        store.handle_event(UpsertEvent(key="pod-a", obj=make_pod("pod-a", cell="cell-a")))
        store.handle_event(UpsertEvent(key="pod-b", obj=make_pod("pod-b", cell="cell-b")))
        store.handle_event(DeleteEvent(key="pod-b", last_obj=None))

        assert store.parent_keys() == {"cell-a"}


class TestUnknownEvent:
    # [CLASS DOC] Unclassifiable events are an error.
    def test_an_event_outside_the_union_fails_fast(self):
        # [FUNCTION DOC] Handling a plain object that is not a known event type fails an
        # assertion.
        """A source event the store cannot classify must raise rather than be silently ignored."""
        store = make_store()

        with pytest.raises(AssertionError):
            store.handle_event(object())
