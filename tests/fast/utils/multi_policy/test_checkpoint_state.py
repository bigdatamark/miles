# [MODULE DOC]
# Tests for the multi-policy global checkpoint state record: a small model holding the leader policy
# id and a mapping from policy id to the rollout id each policy had reached, saved as a json file
# named after the leader's rollout id inside a dedicated state directory under the checkpoint root,
# and loaded back by leader rollout id. All tests use a temporary directory.
import pytest

from miles.utils.multi_policy.checkpoint_state import MULTI_POLICY_STATE_DIRNAME, MultiPolicyCheckpointState


class TestMultiPolicyCheckpointState:
    # [CLASS DOC] Groups the save/load round-trip, per-policy position, missing-file and corrupt-record
    # tests of the checkpoint state record.
    def test_the_global_state_is_indexed_by_the_leader_rollout_id(self, tmp_path):
        # [FUNCTION DOC] Saves a state with leader a and rollout ids a=7, b=5 into a temporary
        # directory. Asserts a file named after the leader's rollout id (7) exists in the state
        # directory, and that loading with leader rollout id 7 returns an equal state.
        """DataSource and RolloutExecutor are global, so one index must identify the whole run."""
        state = MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 7, "b": 5})

        state.save(tmp_path)

        assert (tmp_path / MULTI_POLICY_STATE_DIRNAME / "7.json").exists()
        assert MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=7) == state

    def test_the_policies_are_recorded_at_the_positions_they_really_reached(self, tmp_path):
        # [FUNCTION DOC] Saves a state whose policies sit at different rollout ids (a=7, b=5), loads it
        # back by the leader's id, and asserts the per-policy rollout id mapping is preserved
        # unchanged (not collapsed to the leader's id).
        """Fully async policies run at their own pace, so their rollout ids need not agree."""
        MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 7, "b": 5}).save(tmp_path)

        assert MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=7).rollout_ids == {"a": 7, "b": 5}

    def test_a_state_of_another_rollout_is_not_read_back(self, tmp_path):
        # [FUNCTION DOC] After saving a state for leader rollout 7, loading with leader rollout id 5
        # (a neighbouring, never saved id, which is only a follower's position) returns nothing.
        """Each global checkpoint has its own record, so a resume must not pick up a neighbour's."""
        MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 7, "b": 5}).save(tmp_path)

        assert MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=5) is None

    def test_a_missing_state_file_reads_as_none(self, tmp_path):
        # [FUNCTION DOC] Loading from an empty directory that never had a state saved returns nothing
        # rather than raising.
        """A run that never saved must start fresh instead of exploding on load."""
        assert MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=3) is None

    def test_a_record_that_disagrees_with_its_own_file_name_is_refused(self, tmp_path):
        # [FUNCTION DOC] Saves a valid record (which names rollout 7 for the leader), then writes the
        # same record's serialized json into a file named for rollout 9. Loading with leader rollout
        # id 9 must raise an assertion error whose message says it is the record of rollout 9, i.e.
        # the loader cross-checks the leader's recorded rollout id against the requested one.
        """The file name is how a resume finds the record, so a record naming another rollout is corrupt."""
        MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 7, "b": 5}).save(tmp_path)
        path = tmp_path / MULTI_POLICY_STATE_DIRNAME / "9.json"
        path.write_text(
            MultiPolicyCheckpointState(leader_model_id="a", rollout_ids={"a": 7, "b": 5}).model_dump_json()
        )

        with pytest.raises(AssertionError, match="record of rollout 9"):
            MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=9)
