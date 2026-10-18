# [MODULE DOC]
# CPU tests (skipped when SGLang is not installed) of the helper in the disk-delta protocol that
# announces a new weight version to rollout engines which have not yet reported a known version
# ("cold" engines). Engines are fake objects reporting a fixed version and logging update calls.
import pytest


def _helpers():
    # [FUNCTION DOC] HELPER: Skips the test if SGLang cannot be imported; otherwise imports and
    # returns a pair: the sentinel value meaning "weight version unset" and the function under test
    # (update weight version if unset). Deferred import keeps collection working without SGLang.
    pytest.importorskip("sglang")
    from miles.backends.training_utils.weight_update.protocols.delta import (
        _UNSET_WEIGHT_VERSION,
        _update_weight_version_if_unset,
    )

    return _UNSET_WEIGHT_VERSION, _update_weight_version_if_unset


class TestUpdateWeightVersionIfUnset:
    # [CLASS DOC] Tests of the cold-engine version announcement. Each call log entry is a triple of
    # engine name, announced version and the abort-all-requests flag.
    def test_publishes_the_version_to_every_cold_engine(self):
        # [FUNCTION DOC] Two engines reporting the unset sentinel both receive the announcement of
        # the given version string, in order, with the abort flag false.
        """Paths whose transport cannot carry the version announce it to the engines itself."""
        unset_version, update_weight_version_if_unset = _helpers()
        calls = []

        update_weight_version_if_unset(
            [_Engine(unset_version, calls, name="a"), _Engine(unset_version, calls, name="b")],
            "ab12cd34-00000007",
        )

        assert calls == [("a", "ab12cd34-00000007", False), ("b", "ab12cd34-00000007", False)]

    def test_announcing_does_not_abort_running_requests(self):
        # [FUNCTION DOC] The single announcement is made with the abort-all-requests flag false.
        """Announcing a version changes no weights, so it must not cancel in-flight generation."""
        unset_version, update_weight_version_if_unset = _helpers()
        calls = []

        update_weight_version_if_unset([_Engine(unset_version, calls)], "ab12cd34-00000007")

        assert [abort for _name, _version, abort in calls] == [False]

    def test_an_engine_that_never_reported_a_version_is_cold(self):
        # [FUNCTION DOC] An engine reporting no version at all (none) is treated like an unset one and
        # receives the announcement.
        """A missing version is as unset as the launch default."""
        _unset, update_weight_version_if_unset = _helpers()
        calls = []

        update_weight_version_if_unset([_Engine(None, calls)], "ab12cd34-00000007")

        assert [version for _name, version, _abort in calls] == ["ab12cd34-00000007"]

    def test_leaves_an_engine_that_already_serves_a_known_version_alone(self):
        # [FUNCTION DOC] An engine already reporting a concrete version receives no announcement.
        """After a trainer failover the engines still hold their weights; claiming a version would lie."""
        _unset, update_weight_version_if_unset = _helpers()
        calls = []

        update_weight_version_if_unset([_Engine("ab12cd34-00000004", calls)], "ab12cd34-00000007")

        assert calls == []

    def test_announces_only_to_the_engines_that_are_still_cold(self):
        # [FUNCTION DOC] Given a warm engine, a cold (sentinel) engine and an engine that never
        # reported, only the latter two are announced to, in list order.
        """Partial recovery leaves survivors holding real weights; only the replacements need telling."""
        unset_version, update_weight_version_if_unset = _helpers()
        calls = []
        engines = [
            _Engine("ab12cd34-00000004", calls, name="warm"),
            _Engine(unset_version, calls, name="cold"),
            _Engine(None, calls, name="never_told"),
        ]

        update_weight_version_if_unset(engines, "ab12cd34-00000007")

        assert [name for name, _version, _abort in calls] == ["cold", "never_told"]

    def test_no_engines_is_a_noop(self):
        # [FUNCTION DOC] An empty engine list completes without error and without effects.
        """An updater with no engines attached publishes nothing rather than failing."""
        _unset, update_weight_version_if_unset = _helpers()

        update_weight_version_if_unset([], "ab12cd34-00000007")


class _Engine:
    # [CLASS DOC] HELPER: Fake rollout engine with a fixed reported weight version that logs
    # version announcements into a shared list.
    def __init__(self, reported_version, calls, *, name=""):
        # [FUNCTION DOC] HELPER: Stores the version to report, the shared call log and an optional
        # name (default empty) used to identify the engine in log entries.
        self._reported_version = reported_version
        self._calls = calls
        self._name = name

    async def get_weight_version(self):
        # [FUNCTION DOC] HELPER: Async; returns the configured reported version (may be none).
        return self._reported_version

    async def update_weight_version(self, weight_version, abort_all_requests=True):
        # [FUNCTION DOC] HELPER: Async; appends a triple of engine name, announced version and the
        # abort flag (default true) to the shared log. Tests rely on the caller passing false
        # explicitly, since the default here is true.
        self._calls.append((self._name, weight_version, abort_all_requests))
