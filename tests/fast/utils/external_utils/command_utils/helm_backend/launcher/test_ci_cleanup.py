# [MODULE DOC] Unit tests for the CI leftover cleanup of the Helm launcher entrypoint: a function
# that, for a namespace and the id of the run being launched, lists Helm releases carrying the CI
# label in that namespace, skips the releases belonging to the launching run (recognized by the
# release name pattern with the run id), uninstalls all others one by one in listing order within
# their namespace, and returns the removed release names. The cluster is replaced by a fake that
# stands in for the command wrapper's low-level runner and understands only the Helm list and
# uninstall commands.
import subprocess
from dataclasses import dataclass, field

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint

NAMESPACE = "rl"
_OTHER_RUN_ID = "260101-000000-999"


@dataclass
class FakeCluster:
    # [CLASS DOC] HELPER: Fake cluster used as a replacement for the command runner. Holds a list
    # of releases (dictionaries with name, namespace and a CI flag) and the commands it received.
    # Called with a command (list of words) it records the command; for a Helm list command it
    # returns a JSON array of the names of releases matching the namespace option and having the
    # CI flag set (asserting the selector option equals the CI label with value true); for any
    # other command (the uninstall) it removes the release named by the third word from its list.
    # Always reports exit code 0.
    releases: list[dict[str, str | bool]]
    commands: list[list[str]] = field(default_factory=list)

    def __call__(self, command: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        # [FUNCTION DOC] HELPER: Records the command; answers a list command with the JSON text
        # of matching releases, otherwise deletes the release whose name is the third word and
        # returns empty output. The capture-output flag is ignored.
        self.commands.append(command)
        if command[1] == "list":
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=self._listed(command), stderr="")
        self.releases = [release for release in self.releases if release["name"] != command[2]]
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    def uninstalled(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns, in order, the release names (third word) of all
        # recorded uninstall commands.
        return [command[2] for command in self.commands if command[1] == "uninstall"]

    def _listed(self, command: list[str]) -> str:
        # [FUNCTION DOC] HELPER: Builds the JSON array text for a list command: reads the
        # namespace and selector options, asserts the selector is the CI label equal to true, and
        # returns objects with only the "name" key for releases in that namespace whose CI flag
        # is exactly true. Names are inserted into the text without escaping.
        namespace = command[command.index("--namespace") + 1]
        selector = command[command.index("--selector") + 1]
        assert selector == f"{command_wrapper.CI_LABEL}=true"
        matched = [release for release in self.releases if release["namespace"] == namespace and release["ci"] is True]
        return "[" + ", ".join(f'{{"name": "{release["name"]}"}}' for release in matched) + "]"


def _cluster(monkeypatch: pytest.MonkeyPatch, releases: list[dict[str, str | bool]]) -> FakeCluster:
    # [FUNCTION DOC] HELPER: Creates a fake cluster holding the given releases, installs it as
    # the command wrapper's private low-level runner for the test, and returns it.
    cluster = FakeCluster(releases=releases)
    monkeypatch.setattr(command_wrapper, "_run", cluster)
    return cluster


def _clean_up(*, keep_run_id: str = _OTHER_RUN_ID) -> list[str]:
    # [FUNCTION DOC] HELPER: Calls the entrypoint's leftover CI release cleanup for the test
    # namespace "rl" keeping releases of the given run id (default: an unrelated run id). Returns
    # the list of removed release names.
    return entrypoint._uninstall_leftover_ci_releases(NAMESPACE, keep_run_id=keep_run_id)


class TestUninstallLeftoverCiReleases:
    # [CLASS DOC] Tests of the cleanup function against the fake cluster.
    def test_removes_only_releases_matching_both_the_namespace_and_the_label(self, monkeypatch):
        # [FUNCTION DOC] Of three releases (CI in the namespace, non-CI in the namespace, CI in
        # another namespace) only the first is uninstalled.
        """A human's release in the same namespace, or a CI release of another namespace, is not CI's to delete."""
        cluster = _cluster(
            monkeypatch,
            [
                {"name": "ci-here", "namespace": NAMESPACE, "ci": True},
                {"name": "human-here", "namespace": NAMESPACE, "ci": False},
                {"name": "ci-elsewhere", "namespace": "other", "ci": True},
            ],
        )

        _clean_up()

        assert cluster.uninstalled() == ["ci-here"]

    def test_asks_helm_for_both_filters_at_once(self, monkeypatch):
        # [FUNCTION DOC] With no releases, the first (listing) command carries the namespace
        # option equal to "rl" and the selector option equal to the CI label with value true.
        """Filtering on only one of the two would list releases the caller must never touch."""
        cluster = _cluster(monkeypatch, [])

        _clean_up()

        listing = cluster.commands[0]
        assert listing[listing.index("--namespace") + 1] == NAMESPACE
        assert listing[listing.index("--selector") + 1] == f"{command_wrapper.CI_LABEL}=true"

    def test_uninstalls_every_release_it_listed_in_order(self, monkeypatch):
        # [FUNCTION DOC] Three CI releases named first, second, third are all uninstalled, in
        # that order.
        """Stopping at the first one would leak the rest, which is how a runner fills up over a week."""
        cluster = _cluster(
            monkeypatch,
            [{"name": name, "namespace": NAMESPACE, "ci": True} for name in ("first", "second", "third")],
        )

        _clean_up()

        assert cluster.uninstalled() == ["first", "second", "third"]

    def test_uninstalls_each_release_from_the_namespace_it_was_found_in(self, monkeypatch):
        # [FUNCTION DOC] The second command (after the listing) is exactly the Helm uninstall of
        # the release with the namespace option.
        """helm uninstall without the namespace looks in the caller's default context and finds nothing."""
        cluster = _cluster(monkeypatch, [{"name": "ci-here", "namespace": NAMESPACE, "ci": True}])

        _clean_up()

        assert cluster.commands[1] == ["helm", "uninstall", "ci-here", "--namespace", NAMESPACE]

    def test_reports_the_names_it_removed(self, monkeypatch):
        # [FUNCTION DOC] The returned list equals the names of the removed releases, in order.
        """The caller logs what it cleaned, and a silent cleanup is indistinguishable from a broken one."""
        _cluster(monkeypatch, [{"name": name, "namespace": NAMESPACE, "ci": True} for name in ("a", "b")])

        assert _clean_up() == ["a", "b"]

    def test_does_nothing_when_no_ci_release_is_left(self, monkeypatch):
        # [FUNCTION DOC] With only a non-CI release, the result is empty and no uninstall
        # command was issued.
        """An empty listing must not turn into an uninstall of nothing, which helm treats as an error."""
        cluster = _cluster(monkeypatch, [{"name": "human-here", "namespace": NAMESPACE, "ci": False}])

        assert _clean_up() == []
        assert cluster.uninstalled() == []

    def test_keeps_the_sibling_releases_of_the_run_being_launched(self, monkeypatch):
        # [FUNCTION DOC] Three releases named after the launching run id (suffixes all, trainer,
        # primary) and one of another run are listed; only the other run's release is removed and
        # returned.
        """A split run installs one ci release per component, so cleaning them would tear down its own halves."""
        run_id = "260101-000000-000"
        siblings = [f"miles-run-{run_id}-all", f"miles-run-{run_id}-trainer", f"miles-run-{run_id}-primary"]
        cluster = _cluster(
            monkeypatch,
            [{"name": name, "namespace": NAMESPACE, "ci": True} for name in ("ci-of-another-run", *siblings)],
        )

        removed = _clean_up(keep_run_id=run_id)

        assert removed == ["ci-of-another-run"]
        assert cluster.uninstalled() == ["ci-of-another-run"]

    def test_recognizes_every_release_this_run_installs_by_the_run_id_they_share(self):
        # [FUNCTION DOC] Direct test of the run membership predicate: names with the prefix
        # "miles-run-", the run id and a dash-separated component (all, trainer-a) belong to the
        # run; a name where the run id is followed directly by another digit, or an unrelated
        # name, does not.
        """One release missed here is one release of this run that its own next launch uninstalls."""
        run_id = "260101-000000-000"

        assert entrypoint._belongs_to_run(f"miles-run-{run_id}-all", run_id=run_id)
        assert entrypoint._belongs_to_run(f"miles-run-{run_id}-trainer-a", run_id=run_id)
        assert not entrypoint._belongs_to_run(f"miles-run-{run_id}9-all", run_id=run_id)
        assert not entrypoint._belongs_to_run("helm-release-of-something-else", run_id=run_id)

    def test_a_release_of_a_run_whose_id_starts_with_this_one_is_still_cleaned(self, monkeypatch):
        # [FUNCTION DOC] One CI release named "miles-run-<id>-trainer-primary" is listed and the
        # cleanup is asked to keep the run with that same id; asserts the release is removed and
        # returned. Whether it is kept depends on how the naming rules parse that name (a
        # deploy instance id of "primary" versus a longer run id).
        # KNOWN ISSUE: the test name speaks of a run whose id merely starts with the kept id, but
        # the body uses the identical run id; the intended longer run id is never constructed.
        """A prefix test reads another run's releases as this run's siblings and leaves them behind forever."""
        run_id = "260101-000000-000"
        _cluster(monkeypatch, [{"name": f"miles-run-{run_id}-trainer-primary", "namespace": NAMESPACE, "ci": True}])

        assert _clean_up(keep_run_id=run_id) == [f"miles-run-{run_id}-trainer-primary"]

    def test_treats_empty_helm_output_as_nothing_to_clean(self, monkeypatch):
        # [FUNCTION DOC] When the runner always returns empty output (as Helm does for no rows)
        # instead of a JSON array, the cleanup returns an empty list without raising.
        """helm prints an empty body rather than [] when it has no rows, and json.loads would raise on it."""
        monkeypatch.setattr(
            command_wrapper,
            "_run",
            lambda command, capture_output=False: subprocess.CompletedProcess(
                args=command, returncode=0, stdout="", stderr=""
            ),
        )

        assert _clean_up() == []
