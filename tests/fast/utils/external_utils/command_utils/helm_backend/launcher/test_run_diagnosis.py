# [MODULE DOC] Tests for the launcher entrypoint's run diagnosis helpers: finding all helm
# releases that belong to a run (the primary, trainer and inference releases of a split run) by
# listing the namespace with a name prefix filter, falling back to the followed release when helm
# fails, building the kubectl label selector that covers several releases, and treating release
# names that the naming rules cannot parse as belonging to no run. The helm process runner is
# replaced by a fake that returns a json list of release names and records the commands.
import json
import re
import subprocess

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL

RUN_ID = "260101-000000-000"
OTHER_RUN_ID = "260101-111111-111"


def _release(component: DeployComponent, instance_id: str | None = None, run_id: str = RUN_ID) -> str:
    # [FUNCTION DOC] HELPER: Builds and serializes a release name from the run id (default the
    # first constant), deploy component and optional instance id. Returns the text.
    return ReleaseName(run_id=run_id, deploy_component=component, deploy_instance_id=instance_id).serialize()


PRIMARY = _release(DeployComponent.PRIMARY)
TRAINER = _release(DeployComponent.TRAINER)
ENGINE = _release(DeployComponent.INFERENCE, "e0")
STRANGER = _release(DeployComponent.ALL, run_id=OTHER_RUN_ID)


def _helm_listing(monkeypatch: pytest.MonkeyPatch, releases: list[str]) -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Replaces the command wrapper's low level process runner with a fake
    # that records every command and answers with a successful process whose output is a json list
    # of objects carrying each given release name. Returns the (live) list of recorded commands.
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool) -> subprocess.CompletedProcess:
        # [FUNCTION DOC] HELPER: Records the command and returns exit code zero with the json
        # release list as output and empty error output.
        commands.append(command)
        body = json.dumps([{"name": release} for release in releases])
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=body, stderr="")

    monkeypatch.setattr(command_wrapper, "_run", fake_run)
    return commands


class TestTheReleasesADiagnosisCovers:
    # [CLASS DOC] Tests of the entrypoint's releases-of-run lookup.
    def test_covers_every_deployment_the_run_installed(self, monkeypatch):
        # [FUNCTION DOC] With helm listing three releases of the run plus one of another run, the
        # lookup returns exactly the three, sorted.
        """A split run fails in whichever deployment broke, and that is rarely the one being followed."""
        _helm_listing(monkeypatch, [PRIMARY, TRAINER, ENGINE, STRANGER])

        found = entrypoint._releases_of_run(release=PRIMARY, namespace="rl")

        assert found == sorted([PRIMARY, TRAINER, ENGINE])

    def test_leaves_another_run_in_the_namespace_alone(self, monkeypatch):
        # [FUNCTION DOC] A release of another run listed by helm is excluded from the result.
        """Namespaces hold more than one run, and describing someone else's pods reads their experiment."""
        _helm_listing(monkeypatch, [PRIMARY, STRANGER])

        assert STRANGER not in entrypoint._releases_of_run(release=PRIMARY, namespace="rl")

    def test_asks_helm_for_the_whole_namespace_because_a_run_carries_no_label(self, monkeypatch):
        # [FUNCTION DOC] The helm command has no selector option and passes the given namespace
        # as the value after the namespace option.
        """Releases are labelled per release, so the siblings can only be found by listing and parsing."""
        commands = _helm_listing(monkeypatch, [PRIMARY])

        entrypoint._releases_of_run(release=PRIMARY, namespace="rl")

        assert "--selector" not in commands[0]
        assert commands[0][commands[0].index("--namespace") + 1] == "rl"

    def test_falls_back_to_the_followed_release_when_helm_cannot_be_asked(self, monkeypatch):
        # [FUNCTION DOC] When the process runner raises a runtime error, the lookup swallows it
        # and returns a list holding only the followed release.
        """A diagnosis of one deployment beats no diagnosis at all when the run has already failed."""

        def fake_run(command: list[str], capture_output: bool) -> subprocess.CompletedProcess:
            # [FUNCTION DOC] HELPER: Always raises a runtime error saying helm is unreachable.
            raise RuntimeError("helm is not reachable")

        monkeypatch.setattr(command_wrapper, "_run", fake_run)

        assert entrypoint._releases_of_run(release=PRIMARY, namespace="rl") == [PRIMARY]

    def test_keeps_the_followed_release_even_when_helm_forgot_it(self, monkeypatch):
        # [FUNCTION DOC] An empty helm listing still yields the followed release alone.
        """The release being diagnosed is the one known to have failed, listed or not."""
        _helm_listing(monkeypatch, [])

        assert entrypoint._releases_of_run(release=PRIMARY, namespace="rl") == [PRIMARY]

    def test_asks_helm_only_for_the_releases_of_this_run(self, monkeypatch):
        # [FUNCTION DOC] The value after the filter option is a caret anchored, regular expression
        # escaped form of the run's release name prefix.
        """helm truncates a busy namespace to 256 releases, which can drop the sibling that actually failed."""
        commands = _helm_listing(monkeypatch, [PRIMARY])

        entrypoint._releases_of_run(release=PRIMARY, namespace="rl")

        assert commands[0][commands[0].index("--filter") + 1] == f"^{re.escape(ReleaseName.run_prefix(run_id=RUN_ID))}"

    def test_the_filter_is_a_regex_the_run_id_cannot_break_out_of(self):
        # [FUNCTION DOC] The escaped prefix pattern followed by any suffix fully matches the
        # primary release name.
        """helm reads it as a regular expression, and a run id is not written to be one."""
        assert re.fullmatch(f"^{re.escape(ReleaseName.run_prefix(run_id=RUN_ID))}.*", PRIMARY)


class TestTheSelectorItDescribesWith:
    # [CLASS DOC] Tests of the kubectl label selector builder for several releases.
    def test_matches_every_release_of_the_run_at_once(self):
        # [FUNCTION DOC] For two releases the selector is the instance label, the word in, and a
        # parenthesized comma separated list of the release names without spaces.
        """One kubectl call per describe keeps the failure path short while covering all deployments."""
        selector = Kubectl.releases_selector([PRIMARY, TRAINER])

        assert selector == f"{INSTANCE_LABEL} in ({PRIMARY},{TRAINER})"

    def test_stays_an_equality_for_an_unsplit_run(self):
        # [FUNCTION DOC] For a single release the multi release selector equals the single
        # release selector.
        """An all-in-one run has one release, and the plain selector is what its snapshots already carry."""
        assert Kubectl.releases_selector([PRIMARY]) == Kubectl.release_selector(PRIMARY)


# Legal for helm, but not for miles: the run id is one character longer than the naming rules allow.
UNPARSABLE_NEIGHBOUR = "miles-run-" + "a" * 34 + "-all"


class TestAReleaseTheMilesRulesCannotRead:
    # [CLASS DOC] Tests that release names the naming rules reject (here a run id one character
    # too long) or that use another scheme belong to no run and do not break the lookup.
    def test_it_belongs_to_no_run(self):
        # [FUNCTION DOC] The belongs-to-run check returns false for the unparsable name.
        """ReleaseName.parse builds a validated model, so such a name raised out of the candidate loop."""
        assert entrypoint._belongs_to_run(UNPARSABLE_NEIGHBOUR, run_id=RUN_ID) is False

    def test_a_release_of_another_naming_scheme_belongs_to_no_run(self):
        # [FUNCTION DOC] The check returns false for a foreign chart's release name.
        """A namespace holds releases of other charts, and none of them is part of this run."""
        assert entrypoint._belongs_to_run("someone-elses-release", run_id=RUN_ID) is False

    def test_a_release_of_this_run_still_belongs_to_it(self):
        # [FUNCTION DOC] The check returns true for the run's primary release.
        """Reading an unreadable name as belonging to nothing may not cost the readable ones."""
        assert entrypoint._belongs_to_run(PRIMARY, run_id=RUN_ID) is True

    def test_such_a_neighbour_does_not_replace_the_diagnosis_of_the_failed_run(self, monkeypatch):
        # [FUNCTION DOC] With an unparsable neighbour in the helm listing, the lookup does not
        # raise and returns the run's two releases, sorted.
        """That loop runs outside both try blocks, so one neighbour replaced the diagnosis with a traceback."""
        _helm_listing(monkeypatch, [PRIMARY, TRAINER, UNPARSABLE_NEIGHBOUR])

        assert entrypoint._releases_of_run(release=PRIMARY, namespace="rl") == sorted([PRIMARY, TRAINER])

    def test_such_a_neighbour_is_not_diagnosed_as_part_of_the_run(self, monkeypatch):
        # [FUNCTION DOC] The unparsable neighbour is absent from the lookup's result.
        """Describing someone else's pods reads an experiment this run has nothing to do with."""
        _helm_listing(monkeypatch, [PRIMARY, UNPARSABLE_NEIGHBOUR])

        assert UNPARSABLE_NEIGHBOUR not in entrypoint._releases_of_run(release=PRIMARY, namespace="rl")
