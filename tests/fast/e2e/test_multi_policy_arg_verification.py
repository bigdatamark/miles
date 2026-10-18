# [MODULE DOC]
# Tests for the multi-policy end-to-end check that every trainer rank was built with its own
# policy's argument overrides (function assert_ranks_trained_with_policy_args in the multi-policy
# end-to-end support module, imported as e2e here). The check reads environment-report events from
# an events directory (the reader is replaced to return synthetic events), where each trainer rank
# reports its parsed argument values, its policy namespace id and its process identity. A constant
# two-policy megatron config (solver with 24 layers, verifier with 28) drives synthetic reports,
# two ranks each reporting twice. Tests cover the passing path and the failure messages for wrong
# overrides, missing policies, missing ranks, repeated reports masking a missing rank, identity
# mismatches, a shrunken verified-argument count, and ignoring non-trainer events.
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests.e2e import conftest_multi_policy as e2e

from miles.utils.audit_utils.event_logger.models import (
    EnvReport,
    EnvReportArgsDump,
    EnvReportEvent,
    EnvReportProcessFacts,
    MetricEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainProcessIdentity

MEGATRON_CONFIG = dict(
    trainers=[
        dict(model_id="solver", overrides=dict(hf_checkpoint="/models/solver", num_layers=24)),
        dict(model_id="verifier", overrides=dict(hf_checkpoint="/models/verifier", num_layers=28)),
    ]
)


def _make_report(*, model_id: str | None, rank: int = 0, values: dict[str, Any]) -> EnvReportEvent:
    # [FUNCTION DOC] HELPER: Builds an environment report event dated mid August 2026 from a
    # trainer actor process with the given policy id, cell index zero and the given rank within the
    # cell. The report carries a process section (host node-0, empty argv and environment) whose
    # argument dump holds the given values and no skipped names, and empty version, package, repo
    # and pip lists with packages not probed.
    return EnvReportEvent(
        timestamp=datetime(2026, 8, 14, tzinfo=UTC),
        source=TrainProcessIdentity(component="actor", model_id=model_id, cell_index=0, rank_within_cell=rank),
        report=EnvReport(
            process=EnvReportProcessFacts(
                hostname="node-0",
                argv=[],
                args=EnvReportArgsDump(values=values, skipped_names=[]),
                env_vars={},
                launcher_env_report=None,
            ),
            key_versions={},
            editable_packages=[],
            git_repos=[],
            full_pip_list=[],
            packages_probed=False,
        ),
    )


def _reports_of(model_id: str, **overrides: Any) -> list[EnvReportEvent]:
    # [FUNCTION DOC] HELPER: Finds the single trainer of the config with the given model id (the
    # unpacking raises a value error if none or several). Builds argument values from that trainer's
    # overrides, adds the trainer model id, then applies the keyword overrides last. Returns four
    # reports, two for each of ranks zero and one, in rank-major order.
    [trainer] = [entry for entry in MEGATRON_CONFIG["trainers"] if entry["model_id"] == model_id]
    values = {**trainer["overrides"], "trainer_model_id": model_id, **overrides}
    return [_make_report(model_id=model_id, rank=rank, values=values) for rank in (0, 1) for _ in range(2)]


@pytest.fixture(autouse=True)
def _pin_verified_argument_count(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture setting the module's expected number of verified
    # arguments to two for both solver and verifier, matching the two overrides in the config.
    monkeypatch.setattr(e2e, "NUM_VERIFIED_ARGS_PER_POLICY", {"solver": 2, "verifier": 2})


_EXPECTED_NUM_RANKS: int = 2


def _assert_with(monkeypatch, events: list) -> None:
    # [FUNCTION DOC] HELPER: Replaces the events reader so it returns the given events regardless
    # of directory, then runs the verification on a fake events directory with the config and two
    # expected ranks per policy. Raises whatever the verification raises.
    monkeypatch.setattr(e2e, "read_events", lambda events_dir: events)
    e2e.assert_ranks_trained_with_policy_args(
        Path("/events"), megatron_config=MEGATRON_CONFIG, expected_num_ranks=_EXPECTED_NUM_RANKS
    )


class TestAssertEveryRankTrainedWithItsOwnPolicyArgs:
    # [CLASS DOC] Tests of the verification's pass and failure cases.
    def test_a_run_where_every_rank_carries_its_own_overrides_passes(self, monkeypatch):
        # [FUNCTION DOC] Valid reports for both policies; asserts no error.
        """The happy path has to stay reachable, or every refusal below proves nothing."""
        _assert_with(monkeypatch, _reports_of("solver") + _reports_of("verifier"))

    def test_a_rank_built_with_another_policys_arguments_is_caught(self, monkeypatch):
        # [FUNCTION DOC] Verifier reports carry the solver's layer count; asserts an assertion
        # error saying the policy was built with other values.
        """This is the whole point: a policy handed another one's shape trains a second copy of that model."""
        events = _reports_of("solver") + _reports_of("verifier", num_layers=24)

        with pytest.raises(AssertionError, match="was built with"):
            _assert_with(monkeypatch, events)

    def test_a_policy_that_reported_nothing_is_caught(self, monkeypatch):
        # [FUNCTION DOC] Only solver reports exist; asserts an assertion error saying a policy
        # never actually trained.
        """A policy whose ranks never started would otherwise leave a run that trains half of what it claims."""
        with pytest.raises(AssertionError, match="never actually trained"):
            _assert_with(monkeypatch, _reports_of("solver"))

    def test_a_policy_missing_one_of_its_ranks_is_caught(self, monkeypatch):
        # [FUNCTION DOC] Verifier reports from rank zero only (first two events); asserts an
        # assertion error saying reports came from too few ranks.
        """A surviving rank must not stand in for the whole policy: the missing one is the unverified one."""
        events = _reports_of("solver") + _reports_of("verifier")[:2]

        with pytest.raises(AssertionError, match="reported from ranks"):
            _assert_with(monkeypatch, events)

    def test_a_policy_whose_repeated_reports_stand_in_for_a_missing_rank_is_caught(self, monkeypatch):
        # [FUNCTION DOC] Rank-zero verifier reports duplicated so the total count matches a full
        # policy; asserts the same too-few-ranks assertion error (distinct ranks are what count).
        """A rank reporting as often as the whole policy should must not be counted as the ranks that are missing."""
        rank_zero = _reports_of("verifier")[:2]
        events = _reports_of("solver") + rank_zero + rank_zero

        with pytest.raises(AssertionError, match="reported from ranks"):
            _assert_with(monkeypatch, events)

    def test_every_policy_missing_the_same_rank_is_caught(self, monkeypatch):
        # [FUNCTION DOC] Both policies report only rank zero; asserts an assertion error whose
        # message says the run trains each policy on the expected rank count.
        """Ranks compared only against each other agree on a run that started half the trainer it declared."""
        events = _reports_of("solver")[:2] + _reports_of("verifier")[:2]

        with pytest.raises(AssertionError, match="trains each policy on"):
            _assert_with(monkeypatch, events)

    def test_a_rank_whose_report_disagrees_with_its_own_identity_is_caught(self, monkeypatch):
        # [FUNCTION DOC] Verifier-identity reports whose argument dump names the solver as trainer
        # model id; asserts an assertion error mentioning that the process identity says otherwise.
        """The identity names which policy the rank serves, and the args name which policy it namespaces."""
        events = _reports_of("solver") + _reports_of("verifier", trainer_model_id="solver")

        with pytest.raises(AssertionError, match="while its process identity says"):
            _assert_with(monkeypatch, events)

    def test_a_verification_that_shrank_to_fewer_arguments_is_caught(self, monkeypatch):
        # [FUNCTION DOC] Sets the expected verified argument count to nine per policy though only
        # two overrides exist; asserts an assertion error saying the verification quietly shrank.
        """An override list that quietly lost entries would pass while verifying almost nothing."""
        monkeypatch.setattr(e2e, "NUM_VERIFIED_ARGS_PER_POLICY", {"solver": 9, "verifier": 9})

        with pytest.raises(AssertionError, match="quietly shrank"):
            _assert_with(monkeypatch, _reports_of("solver") + _reports_of("verifier"))

    def test_the_reports_of_other_processes_are_not_mistaken_for_trainer_ranks(self, monkeypatch):
        # [FUNCTION DOC] Prepends to valid reports a report from a rollout executor process
        # (simple identity, reusing a solver report body with empty values) and a metric event from
        # a solver trainer; asserts no error, i.e. non-trainer and non-report events are ignored.
        """Every process of the run writes an env report, and only the trainer ranks carry a policy's args."""
        rollout_executor = EnvReportEvent(
            timestamp=datetime(2026, 8, 14, tzinfo=UTC),
            source=SimpleProcessIdentity(component="rollout_executor"),
            report=_make_report(model_id="solver", values={}).report,
        )
        metric = MetricEvent(
            timestamp=datetime(2026, 8, 14, tzinfo=UTC),
            source=TrainProcessIdentity(component="actor", model_id="solver", cell_index=0, rank_within_cell=0),
            metrics={"solver/rollout/raw_reward": 0.5},
        )

        _assert_with(monkeypatch, [rollout_executor, metric] + _reports_of("solver") + _reports_of("verifier"))
