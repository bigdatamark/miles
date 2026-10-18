# [MODULE DOC] Unit tests for the CI history tracking backend, which appends whitelisted metrics
# as JSON lines (metric key plus a series of step and value pairs) into a record directory named
# by an environment variable: policy-prefixed keys are recorded under the key as logged, separate
# policies stay separate series, unprefixed keys are unchanged, keys outside the whitelist are
# not recorded, and non-numeric values for whitelisted keys raise.
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.tracking_utils.ci_history import RECORD_DIR_ENV, CiHistoryBackend


def _recorded(backend: CiHistoryBackend) -> dict[str, list[list[object]]]:
    # [FUNCTION DOC] HELPER: Reads the backend's record file (private path attribute) and returns
    # a mapping from each line's metric key to its series (later lines of the same key win).
    lines = Path(backend._record_path).read_text().splitlines()
    return {json.loads(line)["metric"]: json.loads(line)["series"] for line in lines}


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CiHistoryBackend:
    # [FUNCTION DOC] HELPER: Fixture that points the record directory environment variable at the
    # temporary path (reverted at teardown), creates a backend and initialises it with an empty
    # argument namespace.
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    collecting = CiHistoryBackend()
    collecting.init(Namespace())
    return collecting


class TestThePolicyAMetricWasLoggedUnder:
    # [CLASS DOC] Tests of key handling for policy-prefixed metrics.
    def test_a_policy_prefixed_metric_is_recorded_under_the_key_it_was_logged_with(self, backend):
        # [FUNCTION DOC] A solver-prefixed grad norm at step 0 is recorded under exactly that key.
        """The whitelist is matched by suffix, and the record then stored it under the bare key."""
        backend.log({"solver/train/grad_norm": 1.5}, step=0)

        assert list(_recorded(backend)) == ["solver/train/grad_norm"]

    def test_two_policies_of_one_run_stay_two_series(self, backend):
        # [FUNCTION DOC] Solver and verifier grad norms at step 0 give two separate series, one
        # point each.
        """One series holding step 0 twice is what made the gate's selector refuse the duplicate step."""
        backend.log({"solver/train/grad_norm": 1.5}, step=0)
        backend.log({"verifier/train/grad_norm": 2.5}, step=0)

        assert _recorded(backend) == {
            "solver/train/grad_norm": [[0, 1.5]],
            "verifier/train/grad_norm": [[0, 2.5]],
        }

    def test_a_single_policy_run_is_recorded_exactly_as_it_was(self, backend):
        # [FUNCTION DOC] An unprefixed grad norm at step 3 is recorded under the bare key with
        # the point (3, 1.5).
        """An unprefixed key is the key a gate spec of an ordinary run names, and it may not move."""
        backend.log({"train/grad_norm": 1.5}, step=3)

        assert _recorded(backend) == {"train/grad_norm": [[3, 1.5]]}

    def test_every_step_of_one_policy_lands_in_that_policy_series(self, backend):
        # [FUNCTION DOC] Steps 0 and 1 of one policy produce one series with both points in order.
        """A gate reads a series per key, and steps of one policy scattered over two keys read as gaps."""
        backend.log({"solver/train/grad_norm": 1.0}, step=0)
        backend.log({"solver/train/grad_norm": 2.0}, step=1)

        assert _recorded(backend) == {"solver/train/grad_norm": [[0, 1.0], [1, 2.0]]}

    def test_a_key_outside_the_whitelist_is_still_never_recorded(self, backend):
        # [FUNCTION DOC] A prefixed key not on the whitelist writes nothing: the record file is
        # never created.
        """Keeping the logged key must not turn the whitelist into a prefix-matching pass-through."""
        backend.log({"solver/train/something_else": 1.0}, step=0)

        assert not Path(backend._record_path).exists()

    def test_a_non_numeric_value_is_still_refused(self, backend):
        # [FUNCTION DOC] A string value for a whitelisted key raises a type error whose message
        # names the metric.
        """An authoring error has to fail loudly rather than be dropped from the record."""
        with pytest.raises(TypeError, match="train/grad_norm"):
            backend.log({"solver/train/grad_norm": "high"}, step=0)


class TestCiHistoryBackend:
    # [CLASS DOC] Test that builds its own backend without the shared fixture.
    def test_a_namespaced_policy_metric_is_captured_under_its_canonical_ci_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Sets the record directory, creates and initialises a backend, logs an
        # alpha-prefixed grad norm of 2.5 at step 7, and asserts the directory holds exactly one
        # record file whose single JSON document has that metric key and the series with the point
        # (7, 2.5).
        """A namespaced target must be persisted under its canonical CI metric identity."""
        monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
        backend = CiHistoryBackend()
        backend.init(SimpleNamespace())

        backend.log(metrics={"alpha/train/grad_norm": 2.5}, step=7)

        [record_path] = tmp_path.glob("*.jsonl")
        assert json.loads(record_path.read_text()) == {
            "metric": "alpha/train/grad_norm",
            "series": [[7, 2.5]],
        }
