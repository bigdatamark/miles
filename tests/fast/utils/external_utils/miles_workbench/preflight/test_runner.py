# [MODULE DOC] Test for the preflight runner's foreign object warnings: a secret that the broad
# all-resources listing omits is still found through its own listing and produces a warning
# naming the namespace and the object. The raw kubectl runner is replaced by a fake that returns
# the foreign secret only for secret listings.
from __future__ import annotations

import logging
import subprocess

import pytest

from miles.utils.external_utils.miles_workbench.preflight import checkers as checkers_module
from miles.utils.external_utils.miles_workbench.preflight.runner import (
    _namespace_listing_checkers,
    _warn_about_foreign_objects,
)


class TestNamespaceForeignObjectWarnings:
    # [CLASS DOC] Test of the runner's foreign object warning step.
    def test_a_foreign_secret_omitted_by_get_all_is_still_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] Builds the private namespace listing checkers for namespace rl and a
        # release, runs each one against the fake kubectl, then runs the private warning step at
        # warning level on the preflight utils logger. The log contains the sentence that
        # namespace rl also holds the foreign secret.
        """A secret omitted from kubectl get all still produces a foreign-object warning."""

        def run_raw(*args: str) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake raw kubectl runner; for a listing whose second argument
            # (the kind) is secret it outputs one foreign secret line, otherwise empty output.
            # Always exits with code 0.
            output = "secret/foreign-secret\n" if args[1] == "secret" else ""
            return subprocess.CompletedProcess(args=list(args), returncode=0, stdout=output, stderr="")

        monkeypatch.setattr(checkers_module.Kubectl, "run_raw", staticmethod(run_raw))
        listings = _namespace_listing_checkers("rl", release="my-release")
        for listing in listings:
            listing.check()

        with caplog.at_level(logging.WARNING, logger="miles.utils.external_utils.miles_workbench.preflight.utils"):
            _warn_about_foreign_objects(namespace="rl", listings=listings)

        assert "namespace rl also holds secret/foreign-secret" in caplog.text
