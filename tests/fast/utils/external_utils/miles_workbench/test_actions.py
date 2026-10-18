# [MODULE DOC] Tests for the workbench actions module: the private run helper turns a failed
# external command into a process exit with the same status code, and the private namespace
# ensure helper warns (instead of failing) and lets installation continue when probes for the
# namespace are inconclusive (forbidden, or unable to connect), including the probe's error text.
from __future__ import annotations

import logging
import subprocess

import pytest

from miles.utils.external_utils.miles_workbench import actions as actions_module


class TestRun:
    # [CLASS DOC] Test of the private run helper.
    def test_a_failed_external_command_exits_with_its_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With binary lookup faked and the process runner returning exit code 7,
        # running a kubectl get pods command raises a system exit whose code is 7.
        """A failed external command exits with the exact status reported by the process."""

        def run_process(command: list[str], *, capture_output: bool, check: bool) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake process runner; returns a completed process for the
            # command with return code 7 and empty output.
            return subprocess.CompletedProcess(args=command, returncode=7, stdout="", stderr="")

        monkeypatch.setattr(actions_module.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(actions_module, "run_process", run_process)

        with pytest.raises(SystemExit) as raised:
            actions_module._run(["kubectl", "get", "pods"])

        assert raised.value.code == 7


class TestEnsureNamespace:
    # [CLASS DOC] Test of the private namespace ensure helper.
    def test_an_inconclusive_namespaced_probe_warns_and_continues(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] The raw kubectl runner is replaced by one that returns two scripted
        # failures in order: a forbidden answer for the namespace get, then an unable to connect
        # answer for the service accounts listing in the namespace. Ensuring namespace rl does not
        # raise, and the warning level log contains both the sentence saying nothing says whether
        # namespace rl exists and the connection error text.
        """An inconclusive namespaced probe warns and lets installation continue."""
        results = iter(
            (
                subprocess.CompletedProcess(
                    args=["kubectl", "get", "namespace", "--", "rl"],
                    returncode=1,
                    stdout="",
                    stderr="Error from server (Forbidden): namespaces is forbidden",
                ),
                subprocess.CompletedProcess(
                    args=["kubectl", "get", "serviceaccounts", "-n", "rl", "-o", "name"],
                    returncode=1,
                    stdout="",
                    stderr="Unable to connect to the server",
                ),
            )
        )

        def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake raw kubectl runner; ignores its arguments and returns the
            # next scripted result (raises a stop iteration error if called a third time).
            return next(results)

        monkeypatch.setattr(actions_module.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(actions_module.Kubectl, "run_raw", staticmethod(run_raw))

        with caplog.at_level(logging.WARNING, logger=actions_module.__name__):
            actions_module._ensure_namespace("rl")

        assert "nothing here says whether namespace rl exists" in caplog.text
        assert "Unable to connect to the server" in caplog.text
