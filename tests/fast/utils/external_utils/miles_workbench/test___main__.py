# [MODULE DOC] Tests for the workbench command line application (a typer app run through the
# library's test runner): empty namespace or release options on the uninstall command are usage
# errors (exit code 2, an invalid value message) raised before any external process runs, and the
# stop command forwards the deploy component and instance id so it uninstalls exactly the
# instance specific release (verified with a fake helm script on PATH that logs its arguments).
from pathlib import Path
from typing import NoReturn

import pytest
from typer.testing import CliRunner

from miles.utils.external_utils.miles_workbench import __main__ as workbench
from miles.utils.external_utils.miles_workbench import actions


class TestKubernetesObjectNames:
    # [CLASS DOC] Tests of command line validation of Kubernetes object names.
    @pytest.mark.parametrize(
        "empty_option",
        [
            ["-n", "", "--release", "workbench"],
            ["--namespace", "training", "--release", ""],
        ],
    )
    def test_an_empty_kubernetes_object_name_is_rejected_before_cluster_access(
        self, monkeypatch: pytest.MonkeyPatch, empty_option: list[str]
    ) -> None:
        # [FUNCTION DOC] Parametrized over an empty namespace option with a release, and a
        # namespace with an empty release option. The binary lookup is faked to find any binary and
        # the process runner is replaced by one that fails the test if reached. Invoking the
        # uninstall command with those options must exit with code 2 and print a message
        # containing the text invalid value.
        """An explicitly empty namespace or release is a usage error before cluster access."""

        def fail_if_process_runs(*args: object, **kwargs: object) -> NoReturn:
            # [FUNCTION DOC] HELPER: Always raises an assertion error saying the cluster process
            # boundary was reached.
            raise AssertionError("cluster process boundary was reached")

        def fake_which(binary: str) -> str:
            # [FUNCTION DOC] HELPER: Fake binary lookup; returns a path under the usr bin directory
            # for any binary name.
            return f"/usr/bin/{binary}"

        monkeypatch.setattr(actions.shutil, "which", fake_which)
        monkeypatch.setattr(actions, "run_process", fail_if_process_runs)

        result = CliRunner().invoke(workbench.app, ["uninstall", *empty_option])

        assert result.exit_code == 2
        assert "Invalid value" in result.output


class TestStop:
    # [CLASS DOC] Test of the stop command.
    def test_stop_forwards_the_deployment_instance_id_to_the_release_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Creates an executable fake helm shell script in a temporary bin directory
        # that appends its arguments to a log file, and puts only that directory plus the system
        # bin directories on the search path. Invoking stop for namespace rl, run demo, inference
        # component and instance east must exit with code 0, and the log must hold exactly one line:
        # a helm uninstall of the release named miles-run, the run, the component and the instance
        # (hyphen joined) in namespace rl. Side effects: file writes and a changed PATH variable
        # (restored by the monkeypatch fixture).
        """Stopping one deployment instance uninstalls exactly its instance-specific release."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        calls_path = tmp_path / "calls.log"
        helm_path = bin_dir / "helm"
        helm_path.write_text(f'#!/usr/bin/env bash\necho "helm $@" >> {calls_path}\n')
        helm_path.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

        result = CliRunner().invoke(
            app=workbench.app,
            args=[
                "stop",
                "--namespace",
                "rl",
                "demo",
                "--deploy-component",
                "inference",
                "--deploy-instance-id",
                "east",
            ],
        )

        assert result.exit_code == 0, result.output
        assert calls_path.read_text().splitlines() == ["helm uninstall miles-run-demo-inference-east --namespace rl"]
