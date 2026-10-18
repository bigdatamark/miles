# [MODULE DOC] Tests for the workbench chart rendering and RBAC planning: rendering the chart
# passes the install's object name as a string setting (so roles and service accounts are not
# rendered with empty names), a failed dependency build result is returned as is without
# rendering, and RBAC planning rejects a rule that grants access to named objects only.
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from miles.utils.external_utils.miles_workbench import render as render_module
from miles.utils.external_utils.miles_workbench.naming import object_name
from miles.utils.external_utils.miles_workbench.options import InstallArgs
from miles.utils.external_utils.miles_workbench.render import rbac_plan_of


def _install_args(*, release: str = "workbench") -> InstallArgs:
    # [FUNCTION DOC] HELPER: Builds install arguments for namespace rl and the given release
    # (default workbench) with RBAC and the leader-worker-set support enabled, no dry run, no
    # values files or overrides, preflight not skipped, and a timeout of 60.
    return InstallArgs(
        namespace="rl",
        release=release,
        rbac=True,
        lws=True,
        dry_run=False,
        values=(),
        overrides=(),
        skip_preflight=False,
        timeout=60,
    )


class TestRenderChartFrom:
    # [CLASS DOC] Tests of the private chart render helper, with the raw helm runner faked.
    def test_the_chart_is_rendered_under_the_object_name_the_install_will_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Rendering for release wb records every helm call's arguments; the last
        # call contains the string-setting option and a setting of the object name equal to the
        # object name computed for release wb.
        """Rendering without it names every Role and ServiceAccount empty, so the preflight can only fail."""
        rendered: list[tuple[str, ...]] = []

        def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake raw helm runner; records the arguments and returns
            # success with empty output.
            rendered.append(arguments)
            return subprocess.CompletedProcess(args=list(arguments), returncode=0, stdout="", stderr="")

        monkeypatch.setattr(render_module.Helm, "run_raw", staticmethod(run_raw))

        render_module._render_chart_from(_install_args(release="wb"), chart_dir=Path("/chart"))

        assert "--set-string" in rendered[-1]
        assert f"objectName={object_name('wb')}" in rendered[-1]

    def test_a_failed_dependency_build_is_returned_without_rendering(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] The fake helm runner returns a failure (exit code 9) for everything except
        # the template command, which would raise an assertion error. Rendering returns that very
        # failure object, so the template command was never run.
        """A failed dependency build is returned without attempting to render the chart."""
        failure = subprocess.CompletedProcess(
            args=["helm", "dependency", "build", "/chart"], returncode=9, stdout="build output", stderr="build error"
        )

        def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake raw helm runner; raises an assertion error if asked to
            # template, otherwise returns the prepared failure.
            if arguments[0] == "template":
                raise AssertionError("rendering must not occur after dependency-build failure")
            return failure

        monkeypatch.setattr(render_module.Helm, "run_raw", staticmethod(run_raw))

        result = render_module._render_chart_from(_install_args(), chart_dir=Path("/chart"))

        assert result is failure


class TestRbacPlanOf:
    # [CLASS DOC] Test of RBAC plan construction from rendered manifests.
    def test_a_named_resource_grant_is_rejected_before_creating_a_whole_resource_plan(self) -> None:
        # [FUNCTION DOC] A role manifest granting get on secrets restricted by a resource name
        # makes plan construction raise an assertion error saying named objects only.
        """A named-object grant is rejected before becoming a whole-resource permission check."""
        rendered = """
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: workbench
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["workbench-token"]
    verbs: ["get"]
"""

        with pytest.raises(AssertionError, match="named objects only"):
            rbac_plan_of(rendered)
