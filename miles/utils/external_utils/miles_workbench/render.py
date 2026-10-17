# [MODULE DOC]
# Renders the workbench helm chart with helm template (running a dependency build first) and
# derives from the output which roles the chart would create and which rules they grant (the
# "rbac plan"). Also builds the helm command line overrides shared by install and render.
from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.miles_workbench.naming import CHART_DIR, object_name
from miles.utils.external_utils.miles_workbench.options import InstallArgs
from miles.utils.external_utils.miles_workbench.preflight.rules import LWS_RESOURCE
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class RbacPlan(FrozenStrictBaseModel):
    # [CLASS DOC] What the chart's rbac would create: whether any role is created and, per role
    # name, a mapping from resource to the sorted verbs that role grants.
    creates_role: bool
    granted_rules_by_role: dict[str, dict[str, tuple[str, ...]]] = {}

    @property
    def role_names(self) -> tuple[str, ...]:
        # [FUNCTION DOC] Property: names of the roles in the plan.
        return tuple(self.granted_rules_by_role)

    @property
    def granted_rules(self) -> dict[str, tuple[str, ...]]:
        # [FUNCTION DOC] Property: rules of all roles merged into one mapping from resource to
        # sorted unique verbs.
        return _merged_rules(self.granted_rules_by_role.values())

    @property
    def grants_leader_worker_sets(self) -> bool:
        # [FUNCTION DOC] Property: true if the merged rules include the leader worker set resource.
        return LWS_RESOURCE in self.granted_rules


def render_chart(args: InstallArgs) -> subprocess.CompletedProcess[str]:
    # [FUNCTION DOC] Renders the chart and returns the completed helm process (failures are
    # reported through the return code, not raised). Normal mode renders from the real chart
    # directory. Dry run mode first copies the whole charts directory into a temporary directory
    # and renders there, so the dependency build does not modify the repository; the temporary
    # copy is deleted afterwards.
    if not args.dry_run:
        return _render_chart_from(args, chart_dir=CHART_DIR)

    with tempfile.TemporaryDirectory() as scratch:
        charts_copy = Path(scratch) / CHART_DIR.parent.name
        shutil.copytree(CHART_DIR.parent, charts_copy)
        return _render_chart_from(args, chart_dir=charts_copy / CHART_DIR.name)


def _render_chart_from(args: InstallArgs, *, chart_dir: Path) -> subprocess.CompletedProcess[str]:
    # [FUNCTION DOC] HELPER: Runs helm dependency build on the chart directory and returns that
    # process if it failed; otherwise returns the helm template process for the release and
    # namespace with the value overrides.
    build = Helm.run_raw("dependency", "build", str(chart_dir))
    if build.returncode != 0:
        return build
    return Helm.run_raw("template", args.release, str(chart_dir), "-n", args.namespace, *helm_value_overrides(args))


def rbac_plan_of(rendered: str) -> RbacPlan:
    # [FUNCTION DOC] Parses the multi-document yaml output of helm template, keeps documents that
    # are mappings of kind Role, and builds a plan: role creation is true if any exist; rules per
    # role name are the merged resource-to-verbs entries of its rules (a role without rules gets an
    # empty mapping). Raises on invalid yaml or the assertion in the rule entry helper.
    roles = [
        document
        for document in yaml.safe_load_all(rendered)
        if isinstance(document, dict) and document.get("kind") == "Role"
    ]
    return RbacPlan(
        creates_role=bool(roles),
        granted_rules_by_role={
            role["metadata"]["name"]: _merged_rules(_rule_entries(rule) for rule in role.get("rules") or [])
            for role in roles
        },
    )


def _merged_rules(rule_sets: Iterable[dict[str, tuple[str, ...]]]) -> dict[str, tuple[str, ...]]:
    # [FUNCTION DOC] HELPER: Unions mappings from resource to verbs; each resulting verb tuple is the
    # sorted set union.
    merged: dict[str, tuple[str, ...]] = {}
    for entries in rule_sets:
        for resource, verbs in entries.items():
            merged[resource] = tuple(sorted({*merged.get(resource, ()), *verbs}))
    return merged


def _rule_entries(rule: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    # [FUNCTION DOC] HELPER: Expands one rbac rule (api groups, resources, verbs) into a mapping
    # from resource key to the rule's verbs. For every group and resource: the resource name is
    # split at the first slash into name and subresource; the key is the name alone for the core
    # (empty) group, otherwise name dot group; a subresource is appended after a slash. Asserts the
    # rule has no resource names restriction, because the checks ask about whole resources.
    assert "resourceNames" not in rule, (
        f"The chart grants {rule} on named objects only, but a preflight check asks kubectl about a whole "
        f"resource; qualify the check with the name before shipping a rule like this"
    )
    verbs = tuple(rule["verbs"])
    entries = {}
    for group in rule["apiGroups"]:
        for resource in rule["resources"]:
            name, _, subresource = resource.partition("/")
            key = name if group == "" else f"{name}.{group}"
            entries[f"{key}/{subresource}" if subresource else key] = verbs
    return entries


def helm_value_overrides(args: InstallArgs) -> list[str]:
    # [FUNCTION DOC] Helm arguments in order: set-string of the object name derived from the
    # release; set rbac create false when rbac is off; set leader worker set rights false when lws
    # is off; then a values file option per values file; then a raw set option per user override
    # (user overrides come last so they win).
    overrides: list[str] = ["--set-string", f"objectName={object_name(args.release)}"]
    if not args.rbac:
        overrides += ["--set", "rbac.create=false"]
    if not args.lws:
        overrides += ["--set", "rbac.leaderWorkerSets=false"]
    for values_file in args.values:
        overrides += ["-f", str(values_file)]
    for override in args.overrides:
        overrides += ["--set", override]
    return overrides
