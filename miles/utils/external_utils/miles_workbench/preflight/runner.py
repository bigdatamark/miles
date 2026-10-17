# [MODULE DOC]
# Orchestrates the preflight checks for a workbench install and reports through the logger. Stages:
# binaries (stop on failure), cluster reachability (stop on failure), namespace existence plus
# workflow permissions plus foreign-object scanning, chart rendering to learn which rbac objects
# the chart would create, then install permissions, grant permissions (permissions the chart's Role
# would hand out, which the installer must hold or be able to escalate/bind) and leader worker set
# cluster checks when the chart grants them. Any failure ends in a system exit with status 1.
from __future__ import annotations

import logging

from miles.utils.external_utils.miles_workbench.options import InstallArgs
from miles.utils.external_utils.miles_workbench.preflight.checkers import (
    BaseChecker,
    BinaryPresenceChecker,
    CheckOutcome,
    CheckResult,
    ClusterReachableChecker,
    LeaderWorkerSetApiChecker,
    LeaderWorkerSetControllerChecker,
    NamespaceListingChecker,
    ResourcePresenceChecker,
    ResourceVerbAvailabilityChecker,
    RoleDelegationChecker,
    Status,
    expand_resource_verbs,
    parallel_execute_checkers,
)
from miles.utils.external_utils.miles_workbench.preflight.rules import (
    CHART_FAMILY,
    CHART_RBAC_RULES,
    CHART_RULES,
    CHART_SERVICE_ACCOUNT_RULES,
    MANAGED_BY,
    NAMESPACE_KINDS,
    WORKFLOW_RULES,
)
from miles.utils.external_utils.miles_workbench.preflight.utils import Verdict, warn
from miles.utils.external_utils.miles_workbench.render import RbacPlan, rbac_plan_of, render_chart

logger = logging.getLogger(__name__)


def run_preflight_checks(args: InstallArgs) -> None:
    # [FUNCTION DOC] Runs all preflight stages and logs each result. Stage 1: kubectl and helm
    # present, else error and exit 1. Stage 2: cluster reachable, else error and exit 1. Stage 3
    # (parallel): the namespace exists, the workflow permission verbs hold, and listings of foreign
    # objects; then warns about foreign objects. Stage 4: renders the chart (helm template) to
    # derive the rbac plan (a render failure is recorded as a failure and assumes a role is created
    # when rbac is on). Stage 5 (parallel): leader worker set api and controller checks if the chart
    # grants those rights, install permission checks (chart rules, plus service account and rbac
    # rules when a role is created), and grant checks (rules the role would hand out). Install and
    # cluster outcomes count toward the verdict; grant outcomes are only reported and then judged
    # collectively. Finally announces the summary and exits 1 on any failure.
    namespace = args.namespace
    verdict = Verdict()

    verdict.absorb(parallel_execute_checkers([BinaryPresenceChecker("kubectl"), BinaryPresenceChecker("helm")]))
    if verdict.failed:
        logger.error("Install the missing binaries before continuing")
        raise SystemExit(1)

    verdict.absorb(parallel_execute_checkers([ClusterReachableChecker()]))
    if verdict.failed:
        logger.error(
            "Fix your kubeconfig, credentials or network before reading anything below as a permission problem"
        )
        raise SystemExit(1)

    listings = _namespace_listing_checkers(namespace, release=args.release)
    verdict.absorb(
        parallel_execute_checkers(
            [
                ResourcePresenceChecker("namespace", namespace),
                *_verb_checkers(namespace, WORKFLOW_RULES),
                *listings,
            ]
        )
    )
    _warn_about_foreign_objects(namespace, listings)

    plan = _render_plan(args, verdict=verdict)

    install_rules = [CHART_RULES]
    if plan.creates_role:
        install_rules += [CHART_SERVICE_ACCOUNT_RULES, CHART_RBAC_RULES]
    install_checkers = _verb_checkers(namespace, *install_rules)
    grant_checkers = _verb_checkers(namespace, plan.granted_rules) if plan.creates_role else []
    cluster_checkers: list[BaseChecker] = []
    if plan.grants_leader_worker_sets:
        cluster_checkers += [LeaderWorkerSetApiChecker(), LeaderWorkerSetControllerChecker()]

    outcomes = parallel_execute_checkers([*cluster_checkers, *install_checkers, *grant_checkers])
    split = len(outcomes) - len(grant_checkers)
    verdict.absorb(outcomes[:split])
    grant_outcomes = outcomes[split:]
    verdict.observe(grant_outcomes)

    if plan.creates_role:
        _judge_grant(args, plan=plan, verdict=verdict, denied=_denied(grant_outcomes))

    verdict.announce()
    if verdict.failed:
        raise SystemExit(1)


def _judge_grant(args: InstallArgs, *, plan: RbacPlan, verdict: Verdict, denied: list[CheckOutcome]) -> None:
    # [FUNCTION DOC] HELPER: Decides if the installer may grant the workbench's role. No denied
    # grant checks: record a pass. Otherwise find roles for which the installer lacks delegation
    # (escalate and bind): if there are none, only warn (kubernetes would accept the role creation).
    # Else record a failure listing denied entries and the roles for which escalate and bind would
    # also do; and if any denied resource concerns leader worker sets, log an error telling that a
    # cluster admin must install the CRDs and grant the rights.
    message = f"may grant the workbench its Role in namespace {args.namespace}"
    if not denied:
        verdict.absorb_result(CheckResult(status=Status.PASS, message=message))
        return

    entries = _denied_entries(denied)
    undelegated = _roles_lacking_delegation(args.namespace, plan=plan, denied=denied)
    if not undelegated:
        warn(
            f"you do not hold every rule the chart grants (denied: {', '.join(entries)}), but for every Role "
            "whose rules you lack you have escalate and bind on that Role, which is what Kubernetes checks "
            "when creating it and its RoleBinding"
        )
        return

    verdict.absorb_result(
        CheckResult(
            status=Status.FAIL,
            message=f"{message} (denied: {', '.join(entries)}; escalate and bind on "
            f"{', '.join(undelegated)} would also do)",
        )
    )
    if any("leaderworkersets" in outcome.checker.resource for outcome in denied):
        logger.error(
            "A cluster admin must install the LWS CRDs and grant you the LeaderWorkerSet rights "
            "before this chart can grant them to the workbench"
        )


def _roles_lacking_delegation(namespace: str, *, plan: RbacPlan, denied: list[CheckOutcome]) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns names of roles in the plan that contain at least one denied
    # resource-verb pair and for which the installer cannot both escalate and bind. Builds the
    # denied verbs per resource, selects roles whose rules include a denied pair, runs escalate and
    # bind delegation checks for those in parallel, and keeps roles with any non-passing check.
    denied_verbs: dict[str, set[str]] = {}
    for outcome in denied:
        denied_verbs.setdefault(outcome.checker.resource, set()).add(outcome.checker.verb)

    lacking = [
        name
        for name, rules in plan.granted_rules_by_role.items()
        if any(verb in denied_verbs.get(resource, set()) for resource, verbs in rules.items() for verb in verbs)
    ]
    delegation = parallel_execute_checkers(
        [
            RoleDelegationChecker(namespace=namespace, verb=verb, role=role)
            for role in lacking
            for verb in ("escalate", "bind")
        ]
    )
    delegated = {
        role
        for role in lacking
        if all(outcome.result.status is Status.PASS for outcome in delegation if outcome.checker.role == role)
    }
    return [role for role in lacking if role not in delegated]


def _render_plan(args: InstallArgs, *, verdict: Verdict) -> RbacPlan:
    # [FUNCTION DOC] HELPER: Renders the chart and records a pass or a failure (with the helm
    # output) in the verdict. On failure returns a plan whose role creation follows the rbac option;
    # on success derives the plan from the rendered documents.
    message = "your values render the chart"
    rendered = render_chart(args)
    if rendered.returncode != 0:
        verdict.absorb_result(
            CheckResult(status=Status.FAIL, message=f"{message} ({(rendered.stdout + rendered.stderr).strip()})")
        )
        return RbacPlan(creates_role=args.rbac)

    verdict.absorb_result(CheckResult(status=Status.PASS, message=message))
    return rbac_plan_of(rendered.stdout)


def _namespace_listing_checkers(namespace: str, *, release: str) -> list[NamespaceListingChecker]:
    # [FUNCTION DOC] HELPER: One listing checker per namespaced object kind and per selector. The
    # selectors find objects not managed by helm, and helm-managed objects whose chart name is not in
    # the miles chart family (workbench and run charts).
    family = ",".join(CHART_FAMILY)
    selectors = [
        f"app.kubernetes.io/managed-by!={MANAGED_BY}",
        f"app.kubernetes.io/managed-by={MANAGED_BY},app.kubernetes.io/name notin ({family})",
    ]
    return [
        NamespaceListingChecker(namespace=namespace, kind=kind, selector=selector, release=release)
        for kind in NAMESPACE_KINDS
        for selector in selectors
    ]


def _warn_about_foreign_objects(namespace: str, listings: list[NamespaceListingChecker]) -> None:
    # [FUNCTION DOC] HELPER: Collects de-duplicated foreign object names from all listings (after
    # they ran) and, if any, logs a warning that the namespace holds them and the workbench role
    # covers the whole namespace.
    foreign: list[str] = []
    for listing in listings:
        foreign += [name for name in listing.foreign if name not in foreign]
    if not foreign:
        return

    warn(
        f"namespace {namespace} also holds {', '.join(foreign)}; the workbench Role covers the "
        f"whole namespace, so confirm this is the namespace you meant before sharing it with them"
    )


def _verb_checkers(namespace: str, *rule_sets: dict[str, tuple[str, ...]]) -> list[ResourceVerbAvailabilityChecker]:
    # [FUNCTION DOC] HELPER: Permission checkers for every unique resource-verb pair in the rule
    # tables.
    return [
        ResourceVerbAvailabilityChecker(namespace, resource_verb)
        for resource_verb in expand_resource_verbs(*rule_sets)
    ]


def _denied(outcomes: list[CheckOutcome]) -> list[CheckOutcome]:
    # [FUNCTION DOC] HELPER: Keeps only outcomes whose status is fail.
    return [outcome for outcome in outcomes if outcome.result.status is Status.FAIL]


def _denied_entries(denied: list[CheckOutcome]) -> list[str]:
    # [FUNCTION DOC] HELPER: Groups denied outcomes by resource (first-seen order) and renders each
    # as the resource followed by its space-joined verbs in parentheses.
    by_resource: dict[str, list[str]] = {}
    for outcome in denied:
        by_resource.setdefault(outcome.checker.resource, []).append(outcome.checker.verb)
    return [f"{resource}({' '.join(verbs)})" for resource, verbs in by_resource.items()]
