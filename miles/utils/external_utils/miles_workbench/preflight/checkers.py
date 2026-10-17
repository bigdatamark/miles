# [MODULE DOC]
# Preflight check vocabulary and implementations. A checker is an object with a check method
# returning a result (pass, fail or unknown, plus a message). Checkers query the cluster through
# kubectl (raw invocation; never raises on kubectl failure, instead interprets the text output) and
# can be executed in parallel with a bounded thread pool. A failed query is "unknown" rather than
# "fail" when the account is merely forbidden to look. Messages state the property being checked, so
# a failure reads as the negation of the message.
from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import NamedTuple

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.miles_workbench.preflight.rules import (
    AVAILABLE_CONDITION,
    CLUSTER_PROVIDED_RESOURCES,
    DEFAULT_TOKEN_PREFIX,
    HELM_RELEASE_RECORD_PREFIX,
    LWS_API_GROUP,
    LWS_CONTROLLER_DEPLOYMENT,
    LWS_CONTROLLER_NAMESPACE,
    LWS_RESOURCE,
    UNSERVED_RESOURCE_MARKERS,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

ROLES_RESOURCE = "roles.rbac.authorization.k8s.io"

_MAX_CONCURRENT_QUERIES = 32
_FORBIDDEN_MARKER = "(Forbidden)"
_ANONYMOUS_IDENTITIES = ("system:anonymous", "system:unauthenticated")


# ============================== vocabulary ==============================


class Status(Enum):
    # [CLASS DOC] Outcome of a check: pass, fail, or unknown (could not be verified).
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class CheckResult(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable result of one check: status and human-readable message.
    status: Status
    message: str


class ResourceVerb(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable (hashable) pair of an access verb and a resource (optionally with a
    # slash-separated subresource), used for permission checks.
    verb: str
    resource: str


class BaseChecker:
    # [CLASS DOC] Interface of all checkers.
    def check(self) -> CheckResult:
        # [FUNCTION DOC] Runs the check and returns its result; subclasses must override (the base
        # raises a not-implemented error).
        raise NotImplementedError


class CheckOutcome(NamedTuple):
    # [CLASS DOC] Pairing of a checker with the result it produced.
    checker: BaseChecker
    result: CheckResult


def parallel_execute_checkers(checkers: Sequence[BaseChecker]) -> list[CheckOutcome]:
    # [FUNCTION DOC] Runs all checkers concurrently in a thread pool of up to 32 workers and returns
    # outcomes in the same order as the input. An empty input returns an empty list without
    # creating a pool. Exceptions raised by a checker propagate when its result is collected.
    if not checkers:
        return []

    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_QUERIES) as pool:
        results = pool.map(lambda checker: checker.check(), checkers)
        return [CheckOutcome(checker, result) for checker, result in zip(checkers, results, strict=True)]


def expand_resource_verbs(*rule_sets: dict[str, tuple[str, ...]]) -> list[ResourceVerb]:
    # [FUNCTION DOC] Flattens rule tables (each a mapping from resource to a tuple of verbs) into a
    # list of unique resource-verb pairs, preserving first-seen order across tables.
    seen: dict[ResourceVerb, None] = {}
    for rules in rule_sets:
        for resource, verbs in rules.items():
            for verb in verbs:
                seen.setdefault(ResourceVerb(verb=verb, resource=resource))
    return list(seen)


# =============================== checkers ===============================


class BinaryPresenceChecker(BaseChecker):
    # [CLASS DOC] Checks that an executable is on the path.
    def __init__(self, binary: str) -> None:
        # [FUNCTION DOC] Stores the binary name.
        self.binary = binary

    def check(self) -> CheckResult:
        # [FUNCTION DOC] Pass if the binary is found on the path, else fail; message says the
        # binary is installed.
        message = f"{self.binary} is installed"
        found = shutil.which(self.binary) is not None
        return CheckResult(status=Status.PASS if found else Status.FAIL, message=message)


class ClusterReachableChecker(BaseChecker):
    # [CLASS DOC] Checks that the cluster answers and the credentials are accepted.
    def check(self) -> CheckResult:
        # [FUNCTION DOC] Runs the kubectl whoami query. If it fails: unknown when the output
        # mentions forbidden or an unserved resource type (older clusters lack the whoami api),
        # otherwise fail. If it succeeds but the output names an anonymous or unauthenticated
        # identity, fail (context served without credentials). Otherwise pass.
        message = "cluster is reachable and your credentials are accepted"
        answer = _query("auth", "whoami")
        if not answer.ok:
            markers = (_FORBIDDEN_MARKER, *UNSERVED_RESOURCE_MARKERS)
            return _failed_query_result(
                message, answer, unverifiable=any(marker in answer.output for marker in markers)
            )
        if any(identity in answer.output for identity in _ANONYMOUS_IDENTITIES):
            return CheckResult(
                status=Status.FAIL,
                message=f"{message}: the cluster served this context without credentials ({answer.output})",
            )
        return CheckResult(status=Status.PASS, message=message)


class ResourcePresenceChecker(BaseChecker):
    # [CLASS DOC] Checks that a named object of a kind exists, optionally within a namespace.
    def __init__(self, kind: str, name: str, namespace: str | None = None) -> None:
        # [FUNCTION DOC] Stores kind, name and optional namespace.
        self.kind = kind
        self.name = name
        self.namespace = namespace

    def check(self) -> CheckResult:
        # [FUNCTION DOC] Runs a kubectl get for the object (name passed after a separator so it
        # cannot be read as a flag). Pass on success; on failure, unknown if forbidden, else fail,
        # with the kubectl output appended.
        scope = ["-n", self.namespace] if self.namespace else []
        answer = _query("get", self.kind, *scope, "--", self.name)
        where = f" in namespace {self.namespace}" if self.namespace else ""
        message = f"{self.kind} {self.name} exists{where}"
        if answer.ok:
            return CheckResult(status=Status.PASS, message=message)
        return _failed_query_result(message, answer, unverifiable=_FORBIDDEN_MARKER in answer.output)


class NamespaceListingChecker(BaseChecker):
    # [CLASS DOC] Lists objects of a kind matching a label selector in a namespace and records those
    # that do not belong to the workbench (foreign objects) in the foreign attribute. Never fails:
    # the result is pass or unknown; callers read the foreign attribute afterwards.
    def __init__(self, namespace: str, kind: str, selector: str, release: str) -> None:
        # [FUNCTION DOC] Stores namespace, kind, selector and release; foreign starts empty.
        self.namespace = namespace
        self.kind = kind
        self.selector = selector
        self.release = release
        self.foreign: tuple[str, ...] = ()

    def check(self) -> CheckResult:
        # [FUNCTION DOC] Lists names (kubectl name output) of the kind matching the selector. On
        # failure returns unknown (a distinct message when the kind is not served by the cluster).
        # On success stores in foreign every listed name that is neither cluster-provided (default
        # service account, root certificate config map, default token secrets) nor one of this
        # release's own helm release record secrets, and returns pass. Note: foreign is mutated as a
        # side effect, from a worker thread.
        message = f"namespace {self.namespace} holds nothing but Miles releases ({self.kind} matching {self.selector})"
        answer = _query("get", self.kind, "-n", self.namespace, "-l", self.selector, "-o", "name")
        if not answer.ok:
            if any(marker in answer.output for marker in UNSERVED_RESOURCE_MARKERS):
                return CheckResult(
                    status=Status.UNKNOWN, message=f"the cluster does not serve {self.kind}, so nothing lists it"
                )
            return CheckResult(
                status=Status.UNKNOWN,
                message=f"could not check whether namespace {self.namespace} holds nothing but Miles releases "
                f"(listing {self.kind} failed: {answer.output})",
            )

        own_record_prefix = f"{HELM_RELEASE_RECORD_PREFIX}{self.release}."
        self.foreign = tuple(
            name
            for name in answer.output.split()
            if not _is_cluster_provided(name) and not name.startswith(own_record_prefix)
        )
        return CheckResult(status=Status.PASS, message=message)


class ResourceVerbAvailabilityChecker(BaseChecker):
    # [CLASS DOC] Checks whether the current account may perform a verb on a resource in a
    # namespace, using kubectl's can-i query.
    def __init__(self, namespace: str, resource_verb: ResourceVerb) -> None:
        # [FUNCTION DOC] Stores namespace and the resource-verb pair.
        self.namespace = namespace
        self.resource_verb = resource_verb

    @property
    def resource(self) -> str:
        # [FUNCTION DOC] Property: the resource of the pair.
        return self.resource_verb.resource

    @property
    def verb(self) -> str:
        # [FUNCTION DOC] Property: the verb of the pair.
        return self.resource_verb.verb

    def check(self) -> CheckResult:
        # [FUNCTION DOC] Splits the resource at the first slash into a target and subresource, runs
        # can-i for the verb with the subresource option when present and the namespace. Pass if
        # kubectl exits successfully (answer yes), otherwise fail (never unknown).
        target, _, subresource = self.resource.partition("/")
        args = ["auth", "can-i", self.verb, target]
        if subresource:
            args.append(f"--subresource={subresource}")
        args += ["-n", self.namespace]

        message = f"may {self.verb} {self.resource} in namespace {self.namespace}"
        return CheckResult(status=Status.PASS if _query(*args).ok else Status.FAIL, message=message)


class RoleDelegationChecker(BaseChecker):
    # [CLASS DOC] Checks whether the account may use a verb (escalate or bind) on a particular role,
    # which lets it create roles granting rights it does not itself hold.
    def __init__(self, namespace: str, verb: str, role: str) -> None:
        # [FUNCTION DOC] Stores namespace, verb and role name.
        self.namespace = namespace
        self.verb = verb
        self.role = role

    def check(self) -> CheckResult:
        # [FUNCTION DOC] Pass if can-i succeeds for the verb on either all roles or the named role
        # (tried in that order); otherwise fail.
        message = f"may {self.verb} role {self.role} in namespace {self.namespace}"
        for target in (ROLES_RESOURCE, f"{ROLES_RESOURCE}/{self.role}"):
            if _query("auth", "can-i", self.verb, target, "-n", self.namespace).ok:
                return CheckResult(status=Status.PASS, message=message)
        return CheckResult(status=Status.FAIL, message=message)


class LeaderWorkerSetApiChecker(BaseChecker):
    # [CLASS DOC] Checks that the cluster serves the leader worker set resource.
    def check(self) -> CheckResult:
        # [FUNCTION DOC] Lists api resources of the leader worker set group. A failed query gives
        # unknown. A listing lacking the resource gives fail naming what was served. Otherwise pass.
        message = f"the cluster serves {LWS_RESOURCE}"
        answer = _query("api-resources", "--api-group", LWS_API_GROUP, "-o", "name")
        if not answer.ok:
            return _failed_query_result(message, answer, unverifiable=True)
        if LWS_RESOURCE not in answer.output.split():
            served = answer.output or "nothing"
            return CheckResult(
                status=Status.FAIL, message=f"{message} (api discovery served {served} in {LWS_API_GROUP})"
            )
        return CheckResult(status=Status.PASS, message=message)


class LeaderWorkerSetControllerChecker(BaseChecker):
    # [CLASS DOC] Checks that the leader worker set controller deployment is available.
    def check(self) -> CheckResult:
        # [FUNCTION DOC] Reads the Available condition status of the controller deployment in its
        # system namespace. Failed query: unknown if forbidden, else fail. Status other than the
        # text True: fail naming the value. Otherwise pass.
        message = f"deployment {LWS_CONTROLLER_DEPLOYMENT} is available in namespace {LWS_CONTROLLER_NAMESPACE}"
        answer = _query(
            "get",
            "deployment.apps",
            "-n",
            LWS_CONTROLLER_NAMESPACE,
            "-o",
            AVAILABLE_CONDITION,
            "--",
            LWS_CONTROLLER_DEPLOYMENT,
        )
        if not answer.ok:
            return _failed_query_result(message, answer, unverifiable=_FORBIDDEN_MARKER in answer.output)
        if answer.output != "True":
            condition = answer.output or "nothing"
            return CheckResult(status=Status.FAIL, message=f"{message} (the Available condition reads {condition})")
        return CheckResult(status=Status.PASS, message=message)


# ================================ queries ===============================


class _Answer(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Result of a kubectl query: success flag and stripped combined text of
    # standard output and error.
    ok: bool
    output: str


def _query(*args: str) -> _Answer:
    # [FUNCTION DOC] HELPER: Runs kubectl with the arguments without raising on failure; ok is true
    # for exit code zero; output is stdout then stderr concatenated and stripped.
    result = Kubectl.run_raw(*args)
    return _Answer(ok=result.returncode == 0, output=(result.stdout + result.stderr).strip())


def _failed_query_result(message: str, answer: _Answer, *, unverifiable: bool) -> CheckResult:
    # [FUNCTION DOC] HELPER: Builds the result for a failed query: unknown (account may not look)
    # when unverifiable, else fail; the kubectl output is included in the message.
    if unverifiable:
        return CheckResult(
            status=Status.UNKNOWN,
            message=f"{message}: this account may not look, so nothing here confirms it ({answer.output})",
        )
    return CheckResult(status=Status.FAIL, message=f"{message} ({answer.output})")


def _is_cluster_provided(name: str) -> bool:
    # [FUNCTION DOC] HELPER: True for names the cluster creates itself in every namespace (listed
    # names, or secrets starting with the default token prefix).
    return name in CLUSTER_PROVIDED_RESOURCES or name.startswith(DEFAULT_TOKEN_PREFIX)
