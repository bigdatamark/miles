#!/usr/bin/env python3
# [MODULE DOC]
# Command handler for CI slash commands posted as pull request comments on the radixark/miles
# repository (run as a GitHub Actions job triggered by the issue comment created event). Supported
# comment commands (the whole trimmed comment must be exactly one command): a label command
# ("/run-ci-*", "/run-on-*" or "/bypass-fastfail") adds that label to the pull request;
# "/clear-labels" removes all CI control labels; "/rerun-failed-ci" reruns the failed jobs of the
# latest failed runs of the pre-commit, PR test and ROCm PR test workflows for the head commit;
# "/rerun-test <test file>" dispatches the run-ci-file workflow for one registered test file.
# Comments containing a command marker but not parsing as exactly one command are errors; comments
# with no marker are ignored. A JSON policy file (strict, version 4) maps each command to a group
# of allowed repository permissions, user ids and author associations, and lists the label names
# the add-label command may add. Authorization is checked against the live GitHub API, never the
# comment payload alone. The script runs in one of several modes selected by environment variables
# (file-run announce/report status, acknowledge with a reaction, preflight authorization that
# writes routing outputs, or the default full processing). Every successful mode prints one compact
# JSON audit mapping to stdout; errors print a workflow error annotation and exit with status one.
# Environment: GITHUB_EVENT_PATH, GITHUB_OUTPUT, CI_COMMAND_API_TOKEN, CI_COMMAND_POLICY_PATH,
# CI_COMMAND_ACKNOWLEDGE, CI_COMMAND_PREFLIGHT, CI_COMMAND_FILE_RUN_STATUS and FILE_RUN_* values.

import datetime
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

API_ROOT = "https://api.github.com"
REPOSITORY = "radixark/miles"
REPOSITORY_ID = 1072725553
CLEAR_COMMAND = "clear-labels"
RERUN_COMMAND = "rerun-failed-ci"
RUN_FILE_COMMAND = "rerun-test"
RUN_FILE_WORKFLOW = "run-ci-file.yml"
RUN_FILE_WORKFLOW_REF = "main"
CLEAR_EXACT_LABELS = frozenset({"nightly", "bypass-fastfail"})
RERUN_WORKFLOWS = (
    ("pre-commit.yml", ".github/workflows/pre-commit.yml"),
    ("pr-test.yml", ".github/workflows/pr-test.yml"),
    ("pr-test-rocm.yml", ".github/workflows/pr-test-rocm.yml"),
)
COMMAND_PATTERN = re.compile(r"/(run-(?:ci|on)-[A-Za-z0-9][A-Za-z0-9_.-]*|bypass-fastfail)")
# Label-family markers and "/rerun-test" must parse as one exact command or
# fail loudly.
COMMAND_MARKERS = (
    "/run-ci",
    "/run-on",
    "/rerun-test",
    "/bypass-fastfail",
    "/clear-labels",
    "/rerun-failed-ci",
)
# Registered test files only: fixed roots, path segments that cannot form
# ".." or an absolute path, and a test_*.py basename. The dispatched workflow
# re-validates the same shape before the path reaches any shell command.
TEST_FILE_PATTERN = re.compile(
    r"tests/(?:e2e|fast|fast-gpu|ci)(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)*/test_[A-Za-z0-9_][A-Za-z0-9_.-]*\.py"
)
RUN_FILE_COMMAND_PATTERN = re.compile(rf"/{RUN_FILE_COMMAND} +({TEST_FILE_PATTERN.pattern})")
# PR-body dependency pins, mirroring the exact patterns the pr-test.yml
# resolve steps grep for; each match is forwarded as one dispatch input.
PR_BODY_PINS = (
    (
        "ci_image_tag",
        re.compile(r"^ci-image-tag:\s+(\S+)", re.MULTILINE),
        re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"),
    ),
    (
        "ci_megatron_pr",
        re.compile(r"^ci-megatron-pr:\s+(\S+)", re.MULTILINE),
        re.compile(r"#[0-9]+|[A-Za-z0-9_][A-Za-z0-9_./-]*"),
    ),
    (
        "ci_megatron_bridge_pr",
        re.compile(r"^ci-megatron-bridge-pr:\s+(\S+)", re.MULTILINE),
        re.compile(r"#[0-9]+|[A-Za-z0-9_][A-Za-z0-9_./-]*"),
    ),
    (
        "ci_sglang_pr",
        re.compile(r"^ci-sglang-pr:\s+(\S+)", re.MULTILINE),
        re.compile(r"#[0-9]+|[A-Za-z0-9_][A-Za-z0-9_./-]*"),
    ),
)
# Shapes a policy entry may take. Widening this does not widen what a comment
# can add: `commands.add_label.allowed_labels` stays an exact default-deny
# allowlist, and this only rejects malformed entries in it.
LABEL_PATTERN = re.compile(r"(?:run-(?:ci|on)-[A-Za-z0-9][A-Za-z0-9_.-]*|bypass-fastfail)")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
WORKFLOW_RUN_URL_PATTERN = re.compile(rf"https://github\.com/{re.escape(REPOSITORY)}/actions/runs/[1-9][0-9]*")
# Suite names come from the resolver, and reach a comment body verbatim.
SUITE_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")
POLICY_PERMISSIONS = frozenset({"write", "admin"})
POLICY_GROUPS = frozenset({"add_label_access", "repo_write_access", "prior_contributor_access"})
# The comment payload's author_association enum, as GitHub defines it. Groups
# may allow a subset; associations outside the enum fail loudly.
AUTHOR_ASSOCIATIONS = frozenset(
    {
        "COLLABORATOR",
        "CONTRIBUTOR",
        "FIRST_TIMER",
        "FIRST_TIME_CONTRIBUTOR",
        "MANNEQUIN",
        "MEMBER",
        "NONE",
        "OWNER",
    }
)


class CommentCommandError(Exception):
    # [CLASS DOC]
    # The single exception type for every expected failure (bad input, policy violation, API
    # error, authorization denial). The main function catches it and turns its message into a
    # workflow error annotation, so messages must never contain secrets.
    pass


class AddLabel(NamedTuple):
    # [CLASS DOC]
    # Parsed request to add one CI label; holds the label name.
    label: str


class ClearLabels(NamedTuple):
    # [CLASS DOC]
    # Parsed request to remove all CI control labels; carries no data.
    pass


class RerunFailedCI(NamedTuple):
    # [CLASS DOC]
    # Parsed request to rerun failed CI jobs for the pull request head; carries no data.
    pass


class RunTestFile(NamedTuple):
    # [CLASS DOC]
    # Parsed request to run one test file through the dispatch workflow; holds the repository
    # relative test file path, already matched against the allowed test path pattern.
    test_file: str


class CommandSpec(NamedTuple):
    # [CLASS DOC]
    # Static registration of one command type in the command registry. Fields: policy_key (the name
    # of the command in the policy file); capability (token capability needed: issues or actions,
    # routed to later workflow steps); handler (function taking a context and request and returning
    # an audit mapping); resource_key (policy field holding a resource allowlist, or none);
    # resource_loader (validator for that policy field, or none); resource_getter (extracts the
    # resource from a request, or none); allows_user_ids (whether the policy group may grant access
    # by explicit user id); audit_key and audit_value (name of the audit field and a function
    # producing its value from a request); success_reaction (comment reaction to add after
    # success, or the word none).
    policy_key: str
    capability: str
    handler: object
    resource_key: object
    resource_loader: object
    resource_getter: object
    allows_user_ids: bool
    audit_key: str
    audit_value: object
    success_reaction: str


class CommandContext(NamedTuple):
    # [CLASS DOC]
    # Everything a command handler needs, assembled by the event processor: the API client; pull
    # request number; comment author id, login and author association; the policy-derived allowed
    # repository permissions, allowed user ids and allowed author associations; the pull request's
    # current label names (frozen set); head commit sha; head repository id, owner login and branch
    # name; and the pull request body text.
    api: object
    pull_number: int
    actor_id: int
    actor_login: str
    author_association: str
    allowed_permissions: frozenset
    allowed_user_ids: frozenset
    allowed_author_associations: frozenset
    current_labels: frozenset
    head_sha: str
    head_repository_id: int
    head_owner_login: str
    head_ref: str
    body: str


def _strict_object(pairs):
    # [FUNCTION DOC] HELPER: JSON object-pairs hook. Builds a dictionary and raises a command error
    # naming the key if any key appears twice.
    result = {}
    for key, value in pairs:
        if key in result:
            raise CommentCommandError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    # [FUNCTION DOC] HELPER: JSON constant hook; always raises a command error naming the
    # non-standard number literal (NaN or infinity) so it is rejected.
    raise CommentCommandError(f"non-standard JSON number: {value}")


def load_json(path):
    # [FUNCTION DOC] Reads a UTF-8 file and parses it as strict JSON (duplicate keys and
    # non-standard number constants rejected). Returns the parsed value. Raises a command error
    # including the path and cause if the file is unreadable, not valid UTF-8 or not valid JSON.
    try:
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CommentCommandError(f"cannot read JSON from {path}: {error}") from error


def _validate_permissions(name, values):
    # [FUNCTION DOC] HELPER: Validates a policy list of repository permission names. It must be a
    # non-empty list of exact strings drawn only from write and admin, with no duplicates; the name
    # is used in error messages. Returns a frozen set.
    if not isinstance(values, list) or not values:
        raise CommentCommandError(f"{name} must be a non-empty array")
    if any(type(value) is not str or value not in POLICY_PERMISSIONS for value in values):
        raise CommentCommandError(f"{name} must contain only write or admin")
    if len(set(values)) != len(values):
        raise CommentCommandError(f"{name} contains duplicate permissions")
    return frozenset(values)


def _validate_author_associations(name, values):
    # [FUNCTION DOC] HELPER: Same as the permissions validator but for GitHub author association
    # names: non-empty list of exact strings from the known association set, no duplicates.
    # Returns a frozen set.
    if not isinstance(values, list) or not values:
        raise CommentCommandError(f"{name} must be a non-empty array")
    if any(type(value) is not str or value not in AUTHOR_ASSOCIATIONS for value in values):
        raise CommentCommandError(f"{name} must contain only GitHub author associations")
    if len(set(values)) != len(values):
        raise CommentCommandError(f"{name} contains duplicate author associations")
    return frozenset(values)


def _validate_labels(name, values):
    # [FUNCTION DOC] HELPER: Validates the policy allowlist of exact label names for the add-label
    # command: a non-empty list, every entry a string fully matching the CI label shape (run-ci-* or
    # run-on-* names, or the bypass fast-fail label), no duplicates. Returns a frozen set.
    if not isinstance(values, list) or not values:
        raise CommentCommandError(f"{name} must be a non-empty array")
    for label in values:
        if not isinstance(label, str) or LABEL_PATTERN.fullmatch(label) is None:
            raise CommentCommandError(f"invalid exact CI label: {label!r}")
    if len(set(values)) != len(values):
        raise CommentCommandError(f"{name} contains duplicate labels")
    return frozenset(values)


def load_policy(path):
    # [FUNCTION DOC] Loads and validates the policy file. Returns a mapping with groups and
    # commands. The top level must have exactly version, groups and commands; version must be the
    # integer four. Groups must be exactly add_label_access, repo_write_access and
    # prior_contributor_access; each has repository_permissions, the first and third also users
    # (a list of objects with an id), and the third also author_associations. Result per group:
    # permissions set, user id set (empty if no users) and author association set (empty if none).
    # Commands must match exactly the policy keys in the command registry; each has a group name
    # naming a defined group plus, for commands with a resource key, that resource loaded through the
    # registered validator. A command whose registration forbids user-id grants may not use a
    # group with any user ids. Raises a command error on any violation.
    # KNOWN ISSUE: user entries are read by indexing their id without checking they are objects or
    # that the ids are positive integers; malformed entries raise raw errors or are accepted.
    raw = load_json(path)
    if not isinstance(raw, dict) or set(raw) != {"version", "groups", "commands"}:
        raise CommentCommandError("policy must contain only version, groups, and commands")
    if type(raw["version"]) is not int or raw["version"] != 4:
        raise CommentCommandError("policy version must be 4")

    raw_groups = raw["groups"]
    if not isinstance(raw_groups, dict) or set(raw_groups) != POLICY_GROUPS:
        raise CommentCommandError(
            "policy groups must contain only add_label_access, repo_write_access, and prior_contributor_access"
        )
    groups = {}
    for name, raw_group in raw_groups.items():
        expected_keys = {"repository_permissions"}
        if name in {"add_label_access", "prior_contributor_access"}:
            expected_keys.add("users")
        if name == "prior_contributor_access":
            expected_keys.add("author_associations")
        if not isinstance(raw_group, dict) or set(raw_group) != expected_keys:
            raise CommentCommandError(f"invalid fields for policy group {name}")
        groups[name] = {
            "repository_permissions": _validate_permissions(
                f"groups.{name}.repository_permissions",
                raw_group["repository_permissions"],
            ),
            "user_ids": (
                frozenset(user["id"] for user in raw_group["users"]) if "users" in raw_group else frozenset()
            ),
            "author_associations": (
                _validate_author_associations(
                    f"groups.{name}.author_associations",
                    raw_group["author_associations"],
                )
                if "author_associations" in raw_group
                else frozenset()
            ),
        }

    specs_by_key = {spec.policy_key: spec for spec in COMMAND_REGISTRY.values()}
    if len(specs_by_key) != len(COMMAND_REGISTRY):
        raise CommentCommandError("command registry contains duplicate policy keys")
    raw_commands = raw["commands"]
    if not isinstance(raw_commands, dict) or set(raw_commands) != set(specs_by_key):
        raise CommentCommandError("policy commands do not match the static command registry")
    commands = {}
    for name, spec in specs_by_key.items():
        raw_command = raw_commands[name]
        expected_keys = {"group"}
        if spec.resource_key is not None:
            expected_keys.add(spec.resource_key)
        if not isinstance(raw_command, dict) or set(raw_command) != expected_keys:
            raise CommentCommandError(f"invalid fields for policy command {name}")
        group_name = raw_command["group"]
        if not isinstance(group_name, str) or group_name not in groups:
            raise CommentCommandError(f"policy command {name} references an unknown group")
        if not spec.allows_user_ids and groups[group_name]["user_ids"]:
            raise CommentCommandError(f"policy command {name} cannot grant access by user ID")
        command = {"group": group_name}
        if spec.resource_key is not None:
            command[spec.resource_key] = spec.resource_loader(
                f"commands.{name}.{spec.resource_key}",
                raw_command[spec.resource_key],
            )
        commands[name] = command

    return {"groups": groups, "commands": commands}


def parse_command(body):
    # [FUNCTION DOC] Parses a comment body (must be a string, else command error). After trimming
    # whitespace the entire text must equal one of: the clear-labels command, the rerun-failed-ci
    # command, a label command (returns an add-label request with the label name), or the rerun-test
    # command followed by spaces and an allowed test file path (returns a run-test-file request).
    # If it matches none but contains any command marker substring anywhere, raises a command error
    # saying exactly one command is required. If no marker is present returns none (ignored).
    if not isinstance(body, str):
        raise CommentCommandError("comment body must be a string")
    command = body.strip()
    if command == f"/{CLEAR_COMMAND}":
        return ClearLabels()
    if command == f"/{RERUN_COMMAND}":
        return RerunFailedCI()
    match = COMMAND_PATTERN.fullmatch(command)
    if match is not None:
        return AddLabel(match.group(1))
    match = RUN_FILE_COMMAND_PATTERN.fullmatch(command)
    if match is not None:
        return RunTestFile(match.group(1))
    if any(marker in command for marker in COMMAND_MARKERS):
        raise CommentCommandError(
            "comment must contain one exact /<label>, /rerun-test <test-file>, /clear-labels, "
            "or /rerun-failed-ci command"
        )
    return None


def _positive_int(value, name):
    # [FUNCTION DOC] HELPER: Returns the value if it is exactly an integer type (booleans rejected)
    # greater than zero; otherwise raises a command error saying the named item must be a positive
    # integer.
    if type(value) is not int or value <= 0:
        raise CommentCommandError(f"{name} must be a positive integer")
    return value


def _validate_workflow_run_url(value):
    # [FUNCTION DOC] HELPER: Returns the value if it is a string fully matching the repository's
    # GitHub Actions run page URL shape (positive numeric run id); else raises a command error.
    if not isinstance(value, str) or WORKFLOW_RUN_URL_PATTERN.fullmatch(value) is None:
        raise CommentCommandError("workflow run URL is invalid")
    return value


def parse_event(event):
    # [FUNCTION DOC] Validates the issue comment event payload and extracts the request. Requires:
    # action is created; the repository id and full name match the expected repository; the issue
    # is attached to a pull request and has a positive number; the comment has a user of type
    # user (not a bot) with positive id and non-empty login; the comment's author association is in
    # the known set; the sender id equals the comment author id. Returns a five-tuple of pull
    # number, author id, author login, author association and the parsed command request (none if
    # the comment is not a command). Raises a command error for any violation, including those from
    # command parsing.
    if not isinstance(event, dict) or event.get("action") != "created":
        raise CommentCommandError("event must be issue_comment.created")

    repository = event.get("repository")
    if not isinstance(repository, dict):
        raise CommentCommandError("event repository is missing")
    if repository.get("id") != REPOSITORY_ID or repository.get("full_name") != REPOSITORY:
        raise CommentCommandError("event repository does not match radixark/miles")

    issue = event.get("issue")
    if not isinstance(issue, dict) or not isinstance(issue.get("pull_request"), dict):
        raise CommentCommandError("comment is not attached to a pull request")
    pull_number = _positive_int(issue.get("number"), "pull request number")

    comment = event.get("comment")
    if not isinstance(comment, dict) or not isinstance(comment.get("user"), dict):
        raise CommentCommandError("comment author is missing")
    if comment["user"].get("type") != "User":
        raise CommentCommandError("comment author must be a human GitHub user")
    actor_id = _positive_int(comment["user"].get("id"), "comment author ID")
    actor_login = comment["user"].get("login")
    if not isinstance(actor_login, str) or not actor_login:
        raise CommentCommandError("comment author login is missing")
    # GitHub stamps the association at comment time; an unknown value fails
    # loudly rather than defaulting into or out of any policy tier.
    author_association = comment.get("author_association")
    if not isinstance(author_association, str) or author_association not in AUTHOR_ASSOCIATIONS:
        raise CommentCommandError("comment author association is invalid")

    sender = event.get("sender")
    if not isinstance(sender, dict) or sender.get("id") != actor_id:
        raise CommentCommandError("event sender does not match the comment author")

    return pull_number, actor_id, actor_login, author_association, parse_command(comment.get("body"))


class GitHubAPI:
    # [CLASS DOC]
    # Minimal GitHub REST client for the single target repository using the standard library.
    # Every method performs network I/O with a fifteen second timeout and converts failures into
    # command errors; the token is only sent as a bearer header and never appears in messages.
    def __init__(self, token):
        # [FUNCTION DOC] Stores the API token; raises a command error if the token is empty or none.
        if not token:
            raise CommentCommandError("GitHub API token is missing")
        self.token = token

    def _request(self, path, *, method="GET", payload=None, expected_statuses=None, expect_json=True):
        # [FUNCTION DOC] HELPER: Performs one API call against the API root with the given path,
        # method (default get) and optional JSON payload (compact encoding), sending the standard
        # accept, authorization, content type and API version headers. If expected statuses are
        # given, the response status must be one of them, else a command error. When JSON is
        # expected (default) returns the decoded body; when not, the body must be blank and none is
        # returned. HTTP errors become a command error naming method, path and status, plus the
        # accepted-permissions header if present; URL errors and timeouts and invalid UTF-8/JSON
        # become command errors with generic messages.
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{API_ROOT}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                if expected_statuses is not None and response.status not in expected_statuses:
                    expected = " or ".join(str(status) for status in expected_statuses)
                    raise CommentCommandError(f"GitHub API returned HTTP {response.status}; expected {expected}")
                body = response.read()
                if not expect_json:
                    if body.strip():
                        raise CommentCommandError("GitHub API returned an unexpected response body")
                    return None
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as error:
            # Name the request so a permission failure points at one call; the
            # token never appears here, and GitHub's accepted-permissions header
            # states which installation permission that call needed.
            detail = f"GitHub API returned HTTP {error.code} for {method} {path}"
            accepted = error.headers.get("X-Accepted-GitHub-Permissions")
            if accepted:
                detail = f"{detail}; accepted permissions: {accepted}"
            raise CommentCommandError(detail) from error
        except urllib.error.URLError as error:
            raise CommentCommandError(f"GitHub API request failed: {error.reason}") from error
        except TimeoutError as error:
            raise CommentCommandError("GitHub API request timed out") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CommentCommandError("GitHub API returned invalid JSON") from error

    def get_pull(self, pull_number):
        # [FUNCTION DOC] Fetches and returns the pull request object for the number.
        return self._request(f"/repos/{REPOSITORY}/pulls/{pull_number}")

    def get_permission(self, actor_login):
        # [FUNCTION DOC] Fetches and returns the collaborator permission object for the login
        # (percent-encoded into the path), containing the permission level and the user.
        encoded_login = urllib.parse.quote(actor_login, safe="")
        return self._request(f"/repos/{REPOSITORY}/collaborators/{encoded_login}/permission")

    def add_label(self, pull_number, label):
        # [FUNCTION DOC] Adds one label to the issue/pull request via a post; returns the resulting
        # full label list from the API. Mutates GitHub state.
        return self._request(
            f"/repos/{REPOSITORY}/issues/{pull_number}/labels",
            method="POST",
            payload={"labels": [label]},
        )

    def remove_label(self, pull_number, label):
        # [FUNCTION DOC] Removes one label (percent-encoded in the path) via a delete; returns the
        # remaining label list from the API. Mutates GitHub state.
        encoded_label = urllib.parse.quote(label, safe="")
        return self._request(
            f"/repos/{REPOSITORY}/issues/{pull_number}/labels/{encoded_label}",
            method="DELETE",
        )

    def list_workflow_runs(self, workflow_file, head_sha):
        # [FUNCTION DOC] Lists all pull-request-event runs of a workflow file for a head sha, paging
        # up to ten pages of one hundred. Validates every page: the response is an object, the total
        # count is an integer between zero and one thousand and constant across pages, runs is a
        # list. Returns the accumulated runs once their count reaches the total (raising if it
        # exceeds it). Raises a command error for an empty page before reaching the total, or if
        # the pages are exhausted (the thousand-run API limit).
        encoded_workflow = urllib.parse.quote(workflow_file, safe="")
        runs = []
        total_count = None
        for page in range(1, 11):
            query = urllib.parse.urlencode(
                {
                    "event": "pull_request",
                    "head_sha": head_sha,
                    "page": page,
                    "per_page": 100,
                }
            )
            result = self._request(f"/repos/{REPOSITORY}/actions/workflows/{encoded_workflow}/runs?{query}")
            if not isinstance(result, dict):
                raise CommentCommandError("GitHub API returned an invalid workflow-run page")
            page_total = result.get("total_count")
            page_runs = result.get("workflow_runs")
            if type(page_total) is not int or page_total < 0 or page_total > 1000:
                raise CommentCommandError("GitHub API returned an invalid workflow-run count")
            if not isinstance(page_runs, list):
                raise CommentCommandError("GitHub API returned invalid workflow runs")
            if total_count is None:
                total_count = page_total
            elif page_total != total_count:
                raise CommentCommandError("workflow-run count changed during pagination")
            runs.extend(page_runs)
            if len(runs) >= total_count:
                if len(runs) != total_count:
                    raise CommentCommandError("GitHub API returned too many workflow runs")
                return runs
            if not page_runs:
                raise CommentCommandError("GitHub API returned an incomplete workflow-run listing")
        raise CommentCommandError("workflow-run listing exceeded the 1000-run API limit")

    def list_pulls_for_head(self, owner_login, head_ref):
        # [FUNCTION DOC] Lists pull requests of any state whose head is owner:branch, paging one
        # hundred at a time up to ten pages; returns the list, stopping at the first short page.
        # Raises a command error if a page is not a list or all ten pages are full.
        pulls = []
        for page in range(1, 11):
            query = urllib.parse.urlencode(
                {
                    "head": f"{owner_login}:{head_ref}",
                    "page": page,
                    "per_page": 100,
                    "state": "all",
                }
            )
            result = self._request(f"/repos/{REPOSITORY}/pulls?{query}")
            if not isinstance(result, list):
                raise CommentCommandError("GitHub API returned an invalid head pull-request page")
            pulls.extend(result)
            if len(result) < 100:
                return pulls
        raise CommentCommandError("head pull-request listing exceeded the 1000-result limit")

    def rerun_failed_jobs(self, run_id):
        # [FUNCTION DOC] Requests a rerun of the failed jobs of a workflow run via a post; the API
        # must answer status 201 with an empty body. Returns nothing. Starts CI jobs.
        self._request(
            f"/repos/{REPOSITORY}/actions/runs/{run_id}/rerun-failed-jobs",
            method="POST",
            expected_statuses=(201,),
            expect_json=False,
        )

    def create_workflow_dispatch(self, workflow_file, ref, inputs):
        # [FUNCTION DOC] Dispatches a workflow file on a ref with the given input mapping, asking
        # for run details in the response (must be status 200). Validates that the answer is an
        # object with a positive workflow run id, a run page URL of the expected shape, and
        # consistent API and page URLs for that run id. Returns the run page URL. Starts a run.
        encoded_workflow = urllib.parse.quote(workflow_file, safe="")
        result = self._request(
            f"/repos/{REPOSITORY}/actions/workflows/{encoded_workflow}/dispatches",
            method="POST",
            payload={"ref": ref, "inputs": inputs, "return_run_details": True},
            expected_statuses=(200,),
        )
        if not isinstance(result, dict):
            raise CommentCommandError("GitHub API returned invalid workflow dispatch details")
        run_id = _positive_int(result.get("workflow_run_id"), "workflow run ID")
        html_url = _validate_workflow_run_url(result.get("html_url"))
        if (
            result.get("run_url") != f"{API_ROOT}/repos/{REPOSITORY}/actions/runs/{run_id}"
            or html_url != f"https://github.com/{REPOSITORY}/actions/runs/{run_id}"
        ):
            raise CommentCommandError("GitHub API returned mismatched workflow dispatch details")
        return html_url

    def add_comment_reaction(self, comment_id, content):
        # [FUNCTION DOC] Adds a reaction (for example thumbs up) to an issue comment; status must be
        # 200 or 201 and the response must echo the same reaction content, else a command error.
        # Returns nothing.
        result = self._request(
            f"/repos/{REPOSITORY}/issues/comments/{comment_id}/reactions",
            method="POST",
            payload={"content": content},
            expected_statuses=(200, 201),
        )
        if not isinstance(result, dict) or result.get("content") != content:
            raise CommentCommandError("GitHub API did not confirm the comment reaction")

    def create_issue_comment(self, pull_number, body):
        # [FUNCTION DOC] Posts a new comment on the pull request (status must be 201) and checks the
        # returned body equals the sent body. Returns the new comment's positive integer id.
        result = self._request(
            f"/repos/{REPOSITORY}/issues/{pull_number}/comments",
            method="POST",
            payload={"body": body},
            expected_statuses=(201,),
        )
        if not isinstance(result, dict) or result.get("body") != body:
            raise CommentCommandError("GitHub API did not confirm the workflow reply")
        return _positive_int(result.get("id"), "comment ID")

    def update_issue_comment(self, comment_id, body):
        # [FUNCTION DOC] Replaces a comment's body via a patch (status must be 200) and checks the
        # returned body equals the sent body, else a command error. Returns nothing.
        result = self._request(
            f"/repos/{REPOSITORY}/issues/comments/{comment_id}",
            method="PATCH",
            payload={"body": body},
            expected_statuses=(200,),
        )
        if not isinstance(result, dict) or result.get("body") != body:
            raise CommentCommandError("GitHub API did not confirm the status update")

    def get_workflow_run(self, run_id):
        # [FUNCTION DOC] Fetches and returns the workflow run object for the id.
        return self._request(f"/repos/{REPOSITORY}/actions/runs/{run_id}")


def _validate_live_pull(pull, pull_number):
    # [FUNCTION DOC] HELPER: Validates a pull request object fetched live. Requires the matching
    # number, open state, a base repository matching the target repository id and name, a head
    # repository with positive id and an owner with non-empty login, a non-empty head branch name,
    # a list of label objects each with a string name, and a head sha of forty lowercase hex
    # characters. Raises a command error naming the missing or invalid part. Returns a five-tuple:
    # frozen set of label names, head sha, head repository id, head owner login, head branch.
    if not isinstance(pull, dict) or pull.get("number") != pull_number:
        raise CommentCommandError("GitHub API returned the wrong pull request")
    if pull.get("state") != "open":
        raise CommentCommandError("pull request is not open")

    base = pull.get("base")
    base_repository = base.get("repo") if isinstance(base, dict) else None
    if not isinstance(base_repository, dict):
        raise CommentCommandError("pull request base repository is missing")
    if base_repository.get("id") != REPOSITORY_ID or base_repository.get("full_name") != REPOSITORY:
        raise CommentCommandError("pull request base repository does not match radixark/miles")

    head = pull.get("head")
    head_repository = head.get("repo") if isinstance(head, dict) else None
    if not isinstance(head_repository, dict):
        raise CommentCommandError("pull request head repository is missing")
    head_repository_id = _positive_int(head_repository.get("id"), "pull request head repository ID")
    head_repository_owner = head_repository.get("owner")
    head_owner_login = head_repository_owner.get("login") if isinstance(head_repository_owner, dict) else None
    if not isinstance(head_owner_login, str) or not head_owner_login:
        raise CommentCommandError("pull request head repository owner is missing")
    head_ref = head.get("ref") if isinstance(head, dict) else None
    if not isinstance(head_ref, str) or not head_ref:
        raise CommentCommandError("pull request head ref is missing")

    labels = pull.get("labels")
    if not isinstance(labels, list) or any(not isinstance(item, dict) for item in labels):
        raise CommentCommandError("pull request labels are invalid")
    names = []
    for item in labels:
        name = item.get("name")
        if not isinstance(name, str):
            raise CommentCommandError("pull request label name is invalid")
        names.append(name)

    head_sha = head.get("sha") if isinstance(head, dict) else None
    if not isinstance(head_sha, str) or SHA_PATTERN.fullmatch(head_sha) is None:
        raise CommentCommandError("pull request head SHA is invalid")

    return frozenset(names), head_sha, head_repository_id, head_owner_login, head_ref


def _pull_body(pull):
    # [FUNCTION DOC] HELPER: Returns the pull request body text, or an empty string if the body is
    # null; raises a command error if it is present but not a string.
    body = pull.get("body")
    if body is None:
        return ""
    if not isinstance(body, str):
        raise CommentCommandError("pull request body is invalid")
    return body


def _command_spec(request):
    # [FUNCTION DOC] HELPER: Looks up the registry entry for a parsed request by its exact type.
    # Raises a command error if the request is none (no recognised command) or its type is not
    # registered. Returns the CommandSpec.
    if request is None:
        raise CommentCommandError("comment does not contain a recognized command")
    spec = COMMAND_REGISTRY.get(type(request))
    if spec is None:
        raise CommentCommandError("request type is not registered")
    return spec


def resolve_policy(event, policy):
    # [FUNCTION DOC] Parses the event, finds the command's registry entry, and applies the loaded
    # policy. Returns a nine-tuple: pull number, author id, author login, author association,
    # request, spec, allowed repository permissions, allowed user ids (empty when the command may
    # not grant by user id) and allowed author associations. For commands with a resource
    # allowlist (add-label), the requested resource must be in the policy list, else a command
    # error saying the label is not exposed. A policy missing the needed keys gives a command error.
    pull_number, actor_id, actor_login, author_association, request = parse_event(event)
    spec = _command_spec(request)
    try:
        command_policy = policy["commands"][spec.policy_key]
        group = policy["groups"][command_policy["group"]]
        allowed_permissions = group["repository_permissions"]
        allowed_user_ids = group["user_ids"] if spec.allows_user_ids else frozenset()
        allowed_author_associations = group["author_associations"]
    except (KeyError, TypeError) as error:
        raise CommentCommandError("resolved command policy is invalid") from error
    if spec.resource_key is not None:
        resource = spec.resource_getter(request)
        if resource not in command_policy[spec.resource_key]:
            raise CommentCommandError("requested label is not exposed by policy")
    return (
        pull_number,
        actor_id,
        actor_login,
        author_association,
        request,
        spec,
        allowed_permissions,
        allowed_user_ids,
        allowed_author_associations,
    )


def require_permission(api, actor_id, actor_login, allowed_permissions):
    # [FUNCTION DOC] Verifies, via a live API lookup by login, that the actor holds an allowed
    # repository permission. Checks the response is an object, its user id is a positive integer
    # equal to the actor id (guards against login reuse), the permission is a string and is in the
    # allowed set. Raises a command error otherwise (including the not-authorized denial).
    permission_result = api.get_permission(actor_login)
    if not isinstance(permission_result, dict):
        raise CommentCommandError("GitHub API returned an invalid repository permission")
    permission_user = permission_result.get("user")
    permission_user_id = permission_user.get("id") if isinstance(permission_user, dict) else None
    if type(permission_user_id) is not int or permission_user_id <= 0:
        raise CommentCommandError("GitHub API returned an invalid repository permission identity")
    if permission_user_id != actor_id:
        raise CommentCommandError("repository permission identity does not match the comment author")
    permission = permission_result.get("permission")
    if not isinstance(permission, str):
        raise CommentCommandError("GitHub API returned an invalid repository permission")
    if permission not in allowed_permissions:
        raise CommentCommandError("comment author is not authorized for the requested operation")


def require_access(
    api,
    actor_id,
    actor_login,
    author_association,
    allowed_permissions,
    allowed_user_ids,
    allowed_author_associations,
):
    # [FUNCTION DOC] Authorization gate; returns nothing on success and raises a command error on
    # denial. Access is granted, in order, if the actor id is in the allowed user ids, or the
    # comment's author association is in the allowed associations (no live lookup), or the live
    # repository permission check passes.
    if actor_id in allowed_user_ids:
        return
    # The association GitHub stamped on the comment admits prior contributors
    # without a live-permission lookup; anyone else needs write or admin.
    if author_association in allowed_author_associations:
        return
    require_permission(api, actor_id, actor_login, allowed_permissions)


def _is_ci_control_label(label):
    # [FUNCTION DOC] HELPER: True if the label name begins with the run-ci prefix or the run-on
    # dash prefix, or is exactly one of the clearable names (nightly, bypass fast-fail).
    # KNOWN ISSUE: the run-ci prefix test has no trailing dash, so it also matches names such as
    # run-ciXYZ that the label allowlist shape would not accept.
    return label.startswith(("run-ci", "run-on-")) or label in CLEAR_EXACT_LABELS


def _latest_failed_run(
    runs,
    workflow_path,
    pull_number,
    head_sha,
    head_repository_id,
    head_ref,
    allow_empty_pull_requests,
):
    # [FUNCTION DOC] HELPER: Given the list of runs of one workflow for a head sha, returns the id
    # of the newest run if it is a completed failure, else none. Each run is validated strictly
    # (object, positive unique id and run number, expected workflow path, pull request event, same
    # head sha, same head branch, head repository id equal to the given one, list of pull request
    # objects with positive numbers); any mismatch raises a command error. A run counts as
    # associated with the pull request if its pull request list contains the number, or, only when
    # allow_empty_pull_requests is true (fork pull requests), if that list is empty. The newest is
    # chosen by highest run number, ties broken by id.
    associated = []
    seen_ids = set()
    for run in runs:
        if not isinstance(run, dict):
            raise CommentCommandError("GitHub API returned an invalid workflow run")
        run_id = _positive_int(run.get("id"), "workflow run ID")
        if run_id in seen_ids:
            raise CommentCommandError("GitHub API returned a duplicate workflow run")
        seen_ids.add(run_id)
        run_number = _positive_int(run.get("run_number"), "workflow run number")
        if run.get("path") != workflow_path:
            raise CommentCommandError("GitHub API returned a workflow run from an unexpected path")
        if run.get("event") != "pull_request" or run.get("head_sha") != head_sha:
            raise CommentCommandError("GitHub API returned a workflow run from an unexpected event or SHA")
        if run.get("head_branch") != head_ref:
            raise CommentCommandError("GitHub API returned a workflow run from an unexpected head ref")
        run_head_repository = run.get("head_repository")
        if not isinstance(run_head_repository, dict) or run_head_repository.get("id") != head_repository_id:
            raise CommentCommandError("GitHub API returned a workflow run from an unexpected repository")
        pull_requests = run.get("pull_requests")
        if not isinstance(pull_requests, list):
            raise CommentCommandError("GitHub API returned invalid workflow-run pull requests")
        pull_numbers = set()
        for pull_request in pull_requests:
            if not isinstance(pull_request, dict):
                raise CommentCommandError("GitHub API returned an invalid workflow-run pull request")
            pull_numbers.add(_positive_int(pull_request.get("number"), "workflow-run pull number"))
        if pull_number in pull_numbers or (not pull_numbers and allow_empty_pull_requests):
            associated.append((run_number, run_id, run))

    if not associated:
        return None
    _, run_id, latest = max(associated, key=lambda item: (item[0], item[1]))
    if latest.get("status") == "completed" and latest.get("conclusion") == "failure":
        return run_id
    return None


def _require_unique_head_pull(
    api,
    pull_number,
    head_sha,
    head_repository_id,
    head_owner_login,
    head_ref,
):
    # [FUNCTION DOC] HELPER: For fork pull requests, confirms that the fork branch (owner and
    # branch) identifies exactly one pull request in the target repository and that it is the
    # current one: same number, open, base repository matches the target, head repository id,
    # owner login, branch and sha all equal the expected values. Raises a command error otherwise;
    # returns nothing.
    pulls = api.list_pulls_for_head(head_owner_login, head_ref)
    if len(pulls) != 1:
        raise CommentCommandError("fork head does not identify exactly one pull request")
    pull = pulls[0]
    if not isinstance(pull, dict) or pull.get("number") != pull_number or pull.get("state") != "open":
        raise CommentCommandError("fork head identifies an unexpected pull request")
    base = pull.get("base")
    base_repository = base.get("repo") if isinstance(base, dict) else None
    head = pull.get("head")
    head_repository = head.get("repo") if isinstance(head, dict) else None
    head_repository_owner = head_repository.get("owner") if isinstance(head_repository, dict) else None
    if (
        not isinstance(base_repository, dict)
        or base_repository.get("id") != REPOSITORY_ID
        or base_repository.get("full_name") != REPOSITORY
        or not isinstance(head_repository, dict)
        or head_repository.get("id") != head_repository_id
        or not isinstance(head_repository_owner, dict)
        or head_repository_owner.get("login") != head_owner_login
        or head.get("ref") != head_ref
        or head.get("sha") != head_sha
    ):
        raise CommentCommandError("fork head pull-request identity does not match the live pull request")


def _handle_rerun_failed_ci(context, request):
    # [FUNCTION DOC] HELPER: Handler for the rerun-failed-ci command. Raises a command error if the
    # request is not exactly the rerun type. For fork pull requests first verifies the unique head
    # pull request. For each of the three rerun workflows finds the latest failed run for the head
    # sha; the candidates are sorted by run id. If there are no candidates, still requires the
    # actor's access (so denial is not masked) and reports that no failed runs exist. For each
    # candidate re-fetches and revalidates the live pull request (head sha, repository, owner and
    # branch must be unchanged), re-verifies the fork identity, re-lists runs and checks the
    # latest failed run is still the same, requires access, then requests the rerun of failed
    # jobs (errors are wrapped with the run id). Returns an audit mapping: actor id, decision
    # (rerun requested or no failed runs), head sha, pull number and the list of rerun run ids.
    # Side effect: restarts CI jobs.
    if type(request) is not RerunFailedCI:
        raise CommentCommandError("rerun handler received the wrong request type")
    (
        api,
        pull_number,
        actor_id,
        actor_login,
        author_association,
        allowed_permissions,
        allowed_user_ids,
        allowed_author_associations,
        _,
        head_sha,
        head_repository_id,
        head_owner_login,
        head_ref,
        _,
    ) = context
    is_fork = head_repository_id != REPOSITORY_ID
    if is_fork:
        _require_unique_head_pull(
            api,
            pull_number,
            head_sha,
            head_repository_id,
            head_owner_login,
            head_ref,
        )
    candidates = []
    for workflow_file, workflow_path in RERUN_WORKFLOWS:
        runs = api.list_workflow_runs(workflow_file, head_sha)
        run_id = _latest_failed_run(
            runs,
            workflow_path,
            pull_number,
            head_sha,
            head_repository_id,
            head_ref,
            is_fork,
        )
        if run_id is not None:
            candidates.append((run_id, workflow_file, workflow_path))

    candidates.sort()
    if not candidates:
        require_access(
            api,
            actor_id,
            actor_login,
            author_association,
            allowed_permissions,
            allowed_user_ids,
            allowed_author_associations,
        )
    for run_id, workflow_file, workflow_path in candidates:
        current_pull = api.get_pull(pull_number)
        (
            _,
            current_head_sha,
            current_head_repository_id,
            current_head_owner_login,
            current_head_ref,
        ) = _validate_live_pull(current_pull, pull_number)
        if (
            current_head_sha != head_sha
            or current_head_repository_id != head_repository_id
            or current_head_owner_login != head_owner_login
            or current_head_ref != head_ref
        ):
            raise CommentCommandError("pull request head changed before rerun")
        if is_fork:
            _require_unique_head_pull(
                api,
                pull_number,
                head_sha,
                head_repository_id,
                head_owner_login,
                head_ref,
            )
        current_run_id = _latest_failed_run(
            api.list_workflow_runs(workflow_file, head_sha),
            workflow_path,
            pull_number,
            head_sha,
            head_repository_id,
            head_ref,
            is_fork,
        )
        if current_run_id != run_id:
            raise CommentCommandError("latest workflow-run state changed before rerun")
        require_access(
            api,
            actor_id,
            actor_login,
            author_association,
            allowed_permissions,
            allowed_user_ids,
            allowed_author_associations,
        )
        try:
            api.rerun_failed_jobs(run_id)
        except CommentCommandError as error:
            raise CommentCommandError(f"could not rerun failed jobs for workflow run {run_id}: {error}") from error

    return {
        "actor_id": actor_id,
        "decision": "ALLOW_RERUN_REQUESTED" if candidates else "ALLOW_NO_FAILED_RUNS",
        "head_sha": head_sha,
        "pull_number": pull_number,
        "workflow_run_ids": [candidate[0] for candidate in candidates],
    }


def _handle_add_label(context, request):
    # [FUNCTION DOC] HELPER: Handler for the add-label command. Checks the request type and the
    # actor's access. If the pull request already has the label, no change is made; otherwise adds
    # it and verifies the returned label list contains it (else command error). Returns an audit
    # mapping: actor id, decision (already present or added), label and pull number.
    if type(request) is not AddLabel:
        raise CommentCommandError("add-label handler received the wrong request type")
    require_access(
        context.api,
        context.actor_id,
        context.actor_login,
        context.author_association,
        context.allowed_permissions,
        context.allowed_user_ids,
        context.allowed_author_associations,
    )
    label = request.label
    if label in context.current_labels:
        decision = "ALLOW_ALREADY_PRESENT"
    else:
        result = context.api.add_label(context.pull_number, label)
        if not isinstance(result, list) or label not in {
            item.get("name") for item in result if isinstance(item, dict)
        }:
            raise CommentCommandError("GitHub API did not confirm the requested label")
        decision = "ALLOW_ADDED"
    return {
        "actor_id": context.actor_id,
        "decision": decision,
        "label": label,
        "pull_number": context.pull_number,
    }


def _handle_clear_labels(context, request):
    # [FUNCTION DOC] HELPER: Handler for the clear-labels command. Checks request type and access,
    # then removes, in sorted order, every current label that is a CI control label. After each
    # removal the API's returned list must be a list of named objects not containing the removed
    # label (else command error naming it); the returned list becomes the remaining labels. At the
    # end no CI control label may remain. Returns an audit mapping: actor id, decision (cleared or
    # already clear), the removed labels list and pull number. Mutates GitHub labels.
    if type(request) is not ClearLabels:
        raise CommentCommandError("clear-labels handler received the wrong request type")
    require_access(
        context.api,
        context.actor_id,
        context.actor_login,
        context.author_association,
        context.allowed_permissions,
        context.allowed_user_ids,
        context.allowed_author_associations,
    )
    labels_to_remove = sorted(label for label in context.current_labels if _is_ci_control_label(label))
    remaining_labels = context.current_labels
    for label in labels_to_remove:
        try:
            result = context.api.remove_label(context.pull_number, label)
        except CommentCommandError as error:
            raise CommentCommandError(f"could not remove CI label {label}: {error}") from error
        if (
            not isinstance(result, list)
            or any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in result)
            or label in {item["name"] for item in result}
        ):
            raise CommentCommandError(f"GitHub API did not confirm removal of {label}")
        remaining_labels = frozenset(item["name"] for item in result)
    if any(_is_ci_control_label(label) for label in remaining_labels):
        raise CommentCommandError("GitHub API did not confirm that all CI labels were removed")
    return {
        "actor_id": context.actor_id,
        "decision": "ALLOW_CLEARED" if labels_to_remove else "ALLOW_ALREADY_CLEAR",
        "labels": labels_to_remove,
        "pull_number": context.pull_number,
    }


def _pr_body_pins(body):
    # [FUNCTION DOC] HELPER: Extracts dependency pins from the pull request body. For each of the
    # four known pins (CI image tag, Megatron PR, Megatron Bridge PR, SGLang PR) searches for a
    # line starting with the pin's key and colon followed by one non-space token. The token must
    # fully match that pin's allowed value shape, else a command error naming the pin. Returns a
    # mapping from dispatch input name to value, containing only pins present.
    pins = {}
    for input_name, body_pattern, value_pattern in PR_BODY_PINS:
        match = body_pattern.search(body)
        if match is None:
            continue
        value = match.group(1)
        if value_pattern.fullmatch(value) is None:
            raise CommentCommandError(
                f"PR body pin {input_name.replace('_', '-')} has an unsupported value: {value!r}"
            )
        pins[input_name] = value
    return pins


def _handle_run_test_file(context, request):
    # [FUNCTION DOC] HELPER: Handler for the rerun-test command. Checks request type. If the head
    # repository is a fork, requires the unique open head pull request identity (the policy tier is
    # otherwise the only gate; the dispatched workflow withholds secrets from fork heads). Requires
    # access. Dispatches the run-ci-file workflow on the main ref with inputs: pull number as text,
    # head sha, test file, plus any pins from the pull request body. Returns an audit mapping:
    # actor id, decision (file run dispatched), head sha, pull number, test file and the new run URL.
    if type(request) is not RunTestFile:
        raise CommentCommandError("rerun-test handler received the wrong request type")
    if context.head_repository_id != REPOSITORY_ID:
        # A fork head is the normal shape of a non-write contributor's PR —
        # same-repository branches require write to push — so the policy tier
        # is the whole gate and fork adds no extra approval. The head identity
        # is pinned to its unique open PR, and the dispatched workflow
        # withholds repository secrets from fork heads.
        _require_unique_head_pull(
            context.api,
            context.pull_number,
            context.head_sha,
            context.head_repository_id,
            context.head_owner_login,
            context.head_ref,
        )
    require_access(
        context.api,
        context.actor_id,
        context.actor_login,
        context.author_association,
        context.allowed_permissions,
        context.allowed_user_ids,
        context.allowed_author_associations,
    )
    inputs = {
        "pull_number": str(context.pull_number),
        "head_sha": context.head_sha,
        "test_file": request.test_file,
    }
    inputs.update(_pr_body_pins(context.body))
    workflow_run_url = context.api.create_workflow_dispatch(
        RUN_FILE_WORKFLOW,
        RUN_FILE_WORKFLOW_REF,
        inputs,
    )
    return {
        "actor_id": context.actor_id,
        "decision": "ALLOW_FILE_RUN_DISPATCHED",
        "head_sha": context.head_sha,
        "pull_number": context.pull_number,
        "test_file": request.test_file,
        "workflow_run_url": workflow_run_url,
    }


def _add_label_resource(request):
    # [FUNCTION DOC] HELPER: Resource getter and audit value producer for add-label: verifies the
    # request is an add-label request (else command error) and returns its label name.
    if type(request) is not AddLabel:
        raise CommentCommandError("add-label resource received the wrong request type")
    return request.label


def _clear_command_value(request):
    # [FUNCTION DOC] HELPER: Audit value producer for clear-labels: verifies the request type (else
    # command error) and returns the slash-prefixed clear-labels command text.
    if type(request) is not ClearLabels:
        raise CommentCommandError("clear-labels audit received the wrong request type")
    return f"/{CLEAR_COMMAND}"


def _rerun_command_value(request):
    # [FUNCTION DOC] HELPER: Audit value producer for rerun-failed-ci: verifies the request type
    # (else command error) and returns the slash-prefixed rerun-failed-ci command text.
    if type(request) is not RerunFailedCI:
        raise CommentCommandError("rerun audit received the wrong request type")
    return f"/{RERUN_COMMAND}"


def _run_file_value(request):
    # [FUNCTION DOC] HELPER: Audit value producer for rerun-test: verifies the request type (else
    # command error) and returns the test file path.
    if type(request) is not RunTestFile:
        raise CommentCommandError("rerun-test audit received the wrong request type")
    return request.test_file


COMMAND_REGISTRY = {
    AddLabel: CommandSpec(
        "add_label",
        "issues",
        _handle_add_label,
        "allowed_labels",
        _validate_labels,
        _add_label_resource,
        True,
        "label",
        _add_label_resource,
        "none",
    ),
    ClearLabels: CommandSpec(
        "clear_labels",
        "issues",
        _handle_clear_labels,
        None,
        None,
        None,
        False,
        "command",
        _clear_command_value,
        "none",
    ),
    RerunFailedCI: CommandSpec(
        "rerun_failed_ci",
        "actions",
        _handle_rerun_failed_ci,
        None,
        None,
        None,
        True,
        "command",
        _rerun_command_value,
        "none",
    ),
    RunTestFile: CommandSpec(
        "run_test_file",
        "actions",
        _handle_run_test_file,
        None,
        None,
        None,
        True,
        "test_file",
        _run_file_value,
        "+1",
    ),
}


def process_event(event, policy, api):
    # [FUNCTION DOC] Full processing of a comment event: resolves the policy for the event, fetches
    # and validates the live pull request, builds the command context (including the pull request
    # body) and invokes the command's registered handler. Returns the handler's audit mapping.
    # Raises a command error from any stage.
    (
        pull_number,
        actor_id,
        actor_login,
        author_association,
        request,
        spec,
        allowed_permissions,
        allowed_user_ids,
        allowed_author_associations,
    ) = resolve_policy(event, policy)
    pull = api.get_pull(pull_number)
    (
        current_labels,
        head_sha,
        head_repository_id,
        head_owner_login,
        head_ref,
    ) = _validate_live_pull(pull, pull_number)
    context = CommandContext(
        api,
        pull_number,
        actor_id,
        actor_login,
        author_association,
        allowed_permissions,
        allowed_user_ids,
        allowed_author_associations,
        current_labels,
        head_sha,
        head_repository_id,
        head_owner_login,
        head_ref,
        _pull_body(pull),
    )
    return spec.handler(context, request)


def authorize_policy(event, policy, api):
    # [FUNCTION DOC] Preflight authorization: resolves the policy for the event and runs the access
    # gate without touching the pull request. Returns a four-tuple of pull number, author id,
    # request and spec. Raises a command error on denial or invalid input.
    (
        pull_number,
        actor_id,
        actor_login,
        author_association,
        request,
        spec,
        allowed_permissions,
        allowed_user_ids,
        allowed_author_associations,
    ) = resolve_policy(event, policy)
    require_access(
        api,
        actor_id,
        actor_login,
        author_association,
        allowed_permissions,
        allowed_user_ids,
        allowed_author_associations,
    )
    return pull_number, actor_id, request, spec


def acknowledge_event(event, api):
    # [FUNCTION DOC] Adds the command's success reaction to the triggering comment. Parses the
    # event; raises a command error if the command defines no success reaction (reaction value
    # none) or the comment id is not a positive integer. Returns an audit mapping: actor id,
    # decision (success reaction confirmed), pull number and reaction. Side effect: posts a
    # reaction.
    pull_number, actor_id, _, _, request = parse_event(event)
    spec = _command_spec(request)
    if spec.success_reaction == "none":
        raise CommentCommandError("command does not define a success reaction")
    comment_id = _positive_int(event["comment"].get("id"), "comment ID")
    api.add_comment_reaction(comment_id, spec.success_reaction)
    return {
        "actor_id": actor_id,
        "decision": "ALLOW_SUCCESS_REACTION_CONFIRMED",
        "pull_number": pull_number,
        "reaction": spec.success_reaction,
    }


def _file_run_marker(run_id):
    # [FUNCTION DOC] HELPER: Returns an HTML comment string embedding the workflow run id, appended
    # to status comments so the owning run is visible in the comment source.
    # Keep the owning workflow run visible in the comment source; the exact
    # comment ID passed between jobs prevents concurrent runs from colliding.
    return f"<!-- rerun-test-run:{run_id} -->"


def _format_duration(seconds):
    # [FUNCTION DOC] HELPER: Formats a duration in seconds (truncated to an integer) as compact
    # text: seconds only below a minute; minutes and two-digit seconds below an hour; otherwise
    # hours, two-digit minutes and two-digit seconds. Negative input raises a command error.
    seconds = int(seconds)
    if seconds < 0:
        raise CommentCommandError("workflow run duration is negative")
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _parse_run_timestamp(value, name):
    # [FUNCTION DOC] HELPER: Parses an ISO-8601 timestamp string from a workflow run (a trailing Z
    # is treated as UTC). Returns a timezone-aware datetime. Raises a command error naming the
    # field if the value is missing or not a non-empty string, unparseable, or lacks a time zone.
    if not isinstance(value, str) or not value:
        raise CommentCommandError(f"workflow run {name} is missing")
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CommentCommandError(f"workflow run {name} is invalid") from error
    if parsed.tzinfo is None:
        raise CommentCommandError(f"workflow run {name} has no timezone")
    return parsed


def _run_elapsed(api, run_id, now):
    # [FUNCTION DOC] HELPER: Fetches the workflow run and returns the seconds (float) from its
    # start time to the given current time. Raises a command error for a non-object response or an
    # invalid start time.
    run = api.get_workflow_run(run_id)
    if not isinstance(run, dict):
        raise CommentCommandError("GitHub API returned an invalid workflow run")
    started = _parse_run_timestamp(run.get("run_started_at"), "start time")
    return (now - started).total_seconds()


# Job results this handler knows how to describe. Anything else fails loudly
# rather than being reported to a reviewer as an unknown-but-fine state.
FILE_RUN_RESULTS = frozenset({"success", "failure", "cancelled", "skipped", ""})
FILE_RUN_STATUS_MODES = frozenset({"announce", "report"})


class FileRunStatus(NamedTuple):
    # [CLASS DOC]
    # Validated inputs for reporting on a dispatched file run: pull request number, test file path,
    # workflow run id, suite name (may be empty), the result of the resolve job and the result of
    # whichever hardware job executed the file (empty if none ran).
    pull_number: int
    test_file: str
    run_id: int
    suite: str
    resolve_result: str
    execute_result: str


def _file_run_status_inputs(environ):
    # [FUNCTION DOC] HELPER: Validates the environment mapping for the file-run status modes and
    # returns a triple: mode, a FileRunStatus and a comment id (none unless the mode is report).
    # Requires: the status mode variable is announce or report; positive integer pull number and run
    # id; a test file fully matching the allowed test path shape; the suite empty or lowercase
    # hyphenated alphanumeric; the resolve, CUDA and CPU results each in the known result set. Of
    # the CUDA and CPU results, those that are neither skipped nor empty are executed; more than one
    # raises a command error; the single one (or empty) is the execute result. In report mode the
    # comment id must be a positive integer. All violations raise a command error.
    mode = environ.get("CI_COMMAND_FILE_RUN_STATUS", "")
    if mode not in FILE_RUN_STATUS_MODES:
        raise CommentCommandError("file run status mode must be announce or report")
    pull_number = _parse_positive_int(environ.get("FILE_RUN_PULL_NUMBER"), "pull request number")
    run_id = _parse_positive_int(environ.get("FILE_RUN_RUN_ID"), "workflow run ID")
    test_file = environ.get("FILE_RUN_TEST_FILE", "")
    if TEST_FILE_PATTERN.fullmatch(test_file) is None:
        raise CommentCommandError("file run test file is invalid")
    suite = environ.get("FILE_RUN_SUITE", "")
    if suite and SUITE_PATTERN.fullmatch(suite) is None:
        raise CommentCommandError("file run suite is invalid")
    results = {
        "resolve": environ.get("FILE_RUN_RESOLVE_RESULT", ""),
        "cuda": environ.get("FILE_RUN_CUDA_RESULT", ""),
        "cpu": environ.get("FILE_RUN_CPU_RESULT", ""),
    }
    for name, value in results.items():
        if value not in FILE_RUN_RESULTS:
            raise CommentCommandError(f"file run {name} result is invalid")
    # Exactly one hardware job executes the file; the other is skipped by its
    # own `if`, so the executed result is the one that is not "skipped".
    executed = [value for value in (results["cuda"], results["cpu"]) if value not in {"skipped", ""}]
    if len(executed) > 1:
        raise CommentCommandError("file run reported two executing jobs")
    comment_id = None
    if mode == "report":
        comment_id = _parse_positive_int(environ.get("FILE_RUN_COMMENT_ID"), "comment ID")
    return (
        mode,
        FileRunStatus(
            pull_number=pull_number,
            test_file=test_file,
            run_id=run_id,
            suite=suite,
            resolve_result=results["resolve"],
            execute_result=executed[0] if executed else "",
        ),
        comment_id,
    )


def _parse_positive_int(value, name):
    # [FUNCTION DOC] HELPER: Parses a text value consisting only of digits into a positive integer
    # via the positive-integer check; raises a command error naming the item if it is not a digit
    # string or is zero.
    if not isinstance(value, str) or not value.isdigit():
        raise CommentCommandError(f"{name} must be a positive integer")
    return _positive_int(int(value), name)


def _file_run_outcome(status):
    # [FUNCTION DOC] HELPER: Maps a file run status to a pair of outcome word and explanation text.
    # Cancelled if either the resolve or execute result is cancelled; failed with a "never started"
    # explanation if resolve did not succeed; passed with empty explanation if execute succeeded;
    # failed with a pointer to the workflow run if execute failed; otherwise failed with a message
    # that no execution job ran.
    """Describe the run for a reviewer: headline, and why when it is not a pass."""
    if status.resolve_result == "cancelled" or status.execute_result == "cancelled":
        return "cancelled", "The run was cancelled."
    if status.resolve_result != "success":
        return "failed", "The run never started: resolving the test file's execution plan failed."
    if status.execute_result == "success":
        return "passed", ""
    if status.execute_result == "failure":
        return "failed", "The execution job failed; inspect the workflow run for the failing step."
    return "failed", "No execution job ran: the resolved plan selected neither the CUDA nor the CPU job."


def announce_file_run(api, status):
    # [FUNCTION DOC] Posts a "running" comment on the pull request containing the test file, a link
    # to the workflow run, the UTC start time and the run marker. Returns an audit mapping: new
    # comment id, decision (file run announced), pull number, run id and test file. Side effect:
    # creates a comment.
    run_url = f"https://github.com/{REPOSITORY}/actions/runs/{status.run_id}"
    body = (
        f"⏳ `{status.test_file}` is **running** — [workflow run]({run_url})\n\n"
        f"Started at {_utc_now().strftime('%Y-%m-%d %H:%M:%S')} UTC; "
        "elapsed time and the result will be recorded here when it finishes.\n\n"
        f"{_file_run_marker(status.run_id)}"
    )
    comment_id = api.create_issue_comment(status.pull_number, body)
    return {
        "comment_id": comment_id,
        "decision": "FILE_RUN_ANNOUNCED",
        "pull_number": status.pull_number,
        "run_id": status.run_id,
        "test_file": status.test_file,
    }


def report_file_run(api, status, comment_id):
    # [FUNCTION DOC] Rewrites the previously announced comment with the final result. Computes the
    # outcome and explanation, picks an icon (check, cross or white circle for passed, failed,
    # cancelled), computes elapsed time from the run's start to now, and builds a headline with the
    # test file, outcome, optional suite, duration and workflow run link, then the explanation
    # (if any) and the run marker, separated by blank lines. Updates the comment. Returns an audit
    # mapping: comment id, decision (upper-case outcome with a file-run prefix), duration text,
    # pull number, run id and test file.
    outcome, explanation = _file_run_outcome(status)
    icon = {"passed": "✅", "failed": "❌", "cancelled": "⚪"}[outcome]
    run_url = f"https://github.com/{REPOSITORY}/actions/runs/{status.run_id}"
    elapsed = _format_duration(_run_elapsed(api, status.run_id, _utc_now()))
    where = f" on `{status.suite}`" if status.suite else ""
    lines = [f"{icon} `{status.test_file}` **{outcome}**{where} in {elapsed} — [workflow run]({run_url})"]
    if explanation:
        lines.append(explanation)
    lines.append(_file_run_marker(status.run_id))
    body = "\n\n".join(lines)
    api.update_issue_comment(comment_id, body)
    return {
        "comment_id": comment_id,
        "decision": f"FILE_RUN_{outcome.upper()}",
        "duration": elapsed,
        "pull_number": status.pull_number,
        "run_id": status.run_id,
        "test_file": status.test_file,
    }


def _utc_now():
    # [FUNCTION DOC] HELPER: Returns the current time as a timezone-aware UTC datetime.
    return datetime.datetime.now(datetime.timezone.utc)


def _write_routing(capability, success_reaction):
    # [FUNCTION DOC] HELPER: Validates that the capability is one of none, issues or actions and
    # the success reaction is none or the thumbs-up name (else command error). If the GitHub output
    # file environment variable is set, appends two lines giving the capability and the success
    # reaction for later workflow steps; if unset does nothing. A write failure raises a command
    # error. Side effect: file append.
    if capability not in {"none", "issues", "actions"}:
        raise CommentCommandError("command registry selected an unknown capability")
    if success_reaction not in {"none", "+1"}:
        raise CommentCommandError("command registry selected an unknown success reaction")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    try:
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(f"capability={capability}\n")
            output.write(f"success_reaction={success_reaction}\n")
    except OSError as error:
        raise CommentCommandError(f"cannot write GITHUB_OUTPUT: {error}") from error


def _write_file_run_comment_id(comment_id):
    # [FUNCTION DOC] HELPER: Validates the comment id as a positive integer and, if the GitHub
    # output file variable is set, appends a line giving the comment id for later steps; does
    # nothing when unset. Write failures raise a command error.
    comment_id = _positive_int(comment_id, "comment ID")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    try:
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(f"comment_id={comment_id}\n")
    except OSError as error:
        raise CommentCommandError(f"cannot write GITHUB_OUTPUT: {error}") from error


def main():
    # [FUNCTION DOC] Entry point; returns the process exit status (zero success, one for a command
    # error). Mode selection by environment: (1) if the file-run status variable is set, validate
    # its inputs, announce (and write the new comment id to the output file) or report, and print the
    # JSON audit; no event file is read. (2) Otherwise load the event JSON from the event path
    # variable. If the acknowledge variable is the word true, add the success reaction. Else if the
    # preflight variable is true: parse the event; an unrecognised comment writes routing none and
    # prints an ignore decision, returning zero; otherwise load the policy, authorize, write routing
    # outputs and print an authorization mapping (actor, pull number and the command's audit field).
    # Else full processing: load the policy and run the event handler. Results are printed as
    # compact sorted-key JSON. A command error prints a workflow error annotation and returns one.
    # Missing required environment variables raise key errors.
    try:
        # The file-run status modes report on a dispatched workflow run, not on
        # a comment event, so they are the one path that reads no event payload.
        if os.environ.get("CI_COMMAND_FILE_RUN_STATUS"):
            mode, status, comment_id = _file_run_status_inputs(os.environ)
            api = GitHubAPI(os.environ["CI_COMMAND_API_TOKEN"])
            if mode == "announce":
                result = announce_file_run(api, status)
                _write_file_run_comment_id(result["comment_id"])
            else:
                result = report_file_run(api, status, comment_id)
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0
        event = load_json(os.environ["GITHUB_EVENT_PATH"])
        if os.environ.get("CI_COMMAND_ACKNOWLEDGE") == "true":
            api = GitHubAPI(os.environ["CI_COMMAND_API_TOKEN"])
            result = acknowledge_event(event, api)
        elif os.environ.get("CI_COMMAND_PREFLIGHT") == "true":
            pull_number, actor_id, _, _, request = parse_event(event)
            if request is None:
                _write_routing("none", "none")
                print(
                    json.dumps(
                        {
                            "actor_id": actor_id,
                            "decision": "IGNORE_UNRECOGNIZED_COMMAND",
                            "pull_number": pull_number,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                return 0
            policy = load_policy(os.environ["CI_COMMAND_POLICY_PATH"])
            api = GitHubAPI(os.environ["CI_COMMAND_API_TOKEN"])
            pull_number, actor_id, request, spec = authorize_policy(event, policy, api)
            authorization = {"actor_id": actor_id, "pull_number": pull_number}
            authorization[spec.audit_key] = spec.audit_value(request)
            _write_routing(spec.capability, spec.success_reaction)
            print(
                json.dumps(
                    authorization,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        else:
            policy = load_policy(os.environ["CI_COMMAND_POLICY_PATH"])
            api = GitHubAPI(os.environ["CI_COMMAND_API_TOKEN"])
            result = process_event(event, policy, api)
    except CommentCommandError as error:
        print(f"::error::{error}")
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
