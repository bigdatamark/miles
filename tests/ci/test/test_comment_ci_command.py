# [MODULE DOC]
# Tests for the PR comment command handler (.github/workflows/scripts/comment_ci_command.py, loaded
# by path as HANDLER) and its policy file (.github/workflows/policies/comment-command-access.json)
# and workflow. The handler parses issue comments on pull requests into one of four exact commands
# (add a run-ci/run-on/bypass label, clear CI labels, rerun failed CI, run one test file), checks
# the actor against a versioned access policy (live repository permission, explicit user ids, or
# the comment's author association depending on the command's group), re-reads the live pull
# request, and then mutates labels, reruns workflow jobs or dispatches a file-run workflow through
# a narrowly scoped GitHub API client. It also has a preflight mode that only writes fixed routing
# outputs (capability and success reaction), and a file-run status mode that announces and reports
# the dispatched file run in a pull request comment.
# These tests use a FakeAPI recording every call, builders for events, pull requests, policies and
# workflow runs, monkeypatched URL opening for the real client, and temporary policy / output files.
# Registers itself as a CPU test in suite stage-a-cpu.
import datetime
import importlib.util
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.labels import KNOWN_LABELS

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]
HANDLER_PATH = ROOT / ".github/workflows/scripts/comment_ci_command.py"
POLICY_PATH = ROOT / ".github/workflows/policies/comment-command-access.json"
WORKFLOW_PATH = ROOT / ".github/workflows/comment-ci-command.yml"


def load_module(name, path):
    # [FUNCTION DOC] HELPER: Import a Python file by path under the given module name and execute its
    # body; return the module. Unlike similar helpers it does not register the module in the module
    # table.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HANDLER = load_module("comment_ci_command", HANDLER_PATH)
ACTOR_ID = 1234
HEAD_SHA = "a" * 40
HEAD_REF = "feature/test"
WRITE_PERMISSIONS = frozenset({"write", "admin"})
RUN_FILE_BODY = "/rerun-test tests/e2e/precision/test_hf_attention_cp_relayout.py"
RUN_FILE_PATH = "tests/e2e/precision/test_hf_attention_cp_relayout.py"
WORKFLOW_RUN_ID = 987654321
WORKFLOW_RUN_API_URL = f"https://api.github.com/repos/radixark/miles/actions/runs/{WORKFLOW_RUN_ID}"
WORKFLOW_RUN_URL = f"https://github.com/radixark/miles/actions/runs/{WORKFLOW_RUN_ID}"
RUN_MARKER = f"<!-- rerun-test-run:{WORKFLOW_RUN_ID} -->"
RUN_STARTED_AT = "2026-08-23T09:25:25Z"
# 3m11s after RUN_STARTED_AT, the shape of a real 4-GPU file run.
REPORT_NOW = datetime.datetime(2026, 8, 23, 9, 28, 36, tzinfo=datetime.timezone.utc)
NEW_COMMENT_ID = 4242
EXISTING_COMMENT_ID = 777
ANNOUNCE_BODY = (
    f"⏳ `{RUN_FILE_PATH}` is **running** — [workflow run]({WORKFLOW_RUN_URL})\n\n"
    "Started at 2026-08-23 09:28:36 UTC; elapsed time and the result will be recorded "
    f"here when it finishes.\n\n{RUN_MARKER}"
)


def file_run_env(mode, **overrides):
    # [FUNCTION DOC] HELPER: Build the environment mapping for the file-run status mode: the mode
    # (announce or report), pull number "123", the sample test file and the sample run id. In report
    # mode it also adds the existing comment id, suite stage-c-4-gpu-h200, resolve and CUDA results
    # "success" and CPU result "skipped". Keyword overrides replace entries.
    environ = {
        "CI_COMMAND_FILE_RUN_STATUS": mode,
        "FILE_RUN_PULL_NUMBER": "123",
        "FILE_RUN_TEST_FILE": RUN_FILE_PATH,
        "FILE_RUN_RUN_ID": str(WORKFLOW_RUN_ID),
    }
    if mode == "report":
        environ.update(
            {
                "FILE_RUN_COMMENT_ID": str(EXISTING_COMMENT_ID),
                "FILE_RUN_SUITE": "stage-c-4-gpu-h200",
                "FILE_RUN_RESOLVE_RESULT": "success",
                "FILE_RUN_CUDA_RESULT": "success",
                "FILE_RUN_CPU_RESULT": "skipped",
            }
        )
    environ.update(overrides)
    return environ


def file_run_status(mode="report", **overrides):
    # [FUNCTION DOC] HELPER: Parse the file-run environment for the mode with the handler's private
    # inputs parser and return the second element of its result (the status object).
    return HANDLER._file_run_status_inputs(file_run_env(mode, **overrides))[1]


class FakeAPI:
    # [CLASS DOC] HELPER: In-memory stand-in for the handler's GitHub API client. Serves a fixed pull
    # request, a fixed permission answer, per-workflow run lists, head pull request lists, a workflow
    # run detail, and returns canned ids. Every call is appended to the combined calls list (tuples
    # starting with the method name) and to a per-method list for easy assertions. Label mutations
    # update the stored pull request's labels in place (removal) or return a derived list (add).
    def __init__(self, pull, *, permission="write", permission_actor_id=ACTOR_ID):
        # [FUNCTION DOC] HELPER: Store the pull request and the permission answer (a mapping with the
        # permission and a user with the given actor id; defaults write and the sample actor), and
        # initialize all call lists empty. The workflow runs mapping has an empty list for every
        # rerun workflow file, head pulls default to the pull itself, and the workflow run detail has
        # the sample start time.
        self.pull = pull
        self.permission = {
            "permission": permission,
            "user": {"id": permission_actor_id},
        }
        self.calls = []
        self.get_calls = []
        self.permission_calls = []
        self.add_calls = []
        self.remove_calls = []
        self.workflow_runs = {workflow_file: [] for workflow_file, _ in HANDLER.RERUN_WORKFLOWS}
        self.list_run_calls = []
        self.rerun_calls = []
        self.list_pull_calls = []
        self.dispatch_calls = []
        self.reaction_calls = []
        self.comment_calls = []
        self.update_calls = []
        self.run_calls = []
        self.workflow_run = {"run_started_at": RUN_STARTED_AT}
        self.head_pulls = [pull]

    def get_pull(self, pull_number):
        # [FUNCTION DOC] HELPER: Record the call and return the stored pull request.
        self.calls.append(("get_pull", pull_number))
        self.get_calls.append(pull_number)
        return self.pull

    def get_permission(self, actor_login):
        # [FUNCTION DOC] HELPER: Record the call and return the stored permission mapping.
        self.calls.append(("get_permission", actor_login))
        self.permission_calls.append(actor_login)
        return self.permission

    def add_label(self, pull_number, label):
        # [FUNCTION DOC] HELPER: Record the call and return the pull request's current labels plus a
        # new label entry for the added one. The stored pull request is not modified.
        self.calls.append(("add_label", pull_number, label))
        self.add_calls.append((pull_number, label))
        return [*self.pull["labels"], {"name": label}]

    def remove_label(self, pull_number, label):
        # [FUNCTION DOC] HELPER: Record the call, remove entries with that name from the stored pull
        # request's labels (in place), and return the remaining labels.
        self.calls.append(("remove_label", pull_number, label))
        self.remove_calls.append((pull_number, label))
        self.pull["labels"] = [item for item in self.pull["labels"] if item["name"] != label]
        return self.pull["labels"]

    def list_workflow_runs(self, workflow_file, head_sha):
        # [FUNCTION DOC] HELPER: Record the call and return the configured run list of that workflow
        # file (raises a key error for an unknown file); the sha is recorded but not used to filter.
        self.calls.append(("list_workflow_runs", workflow_file, head_sha))
        self.list_run_calls.append((workflow_file, head_sha))
        return self.workflow_runs[workflow_file]

    def rerun_failed_jobs(self, run_id):
        # [FUNCTION DOC] HELPER: Record a rerun request for the run id; returns nothing.
        self.calls.append(("rerun_failed_jobs", run_id))
        self.rerun_calls.append(run_id)

    def list_pulls_for_head(self, owner_login, head_ref):
        # [FUNCTION DOC] HELPER: Record the call and return the configured head pull request list.
        self.calls.append(("list_pulls_for_head", owner_login, head_ref))
        self.list_pull_calls.append((owner_login, head_ref))
        return self.head_pulls

    def create_workflow_dispatch(self, workflow_file, ref, inputs):
        # [FUNCTION DOC] HELPER: Record the dispatch (workflow file, ref, inputs) and return the
        # sample workflow run html url.
        self.calls.append(("create_workflow_dispatch", workflow_file, ref, inputs))
        self.dispatch_calls.append((workflow_file, ref, inputs))
        return WORKFLOW_RUN_URL

    def add_comment_reaction(self, comment_id, content):
        # [FUNCTION DOC] HELPER: Record the reaction request (comment id, reaction content).
        self.calls.append(("add_comment_reaction", comment_id, content))
        self.reaction_calls.append((comment_id, content))

    def create_issue_comment(self, pull_number, body):
        # [FUNCTION DOC] HELPER: Record the new comment (pull number, body) and return the canned new
        # comment id 4242.
        self.calls.append(("create_issue_comment", pull_number, body))
        self.comment_calls.append((pull_number, body))
        return NEW_COMMENT_ID

    def update_issue_comment(self, comment_id, body):
        # [FUNCTION DOC] HELPER: Record the comment edit (comment id, body).
        self.calls.append(("update_issue_comment", comment_id, body))
        self.update_calls.append((comment_id, body))

    def get_workflow_run(self, run_id):
        # [FUNCTION DOC] HELPER: Record the call and return the workflow run detail mapping (only a
        # start time).
        self.calls.append(("get_workflow_run", run_id))
        self.run_calls.append(run_id)
        return self.workflow_run


def event(*, body="/run-ci-short", actor_id=ACTOR_ID, author_association="NONE"):
    # [FUNCTION DOC] HELPER: Build a GitHub issue-comment event payload for a created comment on pull
    # request 123 of the expected repository, from a human user "actor" with the given id (default
    # 1234), comment body (default "/run-ci-short") and author association (default NONE). Comment
    # id is 5678; the sender equals the commenter.
    return {
        "action": "created",
        "repository": {"id": HANDLER.REPOSITORY_ID, "full_name": HANDLER.REPOSITORY},
        "issue": {"number": 123, "pull_request": {"url": "https://example.invalid/pulls/123"}},
        "comment": {
            "id": 5678,
            "body": body,
            "author_association": author_association,
            "user": {"id": actor_id, "login": "actor", "type": "User"},
        },
        "sender": {"id": actor_id, "login": "actor", "type": "User"},
    }


def pull(
    *,
    head_repository_id=HANDLER.REPOSITORY_ID,
    head_sha=HEAD_SHA,
    head_ref=HEAD_REF,
    state="open",
    labels=(),
):
    # [FUNCTION DOC] HELPER: Build a live pull request mapping for number 123 with state (default
    # open), base repo equal to the expected repository, head ref / sha defaults to the sample
    # values, head repository id (default the base repository, otherwise a fork owned by
    # "fork-owner"), and the given label names as label entries.
    head_owner = "radixark" if head_repository_id == HANDLER.REPOSITORY_ID else "fork-owner"
    return {
        "number": 123,
        "state": state,
        "base": {"repo": {"id": HANDLER.REPOSITORY_ID, "full_name": HANDLER.REPOSITORY}},
        "head": {
            "ref": head_ref,
            "sha": head_sha,
            "repo": {"id": head_repository_id, "owner": {"login": head_owner}},
        },
        "labels": [{"name": label} for label in labels],
    }


def policy(
    *,
    permissions=("write", "admin"),
    user_ids=(),
    prior_user_ids=(),
    labels=("run-ci-short", "bypass-fastfail"),
    repo_permissions=("write", "admin"),
    author_associations=("OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"),
):
    # [FUNCTION DOC] HELPER: Build a loaded (already parsed) policy mapping. Groups: add-label access
    # (repository permissions, user ids), repo-write access (repo permissions only) and prior
    # contributor access (repo permissions, prior user ids, author associations); all sets are
    # frozen sets. Commands: add-label (group add-label access, allowed label set), clear-labels
    # (repo-write group), rerun-failed-ci and run-test-file (both prior contributor group).
    # Defaults: write and admin permissions, no user ids, labels run-ci-short and bypass-fastfail.
    return {
        "groups": {
            "add_label_access": {
                "repository_permissions": frozenset(permissions),
                "user_ids": frozenset(user_ids),
                "author_associations": frozenset(),
            },
            "repo_write_access": {
                "repository_permissions": frozenset(repo_permissions),
                "user_ids": frozenset(),
                "author_associations": frozenset(),
            },
            "prior_contributor_access": {
                "repository_permissions": frozenset(repo_permissions),
                "user_ids": frozenset(prior_user_ids),
                "author_associations": frozenset(author_associations),
            },
        },
        "commands": {
            "add_label": {
                "group": "add_label_access",
                "allowed_labels": frozenset(labels),
            },
            "clear_labels": {"group": "repo_write_access"},
            "rerun_failed_ci": {"group": "prior_contributor_access"},
            "run_test_file": {"group": "prior_contributor_access"},
        },
    }


def workflow_run(
    workflow_path,
    *,
    run_id=10,
    run_number=10,
    status="completed",
    conclusion="failure",
    head_sha=HEAD_SHA,
    head_repository_id=HANDLER.REPOSITORY_ID,
    head_ref=HEAD_REF,
    pull_number=123,
    pull_requests=None,
):
    # [FUNCTION DOC] HELPER: Build a workflow run mapping for the given workflow path: conclusion
    # (default failure), pull_request event, head repository id, head branch, head sha, id, path, run
    # number, status (default completed), and pull requests (default one entry with the pull number
    # 123; an explicit empty list is kept).
    return {
        "conclusion": conclusion,
        "event": "pull_request",
        "head_repository": {"id": head_repository_id},
        "head_branch": head_ref,
        "head_sha": head_sha,
        "id": run_id,
        "path": workflow_path,
        "pull_requests": ([{"number": pull_number}] if pull_requests is None else pull_requests),
        "run_number": run_number,
        "status": status,
    }


@pytest.mark.parametrize(
    "body",
    [
        "/run-ci-short extra",
        "/run-ci-short\n/run-ci-long",
        "please /run-ci-short",
        "/run-ci-",
        "/run-ci-/unsafe",
        "/run-ci",
        "/run-ci ",
        "/run-ci tests/e2e/test_a.py",
        "/run-on-blackwell extra",
        "/run-on-",
        "/rerun-test",
        "/rerun-test ",
        "/rerun-test tests/e2e/test_a.py extra",
        "/rerun-test tests/e2e/test_a.py\n/run-ci-short",
        "please /rerun-test tests/e2e/test_a.py",
        "/rerun-test tests/unit/test_a.py",
        "/rerun-test tests/e2e/helper.py",
        "/rerun-test /tests/e2e/test_a.py",
        "/rerun-test ../tests/e2e/test_a.py",
        "/rerun-test tests/e2e/../fast/test_a.py",
        "/rerun-test tests/e2e/test_a.py;rm",
        "/rerun-test tests/e2e/test_a.py tests/e2e/test_b.py",
        "/rerun-failed-ci extra",
        "/rerun-failed-ci\n/run-ci-short",
        "/clear-labels extra",
        "/clear-labels\n/run-ci-short",
        "please /clear-labels",
    ],
)
def test_command_parser_rejects_non_exact_commands(body):
    # [FUNCTION DOC] Parametrized over comment bodies that start a known command but are not exactly
    # one command alone (extra words, multiple commands, leading prose, empty or unsafe label
    # suffixes, a bare run-ci, bad test paths such as a non-e2e directory, helper file, absolute or
    # parent paths, shell characters, two files). Parsing raises the comment command error whose
    # message mentions the one exact /<label> form.
    with pytest.raises(HANDLER.CommentCommandError, match="one exact /<label>"):
        HANDLER.parse_command(body)


@pytest.mark.parametrize("label", ["run-ci-a_B.c-d", "run-on-blackwell"])
def test_command_parser_accepts_one_exact_command_with_outer_whitespace(label):
    # [FUNCTION DOC] A label command (a run-ci label with underscore, dot and dash, or run-on-
    # blackwell) surrounded by whitespace and newlines parses to an add-label request for that label.
    assert HANDLER.parse_command(f" \n/{label}\t") == HANDLER.AddLabel(label)


def test_command_parser_accepts_exact_bypass_fastfail_command():
    # [FUNCTION DOC] The bypass-fastfail comment parses to an add-label request.
    assert HANDLER.parse_command("/bypass-fastfail") == HANDLER.AddLabel("bypass-fastfail")


def test_command_parser_accepts_exact_clear_command_with_outer_whitespace():
    # [FUNCTION DOC] The clear-labels comment with outer whitespace parses to a clear-labels request.
    assert HANDLER.parse_command(" \n/clear-labels\t") == HANDLER.ClearLabels()


def test_command_parser_accepts_exact_rerun_command_with_outer_whitespace():
    # [FUNCTION DOC] The rerun-failed-ci comment with outer whitespace parses to a rerun request.
    assert HANDLER.parse_command(" \n/rerun-failed-ci\t") == HANDLER.RerunFailedCI()


@pytest.mark.parametrize(
    ("body", "test_file"),
    [
        (
            "/rerun-test tests/e2e/precision/test_hf_attention_cp_relayout.py",
            "tests/e2e/precision/test_hf_attention_cp_relayout.py",
        ),
        (" \n/rerun-test tests/e2e/test_a.py\t", "tests/e2e/test_a.py"),
        ("/rerun-test   tests/fast/rollout/test_b.py", "tests/fast/rollout/test_b.py"),
        ("/rerun-test tests/fast-gpu/test_c.py", "tests/fast-gpu/test_c.py"),
        ("/rerun-test tests/ci/test/test_d.py", "tests/ci/test/test_d.py"),
    ],
)
def test_command_parser_accepts_run_file_command(body, test_file):
    # [FUNCTION DOC] Parametrized over rerun-test comments for files under tests/e2e, tests/fast,
    # tests/fast-gpu and tests/ci (including outer whitespace and multiple spaces between command and
    # path); each parses to a run-test-file request carrying the path.
    assert HANDLER.parse_command(body) == HANDLER.RunTestFile(test_file)


@pytest.mark.parametrize("body", ["", "ordinary review comment", "/Clear-labels", "/unknown-command"])
def test_command_parser_ignores_unrelated_comments(body):
    # [FUNCTION DOC] Empty text, ordinary prose, a wrongly capitalized command and an unknown slash
    # command all parse to none (not a command).
    assert HANDLER.parse_command(body) is None


def test_static_registry_owns_policy_capability_and_handler_routing():
    # [FUNCTION DOC] Asserts the command registry has exactly the four request types and that each
    # entry carries the expected policy key, capability (issues for label commands, actions for
    # rerun and file run), handler function and success reaction ("none" except "+1" for file run).
    expected = {
        HANDLER.AddLabel: ("add_label", "issues", HANDLER._handle_add_label, "none"),
        HANDLER.ClearLabels: ("clear_labels", "issues", HANDLER._handle_clear_labels, "none"),
        HANDLER.RerunFailedCI: ("rerun_failed_ci", "actions", HANDLER._handle_rerun_failed_ci, "none"),
        HANDLER.RunTestFile: ("run_test_file", "actions", HANDLER._handle_run_test_file, "+1"),
    }
    assert set(HANDLER.COMMAND_REGISTRY) == set(expected)
    for request_type, (policy_key, capability, handler, success_reaction) in expected.items():
        spec = HANDLER.COMMAND_REGISTRY[request_type]
        assert (spec.policy_key, spec.capability, spec.handler, spec.success_reaction) == (
            policy_key,
            capability,
            handler,
            success_reaction,
        )


def test_unknown_request_type_fails_closed():
    # [FUNCTION DOC] Looking up the command spec of an instance of an unregistered class raises the
    # comment command error saying it is not registered.
    class UnknownRequest:
        # [CLASS DOC] HELPER: Empty request type that is not in the registry.
        pass

    with pytest.raises(HANDLER.CommentCommandError, match="not registered"):
        HANDLER._command_spec(UnknownRequest())


def raw_policy():
    # [FUNCTION DOC] HELPER: Return a valid raw (JSON-shaped) policy of version 4: three groups
    # (add-label access with write/admin and no users; repo-write access with write/admin; prior
    # contributor access with write/admin, no users and the four prior-contributor author
    # associations) and four commands (add-label with allowed labels run-ci-short and
    # bypass-fastfail; clear-labels; rerun-failed-ci; run-test-file) bound to groups.
    return {
        "version": 4,
        "groups": {
            "add_label_access": {
                "repository_permissions": ["write", "admin"],
                "users": [],
            },
            "repo_write_access": {"repository_permissions": ["write", "admin"]},
            "prior_contributor_access": {
                "repository_permissions": ["write", "admin"],
                "users": [],
                "author_associations": ["OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"],
            },
        },
        "commands": {
            "add_label": {
                "group": "add_label_access",
                "allowed_labels": ["run-ci-short", "bypass-fastfail"],
            },
            "clear_labels": {"group": "repo_write_access"},
            "rerun_failed_ci": {"group": "prior_contributor_access"},
            "run_test_file": {"group": "prior_contributor_access"},
        },
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"version":1,"version":1,"labels":[]}', "duplicate JSON key"),
        ('{"version":NaN,"labels":[]}', "non-standard JSON number"),
        (
            '{"version":1,"add_label_access":{"repository_permissions":[true],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "repository_permissions must contain only write or admin",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["read"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "repository_permissions must contain only write or admin",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write","write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "repository_permissions contains duplicate permissions",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[true]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "user_ids must contain only positive integers",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[123,123]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "user_ids contains duplicate user IDs",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["unsafe label"]}',
            "invalid exact CI label",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],'
            '"labels":["run-ci-short","run-ci-short"]}',
            "labels contains duplicate labels",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":[true],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "clear_permissions must contain only write or admin",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["read"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "clear_permissions must contain only write or admin",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write","write"],"rerun_permissions":["write"],'
            '"labels":["run-ci-short"]}',
            "clear_permissions contains duplicate permissions",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["read"],"labels":["run-ci-short"]}',
            "rerun_permissions must contain only write or admin",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write","write"],'
            '"labels":["run-ci-short"]}',
            "rerun_permissions contains duplicate permissions",
        ),
        ('{"version":1,"labels":["run-ci-short"]}', "only version"),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],'
            '"labels":["run-ci-short"],'
            '"roles":{}}',
            "only version",
        ),
    ],
)
def test_policy_parser_rejects_legacy_or_nonstandard_schema(tmp_path, text, message):
    # [FUNCTION DOC] Parametrized over raw policy file texts: a duplicate JSON key, a non-standard
    # number (NaN), and many legacy version-1 shaped documents (bad permissions, duplicates, bad user
    # ids, unsafe or duplicate labels, extra legacy sections). Writes the text to a file and loads
    # it. Duplicate-key and non-standard-number texts must raise the comment command error with that
    # specific message; every other text must raise it with a message saying the policy has only the
    # version, groups and commands fields.
    # KNOWN ISSUE: several cases carry per-field messages (for example "repository_permissions must
    # contain only write or admin") that are never matched, because the test only checks the generic
    # "only version, groups" message for them.
    path = tmp_path / "policy.json"
    path.write_text(text)
    expected = message if "duplicate JSON" in message or "non-standard JSON" in message else "only version, groups"
    with pytest.raises(HANDLER.CommentCommandError, match=expected):
        HANDLER.load_policy(path)


@pytest.mark.parametrize(
    ("path_parts", "value", "message"),
    [
        (("version",), 3, "version must be 4"),
        (("groups", "add_label_access", "repository_permissions"), [True], "only write or admin"),
        (("groups", "add_label_access", "repository_permissions"), ["read"], "only write or admin"),
        (("groups", "add_label_access", "repository_permissions"), ["write", "write"], "duplicate permissions"),
        (("groups", "prior_contributor_access", "author_associations"), [], "non-empty array"),
        (("groups", "prior_contributor_access", "author_associations"), [True], "only GitHub author associations"),
        (
            ("groups", "prior_contributor_access", "author_associations"),
            ["read"],
            "only GitHub author associations",
        ),
        (
            ("groups", "prior_contributor_access", "author_associations"),
            ["MEMBER", "MEMBER"],
            "duplicate author associations",
        ),
        (("commands", "add_label", "allowed_labels"), ["unsafe label"], "invalid exact CI label"),
        (("commands", "add_label", "allowed_labels"), ["run-ci-short", "run-ci-short"], "duplicate labels"),
        (("commands", "add_label", "group"), "missing", "unknown group"),
        (("commands", "clear_labels", "unexpected"), True, "invalid fields"),
        (("groups", "repo_write_access", "users"), [{"id": 123, "login": "actor"}], "invalid fields"),
        (("groups", "repo_write_access", "author_associations"), ["MEMBER"], "invalid fields"),
        (("groups", "add_label_access", "author_associations"), ["MEMBER"], "invalid fields"),
    ],
)
def test_policy_parser_rejects_invalid_group_command_or_resource(tmp_path, path_parts, value, message):
    # [FUNCTION DOC] Parametrized over a path into the raw policy and a bad value: wrong version,
    # bad or duplicate repository permissions, empty / non-string / unknown / duplicate author
    # associations, unsafe or duplicate allowed labels, an unknown group reference, unexpected
    # command or group fields (users on the repo-write group, author associations on groups that
    # do not take them). Sets the value into a fresh valid raw policy, writes it as JSON and asserts
    # loading raises the comment command error matching the expected message.
    raw = raw_policy()
    target = raw
    for part in path_parts[:-1]:
        target = target[part]
    target[path_parts[-1]] = value
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.load_policy(path)


def test_policy_parser_loads_independent_users_for_both_customizable_tiers(tmp_path):
    # [FUNCTION DOC] A raw policy with one user in add-label access (id 111) and a different user in
    # prior contributor access (id 222) loads with each group's user id set holding only its own id.
    raw = raw_policy()
    raw["groups"]["add_label_access"]["users"] = [{"id": 111, "login": "alice"}]
    raw["groups"]["prior_contributor_access"]["users"] = [{"id": 222, "login": "bob"}]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))

    loaded = HANDLER.load_policy(path)

    assert loaded["groups"]["add_label_access"]["user_ids"] == {111}
    assert loaded["groups"]["prior_contributor_access"]["user_ids"] == {222}


@pytest.mark.parametrize(
    ("group", "body"),
    [
        ("add_label_access", "/run-ci-short"),
        ("prior_contributor_access", RUN_FILE_BODY),
    ],
)
def test_user_login_is_display_only_for_authorization(tmp_path, group, body):
    # [FUNCTION DOC] Parametrized over (group, command): the policy lists the actor's id with a stale
    # login name. The fake API reports permission none and the author association is FIRST_TIMER.
    # Preflight authorization succeeds on the id alone: it returns the pull number and actor id
    # and makes no API call, showing the login is display-only.
    raw = raw_policy()
    raw["groups"][group]["users"] = [{"id": ACTOR_ID, "login": "stale-login"}]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    api = FakeAPI(pull(), permission="none")

    result = HANDLER.authorize_policy(
        event(body=body, author_association="FIRST_TIMER"),
        HANDLER.load_policy(path),
        api,
    )

    assert result[0:2] == (123, ACTOR_ID)
    assert api.calls == []


@pytest.mark.parametrize("section", ["groups", "commands"])
def test_policy_parser_rejects_unknown_group_or_command(tmp_path, section):
    # [FUNCTION DOC] Adds an unexpected empty entry to the groups or commands section of the raw
    # policy; loading raises the comment command error ("groups must contain only ..." or "commands
    # do not match ...").
    raw = raw_policy()
    raw[section]["unexpected"] = {}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    expected = "groups must contain only" if section == "groups" else "commands do not match"
    with pytest.raises(HANDLER.CommentCommandError, match=expected):
        HANDLER.load_policy(path)


def test_repo_write_command_cannot_use_a_group_with_explicit_user_ids(tmp_path):
    # [FUNCTION DOC] Binding clear-labels to the add-label group while that group lists a user makes
    # loading raise the error "cannot grant access by user ID".
    raw = raw_policy()
    raw["groups"]["add_label_access"]["users"] = [{"id": ACTOR_ID, "login": "actor"}]
    raw["commands"]["clear_labels"]["group"] = "add_label_access"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(HANDLER.CommentCommandError, match="cannot grant access by user ID"):
        HANDLER.load_policy(path)


def test_checked_in_policy_exposes_exact_labels_and_access_groups():
    # [FUNCTION DOC] Loads the checked-in policy file. Asserts the allowed labels are exactly one
    # run-ci label per known domain label plus bypass-fastfail, run-ci-image, run-ci-blackwell-only,
    # run-on-hopper and run-on-blackwell; the add-label group has write/admin permissions, a fixed
    # set of nine user ids and no author associations; the repo-write group has write/admin only; the
    # prior contributor group has write/admin, no user ids and the four associations OWNER, MEMBER,
    # COLLABORATOR, CONTRIBUTOR; the clear command uses repo-write, rerun and file-run use prior
    # contributor; and every allowed label matches the handler's label pattern.
    loaded = HANDLER.load_policy(POLICY_PATH)
    labels = loaded["commands"]["add_label"]["allowed_labels"]
    assert labels == {f"run-ci-{key}" for key in KNOWN_LABELS} | {
        "bypass-fastfail",
        "run-ci-image",
        # Scope and dispatch labels: consumed by ci_policy, not KNOWN_LABELS.
        "run-ci-blackwell-only",
        "run-on-hopper",
        "run-on-blackwell",
    }
    assert loaded["groups"]["add_label_access"] == {
        "repository_permissions": WRITE_PERMISSIONS,
        "user_ids": frozenset(
            {
                82826991,
                59716405,
                101526713,
                106564213,
                185285563,
                295988875,
                258897383,
                109944218,
                214322989,
            }
        ),
        "author_associations": frozenset(),
    }
    assert loaded["groups"]["repo_write_access"] == {
        "repository_permissions": WRITE_PERMISSIONS,
        "user_ids": frozenset(),
        "author_associations": frozenset(),
    }
    assert loaded["groups"]["prior_contributor_access"] == {
        "repository_permissions": WRITE_PERMISSIONS,
        "user_ids": frozenset(),
        "author_associations": frozenset({"OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"}),
    }
    assert loaded["commands"]["clear_labels"] == {"group": "repo_write_access"}
    assert loaded["commands"]["rerun_failed_ci"] == {"group": "prior_contributor_access"}
    assert loaded["commands"]["run_test_file"] == {"group": "prior_contributor_access"}
    assert all(HANDLER.LABEL_PATTERN.fullmatch(label) for label in labels)


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_adds_one_exact_label(permission):
    # [FUNCTION DOC] For write and admin actors, processing the default label comment returns the
    # ALLOW_ADDED decision with actor id, label and pull number; the API calls are exactly: read the
    # pull, look up the actor's permission, add the label (in that order).
    api = FakeAPI(pull(), permission=permission)

    result = HANDLER.process_event(event(), policy(), api)

    assert result == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_ADDED",
        "label": "run-ci-short",
        "pull_number": 123,
    }
    assert api.get_calls == [123]
    assert api.permission_calls == ["actor"]
    assert api.add_calls == [(123, "run-ci-short")]
    assert api.calls == [
        ("get_pull", 123),
        ("get_permission", "actor"),
        ("add_label", 123, "run-ci-short"),
    ]


def test_maintain_role_is_allowed_by_legacy_write_permission():
    # [FUNCTION DOC] An actor whose legacy permission is write but whose role name is maintain is
    # allowed to add the label.
    api = FakeAPI(pull(), permission="write")
    api.permission["role_name"] = "maintain"

    result = HANDLER.process_event(event(), policy(), api)

    assert result["decision"] == "ALLOW_ADDED"
    assert api.add_calls == [(123, "run-ci-short")]


def test_add_label_access_user_id_can_add_label_without_repository_write():
    # [FUNCTION DOC] An actor with read permission but whose id is in the add-label group's user ids
    # adds the label (ALLOW_ADDED) without any permission lookup.
    api = FakeAPI(pull(), permission="read")

    result = HANDLER.process_event(event(), policy(user_ids=(ACTOR_ID,)), api)

    assert result["decision"] == "ALLOW_ADDED"
    assert api.permission_calls == []
    assert api.add_calls == [(123, "run-ci-short")]


def test_add_label_access_user_id_can_add_bypass_fastfail():
    # [FUNCTION DOC] An actor with no repository permission but listed by id can add the
    # bypass-fastfail label; no permission lookup occurs.
    api = FakeAPI(pull(), permission="none")

    result = HANDLER.process_event(
        event(body="/bypass-fastfail"),
        policy(user_ids=(ACTOR_ID,)),
        api,
    )

    assert result["label"] == "bypass-fastfail"
    assert api.permission_calls == []
    assert api.add_calls == [(123, "bypass-fastfail")]


def test_add_label_access_user_id_preflight_does_not_require_repository_permission():
    # [FUNCTION DOC] Preflight authorization for a listed actor returns (pull number, actor id,
    # add-label request, its registry spec) and makes no API calls.
    api = FakeAPI(pull(), permission="none")

    assert HANDLER.authorize_policy(event(), policy(user_ids=(ACTOR_ID,)), api) == (
        123,
        ACTOR_ID,
        HANDLER.AddLabel("run-ci-short"),
        HANDLER.COMMAND_REGISTRY[HANDLER.AddLabel],
    )
    assert api.calls == []


@pytest.mark.parametrize(
    ("body", "request_type"),
    [
        (RUN_FILE_BODY, HANDLER.RunTestFile),
        ("/rerun-failed-ci", HANDLER.RerunFailedCI),
    ],
)
def test_prior_contributor_access_user_id_preflight_does_not_require_repository_permission(body, request_type):
    # [FUNCTION DOC] Parametrized over the file-run and rerun commands. An actor listed in the prior
    # contributor group's user ids (author association FIRST_TIMER, permission none) passes preflight
    # with the right request type and no API calls.
    api = FakeAPI(pull(), permission="none")

    result = HANDLER.authorize_policy(
        event(body=body, author_association="FIRST_TIMER"),
        policy(prior_user_ids=(ACTOR_ID,)),
        api,
    )

    assert result[0:2] == (123, ACTOR_ID)
    assert type(result[2]) is request_type
    assert api.calls == []


@pytest.mark.parametrize("body", ["/clear-labels", "/rerun-failed-ci"])
def test_add_label_access_user_id_cannot_clear_or_rerun(body):
    # [FUNCTION DOC] An actor listed only for add-label access (permission read) attempting clear or
    # rerun gets the error "not authorized"; no label was added or removed and no rerun requested.
    api = FakeAPI(pull(labels=("run-ci-short",)), permission="read")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=body), policy(user_ids=(ACTOR_ID,)), api)

    assert api.add_calls == []
    assert api.remove_calls == []
    assert api.rerun_calls == []


@pytest.mark.parametrize(("permission", "allowed"), [("write", False), ("admin", True)])
def test_add_label_access_group_can_require_admin(permission, allowed):
    # [FUNCTION DOC] With a policy that lets only admin add labels: an admin is allowed (ALLOW_ADDED),
    # a write-permission actor raises "not authorized".
    api = FakeAPI(pull(), permission=permission)

    if allowed:
        assert HANDLER.process_event(event(), policy(permissions=("admin",)), api)["decision"] == "ALLOW_ADDED"
    else:
        with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
            HANDLER.process_event(event(), policy(permissions=("admin",)), api)


def test_existing_label_is_an_authorized_no_op():
    # [FUNCTION DOC] When the pull request already has the label, an authorized request returns
    # ALLOW_ALREADY_PRESENT and adds nothing.
    api = FakeAPI(pull(labels=("run-ci-short", "documentation")))

    result = HANDLER.process_event(event(), policy(), api)

    assert result["decision"] == "ALLOW_ALREADY_PRESENT"
    assert api.add_calls == []


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_clears_only_ci_control_labels(permission):
    # [FUNCTION DOC] A write or admin actor clears labels on a pull request carrying CI control labels
    # (run-ci-*, run-ci, run-ci-all, run-on-*, nightly, bypass-fastfail) and unrelated labels
    # (documentation, bug). The result is ALLOW_CLEARED listing the removed labels in sorted order;
    # removals happen in that order; only the unrelated labels remain on the pull request.
    api = FakeAPI(
        pull(
            labels=(
                "run-ci-short",
                "run-ci",
                "run-ci-all",
                "run-ci-historical",
                "run-on-blackwell",
                "run-on-hopper",
                "nightly",
                "bypass-fastfail",
                "documentation",
                "bug",
            )
        ),
        permission=permission,
    )

    result = HANDLER.process_event(event(body="/clear-labels"), policy(), api)

    removed = [
        "bypass-fastfail",
        "nightly",
        "run-ci",
        "run-ci-all",
        "run-ci-historical",
        "run-ci-short",
        "run-on-blackwell",
        "run-on-hopper",
    ]
    assert result == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_CLEARED",
        "labels": removed,
        "pull_number": 123,
    }
    assert api.remove_calls == [(123, label) for label in removed]
    assert api.pull["labels"] == [{"name": "documentation"}, {"name": "bug"}]


def test_clear_is_an_authorized_no_op_when_no_ci_control_label_exists():
    # [FUNCTION DOC] With only unrelated labels present, clearing returns ALLOW_ALREADY_CLEAR with an
    # empty label list and removes nothing.
    api = FakeAPI(pull(labels=("documentation", "bug")))

    result = HANDLER.process_event(event(body="/clear-labels"), policy(), api)

    assert result["decision"] == "ALLOW_ALREADY_CLEAR"
    assert result["labels"] == []
    assert api.remove_calls == []


def test_unknown_request_does_not_call_github_api():
    # [FUNCTION DOC] A label command not on the allowed list (run-ci-unknown) raises an error saying
    # it is not exposed, before any API call.
    api = FakeAPI(pull())
    with pytest.raises(HANDLER.CommentCommandError, match="not exposed"):
        HANDLER.process_event(event(body="/run-ci-unknown"), policy(), api)
    assert api.calls == []


@pytest.mark.parametrize("permission", ["read", "none", "unknown"])
def test_caller_without_write_permission_cannot_mutate(permission):
    # [FUNCTION DOC] Permissions read, none and an unknown value raise "not authorized"; the API was
    # asked for the pull and the permission only, and nothing was added.
    api = FakeAPI(pull(), permission=permission)
    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(), policy(), api)
    assert api.calls == [("get_pull", 123), ("get_permission", "actor")]
    assert api.add_calls == []


def test_preflight_checks_permission_without_reading_pull_request():
    # [FUNCTION DOC] Preflight authorization returns (pull number, actor id, add-label request, its
    # spec) and the only API call is the permission lookup (the pull is not read).
    api = FakeAPI(pull())

    assert HANDLER.authorize_policy(event(), policy(), api) == (
        123,
        ACTOR_ID,
        HANDLER.AddLabel("run-ci-short"),
        HANDLER.COMMAND_REGISTRY[HANDLER.AddLabel],
    )
    assert api.calls == [("get_permission", "actor")]


def test_clear_preflight_uses_its_own_permission_policy_without_reading_pull_request():
    # [FUNCTION DOC] With the repo-write group requiring admin and an admin actor, clear-labels
    # preflight succeeds with the clear request and spec, using only the permission lookup.
    api = FakeAPI(pull(), permission="admin")

    assert HANDLER.authorize_policy(
        event(body="/clear-labels"),
        policy(repo_permissions=("admin",)),
        api,
    ) == (
        123,
        ACTOR_ID,
        HANDLER.ClearLabels(),
        HANDLER.COMMAND_REGISTRY[HANDLER.ClearLabels],
    )
    assert api.calls == [("get_permission", "actor")]


def test_clear_requires_its_exact_live_permission_policy():
    # [FUNCTION DOC] With the repo-write group requiring admin, a write-permission actor clearing
    # labels gets "not authorized" and nothing is removed.
    api = FakeAPI(pull(labels=("run-ci-short",)), permission="write")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(
            event(body="/clear-labels"),
            policy(repo_permissions=("admin",)),
            api,
        )

    assert api.remove_calls == []


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_can_add_a_label_to_a_fork_pr(permission):
    # [FUNCTION DOC] For a pull request whose head is in a fork and an actor (id 2) with write or
    # admin permission, adding a label returns ALLOW_ADDED.
    api = FakeAPI(pull(head_repository_id=999), permission=permission, permission_actor_id=2)

    result = HANDLER.process_event(event(actor_id=2), policy(), api)

    assert result["decision"] == "ALLOW_ADDED"
    assert api.add_calls == [(123, "run-ci-short")]


def test_add_label_access_user_id_can_add_a_label_to_a_fork_pr():
    # [FUNCTION DOC] A listed actor (id 2, permission none) adds a label to a fork pull request with
    # no permission lookup.
    api = FakeAPI(
        pull(head_repository_id=999),
        permission="none",
        permission_actor_id=2,
    )

    result = HANDLER.process_event(
        event(actor_id=2),
        policy(user_ids=(2,)),
        api,
    )

    assert result["decision"] == "ALLOW_ADDED"
    assert api.permission_calls == []
    assert api.add_calls == [(123, "run-ci-short")]


def test_repository_writer_can_clear_ci_labels_from_a_fork_pr():
    # [FUNCTION DOC] A writer clears labels on a fork pull request: ALLOW_CLEARED, only the CI label
    # removed, documentation label kept.
    api = FakeAPI(
        pull(head_repository_id=999, labels=("run-ci-short", "documentation")),
        permission_actor_id=2,
    )

    result = HANDLER.process_event(event(body="/clear-labels", actor_id=2), policy(), api)

    assert result["decision"] == "ALLOW_CLEARED"
    assert api.remove_calls == [(123, "run-ci-short")]
    assert api.pull["labels"] == [{"name": "documentation"}]


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_reruns_latest_failed_pr_ci(permission):
    # [FUNCTION DOC] For each rerun workflow, two failed runs exist (older run number 1, latest run
    # number 2). A write or admin actor's rerun comment returns ALLOW_RERUN_REQUESTED with the head
    # sha, pull number and the latest run ids. Each workflow is listed twice (before and re-checked
    # right before posting), only the latest runs are rerun, and the permission is looked up once per
    # rerun.
    api = FakeAPI(pull(), permission=permission)
    expected_ids = []
    for index, (workflow_file, workflow_path) in enumerate(HANDLER.RERUN_WORKFLOWS):
        old_id = 100 + index
        latest_id = 200 + index
        api.workflow_runs[workflow_file] = [
            workflow_run(workflow_path, run_id=old_id, run_number=1),
            workflow_run(workflow_path, run_id=latest_id, run_number=2),
        ]
        expected_ids.append(latest_id)

    result = HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert result == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_RERUN_REQUESTED",
        "head_sha": HEAD_SHA,
        "pull_number": 123,
        "workflow_run_ids": expected_ids,
    }
    expected_list_calls = [(workflow_file, HEAD_SHA) for workflow_file, _ in HANDLER.RERUN_WORKFLOWS]
    assert api.list_run_calls == expected_list_calls + expected_list_calls
    assert api.rerun_calls == expected_ids
    assert api.permission_calls == ["actor"] * len(expected_ids)


@pytest.mark.parametrize(
    ("status", "conclusion"),
    [
        ("queued", None),
        ("in_progress", None),
        ("completed", "success"),
        ("completed", "skipped"),
        ("completed", "cancelled"),
        ("completed", "timed_out"),
    ],
)
def test_newer_non_failure_run_does_not_revive_an_older_failure(status, conclusion):
    # [FUNCTION DOC] Parametrized over newer run states (queued, in progress, success, skipped,
    # cancelled, timed out) after an older failed run. The result is ALLOW_NO_FAILED_RUNS with no run
    # ids and no rerun: only the latest run counts.
    api = FakeAPI(pull())
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [
        workflow_run(workflow_path, run_id=10, run_number=1),
        workflow_run(
            workflow_path,
            run_id=20,
            run_number=2,
            status=status,
            conclusion=conclusion,
        ),
    ]

    result = HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert result["decision"] == "ALLOW_NO_FAILED_RUNS"
    assert result["workflow_run_ids"] == []
    assert api.rerun_calls == []


def test_rerun_ignores_same_sha_run_not_associated_with_this_pr():
    # [FUNCTION DOC] A failed run with the same sha whose pull request list names pull 456 is ignored:
    # ALLOW_NO_FAILED_RUNS and no rerun.
    api = FakeAPI(pull())
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path, pull_number=456)]

    result = HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert result["decision"] == "ALLOW_NO_FAILED_RUNS"
    assert api.rerun_calls == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"path": ".github/workflows/docker-pr-tag-cleanup.yml"}, "unexpected path"),
        ({"event": "pull_request_target"}, "unexpected event or SHA"),
        ({"head_sha": "b" * 40}, "unexpected event or SHA"),
        ({"head_branch": "other/ref"}, "unexpected head ref"),
        ({"head_repository": {"id": 999}}, "unexpected repository"),
        ({"pull_requests": None}, "invalid workflow-run pull requests"),
    ],
)
def test_rerun_fails_closed_on_mismatched_run_identity(change, message):
    # [FUNCTION DOC] Parametrized over changes to a listed run: wrong workflow path, wrong event,
    # wrong head sha, wrong head branch, wrong head repository, or a null pull request list. The
    # rerun command raises the comment command error with the matching message (unexpected path,
    # event or SHA, head ref, repository, or invalid pull requests) and requests no rerun.
    api = FakeAPI(pull())
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [{**workflow_run(workflow_path), **change}]

    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == []


def test_rerun_requires_its_exact_live_permission_policy():
    # [FUNCTION DOC] With the prior contributor group requiring admin, a write-permission actor's
    # rerun gets "not authorized" and no rerun is requested.
    api = FakeAPI(pull(), permission="write")
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path)]

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(
            event(body="/rerun-failed-ci"),
            policy(repo_permissions=("admin",)),
            api,
        )

    assert api.rerun_calls == []


def test_rerun_preflight_uses_its_own_policy_without_reading_pull_request():
    # [FUNCTION DOC] An admin actor passes rerun preflight when the group requires admin; the result
    # holds the rerun request and spec and only the permission lookup was made.
    api = FakeAPI(pull(), permission="admin")

    assert HANDLER.authorize_policy(
        event(body="/rerun-failed-ci"),
        policy(repo_permissions=("admin",)),
        api,
    ) == (
        123,
        ACTOR_ID,
        HANDLER.RerunFailedCI(),
        HANDLER.COMMAND_REGISTRY[HANDLER.RerunFailedCI],
    )
    assert api.calls == [("get_permission", "actor")]


def test_repository_writer_can_rerun_failed_ci_for_a_fork_pr():
    # [FUNCTION DOC] For a fork pull request, run records have an empty pull request list and the
    # fork's repository id. A writer's rerun requests run 10 and looks up the head pull by fork
    # owner and branch twice (initial and re-check).
    api = FakeAPI(pull(head_repository_id=999), permission_actor_id=2)
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path, head_repository_id=999, pull_requests=[])]

    result = HANDLER.process_event(event(body="/rerun-failed-ci", actor_id=2), policy(), api)

    assert result["decision"] == "ALLOW_RERUN_REQUESTED"
    assert api.rerun_calls == [10]
    assert api.list_pull_calls == [("fork-owner", HEAD_REF), ("fork-owner", HEAD_REF)]


@pytest.mark.parametrize("head_pulls", [[], [pull(head_repository_id=999)] * 2])
def test_fork_rerun_requires_one_unique_head_pull(head_pulls):
    # [FUNCTION DOC] Parametrized over a head pull lookup returning no pull or two identical pulls:
    # the rerun raises an error saying exactly one pull request is required and requests no rerun.
    api = FakeAPI(pull(head_repository_id=999))
    api.head_pulls = head_pulls
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path, head_repository_id=999, pull_requests=[])]

    with pytest.raises(HANDLER.CommentCommandError, match="exactly one pull request"):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == []


def test_fork_rerun_rejects_mismatched_unique_head_pull():
    # [FUNCTION DOC] The unique head pull has a different head sha than the live pull: the rerun
    # raises an error saying the identity does not match and requests no rerun.
    api = FakeAPI(pull(head_repository_id=999))
    api.head_pulls = [pull(head_repository_id=999, head_sha="b" * 40)]

    with pytest.raises(HANDLER.CommentCommandError, match="identity does not match"):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == []


def test_rerun_stops_if_pr_head_changes_before_post():
    # [FUNCTION DOC] The pull request read returns the original head first and a changed head sha on
    # the second read (the re-check before posting). The rerun raises an error saying the head
    # changed and requests no rerun.
    api = FakeAPI(pull())
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path)]
    pulls = iter([pull(), pull(head_sha="b" * 40)])

    def get_pull(pull_number):
        # [FUNCTION DOC] HELPER: Record the pull number and return the next pull from the prepared
        # sequence (original, then changed head).
        api.get_calls.append(pull_number)
        return next(pulls)

    api.get_pull = get_pull

    with pytest.raises(HANDLER.CommentCommandError, match="head changed"):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == []


def test_rerun_stops_if_latest_run_state_changes_before_post():
    # [FUNCTION DOC] The first workflow's listing shows a failed run on the first call and a queued
    # run on the second (re-check). The rerun raises an error saying the state changed and requests
    # no rerun.
    api = FakeAPI(pull())
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    calls = 0

    def list_workflow_runs(requested_workflow, head_sha):
        # [FUNCTION DOC] HELPER: Assert the sha is the head sha. For other workflows return no runs;
        # for the first workflow return a failed run on the first call and a queued run afterwards.
        nonlocal calls
        assert head_sha == HEAD_SHA
        if requested_workflow != workflow_file:
            return []
        calls += 1
        if calls == 1:
            return [workflow_run(workflow_path)]
        return [workflow_run(workflow_path, status="queued", conclusion=None)]

    api.list_workflow_runs = list_workflow_runs

    with pytest.raises(HANDLER.CommentCommandError, match="state changed"):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == []


def test_rerun_stops_after_partial_failure_without_retry_or_rollback():
    # [FUNCTION DOC] Two workflows have failed runs 10 and 20. The rerun of run 20 raises a timeout
    # error. The command raises an error naming workflow run 20, and the rerun attempts are exactly
    # [10, 20]: no retry, no rollback.
    api = FakeAPI(pull())
    for index, (workflow_file, workflow_path) in enumerate(HANDLER.RERUN_WORKFLOWS[:2]):
        api.workflow_runs[workflow_file] = [workflow_run(workflow_path, run_id=(index + 1) * 10)]

    def rerun_failed_jobs(run_id):
        # [FUNCTION DOC] HELPER: Record the rerun; raise the comment command error "GitHub API request
        # timed out" for run 20.
        api.rerun_calls.append(run_id)
        if run_id == 20:
            raise HANDLER.CommentCommandError("GitHub API request timed out")

    api.rerun_failed_jobs = rerun_failed_jobs

    with pytest.raises(HANDLER.CommentCommandError, match="workflow run 20"):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == [10, 20]


@pytest.mark.parametrize(
    ("bad_event", "message"),
    [
        ({**event(), "sender": {"id": 2}}, "sender does not match"),
        (
            {
                **event(),
                "comment": {**event()["comment"], "user": {"id": ACTOR_ID, "type": "Bot"}},
            },
            "human GitHub user",
        ),
        (
            {
                **event(),
                "comment": {**event()["comment"], "user": {"id": ACTOR_ID, "type": "User"}},
            },
            "login is missing",
        ),
        ({**event(), "repository": {"id": 1, "full_name": "attacker/repo"}}, "event repository"),
    ],
)
def test_untrusted_event_identity_fails_before_api_access(bad_event, message):
    # [FUNCTION DOC] Parametrized over events with a sender id differing from the commenter, a bot
    # commenter, a commenter without a login, and a foreign repository. Processing raises the
    # matching error (sender mismatch, human user required, login missing, event repository) before
    # any API call.
    api = FakeAPI(pull())
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.process_event(bad_event, policy(), api)
    assert api.calls == []


@pytest.mark.parametrize(
    ("permission_result", "message"),
    [
        ([], "invalid repository permission"),
        ({"permission": "write"}, "invalid repository permission identity"),
        ({"permission": "write", "user": {"id": True}}, "invalid repository permission identity"),
        ({"permission": "write", "user": {"id": 2}}, "identity does not match"),
        ({"permission": None, "user": {"id": ACTOR_ID}}, "invalid repository permission"),
    ],
)
def test_invalid_permission_response_fails_before_mutation(permission_result, message):
    # [FUNCTION DOC] Parametrized over malformed permission answers (a list, missing user, boolean
    # user id, a different user id, none permission). Processing raises the matching error; only the
    # pull read and the permission lookup were made; nothing was added.
    api = FakeAPI(pull())
    api.permission = permission_result

    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.process_event(event(), policy(), api)

    assert api.calls == [("get_pull", 123), ("get_permission", "actor")]
    assert api.add_calls == []


@pytest.mark.parametrize(
    ("bad_pull", "message"),
    [
        (pull(state="closed"), "not open"),
        ({**pull(), "head": {"repo": None}}, "head repository is missing"),
        ({**pull(), "base": {"repo": {"id": 1, "full_name": "attacker/repo"}}}, "base repository"),
        ({**pull(), "labels": None}, "labels are invalid"),
    ],
)
def test_unverifiable_live_pull_fails_before_mutation(bad_pull, message):
    # [FUNCTION DOC] Parametrized over a closed pull request, a pull request with a missing head
    # repository, a wrong base repository, and null labels. Processing raises the matching error
    # after exactly one API call (the pull read).
    api = FakeAPI(bad_pull)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.process_event(event(), policy(), api)
    assert api.calls == [("get_pull", 123)]


def test_github_api_uses_only_fixed_repository_and_additive_endpoint(monkeypatch):
    # [FUNCTION DOC] Replaces URL opening with a recorder returning a label list body, then adds a
    # label through the real client. Asserts the single request is a POST to the fixed repository's
    # issues labels endpoint for pull 123, with a JSON body of one label, a bearer authorization
    # header, and a 15 second timeout.
    requests = []

    class Response:
        # [CLASS DOC] HELPER: Fake HTTP response, context manager, whose read returns a JSON list with
        # one label entry.
        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the fixed JSON body bytes.
            return b'[{"name":"run-ci-short"}]'

    def urlopen(request, *, timeout):
        # [FUNCTION DOC] HELPER: Record (request, timeout) and return a response.
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    api = HANDLER.GitHubAPI("secret-token")
    api.add_label(123, "run-ci-short")

    request, timeout = requests[0]
    assert request.full_url == "https://api.github.com/repos/radixark/miles/issues/123/labels"
    assert request.method == "POST"
    assert json.loads(request.data) == {"labels": ["run-ci-short"]}
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert timeout == 15


def test_remove_label_uses_encoded_name_in_a_fixed_repository_path(monkeypatch):
    # [FUNCTION DOC] Removes a label whose name contains slash, space, question mark, hash and
    # percent through the real client with a recording opener. Asserts the request is a DELETE to the
    # fixed repository's labels path with the name fully percent-encoded, no body, and a 15 second
    # timeout.
    requests = []

    class Response:
        # [CLASS DOC] HELPER: Fake HTTP response, context manager, whose read returns an empty JSON
        # list.
        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the bytes of an empty JSON list.
            return b"[]"

    def urlopen(request, *, timeout):
        # [FUNCTION DOC] HELPER: Record (request, timeout) and return a response.
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    HANDLER.GitHubAPI("secret-token").remove_label(123, "run-ci-a/b ?#%")

    request, timeout = requests[0]
    assert request.full_url == (
        "https://api.github.com/repos/radixark/miles/issues/123/labels/" "run-ci-a%2Fb%20%3F%23%25"
    )
    assert request.method == "DELETE"
    assert request.data is None
    assert timeout == 15


def test_list_workflow_runs_uses_fixed_filters_and_complete_pagination(monkeypatch):
    # [FUNCTION DOC] The fake opener serves page 1 with 100 runs and page 2 with one run, both
    # reporting a total of 101. Listing runs of pr-test.yml for the sample sha must return all 101
    # items in order using exactly two GET requests, each with a 15 second timeout, to the fixed
    # repository's workflow runs path with query parameters event=pull_request, head_sha, page
    # (1, 2) and per_page=100.
    requests = []

    class Response:
        # [CLASS DOC] HELPER: Fake response holding a payload that is serialized to JSON on read.
        def __init__(self, payload):
            # [FUNCTION DOC] HELPER: Store the payload.
            self.payload = payload

        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the payload serialized to JSON bytes.
            return json.dumps(self.payload).encode()

    def urlopen(request, *, timeout):
        # [FUNCTION DOC] HELPER: Record (request, timeout); parse the page query parameter and return
        # a response with 100 runs for page 1 and one run (total 101) for any other page.
        requests.append((request, timeout))
        page = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)["page"][0]
        if page == "1":
            return Response({"total_count": 101, "workflow_runs": list(range(100))})
        return Response({"total_count": 101, "workflow_runs": [100]})

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    runs = HANDLER.GitHubAPI("secret-token").list_workflow_runs("pr-test.yml", HEAD_SHA)

    assert runs == list(range(101))
    assert len(requests) == 2
    for index, (request, timeout) in enumerate(requests, start=1):
        parsed = urllib.parse.urlparse(request.full_url)
        assert parsed.path == "/repos/radixark/miles/actions/workflows/pr-test.yml/runs"
        assert urllib.parse.parse_qs(parsed.query) == {
            "event": ["pull_request"],
            "head_sha": [HEAD_SHA],
            "page": [str(index)],
            "per_page": ["100"],
        }
        assert request.method == "GET"
        assert timeout == 15


@pytest.mark.parametrize(
    ("pages", "message"),
    [
        ([{"total_count": True, "workflow_runs": []}], "invalid workflow-run count"),
        ([{"total_count": 1001, "workflow_runs": []}], "invalid workflow-run count"),
        ([{"total_count": 1, "workflow_runs": []}], "incomplete workflow-run listing"),
        (
            [
                {"total_count": 101, "workflow_runs": list(range(100))},
                {"total_count": 102, "workflow_runs": [100]},
            ],
            "count changed",
        ),
    ],
)
def test_list_workflow_runs_fails_closed_on_invalid_pagination(monkeypatch, pages, message):
    # [FUNCTION DOC] Parametrized over page sequences: a boolean total count, a total above the
    # allowed maximum (1001), a total larger than the runs actually returned, and a total that changes
    # between pages. Opening yields the pages in order. Listing raises the comment command error
    # matching the message (invalid count, incomplete listing, count changed).
    class Response:
        # [CLASS DOC] HELPER: Fake response holding a payload that is serialized to JSON on read.
        def __init__(self, payload):
            # [FUNCTION DOC] HELPER: Store the payload.
            self.payload = payload

        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the payload serialized to JSON bytes.
            return json.dumps(self.payload).encode()

    page_iterator = iter(pages)
    monkeypatch.setattr(
        HANDLER.urllib.request,
        "urlopen",
        lambda _request, timeout: Response(next(page_iterator)),
    )

    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").list_workflow_runs("pr-test.yml", HEAD_SHA)


def test_list_pulls_for_head_uses_encoded_all_state_query(monkeypatch):
    # [FUNCTION DOC] Listing pulls for a fork owner and a branch name containing a space returns an
    # empty list and sends one GET to the fixed repository's pulls path with query head
    # "<owner>:<branch>", page 1, per_page 100 and state all, and a 15 second timeout.
    requests = []

    class Response:
        # [CLASS DOC] HELPER: Fake HTTP response, context manager, whose read returns an empty JSON
        # list.
        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the bytes of an empty JSON list.
            return b"[]"

    def urlopen(request, *, timeout):
        # [FUNCTION DOC] HELPER: Record (request, timeout) and return a response.
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    assert HANDLER.GitHubAPI("secret-token").list_pulls_for_head("fork-owner", "feature/a b") == []

    request, timeout = requests[0]
    parsed = urllib.parse.urlparse(request.full_url)
    assert parsed.path == "/repos/radixark/miles/pulls"
    assert urllib.parse.parse_qs(parsed.query) == {
        "head": ["fork-owner:feature/a b"],
        "page": ["1"],
        "per_page": ["100"],
        "state": ["all"],
    }
    assert request.method == "GET"
    assert timeout == 15


def test_rerun_failed_jobs_uses_exact_endpoint_and_accepts_empty_201(monkeypatch):
    # [FUNCTION DOC] The recording opener answers with status 201 and an empty body. The real client's
    # rerun of run 987 must accept it and have sent exactly one POST to the fixed repository's
    # actions runs rerun-failed-jobs endpoint with no request body and a 15 second timeout.
    requests = []

    class Response:
        # [CLASS DOC] HELPER: Fake HTTP response with status 201 and an empty body, as a context
        # manager.
        status = 201

        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return empty bytes.
            return b""

    def urlopen(request, *, timeout):
        # [FUNCTION DOC] HELPER: Record (request, timeout) and return a response.
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    HANDLER.GitHubAPI("secret-token").rerun_failed_jobs(987)

    request, timeout = requests[0]
    assert request.full_url == ("https://api.github.com/repos/radixark/miles/actions/runs/987/rerun-failed-jobs")
    assert request.method == "POST"
    assert request.data is None
    assert timeout == 15


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [(202, b"", "expected 201"), (201, b"{}", "unexpected response body")],
)
def test_rerun_failed_jobs_rejects_unconfirmed_response_without_retry(monkeypatch, status, body, message):
    # [FUNCTION DOC] Parametrized over (202 with empty body -> error "expected 201") and (201 with a
    # non-empty JSON body -> error "unexpected response body"). The rerun call raises the comment
    # command error matching the message and the opener was called exactly once (no retry).
    attempts = 0

    class Response:
        # [CLASS DOC] HELPER: Fake context-manager response returning the parametrized body; the
        # status attribute is set on the instance afterwards.
        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the parametrized body bytes.
            return body

    response = Response()
    response.status = status

    def urlopen(_request, *, timeout):
        # [FUNCTION DOC] HELPER: Count the call, assert the 15 second timeout, and return the shared
        # response.
        nonlocal attempts
        attempts += 1
        assert timeout == 15
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").rerun_failed_jobs(987)
    assert attempts == 1


def test_permission_lookup_encodes_the_login_in_a_fixed_repository_path(monkeypatch):
    # [FUNCTION DOC] Looks up the permission of a login containing slashes and parent-directory
    # segments through the real client with a recording opener. Asserts a single GET to the fixed
    # repository's collaborators path with the login fully percent-encoded (slashes as %2F) followed
    # by the permission suffix, and a 15 second timeout.
    requests = []

    class Response:
        # [CLASS DOC] HELPER: Fake context-manager response returning a write permission for user
        # 1234.
        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the fixed permission JSON bytes.
            return b'{"permission":"write","user":{"id":1234}}'

    def urlopen(request, *, timeout):
        # [FUNCTION DOC] HELPER: Record (request, timeout) and return a response.
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    HANDLER.GitHubAPI("secret-token").get_permission("actor/../../attacker")

    request, timeout = requests[0]
    assert request.full_url == (
        "https://api.github.com/repos/radixark/miles/collaborators/actor%2F..%2F..%2Fattacker/permission"
    )
    assert request.method == "GET"
    assert timeout == 15


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (urllib.error.HTTPError("url", 403, "forbidden", {}, None), "HTTP 403"),
        (urllib.error.HTTPError("url", 404, "missing", {}, None), "HTTP 404"),
        (TimeoutError(), "timed out"),
    ],
)
def test_permission_api_failure_is_not_retried(monkeypatch, exception, message):
    # [FUNCTION DOC] Parametrized over opener failures (HTTP 403, HTTP 404, a timeout). The
    # permission lookup raises the comment command error whose message matches (status or "timed
    # out") and the opener was called exactly once with a 15 second timeout.
    attempts = []

    def urlopen(_request, *, timeout):
        # [FUNCTION DOC] HELPER: Record the timeout, then raise the parametrized exception.
        attempts.append(timeout)
        raise exception

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").get_permission("actor")
    assert attempts == [15]


def test_permission_api_invalid_json_is_not_retried(monkeypatch):
    # [FUNCTION DOC] The opener returns a non-JSON body. The permission lookup raises the comment
    # command error "invalid JSON" and the opener was called exactly once.
    attempts = 0

    class Response:
        # [CLASS DOC] HELPER: Fake context-manager response returning non-JSON bytes.
        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the bytes "not-json".
            return b"not-json"

    def urlopen(_request, *, timeout):
        # [FUNCTION DOC] HELPER: Count the call, assert the 15 second timeout, return a response.
        nonlocal attempts
        attempts += 1
        assert timeout == 15
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match="invalid JSON"):
        HANDLER.GitHubAPI("secret-token").get_permission("actor")
    assert attempts == 1


@pytest.mark.parametrize("operation", ["add", "remove"])
@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (urllib.error.HTTPError("url", 403, "forbidden", {}, None), "HTTP 403"),
        (TimeoutError(), "timed out"),
    ],
)
def test_label_mutation_api_failure_is_not_retried(monkeypatch, operation, exception, message):
    # [FUNCTION DOC] Parametrized over add or remove label and an opener failure (HTTP 403 or a
    # timeout). The call raises the comment command error matching the message and the opener ran
    # exactly once with a 15 second timeout.
    attempts = []

    def urlopen(_request, *, timeout):
        # [FUNCTION DOC] HELPER: Record the timeout, then raise the parametrized exception.
        attempts.append(timeout)
        raise exception

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        api = HANDLER.GitHubAPI("secret-token")
        if operation == "add":
            api.add_label(123, "run-ci-short")
        else:
            api.remove_label(123, "run-ci-short")
    assert attempts == [15]


def test_http_error_names_the_request_and_accepted_permissions(monkeypatch):
    # [FUNCTION DOC] The opener raises an HTTP 403 whose headers list accepted GitHub permissions.
    # Adding a label raises the comment command error with exactly the message "GitHub API returned
    # HTTP 403 for POST /repos/radixark/miles/issues/123/labels; accepted permissions:
    # pull_requests=write", which does not contain the token.
    def urlopen(_request, *, timeout):
        # [FUNCTION DOC] HELPER: Always raise an HTTP 403 error carrying the accepted-permissions
        # header.
        raise urllib.error.HTTPError(
            "url", 403, "forbidden", {"X-Accepted-GitHub-Permissions": "pull_requests=write"}, None
        )

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError) as excinfo:
        HANDLER.GitHubAPI("secret-token").add_label(123, "run-ci-short")
    message = str(excinfo.value)
    assert message == (
        "GitHub API returned HTTP 403 for POST /repos/radixark/miles/issues/123/labels; "
        "accepted permissions: pull_requests=write"
    )
    assert "secret-token" not in message


def test_http_error_without_accepted_permissions_still_names_the_request(monkeypatch):
    # [FUNCTION DOC] The opener raises HTTP 404 with no permission header. The permission lookup
    # raises an error whose message is exactly "GitHub API returned HTTP 404 for GET
    # /repos/radixark/miles/collaborators/actor/permission".
    def urlopen(_request, *, timeout):
        # [FUNCTION DOC] HELPER: Always raise an HTTP 404 error without headers.
        raise urllib.error.HTTPError("url", 404, "missing", {}, None)

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError) as excinfo:
        HANDLER.GitHubAPI("secret-token").get_permission("actor")
    assert str(excinfo.value) == (
        "GitHub API returned HTTP 404 for GET /repos/radixark/miles/collaborators/actor/permission"
    )


def test_unconfirmed_mutation_response_fails_without_rollback():
    # [FUNCTION DOC] The fake add-label returns an empty label list (the label is not confirmed).
    # Processing raises the error "did not confirm".
    api = FakeAPI(pull())
    api.add_label = lambda _pull_number, _label: []
    with pytest.raises(HANDLER.CommentCommandError, match="did not confirm"):
        HANDLER.process_event(event(), policy(), api)


def test_unconfirmed_label_removal_fails_without_rollback():
    # [FUNCTION DOC] The fake remove-label returns the unchanged labels (still containing the CI
    # label). Clearing raises the error "did not confirm removal".
    api = FakeAPI(pull(labels=("run-ci-short", "documentation")))
    api.remove_label = lambda _pull_number, _label: [
        {"name": "run-ci-short"},
        {"name": "documentation"},
    ]

    with pytest.raises(HANDLER.CommentCommandError, match="did not confirm removal"):
        HANDLER.process_event(event(body="/clear-labels"), policy(), api)


def test_clear_stops_after_partial_failure_without_retry_or_rollback():
    # [FUNCTION DOC] Removal of run-ci-b raises a timeout error. Clearing raises an error "could not
    # remove CI label run-ci-b"; removals were attempted for run-ci-a then run-ci-b only; the pull
    # request keeps run-ci-b and documentation (run-ci-a stays removed, no rollback).
    api = FakeAPI(pull(labels=("run-ci-a", "run-ci-b", "documentation")))

    def remove_label(pull_number, label):
        # [FUNCTION DOC] HELPER: Record the call; raise the comment command error "GitHub API request
        # timed out" for run-ci-b; otherwise remove the label from the stored pull and return the
        # remaining labels.
        api.calls.append(("remove_label", pull_number, label))
        api.remove_calls.append((pull_number, label))
        if label == "run-ci-b":
            raise HANDLER.CommentCommandError("GitHub API request timed out")
        api.pull["labels"] = [item for item in api.pull["labels"] if item["name"] != label]
        return api.pull["labels"]

    api.remove_label = remove_label

    with pytest.raises(HANDLER.CommentCommandError, match="could not remove CI label run-ci-b"):
        HANDLER.process_event(event(body="/clear-labels"), policy(), api)

    assert api.remove_calls == [(123, "run-ci-a"), (123, "run-ci-b")]
    assert api.pull["labels"] == [{"name": "run-ci-b"}, {"name": "documentation"}]


def test_clear_rejects_a_final_response_with_a_new_ci_control_label():
    # [FUNCTION DOC] The fake remove-label returns a final label list containing a different CI
    # control label (nightly). Clearing raises an error saying not all CI labels were removed.
    api = FakeAPI(pull(labels=("run-ci-short",)))
    api.remove_label = lambda _pull_number, _label: [{"name": "nightly"}]

    with pytest.raises(HANDLER.CommentCommandError, match="all CI labels were removed"):
        HANDLER.process_event(event(body="/clear-labels"), policy(), api)


@pytest.mark.parametrize(
    ("body", "capability", "success_reaction"),
    [
        ("/run-ci-short", "issues", "none"),
        ("/bypass-fastfail", "issues", "none"),
        ("/clear-labels", "issues", "none"),
        ("/rerun-failed-ci", "actions", "none"),
        (RUN_FILE_BODY, "actions", "+1"),
    ],
)
def test_preflight_writes_only_fixed_routing(monkeypatch, tmp_path, body, capability, success_reaction):
    # [FUNCTION DOC] Parametrized over command bodies with their expected capability (issues for
    # label commands, actions for rerun and file run) and success reaction (none, or +1 for file
    # run). Replaces event loading, policy loading and the API client constructor, sets the event
    # path, policy path, token, preflight flag and output file environment variables, and runs the
    # handler main. Asserts exit code 0, the output file is exactly "capability=<c>" and
    # "success_reaction=<r>" lines, and the only API call was the permission lookup.
    api = FakeAPI(pull())
    output_path = tmp_path / "github-output"
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body=body))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: policy())
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_POLICY_PATH", "policy.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "token")
    monkeypatch.setenv("CI_COMMAND_PREFLIGHT", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 0
    assert output_path.read_text() == (f"capability={capability}\nsuccess_reaction={success_reaction}\n")
    assert api.calls == [("get_permission", "actor")]


def test_preflight_rejects_an_unknown_registry_capability(monkeypatch, tmp_path):
    # [FUNCTION DOC] Temporarily replaces the add-label registry entry with one whose capability is
    # "unknown" and runs preflight for a label comment. Asserts main exits 1 and writes no output
    # file.
    api = FakeAPI(pull())
    output_path = tmp_path / "github-output"
    spec = HANDLER.COMMAND_REGISTRY[HANDLER.AddLabel]
    monkeypatch.setitem(HANDLER.COMMAND_REGISTRY, HANDLER.AddLabel, spec._replace(capability="unknown"))
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event())
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: policy())
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_POLICY_PATH", "policy.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "token")
    monkeypatch.setenv("CI_COMMAND_PREFLIGHT", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 1
    assert not output_path.exists()


def test_preflight_rejects_an_unknown_registry_success_reaction(monkeypatch, tmp_path):
    # [FUNCTION DOC] Temporarily replaces the run-test-file registry entry with one whose success
    # reaction is "heart" and runs preflight for a file-run comment. Asserts main exits 1 and writes
    # no output file.
    api = FakeAPI(pull())
    output_path = tmp_path / "github-output"
    spec = HANDLER.COMMAND_REGISTRY[HANDLER.RunTestFile]
    monkeypatch.setitem(
        HANDLER.COMMAND_REGISTRY,
        HANDLER.RunTestFile,
        spec._replace(success_reaction="heart"),
    )
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body=RUN_FILE_BODY))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: policy())
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_POLICY_PATH", "policy.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "token")
    monkeypatch.setenv("CI_COMMAND_PREFLIGHT", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 1
    assert not output_path.exists()


def test_unrelated_comment_preflight_uses_no_policy_or_api(monkeypatch, tmp_path):
    # [FUNCTION DOC] For an ordinary comment, preflight (with policy loading and API construction
    # set to fail the test) exits 0 and writes "capability=none" and "success_reaction=none".
    output_path = tmp_path / "github-output"
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body="ordinary review comment"))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: pytest.fail("policy must not be loaded"))
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: pytest.fail("API must not be initialized"))
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_PREFLIGHT", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 0
    assert output_path.read_text() == "capability=none\nsuccess_reaction=none\n"


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_dispatches_a_file_run(permission):
    # [FUNCTION DOC] A write or admin actor's file-run comment dispatches exactly one workflow
    # (run-ci-file.yml on ref main) with inputs pull number "123", head sha and the test file, and
    # returns ALLOW_FILE_RUN_DISPATCHED with actor id, head sha, pull number, test file and the
    # workflow run url.
    api = FakeAPI(pull(), permission=permission)

    result = HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)

    assert api.dispatch_calls == [
        (
            "run-ci-file.yml",
            "main",
            {"pull_number": "123", "head_sha": HEAD_SHA, "test_file": RUN_FILE_PATH},
        )
    ]
    assert result == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_FILE_RUN_DISPATCHED",
        "head_sha": HEAD_SHA,
        "pull_number": 123,
        "test_file": RUN_FILE_PATH,
        "workflow_run_url": WORKFLOW_RUN_URL,
    }


def test_file_run_main_reports_the_dispatched_run_without_writing_an_output(monkeypatch, tmp_path, capsys):
    # [FUNCTION DOC] Runs the handler main for a file-run comment with patched loaders and API.
    # Asserts exit 0, the audit record printed to standard output contains the dispatched run url,
    # and no GitHub output file was written (the dispatched run owns the status comment).
    api = FakeAPI(pull())
    output_path = tmp_path / "github-output"
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body=RUN_FILE_BODY))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: policy())
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_POLICY_PATH", "policy.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "actions-token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 0
    # The dispatched run URL stays in the audit record; the dispatched run now
    # owns the pull-request status comment, so no job output carries it.
    assert json.loads(capsys.readouterr().out)["workflow_run_url"] == WORKFLOW_RUN_URL
    assert not output_path.exists()


@pytest.mark.parametrize("bridge_ref", ["#41", "bridge", "v0.7.0", "8cd3466d14d2337c8492827b3712482c2b3e4866"])
def test_file_run_forwards_validated_pr_body_pins(bridge_ref):
    # [FUNCTION DOC] Parametrized over the Megatron-Bridge pin value (a PR number, a branch, a tag,
    # or a commit sha). The pull request body contains lines for image tag pr-42, Megatron pin #77,
    # SGLang pin feature/pin-x and the bridge pin. The dispatched inputs contain the basic three plus
    # ci_image_tag, ci_megatron_pr, ci_sglang_pr and ci_megatron_bridge_pr with those values.
    target = pull()
    target["body"] = (
        "Summary line\nci-image-tag: pr-42\nci-megatron-pr: #77\nci-sglang-pr: feature/pin-x\n"
        f"ci-megatron-bridge-pr: {bridge_ref}\n"
    )
    api = FakeAPI(target)

    HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)

    (_, _, inputs) = api.dispatch_calls[0]
    assert inputs == {
        "pull_number": "123",
        "head_sha": HEAD_SHA,
        "test_file": RUN_FILE_PATH,
        "ci_image_tag": "pr-42",
        "ci_megatron_pr": "#77",
        "ci_sglang_pr": "feature/pin-x",
        "ci_megatron_bridge_pr": bridge_ref,
    }


@pytest.mark.parametrize(
    "line",
    ["ci-image-tag: -bad", "ci-megatron-pr: $(reboot)", "ci-sglang-pr: bad;ref", "ci-megatron-bridge-pr: $(reboot)"],
)
def test_file_run_rejects_an_invalid_pr_body_pin(line):
    # [FUNCTION DOC] Parametrized over a pull request body line with an unsafe value (a tag starting
    # with a hyphen, command substitution, a semicolon). Processing raises an error "unsupported
    # value" and dispatches nothing.
    target = pull()
    target["body"] = f"{line}\n"
    api = FakeAPI(target)

    with pytest.raises(HANDLER.CommentCommandError, match="unsupported value"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)
    assert api.dispatch_calls == []


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_fork_file_run_without_contributor_tier_or_write_is_denied(permission):
    # [FUNCTION DOC] For a fork pull request, an actor with read, triage or no permission and the
    # default author association NONE gets "not authorized" and nothing is dispatched.
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1), permission=permission)

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)
    assert api.dispatch_calls == []


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_dispatches_a_fork_file_run(permission):
    # [FUNCTION DOC] A write or admin actor dispatches a file run for a fork pull request: the head
    # pull is looked up once by fork owner and branch, exactly one dispatch of run-ci-file.yml on
    # main with the three basic inputs occurs, and the decision is ALLOW_FILE_RUN_DISPATCHED.
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1), permission=permission)

    result = HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)

    assert api.list_pull_calls == [("fork-owner", HEAD_REF)]
    assert api.dispatch_calls == [
        (
            "run-ci-file.yml",
            "main",
            {"pull_number": "123", "head_sha": HEAD_SHA, "test_file": RUN_FILE_PATH},
        )
    ]
    assert result["decision"] == "ALLOW_FILE_RUN_DISPATCHED"


@pytest.mark.parametrize("author_association", ["OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"])
def test_prior_contributor_dispatches_a_fork_file_run_without_a_permission_lookup(author_association):
    # [FUNCTION DOC] Parametrized over author associations OWNER, MEMBER, COLLABORATOR, CONTRIBUTOR
    # with permission none: the fork file run is dispatched without any permission lookup, after one
    # head pull lookup.
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1), permission="none")

    result = HANDLER.process_event(event(body=RUN_FILE_BODY, author_association=author_association), policy(), api)

    assert api.permission_calls == []
    assert api.list_pull_calls == [("fork-owner", HEAD_REF)]
    assert result["decision"] == "ALLOW_FILE_RUN_DISPATCHED"


@pytest.mark.parametrize("author_association", ["FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE", "MANNEQUIN"])
def test_first_time_contributor_cannot_dispatch_a_fork_file_run(author_association):
    # [FUNCTION DOC] Parametrized over author associations FIRST_TIME_CONTRIBUTOR, FIRST_TIMER, NONE,
    # MANNEQUIN with permission read: the fork file run gets "not authorized" and nothing is
    # dispatched.
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1), permission="read")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=RUN_FILE_BODY, author_association=author_association), policy(), api)
    assert api.dispatch_calls == []


def test_fork_file_run_requires_one_unique_head_pull():
    # [FUNCTION DOC] When the head pull lookup returns nothing for a fork pull request, the file run
    # raises an error saying exactly one pull request is required and dispatches nothing.
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1))
    api.head_pulls = []

    with pytest.raises(HANDLER.CommentCommandError, match="exactly one pull request"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)
    assert api.dispatch_calls == []


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_file_run_requires_live_write_permission(permission):
    # [FUNCTION DOC] For a same-repository pull request, an actor with read, triage or none
    # permission (association NONE) gets "not authorized" and nothing is dispatched.
    api = FakeAPI(pull(), permission=permission)

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)
    assert api.dispatch_calls == []


def test_add_label_access_user_id_cannot_dispatch_a_file_run():
    # [FUNCTION DOC] An actor listed only in the add-label group (permission read) gets "not
    # authorized" for a file run and nothing is dispatched.
    api = FakeAPI(pull(), permission="read")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(user_ids=(ACTOR_ID,)), api)
    assert api.dispatch_calls == []


@pytest.mark.parametrize("author_association", ["OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"])
def test_prior_contributor_dispatches_a_file_run_without_a_permission_lookup(author_association):
    # [FUNCTION DOC] Parametrized over associations OWNER, MEMBER, COLLABORATOR, CONTRIBUTOR with
    # permission none: a same-repository file run is dispatched with the three basic inputs and no
    # permission lookup; the decision is ALLOW_FILE_RUN_DISPATCHED.
    api = FakeAPI(pull(), permission="none")

    result = HANDLER.process_event(
        event(body=RUN_FILE_BODY, author_association=author_association),
        policy(),
        api,
    )

    assert api.permission_calls == []
    assert api.dispatch_calls == [
        (
            "run-ci-file.yml",
            "main",
            {"pull_number": "123", "head_sha": HEAD_SHA, "test_file": RUN_FILE_PATH},
        )
    ]
    assert result["decision"] == "ALLOW_FILE_RUN_DISPATCHED"


@pytest.mark.parametrize("author_association", ["FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE", "MANNEQUIN"])
def test_first_time_contributor_cannot_dispatch_a_file_run(author_association):
    # [FUNCTION DOC] Parametrized over associations FIRST_TIME_CONTRIBUTOR, FIRST_TIMER, NONE,
    # MANNEQUIN with permission read: "not authorized" and nothing dispatched.
    api = FakeAPI(pull(), permission="read")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(
            event(body=RUN_FILE_BODY, author_association=author_association),
            policy(),
            api,
        )
    assert api.dispatch_calls == []


def test_prior_contributor_access_user_id_dispatches_a_file_run():
    # [FUNCTION DOC] An actor with association FIRST_TIMER and permission none who is listed by id in
    # the prior contributor group dispatches the file run without a permission lookup.
    api = FakeAPI(pull(), permission="none")

    result = HANDLER.process_event(
        event(body=RUN_FILE_BODY, author_association="FIRST_TIMER"),
        policy(prior_user_ids=(ACTOR_ID,)),
        api,
    )

    assert api.permission_calls == []
    assert api.dispatch_calls == [
        (
            "run-ci-file.yml",
            "main",
            {"pull_number": "123", "head_sha": HEAD_SHA, "test_file": RUN_FILE_PATH},
        )
    ]
    assert result["decision"] == "ALLOW_FILE_RUN_DISPATCHED"


@pytest.mark.parametrize("body", ["/run-ci-short", "/clear-labels"])
def test_contributor_association_grants_no_label_command(body):
    # [FUNCTION DOC] Parametrized over an add-label and a clear-labels comment from an actor with
    # association CONTRIBUTOR and permission read: "not authorized", and no label added or removed,
    # no rerun, no dispatch (the contributor tier covers only rerun and file-run).
    api = FakeAPI(pull(labels=("run-ci-short",)), permission="read")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=body, author_association="CONTRIBUTOR"), policy(), api)

    assert api.add_calls == []
    assert api.remove_calls == []
    assert api.rerun_calls == []
    assert api.dispatch_calls == []


@pytest.mark.parametrize("body", ["/run-ci-short", "/clear-labels"])
def test_prior_contributor_access_user_id_grants_no_label_command(body):
    # [FUNCTION DOC] Parametrized over add-label and clear-labels comments from an actor listed only
    # in the prior contributor group (association FIRST_TIMER, permission none): "not authorized" and
    # no mutation of any kind.
    api = FakeAPI(pull(labels=("run-ci-short",)), permission="none")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(
            event(body=body, author_association="FIRST_TIMER"),
            policy(prior_user_ids=(ACTOR_ID,)),
            api,
        )

    assert api.add_calls == []
    assert api.remove_calls == []
    assert api.rerun_calls == []
    assert api.dispatch_calls == []


def test_contributor_reruns_failed_ci_without_a_permission_lookup():
    # [FUNCTION DOC] A CONTRIBUTOR actor (permission none) rerunning failed CI for a pull request
    # with one failed run (id 10) is allowed: no permission lookup, run 10 rerun, decision
    # ALLOW_RERUN_REQUESTED.
    api = FakeAPI(pull(), permission="none")
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path)]

    result = HANDLER.process_event(event(body="/rerun-failed-ci", author_association="CONTRIBUTOR"), policy(), api)

    assert api.permission_calls == []
    assert api.rerun_calls == [10]
    assert result["decision"] == "ALLOW_RERUN_REQUESTED"


def test_prior_contributor_access_user_id_reruns_failed_ci():
    # [FUNCTION DOC] An actor listed by id in the prior contributor group (association FIRST_TIMER,
    # permission none) reruns failed CI without a permission lookup: run 10 is rerun and the decision
    # is ALLOW_RERUN_REQUESTED.
    api = FakeAPI(pull(), permission="none")
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path)]

    result = HANDLER.process_event(
        event(body="/rerun-failed-ci", author_association="FIRST_TIMER"),
        policy(prior_user_ids=(ACTOR_ID,)),
        api,
    )

    assert api.permission_calls == []
    assert api.rerun_calls == [10]
    assert result["decision"] == "ALLOW_RERUN_REQUESTED"


@pytest.mark.parametrize("author_association", [None, True, "", "collaborator", "EVERYONE"])
def test_event_parser_rejects_an_invalid_author_association(author_association):
    # [FUNCTION DOC] Parametrized over a missing author association, a boolean, an empty string, a
    # lowercase value and an unknown value. Parsing the event raises the comment command error "author
    # association is invalid".
    invalid = event(body=RUN_FILE_BODY)
    if author_association is None:
        del invalid["comment"]["author_association"]
    else:
        invalid["comment"]["author_association"] = author_association

    with pytest.raises(HANDLER.CommentCommandError, match="author association is invalid"):
        HANDLER.parse_event(invalid)


def test_create_workflow_dispatch_requests_and_returns_exact_run_details(monkeypatch):
    # [FUNCTION DOC] The recording opener answers 200 with run details (id, api url, html url).
    # Dispatching run-ci-file.yml on ref feature/test with one input via the real client sends one
    # POST to the fixed repository's workflow dispatches endpoint whose JSON body has the ref, the
    # inputs and return_run_details true, with a 15 second timeout, and returns the html url.
    requests = []

    class Response:
        # [CLASS DOC] HELPER: Fake context-manager response with status 200 and a JSON body of run
        # details.
        status = 200

        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the JSON bytes of workflow_run_id, run_url and html_url.
            return json.dumps(
                {
                    "workflow_run_id": WORKFLOW_RUN_ID,
                    "run_url": WORKFLOW_RUN_API_URL,
                    "html_url": WORKFLOW_RUN_URL,
                }
            ).encode("utf-8")

    def urlopen(request, *, timeout):
        # [FUNCTION DOC] HELPER: Record (request, timeout) and return a response.
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    result = HANDLER.GitHubAPI("secret-token").create_workflow_dispatch(
        "run-ci-file.yml", "feature/test", {"test_file": "tests/e2e/test_a.py"}
    )

    request, timeout = requests[0]
    assert request.full_url == (
        "https://api.github.com/repos/radixark/miles/actions/workflows/run-ci-file.yml/dispatches"
    )
    assert request.method == "POST"
    assert json.loads(request.data.decode("utf-8")) == {
        "ref": "feature/test",
        "inputs": {"test_file": "tests/e2e/test_a.py"},
        "return_run_details": True,
    }
    assert timeout == 15
    assert result == WORKFLOW_RUN_URL


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [(204, b"", "expected 200"), (200, b"", "invalid JSON")],
)
def test_create_workflow_dispatch_rejects_unconfirmed_response(monkeypatch, status, body, message):
    # [FUNCTION DOC] Parametrized over (204 with empty body -> "expected 200") and (200 with empty
    # body -> "invalid JSON"). Dispatching raises the comment command error matching the message and
    # the opener ran exactly once (no retry).
    attempts = 0

    class Response:
        # [CLASS DOC] HELPER: Fake context-manager response returning the parametrized body.
        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the parametrized body bytes.
            return body

    response = Response()
    response.status = status

    def urlopen(_request, *, timeout):
        # [FUNCTION DOC] HELPER: Count the call, assert the 15 second timeout, return the shared
        # response.
        nonlocal attempts
        attempts += 1
        assert timeout == 15
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").create_workflow_dispatch("run-ci-file.yml", "feature/test", {})
    assert attempts == 1


@pytest.mark.parametrize(
    ("details", "message"),
    [
        ({}, "workflow run ID must be a positive integer"),
        (
            {
                "workflow_run_id": WORKFLOW_RUN_ID,
                "run_url": "https://api.github.com/repos/radixark/miles/actions/runs/1",
                "html_url": WORKFLOW_RUN_URL,
            },
            "mismatched workflow dispatch details",
        ),
        (
            {
                "workflow_run_id": WORKFLOW_RUN_ID,
                "run_url": WORKFLOW_RUN_API_URL,
                "html_url": "https://example.invalid/actions/runs/987654321",
            },
            "workflow run URL is invalid",
        ),
    ],
)
def test_create_workflow_dispatch_rejects_invalid_run_details(monkeypatch, details, message):
    # [FUNCTION DOC] Parametrized over 200 responses with bad details: missing run id (error "workflow
    # run ID must be a positive integer"), a run api url for a different run (mismatched details), and
    # an html url on a foreign host (invalid workflow run url). Dispatching raises the matching
    # comment command error.
    class Response:
        # [CLASS DOC] HELPER: Fake context-manager response with status 200 and the parametrized
        # details as JSON.
        status = 200

        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the details serialized as JSON bytes.
            return json.dumps(details).encode("utf-8")

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", lambda _request, timeout: Response())
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").create_workflow_dispatch("run-ci-file.yml", "main", {})


def test_create_issue_comment_uses_exact_endpoint_and_confirms_body(monkeypatch):
    # [FUNCTION DOC] The recording opener answers 201 with the created comment (body and id). The
    # real client's create call for pull 123 returns the new comment id and has sent one POST to the
    # fixed repository's issue comments endpoint with a JSON body holding only the body text, within
    # a 15 second timeout.
    requests = []

    class Response:
        # [CLASS DOC] HELPER: Fake context-manager response with status 201 and a JSON body of the
        # comment text and id.
        status = 201

        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the JSON bytes with the announce body and new comment id.
            return json.dumps({"body": ANNOUNCE_BODY, "id": NEW_COMMENT_ID}).encode("utf-8")

    def urlopen(request, *, timeout):
        # [FUNCTION DOC] HELPER: Record (request, timeout) and return a response.
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    assert HANDLER.GitHubAPI("secret-token").create_issue_comment(123, ANNOUNCE_BODY) == NEW_COMMENT_ID

    request, timeout = requests[0]
    assert request.full_url == "https://api.github.com/repos/radixark/miles/issues/123/comments"
    assert request.method == "POST"
    assert json.loads(request.data.decode("utf-8")) == {"body": ANNOUNCE_BODY}
    assert timeout == 15


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (200, {"body": ANNOUNCE_BODY, "id": NEW_COMMENT_ID}, "expected 201"),
        (201, {"body": "wrong", "id": NEW_COMMENT_ID}, "did not confirm"),
        (201, {"body": ANNOUNCE_BODY}, "comment ID"),
    ],
)
def test_create_issue_comment_rejects_unconfirmed_response_without_retry(
    monkeypatch,
    status,
    body,
    message,
):
    # [FUNCTION DOC] Parametrized over (status 200 -> "expected 201"), (201 with a different body ->
    # "did not confirm") and (201 without an id -> "comment ID"). Creating the comment raises the
    # matching comment command error and the opener ran exactly once.
    attempts = 0

    class Response:
        # [CLASS DOC] HELPER: Fake context-manager response returning the parametrized body as JSON.
        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the body mapping serialized as JSON bytes.
            return json.dumps(body).encode("utf-8")

    response = Response()
    response.status = status

    def urlopen(_request, *, timeout):
        # [FUNCTION DOC] HELPER: Count the call, assert the 15 second timeout, return the shared
        # response.
        nonlocal attempts
        attempts += 1
        assert timeout == 15
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").create_issue_comment(123, ANNOUNCE_BODY)
    assert attempts == 1


@pytest.mark.parametrize("status", [200, 201])
def test_add_comment_reaction_accepts_created_or_existing_reaction(monkeypatch, status):
    # [FUNCTION DOC] Parametrized over status 200 (reaction already existed) and 201 (created), both
    # with a body confirming content "+1". Adding the reaction to comment 5678 sends one POST to the
    # fixed repository's issue comment reactions endpoint with a JSON body of the content, within a 15
    # second timeout.
    requests = []

    class Response:
        # [CLASS DOC] HELPER: Fake context-manager response confirming a "+1" reaction; the status is
        # set on the instance afterwards.
        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the JSON bytes of the confirmed reaction.
            return b'{"content":"+1"}'

    response = Response()
    response.status = status

    def urlopen(request, *, timeout):
        # [FUNCTION DOC] HELPER: Record (request, timeout) and return the shared response.
        requests.append((request, timeout))
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    HANDLER.GitHubAPI("secret-token").add_comment_reaction(5678, "+1")

    request, timeout = requests[0]
    assert request.full_url == ("https://api.github.com/repos/radixark/miles/issues/comments/5678/reactions")
    assert request.method == "POST"
    assert json.loads(request.data.decode("utf-8")) == {"content": "+1"}
    assert timeout == 15


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (202, b'{"content":"+1"}', "expected 200 or 201"),
        (204, b"", "expected 200 or 201"),
        (201, b'{"content":"heart"}', "did not confirm"),
        (201, b"not-json", "invalid JSON"),
    ],
)
def test_add_comment_reaction_rejects_unconfirmed_response_without_retry(monkeypatch, status, body, message):
    # [FUNCTION DOC] Parametrized over (202, 204 -> "expected 200 or 201"), (201 with a different
    # reaction -> "did not confirm") and (201 with invalid JSON -> "invalid JSON"). Adding the
    # reaction raises the matching comment command error and the opener ran exactly once.
    attempts = 0

    class Response:
        # [CLASS DOC] HELPER: Fake context-manager response returning the parametrized body.
        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *_args):
            # [FUNCTION DOC] HELPER: Never suppress exceptions.
            return False

        def read(self):
            # [FUNCTION DOC] HELPER: Return the parametrized body bytes.
            return body

    response = Response()
    response.status = status

    def urlopen(_request, *, timeout):
        # [FUNCTION DOC] HELPER: Count the call, assert the 15 second timeout, return the shared
        # response.
        nonlocal attempts
        attempts += 1
        assert timeout == 15
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").add_comment_reaction(5678, "+1")
    assert attempts == 1


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (urllib.error.HTTPError("url", 403, "forbidden", {}, None), "HTTP 403"),
        (TimeoutError(), "timed out"),
    ],
)
def test_comment_reaction_api_failure_is_not_retried(monkeypatch, exception, message):
    # [FUNCTION DOC] Parametrized over opener failures (HTTP 403, a timeout). Adding a reaction raises
    # the matching comment command error and the opener ran exactly once with a 15 second timeout.
    attempts = []

    def urlopen(_request, *, timeout):
        # [FUNCTION DOC] HELPER: Record the timeout, then raise the parametrized exception.
        attempts.append(timeout)
        raise exception

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").add_comment_reaction(5678, "+1")
    assert attempts == [15]


def test_acknowledge_event_reacts_only_for_run_test_file():
    # [FUNCTION DOC] Acknowledging a file-run comment adds exactly one "+1" reaction to comment 5678
    # and returns the ALLOW_SUCCESS_REACTION_CONFIRMED record with actor id, pull number and the
    # reaction.
    api = FakeAPI(pull())

    result = HANDLER.acknowledge_event(event(body=RUN_FILE_BODY), api)

    assert api.reaction_calls == [(5678, "+1")]
    assert result == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_SUCCESS_REACTION_CONFIRMED",
        "pull_number": 123,
        "reaction": "+1",
    }


def test_acknowledge_event_rejects_commands_without_a_success_reaction():
    # [FUNCTION DOC] Acknowledging a rerun-failed-ci comment (whose spec has no success reaction)
    # raises an error saying the command does not define one and adds no reaction.
    api = FakeAPI(pull())

    with pytest.raises(HANDLER.CommentCommandError, match="does not define"):
        HANDLER.acknowledge_event(event(body="/rerun-failed-ci"), api)
    assert api.reaction_calls == []


@pytest.mark.parametrize("comment_id", [None, 0, "5678"])
def test_acknowledge_event_rejects_an_invalid_comment_id(comment_id):
    # [FUNCTION DOC] Parametrized over comment ids none, 0 and the string "5678". Acknowledging
    # raises an error that the comment ID must be a positive integer and adds no reaction.
    api = FakeAPI(pull())
    command_event = event(body=RUN_FILE_BODY)
    command_event["comment"]["id"] = comment_id

    with pytest.raises(HANDLER.CommentCommandError, match="comment ID must be a positive integer"):
        HANDLER.acknowledge_event(command_event, api)
    assert api.reaction_calls == []


def test_acknowledge_mode_reacts_without_loading_policy(monkeypatch, capsys):
    # [FUNCTION DOC] Runs the handler main in acknowledge mode (environment flag set, no policy path)
    # for a file-run comment, with policy loading set to fail the test. Asserts exit 0, the only API
    # call is the "+1" reaction on comment 5678, and the printed audit record is the confirmed
    # reaction record.
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body=RUN_FILE_BODY))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: pytest.fail("policy must not be loaded"))
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "reaction-token")
    monkeypatch.setenv("CI_COMMAND_ACKNOWLEDGE", "true")

    assert HANDLER.main() == 0
    assert api.calls == [("add_comment_reaction", 5678, "+1")]
    assert json.loads(capsys.readouterr().out) == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_SUCCESS_REACTION_CONFIRMED",
        "pull_number": 123,
        "reaction": "+1",
    }


def test_acknowledge_mode_rejects_a_command_without_a_success_reaction(monkeypatch):
    # [FUNCTION DOC] Runs acknowledge mode for a rerun comment; asserts main exits 1 and the API was
    # never called.
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body="/rerun-failed-ci"))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: pytest.fail("policy must not be loaded"))
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "reaction-token")
    monkeypatch.setenv("CI_COMMAND_ACKNOWLEDGE", "true")

    assert HANDLER.main() == 1
    assert api.calls == []


def test_announce_posts_a_running_status_comment_keyed_to_the_run(monkeypatch):
    # [FUNCTION DOC] With the clock fixed at the sample report time, announcing a file run creates one
    # comment on pull 123 whose body is the running message (file path, workflow run link, start time
    # formatted in UTC, and a hidden marker with the run id) and returns the record with the new
    # comment id, decision FILE_RUN_ANNOUNCED, pull number, run id and test file.
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)

    result = HANDLER.announce_file_run(api, file_run_status("announce"))

    pull_number, body = api.comment_calls[0]
    assert pull_number == 123
    assert body == ANNOUNCE_BODY
    assert result == {
        "comment_id": NEW_COMMENT_ID,
        "decision": "FILE_RUN_ANNOUNCED",
        "pull_number": 123,
        "run_id": WORKFLOW_RUN_ID,
        "test_file": RUN_FILE_PATH,
    }


def test_report_updates_the_announced_comment_with_the_result_and_duration(monkeypatch):
    # [FUNCTION DOC] With the clock fixed 3m11s after the run start, reporting a successful run edits
    # the existing comment (no new comment) to "<check mark> `<file>` **passed** on `<suite>` in
    # 3m11s", an em dash, a workflow run link and the hidden run marker, and returns the record with
    # decision FILE_RUN_PASSED, the duration and ids.
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)

    result = HANDLER.report_file_run(api, file_run_status(), EXISTING_COMMENT_ID)

    assert api.comment_calls == []
    comment_id, body = api.update_calls[0]
    assert comment_id == EXISTING_COMMENT_ID
    assert body == (
        f"\u2705 `{RUN_FILE_PATH}` **passed** on `stage-c-4-gpu-h200` in 3m11s "
        f"\u2014 [workflow run]({WORKFLOW_RUN_URL})\n\n{RUN_MARKER}"
    )
    assert result == {
        "comment_id": EXISTING_COMMENT_ID,
        "decision": "FILE_RUN_PASSED",
        "duration": "3m11s",
        "pull_number": 123,
        "run_id": WORKFLOW_RUN_ID,
        "test_file": RUN_FILE_PATH,
    }


@pytest.mark.parametrize(
    ("overrides", "decision", "icon", "explanation"),
    [
        (
            {"FILE_RUN_CUDA_RESULT": "failure"},
            "FILE_RUN_FAILED",
            "\u274c",
            "The execution job failed; inspect the workflow run for the failing step.",
        ),
        (
            {"FILE_RUN_RESOLVE_RESULT": "failure", "FILE_RUN_CUDA_RESULT": "skipped"},
            "FILE_RUN_FAILED",
            "\u274c",
            "The run never started: resolving the test file's execution plan failed.",
        ),
        (
            {"FILE_RUN_CUDA_RESULT": "cancelled"},
            "FILE_RUN_CANCELLED",
            "\u26aa",
            "The run was cancelled.",
        ),
        (
            {"FILE_RUN_RESOLVE_RESULT": "cancelled", "FILE_RUN_CUDA_RESULT": "skipped"},
            "FILE_RUN_CANCELLED",
            "\u26aa",
            "The run was cancelled.",
        ),
        (
            {"FILE_RUN_CUDA_RESULT": "skipped", "FILE_RUN_CPU_RESULT": "skipped"},
            "FILE_RUN_FAILED",
            "\u274c",
            "No execution job ran: the resolved plan selected neither the CUDA nor the CPU job.",
        ),
    ],
)
def test_report_never_calls_a_non_passing_run_a_pass(monkeypatch, overrides, decision, icon, explanation):
    # [FUNCTION DOC] Parametrized over job result overrides: execution job failed, resolve step failed
    # (execution skipped), execution cancelled, resolve cancelled, and both execution jobs skipped.
    # Asserts the edited comment starts with the expected icon (cross mark for failure, white circle
    # for cancelled), never contains "**passed**", contains the matching explanation sentence, and
    # the record's decision is FILE_RUN_FAILED or FILE_RUN_CANCELLED accordingly.
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)

    result = HANDLER.report_file_run(api, file_run_status(**overrides), EXISTING_COMMENT_ID)

    body = api.update_calls[0][1]
    assert body.startswith(icon)
    assert "**passed**" not in body
    assert explanation in body
    assert result["decision"] == decision


def test_report_reads_the_run_start_from_the_api_not_from_an_input(monkeypatch):
    # [FUNCTION DOC] The fake API's workflow run started an hour earlier than the sample. Reporting
    # fetches the run detail exactly once for the run id and the duration is "1h03m11s" (hours,
    # zero-padded minutes and seconds).
    api = FakeAPI(pull())
    api.workflow_run = {"run_started_at": "2026-08-23T08:25:25Z"}
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)

    result = HANDLER.report_file_run(api, file_run_status(), EXISTING_COMMENT_ID)

    assert api.run_calls == [WORKFLOW_RUN_ID]
    assert result["duration"] == "1h03m11s"


@pytest.mark.parametrize(
    "workflow_run",
    [
        {},
        {"created_at": RUN_STARTED_AT},
        {"run_started_at": ""},
        {"run_started_at": "not-a-timestamp"},
        {"run_started_at": "2026-08-23T09:25:25"},
        {"run_started_at": 17},
        "nope",
    ],
)
def test_report_fails_closed_on_an_unusable_run_start(monkeypatch, workflow_run):
    # [FUNCTION DOC] Parametrized over run details lacking a usable start: empty mapping, only a
    # created-at field, empty start, a non-timestamp, a timestamp without timezone marker, a number,
    # or a string instead of a mapping. Reporting raises the comment command error and neither edits
    # nor creates a comment.
    api = FakeAPI(pull())
    api.workflow_run = workflow_run
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)

    with pytest.raises(HANDLER.CommentCommandError):
        HANDLER.report_file_run(api, file_run_status(), EXISTING_COMMENT_ID)
    assert api.update_calls == []
    assert api.comment_calls == []


@pytest.mark.parametrize(
    ("seconds", "formatted"),
    [(0, "0s"), (9, "9s"), (59, "59s"), (60, "1m00s"), (191, "3m11s"), (3600, "1h00m00s"), (3791, "1h03m11s")],
)
def test_duration_formatting_is_stable(seconds, formatted):
    # [FUNCTION DOC] The private duration formatter renders 0 s, 9 s, 59 s as seconds only, 60 s as
    # "1m00s", 191 s as "3m11s", 3600 s as "1h00m00s" and 3791 s as "1h03m11s".
    assert HANDLER._format_duration(seconds) == formatted


def test_negative_duration_is_a_hard_error():
    # [FUNCTION DOC] Formatting a negative duration raises the comment command error mentioning
    # negative.
    with pytest.raises(HANDLER.CommentCommandError, match="negative"):
        HANDLER._format_duration(-1)


@pytest.mark.parametrize(
    ("mode", "overrides", "message"),
    [
        ("announce", {"CI_COMMAND_FILE_RUN_STATUS": "publish"}, "announce or report"),
        ("announce", {"FILE_RUN_PULL_NUMBER": "0"}, "pull request number"),
        ("announce", {"FILE_RUN_PULL_NUMBER": "12a"}, "pull request number"),
        ("announce", {"FILE_RUN_PULL_NUMBER": ""}, "pull request number"),
        ("announce", {"FILE_RUN_RUN_ID": "-1"}, "workflow run ID"),
        ("announce", {"FILE_RUN_TEST_FILE": "tests/../etc/passwd"}, "test file is invalid"),
        ("announce", {"FILE_RUN_TEST_FILE": ""}, "test file is invalid"),
        ("report", {"FILE_RUN_SUITE": "stage a cpu; rm -rf /"}, "suite is invalid"),
        ("report", {"FILE_RUN_RESOLVE_RESULT": "exploded"}, "resolve result is invalid"),
        ("report", {"FILE_RUN_CPU_RESULT": "success"}, "two executing jobs"),
        ("report", {"FILE_RUN_COMMENT_ID": ""}, "comment ID"),
        ("report", {"FILE_RUN_COMMENT_ID": "0"}, "comment ID"),
    ],
)
def test_file_run_status_inputs_fail_closed(mode, overrides, message):
    # [FUNCTION DOC] Parametrized over invalid environment values: an unknown mode, a zero /
    # non-numeric / empty pull number, a negative run id, a test file with parent traversal or empty,
    # an invalid suite name, an unknown resolve result, a CPU result alongside a CUDA result (two
    # executing jobs), and a missing or zero comment id in report mode. Parsing the environment
    # raises the comment command error matching the message.
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER._file_run_status_inputs(file_run_env(mode, **overrides))


def test_status_mode_needs_no_event_and_loads_no_policy(monkeypatch, capsys):
    # [FUNCTION DOC] Runs the handler main in report status mode with the event path unset and event
    # and policy loading set to fail the test; the token and all file-run variables are set. Asserts
    # exit 0 and the printed record has decision FILE_RUN_PASSED.
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: pytest.fail("no event payload may be read"))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: pytest.fail("policy must not be loaded"))
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "status-token")
    for name, value in file_run_env("report").items():
        monkeypatch.setenv(name, value)

    assert HANDLER.main() == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "FILE_RUN_PASSED"


def test_announce_mode_writes_the_created_comment_id(monkeypatch, tmp_path, capsys):
    # [FUNCTION DOC] Runs main in announce mode with a GitHub output file; asserts exit 0, the output
    # file holds exactly "comment_id=<new id>" and the printed record's decision is
    # FILE_RUN_ANNOUNCED.
    api = FakeAPI(pull())
    output_path = tmp_path / "github-output"
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "status-token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    for name, value in file_run_env("announce").items():
        monkeypatch.setenv(name, value)

    assert HANDLER.main() == 0
    assert output_path.read_text() == f"comment_id={NEW_COMMENT_ID}\n"
    assert json.loads(capsys.readouterr().out)["decision"] == "FILE_RUN_ANNOUNCED"


def test_status_mode_reports_a_handler_error_as_a_failed_step(monkeypatch, capsys):
    # [FUNCTION DOC] In report mode with an invalid test file value (and API construction set to
    # fail the test, proving inputs are validated first), main exits 1 and prints the GitHub error
    # annotation "file run test file is invalid".
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: pytest.fail("token is validated first"))
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "status-token")
    for name, value in file_run_env("report", FILE_RUN_TEST_FILE="not-a-test").items():
        monkeypatch.setenv(name, value)

    assert HANDLER.main() == 1
    assert "::error::file run test file is invalid" in capsys.readouterr().out


def test_the_gateway_no_longer_owns_a_reply_path():
    # [FUNCTION DOC] Asserts the handler module has no reply-event function and no workflow-run-url
    # writer: the dispatched run reports its own progress and result.
    # The run itself reports progress and the result; a second static comment
    # from the gateway would duplicate it.
    assert not hasattr(HANDLER, "reply_event")
    assert not hasattr(HANDLER, "_write_workflow_run_url")


def test_workflow_runs_only_trusted_code_with_minimal_permissions():
    # [FUNCTION DOC] Reads the comment-ci-command workflow text and asserts security invariants by
    # substring, count and ordering checks: triggers only on created issue comments and never
    # interpolates the comment body; read-only contents permission; checks out the workflow's own sha
    # pinned by action sha without persisting credentials; exactly one CI App token step (pinned
    # action, client id variable, private key secret), scoped to issues write; no actions permission
    # in that token; the token for the issues command comes from that step while the other jobs use
    # the job token; preflight precedes token minting; label commands fail loudly when the App is
    # disabled; routing outputs (capability, success reaction) drive three jobs (handle-command,
    # actions-command, acknowledge-command) with the actions job queued per issue number without
    # cancellation; the issues token step requests issues and pull-requests write only (no contents,
    # no pull-requests read); the App requirement check comes before minting; the actions job has
    # contents read, actions write and pull-requests read, no issues write and no App token; the
    # acknowledge job has issues and pull-requests write, no actions write and no App token, runs
    # after the actions job, only when it succeeded and the success reaction is "+1", never under an
    # always condition, and sets the acknowledge flag; no reply job or reply / run-url variables
    # remain; no pull_request_target trigger, head ref use, pip install, or contents / workflow
    # actions write permissions.
    workflow = WORKFLOW_PATH.read_text()
    assert "issue_comment:\n    types: [created]" in workflow
    assert "github.event.comment.body" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "ref: ${{ github.sha }}" in workflow
    assert "persist-credentials: false" in workflow
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in workflow
    # This gateway mints a CI App token only for the issues capability: label mutations
    # made with GITHUB_TOKEN would never trigger the labeled CI workflows.
    assert workflow.count("actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1") == 1
    assert workflow.count("client-id: ${{ vars.CI_APP_CLIENT_ID }}") == 1
    assert workflow.count("private-key: ${{ secrets.CI_APP_PRIVATE_KEY }}") == 1
    assert "permission-issues: write" in workflow
    assert "permission-actions" not in workflow
    assert "CI_COMMAND_API_TOKEN: ${{ github.token }}" in workflow
    assert "CI_COMMAND_API_TOKEN: ${{ steps.issues-token.outputs.token }}" in workflow
    assert "CI_APP_TOKEN" not in workflow
    assert workflow.index("CI_COMMAND_PREFLIGHT") < workflow.index("actions/create-github-app-token@")
    # The actions capability never waits on the App gate; label commands fail
    # loudly when the App is not enabled instead of skipping silently.
    assert "vars.CI_APP_ENABLED != 'true'" in workflow
    assert "vars.CI_APP_ENABLED == 'true'" not in workflow
    assert "steps.authorize.outputs.capability != 'none'" in workflow
    assert "steps.authorize.outputs.capability != 'issues'" in workflow
    assert "steps.authorize.outputs.capability != 'actions'" in workflow
    assert "capability: ${{ steps.authorize.outputs.capability }}" in workflow
    assert "success_reaction: ${{ steps.authorize.outputs.success_reaction }}" in workflow
    assert "needs: handle-command" in workflow
    assert "if: needs.handle-command.outputs.capability == 'actions'" in workflow
    assert "group: comment-ci-actions-${{ github.event.issue.number }}" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "queue: max" in workflow
    handle_job = workflow.split("  handle-command:", 1)[1].split("  actions-command:", 1)[0]
    actions_job = workflow.split("  actions-command:", 1)[1].split("  acknowledge-command:", 1)[0]
    acknowledge_job = workflow.split("  acknowledge-command:", 1)[1]
    issues_token = workflow.split("- name: Mint the issues-scoped App token", 1)[1].split(
        "- name: Authorize and run the issues command", 1
    )[0]
    # Exact indented lines, so a commented-out key cannot satisfy the check.
    assert "\n          permission-issues: write\n" in issues_token
    assert "\n          permission-pull-requests: write\n" in issues_token
    assert "permission-pull-requests: read" not in issues_token
    assert "permission-contents" not in issues_token
    assert "Require the CI App for label commands" in handle_job
    assert handle_job.index("Require the CI App for label commands") < handle_job.index(
        "Mint the issues-scoped App token"
    )
    # Each GITHUB_TOKEN job scopes its own permissions; only the actions job
    # may dispatch or rerun, and only the acknowledgement job may write issues.
    assert ("permissions:\n      contents: read\n      actions: write\n      pull-requests: read") in actions_job
    assert "issues: write" not in actions_job
    assert "create-github-app-token" not in actions_job
    assert "CI_COMMAND_API_TOKEN: ${{ github.token }}" in actions_job
    # The acknowledgement job targets a comment on a pull request, and issues-API calls
    # whose target issue is a PR are gated on pull-requests scope.
    assert "permissions:\n      contents: read\n      issues: write\n      pull-requests: write" in acknowledge_job
    assert "actions: write" not in acknowledge_job
    assert "create-github-app-token" not in acknowledge_job
    assert workflow.index("Authorize and run the actions command") < workflow.index(
        "Acknowledge the successful command"
    )
    assert "CI_COMMAND_ACKNOWLEDGE" not in actions_job
    assert "needs: [handle-command, actions-command]" in acknowledge_job
    assert (
        "if: >-\n"
        "      needs.actions-command.result == 'success' &&\n"
        "      needs.handle-command.outputs.success_reaction == '+1'"
    ) in acknowledge_job
    assert "always()" not in acknowledge_job
    assert 'CI_COMMAND_ACKNOWLEDGE: "true"' in acknowledge_job
    assert "CI_COMMAND_REPLY" not in actions_job
    assert "workflow_run_url:" not in actions_job
    assert "reply-command:" not in workflow
    assert "CI_COMMAND_REPLY" not in workflow
    assert "CI_COMMAND_WORKFLOW_RUN_URL" not in workflow
    assert "pull_request_target" not in workflow
    assert "github.event.pull_request.head" not in workflow
    assert "pip install" not in workflow
    assert "contents: write" not in workflow
    assert "\n  actions: write" not in workflow
