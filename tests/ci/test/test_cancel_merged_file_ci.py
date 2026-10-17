# [MODULE DOC] CPU tests for the workflow script that cancels per-file CI runs when a pull request
# merges. The script is loaded by path as a module (HANDLER). A fake GitHub API callable records
# calls, serves paged run listings per status and injects errors. Tests cover: cancelling runs in
# all active states only for the merged PR's file runs; collecting every page before any
# cancellation and deduplicating runs seen in several states; no calls for non-merge events;
# tolerating a run that finishes during cancellation; continuing after cancel and request errors
# and then raising a summary error; failing on the API's result-limit truncation before cancelling
# anything; the real HTTP request helper's URL, method, auth header and timeout; and text checks
# of the two workflow files' trust and queue contracts. Registered as a CPU test.
import importlib.util
import io
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]
SCRIPT = ".github/workflows/scripts/cancel_merged_file_ci.py"
spec = importlib.util.spec_from_file_location("cancel_merged_file_ci", ROOT / SCRIPT)
HANDLER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(HANDLER)
EVENT = {
    "action": "closed",
    "number": 42,
    "pull_request": {"merged": True},
    "repository": {"full_name": "radixark/miles"},
}
PREFIX = "repos/radixark/miles/actions"


def file_run(run_id, *, pull=42, status="queued", **overrides):
    # [FUNCTION DOC] HELPER: Builds a fake workflow-run record for the per-file CI workflow:
    # id, workflow path, manual-dispatch event, status, and a display title of the rerun-test form
    # naming a test file derived from the id and the pull request number. Keyword overrides replace
    # any field.
    return {
        "id": run_id,
        "path": ".github/workflows/run-ci-file.yml",
        "event": "workflow_dispatch",
        "status": status,
        "display_title": f"/rerun-test tests/e2e/test_{run_id}.py (PR #{pull})",
        **overrides,
    }


class FakeAPI:
    # [CLASS DOC] HELPER: Callable stand-in for the GitHub request function. State: runs grouped by
    # status, the status returned when a single run is fetched after a conflict, a mapping from
    # (method, path) to an exception to raise, the ordered list of calls, and the ids cancelled.
    def __init__(self, runs_by_status=None, *, conflict_status=None, errors=None):
        # [FUNCTION DOC] Stores the arguments (defaulting to empty mappings) and initializes empty
        # call and cancelled lists.
        self.runs_by_status = runs_by_status or {}
        self.conflict_status = conflict_status
        self.errors = errors or {}
        self.calls = []
        self.cancelled = []

    def __call__(self, path, *, method="GET"):
        # [FUNCTION DOC] Records the call and dispatches. A configured error for the exact method and
        # path is raised. POST: asserts the path is a run-cancel path, records the cancelled id,
        # returns none. GET without a list query: asserts it is a single-run path and returns a
        # record whose status is the configured conflict status. GET list: asserts the endpoint is
        # the per-file workflow's runs, the event filter is manual dispatch, the page size is 100
        # and no commit filter is used; returns the total count of runs of the requested status and
        # the requested page of up to 100 runs.
        self.calls.append((method, path))
        if (method, path) in self.errors:
            raise self.errors[method, path]
        if method == "POST":
            assert path.startswith(f"{PREFIX}/runs/") and path.endswith("/cancel")
            self.cancelled.append(int(path.split("/")[-2]))
            return None
        if "/runs?" not in path:
            assert path.startswith(f"{PREFIX}/runs/")
            return {"status": self.conflict_status}
        endpoint, query = path.split("?", 1)
        assert endpoint == f"{PREFIX}/workflows/run-ci-file.yml/runs"
        query = urllib.parse.parse_qs(query)
        assert query["event"] == ["workflow_dispatch"]
        assert query["per_page"] == ["100"]
        assert "head_sha" not in query
        runs = self.runs_by_status.get(query["status"][0], [])
        start = (int(query["page"][0]) - 1) * 100
        return {"total_count": len(runs), "workflow_runs": runs[start : start + 100]}


def test_cancels_all_active_states_and_files_only_for_the_merged_pr():
    # [FUNCTION DOC] Sets up one run in each of the five active states plus decoys in the queued
    # list (other PR numbers 420 and 4, a completed run, another workflow path, another event, an
    # unrelated title, a title with trailing text). Asserts the handler returns and cancels exactly
    # the five genuine runs for PR 42, in order.
    statuses = ("requested", "waiting", "pending", "queued", "in_progress")
    api = FakeAPI({status: [file_run(i, status=status)] for i, status in enumerate(statuses, 1)})
    api.runs_by_status["queued"] += [
        file_run(90, pull=420),
        file_run(91, pull=4),
        file_run(92, status="completed"),
        file_run(93, path=".github/workflows/pr-test.yml"),
        file_run(94, event="pull_request"),
        file_run(95, display_title="unrelated (PR #42)"),
        file_run(96, display_title="/rerun-test tests/e2e/test_a.py (PR #42) extra"),
    ]
    assert HANDLER.cancel_merged_file_runs(EVENT, api) == [1, 2, 3, 4, 5]
    assert api.cancelled == [1, 2, 3, 4, 5]


def test_collects_all_pages_before_cancelling_and_deduplicates_state_transitions():
    # [FUNCTION DOC] Provides 101 pending runs and one in-progress copy of run 1. Asserts the
    # handler cancels ids 1 to 101 once each, requested a second page, and made all list requests
    # before the first cancel request.
    runs = [file_run(i) for i in range(1, 102)]
    api = FakeAPI({"pending": runs, "in_progress": [file_run(1, status="in_progress")]})
    assert HANDLER.cancel_merged_file_runs(EVENT, api) == list(range(1, 102))
    assert any("page=2" in path for _, path in api.calls)
    first_post = next(i for i, (method, _) in enumerate(api.calls) if method == "POST")
    assert all(method == "GET" for method, _ in api.calls[:first_post])
    assert all(method == "POST" for method, _ in api.calls[first_post:])


@pytest.mark.parametrize("action,merged", [("closed", False), ("opened", False), ("synchronize", False)])
def test_non_merge_events_do_not_call_github(action, merged):
    # [FUNCTION DOC] Parametrized over closed-unmerged, opened and synchronize events: asserts an
    # empty result and no API calls at all.
    api = FakeAPI()
    event = {**EVENT, "action": action, "pull_request": {"merged": merged}}
    assert HANDLER.cancel_merged_file_runs(event, api) == []
    assert api.calls == []


def test_a_run_finishing_during_cancellation_is_not_an_error():
    # [FUNCTION DOC] Run 1's cancel request fails with a conflict, and fetching it reports it
    # completed. Asserts the handler returns and cancels only run 2, no error is raised, and the
    # status of run 1 was fetched.
    path = f"{PREFIX}/runs/1"
    api = FakeAPI(
        {"in_progress": [file_run(1), file_run(2)]},
        conflict_status="completed",
        errors={("POST", f"{path}/cancel"): urllib.error.HTTPError(path, 409, "conflict", {}, None)},
    )
    assert HANDLER.cancel_merged_file_runs(EVENT, api) == [2]
    assert api.cancelled == [2]
    assert ("GET", path) in api.calls


@pytest.mark.parametrize("code", [403, 409, 500, 502])
def test_cancellation_errors_do_not_prevent_other_runs(code, capsys):
    # [FUNCTION DOC] Parametrized over HTTP codes 403, 409, 500, 502. Four queued runs; cancel
    # requests for runs 1 and 3 fail (and a re-fetch shows the run still in progress). Asserts the
    # handler raises a runtime error naming PR 42 and runs [1, 3], runs 2 and 4 were cancelled,
    # every run was attempted in order, and a message per failed run with the HTTP error was printed.
    api = FakeAPI(
        {"queued": [file_run(i) for i in range(1, 5)]},
        conflict_status="in_progress",
        errors={
            ("POST", f"{PREFIX}/runs/{i}/cancel"): urllib.error.HTTPError("", code, "cancel failed", {}, None)
            for i in (1, 3)
        },
    )
    with pytest.raises(RuntimeError, match=r"Failed to cancel file runs for PR #42: \[1, 3\]"):
        HANDLER.cancel_merged_file_runs(EVENT, api)
    assert api.cancelled == [2, 4]
    assert [path for method, path in api.calls if method == "POST"] == [
        f"{PREFIX}/runs/{i}/cancel" for i in range(1, 5)
    ]
    output = capsys.readouterr().out
    for run_id in (1, 3):
        assert f"Failed to cancel file run {run_id}: HTTP Error {code}" in output


@pytest.mark.parametrize("method", ["POST", "GET"])
@pytest.mark.parametrize("error_type", [urllib.error.URLError, TimeoutError])
def test_request_failures_do_not_prevent_other_runs(method, error_type, capsys):
    # [FUNCTION DOC] Parametrized over which request fails (the cancel POST or the follow-up GET)
    # and the error kind (URL error or timeout). Run 1's cancel first conflicts, then the chosen
    # request raises the error. Asserts a runtime error naming only run 1, that run 2 was still
    # cancelled, and that the printed output contains the failure message with the error text.
    error = error_type("request failed")
    path = f"{PREFIX}/runs/1"
    errors = {("POST", f"{path}/cancel"): urllib.error.HTTPError(path, 409, "conflict", {}, None)}
    errors[method, f"{path}/cancel" if method == "POST" else path] = error
    api = FakeAPI({"queued": [file_run(1), file_run(2)]}, errors=errors)
    with pytest.raises(RuntimeError, match=r"Failed to cancel file runs for PR #42: \[1\]"):
        HANDLER.cancel_merged_file_runs(EVENT, api)
    assert api.cancelled == [2]
    assert f"Failed to cancel file run 1: {error}" in capsys.readouterr().out


def test_api_truncation_fails_before_any_cancellation():
    # [FUNCTION DOC] Provides 1001 queued runs (past the API's 1000-result window). Asserts the
    # handler raises a runtime error mentioning the limit and that nothing was cancelled.
    api = FakeAPI({"queued": [file_run(i) for i in range(1001)]})
    with pytest.raises(RuntimeError, match="1000-run limit"):
        HANDLER.cancel_merged_file_runs(EVENT, api)
    assert api.cancelled == []


@pytest.mark.parametrize("method,status", [("GET", 200), ("POST", 202)])
def test_http_requests_use_the_scoped_token_and_correct_method(monkeypatch, method, status):
    # [FUNCTION DOC] Parametrized over GET (status 200) and POST (status 202). Sets a fake token
    # in the environment and replaces the handler module's URL opener with a checker. Asserts the
    # real request helper returns the parsed JSON body for GET and none for POST (empty body).
    monkeypatch.setenv("GH_TOKEN", "test-token")

    def urlopen(request, *, timeout):
        # [FUNCTION DOC] HELPER: Fake opener asserting the full URL on the GitHub API host, the
        # HTTP method, a bearer authorization header with the test token, and a 30 second timeout;
        # returns an in-memory byte stream with the given status and, for GET, a JSON body saying
        # the run is completed (empty for POST).
        assert request.full_url == f"https://api.github.com/{PREFIX}/runs/1"
        assert request.get_method() == method
        assert request.get_header("Authorization") == "Bearer test-token"
        assert timeout == 30
        response = io.BytesIO(json.dumps({"status": "completed"}).encode() if method == "GET" else b"")
        response.status = status
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    result = HANDLER.github_request(f"{PREFIX}/runs/1", method=method)
    assert result == ({"status": "completed"} if method == "GET" else None)


def test_workflow_uses_trusted_code_and_keeps_file_queue_and_identity_contracts():
    # [FUNCTION DOC] Text-level assertions on the two workflow files. Cancel workflow: triggers on
    # the base-repository closed pull-request event, runs only when merged, has read contents and
    # write actions permissions, checks out the base commit without persisted credentials and
    # sparse-checks-out only the script, runs that script, and never references the pull request
    # head. File workflow: its run name has the rerun-test form with the test file and PR number
    # (matching what the handler parses), and its concurrency group is per PR and test file without
    # cancelling in progress runs and with a maximum-length queue.
    workflow = (ROOT / ".github/workflows/cancel-merged-file-ci.yml").read_text()
    assert "pull_request_target:\n    types: [closed]" in workflow
    assert "if: github.event.pull_request.merged == true" in workflow
    assert "permissions:\n  contents: read\n  actions: write" in workflow
    assert "ref: ${{ github.sha }}" in workflow
    assert "persist-credentials: false" in workflow
    assert f"sparse-checkout: {SCRIPT}" in workflow
    assert f"run: python3 {SCRIPT}" in workflow
    assert "pull_request.head" not in workflow
    file_workflow = (ROOT / ".github/workflows/run-ci-file.yml").read_text()
    assert 'run-name: "/rerun-test ${{ inputs.test_file }} (PR #${{ inputs.pull_number }})"' in file_workflow
    assert "group: run-ci-file-${{ inputs.pull_number }}-${{ inputs.test_file }}" in file_workflow
    assert "cancel-in-progress: false\n  queue: max" in file_workflow
