# [MODULE DOC] Script run by a GitHub Actions workflow when a pull request is closed. If the
# pull request was merged, it finds still-active runs of the per-file CI workflow (run-ci-file,
# started through manual dispatch by the rerun-test command for that pull request) and asks
# GitHub to cancel them, so merged pull requests do not keep consuming runners. Uses the GitHub
# REST API over HTTPS authenticated by the GH_TOKEN environment variable. When run as a script
# it reads the event payload JSON from the file named by GITHUB_EVENT_PATH.
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

WORKFLOW = "run-ci-file.yml"
ACTIVE_STATUSES = ("requested", "waiting", "pending", "queued", "in_progress")


def github_request(path, *, method="GET"):
    # [FUNCTION DOC] Performs one GitHub REST API call against the api host with the given
    # relative path (no leading slash) and a thirty second timeout. Sends JSON accept, bearer
    # token from the GH_TOKEN environment variable (a missing variable raises a key error) and
    # a pinned API version header. No request body is ever sent. For a POST it requires HTTP
    # status 202 (raises a runtime error otherwise) and returns nothing; for any other method
    # it returns the parsed JSON response. HTTP errors propagate as the urllib HTTP error.
    request = urllib.request.Request(
        f"https://api.github.com/{path}",
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if method == "POST":
            if response.status != 202:
                raise RuntimeError(f"Cancellation returned HTTP {response.status}, expected 202")
            return None
        return json.load(response)


def cancel_merged_file_runs(event, request=github_request):
    # [FUNCTION DOC] Cancels active per-file CI runs belonging to a merged pull request.
    # Inputs: the pull request event payload (mapping) and an injectable request callable with
    # the same signature as the GitHub request helper (used for testing). Returns an empty list
    # unless the action is closed and the merged flag is exactly true. Otherwise: builds a
    # pattern that must fully match the run display title, namely the rerun-test command, one
    # non-space token (the test file), and the text "(PR #N)" for this pull request number.
    # For each active status (requested, waiting, pending, queued, in progress) lists
    # workflow-dispatch runs of the per-file workflow, 100 per page, up to 10 pages, stopping
    # a status early when a page has fewer than 100 runs; raises a runtime error if the total
    # count for a status exceeds 1000 (GitHub's listing cap). Keeps runs whose workflow path,
    # event and non-completed status match and whose title matches, deduplicated by run id.
    # Only after all listing is finished (so cancelling does not shift pages) it posts a cancel
    # request for each run. An HTTP 409 conflict where the run is then found completed is
    # reported as already completed and is not an error; any other failure is printed and the
    # run id recorded. If any cancellation failed, raises a runtime error listing the failed
    # run ids at the end. Returns the list of run ids for which cancellation was requested.
    # Side effects: network calls and progress lines printed to standard output.
    if event["action"] != "closed" or event["pull_request"]["merged"] is not True:
        return []

    repo = event["repository"]["full_name"]
    pull_number = event["number"]
    # Dispatch runs belong to main, so head_sha/pull_requests cannot identify
    # their PR. run-ci-file.yml puts the dispatch input in this exact run name.
    title = re.compile(rf"/rerun-test \S+ \(PR #{pull_number}\)")
    runs = {}
    for status in ACTIVE_STATUSES:
        for page in range(1, 11):
            query = urllib.parse.urlencode(
                {"event": "workflow_dispatch", "status": status, "per_page": 100, "page": page}
            )
            result = request(f"repos/{repo}/actions/workflows/{WORKFLOW}/runs?{query}")
            if result["total_count"] > 1000:
                raise RuntimeError(f"File-run listing exceeds GitHub's 1000-run limit for {status}")
            for run in result["workflow_runs"]:
                if (
                    run["path"] == f".github/workflows/{WORKFLOW}"
                    and run["event"] == "workflow_dispatch"
                    and run["status"] != "completed"
                    and title.fullmatch(run["display_title"])
                ):
                    runs[run["id"]] = run
            if len(result["workflow_runs"]) < 100:
                break

    # Finish pagination before cancelling: removals from active-status lists
    # would otherwise shift later pages and leave queued runs behind.
    cancelled = []
    failed = []
    for run_id in runs:
        path = f"repos/{repo}/actions/runs/{run_id}"
        try:
            try:
                request(f"{path}/cancel", method="POST")
            except urllib.error.HTTPError as error:
                if error.code != 409 or request(path)["status"] != "completed":
                    raise
                print(f"File run {run_id} already completed")
            else:
                print(f"Requested cancellation of file run {run_id} for PR #{pull_number}")
                cancelled.append(run_id)
        except Exception as error:
            print(f"Failed to cancel file run {run_id}: {error}")
            failed.append(run_id)
    if failed:
        raise RuntimeError(f"Failed to cancel file runs for PR #{pull_number}: {failed}")
    return cancelled


if __name__ == "__main__":
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    cancel_merged_file_runs(event)
