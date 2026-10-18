# [MODULE DOC]
# Tests for the Lark notification script (.github/workflows/scripts/lark_notify.py, loaded by path
# under a test-only module name as HANDLER) and the workflows that call it. The script renders
# Lark interactive cards for finished CI runs (nightly status with failed jobs, flaky jobs that
# passed on rerun, optional AI failure reasons from ci_failure_analysis; Docker build and wheel
# build failures), reruns failed jobs of a first failed nightly attempt before reporting, and
# downloads job logs through the GitHub API (following the redirect without the token and bounding
# the download to the log tail). Tests use fake GitHub clients, fake blob servers, monkeypatched
# network functions and post functions, and read workflow YAML text for security invariants.
# Helper builders: run and job build GitHub run/job mappings; markdown joins the markdown elements
# of a card; args builds an argument object. Registers itself as a CPU test in suite stage-a-cpu.
import base64
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]
SCRIPT_DIR = ROOT / ".github/workflows/scripts"
HANDLER_PATH = SCRIPT_DIR / "lark_notify.py"
WORKFLOW_PATH = ROOT / ".github/workflows/ci-lark-notify.yml"
sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name, path):
    # [FUNCTION DOC] HELPER: Import a Python file by path under the given module name, registering
    # it in the module table before executing its body; return the module.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HANDLER = load_module("lark_notify_test", HANDLER_PATH)
ANALYSIS = HANDLER.analyze_failures.__globals__["JobAnalysis"]


def run(**overrides):
    # [FUNCTION DOC] HELPER: Build a completed, failed scheduled workflow run mapping (id 123, name
    # "PR Test", attempt 1, timestamps on 2026-09-04, head sha of forty "a", a head commit message,
    # run url and repository radixark/miles) with keyword overrides applied.
    value = {
        "id": 123,
        "name": "PR Test",
        "event": "schedule",
        "status": "completed",
        "conclusion": "failure",
        "run_attempt": 1,
        "created_at": "2026-09-04T15:00:00Z",
        "run_started_at": "2026-09-04T15:01:00Z",
        "updated_at": "2026-09-04T15:03:05Z",
        "head_sha": "a" * 40,
        "head_commit": {"message": "Test commit\nbody"},
        "html_url": "https://github.com/radixark/miles/actions/runs/123",
        "repository": {"id": 1072725553, "full_name": "radixark/miles"},
    }
    value.update(overrides)
    return value


def job(job_id=10, name="unit", conclusion="failure"):
    # [FUNCTION DOC] HELPER: Build a job mapping (id 10, name "unit", conclusion failure by default)
    # with an html url "https://example/jobs/<id>".
    return {"id": job_id, "name": name, "conclusion": conclusion, "html_url": f"https://example/jobs/{job_id}"}


def markdown(card):
    # [FUNCTION DOC] HELPER: Return the content of all markdown elements in the card body joined by
    # newlines.
    return "\n".join(
        element.get("content", "") for element in card["card"]["body"]["elements"] if element.get("tag") == "markdown"
    )


def test_analysis_disabled_preserves_exact_original_card():
    # [FUNCTION DOC] Rendering with a disabled analysis outcome gives a card equal to rendering with
    # no outcome.
    jobs = [job(), job(20, "pass", "success")]
    original = HANDLER.render_ci_status(run(), jobs, None)
    disabled = HANDLER.AnalysisOutcome(enabled=False, reasons={})
    assert HANDLER.render_ci_status(run(), jobs, None, disabled) == original


def test_validated_reason_is_directly_beneath_its_existing_job_link():
    # [FUNCTION DOC] With an enabled outcome holding one analysis (tags megatron and lora, a test
    # name, a reason sentence and related pull request 2754) for job 10, the card markdown contains
    # the job link line followed by indented lines: the tags in brackets with the test name in code
    # style, the reason, and a "related to PR #2754" link to the pull request; exactly three
    # continuation markers appear.
    outcome = HANDLER.AnalysisOutcome(
        enabled=True,
        reasons={
            10: [
                ANALYSIS(
                    reason="The assertion expected 4 but received 3.",
                    tags=("megatron", "lora"),
                    test_name="tests/fast/test_thing.py",
                    related_pull_request=2754,
                )
            ]
        },
    )
    content = markdown(HANDLER.render_ci_status(run(), [job()], None, outcome))
    assert (
        "- [unit](https://example/jobs/10)\n"
        "  ↳ [megatron][lora] `tests/fast/test_thing.py`\n"
        "  ↳ The assertion expected 4 but received 3.\n"
        "  ↳ related to [PR #2754](https://github.com/radixark/miles/pull/2754)" in content
    )
    assert content.count("↳") == 3


def shard(index):
    # [FUNCTION DOC] HELPER: Build a failed job mapping for the CPU shard with the given index (id =
    # index, name "stage-a-cpu (<index>) / run-cpu", url "https://example/jobs/<index>").
    return {
        "id": index,
        "name": f"stage-a-cpu ({index}) / run-cpu",
        "html_url": f"https://example/jobs/{index}",
        "conclusion": "failure",
    }


def test_one_defect_across_every_shard_collapses_to_a_single_row():
    # [FUNCTION DOC] Five shards sharing one identical analysis (reason and a tag) render as a single
    # row: two continuation markers total (tags and reason), all five shard names present, and no
    # separate bullet rows.
    same = ANALYSIS(reason="Collection failed because the hardware list was rejected.", tags=("rollout",))
    content = HANDLER.list_jobs_md([shard(i) for i in range(5)], reasons={i: [same] for i in range(5)})
    assert content.count("↳") == 2, content
    assert content.count("stage-a-cpu") == 5
    assert content.count("\n- ") == 0


def test_a_different_failure_keeps_its_own_row():
    # [FUNCTION DOC] Three shards where two share a reason and one differs render as two rows (one
    # separate bullet) and the different reason text appears.
    same = ANALYSIS(reason="Collection failed because the hardware list was rejected.")
    other = ANALYSIS(reason="The deterministic test exited with code 1.")
    content = HANDLER.list_jobs_md([shard(0), shard(1), shard(2)], reasons={0: [same], 1: [same], 2: [other]})
    assert content.count("\n- ") == 1, content
    assert "The deterministic test exited with code 1." in content


def test_rows_stay_separate_when_no_analysis_is_available():
    # [FUNCTION DOC] Three shards with no reasons at all render as three separate rows (two bullet
    # separators).
    content = HANDLER.list_jobs_md([shard(0), shard(1), shard(2)], reasons=None)
    assert content.count("\n- ") == 2, content


def test_rerun_reasons_apply_only_to_current_failures():
    # [FUNCTION DOC] On attempt 2 with current failing jobs 20 (still failing) and 30 (new) and a
    # previous failure 10 that was fixed (flaky), the outcome carries reasons for jobs 10, 20 and 30.
    # Asserts the card has the flaky section, does not show job 10's stale reason, and shows exactly
    # two continuation markers (for 20 and 30).
    current = [job(20, "still"), job(30, "new")]
    previous = {"fixed": job(10, "fixed"), "still": job(19, "still")}
    outcome = HANDLER.AnalysisOutcome(
        enabled=True,
        reasons={
            10: [ANALYSIS(reason="Wrong old reason.")],
            20: [ANALYSIS(reason="The same assertion still fails.")],
            30: [ANALYSIS(reason="A new timeout occurred.")],
        },
    )
    content = markdown(HANDLER.render_ci_status(run(run_attempt=2), current, previous, outcome))
    assert "Flaky, passed on rerun" in content and "Wrong old reason." not in content
    assert content.count("↳") == 2


def test_jobs_fixed_by_the_rerun_count_as_flaky_not_failed():
    # [FUNCTION DOC] Previously failed "flaky" and "still" jobs; in attempt 2 "still" fails again and
    # "flaky" succeeds. The title is "Nightly Test: FAILED (1 of 2 jobs, 1 flaky)" and the body has
    # sections "Flaky, passed on rerun (1)" and "Still failing (1)".
    previous = {"flaky": job(10, "flaky"), "still": job(19, "still")}
    current = [job(20, "still"), job(11, "flaky", "success")]
    card = HANDLER.render_ci_status(run(run_attempt=2), current, previous)
    assert card["card"]["header"]["title"]["content"] == "Nightly Test: FAILED (1 of 2 jobs, 1 flaky)"
    content = markdown(card)
    assert "**Flaky, passed on rerun (1)**" in content and "**Still failing (1)**" in content


def test_a_rerun_that_fixes_every_failure_passes_with_a_flaky_count():
    # [FUNCTION DOC] A successful attempt 2 whose only job failed before: title "Nightly Test: PASSED
    # (1 job, 1 flaky)" and a green header template.
    previous = {"flaky": job(10, "flaky")}
    card = HANDLER.render_ci_status(run(run_attempt=2, conclusion="success"), [job(11, "flaky", "success")], previous)
    assert card["card"]["header"]["title"]["content"] == "Nightly Test: PASSED (1 job, 1 flaky)"
    assert card["card"]["header"]["template"] == "green"


def test_model_failure_adds_one_note_without_removing_original_rows():
    # [FUNCTION DOC] An enabled but unavailable outcome keeps the job link row, adds exactly one "AI
    # analysis unavailable" note, and no continuation lines.
    outcome = HANDLER.AnalysisOutcome(enabled=True, reasons={}, unavailable=True)
    content = markdown(HANDLER.render_ci_status(run(), [job()], None, outcome))
    assert "[unit](https://example/jobs/10)" in content
    assert content.count("AI analysis unavailable") == 1
    assert "↳" not in content


def test_per_job_missing_log_reason_and_omitted_footer_render_compactly():
    # [FUNCTION DOC] An outcome whose reason is the analyzer's fixed unavailable reason and whose
    # omitted count is 2 renders a continuation line "unavailable" under the job and a footer "AI
    # analysis omitted for 2 additional failed jobs".
    outcome = HANDLER.AnalysisOutcome(
        enabled=True,
        reasons={10: [ANALYSIS(reason=HANDLER.analyze_failures.__globals__["UNAVAILABLE_REASON"])]},
        omitted_count=2,
    )
    content = markdown(HANDLER.render_ci_status(run(), [job()], None, outcome))
    assert "↳ unavailable" in content
    assert "AI analysis omitted for 2 additional failed jobs" in content


class FakeGitHub:
    # [CLASS DOC] HELPER: In-memory GitHub client for the notifier: serves a fixed run, its current
    # jobs and the previous attempt's jobs, and records every call as a tuple in the calls list.
    def __init__(self, current_run, jobs, previous=None):
        # [FUNCTION DOC] HELPER: Store the run, jobs and previous-attempt jobs (default empty list)
        # and an empty call log.
        self.current_run = current_run
        self.jobs = jobs
        self.previous = previous or []
        self.calls = []

    def run(self, run_id):
        # [FUNCTION DOC] HELPER: Record the call and return the fixed run.
        self.calls.append(("run", run_id))
        return self.current_run

    def run_jobs(self, run_id):
        # [FUNCTION DOC] HELPER: Record the call and return the fixed current jobs.
        self.calls.append(("run_jobs", run_id))
        return self.jobs

    def run_attempt_jobs(self, run_id, attempt):
        # [FUNCTION DOC] HELPER: Record the call and return the fixed previous-attempt jobs.
        self.calls.append(("run_attempt_jobs", run_id, attempt))
        return self.previous

    def rerun_failed_jobs(self, run_id):
        # [FUNCTION DOC] HELPER: Record a rerun request; nothing else happens.
        self.calls.append(("rerun_failed_jobs", run_id))

    def rerun_calls(self):
        # [FUNCTION DOC] HELPER: Return the recorded rerun-failed-jobs calls.
        return [call for call in self.calls if call[0] == "rerun_failed_jobs"]


def args(**overrides):
    # [FUNCTION DOC] HELPER: Build a simple object (a dynamically created type instance) whose
    # attributes mimic the parsed command line: run id 123, any_event false, a placeholder webhook,
    # dry_run false and repo radixark/miles, with keyword overrides applied.
    value = {
        "run_id": 123,
        "any_event": False,
        "webhook": "https://lark.invalid",
        "dry_run": False,
        "repo": "radixark/miles",
    }
    value.update(overrides)
    return type("Args", (), value)()


def test_green_card_makes_no_analysis_call_and_posts_once(monkeypatch):
    # [FUNCTION DOC] For a successful run with a successful job, the status command never calls the
    # analyzer, posts exactly one card, and its title ends with "PASSED (1 job)".
    calls = []
    monkeypatch.setattr(HANDLER, "analyze_failures", lambda **kwargs: calls.append(kwargs))
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda card, webhook, dry_run: posted.append(card))
    HANDLER.cmd_ci_status(args(), FakeGitHub(run(conclusion="success"), [job(conclusion="success")]))
    assert calls == [] and len(posted) == 1
    assert posted[0]["card"]["header"]["title"]["content"].endswith("PASSED (1 job)")


def test_failed_rerun_analyzes_only_still_and_new_displayed_jobs(monkeypatch):
    # [FUNCTION DOC] Sets the analysis app token in the environment and replaces the analyzer with a
    # recorder returning a disabled outcome. On attempt 2 with current failures 20 (still) and 30
    # (new) and previous failures 10 (fixed) and 19 (still), asserts the analyzer received exactly
    # jobs 20 and 30 and a GitHub client carrying the app token.
    captured = []
    monkeypatch.setenv("CI_FAILURE_ANALYSIS_GITHUB_TOKEN", "app-token")
    monkeypatch.setattr(
        HANDLER,
        "analyze_failures",
        lambda **kwargs: captured.append(kwargs) or HANDLER.AnalysisOutcome(enabled=False, reasons={}),
    )
    monkeypatch.setattr(HANDLER, "post_card", lambda *unused: None)
    gh = FakeGitHub(
        run(run_attempt=2),
        [job(20, "still"), job(30, "new")],
        [job(10, "fixed"), job(19, "still")],
    )
    HANDLER.cmd_ci_status(args(), gh)
    assert [item["id"] for item in captured[0]["jobs"]] == [20, 30]
    assert captured[0]["gh"].token == "app-token"


def test_unexpected_analyzer_exception_still_posts_original_card_once(monkeypatch):
    # [FUNCTION DOC] The analyzer raises a type error. The status command still posts exactly one
    # card containing the job link and exactly one "AI analysis unavailable" note.
    monkeypatch.setattr(HANDLER, "analyze_failures", lambda **kwargs: (_ for _ in ()).throw(TypeError("bad")))
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda card, webhook, dry_run: posted.append(card))
    HANDLER.cmd_ci_status(args(), FakeGitHub(run(run_attempt=2), [job()], [job()]))
    assert len(posted) == 1
    content = markdown(posted[0])
    assert "[unit](https://example/jobs/10)" in content
    assert content.count("AI analysis unavailable") == 1


def no_analysis():
    # [FUNCTION DOC] HELPER: Return a disabled analysis outcome with no reasons.
    return HANDLER.AnalysisOutcome(enabled=False, reasons={})


def test_first_failed_nightly_attempt_reruns_its_failed_jobs_instead_of_posting(monkeypatch):
    # [FUNCTION DOC] A failed first attempt of a scheduled run: the status command requests a rerun of
    # failed jobs for run 123 exactly once, posts nothing, and (the analyzer fails the test if
    # called) does not analyze before the rerun.
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda *values: posted.append(values))
    monkeypatch.setattr(HANDLER, "analyze_failures", lambda **kwargs: pytest.fail("analysis before the rerun"))
    gh = FakeGitHub(run(), [job(), job(20, "pass", "success")])
    HANDLER.cmd_ci_status(args(), gh)
    assert gh.rerun_calls() == [("rerun_failed_jobs", 123)] and posted == []


def test_a_rerun_attempt_is_reported_and_never_rerun_again(monkeypatch):
    # [FUNCTION DOC] For a failed attempt 2, the status command requests no rerun and posts exactly
    # one card (analysis disabled).
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda card, webhook, dry_run: posted.append(card))
    monkeypatch.setattr(HANDLER, "analyze_failures", lambda **kwargs: no_analysis())
    gh = FakeGitHub(run(run_attempt=2), [job(20, "still")], [job(19, "still")])
    HANDLER.cmd_ci_status(args(), gh)
    assert gh.rerun_calls() == [] and len(posted) == 1


def test_cancelled_and_manually_dispatched_runs_are_not_rerun(monkeypatch):
    # [FUNCTION DOC] A cancelled scheduled run and a manually dispatched run (with the any-event
    # option) both request no rerun, and two cards are posted in total.
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda card, webhook, dry_run: posted.append(card))
    monkeypatch.setattr(HANDLER, "analyze_failures", lambda **kwargs: no_analysis())
    cancelled = FakeGitHub(run(conclusion="cancelled"), [job()])
    HANDLER.cmd_ci_status(args(), cancelled)
    manual = FakeGitHub(run(event="workflow_dispatch"), [job()])
    HANDLER.cmd_ci_status(args(any_event=True), manual)
    assert cancelled.rerun_calls() == [] and manual.rerun_calls() == []
    assert len(posted) == 2


def test_dry_run_announces_the_rerun_without_requesting_it(monkeypatch, capsys):
    # [FUNCTION DOC] In dry-run mode a failed first attempt requests no rerun and posts nothing, and
    # standard output says it "would rerun 1 failed job".
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda *values: posted.append(values))
    gh = FakeGitHub(run(), [job()])
    HANDLER.cmd_ci_status(args(dry_run=True), gh)
    assert gh.rerun_calls() == [] and posted == []
    assert "would rerun 1 failed job" in capsys.readouterr().out


def test_a_refused_rerun_still_posts_the_first_attempt_then_fails(monkeypatch):
    # [FUNCTION DOC] The rerun request raises a runtime error ("403"). The status command first posts
    # the first-attempt card (title ending "FAILED (1 of 1 job)"), then re-raises the error.
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda card, webhook, dry_run: posted.append(card))
    monkeypatch.setattr(HANDLER, "analyze_failures", lambda **kwargs: no_analysis())
    gh = FakeGitHub(run(), [job()])
    monkeypatch.setattr(gh, "rerun_failed_jobs", lambda run_id: (_ for _ in ()).throw(RuntimeError("403")))
    with pytest.raises(RuntimeError, match="403"):
        HANDLER.cmd_ci_status(args(), gh)
    assert len(posted) == 1
    assert posted[0]["card"]["header"]["title"]["content"].endswith("FAILED (1 of 1 job)")


def test_rerun_failed_jobs_posts_to_the_run_with_the_token(monkeypatch):
    # [FUNCTION DOC] Replaces the URL opener with a recorder returning a fake 201 response; the real
    # client's rerun call sends exactly one POST to the repository's actions runs rerun-failed-jobs
    # endpoint for run 123 with an Authorization header "Bearer token".
    requests = []

    class Created:
        # [CLASS DOC] HELPER: Fake HTTP response with status 201 usable as a context manager.
        status = 201

        def __enter__(self):
            # [FUNCTION DOC] HELPER: Return the response itself.
            return self

        def __exit__(self, *unused):
            # [FUNCTION DOC] HELPER: Do not suppress exceptions.
            return False

    monkeypatch.setattr(
        HANDLER.urllib.request, "urlopen", lambda request, timeout: requests.append(request) or Created()
    )
    HANDLER.GitHub("token", "radixark/miles").rerun_failed_jobs(123)
    (request,) = requests
    assert request.get_method() == "POST"
    assert request.full_url == "https://api.github.com/repos/radixark/miles/actions/runs/123/rerun-failed-jobs"
    assert request.get_header("Authorization") == "Bearer token"


def test_rerun_caps_current_failed_rows_across_still_and_new_sections():
    # [FUNCTION DOC] On attempt 2 with 10 still-failing and 10 new failing jobs, the card shows only
    # 15 job links in total and the line "... and 5 more".
    previous = {f"still-{index}": job(index, f"still-{index}") for index in range(1, 11)}
    current = [job(index + 100, f"still-{index}") for index in range(1, 11)] + [
        job(index + 200, f"new-{index}") for index in range(1, 11)
    ]
    content = markdown(HANDLER.render_ci_status(run(run_attempt=2), current, previous))
    assert content.count("https://example/jobs/") == 15
    assert "... and 5 more" in content


def test_non_schedule_and_incomplete_runs_still_skip_before_delivery(monkeypatch, capsys):
    # [FUNCTION DOC] A pull-request-event run and an in-progress run are both skipped: nothing is
    # posted and standard output says the run is "not a scheduled run".
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda *values: posted.append(values))
    HANDLER.cmd_ci_status(args(), FakeGitHub(run(event="pull_request"), []))
    HANDLER.cmd_ci_status(args(), FakeGitHub(run(status="in_progress"), []))
    assert posted == []
    assert "not a scheduled run" in capsys.readouterr().out


def test_dry_run_prints_the_final_card_without_webhook_io(capsys):
    # [FUNCTION DOC] Posting a rendered card in dry-run mode prints the card as JSON to standard
    # output (parsed output equals the card) with no network access.
    card = HANDLER.render_ci_status(run(), [job()], None)
    HANDLER.post_card(card, "https://lark.invalid", True)
    assert json.loads(capsys.readouterr().out) == card


def test_job_log_rejects_non_numeric_or_non_positive_ids():
    # [FUNCTION DOC] Fetching a job log with a boolean, a string, zero or a negative id must raise a
    # value error before any request; any other outcome raises an assertion error naming the id.
    gh = HANDLER.GitHub("token", "radixark/miles")
    for value in (True, "10", 0, -1):
        try:
            gh.job_log(value, 100)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe job id {value!r}")


def test_file_content_accepts_github_line_wrapped_base64(monkeypatch):
    # [FUNCTION DOC] The contents endpoint (replaced) returns base64 text with a newline inserted
    # after eight characters, as GitHub wraps it; the client's file content call decodes it to the
    # original two lines of text.
    gh = HANDLER.GitHub("token", "radixark/miles")
    encoded = base64.b64encode(b"line one\nline two\n").decode()
    wrapped = encoded[:8] + "\n" + encoded[8:]
    monkeypatch.setattr(gh, "get", lambda path, params: {"encoding": "base64", "content": wrapped})
    assert gh.file_content("tests/example.py", "a" * 40, 100) == "line one\nline two\n"


class FakeBlob:
    # [CLASS DOC] HELPER: Fake blob storage serving a byte body as the target of a log redirect. It
    # can ignore Range headers and can omit the Content-Length header; it records each request as
    # (url, range header or none).
    """A redirect target that serves body, honouring an explicit byte range like Azure blob storage."""

    def __init__(self, body, *, honour_range=True, announce_length=True):
        # [FUNCTION DOC] HELPER: Store the body and the two behavior flags (honour range, announce
        # length; both default true) and an empty request list.
        self.body = body
        self.honour_range = honour_range
        self.announce_length = announce_length
        self.requests = []

    def open(self, target, timeout):
        # [FUNCTION DOC] HELPER: Open a url string or request object. Records (url, Range header);
        # when a range "bytes=a-b" is given and honoured, serves that inclusive slice of the body,
        # otherwise the whole body. Headers contain Content-Length of the full body unless announcing
        # is disabled. Returns a FakeBlobResponse.
        url = target if isinstance(target, str) else target.full_url
        header = None if isinstance(target, str) else target.get_header("Range")
        self.requests.append((url, header))
        served = self.body
        if header and self.honour_range:
            start, end = header.removeprefix("bytes=").split("-")
            served = self.body[int(start) : int(end) + 1]
        headers = {"Content-Length": str(len(self.body))} if self.announce_length else {}
        return FakeBlobResponse(served, headers)


class FakeBlobResponse:
    # [CLASS DOC] HELPER: Fake streaming HTTP response over a byte body with a headers mapping and a
    # read offset; usable as a context manager.
    def __init__(self, body, headers):
        # [FUNCTION DOC] HELPER: Store the body and headers; start reading at offset 0.
        self.body = body
        self.headers = headers
        self.offset = 0

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Return the response itself.
        return self

    def __exit__(self, *unused):
        # [FUNCTION DOC] HELPER: Do not suppress exceptions.
        return False

    def read(self, size):
        # [FUNCTION DOC] HELPER: Return up to size bytes from the current offset and advance the
        # offset by the number of bytes returned (empty at the end).
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def redirecting_github(monkeypatch, blob, token="dedicated-app-token"):
    # [FUNCTION DOC] HELPER: Build a real GitHub client (default token "dedicated-app-token") whose
    # authenticated opener always answers the log request with an HTTP 302 redirect to a blob storage
    # url, while the plain url opener is replaced by the fake blob's open. Returns (client, list of
    # the first, authenticated requests).
    first_requests = []

    class Opener:
        # [CLASS DOC] HELPER: Authenticated opener that only produces a redirect.
        def open(self, request, timeout):
            # [FUNCTION DOC] HELPER: Record the request, then raise an HTTP 302 error with a Location
            # header pointing to the blob url.
            first_requests.append(request)
            raise urllib.error.HTTPError(
                request.full_url,
                302,
                "Found",
                {"Location": "https://ci-results.blob.core.windows.net/job/10"},
                None,
            )

    monkeypatch.setattr(HANDLER.urllib.request, "build_opener", lambda *unused: Opener())
    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", blob.open)
    return HANDLER.GitHub(token, "radixark/miles"), first_requests


def test_job_log_redirect_drops_app_token_and_bounds_the_download(monkeypatch):
    # [FUNCTION DOC] Fetching the last 10 bytes of a 15-byte log: the result is the final ten bytes
    # "56789extra"; the first request carried the bearer app token; the blob was requested twice at
    # the redirect url (a length probe, then a ranged request "bytes=5-14"); no blob request url
    # contains an Authorization value.
    blob = FakeBlob(b"0123456789extra")
    gh, first_requests = redirecting_github(monkeypatch, blob)
    assert gh.job_log(10, 10) == "56789extra"
    assert first_requests[0].get_header("Authorization") == "Bearer dedicated-app-token"
    assert [url for url, _ in blob.requests] == ["https://ci-results.blob.core.windows.net/job/10"] * 2
    assert blob.requests[1][1] == "bytes=5-14"
    assert all("Authorization" not in str(url) for url, _ in blob.requests)


def test_job_log_keeps_the_failure_at_the_end_of_an_oversized_log(monkeypatch):
    # [FUNCTION DOC] A log of 4000 noise lines followed by a process-error line, fetched with a 200
    # byte tail limit, still contains the final error line.
    body = b"setup noise\n" * 4000 + b"##[error]Process completed with exit code 1.\n"
    gh, _ = redirecting_github(monkeypatch, FakeBlob(body))
    assert "##[error]Process completed with exit code 1." in gh.job_log(10, 200)


def test_job_log_keeps_the_tail_when_the_backend_ignores_the_range(monkeypatch):
    # [FUNCTION DOC] When the blob ignores range headers and serves the whole body, fetching the last
    # 11 bytes still returns a text ending with "final line" and newline (the client truncates to
    # the tail).
    body = b"noise\n" * 100 + b"final line\n"
    blob = FakeBlob(body, honour_range=False)
    gh, _ = redirecting_github(monkeypatch, blob)
    assert gh.job_log(10, 11).endswith("final line\n")


def test_job_log_streams_the_tail_when_the_backend_reports_no_length(monkeypatch):
    # [FUNCTION DOC] When the blob reports no content length, fetching the last 11 bytes streams the
    # body and returns exactly "final line" and newline using a single blob request.
    body = b"noise\n" * 100 + b"final line\n"
    blob = FakeBlob(body, announce_length=False)
    gh, _ = redirecting_github(monkeypatch, blob)
    assert gh.job_log(10, 11) == "final line\n"
    assert len(blob.requests) == 1


def test_job_log_download_is_bounded_when_the_body_never_ends(monkeypatch):
    # [FUNCTION DOC] The blob serves an endless stream of "x" with no length; the maximum streamed
    # byte constant is lowered to 1 MiB. Fetching the last 32 bytes must terminate and return 32
    # "x" characters (the download is bounded).
    class EndlessResponse(FakeBlobResponse):
        # [CLASS DOC] HELPER: Response whose read always returns the requested number of "x" bytes.
        def read(self, size):
            # [FUNCTION DOC] HELPER: Return size bytes of the letter x, never reaching the end.
            return b"x" * size

    class EndlessBlob(FakeBlob):
        # [CLASS DOC] HELPER: Blob whose open always yields the endless response.
        def open(self, target, timeout):
            # [FUNCTION DOC] HELPER: Record the target and return an endless response with no
            # headers.
            self.requests.append(target)
            return EndlessResponse(b"", {})

    blob = EndlessBlob(b"")
    monkeypatch.setattr(HANDLER, "MAX_LOG_STREAM_BYTES", 1 << 20)
    gh, _ = redirecting_github(monkeypatch, blob)
    assert gh.job_log(10, 32) == "x" * 32


def test_every_configuration_file_the_analyzer_reads_is_checked_out():
    # [FUNCTION DOC] For the analyzer's default policy, schema, prompt and tags paths, computes the
    # path relative to the repository root and asserts the notifier workflow text lists it on its own
    # line (12-space indented) in the sparse checkout.
    analyzer = HANDLER.analyze_failures.__globals__
    workflow = (SCRIPT_DIR.parents[0] / "ci-lark-notify.yml").read_text()
    repo_root = SCRIPT_DIR.parents[2]
    for name in ("DEFAULT_POLICY_PATH", "DEFAULT_SCHEMA_PATH", "DEFAULT_PROMPT_PATH", "DEFAULT_TAGS_PATH"):
        relative = analyzer[name].relative_to(repo_root).as_posix()
        assert f"\n            {relative}\n" in workflow, f"{relative} is missing from sparse-checkout"


def test_notifier_workflow_pins_its_identity_boundaries():
    # [FUNCTION DOC] Reads the notifier workflow text and asserts: it triggers on completion of the
    # "PR Test" workflow, only in radixark/miles on the main branch; has id-token write and an
    # actions write job permission but no contents or pull-requests write; the app token requests
    # only actions, contents and pull-requests read (no issues, no actions write); it references the
    # analysis app client id and no CI app or OpenAI key; checkout and setup-python are pinned by
    # commit sha; credentials are not persisted; the comment command script is not used; and manual
    # dispatch inputs do not offer prompt, model or repository.
    workflow = WORKFLOW_PATH.read_text()
    assert "workflow_run:" in workflow and 'workflows: ["PR Test"]' in workflow
    assert "github.repository == 'radixark/miles'" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "id-token: write" in workflow
    # the job token's only write scope is the one rerun-failed-jobs needs
    assert "\n      actions: write" in workflow
    assert "contents: write" not in workflow and "pull-requests: write" not in workflow
    assert "permission-actions: read" in workflow
    assert "permission-contents: read" in workflow
    assert "permission-pull-requests: read" in workflow
    assert "permission-issues" not in workflow and "permission-actions: write" not in workflow
    assert "CI_FAILURE_ANALYSIS_APP_CLIENT_ID" in workflow
    assert "CI_APP" not in workflow
    assert "OPENAI_API_KEY" not in workflow
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in workflow
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in workflow
    assert "persist-credentials: false" in workflow
    assert "comment_ci_command.py" not in workflow
    dispatch_inputs = workflow.split("workflow_dispatch:", 1)[1].split("env:", 1)[0]
    assert "prompt" not in dispatch_inputs and "model" not in dispatch_inputs and "repository" not in dispatch_inputs


def test_policy_prompt_and_schema_are_git_versioned_and_strict():
    # [FUNCTION DOC] Reads the checked-in analysis policy, response schema and prompt. Asserts the
    # policy is enabled with exactly one model call, the schema and its analyses item schema forbid
    # additional properties, and the prompt marks logs as untrusted and requires exactly one factual
    # sentence.
    policy = json.loads((ROOT / ".github/workflows/policies/ci-failure-analysis.json").read_text())
    schema = json.loads((ROOT / ".github/workflows/policies/ci-failure-response-schema.json").read_text())
    prompt = (ROOT / ".github/workflows/prompts/ci-failure-analysis.md").read_text()
    assert policy["enabled"] is True and policy["max_model_calls"] == 1
    assert schema["additionalProperties"] is False
    assert schema["properties"]["analyses"]["items"]["additionalProperties"] is False
    assert "untrusted" in prompt and "exactly one factual sentence" in prompt


def docker_run(**overrides):
    # [FUNCTION DOC] HELPER: Build an in-progress run mapping named "Docker Build & Push" with no
    # conclusion (from the default run) and keyword overrides applied.
    return run(name="Docker Build & Push", status="in_progress", conclusion=None, **overrides)


def build_job(conclusion="failure"):
    # [FUNCTION DOC] HELPER: Build the job "build-and-push" (id 40, url .../jobs/40) with two steps:
    # "Login to Docker Hub" (success) and "Build and push" (the given conclusion, default failure),
    # and the same conclusion on the job.
    return {
        "id": 40,
        "name": "build-and-push",
        "conclusion": conclusion,
        "html_url": "https://example/jobs/40",
        "steps": [
            {"name": "Login to Docker Hub", "conclusion": "success"},
            {"name": "Build and push", "conclusion": conclusion},
        ],
    }


def test_failed_docker_build_posts_one_card_naming_the_failed_job_and_step(monkeypatch):
    # [FUNCTION DOC] With a successful check job and a failed build job, the build-failure command
    # posts exactly one card with title "Docker Build & Push: FAILED" and a red header. The body
    # lists the failed job as a link followed by " at `Build and push`", omits the successful job,
    # and mentions a scheduled rebuild.
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda card, webhook, dry_run: posted.append(card))
    check = {"id": 30, "name": "check-upstream", "conclusion": "success", "html_url": "https://example/jobs/30"}
    HANDLER.cmd_build_failure(args(), FakeGitHub(docker_run(), [check, build_job()]))
    assert len(posted) == 1
    header = posted[0]["card"]["header"]
    assert header["title"]["content"] == "Docker Build & Push: FAILED" and header["template"] == "red"
    content = markdown(posted[0])
    assert "- [build-and-push](https://example/jobs/40) at `Build and push`" in content
    assert "check-upstream" not in content
    assert "Scheduled rebuild" in json.dumps(posted[0])


def test_push_triggered_build_failure_names_its_branch():
    # [FUNCTION DOC] Rendering the build-failure card for a push event on branch main includes the
    # text "push to main".
    card = HANDLER.render_build_failure(docker_run(event="push", head_branch="main"), [build_job()])
    assert "push to main" in json.dumps(card)


def test_docker_build_notifier_skips_a_run_without_failed_jobs(monkeypatch, capsys):
    # [FUNCTION DOC] With only a successful build job, the build-failure command posts nothing and
    # prints a message containing "no failed job".
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda *values: posted.append(values))
    HANDLER.cmd_build_failure(args(), FakeGitHub(docker_run(), [build_job("success")]))
    assert posted == []
    assert "no failed job" in capsys.readouterr().out


def test_docker_build_workflow_reports_only_failed_automatic_builds():
    # [FUNCTION DOC] Takes the notify-build-failure job text of the docker-build workflow and asserts:
    # it needs the build job, runs only when that job failed, not for manual dispatch, only in
    # radixark/miles; uses the Lark webhook secret and the docker-build-failure subcommand; pins the
    # checkout by sha without persisting credentials; its sparse checkout lists both the notifier and
    # the failure-analysis scripts; and no write permission appears before its steps.
    workflow = (SCRIPT_DIR.parents[0] / "docker-build.yml").read_text()
    job = workflow.split("\n  notify-build-failure:\n", 1)[1]
    assert "needs: [build-and-push]" in job
    assert "needs.build-and-push.result == 'failure'" in job
    assert "github.event_name != 'workflow_dispatch'" in job
    assert "github.repository == 'radixark/miles'" in job
    assert "secrets.LARK_WEBHOOK" in job and "lark_notify.py docker-build-failure" in job
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in job
    assert "persist-credentials: false" in job
    # lark_notify.py imports ci_failure_analysis, so the sparse checkout must carry both
    for relative in (".github/workflows/scripts/lark_notify.py", ".github/workflows/scripts/ci_failure_analysis.py"):
        assert f"\n            {relative}\n" in job
    assert "write" not in job.split("steps:", 1)[0]


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
@pytest.mark.parametrize(
    "failed_name,failed_step",
    [
        ("check", "Compare each source with the commit its release records"),
        ("build-te-x86", "Build transformer_engine_torch in the SGLang base image"),
        ("publish (x86_64, x86)", "Sync into the miles-wheels release"),
    ],
)
def test_wheels_failure_cli_renders_one_card_without_rerunning(monkeypatch, capsys, event, failed_name, failed_step):
    # [FUNCTION DOC] Parametrized over scheduled or manually dispatched runs of "Build Wheels" and
    # three failed job/step pairs. Builds a fake GitHub client with the failed job (steps: a
    # successful setup step and the failed step), a skipped job, a finished job and a notifier job
    # with no conclusion; replaces the client constructor, removes the webhook variable and sets the
    # command line to dry-run wheels-build-failure for run 123. Asserts main returns 0; the printed
    # card has title "Build Wheels: FAILED" and a red template; the body links the failed job with
    # " at `<step>`"; no other job name or setup step appears; the trigger text is "Scheduled
    # rebuild" for schedule and "workflow_dispatch to main" otherwise; the last element's link
    # target is the run url; and no rerun was requested.
    failed = job(name=failed_name)
    failed["steps"] = [
        {"name": "Set up job", "conclusion": "success"},
        {"name": failed_step, "conclusion": "failure"},
    ]
    gh = FakeGitHub(
        run(name="Build Wheels", event=event, head_branch="main", status="in_progress", conclusion=None),
        [failed, job(20, "unchanged", "skipped"), job(30, "finished", "success"), job(40, "notifier", None)],
    )
    monkeypatch.setattr(HANDLER, "GitHub", lambda *unused: gh)
    monkeypatch.delenv("LARK_WEBHOOK", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lark_notify.py", "--token", "test-token", "--dry-run", "wheels-build-failure", "--run-id", "123"],
    )

    assert HANDLER.main() == 0
    card = json.loads(capsys.readouterr().out)
    assert card["card"]["header"] == {
        "title": {"tag": "plain_text", "content": "Build Wheels: FAILED"},
        "template": "red",
    }
    content = markdown(card)
    assert f"- [{failed_name}](https://example/jobs/10) at `{failed_step}`" in content
    assert all(name not in content for name in ["unchanged", "finished", "notifier", "Set up job"])
    trigger = "Scheduled rebuild" if event == "schedule" else "workflow_dispatch to main"
    assert trigger in json.dumps(card)
    assert card["card"]["body"]["elements"][-1]["behaviors"][0]["default_url"] == run()["html_url"]
    assert not gh.rerun_calls()


@pytest.mark.parametrize("conclusion", ["success", "skipped", "cancelled", None])
def test_wheels_without_failed_jobs_do_not_post(monkeypatch, conclusion):
    # [FUNCTION DOC] Parametrized over a lone job whose conclusion is success, skipped, cancelled or
    # none, in a "Build Wheels" run on attempt 2. The build-failure command posts nothing.
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda *values: posted.append(values))
    HANDLER.cmd_build_failure(
        args(), FakeGitHub(run(name="Build Wheels", run_attempt=2), [job(conclusion=conclusion)])
    )
    assert not posted
