# [MODULE DOC]
# Tests for the CI failure analysis workflow script (.github/workflows/scripts/ci_failure_analysis.py),
# which turns failed GitHub Actions jobs into one-sentence, evidence-grounded failure reasons via a
# single model call. The script is loaded by file path under a test-only module name and exposed as
# ANALYZER. The module registers itself as a CPU test in suite stage-a-cpu.
# Contents: fake GitHub / model-client / OIDC-opener doubles and builders for run, job, policy and
# model-response data; then tests of policy loading, prompt hashing, secret redaction and bounded
# log-evidence extraction, response validation (tags, test name, related pull request, reason text,
# evidence refs), the end-to-end analyze_failures flow (disabled policy, missing token, missing logs,
# batching, pull request context, invalid responses, job cap, unauthorized run), and the audit
# classification of errors (OIDC provider failures, API connection errors, allowlisted root causes,
# no leakage of messages or secrets), including strict validation of the GitHub OIDC endpoint URL.
import hashlib
import importlib.util
import io
import json
import re
import ssl
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]
SCRIPT_DIR = ROOT / ".github/workflows/scripts"
ANALYZER_PATH = SCRIPT_DIR / "ci_failure_analysis.py"
POLICY_PATH = ROOT / ".github/workflows/policies/ci-failure-analysis.json"
PROMPT_PATH = ROOT / ".github/workflows/prompts/ci-failure-analysis.md"
SCHEMA_PATH = ROOT / ".github/workflows/policies/ci-failure-response-schema.json"
sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name, path):
    # [FUNCTION DOC] HELPER: Import a Python source file by path under the given module name.
    # Builds a module spec from the file location, creates the module, registers it in the
    # interpreter's module table under the name (before execution, so self-references resolve), runs
    # the module body, and returns the module. Side effect: mutates the module table and executes
    # the file's top-level code. Raises whatever the module body raises.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = load_module("ci_failure_analysis_test", ANALYZER_PATH)
SECRET = "ghp_012345678901234567890123456789012345"
# Split so the repository's detect-private-key hook does not flag this redaction fixture.
PEM_HEADER = "-----BEGIN " + "PRIVATE KEY-----"
PEM_FOOTER = "-----END " + "PRIVATE KEY-----"


def run(**overrides):
    # [FUNCTION DOC] HELPER: Build a workflow-run mapping for the repository radixark/miles (id 1072725553).
    # Defaults: run id 123, name "PR Test", attempt 1, head sha of forty "a" characters. Keyword
    # overrides replace or add top-level keys (the repository mapping is replaced wholesale).
    value = {
        "id": 123,
        "name": "PR Test",
        "run_attempt": 1,
        "head_sha": "a" * 40,
        "repository": {"id": 1072725553, "full_name": "radixark/miles"},
    }
    value.update(overrides)
    return value


def job(job_id=10, name="unit", conclusion="failure"):
    # [FUNCTION DOC] HELPER: Build a job mapping with id, name, conclusion (default failure) and an
    # html url of the form "https://example/jobs/<id>". Defaults: id 10, name "unit".
    return {"id": job_id, "name": name, "conclusion": conclusion, "html_url": f"https://example/jobs/{job_id}"}


def write_policy(tmp_path, **overrides):
    # [FUNCTION DOC] HELPER: Write a policy file in the temporary directory and return its path.
    # Starts from the checked-in policy JSON, forces enabled to true, then applies keyword overrides
    # (which may override enabled). Writes the result as "policy.json" under tmp_path.
    value = json.loads(POLICY_PATH.read_text())
    value.update({"enabled": True, **overrides})
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value))
    return path


class FakeGitHub:
    # [CLASS DOC] HELPER: In-memory stand-in for the GitHub client used by the analyzer. Serves canned
    # job logs (mapping job id to text or an exception to raise), pull requests for a commit, pull
    # request changed files, and file contents (mapping path to text). Every call is recorded as a
    # tuple (method name, arguments...) in the calls list so tests can assert what was fetched.
    def __init__(self, logs=None, pulls=None, files=None, contents=None):
        # [FUNCTION DOC] HELPER: Store the canned data; absent arguments become empty containers.
        # Initializes an empty call log.
        self.logs = logs or {}
        self.pulls = [] if pulls is None else pulls
        self.files = [] if files is None else files
        self.contents = {} if contents is None else contents
        self.calls = []

    def job_log(self, job_id, max_bytes):
        # [FUNCTION DOC] HELPER: Record the call, then return the canned log for the job id.
        # If the canned value is an exception it is raised; if no log exists a runtime error with the
        # message "missing" is raised. The byte limit is recorded but not applied.
        self.calls.append(("job_log", job_id, max_bytes))
        value = self.logs.get(job_id)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise RuntimeError("missing")
        return value

    def pulls_for_commit(self, sha):
        # [FUNCTION DOC] HELPER: Record the call and return the canned list of pull requests,
        # regardless of the sha.
        self.calls.append(("pulls_for_commit", sha))
        return self.pulls

    def pull_files(self, number):
        # [FUNCTION DOC] HELPER: Record the call and return the canned list of changed files,
        # regardless of the pull request number.
        self.calls.append(("pull_files", number))
        return self.files

    def file_content(self, path, sha, max_bytes):
        # [FUNCTION DOC] HELPER: Record the call and return the canned text for the path, truncated to
        # the first max_bytes characters. Raises a key error for a path with no canned content.
        self.calls.append(("file_content", path, sha, max_bytes))
        value = self.contents[path]
        return value[:max_bytes]


class FakeResponses:
    # [CLASS DOC] HELPER: Stand-in for the model client's responses endpoint. Records the keyword
    # arguments of each create call; then raises the configured error, or returns the configured
    # response (calling it with the request keyword arguments when it is callable).
    def __init__(self, response=None, error=None):
        # [FUNCTION DOC] HELPER: Store the response (or response factory) and the optional error;
        # start with an empty list of recorded calls.
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        # [FUNCTION DOC] HELPER: Record the request keyword arguments. Raises the configured error if
        # set; otherwise returns the response, computed from the arguments if the response is callable.
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if callable(self.response):
            return self.response(kwargs)
        return self.response


class FakeClient:
    # [CLASS DOC] HELPER: Fake model client exposing a single responses attribute (a FakeResponses).
    def __init__(self, response=None, error=None):
        # [FUNCTION DOC] HELPER: Create the responses endpoint with the given response and error.
        self.responses = FakeResponses(response, error)


class APIConnectionError(Exception):
    # [CLASS DOC] HELPER: Test exception named like the model SDK's connection error (the analyzer
    # classifies errors by class name). Carries a request attribute holding the request url.
    def __init__(self, message, request_url):
        # [FUNCTION DOC] HELPER: Initialize the base exception with the message and attach a simple
        # request object whose url attribute is the given url.
        super().__init__(message)
        self.request = SimpleNamespace(url=request_url)


class ConnectError(Exception):
    # [CLASS DOC] HELPER: Test exception named like the HTTP library's connect error; carries a
    # request attribute with the url the analyzer uses to classify the failing stage.
    def __init__(self, message, request_url):
        # [FUNCTION DOC] HELPER: Initialize the base exception with the message and attach a simple
        # request object whose url attribute is the given url.
        super().__init__(message)
        self.request = SimpleNamespace(url=request_url)


class SchemaError(Exception):
    # [CLASS DOC] HELPER: Plain exception named like a schema-validation error, used as an
    # allowlisted root cause in the audit classification tests.
    pass


class OIDCResponse:
    # [CLASS DOC] HELPER: Fake HTTP response usable as a context manager; its read returns the byte
    # payload given at construction.
    def __init__(self, payload):
        # [FUNCTION DOC] HELPER: Remember the byte payload.
        self.payload = payload

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Context-manager entry; returns the response itself.
        return self

    def __exit__(self, exc_type, exc, traceback):
        # [FUNCTION DOC] HELPER: Context-manager exit; never suppresses exceptions (returns false).
        return False

    def read(self):
        # [FUNCTION DOC] HELPER: Return the stored byte payload.
        return self.payload


class CapturingOIDCOpener:
    # [CLASS DOC] HELPER: Fake URL opener that records each (request, timeout) pair and then either
    # raises a configured error or returns an OIDCResponse with the configured payload.
    def __init__(self, response=b'{"value":"header.payload.signature"}', error=None):
        # [FUNCTION DOC] HELPER: Store the byte payload (default: a JSON object whose value field is
        # "header.payload.signature") and the optional error; start with no recorded requests.
        self.response = response
        self.error = error
        self.requests = []

    def open(self, request, timeout):
        # [FUNCTION DOC] HELPER: Record the request and timeout; raise the configured error if any,
        # otherwise return a response wrapping the configured payload.
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return OIDCResponse(self.response)


def with_cause(error, cause):
    # [FUNCTION DOC] HELPER: Set the cause link of the error to the given cause (as an explicit
    # exception chain) and return the same error object.
    error.__cause__ = cause
    return error


def valid_response(kwargs):
    # [FUNCTION DOC] HELPER: Fake model response factory. Parses the request input as JSON, and for
    # every job in its jobs list emits one analysis: that job's id, no tags, no test name, a fixed
    # one-sentence reason, category test_failure, confidence high, the job's first evidence ref, and
    # no related pull request. Returns an object with the output text (schema version "1" plus the
    # analyses, serialized) and a usage object whose dump yields 20 input and 10 output tokens.
    packet = json.loads(kwargs["input"])
    analyses = []
    for item in packet["jobs"]:
        analyses.append(
            {
                "job_id": item["job_id"],
                "tags": [],
                "test_name": None,
                "reason": "The assertion failed because the returned value was 3 instead of 4.",
                "category": "test_failure",
                "confidence": "high",
                "evidence_refs": [item["evidence_refs"][0]],
                "related_pull_request": None,
            }
        )
    return SimpleNamespace(
        output_text=json.dumps({"schema_version": "1", "analyses": analyses}),
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 20, "output_tokens": 10}),
    )


def analyze(tmp_path, jobs, gh, client, **policy_overrides):
    # [FUNCTION DOC] HELPER: Run the analyzer's analyze_failures once with the default run, the repo
    # radixark/miles, the given jobs, GitHub double and client double, a temporary policy built from
    # the policy overrides, and the checked-in prompt and schema. The client factory ignores its
    # timeout and returns the given client. Audit lines are collected rather than printed.
    # Returns a pair: the outcome and the list of emitted audit strings.
    emitted = []
    outcome = ANALYZER.analyze_failures(
        run=run(),
        jobs=jobs,
        repo="radixark/miles",
        gh=gh,
        policy_path=write_policy(tmp_path, **policy_overrides),
        prompt_path=PROMPT_PATH,
        schema_path=SCHEMA_PATH,
        client_factory=lambda timeout: client,
        emit=emitted.append,
    )
    return outcome, emitted


def test_checked_in_policy_is_valid_and_enabled():
    # [FUNCTION DOC] Loads the checked-in policy file and asserts it is enabled, allows at most 15
    # jobs, and allows exactly one model call.
    policy = ANALYZER.load_policy(POLICY_PATH)
    assert policy.enabled is True
    assert policy.max_jobs == 15
    assert policy.max_model_calls == 1


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"schema_version":"1","schema_version":"1"}', "duplicate JSON field"),
        (json.dumps({**json.loads(POLICY_PATH.read_text()), "extra": 1}), "unknown policy field"),
        (
            json.dumps({key: value for key, value in json.loads(POLICY_PATH.read_text()).items() if key != "mode"}),
            "missing policy field",
        ),
        (json.dumps({**json.loads(POLICY_PATH.read_text()), "max_jobs": 16}), "max_jobs"),
        (json.dumps({**json.loads(POLICY_PATH.read_text()), "model": "arbitrary-model"}), "unapproved policy model"),
        (json.dumps({**json.loads(POLICY_PATH.read_text()), "enabled": 1}), "enabled"),
    ],
)
def test_policy_rejects_invalid_or_expansive_configuration(tmp_path, contents, message):
    # [FUNCTION DOC] Parametrized over raw policy file contents: duplicate JSON field, an extra
    # unknown field, a missing mode field, max_jobs of 16 (above the cap), an unapproved model name,
    # and a non-boolean enabled. Writes the contents to a temporary file and asserts loading raises
    # the analysis config error with a message matching the expected fragment.
    path = tmp_path / "policy.json"
    path.write_text(contents)
    with pytest.raises(ANALYZER.AnalysisConfigError, match=message):
        ANALYZER.load_policy(path)


def test_prompt_sha_is_the_git_blob_sha():
    # [FUNCTION DOC] Loads the prompt and asserts the prompt text is non-empty and the returned
    # fingerprint equals the git blob hash of the file bytes (SHA-1 over the header "blob <length>",
    # a NUL byte, then the content).
    prompt, blob_sha = ANALYZER.load_prompt(PROMPT_PATH)
    data = PROMPT_PATH.read_bytes()
    assert prompt
    assert blob_sha == hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_redaction_precedes_bounded_evidence_extraction():
    # [FUNCTION DOC] Feeds a raw log containing ANSI colour codes, a bearer authorization header, a
    # GitHub token, an AWS access key id, a signed-URL query secret, a private-key block and a NUL
    # byte around an assertion line. Extracts evidence (job 10, budget 2000) and asserts the
    # assertion text survives while escape and NUL characters and every secret are gone.
    raw = (
        f"\x1b[31mAuthorization: Bearer top-secret\x1b[0m\nTOKEN={SECRET}\n"
        "AWS_ACCESS_KEY_ID=AKIA0123456789ABCDEF\n"
        "https://example.invalid/file?X-Amz-Signature=signed-secret\n"
        f"{PEM_HEADER}\nprivate-material\n{PEM_FOOTER}\n"
        "tests/unit/test_math.py:12: AssertionError: expected 4\x00\n"
    )
    evidence = ANALYZER.extract_log_evidence(raw, 10, 2000)
    assert evidence is not None
    assert "AssertionError" in evidence["text"]
    assert "\x1b" not in evidence["text"] and "\x00" not in evidence["text"]
    assert SECRET not in evidence["text"] and "top-secret" not in evidence["text"]
    assert "AKIA0123456789ABCDEF" not in evidence["text"] and "signed-secret" not in evidence["text"]
    assert "private-material" not in evidence["text"]


def test_log_windows_are_deterministic_and_unicode_safe():
    # [FUNCTION DOC] Builds an 161-line log with an assertion line containing a non-BMP emoji in the
    # middle. Extracts evidence twice with budget 180 and asserts the results are identical, the text
    # is at most 180 characters, and it can be encoded as UTF-8 (no broken characters).
    text = "\n".join(
        [*(f"line {index}" for index in range(80)), "AssertionError: 🐦 failure", *("tail" for _ in range(80))]
    )
    first = ANALYZER.extract_log_evidence(text, 10, 180)
    second = ANALYZER.extract_log_evidence(text, 10, 180)
    assert first == second
    assert len(first["text"]) <= 180
    first["text"].encode("utf-8")


def pytest_collection_log(noise_lines=22):
    # [FUNCTION DOC] HELPER: Return a synthetic log of a pytest collection failure: 60 setup lines, a
    # collection-error marker for tests/fast/ray/test_layout.py, an import-error header, a
    # traceback header, noise_lines (default 22) importlib frames, the decisive module-not-found line,
    # 40 warning lines, a pytest summary line and a process-exit error line.
    """A pytest import failure: the decisive line trails its marker by a long importlib traceback."""
    return "\n".join(
        [
            *(f"setup line {index}" for index in range(60)),
            "_ ERROR collecting tests/fast/ray/test_layout.py _",
            "ImportError while importing test module '/w/tests/fast/ray/test_layout.py'.",
            "Traceback:",
            *("<frozen importlib._bootstrap>:1204: in _gcd_import" for _ in range(noise_lines)),
            "E   ModuleNotFoundError: No module named 'miles.backends.update_weight'",
            *(f"warning {index}" for index in range(40)),
            "=================== 1 error in 4.35s ===================",
            "##[error]Process completed with exit code 1.",
        ]
    )


def test_evidence_reaches_the_decisive_line_past_a_long_traceback():
    # [FUNCTION DOC] Extracts evidence (budget 8400) from the collection-failure log and asserts the
    # decisive module-not-found line, which trails the marker by a long traceback, is included.
    evidence = ANALYZER.extract_log_evidence(pytest_collection_log(), 10, 8400)
    assert "E   ModuleNotFoundError: No module named 'miles.backends.update_weight'" in evidence["text"]


def test_evidence_spends_the_budget_it_is_given():
    # [FUNCTION DOC] Extracts evidence from the same log with budgets 900 and 8400. Asserts the small
    # result is within 900, the large one is longer than the small one and within 8400, and has more
    # lines (a larger budget yields more context).
    text = pytest_collection_log()
    small = ANALYZER.extract_log_evidence(text, 10, 900)
    large = ANALYZER.extract_log_evidence(text, 10, 8400)
    assert len(small["text"]) <= 900 < len(large["text"]) <= 8400
    assert large["text"].count("\n") > small["text"].count("\n")


def test_evidence_never_exceeds_a_budget_smaller_than_the_seed_window():
    # [FUNCTION DOC] Extracts evidence with a tiny budget of 300 characters and asserts the text never
    # exceeds it, even though the initial marker window would be larger.
    evidence = ANALYZER.extract_log_evidence(pytest_collection_log(), 10, 300)
    assert len(evidence["text"]) <= 300


def test_evidence_stops_growing_at_the_edges_of_a_short_log():
    # [FUNCTION DOC] With a two-line log and a large budget, asserts that after the first line (a
    # header added by the extractor) the evidence is exactly the two log lines: window growth stops at
    # the log edges without padding or repeating.
    text = "\n".join(["AssertionError: boom", "second line"])
    evidence = ANALYZER.extract_log_evidence(text, 10, 8400)
    assert evidence["text"].splitlines()[1:] == ["AssertionError: boom", "second line"]


GROUNDING = (
    "E   ModuleNotFoundError: No module named 'miles.backends.megatron_utils.update_weight'\n"
    "tests/fast/ray/test_layout.py:10: in <module>\n"
    "Commits touching miles/backends: refactor(update-weight): move the protocols (#2754)\n"
)


def grounded_item(**overrides):
    # [FUNCTION DOC] HELPER: Build one analysis mapping grounded in the GROUNDING text for job 10:
    # tags weight-update and megatron, test name tests/fast/ray/test_layout.py, a one-sentence
    # reason, category test_failure, confidence high, evidence ref "job:10:log:1-2", related pull
    # request 2754. Keyword overrides replace individual fields.
    item = {
        "job_id": 10,
        "tags": ["weight-update", "megatron"],
        "test_name": "tests/fast/ray/test_layout.py",
        "reason": "Collection failed because the update_weight module no longer exists.",
        "category": "test_failure",
        "confidence": "high",
        "evidence_refs": ["job:10:log:1-2"],
        "related_pull_request": 2754,
    }
    item.update(overrides)
    return item


def validate_grounded(**overrides):
    # [FUNCTION DOC] HELPER: Validate a response with a single grounded item (with overrides) against
    # one job (id 10, single evidence ref), reason limit 280, the loaded tag vocabulary and the
    # GROUNDING text as job 10's evidence. Returns the mapping from job id to analyses; raises a
    # value error when validation rejects the response.
    return ANALYZER.validate_response(
        json.dumps({"schema_version": "1", "analyses": [grounded_item(**overrides)]}),
        [{"job_id": 10, "evidence_refs": ["job:10:log:1-2"]}],
        280,
        ANALYZER.load_tags(),
        {10: GROUNDING},
    )


def test_tag_vocabulary_is_the_only_source_of_the_schema_enum():
    # [FUNCTION DOC] Loads the tag vocabulary and the schema built with it; asserts the schema's tag
    # item enum equals the vocabulary, and the vocabulary is sorted, duplicate-free and contains
    # weight-update.
    tags = ANALYZER.load_tags()
    schema = ANALYZER.load_schema(SCHEMA_PATH, tags)
    assert schema["properties"]["analyses"]["items"]["properties"]["tags"]["items"]["enum"] == tags
    assert tags == sorted(set(tags)) and "weight-update" in tags


def test_grounded_analysis_keeps_tags_test_name_and_cause_pull_request():
    # [FUNCTION DOC] Validates the fully grounded item and asserts the tags (weight-update, megatron
    # in order), the test name and the related pull request 2754 are kept unchanged.
    analysis = validate_grounded()[10][0]
    assert analysis.tags == ("weight-update", "megatron")
    assert analysis.test_name == "tests/fast/ray/test_layout.py"
    assert analysis.related_pull_request == 2754


@pytest.mark.parametrize(
    "overrides",
    [
        {"reason": "Collection failed because the module is gone [job:10:log:1-2]."},
        {"reason": "no terminal punctuation"},
        {"category": "remediation"},
        {"evidence_refs": ["job:other:log:1"]},
    ],
)
def test_a_broken_core_contract_still_rejects_the_whole_response(overrides):
    # [FUNCTION DOC] Parametrized over core-field violations: reason containing an evidence-ref
    # marker, reason without terminal punctuation, an unknown category, evidence ref of a different
    # job. Asserts validation raises a value error for the whole response.
    with pytest.raises(ValueError):
        validate_grounded(**overrides)


@pytest.mark.parametrize(
    ("overrides", "field", "expected"),
    [
        ({"tags": ["deepseek-v9"]}, "tags", ()),
        ({"tags": ["inkling"]}, "tags", ()),
        ({"tags": ["multi-lora"]}, "tags", ()),
        ({"tags": ["megatron", "megatron"]}, "tags", ()),
        ({"tags": ["megatron", "lora", "fsdp"]}, "tags", ()),
        ({"test_name": "tests/fabricated/test_nope.py"}, "test_name", None),
        ({"test_name": "tests/fast/ray/test_layout.py; rm -rf /"}, "test_name", None),
        ({"related_pull_request": 9999}, "related_pull_request", None),
        ({"related_pull_request": -1}, "related_pull_request", None),
        ({"related_pull_request": True}, "related_pull_request", None),
    ],
)
def test_an_ungrounded_decoration_drops_itself_and_keeps_the_row(overrides, field, expected):
    # [FUNCTION DOC] Parametrized over ungrounded optional decorations: unknown tags, duplicate tags,
    # more than two tags, a test name not in the evidence or containing shell text, and related pull
    # requests that are absent from evidence, negative, or boolean. Asserts the offending field
    # falls back to its empty value (empty tags, no test name, no pull request) while the row and
    # its reason survive.
    analysis = validate_grounded(**overrides)[10][0]
    assert getattr(analysis, field) == expected
    assert analysis.reason.startswith("Collection failed")


def test_a_tag_needs_its_whole_phrase_not_scattered_words():
    # [FUNCTION DOC] Calls the private tag validator with a text containing the words of multi-lora
    # scattered. Asserts that tag raises a value error, the single-word tag megatron is accepted from
    # a module name, and weight-update is accepted from the identifier update_weight_from_distributed
    # (word order may differ for compound tags found inside identifiers).
    vocabulary = ANALYZER.load_tags()
    scattered = "lora adapters across multi node runs in megatron_utils"
    with pytest.raises(ValueError):
        ANALYZER._validate_tags(["multi-lora"], vocabulary, scattered)
    assert ANALYZER._validate_tags(["megatron"], vocabulary, scattered) == ("megatron",)
    assert ANALYZER._validate_tags(["weight-update"], vocabulary, "update_weight_from_distributed") == (
        "weight-update",
    )


def test_prompt_states_the_reason_limit_the_policy_enforces():
    # [FUNCTION DOC] Asserts the prompt text contains the phrase "at most N characters" where N is the
    # policy's maximum reason length, keeping prompt and policy consistent.
    prompt, _ = ANALYZER.load_prompt(PROMPT_PATH)
    limit = ANALYZER.load_policy(POLICY_PATH).max_reason_chars
    assert f"at most {limit} characters" in prompt


def test_audit_names_our_own_validation_rule_but_never_foreign_text():
    # [FUNCTION DOC] Builds audit records from two value errors. A message from the declared safe
    # validation set is reported as the validation reason; an arbitrary message is not reported at
    # all.
    ours = ANALYZER._analysis_error_audit(ValueError("tag is not grounded in the evidence"))
    assert ours["validation_reason"] == "tag is not grounded in the evidence"
    assert "validation_reason" not in ANALYZER._analysis_error_audit(ValueError("leaked secret value"))


def test_every_validation_message_is_declared_safe_for_the_audit():
    # [FUNCTION DOC] Reads the analyzer source text, collects every message used in a value error
    # raised with a plain double-quoted literal, and asserts the set is non-empty and a subset of the
    # analyzer's safe validation reasons set. Reads a file; no analyzer code is executed.
    source = ANALYZER_PATH.read_text()
    raised = set(re.findall(r'raise ValueError\("([^"]+)"\)', source))
    assert raised and raised <= ANALYZER.SAFE_VALIDATION_REASONS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("__w/miles/miles/miles/utils/test_utils/runner.py", "miles/utils/test_utils/runner.py"),
        ("__w/miles/miles/tests/e2e/sglang/test_qwen36.py", "tests/e2e/sglang/test_qwen36.py"),
        ("home/runner/work/miles/miles/tests/fast/ray/test_layout.py", "tests/fast/ray/test_layout.py"),
        ("miles/utils/misc.py", "miles/utils/misc.py"),
        ("opt/hostedtoolcache/Python/3.11/site-packages/_pytest/python.py", None),
    ],
)
def test_a_runner_path_resolves_past_the_repeated_repository_name(raw, expected):
    # [FUNCTION DOC] Parametrized over runner-style paths (workspace prefixes with the repository
    # name repeated, plain repo-relative paths, and a hosted tool cache path). Asserts the private
    # safe-path function strips the runner prefix to a repository-relative path, keeps already
    # relative paths, and returns nothing for paths outside the repository.
    assert ANALYZER._safe_path(raw) == expected


SUITE_SUMMARY = """2026-09-09T15:00:00.0Z FAILED:
2026-09-09T15:00:00.0Z   tests/e2e/a/test_one.py (exit code 1)
2026-09-09T15:00:00.0Z   tests/e2e/b/test_two.py (exit code 1)
2026-09-09T15:00:00.0Z ============================================================
"""


def two_failure_jobs():
    # [FUNCTION DOC] HELPER: Return a job list with one job (id 10, one evidence ref) whose failing
    # tests are tests/e2e/a/test_one.py and tests/e2e/b/test_two.py.
    return [
        {
            "job_id": 10,
            "evidence_refs": ["job:10:log:1-2"],
            "failing_tests": ["tests/e2e/a/test_one.py", "tests/e2e/b/test_two.py"],
        }
    ]


def analysis_item(test_name, **overrides):
    # [FUNCTION DOC] HELPER: Build one analysis mapping for job 10 with the given test name, no tags,
    # a valid one-sentence reason, category test_failure, confidence high, evidence ref
    # "job:10:log:1-2" and no related pull request; keyword overrides replace fields.
    item = {
        "job_id": 10,
        "tags": [],
        "test_name": test_name,
        "reason": "Collection failed because the module is gone.",
        "category": "test_failure",
        "confidence": "high",
        "evidence_refs": ["job:10:log:1-2"],
        "related_pull_request": None,
    }
    item.update(overrides)
    return item


def validate_two(items):
    # [FUNCTION DOC] HELPER: Validate a response made of the given analysis items against the
    # two-failure job description, reason limit 280, the tag vocabulary and the GROUNDING evidence.
    # Returns the job-to-analyses mapping; raises a value error on rejection.
    return ANALYZER.validate_response(
        json.dumps({"schema_version": "1", "analyses": items}),
        two_failure_jobs(),
        280,
        ANALYZER.load_tags(),
        {10: GROUNDING},
    )


def failure_block(test_name, body_lines, stamp="2026-09-11T16:56:05.4909230Z "):
    # [FUNCTION DOC] HELPER: Render a "Last output of <test>:" block as a CI log would: a header line
    # followed by each body line prefixed with a pipe and two spaces, every line preceded by the
    # timestamp stamp. Returns the lines joined by newlines.
    lines = [f"{stamp}Last output of {test_name}:"]
    lines += [f"{stamp}  | {line}" for line in body_lines]
    return "\n".join(lines)


def two_block_log():
    # [FUNCTION DOC] HELPER: Return a timestamped log with two failure blocks (tests one and two, each
    # with 40 noise lines then an assertion line) followed by a FAILED summary listing both tests with
    # exit code 1 and a separator line.
    return "\n".join(
        [
            failure_block("tests/e2e/a/test_one.py", [f"noise {i}" for i in range(40)] + ["AssertionError: one"]),
            failure_block("tests/e2e/b/test_two.py", [f"noise {i}" for i in range(40)] + ["AssertionError: two"]),
            "2026-09-11T16:56:05.4909230Z FAILED:",
            "2026-09-11T16:56:05.4909230Z   tests/e2e/a/test_one.py (exit code 1)",
            "2026-09-11T16:56:05.4909230Z   tests/e2e/b/test_two.py (exit code 1)",
            "2026-09-11T16:56:05.4909230Z ============================================================",
        ]
    )


def test_log_evidence_drops_the_timestamp_every_line_carries():
    # [FUNCTION DOC] Extracts evidence (budget 40000) from the two-block log and asserts no timestamp
    # text remains while the first assertion line is still present.
    evidence = ANALYZER.extract_log_evidence(two_block_log(), 10, 40_000)
    assert "2026-09-11T16:56:05" not in evidence["text"]
    assert "AssertionError: one" in evidence["text"]


def test_every_failure_block_survives_whole_when_the_budget_allows():
    # [FUNCTION DOC] With budget 40000, asserts both failure blocks appear (two header occurrences)
    # and both assertion lines are present.
    evidence = ANALYZER.extract_log_evidence(two_block_log(), 10, 40_000)
    assert evidence["text"].count("Last output of") == 2
    assert "AssertionError: one" in evidence["text"] and "AssertionError: two" in evidence["text"]


def test_a_budget_too_small_for_the_blocks_falls_back_to_marker_windows():
    # [FUNCTION DOC] With budget 600, asserts the evidence stays within 600 characters and is not
    # blank, i.e. the extractor falls back to marker windows when whole blocks do not fit.
    evidence = ANALYZER.extract_log_evidence(two_block_log(), 10, 600)
    assert len(evidence["text"]) <= 600
    assert evidence["text"].strip()


def test_the_suite_summary_names_every_failing_test_in_a_job():
    # [FUNCTION DOC] Parses the suite summary text with a limit of 5 and asserts the failed-test
    # extractor returns both test paths in order.
    assert ANALYZER.extract_failed_tests(SUITE_SUMMARY, 5) == [
        "tests/e2e/a/test_one.py",
        "tests/e2e/b/test_two.py",
    ]


def test_a_job_with_two_failures_keeps_both_analyses():
    # [FUNCTION DOC] Validates one analysis per failing test and asserts job 10 keeps both, in order,
    # keyed by their test names.
    analyses = validate_two([analysis_item("tests/e2e/a/test_one.py"), analysis_item("tests/e2e/b/test_two.py")])
    assert [a.test_name for a in analyses[10]] == ["tests/e2e/a/test_one.py", "tests/e2e/b/test_two.py"]


@pytest.mark.parametrize(
    "items",
    [
        [analysis_item("tests/e2e/a/test_one.py")],
        [analysis_item("tests/e2e/a/test_one.py"), analysis_item("tests/e2e/a/test_one.py")],
        [analysis_item("tests/e2e/a/test_one.py"), analysis_item("tests/e2e/c/test_invented.py")],
    ],
)
def test_a_response_must_cover_exactly_the_tests_the_suite_named(items):
    # [FUNCTION DOC] Parametrized over item lists that miss one named test, repeat one test, or
    # include an invented test. Asserts validation raises a value error: coverage of the suite-named
    # failing tests must be exact and without duplicates.
    with pytest.raises(ValueError):
        validate_two(items)


def test_missing_module_paths_cover_deleted_packages_and_module_files():
    # [FUNCTION DOC] Feeds import-failure text (a missing dotted module, a missing name imported from
    # a module, and a third-party module) to the missing-module path extractor. Asserts the first
    # path is the dotted module converted to a slash path without extension; the from-import yields a
    # module file path listed before the package path; third-party modules yield no path.
    text = (
        "ModuleNotFoundError: No module named 'miles.backends.megatron_utils.update_weight'\n"
        "ImportError: cannot import name 'exec_command' from 'miles.utils.misc'\n"
        "No module named 'torch_memory_saver'\n"
    )
    paths = ANALYZER.extract_missing_module_paths(text)
    assert paths[0] == "miles/backends/megatron_utils/update_weight"
    assert "miles/utils/misc.py" in paths and paths.index("miles/utils/misc.py") < paths.index("miles/utils/misc")
    assert not any("torch_memory_saver" in path for path in paths)


def test_absent_test_name_and_cause_pull_request_are_allowed():
    # [FUNCTION DOC] Validates an item with empty tags, no test name and no pull request; asserts all
    # three come back empty without rejection.
    analysis = validate_grounded(tags=[], test_name=None, related_pull_request=None)[10][0]
    assert analysis.tags == () and analysis.test_name is None and analysis.related_pull_request is None


def test_source_locations_keep_only_safe_repository_paths():
    # [FUNCTION DOC] Feeds text with a runner-absolute traceback path, a relative path with line
    # number, a parent-directory path, a url and a temp path to the source-location extractor.
    # Asserts the runner path (ending at the repo test file, line 41) and the relative path (line 22)
    # are found, and no result contains a parent-directory segment, a url host, or a temp prefix.
    text = (
        'File "/home/runner/work/miles/miles/tests/ci/test/test_gate.py", line 41\n'
        "miles/core/train.py:22: failure\n"
        "../secrets/key.py:1\nhttps://evil.invalid/a.py:2\n/tmp/random.py:3"
    )
    locations = ANALYZER.extract_source_locations(text)
    assert any(path.endswith("tests/ci/test/test_gate.py") and line == 41 for path, line in locations)
    assert ("miles/core/train.py", 22) in locations
    assert all(
        ".." not in path and "evil.invalid" not in path and not path.startswith("tmp/") for path, _ in locations
    )


def test_validate_response_accepts_exact_job_and_evidence_contract():
    # [FUNCTION DOC] Validates a minimal well-formed response for job 10 with empty evidence text;
    # asserts it is accepted, the reason is preserved, and tags and related pull request are empty.
    jobs = [{"job_id": 10, "evidence_refs": ["job:10:log:1-2"]}]
    raw = {
        "schema_version": "1",
        "analyses": [
            {
                "job_id": 10,
                "tags": [],
                "test_name": None,
                "reason": "The assertion failed because the result was empty.",
                "category": "test_failure",
                "confidence": "high",
                "evidence_refs": ["job:10:log:1-2"],
                "related_pull_request": None,
            }
        ],
    }
    analyses = ANALYZER.validate_response(json.dumps(raw), jobs, 280, ANALYZER.load_tags(), {10: ""})
    assert analyses[10][0].reason.startswith("The assertion")
    assert analyses[10][0].tags == () and analyses[10][0].related_pull_request is None


def test_strict_response_schema_uses_only_supported_structured_output_keywords():
    # [FUNCTION DOC] Loads the response schema and walks its property and item subschemas collecting
    # keywords. Asserts they are a subset of the keywords supported by strict structured output
    # (id, additional properties, const, enum, items, max/min items, max/min length, properties,
    # required, type) and that unique-items is not used.
    schema = ANALYZER.load_schema(SCHEMA_PATH)
    supported = {
        "$id",
        "additionalProperties",
        "const",
        "enum",
        "items",
        "maxItems",
        "maxLength",
        "minItems",
        "minLength",
        "properties",
        "required",
        "type",
    }

    def schema_keywords(node):
        # [FUNCTION DOC] HELPER: Recursively collect the keyword names of a schema node. Returns the
        # empty set for a non-mapping node; otherwise the node's own keys plus those of every child
        # under properties and of the items subschema.
        if not isinstance(node, dict):
            return set()
        result = set(node)
        for key, value in node.items():
            if key == "properties":
                for child in value.values():
                    result.update(schema_keywords(child))
            elif key == "items":
                result.update(schema_keywords(value))
        return result

    keywords = schema_keywords(schema)
    assert keywords <= supported
    assert "uniqueItems" not in keywords


@pytest.mark.parametrize(
    "refs",
    [
        [],
        ["job:10:log:1-2", "job:10:log:1-2"],
        ["job:other:log:1-2"],
        ["job:10:log:1-2", 123],
    ],
)
def test_local_validation_rejects_duplicate_empty_or_invalid_evidence_refs(refs):
    # [FUNCTION DOC] Parametrized over evidence ref lists that are empty, duplicated, refer to a
    # different job, or contain a non-string. Asserts local validation raises a value error.
    item = {
        "job_id": 10,
        "tags": [],
        "test_name": None,
        "reason": "The assertion failed because the result was empty.",
        "category": "test_failure",
        "confidence": "high",
        "evidence_refs": refs,
        "related_pull_request": None,
    }
    with pytest.raises(ValueError):
        ANALYZER.validate_response(
            json.dumps({"schema_version": "1", "analyses": [item]}),
            [{"job_id": 10, "evidence_refs": ["job:10:log:1-2"]}],
            280,
            ANALYZER.load_tags(),
            {10: ""},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.update(extra="bad"),
        lambda item: item.update(job_id=11),
        lambda item: item.update(reason="One sentence. Another sentence."),
        lambda item: item.update(reason="First cause. second cause."),
        lambda item: item.update(reason="See https://evil.invalid/a."),
        lambda item: item.update(reason="See www.evil.invalid for details."),
        lambda item: item.update(reason="See //evil.invalid for details."),
        lambda item: item.update(reason="The **assertion** failed."),
        lambda item: item.update(reason="x" * 280 + "."),
        lambda item: item.update(reason="missing terminal punctuation"),
        lambda item: item.update(category="remediation"),
        lambda item: item.update(evidence_refs=["job:other:log:1"]),
    ],
)
def test_validate_response_rejects_untrusted_or_unverifiable_output(mutation):
    # [FUNCTION DOC] Parametrized over mutations of a valid item: an extra field, a wrong job id, two
    # sentences, a lowercase second sentence, reasons with a url, a www host or a protocol-relative
    # url, markdown emphasis, an over-long reason, missing terminal punctuation, an unknown category,
    # and a foreign evidence ref. Applies the mutation and asserts validation raises a value error.
    item = {
        "job_id": 10,
        "tags": [],
        "test_name": None,
        "reason": "The assertion failed because the result was empty.",
        "category": "test_failure",
        "confidence": "high",
        "evidence_refs": ["job:10:log:1-2"],
        "related_pull_request": None,
    }
    mutation(item)
    with pytest.raises(ValueError):
        ANALYZER.validate_response(
            json.dumps({"schema_version": "1", "analyses": [item]}),
            [{"job_id": 10, "evidence_refs": ["job:10:log:1-2"]}],
            280,
            ANALYZER.load_tags(),
            {10: ""},
        )


def test_disabled_policy_makes_no_github_or_model_call(tmp_path):
    # [FUNCTION DOC] Runs the analysis with the policy disabled. Asserts the outcome is the disabled
    # outcome with no reasons, and neither the GitHub double nor the model client was called.
    client = FakeClient(response=valid_response)
    gh = FakeGitHub(logs={10: "AssertionError: no"})
    outcome, _ = analyze(tmp_path, [job()], gh, client, enabled=False)
    assert outcome == ANALYZER.AnalysisOutcome(enabled=False, reasons={})
    assert gh.calls == [] and client.responses.calls == []


def test_missing_analysis_app_token_preserves_base_card_contract(tmp_path):
    # [FUNCTION DOC] Runs the analysis with no GitHub client (token unavailable). Asserts the outcome
    # is marked unavailable with no per-job reasons and the first audit line records the github_auth
    # stage.
    outcome, emitted = analyze(tmp_path, [job()], None, FakeClient(response=valid_response))
    assert outcome.unavailable and outcome.reasons == {}
    assert '"stage":"github_auth"' in emitted[0]


def test_all_missing_logs_get_per_row_fallback_without_model_call(tmp_path):
    # [FUNCTION DOC] Runs two jobs whose logs cannot be fetched. Asserts each gets the fixed
    # unavailable reason, the outcome as a whole is not unavailable, no model request is made, and
    # the audit line reports a request count of zero.
    client = FakeClient(response=valid_response)
    outcome, emitted = analyze(tmp_path, [job(10), job(11)], FakeGitHub(), client)
    assert {key: value[0].reason for key, value in outcome.reasons.items()} == {
        10: ANALYZER.UNAVAILABLE_REASON,
        11: ANALYZER.UNAVAILABLE_REASON,
    }
    assert not outcome.unavailable and client.responses.calls == []
    assert '"request_count":0' in emitted[0]


def test_four_jobs_are_batched_into_one_bounded_tool_free_request(tmp_path):
    # [FUNCTION DOC] Runs four jobs (ids 10 to 13) whose logs contain a token and an assertion line.
    # Asserts four reasons result, exactly one model request is made with storage disabled, no tools,
    # model gpt-5.6-luna and a JSON-schema output format; the token appears neither in the request
    # nor in audit output; the packet lists the jobs in order; total evidence text is at most 60000
    # characters; and the audit reports one request and a valid validation result.
    logs = {
        index: f"TOKEN={SECRET}\ntests/ci/test/test_gate.py:12: AssertionError: value {index}"
        for index in range(10, 14)
    }
    client = FakeClient(response=valid_response)
    outcome, emitted = analyze(tmp_path, [job(index) for index in range(10, 14)], FakeGitHub(logs=logs), client)
    assert len(outcome.reasons) == 4 and not outcome.unavailable
    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["store"] is False and request["tools"] == []
    assert request["model"] == "gpt-5.6-luna"
    assert request["text"]["format"]["type"] == "json_schema"
    assert SECRET not in request["input"] and SECRET not in "".join(emitted)
    packet = json.loads(request["input"])
    assert [item["job_id"] for item in packet["jobs"]] == [10, 11, 12, 13]
    assert sum(len(item["text"]) for item in packet["evidence"]) <= 60000
    assert '"request_count":1' in emitted[0] and '"validation":"valid"' in emitted[0]


def test_unique_pr_and_trace_path_add_bounded_context_at_exact_sha(tmp_path):
    # [FUNCTION DOC] Runs one job whose log has a traceback through a repo test file, with exactly one
    # pull request for the commit, a changed-file patch and file contents. Asserts the pull request
    # lookup used the run's head sha, its files were fetched, the source file was fetched at that
    # sha, and the evidence kinds sent to the model are job log, pull request and source.
    log = 'Traceback (most recent call last):\n  File "/work/miles/tests/ci/test/test_gate.py", line 2\nAssertionError'
    gh = FakeGitHub(
        logs={10: log},
        pulls=[{"number": 55, "title": "Change gate", "body": "Updates assertion"}],
        files=[{"filename": "tests/ci/test/test_gate.py", "patch": "@@ -1 +1 @@\n-old\n+new"}],
        contents={"tests/ci/test/test_gate.py": "one\ntwo\nthree\n"},
    )
    client = FakeClient(response=valid_response)
    outcome, _ = analyze(tmp_path, [job()], gh, client)
    assert not outcome.unavailable
    assert ("pulls_for_commit", "a" * 40) in gh.calls
    assert ("pull_files", 55) in gh.calls
    assert any(call[:3] == ("file_content", "tests/ci/test/test_gate.py", "a" * 40) for call in gh.calls)
    evidence = json.loads(client.responses.calls[0]["input"])["evidence"]
    assert {item["kind"] for item in evidence} == {"job_log", "pull_request", "source"}


def test_invalid_model_response_discards_partial_reasons_and_marks_unavailable(tmp_path):
    # [FUNCTION DOC] The model returns an analysis missing required fields and covering only one of
    # two jobs, with an unknown evidence ref. Asserts the outcome is unavailable, no partial reasons
    # are kept, and the audit records a validation error.
    response = SimpleNamespace(
        output_text=json.dumps(
            {
                "schema_version": "1",
                "analyses": [
                    {
                        "job_id": 10,
                        "reason": "The first assertion failed.",
                        "category": "test_failure",
                        "confidence": "high",
                        "evidence_refs": ["unknown"],
                    }
                ],
            }
        )
    )
    outcome, emitted = analyze(
        tmp_path,
        [job(10), job(11)],
        FakeGitHub(logs={10: "AssertionError: one", 11: "AssertionError: two"}),
        FakeClient(response=response),
    )
    assert outcome.unavailable and outcome.reasons == {}
    assert '"validation":"error"' in emitted[0]


def test_policy_cap_omits_hidden_jobs_and_records_count(tmp_path):
    # [FUNCTION DOC] Runs four jobs with the job cap set to 2. Asserts all four ids get a reason, the
    # two beyond the cap get the unavailable reason, the omitted count is 2, and only the first two
    # jobs were sent to the model.
    client = FakeClient(response=valid_response)
    jobs = [job(index) for index in range(10, 14)]
    outcome, _ = analyze(
        tmp_path,
        jobs,
        FakeGitHub(logs={item["id"]: "AssertionError: fail" for item in jobs}),
        client,
        max_jobs=2,
    )
    assert set(outcome.reasons) == {10, 11, 12, 13}
    assert outcome.reasons[12][0].reason == ANALYZER.UNAVAILABLE_REASON
    assert outcome.reasons[13][0].reason == ANALYZER.UNAVAILABLE_REASON
    assert outcome.omitted_count == 2
    assert [item["job_id"] for item in json.loads(client.responses.calls[0]["input"])["jobs"]] == [10, 11]


def test_unauthorized_run_never_touches_github_or_openai(tmp_path):
    # [FUNCTION DOC] Runs the analysis for a workflow run whose repository (id 999, attacker/fork)
    # differs from the expected repo. Asserts the outcome is unavailable and neither GitHub nor the
    # model client received any call.
    client = FakeClient(response=valid_response)
    gh = FakeGitHub(logs={10: "AssertionError"})
    emitted = []
    outcome = ANALYZER.analyze_failures(
        run=run(repository={"id": 999, "full_name": "attacker/fork"}),
        jobs=[job()],
        repo="radixark/miles",
        gh=gh,
        policy_path=write_policy(tmp_path),
        prompt_path=PROMPT_PATH,
        schema_path=SCHEMA_PATH,
        client_factory=lambda timeout: client,
        emit=emitted.append,
    )
    assert outcome.unavailable
    assert gh.calls == [] and client.responses.calls == []


def test_oidc_provider_rejects_unapproved_host_before_sending_token(monkeypatch):
    # [FUNCTION DOC] Sets the OIDC request url to a non-GitHub host and replaces the url opener
    # builder with one that fails if used. Asserts getting a token raises the OIDC provider error with
    # a runtime-error cause, the audit reason is endpoint_validation, and neither the host nor the
    # request token appear in the error message. Modifies environment variables and the opener
    # builder via monkeypatch.
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://evil.invalid/oidc")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: (_ for _ in ()).throw(AssertionError("request must not be sent")),
    )
    provider = ANALYZER._github_actions_oidc_provider("openai-audience")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()
    assert type(caught.value.__cause__).__name__ == "RuntimeError"
    assert ANALYZER._analysis_error_audit(caught.value)["oidc_failure_reason"] == "endpoint_validation"
    assert "evil.invalid" not in str(caught.value)
    assert "sensitive-request-token" not in str(caught.value)


def test_oidc_callback_transport_failure_has_stable_safe_classification(monkeypatch):
    # [FUNCTION DOC] Uses a valid GitHub OIDC url and an opener that raises a url error carrying a
    # secret message. Asserts the provider error has a url-error cause and the audit is exactly the
    # github_oidc stage with error, cause and root types and reason request_error, with the secret
    # absent from the serialized audit.
    secret = "oidc-provider-secret-message"
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://pipelines.actions.githubusercontent.com/oidc?request_data=sensitive-query",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")

    class FailingOpener:
        # [CLASS DOC] HELPER: Opener whose open always fails.
        def open(self, request, timeout):
            # [FUNCTION DOC] HELPER: Always raises a url error whose message is the secret string.
            raise urllib.error.URLError(secret)

    monkeypatch.setattr(ANALYZER.urllib.request, "build_opener", lambda *args: FailingOpener())
    provider = ANALYZER._github_actions_oidc_provider("openai-audience")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()

    assert type(caught.value.__cause__).__name__ == "URLError"
    assert ANALYZER._analysis_error_audit(caught.value) == {
        "stage": "github_oidc",
        "error_type": "GitHubOIDCProviderError",
        "cause_type": "URLError",
        "root_cause_type": "URLError",
        "oidc_failure_reason": "request_error",
    }
    assert secret not in json.dumps(ANALYZER._analysis_error_audit(caught.value))


def test_wrapped_oidc_callback_failure_is_identified_without_emitting_messages(tmp_path):
    # [FUNCTION DOC] The model client raises an API connection error whose cause chain is an OIDC
    # provider error caused by a url error, all with secret messages. Asserts the outcome is
    # unavailable with no reasons, the emitted audit has stage github_oidc with the right error,
    # cause and root types, and none of the secret messages or query secrets are emitted.
    provider_error = with_cause(
        ANALYZER.GitHubOIDCProviderError("provider-secret-message"),
        urllib.error.URLError("oidc-cause-secret-message"),
    )
    error = with_cause(
        APIConnectionError("outer-secret-message", "https://api.openai.com/v1/responses?secret=query-secret"),
        provider_error,
    )
    outcome, emitted = analyze(
        tmp_path,
        [job()],
        FakeGitHub(logs={10: "AssertionError: fail"}),
        FakeClient(error=error),
    )

    audit = json.loads(emitted[0].split("=", 1)[1])
    assert outcome.unavailable and outcome.reasons == {}
    assert audit["stage"] == "github_oidc"
    assert audit["error_type"] == "APIConnectionError"
    assert audit["cause_type"] == "GitHubOIDCProviderError"
    assert audit["root_cause_type"] == "URLError"
    assert not any(
        secret in emitted[0]
        for secret in (
            "provider-secret-message",
            "oidc-cause-secret-message",
            "outer-secret-message",
            "query-secret",
        )
    )


@pytest.mark.parametrize(
    ("cause_url", "expected_stage"),
    [
        ("https://auth.openai.com/oauth/token?secret=wif-query-secret", "openai_wif_exchange"),
        ("https://api.openai.com/v1/responses?secret=api-query-secret", "openai_api_transport"),
    ],
)
def test_downstream_api_connection_error_classifies_cause_without_emitting_details(
    tmp_path, cause_url, expected_stage
):
    # [FUNCTION DOC] Parametrized by the cause's request url: the auth token endpoint maps to stage
    # openai_wif_exchange, the responses endpoint to openai_api_transport. The client raises an API
    # connection error caused by a connect error. Asserts the audit stage, error/cause/root types, and
    # that no secret message or query secret is emitted.
    error = with_cause(
        APIConnectionError("outer-transport-secret", "https://api.openai.com/v1/responses"),
        ConnectError("cause-transport-secret", cause_url),
    )
    outcome, emitted = analyze(
        tmp_path,
        [job()],
        FakeGitHub(logs={10: "AssertionError: fail"}),
        FakeClient(error=error),
    )

    audit = json.loads(emitted[0].split("=", 1)[1])
    assert outcome.unavailable and outcome.reasons == {}
    assert audit["stage"] == expected_stage
    assert audit["error_type"] == "APIConnectionError"
    assert audit["cause_type"] == "ConnectError"
    assert audit["root_cause_type"] == "ConnectError"
    assert "outer-transport-secret" not in emitted[0]
    assert "cause-transport-secret" not in emitted[0]
    assert "wif-query-secret" not in emitted[0]
    assert "api-query-secret" not in emitted[0]


def test_unknown_exception_class_and_message_are_not_emitted(tmp_path):
    # [FUNCTION DOC] The client raises an exception of an unknown locally defined class with a secret
    # message. Asserts the audit has stage openai_api and the generic other-error type for both error
    # and root cause, and neither the class name nor the message is emitted.
    class ArbitrarySecretFailure(Exception):
        # [CLASS DOC] HELPER: Unknown exception type used to prove class names are not leaked.
        pass

    outcome, emitted = analyze(
        tmp_path,
        [job()],
        FakeGitHub(logs={10: "AssertionError: fail"}),
        FakeClient(error=ArbitrarySecretFailure("arbitrary-secret-message")),
    )
    audit = json.loads(emitted[0].split("=", 1)[1])
    assert outcome.unavailable
    assert audit["stage"] == "openai_api"
    assert audit["error_type"] == "OtherError"
    assert audit["root_cause_type"] == "OtherError"
    assert "ArbitrarySecretFailure" not in emitted[0]
    assert "arbitrary-secret-message" not in emitted[0]


def test_nested_oidc_http_error_reports_only_allowlisted_root_and_integer_status():
    # [FUNCTION DOC] Builds an HTTP 403 error with sensitive url, reason, header and body, wrapped in
    # an OIDC provider error inside an API connection error. Serializes the audit through the audit
    # emitter and asserts it contains exactly stage github_oidc, the three type names, root type
    # HTTPError and the integer root status 403, with none of the sensitive strings emitted.
    root = urllib.error.HTTPError(
        "https://pipelines.actions.githubusercontent.com/oidc?jwt=secret-jwt",
        403,
        "sensitive-http-reason",
        {"Authorization": "Bearer sensitive-header-token"},
        io.BytesIO(b"sensitive-response-body"),
    )
    error = with_cause(
        APIConnectionError("outer-sensitive-message", "https://api.openai.com/v1/responses"),
        with_cause(ANALYZER.GitHubOIDCProviderError("provider-sensitive-message"), root),
    )
    emitted = []
    ANALYZER._audit(ANALYZER._analysis_error_audit(error), emitted.append)

    audit = json.loads(emitted[0].split("=", 1)[1])
    assert audit == {
        "stage": "github_oidc",
        "error_type": "APIConnectionError",
        "cause_type": "GitHubOIDCProviderError",
        "root_cause_type": "HTTPError",
        "root_http_status": 403,
    }
    assert isinstance(audit["root_http_status"], int)
    assert not any(
        secret in emitted[0]
        for secret in (
            "secret-jwt",
            "sensitive-http-reason",
            "sensitive-header-token",
            "sensitive-response-body",
            "outer-sensitive-message",
            "provider-sensitive-message",
        )
    )


@pytest.mark.parametrize(
    ("root", "expected_type"),
    [
        (urllib.error.URLError("sensitive-url-error"), "URLError"),
        (ssl.SSLError("sensitive-ssl-error"), "SSLError"),
        (json.JSONDecodeError("sensitive-json-error", "sensitive-json-document", 0), "JSONDecodeError"),
        (SchemaError("sensitive-schema-error"), "SchemaError"),
        (RuntimeError("sensitive-runtime-error"), "RuntimeError"),
    ],
)
def test_nested_root_types_are_allowlisted_without_messages(root, expected_type):
    # [FUNCTION DOC] Parametrized over root causes (url error, ssl error, JSON decode error, schema
    # error, runtime error) placed at the bottom of a chain of API connection error and OIDC provider
    # error. Asserts the audit stage is github_oidc, the root type name equals the expected type, no
    # http status is reported, and no sensitive text is serialized.
    error = with_cause(
        APIConnectionError("sensitive-outer-error", "https://api.openai.com/v1/responses?token=sensitive"),
        with_cause(ANALYZER.GitHubOIDCProviderError("sensitive-provider-error"), root),
    )
    audit = ANALYZER._analysis_error_audit(error)
    serialized = json.dumps(audit)

    assert audit["stage"] == "github_oidc"
    assert audit["error_type"] == "APIConnectionError"
    assert audit["cause_type"] == "GitHubOIDCProviderError"
    assert audit["root_cause_type"] == expected_type
    assert "root_http_status" not in audit
    assert "sensitive" not in serialized


def test_url_error_with_nested_ssl_reason_reports_ssl_as_root():
    # [FUNCTION DOC] A url error whose reason is a certificate-verification error is the cause of an
    # OIDC provider error. Asserts the audit reports the url error as the cause and the certificate
    # verification error as the root, with no sensitive text.
    ssl_error = ssl.SSLCertVerificationError("sensitive-certificate-message")
    url_error = urllib.error.URLError(ssl_error)
    error = with_cause(ANALYZER.GitHubOIDCProviderError("sensitive-provider-message"), url_error)

    audit = ANALYZER._analysis_error_audit(error)
    assert audit == {
        "stage": "github_oidc",
        "error_type": "GitHubOIDCProviderError",
        "cause_type": "URLError",
        "root_cause_type": "SSLCertVerificationError",
    }
    assert "sensitive" not in json.dumps(audit)


@pytest.mark.parametrize("payload", [b"{}", b'{"value":""}', b'{"other":"sensitive-token"}'])
def test_oidc_missing_token_value_has_fixed_reason_without_payload_leak(monkeypatch, payload):
    # [FUNCTION DOC] Parametrized over JSON payloads lacking a usable value field (empty object,
    # empty value, other key). With a valid endpoint url and an opener returning the payload, asserts
    # getting a token raises the OIDC provider error, the audit reason is missing_token_value, the
    # root type is runtime error, and nothing sensitive is serialized.
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://pipelines.actions.githubusercontent.com/oidc?request_data=sensitive-query",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda request, timeout: OIDCResponse(payload)),
    )
    provider = ANALYZER._github_actions_oidc_provider("openai-audience")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()

    audit = ANALYZER._analysis_error_audit(caught.value)
    assert audit["oidc_failure_reason"] == "missing_token_value"
    assert audit["root_cause_type"] == "RuntimeError"
    assert "sensitive" not in json.dumps(audit)


@pytest.mark.parametrize("payload", [b"not-sensitive-json", b'["sensitive-token"]', b"\xff"])
def test_oidc_response_decode_has_fixed_reason_without_payload_leak(monkeypatch, payload):
    # [FUNCTION DOC] Parametrized over undecodable payloads (non-JSON text, a JSON list, invalid
    # UTF-8 bytes). Asserts the OIDC provider error is raised with audit reason response_decode, a
    # root type from the allowed set (JSON decode error, value error, or other error), and no
    # sensitive text in the audit.
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://pipelines.actions.githubusercontent.com/oidc?request_data=sensitive-query",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda request, timeout: OIDCResponse(payload)),
    )
    provider = ANALYZER._github_actions_oidc_provider("openai-audience")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()

    audit = ANALYZER._analysis_error_audit(caught.value)
    assert audit["oidc_failure_reason"] == "response_decode"
    assert audit["root_cause_type"] in {"JSONDecodeError", "ValueError", "OtherError"}
    assert "sensitive" not in json.dumps(audit)


def test_arbitrary_oidc_reason_attribute_cannot_spoof_audit():
    # [FUNCTION DOC] Creates a runtime-error subclass instance and attaches spoofed reason, http
    # status and url attributes. Asserts the audit ignores them and is exactly the generic
    # openai_api stage with other-error types, with no sensitive text.
    class AttributeSpoofError(RuntimeError):
        # [CLASS DOC] HELPER: Ordinary runtime-error subclass used to carry spoofed attributes.
        pass

    error = AttributeSpoofError("sensitive-message")
    error.oidc_failure_reason = "endpoint_validation"
    error.root_http_status = 418
    error.url = "https://evil.invalid/?token=sensitive-token"

    audit = ANALYZER._analysis_error_audit(error)
    assert audit == {
        "stage": "openai_api",
        "error_type": "OtherError",
        "root_cause_type": "OtherError",
    }
    assert "sensitive" not in json.dumps(audit)


def test_oidc_marker_subclass_attribute_cannot_spoof_reason():
    # [FUNCTION DOC] Creates a subclass of the OIDC provider error and sets a spoofed failure reason
    # attribute on an instance. Asserts the audit classifies it as stage github_oidc with its own
    # class treated as the provider error type, and without any failure reason field.
    class MarkerSpoofError(ANALYZER.GitHubOIDCProviderError):
        # [CLASS DOC] HELPER: Subclass of the provider error that is not an exact marker type.
        pass

    error = MarkerSpoofError("sensitive-message")
    error.oidc_failure_reason = "missing_token_value"

    audit = ANALYZER._analysis_error_audit(error)
    assert audit == {
        "stage": "github_oidc",
        "error_type": "GitHubOIDCProviderError",
        "root_cause_type": "GitHubOIDCProviderError",
    }
    assert "sensitive" not in json.dumps(audit)


@pytest.mark.parametrize(
    "url",
    [
        "https://pipelines.actions.githubusercontent.com/oidc/token?request_data=opaque",
        "https://pipelines-ghubeus1.actions.githubusercontent.com/oidc/token?request_data=opaque",
        "https://regional.runner.actions.githubusercontent.com/oidc/token?request_data=opaque",
        "https://PIPELINES.ACTIONS.GITHUBUSERCONTENT.COM:443/oidc/token?request_data=opaque",
    ],
)
def test_github_com_oidc_endpoint_accepts_strict_actions_subdomains(monkeypatch, url):
    # [FUNCTION DOC] Parametrized over accepted endpoint urls (pipelines, regional and runner
    # subdomains of actions.githubusercontent.com, case-insensitive host, explicit port 443). Sets the
    # environment, captures opener handlers, and asserts the provider returns the token from the
    # payload, exactly one request is made with a 15 second timeout, and a no-redirect handler is
    # installed.
    opener = CapturingOIDCOpener()
    handlers = []
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", url)
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: handlers.extend(args) or opener,
    )

    provider = ANALYZER._github_actions_oidc_provider("https://api.openai.com/v1")
    assert provider["get_token"]() == "header.payload.signature"
    assert len(opener.requests) == 1
    assert opener.requests[0][1] == 15
    assert any(isinstance(handler, ANALYZER._NoRedirect) for handler in handlers)


def test_oidc_endpoint_preserves_opaque_path_and_query_while_replacing_audience(monkeypatch):
    # [FUNCTION DOC] Uses an endpoint url with an encoded path and a query containing repeated keys,
    # a valueless flag and two audience parameters (one with an encoded name). Asserts the requested
    # url keeps the path and the other query parameters in order, drops all old audience parameters,
    # and appends the new audience percent-encoded (spaces as plus signs, slash encoded).
    opener = CapturingOIDCOpener()
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://regional.actions.githubusercontent.com/opaque/%2Fpath?first=a%2Fb&repeat=1&repeat=2&audience=old&%61udience=older&flag",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(ANALYZER.urllib.request, "build_opener", lambda *args: opener)

    provider = ANALYZER._github_actions_oidc_provider("new audience/value")
    assert provider["get_token"]() == "header.payload.signature"
    requested_url = opener.requests[0][0].full_url
    assert requested_url == (
        "https://regional.actions.githubusercontent.com/opaque/%2Fpath?"
        "first=a%2Fb&repeat=1&repeat=2&flag&audience=new+audience%2Fvalue"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://actions.githubusercontent.com/oidc/token",
        "https://evilactions.githubusercontent.com/oidc/token",
        "https://pipelines.actions.githubusercontent.com.evil.example/oidc/token",
        "https://actions.githubusercontent.com.evil.example/oidc/token",
        "https://pipelines.actions.githubusercontent.com./oidc/token",
        "https://127.0.0.1/oidc/token",
        "https://[::1]/oidc/token",
        "https://user@pipelines.actions.githubusercontent.com/oidc/token",
        "https://user:password@pipelines.actions.githubusercontent.com/oidc/token",
        "https://pipelines.actions.githubusercontent.com:444/oidc/token",
        "https://pipelines.actions.githubusercontent.com:/oidc/token",
        "https://pipelines.actions.githubusercontent.com:notaport/oidc/token",
        "https://pipelines.actions.githubusercontent.com:65536/oidc/token",
        "https://pipelines.actions.githubusercontent.com/oidc/token#fragment",
        "https://pipelines.actions.githubusercontent.com/oidc/token#",
        "http://pipelines.actions.githubusercontent.com/oidc/token",
        "https://pipelines.actions.githubusercontent.com",
        "https://pipelines.actions.githubusercontent.com/",
        "https://pipelines.actions.githubusercontent.com//other/path",
        "https://pipelines.actions.githubusercontent.com/\\other/path",
        "https://pipelines.actions.githubusercontent.com/oidc path",
        "https://pipelines.actions.githubusercontent.com/oidc\tpath",
        "https://-regional.actions.githubusercontent.com/oidc/token",
        "https://regional-.actions.githubusercontent.com/oidc/token",
        "https://regional..actions.githubusercontent.com/oidc/token",
    ],
)
def test_github_com_oidc_endpoint_rejects_unsafe_or_malformed_urls_before_request(monkeypatch, url):
    # [FUNCTION DOC] Parametrized over rejected urls: bare or look-alike hosts, trailing-dot host, IP
    # literals, userinfo, wrong or malformed ports, fragments, plain http, empty or double-slash or
    # backslash paths, whitespace in the path, and malformed host labels. The opener builder fails
    # if called. Asserts getting a token raises the provider error with reason endpoint_validation
    # and that the url is not echoed in the audit.
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", url)
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: (_ for _ in ()).throw(AssertionError("request must not be sent")),
    )

    provider = ANALYZER._github_actions_oidc_provider("https://api.openai.com/v1")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()
    audit = ANALYZER._analysis_error_audit(caught.value)
    assert audit["oidc_failure_reason"] == "endpoint_validation"
    assert url not in json.dumps(audit)


def test_oidc_provider_does_not_follow_redirects(monkeypatch):
    # [FUNCTION DOC] The opener raises an HTTP 302 error pointing at a foreign host. Asserts exactly
    # one request was made, the no-redirect handler was installed, the audit reason is request_error
    # with root status 302, and neither the redirect host nor sensitive text is in the audit.
    redirect = urllib.error.HTTPError(
        "https://regional.actions.githubusercontent.com/oidc/token",
        302,
        "sensitive-redirect-reason",
        {"Location": "https://evil.invalid/?token=sensitive-token"},
        None,
    )
    opener = CapturingOIDCOpener(error=redirect)
    handlers = []
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://regional.actions.githubusercontent.com/oidc/token?request_data=opaque",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: handlers.extend(args) or opener,
    )

    provider = ANALYZER._github_actions_oidc_provider("https://api.openai.com/v1")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()
    audit = ANALYZER._analysis_error_audit(caught.value)
    assert len(opener.requests) == 1
    assert any(isinstance(handler, ANALYZER._NoRedirect) for handler in handlers)
    assert audit["oidc_failure_reason"] == "request_error"
    assert audit["root_http_status"] == 302
    assert "evil.invalid" not in json.dumps(audit)
    assert "sensitive" not in json.dumps(audit)
