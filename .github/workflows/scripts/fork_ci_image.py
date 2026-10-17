#!/usr/bin/env python3
# [MODULE DOC]
# Trusted-side orchestration for building a CI Docker image for a pull request that comes from a
# fork. It runs from the default branch (with a trusted checkout of this repository as the root
# for imports) in several modes selected by a positional argument: resolve (wait for the PR test
# run to upload a fork image request artifact, validate it, and write the request, pull number
# and merge sha to the GitHub output file), wait-build (wait for the separate publisher workflow
# run that matches the source run to finish successfully), check-source (validate the untrusted
# merge checkout), build (validate, then run the repository docker build script with the OCI
# output destination from the environment), check-current (validate only) and finish (verify the
# published multi-platform image carries the expected inputs hash and, if a forced rebuild was
# requested, remove the rebuild label from the pull request). All GitHub access is through the gh
# command line tool via subprocess. Every validation failure raises a value error; subprocess
# failures propagate. Environment: GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT,
# GITHUB_EVENT_PATH, GITHUB_OUTPUT, REQUEST_JSON (a JSON request) and OCI_OUTPUT.
# The request is a small JSON document (inside a zip artifact) with the pull number, merge sha,
# inputs hash, source run id and attempt, a force rebuild flag and the PR label names.
# doc-dev: docs/developer/ci/02-docker-build.md
"""Validate a fork image request for default-branch build orchestration."""

import argparse
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docker"))

import image_inputs  # noqa: E402
from tests.ci.ci_policy import resolve_policy, resolve_workflow_inputs  # noqa: E402
from tests.ci.file_run import collect_snapshot_tests  # noqa: E402
from tests.ci.hardware import CUDA_STAGES  # noqa: E402
from tests.ci.stage_selection import select_skipped_gpu_stages  # noqa: E402

CPU_JOBS = {f"stage-a-cpu ({partition}) / run-cpu" for partition in range(4)}
REQUEST_KEYS = {"pr", "merge_sha", "inputs_hash", "run_id", "run_attempt", "force_rebuild", "labels"}


def require(condition, message):
    # [FUNCTION DOC] HELPER: Assertion helper; raises a value error with the given message when the
    # condition is false, else does nothing.
    if not condition:
        raise ValueError(message)


def api(path, *, method="GET", collection=None):
    # [FUNCTION DOC] HELPER: Calls the GitHub REST API through the gh command line tool with the
    # given path and HTTP method (default get). Returns the parsed JSON, or none when the output is
    # blank. If a collection key name is given, requests all pages merged into a list of pages and
    # returns the flat list of items found under that key in every page. A failing command raises
    # a subprocess error. Side effect: runs a subprocess and network access.
    command = ["gh", "api", "-X", method, path]
    if collection:
        command += ["--paginate", "--slurp"]
    data = subprocess.check_output(command)
    result = json.loads(data) if data.strip() else None
    if collection:
        return [item for page in result for item in page[collection]]
    return result


def parse_request(data):
    # [FUNCTION DOC] Parses and validates the request artifact bytes. The archive must be at most
    # 65536 bytes, a zip containing exactly one entry named request.json of at most 4096 bytes
    # uncompressed, which must decode to a JSON object with exactly the expected keys (pr, merge
    # sha, inputs hash, run id, run attempt, force rebuild, labels). The pr, run id and attempt must
    # be positive integers, force rebuild a boolean, labels a list of strings, merge sha forty and
    # inputs hash sixty-four lowercase hex characters. Returns the request mapping; raises a value
    # error on violations (a bad zip or JSON raises the library's error).
    require(len(data) <= 65536, "Request archive is too large")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        require(archive.namelist() == ["request.json"], "Expected only request.json")
        info = archive.getinfo("request.json")
        require(info.file_size <= 4096, "Request JSON is too large")
        request = json.loads(archive.read(info))
    require(isinstance(request, dict) and request.keys() == REQUEST_KEYS, "Invalid request fields")
    for key in ("pr", "run_id", "run_attempt"):
        require(type(request[key]) is int and request[key] > 0, f"Invalid {key}")
    require(type(request["force_rebuild"]) is bool, "Invalid force_rebuild")
    require(
        isinstance(request["labels"], list) and all(isinstance(label, str) for label in request["labels"]),
        "Invalid labels",
    )
    for key, length in (("merge_sha", 40), ("inputs_hash", 64)):
        require(
            isinstance(request[key], str) and re.fullmatch(f"[0-9a-f]{{{length}}}", request[key]), f"Invalid {key}"
        )
    return request


def validate_identity(request, run, pr, repository, workflow_id):
    # [FUNCTION DOC] Checks that the request, the live workflow run, the live pull request and the
    # repository agree; raises a value error naming the first failure, returns nothing. Requires:
    # run id and attempt equal the request's; the run is a pull request event and in progress; the
    # run's repository is the given one; the run's workflow id equals the PR test workflow id and
    # its path (ignoring the ref suffix) is the PR test workflow file; the pull request number
    # matches and is open; its base repository is the given one; the head repository exists, differs
    # from the base (a fork), and equals the run's head repository; the PR head sha and branch equal
    # the run's. Finally the run must list, among its referenced workflows, the image build
    # workflow pinned at the request's merge sha with the pull request merge ref and the same sha
    # (reruns keep the original merge sha).
    require(run["id"] == request["run_id"] and run["run_attempt"] == request["run_attempt"], "Stale run attempt")
    require(run["event"] == "pull_request" and run["status"] == "in_progress", "Expected an active PR run")
    require(run["repository"]["full_name"] == repository, "Wrong run repository")
    require(
        run["workflow_id"] == workflow_id and run["path"].split("@")[0] == ".github/workflows/pr-test.yml",
        "Wrong source workflow",
    )
    require(pr["number"] == request["pr"] and pr["state"] == "open", "PR is no longer open")
    require(pr["base"]["repo"]["full_name"] == repository, "Wrong PR base repository")
    head = pr["head"]
    require(head["repo"] and head["repo"]["id"] != pr["base"]["repo"]["id"], "Expected a fork PR")
    require(head["repo"]["id"] == run["head_repository"]["id"], "Wrong fork repository")
    require(head["sha"] == run["head_sha"] and head["ref"] == run["head_branch"], "PR head moved")
    # Reruns keep the original merge SHA even when main has advanced.
    reference = f"{repository}/.github/workflows/_build-pr-ci-image.yml@{request['merge_sha']}"
    require(
        any(
            entry["path"] == reference
            and entry.get("ref") == f"refs/pull/{request['pr']}/merge"
            and entry["sha"] == request["merge_sha"]
            for entry in run.get("referenced_workflows", [])
        ),
        "Merge SHA is not the image workflow's frozen source",
    )


def current_request(request, repository):
    # [FUNCTION DOC] Re-validates a request against live GitHub state and returns a pair of the
    # live pull request object and the CI run policy. Fetches the source run, pull request and PR
    # test workflow, validates identity, then lists PR test runs for the head sha filtered to the
    # same head repository and branch, requiring that the request's run is the newest (highest id)
    # so a newer run supersedes the request. Recomputes the run policy from the PR event labels
    # saved in the request (so reruns and later label removal do not change it). Unless the policy
    # bypasses fast-fail, requires that the four CPU stage-A jobs all exist and concluded with
    # success, taking for each job name the latest attempt not newer than the request's attempt
    # (partial reruns retain earlier successes). Raises a value error on any failure.
    run = api(f"repos/{repository}/actions/runs/{request['run_id']}")
    pr = api(f"repos/{repository}/pulls/{request['pr']}")
    workflow = api(f"repos/{repository}/actions/workflows/pr-test.yml")
    validate_identity(request, run, pr, repository, workflow["id"])
    head = pr["head"]
    query = urlencode({"event": "pull_request", "head_sha": head["sha"], "per_page": 100})
    runs = api(f"repos/{repository}/actions/workflows/pr-test.yml/runs?{query}", collection="workflow_runs")
    matching = [
        item["id"]
        for item in runs
        if item["head_repository"]["id"] == head["repo"]["id"] and item["head_branch"] == head["ref"]
    ]
    require(matching and max(matching) == run["id"], "A newer PR Test run superseded this request")
    # Match the caller's event snapshot, including on reruns and after label removal.
    policy = resolve_workflow_inputs("pull_request", "", json.dumps(request["labels"]))
    if not policy.bypass_fastfail:
        jobs = api(
            f"repos/{repository}/actions/runs/{run['id']}/jobs?filter=all&per_page=100",
            collection="jobs",
        )
        # Partial reruns retain successful dependencies from earlier attempts.
        cpu = {}
        for job in sorted(jobs, key=lambda job: job["run_attempt"]):
            if job["name"] in CPU_JOBS and job["run_attempt"] <= request["run_attempt"]:
                cpu[job["name"]] = job
        require(
            cpu.keys() == CPU_JOBS and all(job["conclusion"] == "success" for job in cpu.values()),
            "CPU A gate did not pass",
        )
    return pr, policy


def resolve_request(event, repository):
    # [FUNCTION DOC] Looks for the fork image request artifact of the workflow run in the event.
    # Lists the run's artifacts, picks those named with the run id and attempt that are not
    # expired; returns none if there are none. Requires exactly one match of at most 65536 bytes,
    # downloads its zip, parses it, requires that its run id and attempt equal the event's run, and
    # revalidates it against live state. Returns the request mapping; raises a value error on
    # violations. Side effect: gh subprocess calls.
    run = event["workflow_run"]
    artifacts = api(f"repos/{repository}/actions/runs/{run['id']}/artifacts?per_page=100", collection="artifacts")
    name = f"fork-ci-image-{run['id']}-{run['run_attempt']}"
    matches = [item for item in artifacts if item["name"] == name and not item["expired"]]
    if not matches:
        return None
    require(len(matches) == 1 and matches[0]["size_in_bytes"] <= 65536, "Invalid request artifact")
    data = subprocess.check_output(["gh", "api", f"repos/{repository}/actions/artifacts/{matches[0]['id']}/zip"])
    request = parse_request(data)
    require(
        request["run_id"] == run["id"] and request["run_attempt"] == run["run_attempt"],
        "Request belongs to another attempt",
    )
    current_request(request, repository)
    return request


def wait_request(event, repository):
    # [FUNCTION DOC] Polls every thirty seconds until the source run uploads a valid request.
    # Each iteration: refetch the run; return none if its attempt changed or it completed; try to
    # resolve a request and return it if found; return none if the run's docker build job (either
    # naming form) of this attempt has already completed (so no request will come); else sleep.
    # Blocks, with no overall timeout.
    source = event["workflow_run"]
    while True:
        run = api(f"repos/{repository}/actions/runs/{source['id']}")
        if run["run_attempt"] != source["run_attempt"] or run["status"] == "completed":
            return None
        request = resolve_request(event, repository)
        if request:
            return request
        jobs = api(
            f"repos/{repository}/actions/runs/{source['id']}/jobs?filter=latest&per_page=100", collection="jobs"
        )
        if any(
            job["name"] in ("docker-build", "docker-build / docker-build")
            and job["run_attempt"] == source["run_attempt"]
            and job["status"] == "completed"
            for job in jobs
        ):
            return None
        time.sleep(30)


def require_active_source(request, repository):
    # [FUNCTION DOC] Fetches the request's source run and requires it still be in progress at the
    # same attempt (else a value error saying the source CI attempt is no longer active). Returns
    # the run mapping.
    run = api(f"repos/{repository}/actions/runs/{request['run_id']}")
    require(
        run["run_attempt"] == request["run_attempt"] and run["status"] == "in_progress",
        "Source CI attempt is no longer active",
    )
    return run


def wait_build(request, repository):
    # [FUNCTION DOC] Waits for the publisher workflow run that builds this request's image. The
    # request only needs the source run id and attempt here. Loops with thirty-second sleeps while
    # the source run stays active: until found, lists publisher workflow runs (workflow_run event,
    # created at or after the source run's creation) whose display title is the fork image title
    # built from run id and attempt and picks the newest, printing its URL; once found, refetches
    # it each loop. When the publisher has completed, requires a success conclusion and that its
    # latest jobs include a successful job named build (value error otherwise). Returns nothing.
    # Blocks; no overall timeout other than the source run ending.
    source = require_active_source(request, repository)
    workflow = api(f"repos/{repository}/actions/workflows/build-fork-ci-image.yml")
    query = urlencode({"event": "workflow_run", "created": f">={source['created_at']}", "per_page": 100})
    title = f"fork-ci-image-{request['run_id']}-{request['run_attempt']}"
    publisher = None
    while True:
        require_active_source(request, repository)
        if publisher is None:
            runs = api(
                f"repos/{repository}/actions/workflows/{workflow['id']}/runs?{query}", collection="workflow_runs"
            )
            matches = [
                run
                for run in runs
                if run["display_title"] == title
                and run["event"] == "workflow_run"
                and run["workflow_id"] == workflow["id"]
            ]
            if matches:
                publisher = max(matches, key=lambda run: run["id"])
                print(f"Waiting for {publisher['html_url']}", flush=True)
        else:
            publisher = api(f"repos/{repository}/actions/runs/{publisher['id']}")
        if publisher and publisher["status"] == "completed":
            require(
                publisher["conclusion"] == "success",
                f"Image publication {publisher['conclusion']}: {publisher['html_url']}",
            )
            jobs = api(
                f"repos/{repository}/actions/runs/{publisher['id']}/jobs?filter=latest&per_page=100", collection="jobs"
            )
            require(
                any(job["name"] == "build" and job["conclusion"] == "success" for job in jobs),
                "Publisher completed without a successful image build",
            )
            return
        time.sleep(30)


def build_image(request, repository, source):
    # [FUNCTION DOC] Builds the image from the validated source checkout by running the repository
    # docker build script in its own session/process group with the cu13 variant, the custom image
    # tag mode with a custom tag naming the pull request, the source as build context, and an OCI
    # output at the destination given by the OCI output environment variable (a missing variable
    # raises a key error). Beforehand and every thirty seconds while it runs, requires the source CI
    # attempt to be still active (this raises a value error and aborts the build). A non-zero exit
    # raises a subprocess error. On any exit path, if the child is still running its process group
    # receives a terminate signal, then after ten seconds a kill signal, and the child is reaped.
    require_active_source(request, repository)
    command = [
        sys.executable,
        str(ROOT / "docker/build.py"),
        "--variant",
        "cu13",
        "--image-tag",
        "custom",
        "--custom-tag",
        f"pr-{request['pr']}",
        "--context",
        str(source),
        "--output",
        f"type=oci,dest={os.environ['OCI_OUTPUT']},tar=false",
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while True:
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                require_active_source(request, repository)
                continue
            if code:
                raise subprocess.CalledProcessError(code, command)
            return
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                # build.py may exit before a buildx child that ignores SIGTERM.
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                process.wait()


def validate_source(request, pr, policy, source):
    # [FUNCTION DOC] Validates the untrusted merge checkout directory against the request; raises
    # a value error on failure and returns nothing. Requires: HEAD equals the requested merge sha;
    # HEAD is a two-parent merge whose second parent is the PR head sha; none of the build input
    # paths (computed at HEAD, plus the Dockerfile and the two dockerignore files) nor any of their
    # parent directories is a symbolic link; the inputs hash computed at HEAD equals the request's;
    # differs from the hash at the first parent (the base); and equals the hash computed from the
    # working tree. Finally derives the run policy from the policy's cadence and raw labels, finds
    # which GPU stages would be skipped using the checkout's registered tests, and requires at
    # least one CUDA stage remains selected.
    # KNOWN ISSUE: relies on a private function of the image inputs module for path listing.
    def git(*args):
        # [FUNCTION DOC] HELPER: Runs a git command in the source directory and returns its
        # whitespace-stripped text output; a failing command raises a subprocess error.
        return subprocess.check_output(["git", *args], cwd=source, text=True).strip()

    require(git("rev-parse", "HEAD") == request["merge_sha"], "Checkout is not the requested merge")
    parents = git("rev-list", "--parents", "-n", "1", "HEAD").split()
    require(len(parents) == 3 and parents[2] == pr["head"]["sha"], "Merge does not contain the fork head")
    # No host-side reader may follow a PR symlink into the trusted checkout or credentials.
    paths = image_inputs._paths_at("HEAD", source) + [
        "docker/Dockerfile",
        ".dockerignore",
        "docker/Dockerfile.dockerignore",
    ]
    for name in paths:
        path = source
        for component in Path(name).parts:
            path /= component
            require(not path.is_symlink(), f"Build input must not be a symlink: {name}")
    current = image_inputs.compute("HEAD", root=source)
    require(current == request["inputs_hash"], "Requested inputs hash does not match trusted computation")
    require(current != image_inputs.compute("HEAD^1", root=source), "Build inputs match the base")
    require(current == image_inputs.compute(root=source), "Build context differs from the frozen inputs")
    run_policy = resolve_policy(policy.cadence, set(policy.raw_labels))
    skipped = select_skipped_gpu_stages(
        event_name="pull_request",
        changed_files=None,
        registrations=collect_snapshot_tests(source),
        run_policy=run_policy,
        raw_labels=policy.raw_labels,
    )
    require(CUDA_STAGES.keys() - set(skipped), "No CUDA tests selected")


def verify_published(request):
    # [FUNCTION DOC] Inspects the published pull request image tag in the docker hub repository via
    # docker buildx imagetools (JSON image format). For each of the linux amd64 and arm64
    # platforms (matching entries named exactly the platform or starting with it plus a slash, such
    # as variant suffixes), requires at least one config and that every config's inputs label equals
    # the request's inputs hash; otherwise a value error naming the platform. Side effect: runs a
    # subprocess with network access.
    manifest = json.loads(
        subprocess.check_output(
            [
                "docker",
                "buildx",
                "imagetools",
                "inspect",
                f"radixark/miles:pr-{request['pr']}",
                "--format",
                "{{ json .Image }}",
            ],
            text=True,
        )
    )
    for platform in ("linux/amd64", "linux/arm64"):
        configs = [config for key, config in manifest.items() if key == platform or key.startswith(platform + "/")]
        require(
            configs
            and all(image_inputs.read_label(json.dumps(config)) == request["inputs_hash"] for config in configs),
            f"Published {platform} inputs do not match",
        )


def main():
    # [FUNCTION DOC] Command line entry point. Takes a mode argument (resolve, wait-build,
    # check-source, build, check-current, finish) and an optional source directory path; reads the
    # repository name from the environment. wait-build: waits on the publisher using only the run id
    # and attempt from the environment. resolve: loads the workflow run event JSON from the event
    # path, waits for a request and, if found, appends request JSON, pull number and merge sha lines
    # to the GitHub output file. Other modes: load the request JSON from the request environment
    # variable, revalidate it live, validate the source directory (resolved to an absolute path;
    # note the source argument is required in practice though declared optional), then for build
    # run the image build, and for finish verify the published image and, if a forced rebuild was
    # requested, delete the rebuild label from the pull request, only warning if that fails.
    # check-source and check-current perform only the common validation. Returns nothing.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["resolve", "wait-build", "check-source", "build", "check-current", "finish"])
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    if args.mode == "wait-build":
        wait_build(
            dict(run_id=int(os.environ["GITHUB_RUN_ID"]), run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"])),
            repository,
        )
        return
    if args.mode == "resolve":
        request = wait_request(json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()), repository)
        if request:
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(
                    f"request={json.dumps(request, separators=(',', ':'))}\npr={request['pr']}\nmerge_sha={request['merge_sha']}\n"
                )
        return

    request = json.loads(os.environ["REQUEST_JSON"])
    pr, policy = current_request(request, repository)
    validate_source(request, pr, policy, args.source.resolve())
    if args.mode == "build":
        build_image(request, repository, args.source.resolve())
    elif args.mode == "finish":
        verify_published(request)
        if request["force_rebuild"]:
            try:
                api(f"repos/{repository}/issues/{request['pr']}/labels/rebuild-ci-image", method="DELETE")
            except subprocess.CalledProcessError:
                print(
                    "::warning::Could not remove the rebuild-ci-image label; remove it by hand or every run will rebuild."
                )


if __name__ == "__main__":
    main()
