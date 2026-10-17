# [MODULE DOC] CPU tests for the wheel-building GitHub workflow. They load the workflow YAML and run
# its embedded shell scripts for real in subprocesses, with fake gh and uploader executables on the
# path, verifying: scoped dispatch resolves only the selected source, scheduled runs skip the
# disabled TransformerEngine source, the router builds only when the release manifest is missing or
# stale, lookup failures stop the run, the publish step passes the superseded-keeping flag only in
# the transition cases, the app token is narrowly scoped, and the failure notifier's wiring. Module
# constants: repository root, parsed workflow, and a table of mock source endpoints with fake
# commit hashes. Registered as a CPU test in the first CPU stage.
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/build-wheels.yml").read_text())
SOURCES = {
    "sgl-router": ("repos/radixark/sgl-router-for-miles/commits/main", "a" * 40),
    "int4_qat": (
        "repos/radixark/miles/commits?sha=main&path=miles/kernels/quant/int4_fake&per_page=1",
        "b" * 40,
    ),
    "te": ("repos/radixark/TransformerEngine/commits/miles-main", "c" * 40),
}


def run_check(tmp_path, wheels, selected, *, event="workflow_dispatch", force="true", failure=None, built=None):
    # [FUNCTION DOC] HELPER: Runs the workflow's "check" step script under bash with a fake gh
    # executable (a small python script written into a temporary bin directory) and returns the
    # completed process plus the parsed key=value lines it wrote to the output file (empty mapping
    # if no output file). The fake gh answers from a JSON table of the selected sources (endpoint to
    # commit hash), returns a manifest URL for release-tag lookups when a built commit is mocked
    # (empty text otherwise), returns the manifest JSON holding that commit, exits with an error
    # for the endpoint named as the failure, and for any other endpoint exits with an error.
    # Environment: workflow env plus event name, wheels input, force flag, TE ref, and temporary
    # summary and output files. Inputs: temp path, wheels selection text, list of selected source
    # names, event (default manual dispatch), force text (default true), failing endpoint, and the
    # commit the mocked release was built from.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "endpoint = args[3] if args[1] == '-H' else args[1]\n"
        "sources = json.loads(os.environ['SOURCES'])\n"
        "if endpoint == os.environ['FAILURE']:\n"
        "    sys.exit('selected lookup unavailable')\n"
        "if endpoint in sources:\n"
        "    print(sources[endpoint])\n"
        "elif '/releases/tags/' in endpoint:\n"
        "    print('https://example.test/manifest' if os.environ['MOCK_RELEASE_COMMIT'] else '')\n"
        "elif endpoint == 'https://example.test/manifest':\n"
        "    print(json.dumps({'commit': os.environ['MOCK_RELEASE_COMMIT']}))\n"
        "else:\n"
        "    sys.exit('unexpected lookup: ' + endpoint)\n"
    )
    gh.chmod(0o755)
    output = tmp_path / "output"
    script = next(step["run"] for step in WORKFLOW["jobs"]["check"]["steps"] if step.get("id") == "check")
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            **WORKFLOW["env"],
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GITHUB_REPOSITORY": "radixark/miles",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "EVENT": event,
            "WHEELS_INPUT": wheels,
            "FORCE": force,
            "TE_REF": "miles-main",
            "SOURCES": json.dumps(dict(SOURCES[name] for name in selected)),
            "FAILURE": failure or "",
            "MOCK_RELEASE_COMMIT": built or "",
        },
        capture_output=True,
        text=True,
    )
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines()) if output.exists() else {}
    return result, outputs


@pytest.mark.parametrize("wheels", SOURCES)
def test_scoped_dispatch_does_not_resolve_other_sources(tmp_path, wheels):
    # [FUNCTION DOC] Parametrized over each source name. Runs the check with only that source
    # available; asserts success, that the router matrix is non-empty only for the router, the int4
    # and TE flags are true only for their own source, and that each sha output is the mock commit
    # for the selected source and empty for the others.
    result, outputs = run_check(tmp_path, wheels, [wheels])
    assert result.returncode == 0, result.stderr
    assert bool(json.loads(outputs["router_matrix"])) == (wheels == "sgl-router")
    assert outputs["int4_x86"] == str(wheels == "int4_qat").lower()
    assert outputs["te_x86"] == str(wheels == "te").lower()
    for name, output in [("sgl-router", "router_sha"), ("int4_qat", "int4_sha"), ("te", "te_sha")]:
        assert outputs[output] == (SOURCES[name][1] if name == wheels else "")


def test_schedule_does_not_resolve_disabled_te_source(tmp_path):
    # [FUNCTION DOC] Runs a scheduled "all" check where only the router and int4 sources are
    # reachable; asserts success, the TE flag false with an empty TE sha, the int4 flag true, and a
    # router matrix of two entries.
    result, outputs = run_check(tmp_path, "all", ["sgl-router", "int4_qat"], event="schedule")
    assert result.returncode == 0, result.stderr
    assert outputs["te_x86"] == "false"
    assert outputs["te_sha"] == ""
    assert outputs["int4_x86"] == "true"
    assert len(json.loads(outputs["router_matrix"])) == 2


@pytest.mark.parametrize("built", [None, "d" * 40, "a" * 40])
def test_router_builds_only_when_manifest_is_missing_or_stale(tmp_path, built):
    # [FUNCTION DOC] Parametrized over the built commit: none (manifest missing), a different hash
    # (stale), and the current hash. With force off, asserts success and that both the router and
    # publish matrices have two entries unless the built commit equals the current one, in which
    # case both are empty.
    result, outputs = run_check(tmp_path, "sgl-router", ["sgl-router"], force="false", built=built)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(outputs["router_matrix"])) == (0 if built == "a" * 40 else 2)
    assert len(json.loads(outputs["publish_matrix"])) == (0 if built == "a" * 40 else 2)


@pytest.mark.parametrize(
    "failure",
    [SOURCES["sgl-router"][0], "repos/radixark/miles-wheels/releases/tags/cu130-torch213-x86_64"],
)
def test_selected_lookup_failure_stops_the_run(tmp_path, failure):
    # [FUNCTION DOC] Parametrized over two failing endpoints (the source commit lookup and the
    # release tag lookup). With force off, asserts a nonzero exit, the failure message on standard
    # error, and that no outputs were written.
    result, outputs = run_check(tmp_path, "sgl-router", ["sgl-router"], force="false", failure=failure)
    assert result.returncode != 0
    assert "selected lookup unavailable" in result.stderr
    assert not outputs


@pytest.mark.parametrize("te_on_schedule,keep", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize(
    "arch,build_arch,sets",
    [
        ("x86_64", "x86", ["sgl-router", "int4_qat", "te"]),
        ("x86_64", "x86", ["te"]),
        ("aarch64", "aarch64", ["sgl-router"]),
    ],
)
@pytest.mark.parametrize("upload_fails", [False, True])
def test_publish_retains_only_te_during_transition(
    tmp_path, te_on_schedule, keep, arch, build_arch, sets, upload_fails
):
    # [FUNCTION DOC] Parametrized over TE-on-schedule, keep flag, architecture with its build arch
    # and wheel sets, and upload failure. Runs the last publish step script under bash with a fake
    # python symlink, a no-op pip, and a fake uploader that logs the wheel files in the wheel
    # directory and its arguments as JSON lines, exiting nonzero when told to fail. Wheel files
    # (one empty file per set name) are placed in the layout the download step implies (flat if the
    # merge option is on, else one directory per architecture and name). Asserts: on upload failure
    # a nonzero exit and exactly one upload attempt; otherwise success, one upload per set, and for
    # each set the exact argument list: upload, CUDA 130, build arch, torch 213, plus the
    # keep-superseded flag if the keep flag is set or the set is TE and TE is not on schedule.
    steps = WORKFLOW["jobs"]["publish"]["steps"]
    download = next(step["with"] for step in steps if step.get("uses") == "actions/download-artifact@v4")
    for name in sets:
        directory = tmp_path / "wheels"
        if not download["merge-multiple"]:
            directory /= f"wheels-{arch}-{name}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.whl").touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    pip = bin_dir / "pip"
    pip.write_text("#!/bin/sh\nexit 0\n")
    pip.chmod(0o755)
    uploader = tmp_path / "miles-wheels" / "build_wheels.py"
    uploader.parent.mkdir()
    uploader.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['UPLOAD_LOG'], 'a') as f:\n"
        "    f.write(json.dumps({'files': sorted(p.name for p in Path(os.environ['WHEEL_DIR']).glob('*.whl')), "
        "'args': sys.argv[1:]}) + '\\n')\n"
        "sys.exit(1 if os.environ['UPLOAD_FAILS'] == 'true' else 0)\n"
    )
    log = tmp_path / "uploads.jsonl"
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", steps[-1]["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "UPLOAD_LOG": str(log),
            "ARCH": arch,
            "BUILD_ARCH": build_arch,
            "CUDA": "130",
            "TORCH": "213",
            "TE_ON_SCHEDULE": str(te_on_schedule).lower(),
            "KEEP_SUPERSEDED": "--keep-superseded" if keep else "",
            "UPLOAD_FAILS": str(upload_fails).lower(),
        },
        capture_output=True,
        text=True,
    )
    uploads = [json.loads(line) for line in log.read_text().splitlines()]
    if upload_fails:
        assert result.returncode != 0
        assert len(uploads) == 1
        return
    assert result.returncode == 0, result.stderr
    assert len(uploads) == len(sets)
    for name in sets:
        upload = next(upload for upload in uploads if upload["files"] == [f"{name}.whl"])
        expected = ["upload", "--cuda", "130", "--arch", build_arch, "--torch", "213"]
        if keep or (name == "te" and not te_on_schedule):
            expected.append("--keep-superseded")
        assert upload["args"] == expected


def test_wheels_publish_scopes_the_ci_app_token():
    # [FUNCTION DOC] Finds the publish job's app-token step and asserts its client id and private
    # key come from the expected repository variable and secret, the owner and the single
    # repository are the expected ones, and the only permission requested is contents write.
    token = next(
        step["with"]
        for step in WORKFLOW["jobs"]["publish"]["steps"]
        if step.get("uses", "").startswith("actions/create-github-app-token@")
    )
    assert token["client-id"] == "${{ vars.CI_APP_CLIENT_ID }}"
    assert token["private-key"] == "${{ secrets.CI_APP_PRIVATE_KEY }}"
    assert token["owner"] == "radixark"
    assert token["repositories"] == "miles-wheels"
    assert {key: value for key, value in token.items() if key.startswith("permission-")} == {
        "permission-contents": "write"
    }


def test_failure_notifier_covers_all_jobs_without_build_or_publish_credentials():
    # [FUNCTION DOC] Asserts on the notify job: it depends on every other job; its condition runs
    # always, on any failed dependency, only in the main repository, and does not test the event
    # name; it runs on the hosted Ubuntu runner with read-only actions and contents permissions;
    # its checkout does not persist credentials and sparse-checks-out exactly the two notification
    # scripts; the final step calls the Lark notifier for wheel build failures with the webhook
    # secret, the workflow token and the run id; and no build or publish app credentials appear
    # anywhere in the job.
    notify = WORKFLOW["jobs"]["notify-build-failure"]
    assert set(notify["needs"]) == set(WORKFLOW["jobs"]) - {"notify-build-failure"}
    assert "always()" in notify["if"]
    assert "contains(needs.*.result, 'failure')" in notify["if"]
    assert "github.repository == 'radixark/miles'" in notify["if"]
    assert "github.event_name" not in notify["if"]
    assert notify["runs-on"] == "ubuntu-latest"
    assert notify["permissions"] == {"actions": "read", "contents": "read"}
    checkout = notify["steps"][0]["with"]
    assert checkout["persist-credentials"] is False
    assert set(checkout["sparse-checkout"].splitlines()) == {
        ".github/workflows/scripts/lark_notify.py",
        ".github/workflows/scripts/ci_failure_analysis.py",
    }
    post = notify["steps"][-1]
    assert "lark_notify.py wheels-build-failure" in post["run"]
    assert post["env"]["LARK_WEBHOOK"] == "${{ secrets.LARK_WEBHOOK }}"
    assert post["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert post["env"]["RUN_ID"] == "${{ github.run_id }}"
    assert "CI_APP" not in json.dumps(notify)
