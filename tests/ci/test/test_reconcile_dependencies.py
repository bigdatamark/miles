# [MODULE DOC]
# Tests of the CI dependency reconcile script. The main test builds fake wheel files in a
# temporary directory, creates a fresh virtual environment and installs from that local wheel
# directory only (package index disabled for pip and uv via environment variables), then runs
# the script as a subprocess. It checks that the image's runtime packages (the torch stand-in and
# an optional cudnn variant: cu12, cu13 or none) are preserved while the requested dependencies
# install; that a requirements change to a dependency-only package succeeds; that requesting a
# different torch version, or reinstalling the runtime via a pip or uv plan, is blocked with the
# message "CI dependency gate blocks reinstalling: <name>"; that an unlisted package of 100 MiB
# or more is automatically protected; and that an unresolvable requirement fails. Further tests
# cover the protected-package size threshold (100 MiB inclusive), refusal of unrecognized or
# truncated uv plan output, and that the GPU workflow's reconcile step runs the gated script on
# two requirements files. Registers itself as a CPU test in stage-a-cpu with a 20 second estimate.
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.ci import reconcile_dependencies
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

SCRIPT = Path(__file__).resolve().parents[1] / "reconcile_dependencies.py"


def wheel(directory, name, version, requires=(), payload_bytes=0):
    # [FUNCTION DOC] HELPER: Write a minimal pure-Python wheel archive for the given package name
    # and version into the directory. The file name uses the name with dashes turned to
    # underscores. The archive holds a package init file defining the version, a metadata file
    # (name, version and one requirement line per entry in requires), a wheel file, optionally a
    # zero-filled weights file of payload_bytes bytes (to make the package "large"), and a record
    # file listing every member with its size (the record itself listed without size).
    normalized = name.replace("-", "_")
    info = f"{normalized}-{version}.dist-info"
    with zipfile.ZipFile(
        directory / f"{normalized}-{version}-py3-none-any.whl", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(f"{normalized}/__init__.py", f'__version__ = "{version}"\n')
        archive.writestr(
            f"{info}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires),
        )
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        if payload_bytes:
            archive.writestr(f"{normalized}/weights.bin", bytes(payload_bytes))
        archive.writestr(
            f"{info}/RECORD",
            "".join(f"{item.filename},,{item.file_size}\n" for item in archive.infolist()) + f"{info}/RECORD,,\n",
        )


@pytest.mark.parametrize("cudnn", ["nvidia-cudnn-cu12", "nvidia-cudnn-cu13", None])
def test_real_resolver_preserves_image_runtime_and_installs_dependencies(tmp_path, cudnn):
    # [FUNCTION DOC] Parametrized over three cudnn flavors (cu12, cu13, none). Steps: (1) build
    # fake wheels: a leaf package, a client requiring the torch stand-in and the leaf, torch
    # (requiring a pinned cudnn 9.20 if any), and two cudnn versions 9.20 and 9.22. (2) Create a
    # virtual environment (subprocess) with index access disabled and local wheel lookup; seed it
    # with torch and the newer cudnn 9.22 without dependencies. (3) With cudnn, check a plain pip
    # dry run would have picked cudnn 9.20 (proving the gate is needed). (4) Run the reconcile
    # script on a requirements file naming the client; assert the client and leaf are installed at
    # 1.0, cudnn stays at 9.22.0.52 (or no cudnn package exists in the none case). (5) A new leaf
    # 2.0 requirement reconciles successfully. (6) A requirement of torch 2.0 fails with the
    # blocked-reinstall message on standard error and torch remains 1.0. (7) The plan checker
    # function, loaded from the script by file path and run in the environment, blocks a forced
    # no-deps reinstall of the runtime package (via uv when a cudnn exists, else pip) with the same
    # message. (8) In the no-cudnn case, an unlisted package with a 100 MiB payload is blocked from
    # upgrade. (9) A requirement on a missing package makes the script exit nonzero.
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel(wheels, "ci-leaf", "1.0")
    wheel(wheels, "ci-client", "1.0", ["torch==1.0", "ci-leaf==1.0"])
    wheel(wheels, "torch", "1.0", [f"{cudnn}==9.20.0.48"] if cudnn else [])
    if cudnn:
        wheel(wheels, cudnn, "9.20.0.48")
        wheel(wheels, cudnn, "9.22.0.52")

    environment = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = str(environment / "bin" / "python")
    env = {
        **os.environ,
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": str(wheels),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "UV_NO_INDEX": "1",
        "UV_FIND_LINKS": str(wheels),
    }
    seeded = ["torch==1.0"] + ([f"{cudnn}==9.22.0.52"] if cudnn else [])
    subprocess.run([python, "-m", "pip", "install", "--no-deps", *seeded], env=env, check=True)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("ci-client==1.0\n")

    if cudnn:
        baseline = subprocess.run(
            [python, "-m", "pip", "install", "--dry-run", "-r", str(requirements)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert f"{cudnn}-9.20.0.48" in baseline.stdout

    subprocess.run([python, str(SCRIPT), str(requirements)], env=env, check=True)
    installed = json.loads(subprocess.check_output([python, "-m", "pip", "list", "--format=json"], env=env, text=True))
    versions = {package["name"]: package["version"] for package in installed}
    assert versions["ci-client"] == versions["ci-leaf"] == "1.0"
    if cudnn:
        assert versions[cudnn] == "9.22.0.52"
    else:
        assert not any(name.startswith("nvidia-cudnn") for name in versions)

    wheel(wheels, "ci-leaf", "2.0")
    requirements.write_text("ci-leaf==2.0\n")
    subprocess.run([python, str(SCRIPT), str(requirements)], env=env, check=True)

    wheel(wheels, "torch", "2.0")
    requirements.write_text("torch==2.0\n")
    blocked = subprocess.run([python, str(SCRIPT), str(requirements)], env=env, capture_output=True, text=True)
    assert blocked.returncode != 0
    assert "CI dependency gate blocks reinstalling: torch" in blocked.stderr
    assert (
        subprocess.check_output(
            [python, "-c", "import importlib.metadata as m; print(m.version('torch'))"], text=True
        ).strip()
        == "1.0"
    )

    runtime = cudnn or "torch"
    version = "9.22.0.52" if cudnn else "1.0"
    command = (
        ["uv", "pip", "install", "--python", python, "--reinstall"]
        if cudnn
        else [python, "-m", "pip", "install", "--force-reinstall"]
    )
    command += ["--no-deps", f"{runtime}=={version}"]
    code = (
        f"import runpy; from pathlib import Path; m=runpy.run_path({str(SCRIPT)!r}); "
        f"m['check_install_plan']({command!r}, Path({str(tmp_path)!r}), use_uv={bool(cudnn)!r})"
    )
    blocked = subprocess.run([python, "-c", code], env=env, capture_output=True, text=True)
    assert blocked.returncode != 0
    assert f"CI dependency gate blocks reinstalling: {runtime}" in blocked.stderr

    if not cudnn:
        wheel(wheels, "ci-unlisted-large", "1.0", payload_bytes=100 * 1024**2)
        wheel(wheels, "ci-unlisted-large", "2.0")
        subprocess.run([python, "-m", "pip", "install", "ci-unlisted-large==1.0"], env=env, check=True)
        requirements.write_text("ci-unlisted-large==2.0\n")
        blocked = subprocess.run([python, str(SCRIPT), str(requirements)], env=env, capture_output=True, text=True)
        assert blocked.returncode != 0
        assert "CI dependency gate blocks reinstalling: ci-unlisted-large" in blocked.stderr

    requirements.write_text("ci-missing-dependency==1.0\n")
    failed = subprocess.run([python, str(SCRIPT), str(requirements)], env=env)
    assert failed.returncode != 0


@pytest.mark.parametrize("size, protected", [(100 * 1024**2 - 1, False), (100 * 1024**2, True)])
def test_large_packages_are_protected_without_a_name_allowlist(monkeypatch, size, protected):
    # [FUNCTION DOC] Parametrized over a package one byte under 100 MiB (not protected) and exactly
    # 100 MiB (protected). Replaces the installed-distribution listing with one fake distribution
    # named with mixed case, dots and underscores holding a single file of that size, and asserts
    # the normalized lowercase dash name is in the protected set exactly when expected.
    distribution = SimpleNamespace(metadata={"Name": "Unknown_Large.Package"}, files=[SimpleNamespace(size=size)])
    monkeypatch.setattr(reconcile_dependencies.importlib.metadata, "distributions", lambda: [distribution])
    assert ("unknown-large-package" in reconcile_dependencies.protected_packages()) == protected


@pytest.mark.parametrize(
    "output",
    ["", "Resolved 1 package", "Would install 2 packages\n + torch==2.0\n", "Would uninstall 1 package\n"],
)
def test_unrecognized_or_truncated_plans_cannot_pass_the_gate(output):
    # [FUNCTION DOC] Parametrized over four uv plan outputs (empty, a resolution line only, an
    # install summary with a listed package but no recognized change format, an uninstall-only
    # summary). Each makes the plan parser raise a runtime error containing "refusing to install".
    with pytest.raises(RuntimeError, match="refusing to install"):
        reconcile_dependencies.uv_changes(output)


def test_gpu_workflow_uses_the_gated_installer():
    # [FUNCTION DOC] Parses the shared CI run workflow definition (three levels above this file),
    # finds the step named "Reconcile Miles dependencies" in the run job and asserts its command
    # words are exactly: python, the reconcile script path, a line-continuation backslash, the
    # multi-LoRA example requirements file, and the root requirements file.
    import yaml

    workflow = Path(__file__).resolve().parents[3] / ".github/workflows/_run-ci.yml"
    steps = yaml.safe_load(workflow.read_text())["jobs"]["run"]["steps"]
    reconcile = next(step for step in steps if step.get("name") == "Reconcile Miles dependencies")
    assert reconcile["run"].split() == [
        "python",
        "tests/ci/reconcile_dependencies.py",
        "\\",
        "examples/multi_lora/requirements.txt",
        "requirements.txt",
    ]
