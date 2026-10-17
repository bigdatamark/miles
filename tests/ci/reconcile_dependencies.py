# [MODULE DOC] Script that installs requirement files in CI without disturbing the container
# image's heavy runtime. It pins any installed cuDNN packages (nvidia-cudnn for CUDA 12 and 13)
# to their current versions through a uv override file (falling back to plain pip when no cuDNN
# package is installed), and before each install runs a dry run whose plan is parsed; if the plan
# would touch a protected package (torch, triton, apex, kernel packages, packages with known
# prefixes, or any installed distribution of 100 MiB or more) the script raises an error telling to
# rebuild the image instead. Command line: one or more requirement file paths.
"""Install CI requirements while retaining the image's cuDNN runtime."""

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

LARGE_PACKAGE_BYTES = 100 * 1024 * 1024
RUNTIME_PACKAGES = {"torch", "triton", "apex", "sglang-kernel"}
RUNTIME_PREFIXES = ("nvidia-", "flash-attn", "flashinfer-", "transformer-engine")


def package_name(name: str) -> str:
    # [FUNCTION DOC] HELPER: Normalizes a package name: runs of dash, underscore or dot become one
    # dash, lower-cased.
    return re.sub(r"[-_.]+", "-", name).lower()


def protected_packages() -> set[str]:
    # [FUNCTION DOC] Scans installed distributions and returns the normalized names that are
    # protected: in the runtime set, starting with a runtime prefix, or whose installed files total
    # at least 100 MiB (files with unknown size count as zero; distributions without a file list
    # count as empty). Reads the environment's package metadata.
    protected = set()
    for distribution in importlib.metadata.distributions():
        name = package_name(distribution.metadata["Name"])
        size = sum(file.size or 0 for file in distribution.files or [])
        if name in RUNTIME_PACKAGES or name.startswith(RUNTIME_PREFIXES) or size >= LARGE_PACKAGE_BYTES:
            protected.add(name)
    return protected


def uv_changes(output: str) -> set[str]:
    # [FUNCTION DOC] Parses the text of a uv dry-run plan and returns the normalized names of all
    # added and removed packages. Raises a runtime error (refusing to install) when no count lines
    # are found and the text is not an explicit no-changes plan, or when a count line appears twice
    # or the number of plus or minus lines does not match the announced install and uninstall
    # counts. A no-changes plan with no change lines returns an empty set.
    changes = re.findall(r"^ ([+-]) ([\w.-]+)(?:==| @ ).+$", output, re.MULTILINE)
    counts = re.findall(r"^Would (uninstall|install) (\d+) packages?$", output, re.MULTILINE)
    if not counts:
        if not changes and "Would make no changes" in output:
            return set()
        raise RuntimeError("Unrecognized uv install plan; refusing to install")
    expected = dict(counts)
    if len(expected) != len(counts) or any(
        sum(sign == symbol for sign, _ in changes) != int(expected.get(action, 0))
        for action, symbol in (("install", "+"), ("uninstall", "-"))
    ):
        raise RuntimeError("Incomplete uv install plan; refusing to install")
    return {package_name(name) for _, name in changes}


def check_install_plan(command: list[str], directory: Path, *, use_uv: bool) -> None:
    # [FUNCTION DOC] Dry-runs the install command and raises a runtime error if the plan would
    # change any protected package. With uv: runs the command plus dry-run and no-color flags,
    # captures and echoes the output, raises on nonzero exit, and parses the plan text. With pip:
    # runs with dry-run and a JSON report in the given directory (raises on failure), requires the
    # report version to be "1", and takes the names of packages to install. The block message lists
    # the sorted intersection and says to rebuild the image. Note that uninstalls reported by uv
    # also count as changes, while pip's report only lists installs.
    protected = protected_packages()
    if use_uv:
        result = subprocess.run([*command, "--dry-run", "--color", "never"], capture_output=True, text=True)
        print(result.stdout + result.stderr, end="", flush=True)
        result.check_returncode()
        changes = uv_changes(result.stdout + result.stderr)
    else:
        report = directory / "pip-plan.json"
        subprocess.run([*command, "--dry-run", "--report", str(report)], check=True)
        plan = json.loads(report.read_text())
        if plan["version"] != "1":
            raise RuntimeError("Unrecognized pip install plan; refusing to install")
        changes = {package_name(item["metadata"]["name"]) for item in plan["install"]}

    # Do not remove or bypass this gate: CI must never reinstall image runtimes or large packages.
    # Update these packages in the Docker image instead, even for same-version reinstalls.
    if blocked := sorted(protected & changes):
        raise RuntimeError(f"CI dependency gate blocks reinstalling: {', '.join(blocked)}. Rebuild the image instead.")


def reconcile(requirements: list[str]) -> None:
    # [FUNCTION DOC] Installs each requirements file in order. First collects pins for the installed
    # cuDNN packages (CUDA 12 and 13 variants; absent ones skipped). In a temporary directory: if
    # any pins exist, writes them to an overrides file and uses uv with the current interpreter and
    # that overrides file (printing the pins); otherwise uses the interpreter's pip module. For each
    # file builds the install command allowing system packages, runs the plan check (uv mode iff
    # pins exist), then runs the install (raising on failure).
    pins = []
    for package in ("nvidia-cudnn-cu12", "nvidia-cudnn-cu13"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        pins.append(f"{package}=={version}")

    with tempfile.TemporaryDirectory(prefix="miles-ci-dependencies-") as directory:
        if pins:
            # TE needs the image's cuDNN even when torch declares an older exact pin.
            # An override replaces that pin; a constraint would only make it conflict.
            overrides = Path(directory) / "overrides.txt"
            overrides.write_text("\n".join(pins) + "\n")
            command = ["uv", "pip", "install", "--python", sys.executable, "--overrides", str(overrides)]
            print(f"Preserving image cuDNN: {', '.join(pins)}", flush=True)
        else:
            command = [sys.executable, "-m", "pip", "install"]

        for path in requirements:
            install = [*command, "--break-system-packages", "-r", path]
            check_install_plan(install, Path(directory), use_uv=bool(pins))
            subprocess.run(install, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", nargs="+")
    reconcile(parser.parse_args().requirements)
