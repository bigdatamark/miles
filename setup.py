# [MODULE DOC] Packaging script for the miles distribution (version 0.1.2, Python 3.10 or newer).
# Installs the miles and miles_plugins packages plus dashboard static files, with base
# requirements read from the requirements text file and optional extras for fsdp, mlflow,
# dashboard, e2b and modal. Replaces the wheel builder with a custom subclass so wheels are marked
# as platform-specific (non-pure) and tagged with the interpreter version, a matching ABI tag and
# a platform tag derived from the operating system and CPU architecture. Executing this module
# runs the setup call immediately (no main guard).
import sys
import platform

from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def _get_platform_tag():
    # [FUNCTION DOC] HELPER: Returns the wheel platform tag. Non-Linux systems give the lower-cased
    # system name. On Linux: x86-64 (either spelling) gives the manylinux1 x86-64 tag, aarch64 or
    # arm64 gives the manylinux2014 aarch64 tag, and any other machine gives a generic linux tag
    # with the lower-cased machine name.
    if platform.system() != "Linux":
        return platform.system().lower()

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "manylinux1_x86_64"
    if machine in ("aarch64", "arm64"):
        return "manylinux2014_aarch64"
    return f"linux_{machine}"


def _fetch_requirements(path):
    # [FUNCTION DOC] HELPER: Reads the requirements file at the path (relative to the current
    # working directory, so the build must run from the repo root) and returns the stripped lines
    # that are non-blank and do not start with a hash; inline comments are not removed.
    with open(path) as fd:
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]


# Custom wheel class to modify the wheel name
class bdist_wheel(_bdist_wheel):
    # [CLASS DOC] Wheel builder subclass that forces platform-specific wheel naming: the wheel is
    # marked non-pure and its tag triple is computed locally instead of by the base class.
    def finalize_options(self):
        # [FUNCTION DOC] Runs the base class finalization, then marks the wheel as not pure.
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        # [FUNCTION DOC] Returns the (python tag, ABI tag, platform tag) triple: the python tag is
        # the interpreter prefix with major and minor version digits, the ABI tag is identical to
        # it, and the platform tag comes from the platform helper.
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = f"{python_version}"
        platform_tag = _get_platform_tag()

        return python_version, abi_tag, platform_tag


# Setup configuration
setup(
    author="miles Team",
    name="miles",
    version="0.1.2",
    packages=find_packages(include=["miles*", "miles_plugins*"]),
    include_package_data=True,
    package_data={"miles.dashboard": ["static/*"]},
    install_requires=_fetch_requirements("requirements.txt"),
    extras_require={
        "fsdp": [
            "torch>=2.0",
        ],
        "mlflow": [
            "mlflow>=2.0",
        ],
        # standalone offline serving; the training image already has these via
        # sglang, and polars is a base requirement (used by the collector)
        "dashboard": [
            "fastapi>=0.135",
            "uvicorn>=0.41",
            "prometheus_client>=0.24",
        ],
        # the openenv e2b sandbox backend, imported only when that backend is
        # selected. 2.12 is the first release that sends the template name in
        # the `name` field, which self-hosted AgentENV reads; older ones send
        # the deprecated `alias` and every Template.build fails with 400
        "e2b": [
            "e2b>=2.12",
        ],
        "modal": [
            "modal>=1.5.5",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    cmdclass={"bdist_wheel": bdist_wheel},
)
