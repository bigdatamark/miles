# [MODULE DOC] Static tests that every training Docker image installs the Kubernetes clients
# (kubectl and helm) through the shared install script rather than hand-copied download blocks:
# each Dockerfile used by a build variant mentions the shared step and the direct download hosts
# do not appear; the shared script pins both client versions with exact version strings and
# verifies two downloads with a checksum check; and exactly three distinct Dockerfiles back the
# build variants. The build script is loaded from its file path (executing it with its own
# directory temporarily on the import path). Constants: repository root, build script and install
# script paths, the default Dockerfile path and the install step name.
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "docker" / "build.py"
INSTALL_SCRIPT = REPO_ROOT / "docker" / "install-kube-tools.sh"
DEFAULT_DOCKERFILE = "docker/Dockerfile"
INSTALL_STEP = "install-kube-tools.sh"


def build_module():
    # [FUNCTION DOC] HELPER: Loads the Docker build script as a module named miles_docker_build
    # from its file path and executes it, with the script's directory inserted at the front of the
    # import path during execution and removed afterwards (also on error). Side effect: runs the
    # script's top-level code. Returns the module.
    spec = importlib.util.spec_from_file_location("miles_docker_build", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)

    script_dir = str(BUILD_SCRIPT.parent)
    sys.path.insert(0, script_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(script_dir)

    return module


def training_dockerfiles() -> list[Path]:
    # [FUNCTION DOC] HELPER: Loads the build module and returns the sorted absolute paths of the
    # distinct Dockerfiles used by its variants (default Dockerfile when a variant names none).
    paths = {config.get("dockerfile", DEFAULT_DOCKERFILE) for config in build_module().VARIANTS.values()}
    return sorted(REPO_ROOT / path for path in paths)


class TestClusterToolsInEveryTrainingImage:
    # [CLASS DOC] Tests of the Kubernetes client installation in training images; parametrization
    # data is computed at collection time by loading the build script.
    @pytest.mark.parametrize("dockerfile", training_dockerfiles(), ids=lambda path: path.name)
    def test_the_image_installs_the_kubernetes_clients(self, dockerfile):
        # [FUNCTION DOC] Parametrized over the training Dockerfiles: the file text mentions the
        # shared install step name.
        """The workbench installs releases with them and the launcher drives the run with them."""
        assert INSTALL_STEP in dockerfile.read_text(), f"{dockerfile.name} builds an image without kubectl and helm"

    @pytest.mark.parametrize("dockerfile", training_dockerfiles(), ids=lambda path: path.name)
    def test_the_image_takes_the_shared_step_rather_than_its_own_copy(self, dockerfile):
        # [FUNCTION DOC] Parametrized over the training Dockerfiles: the text contains neither the
        # Kubernetes download host nor the Helm download host.
        """Two hand-copied install blocks drift, and the one that drifts is the one nobody runs locally."""
        text = dockerfile.read_text()

        assert "dl.k8s.io" not in text
        assert "get.helm.sh" not in text

    def test_the_shared_step_pins_both_clients_and_checks_what_it_downloaded(self):
        # [FUNCTION DOC] Reads the shared install script and asserts it has a line setting the
        # kubectl version to a quoted three-part v-prefixed version, likewise for helm, and that
        # the checksum verification command appears exactly twice.
        """An unpinned or unverified client is a silent cluster-wide behaviour change on the next rebuild."""
        script = INSTALL_SCRIPT.read_text()

        assert re.search(r'^KUBECTL_VERSION="v\d+\.\d+\.\d+"$', script, re.MULTILINE)
        assert re.search(r'^HELM_VERSION="v\d+\.\d+\.\d+"$', script, re.MULTILINE)
        assert script.count("sha256sum -c -") == 2

    def test_every_variant_of_the_build_script_is_covered(self):
        # [FUNCTION DOC] Asserts there are exactly three distinct training Dockerfiles across the
        # build variants, so adding a variant with a new Dockerfile forces updating this test.
        """A new variant pointing at a third Dockerfile must not quietly opt out of this check."""
        assert len(training_dockerfiles()) == 3
