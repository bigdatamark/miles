# [MODULE DOC]
# Pytest configuration for the chart tests: one automatic session-wide fixture that vendors the
# Helm chart dependencies before any chart test renders a chart.
import shutil
import subprocess

import pytest

from tests.fast.charts.utils import chart_directories


@pytest.fixture(scope="session", autouse=True)
def vendored_dependencies():
    # [FUNCTION DOC] HELPER: Session-scoped automatic fixture. Does nothing if the helm executable
    # is not on the path. Otherwise, for every application chart directory that has a dependency
    # lock file, runs a helm dependency build (output captured, not shown) so library charts are
    # vendored into the chart directory. Side effects: spawns subprocesses and writes into the
    # chart directories (vendored chart archives). Raises a called-process error if a build fails.
    if shutil.which("helm") is None:
        return
    for chart_dir in chart_directories():
        if (chart_dir / "Chart.lock").exists():
            subprocess.run(["helm", "dependency", "build", str(chart_dir)], capture_output=True, check=True)
