# [MODULE DOC] HELPER: Shared test support for the environment report tests: a sample of the JSON
# that the package inspection command of pip prints (four packages: two editable installs, one
# plain, one installed from an archive URL), a factory for a minimal argument namespace, a factory
# for a fake completed pip subprocess, and fixtures that mock the pip probe and install or remove
# a process-wide event logger writing into a temporary directory.
import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from miles.utils.audit_utils.event_logger.logger import EventLogger, set_event_logger
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

SAMPLE_PIP_INSPECT = {
    "version": "1",
    "pip_version": "24.0",
    "installed": [
        {
            "metadata": {"name": "miles", "version": "0.2.1"},
            "direct_url": {
                "url": "file:///workspace/miles",
                "dir_info": {"editable": True},
            },
        },
        {
            "metadata": {"name": "sglang", "version": "0.4.0"},
            "direct_url": {
                "url": "file:///workspace/sglang",
                "dir_info": {"editable": True},
            },
        },
        {
            "metadata": {"name": "torch", "version": "2.5.0"},
        },
        {
            "metadata": {"name": "numpy", "version": "1.26.0"},
            "direct_url": {
                "url": "https://files.pythonhosted.org/numpy-1.26.0.tar.gz",
                "archive_info": {},
            },
        },
    ],
}


def make_args(**overrides) -> argparse.Namespace:
    # [FUNCTION DOC] HELPER: Builds an argument namespace with the environment report option
    # empty (no launcher record) and the report interval 3600.0 seconds, then applies the keyword
    # overrides (which may replace those two or add arbitrary extra arguments). Returns the
    # namespace.
    return argparse.Namespace(**{"env_report": "", "env_report_interval_seconds": 3600.0, **overrides})


def mock_pip_inspect() -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Builds a fake finished subprocess for the pip inspect command with
    # exit code 0, empty error output and standard output equal to the JSON text of the sample
    # inspection data. Returns it.
    return subprocess.CompletedProcess(
        args=["pip", "inspect"], returncode=0, stdout=json.dumps(SAMPLE_PIP_INSPECT), stderr=""
    )


@pytest.fixture()
def mocked_pip_inspect():
    # [FUNCTION DOC] HELPER: Fixture. For the duration of a test, replaces the subprocess run
    # function used by the collector module with one returning the fake pip inspect result, so no
    # real pip is executed. Yields nothing; the patch is undone afterwards.
    with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_pip_inspect()):
        yield


@pytest.fixture()
def event_log_dir(tmp_path: Path) -> Path:
    # [FUNCTION DOC] HELPER: Fixture. Installs a global event logger that writes into the
    # temporary directory, with a process identity whose component is "main", and yields that
    # directory. After the test the global event logger is reset to none.
    set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main")))
    yield tmp_path
    set_event_logger(None)


@pytest.fixture()
def without_event_logger():
    # [FUNCTION DOC] HELPER: Fixture. Clears the global event logger before the test and again
    # afterwards, guaranteeing no logger is active during the test. Yields nothing.
    set_event_logger(None)
    yield
    set_event_logger(None)
