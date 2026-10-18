# [MODULE DOC] Tests for the shared command utility helpers: the Mooncake object store argument
# builder (a remote master host reaches the serialized master address together with the port) and
# the default wandb arguments builder (when the wandb key is missing it must configure the
# launcher logger before logging the skip message, and returns an empty string).
import json
import logging
import shlex

import pytest

from miles.utils.external_utils.command_utils import common
from miles.utils.external_utils.command_utils.common import MOONCAKE_INIT_KWARGS_FLAG, get_mooncake_object_store_args


class TestGetMooncakeObjectStoreArgs:
    # [CLASS DOC] Tests of the Mooncake object store argument string builder.
    def test_a_remote_master_host_reaches_the_serialized_store_address(self) -> None:
        # [FUNCTION DOC] The argument string for port 61234 and a given master host, split as a
        # shell command, has an init kwargs flag whose json value has the master server address
        # equal to the host, a colon and the port.
        """A split deployment connects to the master host supplied by its driving release."""
        argv = shlex.split(get_mooncake_object_store_args(master_port=61234, master_host="mooncake.run.svc"))

        kwargs = json.loads(argv[argv.index(MOONCAKE_INIT_KWARGS_FLAG) + 1])

        assert kwargs["master_server_address"] == "mooncake.run.svc:61234"


class TestGetDefaultWandbArgs:
    # [CLASS DOC] Tests of the default wandb arguments builder.
    def test_missing_credentials_configures_logging_before_reporting_the_skip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] With the wandb API key environment variable removed and the logging
        # utility's raw logger configuration replaced by a recorder, a recording handler is added
        # to the common module's logger at info level (its level is not restored afterwards). The
        # builder called for a test script path returns an empty string, and the recorded events
        # are, in order, the logger configuration for the launcher logger name and then the skip
        # message saying the key was not found. The handler is removed in a cleanup block.
        # KNOWN ISSUE: the test sets the common logger's level to info and never restores it, so
        # the change leaks into later tests in the same process.
        """The launcher must make its skip message visible before it emits that message."""
        events: list[str] = []
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setattr(
            "miles.utils.logging_utils.configure_logger_raw", lambda name: events.append(f"configure:{name}")
        )

        class RecordingHandler(logging.Handler):
            # [CLASS DOC] HELPER: Logging handler that appends each record's formatted message to
            # the test's events list.
            def emit(self, record: logging.LogRecord) -> None:
                # [FUNCTION DOC] HELPER: Appends the record's message to the shared events list.
                events.append(record.getMessage())

        handler = RecordingHandler()
        common.logger.addHandler(handler)
        common.logger.setLevel(logging.INFO)
        try:
            assert common.get_default_wandb_args("tests/e2e/test_run.py") == ""
        finally:
            common.logger.removeHandler(handler)

        assert events == ["configure:launcher", "Skip wandb configuration since WANDB_API_KEY is not found"]
