# [MODULE DOC] Contract test for the audit event logger module
# (miles.utils.audit_utils.event_logger.logger): pins the name of the on-disk events directory.
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME


class TestEventsDirectoryName:
    # [CLASS DOC] Holds the single check on the exported events directory name constant.
    def test_events_directory_name_matches_the_on_disk_contract(self) -> None:
        # [FUNCTION DOC] Asserts the exported directory name constant equals exactly "events", so
        # previously stored audit data stays discoverable.
        """The exported events directory name remains compatible with stored audit data."""
        assert EVENTS_DIRNAME == "events"
