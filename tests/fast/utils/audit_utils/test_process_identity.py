# [MODULE DOC] Tests for process identity models (miles.utils.audit_utils.process_identity):
# round-tripping a simple identity through the tagged union type used on the wire.
from pydantic import TypeAdapter

from miles.utils.audit_utils.process_identity import ProcessIdentity, SimpleProcessIdentity


class TestProcessIdentityUnion:
    # [CLASS DOC] Tests of the process identity union type.
    def test_registration_reporter_survives_a_union_json_roundtrip(self) -> None:
        # [FUNCTION DOC] Builds a simple identity for component "registration_reporter", dumps it
        # to JSON, parses it through a type adapter for the union, and asserts the parsed value
        # equals the source, is a simple identity instance, and its name string is
        # "registration_reporter".
        """A registration reporter identity keeps its component and process name through the wire union."""
        adapter = TypeAdapter(ProcessIdentity)
        source = SimpleProcessIdentity(component="registration_reporter")

        parsed = adapter.validate_json(source.model_dump_json())

        assert parsed == source
        assert isinstance(parsed, SimpleProcessIdentity)
        assert parsed.to_name() == "registration_reporter"
