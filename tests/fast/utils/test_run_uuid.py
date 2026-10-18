# [MODULE DOC] Unit tests for the run uuid helpers: generation and validation.
# WELL_FORMED is a module constant: a lowercase hex string built by repeating a 16 character hex
# pattern and truncating it to the pinned run uuid length, used as the known-good sample.
import pytest

from miles.utils.run_uuid import RUN_UUID_LENGTH, generate_run_uuid, validate_run_uuid

WELL_FORMED = ("ab12cd34ef5678ab" * 4)[:RUN_UUID_LENGTH]


class TestGenerateRunUuid:
    # [CLASS DOC] Tests of the generator: round trip through the validator, exact length,
    # uniqueness across many calls, and a minimum width.
    def test_generated_uuid_is_accepted_by_the_validator(self):
        # [FUNCTION DOC] Generates 100 uuids and asserts each is accepted (truthy result) by the
        # validator.
        """What we generate must be what we accept, or an auto-generated launch fails validation."""
        for _ in range(100):
            assert validate_run_uuid(generate_run_uuid())

    def test_generated_uuid_is_exactly_the_pinned_length(self):
        # [FUNCTION DOC] Asserts one generated uuid has exactly the pinned length constant.
        """A longer or shorter uuid silently changes every string that embeds it."""
        assert len(generate_run_uuid()) == RUN_UUID_LENGTH

    def test_two_launches_do_not_share_a_uuid(self):
        # [FUNCTION DOC] Generates 100 uuids into a set and asserts all 100 are distinct.
        """The whole point is that two runs never collide, unlike a human-readable run name."""
        assert len({generate_run_uuid() for _ in range(100)}) == 100

    def test_the_uuid_is_wide_enough_that_collisions_stay_negligible(self):
        # [FUNCTION DOC] Asserts the pinned length constant is at least 16 characters.
        """A collision makes another run's artifacts look native, which is exactly what this defeats."""
        assert RUN_UUID_LENGTH >= 16


class TestValidateRunUuid:
    # [CLASS DOC] Tests of the validator: pass-through of a good value and rejection of
    # malformed strings.
    def test_accepts_a_well_formed_uuid_and_returns_it(self):
        # [FUNCTION DOC] Asserts validating the well-formed sample returns that same string.
        """The validator is used inline in an assignment, so it must pass the value through."""
        assert validate_run_uuid(WELL_FORMED) == WELL_FORMED

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            WELL_FORMED[:-1],
            WELL_FORMED + "0",
            WELL_FORMED.upper(),
            WELL_FORMED[:-1] + "g",
            " " + WELL_FORMED,
            WELL_FORMED + " ",
            WELL_FORMED + "\n",
            "my-experiment",
        ],
    )
    def test_rejects_anything_that_is_not_exactly_the_pinned_lowercase_hex_shape(self, bad):
        # [FUNCTION DOC] Parametrized over bad inputs: empty string, one character too short or
        # too long, uppercase, a non-hex letter, leading or trailing space, trailing newline, and
        # a human-readable name. Asserts the validator raises a value error whose message matches
        # the pattern for an invalid run uuid.
        """A user-supplied uuid is rejected at startup rather than corrupting strings later."""
        with pytest.raises(ValueError, match="invalid run uuid"):
            validate_run_uuid(bad)
