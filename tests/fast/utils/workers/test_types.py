# [MODULE DOC] Unit tests for parse_hot_restart in the worker types module: it turns the text of
# a hot-restart environment variable (comma-separated component names) into a list of
# HotRestartComponent values. Empty text means no hot restart (empty list); whitespace around names
# is ignored; an unknown component raises a value error; a non-empty value that names no component
# (only commas/spaces) raises an assertion error mentioning "names no component" and listing the
# valid names; single valid components parse (combination rules belong to the planner).
from __future__ import annotations

import pytest

from miles.utils.workers.types import HotRestartComponent, parse_hot_restart


class TestParseHotRestart:
    # [CLASS DOC] Tests of ordinary parsing results.
    def test_an_empty_value_asks_for_no_hot_restart(self):
        # [FUNCTION DOC] The empty string parses to an empty list.
        """Every ordinary launch passes this, and it must not plan a restart of anything."""
        assert parse_hot_restart("") == []

    def test_the_two_components_are_parsed_together(self):
        # [FUNCTION DOC] "orchestration,rollout_executor" parses to the orchestration then rollout
        # executor components, in that order.
        """This is the only accepted value, and it names the pair the feature replaces."""
        assert parse_hot_restart("orchestration,rollout_executor") == [
            HotRestartComponent.ORCHESTRATION,
            HotRestartComponent.ROLLOUT_EXECUTOR,
        ]

    def test_whitespace_around_the_names_is_ignored(self):
        # [FUNCTION DOC] Spaces around and between the names give the same two-component result.
        """The value travels through an env var, where a stray space is a typo rather than an intent."""
        assert parse_hot_restart(" orchestration , rollout_executor ") == [
            HotRestartComponent.ORCHESTRATION,
            HotRestartComponent.ROLLOUT_EXECUTOR,
        ]

    def test_a_component_that_cannot_be_hot_restarted_is_refused(self):
        # [FUNCTION DOC] A list containing the unknown name "trainer" raises a value error.
        """Everything else is taken over by the new script rather than replaced with it."""
        with pytest.raises(ValueError):
            parse_hot_restart("orchestration,trainer")

    @pytest.mark.parametrize("value", ["orchestration", "rollout_executor"])
    def test_either_component_alone_is_parsed_and_left_to_the_planner_to_refuse(self, value: str):
        # [FUNCTION DOC] Parametrized over each valid name alone; the result is a one-element list
        # holding the component with that value.
        """Parsing answers what the flag names; which combinations a hot restart supports is the planner's call."""
        assert parse_hot_restart(value) == [HotRestartComponent(value)]


class TestAHotRestartRequestThatNamesNoComponent:
    # [CLASS DOC] Tests of values that are non-empty but name nothing.
    @pytest.mark.parametrize("value", [",", " ", " , ", ",,"])
    def test_a_non_empty_value_naming_nothing_is_refused(self, value: str):
        # [FUNCTION DOC] Parametrized over a comma, a space, a spaced comma and two commas; each
        # raises an assertion error matching "names no component".
        """It parsed to an empty list, which every caller reads as an ordinary launch of the whole fleet."""
        with pytest.raises(AssertionError, match="names no component"):
            parse_hot_restart(value)

    def test_the_refusal_names_the_components_that_could_have_been_asked_for(self):
        # [FUNCTION DOC] The assertion message for a lone comma mentions "orchestration".
        """The value travels through an env var, where the only fix is knowing what may be written there."""
        with pytest.raises(AssertionError, match="orchestration"):
            parse_hot_restart(",")

    def test_an_empty_value_is_still_the_ordinary_launch(self):
        # [FUNCTION DOC] The empty string still parses to an empty list.
        # KNOWN ISSUE: duplicates test_an_empty_value_asks_for_no_hot_restart.
        """Every run that is not being hot restarted passes this, and it must not be refused."""
        assert parse_hot_restart("") == []

    def test_a_value_that_does_name_a_component_is_still_parsed(self):
        # [FUNCTION DOC] "orchestration," (trailing comma) parses to just the orchestration
        # component.
        """The refusal is for a value that names nothing, not for the requests the feature exists for."""
        assert parse_hot_restart("orchestration,") == [HotRestartComponent.ORCHESTRATION]
