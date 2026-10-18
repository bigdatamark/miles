# [MODULE DOC]
# Documentation consistency tests for the fault-injection paragraph of the developer debug
# guide: the paragraph (found by its bold marker, exactly one) must show json action examples
# that load through the real fault-tolerance test action parser, target cells by cell id (not
# the retired cell index), use cell ids that round-trip through the real naming scheme, mention
# exactly the supported action names, and never mention the retired cell index field.
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest

from miles.utils.test_utils.ft_test_actions import _ACTOR_ACTIONS, _CONTROLLER_ACTIONS, FTTestAction, _load_actions
from miles.utils.workers.naming import compute_cell_id, parse_cell_id

REPO_ROOT = Path(__file__).resolve().parents[3]
DEBUG_DOC = REPO_ROOT / "docs" / "developer" / "debug.md"
_PARAGRAPH_MARKER = "**Fault injection.**"
_SUPPORTED_ACTIONS = set(get_args(FTTestAction.model_fields["action"].annotation))
_ALL_ACTIONS = _CONTROLLER_ACTIONS | _ACTOR_ACTIONS


@pytest.fixture(scope="module")
def fault_injection_paragraph() -> str:
    # [FUNCTION DOC] HELPER: Module-scoped fixture. Reads the debug guide, splits it on blank
    # lines and keeps paragraphs starting with the bold fault-injection marker. Asserts exactly
    # one such paragraph exists (message reports the count) and returns its text.
    paragraphs = [p for p in DEBUG_DOC.read_text().split("\n\n") if p.startswith(_PARAGRAPH_MARKER)]
    assert len(paragraphs) == 1, f"expected exactly one {_PARAGRAPH_MARKER} paragraph, found {len(paragraphs)}"
    return paragraphs[0]


@pytest.fixture(scope="module")
def documented_actions(fault_injection_paragraph: str) -> list[dict[str, object]]:
    # [FUNCTION DOC] HELPER: Module-scoped fixture. Finds every innermost brace-delimited blob (no
    # nested braces) in the paragraph, asserts at least one exists, and returns the blobs parsed
    # as json objects (a parse error fails the test).
    blobs = re.findall(r"\{[^{}]*\}", fault_injection_paragraph)
    assert blobs, "the fault injection paragraph no longer shows a JSON action example"
    return [json.loads(blob) for blob in blobs]


class TestDocumentedFaultInjectionExample:
    # [CLASS DOC] Checks that the fault-injection documentation matches the real action parsing
    # and cell naming code.
    def test_the_documented_example_loads_through_the_real_action_parser(
        self, documented_actions: list[dict[str, object]]
    ) -> None:
        # [FUNCTION DOC] Serialises the documented actions back to a json list, feeds it to the
        # real action loader as the inline actions argument (no file path) with the union of
        # controller and actor action names allowed, and asserts as many actions load as were
        # documented (validation errors would raise).
        """Copying the doc example into --ci-ft-test-actions must produce loadable actions, not a validation error."""
        raw = json.dumps(documented_actions)

        actions = _load_actions(SimpleNamespace(ci_ft_test_actions=raw, ci_ft_test_actions_path=None), _ALL_ACTIONS)

        assert len(actions) == len(documented_actions)

    def test_the_documented_example_targets_a_cell_by_id_not_by_index(
        self, documented_actions: list[dict[str, object]]
    ) -> None:
        # [FUNCTION DOC] Every documented action has a cell id field and no cell index field.
        """The flag takes cell_id, so an example still keyed by the retired cell_index would fail on load."""
        for documented_action in documented_actions:
            assert "cell_id" in documented_action
            assert "cell_index" not in documented_action

    def test_the_documented_cell_id_round_trips_through_the_real_id_scheme(
        self, documented_actions: list[dict[str, object]]
    ) -> None:
        # [FUNCTION DOC] For each documented cell id (must be text): parse it with the real
        # parser, recompute the id from the parsed pool id and cell index and assert it equals the
        # original, and assert the parsed cell index is non-negative.
        """A cell_id in the doc must be spelled the way the workers actually name cells."""
        for documented_action in documented_actions:
            cell_id = documented_action["cell_id"]
            assert isinstance(cell_id, str)
            parsed = parse_cell_id(cell_id)
            assert compute_cell_id(pool_id=parsed.pool_id, cell_index=parsed.cell_index) == cell_id
            assert parsed.cell_index >= 0

    def test_the_paragraph_documents_exactly_the_supported_actions(self, fault_injection_paragraph: str) -> None:
        # [FUNCTION DOC] Collects inline code spans consisting only of lowercase letters and
        # underscores from the paragraph, removes the names of the action model's fields, and
        # asserts the remainder equals the set of supported action names (the allowed values of
        # the model's action field).
        """Adding or renaming an action without touching this paragraph leaves the doc lying about the flag."""
        code_spans = re.findall(r"`([^`]+)`", fault_injection_paragraph)
        identifiers = {span for span in code_spans if re.fullmatch(r"[a-z][a-z_]*", span)}

        assert identifiers - set(FTTestAction.model_fields) == _SUPPORTED_ACTIONS

    def test_the_paragraph_never_mentions_the_retired_cell_index_field(self, fault_injection_paragraph: str) -> None:
        # [FUNCTION DOC] Asserts the paragraph text does not contain the retired cell index name.
        """Prose promising a cell_index sentinel sends developers to a command that dies while loading actions."""
        assert "cell_index" not in fault_injection_paragraph
