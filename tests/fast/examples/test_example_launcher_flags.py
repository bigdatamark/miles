# [MODULE DOC]
# Static consistency check between example launcher scripts and the miles argument parser. It scans
# every python file under the examples directory (found three levels above this file's directory
# tree root) for string constants containing flags with miles-owned prefixes (sglang, miles, session
# families) and verifies each is registered by the miles extra-args provider. Constants: the owned
# prefixes tuple and the regex that finds double-dash flags not preceded by a word character or dash.
import argparse
import ast
import re
from pathlib import Path

from miles.utils.arguments import get_miles_extra_args_provider

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples"

MILES_OWNED_FLAG_PREFIXES: tuple[str, ...] = (
    "--sglang-",
    "--miles-",
    "--use-miles-",
    "--session-",
    "--use-session-",
    "--num-session-",
)

_FLAG_PATTERN = re.compile(r"(?<![\w-])--[a-z0-9][a-z0-9-]*")


def test_examples_only_emit_registered_miles_flags() -> None:
    # [FUNCTION DOC] Collects the registered option strings, then for each miles-owned flag found in the
    # examples groups unregistered flags with the repo-relative files that emit them. Asserts that group
    # is empty, printing it on failure.
    """Every miles-owned --flag an examples/ launcher emits must be registered by the miles args provider."""
    registered: set[str] = _registered_option_strings()
    unregistered: dict[str, list[str]] = {}

    for flag, source_path in _iter_miles_owned_flags():
        if flag not in registered:
            unregistered.setdefault(flag, []).append(str(source_path.relative_to(REPO_ROOT)))

    assert not unregistered, f"examples/ emit miles-owned flags that miles does not register: {unregistered}"


def test_launcher_flag_scan_is_not_vacuous() -> None:
    # [FUNCTION DOC] Sanity check of the scan: asserts it finds more than twenty miles-owned flag
    # occurrences, so a broken scan cannot pass the check silently.
    """The scan must actually find miles-owned flags, so a broken scan cannot silently pass the check."""
    found: list[str] = [flag for flag, _ in _iter_miles_owned_flags()]

    assert len(found) > 20, f"scan of {EXAMPLES_ROOT} found suspiciously few miles-owned flags: {sorted(set(found))}"


def _registered_option_strings() -> set[str]:
    # [FUNCTION DOC] HELPER: Builds an argument parser with abbreviation matching off, applies the miles
    # extra-args provider to it, and returns the set of all option strings of its actions. Does not parse
    # any command line.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    get_miles_extra_args_provider()(parser)
    return {option for action in parser._actions for option in action.option_strings}


def _iter_miles_owned_flags() -> list[tuple[str, Path]]:
    # [FUNCTION DOC] HELPER: Despite its name it returns a list, not an iterator. Walks all python files
    # under the examples directory in sorted order, parses each into a syntax tree (a syntax error
    # propagates), and for every string constant applies the flag regex. Matches starting with an owned
    # prefix are collected, in order and with duplicates, as (flag, source path) pairs.
    results: list[tuple[str, Path]] = []
    for source_path in sorted(EXAMPLES_ROOT.rglob("*.py")):
        tree = ast.parse(source=source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            for flag in _FLAG_PATTERN.findall(node.value):
                if flag.startswith(MILES_OWNED_FLAG_PREFIXES):
                    results.append((flag, source_path))
    return results
