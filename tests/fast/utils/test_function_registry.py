# [MODULE DOC]
# Tests for the function registry and the dotted-path function loader. Covers: registering and
# retrieving functions through a scoped temporary registration, duplicate-name rejection, removal
# and exception-safe cleanup, the loader's module import path, none passthrough, registry
# precedence over module paths, rejection of non-callables when synchronous use is required, an
# import-weight guard (the registry module must not pull in ray or the misc utilities module), and
# a source scan asserting nothing in the package or examples imports the loader from the misc
# module.
import os
import re
from pathlib import Path

import pytest
from tests.fast.import_isolation_utils import modules_imported_by

import miles
from miles.utils.function_registry import FunctionRegistry, function_registry, load_function


def _fn_a():
    # [FUNCTION DOC] HELPER: Sample function returning the text "a"; used as a registry value.
    return "a"


def _fn_b():
    # [FUNCTION DOC] HELPER: Sample function returning the text "b"; used as a second registry value.
    return "b"


class TestFunctionRegistry:
    # [CLASS DOC] Tests of the registry class using fresh instances.
    def test_register_and_get(self):
        # [FUNCTION DOC] Inside a temporary registration, fetching the name returns the function.
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            assert registry.get("my_fn") is _fn_a

    def test_register_duplicate_raises(self):
        # [FUNCTION DOC] Registering a second function under an already registered name raises an
        # assertion error.
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            with pytest.raises(AssertionError):
                with registry.temporary("my_fn", _fn_b):
                    pass

    def test_unregister(self):
        # [FUNCTION DOC] After leaving the temporary registration, fetching the name returns none.
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            assert registry.get("my_fn") is _fn_a
        assert registry.get("my_fn") is None

    def test_temporary_cleanup_on_exception(self):
        # [FUNCTION DOC] If the body of a temporary registration raises, the error propagates and
        # the name is still unregistered afterwards.
        registry = FunctionRegistry()
        with pytest.raises(RuntimeError):
            with registry.temporary("temp_fn", _fn_a):
                raise RuntimeError("test")
        assert registry.get("temp_fn") is None


class TestLoadFunction:
    # [CLASS DOC] Tests of the loader, which resolves a name in the global registry first and falls
    # back to importing a dotted module path.
    def test_load_from_module(self):
        # [FUNCTION DOC] Loading the dotted path of the path-join function returns that function.
        import os.path

        assert load_function("os.path.join") is os.path.join

    def test_load_none_returns_none(self):
        # [FUNCTION DOC] Loading none returns none.
        assert load_function(None) is None

    def test_load_from_registry(self):
        # [FUNCTION DOC] A name temporarily registered in the global registry is returned by the
        # loader even though it is not an importable path.
        with function_registry.temporary("test:my_fn", _fn_a):
            assert load_function("test:my_fn") is _fn_a

    def test_registry_takes_precedence(self):
        # [FUNCTION DOC] Registering a function under an importable dotted path makes the loader
        # return the registered function; after the registration ends the real function returns.
        with function_registry.temporary("os.path.join", _fn_b):
            assert load_function("os.path.join") is _fn_b
        assert load_function("os.path.join") is os.path.join

    def test_sync_required_rejects_a_non_callable_registry_entry(self):
        # [FUNCTION DOC] Register a list (non-callable) and load it with the synchronous-required
        # flag; expect a value error whose message says it did not resolve to a callable.
        """A path that resolves to a non-callable must fail loudly before a synchronous caller invokes it."""
        with function_registry.temporary("test:not_callable", [1, 2, 3]):
            with pytest.raises(ValueError, match="did not resolve to a callable"):
                load_function("test:not_callable", sync_required=True)


class TestFunctionRegistryImportWeight:
    # [CLASS DOC] Guard tests on the import footprint of the registry module.
    def test_importing_function_registry_does_not_import_ray(self):
        # [FUNCTION DOC] Using the import-isolation helper, collect the modules imported by the
        # registry module and assert neither ray nor the misc utilities module is among them.
        """load_function lives here rather than in miles.utils.misc so importers avoid ray."""
        modules = modules_imported_by("miles.utils.function_registry")

        assert "ray" not in modules
        assert "miles.utils.misc" not in modules


class TestLoadFunctionImportSites:
    # [CLASS DOC] Source-scan tests of how callers import the loader.
    def test_no_module_takes_load_function_from_misc(self):
        # [FUNCTION DOC] Recursively read every Python source under the package directory and the
        # sibling examples directory (reading files from disk), skipping files named misc.py, and
        # collect those matching a pattern that imports the loader from the misc module, either by
        # a from-import line or by a fully qualified reference. Assert the sorted list of offending
        # relative paths is empty.
        """Callers must import load_function from function_registry so they do not drag ray in through misc."""
        pattern = re.compile(
            r"from miles\.utils\.misc import [^\n]*\bload_function\b|miles\.utils\.misc\.load_function"
        )
        package_root = Path(miles.__file__).parent
        examples_root = package_root.parent / "examples"

        offenders = sorted(
            str(path.relative_to(root))
            for root in (package_root, examples_root)
            for path in root.rglob("*.py")
            if path.name != "misc.py" and pattern.search(path.read_text())
        )

        assert offenders == []
