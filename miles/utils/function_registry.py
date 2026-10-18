# [MODULE DOC]
# Dotted-path function loading with an in-memory override registry. A global registry lets tests
# register in-flight generated functions under a path name for the duration of a with-block; the
# loader consults it first and otherwise imports the module and fetches the attribute.
import importlib
import inspect
from contextlib import contextmanager


# Mainly used for test purpose where `load_function` needs to load many in-flight generated functions
class FunctionRegistry:
    # [CLASS DOC] Mapping from name to object with temporary registration support.
    def __init__(self):
        # [FUNCTION DOC] Starts with an empty registry.
        self._registry: dict[str, object] = {}

    @contextmanager
    def temporary(self, name: str, fn: object):
        # [FUNCTION DOC] Context manager registering the object under the name (assertion error if
        # the name is taken) and always unregistering it on exit. Yields nothing.
        self._register(name, fn)
        try:
            yield
        finally:
            self._unregister(name)

    def get(self, name: str) -> object | None:
        # [FUNCTION DOC] Returns the registered object or none.
        return self._registry.get(name)

    def _register(self, name: str, fn: object) -> None:
        # [FUNCTION DOC] HELPER: Adds an entry; asserts the name is not already registered.
        assert name not in self._registry
        self._registry[name] = fn

    def _unregister(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Removes an entry; asserts it exists.
        assert name in self._registry
        self._registry.pop(name)


function_registry = FunctionRegistry()


# TODO may rename to `load_object` since it can be used to load things like tool_specs
def load_function(path, *, sync_required=False):
    # [FUNCTION DOC] Resolves a dotted path to an object. Empty path returns none. Looks in the
    # global registry first; otherwise splits at the last dot, imports the module part and takes the
    # attribute (import and attribute errors propagate). With sync required, raises a value error if
    # the result is not callable or is a coroutine function. A path without a dot yields an empty
    # module name and fails on import.
    """
    Load a function from registry or module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :param sync_required: Reject coroutine functions, for callers that run the
        loaded function synchronously on an event loop.
    :return: The function object.
    """
    if not path:
        return None

    fn = function_registry.get(path)
    if fn is None:
        module_path, _, attr = path.rpartition(".")
        module = importlib.import_module(module_path)
        fn = getattr(module, attr)
    if sync_required:
        if not callable(fn):
            raise ValueError(f"load_function({path!r}) did not resolve to a callable")
        if inspect.iscoroutinefunction(fn):
            raise ValueError(f"load_function({path!r}) resolved to an async function; a synchronous one is required")
    return fn
