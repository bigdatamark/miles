# [MODULE DOC] Tiny decorator utility: run a fixed callback after a function finishes.
from functools import wraps


def with_defer(deferred_func):
    # [FUNCTION DOC] Decorator factory. Takes a callback with no arguments and returns a decorator
    # that wraps a function so the callback always runs after the function finishes, whether it
    # returned or raised (the exception still propagates; a callback error would replace it). The
    # function's return value is passed through unchanged; metadata is preserved.
    def decorator(fn):
        # [FUNCTION DOC] HELPER: Builds the wrapper around the decorated function.
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the function with all arguments, returns its result,
            # and in a final step calls the deferred callback.
            try:
                return fn(*args, **kwargs)
            finally:
                deferred_func()

        return wrapper

    return decorator
