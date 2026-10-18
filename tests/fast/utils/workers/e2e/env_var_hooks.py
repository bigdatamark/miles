# [MODULE DOC] HELPER:
# Spec-computation entry points for tests of the worker environment-variable hook. One variant
# exports the list of top-level python modules imported in the spec-computing process, another
# has a hook that always fails. Both reuse the standard end-to-end spec builder.
from __future__ import annotations

from tests.fast.utils.workers.e2e.e2e_worker import spec_of
from tests.fast.utils.workers.import_probe import report_imported_top_level_modules

from miles.utils.workers.worker_spec import ServeWorkerSpec

IMPORTED_MODULES_ENV_VAR = "MILES_E2E_IMPORTED_MODULES"
ENV_VAR_FN_FAILURE_MESSAGE = "env var hook refuses to run"


def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    # [FUNCTION DOC] HELPER: Returns a one-element spec list whose environment hook sets the
    # imported-modules variable to the report of imported top-level modules, computed at the
    # time the hook runs (not at spec-building time).
    return [
        spec_of(worker_argv, env_var=lambda context: {IMPORTED_MODULES_ENV_VAR: report_imported_top_level_modules()})
    ]


def compute_failing_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    # [FUNCTION DOC] HELPER: Returns a one-element spec list whose environment hook always
    # raises, to test how the server reports a failing hook.
    return [spec_of(worker_argv, env_var=_raise)]


def _raise(context) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Environment hook that ignores its context and always raises a
    # runtime error carrying the failure message constant.
    raise RuntimeError(ENV_VAR_FN_FAILURE_MESSAGE)
