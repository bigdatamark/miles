# [MODULE DOC]
# Outer launcher of a served worker, run as a module with its own options (spec table function and
# pool id), optionally followed by a double dash and the worker's own arguments. It resolves the
# pool's spec, works out the worker's environment variables from the identity given by the platform
# environment, and then replaces the process (exec) with the inner server module under that
# environment, so the worker process starts with the environment already in place.
from __future__ import annotations

import os
import sys

from miles.utils.workers.argv_utils import python_argv_prefix
from miles.utils.workers.env_vars import PLATFORM_IDENTITY_ENV_VARS
from miles.utils.workers.serving.utils import compute_serve_worker_spec, parse_own_args, split_worker_argv
from miles.utils.workers.serving.worker_identity import read_worker_identity
from miles.utils.workers.worker_spec import WorkerLaunchContext

SERVE_INNER_MODULE = "miles.utils.workers.serving.serve_inner"


def main() -> None:
    # [FUNCTION DOC]
    # Split the command line at the double dash into own options and worker arguments, parse the
    # own options, load the spec of the requested pool (asserting it exists and is a served spec),
    # read the worker identity (cell index, worker index, gpu ids) from the process environment and
    # ask the spec for the environment variables for that launch. Asserts the spec does not set
    # any platform-owned identity variable (that would misreport the worker's identity). Then
    # replaces the current process with a new python interpreter running the inner module with the
    # same own options, a double dash and the worker arguments, with the environment being the
    # current one overlaid with the spec's variables. Logs progress to standard output; does not
    # return on success.
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])
    args = parse_own_args(own_argv)
    _log(f"start own_argv={own_argv} worker_argv={worker_argv}")

    spec = compute_serve_worker_spec(specs_fn=args.specs, pool_id=args.pool_id, worker_argv=worker_argv)
    identity = read_worker_identity(scheduling=spec.scheduling, environ=os.environ)
    env_vars = spec.env_var(
        WorkerLaunchContext(
            cell_index=identity.cell_index,
            worker_in_cell_index=identity.worker_in_cell_index,
            gpu_ids=identity.gpu_ids,
        )
    )
    overridden = sorted(name for name in PLATFORM_IDENTITY_ENV_VARS if name in env_vars)
    assert not overridden, (
        f"spec {args.pool_id} sets {overridden}, which the platform owns; a worker that read the spec's value "
        f"would report the identity of another worker and bind that worker's ports"
    )
    _log(f"pool_id={args.pool_id} env_vars={env_vars}")

    inner_argv = [*python_argv_prefix(), "-m", SERVE_INNER_MODULE, *own_argv, "--", *worker_argv]
    _log(f"exec {inner_argv}")
    os.execve(sys.executable, inner_argv, dict(os.environ) | env_vars)


def _log(message: str) -> None:
    # [FUNCTION DOC] HELPER: Print the message prefixed with the launcher tag to standard output,
    # flushing immediately.
    print(f"[serve] {message}", flush=True)


if __name__ == "__main__":
    main()
