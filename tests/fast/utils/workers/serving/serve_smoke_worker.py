# [MODULE DOC] HELPER: Smoke worker module used by the serving tests as the target of the serve
# entrypoint. It provides a small worker class, a function computing the worker specs (the "specs"
# entry the entrypoint imports by dotted path), and constants naming environment variables and the
# pool id. The specs describe one pool of one cell with one cpu-only worker, a static rpc port taken
# from the worker argv, and environment variables that echo the argv and the set of imported
# top-level modules (so tests can verify the light entrypoint imports).
import os

from tests.fast.utils.workers.import_probe import report_imported_top_level_modules

from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

IMPORTED_MODULES_ENV_VAR = "MILES_SERVE_SMOKE_IMPORTED_MODULES"
SMOKE_EXTRA_ENV_VAR = "MILES_SERVE_SMOKE_EXTRA_ENV_NAME"
POOL_ID = "e2e-pool"
RPC_PORT_FLAG = "--rpc-port"


class SmokeWorker:
    # [CLASS DOC] HELPER: Worker served by the smoke tests. It keeps the worker argv it was built
    # with and exposes an add method, an argv reporter and an environment variable reporter.
    def __init__(self, argv: list[str]):
        # [FUNCTION DOC] HELPER: Stores the worker argv list.
        self._argv = argv

    def demo_sync(self, a: int, b: int) -> int:
        # [FUNCTION DOC] HELPER: Returns a plus b.
        return a + b

    def report_argv(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the stored worker argv list.
        return self._argv

    def report_env(self, name: str) -> str | None:
        # [FUNCTION DOC] HELPER: Returns the value of the named environment variable of the serving
        # process, or nothing when it is not set.
        return os.environ.get(name)


def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    # [FUNCTION DOC] HELPER: Builds the list with a single serve worker spec named by the pool id.
    # The spec has one port entry called rpc with a static port parsed from the worker argv; one cell
    # with one worker and zero gpus; the worker class set to this module's smoke worker by dotted
    # name; constructor keyword arguments holding the worker argv; and an environment variable
    # function (ignoring its context argument) that returns: the comma-joined worker argv, the pool
    # id, the set of imported top-level modules reported by the import probe (evaluated when the
    # function is called), and, only when the extra-env-name variable is set and non-empty in the
    # current process, an entry with that name and the value "0". Raises an assertion error (via the
    # port parser) when the argv lacks the rpc port flag.
    return [
        ServeWorkerSpec(
            name=POOL_ID,
            port_infos=[PortInfo(name="rpc", static_port=rpc_port_of(worker_argv))],
            env_var=lambda context: {
                "MILES_SERVE_SMOKE_ENV": ",".join(worker_argv),
                "MILES_SERVE_SMOKE_POOL_ID": POOL_ID,
                IMPORTED_MODULES_ENV_VAR: report_imported_top_level_modules(),
                **({name: "0"} if (name := os.environ.get(SMOKE_EXTRA_ENV_VAR)) else {}),
            },
            scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
            worker_class=f"{__name__}.SmokeWorker",
            ctor_kwargs=lambda context: dict(argv=worker_argv),
        )
    ]


def rpc_port_of(worker_argv: list[str]) -> int:
    # [FUNCTION DOC] HELPER: Returns the integer that follows the first occurrence of the rpc port
    # flag in the argv. Raises an assertion error naming the flag and the argv when the flag is absent;
    # raises an index error if the flag is last and a value error if the next item is not an integer.
    assert RPC_PORT_FLAG in worker_argv, f"the smoke run's argv must carry {RPC_PORT_FLAG}, got {worker_argv}"
    return int(worker_argv[worker_argv.index(RPC_PORT_FLAG) + 1])
