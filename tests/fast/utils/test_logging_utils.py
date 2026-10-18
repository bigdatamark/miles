# [MODULE DOC]
# Tests for the logging utilities module. Parts: (1) logger configuration: configuring a process
# logger starts environment reporting exactly once per process (unless disabled by a flag), a later
# raw configure call renames the process in the log format, and rebinding forwards final arguments
# to an installed reporter or does nothing when none is installed (the module-level reporter
# global is reset around each of these tests); (2) static checks that every worker class named by
# the specs package (and the worker manager) calls the logger configuration somewhere in its class
# hierarchy source, and that the controller, executor and worker-manager modules name themselves
# with the right component; a module-scoped fixture stubs the GPU-image-only memory saver module
# when it is missing; (3) the strict async warnings configuration, tested in subprocesses and
# in-process: an unawaited coroutine terminates the process with exit code 1 and a fatal message,
# properly awaited coroutines work, and unrelated warnings are not turned into errors. An autouse
# fixture activates the strict configuration for every test in the module and restores the
# unraisable hook afterwards.
# KNOWN ISSUE: the existing module docstring only mentions the strict async warnings configuration,
# while the module also tests logger configuration and worker logging setup.
"""Tests for configure_strict_async_warnings."""

import argparse
import asyncio
import importlib
import inspect
import pkgutil
import subprocess
import sys
import textwrap
import types
import warnings
from unittest.mock import patch

import pytest

from miles.utils import logging_utils
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.logging_utils import configure_logger, configure_strict_async_warnings
from miles.utils.workers.ray_worker_manager import RayWorkerManager

SPECS_PACKAGE = "miles.ray.specs"


def _worker_class_paths() -> list[str]:
    # [FUNCTION DOC] HELPER: Scan every module of the specs package; collect the dotted class paths
    # held in module constants whose names end with the worker-class suffix, plus all values of
    # dictionary constants whose names end with the classes suffix. Asserts at least one path was
    # found (message names the package) and returns the sorted, de-duplicated list. Imports every
    # spec module as a side effect.
    """Every class a spec can name as its worker: the *_WORKER_CLASS constants, plus the mappings
    a spec picks a backend's rank class out of."""
    paths: set[str] = set()
    package = importlib.import_module(SPECS_PACKAGE)
    for module in pkgutil.iter_modules(package.__path__):
        for name, value in vars(importlib.import_module(f"{SPECS_PACKAGE}.{module.name}")).items():
            if name.endswith("_WORKER_CLASS"):
                paths.add(value)
            elif name.endswith("_CLASSES") and isinstance(value, dict):
                paths.update(value.values())

    assert paths, f"no worker classes found in {SPECS_PACKAGE}"
    return sorted(paths)


def _load_class(path: str):
    # [FUNCTION DOC] HELPER: Split a dotted path at the last dot, import the module part and return
    # the named class attribute. Import errors or a missing attribute propagate.
    module_path, _, class_name = path.rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def _class_source(klass: type) -> str | None:
    # [FUNCTION DOC] HELPER: Return the source text of the class, or none if the source cannot be
    # obtained (OS or type errors, such as for built-in classes).
    try:
        return inspect.getsource(klass)
    except (OSError, TypeError):
        return None


def _configures_logger(klass: type) -> bool:
    # [FUNCTION DOC] HELPER: Gather the source of the class and each base class in its resolution
    # order (skipping those without source) and return true if any contains a textual call to the
    # logger-configuration function. It is a text search, so mentions in comments also count.
    sources = [source for base in klass.__mro__ if (source := _class_source(base)) is not None]
    return any("configure_logger(" in source for source in sources)


class TestConfigureLogger:
    # [CLASS DOC] Tests of logger configuration and environment-reporter handling.
    @pytest.fixture(autouse=True)
    def _forget_reporter(self):
        # [FUNCTION DOC] HELPER: Autouse fixture that sets the module's global environment reporter
        # to none before the test and again after it.
        logging_utils._ENV_REPORTER = None
        yield
        logging_utils._ENV_REPORTER = None

    def _configure(self, **overrides) -> None:
        # [FUNCTION DOC] HELPER: Call the logger configuration with arguments that disable debug
        # event saving, a main-component process identity, and any keyword overrides.
        configure_logger(
            argparse.Namespace(save_debug_event_data=None),
            source=SimpleProcessIdentity(component="main"),
            **overrides,
        )

    def test_reports_the_environment_of_this_process(self) -> None:
        # [FUNCTION DOC] With environment reporting start patched, one configure call starts the
        # reporter exactly once.
        """Configuring a process's logger is what makes it record the environment it runs in."""
        with patch("miles.utils.logging_utils.start_env_reporting") as start:
            self._configure()

        assert start.call_count == 1

    def test_a_second_call_does_not_start_a_second_reporter(self) -> None:
        # [FUNCTION DOC] Two configure calls start the reporter only once in total.
        """A process that configures its logger twice would otherwise report everything twice, forever."""
        with patch("miles.utils.logging_utils.start_env_reporting") as start:
            self._configure()
            self._configure()

        assert start.call_count == 1

    def test_a_later_call_renames_the_process(self) -> None:
        # [FUNCTION DOC] With the standard library's basic logging setup patched, call the raw
        # configure with the name "main" and then with "multi_lora_controller"; the second call's
        # log format contains the new name once, the first call's format contains it zero times.
        """A worker configures its logger long after parse_args named the process `main`."""
        with patch.object(logging_utils.logging, "basicConfig") as basic_config:
            logging_utils.configure_logger_raw("main")
            logging_utils.configure_logger_raw("multi_lora_controller")

        assert [call.kwargs["format"].count("multi_lora_controller") for call in basic_config.call_args_list] == [
            0,
            1,
        ]

    def test_a_process_that_is_not_a_run_reports_nothing(self) -> None:
        # [FUNCTION DOC] Configuring with environment reporting turned off never starts the
        # reporter.
        """Probing pip and git costs a process that only borrows the logging setup of a run."""
        with patch("miles.utils.logging_utils.start_env_reporting") as start:
            self._configure(report_env=False)

        assert start.call_count == 0

    def test_rebind_forwards_the_final_args_to_the_installed_reporter(self) -> None:
        # [FUNCTION DOC] Install a fake reporter whose rebind records its argument; rebinding with
        # a final arguments namespace passes exactly that object to the reporter once.
        """A configured reporter must replace its launch-time arguments with the trainer's final ones."""
        final_args = argparse.Namespace(rank=3, world_size=8)
        reporter = types.SimpleNamespace(rebound=[])
        reporter.rebind = reporter.rebound.append
        logging_utils._ENV_REPORTER = reporter

        logging_utils.rebind_env_reporting(final_args)

        assert reporter.rebound == [final_args]

    def test_rebind_without_an_installed_reporter_is_a_noop(self) -> None:
        # [FUNCTION DOC] With no reporter installed, rebinding does not raise and the global
        # reporter stays none.
        """Processes that disabled environment reporting must still finish initialization."""
        logging_utils.rebind_env_reporting(argparse.Namespace(rank=0))

        assert logging_utils._ENV_REPORTER is None


@pytest.fixture(autouse=True, scope="module")
def _stub_gpu_image_imports():
    # [FUNCTION DOC] HELPER: Autouse module-scoped fixture. If the memory-saver module is already
    # loaded, do nothing. Otherwise insert a stub module (exposing a placeholder object under the
    # same name) into the module table for the duration of the module's tests and remove it at
    # teardown. Side effect: modifies the global module table.
    """The megatron actor imports the memory saver at module scope, and that ships with the gpu
    image rather than with miles, so the cpu lane cannot import the class to read its source."""
    if "torch_memory_saver" in sys.modules:
        yield
        return

    stub = types.ModuleType("torch_memory_saver")
    stub.torch_memory_saver = object()
    sys.modules["torch_memory_saver"] = stub
    try:
        yield
    finally:
        del sys.modules["torch_memory_saver"]


class TestServedWorkerLogging:
    # [CLASS DOC] Static source-text checks that served worker processes configure their logger.
    @pytest.mark.parametrize("worker_class_path", _worker_class_paths())
    def test_worker_configures_its_logger(self, worker_class_path: str) -> None:
        # [FUNCTION DOC] Parametrized over every worker class path found in the specs package
        # (computed at collection time, so import errors surface at collection). Paths of
        # torchtitan utilities are skipped when torchtitan is not installed. Assert the loaded
        # class (or a base) calls the logger configuration, with a message naming the path.
        """Every served worker owns a process, and configure_logger is where that process names
        itself, opens its event log and reports its environment."""
        if ".torchtitan_utils." in worker_class_path:
            pytest.importorskip("torchtitan", reason="torchtitan ships with the gpu image, not the cpu lane")
        assert _configures_logger(
            _load_class(worker_class_path)
        ), f"{worker_class_path} runs as its own process but never calls configure_logger"

    @pytest.mark.parametrize(
        ("module_path", "component"),
        [
            ("miles.ray.rollout.inference_controller", "inference_controller"),
            ("miles.ray.rollout.rollout_executor", "rollout_executor"),
            ("miles.utils.workers.ray_worker_manager", "worker_manager"),
        ],
    )
    def test_each_process_names_itself_as_the_component_it_is(self, module_path: str, component: str) -> None:
        # [FUNCTION DOC] Parametrized over three modules (inference controller, rollout executor,
        # worker manager) and their expected component names; assert the module source contains a
        # process-identity construction with exactly that component name.
        """One SimpleProcessIdentity now serves them all, so only the argument tells their events apart."""
        source = inspect.getsource(importlib.import_module(module_path))

        assert f'SimpleProcessIdentity(component="{component}")' in source

    def test_the_worker_manager_configures_its_logger(self) -> None:
        # [FUNCTION DOC] Assert the Ray worker manager class (or a base) calls the logger
        # configuration.
        """The ray worker manager is its own actor process, but no spec names it as a worker class."""
        assert _configures_logger(RayWorkerManager)


async def _dummy_coroutine():
    # [FUNCTION DOC] HELPER: Trivial coroutine returning the integer 42.
    return 42


@pytest.fixture(autouse=True)
def _setup_warning_filter():
    # [FUNCTION DOC] HELPER: Autouse fixture. Saves the unraisable-exception hook, enters a
    # warnings-filter-restoring scope, applies the strict async warnings configuration, runs the
    # test, and restores the original hook afterwards (the warnings scope restores filters).
    # Because it is autouse at module level it also applies to the logger tests above.
    """Activate the filter before each test, restore original filters after."""
    original_hook = sys.unraisablehook
    with warnings.catch_warnings():
        configure_strict_async_warnings()
        yield
    sys.unraisablehook = original_hook


def _run_snippet(code: str) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Run the given Python source (after removing common indentation) in a
    # fresh interpreter subprocess with captured text output and a 60 s timeout; returns the
    # completed-process result (return code, stdout, stderr). Raises a timeout error if exceeded.
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestUnawaitedCoroutineCrashesProcess:
    # [CLASS DOC] Subprocess tests that the strict configuration turns unawaited coroutines into a
    # fatal process exit.
    def test_unawaited_coroutine_exits_with_code_1(self):
        # [FUNCTION DOC] A subprocess enables the strict configuration, calls an async function
        # without awaiting and forces garbage collection. Assert exit code 1, that the following
        # print did not run, and that stderr contains "Fatal async misuse".
        result = _run_snippet(
            """
            import gc
            from miles.utils.logging_utils import configure_strict_async_warnings
            configure_strict_async_warnings()

            async def foo(): pass
            foo()
            gc.collect()
            print("should not reach here")
        """
        )
        assert result.returncode == 1
        assert "should not reach here" not in result.stdout
        assert "Fatal async misuse" in result.stderr

    def test_unawaited_coroutine_del_exits_with_code_1(self):
        # [FUNCTION DOC] A subprocess creates a coroutine object, deletes the reference and
        # collects garbage. Assert exit code 1, the later print did not run, and stderr mentions
        # "coroutine".
        result = _run_snippet(
            """
            import gc
            from miles.utils.logging_utils import configure_strict_async_warnings
            configure_strict_async_warnings()

            async def foo(): pass
            c = foo()
            del c
            gc.collect()
            print("should not reach here")
        """
        )
        assert result.returncode == 1
        assert "should not reach here" not in result.stdout
        assert "coroutine" in result.stderr

    def test_awaited_coroutine_no_crash(self):
        # [FUNCTION DOC] A subprocess runs a coroutine properly to completion and prints its
        # result. Assert exit code 0 and "42" in stdout.
        result = _run_snippet(
            """
            import asyncio
            from miles.utils.logging_utils import configure_strict_async_warnings
            configure_strict_async_warnings()

            async def foo(): return 42
            print(asyncio.run(foo()))
        """
        )
        assert result.returncode == 0
        assert "42" in result.stdout


class TestCorrectUsageNoError:
    # [CLASS DOC] In-process tests that correct coroutine use is not affected by the strict
    # configuration (applied by the autouse fixture).
    def test_properly_awaited_coroutine(self):
        # [FUNCTION DOC] Running the trivial coroutine in a new event loop returns 42.
        result = asyncio.run(_dummy_coroutine())
        assert result == 42

    @pytest.mark.asyncio
    async def test_awaited_in_async_context(self):
        # [FUNCTION DOC] Awaiting the trivial coroutine inside an async test returns 42.
        result = await _dummy_coroutine()
        assert result == 42

    @pytest.mark.asyncio
    async def test_gathered_coroutines(self):
        # [FUNCTION DOC] Gathering two trivial coroutines yields [42, 42].
        results = await asyncio.gather(_dummy_coroutine(), _dummy_coroutine())
        assert results == [42, 42]

    @pytest.mark.asyncio
    async def test_create_task_then_await(self):
        # [FUNCTION DOC] Wrapping the coroutine in a task and awaiting it yields 42.
        task = asyncio.create_task(_dummy_coroutine())
        result = await task
        assert result == 42

    @pytest.mark.asyncio
    async def test_eager_create_task(self):
        # [FUNCTION DOC] Awaiting the project's eager task creator on the coroutine, then awaiting
        # the returned task, yields 42.
        from miles.utils.async_utils import eager_create_task

        task = await eager_create_task(_dummy_coroutine())
        result = await task
        assert result == 42


class TestOtherWarningsUnaffected:
    # [CLASS DOC] Tests that the strict configuration leaves unrelated warnings as warnings.
    def test_unrelated_runtime_warning_not_raised(self):
        # [FUNCTION DOC] After configuring, a runtime warning with unrelated text is emitted as a
        # warning (captured by the expect-warning context), not raised as an error.
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            configure_strict_async_warnings()
            with pytest.warns(RuntimeWarning, match="test warning"):
                warnings.warn("test warning", RuntimeWarning, stacklevel=2)

    def test_deprecation_warning_not_raised(self):
        # [FUNCTION DOC] After configuring, a deprecation warning is emitted as a warning, not
        # raised as an error.
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            configure_strict_async_warnings()
            with pytest.warns(DeprecationWarning):
                warnings.warn("old stuff", DeprecationWarning, stacklevel=2)
