# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

# [MODULE DOC] Builds the command-line apps and the CI runner for fault-tolerance (FT) end-to-end
# tests. Two shapes are supported: a "comparison" test that trains a baseline side and a target
# side (optionally in several named phases) and then compares their dump directories, and a
# "non-comparison" test that trains once and optionally verifies the dumps.
# Dump directories live under a root taken from the environment variable for the dumps root
# (default: a public node directory) plus the run id and test name. For the Kubernetes cluster
# backend, each comparison side's helm release is removed and waited on after the side finishes.
# Public constants: the side names for baseline and target. Callers inject argument builders,
# a comparison function, and optional hooks (config per side, a context manager around the
# target side, a custom side runner, a mode resolver).

import contextlib
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from tests.e2e.ft.conftest_ft.cli_options import (
    DumpDirOption,
    EnableDumperOption,
    ModeOption,
    OptionalModeOption,
    PhaseOption,
)

from tests.e2e.ft.conftest_ft.execution import get_common_train_args, prepare, run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.pod_facts import selected_pods
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import ClusterBackend


BASELINE_SIDE: str = "baseline"
TARGET_SIDE: str = "target"

_DUMPS_ROOT_ENV = "MILES_TEST_DUMPS_ROOT"
_DEFAULT_DUMPS_ROOT = Path("/node_public/dumps")
_RELEASE_POLL_INTERVAL_SECONDS = 1.0
_RELEASE_TIMEOUT_SECONDS = 300.0

BuildArgsFn = Callable[[FTTestMode, str, bool], str]
ConfigForSideFn = Callable[[str, command_utils.ExecuteTrainConfig], command_utils.ExecuteTrainConfig]
TargetSideContextFn = Callable[
    [FTTestMode, str, command_utils.ExecuteTrainConfig], contextlib.AbstractContextManager[None]
]


@dataclass(frozen=True)
class RunSideRequest:
    # [CLASS DOC] Immutable bundle describing one training run of one side. Fields: the side name
    # (baseline or target), the resolved FT test mode, the full training argument string, the
    # side-specific dump directory, the execution config for this side, and whether the dumper
    # is enabled.
    side: str
    mode: FTTestMode
    train_args: str
    dump_dir: str
    config: command_utils.ExecuteTrainConfig
    enable_dumper: bool


RunSideFn = Callable[[RunSideRequest], None]
ReleaseSideFn = Callable[[RunSideRequest], None]
ResolveModeFn = Callable[[str | None], FTTestMode]


def run_one_release(request: RunSideRequest) -> None:
    # [FUNCTION DOC] Default side runner: launches one training run using the request's train
    # arguments, mode, dump directory and config. Returns nothing; side effects are those of
    # the training run (it removes the dump directory first if it exists).
    run_training(train_args=request.train_args, mode=request.mode, dump_dir=request.dump_dir, config=request.config)


def resolve_dump_dir(test_name: str, *, run_id: str) -> str:
    # [FUNCTION DOC] Computes and creates the dump directory for a test. The root is the value of
    # the dumps-root environment variable when set and non-empty, otherwise the default public
    # node dumps directory; the result is root / run id / test name. Creates the directory
    # (and parents) if missing. Returns the path as a string.
    root = os.environ.get(_DUMPS_ROOT_ENV) or _DEFAULT_DUMPS_ROOT
    dump_dir = Path(root) / run_id / test_name
    os.makedirs(dump_dir, exist_ok=True)
    return str(dump_dir)


def _dump_subdir(side: str, phase: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the relative sub-directory for a side: "side/phase" when the
    # phase is non-empty, otherwise just the side name.
    return f"{side}/{phase}" if phase else side


def _release_comparison_side(request: RunSideRequest) -> None:
    # [FUNCTION DOC] HELPER: Default post-side cleanup. Does nothing unless the request's config
    # uses the Kubernetes cluster backend. For Kubernetes, asserts that a namespace is set, builds
    # the serialized release name from run id, deploy component and deploy instance id, and
    # removes that release and waits until it and its pods are gone. May raise an assertion error
    # (no namespace) or a timeout error from the removal.
    config = request.config
    if config.cluster_backend is not ClusterBackend.KUBERNETES:
        return

    assert config.namespace, "A kubernetes comparison side needs a namespace before its release can be removed"
    remove_release_and_wait(
        release=ReleaseName(
            run_id=config.run_id,
            deploy_component=config.deploy_component,
            deploy_instance_id=config.deploy_instance_id,
        ).serialize(),
        namespace=config.namespace,
    )


def remove_release_and_wait(*, release: str, namespace: str) -> None:
    # [FUNCTION DOC] Uninstalls a helm release if present and blocks until both the release
    # manifest is gone and no pods matching the release selector remain. Polls every second
    # (the poll interval constant) with a 300 second deadline (the timeout constant).
    # Raises a timeout error listing whether the release still exists and the sorted names of
    # remaining pods. Side effects: helm uninstall, cluster queries via kubectl/helm wrappers.
    selector = Kubectl.release_selector(release)
    deadline = time.monotonic() + _RELEASE_TIMEOUT_SECONDS

    Helm.uninstall_if_present(release=release, namespace=namespace)
    while True:
        manifest = Helm.get_manifest(release, namespace)
        pods = selected_pods(namespace, selector)
        if manifest is None and not pods:
            return
        if time.monotonic() >= deadline:
            pod_names = sorted(pod.metadata.name for pod in pods)
            raise TimeoutError(
                f"Timed out removing release {release!r} from namespace {namespace!r}; "
                f"release_exists={manifest is not None}, pods={pod_names}"
            )
        time.sleep(_RELEASE_POLL_INTERVAL_SECONDS)


def run_pipeline(
    *,
    test_name: str,
    build_baseline_args: BuildArgsFn,
    build_target_args: BuildArgsFn,
    compare_fn: Callable[[str, FTTestMode], None],
    phases: list[str] | None,
    mode: str | None,
    enable_dumper: bool = True,
    target_side_context: TargetSideContextFn | None = None,
    config_for_side: ConfigForSideFn | None = None,
    run_side: RunSideFn = run_one_release,
    release_side: ReleaseSideFn = _release_comparison_side,
    resolve_mode_fn: ResolveModeFn = resolve_mode,
) -> None:
    # [FUNCTION DOC] Runs the whole comparison pipeline for one mode. Steps: use the given phases
    # or a single empty phase; resolve the mode through the resolver; create the dump directory
    # (named by test name under the current default config's run id) and print it; prepare the
    # models and data once; then for each phase, for baseline then target: derive the side dump
    # directory (dump dir / side [/ phase]), resolve the side config, build the target-side
    # context (a no-op context for baseline or when no hook is given), build the request with
    # train arguments from the corresponding builder, run the side inside the context, and always
    # invoke the release hook afterwards even on failure. After all sides, when the dumper is
    # enabled, call the compare function with the dump dir and mode. The dump directory is always
    # deleted at the end (errors ignored), even on failure. Exceptions from any step propagate.
    """Full pipeline (prepare + every phase's baseline/target + compare) for one mode."""
    effective_phases: list[str] = phases or [""]
    ft_mode: FTTestMode = resolve_mode_fn(mode)
    dump_dir: str = resolve_dump_dir(test_name, run_id=command_utils.default_config().run_id)
    print(f"Dump directory: {dump_dir}")

    prepare(ft_mode)

    try:
        for phase in effective_phases:
            for side, build_args in (
                (BASELINE_SIDE, build_baseline_args),
                (TARGET_SIDE, build_target_args),
            ):
                side_dump = f"{dump_dir}/{_dump_subdir(side, phase)}"
                config = _resolve_config_for_side(side, config_for_side=config_for_side)
                context = (
                    target_side_context(ft_mode, side_dump, config)
                    if side == TARGET_SIDE and target_side_context is not None
                    else contextlib.nullcontext()
                )
                request = RunSideRequest(
                    side=side,
                    mode=ft_mode,
                    train_args=build_args(ft_mode, side_dump, enable_dumper),
                    dump_dir=side_dump,
                    config=config,
                    enable_dumper=enable_dumper,
                )
                try:
                    with context:
                        run_side(request)
                finally:
                    release_side(request)

        if enable_dumper:
            compare_fn(dump_dir, ft_mode)
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


def create_comparison_app_and_run_ci(
    *,
    test_name: str,
    build_baseline_args: BuildArgsFn,
    build_target_args: BuildArgsFn,
    compare_fn: Callable[[str, FTTestMode], None],
    phases: list[str] | None = None,
    target_side_context: TargetSideContextFn | None = None,
    config_for_side: ConfigForSideFn | None = None,
    run_side: RunSideFn = run_one_release,
    resolve_mode_fn: ResolveModeFn = resolve_mode,
) -> tuple[typer.Typer, Callable[[str | None], None]]:
    # [FUNCTION DOC] Factory wiring a comparison test. Takes the test name, baseline and target
    # argument builders (mode, dump dir, dumper flag -> argument string), a compare function,
    # optional phase names, and optional hooks (target-side context, per-side config, side runner,
    # mode resolver). Returns a pair: a typer app with the commands baseline, target, compare,
    # run and generate-data, and a one-shot runner taking an optional mode name. The runner
    # runs the full pipeline with the dumper always enabled and a test name suffixed with the mode
    # (when a mode is given) so concurrent CI modes use distinct dump directories.
    # Command behaviors: baseline/target run a single side (mode, dump dir, phase, dumper flag
    # options; dump dir defaults to the resolved dump directory for the config's run id; models
    # are prepared first; the target side runs inside the target context). compare calls the
    # compare function on a required dump directory. run executes the full pipeline.
    # generate-data (default 12 steps, default output directory under the temp directory)
    # prepares models and runs real-rollout training with the dumper disabled to record debug
    # rollout data; it asserts the mode has a real rollout.
    """Build, from one wiring, the manual typer app and a run_ci(mode) one-shot runner.

    Returns ``(app, run_ci)``: ``app`` exposes run/baseline/target/compare for manual use;
    ``run_ci(mode)`` runs the full pipeline for a single mode (used by the per-mode CI entry
    files), writing dumps under a per-mode test name so concurrent CI modes don't collide.

    For simple (no-phase) tests, leave phases empty.
    For multi-phase tests (e.g. with_failure), provide phase names like ["phase_a", "phase_b"].
    """
    app: typer.Typer = typer.Typer()

    def _execute_one_side(
        side: str,
        build_fn: BuildArgsFn,
        mode: str | None,
        dump_dir: str | None,
        phase: str,
        *,
        enable_dumper: bool = True,
    ) -> None:
        # [FUNCTION DOC] HELPER: Runs exactly one side. Resolves the mode and side config, picks the
        # dump directory (given, or resolved from test name and config run id), appends the
        # side/phase sub-directory, builds train arguments, prepares models and data, then runs the
        # side via the side runner (inside the target context for the target side only).
        ft_mode = resolve_mode_fn(mode)
        config = _resolve_config_for_side(side, config_for_side=config_for_side)
        if dump_dir is None:
            dump_dir = resolve_dump_dir(test_name, run_id=config.run_id)
        sub = _dump_subdir(side, phase)
        full_dump_dir = f"{dump_dir}/{sub}"
        args = build_fn(ft_mode, full_dump_dir, enable_dumper)
        prepare(ft_mode)

        context = (
            target_side_context(ft_mode, full_dump_dir, config)
            if side == TARGET_SIDE and target_side_context is not None
            else contextlib.nullcontext()
        )
        with context:
            run_side(
                RunSideRequest(
                    side=side,
                    mode=ft_mode,
                    train_args=args,
                    dump_dir=full_dump_dir,
                    config=config,
                    enable_dumper=enable_dumper,
                )
            )

    @app.command()
    def baseline(
        mode: OptionalModeOption = None,
        dump_dir: DumpDirOption = None,
        phase: PhaseOption = "",
        enable_dumper: EnableDumperOption = True,
    ) -> None:
        # [FUNCTION DOC] Command: runs only the baseline side with the given options.
        """Run baseline (normal DP) training."""
        _execute_one_side(BASELINE_SIDE, build_baseline_args, mode, dump_dir, phase, enable_dumper=enable_dumper)

    @app.command()
    def target(
        mode: OptionalModeOption = None,
        dump_dir: DumpDirOption = None,
        phase: PhaseOption = "",
        enable_dumper: EnableDumperOption = True,
    ) -> None:
        # [FUNCTION DOC] Command: runs only the target side with the given options.
        """Run target (indep_dp) training."""
        _execute_one_side(TARGET_SIDE, build_target_args, mode, dump_dir, phase, enable_dumper=enable_dumper)

    @app.command()
    def compare(
        dump_dir: Annotated[str, typer.Option(help="Dump base directory")],
        mode: OptionalModeOption = None,
    ) -> None:
        # [FUNCTION DOC] Command: resolves the mode and calls the compare function on an existing
        # dump base directory; no training is run.
        """Compare baseline and target dumps."""
        ft_mode = resolve_mode_fn(mode)
        compare_fn(dump_dir, ft_mode)

    @app.command()
    def run(
        mode: OptionalModeOption = None,
        enable_dumper: EnableDumperOption = True,
    ) -> None:
        # [FUNCTION DOC] Command: runs the full pipeline for the given mode and dumper flag using
        # the closure's wiring.
        """Full pipeline: prepare + all phases + compare."""
        run_pipeline(
            test_name=test_name,
            build_baseline_args=build_baseline_args,
            build_target_args=build_target_args,
            compare_fn=compare_fn,
            phases=phases,
            mode=mode,
            enable_dumper=enable_dumper,
            target_side_context=target_side_context,
            config_for_side=config_for_side,
            run_side=run_side,
            resolve_mode_fn=resolve_mode_fn,
        )

    @app.command()
    def generate_data(
        mode: OptionalModeOption = None,
        num_steps: Annotated[int, typer.Option(help="Number of rollout steps to generate")] = 12,
        output_dir: Annotated[
            str, typer.Option(help="Output directory for rollout data")
        ] = "/tmp/generated_rollout_data",
    ) -> None:
        # [FUNCTION DOC] Command: records debug rollout data into the output directory by training
        # for the requested number of steps with a real rollout and no dumper. Asserts the
        # resolved mode has real rollout engines; prepares models and data first.
        """Generate debug rollout data using real rollout (no dumper)."""
        ft_mode = resolve_mode_fn(mode)
        assert (
            ft_mode.has_real_rollout
        ), f"recording debug rollout data needs real engines, and the mode runs {ft_mode.rollout_num_engines}"
        prepare(ft_mode)
        args = get_common_train_args(ft_mode, dump_dir=output_dir, num_steps=num_steps, enable_dumper=False)
        run_training(train_args=args, mode=ft_mode)

    def run_ci(mode: str | None = None) -> None:
        # [FUNCTION DOC] Returned one-shot runner: runs the full pipeline for the given mode with
        # the dumper enabled; test name gets a mode suffix when a mode is provided.
        """Run one mode's full pipeline (entry point for the per-mode CI files)."""
        run_pipeline(
            test_name=f"{test_name}_{mode}" if mode is not None else test_name,
            build_baseline_args=build_baseline_args,
            build_target_args=build_target_args,
            compare_fn=compare_fn,
            phases=phases,
            mode=mode,
            target_side_context=target_side_context,
            config_for_side=config_for_side,
            run_side=run_side,
            resolve_mode_fn=resolve_mode_fn,
        )

    return app, run_ci


def _resolve_config_for_side(
    side: str, *, config_for_side: ConfigForSideFn | None
) -> command_utils.ExecuteTrainConfig:
    # [FUNCTION DOC] HELPER: Returns the default execution config, passed through the optional
    # per-side config hook (called with the side name and the default config) when provided.
    config = command_utils.default_config()
    return config_for_side(side, config) if config_for_side is not None else config


def create_non_comparison_app(
    *,
    test_name: str,
    build_args: Callable[[FTTestMode, str], str],
    verify_fn: Callable[[str, FTTestMode], None] | None = None,
) -> typer.Typer:
    # [FUNCTION DOC] Factory for a single-run test. Takes the test name, an argument builder
    # (mode, dump dir -> argument string) and an optional verify function (dump dir, mode).
    # Returns a typer app with one command, run, whose required option is the mode name.
    """Generate a typer app with a single 'run' command for non-comparison tests."""
    app: typer.Typer = typer.Typer()

    @app.command()
    def run(
        mode: ModeOption,
    ) -> None:
        # [FUNCTION DOC] Command: resolves the mode with the default resolver, creates and prints the
        # dump directory, prepares models and data, trains once with the built arguments and
        # default config, then calls the verify function if provided. The dump directory is not
        # removed afterwards.
        """Full pipeline: prepare + execute + verify."""
        ft_mode = resolve_mode(mode)
        config = command_utils.default_config()
        dump_dir: str = resolve_dump_dir(test_name, run_id=config.run_id)
        print(f"Dump directory: {dump_dir}")

        prepare(ft_mode)
        args = build_args(ft_mode, dump_dir)
        run_training(train_args=args, mode=ft_mode, config=config)

        if verify_fn is not None:
            verify_fn(dump_dir, ft_mode)

    return app
