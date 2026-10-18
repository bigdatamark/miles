# [MODULE DOC]
# Tests for the ft end-to-end comparison app module: resolving the per-run dump directory (a root
# from an environment variable or a default node-public root, then run id, then test name, created
# on disk), the run pipeline that runs a baseline side then a target side then a comparison, with
# per-side config transforms and resource release between sides, the Kubernetes/Ray release
# handoff of a finished comparison side, and the command-line app for fixed-topology scenarios.
# External effects (dump directory resolution, preparation, training, helm, pod listing, sleeping)
# are replaced by monkeypatched fakes that record events.
import contextlib
import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft import app as app_module
from tests.e2e.ft.conftest_ft.app import _DUMPS_ROOT_ENV, RunSideRequest, resolve_dump_dir, run_pipeline
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from typer.testing import CliRunner

from miles.utils.external_utils import command_utils
from miles.utils.workers.k8s_types import Pod, PodMetadata
from miles.utils.workers.types import ClusterBackend


def test_dump_dir_hangs_off_the_configured_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Sets the dumps-root environment variable to a temp path; asserts the resolved
    # dump dir is root, then run id, then test name.
    """A cluster says where dumps go through the environment its infra file sets."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path / "dumps"))
    assert resolve_dump_dir("scenario_x", run_id="run-a") == str(tmp_path / "dumps" / "run-a" / "scenario_x")


def test_an_empty_configured_root_is_not_a_root(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Sets the root variable to an empty string and stubs directory creation so
    # nothing is created on disk; asserts the default node-public dumps root is used.
    """An unset variable and one set to nothing both mean the cluster configured no root."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, "")
    monkeypatch.setattr("os.makedirs", lambda path, exist_ok: None)

    assert resolve_dump_dir("scenario_x", run_id="run-a") == "/node_public/dumps/run-a/scenario_x"


def test_two_runs_of_one_test_do_not_share_a_dump_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Resolves the dump dir for one test name under two run ids; asserts they differ.
    """The run id in the path is what stops one run's rmtree deleting another's dumps."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path))
    first = resolve_dump_dir("scenario_x", run_id="run-a")

    assert resolve_dump_dir("scenario_x", run_id="run-b") != first


def test_the_dump_directory_exists_when_it_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Asserts the resolved dump dir is an existing directory on disk (created by resolution).
    """Callers write into the returned path without creating it themselves."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path / "dumps"))
    assert Path(resolve_dump_dir("scenario_x", run_id="run-a")).is_dir()


def test_each_comparison_side_can_transform_its_config_before_the_context_and_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Runs the pipeline with a one-cell demo mode, a default config with run id
    # shared-release, a per-side config transform appending the side name to the run id, a
    # recording side runner and a recording target-side context manager. Dump directory resolution
    # and preparation are stubbed. Asserts the two run requests carry the baseline and target run
    # ids, the context was entered once, and the config it received is the very object given to
    # the target request.
    """A side-specific release has to be shared by its target context and the launch it drives."""
    requests: list[RunSideRequest] = []
    contexts: list[command_utils.ExecuteTrainConfig] = []
    mode = FTTestMode(
        model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
    )

    @contextlib.contextmanager
    def target_context(_mode: FTTestMode, _dump_dir: str, config: command_utils.ExecuteTrainConfig) -> Iterator[None]:
        # [FUNCTION DOC] HELPER: Context manager that records the config it was given and yields
        # without doing anything else.
        contexts.append(config)
        yield

    monkeypatch.setattr(app_module, "resolve_dump_dir", lambda _test_name, *, run_id: str(tmp_path / "comparison"))
    monkeypatch.setattr(app_module, "prepare", lambda _mode: None)
    monkeypatch.setattr(
        command_utils, "default_config", lambda: command_utils.ExecuteTrainConfig(run_id="shared-release")
    )

    run_pipeline(
        test_name="scenario_x",
        build_baseline_args=lambda *_args: "",
        build_target_args=lambda *_args: "",
        compare_fn=lambda *_args: None,
        phases=None,
        mode=None,
        target_side_context=target_context,
        config_for_side=lambda side, config: dataclasses.replace(config, run_id=f"{config.run_id}-{side}"),
        run_side=requests.append,
        resolve_mode_fn=lambda _mode: mode,
    )

    assert [request.config.run_id for request in requests] == ["shared-release-baseline", "shared-release-target"]
    assert len(contexts) == 1
    assert contexts[0] is requests[1].config


def test_each_comparison_side_releases_its_resources_before_the_next_side_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Runs the pipeline with recording run, release and compare callbacks; asserts
    # the event order is run baseline, release baseline, run target, release target, compare.
    """A finished baseline cannot overlap the target's GPU requests while its release tears down."""
    events: list[str] = []
    mode = FTTestMode(
        model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
    )

    monkeypatch.setattr(app_module, "resolve_dump_dir", lambda _test_name, *, run_id: str(tmp_path / "comparison"))
    monkeypatch.setattr(app_module, "prepare", lambda _mode: None)

    run_pipeline(
        test_name="scenario_x",
        build_baseline_args=lambda *_args: "",
        build_target_args=lambda *_args: "",
        compare_fn=lambda *_args: events.append("compare"),
        phases=None,
        mode=None,
        run_side=lambda request: events.append(f"run:{request.side}"),
        release_side=lambda request: events.append(f"release:{request.side}"),
        resolve_mode_fn=lambda _mode: mode,
    )

    assert events == ["run:baseline", "release:baseline", "run:target", "release:target", "compare"]


def test_a_failed_comparison_side_is_released_without_starting_the_next_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] The baseline run raises a runtime error. Asserts the pipeline re-raises it and
    # the only events are run baseline and release baseline (no target, no compare).
    """A red verdict still releases its GPUs, but it must not continue into target or compare."""
    events: list[str] = []

    def fail_side(request: RunSideRequest) -> None:
        # [FUNCTION DOC] HELPER: Records the run event for the side then raises a runtime error.
        events.append(f"run:{request.side}")
        raise RuntimeError("baseline failed")

    monkeypatch.setattr(app_module, "resolve_dump_dir", lambda _test_name, *, run_id: str(tmp_path / "comparison"))
    monkeypatch.setattr(app_module, "prepare", lambda _mode: None)

    with pytest.raises(RuntimeError, match="baseline failed"):
        run_pipeline(
            test_name="scenario_x",
            build_baseline_args=lambda *_args: "",
            build_target_args=lambda *_args: "",
            compare_fn=lambda *_args: events.append("compare"),
            phases=None,
            mode=None,
            run_side=fail_side,
            release_side=lambda request: events.append(f"release:{request.side}"),
            resolve_mode_fn=lambda _mode: _mode_fixture(),
        )

    assert events == ["run:baseline", "release:baseline"]


def test_a_failed_release_blocks_the_next_comparison_side(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] The baseline release raises a runtime error. Asserts it propagates out of the
    # pipeline and only run baseline and release baseline were recorded.
    """A target cannot start while the baseline's resource ownership remains unresolved."""
    events: list[str] = []

    def fail_release(request: RunSideRequest) -> None:
        # [FUNCTION DOC] HELPER: Records the release event for the side then raises a runtime error.
        events.append(f"release:{request.side}")
        raise RuntimeError("release failed")

    monkeypatch.setattr(app_module, "resolve_dump_dir", lambda _test_name, *, run_id: str(tmp_path / "comparison"))
    monkeypatch.setattr(app_module, "prepare", lambda _mode: None)

    with pytest.raises(RuntimeError, match="release failed"):
        run_pipeline(
            test_name="scenario_x",
            build_baseline_args=lambda *_args: "",
            build_target_args=lambda *_args: "",
            compare_fn=lambda *_args: events.append("compare"),
            phases=None,
            mode=None,
            run_side=lambda request: events.append(f"run:{request.side}"),
            release_side=fail_release,
            resolve_mode_fn=lambda _mode: _mode_fixture(),
        )

    assert events == ["run:baseline", "release:baseline"]


def test_kubernetes_side_waits_for_its_pods_after_helm_no_longer_lists_the_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Kubernetes config with namespace ci and run id run-baseline. Fakes: helm
    # uninstall records an event, helm manifest lookup returns nothing, the pod listing records the
    # namespace and selector and answers one pod then none, and sleeping records its duration.
    # Releasing the side must uninstall the release (name derived from the run id with an all
    # suffix), then poll pods by the instance label selector, sleep 1.0 second while a pod is still
    # present, and poll again; the asserted event list is exactly that sequence.
    """Helm can forget a release before its terminating GPU pods stop reserving the node."""
    events: list[str] = []
    pod = Pod(metadata=PodMetadata(name="gpu-pod", uid="gpu-pod-uid"))
    pod_answers = iter([[pod], []])
    config = command_utils.ExecuteTrainConfig(
        cluster_backend=ClusterBackend.KUBERNETES,
        namespace="ci",
        run_id="run-baseline",
    )

    monkeypatch.setattr(
        app_module.Helm,
        "uninstall_if_present",
        lambda *, release, namespace: events.append(f"uninstall:{namespace}/{release}"),
    )
    monkeypatch.setattr(app_module.Helm, "get_manifest", lambda release, namespace: None)
    monkeypatch.setattr(
        app_module,
        "selected_pods",
        lambda namespace, selector: events.append(f"pods:{namespace}/{selector}") or next(pod_answers),
    )
    monkeypatch.setattr(app_module.time, "sleep", lambda seconds: events.append(f"sleep:{seconds}"))

    app_module._release_comparison_side(_request(config))

    assert events == [
        "uninstall:ci/miles-run-run-baseline-all",
        "pods:ci/app.kubernetes.io/instance=miles-run-run-baseline-all",
        "sleep:1.0",
        "pods:ci/app.kubernetes.io/instance=miles-run-run-baseline-all",
    ]


def test_kubernetes_side_fails_after_a_bounded_wait_for_terminating_pods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Sets the release timeout to zero seconds and makes the pod listing always
    # return one pod. Asserts releasing the side raises a timeout error naming the stuck pod.
    """A stuck terminating pod fails the handoff instead of polling forever or starting the target."""
    pod = Pod(metadata=PodMetadata(name="stuck-gpu-pod", uid="stuck-gpu-pod-uid"))
    config = command_utils.ExecuteTrainConfig(
        cluster_backend=ClusterBackend.KUBERNETES,
        namespace="ci",
        run_id="run-baseline",
    )

    monkeypatch.setattr(app_module, "_RELEASE_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(app_module.Helm, "uninstall_if_present", lambda **_kwargs: None)
    monkeypatch.setattr(app_module.Helm, "get_manifest", lambda release, namespace: None)
    monkeypatch.setattr(app_module, "selected_pods", lambda namespace, selector: [pod])

    with pytest.raises(TimeoutError, match="stuck-gpu-pod"):
        app_module._release_comparison_side(_request(config))


def test_ray_side_never_calls_kubernetes_release_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Makes the helm uninstall fail the test if called; releases a Ray-backend
    # side; passing means no helm call occurred.
    """Ray owns no Helm release, so the comparison handoff has nothing cluster-side to remove."""
    config = command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)
    monkeypatch.setattr(
        app_module.Helm,
        "uninstall_if_present",
        lambda **_kwargs: pytest.fail("Ray comparison side touched Helm"),
    )

    app_module._release_comparison_side(_request(config))


def test_a_fixed_topology_generate_data_runs_without_a_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Builds a mode with one rollout engine and one gpu per engine, stubs
    # preparation, common train args and training, and creates the comparison command-line app
    # with a mode resolver that ignores its input. Invokes the generate-data subcommand with one
    # step and no mode option through a CLI test runner; asserts exit code zero.
    """A scenario whose topology is fixed asks its user for no --mode on any of its subcommands."""
    mode = dataclasses.replace(_mode_fixture(), rollout_num_engines=1, rollout_gpus_per_engine=1)
    monkeypatch.setattr(app_module, "prepare", lambda _mode: None)
    monkeypatch.setattr(app_module, "get_common_train_args", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(app_module, "run_training", lambda **_kwargs: None)
    app, _ = app_module.create_comparison_app_and_run_ci(
        test_name="scenario_x",
        build_baseline_args=lambda *_args: "",
        build_target_args=lambda *_args: "",
        compare_fn=lambda *_args: None,
        resolve_mode_fn=lambda _mode: mode,
    )

    result = CliRunner().invoke(app, ["generate-data", "--num-steps", "1"])

    assert result.exit_code == 0, result.output


def _mode_fixture() -> FTTestMode:
    # [FUNCTION DOC] HELPER: Returns a minimal one-cell demo mode with empty parallel args.
    return FTTestMode(
        model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
    )


def _request(config: command_utils.ExecuteTrainConfig) -> RunSideRequest:
    # [FUNCTION DOC] HELPER: Builds a baseline-side run request with the demo mode, empty train
    # args, a fixed dump dir, the given config, and the dumper enabled.
    return RunSideRequest(
        side="baseline",
        mode=_mode_fixture(),
        train_args="",
        dump_dir="/dumps/baseline",
        config=config,
        enable_dumper=True,
    )
