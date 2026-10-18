# [MODULE DOC]
# Regression tests that each ft soak scenario (random crash and realistic gsm8k) creates exactly
# one default train config per run and uses that same object for everything: dump directory
# resolution, preparation, the fault injector's api-server host lookup, and training. Otherwise
# the fault injector would address a release that was never installed. The default config factory
# is replaced by one that hands out sentinel run ids, the config's backend factory is replaced by a
# recording backend, and every heavy step of the scenarios is stubbed. A shared record object
# collects which configs were created, prepared, asked for a host, and trained with.
import dataclasses
import itertools
from pathlib import Path

from tests.e2e.ft.conftest_ft import scenario_random_crash, scenario_realistic_gsm8k
from tests.e2e.ft.conftest_ft.fault_injection import state

from miles.utils.external_utils import command_utils


@dataclasses.dataclass
class _Seen:
    # [CLASS DOC] HELPER: Record of configs observed, in four lists: created by the default
    # factory, prepared, used to look up the api server host, and used for training.
    created: list[command_utils.ExecuteTrainConfig] = dataclasses.field(default_factory=list)
    prepared: list[command_utils.ExecuteTrainConfig] = dataclasses.field(default_factory=list)
    asked_for_host: list[command_utils.ExecuteTrainConfig] = dataclasses.field(default_factory=list)
    trained: list[command_utils.ExecuteTrainConfig] = dataclasses.field(default_factory=list)


class _RecordingBackend:
    # [CLASS DOC] HELPER: Fake execution backend bound to one config and a shared record. Reports
    # a host name derived from the run id and records which config trained.
    def __init__(self, config: command_utils.ExecuteTrainConfig, seen: _Seen) -> None:
        # [FUNCTION DOC] HELPER: Stores the bound config and the shared record.
        self.config = config
        self._seen = seen

    def api_server_host(self, config: command_utils.ExecuteTrainConfig) -> str:
        # [FUNCTION DOC] HELPER: Records the config passed in and returns an orchestrator host
        # name built from its run id.
        self._seen.asked_for_host.append(config)
        return f"orchestrator-of-{config.run_id}"

    def execute_train(self, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: Records the backend's bound config (not the arguments) as trained.
        self._seen.trained.append(self.config)


class _StubInjector:
    # [CLASS DOC] HELPER: Fake fault injector handle with an empty event log and a no-op stop-and-join.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates an empty event log.
        self.event_log = state.EventLog()

    def stop_and_join(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        pass


def _install(monkeypatch, seen: _Seen) -> None:
    # [FUNCTION DOC] HELPER: Patches the default config factory to produce configs with run ids
    # sentinel-0, sentinel-1, ... in the test namespace (recording each into the created list), and
    # patches the config's backend factory to return a recording backend bound to the config.
    run_ids = itertools.count()

    def fake_default_config() -> command_utils.ExecuteTrainConfig:
        # [FUNCTION DOC] HELPER: Creates the next sentinel config, records it as created, returns it.
        config = command_utils.ExecuteTrainConfig(run_id=f"sentinel-{next(run_ids)}", namespace="miles-e2e")
        seen.created.append(config)
        return config

    monkeypatch.setattr(command_utils, "default_config", fake_default_config)
    monkeypatch.setattr(
        command_utils.ExecuteTrainConfig, "create_backend", lambda self: _RecordingBackend(self, seen), raising=True
    )


class TestOneConfigPerSoak:
    # [CLASS DOC] Tests that each soak scenario creates and reuses a single config.
    def test_the_random_soak_builds_one_config_and_aims_every_step_at_it(self, monkeypatch, tmp_path: Path) -> None:
        # [FUNCTION DOC] Stubs dump dir resolution (recording the run id it gets), preparation
        # (recording the config), debug rollout data, common and ft args, the injector spawn and
        # the healing assertion. Runs the random-crash scenario for a fake-rollout trainer-kill
        # mode with one step. Asserts exactly one config was created (sentinel-0), the dump dir was
        # resolved with that run id, and preparation, host lookup and training each saw that very
        # config object exactly once.
        """Regression: a second default_config gave the injector a run_id no release was ever installed under."""
        seen = _Seen()
        _install(monkeypatch, seen)
        dump_run_ids: list[str] = []
        monkeypatch.setattr(
            scenario_random_crash,
            "resolve_dump_dir",
            lambda test_name, *, run_id: dump_run_ids.append(run_id) or str(tmp_path / "dump"),
        )
        monkeypatch.setattr(scenario_random_crash, "prepare", lambda mode, *, config: seen.prepared.append(config))
        monkeypatch.setattr(
            scenario_random_crash, "materialize_cyclic_debug_rollout_data", lambda count: str(tmp_path / "rollout")
        )
        monkeypatch.setattr(scenario_random_crash, "get_common_train_args", lambda mode, **kwargs: "")
        monkeypatch.setattr(scenario_random_crash, "get_ft_args", lambda mode: "")
        monkeypatch.setattr(scenario_random_crash, "spawn_fault_injector", lambda **kwargs: _StubInjector())
        monkeypatch.setattr(scenario_random_crash, "assert_healing", lambda ft_components, **kwargs: None)

        scenario_random_crash.run_ci("kill_train__dp4_cp2__fake_rollout__moe_5layer", num_steps=1)

        assert [config.run_id for config in seen.created] == ["sentinel-0"]
        assert dump_run_ids == ["sentinel-0"]
        assert [config is seen.created[0] for config in seen.prepared] == [True]
        assert [config is seen.created[0] for config in seen.asked_for_host] == [True]
        assert [config is seen.created[0] for config in seen.trained] == [True]

    def test_the_gsm8k_soak_builds_one_config_and_aims_every_step_at_it(self, monkeypatch, tmp_path: Path) -> None:
        # [FUNCTION DOC] Sets dummy proxy environment variables, stubs the gsm8k scenario's backend
        # creation (recording the config as prepared and returning a recording backend), dataset
        # preparation, dump dir resolution, injector spawn and healing assertion, then runs one
        # rollout. Asserts one created config, the dump dir resolved with its run id, the backend
        # creation seen twice with that same config object, and host lookup and training each
        # seeing it once.
        """The same bug here would point the injector at one release while training ran under another."""
        seen = _Seen()
        _install(monkeypatch, seen)
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            monkeypatch.setenv(proxy_var, "http://unused")
        monkeypatch.setattr(
            scenario_realistic_gsm8k,
            "create_backend_for_run",
            lambda config: seen.prepared.append(config) or _RecordingBackend(config, seen),
        )
        monkeypatch.setattr(scenario_realistic_gsm8k, "prepare_gsm8k", lambda U: None)
        dump_run_ids: list[str] = []
        monkeypatch.setattr(
            scenario_realistic_gsm8k,
            "resolve_dump_dir",
            lambda test_name, *, run_id: dump_run_ids.append(run_id) or str(tmp_path / "gsm8k"),
        )
        monkeypatch.setattr(scenario_realistic_gsm8k, "spawn_fault_injector", lambda **kwargs: _StubInjector())
        monkeypatch.setattr(scenario_realistic_gsm8k, "assert_healing", lambda ft_components, **kwargs: None)

        scenario_realistic_gsm8k.run_ci(num_rollout=1)

        assert [config.run_id for config in seen.created] == ["sentinel-0"]
        assert dump_run_ids == ["sentinel-0"]
        assert [config is seen.created[0] for config in seen.prepared] == [True, True]
        assert [config is seen.created[0] for config in seen.asked_for_host] == [True]
        assert [config is seen.created[0] for config in seen.trained] == [True]


class TestRelaunchingASoak:
    # [CLASS DOC] Tests for relaunching a soak with a modified config.
    def test_a_relaunch_is_installed_under_the_config_it_was_handed(self, monkeypatch) -> None:
        # [FUNCTION DOC] Builds a copy of the default config with hot restart set to orchestration,
        # launches the gsm8k scenario's launch function with it (backend creation stubbed to a
        # recording backend), and asserts the trained config carries the hot restart setting.
        """A hot restart relaunches the soak with --hot-restart added, and the backend reads that off its config."""
        seen = _Seen()
        _install(monkeypatch, seen)
        monkeypatch.setattr(
            scenario_realistic_gsm8k, "create_backend_for_run", lambda config: _RecordingBackend(config, seen)
        )
        relaunch = dataclasses.replace(command_utils.default_config(), hot_restart="orchestration")

        scenario_realistic_gsm8k._launch_gsm8k(relaunch, train_args="", fully_async=False)

        assert [config.hot_restart for config in seen.trained] == ["orchestration"]
