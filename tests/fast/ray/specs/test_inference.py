# [MODULE DOC]
# Unit tests for the inference-side worker specs: engine pools, router pools, the session server,
# the inference controller and the registration reporter. All tests are CPU only. They build
# argument namespaces and sglang config files, produce the specs, and check scheduling shape
# (cells, workers, gpu slots, head pinning), port schemas, rendered launch commands, per-cell
# metadata, environment variables, pool naming and the controller constructor wiring. Launch
# command rendering of the engine is intercepted by replacing the command builder with a recorder.
from __future__ import annotations

import asyncio
import shlex
import socket
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.backends.sglang_utils.router_args_utils import parse_router_args_argv
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, resolve_sglang_config
from miles.ray.rollout import external_engine_provider as external_engine_provider_module
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.specs import inference as inference_specs
from miles.ray.specs.inference import (
    INFERENCE_CONTROLLER_POOL_ID,
    INFERENCE_CONTROLLER_WORKER_CLASS,
    _compute_router_primary_port_info,
    _compute_session_server_primary_port_info,
    _compute_spec_router,
    compute_engine_pool_id,
    compute_engine_pool_ids,
    compute_inference_controller_provider,
    compute_inference_engine_env_vars,
    compute_router_pool_id,
    inference_controller_worker_name,
    spec_inference_controller,
    spec_session_server,
    specs_inference_engine,
    specs_inference_registration_reporter,
    specs_router,
)
from miles.rollout.session.config import SessionServerConfig
from miles.router.config import MilesRouterConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import SECTION_OF_CATEGORY, LaunchPlan
from miles.utils.function_registry import load_function
from miles.utils.workers.argv_utils import parse_config_argv
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.types import PlatformAccess
from miles.utils.workers.worker_provider.static import StaticWorkerProvider
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    HostAndPort,
    LaunchCommandContext,
    WorkerCtorContext,
    WorkerMetaContext,
)


def _controller_layout() -> LaunchPlan:
    # [FUNCTION DOC] HELPER: Returns a fixed launch plan (run id, state file path, release name,
    # namespace, orchestrator command and worker argv) used to render specs into helm values.
    return LaunchPlan(
        run_id="260101-000000-000",
        state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
        release="miles-run-260101",
        namespace="rl",
        orchestrator_command=["python", "/repo/train.py"],
        worker_argv=["--rollout-num-gpus", "8"],
    )


def _make_model_cfg(*worker_types: str) -> ModelConfig:
    # [FUNCTION DOC] HELPER: Builds a model config named default with one server group per given
    # worker type. Each group has 4 gpus and 4 gpus per engine, a gpu offset of 4 times its position,
    # an engine offset equal to its position, and no offload; weight updates are enabled.
    groups = [
        ServerGroupConfig(
            worker_type=worker_type,
            num_gpus=4,
            num_gpus_per_engine=4,
            gpu_offset=group_index * 4,
            engine_offset=group_index,
            needs_offload=False,
        )
        for group_index, worker_type in enumerate(worker_types)
    ]
    return ModelConfig(name="default", model_path=None, server_groups=groups, update_weights=True)


def _make_router_ctx(
    *, host: str = "127.0.0.1", port: int = 20000, prometheus_port: int = 4001
) -> LaunchCommandContext:
    # [FUNCTION DOC] HELPER: Builds a launch command context for cell 0, worker 0 of a router, whose
    # own addresses are a primary and a prometheus endpoint on the given host and ports. The pool
    # address map and the gpu id lists are empty.
    return LaunchCommandContext(
        cell_index=0,
        worker_in_cell_index=0,
        self_addrs=dict(
            primary=HostAndPort(host=host, port=port),
            prometheus=HostAndPort(host=host, port=prometheus_port),
        ),
        pool_addrs={},
        gpu_ids=[],
        local_gpu_ids=[],
    )


def _refuse_lookup(*_args, **_kwargs):
    # [FUNCTION DOC] HELPER: Stand-in for the socket name lookup that accepts any arguments and
    # always raises an assertion error, proving that no lookup happens for hosts needing none.
    raise AssertionError("a name lookup ran for a host that needs none")


class TestRouterPortPinning:
    # [CLASS DOC] Tests how the router primary port is chosen: unpinned means a preferred port 8000
    # that may move, pinned means a fixed port, and each model is pinned a port apart.
    def test_an_unpinned_router_may_move_off_its_preferred_port(self):
        # [FUNCTION DOC] With no configured router port, the primary port info is the static
        # preferred port 8000 and dynamic allocation is allowed.
        """Nothing outside the job needs to name it, so a busy 8000 must not fail the launch."""
        port_info = _compute_router_primary_port_info(make_args(sglang_router_port=None), model_idx=0)

        assert (port_info.static_port, port_info.allow_dynamic) == (8000, True)

    def test_a_pinned_router_stays_on_the_port_it_was_given(self):
        # [FUNCTION DOC] With a configured router port 31000 for model 0, the static port is that
        # value and dynamic allocation is disallowed.
        """Launchers pin it so a firewall rule or a dial-back host can name the port in advance;
        drifting off it would leave those pointing at nothing."""
        port_info = _compute_router_primary_port_info(make_args(sglang_router_port=31000), model_idx=0)

        assert (port_info.static_port, port_info.allow_dynamic) == (31000, False)

    def test_each_models_router_is_pinned_a_port_apart(self):
        # [FUNCTION DOC] With a pinned port 31000, model 0 and model 1 get static ports 31000 and
        # 31001, i.e. the port is shifted by the model index.
        """Two models pinned to one port would race for the same socket."""
        ports = [
            _compute_router_primary_port_info(make_args(sglang_router_port=31000), model_idx=i).static_port
            for i in range(2)
        ]

        assert ports == [31000, 31001]


class TestComputeSpecRouterLaunchCommand:
    # [CLASS DOC] Tests the launch command rendered by the router spec: miles router versus sgl
    # router selection, prefill/decode (PD) mode, host binding rules and argument round trips. The
    # command string is split with shell rules and parsed back with the router argument parsers.
    def test_pd_disagg_with_miles_router_asserts(self):
        # [FUNCTION DOC] A spec for a model with prefill and decode groups while the miles router is
        # selected must raise an assertion error mentioning missing PD support when its launch
        # command is rendered.
        """Rendering a miles-router launch command for a PD-disaggregated model must fail fast."""
        args = make_args(use_miles_router=True, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("prefill", "decode"))
        with pytest.raises(AssertionError, match="miles router does not support PD"):
            spec.launch_command(_make_router_ctx())

    def test_the_external_pd_flag_launches_a_pd_router_in_front_of_regular_groups(self):
        # [FUNCTION DOC] With external rollout and the external router PD flag on, a model made only
        # of regular groups still gets an sgl router launched in PD mode: the parsed router arguments
        # (after the three leading interpreter and module tokens) have PD disaggregation enabled.
        """External PD is discovered after the router is already up, so this flag is the only thing
        that can put the router in PD mode; a router built without it misroutes every request while
        discovery still reports a healthy prefill/decode fleet."""
        args = make_args(
            use_miles_router=False,
            sglang_router_ip=None,
            sglang_router_port=None,
            rollout_external=True,
            rollout_external_router_pd=True,
        )
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        argv = shlex.split(spec.launch_command(_make_router_ctx()))

        assert parse_router_args_argv(argv[3:]).pd_disaggregation is True

    def test_without_the_external_pd_flag_a_regular_model_keeps_a_regular_router(self):
        # [FUNCTION DOC] Without the external PD flag, a regular-only model gets an sgl router whose
        # parsed arguments have PD disaggregation disabled.
        """Every internal run takes this path, and the new flag defaults to off."""
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        argv = shlex.split(spec.launch_command(_make_router_ctx()))

        assert parse_router_args_argv(argv[3:]).pd_disaggregation is False

    def test_the_external_pd_flag_is_rejected_by_the_miles_router(self):
        # [FUNCTION DOC] With the miles router and the external PD flag on a regular-only model,
        # rendering the launch command raises the same "no PD support" assertion error.
        """The miles router cannot serve PD at all, so the external flag must hit the same guard."""
        args = make_args(
            use_miles_router=True,
            sglang_router_ip=None,
            sglang_router_port=None,
            rollout_external=True,
            rollout_external_router_pd=True,
        )
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        with pytest.raises(AssertionError, match="miles router does not support PD"):
            spec.launch_command(_make_router_ctx())

    def test_sgl_router_launches_the_native_cli(self):
        # [FUNCTION DOC] The rendered command starts the current interpreter with the module option
        # and the upstream sglang router launch module, and carries the port and prometheus port taken
        # from the launch context's own addresses (20000 and 4001).
        """The sgl router runs as the upstream CLI with the addresses from the launch context."""
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))
        argv = shlex.split(spec.launch_command(_make_router_ctx()))
        assert argv[0] == sys.executable
        assert argv[1:3] == ["-m", "sglang_router.launch_router"]
        assert argv[argv.index("--port") + 1] == "20000"
        assert argv[argv.index("--prometheus-port") + 1] == "4001"

    def test_sgl_router_binds_the_ip_a_placed_hostname_resolves_to(self, monkeypatch):
        # [FUNCTION DOC] The socket address lookup is replaced so that any host resolves to 10.0.0.7.
        # With a context host given as a name, the sgl router command must carry the resolved ip as
        # its host argument, because the sgl router parses the host as a socket address.
        """The sgl router parses --host as a socket address, so a ray node named by hostname must reach it as an ip."""
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 0))],
        )
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        argv = shlex.split(spec.launch_command(_make_router_ctx(host="node1234")))

        assert parse_router_args_argv(argv[3:]).host == "10.0.0.7"

    def test_sgl_router_keeps_a_wildcard_bind_host(self, monkeypatch):
        # [FUNCTION DOC] With name lookup forbidden, a context host of 0.0.0.0 must pass through
        # unchanged into the sgl router host argument, with no lookup attempted.
        """Kubernetes places every command worker on 0.0.0.0, which must pass through without a lookup."""
        monkeypatch.setattr(socket, "getaddrinfo", _refuse_lookup)
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        argv = shlex.split(spec.launch_command(_make_router_ctx(host="0.0.0.0")))

        assert parse_router_args_argv(argv[3:]).host == "0.0.0.0"

    def test_miles_router_keeps_the_placed_hostname(self, monkeypatch):
        # [FUNCTION DOC] With name lookup forbidden, the miles router command keeps the placed
        # hostname verbatim: the parsed miles router config has that name as its host.
        """The miles router binds through uvicorn, which resolves names itself, so its host is left alone."""
        monkeypatch.setattr(socket, "getaddrinfo", _refuse_lookup)
        args = make_args(use_miles_router=True, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        argv = shlex.split(spec.launch_command(_make_router_ctx(host="node1234")))

        assert parse_config_argv(MilesRouterConfig, argv[3:]).host == "node1234"

    def test_sgl_router_launch_preserves_prefixed_raw_inputs(self):
        # [FUNCTION DOC] Router arguments given with the router prefix (tls certificate path, a
        # prefill url list, a selector list) on a prefill plus decode model survive rendering and
        # parsing: the cert path, the prefill url with an integer port, the selector mapping and PD
        # mode are all recovered from the command.
        """Raw --router-* aliases and collections survive the full launch-command path."""
        args = make_args(
            use_miles_router=False,
            sglang_router_ip=None,
            sglang_router_port=None,
            router_tls_cert_path="/certs/server.pem",
            router_prefill=[["http://prefill.invalid", "9000"]],
            router_selector=["app=sglang", "role=prefill"],
        )
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("prefill", "decode"))
        argv = shlex.split(spec.launch_command(_make_router_ctx()))
        parsed = parse_router_args_argv(argv[3:])

        assert parsed.server_cert_path == "/certs/server.pem"
        assert parsed.prefill_urls == [("http://prefill.invalid", 9000)]
        assert parsed.selector == {"app": "sglang", "role": "prefill"}
        assert parsed.pd_disaggregation is True

    def test_miles_router_launches_with_a_parseable_config(self):
        # [FUNCTION DOC] The miles router command starts the interpreter on the miles router module
        # and its remaining arguments parse back into a router config whose host, port and max
        # connections match the context (127.0.0.1, 20000) and the argument (100).
        """The miles router command's config payload parses back losslessly."""
        args = make_args(
            use_miles_router=True,
            sglang_router_ip=None,
            sglang_router_port=None,
            miles_router_max_connections=100,
            miles_router_timeout=None,
            miles_router_health_check_failure_threshold=3,
            rollout_health_check_interval=10.0,
        )
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))
        argv = shlex.split(spec.launch_command(_make_router_ctx()))
        assert argv[:3] == [sys.executable, "-m", "miles.router.router"]
        config = parse_config_argv(MilesRouterConfig, argv[3:])
        assert config.host == "127.0.0.1"
        assert config.port == 20000
        assert config.max_connections == 100


class TestComputeSpecSessionServer:
    # [CLASS DOC] Tests the session server spec: launch command content, cpu reservation and the
    # number of cells under the enable and train-only switches.
    def test_launch_command_wires_the_router_backend_and_roundtrips(self):
        # [FUNCTION DOC] With two session server workers, the spec has two cells. For cell 1 with a
        # router pool address in the context, the rendered command runs the session server module and
        # its parsed config has the backend url of that router, the context host and port, and an
        # instance id made of the run uuid and the cell index.
        """The session server command targets the router addr from pool_addrs and its config parses back losslessly."""
        args = make_args(
            use_session_server="v1",
            hf_checkpoint="/fake/model",
            rollout_top_logprobs_num=0,
            rollout_sampling_logprobs_mode="selected",
            session_server_workers=2,
            sglang_router_ip=None,
            sglang_router_port=None,
            miles_router_timeout=None,
            chat_template_path=None,
            tito_model="default",
            apply_chat_template_kwargs=None,
            use_rollout_indexer_replay=False,
            sglang_speculative_algorithm=None,
            num_layers=None,
            moe_router_topk=None,
            save_debug_trajectory_data=None,
            lora_rank=0,
            lora_adapter_path=None,
        )
        spec = spec_session_server(args)
        assert spec.scheduling.num_cells == 2

        ctx = LaunchCommandContext(
            cell_index=1,
            worker_in_cell_index=0,
            self_addrs=dict(primary=HostAndPort(host="127.0.0.1", port=5006)),
            pool_addrs={compute_router_pool_id(0): [dict(primary=HostAndPort(host="127.0.0.1", port=3000))]},
            gpu_ids=[],
            local_gpu_ids=[],
        )
        argv = shlex.split(spec.launch_command(ctx))

        assert argv[:3] == [sys.executable, "-m", "miles.rollout.session.server"]
        config = parse_config_argv(SessionServerConfig, argv[3:])
        assert config.backend_url == "http://127.0.0.1:3000"
        assert config.host == "127.0.0.1"
        assert config.port == 5006
        assert config.instance_id == f"{args.run_uuid}-1"

    def test_it_reserves_no_cpu_on_the_head_node(self):
        # [FUNCTION DOC] The session server spec asks for zero cpus per worker.
        """When pinned to the head, a CPU reservation would leave it pending forever on a head started with --num-cpus=0."""
        spec = spec_session_server(_make_session_server_args())

        assert spec.scheduling.num_cpus_per_worker == 0

    def test_disabled_schedules_zero_cells(self):
        # [FUNCTION DOC] With the session server switched off, the spec schedules zero cells.
        """Disabling the session server removes its cells instead of launching idle servers."""
        args = make_args(use_session_server=False)
        assert spec_session_server(args).scheduling.num_cells == 0

    def test_debug_train_only_schedules_zero_cells(self):
        # [FUNCTION DOC] With the train-only debug flag on, an otherwise enabled session server
        # schedules zero cells because no router exists to point at.
        """Its launch command reads the router address, and --debug-train-only leaves the router unlaunched."""
        args = _make_session_server_args(debug_train_only=True)

        assert spec_session_server(args).scheduling.num_cells == 0

    def test_only_the_debug_train_only_flag_empties_an_enabled_session_server(self):
        # [FUNCTION DOC] Builds the spec with the train-only flag off and on and checks the cell
        # counts are 2 and 0 respectively, so that flag alone changes the count.
        """An enabled session server keeps every requested cell until --debug-train-only takes the router away."""
        cells = {
            debug_train_only: spec_session_server(
                _make_session_server_args(debug_train_only=debug_train_only)
            ).scheduling.num_cells
            for debug_train_only in (False, True)
        }

        assert cells == {False: 2, True: 0}


def _make_session_server_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a test argument namespace with the session server enabled
    # (version v1, 2 workers, no router timeout, no chat template, default tito model, no template
    # kwargs, no lora adapter) and applies the given overrides on top. Returns the namespace.
    return make_args(
        use_session_server="v1",
        session_server_workers=2,
        miles_router_timeout=None,
        chat_template_path=None,
        tito_model="default",
        apply_chat_template_kwargs=None,
        lora_adapter_path=None,
        **overrides,
    )


class TestSessionServerAddressPinning:
    # [CLASS DOC] Tests how the session server primary port and advertised host are determined
    # from configuration.
    def test_an_unpinned_session_server_may_move_off_its_preferred_port(self):
        # [FUNCTION DOC] With no session server port configured, the port info is static 8000, may
        # move dynamically, and is not shifted per cell.
        """Nothing outside the job names it, so a busy 8000 must not fail the launch nor be shifted per cell."""
        port_info = _compute_session_server_primary_port_info(make_args(session_server_port=None))

        assert (port_info.static_port, port_info.allow_dynamic, port_info.offset_by_cell) == (8000, True, False)

    def test_pinned_session_servers_take_consecutive_ports_from_the_configured_one(self):
        # [FUNCTION DOC] With a configured port 5100, the port info is static 5100, not dynamic, and
        # shifted by the cell index so each cell gets a consecutive port.
        """A pinned port must stay put and be shifted per cell, otherwise every session server races for one socket."""
        port_info = _compute_session_server_primary_port_info(make_args(session_server_port=5100))

        assert (port_info.static_port, port_info.allow_dynamic, port_info.offset_by_cell) == (5100, False, True)

    def test_a_configured_session_server_ip_overrides_the_allocated_host(self):
        # [FUNCTION DOC] With a configured session server ip, the parsed config of the rendered
        # command carries that ip as host instead of the context's allocated host, while the port
        # still comes from the context (5006).
        """Operators pin the advertised ip so clients can reach it; binding the ray node ip instead ignores them."""
        args = _make_session_server_args(session_server_ip="10.20.30.40")
        spec = spec_session_server(args)
        ctx = LaunchCommandContext(
            cell_index=0,
            worker_in_cell_index=0,
            self_addrs=dict(primary=HostAndPort(host="127.0.0.1", port=5006)),
            pool_addrs={compute_router_pool_id(0): [dict(primary=HostAndPort(host="127.0.0.1", port=3000))]},
            gpu_ids=[],
            local_gpu_ids=[],
        )

        config = parse_config_argv(SessionServerConfig, shlex.split(spec.launch_command(ctx))[3:])

        assert config.host == "10.20.30.40"
        assert config.port == 5006


class TestSessionServerInterpreterFlags:
    # [CLASS DOC] Tests that the session server command reuses the interpreter flags of the
    # process that built the spec, by faking the original process argument vector.
    def test_the_launch_command_carries_the_parent_interpreter_flags(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With a faked original argument vector carrying the optimize flag and an
        # extra option, the rendered command starts with the interpreter, those flags, the module
        # option and the session server module; the remaining arguments parse to port 5006.
        """A session server launched by a bare interpreter runs with different semantics than its own job."""
        monkeypatch.setattr(
            sys,
            "orig_argv",
            [sys.executable, "-O", "-X", "faulthandler", "-m", "miles.train", "--config", "x.yaml"],
        )
        spec = spec_session_server(_make_session_server_args())
        ctx = LaunchCommandContext(
            cell_index=0,
            worker_in_cell_index=0,
            self_addrs=dict(primary=HostAndPort(host="127.0.0.1", port=5006)),
            pool_addrs={compute_router_pool_id(0): [dict(primary=HostAndPort(host="127.0.0.1", port=3000))]},
            gpu_ids=[],
            local_gpu_ids=[],
        )

        argv = shlex.split(spec.launch_command(ctx))

        assert argv[:6] == [sys.executable, "-O", "-X", "faulthandler", "-m", "miles.rollout.session.server"]
        assert parse_config_argv(SessionServerConfig, argv[6:]).port == 5006

    def test_the_flags_are_captured_when_the_spec_is_built(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The original argument vector is faked while the spec is built, then changed
        # to a different one before the command is rendered. The command must still carry the flags
        # of the first vector, i.e. they are captured at build time.
        """The command is rendered inside the worker manager actor, whose own argv is Ray's worker script."""
        monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-O", "-m", "miles.train"])
        spec = spec_session_server(_make_session_server_args())
        monkeypatch.setattr(sys, "orig_argv", [sys.executable, "/ray/workers/setup_worker.py"])
        ctx = LaunchCommandContext(
            cell_index=0,
            worker_in_cell_index=0,
            self_addrs=dict(primary=HostAndPort(host="127.0.0.1", port=5006)),
            pool_addrs={compute_router_pool_id(0): [dict(primary=HostAndPort(host="127.0.0.1", port=3000))]},
            gpu_ids=[],
            local_gpu_ids=[],
        )

        argv = shlex.split(spec.launch_command(ctx))

        assert argv[:4] == [sys.executable, "-O", "-m", "miles.rollout.session.server"]


class TestSessionServerRouterPoolLookup:
    # [CLASS DOC] Tests how the session server command finds its backend router address in the
    # per-pool address map of the launch context.
    def _make_ctx(self, pool_addrs: dict) -> LaunchCommandContext:
        # [FUNCTION DOC] HELPER: Builds a context for cell 0, worker 0 with a session server own
        # primary address on 127.0.0.1 port 5006 and the given pool address map.
        return LaunchCommandContext(
            cell_index=0,
            worker_in_cell_index=0,
            self_addrs=dict(primary=HostAndPort(host="127.0.0.1", port=5006)),
            pool_addrs=pool_addrs,
            gpu_ids=[],
            local_gpu_ids=[],
        )

    def _make_args(self) -> Namespace:
        # [FUNCTION DOC] HELPER: Session server test arguments with no configured router ip or port.
        return _make_session_server_args(sglang_router_ip=None, sglang_router_port=None)

    def test_the_backend_is_read_under_the_router_specs_own_pool_id(self):
        # [FUNCTION DOC] The pool address map is keyed by the name of the router spec built for
        # model 0; the session server's parsed backend url must be the address stored under it.
        """The key the session server looks up is exactly the name the router pool is registered under."""
        args = self._make_args()
        router_spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))
        ctx = self._make_ctx({router_spec.name: [dict(primary=HostAndPort(host="10.0.0.2", port=3210))]})

        config = parse_config_argv(SessionServerConfig, shlex.split(spec_session_server(args).launch_command(ctx))[3:])

        assert config.backend_url == "http://10.0.0.2:3210"

    def test_an_address_map_keyed_by_worker_names_fails_loudly(self):
        # [FUNCTION DOC] An address map keyed by a worker name (pool id plus cell and worker
        # suffixes) rather than the pool id makes rendering the command raise a key error.
        """The map is keyed per pool, not per worker, so a worker-keyed map must raise instead of launching unwired."""
        args = self._make_args()
        ctx = self._make_ctx(
            {f"{compute_router_pool_id(0)}-0-0": [dict(primary=HostAndPort(host="10.0.0.2", port=3210))]}
        )

        with pytest.raises(KeyError):
            spec_session_server(args).launch_command(ctx)

    def test_another_models_router_pool_is_not_mistaken_for_the_first(self):
        # [FUNCTION DOC] With router pools for model 1 and model 0 both present, the backend url in
        # the parsed config is the one of model 0's router.
        """With several router pools present the session server must still target model 0's router."""
        args = self._make_args()
        ctx = self._make_ctx(
            {
                compute_router_pool_id(1): [dict(primary=HostAndPort(host="10.0.0.3", port=3211))],
                compute_router_pool_id(0): [dict(primary=HostAndPort(host="10.0.0.2", port=3210))],
            }
        )

        config = parse_config_argv(SessionServerConfig, shlex.split(spec_session_server(args).launch_command(ctx))[3:])

        assert config.backend_url == "http://10.0.0.2:3210"


class TestInferenceEngineEnvVars:
    # [CLASS DOC] Tests the environment variables computed for inference engine processes:
    # dumper integration, forced sampler logprob setting, process-level overrides of built-in
    # defaults and the custom all-reduce switch.
    def test_an_enabled_dumper_resolves_the_sglang_environment(self, monkeypatch) -> None:
        # [FUNCTION DOC] With the inference dumper enabled by argument, a fake dumper module is
        # installed (both as an attribute of the utils package and in the module table) whose
        # environment getter records its call and returns three variables. Computing the engine
        # environment must call that getter once with the same args, keep the dumper variables, and
        # force the original-logprob variable to "0" even though the fake returned "1".
        """Enabling inference dumping must load the dumper integration and render its startup environment."""
        args = make_args(dumper_inference=["enable=true", "non_intrusive_mode=all"])
        get_sglang_env_calls: list[Namespace] = []

        def get_sglang_env(call_args: Namespace) -> dict[str, str]:
            # [FUNCTION DOC] HELPER: Fake dumper environment getter; records the received args in
            # the enclosing list and returns a fixed mapping of three dumper related variables.
            get_sglang_env_calls.append(call_args)
            return {
                "DUMPER_SERVER_PORT": "reuse",
                "DUMPER_NON_INTRUSIVE_MODE": "all",
                "SGLANG_RETURN_ORIGINAL_LOGPROB": "1",
            }

        fake_dumper_utils = SimpleNamespace(get_sglang_env=get_sglang_env)
        import miles.utils as miles_utils

        monkeypatch.setattr(miles_utils, "dumper_utils", fake_dumper_utils, raising=False)
        monkeypatch.setitem(
            sys.modules,
            "miles.utils.dumper_utils",
            fake_dumper_utils,
        )

        envs = compute_inference_engine_env_vars(args)

        assert get_sglang_env_calls == [args]
        assert envs["DUMPER_SERVER_PORT"] == "reuse"
        assert envs["DUMPER_NON_INTRUSIVE_MODE"] == "all"
        assert envs["SGLANG_RETURN_ORIGINAL_LOGPROB"] == "0"

    @pytest.mark.parametrize("loss_type", ["policy_loss", "score_centering"])
    def test_training_enforces_sampler_logprobs_on_workers(self, monkeypatch, loss_type):
        # [FUNCTION DOC] Parametrized over two loss types. Even when the process environment sets the
        # original-logprob variable to "1", the computed engine environment has it as "0".
        monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
        envs = compute_inference_engine_env_vars(make_args(loss_type=loss_type))
        assert envs["SGLANG_RETURN_ORIGINAL_LOGPROB"] == "0"

    def test_a_process_level_override_wins_over_the_built_in_default(self, monkeypatch):
        # [FUNCTION DOC] With the jit deepgemm precompile variable set to "true" and the memory saver
        # cuda graph variable set to "false" in the process environment, the computed engine
        # environment keeps those exact values instead of the built-in defaults.
        """The launcher's environment is how operators retune sglang per cluster, so defaults must not overwrite it."""
        monkeypatch.setenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "true")
        monkeypatch.setenv("SGLANG_MEMORY_SAVER_CUDA_GRAPH", "false")

        envs = compute_inference_engine_env_vars(make_args())

        assert envs["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "true"
        assert envs["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "false"

    def test_the_built_in_defaults_apply_without_a_process_override(self, monkeypatch):
        # [FUNCTION DOC] With both variables removed from the process environment, the computed
        # engine environment has the deepgemm precompile variable as "false" and the memory saver
        # cuda graph variable as "true".
        """Without an override the engine must still get miles' own safety values rather than sglang's."""
        monkeypatch.delenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", raising=False)
        monkeypatch.delenv("SGLANG_MEMORY_SAVER_CUDA_GRAPH", raising=False)

        envs = compute_inference_engine_env_vars(make_args())

        assert envs["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "false"
        assert envs["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "true"

    def test_custom_all_reduce_v2_is_disabled_only_for_colocated_multi_gpu_engines(self, monkeypatch):
        # [FUNCTION DOC] With the custom all-reduce v2 variable absent from the process environment,
        # the computed value is "0" only for a colocated run with 2 gpus per engine; it is "1" for a
        # colocated single-gpu engine and for a non-colocated 2-gpu engine.
        """Only a colocated engine spanning several gpus hits the v2 all-reduce conflict; disabling it elsewhere
        silently gives up throughput."""
        monkeypatch.delenv("SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2", raising=False)

        def _value_for(*, colocate: bool, num_gpus_per_engine: int) -> str:
            # [FUNCTION DOC] HELPER: Builds test arguments with the given colocation flag and gpus
            # per engine, and returns the computed custom all-reduce v2 environment value.
            args = make_args(colocate=colocate, rollout_num_gpus_per_engine=num_gpus_per_engine)
            return compute_inference_engine_env_vars(args)["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"]

        assert _value_for(colocate=True, num_gpus_per_engine=2) == "0"
        assert _value_for(colocate=True, num_gpus_per_engine=1) == "1"
        assert _value_for(colocate=False, num_gpus_per_engine=2) == "1"


class TestSpecsInferenceEngine:
    # [CLASS DOC] Tests which engine specs are produced from an sglang config file and the
    # placement group slot offsets they receive.
    def test_pg_slot_offsets_accumulate_and_placeholder_groups_keep_their_slots(self, tmp_path):
        # [FUNCTION DOC] Config with groups: regular 4 gpus at 2 per engine, placeholder 4 gpus,
        # decode 8 gpus at 4 per engine, total 16 rollout gpus. Expects two specs (the placeholder is
        # skipped) named with the engine pool pattern, slot offsets 0 and 8 (the placeholder still
        # occupies its 4 gpus), gpu slots per worker 2 and 4, and every spec in the rollout
        # placement group.
        """Group offsets follow the config order and a skipped placeholder group still occupies its gpu span."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2},
                    {"worker_type": "placeholder", "num_gpus": 4, "num_gpus_per_engine": 4},
                    {"worker_type": "decode", "num_gpus": 8, "num_gpus_per_engine": 4},
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=16)

        specs = specs_inference_engine(args)

        assert [spec.name for spec in specs] == ["inference-engine-all-0-0", "inference-engine-all-0-2"]
        assert [spec.scheduling.pg_slot_offset for spec in specs] == [0, 8]
        assert [spec.scheduling.num_gpu_slots_per_worker for spec in specs] == [2, 4]
        assert all(spec.scheduling.pg_name == "rollout" for spec in specs)

    def test_debug_train_only_produces_no_engine_spec(self, tmp_path):
        # [FUNCTION DOC] With a one-group config but the train-only debug flag and colocation, the
        # list of engine specs is empty.
        """In --debug-train-only the rollout placement group is the trainer's own gpus, so no engine may be specced."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=8,
            colocate=True,
            debug_train_only=True,
        )

        assert specs_inference_engine(args) == []

    def test_external_rollout_produces_no_engine_spec(self, tmp_path):
        # [FUNCTION DOC] With external rollout enabled and an external engine address, the list of
        # engine specs is empty.
        """Externally launched engines are the operator's to run, so miles must not spec its own."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=8,
            rollout_external=True,
            rollout_external_engine_addrs=["host1:8000"],
        )

        assert specs_inference_engine(args) == []


class TestSpecsRouter:
    # [CLASS DOC] Tests which router specs are produced from an sglang config, especially the
    # suppression of the router when the train-only debug flag is set.
    def test_one_router_spec_per_model_is_specced_by_default(self, tmp_path):
        # [FUNCTION DOC] A single-model config yields exactly one router spec named for model 0.
        """Rollout runs still get their router, one per model in the sglang config."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=8)

        assert [spec.name for spec in specs_router(args)] == ["inference-router-0"]

    def test_debug_train_only_produces_no_router_spec(self, tmp_path):
        # [FUNCTION DOC] With colocation and the train-only flag, no router spec is produced.
        """--debug-train-only starts no engines, so a router worker would only wait for workers that never come."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=8,
            colocate=True,
            debug_train_only=True,
        )

        assert specs_router(args) == []

    def test_flipping_only_the_debug_train_only_flag_removes_the_router(self, tmp_path):
        # [FUNCTION DOC] Builds router specs for identical arguments with the train-only flag off and
        # on; expects the router name list to be the single model 0 router versus empty.
        """One identical run specs a router with the flag off and none with it on, so nothing else decides it."""
        names = {
            debug_train_only: [
                spec.name for spec in specs_router(_make_router_args(tmp_path, debug_train_only=debug_train_only))
            ]
            for debug_train_only in (False, True)
        }

        assert names == {False: ["inference-router-0"], True: []}

    def test_debug_train_only_skips_the_router_however_the_router_itself_is_configured(self, tmp_path):
        # [FUNCTION DOC] Even with the miles router selected, a pinned router port and session
        # servers requested, the train-only flag yields no router specs.
        """Pinning a port, picking the miles router or asking for session servers must not resurrect it."""
        args = _make_router_args(
            tmp_path,
            debug_train_only=True,
            use_miles_router=True,
            sglang_router_port=31000,
            use_session_server="v1",
        )

        assert specs_router(args) == []

    def test_debug_train_only_skips_the_router_in_a_disaggregated_run(self, tmp_path):
        # [FUNCTION DOC] With a prefill plus decode config and the train-only flag, no router specs
        # are produced.
        """The skip is keyed on the flag alone, so a prefill/decode config gets no router either."""
        args = _make_router_args(
            tmp_path,
            server_groups=[
                {"worker_type": "prefill", "num_gpus": 4, "num_gpus_per_engine": 1},
                {"worker_type": "decode", "num_gpus": 4, "num_gpus_per_engine": 1},
            ],
            debug_train_only=True,
        )

        assert specs_router(args) == []


def _make_router_args(tmp_path, *, server_groups: list[dict] | None = None, **overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Writes an sglang config file in the given temporary directory (default
    # group: one regular group, 8 gpus, 1 gpu per engine) and returns test arguments pointing at it
    # with 8 rollout gpus plus the given overrides. Side effect: creates the config file.
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(
            server_groups=server_groups or [{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
        )
    )
    return make_args(sglang_config=str(config_path), rollout_num_gpus=8, **overrides)


class TestComputeEnginePools:
    # [CLASS DOC] Tests the list of engine pool ids that the controller watches.
    def test_only_engine_specs_are_named(self, tmp_path):
        # [FUNCTION DOC] For a config with regular, placeholder and decode groups, the pool id list
        # contains exactly the two engine pools (group indices 0 and 2); the placeholder group and
        # any router do not appear.
        """These names are what the controller watches, so a router in the list would be reconciled as an engine."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2},
                    {"worker_type": "placeholder", "num_gpus": 4, "num_gpus_per_engine": 4},
                    {"worker_type": "decode", "num_gpus": 8, "num_gpus_per_engine": 4},
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=16)

        assert compute_engine_pool_ids(args) == ["inference-engine-all-0-0", "inference-engine-all-0-2"]


class TestInferenceSpecPinToHead:
    # [CLASS DOC] Tests head-node pinning of router and session server specs relative to the
    # rollout manager pinning flag.
    @pytest.mark.parametrize("pinned", [False, True])
    def test_the_router_spec_follows_the_rollout_manager_flag(self, pinned: bool):
        # [FUNCTION DOC] Parametrized on the flag. The router spec's pin-to-head setting equals the
        # rollout manager pinning flag.
        """The router is pinned to the head node exactly when the rollout manager is."""
        from miles.ray.specs.inference import _compute_spec_router

        args = _make_pin_args(pinned=pinned)

        router = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        assert router.scheduling.pin_to_head is pinned

    @pytest.mark.parametrize("pinned", [False, True])
    def test_the_session_server_spec_follows_the_rollout_manager_flag(self, pinned: bool):
        # [FUNCTION DOC] Parametrized on the flag. Without an external host, the session server
        # spec's pin-to-head setting equals the rollout manager pinning flag.
        """Without an external host, the session servers are pinned to the head exactly when the rollout manager is."""
        from miles.ray.specs.inference import spec_session_server

        args = _make_pin_args(pinned=pinned)

        session = spec_session_server(args)

        assert session.scheduling.pin_to_head is pinned

    def test_a_shared_external_host_keeps_the_session_servers_on_the_head(self):
        # [FUNCTION DOC] With the rollout manager not pinned but a single external session server
        # host configured, the session server spec is still pinned to the head node.
        """One external host for every instance reaches them only while they all sit on the head."""
        from miles.ray.specs.inference import spec_session_server

        args = _make_pin_args(pinned=False)
        args.session_server_external_host = "100.64.0.1"

        assert spec_session_server(args).scheduling.pin_to_head is True


def _make_pin_args(*, pinned: bool):
    # [FUNCTION DOC] HELPER: Builds test arguments for the head-pinning tests: the sgl router (not
    # the miles router), one session server worker, and the rollout manager pinning flag set to the
    # given value. The other session server related fields are set to neutral values.
    return make_args(
        pin_rollout_manager_to_head=pinned,
        use_miles_router=False,
        use_session_server=True,
        hf_checkpoint="/fake/model",
        session_server_workers=1,
        chat_template_path=None,
        tito_model="default",
        apply_chat_template_kwargs=None,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
        num_layers=None,
        moe_router_topk=None,
        save_debug_trajectory_data=None,
        lora_rank=0,
        lora_adapter_path=None,
        miles_router_timeout=None,
    )


class TestInferenceEnginePortSchema:
    # [CLASS DOC] Tests the port declarations (names, modes, dynamic allocation and consecutive
    # counts) of the engine specs.
    def test_the_master_port_reserves_a_block_for_every_dp_rank(self, tmp_path):
        # [FUNCTION DOC] With data parallel size 3 and 2 gpus per engine, the dist init port is a
        # master-mode port, dynamically allocatable, reserving 33 consecutive ports. The ports with
        # per-worker mode are exactly the primary, nccl and engine info bootstrap ports.
        """sglang needs a contiguous block behind dist_init, so the reservation must grow with dp size."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4, sglang_dp_size=3)

        ports = {info.name: info for info in specs_inference_engine(args)[0].port_infos}

        assert ports["dist_init"].mode == "master"
        assert ports["dist_init"].allow_dynamic is True
        assert ports["dist_init"].num_consecutive == 33
        assert {name for name, info in ports.items() if info.mode == "per_worker"} == {
            "primary",
            "nccl",
            "engine_info_bootstrap",
        }

    def test_the_gate_port_is_allocated_once_per_cell(self, tmp_path):
        # [FUNCTION DOC] The gate port is a master-mode port (one per cell), dynamically
        # allocatable, with a single consecutive port.
        """The out-of-band launch gate lives on the cell's rank-0 engine, like dist_init."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4)

        ports = {info.name: info for info in specs_inference_engine(args)[0].port_infos}

        assert ports["gate"].mode == "master"
        assert ports["gate"].allow_dynamic is True
        assert ports["gate"].num_consecutive == 1

    def test_only_prefill_engines_get_a_disaggregation_bootstrap_port(self, tmp_path):
        # [FUNCTION DOC] With one prefill and one decode group, the ordered port names of the
        # prefill spec are primary, dist init, nccl, disaggregation bootstrap, engine info bootstrap
        # and gate; the decode spec has the same list without the disaggregation bootstrap port.
        """The bootstrap port belongs to the prefill side alone."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "prefill", "num_gpus": 2, "num_gpus_per_engine": 2},
                    {"worker_type": "decode", "num_gpus": 2, "num_gpus_per_engine": 2},
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4)

        prefill, decode = specs_inference_engine(args)

        assert [info.name for info in prefill.port_infos] == [
            "primary",
            "dist_init",
            "nccl",
            "disaggregation_bootstrap",
            "engine_info_bootstrap",
            "gate",
        ]
        assert [info.name for info in decode.port_infos] == [
            "primary",
            "dist_init",
            "nccl",
            "engine_info_bootstrap",
            "gate",
        ]


class TestInferenceEngineGatedLaunch:
    # [CLASS DOC] Tests the arguments the engine spec passes to the engine launch command builder
    # for gating and multi-node ranks. The builder is replaced by a recorder returning a fixed
    # string.
    def test_the_launch_command_is_told_the_cells_own_gate_port(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Renders the launch command with a context whose gate address port is 13007
        # and expects the builder to have received that value as the gated launch port.
        """An engine launched without its gate port would start ungated and ignore the release."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4)
        recorded: dict = {}

        def _record(**kwargs) -> str:
            # [FUNCTION DOC] HELPER: Fake launch command builder; stores all keyword arguments in
            # the enclosing recorded mapping and returns a fixed command string.
            recorded.update(kwargs)
            return "launch-cmd"

        monkeypatch.setattr(inference_specs, "compute_engine_launch_cmd", _record)
        (spec,) = specs_inference_engine(args)
        spec.launch_command(_make_engine_ctx())

        assert recorded["gated_launch_port"] == 13007

    def test_each_node_of_a_multi_node_engine_is_numbered_within_its_own_cell(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Config of 16 gpus at 8 per engine on 4-gpu nodes gives two cells of two
        # workers. Rendering commands for every cell and worker, the recorded node ranks must be 0, 1,
        # 0, 1, i.e. numbered within the cell rather than globally.
        """node_rank is what tells sglang which member of its own two-node group a process is.
        Numbering it globally would launch the second engine as ranks 2 and 3 of a two-node
        group, and both engines would hang in dist_init with no error."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 16, "num_gpus_per_engine": 8}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=16, num_gpus_per_node=4)
        recorded: list[int] = []

        def _record(**kwargs) -> str:
            # [FUNCTION DOC] HELPER: Fake launch command builder; appends the node rank argument
            # to the enclosing list and returns a fixed command string.
            recorded.append(kwargs["node_rank"])
            return "launch-cmd"

        monkeypatch.setattr(inference_specs, "compute_engine_launch_cmd", _record)
        (spec,) = specs_inference_engine(args)
        for cell_index in range(2):
            for worker_in_cell_index in range(2):
                spec.launch_command(_make_engine_ctx(cell_index=cell_index, worker_in_cell_index=worker_in_cell_index))

        assert recorded == [0, 1, 0, 1]


class TestInferenceEngineRandomSeed:
    # [CLASS DOC] Tests the random seed passed to each engine launch: it must equal the run seed
    # plus the global engine actor rank (placeholder groups still consume ranks), stay stable across
    # relaunch, and shift with the run seed. The config has model "default" (regular group of 8
    # gpus at 2 per engine, placeholder group of 4 gpus) and model "reference" (16 gpus at 8 per
    # engine, no weight updates), on 4-gpu nodes with 28 rollout gpus and seed 1000.
    _CONFIG_YAML = (
        "sglang:\n"
        "  - name: default\n"
        "    server_groups:\n"
        "      - worker_type: regular\n"
        "        num_gpus: 8\n"
        "        num_gpus_per_engine: 2\n"
        "      - worker_type: placeholder\n"
        "        num_gpus: 4\n"
        "        num_gpus_per_engine: 4\n"
        "  - name: reference\n"
        "    update_weights: false\n"
        "    server_groups:\n"
        "      - worker_type: regular\n"
        "        num_gpus: 16\n"
        "        num_gpus_per_engine: 8\n"
    )

    def _seeds_by_pool(self, args, monkeypatch) -> dict[str, list[int]]:
        # [FUNCTION DOC] HELPER: Builds the engine specs for args, replaces the launch command
        # builder with a recorder, renders the command for every cell and worker in cell-major order
        # and returns a mapping from pool name to the list of random seeds passed, in launch order.
        recorded: dict[str, list[int]] = {}
        for spec in specs_inference_engine(args):
            seeds = recorded.setdefault(spec.name, [])

            def _record(*, into: list[int] = seeds, **kwargs) -> str:
                # [FUNCTION DOC] HELPER: Fake launch command builder; appends the random seed
                # argument to the bound per-pool list and returns a fixed command string.
                into.append(kwargs["random_seed"])
                return "launch-cmd"

            monkeypatch.setattr(inference_specs, "compute_engine_launch_cmd", _record)
            for cell_index in range(spec.scheduling.num_cells):
                for worker_in_cell_index in range(spec.scheduling.num_workers_per_cell):
                    spec.launch_command(
                        _make_engine_ctx(cell_index=cell_index, worker_in_cell_index=worker_in_cell_index)
                    )
        return recorded

    def _oracle_seeds_by_pool(self, args) -> dict[str, list[int]]:
        # [FUNCTION DOC] HELPER: Independent reference computation of the expected seeds. Walks
        # models and groups in config order with a running global rank; each group has gpus divided by
        # the smaller of gpus per engine and gpus per node actors; non-placeholder groups get seeds
        # run seed + rank + actor offset under their pool id; every group (placeholders too) then
        # advances the rank by its actor count. Returns a mapping from pool id to seed list.
        seeds: dict[str, list[int]] = {}
        global_rank = 0
        for model_idx, model_cfg in enumerate(resolve_sglang_config(args).models):
            for group_index, group_cfg in enumerate(model_cfg.server_groups):
                num_actors = group_cfg.num_gpus // min(group_cfg.num_gpus_per_engine, args.num_gpus_per_node)
                if group_cfg.worker_type != "placeholder":
                    pool_id = compute_engine_pool_id(args, model_idx=model_idx, group_index=group_index)
                    seeds[pool_id] = [args.seed + global_rank + i for i in range(num_actors)]
                global_rank += num_actors
        return seeds

    @pytest.fixture
    def args(self, tmp_path):
        # [FUNCTION DOC] HELPER: Pytest fixture. Writes the class config to a temporary file and
        # returns test arguments with 28 rollout gpus, 4 gpus per node and seed 1000.
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(self._CONFIG_YAML)
        return make_args(sglang_config=str(config_path), rollout_num_gpus=28, num_gpus_per_node=4, seed=1000)

    def test_every_actor_keeps_the_seed_the_pre_refactor_rank_gave_it(self, args, monkeypatch):
        # [FUNCTION DOC] The recorded seeds per pool must equal the independently computed oracle.
        """Reproducing a speculative-decoding run needs each engine actor's seed to stay put."""
        assert self._seeds_by_pool(args, monkeypatch) == self._oracle_seeds_by_pool(args)

    def test_a_skipped_placeholder_group_still_advances_the_numbering(self, args, monkeypatch):
        # [FUNCTION DOC] In model "default" the regular group has 4 actors (ranks 0 to 3) and the
        # placeholder group has 1 actor (rank 4) that launches nothing but still consumes a rank.
        # The first group of model "reference" must therefore get seeds 1005 to 1008.
        """A placeholder group consumed ranks before the refactor, so ignoring it would shift every later seed."""
        seeds = self._seeds_by_pool(args, monkeypatch)

        assert seeds[compute_engine_pool_id(args, model_idx=1, group_index=0)] == [1005, 1006, 1007, 1008]

    def test_no_two_engine_actors_in_the_cluster_share_a_seed(self, args, monkeypatch):
        # [FUNCTION DOC] Flattens the seeds of all pools and requires that they are all distinct.
        """Numbering every pool from the same base would hand two live engines the same RNG stream."""
        seeds = [seed for pool_seeds in self._seeds_by_pool(args, monkeypatch).values() for seed in pool_seeds]

        assert sorted(set(seeds)) == sorted(seeds)

    def test_an_actor_relaunched_onto_other_gpus_keeps_its_seed(self, args, monkeypatch):
        # [FUNCTION DOC] Renders the command of the last engine spec's cell 1, worker 1 twice, the
        # second time with different gpu ids; both renders must pass the same seed 1008.
        """A restarted engine that draws a fresh seed replays a different RNG stream than the run it resumes."""
        recorded: list[int] = []

        def _record(**kwargs) -> str:
            # [FUNCTION DOC] HELPER: Fake launch command builder; appends the random seed argument
            # to the enclosing list and returns a fixed command string.
            recorded.append(kwargs["random_seed"])
            return "launch-cmd"

        monkeypatch.setattr(inference_specs, "compute_engine_launch_cmd", _record)
        spec = specs_inference_engine(args)[-1]
        spec.launch_command(_make_engine_ctx(cell_index=1, worker_in_cell_index=1))
        spec.launch_command(
            _make_engine_ctx(cell_index=1, worker_in_cell_index=1, gpu_ids=[26, 27], local_gpu_ids=[2, 3])
        )

        assert recorded == [1008, 1008]

    def test_shifting_the_run_seed_shifts_every_engine_seed_by_the_same_amount(self, args, monkeypatch):
        # [FUNCTION DOC] Computes seeds for the base arguments and for arguments identical except
        # the run seed raised by 7; every seed of every pool must be exactly 7 larger.
        """The per-engine numbers are an offset on top of --seed, so a rerun with another seed must move as a block."""
        base = self._seeds_by_pool(args, monkeypatch)
        shifted = self._seeds_by_pool(
            make_args(
                sglang_config=args.sglang_config,
                rollout_num_gpus=28,
                num_gpus_per_node=4,
                seed=args.seed + 7,
            ),
            monkeypatch,
        )

        assert shifted == {pool_id: [seed + 7 for seed in seeds] for pool_id, seeds in base.items()}


def _make_engine_ctx(
    *,
    cell_index: int = 0,
    worker_in_cell_index: int = 0,
    gpu_ids: list[int] | None = None,
    local_gpu_ids: list[int] | None = None,
) -> LaunchCommandContext:
    # [FUNCTION DOC] HELPER: Builds a launch command context for an engine worker at the given
    # cell and worker index. Own addresses are on host 10.0.0.1 with ports primary 30000, dist init
    # 9000, nccl 10000, engine info bootstrap 12000 and gate 13007; no pool addresses. The gpu id
    # lists default to [0, 1] when not given (both global and local).
    return LaunchCommandContext(
        cell_index=cell_index,
        worker_in_cell_index=worker_in_cell_index,
        self_addrs=dict(
            primary=HostAndPort(host="10.0.0.1", port=30000),
            dist_init=HostAndPort(host="10.0.0.1", port=9000),
            nccl=HostAndPort(host="10.0.0.1", port=10000),
            engine_info_bootstrap=HostAndPort(host="10.0.0.1", port=12000),
            gate=HostAndPort(host="10.0.0.1", port=13007),
        ),
        pool_addrs={},
        gpu_ids=[0, 1] if gpu_ids is None else gpu_ids,
        local_gpu_ids=[0, 1] if local_gpu_ids is None else local_gpu_ids,
    )


class TestEngineBaseGpuId:
    # [CLASS DOC] Tests the base gpu id passed to the engine launch builder.
    def test_the_launch_command_uses_the_gpu_ids_the_worker_itself_resolved(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With context global gpu ids [6, 7] and local ids [2, 3], the builder must
        # receive base gpu id 2 (the first local id), not 6.
        """The manager runs without a visibility mask, so passing its physical ids would point sglang at
        a device the engine process cannot see."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4)
        recorded: dict = {}

        def _record(**kwargs) -> str:
            # [FUNCTION DOC] HELPER: Fake launch command builder; stores all keyword arguments in
            # the enclosing recorded mapping and returns a fixed command string.
            recorded.update(kwargs)
            return "launch-cmd"

        monkeypatch.setattr(inference_specs, "compute_engine_launch_cmd", _record)
        (spec,) = specs_inference_engine(args)
        spec.launch_command(_make_engine_ctx(gpu_ids=[6, 7], local_gpu_ids=[2, 3]))

        assert recorded["base_gpu_id"] == 2


class TestEngineMetaApiKey:
    # [CLASS DOC] Tests which api key appears in the per-cell metadata of an engine spec: a group
    # override wins over the global argument.
    def _meta_for(self, tmp_path, *, overrides_yaml: str = "", **args_overrides):
        # [FUNCTION DOC] HELPER: Writes a config with one regular group (8 gpus, 1 per engine) plus
        # the given extra yaml text for group overrides, builds test arguments with 8 rollout gpus and
        # the given overrides, builds the single engine spec and returns its metadata for cell 0.
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 1\n" + overrides_yaml
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=8, **args_overrides)
        (spec,) = specs_inference_engine(args)
        return spec.meta(WorkerMetaContext(cell_index=0))

    def test_a_group_api_key_override_wins_over_the_args_key(self, tmp_path):
        # [FUNCTION DOC] A group override giving an api key makes the metadata key equal that value
        # even though the global argument has a different one.
        """The ServerArgs-named api_key override reaches the cell meta ahead of the global args key."""
        meta = self._meta_for(
            tmp_path,
            overrides_yaml="        overrides:\n          api_key: from-override\n",
            sglang_api_key="from-args",
        )
        assert meta["sglang_api_key"] == "from-override"

    def test_the_args_key_is_used_without_an_override(self, tmp_path):
        # [FUNCTION DOC] With no group override, the metadata key equals the global argument value.
        """Without a group override the engine api key falls back to args.sglang_api_key."""
        meta = self._meta_for(tmp_path, sglang_api_key="from-args")
        assert meta["sglang_api_key"] == "from-args"

    def test_an_explicit_empty_override_is_kept_verbatim(self, tmp_path):
        # [FUNCTION DOC] A group override with an empty api key string makes the metadata key the
        # empty string rather than falling back to the global argument.
        """An override disabling the key must win over the args key instead of silently falling back."""
        meta = self._meta_for(
            tmp_path,
            overrides_yaml='        overrides:\n          api_key: ""\n',
            sglang_api_key="from-args",
        )
        assert meta["sglang_api_key"] == ""


class TestTrailingPartialEngineRejection:
    # [CLASS DOC] Tests that a group whose gpu count is not a whole multiple of the multi-node
    # engine width is rejected.
    def _specs_for(self, tmp_path, *, num_gpus: int, num_gpus_per_engine: int):
        # [FUNCTION DOC] HELPER: Writes a one-group regular config with the given gpu counts, builds
        # arguments with that many rollout gpus and 8 gpus per node, and returns the engine specs.
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": num_gpus, "num_gpus_per_engine": num_gpus_per_engine}
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=num_gpus, num_gpus_per_node=8)
        return specs_inference_engine(args)

    def test_a_trailing_partial_multi_node_engine_is_rejected(self, tmp_path):
        # [FUNCTION DOC] 24 gpus with 16 gpus per engine on 8-gpu nodes raises an assertion error
        # mentioning a whole number of engines.
        """24 GPUs cannot host 16-GPU engines on 8-GPU nodes and must fail fast instead of silently flooring."""
        with pytest.raises(AssertionError, match="whole number of"):
            self._specs_for(tmp_path, num_gpus=24, num_gpus_per_engine=16)

    def test_a_whole_number_of_multi_node_engines_passes(self, tmp_path):
        # [FUNCTION DOC] 32 gpus with 16 gpus per engine give exactly one spec with two cells.
        """32 GPUs host exactly two 16-GPU engines and resolve into two cells."""
        (spec,) = self._specs_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert spec.scheduling.num_cells == 2


class TestCrossNodeEngineWidth:
    # [CLASS DOC] Tests validation of engine width relative to the node gpu count: an engine wider
    # than a node must tile whole nodes, narrower ones need not divide the node, the node size comes
    # from the arguments, and the check applies to every non-placeholder group.
    def _specs_for(self, tmp_path, *, num_gpus: int, num_gpus_per_engine: int, num_gpus_per_node: int = 8):
        # [FUNCTION DOC] HELPER: Builds engine specs for a single regular group of the given size
        # and engine width on nodes of the given size (default 8) by delegating to the group helper.
        return self._specs_for_groups(
            tmp_path,
            server_groups=[
                {"worker_type": "regular", "num_gpus": num_gpus, "num_gpus_per_engine": num_gpus_per_engine}
            ],
            num_gpus_per_node=num_gpus_per_node,
        )

    def _specs_for_groups(self, tmp_path, *, server_groups: list[dict], num_gpus_per_node: int = 8):
        # [FUNCTION DOC] HELPER: Writes a config with the given groups, builds arguments whose
        # rollout gpu count is the sum of the group gpu counts, and returns the engine specs.
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(make_sglang_config_yaml(server_groups=server_groups))
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=sum(group["num_gpus"] for group in server_groups),
            num_gpus_per_node=num_gpus_per_node,
        )
        return specs_inference_engine(args)

    def test_a_cross_node_engine_that_does_not_tile_the_node_is_rejected(self, tmp_path):
        # [FUNCTION DOC] 24 gpus at 12 per engine on 8-gpu nodes raises an assertion error saying
        # the width neither fits a node nor tiles whole nodes.
        """A 12-gpu engine on 8-gpu nodes silently specs one node of 12 ranks and then waits for ever."""
        with pytest.raises(AssertionError, match="nor tiles whole nodes"):
            self._specs_for(tmp_path, num_gpus=24, num_gpus_per_engine=12)

    def test_an_engine_that_fits_in_one_node_is_accepted(self, tmp_path):
        # [FUNCTION DOC] 8 gpus at 4 per engine is accepted and has one worker per cell.
        """Anything up to a node wide needs no tiling at all."""
        (spec,) = self._specs_for(tmp_path, num_gpus=8, num_gpus_per_engine=4)
        assert spec.scheduling.num_workers_per_cell == 1

    def test_an_engine_spanning_whole_nodes_is_accepted(self, tmp_path):
        # [FUNCTION DOC] 32 gpus at 16 per engine on 8-gpu nodes is accepted with two workers per
        # cell.
        """A 16-gpu engine covers exactly two 8-gpu nodes, so every rank has a node to run on."""
        (spec,) = self._specs_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert spec.scheduling.num_workers_per_cell == 2

    def test_a_lone_cross_node_engine_is_rejected_even_with_no_leftover_gpus(self, tmp_path):
        # [FUNCTION DOC] 12 gpus at 12 per engine divides evenly yet is still rejected with the
        # same "nor tiles whole nodes" assertion error.
        """One 12-gpu engine on 12 gpus divides evenly, so only the tiling rule can catch its four unlaunched ranks."""
        with pytest.raises(AssertionError, match="nor tiles whole nodes"):
            self._specs_for(tmp_path, num_gpus=12, num_gpus_per_engine=12)

    def test_an_engine_narrower_than_a_node_need_not_divide_the_node(self, tmp_path):
        # [FUNCTION DOC] 12 gpus at 6 per engine on 8-gpu nodes is accepted: one worker per cell and
        # 6 gpu slots per worker.
        """A 6-gpu engine lives inside one 8-gpu node, so the node size need not be a multiple of it."""
        (spec,) = self._specs_for(tmp_path, num_gpus=12, num_gpus_per_engine=6)
        assert (spec.scheduling.num_workers_per_cell, spec.scheduling.num_gpu_slots_per_worker) == (1, 6)

    def test_the_node_width_in_the_check_comes_from_args_not_a_fixed_eight(self, tmp_path):
        # [FUNCTION DOC] 12 gpus at 6 per engine on 4-gpu nodes is rejected with the tiling
        # assertion error, showing that the node size argument drives the check.
        """A 6-gpu engine is fine on 8-gpu nodes but straddles 4-gpu nodes, so the args value must drive the check."""
        with pytest.raises(AssertionError, match="nor tiles whole nodes"):
            self._specs_for(tmp_path, num_gpus=12, num_gpus_per_engine=6, num_gpus_per_node=4)

    def test_an_engine_tiling_a_smaller_node_is_accepted(self, tmp_path):
        # [FUNCTION DOC] 16 gpus at 8 per engine on 4-gpu nodes is accepted, giving 2 cells of 2
        # workers each.
        """An 8-gpu engine covers exactly two 4-gpu nodes, which a check hardcoded to 8-gpu nodes would misread."""
        (spec,) = self._specs_for(tmp_path, num_gpus=16, num_gpus_per_engine=8, num_gpus_per_node=4)
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (2, 2)

    def test_a_later_group_is_checked_and_named_in_the_rejection(self, tmp_path):
        # [FUNCTION DOC] A valid prefill group (8 at 8) followed by an invalid decode group (24 at
        # 12) is rejected, and the assertion message names the decode group and its engine width 12.
        """Every group carries its own width, so a good first group must not excuse a bad second one."""
        with pytest.raises(AssertionError, match="group 'decode'.*num_gpus_per_engine=12"):
            self._specs_for_groups(
                tmp_path,
                server_groups=[
                    {"worker_type": "prefill", "num_gpus": 8, "num_gpus_per_engine": 8},
                    {"worker_type": "decode", "num_gpus": 24, "num_gpus_per_engine": 12},
                ],
            )

    def test_a_placeholder_group_width_is_not_checked(self, tmp_path):
        # [FUNCTION DOC] A placeholder group of 12 gpus at 12 per engine followed by a valid regular
        # group is accepted; only the regular group yields a spec, with one cell.
        """A placeholder group only reserves gpus and launches no ranks, so its width cannot strand any."""
        specs = self._specs_for_groups(
            tmp_path,
            server_groups=[
                {"worker_type": "placeholder", "num_gpus": 12, "num_gpus_per_engine": 12},
                {"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 8},
            ],
        )
        assert [spec.scheduling.num_cells for spec in specs] == [1]


class TestEngineCellChunking:
    # [CLASS DOC] Tests how engine groups are chunked into cells, pods and workers, and the gpu
    # offsets reported in cell metadata.
    def _spec_for(self, tmp_path, *, num_gpus: int, num_gpus_per_engine: int, gpu_offset: int = 0):
        # [FUNCTION DOC] HELPER: Writes a config whose last group is a regular group of the given
        # size and width; when a gpu offset is given a placeholder group of that many gpus precedes
        # it. Builds arguments on 8-gpu nodes and returns the last engine spec.
        config_path = tmp_path / "sglang.yaml"
        groups = []
        if gpu_offset:
            groups.append(
                {"worker_type": "placeholder", "num_gpus": gpu_offset, "num_gpus_per_engine": num_gpus_per_engine}
            )
        groups.append({"worker_type": "regular", "num_gpus": num_gpus, "num_gpus_per_engine": num_gpus_per_engine})
        config_path.write_text(make_sglang_config_yaml(server_groups=groups))
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=num_gpus + gpu_offset, num_gpus_per_node=8)
        return specs_inference_engine(args)[-1]

    def test_a_single_gpu_engine_becomes_its_own_cell(self, tmp_path):
        # [FUNCTION DOC] 8 gpus at 1 per engine give 8 cells of 1 worker each.
        """With one gpu per engine on 8-gpu nodes, the group resolves into eight one-worker cells."""
        spec = self._spec_for(tmp_path, num_gpus=8, num_gpus_per_engine=1)
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (8, 1)

    def test_a_multi_node_engine_chunks_its_node_ranks_into_one_cell(self, tmp_path):
        # [FUNCTION DOC] 32 gpus at 16 per engine give 2 cells of 2 workers each.
        """A 16-gpu engine on 8-gpu nodes spans two workers, so 32 gpus collapse into two cells."""
        spec = self._spec_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (2, 2)

    def test_a_multi_node_engine_gets_one_pod_per_node(self, tmp_path):
        # [FUNCTION DOC] For 32 gpus at 16 per engine the spec has 8 gpus per node, 2 pods per cell
        # and 1 worker per pod.
        """The chart turns this into the pods of a leaderworkerset, so a wrong count mis-sizes every group."""
        spec = self._spec_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)

        assert spec.scheduling.num_gpus_per_node == 8
        assert (spec.scheduling.pods_per_cell(), spec.scheduling.workers_per_pod()) == (2, 1)

    def test_an_engine_inside_one_node_stays_in_one_pod(self, tmp_path):
        # [FUNCTION DOC] For 8 gpus at 4 per engine the spec has 1 pod per cell and 1 worker per
        # pod.
        """A cell that fits a node must not be split, or its ranks would talk over the network for nothing."""
        spec = self._spec_for(tmp_path, num_gpus=8, num_gpus_per_engine=4)

        assert (spec.scheduling.pods_per_cell(), spec.scheduling.workers_per_pod()) == (1, 1)

    def test_single_gpu_cells_carry_contiguous_gpu_offsets(self, tmp_path):
        # [FUNCTION DOC] For 8 single-gpu cells, the metadata gpu offset of cell i is i.
        """Every cell must claim its own gpu span, otherwise two engines share the same devices."""
        spec = self._spec_for(tmp_path, num_gpus=8, num_gpus_per_engine=1)
        offsets = [spec.meta(WorkerMetaContext(cell_index=index))["gpu_offset"] for index in range(8)]
        assert offsets == list(range(8))

    def test_multi_node_cells_advance_by_a_whole_engine(self, tmp_path):
        # [FUNCTION DOC] For 16-gpu engines on 8-gpu nodes, the gpu offsets of cells 0 and 1 are 0
        # and 16 (workers times slots per cell).
        """The per-cell stride is workers x slots, so a 16-gpu engine advances the offset by 16, not by 1."""
        spec = self._spec_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        offsets = [spec.meta(WorkerMetaContext(cell_index=index))["gpu_offset"] for index in range(2)]
        assert offsets == [0, 16]

    def test_the_group_gpu_offset_shifts_every_cell(self, tmp_path):
        # [FUNCTION DOC] A group placed after an 8-gpu placeholder with 16 single-gpu cells has cell
        # gpu offsets 8 through 23.
        """A group placed after another starts counting from that group's end, per cell as well as overall."""
        spec = self._spec_for(tmp_path, num_gpus=16, num_gpus_per_engine=1, gpu_offset=8)
        offsets = [spec.meta(WorkerMetaContext(cell_index=index))["gpu_offset"] for index in range(16)]
        assert offsets == list(range(8, 24))


class TestRouterInterpreterFlags:
    # [CLASS DOC] Tests that router launch commands reuse the interpreter flags of the process
    # that built the spec, for both router kinds.
    @pytest.mark.parametrize(
        ("use_miles_router", "module"),
        [(True, "miles.router.router"), (False, "sglang_router.launch_router")],
        ids=["miles_router", "sglang_router"],
    )
    def test_the_router_launch_command_carries_the_parent_interpreter_flags(
        self, monkeypatch: pytest.MonkeyPatch, use_miles_router: bool, module: str
    ):
        # [FUNCTION DOC] Parametrized over the miles router (its router module) and the sgl router
        # (its launch module). With a faked original argument vector carrying extra interpreter
        # flags, the first six command tokens are the interpreter, the flags, the module option and
        # the router module.
        """A router launched by a bare interpreter runs with different semantics than the job that spawned it."""
        monkeypatch.setattr(
            sys,
            "orig_argv",
            [sys.executable, "-O", "-X", "faulthandler", "-m", "miles.train", "--config", "x.yaml"],
        )
        args = make_args(use_miles_router=use_miles_router, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        argv = shlex.split(spec.launch_command(_make_router_ctx()))

        assert argv[:6] == [sys.executable, "-O", "-X", "faulthandler", "-m", module]


class TestEngineInterpreterFlags:
    # [CLASS DOC] Tests that the engine launch builder receives the parent interpreter flags.
    def test_the_engine_launch_command_carries_the_parent_interpreter_flags(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a faked original argument vector, the interpreter prefix argument
        # given to the builder is the interpreter followed by the optimize flag and the extra
        # option pair, without the module option and script arguments.
        """An engine launched by a bare interpreter runs with different semantics than the job that spawned it."""
        monkeypatch.setattr(
            sys,
            "orig_argv",
            [sys.executable, "-O", "-X", "faulthandler", "-m", "miles.train", "--config", "x.yaml"],
        )
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4)
        recorded: dict = {}

        def _record(**kwargs) -> str:
            # [FUNCTION DOC] HELPER: Fake launch command builder; stores all keyword arguments in
            # the enclosing recorded mapping and returns a fixed command string.
            recorded.update(kwargs)
            return "launch-cmd"

        monkeypatch.setattr(inference_specs, "compute_engine_launch_cmd", _record)
        (spec,) = specs_inference_engine(args)
        spec.launch_command(_make_engine_ctx())

        assert recorded["interpreter_prefix"] == [sys.executable, "-O", "-X", "faulthandler"]


class TestSpecInferenceController:
    # [CLASS DOC] Tests the inference controller spec and the registration reporter spec:
    # scheduling, worker class, platform access, naming, helm rendering and the constructor
    # arguments (engine provider and router providers) obtained from a fake backend capability.
    def _args(self, tmp_path, **overrides) -> Namespace:
        # [FUNCTION DOC] HELPER: Writes an sglang config with one regular group (8 gpus, 4 per
        # engine) and returns test arguments pointing at it with 8 rollout gpus plus the overrides.
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 4}]
            )
        )
        return make_args(sglang_config=str(config_path), rollout_num_gpus=8, **overrides)

    def _ctor_context(self, capability: FakeBackendCapability) -> WorkerCtorContext:
        # [FUNCTION DOC] HELPER: Builds a constructor context for cell 0, worker 0 with no gpus
        # and the given fake backend capability.
        return WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

    def test_every_run_gets_exactly_one_gpuless_controller(self, tmp_path):
        # [FUNCTION DOC] The controller spec carries the controller pool id as its name, has 1 cell
        # of 1 worker, and requests zero gpus and zero gpu slots per worker.
        """It is a control-plane worker on both backends; a gpu request would reserve a whole node for it."""
        spec = spec_inference_controller(self._args(tmp_path))

        assert spec.name == INFERENCE_CONTROLLER_POOL_ID
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 1)
        assert spec.scheduling.num_gpus_per_worker == 0
        assert spec.scheduling.num_gpu_slots_per_worker == 0

    def test_the_worker_class_is_the_controller_itself(self, tmp_path):
        # [FUNCTION DOC] The worker class path of the spec loads to the real inference controller
        # class.
        """The spec names the class a pod or actor constructs, so it must be the real implementation."""
        spec = spec_inference_controller(self._args(tmp_path))

        assert load_function(spec.worker_class) is InferenceController

    def test_it_declares_only_the_platform_reads_its_provider_performs(self, tmp_path):
        # [FUNCTION DOC] The controller spec's platform access is the read-only level.
        """Watching engine pods needs reads, while deleting them remains outside this worker's capability."""
        spec = spec_inference_controller(self._args(tmp_path))

        assert spec.platform_access is PlatformAccess.READ

    def test_the_registration_reporter_declares_only_platform_reads(self, tmp_path):
        # [FUNCTION DOC] For the inference deploy component, exactly one reporter spec is produced
        # and its platform access is the read-only level.
        """The reporter watches engine pods, so it needs reads and none of the orchestrator's other rights."""
        (spec,) = specs_inference_registration_reporter(self._args(tmp_path, deploy_component="inference"))

        assert spec.platform_access is PlatformAccess.READ

    def test_the_worker_name_is_stable(self):
        # [FUNCTION DOC] The controller worker name is the fixed string inference-controller
        # followed by zero-padded cell and worker indices 00000-00000.
        """The driver looks the controller up by name, so this name is part of the release's contract."""
        assert inference_controller_worker_name() == "inference-controller-00000-00000"

    def test_it_renders_into_static_workers_with_its_rpc_port(self, tmp_path):
        # [FUNCTION DOC] Renders the controller spec into helm values with the fixed launch plan.
        # Exactly one static worker entry results; the spec category maps to the static workers
        # section; the entry is named after the controller pool id, exposes an rpc port 8000, passes
        # the pool id on its command line, uses the controller worker class and requests no resources.
        """The release has to contain the controller pod, or the address book would point at nothing."""
        spec = spec_inference_controller(self._args(tmp_path))

        values = build_values([spec], _controller_layout()).as_values()

        (entry,) = values["run"]["staticWorkers"]
        assert SECTION_OF_CATEGORY[spec.category] == "staticWorkers"
        assert entry["name"] == INFERENCE_CONTROLLER_POOL_ID
        assert entry["ports"] == [{"name": "rpc", "port": 8000}]
        assert entry["command"][entry["command"].index("--pool-id") + 1] == INFERENCE_CONTROLLER_POOL_ID
        assert spec.worker_class == INFERENCE_CONTROLLER_WORKER_CLASS
        assert "resources" not in entry

    def test_it_asks_for_a_provider_over_the_engine_pools_it_will_observe(self, tmp_path):
        # [FUNCTION DOC] Building the constructor kwargs asks the capability once for a cells
        # provider over exactly the engine pool ids, and that provider becomes the engine provider.
        """The controller never learns which backend reports those cells, only which pools it wants reported."""
        args = self._args(tmp_path)
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        kwargs = spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert capability.requested_pool_ids == [compute_engine_pool_ids(args)]
        assert kwargs["engine_provider"] is capability.cells_provider

    def test_it_asks_for_one_router_provider_per_model(self, tmp_path):
        # [FUNCTION DOC] For a one-model config, the capability is asked for a static provider for
        # the model 0 router pool only, and the router providers kwarg is a one element list with it.
        """Every model is served by its own router pool, so one provider cannot answer for all of them."""
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        kwargs = spec_inference_controller(self._args(tmp_path)).ctor_kwargs(self._ctor_context(capability))

        assert capability.requested_static_pool_ids == [compute_router_pool_id(0)]
        assert kwargs["router_providers"] == [capability.static_provider]

    def test_a_train_only_run_builds_a_controller_over_an_empty_pool(self, tmp_path):
        # [FUNCTION DOC] With the train-only flag, the capability is asked for a cells provider over
        # an empty list of pool ids.
        """--debug-train-only deploys no engines, so the controller observes no pools at all."""
        args = self._args(tmp_path, debug_train_only=True)
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert capability.requested_pool_ids == [[]]

    def test_the_static_discovery_path_never_asks_the_backend(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With external rollout, an external engine address and the custom provider
        # path naming the built-in static engine provider factory (whose provider class is swapped
        # for a recording stand-in), the engine provider is that stand-in constructed with the args
        # and the capability's cells provider was never requested.
        """External engines belong to no backend, so the capability must never be asked for them."""
        args = self._args(
            tmp_path,
            rollout_external=True,
            rollout_external_engine_addrs=["host1:8000"],
            custom_inference_engine_provider_path=(
                "miles.ray.rollout.external_engine_provider.static_inference_engine_provider"
            ),
        )
        capability = FakeBackendCapability(cells_provider=None, static_provider=object())
        monkeypatch.setattr(
            external_engine_provider_module, "StaticInferenceEngineWorkerProvider", _RecordingStaticProvider
        )

        kwargs = spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert isinstance(kwargs["engine_provider"], _RecordingStaticProvider)
        assert kwargs["engine_provider"].args is args
        assert capability.requested_pool_ids == []

    def test_the_provider_factory_path_is_loaded_unconditionally(self, tmp_path):
        # [FUNCTION DOC] With a custom provider factory path pointing at the fake factory in this
        # module, the engine provider kwarg equals what that factory returns (a tuple of a marker,
        # the args and the capability), proving the path in args is always loaded and called.
        """Provider selection lives in arg validation, so the spec must run whatever path args carry."""
        args = self._args(
            tmp_path,
            rollout_external=True,
            rollout_external_engine_addrs=["host1:8000"],
            custom_inference_engine_provider_path=f"{__name__}._fake_engine_provider_factory",
        )
        capability = FakeBackendCapability(cells_provider=None, static_provider=object())

        kwargs = spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert kwargs["engine_provider"] == ("custom-provider", args, capability)


class _RecordingStaticProvider:
    # [CLASS DOC] HELPER: Fake static worker provider that remembers the args it was built with
    # and hands out a tuple handle naming the requested worker.
    def __init__(self, *, args) -> None:
        # [FUNCTION DOC] HELPER: Stores the given args on the instance.
        self.args = args

    def get_handle(self, worker_name: str) -> tuple[str, str]:
        # [FUNCTION DOC] HELPER: Returns a two element tuple of a marker string and the worker name.
        return ("controller-handle", worker_name)


def _fake_engine_provider_factory(args, *, capability):
    # [FUNCTION DOC] HELPER: Fake custom engine provider factory; returns a tuple of a marker
    # string, the args and the capability so tests can see exactly what it was called with.
    return ("custom-provider", args, capability)


class TestTheEngineEnvironment:
    # [CLASS DOC] Tests an extra environment variable set on every engine.
    def test_every_engine_is_told_to_report_its_own_env_vars(self) -> None:
        # [FUNCTION DOC] The computed engine environment has the variable that exposes the engine's
        # own environment variables set to "1".
        """Without the gate the engine answers /server_info with no env_vars, and the audit is empty."""
        assert compute_inference_engine_env_vars(make_args())["SGLANG_EXPOSE_OWN_ENV_VARS"] == "1"


class TestRegistrationWiring:
    # [CLASS DOC] Tests how deployments of a split run find each other: the registration hub
    # versus the plain engine provider, the reporter's constructor arguments, the controller
    # address provider and engine pool naming by deployment instance.
    @staticmethod
    def _args(tmp_path, **overrides) -> Namespace:
        # [FUNCTION DOC] HELPER: Same as the controller test helper: writes a one-group sglang
        # config (8 gpus, 4 per engine) and returns test arguments with 8 rollout gpus and overrides.
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 4}]
            )
        )
        return make_args(sglang_config=str(config_path), rollout_num_gpus=8, **overrides)

    @staticmethod
    def _ctor_context(capability: FakeBackendCapability) -> WorkerCtorContext:
        # [FUNCTION DOC] HELPER: Builds a constructor context for cell 0, worker 0 with no gpus
        # and the given fake backend capability.
        return WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

    def test_a_run_serving_its_own_engines_keeps_the_engine_provider_it_always_had(self, tmp_path):
        # [FUNCTION DOC] With the default (unsplit) deploy component, the controller's engine
        # provider kwarg is not a registration hub.
        """Every unsplit run must reach its own engines exactly as it did before registration existed."""
        args = self._args(tmp_path)
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        kwargs = spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert not isinstance(kwargs["engine_provider"], RegistrationHub)

    def test_the_reporter_carries_this_deployments_identity_hub_and_engine_provider(self, tmp_path) -> None:
        # [FUNCTION DOC] For the inference deploy component with instance id west and run uuid
        # run-west, the single reporter spec's constructor kwargs carry the args unchanged and a
        # reporter whose run uuid and reporter id come from the args, whose hub endpoint is the
        # controller handle looked up via the static provider under the controller worker name, and
        # whose worker provider is the capability's cells provider.
        """The reporter joins this deployment's engines to the addressed run under its deployment identity."""
        args = self._args(tmp_path, deploy_component="inference", deploy_instance_id="west", run_uuid="run-west")
        engine_provider = object()
        controller_provider = _RecordingStaticProvider(args=args)
        capability = FakeBackendCapability(cells_provider=engine_provider, static_provider=controller_provider)

        (spec,) = specs_inference_registration_reporter(args)
        kwargs = spec.ctor_kwargs(self._ctor_context(capability))

        reporter = kwargs["reporter"]
        assert kwargs["args"] is args
        assert reporter.run_uuid == "run-west"
        assert reporter.reporter_id == "west"
        assert reporter.hub_endpoint == ("controller-handle", inference_controller_worker_name())
        assert reporter.worker_provider is engine_provider

    def test_a_run_that_deploys_no_engines_of_its_own_serves_from_the_registered_ones(self, tmp_path):
        # [FUNCTION DOC] With the primary deploy component, the controller's engine provider kwarg
        # is a registration hub (engines register themselves) instead of a backend provider.
        """The rest of the run must not know which deployment launched an engine it generates from."""
        args = self._args(tmp_path, deploy_component="primary")
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        kwargs = spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert isinstance(kwargs["engine_provider"], RegistrationHub)

    def test_an_engine_deployment_reports_into_the_controller_it_was_given(self, tmp_path):
        # [FUNCTION DOC] For an inference deployment configured with a controller address
        # controller:9000, the controller provider is a static worker provider whose address lookup
        # for the controller worker name returns the rpc port at host controller and port 9000, and
        # the capability was never asked for a static provider.
        """It derives no name of another release, so this address is the only way it finds the run."""
        args = self._args(
            tmp_path,
            deploy_component="inference",
            inference_controller_addr="controller:9000",
        )
        capability = FakeBackendCapability(cells_provider=object())

        provider = compute_inference_controller_provider(args, capability=capability)

        assert isinstance(provider, StaticWorkerProvider)
        addrs = asyncio.run(provider.get_addrs(f"{INFERENCE_CONTROLLER_POOL_ID}-00000-00000"))
        assert addrs[RPC_PORT_NAME] == HostAndPort(host="controller", port=9000)
        assert capability.requested_static_pool_ids == []

    def test_a_run_that_holds_its_controller_addresses_it_by_its_own_release(self, tmp_path):
        # [FUNCTION DOC] With no controller address configured, the controller provider is obtained
        # from the capability by asking for a static provider for the controller pool id only.
        """Naming another release's pods from here is exactly what a split run may not do."""
        args = self._args(tmp_path)
        capability = FakeBackendCapability(static_provider=object())

        compute_inference_controller_provider(args, capability=capability)

        assert capability.requested_static_pool_ids == [INFERENCE_CONTROLLER_POOL_ID]

    def test_an_engine_deployment_names_its_pools_after_the_instance_it_deploys(self, tmp_path):
        # [FUNCTION DOC] For an inference deployment with instance id west, the engine pool id list
        # is a single entry using west as the segment after the engine prefix.
        """Two engine groups of one run install the same pools, and a shared name would collide in the run."""
        args = self._args(tmp_path, deploy_component="inference", deploy_instance_id="west")

        assert compute_engine_pool_ids(args) == ["inference-engine-west-0-0"]

    def test_a_run_deploying_its_own_engines_names_its_pools_after_the_component(self, tmp_path):
        # [FUNCTION DOC] For an unsplit run, the engine pool id list is a single entry whose
        # segment is "all", the default deploy component.
        """Every pool id carries a segment, so the unsplit run falls back to the component it deploys."""
        assert compute_engine_pool_ids(self._args(tmp_path)) == ["inference-engine-all-0-0"]
