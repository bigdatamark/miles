# [MODULE DOC]
# Tests for the SGLang engine launch-command builder: the rendered command preserves interpreter
# prefix tokens ahead of the SGLang server module invocation, renders the worker's own device for
# CPU-only controllers, carries the allocated addressing (host, port, rendezvous address, gate
# port, model path), the base GPU id verbatim, IPv6 host stripping versus bracketed rendezvous
# address, prefill bootstrap port, the actor-given random seed and the API key; LoRA target
# modules reach both the engine and the sync config for several adapter-target shapes; and the
# launch-gate capability assertion accepts or refuses a stubbed server-args type. Skipped when
# SGLang is not installed.
from __future__ import annotations

import dataclasses
import shlex
import sys

import pytest
from tests.fast.backends.sglang_utils.conftest import make_engine_args, tiny_model_path

pytest.importorskip("sglang")

from miles.backends.sglang_utils import sglang_engine
from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv
from miles.backends.sglang_utils.sglang_engine import _assert_launch_gate_served, compute_engine_launch_cmd
from miles.utils.lora.utils import build_lora_config


def _cmd(
    *,
    worker_type: str = "regular",
    args=None,
    interpreter_prefix: list[str] | None = None,
    addr_overrides: dict | None = None,
    base_gpu_id: int = 0,
    random_seed: int = 0,
    **kwargs,
) -> str:
    # [FUNCTION DOC] HELPER: Keyword-only. Builds a launch command string through the engine's
    # command builder with fixed address defaults (host 10.0.0.1, port 30000, NCCL port 20031,
    # engine-info bootstrap 20033, gated launch port 20034, rendezvous 10.0.0.1:20000, no
    # disaggregation bootstrap port), updated by the address overrides mapping; node rank 0, empty
    # SGLang overrides, 1 GPU per engine, the given worker type (regular), base GPU id (0), random
    # seed (0), args (default: the shared engine args), interpreter prefix (default: the current
    # Python executable) and any extra keyword arguments passed through. Returns the command.
    addr_and_ports = dict(
        host="10.0.0.1",
        port=30000,
        nccl_port=20031,
        engine_info_bootstrap_port=20033,
        gated_launch_port=20034,
        dist_init_addr="10.0.0.1:20000",
        disaggregation_bootstrap_port=None,
    )
    addr_and_ports.update(addr_overrides or {})
    return compute_engine_launch_cmd(
        args or make_engine_args(),
        interpreter_prefix=interpreter_prefix or [sys.executable],
        node_rank=0,
        worker_type=worker_type,
        base_gpu_id=base_gpu_id,
        sglang_overrides={},
        num_gpus_per_engine=1,
        dist_init_addr=addr_and_ports["dist_init_addr"],
        nccl_port=addr_and_ports["nccl_port"],
        host=addr_and_ports["host"],
        port=addr_and_ports["port"],
        disaggregation_bootstrap_port=addr_and_ports["disaggregation_bootstrap_port"],
        engine_info_bootstrap_port=addr_and_ports["engine_info_bootstrap_port"],
        gated_launch_port=addr_and_ports["gated_launch_port"],
        random_seed=random_seed,
        **kwargs,
    )


class TestComputeEngineLaunchCmd:
    # [CLASS DOC] Tests of the rendered launch command string (split into tokens with shell rules).
    def test_the_command_preserves_every_interpreter_prefix_token(self):
        # [FUNCTION DOC] With a four-token interpreter prefix (executable, -O, -X, faulthandler),
        # the command's first tokens are the prefix followed by "-m sglang.launch_server".
        """Every interpreter option stays ordered immediately before the SGLang module invocation."""
        interpreter_prefix = [sys.executable, "-O", "-X", "faulthandler"]

        tokens = shlex.split(_cmd(interpreter_prefix=interpreter_prefix))

        assert tokens[: len(interpreter_prefix) + 2] == [*interpreter_prefix, "-m", "sglang.launch_server"]

    def test_a_cpu_only_controller_renders_the_worker_device(self):
        # [FUNCTION DOC] Parses the arguments after the three-token prefix (executable, -m,
        # module) back into server args; the device is "cuda".
        """A controller without an accelerator still renders a CUDA worker command."""
        parsed = parse_server_args_argv(shlex.split(_cmd())[3:])

        assert parsed.device == "cuda"

    def test_the_command_launches_sglang_with_the_allocated_addressing(self):
        # [FUNCTION DOC] The first three tokens are executable, -m, sglang.launch_server; the
        # parsed rest has host 10.0.0.1, port 30000, rendezvous address 10.0.0.1:20000, gated
        # launch port 20034 and the tiny model path as model path.
        """The rendered launch_server command carries the addr map."""
        tokens = shlex.split(_cmd())
        assert tokens[:3] == [sys.executable, "-m", "sglang.launch_server"]
        parsed = parse_server_args_argv(tokens[3:])
        assert parsed.host == "10.0.0.1" and parsed.port == 30000
        assert parsed.dist_init_addr == "10.0.0.1:20000"
        assert parsed.gated_launch_port == 20034
        assert parsed.model_path == str(tiny_model_path())

    def test_the_base_gpu_id_reaches_the_server_unchanged_under_a_visibility_mask(self, monkeypatch):
        # [FUNCTION DOC] With visible devices "4,5,6,7" and base GPU id 6, the parsed command has
        # base GPU id 6.
        """Whoever renders the command may see a different set of devices than the engine will."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

        parsed = parse_server_args_argv(shlex.split(_cmd(base_gpu_id=6))[3:])

        assert parsed.base_gpu_id == 6

    def test_a_bracketed_v6_host_is_stripped_for_the_server_but_kept_in_dist_addr(self):
        # [FUNCTION DOC] Host "[fd00::2]" port 31007 and rendezvous "[fd00::1]:15003": the parsed
        # command has host "fd00::2" (brackets removed) and rendezvous address unchanged.
        """sglang binds a bare v6 host while the rendezvous addr stays bracketed."""
        cmd = _cmd(addr_overrides=dict(host="[fd00::2]", port=31007, dist_init_addr="[fd00::1]:15003"))
        parsed = parse_server_args_argv(shlex.split(cmd)[3:])
        assert parsed.host == "fd00::2"
        assert parsed.dist_init_addr == "[fd00::1]:15003"

    def test_a_prefill_command_carries_the_bootstrap_port(self):
        # [FUNCTION DOC] A prefill worker with bootstrap port 20090 parses back with
        # disaggregation mode "prefill" and that bootstrap port.
        """PD-disaggregation prefill flags survive into the command."""
        cmd = _cmd(worker_type="prefill", addr_overrides=dict(disaggregation_bootstrap_port=20090))
        parsed = parse_server_args_argv(shlex.split(cmd)[3:])
        assert parsed.disaggregation_mode == "prefill"
        assert parsed.disaggregation_bootstrap_port == 20090

    def test_the_command_names_the_seed_its_actor_was_given(self):
        # [FUNCTION DOC] Random seed 4242 parses back as random seed 4242.
        """sglang draws its own seed unless the argv names one, so the seed must survive the rendering."""
        parsed = parse_server_args_argv(shlex.split(_cmd(random_seed=4242))[3:])
        assert parsed.random_seed == 4242

    def test_the_command_carries_the_api_key_from_args(self):
        # [FUNCTION DOC] Engine args with API key "secret" parse back with API key "secret".
        """--sglang-api-key reaches the server through the generic passthrough."""
        cmd = _cmd(args=make_engine_args(sglang_api_key="secret"))
        parsed = parse_server_args_argv(shlex.split(cmd)[3:])
        assert parsed.api_key == "secret"


class TestLoraTargetModules:
    # [CLASS DOC] Test that the adapter target selection reaches both the engine command and the
    # LoRA sync configuration.
    @pytest.mark.parametrize(
        "adapter_targets",
        [
            [f"model.layers.*.self_attn.{projection}_proj" for projection in ("q", "k", "v")],
            [f"model.layers.*.linear_attn.in_proj_{projection}" for projection in ("qkv", "z", "b", "a")],
            [
                "model.layers.*.linear_attn.in_proj_qkv",
                "model.layers.*.linear_attn.in_proj_z",
                "model.layers.*.linear_attn.in_proj_b",
                "model.layers.*.linear_attn.in_proj_a",
                "model.layers.*.linear_attn.out_proj",
            ],
            "all-linear",
        ],
        ids=["qkv", "gdn", "gdn-output", "inkling"],
    )
    @pytest.mark.parametrize("multi_lora", [False, True], ids=["single", "multi"])
    def test_adapter_selection_reaches_engine_and_sync_config(self, adapter_targets, multi_lora):
        # [FUNCTION DOC] Parametrized over four adapter-target shapes (attention q/k/v, linear
        # attention input projections, those plus the output projection, and the string
        # "all-linear") crossed with single/multi LoRA. Engine args have LoRA rank 16, alpha 32,
        # dropout 0, the targets, a separate HF LoRA targets list and 4 multi-LoRA adapters.
        # Asserts the parsed engine command's LoRA target modules equal the target set (the single
        # entry "all" for all-linear), and that the sync LoRA config built from the same targets
        # carries them verbatim as its target modules.
        args = make_engine_args(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.0,
            lora_adapter_targets=adapter_targets,
            hf_lora_targets=["model.language_model.layers.*.self_attn.q_proj"],
            multi_lora=multi_lora,
            multi_lora_n_adapters=4,
        )
        targets = parse_server_args_argv(shlex.split(_cmd(args=args))[3:]).lora_target_modules
        assert set(targets) == ({"all"} if adapter_targets == "all-linear" else set(adapter_targets))
        assert build_lora_config(args, target_modules=adapter_targets)["target_modules"] == adapter_targets


@dataclasses.dataclass
class _SglangWithTheGate:
    # [CLASS DOC] HELPER: Stub server-args record having a model path and a gated launch port field.
    model_path: str = ""
    gated_launch_port: int = 0


@dataclasses.dataclass
class _SglangWithoutTheGate:
    # [CLASS DOC] HELPER: Stub server-args record with only a model path (no gated launch port).
    model_path: str = ""


class TestTheLaunchGateSglangMustServe:
    # [CLASS DOC] Tests of the engine module's check that the installed SGLang supports the
    # launch-gate port.
    @staticmethod
    def _pretend_sglang_is(monkeypatch, server_args: type) -> None:
        # [FUNCTION DOC] HELPER: Replaces the engine module's server-args type with the given stub
        # (via monkeypatch) and clears the cache of the gate assertion function so it re-evaluates.
        monkeypatch.setattr(sglang_engine, "ServerArgs", server_args)
        _assert_launch_gate_served.cache_clear()

    def test_an_sglang_that_serves_the_gate_is_accepted(self, monkeypatch) -> None:
        # [FUNCTION DOC] With the stub that has the gate field, the gate assertion returns normally.
        """The run launches every engine through the gate, so the one field it needs is the whole check."""
        self._pretend_sglang_is(monkeypatch, _SglangWithTheGate)

        _assert_launch_gate_served()

    def test_an_sglang_without_the_gate_is_refused(self, monkeypatch) -> None:
        # [FUNCTION DOC] With the stub lacking the gate field, the gate assertion raises an
        # assertion error mentioning the --gated-launch-port option.
        """An sglang serving nothing on that port leaves each cell waiting out its whole activation
        deadline against an engine that is already up, so it has to be refused at spec time."""
        self._pretend_sglang_is(monkeypatch, _SglangWithoutTheGate)

        with pytest.raises(AssertionError, match="--gated-launch-port"):
            _assert_launch_gate_served()
