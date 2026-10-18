# [MODULE DOC]
# Tests for the SGLang engine's private server-argument computation function: representation of
# the server-args record (a dataclass or a msgspec struct), the random seed (caller-chosen value
# versus group override), base GPU id passthrough regardless of the process's visible-device mask,
# device resolution (default cuda, generic device argument, group override), precedence of the
# group's SGLang overrides over args-derived defaults, and LoRA adapter ownership (loaded by the
# engine only when no trainer pushes weights).
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import msgspec
import pytest

from miles.backends.sglang_utils import sglang_engine
from miles.backends.sglang_utils.sglang_engine import _compute_server_args


def make_args(**overrides: object) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Returns a simple namespace of run arguments with defaults: a fake
    # model path, seed 0, 8 GPUs per node, 1 GPU per engine, no rollout offload, data/pipeline/
    # expert parallel sizes 1, static memory fraction 0.7, routing and indexer replay off, fp16
    # off, no adapter path, debug flags off, one multi-LoRA adapter, and q/k/v attention
    # projection LoRA targets. Keyword overrides replace defaults. Note the LoRA rank and device
    # are absent unless overridden.
    defaults = dict(
        hf_checkpoint="/fake/model",
        seed=0,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        offload_rollout=False,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        sglang_mem_fraction_static=0.7,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fp16=False,
        lora_adapter_path=None,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        multi_lora_n_adapters=1,
        lora_adapter_targets=[f"model.layers.*.self_attn.{projection}_proj" for projection in ("q", "k", "v")],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def compute(args: SimpleNamespace, **overrides: object) -> dict:
    # [FUNCTION DOC] HELPER: Calls the private server-args computation with the given args and
    # keyword defaults: node rank 0, distributed init address 127.0.0.1:1234, NCCL port 5000, host
    # 127.0.0.1, port 30000, worker type "regular", no disaggregation or engine-info bootstrap
    # port, base GPU id 0, no SGLang overrides, no GPU-count override, gated launch port 30001,
    # random seed 0. Keyword overrides replace those defaults. Returns the computed mapping.
    kwargs = dict(
        node_rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=5000,
        host="127.0.0.1",
        port=30000,
        worker_type="regular",
        disaggregation_bootstrap_port=None,
        base_gpu_id=0,
        engine_info_bootstrap_port=None,
        sglang_overrides=None,
        num_gpus_per_engine=None,
        gated_launch_port=30001,
        random_seed=0,
    )
    kwargs.update(overrides)
    return _compute_server_args(args, **kwargs)


@pytest.mark.parametrize(
    "record_factory", [dataclasses.make_dataclass, msgspec.defstruct], ids=["dataclass", "msgspec"]
)
def test_server_args_representation_preserves_launch_values(monkeypatch, record_factory):
    # [FUNCTION DOC] Parametrized over a dataclass and a msgspec struct as the server-args type.
    # Replaces the engine module's server-args type with a generated record of three fields (gated
    # launch port, static memory fraction, random seed). Computes with seed 7 and overrides that
    # set the seed to 99 plus an unknown field. Asserts the result is exactly: gated port 30001,
    # memory fraction 0.7, seed 99 (override wins), and device "cuda" (always added); the unknown
    # override field is dropped.
    """Both record shapes must yield the same launch values, plus the device this renderer always pins."""
    server_args_type = record_factory(
        "ServerArgs",
        [("gated_launch_port", int), ("mem_fraction_static", float), ("random_seed", int)],
    )
    monkeypatch.setattr(sglang_engine, "ServerArgs", server_args_type)

    result = compute(make_args(), random_seed=7, sglang_overrides={"random_seed": 99, "unknown_field": True})

    assert result == {"gated_launch_port": 30001, "mem_fraction_static": 0.7, "random_seed": 99, "device": "cuda"}


class TestRandomSeed:
    # [CLASS DOC] Tests of how the random seed reaches the computed server arguments.
    def test_the_caller_chosen_seed_reaches_the_engine(self):
        # [FUNCTION DOC] With run seed 1234 and caller-chosen seed 7, the computed random seed is 7.
        """A seed sglang picks for itself makes a restarted engine replay a different RNG stream."""
        server_args = compute(make_args(seed=1234), random_seed=7)

        assert server_args["random_seed"] == 7

    def test_a_group_override_still_wins_over_the_seed_the_caller_computed(self):
        # [FUNCTION DOC] A group override of the random seed to 99 beats the caller's seed 7.
        """A group pinning random_seed in its sglang_overrides must keep beating the derived number."""
        server_args = compute(make_args(seed=1234), random_seed=7, sglang_overrides={"random_seed": 99})

        assert server_args["random_seed"] == 99


class TestBaseGpuId:
    # [CLASS DOC] Tests that the base GPU id is passed through without remapping or validation
    # against this process's CUDA visibility mask.
    def test_the_given_base_gpu_id_is_used_verbatim_under_a_visibility_mask(self, monkeypatch):
        # [FUNCTION DOC] With the visible-devices environment variable set to "4,5,6,7" (via
        # monkeypatch), base GPU id 6 is returned as 6.
        """The id already names the engine's own device space, so remapping it here moves the engine."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

        server_args = compute(make_args(), base_gpu_id=6)

        assert server_args["base_gpu_id"] == 6

    def test_a_base_gpu_id_outside_this_processs_visibility_mask_is_not_rejected(self, monkeypatch):
        # [FUNCTION DOC] With visible devices "0,1", base GPU id 7 is accepted and returned as 7.
        """The engine's mask differs from this process's, so judging the id here fails a valid launch."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

        server_args = compute(make_args(), base_gpu_id=7)

        assert server_args["base_gpu_id"] == 7


class TestDeviceResolution:
    # [CLASS DOC] Tests of how the engine device is resolved (default "cuda" unless overridden).
    def test_generic_device_arg_takes_precedence_over_the_cuda_default(self):
        # [FUNCTION DOC] A run argument sglang_device="cpu" makes the computed device "cpu".
        """An explicit generic device remains authoritative over the controller-safe default."""
        server_args = compute(make_args(sglang_device="cpu"))

        assert server_args["device"] == "cpu"

    def test_group_device_override_takes_precedence_over_the_cuda_default(self):
        # [FUNCTION DOC] A group override device="xpu" makes the computed device "xpu".
        """A per-group device override remains authoritative over the controller-safe default."""
        server_args = compute(make_args(), sglang_overrides={"device": "xpu"})

        assert server_args["device"] == "xpu"


class TestSglangOverridePrecedence:
    # [CLASS DOC] Tests that group overrides beat every default derived from run arguments.
    """An override must win over every args-derived default, including the conditional ones."""

    def test_override_wins_over_conditional_args_defaults(self):
        # [FUNCTION DOC] With fp16 and both replay modes on, a dtype override "bfloat16" wins.
        args = make_args(fp16=True, use_rollout_routing_replay=True, use_rollout_indexer_replay=True)

        server_args = compute(args, sglang_overrides={"dtype": "bfloat16"})

        assert server_args["dtype"] == "bfloat16"

    def test_override_wins_over_lora_defaults(self):
        # [FUNCTION DOC] With LoRA rank 8, an override enable_lora=False wins (result is false).
        args = make_args(lora_rank=8)

        server_args = compute(args, sglang_overrides={"enable_lora": False})

        assert server_args["enable_lora"] is False

    @pytest.mark.parametrize("value", [0.5, 0.95])
    def test_override_wins_over_base_sglang_args(self, value):
        # [FUNCTION DOC] Parametrized over 0.5 and 0.95. The run's static memory fraction is 0.7;
        # an override of the memory fraction to the parametrized value wins.
        args = make_args(sglang_mem_fraction_static=0.7)

        server_args = compute(args, sglang_overrides={"mem_fraction_static": value})

        assert server_args["mem_fraction_static"] == value

    def test_no_overrides_keeps_args_derived_values(self):
        # [FUNCTION DOC] With fp16 on, LoRA rank 8 and no overrides: dtype is "float16", LoRA is
        # enabled and the memory fraction is 0.7.
        args = make_args(fp16=True, lora_rank=8)

        server_args = compute(args)

        assert server_args["dtype"] == "float16"
        assert server_args["enable_lora"] is True
        assert server_args["mem_fraction_static"] == 0.7


class TestAdapterOwnership:
    # [CLASS DOC] Tests of who loads the LoRA adapter: the trainer's weight sync or the engine.
    def test_a_trainer_run_leaves_the_adapter_to_the_first_weight_sync(self):
        # [FUNCTION DOC] With LoRA rank 8 and an adapter path in a normal trainer run, LoRA is
        # enabled but the computed arguments carry no LoRA paths.
        """An engine that also loaded the adapter from disk would serve stale weights if the sync were skipped."""
        server_args = compute(make_args(lora_rank=8, lora_adapter_path="/fake/adapter"))

        assert server_args["enable_lora"] is True
        assert not server_args.get("lora_paths")

    @pytest.mark.parametrize(
        "flag", ["debug_rollout_only", "debug_skip_weight_update"], ids=["rollout-only", "skip-sync"]
    )
    def test_without_a_trainer_push_the_engine_loads_the_adapter_itself(self, flag):
        # [FUNCTION DOC] Parametrized over the rollout-only and skip-weight-update debug flags.
        # With LoRA rank 8, an adapter path and the flag set, the computed LoRA paths are exactly
        # one entry "miles_lora=/fake/adapter" (adapter name equals path).
        server_args = compute(make_args(lora_rank=8, lora_adapter_path="/fake/adapter", **{flag: True}))

        assert server_args["lora_paths"] == ["miles_lora=/fake/adapter"]
