# [MODULE DOC] CPU-only unit tests for the FSDP precision policy: policy resolution from a model
# config and args (parameter, reduce and autocast dtypes, fp32 master switch, sync dtype resolver),
# the command-line default for keeping fp32 master weights, the Qwen3 dense formal true-on-policy
# contract (fp32 params with bf16 autocast, rejection of fp16 and of disabled fp32 master), the
# autocast forward context, and recording of per-parameter sync dtypes when casting to fp32 master.
"""Unit tests for the FSDP precision policy (CPU-only)."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils.adaptations.precision import (
    apply_fp32_master,
    precision_forward_context,
    resolve_precision_policy,
)
from miles.backends.fsdp_utils.arguments import load_fsdp_args, parse_fsdp_cli
from miles.backends.training_utils.data.rollout import _rollout_logprob_dtype
from miles.true_on_policy.contracts import QWEN3_DENSE_TRUE_ON_POLICY_V1


def test_resolve_precision_policy_uses_independent_fp32_master_switch_and_dtypes():
    # [FUNCTION DOC] Test: for a dense qwen3 config with bf16 and fp32 master on, the policy keeps
    # fp32 master, parameter dtype bf16 and reduce dtype fp32. For a different model type (glm4 moe
    # lite) with fp16 and fp32 master off, master is off, parameter dtype fp16, reduce dtype fp32,
    # and the policy equals the one resolved for the dense config under the same args (model type
    # does not matter outside the formal contract).
    dense = SimpleNamespace(model_type="qwen3")
    bf16_args = SimpleNamespace(fp16=False, keep_fp32_master=True)

    p = resolve_precision_policy(dense, bf16_args)
    assert p.keep_fp32_master
    assert p.param_dtype == torch.bfloat16 and p.reduce_dtype == torch.float32

    disabled = resolve_precision_policy(
        SimpleNamespace(model_type="glm4_moe_lite"),
        SimpleNamespace(fp16=True, keep_fp32_master=False),
    )
    assert not disabled.keep_fp32_master
    assert disabled.param_dtype == torch.float16 and disabled.reduce_dtype == torch.float32
    assert disabled == resolve_precision_policy(
        dense,
        SimpleNamespace(fp16=True, keep_fp32_master=False),
    )


def test_fp32_master_cli_defaults_enabled_and_can_be_disabled(monkeypatch):
    # [FUNCTION DOC] Test: parsing the FSDP command line with no flags enables fp32 master; passing
    # the negated flag disables it (argv is patched for each parse).
    monkeypatch.setattr(sys, "argv", ["miles"])
    assert parse_fsdp_cli().keep_fp32_master

    monkeypatch.setattr(sys, "argv", ["miles", "--no-keep-fp32-master"])
    assert not parse_fsdp_cli().keep_fp32_master


def test_fsdp_args_expose_effective_compute_precision(monkeypatch):
    # [FUNCTION DOC] Test: for no flags (bf16) and the fp16 flag, loads FSDP args with true-on-policy
    # mode forced on and asserts bf16 equals not fp16, the resolved policy parameter dtype and the
    # rollout log-prob dtype both equal the expected compute dtype (bf16 or fp16).
    for cli_args, expected_dtype in (([], torch.bfloat16), (["--fp16"], torch.float16)):
        monkeypatch.setattr(sys, "argv", ["miles", *cli_args])
        args = load_fsdp_args()
        args.true_on_policy_mode = True

        assert args.bf16 == (not args.fp16)
        assert resolve_precision_policy(None, args).param_dtype is expected_dtype
        assert _rollout_logprob_dtype(args) is expected_dtype


def test_qwen3_formal_true_on_policy_resolves_fp32_params_with_bf16_autocast():
    # [FUNCTION DOC] Test: with qwen3 dense, bf16, fp32 master, true-on-policy mode and the formal
    # Qwen3 dense contract name, the policy has fp32 parameter and reduce dtypes, bf16 autocast
    # dtype, fp32 master on, and a sync dtype resolver present.
    args = SimpleNamespace(
        fp16=False,
        keep_fp32_master=True,
        true_on_policy_mode=True,
        sglang_true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1.name,
    )

    policy = resolve_precision_policy(SimpleNamespace(model_type="qwen3"), args)

    assert policy.param_dtype is torch.float32
    assert policy.reduce_dtype is torch.float32
    assert policy.autocast_dtype is torch.bfloat16
    assert policy.keep_fp32_master
    assert policy.sync_dtype_resolver is not None


@pytest.mark.parametrize(
    ("model_type", "true_on_policy_mode", "contract"),
    [
        ("qwen3", False, QWEN3_DENSE_TRUE_ON_POLICY_V1.name),
        ("qwen3", True, None),
        ("qwen3_moe", True, QWEN3_DENSE_TRUE_ON_POLICY_V1.name),
    ],
)
def test_qwen3_formal_precision_does_not_leak_to_other_modes(model_type, true_on_policy_mode, contract):
    # [FUNCTION DOC] Test (parametrized over: qwen3 with the mode off, qwen3 with no contract,
    # qwen3 moe with the contract): the formal contract is not engaged, so the policy has bf16
    # parameter dtype, no autocast dtype, and no sync dtype resolver.
    policy = resolve_precision_policy(
        SimpleNamespace(model_type=model_type),
        SimpleNamespace(
            fp16=False,
            keep_fp32_master=True,
            true_on_policy_mode=true_on_policy_mode,
            sglang_true_on_policy_contract=contract,
        ),
    )

    assert policy.param_dtype is torch.bfloat16
    assert policy.autocast_dtype is None
    assert policy.sync_dtype_resolver is None


def test_qwen3_formal_true_on_policy_rejects_fp16():
    # [FUNCTION DOC] Test: resolving the policy under the formal contract with fp16 raises a value
    # error saying bf16 training is required.
    with pytest.raises(ValueError, match="requires bf16 training"):
        resolve_precision_policy(
            SimpleNamespace(model_type="qwen3"),
            SimpleNamespace(
                fp16=True,
                keep_fp32_master=True,
                true_on_policy_mode=True,
                sglang_true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1.name,
            ),
        )


def test_qwen3_formal_true_on_policy_rejects_disabled_fp32_master():
    # [FUNCTION DOC] Test: resolving the policy under the formal contract with fp32 master disabled
    # raises a value error saying fp32 master weights are required.
    with pytest.raises(ValueError, match="requires fp32 master weights"):
        resolve_precision_policy(
            SimpleNamespace(model_type="qwen3"),
            SimpleNamespace(
                fp16=False,
                keep_fp32_master=False,
                true_on_policy_mode=True,
                sglang_true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1.name,
            ),
        )


def test_precision_forward_context_uses_policy_autocast(monkeypatch):
    # [FUNCTION DOC] Test: replaces torch's autocast with a recording context manager, resolves the
    # formal Qwen3 policy, enters the precision forward context and asserts autocast was entered
    # exactly once with device type cuda and dtype bf16.
    entered = []

    @contextmanager
    def fake_autocast(*, device_type, dtype):
        # [FUNCTION DOC] HELPER: Recording stand-in for autocast; appends the device type and dtype
        # to the list and yields without doing anything.
        entered.append((device_type, dtype))
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    policy = resolve_precision_policy(
        SimpleNamespace(model_type="qwen3"),
        SimpleNamespace(
            fp16=False,
            keep_fp32_master=True,
            true_on_policy_mode=True,
            sglang_true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1.name,
        ),
    )

    with precision_forward_context(policy):
        pass

    assert entered == [("cuda", torch.bfloat16)]


def test_apply_fp32_master_records_sync_dtypes_before_cast():
    # [FUNCTION DOC] Test: builds a bf16 linear layer plus an extra fp32 parameter, applies the fp32
    # master cast with a resolver that returns fp32 for the weight and the checkpoint dtype
    # otherwise. Asserts all parameters become fp32 and the recorded sync dtypes are fp32 for weight,
    # bf16 for bias (dtype before the cast), and fp32 for the extra parameter.
    m = torch.nn.Linear(4, 4).to(torch.bfloat16)
    m.register_parameter("score_bias", torch.nn.Parameter(torch.zeros(4, dtype=torch.float32)))

    m = apply_fp32_master(
        m,
        lambda name, checkpoint_dtype: torch.float32 if name == "weight" else checkpoint_dtype,
    )

    assert all(p.dtype == torch.float32 for p in m.parameters())
    sync_dtypes = m._fsdp_sync_dtypes
    assert sync_dtypes["weight"] == torch.float32
    assert sync_dtypes["bias"] == torch.bfloat16
    assert sync_dtypes["score_bias"] == torch.float32
