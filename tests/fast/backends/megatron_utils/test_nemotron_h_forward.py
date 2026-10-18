# [MODULE DOC]
# Tests for the loss-mask shim that the Nemotron-H Megatron-bridge plugin installs on the Mamba
# model's forward method: forward calls without multi-token-prediction labels must pass through
# unchanged, and labels must be shifted exactly once without mutating the caller's tensor or
# mapping. The Mamba model module is replaced by a stub exposing a test-supplied model class.
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def install_shim(monkeypatch: pytest.MonkeyPatch, model_class: type) -> None:
    # [FUNCTION DOC] HELPER: Registers (via monkeypatch) a stub Mamba model module exposing the
    # given class, loads the plugin file under a private name from the repository path four
    # directories above this file, executes it, and calls its shim installer twice to verify
    # idempotence (the class's forward method is wrapped in place). Raises an assertion error if
    # the module spec cannot be created.
    stub = ModuleType("megatron.core.models.mamba")
    stub.MambaModel = model_class
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    path = Path(__file__).resolve().parents[4] / "miles_plugins/megatron_bridge/nemotron_h.py"
    spec = importlib.util.spec_from_file_location("nemotron_h_forward_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._install_mamba_model_loss_mask_shim()
    module._install_mamba_model_loss_mask_shim()


@pytest.mark.parametrize("mtp_kwargs", [None, {}, {"mtp_labels": None}])
def test_forward_without_mtp_keyword(monkeypatch: pytest.MonkeyPatch, mtp_kwargs: dict[str, None] | None) -> None:
    # [FUNCTION DOC] Parametrized over the extra multi-token-prediction keyword being absent
    # (none), an empty mapping, or a mapping whose labels are none. Installs the shim on a model
    # whose forward returns its tokens and loss mask, calls forward with tokens, mask and the
    # parametrized keyword, and asserts the very same tokens and mask objects come back (the shim
    # strips the keyword and does not alter data).
    class Model:
        # [CLASS DOC] HELPER: Model stub whose forward accepts tokens and a keyword-only loss mask.
        def forward(self, tokens: torch.Tensor, *, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Returns the tokens and loss mask as a pair, unchanged.
            return tokens, loss_mask

    install_shim(monkeypatch, Model)
    tokens = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[0, 1, 1]])
    result = Model().forward(tokens, loss_mask=mask, mtp_kwargs=mtp_kwargs)
    assert result[0] is tokens
    assert result[1] is mask


def test_forward_shifts_mtp_labels_once_without_mutating_input(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Installs the shim on a model whose forward returns its loss mask and MTP
    # labels. Calls forward with a ones mask and a mapping holding a 2-by-3 labels tensor. Asserts
    # the mask is returned as the same object, the returned labels equal the input rolled left by
    # one position along the last dimension (wrapping the first element to the end), the caller's
    # labels tensor is unchanged, and the caller's mapping still references the original tensor.
    class Model:
        # [CLASS DOC] HELPER: Model stub whose forward takes keyword-only loss mask and MTP labels.
        def forward(self, *, loss_mask: torch.Tensor, mtp_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Returns the loss mask and the labels as a pair, unchanged.
            return loss_mask, mtp_labels

    install_shim(monkeypatch, Model)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.ones_like(labels)
    mtp_kwargs = {"mtp_labels": labels}
    result = Model().forward(loss_mask=mask, mtp_kwargs=mtp_kwargs)
    assert result[0] is mask
    torch.testing.assert_close(result[1], torch.tensor([[2, 3, 1], [5, 6, 4]]))
    torch.testing.assert_close(labels, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert mtp_kwargs["mtp_labels"] is labels
