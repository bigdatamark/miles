# [MODULE DOC]
# Tests for the FSDP train actor's attention bridge during model construction: a non-"triton"
# attention implementation is passed to the model loader unchanged; "triton" is left unchanged
# off ROCm; on ROCm the model is built with the eager implementation (the HuggingFace loader does
# not know "triton" yet), the Triton bridge is registered in the library's attention function
# registry and selected on the model config afterwards. A recording stub replaces the model class.
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor


class _StubModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Trivial module with an empty config namespace.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the module and sets an empty config namespace.
        super().__init__()
        self.config = SimpleNamespace()


class _RecordingModelCls:
    # [CLASS DOC] HELPER: Fake model class whose pretrained loader records the checkpoint path and
    # keyword arguments and returns a stub model.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with no recorded keyword arguments.
        self.kwargs = None

    def from_pretrained(self, checkpoint_path, **kwargs):
        # [FUNCTION DOC] HELPER: Records the keyword arguments plus the checkpoint path under the
        # key checkpoint_path, and returns a new stub model.
        self.kwargs = dict(kwargs, checkpoint_path=checkpoint_path)
        return _StubModel()


def _actor(attn_implementation, model_cls):
    # [FUNCTION DOC] HELPER: Creates an FSDP train actor without running its constructor, with
    # arguments holding the given attention implementation and a model-class getter returning the
    # given class. Returns it.
    actor = object.__new__(FSDPTrainRayActor)
    actor.args = SimpleNamespace(attn_implementation=attn_implementation)
    actor._get_model_cls = lambda: model_cls
    return actor


def test_non_triton_attn_implementation_is_passed_through_unchanged():
    # [FUNCTION DOC] With "flash_attention_2" the build-with-bridge method (given the checkpoint
    # path and a null-context factory) passes that implementation and the path to the loader, and
    # reports 0 as its second return value (the patched count).
    model_cls = _RecordingModelCls()
    actor = _actor("flash_attention_2", model_cls)

    _, patched = actor._build_model_with_attn_bridge("/ckpt", nullcontext)

    assert model_cls.kwargs["checkpoint_path"] == "/ckpt"
    assert model_cls.kwargs["attn_implementation"] == "flash_attention_2"
    assert patched == 0


def test_triton_is_ignored_off_rocm(monkeypatch):
    # [FUNCTION DOC] With the HIP version patched to none (not ROCm), "triton" is passed to the
    # loader unchanged and the patched count is 0.
    monkeypatch.setattr(torch.version, "hip", None)
    model_cls = _RecordingModelCls()
    actor = _actor("triton", model_cls)

    _, patched = actor._build_model_with_attn_bridge("/ckpt", nullcontext)

    # Handed to from_pretrained unchanged, which rejects it — the behaviour a non-ROCm build
    # had before the bridge existed.
    assert model_cls.kwargs["attn_implementation"] == "triton"
    assert patched == 0


def test_triton_on_rocm_loads_eager_then_activates_the_bridge(monkeypatch):
    # [FUNCTION DOC] With the HIP version patched to "6.0.0" and any existing "triton" entry
    # removed from the transformers attention function registry (saved and restored in a finally
    # block: re-added if it existed, else removed). Asserts the loader received "eager", the
    # registry now has a "triton" entry, and the returned model's config selects "triton".
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    monkeypatch.setattr(torch.version, "hip", "6.0.0")
    previous = ALL_ATTENTION_FUNCTIONS.pop("triton", None)
    try:
        model_cls = _RecordingModelCls()
        actor = _actor("triton", model_cls)

        model, _ = actor._build_model_with_attn_bridge("/ckpt", nullcontext)

        # HF cannot construct with an implementation it does not know yet, so the model is
        # built eager and the bridge is registered and selected afterwards.
        assert model_cls.kwargs["attn_implementation"] == "eager"
        assert "triton" in ALL_ATTENTION_FUNCTIONS
        assert model.config._attn_implementation == "triton"
    finally:
        if previous is None:
            ALL_ATTENTION_FUNCTIONS.pop("triton", None)
        else:
            ALL_ATTENTION_FUNCTIONS["triton"] = previous
