# [MODULE DOC]
# Gpu tests (skipped without cuda or the fla package) of the KDA recurrence used by Kimi K3: packed
# sequence boundaries must be honored through the cumulative sequence lengths, and the forward must not
# mutate its inputs. Constants fix the kernel construction: 4 heads, head dim 128, gate lower bound -5.
"""Boundary and aliasing contracts of the KDA recurrence (``kda_recurrence``, which Kimi K3 runs).

Both cases guard failure modes that are silent in training: the run keeps
learning, only worse, so nothing short of a numerical check catches them.
"""

import pytest
import torch

pytest.importorskip("fla")
from miles_plugins.models.linear_attn import kda_recurrence  # noqa: E402


# KDA kernel construction, validated on H200: q/k/v/g scaled bf16, beta uniform on
# [0, 1] (passed as its logit, which the rule squashes back) because it is the delta
# rule's step size and falls outside the operator's domain when negative or above 1.
# (The "KDA goes non-finite with random weights" result is model-level -- random
# *projections* produce out-of-distribution q/k/v/g that compound across layers --
# and does not apply to the kernel in isolation.)
_KDA_HEADS = 4
_KDA_HEAD_DIM = 128
_KDA_LOWER_BOUND = -5.0


def _kda_inputs(seq_len: int, seed: int) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Seeds the global torch generator, then builds on the cuda device a mapping
    # with: q, k, v, g as bfloat16 normal values scaled by 0.5 with shape 1, seq len, heads, head dim;
    # beta logits as the logit of uniform values in [0,1] (clipped away from the ends by 1e-4), float32,
    # shape 1, seq len, heads; A log as float32 normal per head; and dt bias as float32 normal of size
    # heads times head dim. Side effect: reseeds the global generator.
    torch.manual_seed(seed)

    def activation() -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns one bfloat16 cuda tensor of normal values times 0.5.
        return torch.randn(1, seq_len, _KDA_HEADS, _KDA_HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.5

    return {
        "q": activation(),
        "k": activation(),
        "v": activation(),
        "g": activation(),
        "beta_logits": torch.logit(torch.rand(1, seq_len, _KDA_HEADS, device="cuda", dtype=torch.float32), eps=1e-4),
        "A_log": torch.randn(_KDA_HEADS, device="cuda", dtype=torch.float32),
        "dt_bias": torch.randn(_KDA_HEADS * _KDA_HEAD_DIM, device="cuda", dtype=torch.float32),
    }


def _run_kda(inputs: dict[str, torch.Tensor], cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Calls the recurrence with the mapping's tensors, the gate lower bound
    # constant, the given cumulative sequence lengths (none means no boundaries) and no context-parallel
    # context. Returns the output tensor.
    return kda_recurrence(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["beta_logits"],
        inputs["g"],
        inputs["A_log"],
        inputs["dt_bias"],
        gate_lower_bound=_KDA_LOWER_BOUND,
        cu_seqlens=cu_seqlens,
        cp_context=None,
    )


def _require_kda() -> None:
    # [FUNCTION DOC] HELPER: Skips the test if no cuda device is available or the fla kda ops module
    # cannot be imported.
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required to run the KDA kernels")
    pytest.importorskip("fla.ops.kda")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    # [FUNCTION DOC] HELPER: Returns the l2 norm of the difference divided by the l2 norm of the
    # expected tensor, computed in float32, as a python float.
    return ((actual - expected).float().norm() / expected.float().norm()).item()


def test_kda_applies_packed_sequence_boundaries() -> None:
    # [FUNCTION DOC] Packs two sequences of lengths 96 and 160 and runs the recurrence with boundaries
    # 0, 96, 256. For each sequence, slices the inputs (per-head vectors with one dimension are shared),
    # runs it alone, and asserts exact equality (zero tolerance) with the matching slice of the packed
    # output. Then runs without boundaries and asserts the second sequence's output differs by more than
    # 0.1 relative l2 (so the check is not vacuous) while the first sequence's output is exactly equal.
    """A packed batch must reproduce, per sequence, what that sequence produces alone.

    The rule routes boundaries through one of two mutually exclusive channels
    (``cu_seqlens`` off CP, ``cp_context`` under CP). A caller that selects the CP
    channel and drops ``cu_seqlens`` -- the shape the CP-only helper had before the
    two kernel paths were merged -- leaves the recurrence with no boundaries at all,
    so the whole packed microbatch becomes one sequence and every sample inherits its
    predecessors' state. Nothing raises.

    Thresholds are relative because the assertion must not be satisfiable by the
    output being small: these activations sit around 1e-3, so any absolute tolerance
    worth the name swallows the entire signal. Measured on H200: the per-sequence
    equality is exact, and dropping the boundaries moves the second sequence by 0.40
    while leaving the first at exactly 0 -- the first sequence has no predecessor to
    inherit from, which is what makes this leakage rather than noise.
    """
    _require_kda()
    first_len, second_len = 96, 160
    total = first_len + second_len
    packed = _kda_inputs(total, seed=0)
    packed_output = _run_kda(packed, torch.tensor([0, first_len, total], dtype=torch.int32, device="cuda"))

    for offset, length in ((0, first_len), (first_len, second_len)):
        alone = {
            name: tensor[:, offset : offset + length] if tensor.dim() > 1 else tensor
            for name, tensor in packed.items()
        }
        alone_output = _run_kda(alone, torch.tensor([0, length], dtype=torch.int32, device="cuda"))
        torch.testing.assert_close(packed_output[:, offset : offset + length], alone_output, rtol=0, atol=0)

    unbounded_output = _run_kda(packed, None)
    assert _relative_l2(unbounded_output[:, first_len:], packed_output[:, first_len:]) > 1e-1, (
        "dropping cu_seqlens barely moved the second sequence, so boundaries are not reaching the kernel "
        "and the equality above passed vacuously"
    )
    torch.testing.assert_close(unbounded_output[:, :first_len], packed_output[:, :first_len], rtol=0, atol=0)


def test_kda_does_not_mutate_its_inputs() -> None:
    # [FUNCTION DOC] Builds inputs of length 256 (seed 1), clones them, runs the recurrence with one
    # sequence boundary pair, and asserts every input tensor is exactly equal to its clone.
    """The forward must leave q/k/v/g/beta_logits untouched.

    SGLang's vendored ``chunk_kda`` overwrote its ``v`` buffer with a WY-representation
    intermediate. Any caller that reads an input back afterwards -- a manual
    ``save_for_backward`` re-derivation being the case that bit us -- then
    differentiates at a point the forward never evaluated, which put the q/k/g/beta
    gradients ~110x low and near-orthogonal. Exact comparison: aliasing is a
    mutation, not a rounding.
    """
    _require_kda()
    seq_len = 256
    inputs = _kda_inputs(seq_len, seed=1)
    before = {name: tensor.clone() for name, tensor in inputs.items()}

    _run_kda(inputs, torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"))

    for name, tensor in inputs.items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0, msg=f"KDA forward mutated input {name}")
