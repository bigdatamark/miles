# [MODULE DOC] CPU test (stage-a CPU suite, estimated 2 seconds) of the GLM-5 attention's
# query/indexer split. The attention method that produces absorbed query/key/value tensors is
# extracted from the model source file by parsing it and executing only that function in a
# minimal namespace with stubbed collaborators, avoiding GPU-only Megatron imports. The test
# verifies the indexer receives the normalized query without gradients flowing into attention
# parameters, and that the main branch still trains the shared query norm.
"""Exercise the GLM-5 query/indexer split without GPU-only Megatron imports."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from torch import nn

from miles_plugins.models.normalization import rms_norm

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])


@pytest.fixture
def project_queries(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Parses the GLM-5 model source file, finds the method that
    # computes absorbed query/key/value tensors in the DSA multi-latent self-attention class,
    # and builds a namespace with torch, the rms norm function, a stub parallel-state object
    # (no context-parallel group, world size 1, rank 0) and an identity sequence-parallel gather.
    # Puts a stub module providing an identity fused rotary function into the module table for
    # the apex path (reverted by monkeypatch), compiles only that method as a module and executes
    # it in the namespace. Returns the resulting plain function (which takes the model object as
    # its first argument).
    path = Path(__file__).resolve().parents[2] / "miles_plugins/models/glm5/glm5.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DSAMLASelfAttention")
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_absorb_query_key_value_tensors"
    )
    namespace = {
        "torch": torch,
        "rms_norm": rms_norm,
        "parallel_state": SimpleNamespace(
            get_context_parallel_group=lambda: None,
            get_context_parallel_world_size=lambda: 1,
            get_context_parallel_rank=lambda: 0,
        ),
        "gather_from_sequence_parallel_region": lambda tensor, **kwargs: tensor,
    }
    monkeypatch.setitem(
        sys.modules,
        "apex.transformer.functional",
        SimpleNamespace(fused_apply_rotary_pos_emb_thd=lambda tensor, *args: tensor),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


class TupleLinear(nn.Linear):
    # [CLASS DOC] HELPER: Linear layer that mimics Megatron's convention of returning an
    # (output, bias) pair, with the second element always none.
    def forward(self, inputs):
        # [FUNCTION DOC] HELPER: Applies the linear map and returns (result, none).
        return super().forward(inputs), None


class FusedQueryProjection(TupleLinear):
    # [CLASS DOC] HELPER: Stand-in for the fused query up-projection (norm plus linear, four
    # inputs to eight outputs, no bias) with a fixed four-element norm weight parameter. After
    # each call it exposes the normalized input and the full output as attributes.
    def __init__(self, dtype, zero_centered):
        # [FUNCTION DOC] HELPER: Builds the 4-to-8 bias-free linear layer in the given dtype and a
        # layer norm weight parameter with fixed values; stores the zero-centered-gamma flag.
        super().__init__(4, 8, bias=False, dtype=dtype)
        self.layer_norm_weight = nn.Parameter(torch.tensor([0.3, 1.2, -0.7, 2.0], dtype=dtype))
        self.zero_centered = zero_centered

    def forward(self, inputs):
        # [FUNCTION DOC] HELPER: Computes an independent reference RMS norm of the inputs in
        # float32 (epsilon 1e-5, weight plus one when zero-centered), casts back to the input
        # dtype, stores it as the normalized attribute, applies the linear layer, stores and
        # returns the (output, none) pair as the output attribute.
        weight = self.layer_norm_weight.float() + int(self.zero_centered)
        # Independent reference for the fused query RMSNorm.
        self.normalized = torch.nn.functional.rms_norm(inputs.float(), (4,), weight=weight, eps=1e-5).to(inputs.dtype)
        self.output = super().forward(self.normalized)
        return self.output


class IndexerReached(Exception):
    # [CLASS DOC] HELPER: Sentinel exception raised by the stub indexer projection to stop the
    # extracted method as soon as the indexer input has been captured.
    pass


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_indexer_receives_normalized_query_without_attention_gradients(project_queries, dtype, zero_centered):
    # [FUNCTION DOC] Parametrized over float32 and bfloat16 and zero-centered gamma off/on. Seeds
    # the generator (17), builds random hidden states (3 by 1 by 4, scaled by 4, requiring grad),
    # a stub model namespace with tiny projections, identity norms, a stub indexer projection
    # that captures its input and output and raises the sentinel, and a stub rotary embedding.
    # Runs the extracted method expecting the sentinel, then asserts: the captured indexer input
    # has the test dtype, equals the reference normalized query, and does not require grad; after
    # back-propagating a squared-sum of the indexer output only the indexer weight has gradient
    # (hidden states, down projection, norm weight and up-projection weight have none); then
    # back-propagating through the main branch gives the shared norm weight and hidden states
    # gradients; finally, with skip-topk set, the method's trailing three returned values are all
    # none.
    torch.manual_seed(17)
    hidden = (torch.randn(3, 1, 4, dtype=dtype) * 4).requires_grad_()
    q_up = FusedQueryProjection(dtype, zero_centered)
    indexer = TupleLinear(4, 6, bias=False, dtype=dtype)
    captured = {}

    def project_indexer(inputs):
        # [FUNCTION DOC] HELPER: Stub indexer query projection; records the inputs and the
        # indexer layer's output in the captured mapping, then raises the sentinel exception.
        captured["inputs"] = inputs
        captured["output"], _ = indexer(inputs)
        raise IndexerReached

    kv_up = TupleLinear(3, 8, bias=False, dtype=dtype)
    kv_up.layer_norm_weight = nn.Parameter(torch.ones(3, dtype=dtype))
    model = SimpleNamespace(
        config=SimpleNamespace(
            sequence_parallel=False,
            kv_lora_rank=3,
            qk_pos_emb_head_dim=2,
            qk_head_dim=2,
            v_head_dim=2,
            layernorm_epsilon=1e-5,
            layernorm_zero_centered_gamma=zero_centered,
        ),
        rotary_pos_emb=SimpleNamespace(),
        linear_q_down_proj=TupleLinear(4, 4, bias=False, dtype=dtype),
        linear_kv_down_proj=TupleLinear(4, 5, bias=False, dtype=dtype),
        linear_q_up_proj=q_up,
        linear_kv_up_proj=kv_up,
        q_layernorm=nn.Identity(),
        kv_layernorm=nn.Identity(),
        num_attention_heads_per_partition=2,
        q_head_dim=4,
        skip_topk=False,
        wq_b=project_indexer,
    )

    class RotaryEmbedding:
        # [CLASS DOC] HELPER: Stub rotary embedding with a fixed sequence length of 3 and zero
        # embeddings.
        def get_rotary_seq_len(self, *args):
            # [FUNCTION DOC] HELPER: Ignores arguments and returns 3.
            return 3

        def __call__(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Ignores arguments; returns a zero tensor of shape 3 by 1 by 1
            # by 2 in the test dtype and the scale 1.0.
            return torch.zeros(3, 1, 1, 2, dtype=dtype), 1.0

    model.rotary_pos_emb = RotaryEmbedding()
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3]), cu_seqlens_kv=torch.tensor([0, 3]))
    with pytest.raises(IndexerReached):
        project_queries(model, hidden, None, None, packed)

    assert captured["inputs"].dtype == dtype
    torch.testing.assert_close(captured["inputs"], q_up.normalized.detach())
    assert not captured["inputs"].requires_grad
    captured["output"].float().square().sum().backward()
    assert indexer.weight.grad is not None
    assert indexer.weight.grad.abs().sum() > 0
    assert hidden.grad is None
    assert model.linear_q_down_proj.weight.grad is None
    assert q_up.layer_norm_weight.grad is None
    assert q_up.weight.grad is None

    # The main attention branch still trains its shared query norm.
    q_up.output[0].float().square().sum().backward()
    assert q_up.layer_norm_weight.grad is not None
    assert hidden.grad is not None

    model.skip_topk = True
    assert project_queries(model, hidden, None, None, packed)[3:] == (None, None, None)
