# [MODULE DOC] Key/value compressor for DeepSeek-V4.1 attention layers with compression ratio 1 or 2.
# Unlike the V4 compressor it has no overlap, position bias, rotary or quantization inside: those
# are applied by the caller. Ratio 1 is a plain projection plus norm; ratio 2 is a learned
# softmax-gated weighted sum over consecutive pairs of tokens.
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.deepseek_v4.ops.norm import RMSNorm


class DeepSeekV41Compressor(nn.Module):
    # [CLASS DOC] Holds a bias-free bfloat16 key/value projection from hidden size to head dim, a gate
    # projection of the same shape (only for ratio 2), and an RMS norm over head dim whose scale is
    # kept in float32. Works on sequence-major tensors (sequence, batch, channels).
    def __init__(self, config: TransformerConfig, head_dim: int, compress_ratio: int):
        # [FUNCTION DOC] Asserts the ratio is 1 or 2. Creates the projections (gate only when ratio
        # is above one) and the norm using the config's hidden size and layer-norm epsilon. The
        # parameters are created in bfloat16 with default initialization; checkpoint loading is
        # expected to overwrite them.
        super().__init__()
        assert compress_ratio in (1, 2)
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        dim = config.hidden_size
        self.linear_wkv = nn.Linear(dim, head_dim, bias=False, dtype=torch.bfloat16)
        if compress_ratio > 1:
            self.linear_wgate = nn.Linear(dim, head_dim, bias=False, dtype=torch.bfloat16)
        self.norm = RMSNorm(head_dim, config.layernorm_epsilon)
        mark_keep_in_fp32(self.norm.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Input sequence, batch, hidden. Ratio 1: projects with the key/value weight
        # and applies the norm; output has the same sequence length. Ratio 2: truncates the sequence
        # to a multiple of the ratio (dropping any trailing token), upcasts to float32, computes the
        # key/value and gate projections in float32 (weights upcast), reshapes into groups of
        # ratio, takes a weighted sum over the ratio axis with softmax of the gate over that axis,
        # casts to bfloat16 and applies the norm. Output is groups, batch, head dim. No rotary,
        # quantization or position information is applied here.
        seqlen, bsz, _ = x.shape
        ratio = self.compress_ratio
        if ratio == 1:
            return self.norm(F.linear(x, self.linear_wkv.weight))
        groups = seqlen // ratio
        x = x[: groups * ratio].float()
        kv = F.linear(x, self.linear_wkv.weight.float())
        score = F.linear(x, self.linear_wgate.weight.float())
        kv = kv.view(groups, ratio, bsz, self.head_dim)
        score = score.view(groups, ratio, bsz, self.head_dim)
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        return self.norm(kv.to(torch.bfloat16))
