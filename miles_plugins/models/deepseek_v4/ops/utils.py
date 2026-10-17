# [MODULE DOC] HELPER: Shared activation transform for DeepSeek-V4: a scaled Hadamard rotation. The
# fast Hadamard library is an optional import; when missing, the transform fails at call time.
import torch

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Asserts the input is bfloat16 and the Hadamard library is installed, then
    # applies the Hadamard transform along the last dim scaled by the inverse square root of that
    # dimension (orthonormal). Returns a tensor of the same shape and dtype. The last dim must be a
    # size supported by the library (power of two).
    """Scaled Hadamard transform used to redistribute activation energy before
    QAT. Consumed by both the attention compressor and the DSA indexer.
    """
    assert x.dtype == torch.bfloat16
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)
