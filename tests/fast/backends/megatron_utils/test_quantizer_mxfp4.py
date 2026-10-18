# [MODULE DOC]
# Golden-value test for the MXFP4 dequantization utility (4-bit floating-point values packed two
# per byte, with one power-of-two 8-bit exponent scale per group).
import torch

from miles.utils.mxfp4 import dequantize_mxfp4


def test_dequantize_mxfp4_decodes_nibbles_and_e8m0_scales() -> None:
    # [FUNCTION DOC] Packs one row of four bytes (hex 10, 32, 54, 76; each byte holds the low
    # nibble first, giving codes 0..7) with two scale bytes 127 and 128 (exponent bias 127: scale
    # factors 1 and 2) and group size 4 (four values, i.e. two bytes, per scale). Expects a
    # bfloat16 row 0, 0.5, 1, 1.5 (first group, scale 1) then 4, 6, 8, 12 (second group, codes
    # 4..7 = 2, 3, 4, 6 times scale 2). Asserts exact equality (zero tolerances).
    """Golden values pin the wire convention a pack/unpack round trip cannot catch."""
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76]], dtype=torch.uint8)
    scales = torch.tensor([127, 128], dtype=torch.uint8)
    expected = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 4.0, 6.0, 8.0, 12.0]],
        dtype=torch.bfloat16,
    )

    actual = dequantize_mxfp4(packed, scales, group_size=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
