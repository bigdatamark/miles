# [MODULE DOC] HELPER: Test helper that fabricates Ray node ids. Ray only accepts hexadecimal
# node ids, so the helper renders an integer index as a zero padded lower case hex string of a
# fixed width of 56 digits.
from __future__ import annotations

_NODE_ID_HEX_DIGITS = 56


def fake_ray_node_id(index: int) -> str:
    # [FUNCTION DOC] HELPER: Returns the index formatted as lower case hex, left padded with zeros
    # to 56 digits. Pure; a non-negative index is expected (a negative one would produce a sign
    # character and an invalid id).
    """Ray validates node ids as hex, so a readable "node-0" is rejected before scheduling."""
    return f"{index:0{_NODE_ID_HEX_DIGITS}x}"
