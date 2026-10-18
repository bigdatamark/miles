# [MODULE DOC]
# CPU tests comparing the first-fit-decreasing bin packer with the plain first-fit packer from the
# sequence-length balancing module. Both take a list of sample lengths and a bin capacity and return
# bins as lists of sample indices. Tests check the shared contract (each sample packed once, bins
# within capacity unless a single oversized sample), that decreasing order never uses more bins, and
# that it saves at least four percent of bins on a mid-length distribution. All use seeded random
# generators, so results are deterministic.
"""CPU tests for first-fit-decreasing packing and its bin-count gain over first-fit."""

from __future__ import annotations

import random

from miles.utils.seqlen_balancing import first_fit_decreasing_pack, first_fit_pack


def _check_valid(bins, lengths, cap):
    # [FUNCTION DOC] HELPER: Assert a packing is valid: the sorted concatenation of all bins equals
    # every sample index exactly once, and every bin whose total length exceeds the capacity holds
    # only one sample (an oversized singleton).
    seen = sorted(i for b in bins for i in b)
    assert seen == list(range(len(lengths))), "every sample packed exactly once"
    for b in bins:
        total = sum(lengths[i] for i in b)
        assert total <= cap or len(b) == 1, f"bin over cap with >1 sample: {total}"


def test_ffd_same_contract_as_first_fit():
    # [FUNCTION DOC] Fifty seeded random trials with 1 to 64 samples, capacity 50 to 400 and
    # lengths up to twice the capacity (so oversized samples occur); assert the decreasing packer
    # output passes the validity check.
    rng = random.Random(3)
    for _ in range(50):
        n = rng.randint(1, 64)
        cap = rng.randint(50, 400)
        lengths = [rng.randint(1, cap * 2) for _ in range(n)]  # includes oversized
        _check_valid(first_fit_decreasing_pack(lengths, cap), lengths, cap)


def test_ffd_never_more_bins_than_first_fit():
    # [FUNCTION DOC] Two hundred seeded random trials with 2 to 96 samples, capacity 100 to 1000
    # and lengths within capacity; assert the decreasing packer never uses more bins than the
    # plain first-fit packer.
    rng = random.Random(11)
    for _ in range(200):
        n = rng.randint(2, 96)
        cap = rng.randint(100, 1000)
        lengths = [rng.randint(1, cap) for _ in range(n)]
        ff = len(first_fit_pack(lengths, cap))
        ffd = len(first_fit_decreasing_pack(lengths, cap))
        assert ffd <= ff, f"FFD produced more bins ({ffd}) than FF ({ff})"


def test_ffd_gain_on_mid_length_distribution():
    # [FUNCTION DOC] Over 500 seeded trials of 32 samples with lengths 2000 to 6000 and capacity
    # 9216, sum bin counts for both packers; assert the decreasing packer total is at most 96
    # percent of the first-fit total (at least a four percent saving).
    """FFD's gain concentrates where item sizes are a sizable fraction of the
    cap (long-response RL with roomy token budgets): measured ~3-7% fewer bins
    for mean-length ~cap/4..cap/2 regimes, 0% for short-response regimes where
    both packers are already optimal. Assert the mid regime keeps a >=4% edge."""
    rng = random.Random(42)
    cap = 9216
    total_ff = total_ffd = 0
    for _ in range(500):
        lengths = [rng.randint(2000, 6000) for _ in range(32)]
        total_ff += len(first_fit_pack(lengths, cap))
        total_ffd += len(first_fit_decreasing_pack(lengths, cap))
    assert total_ffd <= total_ff * 0.96, f"expected >=4% bin savings, got FF={total_ff} FFD={total_ffd}"
