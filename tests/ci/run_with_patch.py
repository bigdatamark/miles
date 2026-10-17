#!/usr/bin/env python3
# [MODULE DOC] Script-style wrapper (runs on import, no functions to call) for ROCm containers with
# an older aiter library. It imports the aiter shuffle ops module; if that module lacks the
# shuffle-scale function, it adds one that forwards to the older a16w4 variant (dropping the
# interleave argument) and prints a notice. Then it takes the test file path from the first
# command-line argument, rewrites the argument list to contain only that path, and executes the
# file's source in this module's namespace. Needs the aiter package; exits with an index error if
# no argument is given.
"""Run a miles test with aiter shuffle_scale monkey-patch for older containers."""

import sys

# Monkey-patch aiter.ops.shuffle before anything imports from sglang
import aiter.ops.shuffle as _shuffle

if not hasattr(_shuffle, "shuffle_scale"):
    import functools

    @functools.wraps(_shuffle.shuffle_scale_a16w4)
    def shuffle_scale(src, experts_cnt=None, is_guinterleave=False, gate_up=False):
        # [FUNCTION DOC] HELPER: Shim with the newer signature that calls the older a16w4 function
        # with the source, expert count and gate-up flag and returns its result.
        # KNOWN ISSUE: the interleave argument is accepted but ignored.
        """Compatibility shim: older aiter has shuffle_scale_a16w4 only."""
        return _shuffle.shuffle_scale_a16w4(src, experts_cnt, gate_up)

    _shuffle.shuffle_scale = shuffle_scale
    print("[patch] Added shuffle_scale compatibility shim to aiter.ops.shuffle", flush=True)

# Execute the target test file
test_file = sys.argv[1]
sys.argv = [test_file]
exec(open(test_file).read())
