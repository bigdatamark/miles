# [MODULE DOC] HELPER: Custom rollout generate function used by the router-equivalence end-to-end
# test. It wraps the stock single-turn generate function and, when an environment variable names a
# dump file, appends one JSON line per returned sample with the fields compared between router
# variants. The module also copies the wrapped function's optional add-arguments hook (or none)
# onto the new function so command-line argument registration still works. If the environment
# variable is unset the wrapper logs a warning and dumps nothing.
"""Custom generate function for router-equivalence e2e test.

Wraps the stock ``single_turn.generate`` and, after each rollout, appends a
JSON record to ``$MILES_ROUTER_EQ_DUMP_PATH`` capturing the fields that
must match byte-for-byte between two runs using different routers:

- ``tokens`` (the full input + output token ids)
- ``rollout_log_probs`` (per-output-token logprob)
- ``rollout_routed_experts`` (shape + base64-encoded int32 bytes)

The dump is later loaded by ``test_r3_router_equivalence`` and diffed
between a ``--use-miles-router`` run and a sglang-router run.
"""

import base64
import json
import logging
import os
from pathlib import Path

import numpy as np

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as _base_generate
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_DUMP_PATH_ENV = "MILES_ROUTER_EQ_DUMP_PATH"


def _dump_sample(sample: Sample) -> dict:
    # [FUNCTION DOC] HELPER: Converts a sample into a JSON-serializable mapping with keys: index;
    # status as text; response length; tokens as a list (or none); rollout log probs as a list (or
    # none); routed-experts shape as a list of ints and routed-experts payload as base64 text of the
    # C-contiguous int32 bytes (both none when the sample has no routed experts). The routed
    # experts are cast to 32-bit integers before encoding.
    re = sample.rollout_routed_experts
    if re is not None:
        arr = np.ascontiguousarray(re, dtype=np.int32)
        experts_shape = list(arr.shape)
        experts_b64 = base64.b64encode(arr.tobytes()).decode("ascii")
    else:
        experts_shape = None
        experts_b64 = None

    return {
        "index": sample.index,
        "status": str(sample.status),
        "response_length": sample.response_length,
        "tokens": list(sample.tokens) if sample.tokens is not None else None,
        "rollout_log_probs": list(sample.rollout_log_probs) if sample.rollout_log_probs is not None else None,
        "rollout_routed_experts_shape": experts_shape,
        "rollout_routed_experts_b64": experts_b64,
    }


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # [FUNCTION DOC] Awaits the stock single-turn generate function on the same input. If the dump
    # path environment variable is unset or empty, logs a warning and returns the output unchanged.
    # Otherwise normalizes the output's samples to a list (a single sample is wrapped), creates the
    # dump file's parent directory, appends one JSON line per sample (append mode, so repeated calls
    # accumulate), and returns the original output unchanged. Side effects: filesystem writes.
    out = await _base_generate(input)

    dump_path = os.environ.get(_DUMP_PATH_ENV)
    if not dump_path:
        logger.warning("%s not set; not dumping", _DUMP_PATH_ENV)
        return out

    samples = out.samples if isinstance(out.samples, list) else [out.samples]

    Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "a") as f:
        for s in samples:
            f.write(json.dumps(_dump_sample(s)) + "\n")

    return out


generate.add_arguments = getattr(_base_generate, "add_arguments", None)
