# [MODULE DOC] HELPER: Benchmarking generate functions. Currently one wrapper around the base generate
# that forces a long, random output length with end-of-sequence stopping disabled, useful for load and
# throughput experiments.
import logging
import random
from argparse import Namespace
from copy import deepcopy
from typing import Any

from miles.rollout.sglang_rollout import generate as _generate_base
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def generate_with_random_osl(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    # [FUNCTION DOC] HELPER: Legacy-style generate (args, sample, sampling parameters) returning the
    # sample. Copies the sampling parameters (input unchanged), sets ignore end-of-sequence to true and
    # sets the maximum new tokens to a random integer in [32768, 65536) (upper bound exclusive, drawn
    # per call; hard-coded, TODO in code to make configurable), calls the base sglang rollout generate,
    # logs the response length and returns its result.
    # TODO: make it configurable after we have an enhanced arg parser
    min_osl = 32 * 1024
    max_osl = 64 * 1024

    modified_sampling_params = deepcopy(sampling_params)
    modified_sampling_params["ignore_eos"] = True
    modified_sampling_params["max_new_tokens"] = random.randrange(min_osl, max_osl)

    ans = await _generate_base(args, sample, modified_sampling_params)

    logger.info(f"generate_with_random_osl {ans.response_length=}")
    return ans
