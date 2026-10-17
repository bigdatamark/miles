# [MODULE DOC] Sanity checks on the weight version bookkeeping: that the rollout side is told about
# published weight versions often enough, and that training samples carry only numeric versions
# stamped by miles (not the inference server's placeholder). Checks are skipped in debug and LoRA
# modes. All failures are assertion errors.
import re
from argparse import Namespace
from typing import TYPE_CHECKING

from miles.utils.lora.utils import is_lora_enabled

if TYPE_CHECKING:
    from miles.utils.types import Sample

SGLANG_DEFAULT_WEIGHT_VERSION = "default"

_NUMERIC_VERSION_PATTERN = re.compile(r"[0-9]+")

_MIN_ROLLOUTS_WITHOUT_PUBLISHED_WEIGHT_VERSION = 3


def max_rollouts_without_published_weight_version(args: Namespace) -> int:
    # [FUNCTION DOC] Returns the larger of 3 and the weight update interval plus one.
    return max(_MIN_ROLLOUTS_WITHOUT_PUBLISHED_WEIGHT_VERSION, args.update_weights_interval + 1)


def assert_weight_version_is_published(args: Namespace, *, rollouts_since_publish: int) -> None:
    # [FUNCTION DOC] Returns without checking if rollout-only, train-only, skip-weight-update debug
    # flags or LoRA are on. Otherwise asserts the number of rollouts served since the last version
    # publication does not exceed the maximum allowed, with a message explaining the driver must publish
    # the version after every weight update.
    if args.debug_rollout_only or args.debug_train_only or args.debug_skip_weight_update or is_lora_enabled(args):
        return

    assert rollouts_since_publish <= max_rollouts_without_published_weight_version(args), (
        f"the rollout executor served {rollouts_since_publish} rollouts without anyone calling "
        f"set_weight_version, so its notion of the served weight version is frozen while training advances; "
        f"weight staleness accounting is silently disabled. The driver must publish the version returned by "
        f"TrainerController.update_weights to the rollout executor after every weight update"
    )


def assert_samples_weight_version_sane(args: Namespace, samples: list["Sample"]) -> None:
    # [FUNCTION DOC] Returns without checking if rollout-only or skip-weight-update debug flags or LoRA
    # are on (note: train-only mode is not exempted here, unlike the publication check). Otherwise, for
    # every weight version span of every sample, asserts the version is not the inference server's
    # "default" placeholder and consists entirely of decimal digits.
    if args.debug_rollout_only or args.debug_skip_weight_update or is_lora_enabled(args):
        return

    for sample in samples:
        for span in sample.all_weight_version_spans:
            assert span.version != SGLANG_DEFAULT_WEIGHT_VERSION, (
                f"sample index={sample.index} tokens [{span.abs_start}, {span.abs_end}) were generated under "
                f"weight version {SGLANG_DEFAULT_WEIGHT_VERSION!r}, the sglang placeholder for an engine whose "
                f"weights were never updated; training data must only come from engines that received a weight update"
            )
            assert _NUMERIC_VERSION_PATTERN.fullmatch(span.version), (
                f"sample index={sample.index} tokens [{span.abs_start}, {span.abs_end}) carry weight version "
                f"{span.version!r}, which is not the numeric version miles stamps on weight updates; "
                f"the engine serving this sample got its weights from somewhere miles does not know about"
            )
