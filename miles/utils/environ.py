# [MODULE DOC] Reads behavior switches and hardware-dependent defaults for environment variables.
# Holds a module flag recording that the legacy-rollout notice has been printed.
import os

_printed_legacy_rollout_v1 = False


def use_legacy_rollout_v1() -> bool:
    # [FUNCTION DOC] Reads MILES_USE_LEGACY_ROLLOUT_V1 (default "0") as an integer and returns
    # whether it is non-zero; a non-integer value raises a value error. The first time the result
    # is true, prints a one-line deprecation notice to standard output and sets the module flag so
    # it prints only once per process.
    result = bool(int(os.environ.get("MILES_USE_LEGACY_ROLLOUT_V1", "0")))

    global _printed_legacy_rollout_v1
    if result and not _printed_legacy_rollout_v1:
        print("MILES_USE_LEGACY_ROLLOUT_V1=1 is enabled: using the deprecated v1 rollout path")
        _printed_legacy_rollout_v1 = True

    return result


def default_fp8_block_scaling_fp32_scales() -> str:
    # [FUNCTION DOC] Returns the string "1" or "0" for use as an environment variable value.
    # Imports torch lazily. Without CUDA returns "1". Otherwise reads the compute capability of the
    # current device: major version 10 or higher (Blackwell) gives "0" (FP32 scales disabled),
    # lower gives "1".
    """Default for NVTE_FP8_BLOCK_SCALING_FP32_SCALES, decided by hardware.

    On Blackwell (SM100+), TE emulates the blockwise FP8 recipe with MXFP8,
    which requires power-of-two scales, so FP32 scales must stay disabled.
    """
    import torch

    if not torch.cuda.is_available():
        return "1"
    major, _minor = torch.cuda.get_device_capability()
    return "0" if major >= 10 else "1"
