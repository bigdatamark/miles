# [MODULE DOC] Lookup of the theoretical peak bf16 throughput (in TFLOPS) of GPUs by device name,
# used for utilization estimates. Holds a table of known device families (A100, H100 incl. PCIe
# variant, H200, GH200, B200, B300, GB200, GB300), matching rules that tolerate arbitrary
# vendor name decoration, and a cached lookup for the local CUDA device.
from __future__ import annotations

import re
from functools import cache

_PEAK_BF16_TFLOPS: dict[str, float] = {
    "A100": 312.0,
    "H100": 989.0,
    "H100 PCIE": 756.0,
    "H200": 989.0,
    "GH200": 989.0,
    "B200": 2250.0,
    "B300": 2250.0,
    "GB200": 2500.0,
    "GB300": 2500.0,
}

_MATCH_ORDER: tuple[str, ...] = tuple(sorted(_PEAK_BF16_TFLOPS, key=len, reverse=True))


def _words(device_name: str) -> str:
    # [FUNCTION DOC] HELPER: Normalizes a device name: every run of non-alphanumeric characters
    # becomes one space, text is upper-cased and trimmed, and one space is added at each end so
    # whole-word matching can be done by searching for a key padded with spaces.
    return f" {re.sub(r'[^A-Za-z0-9]+', ' ', device_name).upper().strip()} "


def peak_bf16_tflops(device_name: str) -> float | None:
    # [FUNCTION DOC] Returns the peak bf16 TFLOPS for a device name, or none when no known family
    # matches. The name is normalized, then table keys are tried longest first (so the PCIe H100
    # key wins over plain H100 and GH200 is never mistaken for H200); a key matches only as a
    # whole run of words inside the name. Pure.
    name = _words(device_name)
    for key in _MATCH_ORDER:
        if f" {key} " in name:
            return _PEAK_BF16_TFLOPS[key]
    return None


@cache
def _current_device_name() -> str | None:
    # [FUNCTION DOC] HELPER: Imports torch lazily; returns none when CUDA is unavailable, else the
    # name of the current CUDA device. The result is cached for the process lifetime, so later
    # device changes are not noticed.
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name()


def local_peak_bf16_tflops() -> float | None:
    # [FUNCTION DOC] Peak bf16 TFLOPS of the local current GPU, or none if there is no CUDA device
    # or the device is not in the table.
    device_name = _current_device_name()
    return peak_bf16_tflops(device_name) if device_name else None
