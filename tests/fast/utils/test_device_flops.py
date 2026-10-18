# [MODULE DOC] Unit tests for the device peak-throughput lookup: peak_bf16_tflops maps a GPU device
# name to its peak BF16 TFLOPS from a table (case-insensitive substring matching where the longest
# matching key wins; PCIe H100 is lower than other H100 variants; unknown or empty names give none),
# and local_peak_bf16_tflops applies it to the current device name (monkeypatched here; none when
# there is no device). Pure CPU tests.
from __future__ import annotations

import pytest

from miles.utils import device_flops
from miles.utils.device_flops import local_peak_bf16_tflops, peak_bf16_tflops


@pytest.mark.parametrize(
    "device_name, expected",
    [
        ("NVIDIA A100-SXM4-80GB", 312.0),
        ("NVIDIA A100 80GB PCIe", 312.0),
        ("NVIDIA H100 80GB HBM3", 989.0),
        ("NVIDIA H200", 989.0),
        ("NVIDIA B200", 2250.0),
        ("NVIDIA GB300", 2500.0),
    ],
)
def test_known_devices_resolve(device_name, expected):
    # [FUNCTION DOC] Parametrized over six device names (A100 variants 312, H100/H200 989, B200 2250,
    # GB300 2500): the lookup returns the expected peak value.
    assert peak_bf16_tflops(device_name) == expected


@pytest.mark.parametrize(
    "device_name, expected",
    [
        ("NVIDIA H100 PCIe", 756.0),
        ("NVIDIA H100-PCIE-80GB", 756.0),
        ("NVIDIA H100 80GB HBM3", 989.0),
        ("NVIDIA H100 NVL", 989.0),
        ("NVIDIA H200 NVL", 989.0),
        ("NVIDIA A100-PCIE-40GB", 312.0),
    ],
)
def test_only_pcie_h100_clocks_below_its_family(device_name, expected):
    # [FUNCTION DOC] Parametrized: PCIe H100 names give 756; SXM-style H100, H100 NVL and H200 NVL
    # give 989; an A100 PCIe name still gives 312 (the PCIe distinction applies only to H100).
    assert peak_bf16_tflops(device_name) == expected


def test_grace_hopper_is_not_read_as_an_h200():
    # [FUNCTION DOC] The name "NVIDIA GH200 480GB" resolves to 989 (its own entry; the test name
    # says it must not be mistaken for an H200, though both values are equal).
    assert peak_bf16_tflops("NVIDIA GH200 480GB") == 989.0


def test_longer_key_wins_over_the_substring_it_contains():
    # [FUNCTION DOC] GB200 and GB300 resolve to 2500 although they contain the shorter keys B200 and
    # B300 (2250); a plain B300 name resolves to 2250. The longest matching key wins.
    assert peak_bf16_tflops("NVIDIA GB200") == 2500.0
    assert peak_bf16_tflops("NVIDIA GB300") == 2500.0
    assert peak_bf16_tflops("NVIDIA B300") == 2250.0


def test_matching_is_case_insensitive():
    # [FUNCTION DOC] A fully lower-case H100 device name still resolves to 989.
    assert peak_bf16_tflops("nvidia h100 80gb hbm3") == 989.0


def test_unknown_device_returns_none_rather_than_a_default():
    # [FUNCTION DOC] A consumer GPU name and an empty string both give none (no default guess).
    assert peak_bf16_tflops("NVIDIA GeForce RTX 4090") is None
    assert peak_bf16_tflops("") is None


def test_local_lookup_uses_the_table(monkeypatch):
    # [FUNCTION DOC] With the current-device-name function patched to return an H100 name, the local
    # lookup returns 989.
    monkeypatch.setattr(device_flops, "_current_device_name", lambda: "NVIDIA H100 80GB HBM3")
    assert local_peak_bf16_tflops() == 989.0


def test_none_without_a_device(monkeypatch):
    # [FUNCTION DOC] With the current-device-name function patched to return none, the local lookup
    # returns none.
    monkeypatch.setattr(device_flops, "_current_device_name", lambda: None)
    assert local_peak_bf16_tflops() is None
