# [MODULE DOC]
# Tests for FSDP argument loading with a YAML config file: precedence is dataclass default, then
# config file, then command line (for numbers and for booleans in both directions); unknown config
# keys raise a value error naming the key (with a "did you mean" suggestion for near misses);
# valid configs pass the key check; fields the config does not mention keep their defaults. The
# process argument vector is faked via monkeypatch.
import sys
from pathlib import Path

import pytest
import yaml

from miles.backends.fsdp_utils.arguments import FSDPArgs, load_fsdp_args


def _config(tmp_path: Path, **entries) -> str:
    # [FUNCTION DOC] HELPER: Writes the keyword entries as a YAML mapping to fsdp.yaml in the
    # temporary directory and returns the file path as a string.
    path = tmp_path / "fsdp.yaml"
    path.write_text(yaml.safe_dump(entries))
    return str(path)


def _load(monkeypatch: pytest.MonkeyPatch, *argv: str):
    # [FUNCTION DOC] HELPER: Patches the process argument vector to the program name plus the
    # given arguments (via monkeypatch) and returns the loaded FSDP arguments.
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return load_fsdp_args()


def test_config_beats_the_dataclass_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # [FUNCTION DOC] Asserts the dataclass learning rate default is not 5e-5, then a config with
    # 5e-5 yields 5e-5.
    assert FSDPArgs.lr != 5e-5
    args = _load(monkeypatch, "--config", _config(tmp_path, lr=5e-5))
    assert args.lr == 5e-5


def test_cli_beats_the_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # [FUNCTION DOC] Config learning rate 5e-5 with CLI learning rate 7e-5 yields 7e-5.
    args = _load(monkeypatch, "--config", _config(tmp_path, lr=5e-5), "--lr", "7e-5")
    assert args.lr == 7e-5


def test_config_toggles_bools_both_ways(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # [FUNCTION DOC] Asserts dataclass defaults (gradient checkpointing false, FP32 master true),
    # then a config setting gradient checkpointing true and FP32 master false yields true and false.
    assert FSDPArgs.gradient_checkpointing is False
    assert FSDPArgs.keep_fp32_master is True
    args = _load(
        monkeypatch,
        "--config",
        _config(tmp_path, gradient_checkpointing=True, keep_fp32_master=False),
    )
    assert args.gradient_checkpointing is True
    assert args.keep_fp32_master is False


def test_cli_beats_the_config_for_bools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # [FUNCTION DOC] Config sets gradient checkpointing false while the CLI flag turns it on: the
    # result is true.
    args = _load(
        monkeypatch,
        "--config",
        _config(tmp_path, gradient_checkpointing=False),
        "--gradient-checkpointing",
    )
    assert args.gradient_checkpointing is True


def test_a_key_the_parser_does_not_know_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # [FUNCTION DOC] A config with an unknown key "some_plugin_knob" makes loading raise a value
    # error naming that key.
    with pytest.raises(ValueError, match="some_plugin_knob"):
        _load(monkeypatch, "--config", _config(tmp_path, some_plugin_knob=42))


def test_a_misspelled_key_names_the_field_it_almost_matched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # [FUNCTION DOC] A config key "weigth_decay" makes loading raise a value error whose message
    # suggests the field weight_decay ("did you mean 'weight_decay'").
    with pytest.raises(ValueError, match="did you mean 'weight_decay'"):
        _load(monkeypatch, "--config", _config(tmp_path, weigth_decay=0.1))


def test_a_valid_config_is_not_tripped_by_the_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # [FUNCTION DOC] A config with learning rate 5e-5, weight decay 0.1 and gradient checkpointing
    # true loads without error and yields exactly those values.
    args = _load(monkeypatch, "--config", _config(tmp_path, lr=5e-5, weight_decay=0.1, gradient_checkpointing=True))
    assert (args.lr, args.weight_decay, args.gradient_checkpointing) == (5e-5, 0.1, True)


def test_fields_untouched_by_the_config_keep_their_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # [FUNCTION DOC] A config setting only the learning rate leaves weight decay and FP32 master
    # at their dataclass defaults.
    args = _load(monkeypatch, "--config", _config(tmp_path, lr=5e-5))
    assert args.weight_decay == FSDPArgs.weight_decay
    assert args.keep_fp32_master == FSDPArgs.keep_fp32_master


def test_no_config_still_lands_on_the_dataclass_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Loading with no arguments yields the dataclass defaults for learning rate and
    # weight decay.
    args = _load(monkeypatch)
    assert args.lr == FSDPArgs.lr
    assert args.weight_decay == FSDPArgs.weight_decay
