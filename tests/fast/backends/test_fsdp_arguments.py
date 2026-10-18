# [MODULE DOC]
# Tests for the FSDP command-line parser that is generated from the FSDP arguments dataclass:
# parser defaults equal dataclass defaults (and every field must have a plain default), boolean
# fields get a positive flag and a "--no-" negative flag so both true-default and false-default
# booleans can be set either way. The process argument vector is faked via monkeypatch.
import dataclasses
import sys

import pytest

from miles.backends.fsdp_utils.arguments import FSDPArgs, parse_fsdp_cli


def _parse(monkeypatch: pytest.MonkeyPatch, *argv: str):
    # [FUNCTION DOC] HELPER: Patches the process argument vector to the program name followed by
    # the given arguments (via monkeypatch) and returns the namespace parsed by the FSDP parser.
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return parse_fsdp_cli()


def test_cli_defaults_match_the_dataclass(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Parses no arguments. For every dataclass field asserts the field declares a
    # plain default (not the missing-value sentinel; otherwise the parser would register the
    # sentinel as the CLI default and the message tells to teach the parser about default
    # factories), and that the parsed value equals that default.
    args = _parse(monkeypatch)
    for field in dataclasses.fields(FSDPArgs):
        assert field.default is not dataclasses.MISSING, (
            f"{field.name} declares no plain default, so parse_fsdp_cli registers "
            f"the dataclasses.MISSING sentinel as its CLI default; teach the parser "
            f"about default_factory before adding a field like this"
        )
        assert getattr(args, field.name) == field.default, (
            f"CLI default for --{field.name.replace('_', '-')} is {getattr(args, field.name)!r}, "
            f"but the dataclass declares {field.default!r}"
        )


def test_true_default_bools_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] The negative flags for state-dict CPU offload, checkpoint LR scheduler use and
    # FP32 master keeping (all true by default) set those three fields to false.
    args = _parse(
        monkeypatch,
        "--no-fsdp-state-dict-cpu-offload",
        "--no-use-checkpoint-lr-scheduler",
        "--no-keep-fp32-master",
    )
    assert args.fsdp_state_dict_cpu_offload is False
    assert args.use_checkpoint_lr_scheduler is False
    assert args.keep_fp32_master is False


def test_false_default_bools_still_turn_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] The positive flags for gradient checkpointing and fp16 (false by default)
    # set those fields to true.
    args = _parse(monkeypatch, "--gradient-checkpointing", "--fp16")
    assert args.gradient_checkpointing is True
    assert args.fp16 is True


def test_every_bool_gets_both_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] For every dataclass field whose type is exactly boolean, derives the flag
    # name by replacing underscores with dashes and asserts the positive flag sets the field to
    # true and the "--no-" form sets it to false.
    for field in dataclasses.fields(FSDPArgs):
        if field.type is not bool:
            continue
        flag = field.name.replace("_", "-")
        assert _parse(monkeypatch, f"--{flag}").__dict__[field.name] is True
        assert _parse(monkeypatch, f"--no-{flag}").__dict__[field.name] is False
