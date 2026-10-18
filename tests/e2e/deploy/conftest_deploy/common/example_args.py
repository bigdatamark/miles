# [MODULE DOC]
# HELPER: Argument-building helpers shared by deploy end-to-end scenarios that wrap an example
# launcher. Provides: building an example's script-argument object from a base execute-train
# configuration plus overrides; building the deterministic-run training arguments string
# (deterministic rollout arguments, debug dump arguments, deterministic collectives, radix cache
# disabled, and train environment variables of the fault-tolerance test mode); checking that the
# parallelism a test mode declares matches the argument string the example actually built;
# and zeroing or replacing the value of a flag in a training argument string. A module constant
# holds the weight-decay flag name.
import dataclasses
import shlex
from typing import TypeVar

from tests.e2e.ft.conftest_ft.execution import DETERMINISTIC_ROLLOUT_ARGS, get_debug_dump_args, get_train_env_vars_arg
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import ArgvManipulator

WEIGHT_DECAY_FLAG: str = "--weight-decay"

ScriptArgsT = TypeVar("ScriptArgsT", bound=ExecuteTrainConfig)


def build_script_args(
    config: ExecuteTrainConfig, *, script_args_class: type[ScriptArgsT], **overrides: object
) -> ScriptArgsT:
    # [FUNCTION DOC] HELPER: Convert the execute-train configuration dataclass to a field mapping
    # and construct the given script-arguments class from it plus the keyword overrides. Raises a
    # type error (from the constructor) if an override duplicates a field of the configuration or
    # a field is unknown to the class. Returns the new script-arguments object.
    return script_args_class(**dataclasses.asdict(config), **overrides)


def build_deterministic_test_args(*, mode: FTTestMode, dump_dir: str, enable_dumper: bool) -> str:
    # [FUNCTION DOC] HELPER: Return one argument string, concatenated in order: the deterministic
    # rollout arguments, the debug dump arguments for the dump directory (dumper on or off), the
    # deterministic-collective debug flag, the flag disabling the SGLang radix cache, and the
    # environment-variable argument for the given test mode in deterministic form. Each piece is
    # expected to end with a trailing space.
    return (
        DETERMINISTIC_ROLLOUT_ARGS
        + get_debug_dump_args(dump_dir=dump_dir, enable_dumper=enable_dumper)
        + "--debug-deterministic-collective "
        + "--sglang-disable-radix-cache "
        + get_train_env_vars_arg(mode, deterministic=True)
    )


def assert_example_parallelism_matches(mode: FTTestMode, *, train_args: str) -> None:
    # [FUNCTION DOC] HELPER: Check that the topology a test mode declares is what the example's
    # training argument string really contains. Splits both strings into shell words. For every
    # declared word starting with two dashes: assert the flag is defined in the built arguments;
    # then, if the next declared word exists and is not itself a flag, assert the built values of
    # that flag are exactly the single declared value. Flags declared without a following value
    # (or followed by another flag) are only checked for presence. Raises assertion errors saying
    # the mode describes a topology nobody launched.
    argv = shlex.split(train_args)
    declared = shlex.split(mode.parallel_args)
    token_after_flag = dict(zip(declared, declared[1:], strict=False))

    for flag in [one for one in declared if one.startswith("--")]:
        assert ArgvManipulator.is_defined(argv, flag), (
            f"this scenario's mode declares the parallelism {mode.parallel_args!r}, while the example builds no "
            f"{flag} at all: the mode describes a topology nobody launched"
        )

        value = token_after_flag.get(flag, "")
        if not value or value.startswith("--"):
            continue

        built_values = ArgvManipulator.get(argv, flag)
        assert built_values == [value], (
            f"this scenario's mode declares the parallelism {mode.parallel_args!r}, while the example builds "
            f"{flag} as {built_values}: the mode describes a topology nobody launched"
        )


def without_weight_decay(train_args: str) -> str:
    # [FUNCTION DOC] HELPER: Return the training argument string with the weight-decay flag's value
    # replaced by "0" (added if the flag is not present, per the argument manipulator's set
    # behavior).
    return with_replaced_value(train_args, flag=WEIGHT_DECAY_FLAG, value="0")


def with_replaced_value(train_args: str, *, flag: str, value: str) -> str:
    # [FUNCTION DOC] HELPER: Split the argument string into shell words, set the flag to the value
    # via the argument manipulator, re-join with shell quoting, and append one trailing space.
    # Returns the new string.
    return shlex.join(ArgvManipulator.set(shlex.split(train_args), flag, value)) + " "
