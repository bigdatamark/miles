# [MODULE DOC] HELPER: Shared helpers for fast tests that inspect training command-line strings,
# especially for split-deployment runs (separate trainer and inference controllers) versus unsplit
# ones. Defines the flag names whose values legitimately differ between the commands of one split
# run, and between a split run and an unsplit run, plus small functions to read flag values out of
# a command-line string and to drop the differing flags for comparison. Pure functions; no I/O.
import shlex

from examples.infra_features.split_deployment.address_book import INIT_EXPECTED_NUM_CELLS_FLAG

from miles.ray.specs.inference import INFERENCE_CONTROLLER_ADDR_FLAG
from miles.ray.specs.train import TRAINER_CONTROLLER_ADDRS_FLAG
from miles.utils.external_utils.command_utils.common import MOONCAKE_INIT_KWARGS_FLAG, ArgvManipulator

ROLLOUT_NUM_GPUS_FLAG: str = "--rollout-num-gpus"

FLAGS_A_COMMAND_OF_ONE_SPLIT_RUN_MAY_DIFFER_ON: tuple[str, ...] = (
    INFERENCE_CONTROLLER_ADDR_FLAG,
    INIT_EXPECTED_NUM_CELLS_FLAG,
    TRAINER_CONTROLLER_ADDRS_FLAG,
    ROLLOUT_NUM_GPUS_FLAG,
)

FLAGS_A_SPLIT_RUN_MAY_DIFFER_FROM_AN_UNSPLIT_ONE_ON: tuple[str, ...] = (
    *FLAGS_A_COMMAND_OF_ONE_SPLIT_RUN_MAY_DIFFER_ON,
    MOONCAKE_INIT_KWARGS_FLAG,
)


def value_of(train_args: str, flag: str) -> str:
    # [FUNCTION DOC] HELPER: Return the single value of a flag in a command-line string.
    # Splits the string with shell-style quoting rules and looks the flag up through the command
    # utilities' argument manipulator. Asserts the flag is declared exactly once (assertion error
    # names the count otherwise) and returns its one value token.
    values = ArgvManipulator.get(shlex.split(train_args), flag)
    assert len(values) == 1, f"{flag} is declared {len(values)} time(s) in these arguments"
    return values[0]


def values_after(train_args: str, flag: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Return all value tokens following a flag, up to the next token that
    # begins with two dashes (or the end). Splits with shell-style quoting rules and starts after
    # the first occurrence of the flag. Raises a value error if the flag is absent. Only the first
    # occurrence is considered.
    tokens = shlex.split(train_args)
    kept: list[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        kept.append(token)
    return kept


def shared_argv(train_args: str, *, differing_flags: tuple[str, ...]) -> list[str]:
    # [FUNCTION DOC] HELPER: Return the tokens of a command line with the given flags and their
    # values removed. Scans tokens in order: a token equal to a differing flag turns skipping on and
    # is dropped; any other token starting with two dashes turns skipping off and is kept; any
    # other token is kept only while not skipping. So values attached to a differing flag are
    # dropped until the next flag. Returns the kept tokens in order, to compare commands.
    kept: list[str] = []
    skipping = False
    for token in shlex.split(train_args):
        if token in differing_flags:
            skipping = True
        elif token.startswith("--"):
            skipping = False
            kept.append(token)
        elif not skipping:
            kept.append(token)
    return kept
