# [MODULE DOC] Parallelism layout description for the run_megatron tool: a frozen configuration of
# tensor, pipeline, context, expert and expert-tensor parallel sizes with derived values, plus a
# parser for layout strings written like command-line flags. Field names are listed in a
# module constant in the order tp, pp, cp, ep, etp.
"""Parallel configuration utilities for run_megatron CLI."""

from __future__ import annotations

import argparse
import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles.utils.debug_utils.run_megatron.cli.commands.args import RunArgs

_FIELD_NAMES: tuple[str, ...] = ("tp", "pp", "cp", "ep", "etp")


@dataclasses.dataclass(frozen=True)
class ParallelConfig:
    # [CLASS DOC] Immutable parallel layout: tp, pp, cp (default 1 each), ep (expert parallel,
    # default none meaning "same as tp") and etp (default 1). The process count is the product of
    # tp, pp and cp; the expert parallel size must divide it.
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int | None = None
    etp: int = 1

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validation: raises a value error (message shows the product breakdown and
        # the effective expert parallel size) if the process count is not divisible by the
        # effective expert parallel size. A zero effective size would raise a division error.
        if self.nproc % self.effective_ep != 0:
            raise ValueError(
                f"nproc ({self.nproc} = tp*pp*cp = {self.tp}*{self.pp}*{self.cp}) "
                f"is not divisible by effective EP ({self.effective_ep})"
            )

    @property
    def effective_ep(self) -> int:
        # [FUNCTION DOC] Property: the expert parallel size if set, otherwise the tensor parallel
        # size.
        return self.ep if self.ep is not None else self.tp

    @property
    def nproc(self) -> int:
        # [FUNCTION DOC] Property: number of worker processes, tp times pp times cp (expert
        # parallelism does not add processes).
        return self.tp * self.pp * self.cp

    @classmethod
    def from_parsed_args(cls, parsed: dict[str, int]) -> ParallelConfig:
        # [FUNCTION DOC] Class method: builds a config from a mapping of field name to size,
        # taking each of the five fields from the mapping or, when missing, the field default.
        # Extra keys are ignored. May raise the validation error.
        defaults: dict[str, object] = {f.name: f.default for f in dataclasses.fields(cls)}
        return cls(**{name: parsed.get(name, defaults[name]) for name in _FIELD_NAMES})  # type: ignore[arg-type]

    @classmethod
    def from_run_args(cls, args: RunArgs) -> ParallelConfig:
        # [FUNCTION DOC] Class method: builds a config from the five same-named attributes of a
        # run-arguments object. May raise the validation error.
        return cls(**{name: getattr(args, name) for name in _FIELD_NAMES})  # type: ignore[arg-type]

    def __str__(self) -> str:
        # [FUNCTION DOC] Human-readable form: the five fields as name=value pairs separated by
        # commas, followed by the process count (an unset ep shows as none).
        field_str: str = ", ".join(f"{name}={getattr(self, name)}" for name in _FIELD_NAMES)
        return f"{field_str}, nproc={self.nproc}"

    def dir_name(self) -> str:
        # [FUNCTION DOC] Builds a directory name from the non-default settings, in field order,
        # each as name immediately followed by value, joined by underscores. Omitted: tp when it
        # is none (never in practice), pp and cp when 1, ep when unset or equal to tp, etp when 1.
        # Note tp is always included since its skip value is none. A fully default config still
        # yields "tp1".
        """Build directory name from parallel config, e.g. 'tp2_cp2_ep2'."""
        _SKIP: dict[str, object] = {"tp": None, "pp": 1, "cp": 1, "ep": self.tp, "etp": 1}
        parts: list[str] = [
            f"{name}{getattr(self, name)}"
            for name in _FIELD_NAMES
            if getattr(self, name) is not None and getattr(self, name) != _SKIP[name]
        ]
        return "_".join(parts)


def parse_parallel_args(args_str: str) -> dict[str, int]:
    # [FUNCTION DOC] Splits the string on whitespace and parses it with an argument parser that
    # knows integer options for tp, pp, cp, ep and etp. Returns a mapping containing only the
    # options that were given. An unknown option or non-integer value makes the argument parser
    # print usage and exit the process (system exit) rather than raise a normal error.
    """Parse a parallel config string like '--tp 2 --cp 2' into a dict."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    for flag in _FIELD_NAMES:
        parser.add_argument(f"--{flag}", type=int)
    namespace: argparse.Namespace = parser.parse_args(args_str.split())
    return {k: v for k, v in vars(namespace).items() if v is not None}
