# [MODULE DOC]
# Tests for the SGLang command-line argument builder: the multi-policy router option defaults to
# none, and server-argument fields owned by the address allocator (the gated-launch port, plus the
# existing port, NCCL port and base GPU id entries) must not be exposed as command-line flags, in
# both the normal and evaluation prefixes, while ordinary server arguments stay exposed. Skipped
# entirely when SGLang is not installed.
from __future__ import annotations

import argparse


import pytest

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.arguments import add_sglang_arguments, collect_eval_sglang_overrides
from miles.utils.workers.argv_utils import _record_field_names


def _sglang_flags() -> set[str]:
    # [FUNCTION DOC] HELPER: Adds the SGLang arguments to a fresh parser and returns the set of
    # all option strings (long and short flags) of all its actions.
    parser = add_sglang_arguments(argparse.ArgumentParser())
    return {option for action in parser._actions for option in action.option_strings}


def _parse_sglang_args(argv: list[str]) -> argparse.Namespace:
    # [FUNCTION DOC] HELPER: Parses the given argument list with a fresh parser that has the
    # SGLang arguments added, and returns the namespace (argparse exits the process on errors).
    return add_sglang_arguments(argparse.ArgumentParser()).parse_args(argv)


class TestSglangModelRoutersDefault:
    # [CLASS DOC] Test of the default of the multi-policy router option.
    def test_parsing_without_model_routers_sets_none(self):
        # [FUNCTION DOC] Parsing an empty argument list leaves the model-routers attribute none.
        """Parsing without multi-policy routers exposes a safe None default."""
        args = _parse_sglang_args([])

        assert args.sglang_model_routers is None


class TestAllocatorOwnedServerArgs:
    # [CLASS DOC] Tests that server-argument fields assigned by the address allocator are skipped
    # when generating command-line flags.
    def test_the_launch_gate_port_is_not_exposed_on_the_cli(self):
        # [FUNCTION DOC] Neither the normal nor the evaluation gated-launch-port flag is among the
        # parser's option strings.
        """The gate port comes from the addr allocator, so a flag for it could only point the engine elsewhere."""
        flags = _sglang_flags()

        assert "--sglang-gated-launch-port" not in flags
        assert "--eval-sglang-gated-launch-port" not in flags

    def test_a_tunable_server_arg_is_still_exposed(self):
        # [FUNCTION DOC] The static memory fraction flag is still among the parser's options.
        """The skip list must stay narrow: ordinary ServerArgs fields remain reachable from the cli."""
        assert "--sglang-mem-fraction-static" in _sglang_flags()

    def test_a_launch_gate_port_flag_is_a_hard_cli_error(self):
        # [FUNCTION DOC] Passing either gated-launch-port flag with value 13007 on the command line
        # makes the parser exit (system-exit error) instead of accepting it.
        """Passing a gate port on the command line must fail loudly instead of being accepted and ignored."""
        for flag in ("--sglang-gated-launch-port", "--eval-sglang-gated-launch-port"):
            with pytest.raises(SystemExit):
                _parse_sglang_args([flag, "13007"])

    def test_parsing_leaves_no_launch_gate_port_attribute_on_the_namespace(self):
        # [FUNCTION DOC] After parsing an empty list, the namespace has neither the normal nor the
        # evaluation gated-launch-port attribute, and the evaluation override collection does not
        # contain a gated launch port entry.
        """The engine copies every args.sglang_<field> onto ServerArgs, so a parsed attribute would travel to it."""
        args = _parse_sglang_args([])

        assert not hasattr(args, "sglang_gated_launch_port")
        assert not hasattr(args, "eval_sglang_gated_launch_port")
        assert "gated_launch_port" not in collect_eval_sglang_overrides(args)

    def test_the_other_allocator_owned_server_args_stay_off_the_cli(self):
        # [FUNCTION DOC] The port, NCCL port, base GPU id and evaluation port flags are all absent
        # from the parser's options.
        """The gate port joins the existing skip list instead of replacing the endpoint entries already there."""
        flags = _sglang_flags()

        for flag in ("--sglang-port", "--sglang-nccl-port", "--sglang-base-gpu-id", "--eval-sglang-port"):
            assert flag not in flags

    def test_the_skipped_launch_gate_port_names_a_real_server_args_field(self):
        # [FUNCTION DOC] The set of field names of SGLang's real server-arguments record includes
        # the gated launch port, guarding against an upstream rename that would make the skip
        # entry stale.
        """A renamed upstream field would leave the skip entry stale and quietly re-expose the flag."""
        assert "gated_launch_port" in set(_record_field_names(ServerArgs))
