# [MODULE DOC] Converts a dictionary of SGLang server arguments to a command-line argument list (with
# a round-trip check) and parses such a list back, used to launch SGLang servers as subprocesses.
# Some fields (trust-remote-code, model path, host, port, device) are always rendered even when
# equal to their defaults.
import argparse

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_device

from miles.utils.workers.argv_utils import render_cli_argv

_ALWAYS_RENDER_FIELDS = ("trust_remote_code", "model_path", "host", "port", "device")


def server_args_to_argv(server_args_dict: dict) -> list[str]:
    # [FUNCTION DOC] Returns the command-line token list for the given server-argument dictionary
    # (the input is not mutated). The device entry is replaced by an explicit device type: the
    # given one, or SGLang's detected device if unset, with any index after a colon removed. The
    # shared renderer is given the server-argument object built from the dictionary as the
    # expected result, SGLang's own parser, the parsed-to-object converter and the always-render
    # field list, and verifies the round trip (failures raise from the renderer or from the
    # server-argument constructor).
    server_args_dict = {**server_args_dict, "device": _explicit_device(server_args_dict.get("device"))}
    return render_cli_argv(
        server_args_dict,
        expected_obj=ServerArgs(**server_args_dict),
        make_parser=_make_cli_parser,
        from_parsed=ServerArgs.from_cli_args,
        always_render_fields=_ALWAYS_RENDER_FIELDS,
    )


def parse_server_args_argv(argv: list[str]) -> ServerArgs:
    # [FUNCTION DOC] Parses a token list with SGLang's parser and returns the server-arguments
    # object built from the result.
    return ServerArgs.from_cli_args(_make_cli_parser().parse_args(argv))


def _make_cli_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: A fresh parser with SGLang's server options registered.
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser


def _explicit_device(device: str | None) -> str:
    # [FUNCTION DOC] HELPER: Returns the device type text before any colon, using the given device
    # or, if it is none or empty, SGLang's detected device.
    return (device or get_device()).split(":")[0]
