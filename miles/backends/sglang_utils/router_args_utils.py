# [MODULE DOC] Helpers that turn miles' parsed arguments into SGLang router arguments and back into
# a command-line argument list (and parse such a list back), so the router can be launched as a
# subprocess with a verified round trip. Router options in the parsed miles arguments carry a
# "router_" destination prefix.
import argparse
from collections.abc import Mapping

from sglang_router.launch_router import RouterArgs

from miles.utils.workers.argv_utils import render_cli_argv

_ROUTER_DEST_PREFIX = "router_"


def compute_sglang_router_args(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    prometheus_port: int,
    has_pd_disaggregation: bool,
) -> dict[str, object]:
    # [FUNCTION DOC] Builds the router argument mapping (router field name to value). Starts from
    # all router options present in the parsed miles arguments (prefix stripped), then forces host,
    # port, prometheus port, log level "warn" and the request timeout from the miles router timeout
    # option. If a router policy is set in the miles arguments it overrides the policy field; if the
    # caller says prefill/decode disaggregation is in use, sets the disaggregation field to true.
    # Host, port, prometheus port and the disaggregation flag are keyword-only inputs.
    router_args = _extract_router_cli_values(args)
    router_args.update(
        host=host,
        port=port,
        prometheus_port=prometheus_port,
        log_level="warn",
        request_timeout_secs=args.sglang_router_request_timeout_secs,
    )

    if args.sglang_router_policy:
        router_args["policy"] = args.sglang_router_policy

    if has_pd_disaggregation:
        router_args["pd_disaggregation"] = True

    return router_args


def router_args_to_argv(router_args: Mapping[str, object]) -> list[str]:
    # [FUNCTION DOC] Renders the router argument mapping to a list of command-line tokens using the
    # shared renderer. The expected router object is built from the mapping, the parser is the
    # router's own parser, and the renderer verifies that parsing the produced list gives back an
    # equal object (failures raise from the renderer).
    return render_cli_argv(
        router_args,
        expected_obj=RouterArgs.from_cli_args(argparse.Namespace(**router_args)),
        make_parser=_make_cli_parser,
        from_parsed=RouterArgs.from_cli_args,
    )


def parse_router_args_argv(argv: list[str]) -> RouterArgs:
    # [FUNCTION DOC] Parses a list of command-line tokens with the router's parser and converts the
    # result to a router arguments object. Exits via the parser (system exit) on invalid input.
    return RouterArgs.from_cli_args(_make_cli_parser().parse_args(argv))


def _make_cli_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: A fresh parser with the router's unprefixed options registered.
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    return parser


def _extract_router_cli_values(args: argparse.Namespace) -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Learns the router option names by parsing an empty argument list with
    # the prefixed router parser (host and port excluded), then reads each such prefixed attribute
    # from the given parsed arguments (an attribute error if missing) and returns a mapping with
    # the prefix removed.
    prefixed_defaults = vars(_make_prefixed_cli_parser().parse_args([]))
    return {name.removeprefix(_ROUTER_DEST_PREFIX): getattr(args, name) for name in prefixed_defaults}


def _make_prefixed_cli_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: A fresh parser with the router options registered with the router
    # prefix on destinations, excluding the host and port options.
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    return parser
