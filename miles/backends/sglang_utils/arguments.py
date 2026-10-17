# [MODULE DOC] Command-line argument definitions for SGLang rollout engines. Registers the router
# options, re-registers every option of SGLang's own server arguments under a "--sglang-" prefix
# (with destination names prefixed by "sglang_") and a second time under "--eval-sglang-" (with
# "eval_sglang_" destinations, defaults suppressed so unset values are inherited from the rollout
# settings), skips a fixed list of options miles controls itself, adds the engine-deployment YAML
# option, and provides post-parse validation that derives dependent settings.
import argparse

from sglang.srt.server_args import ServerArgs
from miles.utils.http_utils import wrap_ipv6


# TODO: use all sglang router arguments with `--sglang-router` prefix
def add_sglang_router_arguments(parser):
    # [FUNCTION DOC] Registers on the given parser: router IP and port (strings/ints, default
    # none), a router policy name (default none), and the router request timeout in seconds
    # (integer, default 14400). Also sets the parser default for a model-routers attribute to
    # none (no option registers it; it is only a default). Returns the same parser.
    """
    Add arguments to the parser for the SGLang router.
    """
    parser.add_argument(
        "--sglang-router-ip",
        type=str,
        default=None,
        help="IP address of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-port",
        type=int,
        default=None,
        help="Port of the SGLang router",
    )
    parser.set_defaults(sglang_model_routers=None)
    parser.add_argument(
        "--sglang-router-policy",
        type=str,
        default=None,
        help="Routing policy for the SGLang router (e.g., 'consistent_hashing', 'round_robin')",
    )
    parser.add_argument(
        "--sglang-router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout for requests to the SGLang router in seconds",
    )
    return parser


_SKIPPED_SERVER_ARGS = [
    "model_path",
    "config",
    "trust_remote_code",
    "random_seed",
    # memory
    "enable_memory_saver",
    # distributed
    # tp_size stays exposed: scripts pass --sglang-tp-size, but the value is
    # overridden from --rollout-num-gpus-per-engine in validate_args below.
    "port",
    "nnodes",
    "node_rank",
    "dist_init_addr",
    "gpu_id_step",
    "base_gpu_id",
    "nccl_port",
    "gated_launch_port",
    "skip_server_warmup",
    "enable_return_routed_experts",
    "enable_return_indexer_topk",
]

# tp_size comes from --eval-num-gpus-per-engine, which also places the engines.
_EVAL_SKIPPED_SERVER_ARGS = _SKIPPED_SERVER_ARGS + ["tp_size"]


def _add_prefixed_server_args(parser, *, flag_prefix: str, dest_prefix: str, skipped_args: list[str], inherit: bool):
    # [FUNCTION DOC] HELPER: Temporarily replaces the parser's add-argument method with a wrapper,
    # lets SGLang's server arguments register themselves through it, then restores the original
    # (not restored if registration raises). Keyword-only inputs: flag prefix, destination prefix,
    # a list of destination names to skip, and the inherit flag. The wrapper (a) works out each
    # option's canonical name from an explicit destination, else from the first long flag with
    # dashes turned to underscores, and silently drops options in the skip list; (b) prefixes every
    # flag (long or short) with two dashes, the flag prefix and the stripped flag; (c) prefixes an
    # explicit destination unless it already starts with the destination prefix; (d) when no
    # destination is given, forces destinations under the destination prefix for the four
    # parallel-size options (tensor, data, pipeline, expert) so they use SGLang's short field names;
    # (e) when inheriting, sets the default to suppress (so unset options leave no attribute) and
    # converts store-true / store-false actions to the boolean optional action (dropping constant
    # and nargs) so an override can also turn a feature off. Positional arguments pass through.
    """Register ``ServerArgs.add_cli_args`` as ``--{flag_prefix}-*`` -> ``args.{dest_prefix}*``.

    ``inherit=True`` defaults to SUPPRESS, so an unset flag leaves no attribute to inherit over.
    """
    old_add_argument = parser.add_argument

    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        # [FUNCTION DOC] HELPER: The replacement add-argument function; behavior is described in the
        # enclosing function. Returns nothing (skipped options are not registered at all, and the
        # underlying registration's result is discarded).
        # KNOWN ISSUE: an option given only by short flags and no destination is never matched
        # against the skip list, so it can never be skipped.
        """
        Add arguments to the parser, ensuring that the server arguments are prefixed and skippable.
        """
        # Determine the canonical name for skip check (e.g., "model_path")
        canonical_name_for_skip_check = None
        if "dest" in kwargs:
            canonical_name_for_skip_check = kwargs["dest"]
        else:
            for flag_name_candidate in name_or_flags:
                if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                    # Derive from first long flag: --foo-bar -> foo_bar
                    stem = flag_name_candidate[2:]
                    canonical_name_for_skip_check = stem.replace("-", "_")
                    break
            # If no long flag and no dest, skip logic might not catch it unless short flags imply a dest.

        if canonical_name_for_skip_check and canonical_name_for_skip_check in skipped_args:
            return  # Skip this entire argument definition

        # If not skipped, proceed to prefix flags and dest
        new_name_or_flags_list = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")  # "foo-bar" from "--foo-bar", or "f" from "-f"
                prefixed_item = f"--{flag_prefix}-{original_flag_stem}"
                new_name_or_flags_list.append(prefixed_item)
            else:
                # Positional arguments or non-string items
                new_name_or_flags_list.append(item_flag)

        # Prepare kwargs for the actual add_argument call.
        # Make a copy to avoid modifying the original kwargs dict.
        final_kwargs = kwargs.copy()

        # If 'dest' is explicitly provided and is a string, prefix it.
        # This ensures the attribute on the args namespace becomes, e.g., args.sglang_dest_name.
        if "dest" in final_kwargs and isinstance(final_kwargs["dest"], str):
            original_dest = final_kwargs["dest"]
            # Avoid double prefixing if dest somehow already starts with the prefix
            if not original_dest.startswith(dest_prefix):
                final_kwargs["dest"] = f"{dest_prefix}{original_dest}"
        elif "dest" not in final_kwargs:
            # argparse derives dest from the first alias, so store parallel sizes under SGLang's short field names.
            for item_flag in name_or_flags:
                if not isinstance(item_flag, str) or not item_flag.startswith("--"):
                    continue
                canonical_dest = item_flag[2:].replace("-", "_")
                if canonical_dest in ("tp_size", "dp_size", "pp_size", "ep_size"):
                    final_kwargs["dest"] = f"{dest_prefix}{canonical_dest}"
                    break

        if inherit:
            final_kwargs["default"] = argparse.SUPPRESS
            if final_kwargs.get("action") in ("store_true", "store_false"):
                # store_true can only turn a field on; an override must also turn one off.
                final_kwargs["action"] = argparse.BooleanOptionalAction
                final_kwargs.pop("const", None)
                final_kwargs.pop("nargs", None)

        old_add_argument(*new_name_or_flags_list, **final_kwargs)

    parser.add_argument = new_add_argument_wrapper
    ServerArgs.add_cli_args(parser)
    parser.add_argument = old_add_argument


def collect_eval_sglang_overrides(args) -> dict:
    # [FUNCTION DOC] Returns a mapping from server-argument field name to value for every parsed
    # attribute whose name starts with the eval prefix (prefix removed). Because inherited options
    # use suppressed defaults, only options the user actually set appear.
    """``ServerArgs`` fields set via ``--eval-sglang-*``; absent means inherit ``--sglang-*``."""
    return {
        key.removeprefix("eval_sglang_"): value for key, value in vars(args).items() if key.startswith("eval_sglang_")
    }


def add_sglang_arguments(parser):
    # [FUNCTION DOC] Registers all SGLang-related options on the parser and returns it: the router
    # options; a server concurrency integer (default 512); SGLang's server options under the
    # "--sglang-" prefix with normal defaults (skipping the miles-controlled list); the same options
    # under "--eval-sglang-" with inheriting behavior (additionally skipping the tensor-parallel
    # size, which is derived from the eval GPUs per engine); and a path option for the engine
    # deployment YAML (default none; mutually exclusive with the prefill server count option, which
    # is not enforced here).
    """
    Add arguments to the parser for the SGLang server.
    """
    parser = add_sglang_router_arguments(parser)
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)

    _add_prefixed_server_args(
        parser, flag_prefix="sglang", dest_prefix="sglang_", skipped_args=_SKIPPED_SERVER_ARGS, inherit=False
    )
    _add_prefixed_server_args(
        parser,
        flag_prefix="eval-sglang",
        dest_prefix="eval_sglang_",
        skipped_args=_EVAL_SKIPPED_SERVER_ARGS,
        inherit=True,
    )

    parser.add_argument(
        "--sglang-config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config for SGLang engine deployment. "
            "Defines server_groups with worker_type (regular/prefill/decode/placeholder), "
            "num_gpus per group, and optional per-group 'overrides' dict of "
            "ServerArgs field names that override the base --sglang-* CLI args. "
            "Placeholder groups reserve GPU slots without creating engines. "
            "A 'name: eval' model is filled in from the --eval-* args (model_path, "
            "num_gpus_per_engine, --eval-sglang-* overrides) wherever the YAML leaves them unset. "
            "Mutually exclusive with --prefill-num-servers."
        ),
    )

    return parser


def validate_args(args):
    # [FUNCTION DOC] Post-parse validation and derivation; mutates the parsed arguments and returns
    # nothing. Rules in order: the server tensor-parallel size is overwritten by rollout GPUs per
    # engine (any user-supplied value is ignored); true-on-policy mode turns on deterministic
    # inference; when logprob recomputation via prefill is set (attribute may be absent), turns on
    # prefill-only deterministic inference and deterministic inference; a data-parallel size above
    # one asserts that data-parallel attention is enabled; data-parallel attention turns on the
    # router's data-parallel-aware mode; with no router policy and a session server, sets the
    # policy to manual and changes a random router assignment mode to min-load; a router IP, if set,
    # is wrapped for IPv6 literal handling.
    args.sglang_tp_size = args.rollout_num_gpus_per_engine

    if args.true_on_policy_mode:
        args.sglang_enable_deterministic_inference = True

    if getattr(args, "recompute_logprobs_via_prefill", False):
        args.sglang_enable_prefill_only_deterministic_inference = True
        args.sglang_enable_deterministic_inference = True

    if args.sglang_dp_size > 1:
        assert args.sglang_enable_dp_attention

    if args.sglang_enable_dp_attention:
        args.router_dp_aware = True

    if args.sglang_router_policy is None and args.use_session_server:
        args.sglang_router_policy = "manual"
        if args.router_assignment_mode == "random":
            args.router_assignment_mode = "min_load"

    if getattr(args, "sglang_router_ip", None):
        args.sglang_router_ip = wrap_ipv6(args.sglang_router_ip)
