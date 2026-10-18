# [MODULE DOC]
# Tests for data-parallel-aware routing: enabling SGLang's data-parallel attention in the
# arguments must turn on the router's dp-aware mode in argument validation (and leave it off
# otherwise), and the router worker URL helper must strip the data-parallel rank suffix ("@<n>")
# so that per-rank entries collapse to one addressable server, while leaving plain URLs, IPv6
# URLs and URLs whose "@" is user info untouched. Registered as a CPU test in the first CPU stage
# with an estimated 20 seconds and no labels.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import argparse

from miles.backends.sglang_utils.arguments import add_sglang_arguments, validate_args
from miles.utils.http_utils import router_worker_base_urls


def _args(argv):
    # [FUNCTION DOC] HELPER: Parses the argument list with the SGLang arguments added, then sets
    # 4 GPUs per engine, true-on-policy mode off, session server off, router assignment mode
    # "random" and router dp-aware false (the last two are registered by the router's own
    # argument builder, not by the SGLang one, so they are set by hand). Runs the SGLang argument
    # validation (which may change attributes in place) and returns the namespace.
    parser = add_sglang_arguments(argparse.ArgumentParser())
    args = parser.parse_args(argv)
    args.rollout_num_gpus_per_engine = 4
    args.true_on_policy_mode = False
    args.use_session_server = False
    # Registered by RouterArgs.add_cli_args, not by add_sglang_arguments.
    args.router_assignment_mode = "random"
    args.router_dp_aware = False
    validate_args(args)
    return args


def test_dp_attention_turns_on_router_dp_aware():
    # [FUNCTION DOC] With data-parallel attention enabled and data-parallel size 4, validation
    # leaves router dp-aware true.
    assert _args(["--sglang-enable-dp-attention", "--sglang-dp-size", "4"]).router_dp_aware is True


def test_router_dp_aware_untouched_without_dp_attention():
    # [FUNCTION DOC] With no arguments, validation leaves router dp-aware false.
    assert _args([]).router_dp_aware is False


def test_dp_rank_suffix_stripped_and_engines_deduplicated():
    # [FUNCTION DOC] Two entries for the same host and port with rank suffixes 0 and 1 reduce to a
    # single URL without suffix (deduplicated, order preserved).
    # dp-aware routing reports one entry per DP rank; they share one addressable server.
    assert router_worker_base_urls(["http://h:8080@0", "http://h:8080@1"]) == ["http://h:8080"]


def test_plain_worker_urls_pass_through_in_order():
    # [FUNCTION DOC] Two plain URLs with different ports are returned unchanged and in order.
    assert router_worker_base_urls(["http://h:8080", "http://h:8081"]) == ["http://h:8080", "http://h:8081"]


def test_ipv6_dp_rank_suffix_stripped():
    # [FUNCTION DOC] A bracketed IPv6 URL with a rank suffix has just the suffix removed.
    assert router_worker_base_urls(["http://[::1]:8080@0"]) == ["http://[::1]:8080"]


def test_non_numeric_suffix_is_not_a_dp_rank():
    # [FUNCTION DOC] A URL with user:password user info before the host is returned unchanged:
    # the "@" there is not a data-parallel rank marker (the part after it is not a number).
    # userinfo, not a rank: stripping it would change which host is addressed.
    assert router_worker_base_urls(["http://user:pass@h:8080"]) == ["http://user:pass@h:8080"]
