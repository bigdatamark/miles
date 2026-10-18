# [MODULE DOC]
# Tests for the dashboard command-line argument helpers: defaults of the registered options,
# validation of the dependent arguments, and conversion of parsed arguments into the collector
# configuration. Pure in-process, no I/O.
import argparse
import logging

import pytest

from miles.dashboard.args import add_dashboard_arguments, collector_config_from_args, validate_dashboard_args
from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST


def parse(argv):
    # [FUNCTION DOC] HELPER: Parses an argument list with a fresh parser containing only the
    # dashboard options; returns the resulting namespace.
    parser = argparse.ArgumentParser()
    add_dashboard_arguments(parser)
    return parser.parse_args(argv)


def full_args(**overrides):
    # [FUNCTION DOC] HELPER: Parses with the dashboard enabled, then sets the external attributes
    # the dashboard reads (dump directory "/tmp/dump", rollout entropy on, group "my-run",
    # colocate on) and applies the given overrides as attribute assignments. Returns the namespace.
    args = parse(["--use-miles-dashboard"])
    args.dump_details = "/tmp/dump"
    args.use_rollout_entropy = True
    args.wandb_group = "my-run"
    args.colocate = True
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_defaults():
    # [FUNCTION DOC] Asserts defaults with no arguments: dashboard off, flush interval 5.0,
    # gpu sample interval 1.0, scrape mode auto, no metric list override, prometheus forwarding off.
    args = parse([])
    assert args.use_miles_dashboard is False
    assert args.dashboard_flush_interval == 5.0
    assert args.dashboard_gpu_sample_interval == 1.0
    assert args.dashboard_sglang_scrape_mode == "auto"
    assert args.dashboard_sglang_metrics is None
    assert args.dashboard_forward_prometheus is False


def test_validate_requires_dump_details():
    # [FUNCTION DOC] With the dashboard on and no dump directory, validation must raise an
    # assertion error whose message mentions dump-details.
    args = full_args(dump_details=None)
    with pytest.raises(AssertionError, match="dump-details"):
        validate_dashboard_args(args)


def test_validate_warns_without_entropy(caplog):
    # [FUNCTION DOC] Validation with rollout entropy disabled logs a warning mentioning
    # use-rollout-entropy; after clearing captured logs, validation with entropy enabled logs
    # no records at all.
    with caplog.at_level(logging.WARNING):
        validate_dashboard_args(full_args(use_rollout_entropy=False))
    assert any("use-rollout-entropy" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        validate_dashboard_args(full_args())
    assert caplog.records == []


def test_validate_disabled_checks_nothing():
    # [FUNCTION DOC] Validation on arguments with the dashboard disabled (and without a dump
    # directory attribute) must succeed without raising.
    validate_dashboard_args(parse([]))  # no dump_details attribute needed


def test_collector_config_from_args():
    # [FUNCTION DOC] Builds a config from full arguments with start time 123.0 and asserts: the
    # dashboard directory is the dump directory plus "dashboard", run name equals the wandb
    # group, start time carried through, metric whitelist equals the scraper default, the
    # argument snapshot includes colocate and omits attributes absent from the namespace.
    config = collector_config_from_args(full_args(), start_ts=123.0)
    assert config.dashboard_dir == "/tmp/dump/dashboard"
    assert config.run_name == "my-run"
    assert config.start_ts == 123.0
    assert config.metric_whitelist == DEFAULT_METRIC_WHITELIST
    assert config.args_snapshot["colocate"] is True
    assert "hf_checkpoint" not in config.args_snapshot  # absent attrs are skipped


def test_collector_config_whitelist_override_and_run_name_fallback():
    # [FUNCTION DOC] With no wandb group and a comma-separated metric list option, asserts the
    # whitelist becomes a tuple of the two names in order and the run name falls back to
    # "miles-run".
    args = full_args(wandb_group=None)
    args.dashboard_sglang_metrics = "sglang_gen_throughput,sglang_token_usage"
    config = collector_config_from_args(args, start_ts=0.0)
    assert config.metric_whitelist == ("sglang_gen_throughput", "sglang_token_usage")
    assert config.run_name == "miles-run"
