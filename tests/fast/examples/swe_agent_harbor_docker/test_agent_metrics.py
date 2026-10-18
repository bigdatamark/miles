# [MODULE DOC]
# Tests of the agent metric aggregation function in the swe-agent harbor docker example's generate
# script. The script is loaded by file path (the directory name is not an importable identifier)
# through a module-scoped fixture. Constants: the repo root, the script path, a fully instrumented
# agent metrics mapping, a totals-only mapping (eval, run and total time only), and the list of
# aggregated keys that the totals-only agent must not produce. Aggregation yields keys prefixed with
# the agent namespace (mean, sum, min, max variants) and only counts agents that reported a key.
"""Agent metrics must reflect what an agent measured, not a default of zero.

The example directory name is not a Python identifier, so the module is loaded
by path the same way the other example tests do it.
"""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
GENERATE_SCRIPT = REPO_ROOT / "examples" / "swe-agent-harbor-docker" / "generate.py"

# An agent that instruments every turn reports the full key set.
FULLY_INSTRUMENTED = {
    "turns": 30,
    "tool_calls": 47,
    "model_query_time_sum": 120.0,
    "env_execution_time_sum": 60.0,
    "eval_time": 12.0,
    "agent_run_time": 400.0,
    "time_per_turn": 13.3,
    "model_query_time_avg": 4.0,
    "env_execution_time_avg": 2.0,
    "model_time_ratio": 0.3,
    "env_time_ratio": 0.15,
    "eval_time_ratio": 0.03,
    "total_time": 500.0,
}
# Another agent reports only wall-clock totals: no turn, tool-call or timing breakdown.
TOTALS_ONLY = {"eval_time": 42.8, "agent_run_time": 486.8, "total_time": 559.3}

UNREPORTED_BY_TOTALS_ONLY = [
    "agent/turns_mean",
    "agent/turns_sum",
    "agent/tool_calls_mean",
    "agent/tool_calls_sum",
    "agent/model_query_time_sum_mean",
    "agent/env_execution_time_sum_mean",
    "agent/time_per_turn",
    "agent/model_query_time_avg",
    "agent/env_execution_time_avg",
    "agent/model_time_ratio",
    "agent/env_time_ratio",
    "agent/eval_time_ratio",
]


@pytest.fixture(scope="module")
def generate_module() -> ModuleType:
    # [FUNCTION DOC] HELPER: Module-scoped fixture. Loads the example generate script as a module from
    # its file path under a synthetic module name, executes it (import side effects of the script run
    # once), and returns the module object.
    spec = importlib.util.spec_from_file_location("swe_agent_harbor_docker_generate", GENERATE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_with(agent_metrics: dict) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Returns a minimal sample stand-in whose metadata mapping holds the given
    # agent metrics under the agent-metrics key.
    return SimpleNamespace(metadata={"agent_metrics": agent_metrics})


def test_fully_instrumented_agent_reports_every_metric(generate_module: ModuleType) -> None:
    # [FUNCTION DOC] Aggregates one fully instrumented sample and asserts the mean turns (30), mean and
    # sum of tool calls (47), time per turn (13.3) and mean total time (500.0) keys.
    metrics = generate_module.aggregate_agent_metrics([sample_with(FULLY_INSTRUMENTED)])

    assert metrics["agent/turns_mean"] == 30
    assert metrics["agent/tool_calls_mean"] == 47
    assert metrics["agent/tool_calls_sum"] == 47
    assert metrics["agent/time_per_turn"] == pytest.approx(13.3)
    assert metrics["agent/total_time_mean"] == pytest.approx(500.0)


def test_unreported_keys_are_omitted_rather_than_logged_as_zero(generate_module: ModuleType) -> None:
    # [FUNCTION DOC] Aggregates one totals-only sample; asserts none of the turn, tool-call, timing
    # breakdown or ratio keys appear, while the mean eval time, agent run time and total time keys do.
    """A zero here is indistinguishable from an agent that made no tool calls."""
    metrics = generate_module.aggregate_agent_metrics([sample_with(TOTALS_ONLY)])

    for key in UNREPORTED_BY_TOTALS_ONLY:
        assert key not in metrics, f"{key} was logged despite never being measured"

    # What the agent did report still comes through.
    assert metrics["agent/eval_time_mean"] == pytest.approx(42.8)
    assert metrics["agent/agent_run_time_mean"] == pytest.approx(486.8)
    assert metrics["agent/total_time_mean"] == pytest.approx(559.3)


def test_mixed_batch_averages_only_over_agents_that_measured_the_key(generate_module: ModuleType) -> None:
    # [FUNCTION DOC] Aggregates a batch of one instrumented and one totals-only sample. Keys only the
    # instrumented one reports keep its value (not halved); the total time key averages over both
    # samples, with min and max being the two values.
    """One silent agent must not halve the tool-call count of the batch."""
    metrics = generate_module.aggregate_agent_metrics([sample_with(FULLY_INSTRUMENTED), sample_with(TOTALS_ONLY)])

    assert metrics["agent/tool_calls_mean"] == 47
    assert metrics["agent/turns_mean"] == 30
    assert metrics["agent/time_per_turn"] == pytest.approx(13.3)
    # Keys both agents report still average across the whole batch.
    assert metrics["agent/total_time_mean"] == pytest.approx((500.0 + 559.3) / 2)
    assert metrics["agent/total_time_min"] == pytest.approx(500.0)
    assert metrics["agent/total_time_max"] == pytest.approx(559.3)


def test_zero_is_kept_when_an_agent_actually_measures_zero(generate_module: ModuleType) -> None:
    # [FUNCTION DOC] An instrumented sample with zero tool calls still yields zero mean and zero sum
    # tool-call keys: a real zero is kept, not omitted.
    """Omitting unreported keys must not swallow a genuine zero measurement."""
    metrics = generate_module.aggregate_agent_metrics([sample_with({**FULLY_INSTRUMENTED, "tool_calls": 0})])

    assert metrics["agent/tool_calls_mean"] == 0
    assert metrics["agent/tool_calls_sum"] == 0


def test_samples_without_agent_metrics_are_ignored(generate_module: ModuleType) -> None:
    # [FUNCTION DOC] Asserts aggregation returns an empty mapping for a sample with empty agent
    # metrics, for a sample with empty metadata, and for an empty sample list.
    assert generate_module.aggregate_agent_metrics([sample_with({})]) == {}
    assert generate_module.aggregate_agent_metrics([SimpleNamespace(metadata={})]) == {}
    assert generate_module.aggregate_agent_metrics([]) == {}
