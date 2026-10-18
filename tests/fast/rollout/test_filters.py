# [MODULE DOC] Tests for the rollout filter hub: the filter output type and its legacy alias, the
# deprecated dynamic-sampling filter wrappers that forward to canonical filters, the sample iterator
# over flat or nested groups, the common filters (aborted, missing reward, nonzero reward spread),
# the pre-put filter pipeline (ordering and error propagation), and the weight-version staleness
# helpers. Pure CPU tests with synthetic samples; registered in the first CPU stage, 20 second
# estimate.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

import warnings
from argparse import Namespace

import pytest

from miles.rollout.filter_hub import dynamic_sampling_filters
from miles.rollout.filter_hub.base_types import DynamicFilterOutput, FilterOutput, iter_samples
from miles.rollout.filter_hub.common_filters import (
    apply_aborted_filter,
    apply_missing_reward_filter,
    apply_preput_filters,
    apply_reward_nonzero_std_filter,
    group_staleness,
    group_weight_version_stats,
)
from miles.utils.function_registry import load_function
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def make_sample(
    *,
    reward=1.0,
    status=Sample.Status.COMPLETED,
    weight_versions=(),
) -> Sample:
    # [FUNCTION DOC] HELPER: Builds a sample with the given reward and status. Each entry of the
    # weight versions sequence becomes one per-call record holding a single span with that version
    # and token range 0 to 1. Returns the sample.
    return Sample(
        reward=reward,
        status=status,
        weight_versions=[
            WeightVersionsPerCall(spans=[WeightVersionSpan(version=version, abs_start=0, abs_end=1)])
            for version in weight_versions
        ],
    )


def test_dynamic_filter_output_is_a_compatibility_alias():
    # [FUNCTION DOC] Asserts the dynamic filter output name refers to the very same type as the
    # general filter output.
    assert DynamicFilterOutput is FilterOutput


@pytest.mark.parametrize(
    ("legacy_name", "canonical"),
    [
        ("check_no_aborted", apply_aborted_filter),
        ("check_reward_nonzero_std", apply_reward_nonzero_std_filter),
    ],
)
def test_legacy_filters_forward_without_runtime_deprecation_warnings(monkeypatch, legacy_name, canonical):
    # [FUNCTION DOC] Parametrized over the legacy names check_no_aborted and check_reward_nonzero_std
    # and their canonical replacements. Loads the legacy function through the function registry by
    # dotted path and asserts it keeps its name and its deprecation marker mentions the canonical
    # name, while the canonical function has no such marker. Then replaces the canonical name in the
    # legacy module with a recording function: calling the legacy filter (with warnings turned into
    # errors) must pass the same args and samples objects and the extra keyword through, and return
    # the replacement's result object itself. Finally replaces it with a function raising a value
    # error and asserts the legacy filter (called with keywords) re-raises that same exception.
    legacy = load_function(f"miles.rollout.filter_hub.dynamic_sampling_filters.{legacy_name}")
    assert legacy.__name__ == legacy_name
    assert canonical.__name__ in legacy.__deprecated__
    assert not hasattr(canonical, "__deprecated__")

    args, samples, extra = Namespace(reward_key=None), [make_sample()], object()
    result = FilterOutput(keep=False, reason="unchanged")

    def replacement(forwarded_args, forwarded_samples, **kwargs):
        # [FUNCTION DOC] HELPER: Asserts the forwarded args and samples are the identical objects the
        # test passed and that the only extra keyword is the test's extra object, then returns the
        # pre-built result.
        assert forwarded_args is args
        assert forwarded_samples is samples
        assert kwargs == {"extra": extra}
        return result

    monkeypatch.setattr(dynamic_sampling_filters, canonical.__name__, replacement)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert legacy(args, samples, extra=extra) is result

    error = ValueError("filter failed")

    def fail(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Ignores its arguments and raises the pre-built value error.
        raise error

    monkeypatch.setattr(dynamic_sampling_filters, canonical.__name__, fail)
    with pytest.raises(ValueError) as raised:
        legacy(args=args, samples=samples, extra=extra)
    assert raised.value is error


@pytest.mark.parametrize(
    ("legacy_name", "samples", "expected"),
    [
        ("check_no_aborted", [make_sample()], FilterOutput(keep=True)),
        (
            "check_no_aborted",
            [[make_sample(status=Sample.Status.ABORTED)]],
            FilterOutput(keep=False, reason="group_has_aborted"),
        ),
        (
            "check_reward_nonzero_std",
            [make_sample(reward=1.0), make_sample(reward=2.0)],
            FilterOutput(keep=True),
        ),
        (
            "check_reward_nonzero_std",
            [[make_sample(reward=1.0), make_sample(reward=1.0)]],
            FilterOutput(keep=False, reason="zero_std_1.0"),
        ),
    ],
)
def test_legacy_filters_preserve_structured_results(legacy_name, samples, expected):
    # [FUNCTION DOC] Parametrized cases: no-aborted keeps a flat completed group; no-aborted drops a
    # nested group holding an aborted sample with reason "group_has_aborted"; nonzero-std keeps a flat
    # group with rewards 1 and 2; nonzero-std drops a nested group of equal rewards with reason
    # "zero_std_1.0". Calls the legacy filter (with an unused extra keyword) and asserts the result
    # is a dynamic filter output equal to the expected output.
    legacy = load_function(f"miles.rollout.filter_hub.dynamic_sampling_filters.{legacy_name}")
    result = legacy(Namespace(reward_key=None), samples, ignored=True)
    assert isinstance(result, DynamicFilterOutput)
    assert result == expected


def test_iter_samples_preserves_flat_and_mixed_nested_order():
    # [FUNCTION DOC] Four samples indexed 0 to 3: iterating a flat list yields them in order, and
    # iterating a list mixing single samples with a nested sub-list also yields them flattened in the
    # original order.
    samples = [Sample(index=index) for index in range(4)]

    assert list(iter_samples(samples)) == samples
    assert list(iter_samples([samples[0], samples[1:3], samples[3]])) == samples


@pytest.mark.parametrize(
    ("filter_fn", "group", "expected"),
    [
        (
            apply_aborted_filter,
            [make_sample(status=Sample.Status.ABORTED)],
            FilterOutput(keep=False, reason="group_has_aborted"),
        ),
        (
            apply_missing_reward_filter,
            [make_sample(reward=None)],
            FilterOutput(keep=False, reason="group_has_missing_reward"),
        ),
    ],
)
def test_common_filters_return_structured_drop(filter_fn, group, expected):
    # [FUNCTION DOC] Parametrized: the aborted filter drops a group with an aborted sample (reason
    # "group_has_aborted"); the missing-reward filter drops a group with a none reward (reason
    # "group_has_missing_reward"). Extra keywords are accepted and ignored.
    assert filter_fn(Namespace(reward_key=None), group, ignored=True) == expected


@pytest.mark.parametrize("reward", [None, 1.0], ids=["missing-reward", "numeric-reward"])
def test_preput_aborted_wins_before_missing_reward_and_dynamic_filter(reward):
    # [FUNCTION DOC] Parametrized over a missing and a numeric reward, with an aborted sample. The
    # pre-put pipeline must return the aborted drop ("group_has_aborted") and never call the custom
    # dynamic filter.
    custom_calls = []

    def dynamic_filter(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Records its call and returns true (keep).
        custom_calls.append((args, kwargs))
        return True

    output = apply_preput_filters(
        Namespace(reward_key=None),
        dynamic_filter,
        [make_sample(reward=reward, status=Sample.Status.ABORTED)],
    )

    assert output == FilterOutput(keep=False, reason="group_has_aborted")
    assert custom_calls == []


@pytest.mark.parametrize(
    ("reward_key", "group"),
    [
        (None, [make_sample(reward=None)]),
        ("score", [make_sample(reward={"score": None})]),
        (None, [make_sample(), [make_sample(reward=None)]]),
    ],
    ids=["raw", "selected", "nested"],
)
def test_preput_missing_reward_wins_before_dynamic_filter(reward_key, group):
    # [FUNCTION DOC] Parametrized over a raw none reward, a none reward selected by key from a reward
    # mapping, and a none reward inside a nested group. The pre-put pipeline must return the
    # missing-reward drop ("group_has_missing_reward") without calling the dynamic filter.
    custom_calls = []

    def dynamic_filter(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Records its call and returns true (keep).
        custom_calls.append((args, kwargs))
        return True

    output = apply_preput_filters(
        Namespace(reward_key=reward_key),
        dynamic_filter,
        group,
    )

    assert output == FilterOutput(keep=False, reason="group_has_missing_reward")
    assert custom_calls == []


def test_preput_selected_reward_missing_key_propagates():
    # [FUNCTION DOC] When the reward key selects a key absent from the sample's reward mapping, the
    # pre-put pipeline raises a key error naming it, and the dynamic filter is never called.
    custom_calls = []

    def dynamic_filter(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Records its call and returns true (keep).
        custom_calls.append((args, kwargs))
        return True

    with pytest.raises(KeyError, match="score"):
        apply_preput_filters(
            Namespace(reward_key="score"),
            dynamic_filter,
            [make_sample(reward={"other": 1.0})],
        )

    assert custom_calls == []


def test_preput_preserves_filter_output_identity():
    # [FUNCTION DOC] A dynamic filter returning a filter output object (keep false, custom reason)
    # must have that exact same object returned by the pre-put pipeline.
    output = FilterOutput(keep=False, reason="custom_reason")

    assert apply_preput_filters(Namespace(reward_key=None), lambda *args, **kwargs: output, [make_sample()]) is output


def test_preput_normalizes_legacy_false():
    # [FUNCTION DOC] A legacy dynamic filter returning the boolean false is normalized by the
    # pipeline into a filter output with keep false and no reason.
    output = apply_preput_filters(Namespace(reward_key=None), lambda *args, **kwargs: False, [make_sample()])

    assert output == FilterOutput(keep=False)


def test_preput_passes_kwargs_and_returns_keep_output():
    # [FUNCTION DOC] The pipeline calls the dynamic filter with the same args and group objects plus
    # any extra keyword (a marker object), and returns the keep output the filter produced. Asserts
    # the single recorded call and the output.
    args = Namespace(reward_key=None)
    group = [make_sample()]
    marker = object()
    calls = []

    def dynamic_filter(received_args, received_group, **kwargs):
        # [FUNCTION DOC] HELPER: Records (args, group, keywords) and returns a keep output.
        calls.append((received_args, received_group, kwargs))
        return FilterOutput(keep=True)

    output = apply_preput_filters(
        args,
        dynamic_filter,
        group,
        marker=marker,
    )

    assert output == FilterOutput(keep=True)
    assert calls == [(args, group, {"marker": marker})]


def test_preput_propagates_dynamic_filter_exception():
    # [FUNCTION DOC] A runtime error raised by the dynamic filter propagates unchanged out of the
    # pre-put pipeline.
    def dynamic_filter(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Always raises a runtime error saying the dynamic filter failed.
        raise RuntimeError("dynamic filter failed")

    with pytest.raises(RuntimeError, match="dynamic filter failed"):
        apply_preput_filters(
            Namespace(reward_key=None),
            dynamic_filter,
            [make_sample()],
        )


def test_group_staleness_uses_oldest_version_across_nested_samples():
    # [FUNCTION DOC] For a nested group with versions "9", "4" and "8" (plus a sample without
    # versions) and current version 10, group staleness is 6 (current minus the oldest version, 4).
    # It is none when the current version is none or when no sample carries a version, and negative
    # (-2) when the only version (12) is newer than the current one.
    group = [
        make_sample(weight_versions=("9",)),
        [make_sample(weight_versions=("4", "8")), make_sample(weight_versions=())],
    ]

    assert group_staleness(group, current_version=10) == 6
    assert group_staleness(group, current_version=None) is None
    assert group_staleness([make_sample()], current_version=10) is None
    assert group_staleness([make_sample(weight_versions=("12",))], current_version=10) == -2


def test_group_weight_version_stats_preserve_sample_and_token_populations():
    # [FUNCTION DOC] Builds one sample with two spans (version "4" over tokens 2 to 5, that is 3
    # tokens, and version "6" over tokens 5 to 6, that is 1 token) and a second, nested sample whose
    # only version is non-numeric. Asserts the stats count 2 samples, 1 versioned sample and 4
    # versioned tokens; at current version 10 the oldest lag is 6, newest lag 4, and token-weighted
    # lag 5.5 ((3*6 + 1*4) / 4); with a none current version the oldest and token-weighted lags are
    # none.
    first = make_sample()
    first.weight_versions = [
        WeightVersionsPerCall(
            spans=[
                WeightVersionSpan(version="4", abs_start=2, abs_end=5),
                WeightVersionSpan(version="6", abs_start=5, abs_end=6),
            ]
        )
    ]
    second = make_sample(weight_versions=("not-numeric",))

    stats = group_weight_version_stats([first, [second]])

    assert stats.sample_count == 2
    assert stats.versioned_sample_count == 1
    assert stats.versioned_token_count == 4
    assert stats.oldest_lag(current_version=10) == 6
    assert stats.newest_lag(current_version=10) == 4
    assert stats.token_weighted_lag(current_version=10) == 5.5
    assert stats.oldest_lag(current_version=None) is None
    assert stats.token_weighted_lag(current_version=None) is None
