# [MODULE DOC] Tests of the precedence used to resolve a sample's reward configuration (a pair
# of custom reward function path and reward model type): fields set on the sample's reward spec
# win; unset fields fall back first to the sample's metadata (for the type) and then to the
# process-wide arguments. Returns are pairs of (custom path, type).
"""_resolve_reward_config precedence: reward_spec fields win when set, and
unset fields fall through to sample metadata and process-wide args. A
reward_spec with empty fields (an adapter config without its own rm_type)
must not disable the fallbacks."""

from types import SimpleNamespace

from miles.rollout.rm_hub import _resolve_reward_config
from miles.utils.types import RewardSpec, Sample


def _args(**kwargs) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds an arguments namespace with no reward type, no custom reward
    # path and any extra keyword fields. KNOWN ISSUE: this helper is never used by any test in
    # this module (dead code).
    return SimpleNamespace(rm_type=None, custom_rm_path=None, **{k: v for k, v in kwargs.items()})


def test_spec_fields_win_over_args():
    # [FUNCTION DOC] A sample whose reward spec sets type math and a custom path, with different
    # values in the arguments, resolves to the spec's custom path and type.
    sample = Sample(prompt="p", reward_spec=RewardSpec(rm_type="math", custom_rm_path="pkg.fn"))
    args = SimpleNamespace(rm_type="deepscaler", custom_rm_path="other.fn")
    assert _resolve_reward_config(args, sample) == ("pkg.fn", "math")


def test_empty_spec_falls_back_to_args():
    # [FUNCTION DOC] A sample with an all-empty reward spec resolves to no custom path and the
    # arguments' type (math).
    sample = Sample(prompt="p", reward_spec=RewardSpec(rm_type=None, custom_rm_path=None))
    args = SimpleNamespace(rm_type="math", custom_rm_path=None)
    assert _resolve_reward_config(args, sample) == (None, "math")


def test_empty_spec_falls_back_to_sample_metadata_before_args():
    # [FUNCTION DOC] With an empty reward spec and metadata naming type gpqa while the arguments
    # name math, resolves to no custom path and type gpqa (metadata beats arguments).
    sample = Sample(prompt="p", reward_spec=RewardSpec(rm_type=None, custom_rm_path=None))
    sample.metadata = {"rm_type": "gpqa"}
    args = SimpleNamespace(rm_type="math", custom_rm_path=None)
    assert _resolve_reward_config(args, sample) == (None, "gpqa")


def test_no_spec_keeps_metadata_then_args_chain():
    # [FUNCTION DOC] With no reward spec and empty metadata, resolves to no custom path and the
    # arguments' type (math).
    sample = Sample(prompt="p")
    sample.metadata = {}
    args = SimpleNamespace(rm_type="math", custom_rm_path=None)
    assert _resolve_reward_config(args, sample) == (None, "math")
