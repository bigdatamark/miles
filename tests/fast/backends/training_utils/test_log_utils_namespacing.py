# [MODULE DOC] CPU tests of metric namespacing by trainer model id in the training log utilities:
# train-step metrics and their step axis are prefixed with "<id>/" when an id is set (and keep
# plain names otherwise), gathered metrics (rollout) carry the same namespace and equal what the
# tracker receives, the CPU memory metric gets its own namespaced step axis, the namespace-stripping
# helper restores plain names for CI checkers (and returns the same object when no id is set), and
# a namespaced rollout log-prob divergence still trips the CI log-prob checker. Tracking calls are
# captured through a patched tracker.
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.metrics import log_utils
from miles.utils.metric_utils import strip_metrics_namespace


def _capture(monkeypatch) -> list[tuple[dict, str]]:
    # [FUNCTION DOC] HELPER: Replaces the tracker's log function with one that appends (payload,
    # step key) to a list (arguments ignored). Returns that live list.
    calls: list[tuple[dict, str]] = []
    monkeypatch.setattr(log_utils.tracking, "log", lambda _args, payload, step_key: calls.append((payload, step_key)))
    return calls


@pytest.fixture()
def source_rank(monkeypatch) -> None:
    # [FUNCTION DOC] HELPER: Fixture; patches the parallel state getter to a fake whose effective
    # dp-with-cp axis has rank 0, size 1 and no gloo groups (the source rank), and patches the
    # object gather to return a one-element list holding the given object.
    parallel_state = SimpleNamespace(effective_dp_cp=SimpleNamespace(rank=0, size=1, gloo_groups_inner_to_outer=[]))
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(log_utils.MultiPGUtil, "gather_object", staticmethod(lambda obj, groups_inner_to_outer: [obj]))


class TestLogTrainStep:
    # [CLASS DOC] Tests for the train-step logging function.
    def test_a_policy_run_namespaces_every_train_metric_and_its_step_axis(self, monkeypatch):
        # [FUNCTION DOC] Test: with model id "alpha", loss 0.5, grad norm 2.0, rollout 2, step 1 and
        # 4 steps per rollout, exactly one tracker call is made with step key "alpha/train/step" and
        # payload {alpha/train/loss: 0.5, alpha/train/grad_norm: 2.0, alpha/train/step: 9}; the
        # function returns that same payload.
        """Two policies train at their own pace, so sharing train/step would interleave their curves."""
        calls = _capture(monkeypatch)

        log_dict = log_utils.log_train_step(
            args=Namespace(trainer_model_id="alpha"),
            loss_dict={"loss": 0.5},
            grad_norm=2.0,
            rollout_id=2,
            step_id=1,
            num_steps_per_rollout=4,
            should_log=True,
        )

        [(payload, step_key)] = calls
        assert step_key == "alpha/train/step"
        assert payload == {"alpha/train/loss": 0.5, "alpha/train/grad_norm": 2.0, "alpha/train/step": 9}
        assert log_dict == payload

    def test_a_run_without_a_policy_id_keeps_the_names_it_had(self, monkeypatch):
        # [FUNCTION DOC] Test: with no model id, the same inputs give step key "train/step", and
        # both the return value and the tracker payload are {train/loss, train/grad_norm,
        # train/step: 9} with unprefixed names.
        """Single policy runs feed dashboards written against the unprefixed names."""
        calls = _capture(monkeypatch)

        log_dict = log_utils.log_train_step(
            args=Namespace(trainer_model_id=None),
            loss_dict={"loss": 0.5},
            grad_norm=2.0,
            rollout_id=2,
            step_id=1,
            num_steps_per_rollout=4,
            should_log=True,
        )

        [(payload, step_key)] = calls
        assert step_key == "train/step"
        assert log_dict == payload == {"train/loss": 0.5, "train/grad_norm": 2.0, "train/step": 9}


class TestGatherLogData:
    # [CLASS DOC] Tests for the gather-and-log function.
    @pytest.mark.parametrize("trainer_model_id", [None, "alpha"])
    def test_the_reduced_metrics_carry_the_namespace_of_their_policy(self, source_rank, monkeypatch, trainer_model_id):
        # [FUNCTION DOC] Test (parametrized: no id, id "alpha"): gathering "rollout" metrics for
        # step 3 with one log-prob entry on the source rank makes one tracker call whose step key is
        # "<namespace>rollout/step" and whose payload, equal to the returned reduced dictionary, has
        # the prefixed log-prob key and the prefixed step key valued 3.
        """The answer is the same object tracking receives, so a reader of either sees one naming scheme."""
        calls = _capture(monkeypatch)
        args = Namespace(trainer_model_id=trainer_model_id, wandb_always_use_train_step=False)

        reduced = log_utils.gather_log_data("rollout", args, 3, {"log_probs": -1.0})

        namespace = "" if trainer_model_id is None else f"{trainer_model_id}/"
        [(payload, step_key)] = calls
        assert step_key == f"{namespace}rollout/step"
        assert reduced == payload == {f"{namespace}rollout/log_probs": -1.0, f"{namespace}rollout/step": 3}


class TestLogCpuMemory:
    # [CLASS DOC] Tests for the CPU memory logging function.
    def test_the_memory_series_of_a_policy_carries_its_own_step_axis(self, monkeypatch):
        # [FUNCTION DOC] Test: logging CPU memory for rollout 3, id "alpha", label "before" makes one
        # tracker call with step key "alpha/rollout/step"; the payload's step entry is 3 and its keys
        # are exactly the prefixed CPU-memory-before-GB metric and the prefixed step.
        """The memory point is logged per rollout, so it needs the same policy step axis as the other perf points."""
        calls = _capture(monkeypatch)

        log_utils.log_cpu_memory(
            rollout_id=3, args=Namespace(trainer_model_id="alpha", wandb_always_use_train_step=False), label="before"
        )

        [(payload, step_key)] = calls
        assert step_key == "alpha/rollout/step"
        assert payload["alpha/rollout/step"] == 3
        assert set(payload) == {"alpha/perf/cpu_memory_before_gb", "alpha/rollout/step"}


class TestStripMetricsNamespace:
    # [CLASS DOC] Tests for the namespace-stripping helper.
    def test_the_ci_checkers_see_the_names_they_are_written_against(self):
        # [FUNCTION DOC] Test: stripping the "alpha" namespace from a dictionary with prefixed ppo
        # KL and step keys returns the same entries with plain "train/..." keys.
        """Every CI assertion indexes train/ppo_kl, whichever policy produced the step."""
        namespaced = {"alpha/train/ppo_kl": 0.1, "alpha/train/step": 4}

        assert strip_metrics_namespace(namespaced, trainer_model_id="alpha") == {
            "train/ppo_kl": 0.1,
            "train/step": 4,
        }

    def test_a_single_policy_dict_is_handed_back_untouched(self):
        # [FUNCTION DOC] Test: with no model id, the helper returns the very same dictionary object.
        """Nothing was prefixed, so stripping must not copy or rename anything."""
        log_dict = {"train/ppo_kl": 0.1}

        assert strip_metrics_namespace(log_dict, trainer_model_id=None) is log_dict


class TestLogRolloutData:
    # [CLASS DOC] Tests for the rollout data logging function.
    def test_namespaced_rollout_metrics_still_trigger_the_logprob_checker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Test: with a fake single-rank parallel state patched into the log
        # utilities and the context-parallel module, gather returning its input, and the tracker
        # a no-op, logs rollout data for one sample (length 2, mask ones) with policy log-probs 0
        # and reference log-probs 1 under CI-test args (log-prob checker enabled, model id "alpha",
        # optional extra logs off). Logging raises an assertion error whose message contains "CI
        # check failed", i.e. the checker still sees divergence despite namespaced keys.
        """A policy namespace must not hide divergent rollout and reference log probabilities."""
        parallel_state = SimpleNamespace(
            tp=SimpleNamespace(rank=0),
            cp=SimpleNamespace(size=1),
            intra_dp=SimpleNamespace(size=1),
            effective_dp_cp=SimpleNamespace(rank=0, size=1, gloo_groups_inner_to_outer=[]),
            is_pp_last_stage=True,
        )
        monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
        monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
        monkeypatch.setattr(
            log_utils.MultiPGUtil, "gather_object", staticmethod(lambda obj, groups_inner_to_outer: [obj])
        )
        monkeypatch.setattr(log_utils.tracking, "log", lambda args, payload, step_key: None)
        rollout_data = {
            "total_lengths": [2],
            "response_lengths": [2],
            "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
            "log_probs": [torch.tensor([0.0, 0.0])],
            "ref_log_probs": [torch.tensor([1.0, 1.0])],
        }
        args = Namespace(
            ci_test=True,
            ci_disable_logprobs_checker=False,
            trainer_model_id="alpha",
            true_on_policy_mode=False,
            use_rollout_routing_replay=False,
            qkv_format="thd",
            wandb_always_use_train_step=False,
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        )

        with pytest.raises(AssertionError, match="CI check failed"):
            log_utils.log_rollout_data(rollout_id=0, args=args, rollout_data=rollout_data)
