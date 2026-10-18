# [MODULE DOC]
# Tests for per-policy metric namespacing helpers in the metric utilities module. The namespacing
# function takes a metrics mapping, an optional trainer model id, a step-axis name and a step value,
# and returns the logged mapping (with the step added under the step name) plus the name of the step
# key; with a model id every key gets "<id>/" prepended. The stripping function removes one
# policy's prefix from its keys and leaves other policies' keys unchanged.
from miles.utils.metric_utils import namespace_metrics, strip_metrics_namespace


class TestNamespaceMetrics:
    # [CLASS DOC] Tests of adding a policy namespace to metrics.
    def test_a_single_policy_run_keeps_the_names_it_had(self):
        # [FUNCTION DOC] With no model id, keys are unchanged, the step is added under the plain
        # step name, and that name is returned as the step key.
        """Every existing dashboard query is written against the unprefixed names."""
        log_dict, step_key = namespace_metrics(
            {"rollout/reward": 1.0}, trainer_model_id=None, step_name="rollout/step", step=7
        )

        assert log_dict == {"rollout/reward": 1.0, "rollout/step": 7}
        assert step_key == "rollout/step"

    def test_every_policy_gets_its_own_namespace_and_step_axis(self):
        # [FUNCTION DOC] With model id "alpha", every metric key and the step key are prefixed with
        # "alpha/", and the prefixed step key is returned.
        """Two policies advance at their own pace, so one shared step axis would interleave their curves."""
        log_dict, step_key = namespace_metrics(
            {"rollout/reward": 1.0}, trainer_model_id="alpha", step_name="rollout/step", step=7
        )

        assert log_dict == {"alpha/rollout/reward": 1.0, "alpha/rollout/step": 7}
        assert step_key == "alpha/rollout/step"


class TestStripMetricsNamespace:
    # [CLASS DOC] Tests of removing one policy's namespace from metrics.
    def test_another_policys_namespace_is_left_unchanged(self) -> None:
        # [FUNCTION DOC] Stripping "alpha" from a mapping holding alpha and beta keys removes the
        # prefix from alpha's key only; beta's keys keep their prefix.
        """Normalizing one policy must preserve every key owned by another policy."""
        metrics = {"alpha/train/loss": 1.0, "beta/train/loss": 2.0, "beta/train/step": 7}

        assert strip_metrics_namespace(metrics, trainer_model_id="alpha") == {
            "train/loss": 1.0,
            "beta/train/loss": 2.0,
            "beta/train/step": 7,
        }
