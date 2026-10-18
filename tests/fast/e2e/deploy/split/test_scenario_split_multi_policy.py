# [MODULE DOC]
# Tests for the split multi-policy (solver and verifier) deployment scenario: the deployments the
# scenario installs match the example's declared identities and per-deployment arguments (five
# releases, two of them trainers), and the verification helpers over event logs written with the
# real event logger: the leader policy must have trained every declared rollout (multiple
# optimizer steps per rollout allowed), and each policy's trainer must score its engines' output
# (small mean log-probability differences accepted; large, undefined, too few or single-rollout
# reports rejected).
import dataclasses
from pathlib import Path

import pytest
from examples.infra_features.split_deployment.run_solver_verifier_gsm8k_split import (
    build_deployment_train_args,
    compute_deployment_identities,
)
from examples.multi_policy.run_solver_verifier_gsm8k import LEADER_MODEL_ID, ScriptArgs
from tests.e2e.deploy.conftest_deploy.split import scenario_split_multi_policy as scenario

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.workers.types import ClusterBackend, DeployComponent

NAMESPACE: str = "rl"
RUN_ID: str = "demo"
RUN_UUID: str = "0123456789abcdef"
NUM_DEPLOYMENTS_OF_THE_RUN: int = 5
NUM_TRAINER_DEPLOYMENTS: int = 2


@pytest.fixture
def args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Fixture returning script arguments for a kubernetes backend with the
    # demo namespace, run id and run uuid.
    return ScriptArgs(cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID, run_uuid=RUN_UUID)


class TestBuildDeployments:
    # [CLASS DOC] Tests of the scenario's deployment list against the example.
    def test_the_scenario_installs_every_part_the_example_declares_and_no_other(self, args):
        # [FUNCTION DOC] The (component, instance id) pairs of the scenario's deployments equal
        # the example's computed deployment identities, in order.
        """A scenario naming its own topology would deploy a run the example's README never describes."""
        deployments = scenario._build_deployments(args)

        assert [
            (one.deploy_component, one.deploy_instance_id) for one in deployments
        ] == compute_deployment_identities(args)

    def test_the_scenario_deploys_the_two_policy_five_release_shape_its_spec_describes(self, args):
        # [FUNCTION DOC] There are exactly 5 deployments of which exactly 2 are trainers.
        """Every assertion here follows the example, so a run that shrank to one policy would satisfy them all."""
        deployments = scenario._build_deployments(args)
        trainers = [one for one in deployments if one.deploy_component is DeployComponent.TRAINER]

        assert len(deployments) == NUM_DEPLOYMENTS_OF_THE_RUN
        assert len(trainers) == NUM_TRAINER_DEPLOYMENTS

    def test_every_part_is_launched_with_the_arguments_the_example_composes_for_it(self, args):
        # [FUNCTION DOC] There are 5 deployments, and each one's arguments equal the example's
        # deployment argument builder applied to the script arguments with that deployment's
        # component and instance id substituted.
        """Arguments rebuilt here rather than imported would drift from the commands a reader types."""
        deployments = scenario._build_deployments(args)

        assert len(deployments) == NUM_DEPLOYMENTS_OF_THE_RUN
        for one in deployments:
            assert one.train_args == build_deployment_train_args(
                dataclasses.replace(
                    args, deploy_component=one.deploy_component, deploy_instance_id=one.deploy_instance_id
                )
            )


class TestVerify:
    # [CLASS DOC] Tests of the verification helpers over written event logs.
    def test_a_run_whose_leader_trained_every_rollout_it_declared_passes(self, dump_dir, args):
        # [FUNCTION DOC] Grad-norm metrics for the leader policy at every declared rollout pass
        # the leader-ran-all-rollouts check.
        """The happy path has to stay reachable, or the refusals below prove nothing."""
        _write_metrics(dump_dir, _leader_grad_norms(range(args.num_rollout)))

        scenario._assert_leader_policy_ran_all_rollouts(dump_dir, num_rollout=args.num_rollout)

    def test_a_run_whose_leader_skipped_a_rollout_is_caught(self, dump_dir, args):
        # [FUNCTION DOC] Metrics missing rollout 1: the check raises an assertion error matching
        # "a deployment failed rather than".
        """A rollout the leader never trained is a deployment that died, dressed up as a shorter run."""
        _write_metrics(dump_dir, _leader_grad_norms([one for one in range(args.num_rollout) if one != 1]))

        with pytest.raises(AssertionError, match="a deployment failed rather than"):
            scenario._assert_leader_policy_ran_all_rollouts(dump_dir, num_rollout=args.num_rollout)

    def test_a_leader_that_trained_one_rollout_over_several_steps_still_passes(self, dump_dir, args):
        # [FUNCTION DOC] Each rollout id logged three times (several optimizer steps) still
        # passes.
        """How many optimizer steps a rollout takes is the workload's business, not this assertion's."""
        _write_metrics(dump_dir, _leader_grad_norms(list(range(args.num_rollout)) * 3))

        scenario._assert_leader_policy_ran_all_rollouts(dump_dir, num_rollout=args.num_rollout)

    def test_a_policy_scoring_what_its_own_engines_generated_passes(self, dump_dir):
        # [FUNCTION DOC] Log-probability absolute differences of 0.01, 0.02 and 0.015 for the
        # leader model id pass the trainer-scores-engine-output check.
        """Trainer and engines serving the same weights agree on the log probability of the same tokens."""
        _write_metrics(dump_dir, _logprob_diffs(LEADER_MODEL_ID, [0.01, 0.02, 0.015]))

        scenario._assert_trainer_scores_engine_output(dump_dir, model_id=LEADER_MODEL_ID)

    def test_a_policy_whose_trainer_and_engines_disagree_is_caught(self, dump_dir):
        # [FUNCTION DOC] A difference of 5.0 among small ones raises matching "different weights".
        """A trainer scoring another policy's tokens is exactly what one release per policy could get wrong."""
        _write_metrics(dump_dir, _logprob_diffs(LEADER_MODEL_ID, [0.01, 0.02, 5.0]))

        with pytest.raises(AssertionError, match="different weights"):
            scenario._assert_trainer_scores_engine_output(dump_dir, model_id=LEADER_MODEL_ID)

    def test_a_policy_drifting_far_less_than_a_wrong_policy_would_is_still_caught(self, dump_dir):
        # [FUNCTION DOC] A maximum difference of 0.2 (moderate) raises matching "different
        # weights": the threshold is tight, far below what a wrong policy would produce.
        """A trainer and its engines on the same weights agree far closer than this, so this much is already wiring."""
        _write_metrics(dump_dir, _logprob_diffs(LEADER_MODEL_ID, [0.01, 0.02, 0.2]))

        with pytest.raises(AssertionError, match="different weights"):
            scenario._assert_trainer_scores_engine_output(dump_dir, model_id=LEADER_MODEL_ID)

    def test_a_policy_whose_comparison_came_out_undefined_is_caught(self, dump_dir):
        # [FUNCTION DOC] A not-a-number difference among reports raises an assertion error whose
        # message mentions that the maximum function passes over it.
        """max() passes over a nan, so the worst value it reports would be the one healthy rollout."""
        _write_metrics(dump_dir, _logprob_diffs(LEADER_MODEL_ID, [0.01, float("nan")]))

        with pytest.raises(AssertionError, match="max\\(\\) passes over"):
            scenario._assert_trainer_scores_engine_output(dump_dir, model_id=LEADER_MODEL_ID)

    def test_a_policy_that_reported_the_comparison_almost_never_is_caught(self, dump_dir):
        # [FUNCTION DOC] A single report raises matching "nothing compares".
        """One agreeing rollout would let a run whose engines never generated pass as one that did."""
        _write_metrics(dump_dir, _logprob_diffs(LEADER_MODEL_ID, [0.01]))

        with pytest.raises(AssertionError, match="nothing compares"):
            scenario._assert_trainer_scores_engine_output(dump_dir, model_id=LEADER_MODEL_ID)

    def test_one_rollout_reported_over_several_steps_is_not_several_rollouts(self, dump_dir):
        # [FUNCTION DOC] Three reports all at rollout 0 raise matching "reported for only 1
        # rollout" (optimizer steps within a rollout prove the comparison once).
        """Optimizer steps within one rollout all score the same generated tokens, so they prove it once."""
        _write_metrics(dump_dir, _logprob_diffs_of_rollouts(LEADER_MODEL_ID, [(0, 0.01), (0, 0.02), (0, 0.015)]))

        with pytest.raises(AssertionError, match="reported for only 1 rollout"):
            scenario._assert_trainer_scores_engine_output(dump_dir, model_id=LEADER_MODEL_ID)


@pytest.fixture
def dump_dir(tmp_path) -> str:
    # [FUNCTION DOC] HELPER: Fixture returning a "run" subdirectory path under the temporary path
    # as text (not created here).
    return str(tmp_path / "run")


def _leader_grad_norms(rollout_ids) -> list[tuple[int, dict[str, float]]]:
    # [FUNCTION DOC] HELPER: For each rollout id, returns a pair of the id and a metrics mapping
    # with the leader model's train grad norm set to 0.5.
    return [(rollout_id, {f"{LEADER_MODEL_ID}/train/grad_norm": 0.5}) for rollout_id in rollout_ids]


def _logprob_diffs(model_id: str, values: list[float]) -> list[tuple[int, dict[str, float]]]:
    # [FUNCTION DOC] HELPER: Pairs the values with consecutive rollout ids from 0 and returns the
    # log-probability difference metric points for the model.
    return _logprob_diffs_of_rollouts(model_id, list(enumerate(values)))


def _logprob_diffs_of_rollouts(model_id: str, points: list[tuple[int, float]]) -> list[tuple[int, dict[str, float]]]:
    # [FUNCTION DOC] HELPER: Converts (rollout id, value) pairs into (rollout id, metrics mapping)
    # pairs where the metric is the model's train rollout log-probability absolute difference.
    key = f"{model_id}/train/train_rollout_logprob_abs_diff"
    return [(rollout_id, {key: value}) for rollout_id, value in points]


def _write_metrics(dump_dir: str, points: list[tuple[int, dict[str, float]]]) -> None:
    # [FUNCTION DOC] HELPER: Writes one metric event per (rollout id, metrics) point (attempt 0)
    # into the "events" subdirectory of the dump directory through the real event logger, without
    # printing, then closes the logger.
    event_logger = EventLogger(
        log_dir=Path(dump_dir) / "events", source=SimpleProcessIdentity(component="main"), file_name="main.jsonl"
    )
    for rollout_id, metrics in points:
        event_logger.log(MetricEvent, dict(rollout_id=rollout_id, attempt=0, metrics=metrics), print_log=False)
    event_logger.close()
