# [MODULE DOC]
# Unit tests for the launcher recipe of the solver/verifier gsm8k multi-policy example. They check the
# training command line built by the build-train-args function, and the per-policy megatron and sglang
# config documents produced by the compute functions, without starting any run. Module constants: the
# run id used for every script-args object, and the pair of flags whose values are pseudo-file configs.
# Fixtures: an autouse one that removes the wandb api key from the environment, one building the
# script-args object, and one parsing the built command line into a flag-to-value mapping.
import shlex

import pytest
from examples.multi_policy.run_solver_verifier_gsm8k import (
    LEADER_MODEL_ID,
    MODEL_IDS,
    SHARED_TRAINER_OVERRIDES,
    ScriptArgs,
    build_train_args,
    compute_events_dir,
    compute_megatron_config,
    compute_rollout_data_path_template,
    compute_save_dir,
    compute_sglang_config,
    compute_trainer_id,
)
from tests.e2e.conftest_multi_policy import NUM_VERIFIED_ARGS_PER_POLICY

from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX

RUN_ID: str = "demo"
CONFIG_FLAGS: tuple[str, ...] = ("--megatron-config", "--sglang-config")


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture. Removes the wandb api key from the environment (no error
    # if absent) so the built arguments never depend on the developer's wandb login. Returns nothing.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


@pytest.fixture
def args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Fixture. Returns a script-args object with the fixed run id and all other
    # fields at their defaults.
    return ScriptArgs(run_id=RUN_ID)


@pytest.fixture
def flags(args: ScriptArgs) -> dict[str, str | None]:
    # [FUNCTION DOC] HELPER: Fixture. Builds the train command line for the default script args and
    # returns it parsed into a mapping from flag to its value (or none for value-less flags).
    return _flags_of(build_train_args(args))


class TestBuildTrainArgs:
    # [CLASS DOC] Tests of the build-train-args function: the complete flag set of the recipe, the
    # retract pause mode, config hand-over as pseudo files, and the rollout gpu count override.
    def test_the_recipe_is_exactly_these_flags_and_values(self, flags, args):
        # [FUNCTION DOC] Asserts that the parsed flags, excluding the two config flags, equal one exact
        # mapping of about seventy flags to values. Values derived from the script args are the
        # solver (leader model) checkpoint path with a trailing slash for hf-checkpoint and ref-load,
        # the save directory, save interval, data directory based gsm8k train parquet prompt data, number
        # of rollouts, the events directory, the rollout data path template, and the actor and rollout gpu
        # counts. The remainder are literal constants (batch sizes, lengths, parallel sizes of one,
        # grpo, bridge mode, flash attention and so on). Value-less flags map to none.
        """A flag silently dropped from an 80-line recipe surfaces as a 4-gpu run that hangs, not as a red test."""
        solver_path = args.model_path_of_model_id[LEADER_MODEL_ID]

        assert {flag: value for flag, value in flags.items() if flag not in CONFIG_FLAGS} == {
            "--hf-checkpoint": f"{solver_path}/",
            "--ref-load": f"{solver_path}/",
            "--save": str(compute_save_dir(args)),
            "--save-interval": str(args.save_interval),
            "--custom-generate-function-path": "examples.multi_policy.solver_verifier.generate",
            "--fully-async": None,
            "--prompt-data": f"{args.data_dir}/gsm8k/train.parquet",
            "--input-key": "messages",
            "--label-key": "label",
            "--rollout-shuffle": None,
            "--num-rollout": str(args.num_rollout),
            "--rollout-batch-size": "32",
            "--n-samples-per-prompt": "8",
            "--rollout-max-response-len": "1024",
            "--rollout-temperature": "1",
            "--global-batch-size": "256",
            "--dynamic-sampling-filter-path": "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
            "--reward-key": "reward_value",
            "--log-reward-category": "outcome",
            "--pause-generation-mode": "retract",
            "--optimizer": "adam",
            "--lr": "1e-6",
            "--eval-interval": "20",
            "--eval-prompt-data": "gsm8k",
            "--n-samples-per-eval-prompt": "1",
            "--eval-max-response-len": "1024",
            "--eval-top-k": "1",
            "--custom-eval-rollout-log-function-path": "examples.multi_policy.solver_verifier.split_eval_data_by_policy",
            "--advantage-estimator": "grpo",
            "--use-kl-loss": None,
            "--tensor-model-parallel-size": "1",
            "--sequence-parallel": None,
            "--pipeline-model-parallel-size": "1",
            "--context-parallel-size": "1",
            "--expert-model-parallel-size": "1",
            "--expert-tensor-parallel-size": "1",
            "--use-dynamic-batch-size": None,
            "--max-tokens-per-gpu": "9216",
            "--rollout-num-gpus-per-engine": "1",
            "--sglang-mem-fraction-static": "0.65",
            "--sglang-enable-metrics": None,
            "--ci-test": None,
            "--save-debug-event-data": str(compute_events_dir(args)),
            "--save-debug-rollout-data": compute_rollout_data_path_template(args),
            "--attention-dropout": "0.0",
            "--hidden-dropout": "0.0",
            "--accumulate-allreduce-grads-in-fp32": None,
            "--attention-softmax-in-fp32": None,
            "--attention-backend": "flash",
            "--actor-num-nodes": "1",
            "--actor-num-gpus-per-node": str(args.actor_num_gpus_per_policy),
            "--rollout-num-gpus": str(args.rollout_num_gpus),
            "--megatron-to-hf-mode": "bridge",
        }

    def test_the_run_retracts_what_it_has_in_flight_when_generation_pauses(self, flags):
        # [FUNCTION DOC] Asserts the fully-async flag is present without a value and the pause
        # generation mode is the retract mode.
        """in_place only ever bought this fully async run its way around the pre-refactor flush_cache deadlock."""
        assert flags["--fully-async"] is None
        assert flags["--pause-generation-mode"] == "retract"

    def test_the_per_policy_configs_are_handed_over_as_files_of_their_own(self, flags):
        # [FUNCTION DOC] Asserts both config flags (megatron and sglang) are present, and that each
        # value starts with the pseudo-file prefix, i.e. the config is passed inline as a pseudo file.
        """Their contents are asserted below; here they only have to reach the run at all."""
        assert sorted(flag for flag in flags if flag in CONFIG_FLAGS) == sorted(CONFIG_FLAGS)
        assert all(flags[flag].startswith(PSEUDO_FILE_PREFIX) for flag in CONFIG_FLAGS)

    def test_a_run_told_how_many_engine_gpus_to_expect_declares_that_instead(self, args):
        # [FUNCTION DOC] Builds the command line with an explicit rollout gpu count of one and asserts
        # the rollout-num-gpus flag then has the value one rather than the script-args default.
        """A split deployment installs the engines of one policy, and counts only those."""
        flags = _flags_of(build_train_args(args, rollout_num_gpus=1))

        assert flags["--rollout-num-gpus"] == "1"


class TestComputeMegatronConfig:
    # [CLASS DOC] Tests of the compute-megatron-config function, whose result holds a list of
    # trainer entries, each with a model id, a trainer id and a mapping of overrides.
    def test_the_run_trains_exactly_the_policies_the_example_declares(self, args):
        # [FUNCTION DOC] Asserts the trainers' model ids equal the declared model id list in order, and
        # that each trainer id equals the compute-trainer-id result for its model id.
        """A trainer short of the model ids is a policy nothing ever trains, and the run still starts."""
        trainers = compute_megatron_config(args)["trainers"]

        assert [one["model_id"] for one in trainers] == MODEL_IDS
        assert [one["trainer_id"] for one in trainers] == [compute_trainer_id(one) for one in MODEL_IDS]

    def test_every_policy_is_given_the_arguments_the_e2e_test_counts(self, args):
        # [FUNCTION DOC] Asserts the number of overrides per model id equals the verified-argument
        # count mapping imported from the multi-policy e2e conftest.
        """The e2e assertion verifies a fixed number of overrides per policy, and would shrink with this one."""
        trainers = compute_megatron_config(args)["trainers"]

        assert {one["model_id"]: len(one["overrides"]) for one in trainers} == NUM_VERIFIED_ARGS_PER_POLICY

    def test_every_policy_is_given_the_shared_hyperparameters_of_this_recipe(self, args):
        # [FUNCTION DOC] For every trainer, asserts the subset of its overrides keyed by the shared
        # trainer override names equals the shared overrides mapping exactly.
        """A policy trained on other hyperparameters than its neighbour is not the comparison this run makes."""
        for trainer in compute_megatron_config(args)["trainers"]:
            assert {key: trainer["overrides"][key] for key in SHARED_TRAINER_OVERRIDES} == SHARED_TRAINER_OVERRIDES

    def test_a_run_of_one_policy_carries_only_that_policy(self, args):
        # [FUNCTION DOC] Computes the config restricted to the leader model id and asserts the only
        # trainer has that model id.
        """A split deployment installs a trainer per policy, and hands each command its own config."""
        trainers = compute_megatron_config(args, model_ids=[LEADER_MODEL_ID])["trainers"]

        assert [one["model_id"] for one in trainers] == [LEADER_MODEL_ID]


class TestComputeSglangConfig:
    # [CLASS DOC] Tests of the compute-sglang-config function, whose result holds a list of models,
    # each with a name, a model path and a list of server groups carrying gpu counts.
    def test_the_run_serves_exactly_the_policies_the_example_declares(self, args):
        # [FUNCTION DOC] Asserts model names equal the declared model id list in order, and model paths
        # equal each id's path from the script args' model-path mapping.
        """A policy without engines generates nothing, and only its own reward would ever show it."""
        models = compute_sglang_config(args)["sglang"]

        assert [one["name"] for one in models] == MODEL_IDS
        assert [one["model_path"] for one in models] == [args.model_path_of_model_id[one] for one in MODEL_IDS]

    def test_every_policy_is_served_by_the_engine_gpus_the_run_declares(self, args):
        # [FUNCTION DOC] Collects the gpu count of every server group of every model, asserts it is the
        # per-model rollout gpu count once per declared model, and that the sum equals the total rollout
        # gpu count of the script args.
        """Engines short of the gpus the run counts leave it waiting for cells nothing installs."""
        models = compute_sglang_config(args)["sglang"]
        declared = [group["num_gpus"] for one in models for group in one["server_groups"]]

        assert declared == [args.rollout_num_gpus_per_model] * len(MODEL_IDS)
        assert sum(declared) == args.rollout_num_gpus

    def test_a_run_of_one_policy_serves_only_that_policy(self, args):
        # [FUNCTION DOC] Computes the config restricted to the leader model id and asserts the only
        # model name is that id.
        """A split deployment installs the engines of one policy, and hands that command its own config."""
        models = compute_sglang_config(args, model_ids=[LEADER_MODEL_ID])["sglang"]

        assert [one["name"] for one in models] == [LEADER_MODEL_ID]


def _flags_of(train_args: str) -> dict[str, str | None]:
    # [FUNCTION DOC] HELPER: Splits the command line string with shell-style quoting rules into tokens
    # and returns a mapping from each double-dash token to the next token, or none when it is the last
    # token or the next token is itself a flag. Non-flag tokens that are not a flag's value are skipped.
    # Raises an assertion error if the same flag appears twice.
    tokens = shlex.split(train_args)
    flags: dict[str, str | None] = {}
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue

        assert token not in flags, f"{token} is declared twice in these arguments, and only one of them can win"
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        flags[token] = None if following is None or following.startswith("--") else following
    return flags
