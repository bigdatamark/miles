# [MODULE DOC]
# Tests for the Megatron trainer configuration module: resolving the optional multi-trainer yaml
# (or inline base64 payload) into a configuration of trainers (ids, roles, model ids, overrides),
# validation (unique DNS-label ids, length limits, reserved names, strict keys), typing and
# whitelisting of per-trainer argument overrides, per-trainer argument overlays, checkpoint
# directory derivation and fresh-versus-resume load resolution, the synthesized critic trainer,
# and consistency of the override whitelists with the real Megatron argument parser. Command-line
# arguments are modeled as a namespace built from shared parser defaults plus per-test values.
import argparse
import re
import sys
from argparse import Namespace
from types import SimpleNamespace

import pydantic
import pytest
import yaml
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.backends.megatron_utils import megatron_config as megatron_config_module
from miles.backends.megatron_utils.megatron_config import (
    MODEL_DEFINITION_ARGS,
    PER_POLICY_ARGS,
    _has_megatron_checkpoint,
    _resolve_overrides,
    compute_trainer_args,
    compute_trainer_checkpoint_dir,
    get_megatron_arg_parser,
    resolve_args_checkpoint_load,
    resolve_megatron_config,
)
from miles.utils.external_utils.model_args_utils import load_model_args
from miles.utils.workers.naming import TRAINER_ID_MAX_LENGTH


def _write_yaml(data: dict, tmp_path) -> str:
    # [FUNCTION DOC] HELPER: Dumps the given mapping as yaml to a file named megatron.yaml in the
    # temporary directory and returns its path as a string.
    path = tmp_path / "megatron.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


def _model_args(args: Namespace, *, model_id: str) -> Namespace:
    # [FUNCTION DOC] HELPER: Resolves the trainer configuration from the arguments, looks up the
    # trainer for the model id (first trainer with that id; unknown id raises a key error), and
    # returns the base arguments overlaid with that trainer's settings.
    return compute_trainer_args(args, resolve_megatron_config(args).get(model_id))


def _make_args(megatron_config: str | None = None, **overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a run-argument namespace. Starts from the shared parser
    # defaults, then applies a baseline of test values (learning rate 1e-6, tensor parallel 1,
    # core conversion mode, GRPO advantage estimator, no critic, base checkpoint path, rollout
    # batch size 8 with 4 samples per prompt, world size 8, a tokenizer path equal to the base
    # checkpoint, and so on), then the optional config path and the keyword overrides, which win.
    defaults = dict(
        megatron_config=megatron_config,
        lr=1e-6,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        hf_checkpoint="/models/base",
        global_batch_size=None,
        eps_clip_high=None,
        save=None,
        load=None,
        advantage_estimator="grpo",
        use_critic=False,
        num_steps_per_rollout=None,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        megatron_to_hf_mode="core",
        ref_load=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        ref_ckpt_step=None,
        ckpt_step=None,
        start_rollout_id=None,
        optimizer="adam",
        use_distributed_optimizer=True,
        world_size=8,
        debug_disable_optimizer=False,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        kl_coef=0.1,
        use_opd=True,
        disable_param_buffers_cpu_backup=True,
        lr_warmup_iters=10,
        critic_load=None,
        critic_save=None,
        critic_lr=None,
        critic_lr_warmup_iters=None,
        fp16=False,
        seq_length=4096,
        vocab_size=None,
        padded_vocab_size=None,
        tokenizer_model="/models/base",
        tokenizer_type="HuggingFaceTokenizer",
        multi_lora_n_adapters=0,
    )
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


class TestResolveMegatronConfig:
    # [CLASS DOC] Tests for resolving the trainer configuration from the arguments: the legacy
    # synthesized single actor, yaml parsing, id derivation and validation, and lookups.
    def test_a_run_without_the_flag_synthesizes_a_plain_actor_trainer(self):
        # [FUNCTION DOC] No config path: expects exactly one trainer with id and role actor, no
        # model id and no overrides; the model id list is empty and the run is not multi-policy.
        """Legacy single policy runs must keep working, with no model id anywhere downstream."""
        config = resolve_megatron_config(_make_args())

        assert [(t.trainer_id, t.role, t.model_id, t.overrides) for t in config.trainers] == [
            ("actor", "actor", None, {})
        ]
        assert config.model_ids == []
        assert not config.is_multi_policy

    def test_the_legacy_megatron_key_is_still_accepted(self, tmp_path):
        # [FUNCTION DOC] A yaml whose top-level key is the legacy name for the trainer list, with
        # two entries, resolves to model ids a and b in order.
        """Configs written against the first name of the field must keep resolving."""
        path = _write_yaml({"megatron": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)

        assert resolve_megatron_config(_make_args(path)).model_ids == ["a", "b"]

    def test_the_yaml_model_ids_become_the_trainer_model_ids(self, tmp_path):
        # [FUNCTION DOC] Two trainers (the first with a learning-rate override). Expects model ids
        # a, b in order, leader model id a (the first), and the run flagged multi-policy.
        """The `model_id` field is the source of truth for trainer_model_id and spec names."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"lr": 1e-5}}, {"model_id": "b"}]}, tmp_path)

        config = resolve_megatron_config(_make_args(path))

        assert config.model_ids == ["a", "b"]
        assert config.leader_model_id == "a"
        assert config.is_multi_policy

    def test_the_eval_model_id_is_refused(self, tmp_path):
        # [FUNCTION DOC] A trainer with model id eval makes resolution raise a pydantic validation
        # error mentioning shared eval rollouts.
        """A policy named eval would overwrite the shared evaluation rollout artifacts."""
        path = _write_yaml({"trainers": [{"model_id": "eval"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="shared eval rollouts"):
            resolve_megatron_config(_make_args(path))

    def test_the_first_model_is_the_leader_policy(self, tmp_path):
        # [FUNCTION DOC] With trainers listed as second then first, the leader model id is the
        # first listed ("second"): leadership is positional, not alphabetical.
        """The leader owns the global checkpoint index, so its identity must be positional and stable."""
        path = _write_yaml({"trainers": [{"model_id": "second"}, {"model_id": "first"}]}, tmp_path)

        assert resolve_megatron_config(_make_args(path)).leader_model_id == "second"

    def test_an_inline_base64_payload_is_accepted(self, tmp_path):
        # [FUNCTION DOC] Passes an encoded inline payload (built by the shared encoder fixture for
        # a single model id "solo") instead of a file path; expects model ids to be solo only.
        # The temporary path fixture is unused.
        """Launchers that cannot ship a file still need to pass the config."""
        config = resolve_megatron_config(_make_args(encode_megatron_config("solo")))

        assert config.model_ids == ["solo"]

    def test_duplicate_trainer_ids_are_refused(self, tmp_path):
        # [FUNCTION DOC] Two entries with the same model id (hence the same derived trainer id)
        # raise a validation error saying trainer ids must be unique.
        """Two entries sharing a trainer id would land in one controller and one engine pool."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "a"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="trainer ids must be unique"):
            resolve_megatron_config(_make_args(path))

    def test_a_trainer_id_defaults_to_the_model_id_and_the_role(self, tmp_path):
        # [FUNCTION DOC] Without explicit ids, trainer ids are model id, a dash, then role
        # (a-actor, b-actor) and both roles default to actor.
        """The trainer id addresses a pool, so its default must stay the name every deployment already uses."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)

        config = resolve_megatron_config(_make_args(path))

        assert [trainer.trainer_id for trainer in config.trainers] == ["a-actor", "b-actor"]
        assert [trainer.role for trainer in config.trainers] == ["actor", "actor"]

    def test_an_explicit_trainer_id_wins_over_the_derived_one(self, tmp_path):
        # [FUNCTION DOC] An explicit trainer id in the yaml is kept verbatim.
        """A deployment that already named its pools must be able to keep those names."""
        path = _write_yaml({"trainers": [{"model_id": "a", "trainer_id": "legacy-actor"}]}, tmp_path)

        assert resolve_megatron_config(_make_args(path)).trainers[0].trainer_id == "legacy-actor"

    def test_an_explicit_trainer_id_colliding_with_a_derived_one_is_refused(self, tmp_path):
        # [FUNCTION DOC] Model b gets an explicit trainer id equal to the id derived for model a
        # (a-actor). Uniqueness is checked after derivation, so a validation error is raised.
        """Uniqueness has to hold across both spellings, or two trainers would share one engine pool."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b", "trainer_id": "a-actor"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="trainer ids must be unique"):
            resolve_megatron_config(_make_args(path))

    def test_a_trainer_id_that_is_not_a_dns_label_is_refused(self, tmp_path):
        # [FUNCTION DOC] An explicit trainer id with uppercase letters and an underscore raises a
        # validation error mentioning trainer ids.
        """A trainer id is embedded in Kubernetes pool names, which must be lowercase DNS labels."""
        path = _write_yaml({"trainers": [{"model_id": "a", "trainer_id": "Legacy_Actor"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="trainer ids"):
            resolve_megatron_config(_make_args(path))

    def test_a_trainer_id_too_long_for_its_controller_pool_is_refused(self, tmp_path):
        # [FUNCTION DOC] A lowercase trainer id one character longer than the maximum length
        # constant raises a validation error containing "longer than".
        """A trainer id must leave room for the controller suffix in its Kubernetes pool name."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "trainer_id": "a" * (TRAINER_ID_MAX_LENGTH + 1)}]}, tmp_path
        )

        with pytest.raises(pydantic.ValidationError, match="longer than"):
            resolve_megatron_config(_make_args(path))

    def test_several_entries_of_one_model_id_are_not_a_multi_policy_run(self, tmp_path):
        # [FUNCTION DOC] One yaml model with the critic enabled expands to an actor and a critic
        # trainer (ids a-actor, a-critic) sharing model id a; the model id list and leader are
        # just a, and the run is not multi-policy.
        """An actor and a critic of one policy share its id, and one policy is not several policies."""
        path = _write_yaml({"trainers": [{"model_id": "a"}]}, tmp_path)

        config = resolve_megatron_config(_make_args(path, use_critic=True))

        assert [trainer.trainer_id for trainer in config.trainers] == ["a-actor", "a-critic"]
        assert config.model_ids == ["a"]
        assert config.leader_model_id == "a"
        assert not config.is_multi_policy

    def test_an_unknown_yaml_key_is_refused(self, tmp_path):
        # [FUNCTION DOC] A misspelled key (override instead of overrides) raises an error naming
        # the key; the config model is strict about extra fields.
        """A strict model turns a typo into a startup error instead of a silently ignored setting."""
        path = _write_yaml({"trainers": [{"model_id": "a", "override": {"lr": 1e-5}}]}, tmp_path)

        with pytest.raises(Exception, match="override"):
            resolve_megatron_config(_make_args(path))

    def test_getting_an_unknown_model_id_fails_loudly(self, tmp_path):
        # [FUNCTION DOC] Looking up a model id absent from the config raises a key error saying
        # unknown trainer model id.
        """Callers routing by model id must not silently fall back to another policy."""
        path = _write_yaml({"trainers": [{"model_id": "a"}]}, tmp_path)

        with pytest.raises(KeyError, match="Unknown trainer model id"):
            resolve_megatron_config(_make_args(path)).get("b")

    def test_a_config_declaring_no_trainer_is_refused(self, tmp_path):
        # [FUNCTION DOC] An empty trainer list raises an assertion error saying the config must
        # declare at least one trainer.
        """An empty list would resolve to a run with nothing to train, and fail much later and less clearly."""
        path = _write_yaml({"trainers": []}, tmp_path)

        with pytest.raises(AssertionError, match="must declare at least one trainer"):
            resolve_megatron_config(_make_args(path))

    def test_getting_a_model_id_answers_its_first_trainer(self, tmp_path):
        # [FUNCTION DOC] With the critic enabled, trainers are ordered actor then critic, and
        # looking up the shared model id returns the actor.
        """Callers ask by model id and expect the actor: the critic of that policy is addressed by role."""
        path = _write_yaml({"trainers": [{"model_id": "a"}]}, tmp_path)

        config = resolve_megatron_config(_make_args(path, use_critic=True))

        assert [trainer.role for trainer in config.trainers] == ["actor", "critic"]
        assert config.get("a").role == "actor"

    def test_a_run_without_the_flag_has_no_leader_model_id(self):
        # [FUNCTION DOC] A run without a config path has no leader model id (none).
        """A single policy run has no leader to index the trainers by, and must answer None rather than invent one."""
        assert resolve_megatron_config(_make_args()).leader_model_id is None


class TestDerivedPerPolicyArgs:
    # [CLASS DOC] Tests for validation of model ids that are embedded in derived names and paths.
    def test_a_model_id_that_escapes_its_checkpoint_directory_is_refused(self, tmp_path):
        # [FUNCTION DOC] A model id with a parent-directory path segment raises a validation error
        # saying the ids are not usable as Kubernetes pool names (the DNS-label rule also keeps it
        # a single path component).
        """A model id is pasted into --save and --load, so it must stay one path component."""
        path = _write_yaml({"trainers": [{"model_id": "../evil"}, {"model_id": "b"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="not usable as Kubernetes pool names"):
            resolve_megatron_config(_make_args(path))

    @pytest.mark.parametrize("model_id", ["policy_a", "PolicyA", "-policy", "policy-", "policy.a"])
    def test_a_model_id_that_is_not_a_dns_label_is_refused(self, tmp_path, model_id):
        # [FUNCTION DOC] Parametrized over ids with an underscore, uppercase letters, a leading or
        # trailing dash, or a dot; each raises the same validation error as above.
        """A model id is embedded in Kubernetes pool names, which must be lowercase DNS labels."""
        path = _write_yaml({"trainers": [{"model_id": model_id}, {"model_id": "b"}]}, tmp_path)

        with pytest.raises(pydantic.ValidationError, match="not usable as Kubernetes pool names"):
            resolve_megatron_config(_make_args(path))

    @pytest.mark.parametrize("model_id", ["default", "policy-a", "a1", "a-b-c"])
    def test_lowercase_dns_labels_are_accepted(self, tmp_path, model_id):
        # [FUNCTION DOC] Parametrized over valid lowercase labels (including digits and inner
        # dashes); each resolves, giving model ids [id, "other"].
        """The ids the docs and examples use must survive validation."""
        path = _write_yaml({"trainers": [{"model_id": model_id}, {"model_id": "other"}]}, tmp_path)

        assert resolve_megatron_config(_make_args(path)).model_ids == [model_id, "other"]


class TestOverrideCoercion:
    # [CLASS DOC] Tests for typing and validating per-trainer override values at resolution time.
    def test_a_value_is_typed_by_the_declared_argument_not_by_the_yaml_scalar(self, tmp_path):
        # [FUNCTION DOC] String yaml values for the learning rate ("5e-7", which yaml does not
        # parse as a float) and the global batch size ("128") are converted using the parser's
        # declared types, giving a float and an integer in the resolved overrides.
        """YAML reads `5e-7` as a string, so an untyped overlay would train against a string learning rate."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"lr": "5e-7", "global_batch_size": "128"}}]}, tmp_path
        )

        overrides = resolve_megatron_config(_make_args(path)).get("a").overrides

        assert overrides == {"lr": 5e-7, "global_batch_size": 128}

    def test_a_boolean_argument_given_a_non_boolean_is_refused(self, tmp_path):
        # [FUNCTION DOC] A flag argument (sequence parallel) given a non-boolean string raises an
        # assertion error saying it is not a boolean.
        """`sequence_parallel: yes-please` would otherwise become a truthy string."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"sequence_parallel": "yes-please"}}]}, tmp_path
        )

        with pytest.raises(AssertionError, match="not a boolean"):
            resolve_megatron_config(_make_args(path))

    def test_an_override_without_a_value_is_refused(self, tmp_path):
        # [FUNCTION DOC] An override whose yaml value is null raises an assertion error saying it
        # has no value.
        """A key written with an empty YAML value reads as None, which no argument can be set to."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"eps_clip_high": None}}]}, tmp_path)

        with pytest.raises(AssertionError, match="no value"):
            resolve_megatron_config(_make_args(path))

    def test_an_argument_outside_the_per_policy_whitelist_is_refused(self, tmp_path):
        # [FUNCTION DOC] Overriding the number of rollouts (not in either whitelist) raises an
        # assertion error naming that argument.
        """Rhythm arguments are read from the base command line, so accepting them here would do nothing."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"num_rollout": 3}}]}, tmp_path)

        with pytest.raises(AssertionError, match="num_rollout"):
            resolve_megatron_config(_make_args(path))


class TestModelDefinitionOverrides:
    # [CLASS DOC] Tests for the private override resolver with model-definition arguments, using a
    # small fake parser instead of the real Megatron parser (installed by an autouse fixture).
    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        # [FUNCTION DOC] HELPER: Builds a help-less parser declaring: float learning rate,
        # integer layer count, integer expert count, integer router top-k, a store-true flag for
        # untied embeddings and output weights, a multi-value spec, and an integer rollout count.
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--num-layers", type=int)
        parser.add_argument("--num-experts", type=int)
        parser.add_argument("--moe-router-topk", type=int)
        parser.add_argument("--untie-embeddings-and-output-weights", action="store_true")
        parser.add_argument("--spec", nargs="*")
        parser.add_argument("--num-rollout", type=int)
        return parser

    @pytest.fixture(autouse=True)
    def _parser_of_run(self, monkeypatch):
        # [FUNCTION DOC] HELPER: Autouse fixture replacing the config module's parser getter with
        # the fake parser factory for every test in the class (undone by monkeypatch).
        monkeypatch.setattr(megatron_config_module, "get_megatron_arg_parser", self._parser)

    def test_a_model_definition_argument_this_run_declares_is_accepted_and_typed(self):
        # [FUNCTION DOC] String expert count and router top-k values come back as integers.
        """A policy carries the shape of its own model, of which PER_POLICY_ARGS lists only a part."""
        overrides = _resolve_overrides({"num_experts": "8", "moe_router_topk": "4"}, model_id="a")

        assert overrides == {"num_experts": 8, "moe_router_topk": 4}

    def test_a_model_definition_flag_stays_a_boolean(self):
        # [FUNCTION DOC] A true value for a store-true argument is returned as the boolean true.
        """store_true arguments carry no value on a command line, so the overlay must keep the boolean."""
        overrides = _resolve_overrides({"untie_embeddings_and_output_weights": True}, model_id="a")

        assert overrides == {"untie_embeddings_and_output_weights": True}

    def test_a_model_definition_argument_this_run_does_not_declare_is_refused(self):
        # [FUNCTION DOC] A model-definition name the fake parser does not declare (the KV LoRA
        # rank) raises an assertion error saying the policy may not override it.
        """The set is intersected with the parser, so a name this run cannot type admits nothing."""
        with pytest.raises(AssertionError, match="may not override"):
            _resolve_overrides({"kv_lora_rank": 512}, model_id="a")

    def test_an_argument_in_neither_whitelist_is_refused(self):
        # [FUNCTION DOC] The rollout count is declared by the fake parser but in neither
        # whitelist, so overriding it raises an assertion error naming it.
        """The parser declares --num-rollout, yet the rhythm of a run is read from the base command line."""
        with pytest.raises(AssertionError, match="num_rollout"):
            _resolve_overrides({"num_rollout": 3}, model_id="a")

    def test_a_list_reaches_an_argument_that_takes_several_values(self):
        # [FUNCTION DOC] A two-element list (module path, function name) for the multi-value spec
        # argument is passed through as the same list.
        """--spec names a module and a function, which is how such an argument arrives from yaml."""
        overrides = _resolve_overrides({"spec": ["miles_plugins.models.glm5.glm5", "get_glm5_spec"]}, model_id="a")

        assert overrides == {"spec": ["miles_plugins.models.glm5.glm5", "get_glm5_spec"]}

    def test_a_list_written_where_a_single_value_is_declared_is_refused(self):
        # [FUNCTION DOC] A list for a single-valued integer argument raises an assertion error
        # saying it takes a single value.
        """One integer argument cannot be given two, and the overlay is the only place that can say so."""
        with pytest.raises(AssertionError, match="takes a single value"):
            _resolve_overrides({"num_experts": [8, 4]}, model_id="a")

    def test_a_training_knob_is_still_accepted_beside_a_model_definition_argument(self):
        # [FUNCTION DOC] Learning rate (per-policy whitelist) and layer count (model definition)
        # given as strings are both accepted and typed (float and integer).
        """The model definition arguments are admitted on top of PER_POLICY_ARGS, not instead of it."""
        overrides = _resolve_overrides({"lr": "5e-7", "num_layers": "12"}, model_id="a")

        assert overrides == {"lr": 5e-7, "num_layers": 12}

    def test_the_deprecated_parser_epsilon_destination_is_whitelisted(self) -> None:
        # [FUNCTION DOC] Asserts the model-definition whitelist contains the norm epsilon name
        # that the deprecated parser uses as its destination.
        """The deprecated parser stores its epsilon option under norm_epsilon, which policies may override."""
        assert "norm_epsilon" in MODEL_DEFINITION_ARGS


class TestResolveOverrides:
    # [CLASS DOC] Tests for the private override resolver's short-circuit behavior.
    def test_an_empty_override_map_never_builds_the_parser(self, monkeypatch):
        # [FUNCTION DOC] Replaces the parser getter with one that fails the test when called;
        # resolving an empty override mapping must return an empty mapping without calling it.
        """Building the parser imports and runs megatron's whole argument stack, per trainer that overrides nothing."""
        monkeypatch.setattr(
            megatron_config_module,
            "get_megatron_arg_parser",
            lambda: pytest.fail("the parser was built for a trainer that overrides nothing"),
        )

        assert _resolve_overrides({}, model_id="a") == {}


class TestGetMegatronArgParser:
    # [CLASS DOC] Tests for the function that obtains the real Megatron argument parser.
    def test_a_parser_that_never_reaches_the_provider_fails_loudly(self, monkeypatch):
        # [FUNCTION DOC] Substitutes the module providing the argument parsing function with a
        # stub that returns an empty namespace without invoking the extra-arguments provider.
        # Asserts the getter raises an assertion error saying it returned without calling the
        # provider.
        """Silently answering an empty parser would type every override as a string."""
        monkeypatch.setitem(
            sys.modules,
            "miles.backends.megatron_utils.arguments",
            SimpleNamespace(parse_args=lambda extra_args_provider: Namespace()),
        )

        with pytest.raises(AssertionError, match="returned without calling the extra args provider"):
            get_megatron_arg_parser()


def _make_checkpoint_args(tmp_path, **overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a namespace (shared parser defaults plus test values) for the
    # checkpoint-load resolution tests: core conversion mode, load under tmp save, reference load
    # under tmp ref, HuggingFace checkpoint under tmp hf, no step settings, optimizer and rng
    # loading enabled, not finetune, no start rollout id; keyword overrides win.
    defaults = dict(
        megatron_to_hf_mode="core",
        load=str(tmp_path / "save"),
        ref_load=str(tmp_path / "ref"),
        hf_checkpoint=str(tmp_path / "hf"),
        ref_ckpt_step=None,
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        start_rollout_id=None,
    )
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


def _write_megatron_checkpoint(tmp_path) -> str:
    # [FUNCTION DOC] HELPER: Creates the "save" directory under the temporary path with a
    # latest-checkpointed-iteration tracker file containing 10, and returns the directory path.
    """A directory megatron recognizes: it exists and carries the iteration tracker file."""
    path = tmp_path / "save"
    path.mkdir()
    (path / "latest_checkpointed_iteration.txt").write_text("10")
    return str(path)


class TestResolveArgsCheckpointLoad:
    # [CLASS DOC] Tests for the in-place resolution of load-related arguments into either a fresh
    # start (from reference or HuggingFace weights) or a resume from an existing checkpoint.
    def test_a_fresh_bridge_run_falls_back_to_the_reference_weights(self, tmp_path):
        # [FUNCTION DOC] Bridge mode with no checkpoint at the load path: the load path becomes the
        # reference load path and the start rollout id becomes 0.
        """--load points at a directory the first run has not written yet, which load_checkpoint asserts on."""
        args = _make_checkpoint_args(tmp_path, megatron_to_hf_mode="bridge")

        resolve_args_checkpoint_load(args)

        assert args.load == str(tmp_path / "ref")
        assert args.start_rollout_id == 0

    def test_a_bridge_run_without_reference_weights_falls_back_to_the_hf_checkpoint(self, tmp_path):
        # [FUNCTION DOC] Bridge mode, no checkpoint and no reference load: the load path becomes
        # the HuggingFace checkpoint path.
        """Without --ref-load the HF weights are the only thing the bridge can start from."""
        args = _make_checkpoint_args(tmp_path, megatron_to_hf_mode="bridge", ref_load=None)

        resolve_args_checkpoint_load(args)

        assert args.load == str(tmp_path / "hf")

    def test_a_bridge_run_with_a_real_checkpoint_resumes_from_it(self, tmp_path):
        # [FUNCTION DOC] Bridge mode with a real Megatron checkpoint at the load path: the load
        # path is kept and the start rollout id stays unset (none), which signals a resume.
        """A bridge resume keeps --load, and start_rollout_id staying None is what says 'resume'."""
        load = _write_megatron_checkpoint(tmp_path)
        args = _make_checkpoint_args(tmp_path, megatron_to_hf_mode="bridge", load=load)

        resolve_args_checkpoint_load(args)

        assert args.load == load
        assert args.start_rollout_id is None

    def test_a_fresh_core_run_finetunes_from_the_reference_weights(self, tmp_path):
        # [FUNCTION DOC] Core mode with no checkpoint: optimizer loading, rng loading are turned
        # off and finetune turned on, the load path becomes the reference path, start rollout 0.
        """Loading an optimizer state and rng that were never written aborts the very first step."""
        args = _make_checkpoint_args(tmp_path)

        resolve_args_checkpoint_load(args)

        assert (args.no_load_optim, args.no_load_rng, args.finetune) == (True, True, True)
        assert args.load == str(tmp_path / "ref")
        assert args.start_rollout_id == 0

    def test_a_fresh_core_run_takes_the_step_of_the_reference_checkpoint(self, tmp_path):
        # [FUNCTION DOC] Core mode fresh run with a reference checkpoint step of 7: the checkpoint
        # step argument becomes 7.
        """--ref-ckpt-step names which iteration of the reference weights to read."""
        args = _make_checkpoint_args(tmp_path, ref_ckpt_step=7)

        resolve_args_checkpoint_load(args)

        assert args.ckpt_step == 7

    def test_a_core_run_with_a_real_checkpoint_is_left_untouched(self, tmp_path):
        # [FUNCTION DOC] Core mode with a real checkpoint (and a reference step of 7): load path
        # kept, optimizer/rng loading and finetune unchanged (on, on, off), checkpoint step and
        # start rollout id remain unset.
        """A resume must keep its optimizer state, and start_rollout_id staying None is what says 'resume'."""
        load = _write_megatron_checkpoint(tmp_path)
        args = _make_checkpoint_args(tmp_path, load=load, ref_ckpt_step=7)

        resolve_args_checkpoint_load(args)

        assert args.load == load
        assert (args.no_load_optim, args.no_load_rng, args.finetune) == (False, False, False)
        assert (args.ckpt_step, args.start_rollout_id) == (None, None)


class TestHasMegatronCheckpoint:
    # [CLASS DOC] Tests for the private checkpoint-presence check, which looks for the tracker
    # file in a directory path and tolerates missing or unset paths.
    def test_a_directory_holding_the_tracker_file_is_a_checkpoint(self, tmp_path):
        # [FUNCTION DOC] A directory containing the tracker file: true.
        """This is the one shape both branches treat as a resume."""
        assert _has_megatron_checkpoint(_write_megatron_checkpoint(tmp_path)) is True

    def test_a_directory_without_the_tracker_file_is_not_a_checkpoint(self, tmp_path):
        # [FUNCTION DOC] An existing empty directory: false.
        """A --save directory exists from the moment the run starts, long before it holds a checkpoint."""
        (tmp_path / "save").mkdir()

        assert _has_megatron_checkpoint(str(tmp_path / "save")) is False

    def test_a_missing_directory_is_not_a_checkpoint(self, tmp_path):
        # [FUNCTION DOC] A path that does not exist: false.
        """The first run of a job passes a --load path nothing has created yet."""
        assert _has_megatron_checkpoint(str(tmp_path / "nope")) is False

    def test_no_load_directory_at_all_is_not_a_checkpoint(self):
        # [FUNCTION DOC] An unset (none) path: false, without touching the file system.
        """--load is optional, and None must not reach os.path.exists."""
        assert _has_megatron_checkpoint(None) is False


class TestComputeTrainerArgs:
    # [CLASS DOC] Tests for building a trainer's argument namespace by overlaying its overrides
    # on the base arguments, including the rejections of disallowed overrides.
    def test_each_policy_overlays_its_own_args_on_the_base_arguments(self, tmp_path):
        # [FUNCTION DOC] Policy a overrides learning rate and tensor parallel size, policy b has
        # none (the yaml uses the legacy top-level key). Asserts a sees its overrides, b sees the
        # base values, and the base arguments themselves are unchanged.
        """Per-policy megatron args are the whole point of the flag; the base args stay untouched."""
        path = _write_yaml(
            {
                "megatron": [
                    {"model_id": "a", "overrides": {"lr": 5e-7, "tensor_model_parallel_size": 2}},
                    {"model_id": "b"},
                ]
            },
            tmp_path,
        )
        args = _make_args(path)

        model_a = _model_args(args, model_id="a")
        model_b = _model_args(args, model_id="b")

        assert (model_a.lr, model_a.tensor_model_parallel_size) == (5e-7, 2)
        assert (model_b.lr, model_b.tensor_model_parallel_size) == (1e-6, 1)
        assert (args.lr, args.tensor_model_parallel_size) == (1e-6, 1)

    def test_a_boolean_override_is_kept_as_a_boolean(self, tmp_path):
        # [FUNCTION DOC] A true override of sequence parallel yields exactly the boolean true in
        # the overlaid arguments.
        """store_true arguments are booleans in the overlay, not the strings a command line would carry."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"sequence_parallel": True}}, {"model_id": "b"}]},
            tmp_path,
        )

        assert _model_args(_make_args(path), model_id="a").sequence_parallel is True

    def test_an_unknown_argument_is_refused(self, tmp_path):
        # [FUNCTION DOC] An override of a nonexistent argument raises an assertion error naming
        # it at config resolution time.
        """A per-policy typo would otherwise be dropped and the policy would train with base settings."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"no_such_flag": 3}}]}, tmp_path)

        with pytest.raises(AssertionError, match="no_such_flag"):
            resolve_megatron_config(_make_args(path))

    def test_a_whitelisted_argument_this_run_does_not_declare_is_refused(self, tmp_path):
        # [FUNCTION DOC] A whitelisted name (minimum learning rate) that the run's argument
        # namespace does not carry makes computing the trainer args raise an assertion error
        # saying the run does not know it.
        """A whitelist entry is not a promise that every backend's parser declares it."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"min_lr": 1e-8}}, {"model_id": "b"}]}, tmp_path
        )

        with pytest.raises(AssertionError, match="does not know"):
            _model_args(_make_args(path), model_id="a")

    def test_a_policy_that_overrides_a_shared_rollout_argument_is_refused(self, tmp_path):
        # [FUNCTION DOC] Overriding the advantage estimator (shared with the rollout side) makes
        # computing the trainer args raise an assertion error mentioning rollout arguments.
        """The rollout side converts, trims and splits every policy's data under the run's own arguments."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"advantage_estimator": "ppo"}}, {"model_id": "b"}]}, tmp_path
        )

        with pytest.raises(AssertionError, match="rollout arguments"):
            _model_args(_make_args(path), model_id="a")

    def test_a_policy_that_overrides_the_global_batch_size_is_refused(self, tmp_path):
        # [FUNCTION DOC] Overriding the global batch size raises an assertion error naming it when
        # the trainer args are computed (not at config resolution time).
        """Every policy trains at the run's global batch size, so an override cannot be honored."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"global_batch_size": 128}}]}, tmp_path)

        with pytest.raises(AssertionError, match="global_batch_size"):
            _model_args(_make_args(path), model_id="a")


class TestTrainerCheckpointDirs:
    # [CLASS DOC] Tests for derivation of per-trainer save, load and HuggingFace save directories
    # as a "trainers" subdirectory keyed by trainer id under the base directory.
    def test_a_multi_policy_run_gives_every_trainer_its_own_checkpoint_dir(self, tmp_path):
        # [FUNCTION DOC] Creates old checkpoint trees with trackers for a-actor and b-actor under
        # the old load base; with base save /ckpt/run and that load base, each model's save is
        # the base save plus trainers plus its trainer id, and its load is the matching old tree.
        """A shared --save makes two policies write the same iter_* directory and overwrite each other."""
        old = tmp_path / "old"
        for trainer_id in ("a-actor", "b-actor"):
            trainer_dir = old / "trainers" / trainer_id
            trainer_dir.mkdir(parents=True)
            (trainer_dir / "latest_checkpointed_iteration.txt").write_text("7")
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)
        args = _make_args(path, save="/ckpt/run", load=str(old))

        model_a = _model_args(args, model_id="a")
        model_b = _model_args(args, model_id="b")

        assert (model_a.save, model_a.load) == ("/ckpt/run/trainers/a-actor", str(old / "trainers" / "a-actor"))
        assert (model_b.save, model_b.load) == ("/ckpt/run/trainers/b-actor", str(old / "trainers" / "b-actor"))

    def test_a_checkpoint_dir_is_keyed_by_the_trainer_id_and_not_by_the_model_id(self, tmp_path):
        # [FUNCTION DOC] Model a has an explicit trainer id a-second, model b the derived b-actor.
        # Computing args for every trainer yields saves under trainers/a-second and
        # trainers/b-actor respectively.
        """A trainer id is unique where a model id is not, so keying the directory by the model would collide."""
        path = _write_yaml({"trainers": [{"model_id": "a", "trainer_id": "a-second"}, {"model_id": "b"}]}, tmp_path)
        args = _make_args(path, save="/ckpt/run")

        saves = [compute_trainer_args(args, trainer).save for trainer in resolve_megatron_config(args).trainers]

        assert saves == ["/ckpt/run/trainers/a-second", "/ckpt/run/trainers/b-actor"]

    def test_an_explicitly_named_policy_gets_its_own_checkpoint_dir(self, tmp_path):
        # [FUNCTION DOC] A single-policy yaml with an old checkpoint tree for a-actor. The
        # resolved save, load and HuggingFace save directories all gain the trainers/a-actor
        # suffix (load pointing into the old tree).
        """A split trainer must not share its checkpoint paths with another policy's release."""
        old = tmp_path / "old"
        trainer_dir = old / "trainers" / "a-actor"
        trainer_dir.mkdir(parents=True)
        (trainer_dir / "latest_checkpointed_iteration.txt").write_text("7")
        path = _write_yaml({"trainers": [{"model_id": "a"}]}, tmp_path)
        args = _make_args(path, save="/ckpt/run", load=str(old), save_hf="/ckpt/hf")

        model = _model_args(args, model_id="a")

        assert (model.save, model.load, model.save_hf) == (
            "/ckpt/run/trainers/a-actor",
            str(trainer_dir),
            "/ckpt/hf/trainers/a-actor",
        )

    def test_a_named_critic_derives_its_dir_from_the_critic_checkpoint(self, tmp_path):
        # [FUNCTION DOC] With the critic enabled and distinct actor and critic save/load bases,
        # the synthesized critic trainer (alpha-critic) derives its save from the critic save base
        # and its load from the critic load base (which holds a tracker), each under
        # trainers/alpha-critic, not from the actor bases.
        """A named critic must namespace its own checkpoint base rather than the policy's base."""
        old = tmp_path / "critic-old"
        trainer_dir = old / "trainers" / "alpha-critic"
        trainer_dir.mkdir(parents=True)
        (trainer_dir / "latest_checkpointed_iteration.txt").write_text("7")
        path = _write_yaml({"trainers": [{"model_id": "alpha"}]}, tmp_path)
        args = _make_args(
            path,
            use_critic=True,
            save="/ckpt/actor",
            load="/old/actor",
            critic_save="/ckpt/critic",
            critic_load=str(old),
        )
        [_, critic] = resolve_megatron_config(args).trainers

        model = compute_trainer_args(args, critic)

        assert (model.save, model.load) == ("/ckpt/critic/trainers/alpha-critic", str(trainer_dir))

    def test_an_unset_checkpoint_dir_stays_unset(self, tmp_path):
        # [FUNCTION DOC] With no save directory in the base arguments, the trainer's save is none.
        """A run without --save must not grow a derived path out of None."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)

        assert _model_args(_make_args(path), model_id="a").save is None

    def test_the_derived_dir_is_the_trainer_id_under_a_trainers_directory(self):
        # [FUNCTION DOC] The directory derivation function with base /ckpt/run and trainer id
        # policy-b-actor returns /ckpt/run/trainers/policy-b-actor.
        """The layout is a user visible contract: it is where a resume looks for a trainer's checkpoints."""
        assert (
            compute_trainer_checkpoint_dir(base_dir="/ckpt/run", trainer_id="policy-b-actor")
            == "/ckpt/run/trainers/policy-b-actor"
        )

    def test_a_policy_cannot_name_its_own_checkpoint_directory(self, tmp_path):
        # [FUNCTION DOC] A per-policy override of the load argument raises an assertion error
        # saying the policy sets load, which it may not override.
        """The per trainer directory is derived from the base --load after the overlay, which would drop an override."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"load": "/ckpt/a"}}, {"model_id": "b"}]}, tmp_path
        )

        with pytest.raises(AssertionError, match="sets 'load', which it may not override"):
            resolve_megatron_config(_make_args(path))


class TestPerPolicyCheckpointResolution:
    # [CLASS DOC] Tests for combining per-trainer directory derivation with fresh-versus-resume
    # load resolution.
    def test_a_fresh_policy_falls_back_to_the_reference_weights_without_the_policy_subdirectory(self, tmp_path):
        # [FUNCTION DOC] No tracker exists anywhere: the policy's save is the derived trainer
        # directory, but its load is the shared reference path unchanged (no subdirectory),
        # finetune is on and the start rollout id is 0.
        """The --ref-load fallback holds shared reference weights, not a per policy checkpoint tree."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)
        args = _make_args(path, save="/ckpt/run", load="/ckpt/run", ref_load="/models/ref")

        model_args = _model_args(args, model_id="a")

        assert model_args.save == "/ckpt/run/trainers/a-actor"
        assert model_args.load == "/models/ref"
        assert (model_args.finetune, model_args.start_rollout_id) == (True, 0)

    def test_a_policy_with_its_own_tracker_resumes_from_its_own_directory(self, tmp_path):
        # [FUNCTION DOC] A tracker exists under the base's trainers/a-actor. The policy's load is
        # that directory, finetune is off and the start rollout id is unset (a resume).
        """The tracker of a policy lives under its own subdirectory, so the root never looks resumable."""
        root = tmp_path / "run"
        trainer_dir = root / "trainers" / "a-actor"
        trainer_dir.mkdir(parents=True)
        (trainer_dir / "latest_checkpointed_iteration.txt").write_text("7")
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)
        args = _make_args(path, save=str(root), load=str(root), ref_load="/models/ref")

        model_args = _model_args(args, model_id="a")

        assert model_args.load == str(trainer_dir)
        assert (model_args.finetune, model_args.start_rollout_id) == (False, None)

    def test_a_fresh_bridge_policy_falls_back_to_its_own_hf_checkpoint(self, tmp_path):
        # [FUNCTION DOC] Bridge mode, no tracker, no reference load, and each policy overriding
        # its own HuggingFace checkpoint (/models/a, /models/b): each policy's load is its own
        # checkpoint path.
        """In bridge mode a policy starts from its own hugging face checkpoint, not from another policy's."""
        path = _write_yaml(
            {
                "trainers": [
                    {"model_id": "a", "overrides": {"hf_checkpoint": "/models/a"}},
                    {"model_id": "b", "overrides": {"hf_checkpoint": "/models/b"}},
                ]
            },
            tmp_path,
        )
        args = _make_args(path, megatron_to_hf_mode="bridge", save="/ckpt/run", load="/ckpt/run")

        assert _model_args(args, model_id="a").load == "/models/a"
        assert _model_args(args, model_id="b").load == "/models/b"


class TestPerPolicyDerivedDefaults:
    # [CLASS DOC] Tests for defaults derived from overrides, currently the tokenizer model path.
    def test_a_policy_checkpoint_override_repoints_the_tokenizer(self, tmp_path):
        # [FUNCTION DOC] Policy a overrides its HuggingFace checkpoint; its tokenizer model
        # becomes that path, while policy b keeps the base tokenizer path.
        """The tokenizer latched onto the base checkpoint at parse time, so a policy of its own needs its own."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"hf_checkpoint": "/models/a"}}, {"model_id": "b"}]},
            tmp_path,
        )
        args = _make_args(path)

        assert _model_args(args, model_id="a").tokenizer_model == "/models/a"
        assert _model_args(args, model_id="b").tokenizer_model == "/models/base"

    def test_a_tokenizer_named_on_the_command_line_is_left_alone(self, tmp_path):
        # [FUNCTION DOC] When the base tokenizer path differs from the base checkpoint (it was
        # chosen explicitly), a policy overriding its checkpoint keeps that explicit tokenizer.
        """That tokenizer was chosen rather than derived, so no policy may re-point it at its own checkpoint."""
        path = _write_yaml(
            {"trainers": [{"model_id": "a", "overrides": {"hf_checkpoint": "/models/a"}}, {"model_id": "b"}]},
            tmp_path,
        )
        args = _make_args(path, tokenizer_model="/models/shared")

        assert _model_args(args, model_id="a").tokenizer_model == "/models/shared"


class TestMultiPolicyIds:
    # [CLASS DOC] Tests for the model id and trainer id attributes placed on trainer arguments.
    def test_an_explicitly_named_policy_keeps_its_trainer_model_id(self, tmp_path):
        # [FUNCTION DOC] A single named policy a: its trainer arguments carry trainer model id a.
        """A split trainer keeps its policy identity even when its release carries no neighbouring policy."""
        path = _write_yaml({"trainers": [{"model_id": "a"}]}, tmp_path)

        assert _model_args(_make_args(path), model_id="a").trainer_model_id == "a"

    def test_a_run_without_a_megatron_config_carries_no_trainer_model_id(self):
        # [FUNCTION DOC] The synthesized legacy actor's trainer arguments have trainer model id
        # none.
        """The unnamed legacy actor keeps using the unnamespaced single-policy key."""
        args = _make_args()
        [trainer] = resolve_megatron_config(args).trainers

        assert compute_trainer_args(args, trainer).trainer_model_id is None

    def test_each_policy_of_a_multi_policy_run_carries_its_own_id(self, tmp_path):
        # [FUNCTION DOC] In a two-policy run, policy b's trainer arguments carry trainer model id b.
        """Metrics, routers and checkpoints are all namespaced by this value."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)

        assert _model_args(_make_args(path), model_id="b").trainer_model_id == "b"

    def test_each_trainer_carries_its_trainer_id_into_the_worker(self, tmp_path):
        # [FUNCTION DOC] A policy with explicit trainer id alpha-actor: its trainer arguments
        # carry that trainer id.
        """The worker must derive its cell id from the trainer-specific pool that launched it."""
        path = _write_yaml({"trainers": [{"model_id": "a", "trainer_id": "alpha-actor"}]}, tmp_path)

        assert _model_args(_make_args(path), model_id="a").trainer_id == "alpha-actor"


class TestSynthesizedCriticTrainer:
    # [CLASS DOC] Tests for the critic trainer that resolution synthesizes beside the actor when
    # the use-critic flag is set, including its overlay of critic-specific arguments.
    def test_arguments_that_do_not_carry_use_critic_yet_still_resolve(self):
        # [FUNCTION DOC] Deletes the use-critic attribute from the arguments (it is derived later
        # in validation); resolution must still succeed with a single actor trainer.
        """use_critic is derived while the arguments are validated, and the config is resolved before that."""
        args = _make_args()
        del args.use_critic

        assert [trainer.role for trainer in resolve_megatron_config(args).trainers] == ["actor"]

    def test_a_run_without_the_flag_synthesizes_the_critic_beside_the_actor(self):
        # [FUNCTION DOC] No config path but use-critic on: two trainers, actor then critic, with
        # ids actor and critic and no model id.
        """The critic used to be assembled in specs and in the worker; the config is now the only source."""
        config = resolve_megatron_config(_make_args(use_critic=True))

        assert [(t.trainer_id, t.role, t.model_id) for t in config.trainers] == [
            ("actor", "actor", None),
            ("critic", "critic", None),
        ]

    def test_training_several_policies_with_a_critic_is_refused(self, tmp_path):
        # [FUNCTION DOC] Two policies plus use-critic raises an assertion error saying the config
        # does not support the use-critic option.
        """A critic is synthesized beside one policy, so several policies plus a critic has no defined meaning."""
        path = _write_yaml({"trainers": [{"model_id": "a"}, {"model_id": "b"}]}, tmp_path)

        with pytest.raises(AssertionError, match="does not support --use-critic"):
            resolve_megatron_config(_make_args(path, use_critic=True))

    def test_a_config_that_declares_its_own_critic_is_refused(self, tmp_path):
        # [FUNCTION DOC] A yaml declaring a critic-role trainer for model a while use-critic is on
        # raises an assertion error saying the config declares a critic for it.
        """The critic checkpoint, learning rate and neutralized knobs only reach the synthesized critic, so a
        declared one would run as a plain trainer under a critic's name."""
        path = _write_yaml(
            {
                "trainers": [
                    {"model_id": "a"},
                    {"model_id": "a", "role": "critic", "overrides": {"lr": 5e-7}},
                ]
            },
            tmp_path,
        )

        with pytest.raises(AssertionError, match="declares a critic for"):
            resolve_megatron_config(_make_args(path, use_critic=True))

    def test_the_critic_of_a_named_policy_inherits_its_id_and_its_overlay(self, tmp_path):
        # [FUNCTION DOC] Policy alpha with an eps_clip override and use-critic on: the second
        # trainer has id alpha-critic, model id alpha, role critic, and carries the policy's
        # eps_clip override of 0.3.
        """The critic trains the same policy, so it must be addressed by that policy and see its settings."""
        path = _write_yaml({"trainers": [{"model_id": "alpha", "overrides": {"eps_clip": 0.3}}]}, tmp_path)

        [_, critic] = resolve_megatron_config(_make_args(path, use_critic=True)).trainers

        assert (critic.trainer_id, critic.model_id, critic.role) == ("alpha-critic", "alpha", "critic")
        assert critic.overrides["eps_clip"] == 0.3

    def test_the_synthesized_critic_reproduces_the_legacy_worker_swap(self):
        # [FUNCTION DOC] With critic load/save/learning-rate/warmup values set, the critic's
        # trainer arguments have KL coefficient 0, on-policy-distillation off, parameter-buffer
        # CPU backup disabling off, load and save taken from the critic ones, learning rate 2e-6
        # and warmup iterations 3.
        """The worker no longer remaps critic_* onto the standard fields, so the overlay must do it."""
        args = _make_args(
            use_critic=True,
            save="/ckpt/run",
            load="/ckpt/run",
            critic_load="/ckpt/critic",
            critic_save="/ckpt/run_critic",
            critic_lr=2e-6,
            critic_lr_warmup_iters=3,
        )

        critic_args = compute_trainer_args(args, resolve_megatron_config(args).trainers[1])

        assert (critic_args.kl_coef, critic_args.use_opd, critic_args.disable_param_buffers_cpu_backup) == (
            0,
            False,
            False,
        )
        assert (critic_args.load, critic_args.save, critic_args.lr, critic_args.lr_warmup_iters) == (
            "/ckpt/critic",
            "/ckpt/run_critic",
            2e-6,
            3,
        )

    def test_the_actor_of_a_critic_run_keeps_its_own_checkpoint_and_schedule(self):
        # [FUNCTION DOC] In a critic run, the actor trainer's arguments keep the base load, save,
        # learning rate (1e-6) and KL coefficient (0.1); no critic value leaks into them.
        """The two trainers share one command line, so a leaked critic override would retrain the actor."""
        args = _make_args(use_critic=True, save="/ckpt/run", load="/ckpt/run", critic_load="/ckpt/critic")

        actor_args = compute_trainer_args(args, resolve_megatron_config(args).trainers[0])

        assert (actor_args.load, actor_args.save, actor_args.lr, actor_args.kl_coef) == (
            "/ckpt/run",
            "/ckpt/run",
            1e-6,
            0.1,
        )

    def test_an_internally_synthesized_override_bypasses_the_user_whitelist(self):
        # [FUNCTION DOC] The synthesized critic's overrides include the KL coefficient and load
        # names, although neither is in the per-policy whitelist.
        """kl_coef and load are not per-policy yaml arguments, yet the critic must still be able to set them."""
        args = _make_args(use_critic=True, critic_load="/ckpt/critic")

        overrides = set(resolve_megatron_config(args).trainers[1].overrides)

        assert {"kl_coef", "load"} <= overrides
        assert not {"kl_coef", "load"} & set(PER_POLICY_ARGS)

    def test_a_critic_without_its_own_schedule_inherits_the_unset_values(self, tmp_path):
        # [FUNCTION DOC] Policy alpha overrides its learning rate but no critic learning rate or
        # warmup is given: the critic's learning rate and warmup iterations are both none (the
        # policy's own learning rate does not apply to the critic).
        """--critic-lr is unset by default, and the critic takes it as it is: the policy's own lr does not apply."""
        path = _write_yaml({"trainers": [{"model_id": "alpha", "overrides": {"lr": 5e-7}}]}, tmp_path)
        args = _make_args(path, use_critic=True)

        critic_args = compute_trainer_args(args, resolve_megatron_config(args).trainers[1])

        assert (critic_args.lr, critic_args.lr_warmup_iters) == (None, None)

    def test_the_critic_overlay_names_exactly_the_fields_the_worker_used_to_swap(self):
        # [FUNCTION DOC] The set of the synthesized critic's override names equals exactly: KL
        # coefficient, on-policy distillation flag, parameter-buffer CPU-backup flag, load, save,
        # learning rate and warmup iterations.
        """A new critic_* argument that nobody wires in here would be read from the command line and ignored."""
        overrides = resolve_megatron_config(_make_args(use_critic=True)).trainers[1].overrides

        assert set(overrides) == {
            "kl_coef",
            "use_opd",
            "disable_param_buffers_cpu_backup",
            "load",
            "save",
            "lr",
            "lr_warmup_iters",
        }

    def test_a_policy_override_of_a_critic_neutralized_field_loses_to_the_neutralization(self, tmp_path):
        # [FUNCTION DOC] Policy overrides learning rate (5e-7) and eps_clip (0.3), with critic
        # learning rate 2e-6. In the critic's overrides the learning rate is the critic's 2e-6
        # (critic-specific values win) while eps_clip stays 0.3.
        """The overlay order is what neutralizes the critic, so a policy override of the same field must not win."""
        path = _write_yaml({"trainers": [{"model_id": "alpha", "overrides": {"lr": 5e-7, "eps_clip": 0.3}}]}, tmp_path)

        overrides = resolve_megatron_config(_make_args(path, use_critic=True, critic_lr=2e-6)).trainers[1].overrides

        assert (overrides["lr"], overrides["eps_clip"]) == (2e-6, 0.3)


class TestBaseArgumentsAreNotMutated:
    # [CLASS DOC] Tests that overlaying a trainer does not share mutable state with the base.
    def test_the_base_arguments_are_never_mutated_by_an_overlay(self, tmp_path):
        # [FUNCTION DOC] Base arguments carry a training environment-variable mapping. After
        # computing a trainer's arguments and mutating that mapping on the result, the base
        # mapping must still hold its original value (the overlay deep-copies).
        """A shallow copy would let one trainer's overlay reach every other trainer through a shared dict."""
        path = _write_yaml({"trainers": [{"model_id": "a", "overrides": {"lr": 5e-7}}, {"model_id": "b"}]}, tmp_path)
        args = _make_args(path, train_env_vars={"NCCL_DEBUG": "WARN"})

        model = _model_args(args, model_id="a")
        model.train_env_vars["NCCL_DEBUG"] = "INFO"

        assert args.train_env_vars == {"NCCL_DEBUG": "WARN"}


class TestPerPolicyArgsCoverage:
    # [CLASS DOC] Tests tying the two whitelists to the real Megatron argument parser and to the
    # model-script argument strings shipped with the repository.
    @pytest.mark.parametrize("model_type", ["qwen2.5-0.5B", "qwen3-0.6B"])
    def test_the_whitelist_admits_every_argument_of_a_model_script(self, model_type):
        # [FUNCTION DOC] For each model type, loads its model-script argument string, takes the
        # dashed flags, asserts each is declared by the real parser, then asserts the parser
        # destinations of all those flags are within the union of both whitelists.
        """A policy that cannot override one of its own architecture arguments would train another model's shape."""
        options = get_megatron_arg_parser()._option_string_actions
        flags = [token for token in load_model_args(model_type).split() if token.startswith("--")]
        unknown = [flag for flag in flags if flag not in options]
        assert not unknown, f"{model_type} passes {unknown}, which the megatron parser does not declare"

        assert {options[flag].dest for flag in flags} <= PER_POLICY_ARGS | MODEL_DEFINITION_ARGS

    @pytest.mark.parametrize("model_type", ["qwen2.5-0.5B", "qwen3-0.6B"])
    def test_a_model_script_names_only_model_definition_arguments(self, model_type):
        # [FUNCTION DOC] For each model type, asserts the parser destinations of every flag in
        # its model script are within the model-definition whitelist alone.
        """PER_POLICY_ARGS holds the training knobs, so nothing describing a model shape may hide in it."""
        options = get_megatron_arg_parser()._option_string_actions
        flags = [token for token in load_model_args(model_type).split() if token.startswith("--")]

        assert {options[flag].dest for flag in flags} <= MODEL_DEFINITION_ARGS

    def test_the_whitelist_names_arguments_the_parser_actually_produces(self):
        # [FUNCTION DOC] Collects all destination names of the real parser's actions and asserts
        # the per-policy whitelist is a subset of them.
        """The override keys are compared against this set and then looked up by the same name in the
        parser, so a flag spelled the way it appears on a command line admits nothing at all: a config
        carrying the argument's real name is refused, and the name in the set can never be reached."""
        parsed = {action.dest for action in get_megatron_arg_parser()._actions}

        assert PER_POLICY_ARGS <= parsed


class TestOverrideWhitelistShape:
    # [CLASS DOC] Structural checks on the two whitelist sets, independent of any parser.
    def test_the_two_whitelists_are_disjoint(self):
        # [FUNCTION DOC] Asserts the per-policy and model-definition whitelists share no names.
        """A name in both no longer says whether it is a training knob or part of a model definition."""
        assert not PER_POLICY_ARGS & MODEL_DEFINITION_ARGS

    def test_every_whitelisted_name_is_spelled_the_way_a_destination_is(self):
        # [FUNCTION DOC] Asserts every name in the union of both whitelists is lowercase letters,
        # digits and underscores starting with a letter (an argparse destination spelling, not a
        # dashed command-line flag). The list of offenders must be empty.
        """Both sets are matched against argparse destinations, so a command line spelling admits nothing."""
        misspelled = [
            name
            for name in sorted(PER_POLICY_ARGS | MODEL_DEFINITION_ARGS)
            if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
        ]

        assert misspelled == []
