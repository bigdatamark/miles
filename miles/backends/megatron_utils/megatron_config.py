# [MODULE DOC]
# Parsing and resolution of the --megatron-config file, which declares one or more Megatron
# trainers (policy models and their roles) with per-trainer argument overrides. Flow: the raw file
# (yaml, via the file-argument resolver; top-level key "trainers" or alias "megatron") is validated
# into frozen raw records; each is resolved into a trainer config with a trainer id (default
# model id plus dash plus role) and overrides type-coerced through the real Megatron/Miles argument
# parser, restricted to an allow-list (per-policy args, plus model-definition args that the parser
# declares); a critic trainer is synthesized when use-critic is on; validators enforce id syntax,
# uniqueness, length, reserved names and one actor per model. Per-trainer args are derived by
# deep-copying the run args, applying overrides, namespacing checkpoint directories per trainer,
# and resolving the checkpoint-load fallback chain. Also holds the constants: role names, the
# trainer checkpoint subdirectory name, the model id pattern, the reserved id for shared eval
# rollouts, and the two allow-lists.
import argparse
import copy
import logging
import os
import re
from argparse import Namespace
from pathlib import Path
from typing import Any, Literal

import pydantic
import yaml

from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.argv_utils import coerce_dict_to_args, declared_arg_dests
from miles.utils.workers.naming import DNS_LABEL_PATTERN, TRAINER_ID_MAX_LENGTH

logger = logging.getLogger(__name__)


# ---------------------------- constants -----------------------------

ACTOR_ROLE = "actor"
CRITIC_ROLE = "critic"
DEFAULT_MODEL_ROLE = ACTOR_ROLE
TrainerRole = Literal["actor", "critic"]
TRAINER_CHECKPOINT_DIRNAME = "trainers"
MODEL_ID_PATTERN = re.compile(rf"\A{DNS_LABEL_PATTERN}\Z")
RESERVED_MODEL_ID = "eval"

PER_POLICY_ARGS: frozenset[str] = frozenset(
    {
        "hf_checkpoint",
        "ref_load",
        "megatron_to_hf_mode",
        "optimizer",
        "lr",
        "min_lr",
        "lr_decay_style",
        "lr_warmup_iters",
        "lr_warmup_fraction",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "clip_grad",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "sequence_parallel",
        "global_batch_size",
        "micro_batch_size",
        "max_tokens_per_gpu",
        "use_dynamic_batch_size",
        "advantage_estimator",
        "use_kl_loss",
        "kl_loss_coef",
        "kl_loss_type",
        "entropy_coef",
        "eps_clip",
        "eps_clip_high",
    }
)

MODEL_DEFINITION_ARGS: frozenset[str] = frozenset(
    {
        "activation_func_clamp_value",
        "add_bias_linear",
        "add_qkv_bias",
        "apply_layernorm_1p",
        "attention_dropout",
        "attention_output_gate",
        "attention_softmax_in_fp32",
        "beta_fast",
        "beta_slow",
        "custom_model_provider_path",
        "dsa_indexer_head_dim",
        "dsa_indexer_n_heads",
        "dsa_indexer_topk",
        "dsv4_compress_ratios",
        "dsv4_compress_rope_theta",
        "dsv4_hc_mult",
        "dsv4_hc_sinkhorn_iters",
        "dsv4_n_hash_layers",
        "dsv4_o_groups",
        "dsv4_o_lora_rank",
        "dsv4_window_size",
        "enable_experimental",
        "experimental_attention_variant",
        "ffn_hidden_size",
        "group_query_attention",
        "hidden_dropout",
        "hidden_size",
        "kv_channels",
        "kv_lora_rank",
        "layernorm_epsilon",
        "make_vocab_size_divisible_by",
        "max_position_embeddings",
        "moe_aux_loss_coeff",
        "moe_ffn_hidden_size",
        "moe_grouped_gemm",
        "moe_latent_size",
        "moe_layer_freq",
        "moe_permute_fusion",
        "moe_router_bias_update_rate",
        "moe_router_dtype",
        "moe_router_enable_expert_bias",
        "moe_router_group_topk",
        "moe_router_load_balancing_type",
        "moe_router_num_groups",
        "moe_router_pre_softmax",
        "moe_router_score_function",
        "moe_router_topk",
        "moe_router_topk_scaling_factor",
        "moe_shared_expert_gate",
        "moe_shared_expert_intermediate_size",
        "moe_token_dispatcher_type",
        "moe_token_drop_policy",
        "mscale",
        "mscale_all_dim",
        "mtp_num_layers",
        "multi_latent_attention",
        "norm_epsilon",
        "normalization",
        "num_attention_heads",
        "num_experts",
        "num_layers",
        "num_query_groups",
        "original_max_position_embeddings",
        "position_embedding_type",
        "post_mlp_layernorm",
        "post_self_attn_layernorm",
        "q_lora_rank",
        "qk_head_dim",
        "qk_layernorm",
        "qk_pos_emb_head_dim",
        "rope_type",
        "rotary_base",
        "rotary_interleaved",
        "rotary_percent",
        "rotary_scaling_factor",
        "seq_length",
        "softmax_type",
        "spec",
        "swiglu",
        "untie_embeddings_and_output_weights",
        "use_rope_scaling",
        "use_rotary_position_embeddings",
        "v_head_dim",
        "vocab_size",
        "window_attn_skip_freq",
        "window_size",
    }
)


# ---------------------------- raw config -----------------------------


class _RawMegatronTrainerConfig(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Immutable, extra-fields-forbidden record of one trainer entry as written in
    # the file: model id (required), role (actor or critic; default actor), optional explicit
    # trainer id, and a mapping of argument overrides (default empty, values untyped).
    model_id: str
    role: TrainerRole = DEFAULT_MODEL_ROLE
    trainer_id: str | None = None
    overrides: dict[str, Any] = {}


class _RawMegatronConfig(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Immutable record of the whole file: a list of raw trainer entries, read
    # from the key "trainers" or its alias "megatron".
    trainers: list[_RawMegatronTrainerConfig] = pydantic.Field(
        validation_alias=pydantic.AliasChoices("trainers", "megatron")
    )

    @classmethod
    def from_file_arg(cls, value: str) -> "_RawMegatronConfig":
        # [FUNCTION DOC] HELPER: Resolves the command-line value to text (a path or inline content,
        # per the file-argument resolver), parses it as yaml with the safe loader and validates it
        # into the record. Raises validation errors on a bad shape.
        return cls.model_validate(yaml.safe_load(resolve_file_arg(value)))


# ---------------------------- resolved config -----------------------------


class MegatronTrainerConfig(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Immutable resolved trainer: trainer id (identifies the trainer controller and its engine
    # pool), model id (none for the implicit single default trainer), role (actor or critic) and the
    # type-coerced overrides mapping.
    trainer_id: str
    model_id: str | None
    role: TrainerRole
    overrides: dict[str, Any]

    @classmethod
    def resolve(cls, raw: _RawMegatronTrainerConfig) -> "MegatronTrainerConfig":
        # [FUNCTION DOC] Builds the resolved trainer from a raw entry: the trainer id is the raw one
        # if given, else the model id, a dash and the role; overrides are coerced and validated
        # through the override resolver.
        return cls(
            trainer_id=raw.trainer_id if raw.trainer_id is not None else f"{raw.model_id}-{raw.role}",
            model_id=raw.model_id,
            role=raw.role,
            overrides=_resolve_overrides(raw.overrides, model_id=raw.model_id),
        )


class MegatronConfig(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Immutable resolved configuration: the ordered list of trainers (the first is the leader).
    # Validated at construction: model ids and trainer ids syntactically valid, trainer ids unique
    # and short enough, no model uses the reserved eval id, and at most one actor per model id.
    trainers: list[MegatronTrainerConfig]

    @pydantic.model_validator(mode="after")
    def _validate_ids(self) -> "MegatronConfig":
        # [FUNCTION DOC] HELPER: Post-construction validator: checks model id syntax, that none is
        # the reserved id, and trainer id syntax/uniqueness/length. Raises assertion errors. Returns
        # the config itself.
        _assert_valid_ids(self.model_ids, kind="model")
        _assert_no_reserved_model_id(self.model_ids)
        _assert_valid_trainer_ids([t.trainer_id for t in self.trainers])
        return self

    @pydantic.model_validator(mode="after")
    def _validate_one_actor_per_model(self) -> "MegatronConfig":
        # [FUNCTION DOC] HELPER: Post-construction validator: asserts no two actor-role trainers
        # share a model id (the run keys trainers by model id, so duplicates would be dropped
        # silently). Returns the config itself.
        actor_model_ids = [t.model_id for t in self.trainers if t.role == ACTOR_ROLE]
        assert len(set(actor_model_ids)) == len(actor_model_ids), (
            f"--megatron-config declares several actors for the same model id ({actor_model_ids}); the run "
            f"keys its trainers by model id, so all but the last one would be launched and then ignored"
        )
        return self

    @property
    def model_ids(self) -> list[str]:
        # [FUNCTION DOC] Property: the distinct non-none model ids of the trainers, in first-seen
        # order.
        return list(dict.fromkeys(t.model_id for t in self.trainers if t.model_id is not None))

    @property
    def leader_model_id(self) -> str | None:
        # [FUNCTION DOC] Property: the model id of the first trainer (may be none). Index error if
        # there are no trainers.
        return self.trainers[0].model_id

    @property
    def is_multi_policy(self) -> bool:
        # [FUNCTION DOC] Property: true when more than one distinct model id is present.
        return len(set(self.model_ids)) > 1

    def get(self, model_id: str) -> MegatronTrainerConfig:
        # [FUNCTION DOC] Returns the first trainer whose model id equals the argument (so for a model
        # with an actor and a critic, the earlier-listed one). Raises a key error listing known ids
        # when none matches.
        for trainer in self.trainers:
            if trainer.model_id == model_id:
                return trainer
        raise KeyError(f"Unknown trainer model id {model_id!r}, known ids: {self.model_ids}")


def resolve_megatron_config(args) -> MegatronConfig:
    # [FUNCTION DOC] Public entry: builds the validated configuration from the parsed run args
    # (using the megatron_config, use_critic and critic_* attributes).
    return MegatronConfig(trainers=_compute_trainers(args))


def _compute_trainers(args) -> list[MegatronTrainerConfig]:
    # [FUNCTION DOC] HELPER: Without a config file, yields one default actor trainer (id "actor", no
    # model id, no overrides). With one: asserts it declares no critic, resolves each entry, and
    # asserts at least one trainer. If use-critic is set: asserts all trainers share one model id
    # (multi-policy with critic unsupported) and appends a synthesized critic derived from the first
    # trainer. Returns the list.
    if (raw := _resolve_raw_megatron_config(args.megatron_config)) is None:
        trainers = [MegatronTrainerConfig(trainer_id=ACTOR_ROLE, model_id=None, role=ACTOR_ROLE, overrides={})]
    else:
        _assert_no_declared_critic(raw)
        trainers = [MegatronTrainerConfig.resolve(raw=t) for t in raw.trainers]
        assert trainers, "--megatron-config must declare at least one trainer"

    if getattr(args, "use_critic", False):
        assert (
            len({trainer.model_id for trainer in trainers}) == 1
        ), "training several policy models does not support --use-critic"
        trainers = [*trainers, _compute_critic_trainer(args, policy=trainers[0])]

    return trainers


def _compute_critic_trainer(args, *, policy: MegatronTrainerConfig) -> MegatronTrainerConfig:
    # [FUNCTION DOC] HELPER: Builds the critic trainer for a policy: id is "critic" when the policy
    # has no model id, else the model id plus "-critic"; same model id; critic role; overrides are
    # the policy's overrides updated with (and overridden by) the critic-specific ones.
    model_id = policy.model_id
    return MegatronTrainerConfig(
        trainer_id=CRITIC_ROLE if model_id is None else f"{model_id}-{CRITIC_ROLE}",
        model_id=model_id,
        role=CRITIC_ROLE,
        overrides={**policy.overrides, **_compute_critic_overrides(args)},
    )


def _compute_critic_overrides(args) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: The fixed overrides applied to a synthesized critic: KL coefficient 0,
    # on-policy distillation off, parameter-buffer CPU backup not disabled, and load path, save path,
    # learning rate and warmup iterations taken from the critic_* attributes of args.
    return {
        "kl_coef": 0,
        "use_opd": False,
        "disable_param_buffers_cpu_backup": False,
        "load": args.critic_load,
        "save": args.critic_save,
        "lr": args.critic_lr,
        "lr_warmup_iters": args.critic_lr_warmup_iters,
    }


def _resolve_raw_megatron_config(value: str | None) -> "_RawMegatronConfig | None":
    # [FUNCTION DOC] HELPER: None if the argument is none, else the parsed raw configuration.
    if value is None:
        return None
    return _RawMegatronConfig.from_file_arg(value)


def _assert_no_declared_critic(raw: "_RawMegatronConfig") -> None:
    # [FUNCTION DOC] HELPER: Asserts no entry in the file has the critic role (assertion error
    # listing their model ids), since critic-specific overrides only apply to a synthesized critic.
    # TODO: accept a declared critic once the critic overrides are applied to it too, not only to a synthesized one
    declared = [t.model_id for t in raw.trainers if t.role == CRITIC_ROLE]
    assert not declared, (
        f"--megatron-config declares a critic for {declared}, which is not supported yet: the critic "
        f"checkpoint, learning rate and neutralized knobs are only applied to the critic the run "
        f"synthesizes itself from --use-critic"
    )


# ---------------------------- per policy args -----------------------------


_ROLLOUT_SHARED_ARGS: frozenset[str] = frozenset(
    {"advantage_estimator", "max_tokens_per_gpu", "micro_batch_size", "use_dynamic_batch_size"}
)


def compute_trainer_args(args: Namespace, trainer: MegatronTrainerConfig) -> Namespace:
    # [FUNCTION DOC]
    # Returns a deep copy of the run args specialized for one trainer; the input is not mutated.
    # Assertions: the trainer does not override global_batch_size, nor any rollout-shared argument
    # (advantage estimator, max tokens per GPU, micro batch size, dynamic batch size flag), nor any
    # name the copied args lack. Steps: set trainer id and trainer model id; apply each override;
    # apply derived overrides (the tokenizer model follows an overridden HF checkpoint if it
    # equaled the base HF checkpoint); for trainers with a model id, nest save, load and HF-save
    # directories under a per-trainer subdirectory (none stays none); when a megatron config was
    # given, resolve the checkpoint-load fallbacks.
    # TODO: support policies with different global batch sizes.
    assert "global_batch_size" not in trainer.overrides, (
        f"--megatron-config trainer {trainer.trainer_id!r} overrides global_batch_size; every policy has to "
        f"share the run's"
    )

    # TODO: let policies differ in these once the rollout side reads per-trainer arguments.
    forbidden = _ROLLOUT_SHARED_ARGS & set(trainer.overrides)
    assert not forbidden, (
        f"--megatron-config trainer {trainer.trainer_id!r} overrides {sorted(forbidden)}; every policy has to "
        f"share the run's rollout arguments"
    )

    ans = copy.deepcopy(args)
    ans.trainer_id = trainer.trainer_id
    ans.trainer_model_id = trainer.model_id

    for key, value in trainer.overrides.items():
        assert hasattr(ans, key), (
            f"--megatron-config trainer {trainer.trainer_id!r} overrides {key!r}, which this run's argument "
            f"parser does not know"
        )
        setattr(ans, key, value)

    _apply_critical_derived_overrides(ans, base=args, trainer=trainer)

    if trainer.model_id is not None:
        ans.save = compute_trainer_checkpoint_dir(base_dir=ans.save, trainer_id=trainer.trainer_id)
        ans.load = compute_trainer_checkpoint_dir(base_dir=ans.load, trainer_id=trainer.trainer_id)
        ans.save_hf = compute_trainer_checkpoint_dir(base_dir=ans.save_hf, trainer_id=trainer.trainer_id)

    # TODO: a --use-critic critic keeps the actor's requested_load, so a hot restart reads the actor's checkpoint.
    if args.megatron_config is not None:
        resolve_args_checkpoint_load(ans)

    return ans


def _apply_critical_derived_overrides(ans: Namespace, *, base: Namespace, trainer: MegatronTrainerConfig) -> None:
    # [FUNCTION DOC] HELPER: Mutates the trainer args: if the trainer overrides the HF checkpoint
    # and the base args' tokenizer model was the base HF checkpoint, points the tokenizer model to
    # the new HF checkpoint.
    # TODO: most derived defaults are still computed from the base args; revisit after the arguments refactor
    if "hf_checkpoint" in trainer.overrides and base.tokenizer_model == base.hf_checkpoint:
        ans.tokenizer_model = ans.hf_checkpoint


# ---------------------------- checkpoint dirs -----------------------------


def compute_trainer_checkpoint_dir(*, base_dir: str | None, trainer_id: str) -> str | None:
    # [FUNCTION DOC] Returns none for no base directory, else the text of base directory joined with
    # the constant subdirectory "trainers" and the trainer id.
    if base_dir is None:
        return None
    return str(Path(base_dir) / TRAINER_CHECKPOINT_DIRNAME / trainer_id)


def resolve_args_checkpoint_load(args: Namespace) -> None:
    # [FUNCTION DOC]
    # Mutates args. Records the originally requested load path as requested_load. If the load
    # directory holds no Megatron checkpoint (exists with the tracker file): in bridge mode sets the
    # load path to the reference load path, else the HF checkpoint, and start rollout id 0; in other
    # modes sets no-load-optim, no-load-rng and finetune, sets load to the reference load path
    # (possibly none), uses the reference checkpoint step as the checkpoint step if given, and
    # start rollout id 0. If a checkpoint exists nothing else changes.
    # TODO: refactor
    args.requested_load = args.load

    # TODO: During loading, we need to set the start_rollout_id here.
    if args.megatron_to_hf_mode == "bridge":
        # Fresh runs pass a not-yet-created `--load` dir; fall back to the reference
        # weights (loaded via the HF bridge) instead of asserting in load_checkpoint.
        # Mirrors the non-bridge branch below.
        if not _has_megatron_checkpoint(args.load):
            args.load = args.ref_load or args.hf_checkpoint
            args.start_rollout_id = 0
    else:
        if not _has_megatron_checkpoint(args.load):
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
            args.load = args.ref_load
            if args.ref_ckpt_step is not None:
                args.ckpt_step = args.ref_ckpt_step
            args.start_rollout_id = 0


def _has_megatron_checkpoint(load_dir: str | None) -> bool:
    # [FUNCTION DOC] HELPER: True if the directory is given, exists, and contains the tracker file
    # latest_checkpointed_iteration.txt.
    return (
        load_dir is not None
        and os.path.exists(load_dir)
        and os.path.exists(os.path.join(load_dir, "latest_checkpointed_iteration.txt"))
    )


# ---------------------------- validation -----------------------------


def _assert_valid_trainer_ids(trainer_ids: list[str]) -> None:
    # [FUNCTION DOC] HELPER: Asserts the ids are unique, each matches the id pattern, and none
    # exceeds the maximum trainer id length; assertion messages list the offenders.
    assert len(set(trainer_ids)) == len(trainer_ids), (
        f"--megatron-config trainer ids must be unique, got {trainer_ids}; a trainer id addresses one trainer "
        f"controller and its engine pool, so two entries sharing it would land in the same pool"
    )
    _assert_valid_ids(trainer_ids, kind="trainer")

    too_long_trainer_ids = [trainer_id for trainer_id in trainer_ids if len(trainer_id) > TRAINER_ID_MAX_LENGTH]
    assert not too_long_trainer_ids, (
        f"--megatron-config trainer ids {too_long_trainer_ids} are longer than {TRAINER_ID_MAX_LENGTH} "
        f"characters; shorten them"
    )


def _assert_no_reserved_model_id(model_ids: list[str]) -> None:
    # [FUNCTION DOC] HELPER: Asserts the reserved id "eval" (naming the shared eval rollouts) is not
    # among the model ids.
    assert (
        RESERVED_MODEL_ID not in model_ids
    ), f"--megatron-config declares the model id {RESERVED_MODEL_ID!r}, which names the shared eval rollouts"


def _assert_valid_ids(ids: list[str], *, kind: str) -> None:
    # [FUNCTION DOC] HELPER: Asserts every id fully matches the DNS-label-style pattern (usable as
    # pool names and path components); the message names the kind ("model" or "trainer").
    bad_ids = [identifier for identifier in ids if MODEL_ID_PATTERN.match(identifier) is None]
    assert not bad_ids, (
        f"--megatron-config {kind} ids {bad_ids} are not usable as Kubernetes pool names or path components: "
        f"each must match {MODEL_ID_PATTERN.pattern}"
    )


# ---------------------------- override coercion -----------------------------


def _resolve_overrides(overrides: dict[str, Any], *, model_id: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns empty overrides unchanged (no parser built). Otherwise builds
    # the argument parser and coerces each override value to the type the parser declares for it,
    # allowing only per-policy args and those model-definition args the parser actually declares;
    # the error context names the model id. Returns the coerced mapping.
    if not overrides:
        return overrides

    parser = get_megatron_arg_parser()
    return coerce_dict_to_args(
        overrides,
        parser=parser,
        allowed_names=PER_POLICY_ARGS | (MODEL_DEFINITION_ARGS & declared_arg_dests(parser)),
        context=f"--megatron-config model {model_id!r}",
    )


def get_megatron_arg_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC]
    # Obtains the full argument parser (Megatron's plus the project's extras) without parsing real
    # arguments: calls Megatron's parse-args with an extra-args provider that adds the project's
    # arguments to the parser and then deliberately raises a private exception carrying the parser;
    # the function catches it and returns the parser. If parse-args returns without calling the
    # provider, raises an assertion error. Local imports avoid a cycle. Heavy: builds a throwaway
    # parser each call, and parse-args may touch global state.
    # TODO: revisit once the args refactor lands; this throwaway construction may then be optimized
    from miles.backends.megatron_utils.arguments import parse_args
    from miles.utils.arguments import get_miles_extra_args_provider

    class ParserCaptured(Exception):
        # [CLASS DOC] HELPER: Control-flow exception used to smuggle the built parser out of the
        # parse call; holds the parser as an attribute.
        def __init__(self, parser: argparse.ArgumentParser) -> None:
            # [FUNCTION DOC] HELPER: Stores the parser and sets a fixed message.
            super().__init__("the throwaway parser was captured before anything was parsed")
            self.parser = parser

    def capture(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # [FUNCTION DOC] HELPER: Extra-args provider: adds the project's extra arguments to the given
        # parser and always raises the capture exception with it (never returns).
        raise ParserCaptured(get_miles_extra_args_provider()(parser))

    try:
        parse_args(extra_args_provider=capture)
    except ParserCaptured as captured:
        return captured.parser
    raise AssertionError(
        "megatron's parse_args returned without calling the extra args provider, so the arguments this "
        "run declares could not be read; --megatron-config overrides cannot be typed"
    )
