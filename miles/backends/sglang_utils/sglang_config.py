# [MODULE DOC] Configuration models for SGLang engine deployment, in two layers. Raw models (names
# starting with an underscore) mirror the user's YAML (or are built from legacy arguments): a list
# of models, each with server groups giving worker type and GPU counts. Resolved models (server
# group, model, SGLang config) add per-group defaults taken from the model and the parsed arguments,
# absolute GPU and engine offsets laid out consecutively across all models, whether a group must
# offload (shares GPUs with training), and the merged server-argument overrides. Also contains the
# logic choosing the raw config source (YAML file, legacy prefill server count, or a single default
# regular group) and appending/filling an evaluation model. All models are immutable and strict
# about unknown fields.
"""Configuration models for SGLang engine deployment."""

import logging
from dataclasses import dataclass
from typing import Literal

import pydantic
import yaml

from miles.backends.sglang_utils.arguments import collect_eval_sglang_overrides
from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.lora.utils import is_multi_lora_enabled
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)


# ---------------------------- raw config -----------------------------


class _RawServerGroupConfig(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: User-facing description of one group of identical engines: worker type
    # (regular, prefill, decode, or placeholder that only reserves GPUs), total GPUs, optional GPUs
    # per engine, and a dictionary of server-argument overrides (default empty). No validation of
    # positivity happens here; the resolved model checks it.
    """Configuration for a single server group.

    Attributes:
        worker_type: One of "regular", "prefill", "decode", or "placeholder".
                     "placeholder" reserves GPU slots without creating engines.
        num_gpus: Total number of GPUs for this group.
        num_gpus_per_engine: GPUs per engine for this group.  Overrides the
                             model-level or global ``--rollout-num-gpus-per-engine``.
        overrides: Optional dict of SGLang ``ServerArgs`` field overrides.
                   These are applied on top of the base CLI ``--sglang-*``
                   arguments in ``_compute_server_args``.
    """

    worker_type: Literal["regular", "prefill", "decode", "placeholder"]
    num_gpus: int
    num_gpus_per_engine: int | None = None
    overrides: dict = {}


class _RawModelConfig(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: User-facing description of one deployed model: name, optional model path,
    # optional default GPUs per engine, list of raw server groups (accepts either the key "server
    # groups" or the older "engine groups" in the input; default empty), and an optional
    # update-weights flag whose none value means "infer later".
    """Configuration for a single model deployment.

    Attributes:
        name: Unique name for this model (e.g. "actor", "reward").
        model_path: HF checkpoint path.  Falls back to ``args.hf_checkpoint``.
        num_gpus_per_engine: Default GPUs per engine for all groups in this
                             model.  Individual groups can override.
        server_groups: Server group configurations for this model.
        update_weights: Whether this model receives weight updates from
                        training.  Set to ``False`` for frozen models
                        (reference, reward, etc.).  When ``None`` (default),
                        automatically inferred in ``resolve()``: ``True`` if
                        model_path matches ``args.hf_checkpoint``, ``False``
                        otherwise.
    """

    name: str
    model_path: str | None = None
    num_gpus_per_engine: int | None = None
    server_groups: list[_RawServerGroupConfig] = pydantic.Field(
        default_factory=list,
        validation_alias=pydantic.AliasChoices("server_groups", "engine_groups"),
    )
    update_weights: bool | None = None

    @property
    def total_num_gpus(self) -> int:
        # [FUNCTION DOC] Sum of the GPU counts of all this model's server groups (placeholders
        # included).
        return sum(g.num_gpus for g in self.server_groups)


class _RawSglangConfig(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Top-level raw deployment description: a required list of raw models
    # (accepted under either the key "models" or "sglang"). Can be loaded from a YAML file or inline
    # encoded payload, or built from the legacy prefill-server count option.
    """Configuration for SGLang engine deployment.

    Loaded from ``--sglang-config``: either a YAML file path or an inline ``base64:`` payload.

    **Config format**::

        sglang:
          - name: actor
            model_path: /path/to/actor
            update_weights: true          # receives training weight updates (default)
            num_gpus_per_engine: 2
            server_groups:
              - worker_type: prefill
                num_gpus: 4
                num_gpus_per_engine: 2
              - worker_type: decode
                num_gpus: 8
                num_gpus_per_engine: 4
          - name: ref
            model_path: /path/to/ref
            update_weights: false          # frozen, no weight updates
            server_groups:
              - worker_type: regular
                num_gpus: 4

    Each model gets its own router.  ``placeholder`` groups reserve GPU
    slots without creating engines.  ``overrides`` are ``ServerArgs``
    field names applied on top of the base ``--sglang-*`` CLI args.

    Set ``update_weights: false`` for frozen models (reference, reward,
    etc.) that should not receive weight updates from training.

    .. note::

       ``engine_groups`` is accepted as a backward-compatible alias for
       ``server_groups`` in the YAML config.
    """

    models: list[_RawModelConfig] = pydantic.Field(validation_alias=pydantic.AliasChoices("models", "sglang"))

    @classmethod
    def from_file_arg(cls, value: str) -> "_RawSglangConfig":
        # [FUNCTION DOC] Resolves the argument value to text (a file path or an inline encoded
        # payload, via the shared helper), parses it as safe YAML, and validates it into a config.
        # Raises on I/O, YAML or validation errors.
        return cls.model_validate(yaml.safe_load(resolve_file_arg(value)))

    @staticmethod
    def from_prefill_num_servers(args) -> "_RawSglangConfig":
        # [FUNCTION DOC] Static. Builds one model named "default" with a prefill group of
        # (prefill server count times rollout GPUs per engine) GPUs and a decode group of all
        # remaining rollout GPUs. Asserts that decode GPUs are positive (assertion error shows
        # totals). GPUs per engine for the groups is left unset (resolved later from arguments).
        """Build a config equivalent to the legacy --prefill-num-servers flag."""
        total_gpus = args.rollout_num_gpus
        prefill_gpus = args.prefill_num_servers * args.rollout_num_gpus_per_engine
        decode_gpus = total_gpus - prefill_gpus
        assert decode_gpus > 0, f"No decode GPUs: total {total_gpus}, prefill {prefill_gpus}"
        return _RawSglangConfig(
            models=[
                _RawModelConfig(
                    name="default",
                    server_groups=[
                        _RawServerGroupConfig(worker_type="prefill", num_gpus=prefill_gpus),
                        _RawServerGroupConfig(worker_type="decode", num_gpus=decode_gpus),
                    ],
                )
            ]
        )

    @property
    def total_num_gpus(self) -> int:
        # [FUNCTION DOC] Sum of GPU counts over all models.
        return sum(m.total_num_gpus for m in self.models)


# ---------------------------- resolved config -----------------------------


class ServerGroupConfig(FrozenStrictBaseModel):
    # [CLASS DOC] Resolved server group. Fields: worker type; positive total GPUs; positive GPUs per
    # engine; non-negative starting GPU offset and starting engine offset within the whole rollout
    # layout; the full server-argument overrides (model path always present, see resolve); and
    # whether the engines must offload memory to share GPUs with training.
    worker_type: Literal["regular", "prefill", "decode", "placeholder"]
    num_gpus: int = pydantic.Field(gt=0)
    num_gpus_per_engine: int = pydantic.Field(gt=0)
    gpu_offset: int = pydantic.Field(ge=0)
    engine_offset: int = pydantic.Field(ge=0)
    overrides: dict = pydantic.Field(default_factory=dict)
    needs_offload: bool

    @property
    def model_path(self) -> str:
        # [FUNCTION DOC] Returns the model path stored in the overrides (key error if absent, which
        # does not happen for groups built by resolve).
        return self.overrides["model_path"]

    @classmethod
    def resolve(
        cls,
        raw: _RawServerGroupConfig,
        args,
        default_gpus_per_engine: int,
        default_model_path: str,
        offset_cursor: "_OffsetCursor",
    ) -> "ServerGroupConfig":
        # [FUNCTION DOC] Class method building a resolved group from a raw group, the parsed
        # arguments, the default GPUs per engine, the default model path and a shared mutable offset
        # cursor. Asserts the overrides do not set host, port or gated launch port (engine URLs
        # come from the address allocator). Computes the absolute start of the group as the rollout
        # placement offset plus the cursor's GPU offset; the group needs offload when rollout
        # offloading is on and that start falls below the number of training GPUs. The new group
        # takes GPUs per engine from the raw group or the default; its overrides are merged with
        # precedence raw overrides over (disable memory saver, only when offloading is on but this
        # group does not need it) over the model path default. Then advances the cursor: GPUs by
        # the group's GPUs, engines by GPUs divided by the smaller of GPUs per engine and GPUs per
        # node (so multi-node engines count one per node). Returns the group.
        assert not ({"host", "port", "gated_launch_port"} & set(raw.overrides)), (
            f"sglang_overrides must not override host/port ({raw.overrides=}): the rollout process derives "
            f"each engine's url from the addr allocator, so an override would make it talk to the wrong endpoint"
        )

        rollout_pg_offset = _compute_rollout_offset(args)
        megatron_num_gpus = _compute_megatron_num_gpus(args)

        gpu_offset = offset_cursor.gpu
        group_abs_start = rollout_pg_offset + gpu_offset
        needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus

        ans = cls(
            worker_type=raw.worker_type,
            num_gpus=raw.num_gpus,
            num_gpus_per_engine=x if (x := raw.num_gpus_per_engine) is not None else default_gpus_per_engine,
            gpu_offset=gpu_offset,
            engine_offset=offset_cursor.engine,
            overrides={
                "model_path": default_model_path,
                **({"enable_memory_saver": False} if args.offload_rollout and not needs_offload else {}),
                **raw.overrides,
            },
            needs_offload=needs_offload,
        )

        offset_cursor.gpu += raw.num_gpus
        offset_cursor.engine += raw.num_gpus // min(ans.num_gpus_per_engine, args.num_gpus_per_node)
        return ans


class ModelConfig(FrozenStrictBaseModel):
    # [CLASS DOC] Resolved model: name, the model path exactly as written in the raw config (may be
    # none even though groups carry an effective path), resolved server groups, and a resolved
    # boolean saying whether it receives training weight updates.
    name: str
    model_path: str | None
    server_groups: list[ServerGroupConfig]
    update_weights: bool

    @classmethod
    def resolve(cls, raw: _RawModelConfig, args, offset_cursor: "_OffsetCursor") -> "ModelConfig":
        # [FUNCTION DOC] Class method resolving a raw model. Default model path is the raw path else
        # the Hugging Face checkpoint argument; default GPUs per engine is the raw value else the
        # rollout GPUs per engine argument. Resolves each group in order with these defaults and the
        # shared cursor. If there are groups, asserts they all ended up with one model path (the
        # effective path); with no groups the effective path is the default. Update-weights rule:
        # with multi-LoRA enabled it must not be explicitly true, the effective path must equal the
        # checkpoint argument (assertion errors), and the value becomes false; otherwise if unset,
        # it becomes false (with a warning log) when the effective path differs from the checkpoint,
        # else true; an explicit value is kept. Returns the resolved model.
        """Resolve per-group defaults from model-level then args-level values."""
        default_model_path = p if (p := raw.model_path) is not None else args.hf_checkpoint
        default_gpus_per_engine = n if (n := raw.num_gpus_per_engine) is not None else args.rollout_num_gpus_per_engine
        server_groups = [
            ServerGroupConfig.resolve(
                g,
                args,
                default_gpus_per_engine=default_gpus_per_engine,
                default_model_path=default_model_path,
                offset_cursor=offset_cursor,
            )
            for g in raw.server_groups
        ]

        if server_groups:
            model_paths = {g.overrides["model_path"] for g in server_groups}
            assert len(model_paths) == 1, (
                f"Model '{raw.name}' has server groups with different model_path values: "
                f"{model_paths}. All server groups within a model must use the same model_path."
            )
            effective_model_path = model_paths.pop()
        else:
            effective_model_path = default_model_path

        update_weights = raw.update_weights
        if is_multi_lora_enabled(args):
            assert update_weights is not True, "Tinker loads a frozen base; update_weights must be false"
            assert effective_model_path == args.hf_checkpoint, "Tinker engines must load the trainer's base checkpoint"
            update_weights = False
        elif update_weights is None:
            if effective_model_path != args.hf_checkpoint:
                logger.warning(
                    f"Model '{raw.name}' uses model_path='{effective_model_path}' which differs "
                    f"from hf_checkpoint='{args.hf_checkpoint}'. Defaulting update_weights to False. "
                    f"Set update_weights explicitly in the config to suppress this warning."
                )
                update_weights = False
            else:
                update_weights = True

        return cls(
            name=raw.name,
            model_path=raw.model_path,
            server_groups=server_groups,
            update_weights=update_weights,
        )

    @property
    def has_pd_disaggregation(self) -> bool:
        # [FUNCTION DOC] True if any group of this model is a prefill or decode group.
        return any(g.worker_type in ("prefill", "decode") for g in self.server_groups)

    @property
    def num_server_cells(self) -> int:
        # [FUNCTION DOC] Number of engines (cells) of this model: for every non-placeholder group,
        # GPUs divided (integer division) by GPUs per engine, summed.
        return sum(
            group.num_gpus // group.num_gpus_per_engine
            for group in self.server_groups
            if group.worker_type != "placeholder"
        )


class SglangConfig(FrozenStrictBaseModel):
    # [CLASS DOC] Resolved deployment: the list of resolved models, laid out consecutively.
    models: list[ModelConfig]

    @classmethod
    def resolve(cls, raw: _RawSglangConfig, args) -> "SglangConfig":
        # [FUNCTION DOC] Class method. Starts a cursor at GPU zero and engine zero, resolves each
        # raw model in order (sharing the cursor), and asserts the final GPU offset equals the raw
        # total GPU count. Returns the resolved config.
        offset_cursor = _OffsetCursor(gpu=0, engine=0)
        model_configs = [ModelConfig.resolve(m, args, offset_cursor) for m in raw.models]

        assert offset_cursor.gpu == raw.total_num_gpus
        return cls(models=model_configs)

    @property
    def has_pd_disaggregation(self) -> bool:
        # [FUNCTION DOC] True if any model uses prefill/decode disaggregation.
        return any(m.has_pd_disaggregation for m in self.models)


@dataclass
class _OffsetCursor:
    # [CLASS DOC] HELPER: Mutable running counters (next free GPU offset and next engine offset)
    # advanced as groups are resolved.
    gpu: int
    engine: int


def resolve_sglang_config(args) -> SglangConfig:
    # [FUNCTION DOC] Entry point: computes the raw config from the parsed arguments, then resolves
    # it. Returns the resolved deployment.
    """Build a SglangConfig from args, choosing the right source."""
    raw = _compute_raw_sglang_config(args)
    return SglangConfig.resolve(raw, args)


def _compute_raw_sglang_config(args) -> _RawSglangConfig:
    # [FUNCTION DOC] HELPER: Chooses the raw config source. If the run does not start inference
    # engines: an empty config. If a deployment YAML argument is set: loads it, asserts its total
    # GPUs equal rollout GPUs (zero if unset) plus eval GPUs; with no eval GPUs returns it as is,
    # otherwise asserts exactly one model is named "eval" with exactly the eval GPU count and fills
    # that model from the eval arguments. Otherwise: zero rollout GPUs gives an empty config; a
    # prefill server count gives the legacy prefill/decode config; else one "default" model with a
    # single regular group of all rollout GPUs. Then, if eval GPUs are set, appends a filled "eval"
    # model with one regular group. Returns the raw config.
    if not args.starts_inference_engines:
        return _RawSglangConfig(models=[])

    eval_num_gpus = args.eval_num_gpus
    rollout_num_gpus = args.rollout_num_gpus or 0

    if getattr(args, "sglang_config", None) is not None:
        config = _RawSglangConfig.from_file_arg(args.sglang_config)
        expected = rollout_num_gpus + eval_num_gpus
        actual = config.total_num_gpus
        assert (
            actual == expected
        ), f"sglang_config total GPUs ({actual}) != rollout_num_gpus + eval_num_gpus ({expected})"
        if eval_num_gpus == 0:
            return config
        eval_models = [m for m in config.models if m.name == "eval"]
        assert len(eval_models) == 1 and eval_models[0].total_num_gpus == eval_num_gpus, (
            f"--eval-num-gpus {eval_num_gpus} requires the sglang_config YAML to contain "
            f"exactly one model named 'eval' with that many GPUs."
        )
        return _RawSglangConfig(
            models=[_compute_eval_raw_model(m, args) if m.name == "eval" else m for m in config.models]
        )

    if rollout_num_gpus == 0:
        config = _RawSglangConfig(models=[])
    elif args.prefill_num_servers is not None:
        config = _RawSglangConfig.from_prefill_num_servers(args)
    else:
        config = _RawSglangConfig(
            models=[
                _RawModelConfig(
                    name="default",
                    server_groups=[_RawServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
                )
            ]
        )

    if eval_num_gpus == 0:
        return config

    eval_model = _compute_eval_raw_model(
        _RawModelConfig(
            name="eval",
            server_groups=[_RawServerGroupConfig(worker_type="regular", num_gpus=eval_num_gpus)],
        ),
        args,
    )
    return _RawSglangConfig(models=[*config.models, eval_model])


def _eval_sglang_overrides(args) -> dict:
    # [FUNCTION DOC] HELPER: Builds the server-argument overrides for the eval engines. Always turns
    # off returning routed experts and indexer top-k. If eval GPUs per engine differ from rollout
    # GPUs per engine, also forces data, pipeline, expert and attention-context parallel sizes to
    # one (with an info log), because inheriting them across a different tensor-parallel size would
    # make an unbootable engine. Finally the user's eval-prefixed options take precedence over
    # these derived values. Returns the merged mapping.
    """Eval-fleet engine settings; anything absent is inherited from the rollout engines."""
    overrides = {
        # Eval samples never feed training, so the replay side-channels are pure overhead.
        "enable_return_routed_experts": False,
        "enable_return_indexer_topk": False,
    }
    if args.eval_num_gpus_per_engine != args.rollout_num_gpus_per_engine:
        # Inheriting these across a different tp gives an engine SGLang refuses to boot.
        tp_coupled = ("dp_size", "pp_size", "ep_size", "attn_cp_size")
        overrides |= dict.fromkeys(tp_coupled, 1)
        logger.info(
            f"Eval tp={args.eval_num_gpus_per_engine} != rollout tp={args.rollout_num_gpus_per_engine}; "
            f"{', '.join(tp_coupled)} default to 1. Override with --eval-sglang-*."
        )
    return overrides | collect_eval_sglang_overrides(args)


def _compute_eval_raw_model(raw: _RawModelConfig, args) -> _RawModelConfig:
    # [FUNCTION DOC] HELPER: Returns a copy of the raw model with eval defaults filled in. The
    # update-weights flag defaults to false when unset (the eval engines are synced by snapshot
    # only, never joining the training broadcast group). Each group gets GPUs per engine from its
    # own value if truthy else the eval GPUs-per-engine argument, and overrides equal to the eval
    # overrides merged under the group's own overrides (group wins). The input is not mutated.
    # KNOWN ISSUE: a group GPUs-per-engine of zero would be treated as unset by the truthiness test.
    """Fill the eval model from the ``--eval-*`` args: YAML > ``--eval-sglang-*`` > ``--sglang-*``."""
    overrides = _eval_sglang_overrides(args)
    return raw.model_copy(
        update=dict(
            # Never joins the training broadcast group; the fleet is synced by snapshot only.
            update_weights=False if raw.update_weights is None else raw.update_weights,
            server_groups=[
                group.model_copy(
                    update=dict(
                        num_gpus_per_engine=group.num_gpus_per_engine or args.eval_num_gpus_per_engine,
                        overrides=overrides | group.overrides,
                    )
                )
                for group in raw.server_groups
            ],
        )
    )


def _compute_rollout_offset(args) -> int:
    # [FUNCTION DOC] HELPER: Zero when rollout-only debugging, colocation, or no engines are
    # started; the critic's total GPUs when only the critic trains; otherwise the actor's total
    # GPUs (nodes times GPUs per node).
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_rollout_only or args.colocate or not args.starts_inference_engines:
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    # [FUNCTION DOC] HELPER: Zero in rollout-only debugging; the critic's total GPUs when only the
    # critic trains; otherwise the actor's total GPUs (nodes times GPUs per node).
    # KNOWN ISSUE: the docstring says actor plus critic, but the critic is never added to the actor
    # count.
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if getattr(args, "debug_rollout_only", False):
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num
