# [MODULE DOC] Factories that produce the per-pipeline-stage model builder function Megatron's
# get-model helper expects. Three sources are supported, chosen in order: a user-supplied custom
# provider function loaded by path, a Megatron Bridge provider built from a HuggingFace
# checkpoint, and the default Megatron GPT model built from command-line arguments.
# Each returned builder takes pre/post-process flags (whether this pipeline stage holds the
# embedding / output layer), an optional virtual-pipeline stage, a config (must be none because
# the config is derived from arguments) and an optional process-group collection.
# For the critic role the output layer is replaced by a one-output linear head. Also contains
# optional witness installation and freezing of the native DSA indexer.
# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
import logging
from contextlib import nullcontext
from typing import Literal

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from miles.utils.audit_utils.witness.module import install_witness
from miles.utils.function_registry import load_function
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)


def _apply_bridge_runtime_config(provider, args: argparse.Namespace) -> None:
    # [FUNCTION DOC] HELPER: Copies a fixed list of training-time settings from the arguments onto
    # the provider object, mutating it and returning nothing (rationale in the docstring).
    # Always copied: tensor, pipeline, expert, expert-tensor and context parallel sizes; sequence
    # parallel; per-token loss and variable sequence lengths flags; fp32 softmax in attention,
    # gradient accumulation fusion, fp32 residual, deterministic mode; activation recompute
    # granularity/method/layer count/modules; CPU offloading layer count and distributed saved
    # activations; tensor-parallel comm overlap; fp8 mode and recipe; attention backend; MoE token
    # dispatcher type.
    # Conditionally copied: MTP head detaching is switched on when MTP training is enabled; CPU
    # offloading flag only when the arguments carry it; first/last pipeline stage layer counts
    # only when set (names differ from the provider fields); MoE router bias update rate and aux
    # loss coefficient only when set. Finally applies the DSA backend arguments through a helper
    # imported locally (a test executes this function in a bare namespace, so the import must stay
    # inside).
    """Copy the runtime config from args onto a bridge-built provider.

    Bridge mode builds the model from the HF checkpoint and skips
    core_transformer_config_from_args, so command-line args never reach the
    provider. We copy only some fields, not all of args: the provider already
    holds the right values from the HF checkpoint, while args only has default
    values for model shape, dtype, and fields the provider set on purpose.
    Copying those would quietly break the model -- the bridge only logs a
    warning and keeps going, it does not fail. So we copy just the training,
    parallelism, memory, and numerics settings that really come from args. Put
    new training flags here, not spread across the code.
    """
    # parallelism / sharding
    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.sequence_parallel = args.sequence_parallel
    provider.context_parallel_size = args.context_parallel_size

    # loss / sequence handling
    provider.calculate_per_token_loss = args.calculate_per_token_loss  # CP>1 VL models assert this
    provider.variable_seq_lengths = args.variable_seq_lengths

    # Match the non-bridge path: MTP must only train its own draft parameters.
    if getattr(args, "enable_mtp_training", False):
        provider.mtp_detach_heads = True

    # numerics (training infra, not model-defining)
    provider.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
    provider.gradient_accumulation_fusion = args.gradient_accumulation_fusion
    provider.fp32_residual_connection = args.fp32_residual_connection
    provider.deterministic_mode = args.deterministic_mode

    # activation recompute (silently dropped before -> no checkpointing -> OOM at long context)
    provider.recompute_granularity = args.recompute_granularity
    provider.recompute_method = args.recompute_method
    provider.recompute_num_layers = args.recompute_num_layers
    provider.recompute_modules = args.recompute_modules

    # activation / memory offload
    provider.cpu_offloading_num_layers = args.cpu_offloading_num_layers
    provider.distribute_saved_activations = args.distribute_saved_activations
    # cpu_offloading is derived, set only when cpu_offloading_num_layers>0; guard its presence.
    if hasattr(args, "cpu_offloading"):
        provider.cpu_offloading = args.cpu_offloading

    # communication overlap
    provider.tp_comm_overlap = args.tp_comm_overlap

    # fp8
    provider.fp8 = args.fp8
    provider.fp8_recipe = args.fp8_recipe

    # attention kernel selection
    provider.attention_backend = args.attention_backend

    # MoE token dispatcher (same-name, always present)
    provider.moe_token_dispatcher_type = args.moe_token_dispatcher_type

    # arg name != provider field; arg default None, so propagate only when the user set it
    if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
        provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
    if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
        provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers

    # MoE training knobs: override only when explicitly set, else keep the provider's value
    if getattr(args, "moe_router_bias_update_rate", None) is not None:
        provider.moe_router_bias_update_rate = args.moe_router_bias_update_rate
    if getattr(args, "moe_aux_loss_coeff", None) is not None:
        provider.moe_aux_loss_coeff = args.moe_aux_loss_coeff

    # Imported here: test_bridge_mtp_detachment.py exec()s this function's AST in a bare namespace (no module names).
    from miles.utils.megatron_bridge_utils import apply_dsa_backend_args

    apply_dsa_backend_args(provider, args)


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class LinearForLastLayer(torch.nn.Linear):
    # [CLASS DOC] Plain linear output head used instead of the language-model output layer for the
    # critic role (value head). It mimics the call signature of Megatron's output layer: forward
    # returns a pair of (logits in fp32, none). Under sequence parallelism, the weight is flagged
    # so its gradient is reduced over the sequence-parallel group and the logits are gathered
    # back over the sequence dimension.
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        # [FUNCTION DOC] Creates a linear layer of the given input and output sizes with optional
        # bias (default true). Remembers the sequence-parallel flag from the config and, when on,
        # marks the weight as sequence-parallel. Initializes weights from a normal distribution
        # (mean 0, standard deviation 0.02) and zeroes the bias.
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        # [FUNCTION DOC] Applies the linear layer to the input, casts the result to fp32, and if
        # sequence parallelism is on gathers the logits across the sequence-parallel region (with
        # no output-gradient parallelism). Returns the logits and none.
        # KNOWN ISSUE: the weight and runtime-gather-output parameters are accepted for interface
        # compatibility but ignored.
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # [FUNCTION DOC] Returns the model builder function for the given arguments and role (actor by
    # default). Selection order: (1) if a custom model provider path is set, a wrapper that loads
    # that function on each call, forwards pre/post-process flags (and the virtual stage only if
    # the custom function accepts it), swaps in the critic head on the post-process stage for the
    # critic role, and installs witness if enabled; (2) if the conversion mode is bridge, loads the
    # Megatron Bridge provider from the HuggingFace checkpoint path (trusting remote code) without
    # weights, copies runtime config from arguments, disables embedding tying for the critic,
    # finalizes it, and returns a wrapper that builds from it; (3) otherwise the default builder.
    # The returned builders assert that no config is passed in.
    # Side effects: the bridge path performs file/network reads of the HF checkpoint config at
    # selection time.
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True,
            post_process: bool = True,
            vp_stage: int | None = None,
            config: TransformerConfig | None = None,
            pg_collection=None,
        ) -> GPTModel:
            # [FUNCTION DOC] Builder for the custom-provider path. Asserts config is none, loads the
            # user function from the registry by path, calls it with pre/post-process flags (plus
            # the virtual stage if its signature has such a parameter), replaces the output layer
            # with the one-output head when this is the last stage of a critic, installs witness
            # when enabled, and returns the model. The process-group collection argument is unused.
            assert config is None, "miles builds the config from args, so it expects config to be None"
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)
            # Apply critic output layer if needed
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size, output_size=1, config=model.config
                )
            _maybe_install_witness(args, model)
            return model

        return wrapped_model_provider

    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        _apply_bridge_runtime_config(provider, args)
        if role == "critic":
            provider.share_embeddings_and_output_weights = False
        provider.finalize()

        def wrapped_bridge_provider(
            pre_process: bool = True,
            post_process: bool = True,
            vp_stage: int | None = None,
            config: TransformerConfig | None = None,
            pg_collection=None,
        ) -> GPTModel:
            # [FUNCTION DOC] Builder for the bridge path. Asserts config is none; if a process-group
            # collection is given, stores it on the shared provider (needed by pipeline-parallel
            # providers). Calls the provider's provide method with the stage flags, swaps the
            # output layer for the critic head on the last stage of a critic, asserts witness is
            # not enabled, and replaces the model's forward so that when it returns a tuple only
            # the first element (logits) is returned. Returns the model.
            assert config is None, "miles builds the config from args, so it expects config to be None"
            # PP>1 paths in megatron.bridge providers (e.g. mamba_provider) read
            # self._pg_collection.pp during provide(); without forwarding the
            # caller's pg_collection here, those code paths hit AttributeError.
            if pg_collection is not None:
                provider._pg_collection = pg_collection
            model = provider.provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size, output_size=1, config=model.config
                )
            assert not getattr(args, "enable_witness", False), "Witness is not supported yet in this mode"
            # Gemma-4 forward returns (logits, loss_mask); keep logits only.
            _bridge_forward = model.forward

            def _logits_only_forward(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls the original forward with all given arguments and
                # returns only the first element if the result is a tuple, else the result as is.
                out = _bridge_forward(*args, **kwargs)
                return out[0] if isinstance(out, tuple) else out

            model.forward = _logits_only_forward
            return model

        return wrapped_bridge_provider

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
        config: TransformerConfig | None = None,
        pg_collection=None,
    ) -> GPTModel:
        # [FUNCTION DOC] Default builder from arguments. Asserts config is none, then derives the
        # transformer config from arguments (enabling MTP head detaching when MTP training is on).
        # Layer spec: if a spec path is given it is imported (and called with arguments, config and
        # virtual stage when callable); else for MoE models a decoder block spec is built (passing
        # the stage when set); else a single-layer spec using Transformer Engine or the local
        # implementation with the MoE, grouped-GEMM, qk-layernorm, latent-attention (and for local:
        # normalization and kitchen) options.
        # With fp8 parameter gather, wraps model construction in Transformer Engine's fp8 model
        # init context (preserving high-precision initial values when supported); raises a runtime
        # error if that import fails. Builds the GPT model with vocabulary, position, rotary, rope
        # scaling, tied-embedding (disabled for critic or when untie is requested) and fp16 loss
        # settings. With MTP layers configured, also builds an MTP block spec; while doing so,
        # temporarily disables the routing replay manager (when rollout routing replay is on) and
        # restores its previous state afterwards. Then freezes the native DSA indexer if
        # applicable, swaps in the critic head on the last stage of a critic, installs witness if
        # enabled, and returns the model.
        # KNOWN ISSUE: the docstring mentions legacy models, which this code never builds, and the
        # process-group collection argument is unused.
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        assert config is None, "miles builds the config from args, so it expects config to be None"
        config = core_transformer_config_from_args(args)

        # `enable_mtp_training` comes from miles' arg parser; megatron-only arg contexts
        # (e.g. the run_megatron debug worker) won't have it, so default to False.
        if getattr(args, "enable_mtp_training", False):
            # Detach the MTP heads so RL MTP gradients do not flow into the shared
            # output layer / embedding.
            config.mtp_detach_heads = True

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        normalization=args.normalization,
                        use_kitchen=config.use_kitchen,
                        use_kitchen_attention=config.use_kitchen_attention,
                        kitchen_attention_backend=config.kitchen_attention_backend,
                    )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except Exception as e:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": role != "critic" and not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            # hard code here to skip r3 registration for mtp layers
            # getattr is required to avoid ckpt conversion errors
            if getattr(args, "use_rollout_routing_replay", False):
                prev_routing_replay_enabled = routing_replay_manager.enabled
                routing_replay_manager.enabled = False
                logger.warning(
                    "Rollout routing replay is not applicable for MTP modules, so skipped replay registration"
                )
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec
            if getattr(args, "use_rollout_routing_replay", False):
                # restore instead of forcing True: the critic role keeps the manager disabled
                routing_replay_manager.enabled = prev_routing_replay_enabled

        with build_model_context(**build_model_context_args):
            model = GPTModel(**kwargs)

        _maybe_freeze_native_dsa_indexer(args, model)

        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

        _maybe_install_witness(args, model)

        return model

    return model_provider


def _maybe_freeze_native_dsa_indexer(args: argparse.Namespace, model: GPTModel) -> None:
    # [FUNCTION DOC] HELPER: When the DSA implementation is Megatron's native one and the indexer
    # loss coefficient is zero, turns off gradients for every parameter whose name contains the
    # attention indexer path, so DDP and weight decay skip them. No-op otherwise.
    if args.dsa_impl == "megatron" and model.config.dsa_indexer_loss_coeff == 0:
        # Native DSA runs its indexer under no_grad without the auxiliary objective.
        # Exclude these unused parameters from DDP and optimizer weight decay.
        for name, parameter in model.named_parameters():
            if ".self_attention.core_attention.indexer." in name:
                parameter.requires_grad_(False)


def _maybe_install_witness(
    args: argparse.Namespace,
    model: GPTModel,
) -> None:
    # [FUNCTION DOC] HELPER: If the witness feature is enabled in the arguments, installs witness
    # on the model with the configured buffer size and the model config's sequence-parallel flag
    # (default false). Mutates the model; otherwise does nothing.
    if getattr(args, "enable_witness", False):
        install_witness(
            model,
            buffer_size=args.witness_buffer_size,
            sequence_parallel=getattr(model.config, "sequence_parallel", False),
        )
