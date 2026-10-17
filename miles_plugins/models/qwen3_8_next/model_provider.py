# [MODULE DOC] Model provider for Qwen3.8-Next. Builds the standard Miles Megatron model, then
# registers forward hooks on it that compute and publish the per-layer-embedding (PLE) n-gram ids
# as a process-global side channel, since the input token ids never reach the transformer layer
# that needs them. Hooks are only installed on the pipeline stage that hosts the PLE embedding.
"""Custom provider: miles' default model + forward hooks publishing the PLE
n-gram side channel (input token ids never reach a transformer layer).
"""

import logging

from megatron.core.models.gpt import GPTModel

from miles_plugins.models.qwen3_8_next.ops.ple import (
    Qwen38NextFrozenNGramEmbedding,
    build_ngram_contexts_packed,
    clear_ple_batch,
    publish_ple_batch,
)

logger = logging.getLogger(__name__)


def _find_ple_embedding(model):
    # [FUNCTION DOC] HELPER: scan the model's named sub-modules in order and return the first
    # frozen n-gram embedding module, or None when this model chunk does not contain one.
    for _, module in model.named_modules():
        if isinstance(module, Qwen38NextFrozenNGramEmbedding):
            return module
    return None


def _install_ple_context_hooks(model: GPTModel) -> None:
    # [FUNCTION DOC] HELPER: if the model hosts the PLE embedding, register a forward pre-hook and
    # a forward post-hook on the whole model and log an info message; otherwise do nothing.
    ple_embedding = _find_ple_embedding(model)
    if ple_embedding is None:
        return

    def pre_hook(_module, args, kwargs):
        # [FUNCTION DOC] HELPER: forward pre-hook (receives keyword arguments). Finds input ids from
        # the keyword argument or else the first positional argument; returns without doing anything
        # if absent. Takes the query cumulative sequence lengths from the packed sequence params if
        # given. Flattens the ids, builds packed n-gram contexts (using the embedding's n-gram size
        # and end-of-sequence token id, respecting sequence boundaries), computes n-gram ids via the
        # embedding, and publishes ids plus cumulative lengths to the global side channel.
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            return
        packed = kwargs.get("packed_seq_params")
        cu_seqlens = getattr(packed, "cu_seqlens_q", None) if packed is not None else None
        flat = input_ids.reshape(-1)
        contexts = build_ngram_contexts_packed(flat, cu_seqlens, ple_embedding.ngram_size, ple_embedding.eos_token_id)
        ngram_ids = ple_embedding.compute_ngram_ids(contexts)
        publish_ple_batch(ngram_ids, cu_seqlens)

    def post_hook(_module, _args, _output):
        # [FUNCTION DOC] HELPER: forward post-hook; clears the published PLE batch after the model
        # forward finishes. Returns nothing (leaves the output unchanged).
        clear_ple_batch()

    model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    model.register_forward_hook(post_hook)
    logger.info("PLE n-gram context hooks installed on the stage hosting the PLE layer")


def get_qwen3_8_next_model_provider(pre_process: bool = True, post_process: bool = True, vp_stage=None):
    # [FUNCTION DOC] Custom model provider entry point (Megatron calls it per pipeline / virtual
    # stage). Fetches the global Megatron args, temporarily clears the custom-provider-path argument
    # (restored in a finally block) so that the default Miles provider is resolved instead of
    # recursing into this function, builds the model with the same pre/post-process flags and virtual
    # stage, installs the PLE hooks, and returns the model. Side effect: temporary mutation of the
    # global args object.
    from megatron.training import get_args

    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    args = get_args()
    saved = args.custom_model_provider_path
    args.custom_model_provider_path = None
    try:
        base_provider = get_model_provider_func(args)
    finally:
        args.custom_model_provider_path = saved

    model = base_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    _install_ple_context_hooks(model)
    return model
