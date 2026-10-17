# [PACKAGE DOC] Weight-update support for the Megatron backend: modules that export the trained
# Megatron weights (and LoRA adapters) as Hugging Face-named tensors, gathering them across tensor,
# expert and pipeline parallel ranks, optionally converting or quantizing them, so they can be
# pushed to the rollout (SGLang) engines. This package marker itself defines no names; import the
# concrete modules (the shared iterator base and factory, the direct and bridge iterators, and the
# expert gather helper) directly.
