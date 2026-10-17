# [PACKAGE DOC] DeepSeek-V4.1 model plugin for Megatron-based training in miles. Builds on the
# DeepSeek-V4 plugin (shared rotary, context-parallel and quantization helpers) and adds the V4.1
# specifics: attention layers that share compressed key/value latents and indexer keys across layers
# (source layers), a candidate-block mechanism for top-k selection, an optional "engram" hashed n-gram
# embedding module, hyper-connection (multi-residual-stream) layer variants, and optional carrying of
# cross-layer state through pipeline stages. Submodules: the main model/spec module, the engram module,
# and the ops subpackage (compressor, indexer, key/value norm kernels). This file has no code.
