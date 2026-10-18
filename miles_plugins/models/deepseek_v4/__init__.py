# [PACKAGE DOC]
# DeepSeek-V4 model plugin for the Miles training backend: command-line argument declarations and
# validation (arguments), the attention layer and layer spec builder (deepseek_v4), and the operator
# helpers it uses (ops: compressor, indexer, rotary embeddings, context-parallel and packed-sequence
# index utilities, quantization-aware training simulation). The package itself imports nothing, so
# argument parsing can run before heavy kernel libraries are loadable.

