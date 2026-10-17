# [PACKAGE DOC] Custom operators for the Qwen3.8-Next plugin: the QSA (indexer-selected sparse)
# full-attention layer (attention), the QSA indexer that picks which key blocks each query attends
# (qsa_indexer), and per-layer embeddings, PLE (ple): n-gram hashing, a thread-local side channel
# for n-gram ids, and the frozen host-resident embedding table. The heavy compute lives in kernels
# under the miles kernels package; this package holds the Megatron module wrappers. This __init__
# defines nothing.
"""Qwen3.8-Next ops: QSA attention and indexer, PLE, and their triton kernels."""
