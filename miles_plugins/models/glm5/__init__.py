# [PACKAGE DOC] GLM-5 family (also usable for DeepSeek-V3.2 style models) plugin using deepseek sparse
# attention (DSA): an indexer picks the top-k keys per query and a sparse multi-latent attention
# attends only to them. Submodules: arguments (command-line options and normalization that route to the
# in-repo or native Megatron DSA implementation), glm5 (the in-repo attention layer and spec builder,
# including optional cross-layer sharing of top-k picks), and megatron_spec (spec provider that uses
# Megatron's native DSA). This file has no code.
