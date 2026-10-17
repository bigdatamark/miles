# [PACKAGE DOC] Qwen3.8-Next model plugin package for Megatron. Holds the replicated Gated DeltaNet
# attention (gdn), the hyper-connection modules for Megatron's residual-stream spec slots
# (hyper_connection), the custom model provider that installs per-layer-embedding (PLE) n-gram
# hooks (model_provider), the spec / layer construction entry point (qwen3_8_next), and the custom
# ops subpackage (ops: attention, PLE embeddings, and the QSA indexer). This __init__ itself defines
# nothing and has no import-time side effects.
