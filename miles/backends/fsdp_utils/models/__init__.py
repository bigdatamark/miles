# [PACKAGE DOC]
# Model-specific adaptations for the FSDP backend. Each module here holds patches or helpers for one
# Hugging Face model family (NemotronH, Qwen3 dense, Qwen3.5, Qwen3 MoE) plus shared MoE replay
# routers; they are imported and applied by the FSDP actor and plugin code when the matching model
# is loaded. The package init itself defines nothing.
