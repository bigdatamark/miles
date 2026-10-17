# [PACKAGE DOC] GLM-5.3 ("next") hybrid model plugin for Megatron. The model interleaves two kinds of
# attention layers: full-attention layers that use deepseek-style sparse attention with a pooled-key
# indexer and no rotary embedding (module dsa), and linear-attention layers using Kimi delta attention
# (module kda); it also uses multi-stream hyper-connections with a mean output contraction. Module
# glm5_next builds the layer specs and config from the Hugging Face config; subpackage ops holds the
# pooled-key top-k selection. This file has no code.
