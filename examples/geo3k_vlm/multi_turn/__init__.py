# [PACKAGE DOC] Multi-turn vision-language example package for the geo3k geometry problem set.
# Contains: the abstract interaction environment contract (base_env), a geo3k environment in which
# the model calls a scoring tool with its answer (env_geo3k), the custom multi-turn rollout function
# that alternates SGLang generation with environment steps while tracking tokens, loss mask, log
# probabilities and multimodal inputs (rollout), a config file with the turn limit, and a launcher
# script (run_geo3k_vlm_multi_turn). The package marker itself defines no code.
# KNOWN ISSUE: the original one-line comment below calls this a Sokoban example, but the package
# implements geo3k, not Sokoban.
# Multi-turn VLM Sokoban example package
