# [PACKAGE DOC]
# Version 2 of the session server: instead of a single linear conversation per session, it keeps
# a tree of generations (nodes hold the request/response record and token ids; a request attaches
# under the node whose history it extends), so branching conversations such as retries or
# sub-agents are all kept. Modules: core (the v2 session core overriding chat completion, get
# session and sample collection), session_state (registry, tree-session state, request preparation
# and commit), tree_trajectory (node and tree data structures), utils (leaf sample assembly and tree
# metadata), metrics (per-session rollout metrics), and the picker_hub and postprocessor_hub
# packages (pluggable hooks choosing which leaves become samples and post-processing them).
# Enabled with the use-session-server launch option set to v2. The init itself defines nothing.
"""Session server v2: tree-of-trajectories serving, opt-in via --use-session-server v2.

v1 (linear_trajectory + core) stays the default and is untouched; this package
holds the entire tree implementation.
"""
