# [PACKAGE DOC]
# Session server package: an HTTP service that sits between agent frameworks (OpenAI-style chat
# completion clients, optionally Anthropic-style messages clients) and the SGLang inference
# backend during rollouts. It keeps per-session state of the conversation as token ids (TITO,
# token-in token-out), so that every request is sent to the engine with a prefix-consistent token
# sequence, records each turn's tokens, log-probs and replay data, and later turns a session's
# trajectory into training samples. Sub-modules: config (server configuration), core (shared HTTP
# helpers), errors (typed errors with HTTP status codes), linear_trajectory and sessions (the
# linear, append-only session implementation), server (HTTP app), request_args, types, the
# samples subpackage (sample encoding and merging) and the v2 subpackage (tree-structured
# trajectories). The package init itself defines nothing.
