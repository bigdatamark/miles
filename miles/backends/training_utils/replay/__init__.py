# [PACKAGE DOC]
# Routing-replay support for the training backend (for mixture-of-experts models): the data module
# aligns rollout-recorded expert routing tensors with the local token layout and registers them in
# per-module replay objects, and the routing_replay module is a thin facade over the shared routing
# replay manager (enable, fill, stage switching, rewind, reset). The package file itself exports
# nothing.
