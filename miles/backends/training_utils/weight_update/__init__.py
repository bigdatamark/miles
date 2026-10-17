# [PACKAGE DOC]
# Trainer-to-inference weight synchronization. Contains the updater that orchestrates syncing
# (updater, session, protocol and the concrete transfer protocols under protocols), connection
# staleness tracking (conn_status), publishing of weight snapshots (snapshot_publisher), shared
# helpers (utils) and the backend-neutral iterator that streams the training model's weights as
# size-bounded buckets of HF-named tensors (hf_weight_iterator). The package file exports nothing.
