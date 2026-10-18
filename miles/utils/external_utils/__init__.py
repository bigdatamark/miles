# [PACKAGE DOC] Utilities for running Miles on external infrastructure rather than inside the
# trainer process: colocate_pairing (a Kubernetes controller that pins inference pods onto the
# nodes of the trainer pods they share GPUs with) and command_utils (building and launching
# training commands through pluggable backends such as helm).
