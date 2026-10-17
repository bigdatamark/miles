# [PACKAGE DOC]
# Fault-tolerance helpers for the Megatron backend: independent data-parallel groups across
# training cells (creation, reconfiguration, cross-cell gradient all-reduce), in-memory
# checkpoints, and checkpoint transfer between cells to heal a replaced cell. The package init
# itself defines nothing.
