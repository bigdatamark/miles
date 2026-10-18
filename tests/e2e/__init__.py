# [PACKAGE DOC]
# Package marker for the end-to-end (GPU, model-downloading) test tree. Each test file below it
# is a self-contained script that registers itself with the CI suite declarations, prepares
# models and datasets under the configured model and data directories, and launches a real
# training or rollout job; the subpackages group tests by feature area (agentic, checkpoint,
# deploy, FSDP, fault tolerance, and others). The package itself defines no code, and exists so
# that shared helper modules in this directory (directory-path helpers, dumper and multi-policy
# fixture modules) can be imported with the full dotted path from the repository root.
