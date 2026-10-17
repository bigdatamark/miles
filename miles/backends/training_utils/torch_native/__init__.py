# [PACKAGE DOC]
# Backend-agnostic training actor for plain PyTorch-native training stacks (non-Megatron): the actor
# base class drives the rollout training loop (log-prob passes, advantage computation, optimizer
# steps, weight update, sleep/wake offload) while concrete backends supply the model parts,
# optimizers and a step runner. offload moves model and optimizer state between GPU and pinned host
# memory; step_runner defines the forward/backward runner protocol. The package file exports nothing.
