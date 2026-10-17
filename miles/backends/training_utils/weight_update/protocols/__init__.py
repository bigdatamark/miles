# [PACKAGE DOC]
# Concrete weight transfer protocols implementing the base protocol contract: broadcast (NCCL
# broadcast to non-colocated engines), cuda_ipc (colocated engines via CUDA IPC handles, optionally
# plus a distributed tail), delta (disk-based delta publishing), p2p (peer-to-peer transfer, with
# its helpers in p2p_transfer_utils). The package file exports nothing; the protocol factory imports
# the modules lazily.
