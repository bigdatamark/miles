# [PACKAGE DOC] Optimizer plugins for Miles training. Currently holds the NVMe streaming module
# (nvme_stream), which moves Megatron distributed-optimizer state (fp32 main parameters and Adam
# moments) to files on node-local disk and streams it through the GPU one bucket at a time, plus
# a file-backed variant for Muon's state. This __init__ defines nothing and has no import-time
# side effects.
