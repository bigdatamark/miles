# [PACKAGE DOC] Multi-stream (hyper-connection) kernels where each token carries several parallel
# residual streams: the hc module (grouped-norm gated mixing of streams into one vector and
# write-back of a block output into every stream) and the mhc module (per-token stream mixing matrix
# applied to the residual plus the block output, and weighted aggregation of streams). This package
# file exports nothing; import from the submodules.
