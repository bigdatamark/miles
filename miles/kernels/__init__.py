# [PACKAGE DOC] Custom GPU kernels used by miles training models. Grouped by subpackage:
# activation (fp32-accumulating SwiGLU and short convolution), attention (dense backward, DSA
# sparse attention with indexer, and QSA sparse attention), embedding (per-layer-embedding gather
# and gated convolution), hyper_connection (hyper-connection mixing), moe (fused expert
# computation forward and backward), norm (grouped RMS normalization), position (rotary position
# embedding) and quant (fake quantization and FP8/NVFP4/INT4 helpers). This top-level package
# itself defines nothing and imports nothing.
