# [PACKAGE DOC] Code that runs inside each torchrun process of the standalone Megatron tool: the
# worker entry point (main), input batch construction and loss (batch), per-token logprob output
# files (output), routing replay save and load (replay), the dataclass of script options passed
# from the command-line layer (script_args), and top-k prediction printing (top_k_print).
# The worker is not meant to be started by hand; the run command builds its command line.
