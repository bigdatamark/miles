# [PACKAGE DOC] Debug tool for running a standalone Megatron forward (optionally backward) pass on a
# HuggingFace checkpoint under a chosen parallelism layout, dumping activations and per-token
# logprobs, and comparing two layouts (a baseline and a target) against each other.
# Layout: cli (typer command-line app with run, compare, run-and-compare and show-model-args
# commands plus helpers that build torchrun commands, prompts and paths), worker (the script that
# torchrun launches on every rank), and logprob_comparator (compares saved per-token logprobs).
# Run as a module: the main entry point defers to the cli app.
