# [PACKAGE DOC] End-to-end test package for the "short" CI cases: quick single-node runs of the
# miles training loop on small models (Qwen2.5-0.5B, Qwen3-0.6B and similar) covering rollout
# variants (compact, external, fully asynchronous, legacy API, different communication and store
# backends), on-policy distillation, FSDP colocation, SFT snapshot evaluation, multi-policy
# training, the Megatron run script and the tensor dumper with its comparator. The package itself
# defines no code and exports nothing; it only makes the test modules importable by dotted path.
