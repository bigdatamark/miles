# [PACKAGE DOC] Example package for the fully asynchronous training feature: rollout generation runs
# continuously and decoupled from the training steps. Contains launcher scripts that train Qwen3
# models in fully asynchronous mode (a 30B mixture-of-experts one and a 4B one with evaluation) and
# an external evaluation function module. The package itself defines no code; the empty marker file
# only lets sibling modules be imported by dotted path.
