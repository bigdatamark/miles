# [PACKAGE DOC]
# HELPER: Code for the split-deployment deploy tests, where one training run is installed as
# several separate helm releases (trainers, rollout engines, and the orchestration script that
# drives the run). Modules: the split deployment module (deployment records, installing the
# releases in order and always cleaning them up, with validation that the deployments form one
# run), the deterministic scenario (baseline unsplit versus target split comparison) and the
# multi-policy scenario (solver and verifier policies on GSM8K). The package itself defines no
# code.
