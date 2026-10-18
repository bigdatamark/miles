# [PACKAGE DOC]
# HELPER: Shared scenario and assertion code for the deploy end-to-end tests. Subpackages: "common"
# (helpers shared by all deploy scenarios: example argument building and cluster checks),
# "hot_restart" (driving, observing and verifying hot restarts, the take-over of a live run's
# orchestration script by upgrading its helm release), and "split" (split-deployment scenarios in
# which trainers and rollout engines run as separate deployments). The package itself defines no
# code.
