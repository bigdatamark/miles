# [PACKAGE DOC]
# End-to-end tests of deployed training runs: runs installed as helm releases on a Kubernetes
# cluster (only that cluster backend is supported). The test files in this directory are thin
# entry points (hot restart with and without a checkpoint, the realistic GSM8K hot-restart soak,
# split deployment, multi-policy split deployment); their reusable scenario code, observers and
# assertions live in the helper package named conftest_deploy next to them. The package itself
# defines no code, and exists so these modules can be imported by dotted path.
