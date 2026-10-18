# [PACKAGE DOC] Root package of the fast, CPU-only unit-test suite. Per the CI registration rules,
# tests here need no registration declaration, run on GitHub-hosted CPU runners as the first stage
# ("stage-a-cpu", which gates the CUDA stages), and are expected to finish in seconds. Subpackages
# mirror the source layout under test (backends, rollout, router, ray, utils, models, optimizers,
# plugins, dashboard, tools, tinker and others); the package marker also makes shared helper modules
# in this directory (for example the distributed-utils, source-scan and training-args helpers)
# importable as part of the tests.fast namespace. It defines no code of its own.
