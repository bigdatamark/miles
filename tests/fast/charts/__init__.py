# [PACKAGE DOC]
# CPU tests of the Helm charts under the repository charts directory: the shared library chart
# (miles_common), the run chart (miles_run) and the workbench chart (miles_workbench), plus a
# cross-chart test of the shared infra values contract. Shared helpers (rendering charts with helm,
# picking objects from rendered output, running the workbench tool) live in utils, and conftest
# vendors chart dependencies once per session. Most tests need the helm executable and are skipped
# without it (a failure in CI).
