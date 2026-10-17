# [MODULE DOC] HELPER:
# Names of the environment variables through which the platform tells a worker process who and
# where it is: its subprocess index under the supervisor, cell index, pod index, base GPU id, and
# the Kubernetes namespace and release. The identity tuple groups the three index variables that
# identify a process within the deployment (subprocess, cell, pod).
from __future__ import annotations

SUBPROCESS_INDEX_ENV_VAR = "MILES_SUPERVISOR_SUBPROCESS_INDEX"
CELL_INDEX_ENV_VAR = "MILES_CELL_INDEX"
POD_INDEX_ENV_VAR = "MILES_POD_INDEX"
BASE_GPU_ID_ENV_VAR = "MILES_BASE_GPU_ID"
NAMESPACE_ENV_VAR = "MILES_K8S_NAMESPACE"
RELEASE_ENV_VAR = "MILES_K8S_RELEASE"

PLATFORM_IDENTITY_ENV_VARS = (SUBPROCESS_INDEX_ENV_VAR, CELL_INDEX_ENV_VAR, POD_INDEX_ENV_VAR)
