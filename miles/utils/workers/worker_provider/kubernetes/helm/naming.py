# [MODULE DOC]
# Mirrors the helm chart's naming of Kubernetes objects so code outside the chart can compute the
# names and host addresses of statically addressed pools. Object names may be at most 63 characters;
# the component part of the name gets a budget leaving room for the cell index and revision hash
# suffixes that Kubernetes appends. Names over budget are cut and given a short digest of the release
# (or component) so they stay unique. Must stay in sync with the chart's templates.
from __future__ import annotations

import hashlib

from miles.utils.workers.worker_spec import BaseWorkerSpec, HostAndPort, NamedHostAndPorts

CHART_NAME = "miles-run"
UNINSTALLER_SERVICE_ACCOUNT = "miles-uninstaller"

MAX_OBJECT_NAME_LENGTH = 63
LONGEST_CELL_INDEX_SUFFIX = "-99999"
LONGEST_REVISION_HASH_SUFFIX = "-0123456789"
COMPONENT_NAME_BUDGET = MAX_OBJECT_NAME_LENGTH - len(LONGEST_CELL_INDEX_SUFFIX) - len(LONGEST_REVISION_HASH_SUFFIX)

RELEASE_DIGEST_LENGTH = 6
MIN_RELEASE_PREFIX_LENGTH = len("x-") + 2 * RELEASE_DIGEST_LENGTH


def static_cell_addrs(
    *, spec: BaseWorkerSpec, release: str, cell_index: int, worker_in_pod_index: int = 0
) -> NamedHostAndPorts:
    # [FUNCTION DOC] Endpoints of a statically addressed cell: the pool's stable host name for the
    # cell index with each port the spec declares, at the port's effective static port for the given
    # worker index in the pod (default zero). The port's cell offset flag is not applied.
    host = static_worker_host(release, spec.name, cell_index)
    return {
        port.name: HostAndPort(host=host, port=port.effective_static_port(worker_in_pod_index=worker_in_pod_index))
        for port in spec.port_infos
    }


def static_worker_host(release: str, component: str, cell_index: int) -> str:
    # [FUNCTION DOC] Host name of the first pod of a cell of a component: the component object name,
    # a dash and the cell index (the stateful pod name), a dot, then the component object name again
    # (the headless service name).
    name = component_name(release, component)
    return f"{name}-{cell_index}.{name}"


def component_name(release: str, component: str) -> str:
    # [FUNCTION DOC] Object name of a component within a release: the (possibly shortened) release
    # prefix, a dash, and the (possibly shortened) component name. The component is first limited to
    # the overall budget minus the minimum release prefix length and a dash, then the release prefix
    # gets whatever budget remains.
    component_budget = COMPONENT_NAME_BUDGET - MIN_RELEASE_PREFIX_LENGTH - 1
    component = _fit_to_budget(component, budget=component_budget, digest_of=component)
    budget = COMPONENT_NAME_BUDGET - (len(component) + 1)
    return f"{release_prefix(release, chart_name=CHART_NAME, budget=budget)}-{component}"


def release_prefix(release: str, *, chart_name: str, budget: int) -> str:
    # [FUNCTION DOC] Release prefix of object names: the release itself if it already contains the
    # chart name, else the release plus a dash and the chart name; then shortened to the budget with a
    # digest of the release.
    name = release if chart_name in release else f"{release}-{chart_name}"
    return _fit_to_budget(name, budget=budget, digest_of=release)


def _fit_to_budget(value: str, *, budget: int, digest_of: str) -> str:
    # [FUNCTION DOC] HELPER: Return the value unchanged if within the budget. Otherwise cut it so that
    # a dash plus a six-byte blake2b hexadecimal digest of the digest source fit within the budget,
    # remove one trailing dash left by the cut, and append the dash and digest.
    if len(value) <= budget:
        return value
    digest = hashlib.blake2b(digest_of.encode(), digest_size=RELEASE_DIGEST_LENGTH).hexdigest()
    return f"{_trim_suffix(value[: budget - (len(digest) + 1)], '-')}-{digest}"


def _trim_suffix(value: str, suffix: str) -> str:
    # [FUNCTION DOC] HELPER: Remove the suffix once if the value ends with it (and the suffix is
    # non-empty); otherwise return the value.
    return value[: -len(suffix)] if suffix and value.endswith(suffix) else value
