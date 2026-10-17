# [MODULE DOC]
# Constants and naming for the workbench: chart directory (found four levels above this package,
# under charts), chart name, default release name, package and program names, plus helpers that
# compute run release names and the workbench's kubernetes object name.
from __future__ import annotations

from pathlib import Path

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    WORKBENCH_OBJECT_NAME_MAX,
)
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import release_prefix

CHART_DIR = Path(__file__).resolve().parents[4] / "charts" / "miles-workbench"
CHART_NAME = "miles-workbench"

DEFAULT_RELEASE = "workbench"

PACKAGE = "miles.utils.external_utils.miles_workbench"
PROGRAM_NAME = f"python -m {PACKAGE}"


def run_release_name(
    run_id: str, deploy_component: DeployComponent = DeployComponent.ALL, deploy_instance_id: str | None = None
) -> str:
    # [FUNCTION DOC] Serialized helm release name of a run for the deploy component (default all)
    # and optional instance id. Validation errors from the release name model propagate.
    return ReleaseName(
        run_id=run_id, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id
    ).serialize()


def object_name(release: str) -> str:
    # [FUNCTION DOC] Name of the workbench stateful set for the release: the chart's release prefix
    # computed with the workbench chart name and the maximum object name length budget.
    return release_prefix(release, chart_name=CHART_NAME, budget=WORKBENCH_OBJECT_NAME_MAX)
