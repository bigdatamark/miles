# [MODULE DOC]
# Shared pieces for building helm values: the three chart section names and the mapping from a pool
# category to its section, the launch plan model, mooncake store helpers (plan extraction from
# parsed arguments, cluster address rewriting, object names, chart section), infra values loading
# from a chart plus override files, and namespace variable resolution in paths.
from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.external_utils.command_utils.common import (
    MOONCAKE_BACKEND_NAME,
    MOONCAKE_INIT_KWARGS_FLAG,
    ArgvManipulator,
)
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    InfraValues,
    MooncakeSection,
)
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.object_store_config import MOONCAKE_MASTER_ADDRESS_KEY
from miles.utils.pydantic_utils import FrozenStrictBaseModel

STATIC_WORKERS_SECTION = "staticWorkers"
INFERENCE_ENGINES_SECTION = "inferenceEngines"
TRAINER_ENGINES_SECTION = "trainerEngines"

SECTION_OF_CATEGORY = {
    None: STATIC_WORKERS_SECTION,
    POOL_CATEGORY_INFERENCE_ENGINE: INFERENCE_ENGINES_SECTION,
    POOL_CATEGORY_TRAINER_ENGINE: TRAINER_ENGINES_SECTION,
}

_MOONCAKE_COMPONENT = "mooncake-master"
_INFRA_KEY = "infra"
_VALUES_FILE_NAME = "values.yaml"
_NAMESPACE_VARIABLE = "${NAMESPACE}"
_PATH_VARIABLE = re.compile(r"\$\{[^}]*\}")


class MooncakePlan(FrozenStrictBaseModel):
    # [CLASS DOC] Mooncake store plan: the user's init keyword arguments (mapping) and the port
    # parsed from the configured master address.
    init_kwargs: dict[str, Any]
    port: int


class LaunchPlan(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable description of one helm launch: run id, release and namespace, path of
    # the state file the orchestrator writes, orchestrator command, worker argument vector,
    # environment, optional launch record path, colocate flag, optional mooncake plan, per-role
    # prepare commands, extra raw manifests, an optional restart stamp and the set of component names
    # that receive the stamp.
    run_id: str
    release: str
    namespace: str
    state_file: str
    orchestrator_command: list[str]
    worker_argv: list[str]
    env: dict[str, str] = {}
    launch_record: str | None = None
    colocate: bool = False
    mooncake_plan: MooncakePlan | None = None
    prepare_cmd: dict[str, str] = {}
    extra_manifests: list[str] = []
    restart_at: str | None = None
    stamped_components: frozenset[str] = frozenset()

    def rendered_restart_at(self, component: str) -> str | None:
        # [FUNCTION DOC] Returns the restart stamp if the component is among the stamped components,
        # otherwise none.
        if component not in self.stamped_components:
            return None
        return self.restart_at


class MooncakeInfo:
    # [CLASS DOC] Namespace of static helpers around the mooncake store (no state).
    @staticmethod
    def plan_of_args(args: Namespace) -> MooncakePlan | None:
        # [FUNCTION DOC] Derives a mooncake plan from parsed arguments. Returns none unless the
        # object store backend is mooncake. Otherwise requires non-empty store init keyword arguments
        # and a master address string containing a colon; the port is the integer after the last
        # colon. Raises assertion errors when the init arguments are missing or the address has no
        # port, and a value error if the port is not an integer.
        if args.object_store_backend != MOONCAKE_BACKEND_NAME:
            return None

        init_kwargs = args.mooncake_store_init_kwargs
        assert (
            init_kwargs
        ), f"{MOONCAKE_INIT_KWARGS_FLAG} is missing, so the mooncake master address cannot be rewritten"
        address = init_kwargs.get(MOONCAKE_MASTER_ADDRESS_KEY)
        assert (
            isinstance(address, str) and ":" in address
        ), f"{MOONCAKE_MASTER_ADDRESS_KEY} is {address!r} and carries no port, so the in-cluster address cannot be built"
        return MooncakePlan(init_kwargs=init_kwargs, port=int(address.rsplit(":", 1)[1]))

    @staticmethod
    def with_cluster_master(train_argv: list[str], *, plan: MooncakePlan | None, host: str) -> list[str]:
        # [FUNCTION DOC] Returns the training argument vector with the mooncake init keyword flag
        # set to a json rendering of the cluster init arguments (master address rewritten to the
        # given host and plan port). Unchanged when there is no plan.
        if plan is None:
            return train_argv

        rendered = json.dumps(MooncakeInfo.cluster_init_kwargs(plan, host=host))
        return ArgvManipulator.set(train_argv, MOONCAKE_INIT_KWARGS_FLAG, rendered)

    @staticmethod
    def cluster_init_kwargs(plan: MooncakePlan, *, host: str) -> dict[str, Any]:
        # [FUNCTION DOC] Copy of the plan's init arguments with the master address replaced by the
        # host and the plan's port joined by a colon.
        return {**plan.init_kwargs, MOONCAKE_MASTER_ADDRESS_KEY: f"{host}:{plan.port}"}

    @staticmethod
    def master_service_host(release: str, namespace: str) -> str:
        # [FUNCTION DOC] Fully qualified in-cluster service host name of the mooncake master for the
        # release in the namespace.
        return RunNames.service_fqdn(name=MooncakeInfo.master_object_name(release), namespace=namespace)

    @staticmethod
    def master_object_name(release: str) -> str:
        # [FUNCTION DOC] Kubernetes object name of the mooncake master component for the release.
        return naming.component_name(release, _MOONCAKE_COMPONENT)

    @staticmethod
    def section(plan: MooncakePlan) -> MooncakeSection:
        # [FUNCTION DOC] Chart section enabling mooncake with the plan's port as rpc port.
        return MooncakeSection(enabled=True, rpc_port=plan.port)


class InfraInfo:
    # [CLASS DOC] Namespace of static helpers reading the infra section of helm values.
    @staticmethod
    def load(chart: str | Path, helm_values_files: list[str]) -> InfraValues:
        # [FUNCTION DOC] Loads the chart's default values merged with the override files in order,
        # takes the infra key and validates it into infra values. Reads files; raises validation
        # errors for an invalid or missing infra section.
        return InfraValues.model_validate(_load_helm_values(chart, helm_values_files).get(_INFRA_KEY))

    @staticmethod
    def shared_root(infra: InfraValues, *, namespace: str) -> str:
        # [FUNCTION DOC] Returns the runs root path with the namespace variable substituted and
        # trailing slashes removed. Raises an assertion error if the runs root is unset or empty, and
        # a value error for unknown variables.
        runs_root = infra.paths.runs_root if infra.paths is not None else None
        assert runs_root, (
            "infra.paths.runsRoot is unset, so the launcher has no container path to write a run's directory to; "
            "set it to an absolute path under one of the infra.volumes mounts"
        )
        return resolve_helm_yaml_entry(runs_root, namespace).rstrip("/")


def resolve_helm_yaml_entry(raw_value: str, namespace: str) -> str:
    # [FUNCTION DOC] Replaces the namespace variable (dollar and braces form) in a values entry by
    # the namespace. Any other braced variable reference raises a value error, since leaving it in
    # place would make every namespace share one literal directory.
    for reference in _PATH_VARIABLE.findall(raw_value):
        if reference != _NAMESPACE_VARIABLE:
            raise ValueError(
                f"{raw_value} names the unknown variable {reference}: {_NAMESPACE_VARIABLE} is the only variable "
                "a path may name, and an unknown one left in place would give every namespace the same literal "
                "directory instead of one of its own"
            )
    return raw_value.replace(_NAMESPACE_VARIABLE, namespace)


def _load_helm_values(chart: str | Path, values_files: list[str] | list[Path]) -> Any:
    # [FUNCTION DOC] HELPER: Reads the default values file of the chart directory and merges each
    # override file over it in order. Reads files from disk; yaml parse errors propagate.
    def load(values_file: Path) -> Any:
        # [FUNCTION DOC] HELPER: Parses a yaml file; an empty file yields an empty mapping.
        return yaml.safe_load(values_file.read_text()) or {}

    def merge(base: Any, override: Any) -> Any:
        # [FUNCTION DOC] HELPER: Recursive helm-style merge. If either side is not a mapping, the
        # override wins unless it is none (then the base is kept). For two mappings: result starts
        # as a copy of base; an override key with a none value deletes that key, other keys are
        # merged recursively. Lists are replaced, not concatenated.
        if not isinstance(override, dict) or not isinstance(base, dict):
            return base if override is None else override

        result = dict(base)
        for key, value in override.items():
            if value is None:
                result.pop(key, None)
            else:
                result[key] = merge(result.get(key), value)
        return result

    values = load(Path(chart) / _VALUES_FILE_NAME)
    for values_file in values_files:
        values = merge(values, load(Path(values_file)))
    return values
