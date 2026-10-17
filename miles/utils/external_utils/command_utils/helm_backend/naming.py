# [MODULE DOC]
# Naming and file layout conventions of helm-backed runs. Release names have the form chart name,
# run id, deploy component and optional deploy instance id joined by dashes, and must fit helm's
# 53 character limit (hence the maximum run id length derived from the longest component name).
# Also defines names of run-level kubernetes objects and the on-disk layout of a run directory
# (state, values and launch record files, each tagged with a unique launch token).
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

from pydantic import model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.types import DeployComponent, PlatformAccess
from miles.utils.workers.worker_provider.kubernetes.helm.naming import CHART_NAME, component_name

ORCHESTRATOR_COMPONENT = "orchestrator"

_HELM_RELEASE_NAME_MAX = 53
_LONGEST_COMPONENT_VALUE = max((component.value for component in DeployComponent), key=len)
_LONGEST_COMPONENT_SUFFIX = len(f"-{_LONGEST_COMPONENT_VALUE}")
RUN_ID_MAX_LENGTH = _HELM_RELEASE_NAME_MAX - len(f"{CHART_NAME}-") - _LONGEST_COMPONENT_SUFFIX
_COMPONENT_VALUES = frozenset(component.value for component in DeployComponent)

_UNINSTALL_COMPONENT = "uninstall"
_UNINSTALL_MANIFEST_COMPONENT = "uninstall-manifest"

_RUNS_DIR_NAME = "miles-runs"
_STATE_DIR_NAME = "state"
_VALUES_DIR_NAME = "values"
_RECORDS_DIR_NAME = "launches"
_RECORD_FILE_GLOB = "launch-*.json"
_RECORDED_STATE_FILE_KEY = "state_file"
_SUPERSEDED_MARKER_SUFFIX = ".superseded"


def platform_account_name(*, release: str, access: PlatformAccess) -> str:
    # [FUNCTION DOC] Name of the platform service account for the release and access level (the
    # component name with a "platform-" prefixed access value). Raises an assertion error for the
    # no-access level, since such workers use no dedicated account.
    assert access is not PlatformAccess.NONE, "a worker that never reaches the platform runs on no account of its own"
    return component_name(release, f"platform-{access.value}")


class ReleaseName(FrozenStrictBaseModel):
    # [CLASS DOC] Structured helm release name: run id, deploy component and optional deploy
    # instance id (none for a component deployed once). Validated on construction to fit the
    # release length limit and to be unambiguously parseable back.
    run_id: str
    deploy_component: DeployComponent
    deploy_instance_id: str | None

    @model_validator(mode="after")
    def _fits_helm_release(self) -> ReleaseName:
        # [FUNCTION DOC] HELPER: After-validator. Asserts: run id within the maximum length; if an
        # instance id is present, it is non-empty, fits the remaining character budget, and none of
        # its dash-separated tokens equals a component name (would break parsing); and the full
        # serialized name is within the 53 character limit. Returns the model.
        assert (
            len(self.run_id) <= RUN_ID_MAX_LENGTH
        ), f"run_id {self.run_id!r} is {len(self.run_id)} characters, at most {RUN_ID_MAX_LENGTH}"
        if self.deploy_instance_id is not None:
            assert self.deploy_instance_id, "deploy_instance_id is empty; a component deployed once carries None"
            budget = _deploy_instance_id_budget(run_id=self.run_id)
            assert len(self.deploy_instance_id) <= budget, (
                f"deploy_instance_id {self.deploy_instance_id!r} is {len(self.deploy_instance_id)} characters, and "
                f"run_id {self.run_id!r} leaves {budget} for it in the release name"
            )
            intersected = sorted(_COMPONENT_VALUES.intersection(self.deploy_instance_id.split("-")))
            assert not intersected, (
                f"--deploy-instance-id {self.deploy_instance_id!r} carries the component name(s) {intersected}, "
                f"which a release name could not be parsed back apart on"
            )
        assert (
            len(name := self.serialize()) <= _HELM_RELEASE_NAME_MAX
        ), f"release {name!r} is {len(name)} characters, at most {_HELM_RELEASE_NAME_MAX}"
        return self

    def serialize(self) -> str:
        # [FUNCTION DOC] Joins chart name, run id, component value and (if present) instance id with
        # dashes.
        parts = [CHART_NAME, self.run_id, self.deploy_component.value]
        if self.deploy_instance_id is not None:
            parts.append(self.deploy_instance_id)
        return "-".join(parts)

    @staticmethod
    def run_prefix(*, run_id: str) -> str:
        # [FUNCTION DOC] Prefix shared by all releases of a run: chart name, run id and a trailing
        # dash.
        return f"{CHART_NAME}-{run_id}-"

    @classmethod
    def parse(cls, release: str) -> ReleaseName | None:
        # [FUNCTION DOC] Inverse of serialize. Returns none if the name lacks the chart prefix.
        # Splits the rest by dashes and picks the LAST token that is a component name (run ids may
        # contain component-like words); if none is found or it is the first token (empty run id),
        # returns none. Run id is the tokens before it, instance id the tokens after it (none if
        # empty). Construction validation errors (assertions) may propagate for over-long names.
        if not release.startswith(f"{CHART_NAME}-"):
            return None

        tokens = release.removeprefix(f"{CHART_NAME}-").split("-")
        index = max((i for i, token in enumerate(tokens) if token in _COMPONENT_VALUES), default=0)
        if index == 0:
            return None

        return cls(
            run_id="-".join(tokens[:index]),
            deploy_component=DeployComponent(tokens[index]),
            deploy_instance_id="-".join(tokens[index + 1 :]) or None,
        )


def _deploy_instance_id_budget(*, run_id: str) -> int:
    # [FUNCTION DOC] HELPER: Characters left for the instance id in a release name given the run id,
    # assuming the longest component name.
    return _HELM_RELEASE_NAME_MAX - len(f"{CHART_NAME}-{run_id}-{_LONGEST_COMPONENT_VALUE}-")


class RunNames:
    # [CLASS DOC] Namespace of static helpers naming run-level kubernetes objects.
    @staticmethod
    def service_fqdn(*, name: str, namespace: str) -> str:
        # [FUNCTION DOC] In-cluster service host: name, namespace and the svc suffix, dot-joined.
        return f"{name}.{namespace}.svc"

    @staticmethod
    def orchestrator_object(*, release: str) -> str:
        # [FUNCTION DOC] Object name of the orchestrator component for the release.
        return component_name(release, ORCHESTRATOR_COMPONENT)

    @staticmethod
    def orchestrator_host(*, release: str, namespace: str) -> str:
        # [FUNCTION DOC] In-cluster service host of the orchestrator.
        return RunNames.service_fqdn(name=RunNames.orchestrator_object(release=release), namespace=namespace)

    @staticmethod
    def uninstall_job(*, release: str) -> str:
        # [FUNCTION DOC] Object name of the uninstall job for the release.
        return component_name(release, _UNINSTALL_COMPONENT)

    @staticmethod
    def uninstall_manifest(*, release: str) -> str:
        # [FUNCTION DOC] Object name used for the uninstall manifest of the release.
        return component_name(release, _UNINSTALL_MANIFEST_COMPONENT)


class RunFiles:
    # [CLASS DOC] Namespace of static helpers for the files of a run directory. Files that are
    # created per launch embed a unique launch token (timestamp plus random number), so lexical
    # ordering of names follows launch order. Layout under the run directory: values, state and
    # launches subdirectories. The helpers only compute paths; they do not create files, except
    # that the latest state file helper reads launch records.
    @staticmethod
    def run_dir(*, shared_root: str | Path, run_id: str) -> Path:
        # [FUNCTION DOC] Run directory: shared root, then the runs directory name, then run id.
        return Path(shared_root) / _RUNS_DIR_NAME / run_id

    @staticmethod
    def new_values_file(*, run_directory: str | Path) -> Path:
        # [FUNCTION DOC] Fresh values file path (yaml) under the values subdirectory.
        return Path(run_directory) / _VALUES_DIR_NAME / f"values-{_new_launch_token()}.yaml"

    @staticmethod
    def new_state_file(*, run_directory: str | Path) -> Path:
        # [FUNCTION DOC] Fresh orchestrator state file path under the state subdirectory.
        return _orchestrator_state_path(run_directory, _new_launch_token())

    @staticmethod
    def new_record_file(*, run_directory: str | Path) -> Path:
        # [FUNCTION DOC] Fresh launch record file path (json) under the launches subdirectory.
        return Path(run_directory) / _RECORDS_DIR_NAME / f"launch-{_new_launch_token()}.json"

    @staticmethod
    def superseded_marker(*, state_file: str | Path) -> Path:
        # [FUNCTION DOC] Path of the marker next to a state file, signalling that a later launch
        # took the run over (state file path plus a fixed suffix).
        return Path(f"{state_file}{_SUPERSEDED_MARKER_SUFFIX}")

    @staticmethod
    def latest_state_file(*, run_directory: str | Path) -> Path | None:
        # [FUNCTION DOC] Reads the lexically last launch record in the launches subdirectory and
        # returns the state file path it names; none if there are no records or the recorded name is
        # empty. Raises on unreadable or malformed json or a missing key.
        """The newest launch's state file, written or not; launches sort by the token in their name."""
        recorded = sorted((Path(run_directory) / _RECORDS_DIR_NAME).glob(_RECORD_FILE_GLOB))
        if not recorded:
            return None
        named = json.loads(recorded[-1].read_text())[_RECORDED_STATE_FILE_KEY]
        return Path(named) if named else None


def _orchestrator_state_path(run_directory: str | Path, launch_token: str) -> Path:
    # [FUNCTION DOC] HELPER: State file path inside the state subdirectory named by the token.
    return Path(run_directory) / _STATE_DIR_NAME / f"orchestrator-{launch_token}.state"


def _new_launch_token() -> str:
    # [FUNCTION DOC] HELPER: Unique, time-sortable token: current local time as two digit year,
    # month, day, hour, minute, second and microseconds (dash separated), then a random integer of
    # up to six digits zero padded. Uses the wall clock and a fresh random generator.
    return f"{datetime.now().strftime('%y%m%d-%H%M%S-%f')}-{random.Random().randint(0, 999999):06d}"
