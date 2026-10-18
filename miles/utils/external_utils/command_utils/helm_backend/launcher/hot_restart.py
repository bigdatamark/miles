# [MODULE DOC] Planning of hot restarts: restarting the orchestration script together with the
# rollout executor of an already installed run by stamping a restart time onto their stateful sets
# (so the pods are rolled) while permitting only those objects to differ from the installed
# manifest. Without a hot restart request, the stamp found on the installed manifest is carried
# over so a later plain upgrade does not undo or re-trigger it.
from __future__ import annotations

import datetime

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    STATEFUL_SET_KIND,
    Manifest,
    ManifestObjectKey,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.types import DeployComponent, HotRestartComponent

_COMPONENT_NAMES = {
    HotRestartComponent.ORCHESTRATION: naming.ORCHESTRATOR_COMPONENT,
    HotRestartComponent.ROLLOUT_EXECUTOR: ROLLOUT_EXECUTOR_POOL_ID,
}


class HotRestartPlan(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable plan: the restart timestamp text (none if no restart stamp), the set of
    # component names that carry the stamp, and the set of manifest object keys that may differ
    # from the installed manifest (all default empty or none).
    restart_at: str | None = None
    stamped_components: frozenset[str] = frozenset()
    allow_diff_object_keys: frozenset[ManifestObjectKey] = frozenset()


def plan_hot_restart(
    *,
    components: list[HotRestartComponent],
    deploy_component: DeployComponent,
    release: str,
    installed_manifest: Manifest | None,
) -> HotRestartPlan:
    # [FUNCTION DOC] Keyword-only. With no components requested, returns the plan that carries the
    # installed stamp (see the carry helper). Otherwise asserts: the request names exactly the
    # orchestration script and the rollout executor (no subset); the release being deployed carries
    # the orchestration script; and a release is already installed (a hot restart takes over a live
    # run). Then returns a plan whose restart time is the current UTC time in ISO format with
    # microseconds, whose stamped components are the stateful set names of the requested
    # components, and whose allowed-difference keys are the stateful set object keys of those
    # components in the release.
    if not components:
        return _carry_installed_stamp(installed_manifest, release=release)

    assert set(components) == set(HotRestartComponent), (
        f"--hot-restart names {[one.value for one in components]}, and a hot restart currently only supports "
        f"restarting the orchestration script together with the rollout executor: the new script cannot drive the "
        f"executor its predecessor initialized, and an executor replaced under a live script kills its run"
    )
    assert deploy_component.deploys_orchestration_script(), (
        f"--hot-restart replaces the orchestration script and the rollout executor, and a --deploy-component "
        f"{deploy_component.value} release deploys neither of them; run it against the release that carries them "
        f"({DeployComponent.ALL.value} or {DeployComponent.PRIMARY.value})"
    )
    assert installed_manifest is not None, (
        f"--hot-restart takes over the trainers and the inference side of a run that is already up, and no release "
        f"{release!r} is installed; installing it here would build a second orchestration script beside the one "
        f"still driving those trainers, so launch this run normally instead"
    )

    names = [_COMPONENT_NAMES[component] for component in components]
    return HotRestartPlan(
        restart_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds"),
        stamped_components=frozenset(names),
        allow_diff_object_keys=frozenset(_stateful_set_key(release, name) for name in names),
    )


def compute_orchestrator_object_key(release: str) -> ManifestObjectKey:
    # [FUNCTION DOC] The manifest object key (stateful set kind plus component name) of the
    # release's orchestrator.
    return _stateful_set_key(release, naming.ORCHESTRATOR_COMPONENT)


def _stateful_set_key(release: str, component: str) -> ManifestObjectKey:
    # [FUNCTION DOC] HELPER: The object key of the component's stateful set: kind stateful set and
    # the component object name derived from the release and component.
    return ManifestObjectKey(kind=STATEFUL_SET_KIND, name=naming.component_name(release, component))


def _carry_installed_stamp(installed_manifest: Manifest | None, *, release: str) -> HotRestartPlan:
    # [FUNCTION DOC] HELPER: Returns an empty plan if nothing is installed or the installed
    # orchestrator has no restart stamp. Otherwise reads the restart stamp of both components'
    # objects from the installed manifest and returns a plan with the orchestrator's stamp and the
    # set of components whose stamp equals it; no differences are allowed.
    if installed_manifest is None:
        return HotRestartPlan()

    stamps = {
        name: installed_manifest.restart_at(object_name=naming.component_name(release, name))
        for name in _COMPONENT_NAMES.values()
    }
    if (stamp := stamps[naming.ORCHESTRATOR_COMPONENT]) is None:
        return HotRestartPlan()

    return HotRestartPlan(
        restart_at=stamp,
        stamped_components=frozenset(name for name, found in stamps.items() if found == stamp),
    )
