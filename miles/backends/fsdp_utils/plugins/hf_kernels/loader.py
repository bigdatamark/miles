# [MODULE DOC]
# Collective resolution of Hub kernels. Defines the kernel spec dataclass, the function that loads
# and validates the slot-to-spec mapping from the run arguments, and the per-run state class that
# (1) makes all ranks agree on the mapping, (2) downloads and imports each distinct repo with local
# rank zero going first on each node, (3) all-gathers each rank's per-slot status and disables any
# slot that failed or whose loaded build identity differs between ranks (or raises when strict),
# and (4) hands out resolved functions and binds them on models. Collectives use the project's gloo
# group when available, falling back to the default group; without an initialized distributed
# runtime everything degrades to single-process behavior. The kernels package is imported lazily.
"""Module-level compute kernels resolved from the Hugging Face Hub via the `kernels` package.

`HubKernels.prepare()` collectively agrees on a `slot -> HubKernelSpec` mapping and resolves each
repo once per run; `bind()` then rebinds the free functions HF modeling code looks up per forward.
Opt-in through `--kernel-backend hub`; under the default `native` nothing here imports `kernels`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.function_registry import load_function

logger = logging.getLogger(__name__)

# Attributed to miles rather than to `kernels` itself in the Hub's download telemetry.
_USER_AGENT = {"framework": "miles"}


@dataclass(frozen=True)
class HubKernelSpec:
    # [CLASS DOC]
    # Immutable (hashable) description of one Hub kernel repo: repo id, optional integer version
    # or optional revision string (at most one of the two), and the non-empty tuple of function
    # names to fetch from the loaded module. Used as part of dictionary keys and for de-duplication.
    """One Hub kernel repo plus the module-level functions miles pulls off it.

    `version` and `revision` are mutually exclusive, matching `kernels.get_kernel`: a `version`
    resolves through the repo's `vN` branch, a `revision` pins a branch, tag or commit SHA.
    """

    repo_id: str
    version: int | None = None
    revision: str | None = None
    functions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validation after construction. Raises a value error when both version and revision are
        # given, or when the function-name tuple is empty; messages are prefixed with the repo id.
        if self.version is not None and self.revision is not None:
            raise ValueError(f"{self.repo_id}: pass either version or revision to HubKernelSpec, not both")
        if not self.functions:
            raise ValueError(f"{self.repo_id}: HubKernelSpec needs at least one function name")

    def describe(self) -> str:
        # [FUNCTION DOC]
        # Human-readable label for logs: the repo id, followed by an at-sign and the revision if
        # set, or an at-sign, the letter v and the version number if a version is set, or nothing.
        pin = f"@{self.revision}" if self.revision else (f"@v{self.version}" if self.version is not None else "")
        return f"{self.repo_id}{pin}"


def load_mapping(args) -> dict[str, HubKernelSpec]:
    # [FUNCTION DOC]
    # Chooses the provider: the function loaded from args.kernel_mapping_path (via the function
    # registry, required to be synchronously loadable) if set, else the default presets provider.
    # Calls it with args (a falsy result is treated as an empty mapping) and validates every entry:
    # the value must be a kernel spec (type error otherwise), the slot must be a known slot name
    # (value error listing known ones), and the spec's functions must include every function
    # required for that slot (value error listing the missing ones, sorted). Returns the mapping.
    # The presets import is local to avoid a circular import.
    """Resolve this run's `slot -> HubKernelSpec` mapping from `--kernel-mapping-path` or the presets."""
    # Local import: presets defines specs in terms of HubKernelSpec, so this module cannot
    # import presets at the top.
    from miles.backends.fsdp_utils.plugins.hf_kernels.presets import REQUIRED_SLOT_FUNCTIONS, default_module_kernels

    if args.kernel_mapping_path:
        provider = load_function(args.kernel_mapping_path, sync_required=True)
    else:
        provider = default_module_kernels

    mapping = provider(args) or {}
    for slot, spec in mapping.items():
        if not isinstance(spec, HubKernelSpec):
            raise TypeError(
                f"kernel mapping slot {slot!r} must be a HubKernelSpec, got {type(spec).__name__}; "
                f"see miles/backends/fsdp_utils/plugins/hf_kernels/presets.py"
            )
        if slot not in REQUIRED_SLOT_FUNCTIONS:
            raise ValueError(f"unknown kernel mapping slot {slot!r}; expected one of {tuple(REQUIRED_SLOT_FUNCTIONS)}")
        missing = set(REQUIRED_SLOT_FUNCTIONS[slot]) - set(spec.functions)
        if missing:
            raise ValueError(
                f"kernel mapping slot {slot!r} must declare all required functions; missing {sorted(missing)}"
            )
    return mapping


@dataclass
class HubKernels:
    # [CLASS DOC]
    # Mutable per-run state. Fields: strict flag; the agreed slot-to-spec mapping; a cache from
    # (repo id, version, revision) to the imported module or none on failure; the error text for
    # failed modules; and the per-slot failure messages determined collectively. A default instance
    # (empty mapping) is inert: binding returns nothing and every slot resolves to none.
    """Per-run hub kernel state: the mapping every rank agreed on and the resolved modules.

    Built once per actor by `prepare()` and shared by the policy and the ref model, so each repo
    is downloaded and imported once.
    """

    strict: bool = False
    _mapping: dict[str, HubKernelSpec] = field(default_factory=dict)
    _modules: dict[tuple[str, int | None, str | None], ModuleType | None] = field(default_factory=dict)
    _module_errors: dict[tuple[str, int | None, str | None], str] = field(default_factory=dict)
    _slot_failures: dict[str, str] = field(default_factory=dict)

    @classmethod
    def prepare(cls, args) -> HubKernels:
        # [FUNCTION DOC]
        # Collective factory; all ranks must call it. Returns an inert instance unless
        # args.kernel_backend equals "hub". Otherwise: creates state with strict taken from
        # args.kernel_strict; agrees on the mapping across ranks (raises a value error if any rank
        # failed to load it or ranks differ); returns early if the mapping is empty. Then runs two
        # rounds separated by barriers: first node-local-rank-zero processes resolve (download and
        # import) every distinct spec, then all the others do (so downloads happen once per node).
        # Then all-gathers each rank's per-slot (identity, error) status; a slot fails if any rank
        # reported an error, or, absent errors, if build identities differ across ranks. Failed
        # slots get a message recorded and a warning logged (native kernel kept on every rank);
        # agreed slots are logged. If strict and any slot failed, raises a runtime error joining
        # the messages. Side effects: network download, imports, distributed collectives.
        """Collectively resolve the mapping before either model is bound. Inert unless `--kernel-backend hub`.

        Every rank participates, including ranks with an empty or invalid mapping. Local leaders
        download first; each slot must expose its functions and carry the same repo/revision/build
        identity on every rank, or every rank keeps its native kernel for that slot (non-strict) /
        raises (strict).
        """
        if args.kernel_backend != "hub":
            return cls()

        hub = cls(strict=args.kernel_strict)
        hub._mapping = _agree_on_configuration(args)
        if not hub._mapping:
            return hub

        for leader_turn in (True, False):
            if leader_turn == _is_download_leader():
                for spec in dict.fromkeys(hub._mapping.values()):
                    hub._resolve_module(spec)
            _barrier()

        outcomes = _all_gather_object({slot: hub._slot_status(spec) for slot, spec in hub._mapping.items()})
        for slot, spec in hub._mapping.items():
            failures = [f"rank {rank}: {status[slot][1]}" for rank, status in enumerate(outcomes) if status[slot][1]]
            identities = [status[slot][0] for status in outcomes]
            if not failures and any(identity != identities[0] for identity in identities[1:]):
                failures = [f"repo/revision/build differs across ranks: {identities}"]
            if failures:
                message = f"slot {slot!r} ({spec.describe()}): {'; '.join(failures)}"
                hub._slot_failures[slot] = message
                logger.warning("[hf kernels] %s; keeping the native kernel on every rank", message)
            else:
                logger.info("[hf kernels] slot %r agreed across all ranks: %s", slot, identities[0])

        if hub.strict and hub._slot_failures:
            raise RuntimeError("--kernel-strict: " + "; ".join(hub._slot_failures.values()))
        return hub

    def bind(self, model) -> dict[str, int]:
        # [FUNCTION DOC]
        # Returns an empty mapping when no kernel mapping is active. Otherwise runs the two
        # binders (GatedDeltaNet and NemotronH; imported locally to avoid a cycle), drops the
        # entries with zero modules, logs and returns a mapping from architecture name to the
        # number of modules patched. Mutates the model's modules.
        """Rebind every arch's hub-backed kernels on `model`; returns per-arch patched-module counts."""
        if not self._mapping:
            return {}

        # Local import: binders resolve slots through this module.
        from miles.backends.fsdp_utils.plugins.hf_kernels.binders import bind_gated_deltanet, bind_nemotron_h

        bound = {"gated_deltanet": bind_gated_deltanet(model, self), "nemotron_h": bind_nemotron_h(model, self)}
        bound = {arch: n for arch, n in bound.items() if n}
        if bound:
            logger.info("[hf kernels] bound module kernels: %s", bound)
        return bound

    def resolve_slot(self, slot: str) -> dict[str, Callable] | None:
        # [FUNCTION DOC]
        # Returns none if the slot is absent from the mapping or recorded as failed. Otherwise
        # looks up the cached module for the slot's spec (a key error if prepare never resolved it)
        # and returns a mapping from each declared function name to the attribute of that name on
        # the module.
        """One slot's functions, or `None` when the native kernel should stand.

        `None` covers every reason a slot can be inactive -- hub kernels off, slot not in the
        mapping, no rank-wide agreement -- so binders only branch once. A strict run never gets
        here with a failed slot: `prepare()` already raised.
        """
        spec = self._mapping.get(slot)
        if spec is None or slot in self._slot_failures:
            return None
        module = self._modules[(spec.repo_id, spec.version, spec.revision)]
        return {name: getattr(module, name) for name in spec.functions}

    def _resolve_module(self, spec: HubKernelSpec) -> ModuleType | None:
        # [FUNCTION DOC] HELPER: Memoized load of the module for a spec, keyed by repo id, version
        # and revision. On a cache hit returns the cached module (possibly none). Otherwise lazily
        # imports the kernels package and fetches the kernel with the repo id, revision, version and
        # a framework user-agent; any exception is swallowed, its type name and message stored as
        # the module error, and none cached. Side effects: network download and dynamic import.
        key = (spec.repo_id, spec.version, spec.revision)
        if key in self._modules:
            return self._modules[key]
        try:
            # Lazy: importing `kernels` must never run for a native job or break `import miles`
            # on a node with no matching build.
            from kernels import get_kernel

            module = get_kernel(spec.repo_id, revision=spec.revision, version=spec.version, user_agent=_USER_AGENT)
        except Exception as exc:
            self._module_errors[key] = f"{type(exc).__name__}: {exc}"
            module = None
        self._modules[key] = module
        return module

    def _slot_status(self, spec: HubKernelSpec) -> tuple[tuple[str, str, str] | None, str | None]:
        # [FUNCTION DOC] HELPER: Returns (identity, error) for a spec on this rank; exactly one is
        # none. Errors: the stored load error if the module failed to load; a message listing
        # declared functions that are missing or not callable on the module; the kernels package
        # lacking its loaded-kernels registry; or no registry entry for this exact module with repo
        # info (e.g. a local override), reported as unable to establish provenance. On success the
        # identity is a triple of the registry's repo id, revision and build metadata id.
        """This rank's `(build identity, error)` for one spec, gathered for the collective decision."""
        module = self._resolve_module(spec)
        if module is None:
            return None, self._module_errors[(spec.repo_id, spec.version, spec.revision)]
        missing = [name for name in spec.functions if not callable(getattr(module, name, None))]
        if missing:
            return None, f"{spec.describe()} does not expose callable {missing}"

        # The public registry identifies the module actually loaded, including moving refs and
        # version-specific builds, rather than just the requested pin.
        try:
            from kernels import get_loaded_kernels
        except ImportError as exc:
            return None, f"kernels.get_loaded_kernels unavailable: {exc}"

        for loaded in get_loaded_kernels():
            if loaded.module is module and loaded.repo_info is not None:
                return (loaded.repo_info.repo_id, loaded.repo_info.revision, loaded.metadata.id), None
        return None, f"cannot establish Hub provenance for {spec.describe()} (local overrides are unsupported)"


def _agree_on_configuration(args) -> dict[str, HubKernelSpec]:
    # [FUNCTION DOC] HELPER: Each rank loads its mapping, capturing any exception as an error
    # string (so all ranks still reach the collective). All-gathers (mapping, strict flag, error)
    # from every rank. Raises a value error listing per-rank errors if any exist, or if the mapping
    # or strict flag differ from rank zero's on any rank. Returns the local mapping (identical on
    # all ranks at that point). Collective: every rank must call it.
    mapping, error = {}, None
    try:
        mapping = load_mapping(args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    configs = _all_gather_object((mapping, args.kernel_strict, error))
    errors = [f"rank {rank}: {config[2]}" for rank, config in enumerate(configs) if config[2]]
    if errors:
        raise ValueError("invalid hub kernel mapping: " + "; ".join(errors))
    if any(config[:2] != configs[0][:2] for config in configs[1:]):
        raise ValueError("hub kernel mapping and --kernel-strict must match across all ranks")
    return mapping


def _is_download_leader() -> bool:
    # [FUNCTION DOC] HELPER: True if this process should download first: when distributed is
    # initialized, whether the LOCAL_RANK environment variable (default zero) equals zero; always
    # true when not distributed.
    # One leader per node: local rank 0 covers a node-local cache and a shared one alike.
    return int(os.environ.get("LOCAL_RANK", 0)) == 0 if _distributed() else True


def _distributed() -> bool:
    # [FUNCTION DOC] HELPER: True when the distributed library is available and a default process
    # group has been initialized.
    return dist.is_available() and dist.is_initialized()


def _collective_group():
    # [FUNCTION DOC] HELPER: Returns the project's gloo process group, or none (meaning the default
    # group) when that group is not available (the lookup raises a runtime error).
    # Standalone Gloo harnesses have no miles gloo group; the default process group serves there.
    try:
        return get_gloo_group()
    except RuntimeError:
        return None


def _all_gather_object(value) -> list:
    # [FUNCTION DOC] HELPER: Gathers one picklable object from every rank over the collective
    # group and returns the list ordered by rank; without distributed, returns a one-element list.
    if not _distributed():
        return [value]
    group = _collective_group()
    values = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(values, value, group=group)
    return values


def _barrier() -> None:
    # [FUNCTION DOC] HELPER: Blocks until all ranks in the collective group arrive; no-op without
    # distributed.
    if _distributed():
        dist.barrier(group=_collective_group())
