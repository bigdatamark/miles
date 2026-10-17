# [MODULE DOC] Static taxonomy of CUDA CI hardware: the ordered tuple of known GPU generations
# (hopper first, which is the auto-dispatch preference), the table of CUDA stages (arch, GPU count
# and runner labels) that is the single source for suite names, and pure functions that pick a
# test's default arch, the smallest stage on another arch that can hold a test, and the full set of
# stages a registration runs on under a dispatch policy. Standard library only.
"""GPU generations a CUDA test can run on, and what each CUDA stage provides.

* An *arch* is a GPU generation, not a SKU: `h100` and `h200` differ in memory,
  which the suite name already encodes, not in the kernels they support.
* `register_cuda_ci(hardware=[...])` declares the arches a test's kernel and
  precision paths support. Which stage runs it is a separate question, answered
  by `dispatch_targets` from the test's home suite plus the run's dispatch policy.
* `CUDA_STAGES` is the single source of truth for the CUDA stage taxonomy;
  `run_suite.CI_SUITES` and `file_run.CUDA_SUITE_RUNS_ON` derive from it.
* Stdlib only. `tests/ci/file_run.py` imports this chain on a bare hosted runner
  before any dependency install, so it cannot reach into `miles.*`.
"""

from dataclasses import dataclass

__all__ = [
    "KNOWN_ARCHES",
    "CUDA_STAGES",
    "CudaStage",
    "auto_arch",
    "dispatch_targets",
    "target_stage",
]

# Order is the auto-dispatch preference: a test supporting several arches runs
# on the first one listed unless a PR asks otherwise. Hopper leads because one
# Blackwell host against four-plus Hopper hosts makes any balanced split
# saturate the Blackwell queue.
KNOWN_ARCHES: tuple[str, ...] = ("hopper", "blackwell")


@dataclass(frozen=True)
class CudaStage:
    # [CLASS DOC] Immutable description of one CUDA stage: its GPU arch, GPU count, and the tuple
    # of runner labels used to select a machine.
    arch: str
    num_gpus: int
    runs_on: tuple[str, ...]


CUDA_STAGES: dict[str, CudaStage] = {
    "stage-b-2-gpu-h200": CudaStage("hopper", 2, ("h200", "2gpu")),
    "stage-c-8-gpu-h100": CudaStage("hopper", 8, ("h100", "8gpu")),
    "stage-c-8-gpu-h200": CudaStage("hopper", 8, ("h200", "8gpu")),
    "stage-c-4-gpu-h200": CudaStage("hopper", 4, ("h200", "4gpu")),
    "stage-c-2-gpu-h200": CudaStage("hopper", 2, ("h200", "2gpu")),
    "stage-c-8-gpu-b200": CudaStage("blackwell", 8, ("b200", "8gpu")),
}


def auto_arch(hardware: list[str]) -> str:
    # [FUNCTION DOC] Returns the first known arch (in preference order) present in the given
    # list. Raises a value error if none of the entries is a known arch (including an empty list).
    """The arch a test runs on when nothing asks for a different one."""
    supported = set(hardware)
    for arch in KNOWN_ARCHES:
        if arch in supported:
            return arch
    raise ValueError(f"no known arch in {sorted(supported)}")


def target_stage(home_suite: str, arch: str) -> str | None:
    # [FUNCTION DOC] Returns the home suite itself when the requested arch is the home arch;
    # otherwise the stage on the requested arch with at least as many GPUs as the home stage and the
    # fewest GPUs (ties broken by smaller name), or none if there is none. Raises a key error for
    # an unknown home suite.
    """The stage on `arch` that runs a test homed at `home_suite`.

    On the test's own arch that is its home stage. On another arch it is the
    smallest stage with enough GPUs: a test declares its own budget through
    `ray start --num-gpus` / `torchrun --nproc-per-node` rather than reading the
    devices it can see, so a larger stage just leaves the surplus idle. None
    when that arch has no stage big enough.

    Derived rather than tabulated. A hand-written home -> destination map would
    encode a fleet shape that does not exist yet; this rule gives today's
    degenerate answer (one Blackwell stage absorbs everything) and a 1:1 mirror
    once the Blackwell fleet is partitioned, from the same three lines.
    """
    home = CUDA_STAGES[home_suite]
    if home.arch == arch:
        return home_suite
    fits = [name for name, stage in CUDA_STAGES.items() if stage.arch == arch and stage.num_gpus >= home.num_gpus]
    # Tie-break on the name so two stages of equal width choose deterministically.
    return min(fits, key=lambda name: (CUDA_STAGES[name].num_gpus, name), default=None)


def dispatch_targets(
    home_suite: str,
    hardware: list[str],
    *,
    dispatch_arches: frozenset[str],
    absorb: bool,
) -> set[str]:
    # [FUNCTION DOC] Computes the set of stages a registration runs on. Arches considered: the
    # intersection of the supported arches and the requested dispatch arches, or, when no arches
    # are requested, just the auto arch of the supported ones (value error if none known). For each
    # arch: the home suite if it is the home arch; otherwise, only when absorb is true, the target
    # stage on that arch if one exists. Returns a possibly empty set.
    """Every CUDA stage that runs this registration under one dispatch policy.

    Empty `dispatch_arches` is AUTO: the test runs once, on `auto_arch`, which
    the home-stage invariant makes equal to its home stage's arch. `absorb` is
    what allows a test to leave its home stage at all; without it a requested
    arch that is not the test's own simply selects nothing.
    """
    supported = set(hardware)
    arches = (supported & dispatch_arches) if dispatch_arches else {auto_arch(supported)}

    targets: set[str] = set()
    for arch in arches:
        if CUDA_STAGES[home_suite].arch == arch:
            targets.add(home_suite)
        elif absorb:
            destination = target_stage(home_suite, arch)
            if destination is not None:
                targets.add(destination)
    return targets
