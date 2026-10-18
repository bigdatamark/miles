# [MODULE DOC]
# Tests for tests.ci.hardware, the table of CUDA stages and GPU architectures (hopper, blackwell):
# consistency of the stage table with the run-suite inventory, architecture preference
# (auto_arch), which stages a registration executes in under a set of requested architectures
# (dispatch_targets, with absorb controlling whether a test may leave its home stage), and the
# routing of a home stage to the stage serving another architecture (target_stage). Defines
# constants for three sample registrations (portable, hopper-only, blackwell-only) as pairs of home
# stage and hardware list, and architecture sets. Registers itself as a CPU test in suite
# stage-a-cpu.
"""Locks the arch table against the suite inventory it describes."""

import pytest
from tests.ci.ci_register import HWBackend, register_cpu_ci
from tests.ci.hardware import CUDA_STAGES, KNOWN_ARCHES, auto_arch, dispatch_targets, target_stage
from tests.ci.run_suite import CI_SUITES

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def test_every_cuda_suite_declares_an_arch():
    # [FUNCTION DOC] Asserts the stage table's suite names equal the run-suite module's CUDA suites,
    # so no suite exists without an architecture declaration.
    # Drift here is what let a suite name sit in test files for months with no
    # workflow job behind it.
    assert set(CUDA_STAGES) == set(CI_SUITES[HWBackend.CUDA])


def test_every_declared_arch_is_known():
    # [FUNCTION DOC] Asserts every stage's architecture belongs to the known architecture list.
    assert {stage.arch for stage in CUDA_STAGES.values()} <= set(KNOWN_ARCHES)


def test_hopper_is_the_default_arch():
    # [FUNCTION DOC] Asserts the first (most preferred) known architecture is hopper.
    # One B200 host against four-plus Hopper hosts: a test that could run on
    # either must not land on the scarce one by default.
    assert KNOWN_ARCHES[0] == "hopper"


@pytest.mark.parametrize(
    ("hardware", "expected"),
    [
        (["hopper"], "hopper"),
        (["blackwell"], "blackwell"),
        (["hopper", "blackwell"], "hopper"),
        (["blackwell", "hopper"], "hopper"),
    ],
)
def test_auto_arch_follows_preference_not_declaration_order(hardware, expected):
    # [FUNCTION DOC] Parametrized over hardware lists; the chosen architecture is the most preferred
    # one present (hopper whenever listed, regardless of order), else blackwell.
    assert auto_arch(hardware) == expected


def test_auto_arch_rejects_an_unknown_set():
    # [FUNCTION DOC] Hardware list with only ampere raises a value error saying there is no known
    # arch.
    with pytest.raises(ValueError, match="no known arch"):
        auto_arch(["ampere"])


# --- dispatch: which stages a registration executes in ----------------------

PORTABLE = ("stage-c-8-gpu-h200", ["hopper", "blackwell"])
HOPPER_ONLY = ("stage-c-4-gpu-h200", ["hopper"])
BLACKWELL_ONLY = ("stage-c-8-gpu-b200", ["blackwell"])

HOPPER = frozenset({"hopper"})
BLACKWELL = frozenset({"blackwell"})
BOTH = frozenset({"hopper", "blackwell"})


@pytest.mark.parametrize(
    ("registration", "dispatch", "absorb", "expected"),
    [
        # AUTO: everything at home, exactly once. This is the shape every run
        # without an explicit `run-on-*` takes, including nightly and weekly.
        (PORTABLE, frozenset(), False, {"stage-c-8-gpu-h200"}),
        (HOPPER_ONLY, frozenset(), False, {"stage-c-4-gpu-h200"}),
        (BLACKWELL_ONLY, frozenset(), False, {"stage-c-8-gpu-b200"}),
        # One arch requested: tests that cannot run there select nothing.
        (PORTABLE, HOPPER, True, {"stage-c-8-gpu-h200"}),
        (HOPPER_ONLY, HOPPER, True, {"stage-c-4-gpu-h200"}),
        (BLACKWELL_ONLY, HOPPER, True, set()),
        (PORTABLE, BLACKWELL, True, {"stage-c-8-gpu-b200"}),
        (HOPPER_ONLY, BLACKWELL, True, set()),
        (BLACKWELL_ONLY, BLACKWELL, True, {"stage-c-8-gpu-b200"}),
        # Both arches: a portable test runs twice, once per generation.
        (PORTABLE, BOTH, True, {"stage-c-8-gpu-h200", "stage-c-8-gpu-b200"}),
        (HOPPER_ONLY, BOTH, True, {"stage-c-4-gpu-h200"}),
        (BLACKWELL_ONLY, BOTH, True, {"stage-c-8-gpu-b200"}),
        # `run-ci-blackwell-only`: an arch without permission to leave home,
        # which is what makes it select only the Blackwell-exclusive tests.
        (PORTABLE, BLACKWELL, False, set()),
        (HOPPER_ONLY, BLACKWELL, False, set()),
        (BLACKWELL_ONLY, BLACKWELL, False, {"stage-c-8-gpu-b200"}),
    ],
)
def test_dispatch_targets(registration, dispatch, absorb, expected):
    # [FUNCTION DOC] Parametrized table of (registration, requested arches, absorb flag, expected set
    # of stages). With no requested arch, the test runs only at home. With one arch requested and
    # absorb allowed, tests that support it run on that arch's stage (hopper-only tests do not move
    # to blackwell, blackwell-only do not move to hopper); both arches run a portable test twice.
    # With absorb disallowed (the blackwell-only label), only tests homed on the requested arch run.
    # Asserts the function returns exactly the expected set.
    home, hardware = registration
    assert dispatch_targets(home, hardware, dispatch_arches=dispatch, absorb=absorb) == expected


def test_absorb_false_never_leaves_the_home_stage():
    # [FUNCTION DOC] Exhaustive property check: for every stage, every hardware list whose automatic
    # architecture matches the stage's architecture (others are excluded by the home-stage
    # invariant), and every requested-arch set (none, hopper, blackwell, both), dispatch with absorb
    # disallowed returns a subset of the home stage.
    # The safety property every non-`run-on-*` run relies on.
    for home, stage in CUDA_STAGES.items():
        for hardware in (["hopper"], ["blackwell"], ["hopper", "blackwell"]):
            if stage.arch != auto_arch(hardware):
                continue  # excluded by the home-stage invariant
            for dispatch in (frozenset(), HOPPER, BLACKWELL, BOTH):
                targets = dispatch_targets(home, hardware, dispatch_arches=dispatch, absorb=False)
                assert targets <= {home}, (home, hardware, dispatch)


@pytest.mark.parametrize(
    ("home", "arch", "expected"),
    [
        ("stage-c-8-gpu-h200", "hopper", "stage-c-8-gpu-h200"),
        # Both 8-GPU Hopper stages collapse onto the one Blackwell stage: the
        # h100/h200 split is a memory tier that Blackwell does not have.
        ("stage-c-8-gpu-h100", "blackwell", "stage-c-8-gpu-b200"),
        ("stage-c-8-gpu-h200", "blackwell", "stage-c-8-gpu-b200"),
        # Narrower stages route up; a test brings its own GPU budget.
        ("stage-b-2-gpu-h200", "blackwell", "stage-c-8-gpu-b200"),
        ("stage-c-4-gpu-h200", "blackwell", "stage-c-8-gpu-b200"),
    ],
)
def test_target_stage(home, arch, expected):
    # [FUNCTION DOC] Parametrized routing table: a hopper request for an h200 home stays; both
    # 8-GPU Hopper homes and the narrower stages route to the single 8-GPU blackwell stage. Asserts
    # the target stage equals the expected stage name.
    assert target_stage(home, arch) == expected


def test_target_stage_is_none_when_no_stage_is_wide_enough():
    # [FUNCTION DOC] Routing to an architecture with no stages returns none.
    assert target_stage("stage-c-8-gpu-h200", "nonexistent-arch") is None


def test_blackwell_work_never_reaches_a_hopper_stage():
    # [FUNCTION DOC] Asserts that routing the blackwell 8-GPU stage to hopper answers with the
    # 8-GPU h100 stage (the routing rule alone would allow it), but that dispatching a blackwell-only
    # registration for a hopper request with absorb allowed yields an empty set, because the
    # registration's own architectures do not intersect the request.
    """`target_stage` alone would route it; `dispatch_targets` never asks.

    Both 8-GPU Hopper stages are wide enough to hold `stage-c-8-gpu-b200`'s
    work, so the routing rule does answer. The guarantee comes one level up:
    the home-stage invariant makes a Blackwell-homed test Blackwell-exclusive,
    so intersecting its arches with a Hopper request yields nothing and routing
    is never consulted.
    """
    assert target_stage("stage-c-8-gpu-b200", "hopper") == "stage-c-8-gpu-h100"
    assert dispatch_targets("stage-c-8-gpu-b200", ["blackwell"], dispatch_arches=HOPPER, absorb=True) == set()
