# [MODULE DOC] Two-process CPU (Gloo) test that HubKernels.prepare reaches the same outcome on both
# ranks. The pytest entry launches this same file as a subprocess in worker mode, which spawns two
# ranks; each rank loops over scenarios (success, leader/follower fetch failure, missing function,
# differing revision or build id, invalid / differing / empty mapping, differing strict flag),
# injects the fault on one rank via the fake kernels package from the sibling single-process test,
# and checks prepare either succeeds consistently or raises on both ranks with identical messages.
# Run directly without arguments it invokes pytest on itself; with the worker flag it is a worker.
"""Two-rank Gloo regressions for `HubKernels.prepare`; no CUDA devices or Hub requests."""

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

_SCENARIOS = (
    "success",
    "leader_failure",
    "follower_failure",
    "missing_function",
    "different_revision",
    "different_build",
    "invalid_mapping",
    "different_mapping",
    "empty_mapping",
    "different_strict",
)


def _worker(rank: int, rendezvous: str, strict: bool) -> None:
    # [FUNCTION DOC] HELPER: Per-rank body spawned by the multiprocess launcher. Inputs: rank, a
    # file path used for the file-based rendezvous, and the strict flag. Sets the local-rank
    # environment variable, initializes a two-rank Gloo group (15 second timeout) plus the project's
    # Gloo group, then for each scenario: builds args, installs fake kernels, wraps the fake fetch
    # so that rank 0 fails for the fla repo in "leader_failure" and rank 1 in "follower_failure"
    # (an OS error), and on rank 1 only deletes the chunk function ("missing_function"), changes
    # the recorded revision ("different_revision") or build id ("different_build"), replaces mapping
    # loading with one that raises ("invalid_mapping"), returns a reduced mapping
    # ("different_mapping": conv slot only; "empty_mapping": empty), or flips strict
    # ("different_strict"). Then calls prepare, capturing the message of a value or runtime error.
    # Expected to fail: configuration scenarios always, other non-success scenarios only when
    # strict; asserted per rank. When not failing, binds two fresh fake models (policy and
    # reference path) expecting two layers bound, conv slot resolved, and the chunk slot resolved
    # only in "success" (hub function bound in success, stand-in otherwise). Finally gathers errors
    # across ranks and asserts they are equal; rank 0 prints a PASS line per scenario and strict
    # value. Always destroys the process group. Side effects: environment, process group, monkeypatches.
    from tests.fast.backends.test_fsdp_hub_kernels import _all_hub_modules, _build_gdn_model, _make_args, _stub_kernels

    from miles.backends.fsdp_utils.plugins.hf_kernels import loader
    from miles.backends.fsdp_utils.plugins.hf_kernels.presets import SLOT_CAUSAL_CONV1D, SLOT_GATED_DELTA_RULE
    from miles.utils.distributed_utils import init_gloo_group

    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    init_gloo_group()
    try:
        for scenario in _SCENARIOS:
            args = _make_args(kernel_strict=strict)
            modules = _all_hub_modules()
            with pytest.MonkeyPatch.context() as patch:
                _stub_kernels(patch, modules=modules)
                fake = sys.modules["kernels"]
                original_get = fake.get_kernel
                mapping = loader.load_mapping(args)
                error = None

                def get_kernel(repo_id, *, scenario=scenario, original_get=original_get, fake=fake, **kwargs):
                    # [FUNCTION DOC] HELPER: Fault-injecting wrapper over the fake fetch for the
                    # current scenario (bound through defaults so each loop pass keeps its own).
                    # Raises an OS error for the fla repo on the failing rank; otherwise calls the
                    # original, and on rank 1 for fla mutates the result or its recorded metadata
                    # per scenario. Returns the module.
                    if repo_id == "kernels-community/fla":
                        if (scenario == "leader_failure" and rank == 0) or (
                            scenario == "follower_failure" and rank == 1
                        ):
                            raise OSError("simulated rank-local import failure")
                    result = original_get(repo_id, **kwargs)
                    if repo_id == "kernels-community/fla" and rank == 1:
                        if scenario == "missing_function":
                            del result.chunk_gated_delta_rule
                        for loaded in fake.get_loaded_kernels():
                            if loaded.module is result:
                                if scenario == "different_revision":
                                    loaded.repo_info.revision = "another-commit"
                                if scenario == "different_build":
                                    loaded.metadata.id = "another-build"
                    return result

                fake.get_kernel = get_kernel
                if rank == 1:
                    if scenario == "invalid_mapping":

                        def invalid_mapping(args):
                            # [FUNCTION DOC] HELPER: Replacement mapping loader that always raises a
                            # value error, simulating an invalid slot declaration on one rank.
                            raise ValueError("simulated invalid slot declaration")

                        patch.setattr(loader, "load_mapping", invalid_mapping)
                    elif scenario in ("different_mapping", "empty_mapping"):
                        altered = (
                            {} if scenario == "empty_mapping" else {SLOT_CAUSAL_CONV1D: mapping[SLOT_CAUSAL_CONV1D]}
                        )
                        patch.setattr(loader, "load_mapping", lambda args, altered=altered: altered)
                    elif scenario == "different_strict":
                        args.kernel_strict = not strict

                hub = None
                try:
                    hub = loader.HubKernels.prepare(args)
                except (ValueError, RuntimeError) as exc:
                    error = str(exc)

                configuration_failure = scenario in (
                    "invalid_mapping",
                    "different_mapping",
                    "empty_mapping",
                    "different_strict",
                )
                should_fail = configuration_failure or (strict and scenario != "success")
                assert (error is not None) == should_fail, (rank, scenario, strict, error)
                if not should_fail:
                    # Repeated binding exercises the policy and reference model path after consensus.
                    for _ in range(2):
                        model = _build_gdn_model()
                        assert hub.bind(model) == {"gated_deltanet": 2}
                        assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is not None
                        chosen = hub.resolve_slot(SLOT_GATED_DELTA_RULE)
                        assert (chosen is not None) == (scenario == "success")
                        if scenario == "success":
                            assert (
                                model.layers[0].chunk_gated_delta_rule
                                is modules["kernels-community/fla"].chunk_gated_delta_rule
                            )
                        else:
                            assert model.layers[0].chunk_gated_delta_rule.__name__ == "_torch_chunk_stand_in"
                errors = [None, None]
                dist.all_gather_object(errors, error)
                assert errors[0] == errors[1], (scenario, errors)
                if rank == 0:
                    print(f"PASS strict={strict} {scenario}", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strict", [False, True])
def test_hub_preparation_agrees_across_ranks(tmp_path: Path, strict: bool) -> None:
    # [FUNCTION DOC] Test (parametrized over strict): runs this file in a subprocess with the
    # worker flag, a rendezvous path inside the temp directory and the strict flag as 0/1, with the
    # repository root (three directories above this file) prepended to the python path and a 60
    # second timeout. Asserts the subprocess exits with code zero (showing its output otherwise)
    # and that its output contains the PASS line for every scenario at this strict value.
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, __file__, "--worker", str(tmp_path / "rendezvous"), str(int(strict))],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for scenario in _SCENARIOS:
        assert f"PASS strict={strict} {scenario}" in result.stdout


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        mp.spawn(_worker, args=(sys.argv[2], bool(int(sys.argv[3]))), nprocs=2, join=True)
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
