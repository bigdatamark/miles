# [MODULE DOC] CPU unit tests for the tensor backuper in its main-cast variant: model parameters
# are not copied to host memory; instead the high-precision main copies are re-cast into the
# parameters on restore (through the optimizer's copy routine and a parameter all-gather), while
# extra tensors (not rematerializable) are backed up as pinned host copies. Fake optimizer, model
# chunk and a setup class stand in for the real training stack. Registers as a stage-a CPU CI
# test with a 60 second estimate.
from types import SimpleNamespace

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from miles.utils.tensor_backper import MainCastContext, TensorBackuper, _TensorBackuperMainCast


@pytest.fixture(autouse=True)
def _cpu_fallback(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture. When CUDA is available it does nothing. Otherwise it
    # patches (undone at teardown) the CUDA synchronize function to a no-op and the empty-like
    # tensor factory to a wrapper that drops the pin-memory keyword, so pinned-memory backups work
    # on CPU-only machines.
    if torch.cuda.is_available():
        yield
        return
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    real_empty_like = torch.empty_like

    def empty_like_no_pin(tensor, **kwargs):
        # [FUNCTION DOC] HELPER: Removes the pin-memory keyword if present and delegates to the
        # real empty-like factory; returns its tensor.
        kwargs.pop("pin_memory", None)
        return real_empty_like(tensor, **kwargs)

    monkeypatch.setattr(torch, "empty_like", empty_like_no_pin)
    yield


class _FakeOptimizer:
    # [CLASS DOC] HELPER: Fake optimizer holding named main (float32) tensors for the owned
    # shards, a staging buffer mapping and per-name shard slices. Its copy routine casts each main
    # tensor to bfloat16 and writes it into the owned slice of the staging buffer.
    """Like _copy_main_params_to_model_params: writes only this rank's owned shard."""

    def __init__(self, mains, staging, shards):
        # [FUNCTION DOC] HELPER: Stores the three mappings (main tensors, staging buffers, shard
        # slices), all keyed by parameter name, without copying.
        self._mains = mains
        self._staging = staging
        self._shards = shards

    def _copy_main_params_to_model_params(self):
        # [FUNCTION DOC] HELPER: For each named main tensor, converts to bfloat16 and assigns it
        # into the staging buffer at the name's shard slice (mutates staging in place).
        for name, main in self._mains.items():
            self._staging[name][self._shards[name]] = main.to(torch.bfloat16)


class _FakeModelChunk:
    # [CLASS DOC] HELPER: Fake model chunk that simulates the parameter all-gather by copying
    # whole staging buffers into the live parameters.
    """start_param_sync = param all-gather: staging buffers -> live params."""

    def __init__(self, params, staging):
        # [FUNCTION DOC] HELPER: Stores the live parameter mapping and the staging buffer mapping.
        self._params = params
        self._staging = staging

    def start_param_sync(self, force_sync=False):
        # [FUNCTION DOC] HELPER: Asserts the force-sync flag is true (the backuper must request a
        # synchronous sync), then copies every staging buffer into the same-named live parameter
        # in place.
        assert force_sync
        for name, param in self._params.items():
            param.copy_(self._staging[name])


class _Setup:
    # [CLASS DOC] HELPER: Test scaffold that builds, from a seeded random generator, named float32
    # main tensors, bfloat16 live parameters derived from them, staging buffers cloned from the
    # parameters, per-name shard slices (default the full tensor), and a few extra float32 tensors
    # that are not parameters. It wires a fake optimizer and model chunk into a main-cast context
    # and creates the backuper under test from a source getter yielding parameters then extras.
    def __init__(self, num_params=3, numel=16, num_extras=1, check=False, shards=None):
        # [FUNCTION DOC] HELPER: Builds all tensors and the backuper. Parameters are named p0, p1
        # and so on with the given element count; extras are named extra0 and so on with 4
        # elements. The optimizer sees only the owned slice of each main tensor. The context's
        # cast routine calls every chained optimizer's copy routine (read at call time, so the
        # chain can be swapped later); the rematerializable ids are the ids of the live parameter
        # tensors; the check flag enables hash verification. Seeded with zero for determinism.
        generator = torch.Generator().manual_seed(0)
        self.mains = {f"p{i}": torch.randn(numel, generator=generator) for i in range(num_params)}
        self.params = {name: main.to(torch.bfloat16) for name, main in self.mains.items()}
        self.staging = {name: param.clone() for name, param in self.params.items()}
        self.shards = shards or {name: slice(None) for name in self.mains}
        self.extras = {f"extra{i}": torch.randn(4, generator=generator) for i in range(num_extras)}
        owned = {name: main[self.shards[name]] for name, main in self.mains.items()}
        self.optimizer = SimpleNamespace(chained_optimizers=[_FakeOptimizer(owned, self.staging, self.shards)])
        self.model_chunk = _FakeModelChunk(self.params, self.staging)
        ctx = MainCastContext(
            cast_main_to_params=lambda: [
                opt._copy_main_params_to_model_params() for opt in self.optimizer.chained_optimizers
            ],
            model_chunks=[self.model_chunk],
            extras_getter=lambda: iter(self.extras.items()),
            rematerializable_ids={id(t) for t in self.params.values()},
            check=check,
        )
        self.backuper = TensorBackuper.create(
            source_getter=lambda: iter({**self.params, **self.extras}.items()),
            main_cast_ctx=ctx,
        )

    def corrupt_live_tensors(self):
        # [FUNCTION DOC] HELPER: Overwrites every live parameter and extra tensor in place with
        # not-a-number values to simulate lost weights.
        for param in self.params.values():
            param.fill_(float("nan"))
        for extra in self.extras.values():
            extra.fill_(float("nan"))


def test_create_returns_main_cast_variant():
    # [FUNCTION DOC] Asserts the factory, given a main-cast context, returns the main-cast
    # backuper variant.
    assert isinstance(_Setup().backuper, _TensorBackuperMainCast)


def test_round_trip_restores_bit_identical_weights():
    # [FUNCTION DOC] Records copies of all parameters and extras, backs up under the actor tag,
    # corrupts all live tensors, restores, and asserts every tensor is bit-identical to the copy.
    setup = _Setup()
    expected = {name: t.clone() for name, t in {**setup.params, **setup.extras}.items()}
    setup.backuper.backup("actor")
    setup.corrupt_live_tensors()
    setup.backuper.restore("actor")
    for name, tensor in {**setup.params, **setup.extras}.items():
        assert torch.equal(tensor, expected[name]), name


def test_restore_only_covers_owned_shard_and_relies_on_param_sync():
    # [FUNCTION DOC] Uses shards covering only the first half of each tensor, so the optimizer
    # cast writes only that half of staging while the other half keeps its value (standing in for
    # another data-parallel rank's cast). After backup, corruption and restore, asserts the full
    # parameters equal the originals, showing the parameter sync fills in the unowned half.
    numel = 16
    shards = {f"p{i}": slice(0, numel // 2) for i in range(3)}
    setup = _Setup(numel=numel, shards=shards)
    expected = {name: param.clone() for name, param in setup.params.items()}
    setup.backuper.backup("actor")
    setup.corrupt_live_tensors()
    # The unowned staging halves stand in for the other DP rank's cast.
    setup.backuper.restore("actor")
    for name, param in setup.params.items():
        assert torch.equal(param, expected[name]), name


def test_chained_optimizer_casts_every_inner_optimizer():
    # [FUNCTION DOC] Replaces the chained optimizer list with one fake optimizer per parameter,
    # each owning a single main tensor, then does backup, corruption and restore and asserts every
    # parameter is restored, i.e. all inner optimizers were invoked.
    setup = _Setup()
    names = list(setup.mains)
    inner = [_FakeOptimizer({n: setup.mains[n]}, setup.staging, setup.shards) for n in names]
    setup.optimizer.chained_optimizers = inner  # the ctx closure reads through
    expected = {name: param.clone() for name, param in setup.params.items()}
    setup.backuper.backup("actor")
    setup.corrupt_live_tensors()
    setup.backuper.restore("actor")
    for name, param in setup.params.items():
        assert torch.equal(param, expected[name]), name


def test_get_returns_pinned_backup_for_extras_and_live_tensors_for_params():
    # [FUNCTION DOC] After a backup, fetching the actor tensors must return, for parameters, the
    # live tensors themselves (same memory address), and for extras, a separate backup copy
    # (different address) with equal values; extras are paused during weight updates so the
    # backup must be served.
    setup = _Setup()
    setup.backuper.backup("actor")
    got = setup.backuper.get("actor")
    for name, param in setup.params.items():
        assert got[name].data_ptr() == param.data_ptr(), name
    for name, extra in setup.extras.items():
        # Extras are paused during update_weights, so get() must hand out the backup.
        assert got[name].data_ptr() != extra.data_ptr(), name
        assert torch.equal(got[name], extra), name


def test_check_verifies_first_cycles_and_raises_on_corruption():
    # [FUNCTION DOC] With checking enabled, backs up, then perturbs one main tensor element so the
    # re-cast parameters differ; asserts restore raises a runtime error saying the result is not
    # bit-identical.
    setup = _Setup(check=True)
    setup.backuper.backup("actor")
    setup.mains["p0"][0] += 1.0
    with pytest.raises(RuntimeError, match="not bit-identical"):
        setup.backuper.restore("actor")


def test_check_stops_after_check_num_cycles():
    # [FUNCTION DOC] With checking enabled, runs the configured number of backup and restore
    # cycles, then one more backup; asserts the stored expected hashes are cleared (none), and a
    # subsequent perturbation of a main tensor no longer raises on restore.
    setup = _Setup(check=True)
    for _ in range(setup.backuper._check_num_cycles):
        setup.backuper.backup("actor")
        setup.backuper.restore("actor")
    setup.backuper.backup("actor")
    assert setup.backuper._expected_hashes is None
    setup.mains["p0"][0] += 1.0
    setup.backuper.restore("actor")


def test_no_check_computes_no_hashes():
    # [FUNCTION DOC] With checking disabled, asserts a backup stores no expected hashes and that
    # perturbing a main tensor does not make restore raise.
    setup = _Setup(check=False)
    setup.backuper.backup("actor")
    assert setup.backuper._expected_hashes is None
    setup.mains["p0"][0] += 1.0
    setup.backuper.restore("actor")


def test_check_detects_tensor_set_change():
    # [FUNCTION DOC] With checking enabled, backs up, then adds a new parameter and staging buffer
    # to the source; asserts restore raises an assertion error saying the tensor set changed.
    setup = _Setup(check=True)
    setup.backuper.backup("actor")
    setup.params["p_new"] = torch.zeros(4, dtype=torch.bfloat16)
    setup.staging["p_new"] = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="changed the tensor set"):
        setup.backuper.restore("actor")


def test_get_rejects_unknown_tensor_in_source():
    # [FUNCTION DOC] After a backup, adds an unexpected tensor named stray_buffer to the source;
    # asserts fetching the actor tensors raises an assertion error mentioning that name.
    setup = _Setup()
    setup.backuper.backup("actor")
    setup.params["stray_buffer"] = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="stray_buffer"):
        setup.backuper.get("actor")


def test_non_actor_tag_keeps_full_pinned_copy():
    # [FUNCTION DOC] Backs up under the ref tag and asserts the known tags are actor then ref.
    # After corruption, restoring ref brings every parameter and extra back to its recorded value,
    # and fetching ref returns tensors at different memory addresses from the live ones (a full
    # pinned host copy, not live tensors).
    setup = _Setup()
    ref_values = {name: t.clone() for name, t in {**setup.params, **setup.extras}.items()}
    setup.backuper.backup("ref")
    assert setup.backuper.backup_tags == ["actor", "ref"]
    setup.corrupt_live_tensors()
    setup.backuper.restore("ref")
    for name, tensor in {**setup.params, **setup.extras}.items():
        assert torch.equal(tensor, ref_values[name]), name
    got = setup.backuper.get("ref")
    for name in ref_values:
        # Pinned host copies, never the live tensors.
        assert got[name].data_ptr() != {**setup.params, **setup.extras}[name].data_ptr(), name


def test_actor_restore_wins_after_ref_switch():
    # [FUNCTION DOC] Backs up actor, overwrites all parameters with 7.0 (as if loading ref
    # weights), backs up ref, restores ref, then restores actor; asserts all parameters and extras
    # equal the original actor values.
    # A ref switch overwrites the live params; the actor restore must win them back.
    setup = _Setup()
    actor_values = {name: t.clone() for name, t in {**setup.params, **setup.extras}.items()}
    setup.backuper.backup("actor")
    for param in setup.params.values():
        param.fill_(7.0)  # stand-in for loading ref weights into the model
    setup.backuper.backup("ref")
    setup.backuper.restore("ref")
    setup.backuper.restore("actor")
    for name, tensor in {**setup.params, **setup.extras}.items():
        assert torch.equal(tensor, actor_values[name]), name
