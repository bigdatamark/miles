# [MODULE DOC] CPU-only tests for the Hub kernel plugin of the FSDP backend (HubKernels, its
# loader/presets, and argument validation). The external kernels package is replaced by a fake
# module placed into the module registry, so nothing touches the network or a GPU. Covered: native
# backend inertness, rejection of hub mode in bit-exact modes, strict vs fallback behaviour,
# mapping providers, per-slot independence, binding onto fake gated-delta-net layers and Nemotron
# attention mixers, and Hub provenance checks. Also exports shared helpers (argument builder, fake
# kernels installer, fake gated delta net model builder) that the distributed test reuses.
"""Hub compute kernels for the FSDP backend (CPU-only, no GPU, no network).

`kernels` is stubbed through sys.modules throughout, so these never touch the Hub. What they pin:

  * `--kernel-backend native` (the default) resolves nothing and never imports `kernels`;
  * the numerics guard that keeps hub kernels out of --true-on-policy-mode / --deterministic-mode;
  * strict vs. fallback when a repo or a function is missing from the build;
  * and, the reason this feature exists, that a Hub-bound `causal_conv1d_fn` still receives the
    packed-document `seq_idx` that `models/qwen3_5.py` injects -- i.e. the binder composes with the
    packing wrapper rather than defeating it.
"""

import sys
import types
from argparse import Namespace

import pytest
import torch
import torch.nn as nn

from miles.backends.fsdp_utils.arguments import validate_kernel_backend_args
from miles.backends.fsdp_utils.plugins.hf_kernels import HubKernels, HubKernelSpec
from miles.backends.fsdp_utils.plugins.hf_kernels.loader import load_mapping
from miles.backends.fsdp_utils.plugins.hf_kernels.presets import (
    CAUSAL_CONV1D,
    SLOT_CAUSAL_CONV1D,
    SLOT_FLASH_ATTN_VARLEN,
    SLOT_GATED_DELTA_RULE,
)


def _make_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a plain namespace of arguments with hub kernel backend, empty
    # mapping path, non-strict, and both bit-exact modes off; keyword overrides replace any of these
    # defaults (extra keys are added). Returns the namespace.
    base = dict(
        kernel_backend="hub",
        kernel_mapping_path="",
        kernel_strict=False,
        true_on_policy_mode=False,
        deterministic_mode=False,
    )
    base.update(overrides)
    return Namespace(**base)


def _stub_kernels(monkeypatch, *, module=None, modules=None, raises=None, calls=None):
    # [FUNCTION DOC] HELPER: Installs a fake kernels package into the module registry (undone by the
    # monkeypatch fixture). Its get_kernel records (repo, revision, version) into the optional calls
    # list, raises the given exception if provided, else returns the module from the per-repo
    # mapping (raising a file-not-found error for repos absent from it) or the single shared module.
    # Every successful fetch is appended to an internal list served by the fake get_loaded_kernels,
    # with repo id, revision (or a synthesized "commit-v<version>") and a build id "build:<repo>"
    # so provenance checks can pass.
    """Install a fake `kernels` package.

    `module` serves every repo; `modules` maps repo_id -> module so a test can make one repo
    resolve and another 404, which is how the per-slot independence is exercised.
    """

    loaded = []

    def get_kernel(repo_id, revision=None, version=None, user_agent=None):
        # [FUNCTION DOC] HELPER: Fake kernel fetch; see the enclosing installer for behaviour. Returns
        # the stub module for the repo or raises the configured error.
        if calls is not None:
            calls.append((repo_id, revision, version))
        if raises is not None:
            raise raises
        if modules is not None:
            if repo_id not in modules:
                raise FileNotFoundError(f"no stub build for {repo_id}")
            result = modules[repo_id]
        else:
            result = module
        loaded.append(
            types.SimpleNamespace(
                module=result,
                repo_info=types.SimpleNamespace(repo_id=repo_id, revision=revision or f"commit-v{version}"),
                metadata=types.SimpleNamespace(id=f"build:{repo_id}"),
            )
        )
        return result

    fake = types.ModuleType("kernels")
    fake.get_kernel = get_kernel
    fake.get_loaded_kernels = lambda: loaded
    monkeypatch.setitem(sys.modules, "kernels", fake)


def _kernel_module(**functions):
    # [FUNCTION DOC] HELPER: Builds an empty fake module and sets each keyword argument as an
    # attribute (name to callable). Returns the module.
    mod = types.ModuleType("fake_hub_kernel")
    for name, fn in functions.items():
        setattr(mod, name, fn)
    return mod


def _noop(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Accepts any arguments and returns nothing; placeholder kernel function.
    return None


def _all_hub_modules(**overrides):
    # [FUNCTION DOC] HELPER: Returns a mapping from repo id to fake module for the three repos in the
    # shipped default mapping (fla with both gated delta rule functions, causal-conv1d with the conv
    # and update functions, flash-attn2 with the varlen function), all no-op callables. Keyword
    # overrides are keyed by repo id and replace or add entries.
    """One stub module per repo in the shipped mapping, with every advertised function present."""
    modules = {
        "kernels-community/fla": _kernel_module(chunk_gated_delta_rule=_noop, fused_recurrent_gated_delta_rule=_noop),
        "kernels-community/causal-conv1d": _kernel_module(causal_conv1d_fn=_noop, causal_conv1d_update=_noop),
        "kernels-community/flash-attn2": _kernel_module(flash_attn_varlen_func=_noop),
    }
    modules.update(overrides)
    return modules


# --------------------------------------------------------------------------------------- args


def test_native_is_the_default_and_stays_inert(monkeypatch):
    # [FUNCTION DOC] Test: with the native backend, preparing hub kernels with the kernels package
    # removed from the registry succeeds; binding a plain model yields an empty result, resolving
    # the conv slot yields nothing, and the kernels package is never imported.
    args = _make_args(kernel_backend="native")
    monkeypatch.delitem(sys.modules, "kernels", raising=False)

    hub = HubKernels.prepare(args)

    assert hub.bind(nn.Sequential(nn.Linear(4, 4))) == {}
    assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is None
    # The whole point of the lazy import: a native run must not need `kernels` to exist at all.
    assert "kernels" not in sys.modules


@pytest.mark.parametrize("mode", ["true_on_policy_mode", "deterministic_mode"])
def test_hub_backend_is_rejected_in_the_bit_exact_modes(mode):
    # [FUNCTION DOC] Test (parametrized over true-on-policy and deterministic modes): validation of
    # hub backend args with the mode on raises a value error mentioning incompatibility.
    with pytest.raises(ValueError, match="incompatible with --true-on-policy-mode"):
        validate_kernel_backend_args(_make_args(**{mode: True}))


@pytest.mark.parametrize("mode", ["true_on_policy_mode", "deterministic_mode"])
def test_default_mapping_is_empty_in_the_bit_exact_modes(mode):
    # [FUNCTION DOC] Test (parametrized over both bit-exact modes): loading the mapping with the
    # mode on yields an empty mapping even without validation.
    """Defence in depth: even if validation is bypassed, the preset hands back nothing to bind."""
    assert load_mapping(_make_args(**{mode: True})) == {}


def test_unknown_backend_is_rejected():
    # [FUNCTION DOC] Test: an unknown backend name makes validation raise a value error saying the
    # backend must be one of the allowed values.
    with pytest.raises(ValueError, match="--kernel-backend must be one of"):
        validate_kernel_backend_args(_make_args(kernel_backend="hubb"))


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(kernel_strict=True), "--kernel-strict only applies"),
        (dict(kernel_mapping_path="pkg.mod.fn"), "--kernel-mapping-path only applies"),
    ],
)
def test_hub_only_flags_are_rejected_under_native(overrides, message):
    # [FUNCTION DOC] Test (parametrized): the strict flag or a mapping path combined with the native
    # backend makes validation raise a value error with the matching "only applies" message.
    with pytest.raises(ValueError, match=message):
        validate_kernel_backend_args(_make_args(kernel_backend="native", **overrides))


def test_native_and_hub_both_validate_clean():
    # [FUNCTION DOC] Test: plain native args and hub args with strict on both pass validation
    # without raising.
    validate_kernel_backend_args(_make_args(kernel_backend="native"))
    validate_kernel_backend_args(_make_args(kernel_backend="hub", kernel_strict=True))


# ------------------------------------------------------------------------------------ presets


def test_default_mapping_pins_the_expected_repos():
    # [FUNCTION DOC] Test: the default mapping has exactly the three slots (gated delta rule, causal
    # conv1d, flash attention varlen), pinned to repos kernels-community/fla (version 1),
    # kernels-community/causal-conv1d (version 1) and kernels-community/flash-attn2 (version 2); the
    # fla slot lists both chunk and fused recurrent functions, and the other slots include the conv
    # and varlen functions respectively.
    mapping = load_mapping(_make_args())

    assert set(mapping) == {SLOT_GATED_DELTA_RULE, SLOT_CAUSAL_CONV1D, SLOT_FLASH_ATTN_VARLEN}
    assert (mapping[SLOT_GATED_DELTA_RULE].repo_id, mapping[SLOT_GATED_DELTA_RULE].version) == (
        "kernels-community/fla",
        1,
    )
    assert (mapping[SLOT_CAUSAL_CONV1D].repo_id, mapping[SLOT_CAUSAL_CONV1D].version) == (
        "kernels-community/causal-conv1d",
        1,
    )
    assert (mapping[SLOT_FLASH_ATTN_VARLEN].repo_id, mapping[SLOT_FLASH_ATTN_VARLEN].version) == (
        "kernels-community/flash-attn2",
        2,
    )
    # Every mapped function is one a binder actually reaches for.
    assert mapping[SLOT_GATED_DELTA_RULE].functions == (
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    )
    assert "causal_conv1d_fn" in mapping[SLOT_CAUSAL_CONV1D].functions
    assert "flash_attn_varlen_func" in mapping[SLOT_FLASH_ATTN_VARLEN].functions


def _custom_mapping(args):
    # [FUNCTION DOC] HELPER: Mapping provider (addressed by dotted path in tests) returning a single
    # conv slot spec for repo me/my-conv1d at revision main with the conv and update functions.
    return {
        SLOT_CAUSAL_CONV1D: HubKernelSpec(
            repo_id="me/my-conv1d", revision="main", functions=("causal_conv1d_fn", "causal_conv1d_update")
        )
    }


def _conv_only_mapping(args):
    # [FUNCTION DOC] HELPER: Mapping provider returning only the conv slot with the shipped preset.
    return {SLOT_CAUSAL_CONV1D: CAUSAL_CONV1D}


def _broken_mapping(args):
    # [FUNCTION DOC] HELPER: Mapping provider returning a plain string instead of a spec, to test
    # type rejection.
    return {SLOT_CAUSAL_CONV1D: "kernels-community/causal-conv1d"}


def test_kernel_mapping_path_substitutes_the_whole_mapping():
    # [FUNCTION DOC] Test: pointing the mapping path at the custom provider replaces the default
    # mapping entirely; only the conv slot remains and its repo is me/my-conv1d.
    args = _make_args(kernel_mapping_path=f"{__name__}._custom_mapping")
    mapping = load_mapping(args)

    assert set(mapping) == {SLOT_CAUSAL_CONV1D}
    assert mapping[SLOT_CAUSAL_CONV1D].repo_id == "me/my-conv1d"


def test_kernel_mapping_path_rejects_a_non_spec_value():
    # [FUNCTION DOC] Test: a provider returning a non-spec value makes mapping load raise a type
    # error saying the value must be a spec.
    args = _make_args(kernel_mapping_path=f"{__name__}._broken_mapping")
    with pytest.raises(TypeError, match="must be a HubKernelSpec"):
        load_mapping(args)


def test_spec_rejects_both_pins_and_no_functions():
    # [FUNCTION DOC] Test: constructing a spec with both a version and a revision raises a value
    # error ("not both"); a spec with no functions raises a value error requiring at least one.
    with pytest.raises(ValueError, match="not both"):
        HubKernelSpec(repo_id="a/b", version=1, revision="main", functions=("f",))
    with pytest.raises(ValueError, match="at least one function"):
        HubKernelSpec(repo_id="a/b", version=1)


# ---------------------------------------------------------------------------------- resolution


def test_prepare_downloads_once_and_resolve_reads_the_cache(monkeypatch):
    # [FUNCTION DOC] Test: with a conv-only mapping and a stub module exposing both functions,
    # prepare fetches the repo once (recorded as repo, no revision, version 1); two resolves of the
    # conv slot return equal results containing both functions and cause no further fetches.
    calls = []
    _stub_kernels(
        monkeypatch,
        module=_kernel_module(causal_conv1d_fn=lambda **kw: None, causal_conv1d_update=lambda: None),
        calls=calls,
    )
    hub = HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._conv_only_mapping"))

    first = hub.resolve_slot(SLOT_CAUSAL_CONV1D)
    second = hub.resolve_slot(SLOT_CAUSAL_CONV1D)

    assert set(first) == {"causal_conv1d_fn", "causal_conv1d_update"}
    assert first == second
    # One download for the policy model and the ref model together.
    assert calls == [("kernels-community/causal-conv1d", None, 1)]


def test_unresolvable_repo_falls_back_to_the_native_kernel(monkeypatch, caplog):
    # [FUNCTION DOC] Test: when the fetch raises a file-not-found error and strict is off, prepare
    # logs a warning containing "keeping the native kernel"; the slot resolves to nothing twice and
    # the failure is cached so only one fetch is attempted.
    calls = []
    _stub_kernels(monkeypatch, raises=FileNotFoundError("no build for torch2.9-cu130"), calls=calls)

    with caplog.at_level("WARNING"):
        hub = HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._conv_only_mapping"))

    assert "keeping the native kernel" in caplog.text
    # The failure is remembered: re-resolving per model would just re-pay the Hub round trip.
    assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is None
    assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is None
    assert len(calls) == 1


def test_unresolvable_repo_raises_under_strict(monkeypatch):
    # [FUNCTION DOC] Test: with strict on and a failing fetch, prepare raises a runtime error that
    # mentions the strict flag.
    _stub_kernels(monkeypatch, raises=FileNotFoundError("no build for torch2.9-cu130"))

    with pytest.raises(RuntimeError, match="--kernel-strict"):
        HubKernels.prepare(_make_args(kernel_strict=True))


def test_a_build_missing_the_function_falls_back(monkeypatch, caplog):
    # [FUNCTION DOC] Test: a build lacking the update function leads, non-strict, to a warning
    # ("does not expose callable") and the slot resolving to nothing; with strict on, prepare
    # raises a runtime error with that same message.
    _stub_kernels(monkeypatch, module=_kernel_module(causal_conv1d_fn=lambda: None))  # no causal_conv1d_update

    with caplog.at_level("WARNING"):
        hub = HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._conv_only_mapping"))
    assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is None
    assert "does not expose callable" in caplog.text

    with pytest.raises(RuntimeError, match="does not expose callable"):
        HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._conv_only_mapping", kernel_strict=True))


def test_prepare_is_inert_under_native(monkeypatch):
    # [FUNCTION DOC] Test: with the native backend, prepare performs no fetches.
    calls = []
    _stub_kernels(monkeypatch, module=_kernel_module(), calls=calls)

    HubKernels.prepare(_make_args(kernel_backend="native"))

    assert calls == []


def test_prepare_warms_every_mapped_repo_once(monkeypatch):
    # [FUNCTION DOC] Test: with the default mapping, prepare fetches each of the three repos exactly
    # once, and subsequently resolving all three slots succeeds (cache hits only).
    calls = []
    _stub_kernels(monkeypatch, modules=_all_hub_modules(), calls=calls)

    hub = HubKernels.prepare(_make_args())
    for slot in (SLOT_GATED_DELTA_RULE, SLOT_CAUSAL_CONV1D, SLOT_FLASH_ATTN_VARLEN):
        assert hub.resolve_slot(slot) is not None

    # Downloaded once each by prepare; the per-slot resolves are all cache hits.
    assert sorted(repo for repo, _, _ in calls) == [
        "kernels-community/causal-conv1d",
        "kernels-community/fla",
        "kernels-community/flash-attn2",
    ]


_mapping_provider_calls = []


def _counting_mapping(args):
    # [FUNCTION DOC] HELPER: Mapping provider that records each invocation in a module-level list
    # and returns the conv-only preset mapping.
    _mapping_provider_calls.append(None)
    return {SLOT_CAUSAL_CONV1D: CAUSAL_CONV1D}


def test_binding_uses_the_mapping_agreed_in_prepare(monkeypatch):
    # [FUNCTION DOC] Test: after prepare with the counting provider, binding a two-layer fake gated
    # delta net model reports two bound layers, the conv slot resolves, and the provider was called
    # exactly once overall (binding does not re-invoke it).
    """The provider runs once, in prepare; binding must not re-invoke a possibly drifting provider."""
    _mapping_provider_calls.clear()
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    hub = HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._counting_mapping"))

    assert hub.bind(_build_gdn_model()) == {"gated_deltanet": 2}
    assert hub.resolve_slot(SLOT_CAUSAL_CONV1D) is not None
    assert len(_mapping_provider_calls) == 1


# ------------------------------------------------------------------------------ GatedDeltaNet


class FakeGatedDeltaNet(nn.Module):
    # [CLASS DOC] HELPER: Fake layer exposing the attributes the hub binder and the packing forward
    # patch touch: optional conv function and conv update, plus chunk and recurrent gated delta rule
    # functions defaulting to a pure-torch stand-in that ignores sequence boundaries.
    """Mirrors the parts of Qwen3NextGatedDeltaNet the binder and the packing wrapper touch.

    `native_chunk` defaults to a stand-in for transformers' `torch_chunk_gated_delta_rule`: it
    accepts `cu_seqlens` into `**kwargs` and ignores it, which is exactly the silent degradation
    the `fla` slot exists to remove.
    """

    def __init__(self, native_conv=None, native_chunk=None):
        # [FUNCTION DOC] HELPER: Stores the given native conv function, no conv update, and uses the
        # given native chunk function (or the stand-in) for both the chunk and the recurrent slots.
        super().__init__()
        self.causal_conv1d_fn = native_conv
        self.causal_conv1d_update = None
        self.chunk_gated_delta_rule = native_chunk or _torch_chunk_stand_in
        self.recurrent_gated_delta_rule = native_chunk or _torch_chunk_stand_in

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Calls the current chunk function with no boundaries (discarding the
        # result), then, if no conv function is set, returns the input unchanged; otherwise returns
        # the result of the conv function called with keyword arguments x, no weight/bias, silu
        # activation and no sequence index.
        self.chunk_gated_delta_rule(x, g=None, beta=None, cu_seqlens=None)
        if self.causal_conv1d_fn is None:
            # transformers' fallback: no seq_idx, so packed documents bleed into each other.
            return x
        return self.causal_conv1d_fn(x=x, weight=None, bias=None, activation="silu", seq_idx=None)


def _torch_chunk_stand_in(query, g=None, beta=None, **kwargs):
    # [FUNCTION DOC] HELPER: Stand-in for the pure-torch chunked gated delta rule; ignores extra
    # keyword arguments (including sequence boundaries) and returns the query unchanged.
    """transformers' pure-torch GDN fallback: swallows cu_seqlens and never resets per document."""
    return query


def _build_gdn_model(native_conv=None, native_chunk=None, n_layers=2):
    # [FUNCTION DOC] HELPER: Builds a module whose layers attribute is a list of n_layers (default
    # two) fake gated delta net layers sharing the given native conv and chunk functions. Returns it.
    model = nn.Module()
    model.layers = nn.ModuleList([FakeGatedDeltaNet(native_conv, native_chunk) for _ in range(n_layers)])
    return model


def test_binder_rebinds_every_gated_deltanet(monkeypatch):
    # [FUNCTION DOC] Test: binding with all hub modules reports two gated delta net layers bound; on
    # every layer the chunk function is the fla chunk function, the recurrent function is the fla
    # fused recurrent function (name mapped across), and the conv and conv update functions are the
    # causal-conv1d ones.
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    model = _build_gdn_model()

    assert HubKernels.prepare(_make_args()).bind(model) == {"gated_deltanet": 2}
    fla = sys.modules["kernels"].get_kernel("kernels-community/fla")
    conv = sys.modules["kernels"].get_kernel("kernels-community/causal-conv1d")
    for layer in model.layers:
        assert layer.chunk_gated_delta_rule is fla.chunk_gated_delta_rule
        # transformers stores the recurrent kernel without the `fused_` prefix; the slot maps across.
        assert layer.recurrent_gated_delta_rule is fla.fused_recurrent_gated_delta_rule
        assert layer.causal_conv1d_fn is conv.causal_conv1d_fn
        assert layer.causal_conv1d_update is conv.causal_conv1d_update


def test_gated_deltanet_slots_resolve_independently(monkeypatch):
    # [FUNCTION DOC] Test: with the causal-conv1d repo missing, binding still rebinds the chunk
    # function from fla on the layers (two layers reported) while the conv function stays unset.
    """One Hub repo being unavailable must not cost the run the other slot."""
    modules = _all_hub_modules()
    del modules["kernels-community/causal-conv1d"]
    _stub_kernels(monkeypatch, modules=modules)
    model = _build_gdn_model(native_conv=None)

    assert HubKernels.prepare(_make_args()).bind(model) == {"gated_deltanet": 2}
    fla = sys.modules["kernels"].get_kernel("kernels-community/fla")
    assert model.layers[0].chunk_gated_delta_rule is fla.chunk_gated_delta_rule
    assert model.layers[0].causal_conv1d_fn is None  # untouched: no build to bind


def test_binder_is_inert_under_native(monkeypatch):
    # [FUNCTION DOC] Test: with the native backend, binding returns an empty result and leaves the
    # conv unset and the chunk function as the stand-in.
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    model = _build_gdn_model(native_conv=None)

    assert HubKernels.prepare(_make_args(kernel_backend="native")).bind(model) == {}
    assert model.layers[0].causal_conv1d_fn is None
    assert model.layers[0].chunk_gated_delta_rule is _torch_chunk_stand_in


def test_binder_leaves_the_native_kernels_when_every_hub_repo_fails(monkeypatch):
    # [FUNCTION DOC] Test: when every fetch fails (non-strict), binding returns an empty result and
    # the layers keep their native conv function and stand-in chunk function.
    def native(**kwargs):
        # [FUNCTION DOC] HELPER: Placeholder native conv function; returns nothing.
        return None

    _stub_kernels(monkeypatch, raises=FileNotFoundError("nope"))
    model = _build_gdn_model(native_conv=native)

    assert HubKernels.prepare(_make_args()).bind(model) == {}
    assert model.layers[0].causal_conv1d_fn is native
    assert model.layers[0].chunk_gated_delta_rule is _torch_chunk_stand_in


def test_hub_kernels_still_receive_the_packed_document_boundaries(monkeypatch):
    # [FUNCTION DOC] Test: hub chunk and conv functions record the cumulative sequence lengths and
    # the per-token sequence index they receive. A throwaway subclass of the fake layer gets the
    # packing forward patch applied, a one-layer model is bound to hub kernels, and boundaries
    # (lengths 0,2,4 and index 0,0,1,1) are set on the layer before a forward over a zero tensor of
    # shape 1 by 4. Asserts both kernels received the boundaries unchanged, and that the layer's
    # chunk and conv handles afterwards are the original hub functions (the patch restores them).
    """The reason this feature exists.

    `_patch_gdn_forward` injects `cu_seqlens` into the recurrence and `seq_idx` into the conv. The
    native fallbacks defeat that -- the torch GDN kernel swallows `cu_seqlens` into `**kwargs`, and
    a missing conv wheel leaves nothing to wrap. Bind both from the Hub and the boundaries must
    arrive at the kernels unchanged.
    """
    from miles.backends.fsdp_utils.models.qwen3_5 import _patch_gdn_forward

    seen = {}

    def hub_chunk(query, g=None, beta=None, **kwargs):
        # [FUNCTION DOC] HELPER: Fake hub chunk function; records the sequence-boundary keyword in
        # the seen dictionary and returns the query.
        seen["cu_seqlens"] = kwargs.get("cu_seqlens")
        return query

    def hub_conv(**kwargs):
        # [FUNCTION DOC] HELPER: Fake hub conv function; records the sequence index keyword in the
        # seen dictionary and returns the x argument.
        seen["seq_idx"] = kwargs.get("seq_idx")
        return kwargs["x"]

    _stub_kernels(
        monkeypatch,
        modules=_all_hub_modules(
            **{
                "kernels-community/fla": _kernel_module(
                    chunk_gated_delta_rule=hub_chunk, fused_recurrent_gated_delta_rule=_noop
                ),
                "kernels-community/causal-conv1d": _kernel_module(
                    causal_conv1d_fn=hub_conv, causal_conv1d_update=_noop
                ),
            }
        ),
    )

    # Patch a throwaway subclass: _patch_gdn_forward rewrites the class forward permanently.
    gdn_cls = type("PackedGatedDeltaNet", (FakeGatedDeltaNet,), {})
    _patch_gdn_forward(gdn_cls)

    model = nn.Module()
    model.layers = nn.ModuleList([gdn_cls()])
    assert HubKernels.prepare(_make_args()).bind(model) == {"gated_deltanet": 1}

    layer = model.layers[0]
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    seq_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
    layer._gdn_cu_seqlens = cu_seqlens
    layer._gdn_seq_idx = seq_idx
    layer(torch.zeros(1, 4))

    assert torch.equal(seen["cu_seqlens"], cu_seqlens)
    assert torch.equal(seen["seq_idx"], seq_idx)
    # The wrapper restores the handles it swapped, so the next forward starts from the Hub kernels
    # again rather than from a stack of boundary-injecting wrappers.
    assert layer.chunk_gated_delta_rule is hub_chunk
    assert layer.causal_conv1d_fn is hub_conv


# --------------------------------------------------------------------------------- NemotronH


class FakeAttnMixer(nn.Module):
    # [CLASS DOC] HELPER: Empty module standing in for a Nemotron attention mixer; the binder
    # attaches the hub varlen function to it.
    pass


class FakeAttnBlock(nn.Module):
    # [CLASS DOC] HELPER: Fake Nemotron block whose block type is attention and whose mixer is a
    # fake attention mixer.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Sets the block type to attention and creates the mixer.
        super().__init__()
        self.block_type = "attention"
        self.mixer = FakeAttnMixer()


def test_binder_stashes_varlen_on_each_nemotron_attention_mixer(monkeypatch):
    # [FUNCTION DOC] Test: with a model of three fake attention blocks and a stub flash-attn2 repo,
    # binding reports three Nemotron layers bound and each mixer carries the hub varlen function
    # under the well-known attribute name.
    from miles.backends.fsdp_utils.models.nemotron_h import HUB_VARLEN_ATTR

    def varlen(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake hub varlen attention function; returns nothing.
        return None

    _stub_kernels(
        monkeypatch,
        modules=_all_hub_modules(**{"kernels-community/flash-attn2": _kernel_module(flash_attn_varlen_func=varlen)}),
    )

    model = nn.Module()
    model.blocks = nn.ModuleList([FakeAttnBlock() for _ in range(3)])

    assert HubKernels.prepare(_make_args()).bind(model) == {"nemotron_h": 3}
    for block in model.blocks:
        assert getattr(block.mixer, HUB_VARLEN_ATTR) is varlen


def test_binders_skip_architectures_that_are_not_present(monkeypatch):
    # [FUNCTION DOC] Test: binding a model with no recognized architecture returns an empty result.
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    plain = nn.Sequential(nn.Linear(4, 4))

    assert HubKernels.prepare(_make_args()).bind(plain) == {}


class FakeNemotronAttn(nn.Module):
    # [CLASS DOC] HELPER: Minimal Nemotron attention mixer (head dim, four bias-free linear
    # projections for q, k, v, o) whose un-patched forward just counts calls and returns the hidden
    # states with no attention weights.
    """Enough of the NemotronH attention mixer for `_patch_attn_forward` to rewrite its forward."""

    def __init__(self, hidden=8, heads=2):
        # [FUNCTION DOC] HELPER: Sets head dim to hidden divided by heads, creates the projections
        # of size hidden by hidden, and zeroes the dense-call counter.
        super().__init__()
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.dense_calls = 0

    def forward(self, hidden_states, *args, **kwargs):
        # [FUNCTION DOC] HELPER: The un-patched dense path; increments the dense-call counter and
        # returns the hidden states together with no attention weights.
        self.dense_calls += 1  # the un-patched dense path
        return hidden_states, None


def _patch_nemotron_attn(monkeypatch, native_varlen):
    # [FUNCTION DOC] HELPER: Places a fake flash_attn module (whose varlen function is the given
    # one, possibly nothing) in the module registry, creates a fresh subclass of the fake attention
    # mixer, applies the Nemotron attention forward patch to it, and returns the class.
    """Install `_patch_attn_forward` on a fresh class with `flash_attn` stubbed to `native_varlen`."""
    from miles.backends.fsdp_utils.models import nemotron_h as nemotron_h_module

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_varlen_func = native_varlen
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    cls = type("PatchTargetAttn", (FakeNemotronAttn,), {})
    nemotron_h_module._patch_attn_forward(cls)
    return cls


def _run_packed_forward(mixer):
    # [FUNCTION DOC] HELPER: Sets packing boundaries on the mixer (cumulative lengths 0,2,4 and max
    # sequence length 2), runs it on a zero tensor of shape 1 by 4 by 8 and returns the output.
    mixer._packing_cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    mixer._packing_max_seqlen = 2
    return mixer(torch.zeros(1, 4, 8))


def test_nemotron_attention_still_uses_the_native_varlen_kernel(monkeypatch):
    # [FUNCTION DOC] Test: with no hub function bound, a packed forward of the patched mixer calls
    # the native varlen function exactly once and never the dense path.
    """The Hub handle is an override, not a replacement: with no --kernel-backend hub the wheel wins."""
    used = []

    def native(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake native varlen function; records use and returns zeros of shape
        # 4 by 2 by 4.
        used.append("native")
        return torch.zeros(4, 2, 4)

    mixer = _patch_nemotron_attn(monkeypatch, native)()
    _run_packed_forward(mixer)

    assert used == ["native"]
    assert mixer.dense_calls == 0


def test_nemotron_attention_prefers_the_hub_varlen_kernel_when_bound(monkeypatch):
    # [FUNCTION DOC] Test: when a hub varlen function is attached to the mixer, a packed forward
    # uses only the hub function (not native, not dense).
    from miles.backends.fsdp_utils.models.nemotron_h import HUB_VARLEN_ATTR

    used = []

    def native(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake native varlen function; records use and returns zeros.
        used.append("native")
        return torch.zeros(4, 2, 4)

    def from_hub(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake hub varlen function; records use and returns zeros of shape 4
        # by 2 by 4.
        used.append("hub")
        return torch.zeros(4, 2, 4)

    mixer = _patch_nemotron_attn(monkeypatch, native)()
    setattr(mixer, HUB_VARLEN_ATTR, from_hub)
    _run_packed_forward(mixer)

    assert used == ["hub"]
    assert mixer.dense_calls == 0


def test_nemotron_attention_falls_back_to_dense_when_no_varlen_kernel_exists(monkeypatch):
    # [FUNCTION DOC] Test: with neither a native varlen function nor a hub binding, a packed forward
    # runs the dense path exactly once.
    """Pre-existing behaviour, unchanged: no wheel and no hub binding means the dense forward."""
    mixer = _patch_nemotron_attn(monkeypatch, None)()
    _run_packed_forward(mixer)

    assert mixer.dense_calls == 1


# ------------------------------------------------------------------------------ custom mappings


def _incomplete_mapping(args):
    # [FUNCTION DOC] HELPER: Mapping provider returning one spec for the slot named by the test slot
    # argument, version 1 of me/custom, declaring only the conv function (so incomplete for every
    # slot).
    return {args.test_slot: HubKernelSpec(repo_id="me/custom", version=1, functions=("causal_conv1d_fn",))}


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("slot", [SLOT_CAUSAL_CONV1D, SLOT_GATED_DELTA_RULE, SLOT_FLASH_ATTN_VARLEN])
def test_incomplete_custom_slot_is_rejected_before_downloading(monkeypatch, strict, slot):
    # [FUNCTION DOC] Test (parametrized over strict and each slot): a custom mapping whose spec
    # lacks required functions makes prepare raise a value error ("must declare all required
    # functions") and no fetch happens.
    calls = []
    _stub_kernels(monkeypatch, modules=_all_hub_modules(), calls=calls)
    args = _make_args(kernel_mapping_path=f"{__name__}._incomplete_mapping", kernel_strict=strict, test_slot=slot)

    with pytest.raises(ValueError, match="must declare all required functions"):
        HubKernels.prepare(args)

    assert calls == []


def test_unknown_custom_slot_is_rejected():
    # [FUNCTION DOC] Test: a custom mapping using a misspelled slot name makes mapping load raise a
    # value error ("unknown kernel mapping slot").
    args = _make_args(kernel_mapping_path=f"{__name__}._incomplete_mapping", test_slot="causal_conv1d_typo")
    with pytest.raises(ValueError, match="unknown kernel mapping slot"):
        load_mapping(args)


def test_complete_custom_mapping_binds_policy_and_reference_without_changing_weights(monkeypatch):
    # [FUNCTION DOC] Test: with a complete custom conv mapping, prepare once and bind two separately
    # built models (policy and reference stand-ins, each with an extra linear layer). Each bind
    # reports two layers; conv and update functions are the custom ones, the chunk function stays
    # the stand-in, and the state dictionary keys and values are bit-identical before and after.
    # Only one fetch (repo me/my-conv1d, revision main, no version) is made in total.
    conv, update = lambda **kwargs: None, lambda **kwargs: None
    calls = []
    _stub_kernels(monkeypatch, module=_kernel_module(causal_conv1d_fn=conv, causal_conv1d_update=update), calls=calls)
    hub = HubKernels.prepare(_make_args(kernel_mapping_path=f"{__name__}._custom_mapping"))

    for _ in range(2):
        model = _build_gdn_model()
        model.proj = nn.Linear(4, 4)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        assert hub.bind(model) == {"gated_deltanet": 2}
        for layer in model.layers:
            assert layer.causal_conv1d_fn is conv
            assert layer.causal_conv1d_update is update
            assert layer.chunk_gated_delta_rule is _torch_chunk_stand_in
        assert model.state_dict().keys() == before.keys()
        for key, expected in before.items():
            torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
    assert calls == [("me/my-conv1d", "main", None)]


@pytest.mark.parametrize("strict", [False, True])
def test_prepare_rejects_kernels_without_hub_provenance(monkeypatch, strict):
    # [FUNCTION DOC] Test (parametrized over strict): when the loaded-kernels listing is empty, so
    # provenance cannot be established, strict prepare raises a runtime error ("cannot establish
    # Hub provenance"); non-strict prepare succeeds but binding a model yields an empty result.
    _stub_kernels(monkeypatch, modules=_all_hub_modules())
    sys.modules["kernels"].get_loaded_kernels = lambda: []
    args = _make_args(kernel_strict=strict)
    if strict:
        with pytest.raises(RuntimeError, match="cannot establish Hub provenance"):
            HubKernels.prepare(args)
    else:
        hub = HubKernels.prepare(args)
        assert hub.bind(_build_gdn_model()) == {}
