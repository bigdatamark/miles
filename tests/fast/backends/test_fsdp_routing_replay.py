# [MODULE DOC] CPU tests for the FSDP routing-replay adaptation module and a few related helpers:
# discovery of MoE router modules by class name with global layer indices, installation through
# registered adapters (stream index, once-per-layer calls, disabled manager no-op, errors for no
# matching adapter or no layers), adapter registration for every supported model type, enabling and
# log-prob stage selection driven by arguments, the stage context manager (set, restore, nest), a
# source-level check that reference model creation does not install replay, and the layer-count
# resolver for flat and nested text configs. Uses fake router/dense modules and resets the global
# manager and adapter registry around every test. Registered as a CPU test in the first CPU stage.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch.nn as nn

from miles.backends.fsdp_utils.adaptations import routing_replay
from miles.utils.arguments import resolve_fsdp_num_layers
from miles.utils.replay_base import routing_replay_manager


class _FakeRouter(nn.Module):
    # [CLASS DOC] HELPER: Identity module whose class name is what the discovery looks for.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input unchanged.
        return x


class _FakeDense(nn.Module):
    # [CLASS DOC] HELPER: Identity module standing in for a dense (non-MoE) gate.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input unchanged.
        return x


def _model_with_layers(kinds):
    # [FUNCTION DOC] HELPER: Builds a nested fake model (model.model.layers) from a list of layer
    # kinds; each layer has an mlp whose gate is a fake router for the MoE kind and a fake dense
    # module otherwise. Returns the outer model.
    """kinds[i] == 'moe' builds a layer whose .mlp.gate is a _FakeRouter; 'dense' builds _FakeDense."""
    layers = nn.ModuleList()
    for kind in kinds:
        mlp = nn.Module()
        mlp.gate = _FakeRouter() if kind == "moe" else _FakeDense()
        layer = nn.Module()
        layer.mlp = mlp
        layers.append(layer)
    inner = nn.Module()
    inner.layers = layers
    model = nn.Module()
    model.model = inner
    return model


@pytest.fixture(autouse=True)
def _reset_manager():
    # [FUNCTION DOC] HELPER: Autouse fixture. Saves the adapter registry, resets the global manager
    # (disabled, no replays, no current) before the test; afterwards restores the registry contents
    # in place and resets the manager again, including setting the stage back to fallthrough.
    saved_adapters = list(routing_replay._ADAPTERS)
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    yield
    routing_replay._ADAPTERS[:] = saved_adapters
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    routing_replay_manager.stage = routing_replay.FALLTHROUGH


def test_discover_returns_global_layer_index_skipping_dense():
    # [FUNCTION DOC] Test: with one dense layer followed by four MoE layers, discovery returns
    # entries with global layer indices 1 to 4 (dense layer skipped, index not renumbered).
    model = _model_with_layers(["dense"] + ["moe"] * 4)
    found = routing_replay.discover_moe_modules(model, "_FakeRouter")
    assert [idx for idx, _ in found] == [1, 2, 3, 4]


def test_discover_is_sorted_by_layer_index():
    # [FUNCTION DOC] Test: with twelve MoE layers discovery returns indices 0 to 11 in numeric order
    # (not lexicographic).
    model = _model_with_layers(["moe"] * 12)
    found = routing_replay.discover_moe_modules(model, "_FakeRouter")
    assert [idx for idx, _ in found] == list(range(12))


def test_discover_finds_layers_behind_an_extra_wrapper():
    # [FUNCTION DOC] Test: when the model is nested under an extra language-model wrapper,
    # discovery still finds layers 0, 1, 2.
    model = _model_with_layers(["moe"] * 3)
    outer = nn.Module()
    outer.language_model = model
    assert [idx for idx, _ in routing_replay.discover_moe_modules(outer, "_FakeRouter")] == [0, 1, 2]


def test_install_assigns_stream_idx_equal_to_global_layer_index():
    # [FUNCTION DOC] Test: with the manager enabled and a matching adapter registered, installing on
    # a model of dense, moe, moe returns a count of two and the manager's replays have stream
    # indices 1 and 2 (global layer indices).
    routing_replay_manager.enabled = True
    routing_replay.register_routing_replay_adapter(
        routing_replay.RoutingReplayAdapter(
            name="fake", applies_to=lambda cfg: True, module_cls_name="_FakeRouter", install=lambda m: None
        )
    )
    model = _model_with_layers(["dense", "moe", "moe"])

    count = routing_replay.install(model, SimpleNamespace(model_type="fake"))

    assert count == 2
    assert [r.stream_idx for r in routing_replay_manager.replays] == [1, 2]


def test_install_calls_the_adapter_once_per_moe_layer():
    # [FUNCTION DOC] Test: an adapter keyed on model type records each module passed to its install
    # callback; for one dense and three MoE layers it is called three times, each with a fake router.
    routing_replay_manager.enabled = True
    installed = []
    routing_replay.register_routing_replay_adapter(
        routing_replay.RoutingReplayAdapter(
            name="fake_counted",
            applies_to=lambda cfg: getattr(cfg, "model_type", None) == "fake_counted",
            module_cls_name="_FakeRouter",
            install=installed.append,
        )
    )
    model = _model_with_layers(["dense", "moe", "moe", "moe"])

    routing_replay.install(model, SimpleNamespace(model_type="fake_counted"))

    assert len(installed) == 3
    assert all(isinstance(m, _FakeRouter) for m in installed)


def test_install_is_a_noop_when_manager_disabled():
    # [FUNCTION DOC] Test: with the manager disabled, install returns zero and adds no replays even
    # though a matching adapter and MoE layers exist.
    routing_replay_manager.enabled = False
    routing_replay.register_routing_replay_adapter(
        routing_replay.RoutingReplayAdapter(
            name="fake_off", applies_to=lambda cfg: True, module_cls_name="_FakeRouter", install=lambda m: None
        )
    )
    model = _model_with_layers(["moe", "moe"])

    assert routing_replay.install(model, SimpleNamespace(model_type="fake_off")) == 0
    assert routing_replay_manager.replays == []


def test_install_raises_when_no_adapter_matches():
    # [FUNCTION DOC] Test: with only an adapter that does not apply to the config's model type,
    # install raises a value error mentioning that no routing-replay adapter exists.
    routing_replay_manager.enabled = True
    routing_replay.register_routing_replay_adapter(
        routing_replay.RoutingReplayAdapter(
            name="picky",
            applies_to=lambda cfg: getattr(cfg, "model_type", None) == "something_else",
            module_cls_name="_FakeRouter",
            install=lambda m: None,
        )
    )
    with pytest.raises(ValueError, match="no routing-replay adapter"):
        routing_replay.install(_model_with_layers(["moe"]), SimpleNamespace(model_type="unknown_arch"))


def test_install_raises_when_adapter_matches_but_finds_no_layers():
    # [FUNCTION DOC] Test: an adapter that applies but whose module class name is absent from the
    # model makes install raise a value error saying it found no MoE layers.
    routing_replay_manager.enabled = True
    routing_replay.register_routing_replay_adapter(
        routing_replay.RoutingReplayAdapter(
            name="empty",
            applies_to=lambda cfg: getattr(cfg, "model_type", None) == "empty",
            module_cls_name="_NotPresent",
            install=lambda m: None,
        )
    )
    with pytest.raises(ValueError, match="found no MoE layers"):
        routing_replay.install(_model_with_layers(["moe"]), SimpleNamespace(model_type="empty"))


def test_specs_register_adapters_for_every_supported_model_type():
    # [FUNCTION DOC] Test: importing the specs module (side-effect registration) makes the
    # resolver return an adapter for qwen3_moe (router class Qwen3MoeTopKRouter), qwen3_5_moe_text
    # (Qwen3_5MoeTopKRouter) and glm4_moe_lite (Glm4MoeLiteMoE).
    import miles.backends.fsdp_utils.adaptations.specs  # noqa: F401

    expected = {
        "qwen3_moe": "Qwen3MoeTopKRouter",
        "qwen3_5_moe_text": "Qwen3_5MoeTopKRouter",
        "glm4_moe_lite": "Glm4MoeLiteMoE",
    }
    for model_type, module_cls_name in expected.items():
        adapter = routing_replay.resolve_routing_replay_adapter(SimpleNamespace(model_type=model_type))
        assert adapter is not None, model_type
        assert adapter.module_cls_name == module_cls_name


def test_dense_archs_do_not_resolve_to_a_moe_adapter():
    # [FUNCTION DOC] Test: after importing the specs, dense model types qwen3_5_text and qwen3
    # resolve to no adapter.
    import miles.backends.fsdp_utils.adaptations.specs  # noqa: F401

    for model_type in ("qwen3_5_text", "qwen3"):
        assert routing_replay.resolve_routing_replay_adapter(SimpleNamespace(model_type=model_type)) is None


def test_enable_follows_use_routing_replay():
    # [FUNCTION DOC] Test: enable returns true when the use-routing-replay arg is set and false when
    # not.
    assert routing_replay.enable(Namespace(use_routing_replay=True, ci_test=False)) is True
    assert routing_replay.enable(Namespace(use_routing_replay=False, ci_test=False)) is False


def test_enable_turns_on_the_replay_check_only_under_ci_test():
    # [FUNCTION DOC] Test: enabling with the ci-test arg set turns the manager's replay-result check
    # on; enabling again with it unset turns the check off.
    routing_replay.enable(Namespace(use_routing_replay=True, ci_test=True))
    assert routing_replay_manager.enable_check_replay_result is True

    routing_replay.enable(Namespace(use_routing_replay=True, ci_test=False))
    assert routing_replay_manager.enable_check_replay_result is False


def test_log_prob_stage_is_fallthrough_when_disabled():
    # [FUNCTION DOC] Test: with the manager disabled, the log-prob stage is fallthrough even when
    # rollout routing replay is requested.
    routing_replay_manager.enabled = False
    assert routing_replay.log_prob_stage(Namespace(use_rollout_routing_replay=True)) == routing_replay.FALLTHROUGH


def test_log_prob_stage_replays_when_routing_came_from_the_rollout():
    # [FUNCTION DOC] Test: enabled manager plus the rollout-routing arg gives the replay-forward
    # stage.
    routing_replay_manager.enabled = True
    assert routing_replay.log_prob_stage(Namespace(use_rollout_routing_replay=True)) == routing_replay.REPLAY_FORWARD


def test_log_prob_stage_records_for_the_non_rollout_variant():
    # [FUNCTION DOC] Test: enabled manager without the rollout-routing arg gives the record stage
    # (nothing has been filled to replay yet).
    # --use-routing-replay alone has nothing to replay yet: the queues are only filled by fill(),
    # which is gated on use_rollout_routing_replay. Replaying here would pop an empty queue.
    routing_replay_manager.enabled = True
    assert routing_replay.log_prob_stage(Namespace(use_rollout_routing_replay=False)) == routing_replay.RECORD


def test_fill_is_skipped_for_the_non_rollout_variant():
    # [FUNCTION DOC] Test: fill with the rollout-routing arg off and all other arguments empty does
    # nothing and does not raise.
    # Passing None for every downstream argument: reaching fill_replay_data would raise.
    routing_replay.fill(Namespace(use_rollout_routing_replay=False), None, None, None, None)


def test_stage_sets_and_restores():
    # [FUNCTION DOC] Test: inside the stage context the manager has the requested stage; on exit the
    # previous stage is restored.
    routing_replay_manager.stage = routing_replay.REPLAY_BACKWARD
    with routing_replay.stage(routing_replay.REPLAY_FORWARD):
        assert routing_replay_manager.stage == routing_replay.REPLAY_FORWARD
    assert routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD


def test_stage_restores_on_exception():
    # [FUNCTION DOC] Test: an exception raised inside the stage context propagates and the previous
    # stage is still restored.
    routing_replay_manager.stage = routing_replay.REPLAY_BACKWARD
    with pytest.raises(RuntimeError):
        with routing_replay.stage(routing_replay.REPLAY_FORWARD):
            raise RuntimeError("boom")
    assert routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD


def test_stage_nests():
    # [FUNCTION DOC] Test: nested stage contexts each restore their immediately previous stage,
    # ending back at fallthrough.
    routing_replay_manager.stage = routing_replay.FALLTHROUGH
    with routing_replay.stage(routing_replay.REPLAY_BACKWARD):
        with routing_replay.stage(routing_replay.REPLAY_FORWARD):
            assert routing_replay_manager.stage == routing_replay.REPLAY_FORWARD
        assert routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD
    assert routing_replay_manager.stage == routing_replay.FALLTHROUGH


def test_ref_model_creation_does_not_install_routing_replay():
    # [FUNCTION DOC] Test: a textual check that the source of the actor's reference-model creation
    # method does not mention the routing replay install call.
    import inspect

    from miles.backends.fsdp_utils.actor import FSDPTrainRayActor

    source = inspect.getsource(FSDPTrainRayActor._create_ref_model)
    assert "routing_replay.install" not in source


def test_resolve_num_layers_from_flat_config():
    # [FUNCTION DOC] Test: a config with a hidden-layer count returns that count (48).
    cfg = SimpleNamespace(num_hidden_layers=48)
    assert resolve_fsdp_num_layers(cfg) == 48


def test_resolve_num_layers_unwraps_text_config():
    # [FUNCTION DOC] Test: a config with the count only inside a text sub-config returns it (40).
    cfg = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=40))
    assert resolve_fsdp_num_layers(cfg) == 40


def test_resolve_num_layers_prefers_text_config_when_both_present():
    # [FUNCTION DOC] Test: when both top-level (1) and text sub-config (40) counts exist, the text
    # sub-config count wins.
    cfg = SimpleNamespace(num_hidden_layers=1, text_config=SimpleNamespace(num_hidden_layers=40))
    assert resolve_fsdp_num_layers(cfg) == 40


def test_resolve_num_layers_uses_get_text_config_when_available():
    # [FUNCTION DOC] Test: a config exposing a text-config getter uses its result (47) over the
    # top-level count (1).
    text = SimpleNamespace(num_hidden_layers=47)
    cfg = SimpleNamespace(num_hidden_layers=1, get_text_config=lambda: text)
    assert resolve_fsdp_num_layers(cfg) == 47


def test_resolve_num_layers_falls_back_when_text_config_lacks_depth():
    # [FUNCTION DOC] Test: a text sub-config lacking the count falls back to the top-level count
    # (32).
    cfg = SimpleNamespace(num_hidden_layers=32, text_config=SimpleNamespace())
    assert resolve_fsdp_num_layers(cfg) == 32


def test_resolve_num_layers_returns_none_when_absent():
    # [FUNCTION DOC] Test: a config with no layer count anywhere resolves to none.
    assert resolve_fsdp_num_layers(SimpleNamespace()) is None
