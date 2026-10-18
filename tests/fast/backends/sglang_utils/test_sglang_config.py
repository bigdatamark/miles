# [MODULE DOC]
# Tests for the SGLang deployment configuration resolver: parsing a YAML (from a file path or an
# inline base64 payload) or the legacy prefill-server-count shortcut into models with server
# groups, resolving per-engine GPU counts (group over model over args), validating keys and
# values, computing GPU offsets and engine (worker) offsets across groups and models (including
# placeholder groups, cross-node engines, the generated eval model), deciding which groups need
# offload and injecting the memory-saver override, validating the eval model, computing rollout
# placement offsets and the Megatron GPU span, and rejecting host/port overrides owned by the
# address allocator. Arguments are namespaces built by a local factory with 8-GPU defaults.
from __future__ import annotations

from argparse import Namespace

import pytest

from miles.backends.sglang_utils.sglang_config import (
    ServerGroupConfig,
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    resolve_sglang_config,
)
from miles.utils.external_utils.command_utils.common import encode_pseudo_file


def _make_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds an arguments namespace with defaults: no config file, no
    # prefill server count, 8 rollout GPUs at 1 per engine, 8 GPUs per node, no eval GPUs (1 per
    # eval engine), a fake checkpoint path, no offload, no debug modes, not colocated, 1 actor node
    # of 8 GPUs, no critic. Overrides replace defaults. Then derives "starts inference engines"
    # (unless given) as true unless train-only debug is on without eval GPUs, and, if train-only
    # debug is on, forces the rollout GPU count to 0. Returns the namespace.
    defaults = dict(
        sglang_config=None,
        prefill_num_servers=None,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        hf_checkpoint="/ckpt/actor",
        offload_rollout=False,
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        critic_num_nodes=0,
        critic_num_gpus_per_node=0,
        use_critic=False,
        critic_train_only=False,
    )
    defaults.update(overrides)
    defaults.setdefault("starts_inference_engines", not defaults["debug_train_only"] or defaults["eval_num_gpus"] > 0)
    if defaults["debug_train_only"]:
        defaults["rollout_num_gpus"] = 0
    return Namespace(**defaults)


def _resolve_yaml(tmp_path, yaml_text: str, **args_overrides):
    # [FUNCTION DOC] HELPER: Writes the YAML text to config.yaml in the temporary directory, builds
    # arguments pointing at that file with the given overrides, and returns the resolved SGLang
    # configuration (errors from resolution propagate).
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text)
    return resolve_sglang_config(_make_args(sglang_config=str(cfg_path), **args_overrides))


class TestNumGpusPerEnginePrecedence:
    # [CLASS DOC] Tests of how the per-engine GPU count is resolved (group, then model, then args).
    def test_group_level_value_wins_over_model_and_args(self, tmp_path):
        # [FUNCTION DOC] Model-level 4, group-level 2, args 1: the group resolves to 2.
        """A group's own num_gpus_per_engine beats the model-level and args-level defaults."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    num_gpus_per_engine: 4\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 2\n",
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=1,
        )
        assert cfg.models[0].server_groups[0].num_gpus_per_engine == 2

    def test_model_then_global_num_gpus_per_engine_defaults_are_resolved(self, tmp_path):
        # [FUNCTION DOC] Model "actor" with model-level 4 and model "ref" with none, args value 2
        # and 12 rollout GPUs: the actor's group resolves to 4 and the ref's group to 2.
        """A group without its own value takes the model-level value, or the args-level one when the model is silent."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    num_gpus_per_engine: 4\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: ref\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n",
            rollout_num_gpus=12,
            rollout_num_gpus_per_engine=2,
        )
        assert cfg.models[0].server_groups[0].num_gpus_per_engine == 4
        assert cfg.models[1].server_groups[0].num_gpus_per_engine == 2

    def test_a_group_asking_for_zero_gpus_per_engine_is_rejected_not_defaulted(self, tmp_path):
        # [FUNCTION DOC] A group with GPUs per engine 0 raises a value error "greater than 0"
        # instead of falling back to a default.
        """Falling back on any falsy value hid the typo and silently started engines with the wrong tp topology."""
        with pytest.raises(ValueError, match="greater than 0"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "        num_gpus_per_engine: 0\n",
                rollout_num_gpus=8,
                rollout_num_gpus_per_engine=1,
            )

    def test_a_model_asking_for_zero_gpus_per_engine_is_rejected_not_defaulted(self, tmp_path):
        # [FUNCTION DOC] A model-level GPUs per engine of 0 raises the same value error.
        """The model-level default took the same falsy fallback, so its groups silently inherited the args value."""
        with pytest.raises(ValueError, match="greater than 0"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    num_gpus_per_engine: 0\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n",
                rollout_num_gpus=8,
                rollout_num_gpus_per_engine=1,
            )


class TestModelPathConsistency:
    # [CLASS DOC] Test that all groups of one model must resolve to the same model path.
    def test_groups_for_one_model_cannot_resolve_to_different_model_paths(self, tmp_path):
        # [FUNCTION DOC] Model path /model/a with a second group overriding it to /model/b raises
        # an assertion error mentioning different model_path values.
        """One model serving two different checkpoints would silently mis-route, so it fails at resolve time."""
        with pytest.raises(AssertionError, match="different model_path values"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    model_path: /model/a\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 4\n"
                "      - worker_type: regular\n"
                "        num_gpus: 4\n"
                "        overrides:\n"
                "          model_path: /model/b\n",
                rollout_num_gpus=8,
            )


class TestResolvedServerGroupValidation:
    # [CLASS DOC] Tests of validation when constructing a resolved server-group config directly.
    def test_a_resolved_group_with_zero_gpus_is_rejected(self):
        # [FUNCTION DOC] Constructing a regular group with 0 GPUs raises a value error
        # "greater than 0".
        """A group reserving no GPUs cannot host an engine and must fail construction."""
        with pytest.raises(ValueError, match="greater than 0"):
            ServerGroupConfig(
                worker_type="regular",
                num_gpus=0,
                num_gpus_per_engine=2,
                gpu_offset=0,
                engine_offset=0,
                needs_offload=False,
            )

    def test_a_resolved_group_with_non_positive_gpus_per_engine_is_rejected(self):
        # [FUNCTION DOC] Constructing a regular group with 8 GPUs and -1 GPUs per engine raises a
        # value error "greater than 0".
        """A non-positive engine width would make the engine count division meaningless."""
        with pytest.raises(ValueError, match="greater than 0"):
            ServerGroupConfig(
                worker_type="regular",
                num_gpus=8,
                num_gpus_per_engine=-1,
                gpu_offset=0,
                engine_offset=0,
                needs_offload=False,
            )


class TestNumServerCells:
    # [CLASS DOC] Test of the model's server cell count.
    def test_non_placeholder_engine_cells_are_counted(self, tmp_path):
        # [FUNCTION DOC] Groups: regular 8 GPUs at 4 per engine (2 engines), prefill 4 GPUs at 2
        # (2 engines), placeholder 4 GPUs at 1 (not counted). The model's server cell count is 4.
        """The server cell count includes every engine except placeholder reservations."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 4\n"
            "      - worker_type: prefill\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n"
            "      - worker_type: placeholder\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n",
            rollout_num_gpus=16,
        )

        assert cfg.models[0].num_server_cells == 4


class TestOverridesResolution:
    # [CLASS DOC] Tests that per-group overrides are carried verbatim.
    def test_an_explicit_model_path_override_is_not_replaced(self, tmp_path):
        # [FUNCTION DOC] A group's model path override differing from the model-level path (the
        # model has one group, so no consistency conflict) is kept as given.
        """A group's own model_path override is parsed verbatim."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    model_path: /model/level/path\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        overrides:\n"
            "          model_path: /group/level/path\n",
            rollout_num_gpus=8,
        )
        assert cfg.models[0].server_groups[0].overrides["model_path"] == "/group/level/path"

    def test_unrelated_override_keys_pass_through(self, tmp_path):
        # [FUNCTION DOC] A static memory fraction override of 0.5 appears unchanged in the group's
        # overrides.
        """Arbitrary ServerArgs overrides are parsed verbatim."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        overrides:\n"
            "          mem_fraction_static: 0.5\n",
            rollout_num_gpus=8,
        )
        overrides = cfg.models[0].server_groups[0].overrides
        assert overrides["mem_fraction_static"] == 0.5


class TestYamlShapeValidation:
    # [CLASS DOC] Tests of structural validation of the YAML document.
    def test_engine_groups_is_accepted_as_an_alias_for_server_groups(self, tmp_path):
        # [FUNCTION DOC] A model using the alias key for its groups resolves with the group's
        # GPU count 8.
        """The documented engine_groups spelling keeps parsing."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n  - name: actor\n    engine_groups:\n      - worker_type: regular\n        num_gpus: 8\n",
            rollout_num_gpus=8,
        )
        assert cfg.models[0].server_groups[0].num_gpus == 8

    def test_a_yaml_without_the_sglang_key_is_rejected(self, tmp_path):
        # [FUNCTION DOC] A YAML whose top-level key is something else raises an assertion or value
        # error whose message mentions sglang or models.
        """A config missing the top-level sglang key fails loudly."""
        with pytest.raises((AssertionError, ValueError), match="sglang|models"):
            _resolve_yaml(tmp_path, "other_key:\n  - name: actor\n", rollout_num_gpus=8)

    def test_an_unknown_model_key_is_rejected(self, tmp_path):
        # [FUNCTION DOC] An unknown key at model level raises a value error naming the key.
        """Typos at the model level fail parsing instead of being silently dropped."""
        with pytest.raises(ValueError, match="typo_key"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    typo_key: 1\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n",
                rollout_num_gpus=8,
            )

    def test_an_unknown_server_group_key_is_rejected(self, tmp_path):
        # [FUNCTION DOC] An unknown key at server-group level raises a value error naming the key.
        """Typos at the server-group level fail parsing instead of dropping an intended SGLang override."""
        with pytest.raises(ValueError, match="typo_key"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "        typo_key: 1\n",
                rollout_num_gpus=8,
            )

    def test_giving_both_group_spellings_is_rejected(self, tmp_path):
        # [FUNCTION DOC] A model with both server-groups and engine-groups raises a value error
        # mentioning engine_groups.
        """server_groups plus engine_groups on one model is ambiguous and fails parsing."""
        with pytest.raises(ValueError, match="engine_groups"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "    engine_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n",
                rollout_num_gpus=8,
            )


class TestPrefillNumServersPath:
    # [CLASS DOC] Tests of the legacy path that builds prefill and decode groups from a prefill
    # server count instead of YAML.
    @pytest.mark.parametrize("multi_lora", [False, True])
    def test_prefill_num_servers_counts_engines_not_gpus(self, multi_lora):
        # [FUNCTION DOC] Parametrized over multi-LoRA off/on. 16 rollout GPUs, 3 prefill servers at
        # 2 GPUs per engine: groups are prefill with 6 GPUs then decode with 10; the model's
        # update-weights flag is true exactly when multi-LoRA is off.
        """prefill_num_servers is a server count, so its GPU span scales with the engine width."""
        cfg = resolve_sglang_config(
            _make_args(
                rollout_num_gpus=16, prefill_num_servers=3, rollout_num_gpus_per_engine=2, multi_lora=multi_lora
            )
        )
        groups = cfg.models[0].server_groups
        assert [(group.worker_type, group.num_gpus) for group in groups] == [("prefill", 6), ("decode", 10)]
        assert cfg.models[0].update_weights is not multi_lora

    def test_prefill_consuming_all_gpus_is_rejected(self):
        # [FUNCTION DOC] 4 rollout GPUs with 4 prefill servers at 1 GPU each leaves nothing for
        # decode: raises an assertion error "No decode GPUs".
        """prefill_num_servers leaving no decode gpus fails loudly."""
        args = _make_args(rollout_num_gpus=4, prefill_num_servers=4, rollout_num_gpus_per_engine=1)
        with pytest.raises(AssertionError, match="No decode GPUs"):
            resolve_sglang_config(args)


class TestGpuOffset:
    # [CLASS DOC] Test of the cumulative GPU offset of groups.
    def test_gpu_offsets_accumulate_across_groups_and_models_including_placeholders(self, tmp_path):
        # [FUNCTION DOC] Model actor: regular 4 GPUs, placeholder 4 GPUs; model ref: regular 8
        # GPUs. Offsets are [0, 4] for the actor groups and 8 for the ref group.
        """Each group's gpu_offset equals the num_gpus sum of all preceding groups, counting placeholders."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "      - worker_type: placeholder\n"
            "        num_gpus: 4\n"
            "  - name: ref\n"
            "    update_weights: false\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n",
            rollout_num_gpus=16,
        )
        assert [group.gpu_offset for group in cfg.models[0].server_groups] == [0, 4]
        assert cfg.models[1].server_groups[0].gpu_offset == 8


class TestEngineOffset:
    # [CLASS DOC] Tests of the engine (worker) offset numbering across groups and models.
    def test_engine_offsets_count_the_workers_of_every_preceding_group(self, tmp_path):
        # [FUNCTION DOC] 4 GPUs per node. Actor: regular 8 GPUs at 2 per engine (4 workers),
        # placeholder 4 GPUs at 4 per engine (1 worker); ref: 16 GPUs at 8 per engine. Asserts the
        # actor groups' engine offsets are [0, 4] and the ref group's is 5.
        """Groups of different engine widths contribute different worker counts, so a gpu offset alone cannot number them."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 2\n"
            "      - worker_type: placeholder\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 4\n"
            "  - name: ref\n"
            "    update_weights: false\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 16\n"
            "        num_gpus_per_engine: 8\n",
            rollout_num_gpus=28,
            num_gpus_per_node=4,
        )
        assert [group.engine_offset for group in cfg.models[0].server_groups] == [0, 4]
        assert cfg.models[1].server_groups[0].engine_offset == 5

    def test_a_group_whose_engine_spans_nodes_contributes_one_worker_per_node(self, tmp_path):
        # [FUNCTION DOC] 4 GPUs per node. First group 16 GPUs at 8 per engine (2 engines of 2 nodes
        # each, so 4 workers); second group 4 GPUs at 2 per engine. Engine offsets are [0, 4].
        """A cross-node engine is launched by one actor per node, so it consumes that many numbers, not one."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 16\n"
            "        num_gpus_per_engine: 8\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n",
            rollout_num_gpus=20,
            num_gpus_per_node=4,
        )
        assert [group.engine_offset for group in cfg.models[0].server_groups] == [0, 4]

    def test_prefill_and_decode_groups_are_numbered_in_that_order(self):
        # [FUNCTION DOC] Legacy path with 16 GPUs, 3 prefill servers at 2 GPUs per engine: the
        # prefill group's engine offset is 0 and the decode group's is 3.
        """The legacy --prefill-num-servers layout has no YAML to carry offsets, so the cursor must number it too."""
        cfg = resolve_sglang_config(
            _make_args(rollout_num_gpus=16, prefill_num_servers=3, rollout_num_gpus_per_engine=2)
        )

        assert [group.engine_offset for group in cfg.models[0].server_groups] == [0, 3]

    def test_the_generated_eval_model_is_numbered_after_every_rollout_engine(self):
        # [FUNCTION DOC] 8 rollout GPUs at 2 per engine (4 engines) plus 4 eval GPUs at 2 per
        # engine: the rollout model's group offset is 0 and the generated eval model's is 4.
        """The eval fleet is appended without YAML, and reusing the rollout numbers would clone their RNG streams."""
        cfg = resolve_sglang_config(
            _make_args(
                rollout_num_gpus=8,
                rollout_num_gpus_per_engine=2,
                eval_num_gpus=4,
                eval_num_gpus_per_engine=2,
            )
        )

        assert cfg.models[0].server_groups[0].engine_offset == 0
        assert cfg.models[1].server_groups[0].engine_offset == 4


class TestNeedsOffload:
    # [CLASS DOC] Tests of the per-group offload decision and the injected memory-saver override.
    @pytest.mark.parametrize("multi_lora", [False, True])
    def test_no_offload_flag_means_no_group_needs_offload(self, tmp_path, multi_lora):
        # [FUNCTION DOC] Parametrized over multi-LoRA off/on, rollout offload off: the group does
        # not need offload, no memory-saver override is injected, and the model's update-weights
        # flag is true exactly when multi-LoRA is off.
        """With offload_rollout off, no group offloads and no memory-saver override is injected."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n  - name: actor\n    server_groups:\n      - worker_type: regular\n        num_gpus: 8\n",
            rollout_num_gpus=8,
            multi_lora=multi_lora,
        )
        group = cfg.models[0].server_groups[0]
        assert group.needs_offload is False
        assert "enable_memory_saver" not in group.overrides
        assert cfg.models[0].update_weights is not multi_lora

    def test_groups_overlapping_megatron_offload_and_the_rest_disable_memory_saver(self, tmp_path):
        # [FUNCTION DOC] Colocated with offload on, Megatron using 8 GPUs, two 8-GPU groups: the
        # first (overlapping Megatron GPUs) needs offload and gets no memory-saver override; the
        # second does not need offload and gets enable-memory-saver false.
        """Only groups starting inside the megatron gpu range offload; later ones get enable_memory_saver=False."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n",
            rollout_num_gpus=16,
            offload_rollout=True,
            colocate=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
        )
        first, second = cfg.models[0].server_groups
        assert first.needs_offload is True
        assert "enable_memory_saver" not in first.overrides
        assert second.needs_offload is False
        assert second.overrides["enable_memory_saver"] is False

    def test_the_gpu_cursor_accumulates_across_models(self, tmp_path):
        # [FUNCTION DOC] Two 8-GPU models in a colocated offload setup with Megatron on 8 GPUs: the
        # first model's group needs offload; the second model's does not.
        """A later model's group starts where the previous model's gpus ended."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: ref\n"
            "    update_weights: false\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n",
            rollout_num_gpus=16,
            offload_rollout=True,
            colocate=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
        )
        assert cfg.models[0].server_groups[0].needs_offload is True
        assert cfg.models[1].server_groups[0].needs_offload is False

    def test_a_disaggregated_rollout_group_past_the_megatron_gpus_does_not_offload(self, tmp_path):
        # [FUNCTION DOC] Offload on but not colocated, Megatron using 8 GPUs, one 8-GPU group: the
        # group does not need offload and gets enable-memory-saver false.
        """Without colocate the rollout pg offset pushes the group past the megatron gpus, so it disables memory saver."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n  - name: actor\n    server_groups:\n      - worker_type: regular\n        num_gpus: 8\n",
            rollout_num_gpus=8,
            offload_rollout=True,
            colocate=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
        )
        group = cfg.models[0].server_groups[0]
        assert group.needs_offload is False
        assert group.overrides["enable_memory_saver"] is False

    def test_an_explicit_memory_saver_override_wins(self, tmp_path):
        # [FUNCTION DOC] Colocated offload with Megatron using 0 GPUs and a group overriding
        # enable-memory-saver to true: the group does not need offload and the override stays true.
        """A user-provided enable_memory_saver survives the conditional injection."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        overrides:\n"
            "          enable_memory_saver: true\n",
            rollout_num_gpus=8,
            offload_rollout=True,
            colocate=True,
            actor_num_nodes=0,
            actor_num_gpus_per_node=0,
        )
        group = cfg.models[0].server_groups[0]
        assert group.needs_offload is False
        assert group.overrides["enable_memory_saver"] is True


class TestYamlEvalModel:
    # [CLASS DOC] Tests of the eval model in YAML configurations.
    def test_a_yaml_without_an_eval_model_is_rejected_when_eval_gpus_are_requested(self, tmp_path):
        # [FUNCTION DOC] 2 eval GPUs requested but only a model named default in the YAML: raises
        # an assertion error "exactly one model named 'eval'".
        """--eval-num-gpus with no eval model in the YAML would silently run evals on the rollout fleet."""
        with pytest.raises(AssertionError, match="exactly one model named 'eval'"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: default\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 10\n",
                rollout_num_gpus=8,
                eval_num_gpus=2,
            )

    def test_a_yaml_eval_model_sized_differently_from_eval_num_gpus_is_rejected(self, tmp_path):
        # [FUNCTION DOC] 6 rollout and 4 eval GPUs requested (total 10), while the YAML has a
        # default model of 8 GPUs and an eval model of 2 (also total 10): the grand total matches
        # but the eval model's own size differs from the eval GPU count, so the same assertion
        # error is raised.
        """A matching grand total is not enough; the eval model itself must own exactly --eval-num-gpus."""
        with pytest.raises(AssertionError, match="exactly one model named 'eval'"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: default\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "  - name: eval\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 2\n",
                rollout_num_gpus=6,
                eval_num_gpus=4,
            )

    def test_yaml_eval_update_weights_is_not_overridden(self, tmp_path):
        # [FUNCTION DOC] An eval model explicitly setting update-weights true keeps true (the
        # eval default of false does not replace it).
        """An explicit update_weights in the YAML survives the eval model's False default."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: eval\n"
            "    update_weights: true\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 2\n",
            rollout_num_gpus=8,
            eval_num_gpus=2,
        )
        [eval_model] = [model for model in cfg.models if model.name == "eval"]
        assert eval_model.update_weights is True

    def test_yaml_eval_group_num_gpus_per_engine_wins_over_eval_cli(self, tmp_path):
        # [FUNCTION DOC] The eval group states 2 GPUs per engine while the eval CLI value is 4:
        # the group keeps 2.
        """A group that states its own engine width keeps it instead of taking --eval-num-gpus-per-engine."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: eval\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n",
            rollout_num_gpus=8,
            eval_num_gpus=4,
            eval_num_gpus_per_engine=4,
        )
        [eval_model] = [model for model in cfg.models if model.name == "eval"]
        assert eval_model.server_groups[0].num_gpus_per_engine == 2


class TestRolloutOffset:
    # [CLASS DOC] Tests for the private rollout placement offset computation.
    def test_debug_train_only_without_eval_fleet_has_zero_rollout_placement_offset(self):
        # [FUNCTION DOC] Train-only debug, not colocated, 2 actor nodes of 8 GPUs, no eval GPUs:
        # the offset is 0.
        args = _make_args(debug_train_only=True, colocate=False, actor_num_nodes=2, actor_num_gpus_per_node=8)
        assert _compute_rollout_offset(args) == 0

    def test_debug_train_only_places_the_eval_fleet_after_the_actor_bundles(self):
        # [FUNCTION DOC] Train-only debug with 2 actor nodes of 8 GPUs and 4 eval GPUs: the offset
        # is 16 (after all actor bundles).
        args = _make_args(
            debug_train_only=True, colocate=False, actor_num_nodes=2, actor_num_gpus_per_node=8, eval_num_gpus=4
        )
        assert _compute_rollout_offset(args) == 16

    def test_debug_rollout_only_has_zero_rollout_placement_offset(self):
        # [FUNCTION DOC] Rollout-only debug with 2 actor nodes of 8 GPUs: the offset is 0.
        """In rollout-only debug runs no megatron bundles are reserved ahead of the rollout ones."""
        args = _make_args(debug_rollout_only=True, colocate=False, actor_num_nodes=2, actor_num_gpus_per_node=8)
        assert _compute_rollout_offset(args) == 0


class TestMegatronNumGpus:
    # [CLASS DOC] Test for the private Megatron GPU span computation.
    def test_compute_megatron_num_gpus_for_critic_train_only(self):
        # [FUNCTION DOC] Critic-train-only with a 1-node 8-GPU actor and a 1-node 4-GPU critic:
        # the span is 4 (the critic's GPUs only).
        """With only the critic training, the megatron span is the critic's own gpus, not the actor's."""
        args = _make_args(
            critic_train_only=True,
            debug_rollout_only=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_megatron_num_gpus(args) == 4


class TestHostPortOverrideRejection:
    # [CLASS DOC] Tests that allocator-owned address fields cannot be overridden in any channel.
    def test_a_port_override_is_rejected_at_resolve_time(self, tmp_path):
        # [FUNCTION DOC] A group override of port raises an assertion error "must not override
        # host/port".
        """Overriding the allocator-owned port must fail fast instead of desyncing engine and controller."""
        with pytest.raises(AssertionError, match="must not override host/port"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "        overrides:\n"
                "          port: 12345\n",
                rollout_num_gpus=8,
            )

    def test_a_host_override_is_rejected_at_resolve_time(self, tmp_path):
        # [FUNCTION DOC] A group override of host raises the same assertion error.
        """Overriding the allocator-owned host must fail fast as well."""
        with pytest.raises(AssertionError, match="must not override host/port"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "        overrides:\n"
                "          host: 10.0.0.1\n",
                rollout_num_gpus=8,
            )

    def test_a_gated_launch_port_override_is_rejected_at_resolve_time(self, tmp_path):
        # [FUNCTION DOC] A group override of the gated launch port raises the same assertion error.
        """The launch gate port belongs to the allocator; an override would make the engine wait on the wrong socket."""
        with pytest.raises(AssertionError, match="must not override host/port"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "        overrides:\n"
                "          gated_launch_port: 13007\n",
                rollout_num_gpus=8,
            )

    def test_a_gated_launch_port_override_in_a_later_group_is_rejected(self, tmp_path):
        # [FUNCTION DOC] The forbidden override sits in the third group of the second model,
        # beside a valid override; resolution still raises the same assertion error.
        """The check runs per group, so a gate port hidden behind valid groups and valid keys still fails fast."""
        with pytest.raises(AssertionError, match="must not override host/port"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 4\n"
                "  - name: ref\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 2\n"
                "      - worker_type: regular\n"
                "        num_gpus: 2\n"
                "        overrides:\n"
                "          mem_fraction_static: 0.5\n"
                "          gated_launch_port: 13007\n",
                rollout_num_gpus=8,
            )

    def test_a_gated_launch_port_reaching_the_eval_fleet_from_the_cli_is_rejected(self, tmp_path):
        # [FUNCTION DOC] The arguments carry an eval gated-launch-port value (13007) while an eval
        # model exists with 2 GPUs: resolution raises the same assertion error.
        """An --eval-sglang-* gate port becomes an eval override, and that channel must be refused too."""
        with pytest.raises(AssertionError, match="must not override host/port"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 8\n"
                "  - name: eval\n"
                "    server_groups:\n"
                "      - worker_type: regular\n"
                "        num_gpus: 2\n",
                rollout_num_gpus=8,
                eval_num_gpus=2,
                eval_sglang_gated_launch_port=13007,
            )


class TestSglangConfigFileArg:
    # [CLASS DOC] Tests of the two accepted forms of the config argument: a file path or an inline
    # base64 payload. The class constant holds a one-model, one-regular-group, 8-GPU YAML text.
    _YAML = (
        "sglang:\n" "  - name: actor\n" "    server_groups:\n" "      - worker_type: regular\n" "        num_gpus: 8\n"
    )

    def test_a_path_is_still_read_from_disk(self, tmp_path):
        # [FUNCTION DOC] Resolving from a file path yields one model named actor.
        """The flag has always named a file, and every existing launch script still passes one."""
        cfg = _resolve_yaml(tmp_path, self._YAML)

        assert [model.name for model in cfg.models] == ["actor"]

    def test_an_inline_base64_payload_is_accepted(self, tmp_path):
        # [FUNCTION DOC] Passing the YAML encoded as a pseudo-file payload instead of a path also
        # yields one model named actor. The temporary path fixture is unused.
        """A test or a launcher that has no writable shared path must be able to inline the config."""
        cfg = resolve_sglang_config(_make_args(sglang_config=encode_pseudo_file(self._YAML)))

        assert [model.name for model in cfg.models] == ["actor"]
