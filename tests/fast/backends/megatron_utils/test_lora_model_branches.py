# [MODULE DOC]
# Mock-based tests for LoRA-related branching in the Megatron model module: the helper that wraps
# a single model into a list, the predicate deciding whether the forward pre-hook is disabled, the
# routing in the model-and-optimizer setup function between LoRA setup via the bridge and the
# ordinary model builder, and the routing in the save function between LoRA and regular
# checkpoint saving. All heavy collaborators are patched; no GPU is required.
"""Mock-based tests for LoRA branch logic in miles.backends.megatron_utils.model.

Validates that setup_model_and_optimizer, save, and save_hf_model correctly
route to LoRA-specific code paths depending on configuration — without GPU.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _ensure_model_list
# ---------------------------------------------------------------------------


class TestEnsureModelList:
    # [CLASS DOC] Tests for the private helper that guarantees a list of model chunks.
    def test_list_passthrough(self):
        # [FUNCTION DOC] A list input is returned as the very same list object (identity).
        from miles.backends.megatron_utils.model import _ensure_model_list

        models = [MagicMock(), MagicMock()]
        assert _ensure_model_list(models) is models

    def test_non_list_wrapped(self):
        # [FUNCTION DOC] A single non-list object is returned wrapped in a list whose first
        # element is that object.
        from miles.backends.megatron_utils.model import _ensure_model_list

        model = MagicMock()
        result = _ensure_model_list(model)
        assert isinstance(result, list)
        assert result[0] is model


# ---------------------------------------------------------------------------
# should_disable_forward_pre_hook
# ---------------------------------------------------------------------------


class TestShouldDisableForwardPreHook:
    # [CLASS DOC] Tests for the predicate that is true only when both the distributed optimizer
    # and overlapped parameter gather are enabled in the arguments.
    def test_both_true(self):
        # [FUNCTION DOC] Both flags true: the predicate returns true.
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = Namespace(use_distributed_optimizer=True, overlap_param_gather=True)
        assert should_disable_forward_pre_hook(args) is True

    def test_optimizer_false(self):
        # [FUNCTION DOC] Distributed optimizer off, overlap on: the predicate returns false.
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = Namespace(use_distributed_optimizer=False, overlap_param_gather=True)
        assert should_disable_forward_pre_hook(args) is False

    def test_overlap_false(self):
        # [FUNCTION DOC] Distributed optimizer on, overlap off: the predicate returns false.
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = Namespace(use_distributed_optimizer=True, overlap_param_gather=False)
        assert should_disable_forward_pre_hook(args) is False


# ---------------------------------------------------------------------------
# setup_model_and_optimizer — LoRA branch routing
# ---------------------------------------------------------------------------

_MODEL_MODULE = "miles.backends.megatron_utils.model"


class TestSetupModelAndOptimizerLoraBranch:
    # [CLASS DOC] Tests that the setup function routes to the LoRA bridge setup only for a
    # LoRA-enabled actor in bridge mode; optimizer and scheduler factories are mocked.
    """Verify that LoRA-enabled actor + bridge mode routes to _setup_lora_model_via_bridge."""

    def _make_args(self, lora_rank=32, role="actor", mode="bridge"):
        # [FUNCTION DOC] HELPER: Builds an arguments namespace with the given LoRA rank and
        # conversion mode (default 32, bridge), no adapter path, no custom provider, no upcycling,
        # optimizer enabled, a load path, and a full set of learning-rate and weight-decay
        # scheduler fields (constant learning rate 1e-5, batch size 32, 10 rollouts).
        # KNOWN ISSUE: the role parameter is accepted but never used.
        return Namespace(
            lora_rank=lora_rank,
            lora_adapter_path=None,
            custom_model_provider_path=None,
            megatron_to_hf_mode=mode,
            model_name=None,
            moe_use_upcycling=False,
            debug_disable_optimizer=False,
            stream_optimizer_state_to_disk=False,
            load="/some/path",
            pretrained_checkpoint=None,
            # optimizer fields
            num_rollout=10,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            global_batch_size=32,
            lr_decay_iters=None,
            lr_wsd_decay_iters=None,
            lr_warmup_fraction=None,
            lr_warmup_iters=0,
            lr_warmup_init=0,
            lr=1e-5,
            min_lr=0,
            lr_decay_style="constant",
            start_weight_decay=0,
            end_weight_decay=0,
            weight_decay_incr_style="constant",
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=False,
            lr_wsd_decay_style="linear",
            use_gloo_process_groups=False,
        )

    @patch(f"{_MODEL_MODULE}.get_optimizer_param_scheduler")
    @patch(f"{_MODEL_MODULE}.get_megatron_optimizer")
    @patch(f"{_MODEL_MODULE}._setup_lora_model_via_bridge")
    def test_lora_actor_bridge_routes_to_lora_setup(self, mock_lora_setup, mock_opt, mock_sched):
        # [FUNCTION DOC] LoRA rank 32, bridge mode, actor role. Asserts the LoRA bridge setup is
        # called exactly once with the arguments object.
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_lora_setup.return_value = [MagicMock()]
        mock_opt.return_value = MagicMock(param_groups=[])
        mock_sched.return_value = MagicMock()

        args = self._make_args(lora_rank=32, role="actor", mode="bridge")
        model, _, _ = setup_model_and_optimizer(args, role="actor")

        mock_lora_setup.assert_called_once_with(args)

    @patch(f"{_MODEL_MODULE}.get_optimizer_param_scheduler")
    @patch(f"{_MODEL_MODULE}.get_megatron_optimizer")
    @patch(f"{_MODEL_MODULE}.get_model")
    @patch(f"{_MODEL_MODULE}.get_model_provider_func")
    @patch(f"{_MODEL_MODULE}._setup_lora_model_via_bridge")
    def test_lora_critic_skips_lora_setup(self, mock_lora_setup, mock_provider, mock_get_model, mock_opt, mock_sched):
        # [FUNCTION DOC] LoRA rank 32 but the critic role. Asserts the LoRA bridge setup is not
        # called and the ordinary model builder is called once.
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_get_model.return_value = [MagicMock()]
        mock_opt.return_value = MagicMock(param_groups=[])
        mock_sched.return_value = MagicMock()

        args = self._make_args(lora_rank=32, role="critic", mode="bridge")
        setup_model_and_optimizer(args, role="critic")

        mock_lora_setup.assert_not_called()
        mock_get_model.assert_called_once()

    @patch(f"{_MODEL_MODULE}.get_optimizer_param_scheduler")
    @patch(f"{_MODEL_MODULE}.get_megatron_optimizer")
    @patch(f"{_MODEL_MODULE}.get_model")
    @patch(f"{_MODEL_MODULE}.get_model_provider_func")
    @patch(f"{_MODEL_MODULE}._setup_lora_model_via_bridge")
    def test_non_lora_skips_lora_setup(self, mock_lora_setup, mock_provider, mock_get_model, mock_opt, mock_sched):
        # [FUNCTION DOC] LoRA rank 0, actor role. Asserts the LoRA bridge setup is not called and
        # the ordinary model builder is called once.
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_get_model.return_value = [MagicMock()]
        mock_opt.return_value = MagicMock(param_groups=[])
        mock_sched.return_value = MagicMock()

        args = self._make_args(lora_rank=0, role="actor", mode="bridge")
        setup_model_and_optimizer(args, role="actor")

        mock_lora_setup.assert_not_called()
        mock_get_model.assert_called_once()

    @patch(f"{_MODEL_MODULE}.get_optimizer_param_scheduler")
    @patch(f"{_MODEL_MODULE}.get_megatron_optimizer")
    @patch(f"{_MODEL_MODULE}.get_model")
    @patch(f"{_MODEL_MODULE}._setup_lora_model_via_bridge")
    def test_non_inkling_lora_raw_mode_is_rejected(self, mock_lora_setup, mock_get_model, mock_opt, mock_sched):
        # [FUNCTION DOC] LoRA rank 32 with raw conversion mode and no model name. Asserts setup
        # raises an assertion error saying native LoRA injection is only implemented for Inkling,
        # and that neither the LoRA bridge setup nor the ordinary model builder was called.
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_get_model.return_value = [MagicMock()]
        mock_opt.return_value = MagicMock(param_groups=[])
        mock_sched.return_value = MagicMock()

        args = self._make_args(lora_rank=32, role="actor", mode="raw")
        with pytest.raises(AssertionError, match="Native LoRA injection is only implemented for Inkling"):
            setup_model_and_optimizer(args, role="actor")

        mock_lora_setup.assert_not_called()
        mock_get_model.assert_not_called()


# ---------------------------------------------------------------------------
# save — LoRA vs regular branch
# ---------------------------------------------------------------------------


class TestSaveLoRaBranch:
    # [CLASS DOC] Tests for the save function's choice between LoRA and regular checkpoint
    # saving; hash saving, pre-hook toggling and arguments retrieval are all patched out.
    @patch(f"{_MODEL_MODULE}.save_model_hashes")
    @patch(f"{_MODEL_MODULE}.enable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.disable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.should_disable_forward_pre_hook", return_value=False)
    @patch(f"{_MODEL_MODULE}.get_args")
    @patch(f"{_MODEL_MODULE}.save_checkpoint_with_lora")
    @patch(f"{_MODEL_MODULE}.is_lora_model", return_value=True)
    def test_lora_model_calls_lora_save(
        self, mock_is_lora, mock_save_lora, mock_get_args, mock_should, mock_disable, mock_enable, mock_save_hashes
    ):
        # [FUNCTION DOC] Model detected as LoRA: saving iteration 42 calls the LoRA save once.
        from miles.backends.megatron_utils.model import save

        model = [MagicMock()]
        save(42, model, MagicMock(), MagicMock())

        mock_save_lora.assert_called_once()

    @patch(f"{_MODEL_MODULE}.save_model_hashes")
    @patch(f"{_MODEL_MODULE}.enable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.disable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.should_disable_forward_pre_hook", return_value=False)
    @patch(f"{_MODEL_MODULE}.get_args")
    @patch(f"{_MODEL_MODULE}.save_checkpoint")
    @patch(f"{_MODEL_MODULE}.is_lora_model", return_value=False)
    def test_non_lora_model_calls_regular_save(
        self, mock_is_lora, mock_save_ckpt, mock_get_args, mock_should, mock_disable, mock_enable, mock_save_hashes
    ):
        # [FUNCTION DOC] Model not detected as LoRA: saving iteration 42 calls the regular
        # checkpoint save once.
        from miles.backends.megatron_utils.model import save

        model = [MagicMock()]
        save(42, model, MagicMock(), MagicMock())

        mock_save_ckpt.assert_called_once()
