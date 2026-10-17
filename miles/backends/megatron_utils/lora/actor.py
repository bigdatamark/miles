# [MODULE DOC]
# Multi-adapter LoRA train actor: a specialization of the Megatron train actor that keeps several
# LoRA adapters ("slots") in one model, each with its own optimizer, and exposes slot-level
# operations (forward-backward, forward-only, optimizer step, load, save, export, unload) used by
# a Tinker-style training service. Operations delegate to the LoRA model and checkpoint modules.
from contextlib import ExitStack

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.lora import checkpoint as lora_checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.optimizer import SlotOptimizer
from miles.backends.training_utils.data.rollout import get_rollout_data
from miles.utils.lora.utils import AdapterSpec
from miles.utils.object_store import StoreObjectRef
from miles.utils.tracking_utils.structured_log import with_logs


class MultiLoRATrainRayActor(MegatronTrainRayActor):
    # [CLASS DOC]
    # Train actor whose weight updates are replaced by per-slot adapter publishing. State: a
    # mapping from slot number to that slot's optimizer wrapper, created on slot load and removed
    # on unload. Other behavior (init, sleep, wake) is inherited.
    def _init_training_state(self) -> None:
        # [FUNCTION DOC] HELPER: Override of the base hook: starts with no slot optimizers and
        # creates a snapshot publisher but no weight updater (weights are not pushed to engines by
        # the regular path; adapters are exported per slot instead).
        self.slot_optimizers: dict[int, SlotOptimizer] = {}
        self._init_weight_updater_and_publisher(update_weights=False, publish_snapshots=True)

    @with_logs
    def forward_backward(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        # [FUNCTION DOC] Bumps heartbeat, fetches the batch's rollout data from the object store
        # (kept open until return), and runs the LoRA forward-and-backward pass for the batch id on
        # the model, accumulating gradients per slot. Returns the result mapping from the LoRA
        # model module (per-datum losses/metrics).
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref)
            stack.enter_context(store_get_result)
            return lora_model.run_forward_backward(self.args, batch_id, self.model, rollout_data)

    @with_logs
    def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        # [FUNCTION DOC] Bumps heartbeat and applies an optimizer step for the slots named in the
        # input mapping (slot number to Adam hyperparameters) using the slot optimizers. Returns the
        # per-slot result mapping produced by the LoRA model module.
        self._heartbeat.bump()
        return lora_model.optim_step(self.slot_optimizers, adam_params_by_slot)

    @with_logs
    def forward_only(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        # [FUNCTION DOC] Same as forward-and-backward (heartbeat bump, fetch rollout data, run the
        # LoRA pass) but with the forward-only flag so no gradients are accumulated. Returns the
        # result mapping with the requested per-datum loss.
        # KNOWN ISSUE: shadows the base class's forward-only method of the same name but with a
        # different signature and return type; base-class code calling the inherited one would
        # break on this actor.
        """Same loss pass as forward_backward, without the backward: the Tinker
        forward() contract returns the requested loss per datum."""
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref)
            stack.enter_context(store_get_result)
            return lora_model.run_forward_backward(self.args, batch_id, self.model, rollout_data, forward_only=True)

    @with_logs
    def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        # [FUNCTION DOC] Activates an adapter slot with the given rank and alpha: creates the slot's
        # adapter parameters and optimizer through the LoRA model module and records the optimizer
        # by slot. If a checkpoint path is given, loads the slot's weights from it (and its optimizer
        # state only when load_optimizer is true, the default).
        self.slot_optimizers[slot] = lora_model.load_slot(self.args, self.model, slot, rank, alpha)
        if ckpt_path is not None:
            lora_checkpoint.load_slot(self.model, self.slot_optimizers[slot], ckpt_path, load_optimizer)

    @with_logs
    def save_slot(self, slot: int, path: str, metadata: dict | None = None) -> None:
        # [FUNCTION DOC] Saves the slot's adapter weights and optimizer state, plus optional metadata,
        # to the path via the LoRA checkpoint module. Raises a key error for an unknown slot.
        lora_checkpoint.save_slot(self.model, self.slot_optimizers[slot], path, metadata=metadata)

    @with_logs
    def export_slot(self, slot: int, rank: int, alpha: float, path: str, metadata: dict | None = None) -> None:
        # [FUNCTION DOC] Bumps heartbeat, asserts a snapshot publisher exists, and has it publish
        # the adapter described by (slot, rank, alpha) to the path with optional metadata.
        # Collective over ranks as the publisher gathers adapter weights.
        """Write the slot's adapter as an engine-loadable dir."""
        self._heartbeat.bump()
        assert self.snapshot_publisher is not None, "adapter export requires a snapshot publisher"
        self.snapshot_publisher.publish_adapter(
            AdapterSpec(slot=slot, rank=rank, alpha=alpha), path, metadata=metadata
        )

    @with_logs
    def unload_slot(self, slot: int) -> dict | None:
        # [FUNCTION DOC] Removes the slot's optimizer from the mapping (key error if unknown),
        # unloads the slot's adapter from the model, and returns none.
        slot_optimizer = self.slot_optimizers.pop(slot)
        lora_model.unload_slot(self.model, slot_optimizer)
        return None
