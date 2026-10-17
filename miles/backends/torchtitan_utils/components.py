# [MODULE DOC] Custom torchtitan components used by the RL trainer: a data loader that yields
# nothing (data comes from rollouts, not a dataset) and a checkpoint manager that loads Hugging Face
# checkpoints even when the export omits keys such as a tied output head.
import logging
from dataclasses import dataclass

import torch.distributed.checkpoint as dcp
from torchtitan.components import checkpoint as titan_checkpoint
from torchtitan.components.dataloader import BaseDataLoader

logger = logging.getLogger(__name__)


class EmptyDataLoader(BaseDataLoader):
    # [CLASS DOC] Data loader with no data: iterating yields nothing and checkpoint state is empty.
    # It exists only to satisfy the trainer's data loader slot.
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        # [CLASS DOC] Configuration type for the empty loader; adds no fields to the base config.
        pass

    def __init__(self, config: Config, **kwargs):
        # [FUNCTION DOC] Stores the config; extra keyword arguments are accepted and ignored.
        self.config = config

    def __iter__(self):
        # [FUNCTION DOC] Returns an iterator over an empty tuple (no batches).
        return iter(())

    def state_dict(self):
        # [FUNCTION DOC] Returns an empty mapping (nothing to checkpoint).
        return {}

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC] Ignores the given state; does nothing.
        pass


class TiedCheckpointManager(titan_checkpoint.CheckpointManager):
    # [CLASS DOC] Torchtitan checkpoint manager whose Hugging Face loading tolerates exported keys
    # that the checkpoint does not contain (for example an output head tied to the embedding).
    # Everything else is inherited.
    @dataclass(kw_only=True, slots=True)
    class Config(titan_checkpoint.CheckpointManager.Config):
        # [CLASS DOC] Configuration type for this manager; adds no fields to the base config.
        pass

    def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
        # [FUNCTION DOC] Loads a checkpoint into the given state. When not loading from Hugging Face
        # format, delegates to the parent implementation and returns its result. Otherwise: asserts a
        # state-dict adapter exists; converts the state to Hugging Face naming; if the adapter has a
        # mapping of keys to files, drops from the target any converted key absent from that mapping
        # (logging how many and up to three examples) so the adapter's reverse conversion can
        # reconstruct them; reads the remaining tensors in place via distributed checkpoint load
        # from the adapter's Hugging Face storage reader for the checkpoint id (and the quantized
        # flag); then converts back to native naming with the adapter and loads it into the model
        # entry of the manager's states. Returns nothing on this path.
        if not from_hf:
            return super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)

        assert self.sd_adapter is not None
        hf_state = self.sd_adapter.to_hf(state_dict)
        if self.sd_adapter.fqn_to_index_mapping:
            available = set(self.sd_adapter.fqn_to_index_mapping)
            dropped = sorted(k for k in hf_state if k not in available)
            if dropped:
                logger.info(
                    f"HF checkpoint lacks {len(dropped)} exported key(s) (e.g. {dropped[:3]}); "
                    "deferring to the adapter's from_hf reconstruction"
                )
                hf_state = {k: v for k, v in hf_state.items() if k in available}

        dcp.load(
            hf_state,
            storage_reader=self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized),
        )
        self.states[titan_checkpoint.MODEL].load_state_dict(self.sd_adapter.from_hf(hf_state))
