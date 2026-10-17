# [MODULE DOC] Glue for routing replay on Megatron: distributes recorded per-layer MoE routing
# data to the replay recorders that belong to the MoE layers this rank owns.
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def register_replay_list_moe(replay_list, replay_data, *, models, **_kwargs):
    # [FUNCTION DOC] Finds which global layer indices are MoE layers among those built on this
    # rank, then feeds the replay data of each such layer to the matching replay recorder in
    # order. Inputs: a list of replay recorders (one per local MoE layer, in layer order), replay
    # data indexed by (sample/token position, global layer index, ...), and keyword-only model
    # chunks (unwrapped through their inner module to read the config); extra keyword arguments are
    # ignored. For each chunk (the chunk index is the virtual stage) it computes the layer offset
    # and number of layers to build, and for each global layer in that range: an integer MoE layer
    # frequency keeps layers whose index is divisible by it; a list frequency must have length
    # equal to the total layer count (assertion error otherwise) and keeps layers whose entry is
    # nonzero; any other type keeps all layers. Then recorder number i records the slice of replay
    # data at the i-th kept layer. Mutates the recorders; returns nothing. Raises an index error if
    # there are fewer recorders than kept layers.
    # KNOWN ISSUE: an integer frequency of zero would cause a division-by-zero error; extra
    # recorders beyond the kept layers are silently left unused.
    """Map replay streams to Megatron MoE layers using the local model layout."""
    layer_indices = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)

    for replay_idx, layer_idx in enumerate(layer_indices):
        layer_data = replay_data[:, layer_idx]
        replay_list[replay_idx].record(layer_data)
