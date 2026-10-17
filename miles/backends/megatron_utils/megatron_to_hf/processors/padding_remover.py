# [MODULE DOC] HELPER:
# Removes vocabulary padding rows from embedding and output-head tensors after Megatron export.
import torch

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: If the parameter name (after stripping any leading "module." prefixes)
    # is exactly the word embedding weight or the output layer weight, returns a view of the first
    # vocab_size rows (axis 0); otherwise returns the tensor unchanged. If vocab_size is larger
    # than the row count the full tensor is returned. Pure; no copy.
    """
    Remove vocab padding: param[:vocab_size] for embedding/output layers, else unchanged.
    """
    if strip_param_name_prefix(name) in {"embedding.word_embeddings.weight", "output_layer.weight"}:
        return param[:vocab_size]
    return param
