# [MODULE DOC] Top-k selection callables for DSA indexer scores. Provides a plain torch version, a
# flashinfer version (with determinism and tie-breaking controlled by sglang environment
# variables) and a factory choosing one by backend name. All take a scores tensor (last dimension =
# candidate keys) and a count, and return 32-bit indices of the same leading shape with minus one
# in place of any selected entry whose score is minus infinity (i.e. masked). flashinfer and
# sglang are imported lazily inside the functions that need them.
import torch


_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}


def torch_dsa_topk(logits: torch.Tensor, topk: int) -> torch.Tensor:
    # [FUNCTION DOC] Top-k over the last dimension using the torch built-in (descending score
    # order); converts indices to 32-bit and sets entries whose score is minus infinity to minus one.
    # Torch raises if topk exceeds the number of candidates.
    score, indices = torch.topk(logits, topk, dim=-1)
    indices = indices.to(torch.int32)
    return indices.masked_fill(score == -torch.inf, -1)


def flashinfer_dsa_topk(logits: torch.Tensor, topk: int) -> torch.Tensor:
    # [FUNCTION DOC] Top-k using flashinfer's kernel (imports flashinfer and sglang's environment
    # registry on call). Inputs with more than two dimensions are flattened to rows, and the result
    # is reshaped back to the leading shape with topk as last dimension. The kernel is invoked with
    # unsorted output (order of indices is not by score), the determinism flag read from the sglang
    # environment variable for deterministic flashinfer top-k, the tie-break value from the helper
    # below, and the graph-safe mode for DSA. Indices are converted to 32-bit and entries with
    # minus-infinity scores become minus one.
    import flashinfer
    from sglang.srt.environ import envs

    orig_shape = logits.shape
    if logits.dim() > 2:
        logits = logits.reshape(-1, logits.shape[-1])

    score, indices = flashinfer.top_k(
        logits,
        topk,
        sorted=False,
        deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
        tie_break=flashinfer_tie_break_value(),
        dsa_graph_safe=True,
    )
    indices = indices.to(torch.int32)
    indices = indices.masked_fill(score == -torch.inf, -1)
    if len(orig_shape) > 2:
        indices = indices.reshape(*orig_shape[:-1], topk)
    return indices


def get_dsa_topk_fn(topk_backend: str):
    # [FUNCTION DOC] Returns the top-k callable for the backend name "torch" or "flashinfer";
    # raises a value error for any other name.
    if topk_backend == "torch":
        return torch_dsa_topk
    if topk_backend == "flashinfer":
        return flashinfer_dsa_topk
    raise ValueError(f"Unsupported miles DSA topk backend: {topk_backend}")


def flashinfer_tie_break_value() -> int:
    # [FUNCTION DOC] Reads the sglang environment setting for flashinfer top-k tie breaking
    # (imports sglang's environment registry on call). Unset gives 0 (kernel default); otherwise the
    # value, compared case-insensitively, must be "small" (returns 1) or "large" (returns 2); any
    # other value raises a runtime error that reports the lower-cased value.
    from sglang.srt.environ import envs

    mode = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    if mode is None:
        return 0
    mode = mode.lower()
    if mode not in _FLASHINFER_TIE_BREAK_VALUES:
        raise RuntimeError(
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK must be one of "
            f"{tuple(_FLASHINFER_TIE_BREAK_VALUES)} or unset, got {mode!r}."
        )
    return _FLASHINFER_TIE_BREAK_VALUES[mode]
