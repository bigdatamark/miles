# [MODULE DOC] CPU test that the batch builder's host-side copy of cumulative sequence lengths
# equals the device tensor, for every context-parallel rank at cp sizes 1, 2, 4 and with the
# allgather-cp option off and on. CUDA is stubbed so tensors stay on CPU.
"""`get_batch`'s host copy of `cu_seqlens` must describe the same packed stream as the device tensor."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils


@pytest.mark.parametrize("allgather_cp", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_the_host_cu_seqlens_match_the_device_tensor(
    monkeypatch: pytest.MonkeyPatch, allgather_cp: bool, cp_size: int
):
    # [FUNCTION DOC] Test (parametrized over allgather option and cp size): four sequences of
    # lengths 13, 11, 5, 1 with response lengths half of that (integer division) and all-ones masks.
    # The cuda device query is patched to return the CPU device and tensor cuda-move to the identity.
    # For each cp rank, patches the parallel state in both the rollout and context-parallel
    # modules (tensor parallel size 1), builds one batch over the whole rollout (micro batch size
    # 4, pad multiplier 7, packed format) and asserts the host cumulative sequence lengths tuple
    # equals the device tensor's values.
    """The DSv4 row balancer plans from the host copy to avoid a device sync, so a drift would misroute rows."""
    lengths = [13, 11, 5, 1]
    rollout = {
        "tokens": [torch.arange(1, n + 1) for n in lengths],
        "loss_masks": [torch.ones(n // 2, dtype=torch.int) for n in lengths],
        "total_lengths": lengths,
        "response_lengths": [n // 2 for n in lengths],
    }
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    for cp_rank in range(cp_size):
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(rank=0, size=1))
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda state=state: state)
        monkeypatch.setattr(context_parallel, "get_parallel_state", lambda state=state: state)
        batch = data_utils.get_batch(
            data_utils.DataIterator(rollout, micro_batch_size=len(lengths)),
            list(rollout),
            pad_multiplier=7,
            qkv_format="thd",
            allgather_cp=allgather_cp,
        )
        assert batch["cu_seqlens_host"] == tuple(batch["cu_seqlens"].tolist())
