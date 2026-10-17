# [MODULE DOC] Adapter that lets torchtitan's pipeline/step machinery drive miles' RL loss. The
# adapter is "armed" with a list of microbatches and a closure, and torchtitan calls it like a loss
# function with the model output and a label tensor; the labels are filled with the microbatch index
# so the adapter can recover which microbatch the output belongs to. It also handles vocab-sharded
# logits and context-parallel gathering of the sequence dimension.
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import all_gather_single_autograd
from torch.distributed.tensor import DTensor
from torchtitan.components.loss import BaseLoss
from torchtitan.distributed.context_parallel import cp_shard


def _gather_over_mesh(tensor: torch.Tensor, mesh) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Differentiable all-gather of the tensor along its first dimension over
    # the process group of the given device mesh; returns the concatenated tensor.
    return all_gather_single_autograd(tensor, 0, mesh.get_group())


class RLLossAdapter(BaseLoss):
    # [CLASS DOC] Loss object used by the torchtitan trainer. Lifecycle: set context parallelism and
    # gradient scale once; before each pass call arm with the microbatch list, the closure and the
    # training flag; the schedule then calls the object once per microbatch; afterwards call
    # collect to get per-microbatch results in order. State: the config, armed batches and closure,
    # training flag (default true), results keyed by microbatch index, context-parallel mesh and
    # balancer type (default head-tail) with a cache of restore orderings keyed by sequence length,
    # and the gradient scale (default 1.0).
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        # [CLASS DOC] Configuration type for the adapter; adds no fields to the base config.
        pass

    def __init__(self, config: Config, *, compile_config=None):
        # [FUNCTION DOC] Stores the config and initializes the empty state described for the class.
        # The compile configuration is accepted and ignored. Does not call the parent constructor.
        self.config = config
        self._batches: list | None = None
        self._closure: Callable | None = None
        self._is_training = True
        self._results: dict[int, object] = {}
        self._cp_mesh = None
        self._cp_balancer = "headtail"
        self._cp_restore: dict[int, torch.Tensor] = {}
        self._grad_scale = 1.0

    def set_context_parallel(self, mesh, balancer_type: str) -> None:
        # [FUNCTION DOC] Records the context-parallel device mesh and load-balancer type and clears
        # the cache of restore orderings.
        self._cp_mesh = mesh
        self._cp_balancer = balancer_type
        self._cp_restore = {}

    def set_gradient_scale(self, scale: float) -> None:
        # [FUNCTION DOC] Sets the factor that multiplies the training loss before backward (used to
        # average over data-parallel and context-parallel ranks).
        self._grad_scale = scale

    def _restore_indices(self, seq_len: int, device) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns (cached per sequence length) the index tensor that puts a
        # context-parallel-gathered sequence back into original token order. Computed by sharding
        # the position numbers zero..seq_len-1 with the same context-parallel sharding function and
        # balancer used for the model input, gathering each rank's local positions over the mesh,
        # and taking the argsort of the gathered order.
        cached = self._cp_restore.get(seq_len)
        if cached is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            (local,), _ = cp_shard(self._cp_mesh, (positions,), None, self._cp_balancer)
            order = _gather_over_mesh(local.flatten(), self._cp_mesh)
            cached = order.argsort()
            self._cp_restore[seq_len] = cached
        return cached

    def arm(self, batches: list, closure: Callable, *, is_training: bool) -> None:
        # [FUNCTION DOC] Prepares for one pass: stores the microbatches, the closure and the training
        # flag, and clears previous results. In training mode the closure maps (logits, batch) to
        # (loss, log dictionary); in evaluation mode it maps (logits, batch) to a result object.
        self._batches, self._closure, self._is_training = batches, closure, is_training
        self._results = {}

    def collect(self) -> list:
        # [FUNCTION DOC] Returns the stored results ordered by microbatch index. Raises a runtime
        # error naming the microbatches that were never run. Fails with a type error if never armed.
        missing = [i for i in range(len(self._batches)) if i not in self._results]
        if missing:
            raise RuntimeError(f"the schedule never ran microbatch(es) {missing}")
        return [self._results[i] for i in range(len(self._batches))]

    def _gather_context_parallel(self, pred: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Reassembles full-sequence logits from context-parallel shards. The
        # input has the sequence on the second dimension; swaps the first two dimensions to gather
        # along the sequence over the mesh (differentiable), swaps back, then reorders the sequence
        # dimension with the restore indices for the gathered length. Returns the reordered tensor.
        gathered = _gather_over_mesh(pred.transpose(0, 1).contiguous(), self._cp_mesh)
        gathered = gathered.transpose(0, 1)
        restore = self._restore_indices(gathered.shape[1], gathered.device)
        return gathered.index_select(1, restore)

    def __call__(self, pred, target, global_valid_tokens=None, **kwargs):
        # [FUNCTION DOC] Called by the torchtitan schedule per microbatch with the model output, the
        # label tensor (filled with the microbatch index) and ignored extras (valid-token count and
        # keyword arguments). If the output is a distributed tensor, every placement must be a shard
        # along the last (vocabulary) dimension, else a runtime error; it is converted to the local
        # shard. With context parallelism, gathers the full sequence first. The microbatch index is
        # the first element of the labels. Training mode: runs the closure on logits and that
        # microbatch, stores the log dictionary under the index, and returns the loss times the
        # gradient scale with an empty dictionary. Evaluation mode: stores the closure result and
        # returns a float32 zero scalar on the logits' device with an empty dictionary.
        if isinstance(pred, DTensor):
            for placement in pred.placements:
                if not (placement.is_shard() and placement.dim in (pred.ndim - 1, -1)):
                    raise RuntimeError(
                        f"expected vocab-sharded logits (Shard({pred.ndim - 1})), got {pred.placements}"
                    )
            pred = pred.to_local()

        if self._cp_mesh is not None:
            pred = self._gather_context_parallel(pred)

        index = int(target.flatten()[0])
        batch = self._batches[index]
        if self._is_training:
            loss, log_dict = self._closure(pred, batch)
            self._results[index] = log_dict
            return loss * self._grad_scale, {}
        self._results[index] = self._closure(pred, batch)
        return torch.zeros((), device=pred.device, dtype=torch.float32), {}
