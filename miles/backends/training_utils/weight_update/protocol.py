# [MODULE DOC]
# The contract every weight transfer protocol (trainer to rollout engines) implements, plus the
# factory that picks the concrete protocol from args. The updater drives a protocol through:
# connect (once per topology change), begin_sync, send_bucket (per bucket, on sender ranks only),
# after_base_weights, finalize, after_engines_resumed, then pop_metrics. Concrete protocols are
# imported lazily inside the factory to avoid importing heavy optional dependencies up front.
"""Transfer protocol contract and factory."""

from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from typing import ClassVar

import torch

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


class WeightTransferProtocol(ABC):
    # [CLASS DOC] Abstract base. Class-level declarations: required gather placement (default: no
    # pipeline gather, tensor and expert gathered), whether LoRA adapters are supported (default
    # false), whether the updater should wrap the sync in a weight update session (default true),
    # and (instance-level default false) whether a LoRA sync must also re-push the base weights.
    # Instance state: args, the rollout engine clients (set by connect), whether this rank sends
    # (set by connect; none before), a collective group name (default "miles") and a mapping of
    # accumulated update metrics drained by pop_metrics.
    """Moves HF-named weight buckets from training ranks to rollout engines.

    ``connect`` makes every pairing decision once: it sets ``is_sender`` and
    whatever send channels the protocol needs. The updater then drives
    ``send_bucket`` on sender ranks only; streamed adapter tensors are ordinary
    bucket entries (``{lora_name}:{hf_key}`` names).
    """

    required_placement: ClassVar[WeightUpdatePlacement] = WeightUpdatePlacement(gather_pp=False)
    supports_lora: ClassVar[bool] = False
    use_weight_update_session: ClassVar[bool] = True
    needs_base_resync_for_lora: bool = False

    def __init__(self, args: Namespace) -> None:
        # [FUNCTION DOC] Store args and initialize the instance state described at class level.
        self.args = args
        self.rollout_engines: Sequence[SGLangApiClient] | None = None
        self.is_sender: bool | None = None
        self.group_name = "miles"
        self.update_weight_metrics: dict[str, float] = {}

    @abstractmethod
    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        # [FUNCTION DOC] Abstract. Establish the topology for this sync generation: engine clients,
        # optional per-engine GPU counts and offsets (none means defaults), the trainer's parallel
        # state, the gather placement actually used, and the selector string naming which engine
        # runners participate. Must set the sender flag and any channels (collective groups, IPC
        # groups) the protocol needs. May be collective across training ranks.
        ...

    def begin_sync(
        self,
        weight_version: int,
        iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]],
    ) -> bool:
        # [FUNCTION DOC] Optional hook run before the session frame opens, with the new weight
        # version and the bucket-iterator factory. Returns whether to proceed; default true. A false
        # result skips the round, and must be the same on every rank.
        """Hook before the session frame; return False to skip this round.
        The return value must be identical on every rank."""
        return True

    @abstractmethod
    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # [FUNCTION DOC] Abstract. Deliver one bucket of (HF name, tensor) pairs to the engines; called
        # on sender ranks only. Implementations may free or clear the bucket afterward.
        ...

    def after_base_weights(self) -> None:  # noqa: B027 — optional hook
        # [FUNCTION DOC] Optional no-op hook after the base-weight stream completes.
        """Hook after the base-weight stream completes (e.g. await in-flight writes)."""

    def finalize(self, weight_version: int) -> None:  # noqa: B027 — optional hook
        # [FUNCTION DOC] Optional no-op hook after all buckets are sent, with the weight version.
        """Hook after all sends (e.g. publish + engine reload)."""

    def after_engines_resumed(self) -> None:  # noqa: B027 — optional hook
        # [FUNCTION DOC] Optional no-op hook once engines have applied all buckets and resumed.
        """Hook once the engines have applied every bucket and resumed generation."""

    def pop_metrics(self) -> dict[str, float]:
        # [FUNCTION DOC] Return the accumulated metrics mapping and replace it with a fresh empty one
        # (drain semantics).
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics


def get_weight_transfer_protocol(args: Namespace) -> WeightTransferProtocol:
    # [FUNCTION DOC] Factory. Reads the transfer mode from args (default "broadcast"); raises a value
    # error unless it is one of broadcast, broadcast_packed, p2p, disk-delta. broadcast_packed
    # additionally requires the Megatron train backend and a non-colocated setup (value error
    # otherwise). Selection: colocated runs always use the CUDA-IPC tensor protocol regardless of
    # mode (even p2p or disk-delta); otherwise broadcast and broadcast_packed use the distributed
    # NCCL protocol, disk-delta the disk delta protocol, p2p the peer-to-peer protocol. The
    # protocol class is imported lazily; returns a new instance built from args. The final raise is
    # unreachable.
    mode = getattr(args, "update_weight_transfer_mode", "broadcast")
    if mode not in ("broadcast", "broadcast_packed", "p2p", "disk-delta"):
        raise ValueError(f"Unknown --update-weight-transfer-mode {mode!r}")
    if mode == "broadcast_packed" and (getattr(args, "train_backend", None) != "megatron" or args.colocate):
        raise ValueError("broadcast_packed requires Megatron non-colocated weight transfer")
    if args.colocate:
        from miles.backends.training_utils.weight_update.protocols.cuda_ipc import UpdateWeightFromTensor

        return UpdateWeightFromTensor(args)
    if mode in ("broadcast", "broadcast_packed"):
        from miles.backends.training_utils.weight_update.protocols.broadcast import UpdateWeightFromDistributed

        return UpdateWeightFromDistributed(args)
    if mode == "disk-delta":
        from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

        return UpdateWeightFromDiskDelta(args)
    if mode == "p2p":
        from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P

        return UpdateWeightP2P(args)
    raise ValueError(f"Unknown --update-weight-transfer-mode {mode!r}")
