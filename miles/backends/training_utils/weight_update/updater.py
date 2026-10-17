# [MODULE DOC]
# Single class driving trainer-to-engine weight synchronization independent of the training backend
# and transfer protocol. It combines three collaborators: a transfer protocol (how bytes move), an
# HF weight iterator (what is sent, in size-bounded buckets, built against the protocol's required
# placement) and a connection status manager (when to reconnect). A sync runs on every training
# rank in lockstep: global rank 0 acts as the "driver" for engine-side session RPCs; sender ranks
# transmit buckets; non-sender ranks only join the gathers.
"""Backend-neutral weight-update driver.

The updater owns the lifecycle: it builds the HF weight iterator against the
protocol's required placement, runs the engine session frame, streams base
buckets (senders transmit, other ranks join the gathers), and orchestrates
LoRA adapter pushes.
"""

import logging
import random
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.conn_status import ConnStatusManager
from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.session import (
    begin_weight_update,
    end_weight_update,
    pause_engines,
    register_lora_adapter,
    resume_engines,
    set_weight_version,
)
from miles.backends.training_utils.weight_update.utils import record_lora_checksums
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora.utils import LORA_ADAPTER_NAME
from miles.utils.timer import timer

if TYPE_CHECKING:
    from miles.ray.rollout.inference_controller import UpdatableEngines

logger = logging.getLogger(__name__)


class WeightUpdater:
    # [CLASS DOC] Public entry point for weight sync. Holds args, the parallel state, the protocol
    # chosen by args, the connection status manager, the HF weight iterator, a weights getter, the
    # current weight version (starts at 0, incremented once per executed sync), LoRA settings and the
    # set of adapter names already registered on the current engines (cleared on every connect).
    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        *,
        weights_getter: Callable[[], Mapping[str, torch.Tensor] | None],
        model_name: str,
        quantization_config: dict | None,
        iterator_factory: Callable,
        parallel_state: ParallelState,
        is_lora: bool,
        lora_sync_config: dict | None = None,
    ) -> None:
        # [FUNCTION DOC] Creates the protocol from args and asserts it supports LoRA if is_lora is
        # set (message names the transfer mode). Builds the iterator by calling the factory with
        # args, model, the protocol's required placement (keyword), model name and quantization
        # config. The weights getter returns backend-native named weights or none (live model).
        # If is_lora, a LoRA sync config mapping is required (assertion error otherwise).
        self.args = args
        self.parallel_state = parallel_state
        self.protocol = get_weight_transfer_protocol(args)
        self.conn_status = ConnStatusManager()
        assert (
            not is_lora or self.protocol.supports_lora
        ), f"LoRA weight sync is not supported for {args.update_weight_transfer_mode!r} weight transfer."
        self._hf_weight_iterator = iterator_factory(
            args,
            model,
            required_placement=self.protocol.required_placement,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self.weights_getter = weights_getter
        self.weight_version = 0
        self.is_lora = is_lora
        if is_lora:
            assert lora_sync_config is not None
        self._lora_sync_config = lora_sync_config
        self._registered_adapters: set[str] = set()

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        # [FUNCTION DOC] Connects the protocol to the engines with the iterator's placement and
        # weight update selector; asserts the protocol set its sender flag; clears the set of
        # registered adapters (new engines know no adapters). May be collective, depending on the
        # protocol.
        self.protocol.connect(
            rollout_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            self.parallel_state,
            self._hf_weight_iterator.placement,
            self._hf_weight_iterator.weight_update_selector,
        )
        assert self.protocol.is_sender is not None, "connect() must set is_sender"
        self._registered_adapters.clear()

    def reconnect_if_needed(self, info: "UpdatableEngines") -> bool:
        # [FUNCTION DOC] Asks the status manager whether the engines' snapshot cell id to hash mapping
        # differs, the trainer is stale or no connection was made; if so reconnects and returns true,
        # else returns false. Must be called on all ranks consistently since reconnect is collective.
        if not self.conn_status.needs_reconnect(info.snapshot_cell_id_to_hashes):
            return False
        self.reconnect(info)
        return True

    def reconnect(self, info: "UpdatableEngines") -> None:
        # [FUNCTION DOC] Connects using the engines, GPU counts and offsets in the info, records the
        # reconnect (and the snapshot hashes) in the status manager, and barriers all ranks on the
        # Gloo group.
        self.connect_rollout_engines(
            info.rollout_engines,
            engine_gpu_counts=info.engine_gpu_counts,
            engine_gpu_offsets=info.engine_gpu_offsets,
        )
        self.conn_status.mark_reconnected(info.snapshot_cell_id_to_hashes)
        dist.barrier(group=get_gloo_group())

    def verify_engine_version(self, rollout_engines: Sequence[SGLangApiClient]) -> None:
        # [FUNCTION DOC] CI check. Does nothing if there are no engines or no sync has happened yet
        # (version 0). Otherwise queries one randomly chosen engine's weight version (blocking) and
        # raises a runtime error if its string form differs from the updater's version string. Only
        # one engine is sampled, so it is a spot check.
        if not rollout_engines or self.weight_version == 0:
            return
        engine = random.choice(list(rollout_engines))
        engine_version = async_utils.run(engine.get_weight_version())
        if str(engine_version) != str(self.weight_version):
            raise RuntimeError(f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_version}")

    def pop_metrics(self) -> dict[str, float]:
        # [FUNCTION DOC] Delegates to the protocol's drain of accumulated metrics.
        """Return and clear the protocol's metrics; the actor drains them onto the step log."""
        return self.protocol.pop_metrics()

    @torch.no_grad()
    def update_weights(self) -> None:
        # [FUNCTION DOC] Runs without autograd; collective on all training ranks. Steps: (1) protocol
        # begin-sync hook with the next version number and the base bucket iterator factory; if it
        # returns false return without changing the version; else increment the version. (2) Base
        # weights are synced unless this is LoRA and the protocol does not need a base resync. The
        # adapters to push come from the adapter helper. (3) If the protocol uses the session frame,
        # rank 0 (the driver) pauses the engines, registers new adapters, and opens a weight update
        # session on the iterator's selector; then all ranks barrier. (4) When LoRA checksum checking
        # is on, builds a per-adapter checksum mapping (asserting the iterator gathers pipeline). (5)
        # Inside a timer: senders show a progress bar; iterate the iterator's buckets (base weights
        # if enabled, adapters, materializing only on senders so others merely join the gathers);
        # on senders the driver records LoRA checksums, then the protocol sends the bucket; after the
        # stream call the protocol's after-base-weights hook and barrier. (6) Inside a timer: the
        # protocol finalize hook with the version; with sessions, the driver ends the session
        # (passing checksums), sets the engine weight version and resumes engines; barrier; then the
        # protocol's after-engines-resumed hook. KNOWN ISSUE: the docstring omits the begin-sync
        # skip path; checksums are recorded only on the driver, which is only guaranteed to hold the
        # full adapter when the sender and driver coincide (hence the assertion).
        """Run one weight sync: session frame + base-bucket stream + adapter pushes for LoRA."""
        protocol = self.protocol
        if not protocol.begin_sync(self.weight_version + 1, self._iter_base_buckets):
            return
        self.weight_version += 1

        sync_base = not self.is_lora or protocol.needs_base_resync_for_lora
        adapters = self._get_updated_adapters()

        driver = dist.get_rank() == 0
        if protocol.use_weight_update_session and driver:
            pause_engines(self.args, protocol.rollout_engines)
            self._register_new_lora_adapters(protocol.rollout_engines, adapters)
            begin_weight_update(
                protocol.rollout_engines, self._hf_weight_iterator.weight_update_selector, sync_base=sync_base
            )
        dist.barrier(group=get_gloo_group())

        checksums = {name: {} for name, _ in adapters} if self.is_lora and self.args.check_lora_weight_equal else None
        if checksums is not None:
            assert (
                self._hf_weight_iterator.placement.gather_pp
            ), "the LoRA checksum manifest is recorded on one rank, which must hold the full adapter"
        with timer("update_weights_implementation"):
            pbar = tqdm(desc=f"[{protocol.group_name}] Update weights", total=0) if protocol.is_sender else None
            for bucket in self._hf_weight_iterator.iter_hf_weights(
                self.weights_getter(),
                include_base=sync_base,
                adapters=adapters,
                materialize=protocol.is_sender,
            ):
                if protocol.is_sender:
                    if driver and checksums is not None:
                        record_lora_checksums(bucket, checksums)
                    protocol.send_bucket(bucket)
                    pbar.update(1)
            protocol.after_base_weights()
            dist.barrier(group=get_gloo_group())

        with timer("finalize_and_resume_engines"):
            protocol.finalize(self.weight_version)
            if protocol.use_weight_update_session and driver:
                end_weight_update(protocol.rollout_engines, expected_lora_checksums=checksums)
                set_weight_version(protocol.rollout_engines, self.weight_version)
                resume_engines(protocol.rollout_engines)
            dist.barrier(group=get_gloo_group())
        protocol.after_engines_resumed()

    def _iter_base_buckets(self, *, materialize: bool):
        # [FUNCTION DOC] HELPER: Bucket iterator factory handed to the protocol's begin-sync: base
        # weights only (no adapters) from the current weights getter, with the given materialize flag.
        return self._hf_weight_iterator.iter_hf_weights(self.weights_getter(), materialize=materialize)

    def _get_updated_adapters(self) -> list[tuple[str, object]]:
        # [FUNCTION DOC] HELPER: Empty list when not LoRA; otherwise a single pair of the default
        # LoRA adapter name and none (meaning the live adapter).
        """``(lora_name, adapter_or_None)`` pairs for this sync; the push set is
        identical on every rank so the iterator's collectives align."""
        if not self.is_lora:
            return []
        return [(LORA_ADAPTER_NAME, None)]

    def _register_new_lora_adapters(self, rollout_engines, adapters: list[tuple[str, object]]) -> None:
        # [FUNCTION DOC] HELPER: Called on the driver only. For each adapter name not yet registered:
        # config = the LoRA sync config, extended with the adapter's rank ("r") and alpha
        # ("lora_alpha") when an adapter object is given; registers it on every engine (raising on
        # failure) and records the name as registered.
        """Register adapters the current engine set has not seen, with their
        per-adapter config; eager so the engine validates rank before any bytes move."""
        for lora_name, adapter in adapters:
            if lora_name in self._registered_adapters:
                continue
            config = self._lora_sync_config
            if adapter is not None:
                config = config | {"r": adapter.rank, "lora_alpha": adapter.alpha}
            register_lora_adapter(rollout_engines, lora_name=lora_name, lora_config=config)
            self._registered_adapters.add(lora_name)
