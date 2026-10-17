# [MODULE DOC]
# Backend of the Tinker gateway for Miles: converts internal datums into the trainer's batch data
# format and calls the distributed trainer (slot load/unload/save/export, forward, forward-backward,
# optimizer step), and converts sampling requests into SGLang generate requests sent through the
# router. Module constant: DATUM_TO_BATCH_KEYS maps internal datum keys (weights, advantages,
# sampling log-probs) to trainer batch keys (loss weights, advantages, rollout log-probs).
"""Translate gateway datums to trainer batches and sampling requests to SGLang."""

import asyncio

import httpx

from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC
from miles.tinker.core.types import UserInputError
from miles.utils import object_store
from miles.utils.http_utils import post
from tinker.types.sample_response import MASK_LOGPROB

# internal datum key -> trainer batch key
DATUM_TO_BATCH_KEYS = {"weights": "loss_weights", "advantages": "advantages", "sampling_logprobs": "rollout_log_probs"}


def _pad_to_dp_multiple(slot_datums: list, dp_size: int) -> list:
    # [FUNCTION DOC] HELPER: Input is a list of (slot, datum) pairs. If its length is a multiple of
    # the data-parallel size, return it unchanged. Otherwise append copies of the last pair (same
    # slot, a shallow copy of its datum flagged as padding; the same copy object repeated) until
    # the length is a multiple. Does not mutate the input list. Padding datums get a zero loss mask
    # later.
    """Equalize DP shares with zero-mask datums; their outputs are dropped after the loss pass."""
    remainder = len(slot_datums) % dp_size
    if remainder == 0:
        return slot_datums
    slot, datum = slot_datums[-1]
    filler = dict(datum) | {"padding": True}
    return slot_datums + [(slot, filler)] * (dp_size - remainder)


def _build_train_data(slot_datums: list) -> dict:
    # [FUNCTION DOC] HELPER: Build the trainer batch mapping of parallel per-datum lists from
    # (slot, datum) pairs: tokens, target tokens, loss masks (zeros for padding datums, ones
    # otherwise, one entry per target position), response lengths (the target lengths, so label
    # positions including prompt targets count as response), total lengths (token counts),
    # sample indices (0..n-1), adapter slots, and a dynamic global batch size equal to the datum
    # count. Optional per-datum fields (weights, advantages, sampling log-probs) are copied into
    # their batch keys if the first datum has them (assumes all datums agree). Pure function.
    """Miles response_lengths select label positions here, including prompt targets."""
    datums = [datum for _, datum in slot_datums]
    train_data = {
        "tokens": [datum["tokens"] for datum in datums],
        "target_tokens": [datum["target_tokens"] for datum in datums],
        "loss_masks": [[0 if datum.get("padding") else 1] * datum["target_len"] for datum in datums],
        "response_lengths": [datum["target_len"] for datum in datums],
        "total_lengths": [len(datum["tokens"]) for datum in datums],
        "sample_indices": list(range(len(datums))),
        "adapter_slots": [slot for slot, _ in slot_datums],
        "dynamic_global_batch_size": len(datums),
    }
    for datum_key, batch_key in DATUM_TO_BATCH_KEYS.items():
        if datum_key in datums[0]:
            train_data[batch_key] = [datum[datum_key] for datum in datums]
    return train_data


class MilesBackend:
    # [CLASS DOC]
    # Implements the backend interface the Tinker service expects, on top of a trainer handle
    # (async methods over a group of workers), the SGLang router url, and the trainer's
    # data-parallel size (batches are padded to a multiple of it). Failure convention for slot
    # operations: none on success, or the first worker's error mapping.
    def __init__(self, trainer, router_url: str, dp_size: int = 1) -> None:
        # [FUNCTION DOC] Store the trainer handle, router url and data-parallel size (default 1).
        self.trainer = trainer
        self.router_url = router_url
        self.dp_size = dp_size

    async def trainer_dead(self) -> bool:
        # [FUNCTION DOC] True if any trainer cell has errored (asks the trainer).
        return await self.trainer.has_errored_cell()

    async def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> dict | None:
        # [FUNCTION DOC] Ask all trainer workers to load the adapter slot with the rank and alpha,
        # optionally from a checkpoint path, with or without optimizer state (default with; the flag
        # only matters when a path is given). Returns none on success, else the first non-null
        # worker result (an error mapping).
        return _slot_failure(
            await self.trainer.load_slot(slot, rank, alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer)
        )

    async def unload_slot(self, slot: int) -> dict | None:
        # [FUNCTION DOC] Ask the trainer workers to unload the slot; none on success, else the first
        # worker error mapping.
        return _slot_failure(await self.trainer.unload_slot(slot))

    async def forward_backward(
        self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict
    ) -> list[dict] | dict:
        # [FUNCTION DOC] Run a training forward-backward pass (gradients accumulate on the
        # trainer) over (slot, datum) pairs with a loss function name and config. Returns one
        # output per input datum (loss and per-position log-probs) or an error mapping.
        return await self._execute_batch("forward_backward", batch_id, slot_datums, loss_fn, loss_fn_config)

    async def forward_only(
        self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict
    ) -> list[dict] | dict:
        # [FUNCTION DOC] Same as forward-backward but evaluation only (no gradient accumulation).
        return await self._execute_batch("forward_only", batch_id, slot_datums, loss_fn, loss_fn_config)

    async def _execute_batch(
        self, method: str, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict
    ) -> list[dict] | dict:
        # [FUNCTION DOC] HELPER: Pad the pairs to a data-parallel multiple, build the batch mapping,
        # add the loss function name and config, and call the named trainer method with the batch
        # id and the data placed in the object store. Collect worker results: the first worker
        # result containing an error key is returned as is; otherwise for each datum output (keyed by
        # its sample index, first worker reporting an index wins since data-parallel workers may
        # duplicate) keep the loss as float and log-probs converted from array to list. Returns the
        # outputs for indexes 0..n-1 of the original, unpadded datums in order (a missing index
        # raises a key error), dropping padding outputs.
        train_data = _build_train_data(_pad_to_dp_multiple(slot_datums, self.dp_size))
        train_data["loss_fn"] = loss_fn
        train_data["loss_fn_config"] = loss_fn_config
        worker_results = await self._call_trainer(method, batch_id, train_data)
        by_index: dict[int, dict] = {}
        for worker_result in worker_results:
            if "error" in worker_result:
                return worker_result
            for datum_output in worker_result["per_datum"]:
                index = int(datum_output["sample_index"])
                if index not in by_index:
                    by_index[index] = {
                        "loss": float(datum_output["loss"]),
                        "logprobs": datum_output["logprobs"].tolist(),
                    }
        return [by_index[index] for index in range(len(slot_datums))]

    async def _call_trainer(self, method: str, batch_id: int, train_data: dict) -> list:
        # [FUNCTION DOC] HELPER: Put the batch mapping into the shared object store (using the
        # rollout data value spec), call the named trainer method with the batch id and data
        # reference, return the per-worker results, and always remove the stored object afterwards.
        store = object_store.get_instance()
        data_ref = store.put(value=train_data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        try:
            return await getattr(self.trainer, method)(batch_id=batch_id, data_ref=data_ref)
        finally:
            store.remove(data_ref)

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        # [FUNCTION DOC] Run one optimizer step on the trainer for the slots given by the mapping
        # from slot to Adam parameters. Returns the first worker's mapping from slot to its metrics
        # or error mapping (workers are assumed to agree).
        worker_results = await self.trainer.optim_step(adam_params_by_slot=adam_params_by_slot)
        return worker_results[0]

    async def save_slot(self, slot: int, path: str, metadata: dict | None = None) -> dict | None:
        # [FUNCTION DOC] Ask the trainer to save the slot's training state (adapter and optimizer)
        # to the directory with optional metadata; none on success, else the first error mapping.
        return _slot_failure(await self.trainer.save_slot(slot=slot, path=path, metadata=metadata))

    async def export_slot(
        self, slot: int, rank: int, alpha: float, path: str, metadata: dict | None = None
    ) -> dict | None:
        # [FUNCTION DOC] Ask the trainer to export the slot as an inference adapter (with rank and
        # alpha) to the directory with optional metadata; none on success, else the first error
        # mapping.
        return _slot_failure(
            await self.trainer.export_slot(slot=slot, rank=rank, alpha=alpha, path=path, metadata=metadata)
        )

    # -------- sampling --------

    async def sample(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        # [FUNCTION DOC]
        # Sample from the engines through the router (network I/O). Builds one generate request
        # (may raise a user input error for a missing token limit) and sends num_samples concurrent
        # copies, each with its own seed offset when a seed is pinned. A transport or HTTP error
        # returns an error mapping instead of raising. If any sequence converts to an error
        # mapping (engine abort) that mapping is returned. Otherwise returns a mapping with the
        # list of sequences and, when requested, prompt log-probs and top-k prompt log-probs, both
        # derived from the first response only (all samples share the prompt).
        request = self._generate_request(payload, lora_name, lora_path)
        try:
            responses = await asyncio.gather(
                *[
                    post(f"{self.router_url}/generate", _with_sample_seed(request, index))
                    for index in range(payload["num_samples"])
                ]
            )
        except httpx.HTTPError as error:
            return {"error": str(error)}
        sequences = [_to_sequence(response) for response in responses]
        for sequence in sequences:
            if "error" in sequence:
                return sequence
        result = {"sequences": sequences}
        if payload["prompt_logprobs"]:
            result["prompt_logprobs"] = _prompt_logprobs(responses[0])
        if payload["topk_prompt_logprobs"]:
            result["topk_prompt_logprobs"] = _topk_prompt_logprobs(responses[0], payload["topk_prompt_logprobs"])
        return result

    def _generate_request(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        # [FUNCTION DOC]
        # HELPER: Translate the gateway sampling payload to an SGLang generate request mapping.
        # The max tokens parameter is required (user input error). Sampling params: max new tokens,
        # temperature (default 1.0), top-p (default 1.0), top-k (default -1, disabled), and the
        # seed if not null. Stop handling: null leaves defaults; an empty list disables every stop
        # token including end of sequence (ignore eos); a list whose first element is an integer
        # becomes stop token ids; anything else becomes stop strings. The request has the prompt
        # token ids as input ids and asks for log-probs; if prompt log-probs or top-k prompt
        # log-probs are requested, log-prob start length 0 is added, and the top-k count is added
        # when positive. If an adapter name is given it is used as the LoRA path, and an adapter
        # directory, if given, is passed as a backfill source so the engine can reload an evicted
        # version itself.
        params = payload["sampling_params"]
        max_tokens = params.get("max_tokens")
        if max_tokens is None:
            raise UserInputError("sampling_params.max_tokens is required")
        sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature": params.get("temperature", 1.0),
            "top_p": params.get("top_p", 1.0),
            "top_k": params.get("top_k", -1),
        }
        if params.get("seed") is not None:
            sampling_params["sampling_seed"] = params["seed"]
        stop = params.get("stop")
        if stop is not None:
            if stop == []:
                # tinker defines stop=[] as disabling every stop token, EOS included
                sampling_params["ignore_eos"] = True
            elif isinstance(stop, list) and isinstance(stop[0], int):
                sampling_params["stop_token_ids"] = stop
            else:
                sampling_params["stop"] = stop
        request = {"input_ids": payload["prompt_tokens"], "sampling_params": sampling_params, "return_logprob": True}
        if payload["prompt_logprobs"] or payload["topk_prompt_logprobs"]:
            request["logprob_start_len"] = 0
        if payload["topk_prompt_logprobs"]:
            request["top_logprobs_num"] = payload["topk_prompt_logprobs"]
        if lora_name is not None:
            request["lora_path"] = lora_name
            if lora_path is not None:
                # request-carried backfill source: the engine refills an evicted version itself
                request["lora_backfill_paths"] = {lora_name: lora_path}
        return request


def _slot_failure(worker_results: list) -> dict | None:
    # [FUNCTION DOC] HELPER: The first non-null entry of the per-worker results, or none if every
    # worker succeeded.
    return next((result for result in worker_results if result is not None), None)


def _with_sample_seed(request: dict, index: int) -> dict:
    # [FUNCTION DOC] HELPER: Return a copy of the request (nested sampling params also copied) in
    # which a non-null sampling seed is increased by the sample index; otherwise unchanged.
    """Give each sample a distinct seed when the caller pins the request seed."""
    request = dict(request)
    params = dict(request["sampling_params"])
    if (seed := params.get("sampling_seed")) is not None:
        params["sampling_seed"] = seed + index
    request["sampling_params"] = params
    return request


def _prompt_logprobs(response: dict) -> list[float]:
    # [FUNCTION DOC] HELPER: Per-prompt-position log-probs from the response meta info input token
    # log-probs (first element of each entry); a null value (the first position) becomes not-a-number.
    entries = response["meta_info"]["input_token_logprobs"]
    return [float("nan") if entry[0] is None else float(entry[0]) for entry in entries]


def _topk_prompt_logprobs(response: dict, k: int) -> dict:
    # [FUNCTION DOC] HELPER: Rectangular top-k tables from the response's input top log-probs.
    # For each prompt position (null position treated as empty) keep at most k candidates
    # (log-prob first, token id second in each engine entry) and pad to exactly k with token id 0
    # and the mask log-prob sentinel. Returns a mapping with token id rows and log-prob rows.
    token_ids, logprobs = [], []
    for position in response["meta_info"]["input_top_logprobs"]:
        candidates = position or []
        candidate_token_ids = [entry[1] for entry in candidates][:k]
        candidate_logprobs = [float(entry[0]) for entry in candidates][:k]
        token_ids.append(candidate_token_ids + [0] * (k - len(candidate_token_ids)))
        logprobs.append(candidate_logprobs + [MASK_LOGPROB] * (k - len(candidate_logprobs)))
    return {"token_ids": token_ids, "logprobs": logprobs}


def _to_sequence(response: dict) -> dict:
    # [FUNCTION DOC] HELPER: Convert one engine response into a sequence mapping with generated
    # tokens, their log-probs, and a stop reason (length if the finish type is length, else stop).
    # An aborted generation returns an error mapping instead, so it fails the request rather than
    # passing as a completed sample.
    output_token_logprobs = response["meta_info"]["output_token_logprobs"]
    finish = response["meta_info"]["finish_reason"]["type"]
    if finish == "abort":
        # a truncated sequence must fail the request, not pass as a completed sample
        return {"error": "the engine aborted this sample; resubmit the request"}
    return {
        "tokens": [entry[1] for entry in output_token_logprobs],
        "logprobs": [entry[0] for entry in output_token_logprobs],
        "stop_reason": "length" if finish == "length" else "stop",
    }
