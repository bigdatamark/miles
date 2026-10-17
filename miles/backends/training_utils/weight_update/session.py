# [MODULE DOC]
# Thin synchronous wrappers around the rollout engines' weight-update session RPCs (pause, begin,
# set version, end, resume, plus adapter registration and result checking). Each wrapper fans an
# async call out to every given engine client and blocks until all finish. Callers decide which rank
# invokes them.
"""Engine-side RPCs for a weight-update session.

The session frame is: pause -> begin -> (transfer) -> set version -> end ->
resume. Callers gate driver-only calls (typically global rank 0) themselves.
"""

from argparse import Namespace
from collections.abc import Mapping, Sequence

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils import async_utils


def pause_engines(args: Namespace, rollout_engines: Sequence[SGLangApiClient]) -> None:
    # [FUNCTION DOC] Pauses generation on all engines in the args' pause generation mode and waits;
    # then, unless the mode is in-place, flushes each engine's cache and waits. Raises whatever the
    # waits raise.
    """Quiesce the engines for a weight write.

    in_place pausing freezes requests and resumes them against their existing
    KV cache, so flushing would discard exactly what that mode preserves.
    """
    mode = args.pause_generation_mode
    async_utils.wait_futures([async_utils.submit(client.pause_generation(mode=mode)) for client in rollout_engines])
    if mode != "in_place":
        async_utils.wait_futures([async_utils.submit(client.flush_cache()) for client in rollout_engines])


def resume_engines(rollout_engines: Sequence[SGLangApiClient]) -> None:
    # [FUNCTION DOC] Tell every engine to continue generation and wait for all.
    async_utils.wait_futures([async_utils.submit(client.continue_generation()) for client in rollout_engines])


def begin_weight_update(
    rollout_engines: Sequence[SGLangApiClient], selector: str = "all", *, sync_base: bool = True
) -> None:
    # [FUNCTION DOC] Calls begin-weight-update on every engine with the selector string (default
    # "all") and the sync-base flag (keyword-only, default true), and waits. Responses are not
    # inspected.
    """Open a weight-update session on the selected engines. ``sync_base=False``
    declares an adapter-only session: no quant unpack, base tensors rejected."""
    async_utils.wait_futures(
        [
            async_utils.submit(client.begin_weight_update(selector=selector, sync_base=sync_base))
            for client in rollout_engines
        ]
    )


def end_weight_update(
    rollout_engines: Sequence[SGLangApiClient], *, expected_lora_checksums: Mapping | None = None
) -> None:
    # [FUNCTION DOC] Calls end-weight-update on every engine (passing optional expected LoRA
    # checksums, keyword-only), waits, and raises a runtime error with the engine's message for the
    # first mapping result whose success field is explicitly false. Non-mapping results are ignored.
    """Close the session: re-finalize base weights (sync_base sessions) and apply
    the streamed LoRA stash (optionally verified against a sha256 manifest)."""
    results = async_utils.wait_futures(
        [
            async_utils.submit(client.end_weight_update(expected_lora_checksums=expected_lora_checksums))
            for client in rollout_engines
        ]
    )
    for result in results:
        if isinstance(result, Mapping) and result.get("success") is False:
            raise RuntimeError(f"end_weight_update failed on a rollout engine: {result.get('message')}")


def register_lora_adapter(
    rollout_engines: Sequence[SGLangApiClient], *, lora_name: str, lora_config: Mapping, pinned: bool = False
) -> None:
    # [FUNCTION DOC] Keyword-only name, config mapping (copied to a plain mapping) and pinned flag
    # (default false). Registers the adapter on every engine, waits, and checks results as LoRA
    # results (raises a runtime error on failure).
    """Create-or-refresh an adapter's identity and config on every engine
    (weights zeroed; the bytes follow in the update stream)."""
    futures = [
        async_utils.submit(
            client.register_lora_adapter(lora_name=lora_name, config_dict=dict(lora_config), pinned=pinned)
        )
        for client in rollout_engines
    ]
    check_weight_sync_results(async_utils.wait_futures(futures), is_lora=True)


def set_weight_version(rollout_engines: Sequence[SGLangApiClient], weight_version: int) -> None:
    # [FUNCTION DOC] Sets the weight version (as a decimal string) on every engine and waits.
    # Responses are not inspected.
    async_utils.wait_futures(
        [
            async_utils.submit(client.update_weight_version(weight_version=str(weight_version)))
            for client in rollout_engines
        ]
    )


def check_weight_sync_results(results: list, *, is_lora: bool) -> None:
    # [FUNCTION DOC] For each result: a mapping contributes its success field and error text (error
    # message field, else error field, else "unknown error"); an object with a success attribute
    # contributes that and its error message attribute; anything else is skipped. If success is
    # explicitly false, raises a runtime error that says whether it was LoRA or base-model sync, the
    # error text, and a hint to check SGLang version compatibility. A missing/none success is treated
    # as success.
    """Raise if any engine reported a failed weight-sync RPC."""
    sync_type = "LoRA" if is_lora else "Base model"
    for result in results:
        if isinstance(result, Mapping):
            success = result.get("success")
            error_msg = result.get("error_message") or result.get("error") or "unknown error"
        elif hasattr(result, "success"):
            success = result.success
            error_msg = getattr(result, "error_message", "unknown error")
        else:
            continue

        if success is False:
            raise RuntimeError(
                f"{sync_type} weight sync failed on rollout engine: {error_msg}. "
                f"Check SGLang version compatibility."
            )
