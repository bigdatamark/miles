# [MODULE DOC] HTTP client for the SGLang router's worker-management API. Handles two API
# generations: the legacy API (router version up to 0.2.1, or when the miles router is used) that
# takes the worker URL as a query parameter, and the newer API with a workers collection (delete by
# URL-encoded URL below version 0.3.0, delete by looked-up worker id from 0.3.0 on).
import dataclasses
import logging
from urllib.parse import quote

import httpx
import sglang_router
from packaging.version import parse

from miles.utils.http_utils import GeneralHttpClientProvider

logger = logging.getLogger(__name__)

ROUTER_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def use_legacy_router_api(args) -> bool:
    # [FUNCTION DOC] True when the installed router package version is at most 0.2.1 or the miles
    # router option is set in the arguments.
    return parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router


@dataclasses.dataclass(frozen=True)
class SGLangRouterApiClient:
    # [CLASS DOC] Immutable client holding the router base URL; its coroutines register and
    # unregister workers. All requests use the shared HTTP client and a 30 second timeout (10 second
    # connect).
    router_url: str

    async def add_worker(
        self,
        worker_url: str,
        worker_type: str,
        use_legacy_api: bool,
        bootstrap_port: int | None = None,
    ):
        # [FUNCTION DOC] Registers a worker with the router; raises on a failed status; returns
        # nothing. Legacy API: asserts the worker type is regular (assertion error otherwise) and
        # posts to the add-worker path with the worker URL as an unencoded query parameter. New API:
        # posts a JSON body with URL and worker type to the workers collection, adding the bootstrap
        # port only for prefill workers (sent as given, possibly none).
        if use_legacy_api:
            assert worker_type == "regular", "pd disaggregation is not supported in old router or miles router."
            response = await GeneralHttpClientProvider.client().post(
                f"{self.router_url}/add_worker?url={worker_url}",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        else:
            payload = {
                "url": worker_url,
                "worker_type": worker_type,
            }
            if worker_type == "prefill":
                payload["bootstrap_port"] = bootstrap_port
            response = await GeneralHttpClientProvider.client().post(
                f"{self.router_url}/workers",
                json=payload,
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        response.raise_for_status()

    async def remove_worker(self, worker_url: str, use_legacy_api: bool):
        # [FUNCTION DOC] Unregisters a worker; returns nothing. Legacy API: posts to the
        # remove-worker path with the unencoded URL as a query parameter. New API below router
        # version 0.3.0: sends a delete to the workers path with the fully percent-encoded URL.
        # From 0.3.0: lists the workers, finds the entry with a matching URL and deletes it by its
        # id (logging a warning if not found); any error during listing or deleting is logged as a
        # warning and swallowed. Finally, if a response was obtained, raises on a failed status
        # (so only the legacy and pre-0.3.0 paths and a failed id-delete status propagate errors).
        response = None
        if use_legacy_api:
            response = await GeneralHttpClientProvider.client().post(
                f"{self.router_url}/remove_worker?url={worker_url}",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        elif parse(sglang_router.__version__) < parse("0.3.0"):
            quoted_worker_url = quote(worker_url, safe="")
            response = await GeneralHttpClientProvider.client().delete(
                f"{self.router_url}/workers/{quoted_worker_url}",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        else:
            try:
                all_workers = (
                    await GeneralHttpClientProvider.client().get(
                        f"{self.router_url}/workers",
                        timeout=ROUTER_REQUEST_TIMEOUT,
                    )
                ).json()["workers"]
                for worker in all_workers:
                    if worker["url"] == worker_url:
                        worker_id = worker["id"]
                        response = await GeneralHttpClientProvider.client().delete(
                            f"{self.router_url}/workers/{worker_id}",
                            timeout=ROUTER_REQUEST_TIMEOUT,
                        )
                        break
                else:
                    logger.warning(f"Worker {worker_url} not found in router during shutdown.")
            except Exception as e:
                logger.warning(f"Failed to fetch workers list or remove worker: {e}")

        if response is not None:
            response.raise_for_status()
