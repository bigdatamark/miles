# [MODULE DOC]
# The Miles router process: a FastAPI application that registers SGLang worker urls (add, remove,
# list endpoints compatible with the sglang-router API) and forwards every other request to the
# registered worker with the fewest active requests. A background task periodically polls each
# worker's health endpoint and quarantines workers after consecutive failures (they are never
# re-admitted automatically, because their weights may be stale). Run as a subprocess via the
# run_router entry point or as a script with the config on the command line.
import asyncio
import json
import logging
import sys

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.router.config import MilesRouterConfig
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.workers.argv_utils import parse_config_argv

logger = logging.getLogger(__name__)


def run_router(config: MilesRouterConfig):
    # [FUNCTION DOC] Process entry point (blocks). Configures raw logging for the router, sets the
    # process title so it can be killed by name, builds the router (non-verbose) and serves its
    # app with uvicorn on the configured host and port at info log level.
    """
    Run the Miles router with the specified configuration.
    """
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("miles_router")
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-router")

    # Initialize the router with tokenizer and lazy worker initialization
    miles_router = MilesRouter(config, verbose=False)

    # Start the server
    uvicorn.run(miles_router.app, host=config.host, port=config.port, log_level="info")


class MilesRouter:
    # [CLASS DOC]
    # Least-active-requests HTTP proxy. State: the config; a verbose flag; the web app; a mapping
    # from worker url to the number of requests currently in flight (this mapping's keys are also
    # the registry of known workers); a mapping from worker url to its consecutive health check
    # failures; the set of quarantined (dead) workers excluded from routing; and a shared async
    # HTTP client limited to the configured connections and timeout. All state is in memory and
    # touched only from the event loop, so no locks are used.
    def __init__(self, config: MilesRouterConfig, verbose=False):
        # [FUNCTION DOC] Store config and verbose flag, create the app with a startup hook that
        # launches the health check loop, initialize empty worker bookkeeping, create the HTTP
        # client (max connections and timeout from config; a none timeout disables it), and
        # register routes.
        """Initialize the miles-router with SGLang router address"""
        self.config = config
        self.verbose = verbose

        self.app = FastAPI()
        self.app.router.on_startup.append(self._start_background_health_check)

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=config.max_connections),
            timeout=httpx.Timeout(config.timeout),
        )

        self._setup_routes()

    def _setup_routes(self):
        # [FUNCTION DOC] HELPER: Register POST add-worker, POST remove-worker, GET list-workers and,
        # last (so it does not shadow the others), a catch-all route for GET, POST, PUT and DELETE
        # that proxies to a worker. Note PATCH is not routed.
        """Setup all the HTTP routes except catch-all proxy"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.post("/remove_worker")(self.remove_worker)
        self.app.get("/list_workers")(self.list_workers)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _start_background_health_check(self):
        # [FUNCTION DOC] HELPER: Startup hook: launch the health check loop as a background task.
        # KNOWN ISSUE: the task object is not stored, so it could be garbage collected.
        asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        # [FUNCTION DOC] HELPER: Network call: GET the worker's health path with a 5 second timeout.
        # Returns the pair (url, healthy) where healthy is true only for HTTP 200; any other
        # status or any exception is logged at debug level and gives false.
        """Encapsulated health check logic for better maintainability."""
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return url, True
            logger.debug(f"[miles-router] Worker {url} is unhealthy (Status: {response.status_code})")
        except Exception as e:
            logger.debug(f"[miles-router] Worker {url} health check failed: {e}")
        return url, False

    async def _health_check_loop(self):
        # [FUNCTION DOC]
        # Infinite loop. Each round: sleep for the configured interval; collect registered workers
        # not yet dead (skip the round if none); check them concurrently; for each result whose
        # worker is still registered: an unhealthy result increments its consecutive failure count
        # and, on reaching the threshold, logs a warning and adds it to the dead set (never revived
        # automatically since its weights may be stale); a healthy result resets the count to zero.
        # Logs the healthy count at debug level. Cancellation is logged and re-raised; any other
        # exception is logged with traceback, then the loop sleeps 5 extra seconds and continues.
        """Background loop to monitor worker health and adjust routing pool."""
        interval = self.config.health_check_interval
        threshold = self.config.health_check_failure_threshold

        while True:
            try:
                await asyncio.sleep(interval)

                urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
                if not urls:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                for url, is_healthy in results:
                    if url not in self.worker_request_counts:
                        continue
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            logger.warning(
                                f"[miles-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                            # TODO (chenyang): Connect back 'dead' workers requires a mechanism to sync
                            # model versions to avoid off-policy issues from stale weights, since these
                            # dead workers' parameters may not be refitted.
                    else:
                        self.worker_failure_counts[url] = 0

                logger.debug(
                    f"[miles-router] Health check complete. {len(self.worker_request_counts) - len(self.dead_workers)} workers healthy."
                )

            except asyncio.CancelledError:
                logger.warning("[miles-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[miles-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def proxy(self, request: Request, path: str):
        # [FUNCTION DOC] Catch-all route handler: forward the request via the core proxy and turn
        # the result into an HTTP response.
        """Proxy all other requests to the SGLang router"""
        result = await self.do_proxy(request, path)
        return self.build_proxy_response(result)

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        # [FUNCTION DOC]
        # Forward one request to the least-loaded healthy worker (network I/O) and return a mapping
        # with request body, response body bytes, status code and response headers. Selecting the
        # worker increments its in-flight count (a runtime error is raised if every worker is dead;
        # see the selection helper for the empty pool case), and the count is decremented in a
        # finally block whether or not the request succeeds. Body defaults to the request's body and
        # headers to the request's headers when not given; content length and transfer encoding
        # headers are always dropped before forwarding. The upstream response is fully read.
        # HTTP client errors propagate to the caller (no 502 conversion here).
        """Core proxy logic. Returns dict with request_body, response_body, status_code, headers."""
        worker_url = self._use_url()
        url = f"{worker_url}/{path}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
        if body is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
            content = await response.aread()
            return {
                "request_body": body,
                "response_body": content,
                "status_code": response.status_code,
                "headers": dict(response.headers),
            }
        finally:
            self._finish_url(worker_url)

    def build_proxy_response(self, result: dict) -> Response:
        # [FUNCTION DOC] Convert a proxy result to an HTTP response, dropping content length,
        # transfer encoding, server and date headers (the local server adds its own). If the body
        # parses as JSON, reply with a JSON response carrying the parsed data (re-serialized) and
        # the remaining headers; on any parse failure, pass the raw bytes through with the upstream
        # content type (empty if absent).
        """Build HTTP response from proxy result."""
        content = result["response_body"]
        status_code = result["status_code"]
        headers = result["headers"]
        headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "server", "date")
        }
        content_type = headers.get("content-type", "")
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=status_code, headers=headers)
        except Exception:
            return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)

    async def add_worker(self, request: Request):
        # [FUNCTION DOC] Route handler: register a worker url taken from the query string or JSON
        # body. A missing url replies HTTP 400 with an error message. A new url gets in-flight and
        # failure counts of zero and is removed from the dead set; an already known url is left
        # untouched (a dead known worker stays dead). Replies with success status and the mapping
        # of all worker urls to in-flight counts.
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        worker_url = await self._parse_worker_url(request)
        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        # Add if new, keep a simple request count per worker
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            self.dead_workers.discard(worker_url)
            if self.verbose:
                print(f"[miles-router] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def remove_worker(self, request: Request):
        # [FUNCTION DOC] Route handler: deregister the worker url (same sources as adding). A
        # missing url replies 400. Removes it from the in-flight, failure and dead bookkeeping
        # (unknown urls are ignored), logs, and replies with success and the remaining mapping.
        # Requests already in flight to it finish normally. Note the check here is for none, so an
        # empty string url is treated as present, unlike adding.
        """Remove a worker from the router, using the same URL conventions as add_worker."""
        worker_url = await self._parse_worker_url(request)
        if worker_url is None:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        self.worker_request_counts.pop(worker_url, None)
        self.worker_failure_counts.pop(worker_url, None)
        self.dead_workers.discard(worker_url)
        logger.info(f"[miles-router] Removed worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def _parse_worker_url(self, request: Request) -> str | None:
        # [FUNCTION DOC] HELPER: Read the worker url: the query parameter named url, else the one
        # named worker_url (non-empty); otherwise parse the body as JSON (empty body means an empty
        # object; invalid JSON raises a decode error) and take the url field, else the worker_url
        # field. Returns none or empty if absent.
        if worker_url := request.query_params.get("url") or request.query_params.get("worker_url"):
            return worker_url

        body = await request.body()
        payload = json.loads(body) if body else {}
        return payload.get("url") or payload.get("worker_url")

    async def list_workers(self, request: Request):
        # [FUNCTION DOC] Route handler: reply with a mapping holding the list of all registered
        # worker urls (including dead ones) under the urls key.
        """List all registered workers"""
        return {"urls": list(self.worker_request_counts.keys())}

    def _use_url(self):
        # [FUNCTION DOC] HELPER: Choose the non-dead worker with the smallest in-flight count (ties
        # resolved by registration order), increment its count and return its url. If workers
        # are all dead, raises a runtime error saying no healthy workers are available.
        # KNOWN ISSUE: with no workers registered at all and none dead, the minimum over an empty
        # set raises a plain value error instead of the runtime error.
        """Select worker URL with minimal active requests."""

        if not self.dead_workers:
            # Healthy path: select from all workers
            url = min(self.worker_request_counts, key=self.worker_request_counts.get)
        else:
            # Degraded path: select from workers not in dead_workers
            valid_workers = (w for w in self.worker_request_counts if w not in self.dead_workers)
            try:
                url = min(valid_workers, key=self.worker_request_counts.get)
            except ValueError:
                raise RuntimeError("No healthy workers available in the pool") from None

        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url: str) -> None:
        # [FUNCTION DOC] HELPER: Decrement the in-flight count of the worker. If the worker is no
        # longer registered or its count is already zero, log at info level and do nothing.
        """Mark the request to the given URL as finished"""
        if (count := self.worker_request_counts.get(url)) is None or count == 0:
            logger.info(f"[miles-router] Request to {url} finished after the worker was deregistered; ignoring")
            return
        self.worker_request_counts[url] = count - 1


if __name__ == "__main__":
    run_router(parse_config_argv(MilesRouterConfig, sys.argv[1:]))
