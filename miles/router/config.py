# [MODULE DOC]
# Configuration of the Miles router: an immutable strict model and a function deriving it from the
# parsed launch args.
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class MilesRouterConfig(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Frozen configuration record with all fields required: bind host and port; maximum number of
    # concurrent upstream connections of the proxy client; request timeout in seconds (none means
    # no timeout); seconds between worker health checks; number of consecutive failed health checks
    # after which a worker is considered dead.
    host: str
    port: int
    max_connections: int
    timeout: float | None
    health_check_interval: float
    health_check_failure_threshold: int


def compute_miles_router_config(args, *, host: str, port: int, num_engines: int) -> MilesRouterConfig:
    # [FUNCTION DOC] Build the router config from launch args plus explicit host, port and engine
    # count. Max connections: the launch arg if it is set (not none), otherwise the per-server
    # concurrency times the number of engines. The timeout comes from the router timeout arg, the
    # health check interval from the rollout health check interval arg, and the failure threshold
    # from the router health check failure threshold arg.
    if args.miles_router_max_connections is not None:
        max_connections = args.miles_router_max_connections
    else:
        max_connections = args.sglang_server_concurrency * num_engines

    return MilesRouterConfig(
        host=host,
        port=port,
        max_connections=max_connections,
        timeout=args.miles_router_timeout,
        health_check_interval=args.rollout_health_check_interval,
        health_check_failure_threshold=args.miles_router_health_check_failure_threshold,
    )
