# [MODULE DOC] HELPER: Shared pytest fixtures for the Ray backend command tests. Provides
# lightweight fakes of the Ray scheduling strategy, a remote function execution wrapper and the
# Ray module (init, nodes, get, shutdown), plus a factory fixture that installs the fakes into the
# command module under test.
from collections.abc import Callable
from dataclasses import dataclass, field
import pytest

from miles.utils.external_utils.command_utils.ray_backend import command


@dataclass(frozen=True)
class FakeSchedulingStrategy:
    # [CLASS DOC] HELPER: Immutable stand-in for Ray's node affinity strategy; holds the target
    # node id and a soft flag.
    node_id: str
    soft: bool


@dataclass
class FakeRayExecution:
    # [CLASS DOC] HELPER: Stand-in for a Ray remote function handle. Holds the underlying plain
    # function (taking the command text and capture flag), a record of node ids it was scheduled
    # on in call order, and the strategy set by the latest options call.
    function: Callable[[str, bool], str | None]
    scheduled_node_ids: list[str] = field(default_factory=list)
    strategy: FakeSchedulingStrategy | None = None

    def options(self, *, scheduling_strategy: FakeSchedulingStrategy) -> "FakeRayExecution":
        # [FUNCTION DOC] HELPER: Remembers the scheduling strategy for the next remote call and
        # returns the same object (so options calls can be chained). Mutates the strategy field.
        self.strategy = scheduling_strategy
        return self

    def remote(self, cmd: str, *, capture_output: bool) -> str | None:
        # [FUNCTION DOC] HELPER: Runs the underlying function synchronously in this process.
        # Asserts a strategy was set by options, records its node id in the scheduled list, and
        # returns the function's result for the command and capture flag (it is a plain value, not
        # a Ray object reference).
        assert self.strategy is not None
        self.scheduled_node_ids.append(self.strategy.node_id)
        return self.function(cmd, capture_output)


@dataclass
class FakeRay:
    # [CLASS DOC] HELPER: Stand-in for the Ray module. Holds the node records that nodes returns,
    # an optional error that get raises, a record of init addresses and a shutdown call counter.
    node_records: list[dict[str, object]]
    get_error: Exception | None = None
    init_addresses: list[str] = field(default_factory=list)
    shutdown_count: int = 0

    def init(self, *, address: str) -> None:
        # [FUNCTION DOC] HELPER: Records the address the code connected to; connects to nothing.
        self.init_addresses.append(address)

    def nodes(self) -> list[dict[str, object]]:
        # [FUNCTION DOC] HELPER: Returns the configured node records unchanged.
        return self.node_records

    def get(self, references: list[str | None]) -> list[str | None]:
        # [FUNCTION DOC] HELPER: Raises the configured error if one was set; otherwise returns the
        # given references as the results (they are already plain values).
        if self.get_error is not None:
            raise self.get_error
        return references

    def shutdown(self) -> None:
        # [FUNCTION DOC] HELPER: Increments the shutdown counter.
        self.shutdown_count += 1


@pytest.fixture
def fake_ray_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., tuple[FakeRay, FakeRayExecution]]:
    # [FUNCTION DOC] HELPER: Pytest fixture returning a factory function. Takes the monkeypatch
    # fixture and returns the factory that installs the fakes.
    def create(
        node_records: list[dict[str, object]], *, get_error: Exception | None = None
    ) -> tuple[FakeRay, FakeRayExecution]:
        # [FUNCTION DOC] HELPER: Builds a fake Ray with the node records and optional get error,
        # and a fake execution wrapping the plain function behind the command module's remote
        # per-node execution function. Patches the command module: its ray reference becomes the
        # fake, the remote function's options method becomes the fake's options, and its node
        # affinity strategy class becomes the fake strategy. Returns the pair (fake Ray, fake
        # execution). Patches are undone at test teardown by the monkeypatch fixture.
        fake_ray = FakeRay(node_records=node_records, get_error=get_error)
        ray_execution = FakeRayExecution(function=command._exec_command_on_node._function)
        monkeypatch.setattr(command, "ray", fake_ray)
        monkeypatch.setattr(command._exec_command_on_node, "options", ray_execution.options)
        monkeypatch.setattr(command, "NodeAffinitySchedulingStrategy", FakeSchedulingStrategy)
        return fake_ray, ray_execution

    return create
