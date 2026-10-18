# [MODULE DOC]
# CPU tests of the peer-to-peer weight transfer utilities: building a server args object from a
# remote engine's info mapping (dropping unknown fields), discovering remote weight information over
# fake rollout engines (each distinct engine rank is queried once, and the three returned maps agree
# on session identifiers), and that a replica loads weights inside its parallelism context. Because
# the real Mooncake and SGLang server-args modules may be missing, a module-scoped fixture temporarily
# injects stub modules into the module table and imports the utilities fresh, restoring everything
# afterwards.
import dataclasses
import importlib
import sys
from collections import Counter
from contextlib import contextmanager
from types import ModuleType

import msgspec
import pytest

_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils"


class _FakeServerArgs(msgspec.Struct):
    # [CLASS DOC] HELPER: Stand-in for SGLang's server args, a msgspec struct with one optional
    # string field, the model path.
    """A Struct, like the real ServerArgs since sglang v0.5.20."""

    model_path: str | None = None


_EXTERNAL_SDK_ATTRIBUTES = {
    "mooncake.engine": {"TransferEngine": object},
    "sglang.srt.server_args": {"ServerArgs": _FakeServerArgs},
}


class _FakeRolloutEngine:
    # [CLASS DOC] HELPER: Fake rollout engine answering the three discovery queries with values
    # derived from its index and the queried rank, and logging each call as name and keyword mapping.
    def __init__(self, engine_index: int):
        # [FUNCTION DOC] HELPER: Stores the engine index and creates an empty call log.
        self._engine_index = engine_index
        self.calls: list[tuple[str, dict]] = []

    async def get_remote_instance_transfer_engine_info(self, rank: int):
        # [FUNCTION DOC] HELPER: Async; logs the call and returns a pair: a session id string made
        # from the engine index and rank, and a mapping from a per-rank weight name to a triple
        # (address 0x1000 plus rank, 4, 2).
        self.calls.append(("get_remote_instance_transfer_engine_info", {"rank": rank}))
        return f"session-{self._engine_index}-{rank}", {f"weight-{rank}": (0x1000 + rank, 4, 2)}

    async def get_parallelism_info(self, rank: int):
        # [FUNCTION DOC] HELPER: Async; logs the call and returns a mapping with the tensor-parallel
        # rank equal to the queried rank.
        self.calls.append(("get_parallelism_info", {"rank": rank}))
        return {"tp_rank": rank}

    async def get_server_info(self):
        # [FUNCTION DOC] HELPER: Async; logs the call and returns a mapping whose model path is
        # "/model/" followed by the engine index.
        self.calls.append(("get_server_info", {}))
        return {"model_path": f"/model/{self._engine_index}"}


@contextmanager
def _stubbed_missing_external_sdks():
    # [FUNCTION DOC] HELPER: Context manager. On entry, for each module path in the stub table
    # (Mooncake engine module and SGLang server args module), creates fresh empty stub modules for
    # the final module (always replaced) and for any missing parent packages (given an empty search
    # path), registers them in the module table, links each child as an attribute of its parent, and
    # sets the configured attributes (a transfer engine placeholder and the fake server args) on the
    # final stub. Real already-loaded parent packages are kept but have their child attribute
    # overwritten. On exit (even after errors) restores previous parent attributes (deleting those
    # that did not exist) and previous module table entries (removing those that did not exist).
    # Mutates global interpreter module state for the duration.
    missing = object()
    saved_modules: dict[str, object] = {}
    saved_attributes: list[tuple[ModuleType, str, object]] = []
    for module_name, attributes in _EXTERNAL_SDK_ATTRIBUTES.items():
        parts = module_name.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            saved_modules[name] = sys.modules.get(name, missing)
            if depth == len(parts) or name not in sys.modules:
                module = ModuleType(name)
                if depth < len(parts):
                    module.__path__ = []
                sys.modules[name] = module
            if depth > 1:
                parent = sys.modules[".".join(parts[: depth - 1])]
                attribute = parts[depth - 1]
                saved_attributes.append((parent, attribute, getattr(parent, attribute, missing)))
                setattr(parent, attribute, sys.modules[name])
        for attribute, value in attributes.items():
            setattr(sys.modules[module_name], attribute, value)

    try:
        yield
    finally:
        for parent, attribute, value in reversed(saved_attributes):
            if value is missing:
                delattr(parent, attribute)
            else:
                setattr(parent, attribute, value)
        for name, module in reversed(saved_modules.items()):
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def p2p_transfer_utils():
    # [FUNCTION DOC] HELPER: Module-scoped fixture yielding a freshly imported copy of the P2P
    # transfer utilities module bound to the stubbed external SDKs. Remembers the previously
    # imported module and the package attribute, enters the stub context, removes the cached module
    # and package attribute so the import re-executes, imports and yields it. In cleanup removes
    # the fresh module and restores (or removes) the previous module table entry and package
    # attribute.
    package_name, attribute = _MODULE.rsplit(".", 1)
    package = importlib.import_module(package_name)
    missing = object()
    saved_module = sys.modules.get(_MODULE, missing)
    saved_attribute = getattr(package, attribute, missing)

    with _stubbed_missing_external_sdks():
        sys.modules.pop(_MODULE, None)
        if hasattr(package, attribute):
            delattr(package, attribute)
        try:
            yield importlib.import_module(_MODULE)
        finally:
            sys.modules.pop(_MODULE, None)
            if saved_module is not missing:
                sys.modules[_MODULE] = saved_module
            if saved_attribute is missing:
                if hasattr(package, attribute):
                    delattr(package, attribute)
            else:
                setattr(package, attribute, saved_attribute)


def _make_targets(module, pairs: list[tuple[int, int]]) -> list:
    # [FUNCTION DOC] HELPER: Converts pairs of engine index and engine rank into transfer target
    # metadata records from the given module, numbering the source shard by position in the list.
    return [
        module.TransferTaskP2PMeta(engine_ind=engine_ind, engine_rank=engine_rank, source_shard=source_shard)
        for source_shard, (engine_ind, engine_rank) in enumerate(pairs)
    ]


def _query(module, engines: list[_FakeRolloutEngine], pairs: list[tuple[int, int]]):
    # [FUNCTION DOC] HELPER: Builds targets from the pairs and calls the module's remote weight info
    # query over the engines; returns its result (weight infos, target to session id map, session
    # id to server args map).
    return module.query_remote_weight_infos(engines, _make_targets(module, pairs))


@pytest.mark.parametrize(
    "record_factory", [dataclasses.make_dataclass, msgspec.defstruct], ids=["dataclass", "msgspec"]
)
def test_server_args_from_remote_info_filters_unknown_fields(p2p_transfer_utils, monkeypatch, record_factory):
    # [FUNCTION DOC] Parametrized over a dataclass and a msgspec struct as the server args type
    # (each with one string field, the model path), which replaces the module's server args type.
    # Creating server args from a mapping holding the model path plus an unknown field must return
    # an instance of that type with the model path set, silently dropping the unknown field.
    server_args_type = record_factory("ServerArgs", [("model_path", str)])
    monkeypatch.setattr(p2p_transfer_utils, "ServerArgs", server_args_type)

    result = p2p_transfer_utils.create_server_args_from_dict({"model_path": "/model", "unknown_field": True})

    assert isinstance(result, server_args_type)
    assert result.model_path == "/model"


class TestQueryRemoteWeightInfos:
    # [CLASS DOC] Tests of remote weight info discovery against two fake engines.
    """Remote-info discovery over the rollout engines' HTTP API."""

    def test_repeated_targets_are_queried_once_each(self, p2p_transfer_utils):
        # [FUNCTION DOC] Queries targets (engine, rank): (0,0), (0,1), (0,0), (1,0), where the first
        # is repeated. Asserts engine zero received two calls of each of the three query kinds (one
        # per distinct rank, ranks zero and one for the parallelism query) and engine one received
        # each query kind exactly once, in the order transfer info, parallelism, server info.
        """The same engine rank appears once per source shard, and re-querying it wastes round trips."""
        engines = [_FakeRolloutEngine(0), _FakeRolloutEngine(1)]

        _query(p2p_transfer_utils, engines, [(0, 0), (0, 1), (0, 0), (1, 0)])

        assert Counter(name for name, _kwargs in engines[0].calls) == Counter(
            {
                "get_remote_instance_transfer_engine_info": 2,
                "get_parallelism_info": 2,
                "get_server_info": 2,
            }
        )
        assert sorted(kwargs["rank"] for name, kwargs in engines[0].calls if name == "get_parallelism_info") == [0, 1]
        assert [name for name, _kwargs in engines[1].calls] == [
            "get_remote_instance_transfer_engine_info",
            "get_parallelism_info",
            "get_server_info",
        ]

    def test_the_returned_maps_agree_on_every_session_id(self, p2p_transfer_utils):
        # [FUNCTION DOC] Queries three targets across two engines and asserts the three returned
        # maps: target to session id, session id to pair of weight info and parallelism info, and
        # session id to server args object (instances of the module's server args type whose model
        # path is the owning engine's), all keyed consistently by the same session ids.
        """Every weight, parallelism, and converted server-args entry must match its session ID."""
        engines = [_FakeRolloutEngine(0), _FakeRolloutEngine(1)]

        weight_infos, targets_to_session_id, session_id_to_server_args = _query(
            p2p_transfer_utils, engines, [(0, 0), (0, 1), (1, 0)]
        )

        assert targets_to_session_id == {
            (0, 0): "session-0-0",
            (0, 1): "session-0-1",
            (1, 0): "session-1-0",
        }
        assert weight_infos == {
            "session-0-0": ({"weight-0": (0x1000, 4, 2)}, {"tp_rank": 0}),
            "session-0-1": ({"weight-1": (0x1001, 4, 2)}, {"tp_rank": 1}),
            "session-1-0": ({"weight-0": (0x1000, 4, 2)}, {"tp_rank": 0}),
        }
        assert all(
            isinstance(server_args, p2p_transfer_utils.ServerArgs)
            for server_args in session_id_to_server_args.values()
        )
        assert {
            session_id: server_args.model_path for session_id, server_args in session_id_to_server_args.items()
        } == {
            "session-0-0": "/model/0",
            "session-0-1": "/model/0",
            "session-1-0": "/model/1",
        }


from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import torch
from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P, _ReplicaTarget

_P2P_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p"


def test_replica_loads_inside_its_parallelism_context():
    # [FUNCTION DOC] Builds a P2P protocol without its constructor as a sender with one replica
    # target (mock replica, one fake session, config marker "cfg-0"), mock transfer manager and
    # per-session writer, and a transfer-ready-params function returning one embedding weight. The
    # parallelism context class is patched with a fake context manager that records its config and
    # an inside flag. Sending one bucket must enter the context with the replica's config and call
    # the replica's weight loading while the flag is set; afterwards the flag is cleared. Asserts
    # the recorded sequence is the config followed by a load marker with inside true.
    """sglang's sharded weight loaders read the attention-TP rank at call time, which only
    exists while the replica's ParallelismContext is entered; a load outside it asserts."""
    active = {"inside": False}
    seen = []

    @contextmanager
    def fake_context(config):
        # [FUNCTION DOC] HELPER: Fake context manager: records the config, sets the inside flag for
        # the duration of the with block and clears it afterwards, even on error.
        active["inside"] = True
        seen.append(config)
        try:
            yield
        finally:
            active["inside"] = False

    replica = MagicMock()
    replica.load_weights.side_effect = lambda tensors: seen.append(("load", active["inside"]))

    protocol = object.__new__(UpdateWeightP2P)
    protocol.is_sender = True
    protocol._replica_targets = [_ReplicaTarget(replica, [SimpleNamespace(session_id="s0")], "cfg-0")]
    protocol.transfer_manager = MagicMock()
    protocol._do_p2p_write_one_session = MagicMock()
    ready = [("model.embed_tokens.weight", torch.zeros(1))]
    protocol._get_transfer_ready_params = lambda tensors: (["model.embed_tokens.weight"], ready)

    with patch(f"{_P2P_MODULE}.ParallelismContext", fake_context):
        protocol.send_bucket(list(ready))

    assert seen == ["cfg-0", ("load", True)]
    assert active["inside"] is False
