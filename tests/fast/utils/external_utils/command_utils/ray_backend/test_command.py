# [MODULE DOC] Tests for the Ray backend command helpers, using the fake Ray fixtures from the
# local conftest. Covers running a command on every selected alive Ray node (placeholders for
# rank, node count, master address and node ip substituted, the current node first and others in
# order, dead nodes skipped, CUDA visible devices unset, Ray connected at the auto address and
# always shut down once, including on selection errors and result collection failures) and the
# Mooncake master start failure path (cleanup commands issued and an error about an unreadable
# log).
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from miles.utils.external_utils.command_utils.ray_backend import command


class TestExecCommandAllRayNodes:
    # [CLASS DOC] Tests of the function that runs a command on all (or a number of) alive Ray nodes.
    def test_executes_the_substituted_command_on_each_selected_alive_node(
        self,
        fake_ray_factory: Callable[..., tuple[Any, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # [FUNCTION DOC] Three alive nodes (ids third, current, first; one more dead) with the
        # current node ip faked to the current node's address, the shell runner replaced by one
        # echoing the capture flag and command, and a request for 2 nodes with output capture.
        # Asserts: Ray connected once at the auto address; scheduled node ids are current then
        # first (the current node first, then remaining alive nodes ordered by address, truncated
        # to the requested count); the results are the commands with a prefix unsetting CUDA
        # visible devices and the placeholders replaced (rank 0 and 1, count 2, master address the
        # current node's ip, node ip each node's own address); and Ray was shut down once.
        """Selected alive nodes receive substituted commands in deterministic current-node-first order."""
        nodes = [
            {"Alive": True, "NodeID": "third", "NodeManagerAddress": "10.0.0.3"},
            {"Alive": False, "NodeID": "dead", "NodeManagerAddress": "10.0.0.0"},
            {"Alive": True, "NodeID": "current", "NodeManagerAddress": "10.0.0.2"},
            {"Alive": True, "NodeID": "first", "NodeManagerAddress": "10.0.0.1"},
        ]
        fake_ray, ray_execution = fake_ray_factory(nodes)
        monkeypatch.setattr(command, "get_current_node_ip", lambda: "10.0.0.2")
        monkeypatch.setattr(
            command,
            "run_shell_command",
            lambda cmd, *, capture_output: f"{capture_output}: {cmd}",
        )

        results = command.exec_command_all_ray_nodes(
            "rank={{node_rank}} count={{nnodes}} master={{master_addr}} node={{node_ip}}",
            capture_output=True,
            num_nodes=2,
        )

        assert fake_ray.init_addresses == ["auto"]
        assert ray_execution.scheduled_node_ids == ["current", "first"]
        assert results == [
            "True: unset CUDA_VISIBLE_DEVICES; rank=0 count=2 master=10.0.0.2 node=10.0.0.2",
            "True: unset CUDA_VISIBLE_DEVICES; rank=1 count=2 master=10.0.0.2 node=10.0.0.1",
        ]
        assert fake_ray.shutdown_count == 1

    @pytest.mark.parametrize(
        ("nodes", "num_nodes", "message"),
        [
            ([{"Alive": False, "NodeID": "dead", "NodeManagerAddress": "10.0.0.0"}], None, None),
            ([{"Alive": True, "NodeID": "only", "NodeManagerAddress": "10.0.0.1"}], 2, "only 1 alive"),
        ],
    )
    def test_rejects_an_unschedulable_node_selection_and_shuts_ray_down(
        self,
        fake_ray_factory: Callable[..., tuple[Any, Any]],
        monkeypatch: pytest.MonkeyPatch,
        nodes: list[dict[str, object]],
        num_nodes: int | None,
        message: str | None,
    ) -> None:
        # [FUNCTION DOC] Parametrized over (a) only a dead node with no node count and no expected
        # message, and (b) one alive node with 2 nodes requested and a message about only 1 alive.
        # The command call must raise an assertion error (matching the message when present) and
        # Ray must still have been shut down once. The message parameter being none makes the match
        # accept any text.
        """An impossible node selection raises its assertion and always shuts Ray down."""
        fake_ray, _ = fake_ray_factory(nodes)
        monkeypatch.setattr(command, "get_current_node_ip", lambda: "10.0.0.1")

        with pytest.raises(AssertionError, match=message):
            command.exec_command_all_ray_nodes("true", num_nodes=num_nodes)

        assert fake_ray.shutdown_count == 1

    def test_shuts_ray_down_when_collecting_remote_results_fails(
        self,
        fake_ray_factory: Callable[..., tuple[Any, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # [FUNCTION DOC] With one alive node and Ray's result collection raising a runtime error,
        # the command call propagates that error and Ray was shut down exactly once.
        """A result-collection failure propagates after Ray is shut down exactly once."""
        nodes = [{"Alive": True, "NodeID": "only", "NodeManagerAddress": "10.0.0.1"}]
        fake_ray, _ = fake_ray_factory(nodes, get_error=RuntimeError("remote failed"))
        monkeypatch.setattr(command, "get_current_node_ip", lambda: "10.0.0.1")

        with pytest.raises(RuntimeError, match="remote failed"):
            command.exec_command_all_ray_nodes("true")

        assert fake_ray.shutdown_count == 1


class TestStartMooncakeMaster:
    # [CLASS DOC] Tests of the Mooncake master start helper.
    def test_reports_an_unreadable_log_when_mooncake_startup_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] The port is reported not ready beforehand, shell commands are recorded
        # instead of run, and the wait for server readiness raises a runtime error. Starting the
        # master on port 50061 with a log path in a directory that does not exist must raise a
        # runtime error saying it is unable to read the log file. Exactly two shell commands were
        # issued: the first starts with the cleanup command (a process kill of the master with that
        # port, ignoring errors) followed by a separator and the launch, and the second is exactly
        # the cleanup command.
        """Failed startup cleanup reports when the nonexistent diagnostic log cannot be read."""
        commands: list[str] = []
        log_path = tmp_path / "missing" / "mooncake.log"
        monkeypatch.setattr(command, "_is_tcp_server_ready", lambda host, port: False)
        monkeypatch.setattr(command, "run_shell_command", commands.append)

        def fail_wait(host: str, port: int, *, timeout: float) -> None:
            # [FUNCTION DOC] HELPER: Fake readiness wait; always raises a runtime error saying not ready.
            raise RuntimeError("not ready")

        monkeypatch.setattr(command, "wait_for_server_ready", fail_wait)

        with pytest.raises(RuntimeError, match=r"unable to read .*mooncake\.log"):
            command.start_mooncake_master(rpc_port=50061, log_path=log_path)

        assert len(commands) == 2
        expected_cleanup = "pkill -f '^mooncake_master --rpc_port 50061 ' >/dev/null 2>&1 || true"
        assert commands[0].startswith(f"{expected_cleanup}; ")
        assert commands[1] == expected_cleanup
