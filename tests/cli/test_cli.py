from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"


def _canonical(value: object) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _run_cli(
    home: Path,
    *arguments: str,
    timeout: float = 20.0,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(SOURCE_ROOT)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.pop("ALICE_BRAIN_HERMES_HOME", None)
    environment["ALICE_BRAIN_HOME"] = os.fspath(home.parent / "must-not-be-used")
    environment["HERMES_HOME"] = os.fspath(home.parent / "hermes-home")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "alice_brain_hermes.cli",
            "--home",
            os.fspath(home),
            "--timeout",
            "10",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _success(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    body = json.loads(result.stdout)
    assert type(body) is dict
    assert set(body) == {"command", "data", "ok", "schema_version"}
    assert body["schema_version"] == 1
    assert body["ok"] is True
    assert type(body["command"]) is str and body["command"]
    assert type(body["data"]) is dict
    assert result.stdout == _canonical(body)
    return body


def _error(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode != 0
    assert result.stdout == ""
    body = json.loads(result.stderr)
    assert type(body) is dict
    assert set(body) == {
        "code",
        "data",
        "message",
        "ok",
        "schema_version",
    }
    assert body["schema_version"] == 1
    assert body["ok"] is False
    assert type(body["code"]) is str and body["code"]
    assert type(body["message"]) is str and body["message"]
    assert body["data"] is None or type(body["data"]) is dict
    assert result.stderr == _canonical(body)
    return body


def _stop_if_running(home: Path) -> None:
    result = _run_cli(home, "daemon", "stop", timeout=15)
    if result.returncode != 0:
        raise AssertionError(result.stderr)


def test_help_is_human_readable_and_does_not_create_runtime_home(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"

    result = _run_cli(home, "--help")

    assert result.returncode == 0
    assert result.stderr == ""
    assert "daemon" in result.stdout
    assert "start" in result.stdout
    assert "stop" in result.stdout
    assert "doctor" in result.stdout
    assert "identity" in result.stdout
    assert "trace" in result.stdout
    assert not home.exists()


def test_shared_control_parser_builds_the_standalone_command_surface() -> None:
    from alice_brain_hermes import cli

    parser = cli._MachineArgumentParser(prog="hermes alice-brain")
    returned = cli.configure_control_parser(parser)

    assert returned is parser
    identity = parser.parse_args(["identity", "--brain-id", "brain"])
    trace = parser.parse_args(
        [
            "trace",
            "--brain-id",
            "brain",
            "--after-sequence",
            "4",
            "--limit",
            "25",
        ]
    )
    explicit_identity = parser.parse_args(["identity", "get"])
    explicit_trace = parser.parse_args(["trace", "list"])
    assert identity.alice_brain_command == "identity"
    assert identity.alice_brain_identity_command == "get"
    assert identity.brain_id == "brain"
    assert trace.alice_brain_command == "trace"
    assert trace.alice_brain_trace_command == "list"
    assert trace.after_sequence == 4
    assert trace.limit == 25
    assert explicit_identity.alice_brain_identity_command == "get"
    assert explicit_trace.alice_brain_trace_command == "list"


def test_usage_error_is_machine_json_and_does_not_create_home(tmp_path: Path) -> None:
    home = tmp_path / "runtime"

    error = _error(_run_cli(home, "daemon", "unknown"))

    assert error["code"] == "usage_error"
    assert not home.exists()


def test_stopped_status_is_success_and_has_no_filesystem_side_effect(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"

    result = _success(_run_cli(home, "daemon", "status"))

    assert result["command"] == "daemon.status"
    assert result["data"] == {
        "daemon": None,
        "running": False,
        "runtime_home": os.fspath(home.absolute()),
        "status": "stopped",
    }
    assert not home.exists()


def test_status_payload_rejects_unknown_or_identity_collision_fields() -> None:
    from alice_brain_hermes import cli

    health = {
        "pid": 123,
        "instance_nonce": "nonce",
        "launch_nonce": "launch",
        "package_version": "0.1.0",
        "process_marker": "marker",
        "shutting_down": False,
        "protocol_version": 2,
        "runtime_ready": True,
        "brain_count": 0,
        "engine_count": 0,
        "scheduler_count": 0,
        "running_scheduler_count": 0,
        "degraded_brain_count": 0,
    }
    runtime = {
        "schema_version": 1,
        "runtime_mode": "continuous_daemon",
        "cognition_mode": "local",
        "continuous_runtime": True,
        "brain_ids": [],
        "engine_count": 0,
        "scheduler_count": 0,
        "runtime_ready": True,
        "scheduler_health": {
            "status": "healthy",
            "fail_stopped": False,
            "brain_count": 0,
            "engine_count": 0,
            "scheduler_count": 0,
            "running_scheduler_count": 0,
            "degraded_brain_count": 0,
        },
        "bridge_connection": {
            "state": "never_connected",
            "total_bridges": 0,
            "connected_open_bridges": 0,
            "disconnected_open_bridges": 0,
            "clean_closed_bridges": 0,
            "abandoned_bridges": 0,
        },
        "trace_complete": True,
        "semantic_complete": True,
        "dropped_events": 0,
        "semantic_evidence": {
            "semantic_records": 0,
            "legacy_raw_only_records": 0,
            "semantic_gap_records": 0,
        },
        "host_state_scope": "registered_hook_payloads_only",
        "unobserved_hermes_fields": [
            "chunk_capture",
            "reasoning_capture",
        ],
        "schema_versions": {
            "protocol": 2,
            "observer": 1,
            "record": 1,
            "gap": 1,
            "frame": 3,
            "semantic": 1,
            "sqlite": 6,
        },
    }

    cli._validate_status_payloads(health, runtime)
    with pytest.raises(cli._CliFailure) as failure:
        cli._validate_status_payloads({**health, "unexpected": 999}, runtime)

    assert failure.value.code == "daemon_status_invalid"
    with pytest.raises(cli._CliFailure):
        cli._validate_status_payloads(health, {**runtime, "brain_count": 0})
    changed_schema = dict(runtime)
    changed_schema["schema_versions"] = {
        **runtime["schema_versions"],
        "protocol": 1,
    }
    with pytest.raises(cli._CliFailure):
        cli._validate_status_payloads(health, changed_schema)


def test_runtime_home_precedence_uses_project_env_not_alice_brain_home(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected"
    forbidden = tmp_path / "alice-brain-home"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(SOURCE_ROOT)
    environment["ALICE_BRAIN_HERMES_HOME"] = os.fspath(selected)
    environment["ALICE_BRAIN_HOME"] = os.fspath(forbidden)

    result = subprocess.run(
        [sys.executable, "-m", "alice_brain_hermes.cli", "status"],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    body = _success(result)
    assert body["data"]["runtime_home"] == os.fspath(selected.absolute())
    assert not selected.exists()
    assert not forbidden.exists()


def test_explicit_runtime_home_overrides_project_environment(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    explicit = tmp_path / "explicit"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(SOURCE_ROOT)
    environment["ALICE_BRAIN_HERMES_HOME"] = os.fspath(configured)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alice_brain_hermes.cli",
            "--home",
            os.fspath(explicit),
            "status",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    body = _success(result)
    assert body["data"]["runtime_home"] == os.fspath(explicit.absolute())
    assert not configured.exists()
    assert not explicit.exists()


def test_invalid_existing_discovery_never_masquerades_as_stopped(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    discovery = home / "daemon.json"
    discovery.write_text("{not-json", encoding="utf-8")
    discovery.chmod(0o600)

    error = _error(_run_cli(home, "status"))

    assert error["code"] == "discovery_invalid"


def test_broken_runtime_home_symlink_never_masquerades_as_stopped(
    tmp_path: Path,
    make_symlink: Callable[[Path, Path, bool], bool],
) -> None:
    from alice_brain_hermes import cli

    home = tmp_path / "runtime"
    make_symlink(home, tmp_path / "missing", True)

    with pytest.raises(cli._CliFailure) as captured:
        cli._status(home, timeout_seconds=1)

    assert captured.value.code == "runtime_home_invalid"


def test_broken_discovery_symlink_never_masquerades_as_stopped(
    tmp_path: Path,
    make_symlink: Callable[[Path, Path, bool], bool],
) -> None:
    from alice_brain_hermes import cli

    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    discovery = home / "daemon.json"
    make_symlink(discovery, tmp_path / "missing-discovery", False)

    with pytest.raises(cli._CliFailure) as captured:
        cli._status(home, timeout_seconds=1)

    assert captured.value.code == "discovery_invalid"


def test_start_is_idempotent_status_is_live_and_stop_is_authenticated(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    first_pid = 0
    try:
        first = _success(_run_cli(home, "start"))
        first_data = first["data"]
        first_pid = int(first_data["pid"])
        assert first["command"] == "daemon.start"
        assert first_data["status"] == "running"
        assert first_data["started"] is True
        assert first_data["already_running"] is False
        assert first_data["readiness_verified"] is True
        assert type(first_data["instance_nonce"]) is str
        assert home.is_dir()

        repeated = _success(_run_cli(home, "daemon", "start"))["data"]
        assert repeated["started"] is False
        assert repeated["already_running"] is True
        assert repeated["pid"] == first_pid
        assert repeated["instance_nonce"] == first_data["instance_nonce"]

        status = _success(_run_cli(home, "status"))
        status_data = status["data"]
        assert status["command"] == "daemon.status"
        assert status_data["running"] is True
        assert status_data["status"] == "running"
        assert status_data["daemon"]["instance_nonce"] == first_data["instance_nonce"]
        assert status_data["daemon"]["pid"] == first_pid
        assert status_data["daemon"]["continuous_runtime"] is True
        assert status_data["daemon"]["runtime_mode"] == "continuous_daemon"
        assert status_data["daemon"]["cognition_mode"] == "local"
        assert status_data["daemon"]["scheduler_health"]["status"] == "healthy"
        assert status_data["daemon"]["bridge_connection"]["state"] == (
            "never_connected"
        )
        assert status_data["daemon"]["trace_complete"] is True
        assert status_data["daemon"]["semantic_complete"] is True
        assert status_data["daemon"]["dropped_events"] == 0

        doctor = _success(_run_cli(home, "doctor"))
        assert doctor["data"]["healthy"] is True
        assert {item["status"] for item in doctor["data"]["checks"]} == {"pass"}

        stopped = _success(_run_cli(home, "stop", timeout=20))
        assert stopped["command"] == "daemon.stop"
        assert stopped["data"] == {
            "already_stopped": False,
            "instance_nonce": first_data["instance_nonce"],
            "pid": first_pid,
            "status": "stopped",
            "stop_requested": True,
        }
        assert not (home / "daemon.json").exists()
        assert not list(home.glob("credential-*.key"))
        assert not psutil.pid_exists(first_pid)

        repeated_stop = _success(_run_cli(home, "daemon", "stop"))["data"]
        assert repeated_stop == {
            "already_stopped": True,
            "instance_nonce": None,
            "status": "stopped",
            "stop_requested": False,
        }

        restarted = _success(_run_cli(home, "daemon", "start"))["data"]
        restarted_pid = int(restarted["pid"])
        assert restarted["started"] is True
        assert restarted_pid != first_pid
        assert restarted["instance_nonce"] != first_data["instance_nonce"]
        repeated_restart = _success(_run_cli(home, "start"))["data"]
        assert repeated_restart["started"] is False
        assert repeated_restart["pid"] == restarted_pid
        assert repeated_restart["instance_nonce"] == restarted["instance_nonce"]
        final_stop = _success(_run_cli(home, "stop", timeout=20))["data"]
        assert final_stop["pid"] == restarted_pid
        assert final_stop["instance_nonce"] == restarted["instance_nonce"]
        assert not psutil.pid_exists(restarted_pid)
    finally:
        _stop_if_running(home)


def test_concurrent_real_cli_starts_converge_on_one_healthy_daemon(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    barrier = threading.Barrier(2)

    def start() -> subprocess.CompletedProcess[str]:
        barrier.wait(timeout=5.0)
        return _run_cli(home, "start", timeout=30.0)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(lambda _index: start(), range(2)))
        bodies = tuple(_success(result)["data"] for result in results)
        assert sorted(body["started"] for body in bodies) == [False, True]
        assert len({body["pid"] for body in bodies}) == 1
        assert len({body["instance_nonce"] for body in bodies}) == 1
        status = _success(_run_cli(home, "status"))["data"]
        assert status["running"] is True
        assert status["daemon"]["pid"] == bodies[0]["pid"]
        assert status["daemon"]["runtime_ready"] is True
    finally:
        _stop_if_running(home)


def test_identity_and_trace_commands_use_the_live_typed_rpc(tmp_path: Path) -> None:
    from alice_brain_hermes.protocol.client import DaemonClient

    home = tmp_path / "runtime"
    try:
        _success(_run_cli(home, "daemon", "start"))
        client = DaemonClient.connect(home, timeout_seconds=10.0)
        try:
            created = client.call("brain.create", {"name": "Mira"})
        finally:
            client.close()

        identity = _success(
            _run_cli(home, "identity", "--brain-id", created["brain_id"])
        )
        first = _success(
            _run_cli(
                home,
                "trace",
                "--brain-id",
                created["brain_id"],
                "--limit",
                "1",
            )
        )
        second = _success(
            _run_cli(
                home,
                "trace",
                "--brain-id",
                created["brain_id"],
                "--after-sequence",
                str(first["data"]["next_after_sequence"]),
                "--limit",
                "10",
            )
        )

        assert identity["command"] == "identity.get"
        assert identity["data"]["brain_id"] == created["brain_id"]
        assert identity["data"]["name"] == "Mira"
        assert first["command"] == "trace.list"
        assert first["data"]["returned_count"] == 1
        assert first["data"]["events"][0]["sequence"] == 1
        assert type(first["data"]["has_more"]) is bool
        assert all(
            event["sequence"] > first["data"]["next_after_sequence"]
            for event in second["data"]["events"]
        )
        assert type(second["data"]["has_more"]) is bool
    finally:
        _stop_if_running(home)


def test_daemon_adapter_uses_fresh_no_credential_no_descriptor_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alice_brain_hermes import cli

    captured: list[tuple[Path, list[str], str]] = []
    sentinel = object()

    def create(home: Path, command: list[str], *, launch_nonce: str):
        captured.append((home, command, launch_nonce))
        return sentinel

    monkeypatch.setattr(cli.DmonAdapter, "create", create)

    home = tmp_path / "runtime"
    assert cli._daemon_adapter(home, "launch-nonce") is sentinel
    [(selected_home, command, nonce)] = captured
    assert selected_home == home
    assert nonce == "launch-nonce"
    assert command == [
        sys.executable,
        "-m",
        "alice_brain_hermes.runtime.daemon",
        "--runtime-home",
        os.fspath(home),
        "--launch-nonce",
        "launch-nonce",
    ]
    assert all("credential" not in argument for argument in command)
    assert all("readiness" not in argument for argument in command)


def test_start_requires_authenticated_exact_v2_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alice_brain_hermes import __version__, cli
    from alice_brain_hermes.protocol.models import DaemonDiscoveryV2, LoopbackEndpointV1
    from alice_brain_hermes.runtime.supervisor import DmonProcessHint

    home = tmp_path / "runtime"
    hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    class Adapter:
        launch_nonce = "launch-exact"

        def start(self) -> DmonProcessHint:
            return hint

        def release_parent_guard(self) -> None:
            return None

    discovery = DaemonDiscoveryV2(
        pid=hint.pid,
        process_marker=hint.process_marker,
        instance_nonce="instance-exact",
        launch_nonce="launch-exact",
        endpoint=LoopbackEndpointV1(port=43210),
        credential_ref="credential-instance-exact.key",
    )
    health = {
        "pid": discovery.pid,
        "process_marker": discovery.process_marker,
        "instance_nonce": discovery.instance_nonce,
        "launch_nonce": discovery.launch_nonce,
        "protocol_version": discovery.protocol_version,
        "package_version": __version__,
        "runtime_ready": True,
    }
    monkeypatch.setattr(cli, "_home_exists", lambda _home: False)
    monkeypatch.setattr(cli.secrets, "token_hex", lambda _size: "launch-exact")
    monkeypatch.setattr(cli, "_daemon_adapter", lambda *_args: Adapter())
    monkeypatch.setattr(cli, "_discovery_exists", lambda _home: True)
    monkeypatch.setattr(
        cli,
        "_live_status",
        lambda *_args, **_kwargs: (discovery, health, {"runtime_ready": True}),
    )

    class Process:
        pid = hint.pid

        @staticmethod
        def create_time() -> float:
            return 123.0

    monkeypatch.setattr(cli.psutil, "Process", lambda _pid: Process())

    assert cli._start(home, timeout_seconds=1.0) == {
        "status": "running",
        "started": True,
        "already_running": False,
        "readiness_verified": True,
        "pid": 4321,
        "instance_nonce": "instance-exact",
    }


@pytest.mark.parametrize(
    ("parent_pid", "parent_marker", "accepted"),
    [
        (4321, "psutil-create-time-us:123000000", True),
        (9999, "psutil-create-time-us:123000000", False),
        (4321, "psutil-create-time-us:999000000", False),
    ],
)
def test_authenticated_redirector_child_requires_exact_parent_and_marker(
    monkeypatch: pytest.MonkeyPatch,
    parent_pid: int,
    parent_marker: str,
    accepted: bool,
) -> None:
    from alice_brain_hermes import __version__, cli
    from alice_brain_hermes.protocol.models import DaemonDiscoveryV2, LoopbackEndpointV1
    from alice_brain_hermes.runtime.supervisor import DmonProcessHint

    hint = DmonProcessHint(
        pid=4321,
        process_marker=parent_marker,
    )
    discovery = DaemonDiscoveryV2(
        pid=8765,
        process_marker="psutil-create-time-us:456000000",
        instance_nonce="instance-child",
        launch_nonce="launch-child",
        endpoint=LoopbackEndpointV1(port=43210),
        credential_ref="credential-instance-child.key",
    )
    health = {
        "pid": discovery.pid,
        "process_marker": discovery.process_marker,
        "instance_nonce": discovery.instance_nonce,
        "launch_nonce": discovery.launch_nonce,
        "protocol_version": discovery.protocol_version,
        "package_version": __version__,
        "runtime_ready": True,
    }

    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def ppid(self) -> int:
            assert self.pid == discovery.pid
            return parent_pid

        def create_time(self) -> float:
            return 123.0 if self.pid == hint.pid else 456.0

    monkeypatch.setattr(cli.psutil, "Process", Process)

    assert (
        cli._authenticated_start_identity(
            discovery,
            health,
            {"runtime_ready": True},
            adapter=SimpleNamespace(launch_nonce="launch-child"),
            hint=hint,
        )
        is accepted
    )


def test_start_timeout_exactly_terminates_hint_and_returns_redacted_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alice_brain_hermes import cli
    from alice_brain_hermes.runtime.supervisor import DmonProcessHint

    hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    actions: list[tuple[object, ...]] = []

    class Adapter:
        launch_nonce = "launch-timeout"

        def start(self) -> DmonProcessHint:
            return hint

        def release_parent_guard(self) -> None:
            return None

        def terminate_exact(
            self, selected: DmonProcessHint, *, timeout_seconds: float
        ) -> None:
            actions.append(("terminate", selected, timeout_seconds))

        def remove_meta_hint(self, selected: DmonProcessHint) -> bool:
            actions.append(("meta", selected))
            return True

        def redacted_dmon_output(self) -> str:
            return "<redacted-dmon>"

        def redacted_log_tail(self) -> str:
            return "<redacted-log>"

    monkeypatch.setattr(cli, "_home_exists", lambda _home: False)
    monkeypatch.setattr(cli, "_daemon_adapter", lambda *_args: Adapter())
    monkeypatch.setattr(cli, "_discovery_exists", lambda _home: False)
    monkeypatch.setattr(
        cli,
        "_process_identity_state",
        lambda _pid, _marker: "exact",
    )
    monotonic_values = iter((0.0, 0.0, 0.01))
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    with pytest.raises(cli._CliFailure) as failure:
        cli._start_coordinated(tmp_path / "runtime", timeout_seconds=0.01)

    assert failure.value.code == "startup_timeout"
    assert failure.value.data == {
        "dmon_output": "<redacted-dmon>",
        "log_tail": "<redacted-log>",
    }
    assert actions[0][0] == "terminate"
    assert actions[0][1] == hint
    assert actions[1] == ("meta", hint)


def test_start_cleanup_fails_closed_when_exact_identity_is_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alice_brain_hermes import cli
    from alice_brain_hermes.runtime.supervisor import (
        DmonIdentityError,
        DmonProcessHint,
    )

    hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )

    class Adapter:
        launch_nonce = "launch-unverifiable"

        def start(self) -> DmonProcessHint:
            return hint

        def release_parent_guard(self) -> None:
            return None

        def terminate_exact(self, *_args, **_kwargs) -> None:
            raise DmonIdentityError("unverifiable")

        def redacted_dmon_output(self) -> str:
            return ""

        def redacted_log_tail(self) -> str:
            return ""

    monkeypatch.setattr(cli, "_home_exists", lambda _home: False)
    monkeypatch.setattr(cli, "_daemon_adapter", lambda *_args: Adapter())
    monkeypatch.setattr(cli, "_discovery_exists", lambda _home: False)
    monkeypatch.setattr(
        cli,
        "_process_identity_state",
        lambda _pid, _marker: "exact",
    )
    monotonic_values = iter((0.0, 0.0, 0.01))
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    with pytest.raises(cli._CliFailure) as failure:
        cli._start_coordinated(tmp_path / "runtime", timeout_seconds=0.01)

    assert failure.value.code == "startup_cleanup_unproven"


def test_process_identity_distinguishes_exact_reused_gone_and_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes import cli

    monkeypatch.setattr(cli, "read_process_marker", lambda _pid: "expected")
    assert cli._process_identity_state(10, "expected") == "exact"
    assert cli._process_identity_state(10, "other") == "reused"

    def unreadable(_pid: int) -> str:
        raise PermissionError("unverifiable")

    monkeypatch.setattr(cli, "read_process_marker", unreadable)
    monkeypatch.setattr(cli.psutil, "pid_exists", lambda _pid: False)
    assert cli._process_identity_state(10, "expected") == "gone"
    monkeypatch.setattr(cli.psutil, "pid_exists", lambda _pid: True)
    assert cli._process_identity_state(10, "expected") == "ambiguous"


def test_stop_fails_closed_when_process_identity_is_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes import cli
    from alice_brain_hermes.runtime.supervisor import DmonProcessHint

    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    discovery = SimpleNamespace(
        pid=22,
        process_marker="current-marker",
        instance_nonce="current-nonce",
        launch_nonce="current-launch",
        credential_ref="credential-current.key",
    )

    class Client:
        def __init__(self) -> None:
            self.discovery = discovery

        def shutdown(self) -> dict[str, object]:
            return {"accepted": True}

        def close(self) -> None:
            return None

    discovery_checks = 0

    def discovery_exists(_home: Path) -> bool:
        nonlocal discovery_checks
        discovery_checks += 1
        return discovery_checks == 1

    monkeypatch.setattr(cli, "_validate_existing_home", lambda _home: None)
    monkeypatch.setattr(cli, "_load_discovery", lambda _home: (discovery, "token"))
    monkeypatch.setattr(cli.DaemonClient, "connect", lambda *_args, **_kwargs: Client())
    monkeypatch.setattr(
        cli,
        "_daemon_adapter",
        lambda *_args: SimpleNamespace(
            current_process_hint=lambda: DmonProcessHint(
                pid=11,
                process_marker="supervisor-marker",
            )
        ),
    )
    monkeypatch.setattr(cli, "_discovery_matches_supervisor", lambda *_args: True)
    monkeypatch.setattr(cli, "_discovery_exists", discovery_exists)
    monkeypatch.setattr(
        cli,
        "_process_identity_state",
        lambda _pid, _marker: "ambiguous",
    )
    monotonic_values = iter((0.0, 0.0, 0.01))
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    with pytest.raises(cli._CliFailure) as failure:
        cli._stop_coordinated(home, timeout_seconds=0.01)

    assert failure.value.code == "shutdown_unproven"


def test_stop_waits_for_exact_supervisor_exit_before_artifact_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes import cli
    from alice_brain_hermes.runtime.supervisor import DmonProcessHint

    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    discovery = SimpleNamespace(
        pid=22,
        process_marker="child-marker",
        instance_nonce="current-nonce",
        launch_nonce="current-launch",
        credential_ref="credential-current.key",
    )
    supervisor = DmonProcessHint(pid=11, process_marker="supervisor-marker")
    cleanup_checks: list[DmonProcessHint] = []
    coordination: list[str] = []
    coordinator_held = False
    supervisor_checks = 0
    discovery_checks = 0

    class Client:
        def __init__(self) -> None:
            self.discovery = discovery

        def shutdown(self) -> dict[str, object]:
            return {"accepted": True}

        def close(self) -> None:
            return None

    class Adapter:
        def current_process_hint(self) -> DmonProcessHint:
            return supervisor

        def remove_meta_hint(self, hint: DmonProcessHint) -> bool:
            assert coordinator_held is True
            cleanup_checks.append(hint)
            return True

    class Coordinator:
        def acquire(self, *, timeout_seconds: float) -> None:
            nonlocal coordinator_held
            assert timeout_seconds > 0
            assert coordinator_held is False
            coordinator_held = True
            coordination.append("acquire")

        def release(self) -> None:
            nonlocal coordinator_held
            assert coordinator_held is True
            coordinator_held = False
            coordination.append("release")

    def process_state(pid: int, marker: str) -> str:
        nonlocal supervisor_checks
        if (pid, marker) == (discovery.pid, discovery.process_marker):
            return "gone"
        assert (pid, marker) == (supervisor.pid, supervisor.process_marker)
        supervisor_checks += 1
        return "exact" if supervisor_checks == 1 else "gone"

    def discovery_exists(_home: Path) -> bool:
        nonlocal discovery_checks
        discovery_checks += 1
        return discovery_checks == 1

    monkeypatch.setattr(cli, "_validate_existing_home", lambda _home: None)
    monkeypatch.setattr(cli, "_load_discovery", lambda _home: (discovery, "token"))
    monkeypatch.setattr(cli.DaemonClient, "connect", lambda *_args, **_kwargs: Client())
    monkeypatch.setattr(cli, "_daemon_adapter", lambda *_args: Adapter())
    monkeypatch.setattr(cli, "_discovery_matches_supervisor", lambda *_args: True)
    monkeypatch.setattr(cli, "_discovery_exists", discovery_exists)
    monkeypatch.setattr(cli, "_process_identity_state", process_state)
    monkeypatch.setattr(
        cli.DmonStartCoordinator,
        "create",
        lambda _home: Coordinator(),
    )

    result = cli._stop(home, timeout_seconds=1.0)

    assert result["status"] == "stopped"
    assert supervisor_checks == 2
    assert cleanup_checks == [supervisor]
    assert coordination == ["acquire", "release"]
    assert coordinator_held is False


def test_stop_ignores_unrelated_credential_file(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    unrelated = home / "credential-unproven.key"
    try:
        started = _success(_run_cli(home, "daemon", "start"))["data"]
        unrelated.write_text("not-daemon-authority", encoding="utf-8")
        unrelated.chmod(0o600)

        stopped = _success(_run_cli(home, "daemon", "stop", timeout=20))["data"]

        assert stopped["pid"] == started["pid"]
        assert stopped["instance_nonce"] == started["instance_nonce"]
        assert unrelated.read_text(encoding="utf-8") == "not-daemon-authority"
    finally:
        _stop_if_running(home)


def test_stop_verifies_the_discovery_authenticated_by_its_client(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(SOURCE_ROOT)
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        from alice_brain_hermes import cli

        home = Path(sys.argv[1])
        old = SimpleNamespace(
            pid=11,
            process_marker="old-marker",
            instance_nonce="old-nonce",
            credential_ref="credential-old.key",
        )
        current = SimpleNamespace(
            pid=22,
            process_marker="current-marker",
            instance_nonce="current-nonce",
            launch_nonce="current-launch",
            credential_ref="credential-current.key",
        )
        class Client:
            discovery = current
            def shutdown(self):
                return {"accepted": True}
            def close(self):
                return None

        discovery_checks = []
        process_checks = []
        cli._validate_existing_home = lambda _home: None
        cli._load_discovery = lambda _home: (old, "token")
        cli.DaemonClient.connect = lambda *_args, **_kwargs: Client()
        cli._discovery_exists = (
            lambda _home: not discovery_checks.append(_home)
            and len(discovery_checks) == 1
        )
        cli._process_identity_state = (
            lambda pid, marker: process_checks.append((pid, marker)) or "gone"
        )
        supervisor = SimpleNamespace(pid=33, process_marker="supervisor-marker")
        class Adapter:
            def current_process_hint(self):
                return supervisor
            def remove_meta_hint(self, _hint):
                return True
        cli._daemon_adapter = lambda *_args: Adapter()
        cli._discovery_matches_supervisor = lambda *_args: True

        result = cli._stop(home, timeout_seconds=1.0)
        assert result["pid"] == current.pid
        assert result["instance_nonce"] == current.instance_nonce
        assert process_checks == [
            (current.pid, current.process_marker),
            (supervisor.pid, supervisor.process_marker),
        ]
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script, os.fspath(home)],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_doctor_reports_checks_on_stdout_and_uses_exit_four_for_failures(
    tmp_path: Path,
) -> None:
    home = tmp_path / "missing-runtime"

    result = _run_cli(home, "doctor")

    assert result.returncode == 4
    assert result.stderr == ""
    body = json.loads(result.stdout)
    assert result.stdout == _canonical(body)
    assert body["schema_version"] == 1
    assert body["ok"] is True
    assert body["command"] == "doctor"
    assert body["data"]["healthy"] is False
    assert type(body["data"]["checks"]) is list
    assert body["data"]["checks"]
    assert {item["status"] for item in body["data"]["checks"]} <= {
        "pass",
        "warn",
        "fail",
        "skip",
    }
    assert not home.exists()


def test_doctor_never_claims_green_without_bridge_and_trace_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes import cli

    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(cli, "_validate_existing_home", lambda _home: None)
    monkeypatch.setattr(cli, "_discovery_exists", lambda _home: True)
    monkeypatch.setattr(
        cli,
        "_status",
        lambda _home, *, timeout_seconds: {
            "running": True,
            "daemon": {
                "runtime_ready": True,
                "shutting_down": False,
                "degraded_brain_count": 0,
                "instance_nonce": "nonce",
            },
        },
    )

    data, exit_code = cli._doctor(home, timeout_seconds=1.0)

    assert exit_code == 4
    assert data["healthy"] is False
    integration = next(
        item for item in data["checks"] if item["id"] == "integration_observability"
    )
    assert integration["status"] == "fail"
    assert "bridge_connection" in integration["data"]["missing_fields"]


@pytest.mark.parametrize(
    ("changes", "failed_evidence"),
    [
        ({"trace_complete": False, "semantic_complete": False}, "trace_complete"),
        ({"semantic_complete": False}, "semantic_complete"),
        (
            {
                "trace_complete": False,
                "semantic_complete": False,
                "dropped_events": 2,
            },
            "dropped_events",
        ),
        (
            {
                "bridge_connection": {
                    "state": "degraded",
                    "total_bridges": 1,
                    "connected_open_bridges": 0,
                    "disconnected_open_bridges": 1,
                    "clean_closed_bridges": 0,
                    "abandoned_bridges": 0,
                }
            },
            "bridge_connection",
        ),
    ],
)
def test_doctor_fails_visible_persisted_integration_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    failed_evidence: str,
) -> None:
    from alice_brain_hermes import cli

    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    daemon: dict[str, object] = {
        "runtime_ready": True,
        "shutting_down": False,
        "degraded_brain_count": 0,
        "instance_nonce": "nonce",
        "scheduler_health": {"status": "healthy"},
        "bridge_connection": {
            "state": "never_connected",
            "total_bridges": 0,
            "connected_open_bridges": 0,
            "disconnected_open_bridges": 0,
            "clean_closed_bridges": 0,
            "abandoned_bridges": 0,
        },
        "cognition_mode": "local",
        "trace_complete": True,
        "semantic_complete": True,
        "dropped_events": 0,
        "host_state_scope": "registered_hook_payloads_only",
        "unobserved_hermes_fields": [
            "chunk_capture",
            "reasoning_capture",
        ],
        "schema_versions": {
            "protocol": 2,
            "observer": 1,
            "record": 1,
            "gap": 1,
            "frame": 3,
            "semantic": 1,
            "sqlite": 6,
        },
    }
    daemon.update(changes)
    monkeypatch.setattr(cli, "_validate_existing_home", lambda _home: None)
    monkeypatch.setattr(cli, "_discovery_exists", lambda _home: True)
    monkeypatch.setattr(
        cli,
        "_status",
        lambda _home, *, timeout_seconds: {
            "running": True,
            "daemon": daemon,
        },
    )

    data, exit_code = cli._doctor(home, timeout_seconds=1.0)

    assert exit_code == 4
    integration = next(
        item for item in data["checks"] if item["id"] == "integration_observability"
    )
    assert integration["status"] == "fail"
    assert failed_evidence in integration["data"]["failed_evidence"]
