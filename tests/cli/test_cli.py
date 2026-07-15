from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX daemon lifecycle")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


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
    assert "doctor" in result.stdout
    assert not home.exists()


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
        "instance_nonce": "nonce",
        "process_marker": "marker",
        "shutting_down": False,
        "protocol_version": 1,
        "runtime_ready": True,
        "brain_count": 0,
        "engine_count": 0,
        "scheduler_count": 0,
        "running_scheduler_count": 0,
        "degraded_brain_count": 0,
    }
    runtime = {
        "brain_ids": [],
        "engine_count": 0,
        "scheduler_count": 0,
        "continuous_runtime": True,
    }

    cli._validate_status_payloads(health, runtime)
    with pytest.raises(cli._CliFailure) as failure:
        cli._validate_status_payloads({**health, "pid": 999}, runtime)

    assert failure.value.code == "daemon_status_invalid"


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


@POSIX_ONLY
def test_broken_runtime_home_symlink_never_masquerades_as_stopped(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.symlink_to(tmp_path / "missing", target_is_directory=True)

    error = _error(_run_cli(home, "status"))

    assert error["code"] == "runtime_home_invalid"


@POSIX_ONLY
def test_broken_discovery_symlink_never_masquerades_as_stopped(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    (home / "daemon.json").symlink_to(tmp_path / "missing-discovery")

    error = _error(_run_cli(home, "status"))

    assert error["code"] == "discovery_invalid"


@POSIX_ONLY
def test_start_is_idempotent_status_is_live_and_stop_is_authenticated(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    first_pid = 0
    try:
        first = _success(_run_cli(home, "daemon", "start"))
        first_data = first["data"]
        first_pid = int(first_data["pid"])
        assert first["command"] == "daemon.start"
        assert first_data["status"] == "running"
        assert first_data["started"] is True
        assert first_data["already_running"] is False
        assert first_data["readiness_verified"] is True
        assert type(first_data["instance_nonce"]) is str
        assert stat.S_IMODE(home.stat().st_mode) == 0o700

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

        stopped = _success(_run_cli(home, "daemon", "stop", timeout=20))
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
        assert not Path(f"/proc/{first_pid}").exists()

        repeated_stop = _success(_run_cli(home, "daemon", "stop"))["data"]
        assert repeated_stop == {
            "already_stopped": True,
            "instance_nonce": None,
            "status": "stopped",
            "stop_requested": False,
        }
    finally:
        _stop_if_running(home)


def test_invalid_readiness_terminates_only_the_spawned_child(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(SOURCE_ROOT)
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from alice_brain_hermes import cli

        spawned = object()
        terminated = []
        cli._spawn_daemon = lambda _home: spawned
        cli._read_readiness = (
            lambda _process, *, timeout_seconds: b"not-json\\n"
        )
        cli._terminate_spawned = (
            lambda process, *, timeout_seconds:
            terminated.append((process, timeout_seconds))
        )
        try:
            cli._start(Path(sys.argv[1]), timeout_seconds=2.5)
        except cli._CliFailure as error:
            assert "readiness" in str(error)
        else:
            raise AssertionError("invalid readiness was accepted")
        assert terminated == [(spawned, 2.5)]
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script, os.fspath(tmp_path / "runtime")],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "payload",
    [
        b'{"ready":true,"instance_nonce":"nonce"}\ntrailing',
        b'{"ready":true,"ready":false,"code":"startup_failed"}\n',
        b'{"ready":true,"instance_nonce":"nonce","extra":1}\n',
        b'{"ready":true,"instance_nonce":NaN}\n',
        b'{"ready":false,"code":"unknown"}\n',
    ],
)
def test_readiness_rejects_trailing_duplicate_nonfinite_or_unknown_shapes(
    payload: bytes,
) -> None:
    from alice_brain_hermes import cli

    with pytest.raises(cli._CliFailure, match="readiness"):
        cli._decode_readiness(payload)


def test_readiness_reader_includes_trailing_pipe_bytes() -> None:
    from alice_brain_hermes import cli

    process = SimpleNamespace(
        stdout=BytesIO(b'{"ready":true,"instance_nonce":"nonce"}\nextra')
    )

    payload = cli._read_readiness(process, timeout_seconds=1.0)

    assert payload.endswith(b"e")
    with pytest.raises(cli._CliFailure, match="readiness"):
        cli._decode_readiness(payload)


def test_false_readiness_cannot_leave_spawned_child_unreaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes import cli

    class ResistantChild:
        def wait(self, *, timeout: float) -> int:
            raise subprocess.TimeoutExpired("daemon", timeout)

    child = ResistantChild()
    terminated: list[tuple[object, float]] = []
    monkeypatch.setattr(cli, "_spawn_daemon", lambda _home: child)
    monkeypatch.setattr(
        cli,
        "_read_readiness",
        lambda _process, *, timeout_seconds: (
            b'{"ready":false,"code":"startup_failed"}\n'
        ),
    )
    monkeypatch.setattr(
        cli,
        "_terminate_spawned",
        lambda process, *, timeout_seconds: terminated.append(
            (process, timeout_seconds)
        ),
    )

    with pytest.raises(cli._CliFailure, match="startup failed"):
        cli._start(tmp_path / "runtime", timeout_seconds=2.5)

    assert terminated == [(child, 2.5)]


def test_spawn_cleanup_fails_closed_when_exit_cannot_be_proven() -> None:
    from alice_brain_hermes import cli

    class UnreapableChild:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            raise OSError("denied")

        def wait(self, *, timeout: float) -> int:
            raise subprocess.TimeoutExpired("daemon", timeout)

        def kill(self) -> None:
            raise OSError("denied")

    with pytest.raises(cli._CliFailure) as failure:
        cli._terminate_spawned(UnreapableChild(), timeout_seconds=0.01)

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

    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    discovery = SimpleNamespace(
        pid=22,
        process_marker="current-marker",
        instance_nonce="current-nonce",
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
    monkeypatch.setattr(cli, "_discovery_exists", discovery_exists)
    monkeypatch.setattr(
        cli,
        "_process_identity_state",
        lambda _pid, _marker: "ambiguous",
    )

    with pytest.raises(cli._CliFailure) as failure:
        cli._stop(home, timeout_seconds=0.01)

    assert failure.value.code == "shutdown_unproven"


@POSIX_ONLY
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

        result = cli._stop(home, timeout_seconds=1.0)
        assert result["pid"] == current.pid
        assert result["instance_nonce"] == current.instance_nonce
        assert process_checks == [(current.pid, current.process_marker)]
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
