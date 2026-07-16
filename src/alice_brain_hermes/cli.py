"""Machine-stable operator CLI for the private consciousness daemon."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Literal, NoReturn

import psutil
from pydantic import ValidationError

from alice_brain_hermes import __version__
from alice_brain_hermes.errors import DaemonClientError, DaemonRpcError
from alice_brain_hermes.protocol.client import DaemonClient
from alice_brain_hermes.protocol.diagnostics import (
    TRACE_MAX_PAGE_SIZE,
    IdentitySnapshotV1,
    TracePageV1,
)
from alice_brain_hermes.protocol.models import PROTOCOL_VERSION, DaemonDiscoveryV2
from alice_brain_hermes.protocol.status import DaemonRuntimeStatusV1
from alice_brain_hermes.runtime.discovery import (
    _private_home,
    load_discovery_and_credential,
)
from alice_brain_hermes.runtime.process_marker import read_process_marker
from alice_brain_hermes.runtime.supervisor import (
    DmonAdapter,
    DmonCoordinationTimeout,
    DmonIdentityError,
    DmonProcessHint,
    DmonStartCoordinator,
)

_CLI_SCHEMA_VERSION = 1
_DEFAULT_TIMEOUT_SECONDS = 10.0
_MAX_TIMEOUT_SECONDS = 300.0


class _UsageError(Exception):
    """An argparse failure that must use the machine error channel."""


class _CliFailure(Exception):
    """One sanitized operator-facing failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        data: dict[str, object] | None = None,
        exit_code: int = 5,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
        self.exit_code = exit_code


class _MachineArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _canonical_json(value: object) -> str:
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


def _write_success(command: str, data: dict[str, object]) -> None:
    sys.stdout.write(
        _canonical_json(
            {
                "schema_version": _CLI_SCHEMA_VERSION,
                "ok": True,
                "command": command,
                "data": data,
            }
        )
    )
    sys.stdout.flush()


def _write_error(failure: _CliFailure) -> None:
    sys.stderr.write(
        _canonical_json(
            {
                "schema_version": _CLI_SCHEMA_VERSION,
                "ok": False,
                "code": failure.code,
                "message": failure.message,
                "data": failure.data,
            }
        )
    )
    sys.stderr.flush()


def _timeout(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a number") from error
    if not math.isfinite(result) or not 0 < result <= _MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            "timeout must be finite, positive, and at most 300 seconds"
        )
    return result


def _nonnegative_sequence(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("after-sequence must be an integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("after-sequence cannot be negative")
    return result


def _trace_limit(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be an integer") from error
    if not 1 <= result <= TRACE_MAX_PAGE_SIZE:
        raise argparse.ArgumentTypeError(
            f"limit must be between 1 and {TRACE_MAX_PAGE_SIZE}"
        )
    return result


def _runtime_home(explicit: str | None) -> Path:
    if explicit is not None:
        if not explicit:
            raise _CliFailure(
                "invalid_home",
                "--home must not be empty",
                exit_code=2,
            )
        selected = explicit
    else:
        selected = os.environ.get("ALICE_BRAIN_HERMES_HOME") or os.fspath(
            Path.home() / ".alice-brain-hermes"
        )
    return Path(selected).expanduser().absolute()


def configure_control_parser(
    parser: argparse.ArgumentParser,
    *,
    command_required: bool = True,
) -> argparse.ArgumentParser:
    """Build the shared Python command surface without contacting the runtime."""

    if type(command_required) is not bool:
        raise TypeError("command_required must be a boolean")

    parser.add_argument(
        "--home",
        help=(
            "private runtime home (default: ALICE_BRAIN_HERMES_HOME or "
            "~/.alice-brain-hermes)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=_timeout,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="bounded startup, RPC, and shutdown timeout in seconds",
    )
    commands = parser.add_subparsers(
        dest="alice_brain_command",
        required=command_required,
    )

    daemon = commands.add_parser("daemon", help="run or control the private daemon")
    daemon_commands = daemon.add_subparsers(
        dest="alice_brain_daemon_command", required=True
    )
    daemon_commands.add_parser("run", help="run the daemon in the foreground")
    daemon_commands.add_parser("start", help="start the daemon in the background")
    daemon_commands.add_parser("stop", help="request authenticated daemon shutdown")
    daemon_commands.add_parser("status", help="report live daemon status")

    commands.add_parser("start", help="alias for 'daemon start'")
    commands.add_parser("stop", help="alias for 'daemon stop'")
    commands.add_parser("status", help="alias for 'daemon status'")
    commands.add_parser("doctor", help="diagnose package, home, and daemon health")
    identity = commands.add_parser(
        "identity", help="inspect replay-derived self identity"
    )
    identity.add_argument(
        "alice_brain_identity_command",
        nargs="?",
        choices=("get",),
        default="get",
        help="optional explicit operation (default: get)",
    )
    identity.add_argument(
        "--brain-id",
        help="brain UUID; optional only when exactly one brain exists",
    )
    trace = commands.add_parser("trace", help="inspect the ordered event trace")
    trace.add_argument(
        "alice_brain_trace_command",
        nargs="?",
        choices=("list",),
        default="list",
        help="optional explicit operation (default: list)",
    )
    trace.add_argument(
        "--brain-id",
        help="brain UUID; optional only when exactly one brain exists",
    )
    trace.add_argument(
        "--after-sequence",
        type=_nonnegative_sequence,
        default=0,
        help="exclusive non-negative event sequence cursor",
    )
    trace.add_argument(
        "--limit",
        type=_trace_limit,
        default=100,
        help=f"maximum events to return (1-{TRACE_MAX_PAGE_SIZE})",
    )
    return parser


def _parser() -> _MachineArgumentParser:
    parser = _MachineArgumentParser(
        prog="alice-brain-hermes",
        description="Inspect and control the Alice-brain-Hermes consciousness runtime",
    )
    configure_control_parser(parser)
    return parser


def _discovery_exists(home: Path) -> bool:
    discovery = home / "daemon.json"
    return discovery.exists() or discovery.is_symlink()


def _home_exists(home: Path) -> bool:
    return home.exists() or home.is_symlink()


def _validate_existing_home(home: Path) -> None:
    try:
        _private_home(home)
    except (OSError, ValueError) as error:
        raise _CliFailure(
            "runtime_home_invalid",
            "runtime home is not a verified private directory",
            exit_code=5,
        ) from error


def _load_discovery(home: Path):
    try:
        return load_discovery_and_credential(home)
    except (OSError, ValueError) as error:
        raise _CliFailure(
            "discovery_invalid",
            "daemon discovery or credential is invalid",
            exit_code=5,
        ) from error


def _validate_status_payloads(
    health: dict[str, object],
    runtime: dict[str, object],
) -> None:
    health_fields = {
        "pid",
        "instance_nonce",
        "launch_nonce",
        "package_version",
        "process_marker",
        "shutting_down",
        "protocol_version",
        "runtime_ready",
        "brain_count",
        "engine_count",
        "scheduler_count",
        "running_scheduler_count",
        "degraded_brain_count",
    }
    valid = set(health) == health_fields
    valid = valid and type(health.get("pid")) is int and int(health["pid"]) > 0
    valid = valid and all(
        isinstance(health.get(field), str) and bool(health[field])
        for field in (
            "instance_nonce",
            "launch_nonce",
            "package_version",
            "process_marker",
        )
    )
    valid = valid and all(
        type(health.get(field)) is bool for field in ("shutting_down", "runtime_ready")
    )
    count_fields = (
        "brain_count",
        "engine_count",
        "scheduler_count",
        "running_scheduler_count",
        "degraded_brain_count",
    )
    valid = valid and all(
        type(health.get(field)) is int and int(health[field]) >= 0
        for field in count_fields
    )
    valid = valid and type(health.get("protocol_version")) is int
    health_ready = (
        health.get("brain_count") == health.get("engine_count")
        and health.get("brain_count") == health.get("scheduler_count")
        and health.get("brain_count") == health.get("running_scheduler_count")
    )
    valid = valid and health.get("runtime_ready") is health_ready
    try:
        encoded = json.dumps(
            runtime,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        typed_runtime = DaemonRuntimeStatusV1.model_validate_json(
            encoded,
            strict=True,
        )
    except (TypeError, ValueError, ValidationError):
        valid = False
    else:
        valid = valid and typed_runtime.schema_versions.protocol == health.get(
            "protocol_version"
        )
    if not valid:
        raise _CliFailure(
            "daemon_status_invalid",
            "daemon returned an invalid status payload",
            exit_code=5,
        )


def _live_status(
    home: Path,
    *,
    timeout_seconds: float,
) -> tuple[DaemonDiscoveryV2, dict[str, object], dict[str, object]]:
    _load_discovery(home)
    try:
        client = DaemonClient.connect(home, timeout_seconds=timeout_seconds)
    except DaemonRpcError as error:
        raise _CliFailure(
            error.code,
            error.message,
            data=error.data if isinstance(error.data, dict) else None,
            exit_code=5,
        ) from error
    except DaemonClientError as error:
        raise _CliFailure(
            "daemon_unreachable",
            "daemon discovery exists but its live identity is unreachable",
            exit_code=3,
        ) from error
    try:
        health = client.health()
        runtime = client.call("daemon.status", {})
        _validate_status_payloads(health, runtime)
        return client.discovery, health, runtime
    except DaemonRpcError as error:
        raise _CliFailure(
            error.code,
            error.message,
            data=error.data if isinstance(error.data, dict) else None,
            exit_code=5,
        ) from error
    except DaemonClientError as error:
        raise _CliFailure(
            "daemon_unreachable",
            "daemon stopped responding during status inspection",
            exit_code=3,
        ) from error
    finally:
        with suppress(BaseException):
            client.close()


def _typed_read_rpc(
    home: Path,
    *,
    timeout_seconds: float,
    method: str,
    params: dict[str, object],
    response_model: type[IdentitySnapshotV1] | type[TracePageV1],
) -> dict[str, object]:
    if not _home_exists(home):
        raise _CliFailure(
            "daemon_not_running",
            "Alice-brain-Hermes daemon is not running",
            exit_code=3,
        )
    _validate_existing_home(home)
    if not _discovery_exists(home):
        raise _CliFailure(
            "daemon_not_running",
            "Alice-brain-Hermes daemon is not running",
            exit_code=3,
        )
    _load_discovery(home)
    try:
        client = DaemonClient.connect(home, timeout_seconds=timeout_seconds)
    except DaemonRpcError as error:
        raise _CliFailure(
            error.code,
            error.message,
            data=error.data if isinstance(error.data, dict) else None,
            exit_code=5,
        ) from error
    except DaemonClientError as error:
        raise _CliFailure(
            "daemon_unreachable",
            "daemon discovery exists but its live identity is unreachable",
            exit_code=3,
        ) from error
    try:
        raw = client.call(method, params)
        encoded = json.dumps(
            raw,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        result = response_model.model_validate_json(encoded, strict=True)
        return result.model_dump(mode="json")
    except DaemonRpcError as error:
        raise _CliFailure(
            error.code,
            error.message,
            data=error.data if isinstance(error.data, dict) else None,
            exit_code=5,
        ) from error
    except DaemonClientError as error:
        raise _CliFailure(
            "daemon_unreachable",
            "daemon stopped responding during diagnostic inspection",
            exit_code=3,
        ) from error
    except (ValidationError, TypeError, ValueError) as error:
        raise _CliFailure(
            "daemon_response_invalid",
            "daemon returned an invalid typed diagnostic payload",
            exit_code=5,
        ) from error
    finally:
        with suppress(BaseException):
            client.close()


def _stopped_status(home: Path) -> dict[str, object]:
    return {
        "daemon": None,
        "running": False,
        "runtime_home": os.fspath(home),
        "status": "stopped",
    }


def _status(home: Path, *, timeout_seconds: float) -> dict[str, object]:
    if not _home_exists(home):
        return _stopped_status(home)
    _validate_existing_home(home)
    if not _discovery_exists(home):
        return _stopped_status(home)
    discovery, health, runtime = _live_status(
        home,
        timeout_seconds=timeout_seconds,
    )
    if (
        health["pid"] != discovery.pid
        or health["instance_nonce"] != discovery.instance_nonce
        or health["launch_nonce"] != discovery.launch_nonce
        or health["process_marker"] != discovery.process_marker
        or health["protocol_version"] != discovery.protocol_version
        or health["package_version"] != discovery.package_version
    ):
        raise _CliFailure(
            "daemon_status_invalid",
            "daemon status identity does not match authenticated discovery",
            exit_code=5,
        )
    daemon = {
        "pid": discovery.pid,
        "process_marker": discovery.process_marker,
        "instance_nonce": discovery.instance_nonce,
        "launch_nonce": discovery.launch_nonce,
        "endpoint": discovery.endpoint.model_dump(mode="json"),
        "protocol_version": discovery.protocol_version,
        "package_version": discovery.package_version,
        "shutting_down": health["shutting_down"],
        "runtime_ready": (
            health["runtime_ready"] is True and runtime["runtime_ready"] is True
        ),
        "brain_count": len(runtime["brain_ids"]),
        "engine_count": runtime["engine_count"],
        "scheduler_count": runtime["scheduler_count"],
        "running_scheduler_count": runtime["scheduler_health"][
            "running_scheduler_count"
        ],
        "degraded_brain_count": runtime["scheduler_health"]["degraded_brain_count"],
        "brain_ids": runtime["brain_ids"],
        "runtime_mode": runtime["runtime_mode"],
        "cognition_mode": runtime["cognition_mode"],
        "continuous_runtime": runtime["continuous_runtime"],
        "scheduler_health": runtime["scheduler_health"],
        "bridge_connection": runtime["bridge_connection"],
        "trace_complete": runtime["trace_complete"],
        "semantic_complete": runtime["semantic_complete"],
        "dropped_events": runtime["dropped_events"],
        "semantic_evidence": runtime["semantic_evidence"],
        "unobserved_hermes_fields": runtime["unobserved_hermes_fields"],
        "schema_versions": runtime["schema_versions"],
    }
    return {
        "daemon": daemon,
        "running": True,
        "runtime_home": os.fspath(home),
        "status": "running",
    }


def _daemon_adapter(home: Path, launch_nonce: str) -> DmonAdapter:
    return DmonAdapter.create(
        home,
        [
            sys.executable,
            "-m",
            "alice_brain_hermes.runtime.daemon",
            "--runtime-home",
            os.fspath(home),
            "--launch-nonce",
            launch_nonce,
        ],
        launch_nonce=launch_nonce,
    )


def _failure_diagnostics(adapter: DmonAdapter) -> dict[str, object] | None:
    data: dict[str, object] = {}
    dmon_output = adapter.redacted_dmon_output()
    log_tail = adapter.redacted_log_tail()
    if dmon_output:
        data["dmon_output"] = dmon_output
    if log_tail:
        data["log_tail"] = log_tail
    return data or None


def _start_failure_diagnostics(
    adapter: DmonAdapter,
    error: DmonIdentityError,
) -> dict[str, object] | None:
    data = _failure_diagnostics(adapter) or {}
    if error.meta_hint is not None:
        data["process_hint"] = {
            "pid": error.meta_hint.pid,
            "create_time": error.meta_hint.create_time,
        }
    return data or None


def _cleanup_failed_launch(
    adapter: DmonAdapter,
    hint: DmonProcessHint,
    *,
    timeout_seconds: float,
) -> None:
    try:
        adapter.terminate_exact(hint, timeout_seconds=timeout_seconds)
    except DmonIdentityError as error:
        raise _CliFailure(
            "startup_cleanup_unproven",
            "daemon startup cleanup could not prove exact child termination",
            data=_failure_diagnostics(adapter),
            exit_code=5,
        ) from error
    if not adapter.remove_meta_hint(hint):
        raise _CliFailure(
            "startup_cleanup_unproven",
            "daemon startup artifacts could not be cleaned exactly",
            data=_failure_diagnostics(adapter),
            exit_code=5,
        )


def _authenticated_start_identity(
    discovery: DaemonDiscoveryV2,
    health: dict[str, object],
    runtime: dict[str, object],
    *,
    adapter: DmonAdapter,
    hint: DmonProcessHint,
) -> bool:
    return (
        _discovery_matches_supervisor(discovery, hint)
        and discovery.launch_nonce == adapter.launch_nonce
        and discovery.protocol_version == PROTOCOL_VERSION
        and discovery.package_version == __version__
        and health.get("pid") == discovery.pid
        and health.get("process_marker") == discovery.process_marker
        and health.get("instance_nonce") == discovery.instance_nonce
        and health.get("launch_nonce") == discovery.launch_nonce
        and health.get("protocol_version") == discovery.protocol_version
        and health.get("package_version") == discovery.package_version
        and health.get("runtime_ready") is True
        and runtime.get("runtime_ready") is True
    )


def _exact_process_for_hint(hint: DmonProcessHint) -> psutil.Process | None:
    try:
        process = psutil.Process(hint.pid)
        created = process.create_time()
    except (OSError, psutil.Error):
        return None
    if (
        isinstance(created, bool)
        or not isinstance(created, (int, float))
        or not math.isfinite(float(created))
        or created <= 0
    ):
        return None
    marker = f"psutil-create-time-us:{round(float(created) * 1_000_000)}"
    return process if marker == hint.process_marker else None


def _discovery_matches_supervisor(
    discovery: DaemonDiscoveryV2,
    hint: DmonProcessHint,
) -> bool:
    parent = _exact_process_for_hint(hint)
    if parent is None:
        return False
    if discovery.pid == hint.pid:
        return discovery.process_marker == hint.process_marker
    child_hint = DmonProcessHint(
        pid=discovery.pid,
        process_marker=discovery.process_marker,
    )
    child = _exact_process_for_hint(child_hint)
    if child is None:
        return False
    try:
        parent_pid = child.ppid()
    except (OSError, psutil.Error):
        return False
    return (
        parent_pid == parent.pid
        and _exact_process_for_hint(hint) is not None
        and _exact_process_for_hint(child_hint) is not None
    )


def _running_start_result(
    discovery: DaemonDiscoveryV2,
    *,
    started: bool,
) -> dict[str, object]:
    return {
        "status": "running",
        "started": started,
        "already_running": not started,
        "readiness_verified": True,
        "pid": discovery.pid,
        "instance_nonce": discovery.instance_nonce,
    }


def _start_coordinated(home: Path, *, timeout_seconds: float) -> dict[str, object]:
    if _home_exists(home):
        _validate_existing_home(home)
        if _discovery_exists(home):
            try:
                discovery, _health, _runtime = _live_status(
                    home,
                    timeout_seconds=timeout_seconds,
                )
            except _CliFailure as error:
                if error.code not in {"daemon_unreachable"}:
                    raise
            else:
                return _running_start_result(discovery, started=False)

    launch_nonce = secrets.token_hex(32)
    adapter = _daemon_adapter(home, launch_nonce)
    try:
        hint = adapter.start()
    except DmonIdentityError as error:
        if error.cleanup_unproven:
            raise _CliFailure(
                "startup_cleanup_unproven",
                "daemon launch state remains quarantined pending exact cleanup",
                data=_start_failure_diagnostics(adapter, error),
                exit_code=5,
            ) from error
        raise _CliFailure(
            "startup_failed",
            "python-dmon could not establish the daemon child identity",
            data=_start_failure_diagnostics(adapter, error),
            exit_code=5,
        ) from error

    try:
        return _await_started_daemon(
            home,
            timeout_seconds=timeout_seconds,
            adapter=adapter,
            hint=hint,
        )
    finally:
        adapter.release_parent_guard()


def _await_started_daemon(
    home: Path,
    *,
    timeout_seconds: float,
    adapter: DmonAdapter,
    hint: DmonProcessHint,
) -> dict[str, object]:

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if _discovery_exists(home):
            try:
                discovery, health, runtime = _live_status(
                    home,
                    timeout_seconds=max(0.01, min(remaining, 0.25)),
                )
            except _CliFailure:
                pass
            else:
                if _authenticated_start_identity(
                    discovery,
                    health,
                    runtime,
                    adapter=adapter,
                    hint=hint,
                ):
                    return _running_start_result(discovery, started=True)
                if discovery.launch_nonce == adapter.launch_nonce:
                    _cleanup_failed_launch(
                        adapter,
                        hint,
                        timeout_seconds=max(0.01, remaining),
                    )
                    raise _CliFailure(
                        "startup_failed",
                        "daemon authenticated readiness identity did not match",
                        data=_failure_diagnostics(adapter),
                        exit_code=5,
                    )

        process_state = _process_identity_state(hint.pid, hint.process_marker)
        if process_state in {"gone", "reused"}:
            adapter.remove_meta_hint(hint)
            if _discovery_exists(home):
                try:
                    discovery, _health, _runtime = _live_status(
                        home,
                        timeout_seconds=max(0.01, min(remaining, 0.25)),
                    )
                except _CliFailure:
                    pass
                else:
                    return _running_start_result(discovery, started=False)
            raise _CliFailure(
                "startup_failed",
                "daemon exited before authenticated readiness",
                data=_failure_diagnostics(adapter),
                exit_code=5,
            )
        if remaining <= 0:
            _cleanup_failed_launch(
                adapter,
                hint,
                timeout_seconds=max(0.01, timeout_seconds),
            )
            raise _CliFailure(
                "startup_timeout",
                "daemon did not prove authenticated readiness before the timeout",
                data=_failure_diagnostics(adapter),
                exit_code=5,
            )
        time.sleep(min(0.02, remaining))


def _start(home: Path, *, timeout_seconds: float) -> dict[str, object]:
    started = time.monotonic()
    coordinator = DmonStartCoordinator.create(home)
    try:
        coordinator.acquire(timeout_seconds=timeout_seconds)
    except DmonCoordinationTimeout as error:
        raise _CliFailure(
            "startup_coordination_timeout",
            "another parent still owns daemon launch coordination",
            exit_code=5,
        ) from error
    try:
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise _CliFailure(
                "startup_coordination_timeout",
                "daemon launch coordination exhausted the startup timeout",
                exit_code=5,
            )
        return _start_coordinated(home, timeout_seconds=remaining)
    finally:
        coordinator.release()


def _process_identity_state(
    pid: int,
    process_marker: str,
) -> Literal["exact", "reused", "gone", "ambiguous"]:
    try:
        current = read_process_marker(pid)
    except (OSError, ValueError):
        try:
            exists = psutil.pid_exists(pid)
        except (OSError, psutil.Error):
            return "ambiguous"
        return "ambiguous" if exists else "gone"
    return "exact" if current == process_marker else "reused"


def _stop_coordinated(home: Path, *, timeout_seconds: float) -> dict[str, object]:
    if not _home_exists(home):
        return {
            "already_stopped": True,
            "instance_nonce": None,
            "status": "stopped",
            "stop_requested": False,
        }
    _validate_existing_home(home)
    if not _discovery_exists(home):
        return {
            "already_stopped": True,
            "instance_nonce": None,
            "status": "stopped",
            "stop_requested": False,
        }
    _load_discovery(home)
    try:
        client = DaemonClient.connect(home, timeout_seconds=timeout_seconds)
    except DaemonClientError as error:
        raise _CliFailure(
            "daemon_unreachable",
            "daemon discovery exists but authenticated shutdown is unavailable",
            exit_code=3,
        ) from error
    discovery = client.discovery
    adapter = _daemon_adapter(home, discovery.launch_nonce)
    try:
        supervisor_hint = adapter.current_process_hint()
    except DmonIdentityError as error:
        with suppress(BaseException):
            client.close()
        raise _CliFailure(
            "shutdown_cleanup_unproven",
            "daemon supervisor identity is not verifiable",
            exit_code=5,
        ) from error
    if not _discovery_matches_supervisor(discovery, supervisor_hint):
        with suppress(BaseException):
            client.close()
        raise _CliFailure(
            "shutdown_cleanup_unproven",
            "daemon supervisor identity does not match discovery",
            exit_code=5,
        )
    try:
        client.shutdown()
    except DaemonRpcError as error:
        raise _CliFailure(
            error.code,
            error.message,
            data=error.data if isinstance(error.data, dict) else None,
            exit_code=5,
        ) from error
    except DaemonClientError as error:
        raise _CliFailure(
            "shutdown_failed",
            "daemon shutdown request failed",
            exit_code=5,
        ) from error
    finally:
        with suppress(BaseException):
            client.close()

    deadline = time.monotonic() + timeout_seconds
    credential_path = home / discovery.credential_ref
    last_process_state = "exact"
    last_supervisor_state = "exact"
    while time.monotonic() < deadline:
        last_process_state = _process_identity_state(
            discovery.pid,
            discovery.process_marker,
        )
        if (
            supervisor_hint.pid == discovery.pid
            and supervisor_hint.process_marker == discovery.process_marker
        ):
            last_supervisor_state = last_process_state
        else:
            last_supervisor_state = _process_identity_state(
                supervisor_hint.pid,
                supervisor_hint.process_marker,
            )
        discovery_present = _discovery_exists(home)
        credential_present = credential_path.exists() or credential_path.is_symlink()
        if (
            last_process_state in {"gone", "reused"}
            and last_supervisor_state in {"gone", "reused"}
            and not discovery_present
            and not credential_present
        ):
            cleaned = adapter.remove_meta_hint(supervisor_hint)
            if not cleaned:
                raise _CliFailure(
                    "shutdown_cleanup_unproven",
                    "daemon stopped but supervisor artifacts remain quarantined",
                    exit_code=5,
                )
            return {
                "already_stopped": False,
                "instance_nonce": discovery.instance_nonce,
                "pid": discovery.pid,
                "status": "stopped",
                "stop_requested": True,
            }
        time.sleep(0.02)
    if "ambiguous" in {last_process_state, last_supervisor_state}:
        raise _CliFailure(
            "shutdown_unproven",
            "daemon process tree identity became unverifiable during shutdown",
            data={"instance_nonce": discovery.instance_nonce, "pid": discovery.pid},
            exit_code=5,
        )
    raise _CliFailure(
        "shutdown_timeout",
        "daemon did not prove shutdown before the timeout",
        data={"instance_nonce": discovery.instance_nonce, "pid": discovery.pid},
        exit_code=5,
    )


def _stop(home: Path, *, timeout_seconds: float) -> dict[str, object]:
    started = time.monotonic()
    coordinator = DmonStartCoordinator.create(home)
    try:
        coordinator.acquire(timeout_seconds=timeout_seconds)
    except DmonCoordinationTimeout as error:
        raise _CliFailure(
            "shutdown_coordination_timeout",
            "another parent owns daemon lifecycle coordination",
            exit_code=5,
        ) from error
    try:
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise _CliFailure(
                "shutdown_coordination_timeout",
                "daemon lifecycle coordination exhausted the shutdown timeout",
                exit_code=5,
            )
        return _stop_coordinated(home, timeout_seconds=remaining)
    finally:
        coordinator.release()


def _doctor(home: Path, *, timeout_seconds: float) -> tuple[dict[str, object], int]:
    checks: list[dict[str, object]] = [
        {
            "id": "package",
            "status": "pass",
            "summary": "Alice-brain-Hermes package is importable",
            "data": {"package_version": __version__},
        }
    ]
    if not _home_exists(home):
        checks.extend(
            [
                {
                    "id": "runtime_home",
                    "status": "fail",
                    "summary": "runtime home does not exist",
                    "data": {"path": os.fspath(home)},
                },
                {
                    "id": "daemon",
                    "status": "fail",
                    "summary": "daemon is not running",
                    "data": {},
                },
            ]
        )
    else:
        try:
            _validate_existing_home(home)
        except _CliFailure as error:
            checks.append(
                {
                    "id": "runtime_home",
                    "status": "fail",
                    "summary": error.message,
                    "data": {"path": os.fspath(home)},
                }
            )
        else:
            checks.append(
                {
                    "id": "runtime_home",
                    "status": "pass",
                    "summary": "runtime home is a verified private directory",
                    "data": {"path": os.fspath(home)},
                }
            )
            if not _discovery_exists(home):
                checks.append(
                    {
                        "id": "daemon",
                        "status": "fail",
                        "summary": "daemon is not running",
                        "data": {},
                    }
                )
            else:
                try:
                    status = _status(home, timeout_seconds=timeout_seconds)
                except _CliFailure as error:
                    checks.append(
                        {
                            "id": "daemon",
                            "status": "fail",
                            "summary": error.message,
                            "data": {"code": error.code},
                        }
                    )
                else:
                    daemon = status["daemon"]
                    assert isinstance(daemon, dict)
                    healthy = (
                        daemon.get("runtime_ready") is True
                        and daemon.get("shutting_down") is False
                        and daemon.get("degraded_brain_count") == 0
                        and isinstance(daemon.get("scheduler_health"), dict)
                        and daemon["scheduler_health"].get("status") == "healthy"
                    )
                    checks.append(
                        {
                            "id": "daemon",
                            "status": "pass" if healthy else "fail",
                            "summary": (
                                "daemon identity and continuous runtime are healthy"
                                if healthy
                                else "daemon continuous runtime is degraded"
                            ),
                            "data": {
                                "instance_nonce": daemon.get("instance_nonce"),
                                "runtime_ready": daemon.get("runtime_ready"),
                            },
                        }
                    )
                    required_observability = {
                        "bridge_connection",
                        "cognition_mode",
                        "dropped_events",
                        "scheduler_health",
                        "semantic_complete",
                        "trace_complete",
                        "unobserved_hermes_fields",
                        "schema_versions",
                    }
                    missing = sorted(required_observability - set(daemon))
                    failed_evidence: list[str] = []
                    if not missing:
                        if daemon.get("trace_complete") is not True:
                            failed_evidence.append("trace_complete")
                        if daemon.get("semantic_complete") is not True:
                            failed_evidence.append("semantic_complete")
                        dropped = daemon.get("dropped_events")
                        if type(dropped) is not int or dropped != 0:
                            failed_evidence.append("dropped_events")
                        bridge = daemon.get("bridge_connection")
                        if (
                            not isinstance(bridge, dict)
                            or bridge.get("state") == "degraded"
                        ):
                            failed_evidence.append("bridge_connection")
                    observable = not missing and not failed_evidence
                    checks.append(
                        {
                            "id": "integration_observability",
                            "status": "pass" if observable else "fail",
                            "summary": (
                                "daemon status lacks required bridge/trace evidence"
                                if missing
                                else (
                                    "persisted bridge or semantic evidence is degraded"
                                    if failed_evidence
                                    else "bridge and trace evidence are available"
                                )
                            ),
                            "data": {
                                "missing_fields": missing,
                                "failed_evidence": failed_evidence,
                            },
                        }
                    )
    healthy = not any(item["status"] == "fail" for item in checks)
    return {"healthy": healthy, "checks": checks}, 0 if healthy else 4


def _run_foreground(home: Path) -> int:
    from alice_brain_hermes.runtime.daemon import run_private_daemon

    run_private_daemon(home)
    return 0


def _dispatch(arguments: argparse.Namespace) -> int:
    home = _runtime_home(arguments.home)
    timeout_seconds = arguments.timeout
    command = arguments.alice_brain_command
    if command == "status" or (
        command == "daemon" and arguments.alice_brain_daemon_command == "status"
    ):
        _write_success(
            "daemon.status",
            _status(home, timeout_seconds=timeout_seconds),
        )
        return 0
    if command == "doctor":
        data, exit_code = _doctor(home, timeout_seconds=timeout_seconds)
        _write_success("doctor", data)
        return exit_code
    if command == "identity" and arguments.alice_brain_identity_command == "get":
        params = {} if arguments.brain_id is None else {"brain_id": arguments.brain_id}
        _write_success(
            "identity.get",
            _typed_read_rpc(
                home,
                timeout_seconds=timeout_seconds,
                method="identity.get",
                params=params,
                response_model=IdentitySnapshotV1,
            ),
        )
        return 0
    if command == "trace" and arguments.alice_brain_trace_command == "list":
        params: dict[str, object] = {
            "after_sequence": arguments.after_sequence,
            "limit": arguments.limit,
        }
        if arguments.brain_id is not None:
            params["brain_id"] = arguments.brain_id
        _write_success(
            "trace.list",
            _typed_read_rpc(
                home,
                timeout_seconds=timeout_seconds,
                method="trace.list",
                params=params,
                response_model=TracePageV1,
            ),
        )
        return 0
    if command == "start" or (
        command == "daemon" and arguments.alice_brain_daemon_command == "start"
    ):
        _write_success(
            "daemon.start",
            _start(home, timeout_seconds=timeout_seconds),
        )
        return 0
    if command == "stop" or (
        command == "daemon" and arguments.alice_brain_daemon_command == "stop"
    ):
        _write_success(
            "daemon.stop",
            _stop(home, timeout_seconds=timeout_seconds),
        )
        return 0
    if command == "daemon" and arguments.alice_brain_daemon_command == "run":
        return _run_foreground(home)
    raise AssertionError("argparse accepted an unknown command")


def run_control_namespace(arguments: argparse.Namespace) -> int:
    """Execute one parsed standalone or Hermes plugin namespace."""

    try:
        return _dispatch(arguments)
    except _CliFailure as error:
        failure = error
    except KeyboardInterrupt:
        failure = _CliFailure(
            "interrupted",
            "command was interrupted",
            exit_code=130,
        )
    except Exception:
        failure = _CliFailure(
            "internal_error",
            "command failed unexpectedly",
            exit_code=5,
        )
    _write_error(failure)
    return failure.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except _UsageError as error:
        failure = _CliFailure(
            "usage_error",
            "command arguments are invalid",
            data={"detail": str(error)},
            exit_code=2,
        )
        _write_error(failure)
        return failure.exit_code
    return run_control_namespace(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
