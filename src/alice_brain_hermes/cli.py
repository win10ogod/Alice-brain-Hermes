"""Machine-stable operator CLI for the private consciousness daemon."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
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
from alice_brain_hermes.runtime.discovery import (
    _private_home,
    load_discovery_and_credential,
)
from alice_brain_hermes.runtime.process_marker import read_process_marker

_CLI_SCHEMA_VERSION = 1
_DEFAULT_TIMEOUT_SECONDS = 10.0
_MAX_TIMEOUT_SECONDS = 300.0
_MAX_READINESS_BYTES = 4_096
_READINESS_FAILURE_CODES = frozenset(
    {"runtime_owned", "shutdown_unproven", "startup_failed"}
)


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
        raise argparse.ArgumentTypeError(
            "after-sequence must be an integer"
        ) from error
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
) -> argparse.ArgumentParser:
    """Build the shared Python command surface without contacting the runtime."""

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
    commands = parser.add_subparsers(dest="alice_brain_command", required=True)

    daemon = commands.add_parser("daemon", help="run or control the private daemon")
    daemon_commands = daemon.add_subparsers(
        dest="alice_brain_daemon_command", required=True
    )
    daemon_commands.add_parser("run", help="run the daemon in the foreground")
    daemon_commands.add_parser("start", help="start the daemon in the background")
    daemon_commands.add_parser("stop", help="request authenticated daemon shutdown")
    daemon_commands.add_parser("status", help="report live daemon status")

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
        "instance_nonce",
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
    runtime_fields = {
        "brain_ids",
        "engine_count",
        "scheduler_count",
        "continuous_runtime",
    }
    valid = set(health) == health_fields and set(runtime) == runtime_fields
    valid = valid and all(
        isinstance(health.get(field), str) and bool(health[field])
        for field in ("instance_nonce", "process_marker")
    )
    valid = valid and all(
        type(health.get(field)) is bool
        for field in ("shutting_down", "runtime_ready")
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
    brain_ids = runtime.get("brain_ids")
    valid = valid and isinstance(brain_ids, list)
    valid = valid and all(
        isinstance(item, str) and bool(item) for item in brain_ids or []
    )
    valid = valid and len(set(brain_ids or [])) == len(brain_ids or [])
    valid = valid and all(
        type(runtime.get(field)) is int and int(runtime[field]) >= 0
        for field in ("engine_count", "scheduler_count")
    )
    valid = valid and runtime.get("continuous_runtime") is True
    valid = valid and type(health.get("protocol_version")) is int
    valid = valid and health.get("brain_count") == len(brain_ids or [])
    valid = valid and health.get("engine_count") == runtime.get("engine_count")
    valid = valid and health.get("scheduler_count") == runtime.get(
        "scheduler_count"
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
) -> tuple[object, dict[str, object], dict[str, object]]:
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
        health["instance_nonce"] != discovery.instance_nonce
        or health["process_marker"] != discovery.process_marker
        or health["protocol_version"] != discovery.protocol_version
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
        "endpoint": discovery.endpoint.model_dump(mode="json"),
        "protocol_version": discovery.protocol_version,
        "package_version": discovery.package_version,
        "shutting_down": health["shutting_down"],
        "runtime_ready": health["runtime_ready"],
        "brain_count": health["brain_count"],
        "engine_count": health["engine_count"],
        "scheduler_count": health["scheduler_count"],
        "running_scheduler_count": health["running_scheduler_count"],
        "degraded_brain_count": health["degraded_brain_count"],
        "brain_ids": runtime["brain_ids"],
        "continuous_runtime": runtime["continuous_runtime"],
    }
    return {
        "daemon": daemon,
        "running": True,
        "runtime_home": os.fspath(home),
        "status": "running",
    }


def _read_readiness(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> bytes:
    if process.stdout is None:
        raise _CliFailure("startup_failed", "daemon readiness pipe is unavailable")
    results: queue.Queue[bytes | BaseException] = queue.Queue(maxsize=1)

    def read_once() -> None:
        try:
            first = process.stdout.readline(_MAX_READINESS_BYTES + 1)
            if (
                len(first) <= _MAX_READINESS_BYTES
                and first.endswith(b"\n")
            ):
                # The daemon consumes and closes its one-shot readiness
                # descriptor after the line.  Reading one more byte proves
                # there is no second record or trailing garbage.
                first += process.stdout.read(1)
            results.put(first)
        except BaseException as error:
            results.put(error)

    reader = threading.Thread(
        target=read_once,
        name="alice-brain-hermes-readiness",
        daemon=True,
    )
    reader.start()
    try:
        result = results.get(timeout=timeout_seconds)
    except queue.Empty as error:
        _terminate_spawned(process, timeout_seconds=timeout_seconds)
        raise _CliFailure(
            "startup_timeout",
            "daemon did not report readiness before the timeout",
            exit_code=5,
        ) from error
    finally:
        with suppress(BaseException):
            process.stdout.close()
        reader.join(timeout=min(timeout_seconds, 1.0))
    if reader.is_alive():
        _terminate_spawned(process, timeout_seconds=timeout_seconds)
        raise _CliFailure(
            "startup_cleanup_unproven",
            "daemon readiness reader did not terminate",
            exit_code=5,
        )
    if isinstance(result, BaseException):
        _terminate_spawned(process, timeout_seconds=timeout_seconds)
        raise _CliFailure(
            "startup_failed",
            "daemon readiness could not be read",
            exit_code=5,
        ) from result
    return result


def _decode_readiness(payload: bytes) -> dict[str, object]:
    invalid = (
        not payload
        or len(payload) > _MAX_READINESS_BYTES
        or not payload.endswith(b"\n")
        or payload.count(b"\n") != 1
    )
    if invalid:
        raise _CliFailure(
            "startup_failed",
            "daemon returned an invalid readiness signal",
            exit_code=5,
        )
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate readiness key")
            result[key] = item
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite readiness value")

    try:
        value = json.loads(
            payload[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise _CliFailure(
            "startup_failed",
            "daemon returned an invalid readiness signal",
            exit_code=5,
        ) from error
    if not isinstance(value, dict) or type(value.get("ready")) is not bool:
        raise _CliFailure(
            "startup_failed",
            "daemon returned an invalid readiness signal",
            exit_code=5,
        )
    if value["ready"] is True:
        nonce = value.get("instance_nonce")
        valid = (
            set(value) == {"ready", "instance_nonce"}
            and isinstance(nonce, str)
            and 1 <= len(nonce) <= 128
            and all(
                character.isascii()
                and (character.isalnum() or character in "_-")
                for character in nonce
            )
        )
    else:
        code = value.get("code")
        valid = set(value) == {"ready", "code"} and code in _READINESS_FAILURE_CODES
    if not valid:
        raise _CliFailure(
            "startup_failed",
            "daemon returned an invalid readiness signal",
            exit_code=5,
        )
    return value


def _terminate_spawned(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> None:
    if process.poll() is not None:
        return
    deadline = time.monotonic() + timeout_seconds
    with suppress(OSError):
        process.terminate()
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except (OSError, subprocess.TimeoutExpired):
        with suppress(OSError):
            process.kill()
        with suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
    if process.poll() is None:
        raise _CliFailure(
            "startup_cleanup_unproven",
            "spawned daemon termination could not be proven",
            exit_code=5,
        )


def _spawn_daemon(home: Path) -> subprocess.Popen[bytes]:
    creationflags = 0
    start_new_session = os.name == "posix"
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "alice_brain_hermes.runtime.daemon",
            "--runtime-home",
            os.fspath(home),
            "--readiness-fd",
            "1",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=start_new_session,
        creationflags=creationflags,
    )


def _start(home: Path, *, timeout_seconds: float) -> dict[str, object]:
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
                return {
                    "status": "running",
                    "started": False,
                    "already_running": True,
                    "readiness_verified": True,
                    "pid": discovery.pid,
                    "instance_nonce": discovery.instance_nonce,
                }

    process = _spawn_daemon(home)
    readiness_payload = _read_readiness(process, timeout_seconds=timeout_seconds)
    try:
        readiness = _decode_readiness(readiness_payload)
    except _CliFailure:
        _terminate_spawned(process, timeout_seconds=timeout_seconds)
        raise
    if readiness["ready"] is not True:
        code = readiness.get("code")
        try:
            process.wait(timeout=timeout_seconds)
        except (OSError, subprocess.TimeoutExpired):
            _terminate_spawned(process, timeout_seconds=timeout_seconds)
        if code == "runtime_owned":
            try:
                discovery, _health, _runtime = _live_status(
                    home,
                    timeout_seconds=timeout_seconds,
                )
            except _CliFailure:
                pass
            else:
                return {
                    "status": "running",
                    "started": False,
                    "already_running": True,
                    "readiness_verified": True,
                    "pid": discovery.pid,
                    "instance_nonce": discovery.instance_nonce,
                }
        raise _CliFailure(
            str(code) if isinstance(code, str) and code else "startup_failed",
            "daemon startup failed",
            exit_code=5,
        )

    instance_nonce = readiness.get("instance_nonce")
    if not isinstance(instance_nonce, str) or not instance_nonce:
        _terminate_spawned(process, timeout_seconds=timeout_seconds)
        raise _CliFailure(
            "startup_failed",
            "daemon readiness omitted its instance identity",
            exit_code=5,
        )
    try:
        discovery, health, _runtime = _live_status(
            home,
            timeout_seconds=timeout_seconds,
        )
    except _CliFailure:
        _terminate_spawned(process, timeout_seconds=timeout_seconds)
        raise
    if (
        discovery.pid != process.pid
        or discovery.instance_nonce != instance_nonce
        or health.get("runtime_ready") is not True
    ):
        _terminate_spawned(process, timeout_seconds=timeout_seconds)
        raise _CliFailure(
            "startup_failed",
            "daemon readiness identity could not be verified",
            exit_code=5,
        )
    return {
        "status": "running",
        "started": True,
        "already_running": False,
        "readiness_verified": True,
        "pid": discovery.pid,
        "instance_nonce": discovery.instance_nonce,
    }


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


def _stop(home: Path, *, timeout_seconds: float) -> dict[str, object]:
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
    while time.monotonic() < deadline:
        last_process_state = _process_identity_state(
            discovery.pid,
            discovery.process_marker,
        )
        discovery_present = _discovery_exists(home)
        credential_present = credential_path.exists() or credential_path.is_symlink()
        if (
            last_process_state in {"gone", "reused"}
            and not discovery_present
            and not credential_present
        ):
            return {
                "already_stopped": False,
                "instance_nonce": discovery.instance_nonce,
                "pid": discovery.pid,
                "status": "stopped",
                "stop_requested": True,
            }
        time.sleep(0.02)
    if last_process_state == "ambiguous":
        raise _CliFailure(
            "shutdown_unproven",
            "daemon process identity became unverifiable during shutdown",
            data={"instance_nonce": discovery.instance_nonce, "pid": discovery.pid},
            exit_code=5,
        )
    raise _CliFailure(
        "shutdown_timeout",
        "daemon did not prove shutdown before the timeout",
        data={"instance_nonce": discovery.instance_nonce, "pid": discovery.pid},
        exit_code=5,
    )


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
                        "trace_complete",
                        "unobserved_hermes_fields",
                    }
                    missing = sorted(required_observability - set(daemon))
                    checks.append(
                        {
                            "id": "integration_observability",
                            "status": "fail" if missing else "pass",
                            "summary": (
                                "daemon status lacks required bridge/trace evidence"
                                if missing
                                else "bridge and trace evidence are available"
                            ),
                            "data": {"missing_fields": missing},
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
        command == "daemon"
        and arguments.alice_brain_daemon_command == "status"
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
        params = (
            {}
            if arguments.brain_id is None
            else {"brain_id": arguments.brain_id}
        )
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
    if (
        command == "daemon"
        and arguments.alice_brain_daemon_command == "start"
    ):
        _write_success(
            "daemon.start",
            _start(home, timeout_seconds=timeout_seconds),
        )
        return 0
    if command == "daemon" and arguments.alice_brain_daemon_command == "stop":
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
