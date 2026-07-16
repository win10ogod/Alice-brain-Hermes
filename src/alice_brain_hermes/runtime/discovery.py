"""Portable nonce-bound credentials and atomic daemon discovery records."""

from __future__ import annotations

import json
import os
import secrets
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import psutil

from alice_brain_hermes.errors import RuntimeOwnedError
from alice_brain_hermes.protocol.models import DaemonDiscoveryV2
from alice_brain_hermes.runtime.lease import RuntimeLease

_MAX_DISCOVERY_BYTES = 16_384
_TOKEN_HEX_BYTES = 64
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 2_000
_NONCE_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
_RESERVED_RUNTIME_FILES = frozenset(
    {
        "daemon.json",
        "daemon.lock",
        "runtime.db",
        "runtime.db-journal",
        "runtime.db-shm",
        "runtime.db-wal",
    }
)


@dataclass(frozen=True, slots=True)
class CredentialFile:
    path: Path
    token: str


def _private_home(runtime_home: str | Path) -> Path:
    home = Path(runtime_home).expanduser().absolute()
    if home.is_symlink() or not home.is_dir():
        raise PermissionError("runtime home must be a real directory")
    resolved = home.resolve(strict=True)
    if os.path.normcase(os.fspath(resolved)) != os.path.normcase(
        os.fspath(home.resolve(strict=False))
    ):
        raise PermissionError("runtime home path is not stable")
    return resolved


def _write_exclusive(authority: RuntimeLease, name: str, payload: bytes) -> None:
    path = authority.home_path(name)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
        authority.chmod_home_file(name, 0o600)
        authority.assert_authority()
    except BaseException:
        with suppress(FileNotFoundError, PermissionError):
            authority.unlink_home_file(name, missing_ok=True)
        raise


def create_credential(authority: RuntimeLease) -> CredentialFile:
    if not isinstance(authority, RuntimeLease):
        raise TypeError("credential creation requires live RuntimeLease authority")
    home = authority.assert_authority()
    instance_nonce = authority.instance_nonce
    if (
        not isinstance(instance_nonce, str)
        or not 1 <= len(instance_nonce) <= 128
        or any(character not in _NONCE_ALPHABET for character in instance_nonce)
    ):
        raise ValueError("instance nonce is invalid")
    token = secrets.token_hex(32)
    name = f"credential-{instance_nonce}.key"
    _write_exclusive(authority, name, token.encode("ascii"))
    authority.fsync_home()
    return CredentialFile(path=home / name, token=token)


def publish_discovery(authority: RuntimeLease, record: DaemonDiscoveryV2) -> Path:
    if not isinstance(authority, RuntimeLease):
        raise TypeError("discovery publication requires live RuntimeLease authority")
    home = authority.assert_authority()
    record = DaemonDiscoveryV2.model_validate(record.model_dump(mode="python"))
    if (
        record.instance_nonce != authority.instance_nonce
        or record.launch_nonce != authority.launch_nonce
        or record.process_marker != authority.process_marker
        or record.pid != os.getpid()
    ):
        raise PermissionError("discovery identity does not match lease authority")
    payload = record.canonical_json().encode("utf-8")
    if len(payload) > _MAX_DISCOVERY_BYTES:
        raise ValueError("discovery record exceeds the byte limit")
    temporary = f".daemon-{record.instance_nonce}-{secrets.token_hex(8)}.tmp"
    _write_exclusive(authority, temporary, payload)
    destination = home / "daemon.json"
    try:
        authority.replace_home_file(temporary, destination.name)
        authority.chmod_home_file(destination.name, 0o600)
        authority.fsync_home()
    except BaseException:
        with suppress(FileNotFoundError, PermissionError):
            authority.unlink_home_file(temporary, missing_ok=True)
        raise
    return destination


def _read_private_regular(path: Path, *, label: str, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"{label} must be a private regular file")
    payload = bytearray()
    try:
        with path.open("rb") as stream:
            while len(payload) <= maximum:
                chunk = stream.read(min(65_536, maximum + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
    except OSError as error:
        raise PermissionError(f"{label} changed while it was read") from error
    if path.is_symlink() or not path.is_file() or path.stat().st_size != len(payload):
        raise PermissionError(f"{label} changed while it was read")
    if len(payload) > maximum:
        raise PermissionError(f"{label} exceeds its byte limit")
    return bytes(payload)


def _strict_json_object(payload: bytes) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("discovery contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-finite discovery number")
            ),
        )
    except (UnicodeError, ValueError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("discovery JSON is invalid") from error
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ValueError("discovery JSON structure exceeds limits")
        if isinstance(item, str):
            item.encode("utf-8", errors="strict")
        elif isinstance(item, dict):
            pending.extend((key, depth + 1) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    if not isinstance(value, dict):
        raise ValueError("discovery must be a JSON object")
    return value


def _decode_discovery_record(payload: bytes) -> DaemonDiscoveryV2:
    raw = _strict_json_object(payload)
    if raw.get("schema_version") != 2:
        raise ValueError("only discovery schema v2 is accepted")
    record = DaemonDiscoveryV2.model_validate(raw)
    if payload != record.canonical_json().encode("utf-8"):
        raise ValueError("discovery JSON is not canonical")
    return record


def _load_discovery_record(home: Path) -> DaemonDiscoveryV2:
    payload = _read_private_regular(
        home / "daemon.json", label="discovery", maximum=_MAX_DISCOVERY_BYTES
    )
    return _decode_discovery_record(payload)


def _load_discovery_record_at(authority: RuntimeLease) -> DaemonDiscoveryV2:
    path = authority.home_path("daemon.json")
    payload = _read_private_regular(
        path, label="discovery", maximum=_MAX_DISCOVERY_BYTES
    )
    authority.assert_authority()
    return _decode_discovery_record(payload)


def _validate_credential(path: Path) -> str:
    payload = _read_private_regular(
        path,
        label="credential",
        maximum=_TOKEN_HEX_BYTES,
    )
    try:
        token = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise PermissionError("credential is not ASCII") from error
    if len(token) != _TOKEN_HEX_BYTES or any(
        character not in "0123456789abcdef" for character in token
    ):
        raise PermissionError("credential has an invalid encoding")
    return token


def load_discovery_and_credential(
    runtime_home: str | Path,
) -> tuple[DaemonDiscoveryV2, str]:
    home = _private_home(runtime_home)
    record = _load_discovery_record(home)
    credential_path = home / record.credential_ref
    if credential_path.parent != home:
        raise PermissionError("credential reference escapes runtime home")
    return record, _validate_credential(credential_path)


def cleanup_credential(authority: RuntimeLease, credential: CredentialFile) -> None:
    if not isinstance(authority, RuntimeLease):
        raise TypeError("credential cleanup requires live RuntimeLease authority")
    if not isinstance(credential, CredentialFile):
        raise TypeError("credential cleanup requires a typed CredentialFile")
    expected_name = f"credential-{authority.instance_nonce}.key"
    if credential.path.name != expected_name:
        raise PermissionError("credential does not match lease nonce")
    path = authority.home_path(expected_name)
    try:
        current_token = _validate_credential(path)
    except FileNotFoundError:
        return
    except PermissionError:
        if not path.exists():
            return
        raise
    if current_token != credential.token:
        raise PermissionError("credential content does not match its owner")
    authority.unlink_home_file(expected_name)
    authority.fsync_home()


def _legacy_process_is_dead(raw: dict[str, object]) -> bool:
    pid = raw.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        return not process.is_running()
    except psutil.NoSuchProcess:
        return True
    except (OSError, psutil.Error):
        return False


def _legacy_credential_name(raw: dict[str, object]) -> str:
    nonce = raw.get("instance_nonce")
    credential_ref = raw.get("credential_ref")
    valid_nonce = (
        isinstance(nonce, str)
        and 1 <= len(nonce) <= 128
        and all(character in _NONCE_ALPHABET for character in nonce)
    )
    if not valid_nonce or not isinstance(credential_ref, str):
        raise RuntimeOwnedError("legacy discovery identity is invalid")
    expected = f"credential-{nonce}.key"
    if (
        credential_ref.casefold() in _RESERVED_RUNTIME_FILES
        or Path(credential_ref).name != credential_ref
        or credential_ref != expected
    ):
        raise RuntimeOwnedError("legacy discovery credential reference is invalid")
    return credential_ref


def cleanup_stale_discovery(authority: RuntimeLease) -> None:
    """Clean dead discovery only while both portable locks are held."""
    if not isinstance(authority, RuntimeLease):
        raise TypeError("discovery cleanup requires live RuntimeLease authority")
    path = authority.home_path("daemon.json")
    try:
        payload = _read_private_regular(
            path, label="discovery", maximum=_MAX_DISCOVERY_BYTES
        )
    except PermissionError:
        if not path.exists():
            return
        raise
    raw = _strict_json_object(payload)
    if raw.get("schema_version") == 1:
        credential_ref = _legacy_credential_name(raw)
        if not _legacy_process_is_dead(raw):
            raise RuntimeOwnedError(
                "live or ambiguous legacy discovery refuses startup"
            )
        authority.unlink_home_file(credential_ref, missing_ok=True)
        authority.unlink_home_file("daemon.json", missing_ok=True)
        authority.fsync_home()
        return
    record = _decode_discovery_record(payload)
    if record.instance_nonce == authority.instance_nonce:
        return
    _cleanup_discovery_for_nonce(authority, record.instance_nonce)


def cleanup_discovery(authority: RuntimeLease) -> None:
    if not isinstance(authority, RuntimeLease):
        raise TypeError("discovery cleanup requires live RuntimeLease authority")
    _cleanup_discovery_for_nonce(authority, authority.instance_nonce)


def _cleanup_discovery_for_nonce(authority: RuntimeLease, instance_nonce: str) -> None:
    try:
        record = _load_discovery_record_at(authority)
    except FileNotFoundError:
        return
    except PermissionError:
        if not authority.home_path("daemon.json").exists():
            return
        raise
    if record.instance_nonce != instance_nonce:
        return
    credential = authority.home_path(record.credential_ref)
    if credential.exists():
        _validate_credential(credential)
        authority.unlink_home_file(record.credential_ref)
    current = _load_discovery_record_at(authority)
    if current.instance_nonce == instance_nonce:
        authority.unlink_home_file("daemon.json", missing_ok=True)
    authority.fsync_home()


__all__ = [
    "CredentialFile",
    "cleanup_credential",
    "cleanup_discovery",
    "cleanup_stale_discovery",
    "create_credential",
    "load_discovery_and_credential",
    "publish_discovery",
]
