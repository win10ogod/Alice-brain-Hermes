"""Private nonce-bound credential and atomic daemon discovery records."""

from __future__ import annotations

import json
import os
import secrets
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from alice_brain_hermes.protocol.models import DaemonDiscoveryV1
from alice_brain_hermes.runtime.lease import RuntimeLease

_MAX_DISCOVERY_BYTES = 16_384
_TOKEN_HEX_BYTES = 64
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 2_000
_NONCE_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"


@dataclass(frozen=True, slots=True)
class CredentialFile:
    path: Path
    token: str


def _private_home(runtime_home: str | Path) -> Path:
    if os.name == "nt":
        raise PermissionError(
            "Windows current-user-only DACL cannot be verified; refusing"
        )
    home = Path(runtime_home).expanduser().absolute()
    metadata = home.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError("runtime home must be a real directory")
    if home.resolve(strict=True) != home:
        raise PermissionError("runtime home cannot contain symlink traversal")
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise PermissionError("runtime home must be user-owned mode 0700")
    return home


def _secure_flags(*, exclusive: bool) -> int:
    flags = os.O_WRONLY | os.O_CREAT
    if exclusive:
        flags |= os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _write_exclusive(authority: RuntimeLease, name: str, payload: bytes) -> None:
    descriptor = authority.open_home_file(name, _secure_flags(exclusive=True), 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise PermissionError("private runtime file must be regular and user-owned")
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("private runtime file write made no progress")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        with suppress(FileNotFoundError, PermissionError):
            authority.unlink_home_file(name, missing_ok=True)
        raise
    finally:
        os.close(descriptor)


def create_credential(authority: RuntimeLease) -> CredentialFile:
    if not isinstance(authority, RuntimeLease):
        raise TypeError("credential creation requires live RuntimeLease authority")
    if os.name == "nt":
        raise PermissionError(
            "Windows current-user-only DACL cannot be verified; refusing"
        )
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
    path = home / name
    _write_exclusive(authority, name, token.encode("ascii"))
    authority.fsync_home()
    return CredentialFile(path=path, token=token)


def publish_discovery(authority: RuntimeLease, record: DaemonDiscoveryV1) -> Path:
    if not isinstance(authority, RuntimeLease):
        raise TypeError("discovery publication requires live RuntimeLease authority")
    if os.name == "nt":
        raise PermissionError(
            "Windows current-user-only DACL cannot be verified; refusing"
        )
    home = authority.assert_authority()
    record = DaemonDiscoveryV1.model_validate(record.model_dump(mode="python"))
    if record.instance_nonce != authority.instance_nonce:
        raise PermissionError("discovery nonce does not match lease authority")
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
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PermissionError(f"{label} must be a private regular file")
    if metadata.st_nlink != 1:
        raise PermissionError(f"{label} hardlinks are not permitted")
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PermissionError(f"{label} must be user-owned mode 0600")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        if (
            (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino)
            or not stat.S_ISREG(current.st_mode)
            or current.st_uid != os.getuid()
            or stat.S_IMODE(current.st_mode) != 0o600
            or current.st_nlink != 1
        ):
            raise PermissionError(f"{label} changed while it was opened")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(
                descriptor,
                min(65_536, maximum + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        final = os.fstat(descriptor)
        try:
            final_path = path.lstat()
        except OSError as error:
            raise PermissionError(f"{label} changed while it was read") from error
        if (
            (final.st_dev, final.st_ino) != (current.st_dev, current.st_ino)
            or not stat.S_ISREG(final.st_mode)
            or final.st_uid != os.getuid()
            or stat.S_IMODE(final.st_mode) != 0o600
            or final.st_nlink != 1
            or (final_path.st_dev, final_path.st_ino)
            != (current.st_dev, current.st_ino)
            or not stat.S_ISREG(final_path.st_mode)
            or final_path.st_uid != os.getuid()
            or stat.S_IMODE(final_path.st_mode) != 0o600
            or final_path.st_nlink != 1
            or final.st_size != current.st_size
            or final.st_size != len(payload)
        ):
            raise PermissionError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    if len(payload) > maximum:
        raise PermissionError(f"{label} exceeds its byte limit")
    return bytes(payload)


def _read_private_regular_at(
    authority: RuntimeLease,
    name: str,
    *,
    label: str,
    maximum: int,
) -> bytes:
    metadata = authority.stat_home_file(name, follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PermissionError(f"{label} must be a private regular file")
    if metadata.st_nlink != 1:
        raise PermissionError(f"{label} hardlinks are not permitted")
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PermissionError(f"{label} must be user-owned mode 0600")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = authority.open_home_file(name, flags)
    try:
        current = os.fstat(descriptor)
        if (
            (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino)
            or not stat.S_ISREG(current.st_mode)
            or current.st_uid != os.getuid()
            or stat.S_IMODE(current.st_mode) != 0o600
            or current.st_nlink != 1
        ):
            raise PermissionError(f"{label} changed while it was opened")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(
                descriptor,
                min(65_536, maximum + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        final = os.fstat(descriptor)
        try:
            final_path = authority.stat_home_file(name, follow_symlinks=False)
        except OSError as error:
            raise PermissionError(f"{label} changed while it was read") from error
        if (
            (final.st_dev, final.st_ino) != (current.st_dev, current.st_ino)
            or not stat.S_ISREG(final.st_mode)
            or final.st_uid != os.getuid()
            or stat.S_IMODE(final.st_mode) != 0o600
            or final.st_nlink != 1
            or (final_path.st_dev, final_path.st_ino)
            != (current.st_dev, current.st_ino)
            or not stat.S_ISREG(final_path.st_mode)
            or final_path.st_uid != os.getuid()
            or stat.S_IMODE(final_path.st_mode) != 0o600
            or final_path.st_nlink != 1
            or final.st_size != current.st_size
            or final.st_size != len(payload)
        ):
            raise PermissionError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
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


def load_discovery_and_credential(
    runtime_home: str | Path,
) -> tuple[DaemonDiscoveryV1, str]:
    home = _private_home(runtime_home)
    record = _load_discovery_record(home)
    credential_path = home / record.credential_ref
    if credential_path.parent != home:
        raise PermissionError("credential reference escapes runtime home")
    token_bytes = _read_private_regular(
        credential_path, label="credential", maximum=_TOKEN_HEX_BYTES
    )
    try:
        token = token_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise PermissionError("credential is not ASCII") from error
    if len(token) != _TOKEN_HEX_BYTES or any(
        character not in "0123456789abcdef" for character in token
    ):
        raise PermissionError("credential has an invalid encoding")
    return record, token


def _load_discovery_record(home: Path) -> DaemonDiscoveryV1:
    payload = _read_private_regular(
        home / "daemon.json", label="discovery", maximum=_MAX_DISCOVERY_BYTES
    )
    return _decode_discovery_record(payload)


def _decode_discovery_record(payload: bytes) -> DaemonDiscoveryV1:
    record = DaemonDiscoveryV1.model_validate(_strict_json_object(payload))
    if payload != record.canonical_json().encode("utf-8"):
        raise ValueError("discovery JSON is not canonical")
    return record


def _load_discovery_record_at(authority: RuntimeLease) -> DaemonDiscoveryV1:
    payload = _read_private_regular_at(
        authority,
        "daemon.json",
        label="discovery",
        maximum=_MAX_DISCOVERY_BYTES,
    )
    return _decode_discovery_record(payload)


def _validate_credential_at(authority: RuntimeLease, name: str) -> str:
    payload = _read_private_regular_at(
        authority,
        name,
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


def cleanup_credential(authority: RuntimeLease, credential: CredentialFile) -> None:
    """Remove one exact nonce/token credential or quarantine any replacement."""
    if not isinstance(authority, RuntimeLease):
        raise TypeError("credential cleanup requires live RuntimeLease authority")
    if not isinstance(credential, CredentialFile):
        raise TypeError("credential cleanup requires a typed CredentialFile")
    expected_name = f"credential-{authority.instance_nonce}.key"
    if credential.path.name != expected_name:
        raise PermissionError("credential does not match lease nonce")
    try:
        current_token = _validate_credential_at(authority, expected_name)
    except FileNotFoundError:
        return
    if current_token != credential.token:
        raise PermissionError("credential content does not match its owner")
    authority.assert_authority()
    authority.unlink_home_file(expected_name)
    authority.fsync_home()


def cleanup_stale_discovery(authority: RuntimeLease) -> None:
    """Remove only one strictly proven prior-owner discovery/credential pair."""
    if not isinstance(authority, RuntimeLease):
        raise TypeError("discovery cleanup requires live RuntimeLease authority")
    try:
        record = _load_discovery_record_at(authority)
    except FileNotFoundError:
        return
    if record.instance_nonce == authority.instance_nonce:
        return
    _cleanup_discovery_for_nonce(authority, record.instance_nonce)


def cleanup_discovery(authority: RuntimeLease) -> None:
    """Remove only discovery material owned by this still-held lease."""
    if not isinstance(authority, RuntimeLease):
        raise TypeError("discovery cleanup requires live RuntimeLease authority")
    _cleanup_discovery_for_nonce(authority, authority.instance_nonce)


def _cleanup_discovery_for_nonce(authority: RuntimeLease, instance_nonce: str) -> None:
    try:
        record = _load_discovery_record_at(authority)
    except FileNotFoundError:
        return
    if record.instance_nonce != instance_nonce:
        return
    try:
        _validate_credential_at(authority, record.credential_ref)
    except FileNotFoundError:
        credential_exists = False
    else:
        credential_exists = True
    authority.assert_authority()
    if credential_exists:
        authority.unlink_home_file(record.credential_ref)
    try:
        current = _load_discovery_record_at(authority)
    except FileNotFoundError:
        pass
    else:
        if current.instance_nonce == instance_nonce:
            authority.assert_authority()
            authority.unlink_home_file("daemon.json", missing_ok=True)
    authority.assert_authority()
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
