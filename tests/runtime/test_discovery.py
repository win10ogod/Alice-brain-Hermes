from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from alice_brain_hermes.protocol.models import DaemonDiscoveryV1, LoopbackEndpointV1
from alice_brain_hermes.runtime.discovery import (
    _read_private_regular,
    _strict_json_object,
    cleanup_discovery,
    cleanup_stale_discovery,
    create_credential,
    load_discovery_and_credential,
    publish_discovery,
)
from alice_brain_hermes.runtime.lease import RuntimeLease
from alice_brain_hermes.runtime.process_marker import current_process_marker


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission contract")
def test_discovery_contains_reference_not_token_and_files_are_private(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    with RuntimeLease.acquire(home) as lease:
        credential = create_credential(lease)
        record = DaemonDiscoveryV1(
            pid=os.getpid(),
            process_marker=current_process_marker(),
            instance_nonce=lease.instance_nonce,
            endpoint=LoopbackEndpointV1(host="127.0.0.1", port=43210),
            credential_ref=credential.path.name,
        )

        publish_discovery(lease, record)

        body = json.loads((home / "daemon.json").read_text(encoding="utf-8"))
        assert "token" not in body
        assert credential.token not in (home / "daemon.json").read_text(
            encoding="utf-8"
        )
        assert Path(body["credential_ref"]).name == body["credential_ref"]
        assert credential.path.stat().st_mode & 0o077 == 0
        assert (home / "daemon.json").stat().st_mode & 0o077 == 0
        loaded, token = load_discovery_and_credential(home)
        assert loaded == record
        assert token == credential.token


@pytest.mark.skipif(os.name == "nt", reason="POSIX openat contract")
def test_credential_creation_ancestor_swap_never_mutates_replacement_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    retained = tmp_path / "retained"
    real_open = os.open
    with RuntimeLease.acquire(home) as lease:
        credential_name = f"credential-{lease.instance_nonce}.key"
        swapped = False

        def swap_before_credential_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal swapped
            if path == credential_name and dir_fd is not None and not swapped:
                swapped = True
                home.rename(retained)
                home.mkdir(mode=0o700)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(
            "alice_brain_hermes.runtime.discovery.os.open",
            swap_before_credential_open,
        )

        with pytest.raises(PermissionError, match="authority"):
            create_credential(lease)

        assert swapped is True
        assert not (home / credential_name).exists()
        assert (retained / credential_name).exists()


def test_discovery_rejects_traversal_and_symlinked_credential(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    outside = tmp_path / "outside.key"
    outside.write_text("0" * 64, encoding="ascii")
    link = home / "credential-nonce-a.key"
    link.symlink_to(outside)
    record = DaemonDiscoveryV1(
        pid=os.getpid(),
        process_marker=current_process_marker(),
        instance_nonce="nonce-a",
        endpoint=LoopbackEndpointV1(host="127.0.0.1", port=43210),
        credential_ref=link.name,
    )
    (home / "daemon.json").write_text(record.canonical_json(), encoding="utf-8")
    if os.name != "nt":
        os.chmod(home / "daemon.json", 0o600)

    with pytest.raises(PermissionError, match="credential"):
        load_discovery_and_credential(home)

    with pytest.raises(ValueError):
        DaemonDiscoveryV1(
            pid=os.getpid(),
            process_marker=current_process_marker(),
            instance_nonce="nonce-a",
            endpoint=LoopbackEndpointV1(host="127.0.0.1", port=43210),
            credential_ref="../outside.key",
        )


def test_cleanup_removes_only_current_nonce_files(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    old = home / "credential-old.key"
    old.write_text("0" * 64, encoding="ascii")
    old.chmod(0o600)
    with RuntimeLease.acquire(home) as lease:
        current = create_credential(lease)
        record = DaemonDiscoveryV1(
            pid=os.getpid(),
            process_marker=current_process_marker(),
            instance_nonce=lease.instance_nonce,
            endpoint=LoopbackEndpointV1(host="127.0.0.1", port=43210),
            credential_ref=current.path.name,
        )
        publish_discovery(lease, record)

        with pytest.raises(TypeError, match="RuntimeLease"):
            cleanup_stale_discovery(home)  # type: ignore[arg-type]
        assert current.path.exists()
        assert (home / "daemon.json").exists()

        cleanup_discovery(lease)
        assert not current.path.exists()
        assert not (home / "daemon.json").exists()
        assert old.exists()


@pytest.mark.parametrize("operation", ["client_load", "authority_cleanup"])
def test_noncanonical_discovery_fails_closed_without_mutation(
    tmp_path: Path,
    operation: str,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    with RuntimeLease.acquire(home) as lease:
        credential = create_credential(lease)
        record = DaemonDiscoveryV1(
            pid=os.getpid(),
            process_marker=current_process_marker(),
            instance_nonce=lease.instance_nonce,
            endpoint=LoopbackEndpointV1(port=43210),
            credential_ref=credential.path.name,
        )
        publish_discovery(lease, record)
        noncanonical = json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        ).encode("utf-8")
        assert noncanonical != record.canonical_json().encode("utf-8")
        (home / "daemon.json").write_bytes(noncanonical)
        (home / "daemon.json").chmod(0o600)

        with pytest.raises(ValueError, match="canonical"):
            if operation == "client_load":
                load_discovery_and_credential(home)
            else:
                cleanup_discovery(lease)

        assert (home / "daemon.json").read_bytes() == noncanonical
        assert credential.path.exists()


def test_cleanup_removes_current_discovery_when_credential_is_missing(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    with RuntimeLease.acquire(home) as lease:
        credential = create_credential(lease)
        record = DaemonDiscoveryV1(
            pid=os.getpid(),
            process_marker=current_process_marker(),
            instance_nonce=lease.instance_nonce,
            endpoint=LoopbackEndpointV1(host="127.0.0.1", port=43210),
            credential_ref=credential.path.name,
        )
        publish_discovery(lease, record)
        credential.path.unlink()

        cleanup_discovery(lease)
        assert not (home / "daemon.json").exists()
        assert not credential.path.exists()


@pytest.mark.parametrize(
    "credential_state", ["invalid-content", "insecure-mode", "hardlink", "symlink"]
)
def test_cleanup_quarantines_unsafe_credential_without_unlinking_discovery(
    tmp_path: Path, credential_state: str
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    outside = tmp_path / "outside.key"
    outside.write_text("f" * 64, encoding="ascii")
    outside.chmod(0o600)
    with RuntimeLease.acquire(home) as lease:
        credential = create_credential(lease)
        record = DaemonDiscoveryV1(
            pid=os.getpid(),
            process_marker=current_process_marker(),
            instance_nonce=lease.instance_nonce,
            endpoint=LoopbackEndpointV1(host="127.0.0.1", port=43210),
            credential_ref=credential.path.name,
        )
        publish_discovery(lease, record)
        if credential_state == "invalid-content":
            credential.path.write_text("not-a-valid-token", encoding="ascii")
        elif credential_state == "insecure-mode":
            credential.path.chmod(0o644)
        elif credential_state == "hardlink":
            os.link(credential.path, tmp_path / "credential-hardlink")
        else:
            credential.path.unlink()
            credential.path.symlink_to(outside)

        with pytest.raises(PermissionError, match="credential"):
            cleanup_discovery(lease)

        assert (home / "daemon.json").read_bytes() == (
            record.canonical_json().encode("utf-8")
        )
        assert credential.path.exists()
        assert outside.read_text(encoding="ascii") == "f" * 64


def test_windows_privacy_branch_fails_closed_when_acl_cannot_be_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    with RuntimeLease.acquire(home) as lease:
        monkeypatch.setattr("alice_brain_hermes.runtime.discovery.os.name", "nt")

        with pytest.raises(PermissionError, match="DACL"):
            create_credential(lease)


@pytest.mark.skipif(os.name == "nt", reason="POSIX hardlink contract")
def test_credential_hardlinks_fail_closed(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    with RuntimeLease.acquire(home) as lease:
        credential = create_credential(lease)
        record = DaemonDiscoveryV1(
            pid=os.getpid(),
            process_marker=current_process_marker(),
            instance_nonce=lease.instance_nonce,
            endpoint=LoopbackEndpointV1(host="127.0.0.1", port=43210),
            credential_ref=credential.path.name,
        )
        publish_discovery(lease, record)
        os.link(credential.path, tmp_path / "credential-hardlink")

        with pytest.raises(PermissionError, match="hardlink"):
            load_discovery_and_credential(home)


def test_private_file_reader_loops_over_legal_short_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "private"
    path.write_bytes(b"bounded-private-payload")
    path.chmod(0o600)
    real_read = os.read

    def short_read(descriptor: int, maximum: int) -> bytes:
        return real_read(descriptor, min(maximum, 3))

    monkeypatch.setattr("alice_brain_hermes.runtime.discovery.os.read", short_read)

    assert _read_private_regular(path, label="fixture", maximum=64) == (
        b"bounded-private-payload"
    )


def test_private_file_reader_rechecks_security_metadata_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "private"
    path.write_bytes(b"payload")
    path.chmod(0o600)
    real_fstat = os.fstat
    calls = 0

    def chmod_before_final_check(descriptor: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            os.fchmod(descriptor, 0o644)
        return real_fstat(descriptor)

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.discovery.os.fstat",
        chmod_before_final_check,
    )

    with pytest.raises(PermissionError, match="changed"):
        _read_private_regular(path, label="fixture", maximum=64)


def test_private_file_reader_rejects_final_descriptor_inode_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "private"
    path.write_bytes(b"payload")
    path.chmod(0o600)
    real_fstat = os.fstat
    calls = 0

    def swapped_final_inode(descriptor: int):
        nonlocal calls
        calls += 1
        metadata = real_fstat(descriptor)
        if calls != 2:
            return metadata
        values = list(metadata)
        values[1] += 1
        return os.stat_result(values)

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.discovery.os.fstat", swapped_final_inode
    )

    with pytest.raises(PermissionError, match="changed"):
        _read_private_regular(path, label="fixture", maximum=64)


@pytest.mark.parametrize("depth", [40, 2_000])
def test_discovery_json_depth_and_parser_recursion_are_stable_value_errors(
    depth: int,
) -> None:
    payload = b'{"x":' * depth + b"0" + b"}" * depth

    with pytest.raises(ValueError, match="discovery"):
        _strict_json_object(payload)


def test_private_file_writer_handles_short_and_zero_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    real_write = os.write

    def short_write(descriptor: int, payload: bytes) -> int:
        return real_write(descriptor, payload[: max(1, len(payload) // 3)])

    monkeypatch.setattr("alice_brain_hermes.runtime.discovery.os.write", short_write)
    with RuntimeLease.acquire(home) as lease:
        credential = create_credential(lease)
        assert credential.path.read_text(encoding="ascii") == credential.token
        lease.unlink_home_file(credential.path.name)

        monkeypatch.setattr(
            "alice_brain_hermes.runtime.discovery.os.write",
            lambda _descriptor, _payload: 0,
        )
        with pytest.raises(OSError, match="no progress"):
            create_credential(lease)
        assert not credential.path.exists()
