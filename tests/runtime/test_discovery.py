from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from alice_brain_hermes.protocol.models import DaemonDiscoveryV2, LoopbackEndpointV1
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


def _record(lease: RuntimeLease, credential_name: str) -> DaemonDiscoveryV2:
    return DaemonDiscoveryV2(
        pid=os.getpid(),
        process_marker=current_process_marker(),
        instance_nonce=lease.instance_nonce,
        launch_nonce=lease.launch_nonce,
        endpoint=LoopbackEndpointV1(port=43210),
        credential_ref=credential_name,
    )


def test_discovery_contains_reference_not_token(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    with RuntimeLease.acquire(home) as lease:
        credential = create_credential(lease)
        record = _record(lease, credential.path.name)
        publish_discovery(lease, record)

        body = json.loads((home / "daemon.json").read_text(encoding="utf-8"))
        assert body["schema_version"] == 2
        assert "token" not in body
        assert credential.token not in (home / "daemon.json").read_text(
            encoding="utf-8"
        )
        loaded, token = load_discovery_and_credential(home)
        assert loaded == record
        assert token == credential.token


def test_discovery_rejects_traversal_and_symlinked_credential(
    tmp_path: Path,
    make_symlink: Callable[[Path, Path, bool], bool],
) -> None:
    home = tmp_path / "runtime"
    home.mkdir()
    outside = tmp_path / "outside.key"
    outside.write_text("0" * 64, encoding="ascii")
    link = home / "credential-nonce-a.key"
    make_symlink(link, outside, False)
    record = DaemonDiscoveryV2(
        pid=os.getpid(),
        process_marker=current_process_marker(),
        instance_nonce="nonce-a",
        launch_nonce="launch-a",
        endpoint=LoopbackEndpointV1(port=43210),
        credential_ref=link.name,
    )
    (home / "daemon.json").write_text(record.canonical_json(), encoding="utf-8")

    with pytest.raises(PermissionError, match="credential"):
        load_discovery_and_credential(home)

    with pytest.raises(ValueError):
        DaemonDiscoveryV2(
            pid=os.getpid(),
            process_marker=current_process_marker(),
            instance_nonce="nonce-a",
            launch_nonce="launch-a",
            endpoint=LoopbackEndpointV1(port=43210),
            credential_ref="../outside.key",
        )


def test_cleanup_removes_only_current_nonce_files(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    old = home / "credential-old.key"
    with RuntimeLease.acquire(home) as lease:
        old.write_text("0" * 64, encoding="ascii")
        current = create_credential(lease)
        publish_discovery(lease, _record(lease, current.path.name))

        with pytest.raises(TypeError, match="RuntimeLease"):
            cleanup_stale_discovery(home)  # type: ignore[arg-type]
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
    with RuntimeLease.acquire(home) as lease:
        credential = create_credential(lease)
        record = _record(lease, credential.path.name)
        publish_discovery(lease, record)
        noncanonical = json.dumps(record.model_dump(mode="json"), indent=2).encode()
        (home / "daemon.json").write_bytes(noncanonical)

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
    with RuntimeLease.acquire(home) as lease:
        credential = create_credential(lease)
        publish_discovery(lease, _record(lease, credential.path.name))
        credential.path.unlink()

        cleanup_discovery(lease)
        assert not (home / "daemon.json").exists()


def test_cleanup_quarantines_invalid_credential_without_unlinking_discovery(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    with RuntimeLease.acquire(home) as lease:
        credential = create_credential(lease)
        record = _record(lease, credential.path.name)
        publish_discovery(lease, record)
        credential.path.write_text("not-a-valid-token", encoding="ascii")

        with pytest.raises(PermissionError, match="credential"):
            cleanup_discovery(lease)

        assert (home / "daemon.json").read_bytes() == record.canonical_json().encode()
        assert credential.path.exists()


def test_private_file_reader_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "private"
    path.write_bytes(b"x" * 65)

    with pytest.raises(PermissionError, match="byte limit"):
        _read_private_regular(path, label="fixture", maximum=64)


@pytest.mark.parametrize("depth", [40, 2_000])
def test_discovery_json_depth_and_parser_recursion_are_stable_value_errors(
    depth: int,
) -> None:
    payload = b'{"x":' * depth + b"0" + b"}" * depth

    with pytest.raises(ValueError, match="discovery"):
        _strict_json_object(payload)
