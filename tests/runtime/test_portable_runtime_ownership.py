from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

import portalocker
import psutil
import pytest

from alice_brain_hermes.errors import RuntimeOwnedError
from alice_brain_hermes.protocol.models import (
    DaemonDiscoveryV2,
    LoopbackEndpointV1,
)
from alice_brain_hermes.runtime.discovery import (
    cleanup_stale_discovery,
    create_credential,
    load_discovery_and_credential,
    publish_discovery,
)
from alice_brain_hermes.runtime.lease import RuntimeLease, _external_guard_path
from alice_brain_hermes.runtime.process_marker import (
    current_process_marker,
    read_process_marker,
    verify_process_marker,
)
from alice_brain_hermes.runtime.store import SQLiteLedger


def test_external_and_legacy_guards_are_persistent_and_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"

    lease = RuntimeLease.acquire(home)
    external = _external_guard_path(home)
    legacy = home / "daemon.lock"
    assert external.is_file()
    assert legacy.is_file()
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    lease.release()
    assert external.is_file()
    assert legacy.is_file()
    with RuntimeLease.acquire(home):
        pass
    assert external.is_file()
    assert legacy.is_file()


def test_home_file_mode_uses_generic_fallback_when_nofollow_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    with RuntimeLease.acquire(home) as lease:
        target = lease.home_path("credential-test.key")
        target.write_bytes(b"secret")
        path_type = type(target)
        real_chmod = path_type.chmod
        calls: list[bool] = []

        def chmod(
            path: Path,
            mode: int,
            *,
            follow_symlinks: bool = True,
        ) -> None:
            if path == target:
                calls.append(follow_symlinks)
                if follow_symlinks is False:
                    raise NotImplementedError("nofollow chmod unavailable")
                raise AssertionError("unsafe path-based chmod fallback")
            real_chmod(path, mode, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(path_type, "chmod", chmod)

        lease.chmod_home_file(target.name, 0o600)

        assert calls == [False]
        assert target.read_bytes() == b"secret"


def test_uncertain_legacy_release_retains_external_guard_until_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    events: list[str] = []
    real_legacy_release = lease._legacy_lock.owner.release
    real_external_release = lease._external_lock.owner.release
    failed = False

    def legacy_release() -> None:
        nonlocal failed
        events.append("legacy")
        if not failed:
            failed = True
            raise OSError("injected uncertain legacy release")
        real_legacy_release()

    def external_release() -> None:
        events.append("external")
        real_external_release()

    monkeypatch.setattr(lease._legacy_lock.owner, "release", legacy_release)
    monkeypatch.setattr(lease._external_lock.owner, "release", external_release)

    with pytest.raises(OSError, match="uncertain legacy release"):
        lease.release()
    assert lease.released is False
    assert events == ["legacy"]
    assert lease._legacy_lock.stream is not None
    assert lease._external_lock.stream is not None
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    lease.release()
    assert events == ["legacy", "legacy", "external"]
    assert lease.released is True
    with RuntimeLease.acquire(home):
        pass


def test_normalized_runtime_aliases_share_one_external_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    home.mkdir()
    alias = home.parent / "." / home.name

    assert _external_guard_path(alias) == _external_guard_path(home)
    with RuntimeLease.acquire(alias), pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)


@pytest.mark.parametrize("which", ["external", "legacy"])
def test_either_guard_contention_refuses_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    home.mkdir()
    lock_path = (
        _external_guard_path(home) if which == "external" else home / "daemon.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    held = portalocker.Lock(
        lock_path,
        mode="a+b",
        timeout=0,
        fail_when_locked=True,
    )
    held.acquire()
    try:
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)
    finally:
        held.release()


def test_home_replacement_fails_at_next_sqlite_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    ledger = SQLiteLedger.open(home / "runtime.db", authority=lease)
    replacement = tmp_path / "replacement"
    try:
        try:
            home.rename(replacement)
        except PermissionError:
            with pytest.raises(RuntimeOwnedError):
                RuntimeLease.acquire(home)
            lease.assert_authority()
            return
        home.mkdir()
        with pytest.raises(PermissionError):
            ledger.list_brain_ids()
    finally:
        ledger.close()
        lease.release()


def test_home_replacement_with_matching_locked_guard_fails_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    displaced = tmp_path / "displaced"
    lease = RuntimeLease.acquire(home)
    replacement_guard: portalocker.Lock | None = None
    try:
        try:
            home.rename(displaced)
        except OSError:
            # Some filesystems prevent replacement while the retained lock is open.
            lease.assert_authority()
            return
        home.mkdir()
        replacement_guard = portalocker.Lock(
            home / "daemon.lock",
            mode="a+b",
            timeout=0,
            fail_when_locked=True,
        )
        replacement_guard.acquire()

        with pytest.raises(PermissionError, match="runtime home identity changed"):
            lease.assert_authority()
    finally:
        if replacement_guard is not None:
            replacement_guard.release()
        if displaced.exists():
            (home / "daemon.lock").unlink(missing_ok=True)
            home.rmdir()
            displaced.rename(home)
        lease.release()


def test_runtime_database_symlink_cannot_mutate_outside_target_or_publish_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_symlink: Callable[[Path, Path, bool], bool],
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    outside = tmp_path / "outside.db"
    with sqlite3.connect(outside) as connection:
        connection.execute("CREATE TABLE sentinel(value TEXT NOT NULL)")
        connection.execute("INSERT INTO sentinel(value) VALUES ('unchanged')")
    before = outside.read_bytes()
    home = tmp_path / "runtime"
    owners: list[SQLiteLedger] = []

    with RuntimeLease.acquire(home) as lease:
        runtime_database = home / "runtime.db"
        make_symlink(runtime_database, outside, False)
        try:
            with pytest.raises(PermissionError, match="SQLite runtime path"):
                SQLiteLedger.open(
                    home / "runtime.db",
                    authority=lease,
                    owner_sink=owners.append,
                )
        finally:
            for owner in owners:
                owner.close()

        assert owners == []
        assert not (home / "runtime.db-wal").exists()
        assert not (home / "runtime.db-shm").exists()
    (home / "runtime.db").unlink()
    assert not (home / "runtime.db").exists()
    with RuntimeLease.acquire(home):
        pass
    assert outside.read_bytes() == before
    with sqlite3.connect(outside) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall() == [("sentinel",)]
        assert connection.execute("SELECT value FROM sentinel").fetchall() == [
            ("unchanged",)
        ]


@pytest.mark.parametrize(
    "runtime_name",
    ["runtime.db", "runtime.db-wal", "runtime.db-shm", "runtime.db-journal"],
)
@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory"])
def test_sqlite_rejects_preexisting_unsafe_database_and_sidecar_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_name: str,
    unsafe_kind: str,
    make_symlink: Callable[[Path, Path, bool], bool],
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    outside = tmp_path / f"outside-{runtime_name}"
    owners: list[SQLiteLedger] = []

    with RuntimeLease.acquire(home) as lease:
        unsafe = home / runtime_name
        if unsafe_kind == "symlink":
            outside.write_bytes(b"outside-sentinel")
            make_symlink(unsafe, outside, False)
        else:
            unsafe.mkdir()
        try:
            with pytest.raises(PermissionError, match="SQLite runtime path"):
                SQLiteLedger.open(
                    home / "runtime.db",
                    authority=lease,
                    owner_sink=owners.append,
                )
        finally:
            for owner in owners:
                owner.close()

        assert owners == []
        if runtime_name != "runtime.db":
            assert not (home / "runtime.db").exists()
        if unsafe_kind == "symlink":
            assert outside.read_bytes() == b"outside-sentinel"


def test_process_marker_uses_rounded_psutil_create_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        def __init__(self, pid: int) -> None:
            assert pid == 42

        def create_time(self) -> float:
            return 123.4567894

    monkeypatch.setattr(psutil, "Process", Process)
    assert read_process_marker(42) == "psutil-create-time-us:123456789"
    verify_process_marker(42, "psutil-create-time-us:123456789")
    with pytest.raises(PermissionError, match="does not match"):
        verify_process_marker(42, "psutil-create-time-us:123456788")


def test_process_marker_missing_process_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_pid: int) -> None:
        raise psutil.NoSuchProcess(123)

    monkeypatch.setattr(psutil, "Process", missing)
    with pytest.raises(PermissionError, match="not verifiable"):
        read_process_marker(123)


def test_discovery_v2_round_trip_binds_launch_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    with RuntimeLease.acquire(home, launch_nonce="parent-launch") as lease:
        credential = create_credential(lease)
        record = DaemonDiscoveryV2(
            pid=os.getpid(),
            process_marker=current_process_marker(),
            instance_nonce=lease.instance_nonce,
            launch_nonce=lease.launch_nonce,
            endpoint=LoopbackEndpointV1(port=43210),
            credential_ref=credential.path.name,
        )
        publish_discovery(lease, record)
        loaded, token = load_discovery_and_credential(home)

        assert loaded == record
        assert loaded.schema_version == 2
        assert loaded.launch_nonce == "parent-launch"
        assert token == credential.token


def test_live_v1_refuses_and_dead_v1_is_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    home.mkdir()
    daemon_json = home / "daemon.json"
    daemon_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "instance_nonce": "legacy-live",
                "credential_ref": "credential-legacy-live.key",
            }
        ),
        encoding="utf-8",
    )
    with (
        RuntimeLease.acquire(home) as lease,
        pytest.raises(RuntimeOwnedError, match="legacy discovery"),
    ):
        cleanup_stale_discovery(lease)
    assert daemon_json.exists()

    daemon_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": 2_147_483_647,
                "instance_nonce": "legacy-dead",
                "credential_ref": "credential-legacy-dead.key",
            }
        ),
        encoding="utf-8",
    )
    with RuntimeLease.acquire(home) as lease:
        cleanup_stale_discovery(lease)
    assert not daemon_json.exists()


@pytest.mark.parametrize(
    "credential_ref",
    ["daemon.lock", "daemon.json", "runtime.db", "runtime.db-wal", "../outside"],
)
def test_dead_v1_cannot_remove_reserved_or_noncredential_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    credential_ref: str,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    home.mkdir()
    daemon_json = home / "daemon.json"
    daemon_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pid": 2_147_483_647,
                "instance_nonce": "legacy-dead",
                "credential_ref": credential_ref,
            }
        ),
        encoding="utf-8",
    )
    protected = home / "runtime.db"
    protected.write_bytes(b"protected")

    with (
        RuntimeLease.acquire(home) as lease,
        pytest.raises(RuntimeOwnedError, match="legacy discovery"),
    ):
        cleanup_stale_discovery(lease)

    assert daemon_json.exists()
    assert protected.read_bytes() == b"protected"
    assert (home / "daemon.lock").exists()


def test_runtime_database_is_wal_and_reopens_after_abrupt_connection_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", os.fspath(tmp_path / "data"))
    home = tmp_path / "runtime"
    with RuntimeLease.acquire(home) as lease:
        ledger = SQLiteLedger.open(home / "runtime.db", authority=lease)
        assert ledger.journal_mode == "wal"
        ledger.close()
    with RuntimeLease.acquire(home) as lease:
        reopened = SQLiteLedger.open(home / "runtime.db", authority=lease)
        try:
            assert reopened.journal_mode == "wal"
        finally:
            reopened.close()
