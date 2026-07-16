from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from alice_brain_hermes.errors import RuntimeOwnedError
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.runtime.daemon import HermesDaemonRuntime
from alice_brain_hermes.runtime.lease import RuntimeLease
from alice_brain_hermes.runtime.scheduler import ContinuousScheduler
from alice_brain_hermes.runtime.store import SQLiteLedger


def test_list_brain_ids_is_public_sorted_and_complete(tmp_path: Path) -> None:
    first = new_id()
    second = new_id()
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        ledger.ensure_brain(second)
        ledger.ensure_brain(first)

        assert ledger.list_brain_ids() == sorted((first, second))


def test_scheduler_join_must_prove_writer_exit() -> None:
    scheduler = object.__new__(ContinuousScheduler)
    scheduler._creator_pid = os.getpid()
    scheduler._stop = threading.Event()
    scheduler._thread = threading.current_thread()

    with pytest.raises(RuntimeError, match="current scheduler thread"):
        scheduler.join(timeout=0.01)


def test_runtime_lease_is_held_and_second_owner_fails(tmp_path: Path) -> None:
    runtime_home = tmp_path / "home"

    with RuntimeLease.acquire(runtime_home) as first:
        assert first.path.name == "daemon.lock"
        assert first.external_guard_path.is_file()
        assert first.instance_nonce
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(runtime_home)

    with RuntimeLease.acquire(runtime_home) as recovered:
        assert recovered.instance_nonce != first.instance_nonce


def test_runtime_path_guard_survives_home_rename_and_replacement(
    tmp_path: Path,
) -> None:
    runtime_home = tmp_path / "home"
    displaced_home = tmp_path / "displaced-home"
    lease = RuntimeLease.acquire(runtime_home)
    try:
        runtime_home.rename(displaced_home)
    except PermissionError:
        # Some filesystems prevent namespace replacement while the portable
        # lock file is open. That is already the stronger ownership outcome.
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(runtime_home)
        assert lease.assert_authority() == runtime_home
        lease.release()
        runtime_home.rename(displaced_home)
        displaced_home.rename(runtime_home)
        with RuntimeLease.acquire(runtime_home):
            pass
        return
    runtime_home.mkdir()
    try:
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(runtime_home)
        with pytest.raises(PermissionError):
            lease.assert_authority()
    finally:
        runtime_home.rmdir()
        displaced_home.rename(runtime_home)
        lease.release()

    with RuntimeLease.acquire(runtime_home):
        pass


def test_runtime_home_symlink_is_rejected_before_mutation(
    tmp_path: Path,
    make_symlink: Callable[[Path, Path, bool], bool],
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "runtime-link"
    make_symlink(link, outside, True)

    with pytest.raises(PermissionError, match="symbolic link"):
        RuntimeLease.acquire(link)

    assert not (outside / "daemon.lock").exists()


def test_runtime_acquires_both_locks_before_opening_sqlite(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    observed: list[Path] = []

    def ledger_factory(path: Path) -> SQLiteLedger:
        observed.append(path)
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)
        return SQLiteLedger.open(path)

    runtime = HermesDaemonRuntime.open(
        home,
        ledger_factory=ledger_factory,
        scheduler_interval_seconds=60.0,
    )
    try:
        assert observed == [home / "runtime.db"]
        assert runtime.ledger.journal_mode == "wal"
    finally:
        runtime.close()


def test_custom_factory_is_not_invoked_for_unsafe_runtime_database(
    tmp_path: Path,
    make_symlink: Callable[[Path, Path, bool], bool],
) -> None:
    home = tmp_path / "runtime"
    home.mkdir()
    outside = tmp_path / "outside.db"
    with sqlite3.connect(outside) as connection:
        connection.execute("CREATE TABLE sentinel(value TEXT NOT NULL)")
        connection.execute("INSERT INTO sentinel(value) VALUES ('unchanged')")
    before = outside.read_bytes()
    runtime_database = home / "runtime.db"
    make_symlink(runtime_database, outside, False)
    factory_calls: list[Path] = []

    def ledger_factory(path: Path) -> SQLiteLedger:
        factory_calls.append(path)
        return SQLiteLedger.open(path)

    with pytest.raises(PermissionError, match="SQLite runtime path"):
        HermesDaemonRuntime.open(
            home,
            ledger_factory=ledger_factory,
            scheduler_interval_seconds=60.0,
        )

    assert factory_calls == []
    assert outside.read_bytes() == before
    assert not (home / "runtime.db-wal").exists()
    assert not (home / "runtime.db-shm").exists()
    assert list(home.glob("credential-*.key")) == []
    (home / "runtime.db").unlink()
    assert not (home / "runtime.db").exists()
    with RuntimeLease.acquire(home):
        pass


def test_custom_factory_result_is_closed_if_path_becomes_unsafe_before_adoption(
    tmp_path: Path,
    make_symlink: Callable[[Path, Path, bool], bool],
) -> None:
    home = tmp_path / "runtime"
    outside = tmp_path / "outside-journal"
    outside.write_bytes(b"outside-sentinel")
    returned: list[SQLiteLedger] = []

    def ledger_factory(path: Path) -> SQLiteLedger:
        ledger = SQLiteLedger.open(path)
        returned.append(ledger)
        journal = home / "runtime.db-journal"
        make_symlink(journal, outside, False)
        return ledger

    runtime: HermesDaemonRuntime | None = None
    try:
        with pytest.raises(PermissionError, match="SQLite runtime path"):
            runtime = HermesDaemonRuntime.open(
                home,
                ledger_factory=ledger_factory,
                scheduler_interval_seconds=60.0,
            )
    finally:
        if runtime is not None:
            runtime.close()

    assert len(returned) == 1
    assert returned[0].closed
    assert outside.read_bytes() == b"outside-sentinel"
    assert list(home.glob("credential-*.key")) == []
    (home / "runtime.db-journal").unlink()
    with RuntimeLease.acquire(home):
        pass


def test_runtime_closes_wal_ledger_before_releasing_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = HermesDaemonRuntime.open(
        tmp_path / "runtime", scheduler_interval_seconds=60.0
    )
    released_after_close: list[bool] = []
    real_release = RuntimeLease.release

    def release(lease: RuntimeLease) -> None:
        released_after_close.append(runtime.ledger.closed)
        real_release(lease)

    monkeypatch.setattr(RuntimeLease, "release", release)
    runtime.close()

    assert released_after_close == [True]
    with RuntimeLease.acquire(runtime.runtime_home):
        pass


def test_startup_failure_releases_owner_without_creating_writable_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"

    def fail_open(
        _ledger_type: type[SQLiteLedger],
        path: str | Path,
        **_kwargs: object,
    ) -> SQLiteLedger:
        assert Path(path) == home / "runtime.db"
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)
        raise RuntimeError("injected SQLite startup failure")

    monkeypatch.setattr(SQLiteLedger, "open", classmethod(fail_open))

    with pytest.raises(RuntimeError, match="startup failure"):
        HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

    assert not (home / "runtime.db").exists()
    with RuntimeLease.acquire(home):
        pass
