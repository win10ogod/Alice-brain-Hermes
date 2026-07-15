from __future__ import annotations

import errno
import os
import sqlite3
import threading
from contextlib import suppress
from pathlib import Path

import pytest

from alice_brain_hermes.errors import RuntimeOwnedError
from alice_brain_hermes.runtime import store as store_module
from alice_brain_hermes.runtime.daemon import (
    HermesDaemonRuntime,
    _PreRuntimeOwner,
)
from alice_brain_hermes.runtime.lease import RetainedSQLiteFiles, RuntimeLease


def _is_open(descriptor: int) -> bool:
    try:
        os.fstat(descriptor)
    except OSError:
        return False
    return True


def _capture_retained_files(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[RetainedSQLiteFiles], list[tuple[int, ...]]]:
    captured: list[RetainedSQLiteFiles] = []
    original_descriptors: list[tuple[int, ...]] = []
    real_retain = RuntimeLease.retain_sqlite_files

    def retain(lease: RuntimeLease, name: str = "runtime.db") -> RetainedSQLiteFiles:
        owner = real_retain(lease, name)
        captured.append(owner)
        original_descriptors.append(tuple(owner._descriptors.values()))
        return owner

    monkeypatch.setattr(RuntimeLease, "retain_sqlite_files", retain)
    return captured, original_descriptors


class _StartupPrimaryConnection:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        close_failures: int,
        close_after_error: bool = False,
    ) -> None:
        self._connection = connection
        self._close_failures = close_failures
        self._close_after_error = close_after_error
        self.close_calls = 0
        self._row_factory = None

    @property
    def row_factory(self):
        return self._connection.row_factory

    @row_factory.setter
    def row_factory(self, value) -> None:
        self._connection.row_factory = value

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    def execute(self, _statement: str, *_args, **_kwargs):
        raise RuntimeError("injected schema primary")

    def close(self) -> None:
        self.close_calls += 1
        if self._close_after_error and self.close_calls == 1:
            self._connection.close()
            raise OSError("injected post-connection-close secondary")
        if self.close_calls <= self._close_failures:
            raise OSError("injected connection close secondary")
        self._connection.close()

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()


@pytest.mark.parametrize("failure_stage", ["connect", "verify"])
def test_preconnection_failure_closes_retained_files_before_releasing_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    home = tmp_path / "runtime"
    captured, original_descriptors = _capture_retained_files(monkeypatch)
    baseline = HermesDaemonRuntime.failed_owner_count()

    if failure_stage == "connect":
        monkeypatch.setattr(
            store_module.sqlite3,
            "connect",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected connect primary")
            ),
        )
        expected = "connect primary"
    else:
        real_verify = RetainedSQLiteFiles.verify
        verify_calls = 0

        def verify(
            owner: RetainedSQLiteFiles,
            *,
            allow_missing_transient: bool = False,
        ) -> None:
            nonlocal verify_calls
            verify_calls += 1
            if verify_calls == 2:
                raise RuntimeError("injected verify primary")
            real_verify(
                owner,
                allow_missing_transient=allow_missing_transient,
            )

        monkeypatch.setattr(RetainedSQLiteFiles, "verify", verify)
        expected = "verify primary"

    real_close = os.close
    try:
        with pytest.raises(RuntimeError, match=expected):
            HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

        assert len(captured) == 1
        assert all(not _is_open(descriptor) for descriptor in original_descriptors[0])
        assert HermesDaemonRuntime.failed_owner_count() == baseline
        with RuntimeLease.acquire(home):
            pass
    finally:
        for owner in captured:
            for descriptor in tuple(owner._descriptors.values()):
                if _is_open(descriptor):
                    real_close(descriptor)


def test_failed_open_connection_cleanup_retains_owner_and_primary_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    captured, original_descriptors = _capture_retained_files(monkeypatch)
    real_connect = sqlite3.connect
    proxy: _StartupPrimaryConnection | None = None

    def connect(*args, **kwargs) -> _StartupPrimaryConnection:
        nonlocal proxy
        proxy = _StartupPrimaryConnection(
            real_connect(*args, **kwargs),
            close_failures=2,
        )
        return proxy

    monkeypatch.setattr(store_module.sqlite3, "connect", connect)
    baseline = HermesDaemonRuntime.failed_owner_count()

    try:
        with pytest.raises(RuntimeError, match="injected schema primary") as raised:
            HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

        assert isinstance(raised.value.__cause__, OSError)
        assert proxy is not None and proxy.close_calls == 2
        assert len(captured) == 1
        assert all(_is_open(descriptor) for descriptor in original_descriptors[0])
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)

        assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
        assert proxy.close_calls == 3
        assert HermesDaemonRuntime.failed_owner_count() == baseline
        assert all(not _is_open(descriptor) for descriptor in original_descriptors[0])
        with RuntimeLease.acquire(home):
            pass
    finally:
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            with suppress(BaseException):
                HermesDaemonRuntime.retry_failed_cleanup(home)
        if proxy is not None:
            while proxy.close_calls <= 2:
                try:
                    proxy.close()
                except OSError:
                    continue
            if captured:
                for descriptor in original_descriptors[0]:
                    if _is_open(descriptor):
                        os.close(descriptor)


def test_constructor_failure_closes_adopted_ledger_before_lease_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    _captured_retained, original_descriptors = _capture_retained_files(monkeypatch)
    captured_ledgers: list[store_module.SQLiteLedger] = []
    real_open = store_module.SQLiteLedger.open_retained

    def open_retained(
        _ledger_type: type[store_module.SQLiteLedger],
        owner: RetainedSQLiteFiles,
        *,
        owner_sink=None,
    ) -> store_module.SQLiteLedger:
        ledger = real_open(owner, owner_sink=owner_sink)
        captured_ledgers.append(ledger)
        return ledger

    def fail_constructor(*_args, **_kwargs) -> None:
        raise RuntimeError("injected runtime constructor primary")

    monkeypatch.setattr(
        store_module.SQLiteLedger,
        "open_retained",
        classmethod(open_retained),
    )
    monkeypatch.setattr(HermesDaemonRuntime, "__init__", fail_constructor)
    baseline = HermesDaemonRuntime.failed_owner_count()

    try:
        with pytest.raises(RuntimeError, match="constructor primary"):
            HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

        assert len(captured_ledgers) == 1
        assert captured_ledgers[0].closed is True
        assert all(not _is_open(descriptor) for descriptor in original_descriptors[0])
        assert HermesDaemonRuntime.failed_owner_count() == baseline
        with RuntimeLease.acquire(home):
            pass
    finally:
        for ledger in captured_ledgers:
            if not ledger.closed:
                ledger.close()


def test_connection_close_then_error_is_proven_terminal_without_losing_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    _captured, original_descriptors = _capture_retained_files(monkeypatch)
    real_connect = sqlite3.connect
    proxy: _StartupPrimaryConnection | None = None

    def connect(*args, **kwargs) -> _StartupPrimaryConnection:
        nonlocal proxy
        proxy = _StartupPrimaryConnection(
            real_connect(*args, **kwargs),
            close_failures=0,
            close_after_error=True,
        )
        return proxy

    monkeypatch.setattr(store_module.sqlite3, "connect", connect)
    baseline = HermesDaemonRuntime.failed_owner_count()

    with pytest.raises(RuntimeError, match="injected schema primary") as raised:
        HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

    assert isinstance(raised.value.__cause__, OSError)
    assert proxy is not None and proxy.close_calls == 1
    assert all(not _is_open(descriptor) for descriptor in original_descriptors[0])
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    with RuntimeLease.acquire(home):
        pass


def test_failed_startup_retained_close_keeps_exact_fd_and_lease_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    captured, _original_descriptors = _capture_retained_files(monkeypatch)
    real_connect = sqlite3.connect
    real_close = os.close
    target: int | None = None
    close_failures = 0

    def connect(*args, **kwargs) -> _StartupPrimaryConnection:
        nonlocal target
        assert captured
        target = tuple(captured[0]._descriptors.values())[-1]
        return _StartupPrimaryConnection(
            real_connect(*args, **kwargs),
            close_failures=0,
        )

    def fail_target_twice(descriptor: int) -> None:
        nonlocal close_failures
        if descriptor == target and close_failures < 2:
            close_failures += 1
            raise OSError("injected retained close secondary")
        real_close(descriptor)

    monkeypatch.setattr(store_module.sqlite3, "connect", connect)
    monkeypatch.setattr(os, "close", fail_target_twice)
    baseline = HermesDaemonRuntime.failed_owner_count()

    try:
        with pytest.raises(RuntimeError, match="injected schema primary") as raised:
            HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

        assert isinstance(raised.value.__cause__, OSError)
        assert target is not None and _is_open(target)
        assert target in captured[0]._descriptors.values()
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)

        assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
        assert not _is_open(target)
        assert HermesDaemonRuntime.failed_owner_count() == baseline
        with RuntimeLease.acquire(home):
            pass
    finally:
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            with suppress(BaseException):
                HermesDaemonRuntime.retry_failed_cleanup(home)
        if target is not None and _is_open(target):
            real_close(target)


def test_live_retained_close_failure_is_retried_before_lease_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    retained = runtime.ledger._retained_files
    assert retained is not None
    target = tuple(retained._descriptors.values())[-1]
    real_close = os.close
    failed = False

    def fail_once(descriptor: int) -> None:
        nonlocal failed
        if descriptor == target and not failed:
            failed = True
            raise OSError("injected retained close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", fail_once)
    baseline = HermesDaemonRuntime.failed_owner_count()
    try:
        with pytest.raises(OSError, match="retained close failure"):
            runtime.close()

        assert _is_open(target)
        assert target in retained._descriptors.values()
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)

        assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
        assert not _is_open(target)
        assert HermesDaemonRuntime.failed_owner_count() == baseline
        with RuntimeLease.acquire(home):
            pass
    finally:
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            with suppress(BaseException):
                HermesDaemonRuntime.retry_failed_cleanup(home)
        if _is_open(target):
            real_close(target)


def test_close_error_after_fd_reuse_never_closes_the_reused_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    retained = runtime.ledger._retained_files
    assert retained is not None
    target = tuple(retained._descriptors.values())[-1]
    real_close = os.close
    replacement: int | None = None
    injected = False

    def close_then_reuse(descriptor: int) -> None:
        nonlocal replacement, injected
        if descriptor == target and not injected:
            injected = True
            real_close(descriptor)
            replacement = os.open(os.devnull, os.O_RDONLY)
            assert replacement == descriptor
            raise OSError("injected post-close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", close_then_reuse)
    baseline = HermesDaemonRuntime.failed_owner_count()
    try:
        with pytest.raises(OSError, match="post-close failure"):
            runtime.close()

        assert replacement == target
        os.fstat(replacement)
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1

        assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
        os.fstat(replacement)
        assert HermesDaemonRuntime.failed_owner_count() == baseline
        with RuntimeLease.acquire(home):
            pass
    finally:
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            with suppress(BaseException):
                HermesDaemonRuntime.retry_failed_cleanup(home)
        if replacement is not None and _is_open(replacement):
            real_close(replacement)


def test_close_error_after_fd_was_consumed_is_terminal_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    retained = runtime.ledger._retained_files
    assert retained is not None
    target = tuple(retained._descriptors.values())[-1]
    real_close = os.close
    injected = False

    def close_then_error(descriptor: int) -> None:
        nonlocal injected
        if descriptor == target and not injected:
            injected = True
            real_close(descriptor)
            raise OSError("injected consumed-fd close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", close_then_error)
    baseline = HermesDaemonRuntime.failed_owner_count()
    try:
        with pytest.raises(OSError, match="consumed-fd close failure"):
            runtime.close()

        assert not _is_open(target)
        assert target not in retained._descriptors.values()
        assert retained.closed is True
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1

        assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
        assert not _is_open(target)
        assert HermesDaemonRuntime.failed_owner_count() == baseline
        with RuntimeLease.acquire(home):
            pass
    finally:
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            with suppress(BaseException):
                HermesDaemonRuntime.retry_failed_cleanup(home)


def test_unverifiable_fd_close_failure_keeps_lease_permanently_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    retained = runtime.ledger._retained_files
    assert retained is not None
    target_name, target = tuple(retained._descriptors.items())[-1]
    real_close = os.close
    real_fstat = os.fstat
    close_failed = False

    def fail_close(descriptor: int) -> None:
        nonlocal close_failed
        if descriptor == target:
            close_failed = True
            raise OSError("injected unproven close failure")
        real_close(descriptor)

    def fail_proof(descriptor: int):
        if descriptor == target and close_failed:
            raise OSError(errno.EIO, "injected unverifiable descriptor")
        return real_fstat(descriptor)

    baseline = HermesDaemonRuntime.failed_owner_count()
    try:
        with monkeypatch.context() as injected:
            injected.setattr(os, "close", fail_close)
            injected.setattr(os, "fstat", fail_proof)
            with pytest.raises(OSError, match="unproven close failure"):
                runtime.close()

            assert target_name in retained._unverifiable_descriptors
            assert retained.closed is False
            assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
            with pytest.raises(RuntimeOwnedError):
                RuntimeLease.acquire(home)
            with pytest.raises(PermissionError, match="unverifiable"):
                HermesDaemonRuntime.retry_failed_cleanup(home)

        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)
    finally:
        # There is deliberately no production recovery from an ambiguous
        # numeric descriptor. The test can prove this injected descriptor was
        # never consumed, then remove only its synthetic uncertainty marker.
        retained._unverifiable_descriptors.clear()
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            HermesDaemonRuntime.retry_failed_cleanup(home)


def test_internal_retain_verify_failure_registers_fd_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    real_adopt = RetainedSQLiteFiles.adopt_descriptor
    real_verify = RetainedSQLiteFiles.verify
    real_close = os.close
    target: int | None = None
    close_failures = 0
    verify_calls = 0

    def capture_adopt(self: RetainedSQLiteFiles, name: str, descriptor: int):
        nonlocal target
        metadata = real_adopt(self, name, descriptor)
        if name == "runtime.db-shm":
            target = descriptor
        return metadata

    def fail_first_verify(
        owner: RetainedSQLiteFiles,
        *,
        allow_missing_transient: bool = False,
    ) -> None:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 1:
            raise RuntimeError("injected internal verify primary")
        real_verify(
            owner,
            allow_missing_transient=allow_missing_transient,
        )

    def fail_target_twice(descriptor: int) -> None:
        nonlocal close_failures
        if descriptor == target and close_failures < 2:
            close_failures += 1
            raise OSError("injected internal retain close secondary")
        real_close(descriptor)

    monkeypatch.setattr(
        RetainedSQLiteFiles,
        "adopt_descriptor",
        capture_adopt,
    )
    monkeypatch.setattr(RetainedSQLiteFiles, "verify", fail_first_verify)
    monkeypatch.setattr(os, "close", fail_target_twice)
    baseline = HermesDaemonRuntime.failed_owner_count()
    try:
        with pytest.raises(RuntimeError, match="internal verify primary") as raised:
            HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

        assert isinstance(raised.value.__cause__, OSError)
        assert target is not None and _is_open(target)
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)

        assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
        assert not _is_open(target)
        with RuntimeLease.acquire(home):
            pass
    finally:
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            with suppress(BaseException):
                HermesDaemonRuntime.retry_failed_cleanup(home)
        if target is not None and _is_open(target):
            real_close(target)


def test_partial_retain_build_failure_registers_prior_fd_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    real_open = os.open
    real_close = os.close
    main_descriptor: int | None = None
    close_failures = 0

    def fail_journal(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal main_descriptor
        if path == "runtime.db-journal":
            raise OSError("injected journal open primary")
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "runtime.db":
            main_descriptor = descriptor
        return descriptor

    def fail_main_twice(descriptor: int) -> None:
        nonlocal close_failures
        if descriptor == main_descriptor and close_failures < 2:
            close_failures += 1
            raise OSError("injected partial retain close secondary")
        real_close(descriptor)

    monkeypatch.setattr(os, "open", fail_journal)
    monkeypatch.setattr(os, "close", fail_main_twice)
    baseline = HermesDaemonRuntime.failed_owner_count()
    try:
        with pytest.raises(
            PermissionError, match="SQLite file cannot be created"
        ) as raised:
            HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

        assert isinstance(raised.value.__cause__, OSError)
        assert main_descriptor is not None and _is_open(main_descriptor)
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)

        assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
        assert not _is_open(main_descriptor)
        with RuntimeLease.acquire(home):
            pass
    finally:
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            with suppress(BaseException):
                HermesDaemonRuntime.retry_failed_cleanup(home)
        if main_descriptor is not None and _is_open(main_descriptor):
            real_close(main_descriptor)


@pytest.mark.parametrize("after_assignment", [False, True])
def test_connection_adoption_failure_quarantines_exact_connection_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_assignment: bool,
) -> None:
    home = tmp_path / "runtime"
    _captured, original_descriptors = _capture_retained_files(monkeypatch)
    real_connect = sqlite3.connect
    real_adopt = RetainedSQLiteFiles.adopt_opening_connection
    proxy: _StartupPrimaryConnection | None = None

    def connect(*args, **kwargs) -> _StartupPrimaryConnection:
        nonlocal proxy
        proxy = _StartupPrimaryConnection(
            real_connect(*args, **kwargs),
            close_failures=2,
        )
        return proxy

    def fail_adoption(
        owner: RetainedSQLiteFiles,
        connection: _StartupPrimaryConnection,
    ) -> None:
        if after_assignment:
            real_adopt(owner, connection)
        raise RuntimeError("injected connection adoption primary")

    monkeypatch.setattr(store_module.sqlite3, "connect", connect)
    monkeypatch.setattr(
        RetainedSQLiteFiles,
        "adopt_opening_connection",
        fail_adoption,
    )
    baseline = HermesDaemonRuntime.failed_owner_count()
    try:
        with pytest.raises(RuntimeError, match="connection adoption primary") as raised:
            HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

        assert isinstance(raised.value.__cause__, OSError)
        assert proxy is not None and proxy.close_calls == 2
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
        assert all(_is_open(item) for item in original_descriptors[0])
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)

        assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
        assert proxy.close_calls == 3
        assert all(not _is_open(item) for item in original_descriptors[0])
        with RuntimeLease.acquire(home):
            pass
    finally:
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            with suppress(BaseException):
                HermesDaemonRuntime.retry_failed_cleanup(home)
        if proxy is not None:
            while proxy.close_calls <= 2:
                with suppress(OSError):
                    proxy.close()


@pytest.mark.parametrize("after_assignment", [False, True])
def test_descriptor_adoption_failure_quarantines_fd_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_assignment: bool,
) -> None:
    home = tmp_path / "runtime"
    real_adopt = RetainedSQLiteFiles.adopt_descriptor
    real_close = os.close
    target: int | None = None
    close_injected = False

    def fail_adoption(owner: RetainedSQLiteFiles, name: str, descriptor: int):
        nonlocal target
        target = descriptor
        if after_assignment:
            real_adopt(owner, name, descriptor)
        raise RuntimeError("injected descriptor adoption primary")

    def close_then_raise(descriptor: int) -> None:
        nonlocal close_injected
        if descriptor == target and not close_injected:
            close_injected = True
            real_close(descriptor)
            raise OSError("injected post-descriptor-close secondary")
        real_close(descriptor)

    monkeypatch.setattr(
        RetainedSQLiteFiles,
        "adopt_descriptor",
        fail_adoption,
    )
    monkeypatch.setattr(os, "close", close_then_raise)
    baseline = HermesDaemonRuntime.failed_owner_count()

    with pytest.raises(RuntimeError, match="descriptor adoption primary") as raised:
        HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

    assert isinstance(raised.value.__cause__, OSError)
    assert target is not None and not _is_open(target)
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    with RuntimeLease.acquire(home):
        pass


def test_lease_refuses_release_while_retained_sqlite_owner_is_live(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    retained = lease.retain_sqlite_files()
    try:
        with pytest.raises(PermissionError, match="must close"):
            lease.release()
        assert lease.released is False
    finally:
        retained.close()
        lease.release()

    with RuntimeLease.acquire(home):
        pass


def test_retain_registers_backstop_before_opening_any_sqlite_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    real_register = RuntimeLease._register_retained_files
    real_open = os.open
    sqlite_opens: list[str] = []

    def fail_registration(lease: RuntimeLease, owner) -> None:
        raise RuntimeError("injected provisional registration primary")

    def record_open(path, flags, mode=0o777, *, dir_fd=None):
        if isinstance(path, str) and path.startswith("runtime.db"):
            sqlite_opens.append(path)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        RuntimeLease,
        "_register_retained_files",
        fail_registration,
    )
    monkeypatch.setattr(os, "open", record_open)
    try:
        with pytest.raises(RuntimeError, match="provisional registration primary"):
            lease.retain_sqlite_files()
        assert sqlite_opens == []
    finally:
        monkeypatch.setattr(
            RuntimeLease,
            "_register_retained_files",
            real_register,
        )
        lease.release()

    with RuntimeLease.acquire(home):
        pass


def test_retain_release_race_keeps_lease_until_provisional_owner_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    real_open = os.open
    descriptor_opened = threading.Event()
    continue_retain = threading.Event()
    result: list[RetainedSQLiteFiles] = []
    failures: list[BaseException] = []

    def pause_first_sqlite_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "runtime.db" and not descriptor_opened.is_set():
            descriptor_opened.set()
            assert continue_retain.wait(timeout=5.0)
        return descriptor

    def retain() -> None:
        try:
            result.append(lease.retain_sqlite_files())
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(os, "open", pause_first_sqlite_open)
    worker = threading.Thread(target=retain)
    worker.start()
    assert descriptor_opened.wait(timeout=5.0)
    try:
        with pytest.raises(
            PermissionError, match="retained SQLite resources must close"
        ):
            lease.release()
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)
    finally:
        continue_retain.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        for owner in result:
            owner.close()
        if not lease.released:
            lease.release()

    assert failures == []
    assert len(result) == 1
    with RuntimeLease.acquire(home):
        pass


def test_release_terminal_fence_rejects_retain_during_native_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fcntl

    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    real_flock = fcntl.flock
    unlock_started = threading.Event()
    continue_release = threading.Event()
    release_failures: list[BaseException] = []
    unexpectedly_retained: list[RetainedSQLiteFiles] = []

    def pause_unlock(descriptor: int, operation: int) -> None:
        if descriptor == lease._descriptor and operation == fcntl.LOCK_UN:
            unlock_started.set()
            assert continue_release.wait(timeout=5.0)
        real_flock(descriptor, operation)

    def release() -> None:
        try:
            lease.release()
        except BaseException as error:
            release_failures.append(error)

    monkeypatch.setattr(fcntl, "flock", pause_unlock)
    worker = threading.Thread(target=release)
    worker.start()
    assert unlock_started.wait(timeout=5.0)
    try:
        with pytest.raises(PermissionError, match="release has started"):
            owner = lease.retain_sqlite_files()
            unexpectedly_retained.append(owner)
    finally:
        continue_release.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        for owner in unexpectedly_retained:
            owner.close()

    assert release_failures == []
    assert lease.released is True
    with RuntimeLease.acquire(home):
        pass


def test_retained_close_serializes_the_complete_descriptor_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    retained = lease.retain_sqlite_files()
    target = retained._descriptors["runtime.db-shm"]
    real_close = os.close
    first_close_entered = threading.Event()
    continue_first_close = threading.Event()
    second_close_done = threading.Event()
    failures: list[BaseException] = []
    first_thread: threading.Thread | None = None

    def pause_first_close(descriptor: int) -> None:
        if (
            descriptor == target
            and threading.current_thread() is first_thread
            and not first_close_entered.is_set()
        ):
            first_close_entered.set()
            assert continue_first_close.wait(timeout=5.0)
        real_close(descriptor)

    def close_first() -> None:
        try:
            retained.close()
        except BaseException as error:
            failures.append(error)

    def close_second() -> None:
        try:
            retained.close()
        except BaseException as error:
            failures.append(error)
        finally:
            second_close_done.set()

    monkeypatch.setattr(os, "close", pause_first_close)
    first_thread = threading.Thread(target=close_first)
    second_thread = threading.Thread(target=close_second)
    first_thread.start()
    assert first_close_entered.wait(timeout=5.0)
    second_thread.start()
    try:
        assert not second_close_done.wait(timeout=0.2)
        with pytest.raises(
            PermissionError, match="retained SQLite resources must close"
        ):
            lease.release()
    finally:
        continue_first_close.set()
        first_thread.join(timeout=5.0)
        second_thread.join(timeout=5.0)
        assert not first_thread.is_alive()
        assert not second_thread.is_alive()

    assert failures == []
    assert retained.closed is True
    lease.release()
    with RuntimeLease.acquire(home):
        pass


def test_retained_build_blocks_registered_cleanup_until_adoption_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    real_open = os.open
    first_descriptor_opened = threading.Event()
    continue_build = threading.Event()
    cleanup_done = threading.Event()
    retained_results: list[RetainedSQLiteFiles] = []
    raw_descriptors: list[int] = []
    failures: list[BaseException] = []

    def pause_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "runtime.db" and not first_descriptor_opened.is_set():
            raw_descriptors.append(descriptor)
            first_descriptor_opened.set()
            assert continue_build.wait(timeout=5.0)
        return descriptor

    def retain() -> None:
        try:
            retained_results.append(lease.retain_sqlite_files())
        except BaseException as error:
            failures.append(error)

    def cleanup() -> None:
        try:
            lease.close_registered_retained_files()
        except BaseException as error:
            failures.append(error)
        finally:
            cleanup_done.set()

    monkeypatch.setattr(os, "open", pause_open)
    build_thread = threading.Thread(target=retain)
    cleanup_thread = threading.Thread(target=cleanup)
    build_thread.start()
    assert first_descriptor_opened.wait(timeout=5.0)
    cleanup_thread.start()
    try:
        assert not cleanup_done.wait(timeout=0.2)
        with pytest.raises(
            PermissionError, match="retained SQLite resources must close"
        ):
            lease.release()
    finally:
        continue_build.set()
        build_thread.join(timeout=5.0)
        cleanup_thread.join(timeout=5.0)
        assert not build_thread.is_alive()
        assert not cleanup_thread.is_alive()

    assert failures == []
    assert len(retained_results) == 1
    assert retained_results[0].closed is True
    assert raw_descriptors and not _is_open(raw_descriptors[0])
    lease.release()
    with RuntimeLease.acquire(home):
        pass


def test_ledger_atomically_replaces_retained_owner_for_complete_lifetime(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    retained = lease.retain_sqlite_files()
    ledger = store_module.SQLiteLedger.open_retained(retained)

    with lease._retained_files_lock:
        assert lease._retained_files == {ledger}

    retained.close()
    assert ledger._connection.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(PermissionError, match="retained SQLite resources must close"):
        lease.release()

    ledger.close()
    assert ledger.closed is True
    lease.release()
    with RuntimeLease.acquire(home):
        pass


def test_registered_cleanup_closes_transferred_ledger_before_lease_release(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    retained = lease.retain_sqlite_files()
    ledger = store_module.SQLiteLedger.open_retained(retained)

    lease.close_registered_retained_files()

    assert ledger.closed is True
    with pytest.raises(sqlite3.ProgrammingError):
        ledger._connection.execute("SELECT 1")
    lease.release()
    with RuntimeLease.acquire(home):
        pass


def test_connection_handoff_blocks_cleanup_until_registry_transfer_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    retained = lease.retain_sqlite_files()
    real_connect = sqlite3.connect
    connection_created = threading.Event()
    continue_handoff = threading.Event()
    cleanup_done = threading.Event()
    ledgers: list[store_module.SQLiteLedger] = []
    failures: list[BaseException] = []

    def pause_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection_created.set()
        assert continue_handoff.wait(timeout=5.0)
        return connection

    def open_ledger() -> None:
        try:
            ledgers.append(store_module.SQLiteLedger.open_retained(retained))
        except BaseException as error:
            failures.append(error)

    def cleanup() -> None:
        try:
            lease.close_registered_retained_files()
        except BaseException as error:
            failures.append(error)
        finally:
            cleanup_done.set()

    monkeypatch.setattr(store_module.sqlite3, "connect", pause_connect)
    open_thread = threading.Thread(target=open_ledger)
    cleanup_thread = threading.Thread(target=cleanup)
    open_thread.start()
    assert connection_created.wait(timeout=5.0)
    cleanup_thread.start()
    try:
        assert not cleanup_done.wait(timeout=0.2)
        with pytest.raises(
            PermissionError, match="retained SQLite resources must close"
        ):
            lease.release()
    finally:
        continue_handoff.set()
        open_thread.join(timeout=5.0)
        cleanup_thread.join(timeout=5.0)
        assert not open_thread.is_alive()
        assert not cleanup_thread.is_alive()

    assert failures == []
    assert len(ledgers) == 1 and ledgers[0].closed is True
    lease.release()
    with RuntimeLease.acquire(home):
        pass


def test_factory_ledger_is_adopted_before_retained_validation_can_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    real_verify = RetainedSQLiteFiles.verify
    captured: list[store_module.SQLiteLedger] = []
    verify_calls = 0

    def factory(path: Path) -> store_module.SQLiteLedger:
        ledger = store_module.SQLiteLedger.open(path)
        captured.append(ledger)
        return ledger

    def fail_factory_adoption_verify(
        owner: RetainedSQLiteFiles,
        *,
        allow_missing_transient: bool = False,
    ) -> None:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise RuntimeError("injected factory adoption primary")
        real_verify(
            owner,
            allow_missing_transient=allow_missing_transient,
        )

    monkeypatch.setattr(
        RetainedSQLiteFiles,
        "verify",
        fail_factory_adoption_verify,
    )
    baseline = HermesDaemonRuntime.failed_owner_count()

    with pytest.raises(RuntimeError, match="factory adoption primary"):
        HermesDaemonRuntime.open(
            home,
            ledger_factory=factory,
            scheduler_interval_seconds=60.0,
        )

    assert len(captured) == 1
    assert captured[0].closed is True
    with pytest.raises(sqlite3.ProgrammingError):
        captured[0]._connection.execute("SELECT 1")
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    with RuntimeLease.acquire(home):
        pass


def test_factory_registry_transfer_failure_keeps_ledger_in_startup_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    captured: list[store_module.SQLiteLedger] = []

    def factory(path: Path) -> store_module.SQLiteLedger:
        ledger = store_module.SQLiteLedger.open(path)
        captured.append(ledger)
        return ledger

    def fail_transfer(
        _lease: RuntimeLease,
        _old_owner,
        _new_owner,
    ) -> None:
        raise RuntimeError("injected factory registry transfer primary")

    monkeypatch.setattr(
        RuntimeLease,
        "_replace_retained_files",
        fail_transfer,
    )
    baseline = HermesDaemonRuntime.failed_owner_count()

    with pytest.raises(RuntimeError, match="registry transfer primary"):
        HermesDaemonRuntime.open(
            home,
            ledger_factory=factory,
            scheduler_interval_seconds=60.0,
        )

    assert len(captured) == 1
    assert captured[0].closed is True
    with pytest.raises(sqlite3.ProgrammingError):
        captured[0]._connection.execute("SELECT 1")
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    with RuntimeLease.acquire(home):
        pass


@pytest.mark.parametrize("after_assignment", [False, True])
@pytest.mark.parametrize("close_then_error", [False, True])
def test_factory_ledger_adoption_failure_quarantines_exact_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_assignment: bool,
    close_then_error: bool,
) -> None:
    home = tmp_path / "runtime"
    captured: list[store_module.SQLiteLedger] = []
    real_adopt = _PreRuntimeOwner.adopt_ledger

    def factory(path: Path) -> store_module.SQLiteLedger:
        ledger = store_module.SQLiteLedger.open(path)
        captured.append(ledger)
        if close_then_error:
            real_close = ledger.close

            def close_with_terminal_error() -> None:
                real_close()
                raise OSError("injected post-ledger-close secondary")

            monkeypatch.setattr(ledger, "close", close_with_terminal_error)
        return ledger

    def fail_adoption(
        owner: _PreRuntimeOwner,
        ledger: store_module.SQLiteLedger,
    ) -> None:
        if after_assignment:
            real_adopt(owner, ledger)
        raise RuntimeError("injected factory ledger adoption primary")

    monkeypatch.setattr(_PreRuntimeOwner, "adopt_ledger", fail_adoption)
    baseline = HermesDaemonRuntime.failed_owner_count()
    try:
        with pytest.raises(RuntimeError, match="ledger adoption primary") as raised:
            HermesDaemonRuntime.open(
                home,
                ledger_factory=factory,
                scheduler_interval_seconds=60.0,
            )

        assert len(captured) == 1 and captured[0].closed is True
        with pytest.raises(sqlite3.ProgrammingError):
            captured[0]._connection.execute("SELECT 1")
        if close_then_error:
            assert isinstance(raised.value.__cause__, OSError)
            assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
            with pytest.raises(RuntimeOwnedError):
                RuntimeLease.acquire(home)
            assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
        assert HermesDaemonRuntime.failed_owner_count() == baseline
        with RuntimeLease.acquire(home):
            pass
    finally:
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            with suppress(BaseException):
                HermesDaemonRuntime.retry_failed_cleanup(home)


def test_closed_retained_owner_retries_transient_registry_discard_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    retained = lease.retain_sqlite_files()
    real_discard = lease._discard_retained_files
    discard_calls = 0

    def fail_once(owner) -> None:
        nonlocal discard_calls
        if owner is retained and discard_calls == 0:
            discard_calls += 1
            raise OSError("injected retained registry discard failure")
        real_discard(owner)

    monkeypatch.setattr(lease, "_discard_retained_files", fail_once)
    with pytest.raises(OSError, match="retained registry discard failure"):
        retained.close()

    assert retained.closed is True
    with pytest.raises(PermissionError, match="retained SQLite resources must close"):
        lease.release()

    retained.close()
    assert discard_calls == 1
    lease.release()
    with RuntimeLease.acquire(home):
        pass


def test_closed_ledger_retries_transient_registry_discard_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    lease = RuntimeLease.acquire(home)
    retained = lease.retain_sqlite_files()
    ledger = store_module.SQLiteLedger.open_retained(retained)
    real_discard = lease._discard_retained_files
    discard_calls = 0

    def fail_once(owner) -> None:
        nonlocal discard_calls
        if owner is ledger and discard_calls == 0:
            discard_calls += 1
            raise OSError("injected ledger registry discard failure")
        real_discard(owner)

    monkeypatch.setattr(lease, "_discard_retained_files", fail_once)
    with pytest.raises(OSError, match="ledger registry discard failure"):
        ledger.close()

    assert ledger.closed is True
    with pytest.raises(PermissionError, match="retained SQLite resources must close"):
        lease.release()

    ledger.close()
    assert discard_calls == 1
    lease.release()
    with RuntimeLease.acquire(home):
        pass
