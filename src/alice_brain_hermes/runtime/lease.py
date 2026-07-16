"""Portable two-lock ownership for one Hermes runtime home."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import weakref
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Protocol, Self

import portalocker
from platformdirs import PlatformDirs

from alice_brain_hermes.errors import RuntimeOwnedError
from alice_brain_hermes.runtime.process_marker import current_process_marker

_LOCK_NAME = "daemon.lock"
_LAUNCH_NONCE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class _RetainedResource(Protocol):
    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...


def _normalized_runtime_home(path: str | Path) -> Path:
    requested = Path(path).expanduser().absolute()
    return requested.resolve(strict=False)


def _guard_key(path: str | Path) -> str:
    normalized = os.path.normcase(os.fspath(_normalized_runtime_home(path)))
    return hashlib.sha256(os.fsencode(normalized)).hexdigest()


def _external_guard_path(path: str | Path) -> Path:
    root = (
        Path(PlatformDirs("alice-brain-hermes", appauthor=False).user_data_path)
        / "locks"
    )
    return root / f"{_guard_key(path)}.lock"


def _prepare_home(path: str | Path) -> Path:
    requested = Path(path).expanduser().absolute()
    if requested.is_symlink():
        raise PermissionError("runtime home cannot be a symbolic link")
    requested.mkdir(parents=True, mode=0o700, exist_ok=True)
    if requested.is_symlink() or not requested.is_dir():
        raise PermissionError("runtime home must be a real directory")
    home = requested.resolve(strict=True)
    if os.path.normcase(os.fspath(home)) != os.path.normcase(
        os.fspath(requested.resolve(strict=False))
    ):
        raise PermissionError("runtime home path is not stable")
    try:
        home.chmod(0o700)
    except OSError as error:
        raise PermissionError("runtime home could not be prepared") from error
    return home


class _PortableLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.owner = portalocker.Lock(
            path,
            mode="a+b",
            timeout=0,
            fail_when_locked=True,
        )
        self.stream = None
        self.released = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.stream = self.owner.acquire()
        except portalocker.exceptions.AlreadyLocked as error:
            raise RuntimeOwnedError("runtime is already owned") from error
        except portalocker.exceptions.LockException as error:
            raise PermissionError(
                "runtime ownership lock could not be acquired"
            ) from error
        self.released = False

    def write_record(self, record: dict[str, object]) -> None:
        if self.stream is None:
            raise PermissionError("runtime ownership lock is not held")
        payload = json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.stream.seek(0)
        self.stream.truncate(0)
        self.stream.write(payload)
        self.stream.flush()

    def assert_held(self) -> None:
        if self.stream is None or self.stream.closed:
            raise PermissionError("runtime ownership lock is no longer held")
        if self.path.is_symlink() or not self.path.is_file():
            raise PermissionError("runtime ownership lock path disappeared")
        probe = portalocker.Lock(
            self.path,
            mode="a+b",
            timeout=0,
            fail_when_locked=True,
        )
        try:
            probe.acquire()
        except portalocker.exceptions.AlreadyLocked:
            return
        except portalocker.exceptions.LockException as error:
            raise PermissionError("runtime ownership lock is not verifiable") from error
        else:
            probe.release()
            raise PermissionError("runtime ownership lock identity changed")

    def release(self) -> None:
        if self.released:
            return
        if self.stream is None:
            raise PermissionError("runtime ownership release state is uncertain")
        self.owner.release()
        self.stream = None
        self.released = True


class _FailedAcquireOwner:
    def __init__(self, locks: tuple[_PortableLock, ...]) -> None:
        self._locks = list(locks)

    def cleanup(self) -> None:
        for lock in reversed(self._locks):
            lock.release()
        self._locks.clear()


class RuntimeLease:
    """External and legacy portalocker guards held for the runtime lifetime."""

    _failed_acquire_lock: ClassVar[threading.Lock] = threading.Lock()
    _failed_acquires: ClassVar[dict[Path, _FailedAcquireOwner]] = {}
    _live_lease_lock: ClassVar[threading.Lock] = threading.Lock()
    _live_leases: ClassVar[weakref.WeakSet[RuntimeLease]] = weakref.WeakSet()

    def __init__(
        self,
        *,
        runtime_home: Path,
        runtime_home_identity: os.stat_result,
        external_lock: _PortableLock,
        legacy_lock: _PortableLock,
        instance_nonce: str,
        launch_nonce: str,
        process_marker: str,
    ) -> None:
        self.runtime_home = runtime_home
        self._runtime_home_identity = runtime_home_identity
        self.path = legacy_lock.path
        self.external_guard_path = external_lock.path
        self._external_lock = external_lock
        self._legacy_lock = legacy_lock
        self._creator_pid = os.getpid()
        self.instance_nonce = instance_nonce
        self.launch_nonce = launch_nonce
        self.process_marker = process_marker
        self._resources_lock = threading.Lock()
        self._resources: set[_RetainedResource] = set()
        self._release_lock = threading.Lock()
        self._release_started = False
        self._released = False

    @classmethod
    def acquire(
        cls,
        runtime_home: str | Path,
        *,
        launch_nonce: str | None = None,
    ) -> Self:
        normalized = _normalized_runtime_home(runtime_home)
        cls.retry_failed_acquire_cleanup(normalized)
        if launch_nonce is None:
            launch_nonce = secrets.token_hex(32)
        if (
            not isinstance(launch_nonce, str)
            or _LAUNCH_NONCE.fullmatch(launch_nonce) is None
        ):
            raise ValueError("launch nonce is invalid")

        external = _PortableLock(_external_guard_path(normalized))
        legacy: _PortableLock | None = None
        acquired: list[_PortableLock] = []
        try:
            external.acquire()
            acquired.append(external)
            home = _prepare_home(runtime_home)
            home_identity = home.stat(follow_symlinks=False)
            legacy = _PortableLock(home / _LOCK_NAME)
            legacy.acquire()
            acquired.append(legacy)
            instance_nonce = secrets.token_hex(32)
            process_marker = current_process_marker()
            record = {
                "instance_nonce": instance_nonce,
                "launch_nonce": launch_nonce,
                "pid": os.getpid(),
                "process_marker": process_marker,
            }
            external.write_record(record)
            legacy.write_record(record)
            lease = cls(
                runtime_home=home,
                runtime_home_identity=home_identity,
                external_lock=external,
                legacy_lock=legacy,
                instance_nonce=instance_nonce,
                launch_nonce=launch_nonce,
                process_marker=process_marker,
            )
            lease.assert_authority()
            with cls._live_lease_lock:
                cls._live_leases.add(lease)
            return lease
        except BaseException as primary_error:
            owner = _FailedAcquireOwner(tuple(acquired))
            try:
                owner.cleanup()
            except BaseException:
                with cls._failed_acquire_lock:
                    cls._failed_acquires[normalized] = owner
                raise RuntimeOwnedError(
                    "failed lease acquisition cleanup remains incomplete"
                ) from primary_error
            raise

    @classmethod
    def retry_failed_acquire_cleanup(cls, runtime_home: str | Path) -> bool:
        home = _normalized_runtime_home(runtime_home)
        with cls._failed_acquire_lock:
            owner = cls._failed_acquires.get(home)
            if owner is None:
                return False
            try:
                owner.cleanup()
            except BaseException as error:
                raise RuntimeOwnedError(
                    "failed lease acquisition cleanup remains incomplete"
                ) from error
            del cls._failed_acquires[home]
            return True

    @classmethod
    def failed_acquire_count(cls) -> int:
        with cls._failed_acquire_lock:
            return len(cls._failed_acquires)

    def assert_creator_process(self) -> None:
        if os.getpid() != self._creator_pid:
            raise PermissionError("runtime lease belongs to another process")

    @property
    def released(self) -> bool:
        return self._released

    def assert_authority(self) -> Path:
        self.assert_creator_process()
        if self._released or self._release_started:
            raise PermissionError("runtime lease authority has been released")
        try:
            current_home_identity = self.runtime_home.stat(follow_symlinks=False)
        except OSError as error:
            raise PermissionError("runtime home identity changed") from error
        if not os.path.samestat(
            self._runtime_home_identity,
            current_home_identity,
        ):
            raise PermissionError("runtime home identity changed")
        self._external_lock.assert_held()
        self._legacy_lock.assert_held()
        return self.runtime_home

    @staticmethod
    def _relative_name(name: str) -> str:
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise ValueError("runtime-home file name must be one path component")
        return name

    def home_path(self, name: str) -> Path:
        name = self._relative_name(name)
        return self.assert_authority() / name

    def stat_home_file(self, name: str, *, follow_symlinks: bool = False):
        return self.home_path(name).stat(follow_symlinks=follow_symlinks)

    def unlink_home_file(self, name: str, *, missing_ok: bool = False) -> None:
        path = self.home_path(name)
        self.assert_authority()
        path.unlink(missing_ok=missing_ok)
        self.assert_authority()

    def replace_home_file(self, source: str, destination: str) -> None:
        source_path = self.home_path(source)
        destination_path = self.home_path(destination)
        self.assert_authority()
        os.replace(source_path, destination_path)
        self.assert_authority()

    def chmod_home_file(self, name: str, mode: int) -> None:
        path = self.home_path(name)
        self.assert_authority()
        try:
            before = path.stat(follow_symlinks=False)
        except OSError as error:
            raise PermissionError(
                "runtime-home file identity is unavailable"
            ) from error
        if path.is_symlink() or not path.is_file():
            raise PermissionError("runtime-home file must be a regular file")
        try:
            path.chmod(mode, follow_symlinks=False)
        except NotImplementedError:
            current = path.stat(follow_symlinks=False)
            if (
                path.is_symlink()
                or not path.is_file()
                or not os.path.samestat(before, current)
            ):
                raise PermissionError(
                    "runtime-home file identity changed"
                ) from None
            # A path-based fallback could follow a link introduced after the
            # identity check. Retain the already-private parent protection
            # when the host cannot chmod without following links.
            self.assert_authority()
            return
        after = path.stat(follow_symlinks=False)
        if (
            path.is_symlink()
            or not path.is_file()
            or not os.path.samestat(before, after)
        ):
            raise PermissionError("runtime-home file identity changed")
        self.assert_authority()

    def fsync_home(self) -> None:
        """Portable authority barrier after a flushed file or atomic replace."""
        self.assert_authority()

    def register_resource(self, owner: _RetainedResource) -> None:
        self.assert_creator_process()
        with self._resources_lock:
            if self._release_started or self._released:
                raise PermissionError("runtime lease release has started")
            if owner in self._resources:
                raise RuntimeError("runtime resource is already registered")
            self._resources.add(owner)

    def discard_resource(self, owner: _RetainedResource) -> None:
        self.assert_creator_process()
        with self._resources_lock:
            self._resources.discard(owner)

    def close_registered_resources(self) -> None:
        self.assert_creator_process()
        while True:
            with self._resources_lock:
                owners = tuple(self._resources)
            if not owners:
                return
            first_error: BaseException | None = None
            for owner in owners:
                try:
                    owner.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
            with self._resources_lock:
                if self._resources == set(owners):
                    raise RuntimeError("runtime resource cleanup made no progress")

    def release(self) -> None:
        self.assert_creator_process()
        with self._release_lock:
            if self._released:
                return
            with self._resources_lock:
                if self._resources:
                    raise PermissionError(
                        "runtime resources must close before lease release"
                    )
                self._release_started = True
            self._legacy_lock.release()
            self._external_lock.release()
            self._released = self._legacy_lock.released and self._external_lock.released
            if self._released:
                with type(self)._live_lease_lock:
                    type(self)._live_leases.discard(self)

    def __enter__(self) -> Self:
        self.assert_authority()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.release()


__all__ = ["RuntimeLease"]
