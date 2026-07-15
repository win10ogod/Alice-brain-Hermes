"""Held process lease and private runtime-home validation."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import socket
import stat
import sys
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Protocol, Self

from alice_brain_hermes.errors import RuntimeOwnedError
from alice_brain_hermes.runtime.process_marker import current_process_marker

_LOCK_NAME = "daemon.lock"
_PATH_GUARD_PREFIX = b"\0alice-brain-hermes:"

_RELIABLE_LOCAL_FILESYSTEMS = frozenset(
    {
        "apfs",
        "bcachefs",
        "btrfs",
        "ext2",
        "ext3",
        "ext4",
        "f2fs",
        "hfs",
        "hfsplus",
        "jfs",
        "nilfs2",
        "ntfs3",
        "overlay",
        "ramfs",
        "reiserfs",
        "rootfs",
        "tmpfs",
        "ubifs",
        "ufs",
        "xfs",
        "zfs",
    }
)


def _unescape_mountinfo_field(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _filesystem_type_for_descriptor(descriptor: int) -> str | None:
    """Return the retained directory FD's filesystem type, if provable."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        lines = (
            Path("/proc/self/mountinfo")
            .read_text(encoding="utf-8", errors="strict")
            .splitlines()
        )
    except (OSError, UnicodeError):
        return None
    metadata = os.fstat(descriptor)
    device = f"{os.major(metadata.st_dev)}:{os.minor(metadata.st_dev)}"
    matches: set[str] = set()
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            filesystem_type = fields[separator + 1]
        except (IndexError, ValueError):
            return None
        if fields[2] == device:
            matches.add(filesystem_type)
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _require_reliable_local_filesystem(descriptor: int) -> None:
    filesystem_type = _filesystem_type_for_descriptor(descriptor)
    if filesystem_type not in _RELIABLE_LOCAL_FILESYSTEMS:
        raise PermissionError(
            "runtime home filesystem is not proven local and reliable"
        )
    try:
        readonly = bool(os.fstatvfs(descriptor).f_flag & os.ST_RDONLY)
    except (AttributeError, OSError) as error:
        raise PermissionError(
            "runtime home filesystem properties are not verifiable"
        ) from error
    if readonly:
        raise PermissionError("runtime home filesystem must be writable")


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _open_child_directory(parent_descriptor: int, component: str) -> int:
    try:
        return os.open(
            component,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        try:
            metadata = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError:
            metadata = None
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise PermissionError(
                "runtime home cannot contain symlink traversal"
            ) from error
        raise


def _open_existing_directory_nofollow(path: Path) -> int:
    absolute = path.expanduser().absolute()
    descriptor = os.open(absolute.anchor, _directory_open_flags())
    try:
        for component in absolute.parts[1:]:
            child = _open_child_directory(descriptor, component)
            previous = descriptor
            descriptor = child
            # Transfer ownership before the single close attempt.  If close
            # consumed the old descriptor and then raised, outer cleanup owns
            # only the child and cannot close a reused numeric descriptor.
            os.close(previous)
        return descriptor
    except BaseException:
        current = descriptor
        descriptor = -1
        with suppress(BaseException):
            os.close(current)
        raise


@dataclass(frozen=True, slots=True)
class _VerifiedHome:
    path: Path
    descriptor: int


class _FailedAcquireOwner:
    """Retain an acquired lock until explicit unlock is proven."""

    def __init__(
        self,
        *,
        runtime_home: Path,
        descriptor: int,
        home_descriptor: int,
        path_guard: socket.socket,
    ) -> None:
        self.runtime_home = runtime_home
        self._descriptor = descriptor
        self._home_descriptor = home_descriptor
        self._path_guard: socket.socket | None = path_guard
        self._creator_pid = os.getpid()
        self.lock_unlocked = False
        self._lock_closed = False
        self._home_closed = False
        self._path_guard_closed = False

    def _assert_creator_process(self) -> None:
        if os.getpid() != self._creator_pid:
            raise PermissionError(
                "failed lease-acquisition owner belongs to another process"
            )

    def _abandon_after_fork_child(self) -> None:
        """Close inherited child copies without explicitly unlocking flock."""
        guard = self._path_guard
        self._path_guard = None
        if guard is not None:
            with suppress(BaseException):
                guard.close()
        for attribute in ("_home_descriptor", "_descriptor"):
            descriptor = getattr(self, attribute)
            setattr(self, attribute, -1)
            if descriptor >= 0:
                with suppress(BaseException):
                    os.close(descriptor)

    def cleanup(self) -> None:
        self._assert_creator_process()
        if not self.lock_unlocked:
            import fcntl

            # An unlock error leaves every retained descriptor untouched so a
            # later explicit retry still owns the exact native resources.
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            self.lock_unlocked = True

        first_error: BaseException | None = None
        if not self._home_closed:
            descriptor = self._home_descriptor
            self._home_descriptor = -1
            try:
                os.close(descriptor)
            except BaseException as error:
                first_error = error
            finally:
                # After unlock, a close result is terminal. Retrying a numeric
                # descriptor could close an unrelated descriptor after reuse.
                self._home_closed = True
        if not self._lock_closed:
            descriptor = self._descriptor
            self._descriptor = -1
            try:
                os.close(descriptor)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                self._lock_closed = True
        if not self._path_guard_closed:
            guard = self._path_guard
            self._path_guard = None
            try:
                if guard is not None:
                    guard.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                self._path_guard_closed = True
        if first_error is not None:
            raise first_error


class _CloseableConnection(Protocol):
    def close(self) -> None: ...


class _RetainedResource(Protocol):
    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...


class RetainedSQLiteFiles:
    """Pinned private main/WAL/SHM leaves for one SQLite connection."""

    def __init__(
        self,
        lease: RuntimeLease,
        basename: str,
        descriptors: dict[str, int],
        identities: dict[str, tuple[int, int]],
    ) -> None:
        self._lease = lease
        self.basename = basename
        self._descriptors = descriptors
        self._identities = identities
        self._transient_names = frozenset({f"{basename}-journal"})
        self.logical_path = lease.runtime_home / basename
        self._opening_connection: _CloseableConnection | None = None
        self._unverifiable_descriptors: set[str] = set()
        self._close_lock = threading.RLock()
        self._closed = False

    def _reset_close_lock_after_fork_child(self) -> None:
        self._close_lock = threading.RLock()

    @contextmanager
    def startup_operation(self) -> Iterator[None]:
        """Fence connection construction and transfer against backstop close."""
        self._lease.assert_creator_process()
        with self._close_lock:
            if self._closed:
                raise RuntimeError("retained SQLite owner is already closed")
            yield

    @property
    def connection_path(self) -> Path:
        """Address the already-retained main descriptor through procfs."""
        try:
            descriptor = self._descriptors[self.basename]
        except KeyError:
            raise RuntimeError(
                "retained SQLite main descriptor is not available"
            ) from None
        return Path(f"/proc/self/fd/{descriptor}")

    def adopt_descriptor(self, name: str, descriptor: int):
        """Take ownership before inspecting a newly opened descriptor."""
        self._lease.assert_creator_process()
        with self._close_lock:
            if self._closed or name in self._descriptors:
                raise RuntimeError("retained SQLite descriptor ownership is invalid")
            self._descriptors[name] = descriptor
            metadata = os.fstat(descriptor)
            self._identities[name] = (metadata.st_dev, metadata.st_ino)
            return metadata

    def quarantine_unadopted_descriptor(self, name: str, descriptor: int) -> None:
        """Backstop an FD when normal adoption itself was interrupted."""
        self._lease.assert_creator_process()
        with self._close_lock:
            if self._closed:
                raise RuntimeError("retained SQLite owner is already closed")
            current = self._descriptors.get(name)
            if current is None:
                # Deliberately leave the identity absent. Adoption may have failed
                # after consuming/reusing the number, so cleanup gets one close
                # attempt but may never treat a still-valid number as exact.
                self._descriptors[name] = descriptor
            elif current != descriptor:
                raise RuntimeError("a different SQLite descriptor is already retained")

    @property
    def main_identity(self) -> tuple[int, int]:
        return self._identities[self.basename]

    @property
    def closed(self) -> bool:
        self._lease.assert_creator_process()
        return self._closed

    def adopt_opening_connection(self, connection: _CloseableConnection) -> None:
        """Retain a just-opened connection until its ledger owner exists."""
        self._lease.assert_creator_process()
        with self._close_lock:
            if self._closed or self._opening_connection is not None:
                raise RuntimeError("retained SQLite startup ownership is invalid")
            self._opening_connection = connection

    def quarantine_unadopted_connection(self, connection: _CloseableConnection) -> None:
        """Backstop a connection when normal adoption itself was interrupted."""
        self._lease.assert_creator_process()
        with self._close_lock:
            if self._closed:
                raise RuntimeError("retained SQLite owner is already closed")
            if self._opening_connection is None:
                self._opening_connection = connection
            elif self._opening_connection is not connection:
                raise RuntimeError("a different SQLite connection is already retained")

    def transfer_opening_connection(self, connection: _CloseableConnection) -> None:
        """Transfer the exact connection after the ledger becomes registry owner."""
        self._lease.assert_creator_process()
        with self._close_lock:
            if self._closed or self._opening_connection is not connection:
                raise RuntimeError("SQLite connection ownership does not match")
            self._opening_connection = None

    def confirm_connection_closed(self, connection: _CloseableConnection) -> None:
        """Drop the backstop only after the ledger proves this connection closed."""
        self._lease.assert_creator_process()
        with self._close_lock:
            if self._opening_connection is connection:
                self._opening_connection = None
            elif self._opening_connection is not None:
                raise RuntimeError("SQLite connection ownership does not match")

    def assert_authority(self) -> None:
        """Require the creating process and unchanged configured home path."""
        self._lease.assert_authority()

    def verify(self, *, allow_missing_transient: bool = False) -> None:
        if self._closed:
            raise PermissionError("retained SQLite files are closed")
        self._lease.assert_authority()
        for name, descriptor in self._descriptors.items():
            retained = os.fstat(descriptor)
            try:
                expected = self._identities[name]
            except KeyError:
                raise PermissionError(
                    "retained SQLite descriptor identity is unverified"
                ) from None
            if (
                (retained.st_dev, retained.st_ino) != expected
                or not stat.S_ISREG(retained.st_mode)
                or retained.st_uid != os.getuid()
                or stat.S_IMODE(retained.st_mode) != 0o600
            ):
                raise PermissionError("retained SQLite file identity changed")
            try:
                current = self._lease.stat_home_file(name, follow_symlinks=False)
            except FileNotFoundError:
                if (
                    allow_missing_transient
                    and name in self._transient_names
                    and retained.st_nlink == 0
                ):
                    continue
                raise PermissionError("retained SQLite file path disappeared") from None
            if (
                retained.st_nlink != 1
                or (current.st_dev, current.st_ino) != expected
                or not stat.S_ISREG(current.st_mode)
                or current.st_uid != os.getuid()
                or current.st_nlink != 1
                or stat.S_IMODE(current.st_mode) != 0o600
            ):
                raise PermissionError("retained SQLite file identity changed")

    def verify_connection_path(self, path: str) -> None:
        self._lease.assert_authority()
        if type(path) is not str or not path:
            raise PermissionError("SQLite connection path is not verifiable")
        if os.path.normpath(path) != os.path.normpath(os.fspath(self.logical_path)):
            raise PermissionError(
                "SQLite connection did not retain the expected main file"
            )
        metadata = os.stat(path, follow_symlinks=False)
        if (
            (metadata.st_dev, metadata.st_ino) != self.main_identity
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise PermissionError("SQLite connection main-file identity changed")

    def close(self) -> None:
        self._lease.assert_creator_process()
        with self._close_lock:
            self._close_serialized()

    def _close_serialized(self) -> None:
        if self._closed:
            self._lease._discard_retained_files(self)
            return
        if self._unverifiable_descriptors:
            raise PermissionError(
                "retained SQLite descriptor close remains unverifiable"
            )
        first_error: BaseException | None = None
        if self._opening_connection is not None:
            connection = self._opening_connection
            try:
                connection.close()
            except BaseException as error:
                first_error = error
            else:
                self._opening_connection = None
        if first_error is not None:
            raise first_error
        for name in reversed(tuple(self._descriptors)):
            descriptor = self._descriptors[name]
            expected = self._identities.get(name)
            try:
                os.close(descriptor)
            except BaseException as error:
                if first_error is None:
                    first_error = error
                try:
                    current = os.fstat(descriptor)
                except OSError as verification_error:
                    if verification_error.errno == errno.EBADF:
                        del self._descriptors[name]
                        self._identities.pop(name, None)
                    else:
                        self._unverifiable_descriptors.add(name)
                else:
                    if expected is None:
                        # With no pre-close identity, a still-valid numeric FD
                        # cannot be distinguished from immediate descriptor
                        # reuse. Preserve the lease permanently fail-stop.
                        self._unverifiable_descriptors.add(name)
                    elif (current.st_dev, current.st_ino) != expected:
                        # The original descriptor was consumed and its numeric
                        # value has already been reused. Never close it again.
                        del self._descriptors[name]
                        self._identities.pop(name, None)
                    # Exact identity means the original descriptor is still
                    # owned and remains in the mapping for an explicit retry.
            else:
                del self._descriptors[name]
                self._identities.pop(name, None)
        self._closed = (
            self._opening_connection is None
            and not self._descriptors
            and not self._unverifiable_descriptors
        )
        if self._closed:
            self._lease._discard_retained_files(self)
        if first_error is not None:
            raise first_error


def _validate_private_home(path: Path) -> _VerifiedHome:
    """Create and validate a user-owned home without following any symlink."""
    absolute = path.expanduser().absolute()
    if os.name == "nt":
        raise PermissionError(
            "Windows runtime-home DACL verification is unavailable; refusing"
        )
    if any(component in {".", ".."} for component in absolute.parts[1:]):
        raise PermissionError("runtime home traversal components are forbidden")
    anchor = absolute.anchor
    if not anchor:
        raise PermissionError("runtime home must be absolute")
    descriptor = os.open(anchor, _directory_open_flags())
    current = Path(anchor)
    try:
        for component in absolute.parts[1:]:
            created = False
            raced = False
            try:
                child = _open_child_directory(descriptor, component)
            except FileNotFoundError:
                _require_reliable_local_filesystem(descriptor)
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    raced = True
                if created:
                    created_metadata = os.stat(
                        component,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISDIR(created_metadata.st_mode)
                        or created_metadata.st_uid != os.getuid()
                    ):
                        raise PermissionError(
                            "new runtime home component identity changed"
                        ) from None
                    os.chmod(
                        component,
                        0o700,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                child = _open_child_directory(descriptor, component)
                if created and (
                    os.fstat(child).st_dev,
                    os.fstat(child).st_ino,
                ) != (created_metadata.st_dev, created_metadata.st_ino):
                    with suppress(BaseException):
                        os.close(child)
                    raise PermissionError(
                        "new runtime home component identity changed"
                    ) from None
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                with suppress(BaseException):
                    os.close(child)
                raise PermissionError("runtime home must be a real directory")
            if raced and (
                metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                with suppress(BaseException):
                    os.close(child)
                raise PermissionError(
                    "raced runtime home component must be user-owned mode 0700"
                )
            previous = descriptor
            descriptor = child
            # As above, transfer ownership first and never retry the old
            # numeric descriptor after an uncertain close result.
            os.close(previous)
            current /= component
        _require_reliable_local_filesystem(descriptor)
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid():
            raise PermissionError("runtime home must be owned by the current user")
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise PermissionError("runtime home mode must be exactly 0700")
        return _VerifiedHome(path=absolute, descriptor=descriptor)
    except BaseException:
        current_descriptor = descriptor
        descriptor = -1
        with suppress(BaseException):
            os.close(current_descriptor)
        raise


def _acquire_path_guard(path: Path) -> tuple[socket.socket, bytes]:
    """Bind stable Linux process ownership to the configured absolute path."""
    if not sys.platform.startswith("linux") or not hasattr(socket, "AF_UNIX"):
        raise PermissionError("stable runtime path ownership requires Linux AF_UNIX")
    normalized = os.path.normcase(os.path.abspath(os.path.expanduser(os.fspath(path))))
    digest = hashlib.sha256(os.fsencode(normalized)).hexdigest().encode("ascii")
    address = _PATH_GUARD_PREFIX + str(os.getuid()).encode("ascii") + b":" + digest
    guard = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    guard.set_inheritable(False)
    try:
        guard.bind(address)
    except OSError as error:
        with suppress(BaseException):
            guard.close()
        if error.errno == errno.EADDRINUSE:
            raise RuntimeOwnedError("runtime is already owned") from None
        raise PermissionError("runtime path guard could not be acquired") from error
    return guard, address


class RuntimeLease:
    """An advisory OS lock held for the complete SQLite owner lifetime."""

    _failed_acquire_lock: ClassVar[threading.Lock] = threading.Lock()
    _failed_acquires: ClassVar[dict[Path, _FailedAcquireOwner]] = {}
    _live_lease_lock: ClassVar[threading.Lock] = threading.Lock()
    _live_leases: ClassVar[weakref.WeakSet[RuntimeLease]] = weakref.WeakSet()

    def __init__(
        self,
        path: Path,
        descriptor: int,
        instance_nonce: str,
        process_marker: str,
        *,
        runtime_home: Path,
        home_descriptor: int,
        path_guard: socket.socket,
        path_guard_address: bytes,
    ) -> None:
        self.path = path
        self.runtime_home = runtime_home
        self._descriptor = descriptor
        self._home_descriptor = home_descriptor
        self._path_guard: socket.socket | None = path_guard
        self._path_guard_address = path_guard_address
        self._creator_pid = os.getpid()
        home_metadata = os.fstat(home_descriptor)
        self._home_identity = (home_metadata.st_dev, home_metadata.st_ino)
        lock_metadata = os.fstat(descriptor)
        self._lock_identity = (lock_metadata.st_dev, lock_metadata.st_ino)
        self.instance_nonce = instance_nonce
        self.process_marker = process_marker
        self._lock_unlocked = False
        self._lock_closed = False
        self._home_closed = False
        self._path_guard_closed = False
        self._retained_files_lock = threading.Lock()
        self._retained_files: set[_RetainedResource] = set()
        self._release_lock = threading.Lock()
        self._release_started = False
        self._released = False

    @classmethod
    def acquire(cls, runtime_home: str | Path) -> Self:
        requested_home = Path(runtime_home).expanduser().absolute()
        cls.retry_failed_acquire_cleanup(requested_home)
        path_guard, path_guard_address = _acquire_path_guard(requested_home)
        try:
            verified_home = _validate_private_home(requested_home)
        except BaseException:
            with suppress(BaseException):
                path_guard.close()
            raise
        home = verified_home.path
        path = home / _LOCK_NAME
        flags = os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        created = False
        new_lock_validated = False
        try:
            try:
                descriptor = os.open(
                    _LOCK_NAME,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=verified_home.descriptor,
                )
                created = True
            except FileExistsError:
                descriptor = os.open(
                    _LOCK_NAME,
                    flags,
                    dir_fd=verified_home.descriptor,
                )
        except BaseException:
            with suppress(BaseException):
                os.close(verified_home.descriptor)
            with suppress(BaseException):
                path_guard.close()
            raise
        lock_acquired = False
        try:
            if created:
                os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise PermissionError("runtime lease must be a regular file")
            if metadata.st_uid != os.getuid():
                raise PermissionError("runtime lease must be user-owned")
            if metadata.st_nlink != 1:
                raise PermissionError("runtime lease hardlinks are not permitted")
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise PermissionError("runtime lease mode must be exactly 0600")
            new_lock_validated = True
            if created:
                os.fsync(verified_home.descriptor)
            if os.name == "nt":
                raise PermissionError(
                    "Windows lease ACL verification is unavailable; refusing"
                )
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    raise RuntimeOwnedError("runtime is already owned") from None
                raise
            lock_acquired = True
            nonce = secrets.token_hex(32)
            process_marker = current_process_marker()
            record = {
                "instance_nonce": nonce,
                "pid": os.getpid(),
                "process_marker": process_marker,
            }
            encoded = json.dumps(
                record, allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.ftruncate(descriptor, 0)
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("runtime lease write made no progress")
                offset += written
            os.fsync(descriptor)
            lease = cls(
                path,
                descriptor,
                nonce,
                process_marker,
                runtime_home=home,
                home_descriptor=verified_home.descriptor,
                path_guard=path_guard,
                path_guard_address=path_guard_address,
            )
            lease.assert_authority()
            with cls._live_lease_lock:
                cls._live_leases.add(lease)
            return lease
        except BaseException as primary_error:
            # Preserve the acquisition fault while exhausting every cleanup
            # stage.  Once LOCK_UN succeeds, descriptor-close uncertainty can
            # no longer imply that this failed acquisition still owns the
            # runtime lease.
            if lock_acquired:
                failed_owner = _FailedAcquireOwner(
                    runtime_home=home,
                    descriptor=descriptor,
                    home_descriptor=verified_home.descriptor,
                    path_guard=path_guard,
                )
                try:
                    failed_owner.cleanup()
                except BaseException:
                    if not failed_owner.lock_unlocked:
                        cls._retain_failed_acquire(failed_owner)
                        raise RuntimeOwnedError(
                            "failed lease acquisition cleanup remains incomplete"
                        ) from primary_error
                    # Once explicit unlock completed, uncertain close results
                    # cannot imply continued lock authority and numeric file
                    # descriptors must never be retried.
                raise
            with suppress(BaseException):
                os.close(descriptor)
            if created and not new_lock_validated:
                with suppress(BaseException):
                    os.unlink(_LOCK_NAME, dir_fd=verified_home.descriptor)
            with suppress(BaseException):
                os.close(verified_home.descriptor)
            with suppress(BaseException):
                path_guard.close()
            raise

    @classmethod
    def _retain_failed_acquire(cls, owner: _FailedAcquireOwner) -> None:
        with cls._failed_acquire_lock:
            existing = cls._failed_acquires.get(owner.runtime_home)
            if existing is not None and existing is not owner:
                raise RuntimeOwnedError(
                    "another failed lease acquisition is already retained"
                )
            cls._failed_acquires[owner.runtime_home] = owner

    @classmethod
    def retry_failed_acquire_cleanup(cls, runtime_home: str | Path) -> bool:
        home = Path(runtime_home).expanduser().absolute()
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
            if cls._failed_acquires.get(home) is owner:
                del cls._failed_acquires[home]
            return True

    @classmethod
    def failed_acquire_count(cls) -> int:
        with cls._failed_acquire_lock:
            return len(cls._failed_acquires)

    @classmethod
    def _after_fork_child(cls) -> None:
        """Drop child copies without invoking LOCK_UN on a parent's flock."""
        failed_owners = tuple(cls._failed_acquires.values())
        live_leases = tuple(cls._live_leases)
        cls._failed_acquire_lock = threading.Lock()
        cls._failed_acquires = {}
        cls._live_lease_lock = threading.Lock()
        cls._live_leases = weakref.WeakSet()
        for owner in failed_owners:
            owner._abandon_after_fork_child()
        for lease in live_leases:
            lease._abandon_after_fork_child()

    def _abandon_after_fork_child(self) -> None:
        retained_owners = tuple(self._retained_files)
        guard = self._path_guard
        self._path_guard = None
        if guard is not None:
            with suppress(BaseException):
                guard.close()
        for attribute in ("_home_descriptor", "_descriptor"):
            descriptor = getattr(self, attribute)
            setattr(self, attribute, -1)
            if descriptor >= 0:
                with suppress(BaseException):
                    os.close(descriptor)
        self._retained_files_lock = threading.Lock()
        self._retained_files = set()
        self._release_lock = threading.Lock()
        for owner in retained_owners:
            reset_lock = getattr(owner, "_reset_close_lock_after_fork_child", None)
            if reset_lock is not None:
                reset_lock()

    def assert_creator_process(self) -> None:
        """Reject every inherited authority object before locks or cleanup."""
        if os.getpid() != self._creator_pid:
            raise PermissionError("runtime lease belongs to another process")

    @property
    def released(self) -> bool:
        return self._released

    def assert_authority(self) -> Path:
        """Prove this live lease still names its retained home and lock inodes."""
        self.assert_creator_process()
        if self._released:
            raise PermissionError("runtime lease authority has been released")
        guard = self._path_guard
        try:
            guard_address = None if guard is None else guard.getsockname()
        except OSError as error:
            raise PermissionError(
                "runtime path guard authority is no longer verifiable"
            ) from error
        if guard_address != self._path_guard_address:
            raise PermissionError("runtime path guard authority changed")
        current_home_descriptor: int | None = None
        try:
            retained_home = os.fstat(self._home_descriptor)
            current_home_descriptor = _open_existing_directory_nofollow(
                self.runtime_home
            )
            current_home = os.fstat(current_home_descriptor)
            retained_lock = os.fstat(self._descriptor)
            current_lock = os.stat(
                _LOCK_NAME,
                dir_fd=current_home_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise PermissionError(
                "runtime lease authority is no longer verifiable"
            ) from error
        finally:
            if current_home_descriptor is not None:
                os.close(current_home_descriptor)
        if (
            (retained_home.st_dev, retained_home.st_ino) != self._home_identity
            or (current_home.st_dev, current_home.st_ino) != self._home_identity
            or not stat.S_ISDIR(retained_home.st_mode)
            or not stat.S_ISDIR(current_home.st_mode)
            or retained_home.st_uid != os.getuid()
            or current_home.st_uid != os.getuid()
            or stat.S_IMODE(retained_home.st_mode) != 0o700
            or stat.S_IMODE(current_home.st_mode) != 0o700
            or (retained_lock.st_dev, retained_lock.st_ino) != self._lock_identity
            or (current_lock.st_dev, current_lock.st_ino) != self._lock_identity
            or not stat.S_ISREG(retained_lock.st_mode)
            or not stat.S_ISREG(current_lock.st_mode)
            or retained_lock.st_uid != os.getuid()
            or current_lock.st_uid != os.getuid()
            or retained_lock.st_nlink != 1
            or current_lock.st_nlink != 1
            or stat.S_IMODE(retained_lock.st_mode) != 0o600
            or stat.S_IMODE(current_lock.st_mode) != 0o600
        ):
            raise PermissionError("runtime lease authority identity changed")
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

    def open_home_file(self, name: str, flags: int, mode: int = 0o600) -> int:
        name = self._relative_name(name)
        self.assert_authority()
        return os.open(name, flags, mode, dir_fd=self._home_descriptor)

    def stat_home_file(self, name: str, *, follow_symlinks: bool = False):
        name = self._relative_name(name)
        self.assert_authority()
        return os.stat(
            name,
            dir_fd=self._home_descriptor,
            follow_symlinks=follow_symlinks,
        )

    def unlink_home_file(self, name: str, *, missing_ok: bool = False) -> None:
        name = self._relative_name(name)
        self.assert_authority()
        try:
            os.unlink(name, dir_fd=self._home_descriptor)
        except FileNotFoundError:
            if not missing_ok:
                raise

    def replace_home_file(self, source: str, destination: str) -> None:
        source = self._relative_name(source)
        destination = self._relative_name(destination)
        self.assert_authority()
        os.replace(
            source,
            destination,
            src_dir_fd=self._home_descriptor,
            dst_dir_fd=self._home_descriptor,
        )

    def chmod_home_file(self, name: str, mode: int) -> None:
        name = self._relative_name(name)
        self.assert_authority()
        os.chmod(
            name,
            mode,
            dir_fd=self._home_descriptor,
            follow_symlinks=False,
        )

    def fsync_home(self) -> None:
        self.assert_authority()
        os.fsync(self._home_descriptor)

    def retained_sqlite_path(self, name: str = "runtime.db") -> Path:
        """Name SQLite through the retained Linux directory descriptor."""
        name = self._relative_name(name)
        if not sys.platform.startswith("linux"):
            raise PermissionError(
                "retained SQLite path is unsupported on this platform"
            )
        self.assert_authority()
        retained = Path(f"/proc/self/fd/{self._home_descriptor}")
        try:
            metadata = retained.stat()
        except OSError as error:
            raise PermissionError(
                "retained runtime-home descriptor path is unavailable"
            ) from error
        if (
            metadata.st_dev,
            metadata.st_ino,
        ) != self._home_identity or not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError("retained runtime-home descriptor identity changed")
        return retained / name

    def retain_sqlite_files(self, name: str = "runtime.db") -> RetainedSQLiteFiles:
        """Pin private SQLite main/WAL/SHM inodes before SQLite can open them."""

        name = self._relative_name(name)
        if not sys.platform.startswith("linux"):
            raise PermissionError("retained SQLite files require Linux procfs")
        self.assert_authority()
        target = RetainedSQLiteFiles(self, name, {}, {})
        # This is the begin-operation fence. Registration precedes the first
        # native resource and shares a lock with release's terminal transition.
        target._close_lock.acquire()
        try:
            self._register_retained_files(target)
        except BaseException:
            target._close_lock.release()
            raise
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            for candidate in (
                name,
                f"{name}-journal",
                f"{name}-wal",
                f"{name}-shm",
            ):
                self._relative_name(candidate)
                created = False
                try:
                    descriptor = os.open(
                        candidate,
                        flags | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=self._home_descriptor,
                    )
                    created = True
                except FileExistsError:
                    try:
                        descriptor = os.open(
                            candidate,
                            flags,
                            dir_fd=self._home_descriptor,
                        )
                    except OSError as error:
                        raise PermissionError(
                            "SQLite file cannot be opened without following links"
                        ) from error
                except OSError as error:
                    raise PermissionError(
                        "SQLite file cannot be created without following links"
                    ) from error
                try:
                    metadata = target.adopt_descriptor(candidate, descriptor)
                except BaseException:
                    target.quarantine_unadopted_descriptor(candidate, descriptor)
                    raise
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_nlink != 1
                ):
                    raise PermissionError(
                        "SQLite files must be private regular single-link files"
                    )
                if created or stat.S_IMODE(metadata.st_mode) != 0o600:
                    os.fchmod(descriptor, 0o600)
                    metadata = os.fstat(descriptor)
                if stat.S_IMODE(metadata.st_mode) != 0o600:
                    raise PermissionError(
                        "SQLite file privacy mode could not be enforced"
                    )
                target._identities[candidate] = (
                    metadata.st_dev,
                    metadata.st_ino,
                )
            target.verify()
            self.fsync_home()
            return target
        except BaseException as primary_error:
            traceback = primary_error.__traceback__
            try:
                target.close()
            except BaseException as cleanup_error:
                raise primary_error.with_traceback(traceback) from cleanup_error
            raise
        finally:
            target._close_lock.release()

    def _register_retained_files(self, owner: _RetainedResource) -> None:
        self.assert_creator_process()
        with self._retained_files_lock:
            if self._release_started or self._released:
                raise PermissionError("runtime lease release has started")
            if owner in self._retained_files:
                raise RuntimeError("retained SQLite owner is already registered")
            self._retained_files.add(owner)

    def _unregister_retained_files(self, owner: _RetainedResource) -> None:
        self.assert_creator_process()
        with self._retained_files_lock:
            if owner not in self._retained_files:
                raise RuntimeError("retained SQLite owner is not registered")
            self._retained_files.remove(owner)

    def _discard_retained_files(self, owner: _RetainedResource) -> None:
        """Forget a closed owner, including a failed pre-registration owner."""
        self.assert_creator_process()
        with self._retained_files_lock:
            self._retained_files.discard(owner)

    def _replace_retained_files(
        self,
        old_owner: _RetainedResource,
        new_owner: _RetainedResource,
    ) -> None:
        """Atomically transfer the lease gate without an empty registry state."""
        self.assert_creator_process()
        with self._retained_files_lock:
            if self._release_started or self._released:
                raise PermissionError("runtime lease release has started")
            if old_owner not in self._retained_files:
                raise RuntimeError("retained SQLite owner is not registered")
            if new_owner in self._retained_files:
                raise RuntimeError("replacement SQLite owner is already registered")
            self._retained_files.add(new_owner)
            self._retained_files.remove(old_owner)

    def close_registered_retained_files(self) -> None:
        """Retry every pre-runtime SQLite owner before releasing the lease."""
        self.assert_creator_process()
        while True:
            with self._retained_files_lock:
                owners = tuple(self._retained_files)
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
            with self._retained_files_lock:
                if self._retained_files == set(owners):
                    raise RuntimeError(
                        "retained SQLite cleanup made no ownership progress"
                    )

    def release(self) -> None:
        self.assert_creator_process()
        with self._release_lock:
            if self._released:
                return
            with self._retained_files_lock:
                if self._retained_files:
                    raise PermissionError(
                        "retained SQLite resources must close before lease release"
                    )
                # Atomically prevent new retained operations before native
                # unlock begins. The fence stays terminal across cleanup retry.
                self._release_started = True
            if not self._lock_unlocked:
                import fcntl

                # A failed explicit unlock leaves both live descriptors untouched,
                # preserving fail-stop ownership for an exact later retry.
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
                self._lock_unlocked = True

            first_error: BaseException | None = None
            if not self._home_closed:
                descriptor = self._home_descriptor
                self._home_descriptor = -1
                try:
                    os.close(descriptor)
                except BaseException as error:
                    first_error = error
                finally:
                    self._home_closed = True
            if not self._lock_closed:
                descriptor = self._descriptor
                self._descriptor = -1
                try:
                    os.close(descriptor)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                finally:
                    self._lock_closed = True
            if not self._path_guard_closed:
                guard = self._path_guard
                self._path_guard = None
                try:
                    if guard is not None:
                        guard.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                finally:
                    self._path_guard_closed = True
            self._released = (
                self._lock_closed and self._home_closed and self._path_guard_closed
            )
            if self._released:
                with type(self)._live_lease_lock:
                    type(self)._live_leases.discard(self)
            if first_error is not None:
                raise first_error

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


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=RuntimeLease._after_fork_child)


__all__ = ["RetainedSQLiteFiles", "RuntimeLease"]
