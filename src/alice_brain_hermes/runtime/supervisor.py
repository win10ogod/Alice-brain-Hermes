"""Portable python-dmon adapter for one background Hermes daemon."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import time
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass
from pathlib import Path

import portalocker
import psutil
from dmon.control import start_single
from dmon.types import DmonMeta, DmonTaskConfig
from platformdirs import PlatformDirs

from alice_brain_hermes.runtime.process_marker import read_process_marker

_CAPTURE_BYTES = 4_096
_META_BYTES = 65_536
_LAUNCH_NONCE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_TOKEN_PATTERN = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
try:
    _WORKER_EXECUTABLE: str | None = psutil.Process(os.getpid()).exe()
except (OSError, psutil.Error):
    _WORKER_EXECUTABLE = None
_ANCHORED_UNLINK_CODE = """
import json
import os
import sys
from pathlib import Path

expected = os.stat_result(tuple(json.loads(sys.argv[1])))
names = tuple(json.loads(sys.argv[2]))

def valid():
    current = os.stat(".", follow_symlinks=False)
    if not os.path.samestat(expected, current):
        return False
    if not names or len(names) != len(set(names)):
        return False
    if sorted(os.listdir(".")) != sorted(names):
        return False
    return all(
        os.path.basename(name) == name
        and name not in {".", ".."}
        and not Path(name).is_symlink()
        and Path(name).is_file()
        for name in names
    )

if not valid() or sys.stdin.buffer.read(1) != b"V" or not valid():
    raise SystemExit(2)
if sys.stdin.buffer.read(1) != b"C" or not valid():
    raise SystemExit(3)
for name in names:
    os.unlink(name)
if os.listdir("."):
    raise SystemExit(4)
"""


def _anchored_unlink(
    tasks: tuple[tuple[Path, os.stat_result, tuple[str, ...]], ...],
) -> bool:
    if _WORKER_EXECUTABLE is None or len(tasks) != 1:
        return False
    directory, expected, names = tasks[0]
    try:
        completed = subprocess.run(
            [
                _WORKER_EXECUTABLE,
                "-I",
                "-S",
                "-c",
                _ANCHORED_UNLINK_CODE,
                json.dumps(tuple(expected), separators=(",", ":")),
                json.dumps(names, separators=(",", ":")),
            ],
            cwd=directory,
            input=b"VC",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


class DmonIdentityError(RuntimeError):
    """The supervisor could not establish or preserve exact child identity."""

    def __init__(
        self,
        message: str,
        *,
        meta_hint: DmonMetaHint | None = None,
        cleanup_unproven: bool = False,
    ) -> None:
        super().__init__(message)
        self.meta_hint = meta_hint
        self.cleanup_unproven = cleanup_unproven


class DmonCoordinationTimeout(DmonIdentityError):
    """Another parent still owns this runtime's launch coordination."""


class _PortalockerGuard:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._owner: portalocker.Lock | None = None
        self._stream = None

    def acquire(
        self,
        *,
        timeout_seconds: float,
        fail_when_locked: bool,
    ) -> None:
        parent = self.path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            if parent.is_symlink() or not parent.is_dir():
                raise DmonIdentityError("portable guard directory is unsafe")
            parent.chmod(0o700)
            parent_identity = parent.stat(follow_symlinks=False)
            if self.path.is_symlink() or (
                self.path.exists() and not self.path.is_file()
            ):
                raise DmonIdentityError("portable guard path is unsafe")
        except DmonIdentityError:
            raise
        except OSError as error:
            raise DmonIdentityError("portable guard path is not verifiable") from error

        owner = portalocker.Lock(
            self.path,
            mode="a+b",
            timeout=timeout_seconds,
            check_interval=max(0.001, min(0.02, timeout_seconds)),
            fail_when_locked=fail_when_locked,
        )
        stream = owner.acquire()
        try:
            self.path.chmod(0o600)
            opened = os.fstat(stream.fileno())
            current = self.path.stat(follow_symlinks=False)
            current_parent = parent.stat(follow_symlinks=False)
            valid = (
                not self.path.is_symlink()
                and self.path.is_file()
                and os.path.samestat(opened, current)
                and os.path.samestat(parent_identity, current_parent)
            )
            if not valid:
                raise DmonIdentityError("portable guard identity changed")
        except BaseException:
            owner.release()
            raise
        self._owner = owner
        self._stream = stream

    def release(self) -> None:
        owner = self._owner
        if owner is None:
            return
        owner.release()
        self._owner = None
        self._stream = None


class DmonStartCoordinator:
    """Portable per-runtime parent launch serialization."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._guard = _PortalockerGuard(path)

    @classmethod
    def create(cls, runtime_home: str | Path) -> DmonStartCoordinator:
        home = Path(runtime_home).expanduser().absolute().resolve(strict=False)
        home_key = hashlib.sha256(
            os.fsencode(os.path.normcase(os.fspath(home)))
        ).hexdigest()
        state = (
            Path(
                PlatformDirs(
                    "alice-brain-hermes", appauthor=False
                ).user_state_path
            )
            .expanduser()
            .absolute()
            .resolve(strict=False)
        )
        return cls(state / "dmon" / "start-coordination" / f"{home_key}.lock")

    def acquire(self, *, timeout_seconds: float) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("coordination timeout must be finite and positive")
        try:
            self._guard.acquire(
                timeout_seconds=float(timeout_seconds),
                fail_when_locked=False,
            )
        except portalocker.exceptions.AlreadyLocked as error:
            raise DmonCoordinationTimeout(
                "daemon launch coordination timed out"
            ) from error
        except portalocker.exceptions.LockException as error:
            raise DmonCoordinationTimeout(
                "daemon launch coordination timed out"
            ) from error

    def release(self) -> None:
        self._guard.release()


class _BoundedCapture:
    def __init__(self, maximum: int = _CAPTURE_BYTES) -> None:
        self._maximum = maximum
        self._data = bytearray()

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8", errors="replace")
        self._data.extend(encoded)
        if len(self._data) > self._maximum:
            del self._data[: len(self._data) - self._maximum]
        return len(value)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return bytes(self._data).decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class DmonMetaHint:
    """Bounded recovery hint read from one private dmon metadata file."""

    pid: int
    create_time: float


@dataclass(frozen=True, slots=True)
class DmonProcessHint:
    """A dmon PID hint promoted only after a canonical marker read."""

    pid: int
    process_marker: str


class DmonAdapter:
    """Build one no-shell dmon task and retain only deterministic paths."""

    def __init__(
        self,
        *,
        runtime_home: Path,
        command: tuple[str, ...],
        launch_nonce: str,
        task: str,
        meta_path: Path,
        log_path: Path,
        legacy_log_path: Path,
        parent_guard_path: Path,
    ) -> None:
        self.runtime_home = runtime_home
        self._command = command
        self.launch_nonce = launch_nonce
        self.task = task
        self._task_prefix = task.removesuffix(launch_nonce)
        self.meta_path = meta_path
        self.log_path = log_path
        self.legacy_log_path = legacy_log_path
        self.parent_guard_path = parent_guard_path
        self.last_dmon_output = ""
        self._meta_directory_identity: os.stat_result | None = None
        self._log_directory_identity: os.stat_result | None = None
        self._parent_guard: _PortalockerGuard | None = None

    @classmethod
    def create(
        cls,
        runtime_home: str | Path,
        command: Sequence[str],
        *,
        launch_nonce: str | None = None,
    ) -> DmonAdapter:
        if isinstance(command, (str, bytes, bytearray)) or not isinstance(
            command, Sequence
        ):
            raise TypeError("dmon command must be a sequence of strings")
        copied = tuple(command)
        if not copied:
            raise ValueError("dmon command must not be empty")
        if not all(isinstance(argument, str) for argument in copied):
            raise TypeError("dmon command arguments must be strings")
        if not copied[0]:
            raise ValueError("dmon executable must not be empty")
        if launch_nonce is None:
            launch_nonce = secrets.token_hex(32)
        if (
            not isinstance(launch_nonce, str)
            or _LAUNCH_NONCE.fullmatch(launch_nonce) is None
        ):
            raise ValueError("launch nonce is invalid")

        home = Path(runtime_home).expanduser().absolute().resolve(strict=False)
        home_key = hashlib.sha256(
            os.fsencode(os.path.normcase(os.fspath(home)))
        ).hexdigest()[:16]
        task = f"alice-brain-hermes-{home_key}-{launch_nonce}"
        directories = PlatformDirs("alice-brain-hermes", appauthor=False)
        meta_directory = (
            Path(directories.user_state_path).expanduser().absolute().resolve(strict=False)
            / "dmon"
            / task
        )
        meta_path = meta_directory / f"{task}.meta.json"
        log_path = meta_directory / f"{task}.log"
        legacy_log_path = (
            Path(directories.user_log_path).expanduser().absolute().resolve(strict=False)
            / "dmon"
            / task
            / f"{task}.log"
        )
        parent_guard_path = meta_directory.parent / "parent-guards" / f"{task}.lock"
        return cls(
            runtime_home=home,
            command=copied,
            launch_nonce=launch_nonce,
            task=task,
            meta_path=meta_path,
            log_path=log_path,
            legacy_log_path=legacy_log_path,
            parent_guard_path=parent_guard_path,
        )

    def _acquire_parent_guard(self) -> None:
        if self._parent_guard is not None:
            raise DmonIdentityError("launch parent guard is already held")
        guard = _PortalockerGuard(self.parent_guard_path)
        try:
            guard.acquire(timeout_seconds=0.0, fail_when_locked=True)
        except portalocker.exceptions.AlreadyLocked as error:
            raise DmonIdentityError(
                "launch attempt is already parent-managed",
                cleanup_unproven=True,
            ) from error
        except portalocker.exceptions.LockException as error:
            raise DmonIdentityError(
                "launch parent guard is unavailable",
                cleanup_unproven=True,
            ) from error
        self._parent_guard = guard

    def release_parent_guard(self) -> None:
        guard = self._parent_guard
        if guard is None:
            return
        guard.release()
        self._parent_guard = None

    def _assert_prior_not_parent_managed(self, task: str) -> None:
        guard = _PortalockerGuard(
            self.parent_guard_path.parent / f"{task}.lock"
        )
        try:
            guard.acquire(timeout_seconds=0.0, fail_when_locked=True)
        except portalocker.exceptions.AlreadyLocked as error:
            raise DmonIdentityError(
                "prior dmon launch is still parent-managed",
                cleanup_unproven=True,
            ) from error
        except portalocker.exceptions.LockException as error:
            raise DmonIdentityError(
                "prior dmon parent ownership is unproven",
                cleanup_unproven=True,
            ) from error
        else:
            guard.release()

    @staticmethod
    def _create_launch_directory(path: Path) -> os.stat_result:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir(exist_ok=False)
        except OSError as error:
            raise DmonIdentityError(
                "private dmon launch directory already exists"
            ) from error
        if path.is_symlink() or not path.is_dir():
            raise DmonIdentityError("private dmon launch directory is invalid")
        try:
            path.chmod(0o700)
            return path.stat(follow_symlinks=False)
        except OSError as error:
            raise DmonIdentityError(
                "private dmon launch directory is not verifiable"
            ) from error

    def _prepare_launch_directories(self) -> None:
        meta_directory = self.meta_path.parent
        log_directory = self.log_path.parent
        meta_identity = self._create_launch_directory(meta_directory)
        self._meta_directory_identity = meta_identity
        self._log_directory_identity = meta_identity
        if log_directory != meta_directory:
            raise DmonIdentityError("dmon evidence must share one launch directory")

    def _assert_launch_directories(self) -> None:
        identities = (
            (self.meta_path.parent, self._meta_directory_identity),
            (self.log_path.parent, self._log_directory_identity),
        )
        for path, expected in identities:
            if expected is None:
                raise DmonIdentityError("private dmon launch directory is unowned")
            try:
                current = path.stat(follow_symlinks=False)
            except OSError as error:
                raise DmonIdentityError(
                    "private dmon launch directory identity changed"
                ) from error
            if not os.path.samestat(expected, current):
                raise DmonIdentityError(
                    "private dmon launch directory identity changed"
                )

    def _adopt_existing_directories(self) -> None:
        if (
            self._meta_directory_identity is not None
            and self._log_directory_identity is not None
        ):
            return
        if self.meta_path.parent != self.log_path.parent:
            raise DmonIdentityError("dmon evidence must share one launch directory")
        path = self.meta_path.parent
        if path.is_symlink() or not path.is_dir():
            raise DmonIdentityError("private dmon launch directory is invalid")
        try:
            identity = path.stat(follow_symlinks=False)
        except OSError as error:
            raise DmonIdentityError(
                "private dmon launch directory is not verifiable"
            ) from error
        self._meta_directory_identity = identity
        self._log_directory_identity = identity
        self._assert_launch_directories()

    @staticmethod
    def _regular_file(path: Path, *, label: str) -> None:
        if path.is_symlink() or not path.is_file():
            raise DmonIdentityError(f"python-dmon {label} path is unsafe")

    @staticmethod
    def _read_bounded_file(
        path: Path,
        *,
        directory: Path,
        directory_identity: os.stat_result,
    ) -> bytes:
        try:
            current_directory = directory.stat(follow_symlinks=False)
            before = path.stat(follow_symlinks=False)
        except OSError as error:
            raise DmonIdentityError("python-dmon metadata is unreadable") from error
        if not os.path.samestat(directory_identity, current_directory):
            raise DmonIdentityError(
                "private dmon launch directory identity changed"
            )
        if path.is_symlink() or not path.is_file():
            raise DmonIdentityError("python-dmon metadata path is unsafe")
        try:
            with path.open("rb") as stream:
                opened = os.fstat(stream.fileno())
                if not os.path.samestat(before, opened):
                    raise DmonIdentityError(
                        "python-dmon metadata open identity changed"
                    )
                payload = stream.read(_META_BYTES + 1)
        except DmonIdentityError:
            raise
        except OSError as error:
            raise DmonIdentityError("python-dmon metadata is unreadable") from error
        try:
            after = path.stat(follow_symlinks=False)
            final_directory = directory.stat(follow_symlinks=False)
        except OSError as error:
            raise DmonIdentityError(
                "python-dmon metadata changed while read"
            ) from error
        if not os.path.samestat(opened, after) or not os.path.samestat(
            directory_identity, final_directory
        ):
            raise DmonIdentityError("python-dmon metadata changed while read")
        if len(payload) > _META_BYTES:
            raise DmonIdentityError("python-dmon metadata exceeds its byte limit")
        return payload

    @staticmethod
    def _decode_meta(payload: bytes) -> DmonMeta:
        def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate metadata key")
                value[key] = item
            return value

        try:
            raw = json.loads(
                payload.decode("utf-8", errors="strict"),
                object_pairs_hook=unique_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("non-finite metadata number")
                ),
            )
            if not isinstance(raw, dict):
                raise ValueError("metadata must be an object")
            return DmonMeta(**raw)
        except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise DmonIdentityError("python-dmon metadata is invalid") from error

    def _load_bounded_meta(self) -> DmonMeta:
        self._assert_launch_directories()
        self._regular_file(self.meta_path, label="metadata")
        if self._meta_directory_identity is None:
            raise DmonIdentityError("private dmon launch directory is unowned")
        payload = self._read_bounded_file(
            self.meta_path,
            directory=self.meta_path.parent,
            directory_identity=self._meta_directory_identity,
        )
        return self._decode_meta(payload)

    @staticmethod
    def _meta_hint(meta: DmonMeta) -> DmonMetaHint | None:
        if (
            type(meta.pid) is not int
            or meta.pid <= 0
            or isinstance(meta.create_time, bool)
            or not isinstance(meta.create_time, (int, float))
            or not meta.create_time > 0
            or not math.isfinite(meta.create_time)
        ):
            return None
        return DmonMetaHint(pid=meta.pid, create_time=float(meta.create_time))

    @staticmethod
    def _expected_marker(hint: DmonMetaHint) -> str:
        created_us = round(hint.create_time * 1_000_000)
        if created_us <= 0:
            raise DmonIdentityError("python-dmon create time is invalid")
        return f"psutil-create-time-us:{created_us}"

    def _promote_meta_hint(
        self,
        hint: DmonMetaHint,
    ) -> tuple[str, DmonProcessHint | None]:
        expected = self._expected_marker(hint)
        try:
            current = read_process_marker(hint.pid)
        except (OSError, ValueError):
            try:
                exists = psutil.pid_exists(hint.pid)
            except (OSError, psutil.Error):
                return "ambiguous", None
            return ("ambiguous", None) if exists else ("gone", None)
        if current != expected:
            return "reused", None
        return "exact", DmonProcessHint(pid=hint.pid, process_marker=expected)

    @staticmethod
    def _remove_directory_files(
        directory: Path,
        directory_identity: os.stat_result,
        names: tuple[str, ...],
    ) -> bool:
        if not names or len(names) != len(set(names)):
            return False
        suffix = secrets.token_hex(16)
        quarantine = directory.with_name(f".{directory.name}.cleanup-{suffix}")

        def matches(path: Path, expected: os.stat_result) -> bool:
            try:
                current = path.stat(follow_symlinks=False)
            except OSError:
                return False
            return (
                not path.is_symlink()
                and path.is_dir()
                and os.path.samestat(expected, current)
            )

        def restore(quarantine: Path, original: Path) -> None:
            if quarantine.exists() and not original.exists():
                with suppress(OSError):
                    os.replace(quarantine, original)

        try:
            if (
                quarantine.exists()
                or quarantine.is_symlink()
                or not DmonAdapter._directory_has_exact_files(
                    directory,
                    directory_identity,
                    names,
                )
            ):
                return False
            os.replace(directory, quarantine)
            if not DmonAdapter._directory_has_exact_files(
                quarantine,
                directory_identity,
                names,
            ):
                restore(quarantine, directory)
                return False
            if not _anchored_unlink(
                ((quarantine, directory_identity, names),)
            ):
                try:
                    remaining = tuple(sorted(os.listdir(quarantine)))
                except OSError:
                    restore(quarantine, directory)
                    return False
                # The two-file transaction intentionally removes the log first.
                # If the metadata unlink is transiently blocked, retry only that
                # authenticated final evidence instead of restoring a layout that
                # future recovery cannot interpret.
                if (
                    len(names) == 2
                    and remaining == (names[-1],)
                    and DmonAdapter._directory_has_exact_files(
                        quarantine,
                        directory_identity,
                        remaining,
                    )
                    and _anchored_unlink(
                        ((quarantine, directory_identity, remaining),)
                    )
                ):
                    remaining = ()
                elif len(names) == 2 and remaining == (names[-1],):
                    try:
                        remaining = tuple(sorted(os.listdir(quarantine)))
                    except OSError:
                        restore(quarantine, directory)
                        return False
                if remaining:
                    restore(quarantine, directory)
                    return False
            if not matches(quarantine, directory_identity):
                return False
            try:
                if os.listdir(quarantine):
                    restore(quarantine, directory)
                    return False
                quarantine.rmdir()
            except OSError:
                # An empty random quarantine carries no remaining evidence and
                # is ignored by stable-path recovery.
                pass
        except OSError:
            restore(quarantine, directory)
            return False
        return (
            not directory.exists()
            and not directory.is_symlink()
            and all(
                not (directory / name).exists()
                and not (directory / name).is_symlink()
                for name in names
            )
        )

    @classmethod
    def _remove_attempt_paths(
        cls,
        meta_path: Path,
        log_path: Path,
        *,
        meta_directory_identity: os.stat_result,
        log_directory_identity: os.stat_result,
    ) -> bool:
        directory = meta_path.parent
        if directory != log_path.parent or not os.path.samestat(
            meta_directory_identity,
            log_directory_identity,
        ):
            return False
        if log_path.exists() or log_path.is_symlink():
            names = (log_path.name, meta_path.name)
        else:
            names = (meta_path.name,)
        return cls._remove_directory_files(
            directory,
            meta_directory_identity,
            names,
        )

    @classmethod
    def _remove_legacy_attempt_paths(
        cls,
        meta_path: Path,
        legacy_log_path: Path,
        *,
        meta_directory_identity: os.stat_result,
        log_directory_identity: os.stat_result | None,
    ) -> bool:
        if log_directory_identity is not None and not cls._remove_directory_files(
            legacy_log_path.parent,
            log_directory_identity,
            (legacy_log_path.name,),
        ):
            return False
        return cls._remove_directory_files(
            meta_path.parent,
            meta_directory_identity,
            (meta_path.name,),
        )

    @staticmethod
    def _executable_matches(actual: str, expected: str) -> bool:
        if actual == expected:
            return True
        canonical = shutil.which(expected)
        if canonical is None:
            return False
        expected_path = os.path.normcase(
            os.fspath(Path(canonical).resolve(strict=False))
        )
        actual_path = os.path.normcase(
            os.fspath(Path(actual).resolve(strict=False))
        )
        return actual_path == expected_path

    def _prior_product_metadata_matches(
        self,
        meta: DmonMeta,
        *,
        task: str,
        nonce: str,
        meta_path: Path,
        log_path: Path,
    ) -> bool:
        command = meta.cmd
        try:
            return (
                isinstance(command, list)
                and len(command) == 7
                and all(isinstance(argument, str) for argument in command)
                and self._executable_matches(command[0], self._command[0])
                and command[1:4]
                == [
                    "-m",
                    "alice_brain_hermes.runtime.daemon",
                    "--runtime-home",
                ]
                and Path(command[4]).expanduser().absolute().resolve(strict=False)
                == self.runtime_home
                and command[5:] == ["--launch-nonce", nonce]
                and meta.task == task
                and Path(meta.meta_path).resolve() == meta_path.resolve()
                and Path(meta.log_path).resolve() == log_path.resolve()
                and Path(meta.cwd).resolve() == meta_path.parent.resolve()
                and meta.env == {}
                and meta.override_env is False
                and meta.shell is False
                and meta.log_rotate is False
                and meta.log_max_size == 5
                and meta.rotate_log_path == ""
                and meta.rotate_log_max_size == 5
            )
        except (IndexError, OSError, TypeError, ValueError):
            return False

    def _restore_interrupted_cleanup_directories(self, root: Path) -> None:
        if not root.exists():
            return
        if root.is_symlink() or not root.is_dir():
            raise DmonIdentityError(
                "interrupted dmon cleanup root is unsafe",
                cleanup_unproven=True,
            )
        try:
            root_identity = root.stat(follow_symlinks=False)
        except OSError as error:
            raise DmonIdentityError(
                "interrupted dmon cleanup root is not verifiable",
                cleanup_unproven=True,
            ) from error
        for quarantine in sorted(root.glob(f".{self._task_prefix}*.cleanup-*")):
            visible = quarantine.name.removeprefix(".")
            task, separator, token = visible.rpartition(".cleanup-")
            nonce = task.removeprefix(self._task_prefix)
            if (
                separator != ".cleanup-"
                or len(token) != 32
                or any(character not in "0123456789abcdef" for character in token)
                or not nonce
                or _LAUNCH_NONCE.fullmatch(nonce) is None
                or task != f"{self._task_prefix}{nonce}"
                or quarantine.is_symlink()
                or not quarantine.is_dir()
            ):
                raise DmonIdentityError(
                    "interrupted dmon cleanup identity is invalid",
                    cleanup_unproven=True,
                )
            try:
                quarantine_identity = quarantine.stat(follow_symlinks=False)
                contents = os.listdir(quarantine)
            except OSError as error:
                raise DmonIdentityError(
                    "interrupted dmon cleanup is not verifiable",
                    cleanup_unproven=True,
                ) from error
            if not contents:
                continue
            original = root / task
            if original.exists() or original.is_symlink():
                raise DmonIdentityError(
                    "interrupted dmon cleanup collides with stable state",
                    cleanup_unproven=True,
                )
            try:
                if not os.path.samestat(
                    root_identity,
                    root.stat(follow_symlinks=False),
                ) or not os.path.samestat(
                    quarantine_identity,
                    quarantine.stat(follow_symlinks=False),
                ):
                    raise DmonIdentityError(
                        "interrupted dmon cleanup identity changed"
                    )
                os.replace(quarantine, original)
                if not os.path.samestat(
                    quarantine_identity,
                    original.stat(follow_symlinks=False),
                ):
                    raise DmonIdentityError(
                        "restored dmon cleanup identity changed"
                    )
            except OSError as error:
                raise DmonIdentityError(
                    "interrupted dmon cleanup restoration is unproven",
                    cleanup_unproven=True,
                ) from error

    def _recover_prior_launches(self) -> None:
        root = self.meta_path.parent.parent
        if not root.exists():
            return
        self._restore_interrupted_cleanup_directories(root)
        self._restore_interrupted_cleanup_directories(
            self.legacy_log_path.parent.parent
        )
        if root.is_symlink() or not root.is_dir():
            raise DmonIdentityError(
                "prior dmon launch state is unsafe",
                cleanup_unproven=True,
            )
        candidates = tuple(
            directory
            for directory in sorted(root.glob(f"{self._task_prefix}*"))
            if directory != self.meta_path.parent
        )
        if not candidates:
            return
        try:
            root_identity = root.stat(follow_symlinks=False)
        except OSError as error:
            raise DmonIdentityError(
                "prior dmon launch parent identity is unproven",
                cleanup_unproven=True,
            ) from error
        for directory in candidates:
            self._assert_prior_not_parent_managed(directory.name)
            if directory.is_symlink() or not directory.is_dir():
                raise DmonIdentityError(
                    "prior dmon launch state is unsafe",
                    cleanup_unproven=True,
                )
            try:
                identity = directory.stat(follow_symlinks=False)
                meta_path = directory / f"{directory.name}.meta.json"
                nonce = directory.name.removeprefix(self._task_prefix)
                if (
                    not nonce
                    or _LAUNCH_NONCE.fullmatch(nonce) is None
                    or directory.name != f"{self._task_prefix}{nonce}"
                ):
                    raise DmonIdentityError("prior dmon task identity is invalid")
                payload = self._read_bounded_file(
                    meta_path,
                    directory=directory,
                    directory_identity=identity,
                )
                meta = self._decode_meta(payload)
                current_log_path = directory / f"{directory.name}.log"
                legacy_log_root = self.legacy_log_path.parent.parent
                legacy_log_directory = legacy_log_root / directory.name
                legacy_log_path = legacy_log_directory / f"{directory.name}.log"
                if self._prior_product_metadata_matches(
                    meta,
                    task=directory.name,
                    nonce=nonce,
                    meta_path=meta_path,
                    log_path=current_log_path,
                ):
                    log_path = current_log_path
                    log_identity = identity
                    legacy_layout = False
                    legacy_log_root_identity = None
                    if current_log_path.exists() or current_log_path.is_symlink():
                        self._regular_file(current_log_path, label="log")
                    elif not self._directory_has_exact_files(
                        directory,
                        identity,
                        (meta_path.name,),
                    ):
                        raise DmonIdentityError(
                            "prior partial dmon cleanup state is invalid"
                        )
                elif self._prior_product_metadata_matches(
                    meta,
                    task=directory.name,
                    nonce=nonce,
                    meta_path=meta_path,
                    log_path=legacy_log_path,
                ):
                    log_path = legacy_log_path
                    legacy_layout = True
                    if (
                        legacy_log_directory.exists()
                        or legacy_log_directory.is_symlink()
                    ):
                        if (
                            legacy_log_root.is_symlink()
                            or not legacy_log_root.is_dir()
                        ):
                            raise DmonIdentityError(
                                "prior legacy dmon log root is unsafe"
                            )
                        legacy_log_root_identity = legacy_log_root.stat(
                            follow_symlinks=False
                        )
                        if (
                            legacy_log_directory.is_symlink()
                            or not legacy_log_directory.is_dir()
                        ):
                            raise DmonIdentityError(
                                "prior legacy dmon log directory is unsafe"
                            )
                        log_identity = legacy_log_directory.stat(
                            follow_symlinks=False
                        )
                        self._regular_file(legacy_log_path, label="legacy log")
                    else:
                        legacy_log_root_identity = None
                        log_identity = None
                        if not self._directory_has_exact_files(
                            directory,
                            identity,
                            (meta_path.name,),
                        ):
                            raise DmonIdentityError(
                                "prior partial legacy cleanup state is invalid"
                            )
                else:
                    raise DmonIdentityError(
                        "prior dmon launch metadata is invalid"
                    )
            except (DmonIdentityError, OSError) as error:
                raise DmonIdentityError(
                    "prior dmon launch cleanup is unproven",
                    cleanup_unproven=True,
                ) from error
            hint = self._meta_hint(meta)
            if hint is None:
                raise DmonIdentityError(
                    "prior dmon launch cleanup is unproven",
                    meta_hint=hint,
                    cleanup_unproven=True,
                )
            try:
                if not os.path.samestat(
                    root_identity, root.stat(follow_symlinks=False)
                ):
                    raise DmonIdentityError(
                        "prior dmon launch parent identity changed"
                    )
                if (
                    legacy_layout
                    and legacy_log_root_identity is not None
                    and not os.path.samestat(
                        legacy_log_root_identity,
                        legacy_log_root.stat(follow_symlinks=False),
                    )
                ):
                    raise DmonIdentityError(
                        "prior legacy dmon log parent identity changed"
                    )
            except OSError as error:
                raise DmonIdentityError(
                    "prior dmon launch parent identity is unproven",
                    meta_hint=hint,
                    cleanup_unproven=True,
                ) from error
            state, process_hint = self._promote_meta_hint(hint)
            if state == "exact" and process_hint is not None:
                try:
                    self.terminate_exact(process_hint, timeout_seconds=5.0)
                except DmonIdentityError as error:
                    raise DmonIdentityError(
                        "prior dmon launch cleanup is unproven",
                        meta_hint=hint,
                        cleanup_unproven=True,
                    ) from error
            if state in {"exact", "gone", "reused"}:
                if legacy_layout:
                    removed = self._remove_legacy_attempt_paths(
                        meta_path,
                        log_path,
                        meta_directory_identity=identity,
                        log_directory_identity=log_identity,
                    )
                else:
                    removed = self._remove_attempt_paths(
                        meta_path,
                        log_path,
                        meta_directory_identity=identity,
                        log_directory_identity=log_identity,
                    )
                if not removed:
                    raise DmonIdentityError(
                        "prior dmon launch cleanup is unproven",
                        meta_hint=hint,
                        cleanup_unproven=True,
                    )
                continue
            raise DmonIdentityError(
                "prior dmon launch cleanup is unproven",
                meta_hint=hint,
                cleanup_unproven=True,
            )

    def _command_matches(self, value: object) -> bool:
        if (
            not isinstance(value, list)
            or len(value) != len(self._command)
            or not all(isinstance(argument, str) for argument in value)
            or value[1:] != list(self._command[1:])
        ):
            return False
        if value[0] == self._command[0]:
            return True
        return self._executable_matches(value[0], self._command[0])

    def _metadata_matches(
        self,
        meta: DmonMeta,
        *,
        log_path: Path | None = None,
    ) -> bool:
        expected_log_path = self.log_path if log_path is None else log_path
        try:
            return (
                self._meta_hint(meta) is not None
                and meta.task == self.task
                and Path(meta.meta_path).resolve() == self.meta_path.resolve()
                and Path(meta.log_path).resolve() == expected_log_path.resolve()
                and Path(meta.cwd).resolve() == self.meta_path.parent.resolve()
                and self._command_matches(meta.cmd)
                and meta.env == {}
                and meta.override_env is False
                and meta.shell is False
                and meta.log_rotate is False
                and meta.log_max_size == 5
                and meta.rotate_log_path == ""
                and meta.rotate_log_max_size == 5
            )
        except (OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _directory_matches(path: Path, expected: os.stat_result) -> bool:
        try:
            current = path.stat(follow_symlinks=False)
        except OSError:
            return False
        return (
            not path.is_symlink()
            and path.is_dir()
            and os.path.samestat(expected, current)
        )

    @classmethod
    def _directory_has_exact_files(
        cls,
        path: Path,
        expected: os.stat_result,
        names: tuple[str, ...],
    ) -> bool:
        if not cls._directory_matches(path, expected):
            return False
        try:
            if sorted(os.listdir(path)) != sorted(names):
                return False
        except OSError:
            return False
        return all(
            (path / name).name == name
            and not (path / name).is_symlink()
            and (path / name).is_file()
            for name in names
        )

    def _owned_log_layout(
        self,
        meta: DmonMeta,
    ) -> tuple[Path, os.stat_result | None, bool]:
        meta_identity = self._meta_directory_identity
        if meta_identity is None:
            raise DmonIdentityError("private dmon launch directory is unowned")
        if self._metadata_matches(meta):
            if self.log_path.exists() or self.log_path.is_symlink():
                self._regular_file(self.log_path, label="log")
            elif not self._directory_has_exact_files(
                self.meta_path.parent,
                meta_identity,
                (self.meta_path.name,),
            ):
                raise DmonIdentityError(
                    "partial python-dmon cleanup state is invalid"
                )
            return self.log_path, meta_identity, False
        if not self._metadata_matches(meta, log_path=self.legacy_log_path):
            raise DmonIdentityError("python-dmon supervisor metadata is invalid")
        legacy_directory = self.legacy_log_path.parent
        if not legacy_directory.exists() and not legacy_directory.is_symlink():
            if not self._directory_has_exact_files(
                self.meta_path.parent,
                meta_identity,
                (self.meta_path.name,),
            ):
                raise DmonIdentityError(
                    "partial legacy python-dmon cleanup state is invalid"
                )
            return self.legacy_log_path, None, True
        if legacy_directory.is_symlink() or not legacy_directory.is_dir():
            raise DmonIdentityError("legacy python-dmon log directory is unsafe")
        try:
            legacy_identity = legacy_directory.stat(follow_symlinks=False)
        except OSError as error:
            raise DmonIdentityError(
                "legacy python-dmon log directory is not verifiable"
            ) from error
        self._regular_file(self.legacy_log_path, label="legacy log")
        if not self._directory_matches(legacy_directory, legacy_identity):
            raise DmonIdentityError("legacy python-dmon log identity changed")
        return self.legacy_log_path, legacy_identity, True

    @staticmethod
    def _apply_private_file_mode(path: Path, *, label: str) -> None:
        try:
            before = path.stat(follow_symlinks=False)
            if path.is_symlink() or not path.is_file():
                raise DmonIdentityError(f"python-dmon {label} path is unsafe")
            try:
                path.chmod(0o600, follow_symlinks=False)
            except NotImplementedError:
                current = path.stat(follow_symlinks=False)
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or not os.path.samestat(before, current)
                ):
                    raise DmonIdentityError(
                        f"python-dmon {label} identity changed while securing it"
                    ) from None
                return
            after = path.stat(follow_symlinks=False)
        except DmonIdentityError:
            raise
        except OSError as error:
            raise DmonIdentityError(
                f"python-dmon {label} path is not verifiable"
            ) from error
        if not os.path.samestat(before, after):
            raise DmonIdentityError(
                f"python-dmon {label} identity changed while securing it"
            )

    def _post_launch_failure(
        self,
        error: DmonIdentityError,
    ) -> DmonIdentityError:
        hint: DmonMetaHint | None = None
        metadata_owned = False
        try:
            meta = self._load_bounded_meta()
            hint = self._meta_hint(meta)
            metadata_owned = self._metadata_matches(meta)
        except DmonIdentityError:
            pass

        if hint is not None and metadata_owned:
            state, process_hint = self._promote_meta_hint(hint)
            if state == "exact" and process_hint is not None:
                try:
                    self.terminate_exact(process_hint, timeout_seconds=5.0)
                except DmonIdentityError:
                    return DmonIdentityError(
                        str(error),
                        meta_hint=hint,
                        cleanup_unproven=True,
                    )
                if self._remove_current_attempt():
                    return error
            elif state in {"gone", "reused"}:
                if self._remove_current_attempt():
                    return error

        return DmonIdentityError(
            str(error),
            meta_hint=hint,
            cleanup_unproven=True,
        )

    def _remove_current_attempt(self) -> bool:
        if (
            self._meta_directory_identity is None
            or self._log_directory_identity is None
        ):
            return False
        return self._remove_attempt_paths(
            self.meta_path,
            self.log_path,
            meta_directory_identity=self._meta_directory_identity,
            log_directory_identity=self._log_directory_identity,
        )

    def _start_guarded(self) -> DmonProcessHint:
        self._recover_prior_launches()
        self._prepare_launch_directories()
        config = DmonTaskConfig(
            task=self.task,
            cmd=list(self._command),
            cwd=os.fspath(self.meta_path.parent),
            env={},
            override_env=False,
            log_path=os.fspath(self.log_path),
            log_rotate=False,
            meta_path=os.fspath(self.meta_path),
        )
        capture = _BoundedCapture()
        try:
            with redirect_stdout(capture), redirect_stderr(capture):
                result = start_single(config)
        except BaseException as error:
            self.last_dmon_output = capture.getvalue()
            failure = DmonIdentityError("python-dmon start failed")
            raise self._post_launch_failure(failure) from error
        self.last_dmon_output = capture.getvalue()
        try:
            if result != 0:
                raise DmonIdentityError("python-dmon start failed")
            self._assert_launch_directories()
            self._apply_private_file_mode(self.meta_path, label="metadata")
            self._apply_private_file_mode(self.log_path, label="log")
            meta = self._load_bounded_meta()
            hint = self._meta_hint(meta)
            if hint is None or not self._metadata_matches(meta):
                raise DmonIdentityError("python-dmon metadata is invalid")
            state, process_hint = self._promote_meta_hint(hint)
            if state != "exact" or process_hint is None:
                raise DmonIdentityError(
                    "python-dmon child identity is not verifiable",
                    meta_hint=hint,
                )
            return process_hint
        except DmonIdentityError as error:
            raise self._post_launch_failure(error) from error

    def start(self) -> DmonProcessHint:
        self._acquire_parent_guard()
        try:
            return self._start_guarded()
        except BaseException:
            self.release_parent_guard()
            raise

    @staticmethod
    def _process_is_exact(
        process: psutil.Process,
        hint: DmonProcessHint,
    ) -> bool:
        if process.pid != hint.pid:
            return False
        try:
            created = process.create_time()
        except psutil.NoSuchProcess:
            return False
        except (OSError, psutil.Error) as error:
            raise DmonIdentityError("dmon child identity is not verifiable") from error
        if (
            isinstance(created, bool)
            or not isinstance(created, (int, float))
            or not math.isfinite(float(created))
            or created <= 0
        ):
            raise DmonIdentityError("dmon child create time is invalid")
        marker = f"psutil-create-time-us:{round(float(created) * 1_000_000)}"
        return marker == hint.process_marker

    def _exact_process(self, hint: DmonProcessHint) -> psutil.Process | None:
        try:
            process = psutil.Process(hint.pid)
        except psutil.NoSuchProcess:
            return None
        except (OSError, psutil.Error) as error:
            raise DmonIdentityError("dmon child process is not accessible") from error
        return process if self._process_is_exact(process, hint) else None

    @staticmethod
    def _hint_for_process(process: psutil.Process) -> DmonProcessHint:
        try:
            created = process.create_time()
        except (OSError, psutil.Error) as error:
            raise DmonIdentityError(
                "dmon descendant identity is not verifiable"
            ) from error
        if (
            isinstance(created, bool)
            or not isinstance(created, (int, float))
            or not math.isfinite(float(created))
            or created <= 0
        ):
            raise DmonIdentityError("dmon descendant create time is invalid")
        marker = f"psutil-create-time-us:{round(float(created) * 1_000_000)}"
        return DmonProcessHint(pid=process.pid, process_marker=marker)

    def terminate_exact(
        self,
        hint: DmonProcessHint,
        *,
        timeout_seconds: float,
    ) -> None:
        if not isinstance(hint, DmonProcessHint):
            raise TypeError("cleanup requires a typed dmon process hint")
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("cleanup timeout must be positive")
        deadline = time.monotonic() + float(timeout_seconds)
        process = self._exact_process(hint)
        if process is None:
            return
        try:
            descendants = tuple(process.children(recursive=True))
            descendant_hints = tuple(
                (descendant, self._hint_for_process(descendant))
                for descendant in descendants
            )
        except psutil.NoSuchProcess as error:
            raise DmonIdentityError(
                "dmon process tree identity became unverifiable"
            ) from error
        except (OSError, psutil.Error) as error:
            raise DmonIdentityError(
                "dmon process tree is not accessible"
            ) from error
        if not self._process_is_exact(process, hint):
            return

        tree = ((process, hint), *reversed(descendant_hints))
        signaled: list[tuple[psutil.Process, DmonProcessHint]] = []
        for selected, selected_hint in tree:
            if not self._process_is_exact(selected, selected_hint):
                continue
            try:
                selected.terminate()
            except psutil.NoSuchProcess:
                continue
            except (OSError, psutil.Error) as error:
                raise DmonIdentityError(
                    "dmon process tree termination could not be proven"
                ) from error
            signaled.append((selected, selected_hint))

        timed_out: list[DmonProcessHint] = []
        for selected, selected_hint in signaled:
            try:
                selected.wait(timeout=max(0.001, deadline - time.monotonic()))
            except psutil.NoSuchProcess:
                continue
            except psutil.TimeoutExpired:
                timed_out.append(selected_hint)
            except (OSError, psutil.Error) as error:
                raise DmonIdentityError(
                    "dmon process tree termination could not be proven"
                ) from error

        for selected_hint in timed_out:
            selected = self._exact_process(selected_hint)
            if selected is None or not self._process_is_exact(
                selected, selected_hint
            ):
                continue
            try:
                selected.kill()
                selected.wait(timeout=max(0.001, deadline - time.monotonic()))
            except psutil.NoSuchProcess:
                continue
            except (OSError, psutil.Error) as error:
                raise DmonIdentityError(
                    "dmon process tree termination could not be proven"
                ) from error

    def remove_meta_hint(self, hint: DmonProcessHint) -> bool:
        if not isinstance(hint, DmonProcessHint):
            raise TypeError("metadata cleanup requires a typed process hint")
        paths = (
            self.meta_path,
            self.log_path,
            self.meta_path.parent,
            self.log_path.parent,
            self.legacy_log_path,
            self.legacy_log_path.parent,
        )
        if not any(path.exists() or path.is_symlink() for path in paths):
            return True
        try:
            self._adopt_existing_directories()
            meta = self._load_bounded_meta()
            _log_path, log_directory_identity, legacy_layout = (
                self._owned_log_layout(meta)
            )
        except DmonIdentityError:
            return False
        meta_hint = self._meta_hint(meta)
        if meta_hint is None:
            return False
        try:
            marker = self._expected_marker(meta_hint)
        except DmonIdentityError:
            return False
        expected_log_path = self.legacy_log_path if legacy_layout else self.log_path
        if not (
            meta.pid == hint.pid
            and marker == hint.process_marker
            and self._metadata_matches(meta, log_path=expected_log_path)
        ):
            return False
        if legacy_layout:
            meta_identity = self._meta_directory_identity
            if meta_identity is None:
                return False
            return self._remove_legacy_attempt_paths(
                self.meta_path,
                self.legacy_log_path,
                meta_directory_identity=meta_identity,
                log_directory_identity=log_directory_identity,
            )
        return self._remove_current_attempt()

    def current_process_hint(self) -> DmonProcessHint:
        try:
            self._adopt_existing_directories()
            meta = self._load_bounded_meta()
            _log_path, _log_identity, legacy_layout = self._owned_log_layout(meta)
        except DmonIdentityError as error:
            raise DmonIdentityError(
                "python-dmon supervisor identity is not verifiable"
            ) from error
        hint = self._meta_hint(meta)
        expected_log_path = self.legacy_log_path if legacy_layout else self.log_path
        if hint is None or not self._metadata_matches(
            meta,
            log_path=expected_log_path,
        ):
            raise DmonIdentityError("python-dmon supervisor metadata is invalid")
        state, process_hint = self._promote_meta_hint(hint)
        if state != "exact" or process_hint is None:
            raise DmonIdentityError("python-dmon supervisor process is not exact")
        return process_hint

    def _redact(self, value: str) -> str:
        value = value.replace(os.fspath(self.runtime_home), "<runtime-home>")
        value = value.replace(self.launch_nonce, "<redacted>")
        value = _TOKEN_PATTERN.sub("<redacted>", value)
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) > _CAPTURE_BYTES:
            encoded = encoded[-_CAPTURE_BYTES:]
        return encoded.decode("utf-8", errors="replace")

    def redacted_dmon_output(self) -> str:
        return self._redact(self.last_dmon_output)

    def redacted_log_tail(self) -> str:
        path = self.log_path
        if path.is_symlink() or not path.is_file():
            return ""
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - _CAPTURE_BYTES), os.SEEK_SET)
                payload = stream.read(_CAPTURE_BYTES)
        except OSError:
            return ""
        value = payload.decode("utf-8", errors="replace")
        return self._redact(value)


__all__ = [
    "DmonAdapter",
    "DmonCoordinationTimeout",
    "DmonIdentityError",
    "DmonMetaHint",
    "DmonProcessHint",
    "DmonStartCoordinator",
]
