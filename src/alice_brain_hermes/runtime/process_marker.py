"""Exact, re-readable process creation markers for PID-reuse detection."""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path

_BOOT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_MAX_BOOT_ID_BYTES = 64
_MAX_PROC_STAT_BYTES = 4_096


def _read_bounded_nofollow(path: Path, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PermissionError("process start marker is not verifiable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError("process marker source is not a regular file")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(
                descriptor,
                min(4_096, maximum + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (
            after.st_dev,
            after.st_ino,
        ) or not stat.S_ISREG(after.st_mode):
            raise PermissionError("process marker source changed while read")
    finally:
        os.close(descriptor)
    if not payload or len(payload) > maximum:
        raise PermissionError("process marker source exceeds its byte limit")
    return bytes(payload)


def _parse_linux_start_ticks(payload: str, *, expected_pid: int) -> int:
    """Parse `/proc/<pid>/stat` without being confused by `)` in comm."""
    closing = payload.rfind(")")
    if closing < 0:
        raise PermissionError("process start marker is not verifiable")
    prefix = payload[:closing]
    opening = prefix.find("(")
    if opening < 1:
        raise PermissionError("process start marker is not verifiable")
    try:
        parsed_pid = int(prefix[:opening].strip())
    except ValueError as error:
        raise PermissionError("process start marker is not verifiable") from error
    fields_after_comm = payload[closing + 1 :].split()
    if parsed_pid != expected_pid or len(fields_after_comm) <= 19:
        raise PermissionError("process start marker is not verifiable")
    try:
        ticks = int(fields_after_comm[19])
    except ValueError as error:
        raise PermissionError("process start marker is not verifiable") from error
    if ticks <= 0:
        raise PermissionError("process start marker is not verifiable")
    return ticks


def read_process_marker(pid: int) -> str:
    """Read an exact tagged marker or fail closed when it cannot be proven."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be a positive integer")
    if not sys.platform.startswith("linux"):
        raise PermissionError("exact process creation marker is unavailable")
    try:
        boot_first = (
            _read_bounded_nofollow(
                Path("/proc/sys/kernel/random/boot_id"),
                maximum=_MAX_BOOT_ID_BYTES,
            )
            .decode("ascii", errors="strict")
            .strip()
        )
        stat_first = _read_bounded_nofollow(
            Path(f"/proc/{pid}/stat"),
            maximum=_MAX_PROC_STAT_BYTES,
        ).decode("utf-8", errors="strict")
        boot_second = (
            _read_bounded_nofollow(
                Path("/proc/sys/kernel/random/boot_id"),
                maximum=_MAX_BOOT_ID_BYTES,
            )
            .decode("ascii", errors="strict")
            .strip()
        )
        stat_second = _read_bounded_nofollow(
            Path(f"/proc/{pid}/stat"),
            maximum=_MAX_PROC_STAT_BYTES,
        ).decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise PermissionError("process start marker is not verifiable") from error
    if boot_first != boot_second or _BOOT_ID_PATTERN.fullmatch(boot_first) is None:
        raise PermissionError("process boot marker is not verifiable")
    first_ticks = _parse_linux_start_ticks(stat_first, expected_pid=pid)
    second_ticks = _parse_linux_start_ticks(stat_second, expected_pid=pid)
    if first_ticks != second_ticks:
        raise PermissionError("process creation marker changed while read")
    return f"linux:{boot_first}:{first_ticks}"


def current_process_marker() -> str:
    return read_process_marker(os.getpid())


def verify_process_marker(pid: int, expected: str) -> None:
    if read_process_marker(pid) != expected:
        raise PermissionError("process creation marker does not match")


__all__ = [
    "current_process_marker",
    "read_process_marker",
    "verify_process_marker",
]
