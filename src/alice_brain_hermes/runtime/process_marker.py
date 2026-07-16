"""Portable, re-readable process creation markers for PID-reuse detection."""

from __future__ import annotations

import os
import re

import psutil

_MARKER_PATTERN = re.compile(r"^psutil-create-time-us:([1-9][0-9]*)$")


def read_process_marker(pid: int) -> str:
    """Read a canonical psutil create-time marker or fail closed."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be a positive integer")
    try:
        created_us = round(psutil.Process(pid).create_time() * 1_000_000)
    except (OSError, OverflowError, ValueError, psutil.Error) as error:
        raise PermissionError("process creation marker is not verifiable") from error
    if created_us <= 0:
        raise PermissionError("process creation marker is not verifiable")
    return f"psutil-create-time-us:{created_us}"


def current_process_marker() -> str:
    return read_process_marker(os.getpid())


def verify_process_marker(pid: int, expected: str) -> None:
    if not isinstance(expected, str) or _MARKER_PATTERN.fullmatch(expected) is None:
        raise PermissionError("process creation marker is not canonical")
    if read_process_marker(pid) != expected:
        raise PermissionError("process creation marker does not match")


__all__ = [
    "current_process_marker",
    "read_process_marker",
    "verify_process_marker",
]
