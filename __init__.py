"""Hermes directory-plugin entry point for a Git-installed checkout."""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_ROOT = str(Path(__file__).resolve().parent / "src")
_inserted_source_root = _SOURCE_ROOT not in sys.path
if _inserted_source_root:
    sys.path.insert(0, _SOURCE_ROOT)

try:
    from alice_brain_hermes.hermes_plugin import register
finally:
    if _inserted_source_root:
        sys.path.remove(_SOURCE_ROOT)

__all__ = ["register"]
