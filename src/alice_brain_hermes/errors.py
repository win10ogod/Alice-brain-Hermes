"""Stable domain and persistence errors for Alice-brain-Hermes."""

from __future__ import annotations


class AliceBrainHermesError(Exception):
    """Base class for errors raised by this independent runtime."""


class DomainInvariantError(AliceBrainHermesError, ValueError):
    """Raised when an event would violate a deterministic domain invariant."""


class LedgerError(AliceBrainHermesError):
    """Base class for ledger failures."""


class LedgerClosedError(LedgerError):
    """Raised when an operation is attempted on a closed ledger."""


class LedgerIntegrityError(LedgerError):
    """Raised when persisted data does not match its integrity metadata."""


class EventConflictError(LedgerError):
    """Raised when an event ID is reused with a different immutable body."""


class ExpectedSequenceError(EventConflictError):
    """Raised when a new event loses its exact expected-sequence race."""


class SnapshotConflictError(LedgerError):
    """Raised when a snapshot is stale, future-dated, or not replay-equivalent."""


class SchemaVersionError(LedgerError):
    """Raised when the SQLite or serialized schema is unsupported."""


__all__ = [
    "AliceBrainHermesError",
    "DomainInvariantError",
    "EventConflictError",
    "ExpectedSequenceError",
    "LedgerClosedError",
    "LedgerError",
    "LedgerIntegrityError",
    "SchemaVersionError",
    "SnapshotConflictError",
]
