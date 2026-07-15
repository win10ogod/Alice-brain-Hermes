"""Stable domain and persistence errors for Alice-brain-Hermes."""

from __future__ import annotations


class AliceBrainHermesError(Exception):
    """Base class for errors raised by this independent runtime."""


class DomainInvariantError(AliceBrainHermesError, ValueError):
    """Raised when an event would violate a deterministic domain invariant."""


class DomainCapacityError(DomainInvariantError):
    """Raised when a bounded working set cannot safely admit another item."""


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


class RuntimeOwnershipError(AliceBrainHermesError):
    """Base class for runtime-home ownership and lifecycle failures."""


class RuntimeOwnedError(RuntimeOwnershipError):
    """Raised when another process holds the runtime-home lease."""


class SchedulerShutdownError(RuntimeOwnershipError, RuntimeError):
    """Raised when a continuous writer cannot prove it has stopped."""


class BridgeError(AliceBrainHermesError):
    """Base class for stable bridge-stream failures."""


class BridgeBindingError(BridgeError):
    """Raised when a stream is not bound to the requested server identity."""


class BridgeClosedError(BridgeError):
    """Raised when a closed or abandoned stream receives another record."""


class BridgeCleanClosedError(BridgeClosedError):
    """Raised when attach targets a stream with a proven clean close."""


class BridgeAbandonedError(BridgeClosedError):
    """Raised when attach targets an explicitly abandoned stream."""


class CaptureGapRequiredError(BridgeError):
    """Raised when an observation skips the persisted capture cursor."""


class CaptureSequenceError(BridgeError):
    """Raised when a record does not begin exactly at the capture cursor."""


class IdempotencyConflictError(BridgeError):
    """Raised when one bridge key is reused with a changed immutable body."""


class FrameSizeError(BridgeError):
    """Raised before commit when a consciousness frame exceeds its budget."""


class ResponseSizeError(BridgeError):
    """Raised before commit when its exact success envelope cannot fit."""


class DaemonClientError(AliceBrainHermesError):
    """Base class for authenticated daemon client failures."""


class DaemonRpcError(DaemonClientError):
    """A stable structured error returned by the private daemon."""

    def __init__(self, code: str, message: str, data: object) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.data = data


__all__ = [
    "AliceBrainHermesError",
    "BridgeAbandonedError",
    "BridgeBindingError",
    "BridgeCleanClosedError",
    "BridgeClosedError",
    "BridgeError",
    "CaptureGapRequiredError",
    "CaptureSequenceError",
    "DaemonClientError",
    "DaemonRpcError",
    "DomainCapacityError",
    "DomainInvariantError",
    "EventConflictError",
    "ExpectedSequenceError",
    "FrameSizeError",
    "IdempotencyConflictError",
    "LedgerClosedError",
    "LedgerError",
    "LedgerIntegrityError",
    "ResponseSizeError",
    "RuntimeOwnedError",
    "RuntimeOwnershipError",
    "SchedulerShutdownError",
    "SchemaVersionError",
    "SnapshotConflictError",
]
