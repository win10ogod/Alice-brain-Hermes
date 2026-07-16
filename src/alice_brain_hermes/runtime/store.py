"""Append-only SQLite WAL event ledger with deterministic replay."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from alice_brain_hermes.core.action import ActionPhase
from alice_brain_hermes.core.events import EventEnvelope, new_event
from alice_brain_hermes.core.reducer import reduce_state
from alice_brain_hermes.core.state import (
    FRAME_PROJECTION_RECORD_BUDGET,
    STATE_SCHEMA_VERSION,
    BrainState,
)
from alice_brain_hermes.errors import (
    BridgeAbandonedError,
    BridgeBindingError,
    BridgeCleanClosedError,
    BridgeClosedError,
    CaptureGapRequiredError,
    CaptureSequenceError,
    DomainCapacityError,
    EventConflictError,
    ExpectedSequenceError,
    FrameSizeError,
    IdempotencyConflictError,
    LedgerClosedError,
    LedgerIntegrityError,
    ResponseSizeError,
    SchemaVersionError,
    SnapshotConflictError,
)
from alice_brain_hermes.ids import new_id, validate_id
from alice_brain_hermes.protocol.identity import (
    IDENTITY_NAMING_LEASE_SECONDS,
    IdentityChoiceV1,
    IdentityNameNormalizationError,
    IdentityNamingLeaseStatusV1,
    IdentityNamingLeaseV1,
    identity_name_normalization_key,
)
from alice_brain_hermes.protocol.models import (
    HOOK_EVENT_TYPES,
    MAX_BRIDGE_INTEGER,
    BrainProfileV1,
    BridgeCommitAckV1,
    BridgeCommitAckV2,
    BridgeGapV1,
    BridgeRecordV1,
    BridgeStreamState,
    ConsciousnessFrameV2,
    ConsciousnessFrameV3,
    FrameFreshnessV1,
    HermesObservationV1,
    ObservabilitySnapshotV1,
    validate_bridge_record_json,
    validate_observation,
    validate_observation_json,
)
from alice_brain_hermes.runtime.lease import RuntimeLease
from alice_brain_hermes.runtime.semantic_ingest import (
    HermesSpan,
    SemanticPlan,
    build_raw_event,
    build_semantic_plan,
    match_hermes_span,
    span_context_fingerprint,
)

SQLITE_SCHEMA_VERSION = 6
SEMANTIC_SCHEMA_VERSION = 1
MAX_HERMES_SPANS_PER_STREAM = 256
MAX_PERSISTED_OBSERVABILITY_COUNT = MAX_BRIDGE_INTEGER
MAX_PAGE_SIZE = 10_000
DEFAULT_MAX_FRAME_BYTES = 65_536
DEFAULT_MAX_ACK_BYTES = 4_194_304
_LEGACY_STATE_SCHEMA_VERSIONS = frozenset({1, 2, 3})
_SQLITE_BRIDGE_SCHEMA_VERSIONS = frozenset({3, 4, 5, SQLITE_SCHEMA_VERSION})
_SQLITE_RUNTIME_NAMES = (
    "runtime.db",
    "runtime.db-wal",
    "runtime.db-shm",
    "runtime.db-journal",
)

_SQLITE_MUTATION_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_SAVEPOINT,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_UPDATE,
    }
)
_SQLITE_MUTATING_NO_ARGUMENT_PRAGMAS = frozenset(
    {"incremental_vacuum", "optimize", "shrink_memory", "wal_checkpoint"}
)
_IDENTITY_WORKER_FAILURE_CODE = re.compile(
    r"^(?:invalid_structured_choice|llm_error\.[A-Za-z0-9_]{1,80})$"
)
_IDENTITY_ADAPTER_ID = "alice-brain-hermes-identity-v1"
_IDENTITY_PURPOSE = "identity_self_naming"


def _assert_safe_runtime_sqlite_paths(
    database: Path,
    authority: RuntimeLease,
) -> None:
    """Reject paths SQLite could follow or replace outside the owned home."""
    authority.assert_authority()
    for name in _SQLITE_RUNTIME_NAMES:
        candidate = database.parent / name
        try:
            candidate.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PermissionError(
                f"SQLite runtime path is not verifiable: {name}"
            ) from error
        unsafe = candidate.is_symlink() or not candidate.is_file()
        if unsafe:
            raise PermissionError(f"SQLite runtime path is unsafe: {name}")
    authority.assert_authority()


_CREATE_BASE_SCHEMA = """
CREATE TABLE schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE brains (
    brain_id TEXT PRIMARY KEY,
    next_sequence INTEGER NOT NULL CHECK (next_sequence >= 1)
) WITHOUT ROWID;

CREATE TABLE events (
    brain_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    event_id TEXT NOT NULL UNIQUE,
    body_fingerprint TEXT NOT NULL,
    envelope_fingerprint TEXT NOT NULL,
    envelope_json TEXT NOT NULL,
    PRIMARY KEY (brain_id, sequence),
    FOREIGN KEY (brain_id) REFERENCES brains(brain_id)
) WITHOUT ROWID;

CREATE INDEX events_event_id ON events(event_id);

CREATE TABLE snapshots (
    brain_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK (sequence >= 0),
    schema_version INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    state_json TEXT NOT NULL,
    PRIMARY KEY (brain_id, sequence)
) WITHOUT ROWID;

CREATE INDEX snapshots_latest ON snapshots(brain_id, sequence DESC);
"""

_CREATE_BRIDGE_STREAM_SCHEMA = """
CREATE TABLE bridge_stream (
    bridge_instance_id TEXT PRIMARY KEY,
    brain_id TEXT NOT NULL,
    server_actor_id TEXT NOT NULL,
    server_adapter_id TEXT NOT NULL,
    recovery_token_digest BLOB NOT NULL CHECK (
        typeof(recovery_token_digest) = 'blob'
        AND length(recovery_token_digest) = 32
    ),
    next_capture_seq INTEGER NOT NULL CHECK (next_capture_seq >= 1),
    status TEXT NOT NULL CHECK (status IN ('open', 'clean_closed', 'abandoned')),
    connected_nonce TEXT,
    disconnected_reason TEXT CHECK (disconnected_reason IN (
        'connection_eof', 'daemon_restart', 'clean_close', 'grace_abandonment'
    )),
    disconnected_at TEXT,
    last_seen TEXT NOT NULL,
    closed_final_seq INTEGER CHECK (closed_final_seq >= 0),
    FOREIGN KEY (brain_id) REFERENCES brains(brain_id)
) WITHOUT ROWID;
"""

_CREATE_BRIDGE_RECORD_V4_SCHEMA = """
CREATE TABLE bridge_record (
    bridge_instance_id TEXT NOT NULL,
    first_capture_seq INTEGER NOT NULL CHECK (first_capture_seq >= 1),
    last_capture_seq INTEGER NOT NULL CHECK (last_capture_seq >= first_capture_seq),
    record_kind TEXT NOT NULL CHECK (record_kind IN ('observation', 'gap')),
    record_fingerprint TEXT NOT NULL,
    record_json TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    ledger_sequence INTEGER NOT NULL CHECK (ledger_sequence >= 1),
    ack_json TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    PRIMARY KEY (bridge_instance_id, first_capture_seq),
    FOREIGN KEY (bridge_instance_id) REFERENCES bridge_stream(bridge_instance_id),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
) WITHOUT ROWID;

CREATE INDEX bridge_record_event ON bridge_record(event_id);
"""

_CREATE_PROFILE_SCHEMA = """
CREATE TABLE brain_profile (
    profile_key TEXT PRIMARY KEY,
    profile_fingerprint TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    brain_id TEXT NOT NULL UNIQUE,
    FOREIGN KEY (brain_id) REFERENCES brains(brain_id)
) WITHOUT ROWID;
"""

_CREATE_BRIDGE_RECORD_V5_SCHEMA = """
CREATE TABLE bridge_record (
    bridge_instance_id TEXT NOT NULL,
    first_capture_seq INTEGER NOT NULL CHECK (first_capture_seq >= 1),
    last_capture_seq INTEGER NOT NULL CHECK (last_capture_seq >= first_capture_seq),
    record_kind TEXT NOT NULL CHECK (record_kind IN ('observation', 'gap')),
    record_fingerprint TEXT NOT NULL,
    record_json TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    ledger_sequence INTEGER NOT NULL CHECK (ledger_sequence >= 1),
    semantic_status TEXT NOT NULL CHECK (semantic_status IN (
        'applied', 'not_applicable', 'gap', 'legacy_raw_only'
    )),
    semantic_complete INTEGER NOT NULL CHECK (semantic_complete IN (0, 1)),
    semantic_fingerprint TEXT NOT NULL CHECK (length(semantic_fingerprint) = 64),
    derived_event_count INTEGER NOT NULL CHECK (
        derived_event_count >= 0 AND derived_event_count <= 8
    ),
    derived_first_sequence INTEGER CHECK (derived_first_sequence >= 1),
    derived_last_sequence INTEGER CHECK (derived_last_sequence >= 1),
    ack_json TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    CHECK (
        (derived_event_count = 0
         AND derived_first_sequence IS NULL
         AND derived_last_sequence IS NULL)
        OR
        (derived_event_count > 0
         AND derived_first_sequence = ledger_sequence + 1
         AND derived_last_sequence = ledger_sequence + derived_event_count)
    ),
    PRIMARY KEY (bridge_instance_id, first_capture_seq),
    FOREIGN KEY (bridge_instance_id) REFERENCES bridge_stream(bridge_instance_id),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
) WITHOUT ROWID;
CREATE INDEX bridge_record_event ON bridge_record(event_id);

CREATE TABLE hermes_span (
    bridge_instance_id TEXT NOT NULL,
    span_kind TEXT NOT NULL CHECK (span_kind IN ('tool', 'api')),
    external_id TEXT NOT NULL CHECK (
        length(external_id) >= 1 AND length(external_id) <= 512
    ),
    occurrence_capture_seq INTEGER NOT NULL CHECK (occurrence_capture_seq >= 1),
    context_fingerprint TEXT NOT NULL CHECK (length(context_fingerprint) = 64),
    action_id TEXT CHECK (action_id IS NULL OR (
        length(action_id) >= 1 AND length(action_id) <= 512
    )),
    closed_capture_seq INTEGER CHECK (
        closed_capture_seq IS NULL OR closed_capture_seq >= occurrence_capture_seq
    ),
    PRIMARY KEY (
        bridge_instance_id, span_kind, external_id, occurrence_capture_seq
    ),
    FOREIGN KEY (bridge_instance_id) REFERENCES bridge_stream(bridge_instance_id)
) WITHOUT ROWID;

CREATE INDEX hermes_span_lookup ON hermes_span(
    bridge_instance_id, span_kind, external_id,
    closed_capture_seq, occurrence_capture_seq DESC
);

CREATE TABLE brain_observability (
    brain_id TEXT PRIMARY KEY,
    trace_complete INTEGER NOT NULL CHECK (trace_complete IN (0, 1)),
    semantic_complete INTEGER NOT NULL CHECK (semantic_complete IN (0, 1)),
    dropped_events INTEGER NOT NULL CHECK (dropped_events >= 0),
    semantic_records INTEGER NOT NULL CHECK (semantic_records >= 0),
    legacy_raw_only_records INTEGER NOT NULL CHECK (legacy_raw_only_records >= 0),
    semantic_gap_records INTEGER NOT NULL CHECK (semantic_gap_records >= 0),
    FOREIGN KEY (brain_id) REFERENCES brains(brain_id)
) WITHOUT ROWID;
"""

_CREATE_IDENTITY_SCHEMA = """
CREATE TABLE identity_name_registry (
    brain_id TEXT PRIMARY KEY,
    normalized_name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    source_event_id TEXT NOT NULL UNIQUE,
    FOREIGN KEY (brain_id) REFERENCES brains(brain_id),
    FOREIGN KEY (source_event_id) REFERENCES events(event_id)
) WITHOUT ROWID;

CREATE UNIQUE INDEX identity_name_normalized
ON identity_name_registry(normalized_name);

CREATE TABLE identity_naming_lease (
    lease_id TEXT PRIMARY KEY,
    brain_id TEXT NOT NULL,
    request_sequence INTEGER NOT NULL CHECK (request_sequence >= 1),
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'completed', 'failed', 'superseded'
    )),
    requested_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    request_event_id TEXT NOT NULL UNIQUE,
    choice_fingerprint TEXT CHECK (
        choice_fingerprint IS NULL OR length(choice_fingerprint) = 64
    ),
    choice_json TEXT,
    failure_code TEXT,
    terminal_event_id TEXT UNIQUE,
    terminal_at TEXT,
    CHECK ((choice_fingerprint IS NULL) = (choice_json IS NULL)),
    CHECK (
        (status = 'pending'
         AND choice_json IS NULL
         AND failure_code IS NULL
         AND terminal_event_id IS NULL
         AND terminal_at IS NULL)
        OR
        (status = 'completed'
         AND choice_json IS NOT NULL
         AND failure_code IS NULL
         AND terminal_event_id IS NOT NULL
         AND terminal_at IS NOT NULL)
        OR
        (status = 'failed'
         AND failure_code IS NOT NULL
         AND terminal_event_id IS NOT NULL
         AND terminal_at IS NOT NULL)
        OR
        (status = 'superseded'
         AND choice_json IS NULL
         AND failure_code IN ('expired', 'identity_already_named')
         AND ((failure_code = 'expired' AND terminal_event_id IS NULL)
              OR (failure_code = 'identity_already_named'
                  AND terminal_event_id IS NOT NULL))
         AND terminal_at IS NOT NULL)
    ),
    FOREIGN KEY (brain_id) REFERENCES brains(brain_id),
    FOREIGN KEY (request_event_id) REFERENCES events(event_id),
    FOREIGN KEY (terminal_event_id) REFERENCES events(event_id)
) WITHOUT ROWID;

CREATE UNIQUE INDEX identity_naming_pending ON identity_naming_lease(brain_id)
WHERE status = 'pending';
"""

_CREATE_V2_SCHEMA = _CREATE_BASE_SCHEMA
_CREATE_V4_SCHEMA = (
    _CREATE_BASE_SCHEMA
    + _CREATE_BRIDGE_STREAM_SCHEMA
    + _CREATE_BRIDGE_RECORD_V4_SCHEMA
    + _CREATE_PROFILE_SCHEMA
)
_CREATE_BRIDGE_SCHEMA = (
    _CREATE_BRIDGE_STREAM_SCHEMA
    + _CREATE_BRIDGE_RECORD_V5_SCHEMA
    + _CREATE_PROFILE_SCHEMA
)
_CREATE_V5_SCHEMA = _CREATE_BASE_SCHEMA + _CREATE_BRIDGE_SCHEMA
_CREATE_SCHEMA = _CREATE_V5_SCHEMA + _CREATE_IDENTITY_SCHEMA


@dataclass(frozen=True, slots=True)
class BridgeCommitResult:
    ack: BridgeCommitAckV2
    successor: BrainState | None


@dataclass(frozen=True, slots=True)
class BridgeAbandonResult:
    stream: BridgeStreamState
    successor: BrainState | None


@dataclass(frozen=True, slots=True)
class BrainResolveResult:
    brain_id: str
    created: bool
    foundation: EventEnvelope | None = None


IdentityNamingTerminalStatus = Literal["completed", "failed", "superseded"]


@dataclass(frozen=True, slots=True)
class IdentityNamingClaimResult:
    lease: IdentityNamingLeaseV1 | None
    successor: BrainState | None


@dataclass(frozen=True, slots=True)
class IdentityNamingTerminalResult:
    status: IdentityNamingTerminalStatus
    successor: BrainState | None


def _schema_statements() -> Iterator[str]:
    yield from _statements(_CREATE_SCHEMA)


def _statements(script: str) -> Iterator[str]:
    for statement in script.split(";"):
        if stripped := statement.strip():
            yield stripped


def _normalized_sql(statement: str) -> str:
    return " ".join(statement.split()).casefold()


def _bridge_recovery_token_digest(value: object) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("recovery_token must be one 256-bit lowercase hex token")
    return hashlib.sha256(value.encode("ascii")).digest()


def _schema_objects(script: str) -> dict[str, tuple[str, str]]:
    objects: dict[str, tuple[str, str]] = {}
    pattern = re.compile(
        r"^CREATE\s+(?:UNIQUE\s+)?(TABLE|INDEX)\s+"
        r"([a-zA-Z_][a-zA-Z0-9_]*)\b",
        re.IGNORECASE,
    )
    for statement in _statements(script):
        match = pattern.match(statement)
        if match is None:
            raise RuntimeError("schema contract contains an unsupported statement")
        kind, name = match.groups()
        objects[name] = (kind.casefold(), _normalized_sql(statement))
    return objects


class _RestrictedSQLiteConnection:
    """The exact SQLite subset the ledger is allowed to use after startup."""

    __slots__ = (
        "__close",
        "__commit",
        "__execute",
        "__in_transaction",
        "__rollback",
    )

    def __init__(self, connection: Any) -> None:
        # Store only the five bound capabilities needed by the ledger.  In
        # particular, the facade is not a sqlite3.Connection and cannot be a
        # backup/deserialize target or expose extension/authorizer controls.
        self.__execute = connection.execute
        self.__commit = connection.commit
        self.__rollback = connection.rollback
        self.__close = connection.close
        self.__in_transaction = lambda: connection.in_transaction

    @property
    def in_transaction(self) -> bool:
        return bool(self.__in_transaction())

    def execute(self, statement: str, parameters: Any = ()) -> sqlite3.Cursor:
        return self.__execute(statement, parameters)

    def commit(self) -> None:
        self.__commit()

    def rollback(self) -> None:
        self.__rollback()

    def close(self) -> None:
        self.__close()


class SQLiteLedger:
    """A thread-safe connection facade over a per-brain append-only ledger."""

    def __init__(
        self,
        path: Path,
        connection: _RestrictedSQLiteConnection,
        *,
        authority: RuntimeLease | None = None,
    ) -> None:
        self.path = path
        self._connection = connection
        self._lease_registry = authority
        self._creator_pid = os.getpid()
        self._lock = threading.RLock()
        self._closed = False
        self._connection_closed = False
        self._startup_audited_final_states: dict[str, BrainState] = {}
        self._mutation_seal_installed = False
        self._mutation_seal_poisoned = False
        self._mutation_seal_data_version: int | None = None
        self._authorized_transaction_thread: int | None = None

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        authority: RuntimeLease | None = None,
        owner_sink: Callable[[SQLiteLedger], None] | None = None,
    ) -> SQLiteLedger:
        """Open or initialize a WAL ledger, rejecting unknown schemas."""
        database = Path(path)
        if authority is None:
            database.parent.mkdir(parents=True, exist_ok=True)
        else:
            database = cls._validate_runtime_paths(database, authority)
        return cls._open_database(
            database,
            authority=authority,
            owner_sink=owner_sink,
        )

    @classmethod
    def _validate_runtime_paths(
        cls,
        path: str | Path,
        authority: RuntimeLease,
    ) -> Path:
        if not isinstance(authority, RuntimeLease):
            raise TypeError("authority must be RuntimeLease")
        database = Path(path)
        home = authority.assert_authority()
        if (
            database.name != "runtime.db"
            or database.parent.resolve(strict=True) != home
        ):
            raise PermissionError("SQLite path does not match runtime authority")
        database = home / "runtime.db"
        _assert_safe_runtime_sqlite_paths(database, authority)
        return database

    def _adopt_runtime_authority(self, authority: RuntimeLease) -> None:
        """Bind a custom factory ledger to the already-held runtime lease."""
        self._assert_creator_process()
        if not isinstance(authority, RuntimeLease):
            raise TypeError("authority must be RuntimeLease")
        with self._lock:
            if self._closed or self._lease_registry is not None:
                raise RuntimeError("SQLite ledger authority is already established")
            self._validate_runtime_paths(self.path, authority)
            authority.register_resource(self)
            self._lease_registry = authority
            authority.assert_authority()

    @classmethod
    def _open_database(
        cls,
        database: Path,
        *,
        authority: RuntimeLease | None = None,
        owner_sink: Callable[[SQLiteLedger], None] | None = None,
    ) -> SQLiteLedger:
        connection: Any | None = None
        restricted_connection: _RestrictedSQLiteConnection | None = None
        ledger: SQLiteLedger | None = None
        try:
            if authority is not None:
                _assert_safe_runtime_sqlite_paths(database, authority)
            connection = sqlite3.connect(
                database,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
                cached_statements=0,
            )
            restricted_connection = _RestrictedSQLiteConnection(connection)
            connection.row_factory = sqlite3.Row
            ledger = cls(
                database,
                restricted_connection,
                authority=authority,
            )
            if authority is not None:
                authority.register_resource(ledger)
            if owner_sink is not None:
                owner_sink(ledger)
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA foreign_keys = ON")
            if authority is not None:
                authority.assert_authority()
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            if (
                journal_mode is None
                or not isinstance(journal_mode[0], str)
                or journal_mode[0].casefold() != "wal"
            ):
                raise LedgerIntegrityError("SQLite journal mode is not WAL")
            if authority is not None:
                authority.assert_authority()
            connection.execute("PRAGMA synchronous = FULL")
            startup_data_version = ledger._read_mutation_data_version()
            ledger._initialize_schema()
            connection.set_authorizer(ledger._sqlite_authorizer)
            ledger._install_mutation_seal(expected_data_version=startup_data_version)
            if authority is not None:
                authority.assert_authority()
        except BaseException as primary_error:
            traceback = primary_error.__traceback__
            try:
                if ledger is not None:
                    ledger.close()
                elif restricted_connection is not None:
                    restricted_connection.close()
                elif connection is not None:
                    connection.close()
            except BaseException as cleanup_error:
                raise primary_error.with_traceback(traceback) from cleanup_error
            raise
        if ledger is None:
            raise AssertionError("SQLite ledger construction did not complete")
        return ledger

    def _initialize_schema(self) -> None:
        with self._transaction(immediate=True):
            version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                for statement in _schema_statements():
                    self._connection.execute(statement)
                self._connection.execute(
                    "INSERT INTO schema_metadata(key, value) VALUES (?, ?)",
                    ("schema_version", str(SQLITE_SCHEMA_VERSION)),
                )
                self._connection.execute(
                    f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}"
                )
                self._startup_audited_final_states = self._validate_schema_contract(
                    SQLITE_SCHEMA_VERSION, validate_data=True
                )
                self._validate_schema_metadata(SQLITE_SCHEMA_VERSION)
                return
            if version == 2:
                self._validate_schema_metadata(2)
                final_states = self._validate_schema_contract(2, validate_data=True)
                try:
                    for statement in _statements(_CREATE_BRIDGE_SCHEMA):
                        self._connection.execute(statement)
                    self._rebuild_observability_in_transaction(
                        final_states=final_states
                    )
                    self._install_identity_schema(final_states=final_states)
                except SchemaVersionError:
                    raise
                except Exception as error:
                    raise SchemaVersionError(
                        "SQLite v2 to v6 migration failed"
                    ) from error
                self._connection.execute(
                    "UPDATE schema_metadata SET value = ? WHERE key = ?",
                    (str(SQLITE_SCHEMA_VERSION), "schema_version"),
                )
                self._connection.execute(
                    f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}"
                )
                self._startup_audited_final_states = self._validate_schema_contract(
                    SQLITE_SCHEMA_VERSION, validate_data=True
                )
                self._validate_schema_metadata(SQLITE_SCHEMA_VERSION)
                return
            if version in {3, 4}:
                self._validate_schema_metadata(version)
                self._startup_audited_final_states = self._validate_schema_contract(
                    version, validate_data=True
                )
                try:
                    self._migrate_bridge_schema_to_v5(
                        final_states=self._startup_audited_final_states
                    )
                    self._install_identity_schema(
                        final_states=self._startup_audited_final_states
                    )
                except SchemaVersionError:
                    raise
                except Exception as error:
                    raise SchemaVersionError(
                        f"SQLite v{version} to v6 migration failed"
                    ) from error
                self._connection.execute(
                    "UPDATE schema_metadata SET value = ? WHERE key = ?",
                    (str(SQLITE_SCHEMA_VERSION), "schema_version"),
                )
                self._connection.execute(
                    f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}"
                )
                self._startup_audited_final_states = self._validate_schema_contract(
                    SQLITE_SCHEMA_VERSION, validate_data=True
                )
                self._validate_schema_metadata(SQLITE_SCHEMA_VERSION)
                return
            if version == 5:
                self._validate_schema_metadata(5)
                self._startup_audited_final_states = self._validate_schema_contract(
                    5, validate_data=True
                )
                try:
                    self._install_identity_schema(
                        final_states=self._startup_audited_final_states
                    )
                except SchemaVersionError:
                    raise
                except Exception as error:
                    raise SchemaVersionError(
                        "SQLite v5 to v6 migration failed"
                    ) from error
                self._connection.execute(
                    "UPDATE schema_metadata SET value = ? WHERE key = ?",
                    (str(SQLITE_SCHEMA_VERSION), "schema_version"),
                )
                self._connection.execute(
                    f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}"
                )
                self._startup_audited_final_states = self._validate_schema_contract(
                    SQLITE_SCHEMA_VERSION, validate_data=True
                )
                self._validate_schema_metadata(SQLITE_SCHEMA_VERSION)
                return
            if version != SQLITE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"unsupported SQLite schema version {version}; "
                    f"expected {SQLITE_SCHEMA_VERSION}"
                )
            self._startup_audited_final_states = self._validate_schema_contract(
                SQLITE_SCHEMA_VERSION, validate_data=True
            )
            self._validate_schema_metadata(SQLITE_SCHEMA_VERSION)

    @staticmethod
    def _normalized_identity_name(name: str) -> str:
        return identity_name_normalization_key(name)

    def _last_identity_name_event_in_transaction(
        self,
        brain_id: str,
        name: str,
    ) -> EventEnvelope:
        rows = self._connection.execute(
            "SELECT event_id, brain_id, sequence, body_fingerprint, "
            "envelope_fingerprint, envelope_json FROM events "
            "WHERE brain_id = ? ORDER BY sequence DESC",
            (brain_id,),
        ).fetchall()
        for row in rows:
            event = self._decode_event(row)
            if (
                event.event_type in {"brain.created", "identity.named"}
                and event.payload.get("name") == name
            ):
                return event
        raise LedgerIntegrityError("named identity lacks its source event")

    def _install_identity_schema(
        self,
        *,
        final_states: Mapping[str, BrainState],
    ) -> None:
        for statement in _statements(_CREATE_IDENTITY_SCHEMA):
            self._connection.execute(statement)
        for brain_id, state in final_states.items():
            if state.identity.name is None:
                continue
            source = self._last_identity_name_event_in_transaction(
                brain_id,
                state.identity.name,
            )
            try:
                self._connection.execute(
                    "INSERT INTO identity_name_registry("
                    "brain_id, normalized_name, display_name, source_event_id) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        brain_id,
                        self._normalized_identity_name(state.identity.name),
                        state.identity.name,
                        source.event_id,
                    ),
                )
            except IdentityNameNormalizationError as error:
                raise SchemaVersionError(
                    "identity name migration violates the stable Unicode 14.0 "
                    "normalization boundary"
                ) from error
            except sqlite3.IntegrityError as error:
                raise SchemaVersionError(
                    "identity name migration has a normalized-name collision"
                ) from error

    @staticmethod
    def _identity_timestamp(value: object, *, field: str) -> datetime:
        if not isinstance(value, str):
            raise LedgerIntegrityError(f"identity {field} timestamp is invalid")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise LedgerIntegrityError(
                f"identity {field} timestamp is invalid"
            ) from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise LedgerIntegrityError(f"identity {field} timestamp is naive")
        exact = parsed.astimezone(UTC)
        if exact.isoformat() != value:
            raise LedgerIntegrityError(
                f"identity {field} timestamp is not canonical UTC"
            )
        return exact

    @staticmethod
    def _identity_choice_fingerprint(choice: IdentityChoiceV1) -> str:
        return hashlib.sha256(choice.canonical_json().encode("utf-8")).hexdigest()

    def _identity_status_from_row(
        self,
        row: sqlite3.Row,
    ) -> IdentityNamingLeaseStatusV1:
        choice: IdentityChoiceV1 | None = None
        if row["choice_json"] is not None:
            try:
                choice = IdentityChoiceV1.model_validate_json(row["choice_json"])
            except Exception as error:
                raise LedgerIntegrityError(
                    "identity naming choice is invalid"
                ) from error
            if (
                choice.canonical_json() != row["choice_json"]
                or self._identity_choice_fingerprint(choice)
                != row["choice_fingerprint"]
            ):
                raise LedgerIntegrityError(
                    "identity naming choice integrity check failed"
                )
        elif row["choice_fingerprint"] is not None:
            raise LedgerIntegrityError("identity naming choice fingerprint is orphaned")
        try:
            return IdentityNamingLeaseStatusV1(
                lease_id=row["lease_id"],
                brain_id=row["brain_id"],
                state_sequence=int(row["request_sequence"]),
                status=row["status"],
                requested_at=self._identity_timestamp(
                    row["requested_at"], field="requested_at"
                ),
                expires_at=self._identity_timestamp(
                    row["expires_at"], field="expires_at"
                ),
                choice=choice,
                failure_code=row["failure_code"],
                terminal_event_id=row["terminal_event_id"],
                terminal_at=(
                    None
                    if row["terminal_at"] is None
                    else self._identity_timestamp(row["terminal_at"], field="terminal")
                ),
            )
        except Exception as error:
            raise LedgerIntegrityError(
                "identity naming lease row is invalid"
            ) from error

    def _validate_identity_rows_in_transaction(
        self,
        historical_states: Mapping[str, Mapping[int, BrainState]],
        final_states: Mapping[str, BrainState],
    ) -> None:
        registry_rows = self._connection.execute(
            "SELECT brain_id, normalized_name, display_name, source_event_id "
            "FROM identity_name_registry ORDER BY brain_id"
        ).fetchall()
        registry = {row["brain_id"]: row for row in registry_rows}
        expected_named = {
            brain_id: state.identity.name
            for brain_id, state in final_states.items()
            if state.identity.name is not None
        }
        if set(registry) != set(expected_named):
            raise LedgerIntegrityError("identity name registry coverage is invalid")
        normalized_names = [row["normalized_name"] for row in registry_rows]
        if len(set(normalized_names)) != len(normalized_names):
            raise LedgerIntegrityError(
                "identity name registry normalized names are not unique"
            )
        for brain_id, name in expected_named.items():
            if name is None:
                raise AssertionError("named identity unexpectedly became null")
            row = registry[brain_id]
            source = self._last_identity_name_event_in_transaction(brain_id, name)
            if (
                row["display_name"] != name
                or row["normalized_name"] != self._normalized_identity_name(name)
                or row["source_event_id"] != source.event_id
            ):
                raise LedgerIntegrityError(
                    "identity name registry is not replay-derived"
                )

        reserved_event_ids: set[str] = set()
        event_cursor = self._connection.execute(
            "SELECT event_id, brain_id, sequence, body_fingerprint, "
            "envelope_fingerprint, envelope_json FROM events "
            "ORDER BY brain_id, sequence"
        )
        while event_rows := event_cursor.fetchmany(512):
            for event_row in event_rows:
                event = self._decode_event(event_row)
                if (
                    event.adapter_id == _IDENTITY_ADAPTER_ID
                    or event.payload.get("purpose") == _IDENTITY_PURPOSE
                ):
                    reserved_event_ids.add(event.event_id)

        covered_event_leases: dict[str, str] = {}

        def cover(event: EventEnvelope, lease_id: str) -> None:
            if event.event_id in covered_event_leases:
                raise LedgerIntegrityError(
                    "identity naming evidence is covered by multiple leases"
                )
            covered_event_leases[event.event_id] = lease_id

        lease_rows = self._connection.execute(
            "SELECT lease_id, brain_id, request_sequence, status, requested_at, "
            "expires_at, request_event_id, choice_fingerprint, choice_json, "
            "failure_code, terminal_event_id, terminal_at "
            "FROM identity_naming_lease ORDER BY lease_id"
        ).fetchall()
        for row in lease_rows:
            status = self._identity_status_from_row(row)
            if status.brain_id not in final_states:
                raise LedgerIntegrityError("identity naming lease brain is missing")
            predecessor = historical_states.get(status.brain_id, {}).get(
                status.state_sequence - 1
            )
            if predecessor is None or predecessor.identity.name is not None:
                raise LedgerIntegrityError(
                    "identity naming request did not follow an unnamed state"
                )
            request_row = self._connection.execute(
                "SELECT event_id, brain_id, sequence, body_fingerprint, "
                "envelope_fingerprint, envelope_json FROM events "
                "WHERE event_id = ?",
                (row["request_event_id"],),
            ).fetchone()
            if request_row is None:
                raise LedgerIntegrityError("identity naming request event is missing")
            request = self._decode_event(request_row)
            if (
                request.brain_id != status.brain_id
                or request.actor_id != status.brain_id
                or request.sequence != status.state_sequence
                or request.event_type != "cognition.requested"
                or request.adapter_id != _IDENTITY_ADAPTER_ID
                or request.payload
                != {
                    "schema_version": 1,
                    "purpose": _IDENTITY_PURPOSE,
                    "lease_id": status.lease_id,
                    "requested_at": status.requested_at.isoformat(),
                    "expires_at": status.expires_at.isoformat(),
                }
            ):
                raise LedgerIntegrityError(
                    "identity naming request event does not match its lease"
                )
            cover(request, status.lease_id)
            if status.status == "failed":
                failure_code = status.failure_code
                if failure_code is None:
                    raise LedgerIntegrityError(
                        "identity naming failure lost its reason"
                    )
                valid_failure = (
                    _IDENTITY_WORKER_FAILURE_CODE.fullmatch(failure_code) is not None
                    if status.choice is None
                    else failure_code == "name_conflict"
                )
                if not valid_failure:
                    raise LedgerIntegrityError(
                        "identity naming failure origin is invalid"
                    )
            terminal_id = status.terminal_event_id
            if status.status == "superseded":
                if status.failure_code == "expired":
                    if terminal_id is not None:
                        raise LedgerIntegrityError(
                            "expired identity naming lease has a terminal event"
                        )
                    continue
                if terminal_id is None:
                    raise LedgerIntegrityError(
                        "named identity supersession lacks its source event"
                    )
                source_row = self._connection.execute(
                    "SELECT event_id, brain_id, sequence, body_fingerprint, "
                    "envelope_fingerprint, envelope_json FROM events "
                    "WHERE event_id = ?",
                    (terminal_id,),
                ).fetchone()
                if source_row is None:
                    raise LedgerIntegrityError(
                        "named identity supersession source event is missing"
                    )
                source = self._decode_event(source_row)
                source_name = source.payload.get("name")
                source_state = historical_states.get(status.brain_id, {}).get(
                    source.sequence or -1
                )
                if (
                    source.brain_id != status.brain_id
                    or source.sequence is None
                    or source.sequence <= status.state_sequence
                    or source.event_type not in {"brain.created", "identity.named"}
                    or not isinstance(source_name, str)
                    or source_state is None
                    or source_state.identity.name != source_name
                ):
                    raise LedgerIntegrityError(
                        "named identity supersession source is invalid"
                    )
                continue
            if terminal_id is None:
                continue
            terminal_row = self._connection.execute(
                "SELECT event_id, brain_id, sequence, body_fingerprint, "
                "envelope_fingerprint, envelope_json FROM events "
                "WHERE event_id = ?",
                (terminal_id,),
            ).fetchone()
            if terminal_row is None:
                raise LedgerIntegrityError("identity naming terminal event is missing")
            terminal = self._decode_event(terminal_row)
            if (
                terminal.brain_id != status.brain_id
                or terminal.actor_id != status.brain_id
                or terminal.adapter_id != _IDENTITY_ADAPTER_ID
                or terminal.sequence is None
                or terminal.sequence <= status.state_sequence
            ):
                raise LedgerIntegrityError(
                    "identity naming terminal event changed lease provenance"
                )
            if status.status == "completed":
                if terminal.sequence < 3:
                    raise LedgerIntegrityError(
                        "completed identity naming chain is truncated"
                    )
                causal_rows = self._connection.execute(
                    "SELECT event_id, brain_id, sequence, body_fingerprint, "
                    "envelope_fingerprint, envelope_json FROM events "
                    "WHERE brain_id = ? AND sequence IN (?, ?) "
                    "ORDER BY sequence",
                    (
                        status.brain_id,
                        terminal.sequence - 2,
                        terminal.sequence - 1,
                    ),
                ).fetchall()
                if len(causal_rows) != 2:
                    raise LedgerIntegrityError(
                        "completed identity naming chain is missing"
                    )
                completed, deliberated = (
                    self._decode_event(causal_rows[0]),
                    self._decode_event(causal_rows[1]),
                )
                if status.choice is None or status.terminal_at is None:
                    raise LedgerIntegrityError(
                        "completed identity naming status lost its evidence"
                    )
                terminal_at = status.terminal_at.isoformat()
                choice_fingerprint = self._identity_choice_fingerprint(status.choice)
                if (
                    terminal.event_type != "identity.named"
                    or completed.event_type != "cognition.completed"
                    or deliberated.event_type != "c1.deliberated"
                    or completed.actor_id != status.brain_id
                    or deliberated.actor_id != status.brain_id
                    or completed.adapter_id != _IDENTITY_ADAPTER_ID
                    or deliberated.adapter_id != _IDENTITY_ADAPTER_ID
                    or completed.payload
                    != {
                        "schema_version": 1,
                        "purpose": _IDENTITY_PURPOSE,
                        "lease_id": status.lease_id,
                        "choice_fingerprint": choice_fingerprint,
                        "structured": True,
                        "terminal_at": terminal_at,
                    }
                    or deliberated.payload
                    != {
                        "schema_version": 1,
                        "purpose": _IDENTITY_PURPOSE,
                        "lease_id": status.lease_id,
                        "name": status.choice.name,
                        "reason": status.choice.reason,
                        "source_event_id": completed.event_id,
                        "terminal_at": terminal_at,
                    }
                    or terminal.payload
                    != {
                        "schema_version": 1,
                        "purpose": _IDENTITY_PURPOSE,
                        "lease_id": status.lease_id,
                        "name": status.choice.name,
                        "reason": status.choice.reason,
                        "source_event_id": deliberated.event_id,
                        "terminal_at": terminal_at,
                    }
                ):
                    raise LedgerIntegrityError(
                        "completed identity naming event does not match its choice"
                    )
                cover(completed, status.lease_id)
                cover(deliberated, status.lease_id)
                cover(terminal, status.lease_id)
            else:
                if (
                    status.status != "failed"
                    or status.failure_code is None
                    or status.terminal_at is None
                ):
                    raise LedgerIntegrityError(
                        "identity naming terminal status is invalid"
                    )
                failure_payload: dict[str, object] = {
                    "schema_version": 1,
                    "purpose": _IDENTITY_PURPOSE,
                    "lease_id": status.lease_id,
                    "failure_code": status.failure_code,
                    "terminal_at": status.terminal_at.isoformat(),
                }
                if status.choice is not None:
                    failure_payload["choice_fingerprint"] = (
                        self._identity_choice_fingerprint(status.choice)
                    )
                if (
                    terminal.event_type != "cognition.failed"
                    or terminal.payload != failure_payload
                ):
                    raise LedgerIntegrityError(
                        "failed identity naming event does not match its lease"
                    )
                cover(terminal, status.lease_id)

        if set(covered_event_leases) != reserved_event_ids:
            raise LedgerIntegrityError(
                "identity naming reserved evidence is not bijectively lease-bound"
            )

    @staticmethod
    def _legacy_semantic_fingerprint(
        *,
        record_fingerprint: str,
        raw_event_id: str,
        raw_event_sequence: int,
        through_capture_seq: int,
    ) -> str:
        encoded = json.dumps(
            {
                "record_fingerprint": record_fingerprint,
                "raw_event_id": raw_event_id,
                "raw_event_sequence": raw_event_sequence,
                "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
                "semantic_status": "legacy_raw_only",
                "through_capture_seq": through_capture_seq,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _migrate_bridge_schema_to_v5(
        self, *, final_states: Mapping[str, BrainState]
    ) -> None:
        """Rewrite verified v3/v4 ACKs without inventing semantic backfill."""
        rows = self._connection.execute(
            "SELECT record.bridge_instance_id, record.first_capture_seq, "
            "record.last_capture_seq, record.record_kind, "
            "record.record_fingerprint, record.record_json, record.event_id, "
            "record.ledger_sequence, record.ack_json, record.accepted_at, "
            "stream.brain_id AS brain_id FROM bridge_record AS record "
            "JOIN bridge_stream AS stream ON stream.bridge_instance_id = "
            "record.bridge_instance_id ORDER BY stream.brain_id, "
            "record.ledger_sequence"
        ).fetchall()
        covered_raw_event_ids = {str(row["event_id"]) for row in rows}
        unbounded_gap_sequences: dict[str, list[int]] = {}
        event_cursor = self._connection.execute(
            "SELECT event_id, brain_id, sequence, body_fingerprint, "
            "envelope_fingerprint, envelope_json FROM events "
            "ORDER BY brain_id, sequence"
        )
        while event_rows := event_cursor.fetchmany(512):
            for event_row in event_rows:
                event = self._decode_event(event_row)
                if (
                    event.event_id not in covered_raw_event_ids
                    and event.event_type in {"semantic.gap", "trace.gap"}
                    and event.sequence is not None
                ):
                    unbounded_gap_sequences.setdefault(event.brain_id, []).append(
                        event.sequence
                    )
        migrated_rows: list[tuple[object, ...]] = []
        record_health: dict[str, dict[str, int]] = {}
        for row in rows:
            brain_id = row["brain_id"]
            health = record_health.setdefault(
                brain_id,
                {
                    "records": 0,
                    "legacy": 0,
                    "gaps": 0,
                    "dropped": 0,
                    "unbounded": 0,
                },
            )
            health["unbounded"] = bisect_right(
                unbounded_gap_sequences.get(str(brain_id), []),
                int(row["ledger_sequence"]),
            )
            health["records"] += 1
            health["legacy"] += 1
            if row["record_kind"] == "gap":
                record = validate_bridge_record_json(row["record_json"])
                if not isinstance(record, BridgeGapV1):
                    raise LedgerIntegrityError(
                        "legacy gap row does not contain a typed bridge gap"
                    )
                health["gaps"] += 1
                health["dropped"] += record.dropped_count
            legacy = BridgeCommitAckV1.model_validate_json(row["ack_json"])
            frame_values = legacy.frame.model_dump(mode="python")
            frame_values["schema_version"] = 3
            frame_values["aggregate_semantic_complete"] = False
            frame_values["semantic_schema_version"] = SEMANTIC_SCHEMA_VERSION
            frame_values["semantic_evidence"] = {
                "schema_version": SEMANTIC_SCHEMA_VERSION,
                "semantic_records": health["records"],
                "legacy_raw_only_records": health["legacy"],
                "semantic_gap_records": health["gaps"] + health["unbounded"],
                "dropped_events": health["dropped"],
            }
            frame = ConsciousnessFrameV3.model_validate(frame_values, strict=True)
            semantic_fingerprint = self._legacy_semantic_fingerprint(
                record_fingerprint=row["record_fingerprint"],
                raw_event_id=row["event_id"],
                raw_event_sequence=int(row["ledger_sequence"]),
                through_capture_seq=int(row["last_capture_seq"]),
            )
            migrated_ack = BridgeCommitAckV2(
                record_fingerprint=row["record_fingerprint"],
                raw_event_id=row["event_id"],
                raw_event_sequence=int(row["ledger_sequence"]),
                derived_event_ids=(),
                derived_event_count=0,
                last_event_sequence=int(row["ledger_sequence"]),
                semantic_status="legacy_raw_only",
                semantic_complete=False,
                semantic_fingerprint=semantic_fingerprint,
                frame=frame,
                through_capture_seq=int(row["last_capture_seq"]),
            )
            migrated_rows.append(
                (
                    row["bridge_instance_id"],
                    row["first_capture_seq"],
                    row["last_capture_seq"],
                    row["record_kind"],
                    row["record_fingerprint"],
                    row["record_json"],
                    row["event_id"],
                    row["ledger_sequence"],
                    "legacy_raw_only",
                    0,
                    semantic_fingerprint,
                    0,
                    None,
                    None,
                    migrated_ack.canonical_json(),
                    row["accepted_at"],
                )
            )
        self._connection.execute("DROP INDEX bridge_record_event")
        self._connection.execute("ALTER TABLE bridge_record RENAME TO bridge_record_v4")
        for statement in _statements(_CREATE_BRIDGE_RECORD_V5_SCHEMA):
            self._connection.execute(statement)
        for values in migrated_rows:
            self._connection.execute(
                "INSERT INTO bridge_record("
                "bridge_instance_id, first_capture_seq, last_capture_seq, "
                "record_kind, record_fingerprint, record_json, event_id, "
                "ledger_sequence, semantic_status, semantic_complete, "
                "semantic_fingerprint, derived_event_count, "
                "derived_first_sequence, derived_last_sequence, ack_json, "
                "accepted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
        self._connection.execute("DROP TABLE bridge_record_v4")
        self._rebuild_observability_in_transaction(final_states=final_states)

    def _validate_schema_metadata(self, version: int) -> None:
        expected = [("schema_version", str(version))]
        try:
            rows = self._connection.execute(
                "SELECT key, value FROM schema_metadata ORDER BY key"
            ).fetchall()
        except sqlite3.DatabaseError as error:
            raise SchemaVersionError(
                "SQLite schema metadata is missing or unreadable"
            ) from error
        actual = [(row["key"], row["value"]) for row in rows]
        if actual != expected:
            raise SchemaVersionError(
                f"SQLite v{version} schema metadata does not match the exact contract"
            )

    def _validate_schema_contract(
        self, version: int, *, validate_data: bool
    ) -> dict[str, BrainState]:
        expected = _schema_objects(
            _CREATE_V2_SCHEMA
            if version == 2
            else _CREATE_V4_SCHEMA
            if version in {3, 4}
            else _CREATE_V5_SCHEMA
            if version == 5
            else _CREATE_SCHEMA
        )
        rows = self._connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%'",
        ).fetchall()
        actual = {
            row["name"]: (row["type"], _normalized_sql(row["sql"] or ""))
            for row in rows
        }
        if actual != expected:
            raise SchemaVersionError(
                f"SQLite v{version} structure does not match the exact contract"
            )
        quick = self._connection.execute("PRAGMA quick_check").fetchall()
        if [row[0] for row in quick] != ["ok"]:
            raise SchemaVersionError(
                f"SQLite v{version} integrity check did not return ok"
            )
        if self._connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise SchemaVersionError(
                f"SQLite v{version} foreign-key integrity check failed"
            )
        if not validate_data:
            return {}
        inconsistent = self._connection.execute(
            "SELECT brain.brain_id FROM brains AS brain "
            "WHERE brain.next_sequence != COALESCE(("
            "SELECT MAX(event.sequence) + 1 FROM events AS event "
            "WHERE event.brain_id = brain.brain_id), 1) LIMIT 1"
        ).fetchone()
        if inconsistent is not None:
            raise SchemaVersionError(
                f"SQLite v{version} brain sequence allocation is inconsistent"
            )
        try:
            historical_states, final_states = (
                self._validate_replay_and_snapshots_in_transaction(version)
            )
        except Exception as error:
            raise SchemaVersionError(
                f"SQLite v{version} replay or snapshot data integrity check failed"
            ) from error
        if version in _SQLITE_BRIDGE_SCHEMA_VERSIONS:
            try:
                self._validate_bridge_profile_rows_in_transaction(
                    historical_states=historical_states,
                    final_states=final_states,
                    migrate_legacy_action_frames=(version == 3),
                    legacy_bridge_schema=(version in {3, 4}),
                )
            except Exception as error:
                raise SchemaVersionError(
                    f"SQLite v{version} bridge or profile data integrity check failed"
                ) from error
        if version == SQLITE_SCHEMA_VERSION:
            try:
                self._validate_identity_rows_in_transaction(
                    historical_states,
                    final_states,
                )
            except IdentityNameNormalizationError as error:
                raise SchemaVersionError(
                    "SQLite v6 identity data violates the stable Unicode 14.0 "
                    "normalization boundary"
                ) from error
            except Exception as error:
                raise SchemaVersionError(
                    f"SQLite v{version} identity data integrity check failed"
                ) from error
        return final_states

    @property
    def schema_version(self) -> int:
        self._ensure_open()
        return SQLITE_SCHEMA_VERSION

    @staticmethod
    def _persisted_observability_flag(row: Mapping[str, object], field: str) -> int:
        value = row[field]
        if type(value) is not int or value not in {0, 1}:
            raise LedgerIntegrityError(
                f"persisted observability {field} is not an exact boolean"
            )
        return value

    @staticmethod
    def _persisted_observability_count(row: Mapping[str, object], field: str) -> int:
        value = row[field]
        if (
            type(value) is not int
            or value < 0
            or value > MAX_PERSISTED_OBSERVABILITY_COUNT
        ):
            raise LedgerIntegrityError(
                f"persisted observability {field} is not an exact bounded integer"
            )
        return value

    @staticmethod
    def _checked_observability_add(
        current: int,
        increment: int,
        *,
        field: str,
    ) -> int:
        if (
            type(current) is not int
            or current < 0
            or current > MAX_PERSISTED_OBSERVABILITY_COUNT
        ):
            raise LedgerIntegrityError(
                f"persisted observability {field} is not an exact bounded integer"
            )
        if type(increment) is not int or increment < 0:
            raise ValueError("observability increments must be non-negative integers")
        result = current + increment
        if result > MAX_PERSISTED_OBSERVABILITY_COUNT:
            raise DomainCapacityError(
                f"observability {field} capacity is exhausted; record was not applied"
            )
        return result

    def _observability_row_in_transaction(self, brain_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT brain_id, trace_complete, semantic_complete, dropped_events, "
            "semantic_records, legacy_raw_only_records, semantic_gap_records "
            "FROM brain_observability WHERE brain_id = ?",
            (brain_id,),
        ).fetchone()
        if row is None:
            raise LedgerIntegrityError("semantic observability row is missing")
        self._persisted_observability_flag(row, "trace_complete")
        self._persisted_observability_flag(row, "semantic_complete")
        for field in (
            "dropped_events",
            "semantic_records",
            "legacy_raw_only_records",
            "semantic_gap_records",
        ):
            self._persisted_observability_count(row, field)
        return row

    def observability_snapshot(
        self, brain_id: str | None = None
    ) -> ObservabilitySnapshotV1:
        """Return bounded persisted semantic and bridge health without replay."""
        if brain_id is not None:
            brain_id = validate_id(brain_id)
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            where = " WHERE brain_id = ?" if brain_id is not None else ""
            parameters: tuple[object, ...] = () if brain_id is None else (brain_id,)
            cursor = self._connection.execute(
                "SELECT brain_id, trace_complete, semantic_complete, "
                "dropped_events, semantic_records, legacy_raw_only_records, "
                "semantic_gap_records FROM brain_observability" + where,
                parameters,
            )
            brain_count = 0
            trace_complete = True
            semantic_complete = True
            dropped_events = 0
            semantic_records = 0
            legacy_raw_only_records = 0
            semantic_gap_records = 0
            while rows := cursor.fetchmany(512):
                for row in rows:
                    brain_count += 1
                    trace_complete = trace_complete and bool(
                        self._persisted_observability_flag(row, "trace_complete")
                    )
                    semantic_complete = semantic_complete and bool(
                        self._persisted_observability_flag(row, "semantic_complete")
                    )
                    dropped_events += self._persisted_observability_count(
                        row, "dropped_events"
                    )
                    semantic_records += self._persisted_observability_count(
                        row, "semantic_records"
                    )
                    legacy_raw_only_records += self._persisted_observability_count(
                        row, "legacy_raw_only_records"
                    )
                    semantic_gap_records += self._persisted_observability_count(
                        row, "semantic_gap_records"
                    )
            if brain_id is not None and brain_count != 1:
                raise ValueError("observability brain is not persisted")
            bridge_where = " WHERE brain_id = ?" if brain_id is not None else ""
            bridges = self._connection.execute(
                "SELECT COUNT(*) AS total_bridges, "
                "COALESCE(SUM(CASE WHEN status = 'open' "
                "AND connected_nonce IS NOT NULL THEN 1 ELSE 0 END), 0) "
                "AS connected_open_bridges, "
                "COALESCE(SUM(CASE WHEN status = 'open' "
                "AND connected_nonce IS NULL THEN 1 ELSE 0 END), 0) "
                "AS disconnected_open_bridges, "
                "COALESCE(SUM(CASE WHEN status = 'clean_closed' "
                "THEN 1 ELSE 0 END), 0) AS clean_closed_bridges, "
                "COALESCE(SUM(CASE WHEN status = 'abandoned' "
                "THEN 1 ELSE 0 END), 0) AS abandoned_bridges "
                "FROM bridge_stream" + bridge_where,
                parameters,
            ).fetchone()
            return ObservabilitySnapshotV1(
                brain_id=brain_id,
                brain_count=brain_count,
                trace_complete=trace_complete,
                semantic_complete=semantic_complete,
                dropped_events=dropped_events,
                semantic_records=semantic_records,
                legacy_raw_only_records=legacy_raw_only_records,
                semantic_gap_records=semantic_gap_records,
                total_bridges=int(bridges["total_bridges"]),
                connected_open_bridges=int(bridges["connected_open_bridges"]),
                disconnected_open_bridges=int(bridges["disconnected_open_bridges"]),
                clean_closed_bridges=int(bridges["clean_closed_bridges"]),
                abandoned_bridges=int(bridges["abandoned_bridges"]),
            )

    def _record_semantic_observability(
        self,
        brain_id: str,
        record: BridgeRecordV1,
        plan: SemanticPlan,
    ) -> None:
        dropped = record.dropped_count if isinstance(record, BridgeGapV1) else 0
        gap = int(plan.semantic_status == "gap")
        row = self._observability_row_in_transaction(brain_id)
        updated = self._connection.execute(
            "UPDATE brain_observability SET trace_complete = ?, "
            "semantic_complete = ?, dropped_events = ?, semantic_records = ?, "
            "semantic_gap_records = ? WHERE brain_id = ?",
            (
                int(bool(row["trace_complete"]) and gap == 0),
                int(bool(row["semantic_complete"]) and plan.semantic_complete),
                self._checked_observability_add(
                    int(row["dropped_events"]),
                    dropped,
                    field="dropped_events",
                ),
                self._checked_observability_add(
                    int(row["semantic_records"]),
                    1,
                    field="semantic_records",
                ),
                self._checked_observability_add(
                    int(row["semantic_gap_records"]),
                    gap,
                    field="semantic_gap_records",
                ),
                brain_id,
            ),
        )
        if updated.rowcount != 1:
            raise LedgerIntegrityError("semantic observability row is missing")

    def _semantic_frame_evidence_in_transaction(
        self,
        brain_id: str,
        *,
        pending_record: BridgeRecordV1 | None = None,
        pending_plan: SemanticPlan | None = None,
    ) -> tuple[bool, dict[str, object]]:
        row = self._observability_row_in_transaction(brain_id)
        values = {
            "schema_version": SEMANTIC_SCHEMA_VERSION,
            "semantic_records": int(row["semantic_records"]),
            "legacy_raw_only_records": int(row["legacy_raw_only_records"]),
            "semantic_gap_records": int(row["semantic_gap_records"]),
            "dropped_events": int(row["dropped_events"]),
        }
        complete = bool(row["trace_complete"]) and bool(row["semantic_complete"])
        if (pending_record is None) is not (pending_plan is None):
            raise ValueError("pending semantic frame evidence must be paired")
        if pending_record is not None and pending_plan is not None:
            values["semantic_records"] = self._checked_observability_add(
                int(values["semantic_records"]),
                1,
                field="semantic_records",
            )
            values["legacy_raw_only_records"] = self._checked_observability_add(
                int(values["legacy_raw_only_records"]),
                int(pending_plan.semantic_status == "legacy_raw_only"),
                field="legacy_raw_only_records",
            )
            values["semantic_gap_records"] = self._checked_observability_add(
                int(values["semantic_gap_records"]),
                int(pending_plan.semantic_status == "gap"),
                field="semantic_gap_records",
            )
            if isinstance(pending_record, BridgeGapV1):
                values["dropped_events"] = self._checked_observability_add(
                    int(values["dropped_events"]),
                    pending_record.dropped_count,
                    field="dropped_events",
                )
            complete = complete and pending_plan.semantic_complete
        return complete, values

    def _record_unbounded_gap_observability(self, brain_id: str) -> None:
        row = self._observability_row_in_transaction(brain_id)
        updated = self._connection.execute(
            "UPDATE brain_observability SET trace_complete = 0, "
            "semantic_complete = 0, semantic_gap_records = ? "
            "WHERE brain_id = ?",
            (
                self._checked_observability_add(
                    int(row["semantic_gap_records"]),
                    1,
                    field="semantic_gap_records",
                ),
                brain_id,
            ),
        )
        if updated.rowcount != 1:
            raise LedgerIntegrityError("gap observability row is missing")

    @property
    def foreign_keys_enabled(self) -> bool:
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            return bool(self._connection.execute("PRAGMA foreign_keys").fetchone()[0])

    @property
    def journal_mode(self) -> str:
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            row = self._connection.execute("PRAGMA journal_mode").fetchone()
            if row is None or not isinstance(row[0], str):
                raise LedgerIntegrityError("SQLite journal mode is unreadable")
            return row[0].casefold()

    def _ensure_open(self) -> None:
        self._assert_creator_process()
        with self._lock:
            if self._closed:
                raise LedgerClosedError("ledger is closed")
            if self._lease_registry is not None:
                self._lease_registry.assert_authority()
            self._refresh_mutation_seal()

    @property
    def closed(self) -> bool:
        self._assert_creator_process()
        return self._closed

    def _assert_creator_process(self) -> None:
        if os.getpid() != self._creator_pid:
            raise PermissionError("SQLite ledger belongs to another process")

    def _sqlite_authorizer(
        self,
        action_code: int,
        argument_one: str | None,
        argument_two: str | None,
        _database_name: str | None,
        _trigger_name: str | None,
    ) -> int:
        if not self._mutation_seal_installed:
            return sqlite3.SQLITE_OK
        pragma_mutates = action_code == sqlite3.SQLITE_PRAGMA and (
            argument_two is not None
            or (argument_one or "").casefold() in _SQLITE_MUTATING_NO_ARGUMENT_PRAGMAS
        )
        authorized = self._authorized_transaction_thread == threading.get_ident()
        if (
            action_code in _SQLITE_MUTATION_ACTIONS or pragma_mutates
        ) and not authorized:
            self._mutation_seal_poisoned = True
        return sqlite3.SQLITE_OK

    def _read_mutation_data_version(self) -> int:
        row = self._connection.execute("PRAGMA data_version").fetchone()
        if row is None or isinstance(row[0], bool) or not isinstance(row[0], int):
            raise LedgerIntegrityError(
                "SQLite mutation seal data version is unreadable"
            )
        return int(row[0])

    def _install_mutation_seal(self, *, expected_data_version: int) -> None:
        if self._mutation_seal_installed:
            raise RuntimeError("SQLite mutation seal is already installed")
        current = self._read_mutation_data_version()
        self._mutation_seal_data_version = expected_data_version
        self._mutation_seal_installed = True
        if current != expected_data_version:
            self._mutation_seal_poisoned = True
            raise LedgerIntegrityError(
                "SQLite mutation seal detected a commit during startup audit"
            )

    def _refresh_mutation_seal(self) -> None:
        if not self._mutation_seal_installed:
            return
        if self._mutation_seal_poisoned:
            raise LedgerIntegrityError(
                "SQLite mutation seal detected an unauthorized mutation"
            )
        try:
            current = self._read_mutation_data_version()
        except BaseException as error:
            self._mutation_seal_poisoned = True
            raise LedgerIntegrityError(
                "SQLite mutation seal could not verify the database"
            ) from error
        if current != self._mutation_seal_data_version:
            self._mutation_seal_poisoned = True
            raise LedgerIntegrityError(
                "SQLite mutation seal detected an external commit"
            )
        self._mutation_seal_data_version = current

    @contextmanager
    def _transaction(self, *, immediate: bool) -> Iterator[None]:
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            begin = "BEGIN IMMEDIATE" if immediate else "BEGIN"
            if self._authorized_transaction_thread is not None:
                raise RuntimeError("nested SQLite ledger transactions are forbidden")
            self._authorized_transaction_thread = threading.get_ident()
            cleanup_failed = False
            primary_error: BaseException | None = None
            primary_traceback: TracebackType | None = None
            cleanup_error: BaseException | None = None
            try:
                try:
                    self._connection.execute(begin)
                    self._refresh_mutation_seal()
                    yield
                    if self._lease_registry is not None:
                        self._lease_registry.assert_authority()
                    self._connection.commit()
                    if self._connection.in_transaction:
                        self._mutation_seal_poisoned = True
                        cleanup_failed = True
                        raise LedgerIntegrityError(
                            "SQLite mutation seal detected a transaction "
                            "remaining open after commit"
                        )
                    if self._lease_registry is not None:
                        self._lease_registry.assert_authority()
                except BaseException as error:
                    primary_error = error
                    primary_traceback = error.__traceback__
                    try:
                        if self._connection.in_transaction:
                            self._connection.rollback()
                        if self._connection.in_transaction:
                            raise LedgerIntegrityError(
                                "SQLite transaction cleanup left a transaction open"
                            )
                    except BaseException as error:
                        cleanup_error = error
                        self._mutation_seal_poisoned = True
                        cleanup_failed = True
            finally:
                self._authorized_transaction_thread = None
            if cleanup_error is not None:
                if primary_error is None:
                    raise AssertionError("transaction cleanup lost its primary error")
                raise primary_error.with_traceback(primary_traceback) from cleanup_error
            if not cleanup_failed:
                try:
                    self._refresh_mutation_seal()
                except BaseException as refresh_error:
                    if primary_error is not None:
                        raise primary_error.with_traceback(
                            primary_traceback
                        ) from refresh_error
                    raise
            if primary_error is not None:
                raise primary_error.with_traceback(primary_traceback)

    @staticmethod
    def _normalize_event(event: EventEnvelope) -> EventEnvelope:
        if not isinstance(event, EventEnvelope):
            raise TypeError("append accepts only EventEnvelope instances")
        return event.revalidated()

    @staticmethod
    def _decode_event(row: Mapping[str, Any]) -> EventEnvelope:
        try:
            event = EventEnvelope.model_validate_json(row["envelope_json"])
        except Exception as error:
            raise LedgerIntegrityError("persisted event envelope is invalid") from error
        if (
            event.event_id != row["event_id"]
            or event.brain_id != row["brain_id"]
            or event.sequence != row["sequence"]
        ):
            raise LedgerIntegrityError("event row keys do not match its envelope")
        if event.body_fingerprint() != row["body_fingerprint"]:
            raise LedgerIntegrityError("event body fingerprint does not match its body")
        if event.envelope_fingerprint() != row["envelope_fingerprint"]:
            raise LedgerIntegrityError(
                "event envelope fingerprint does not match its stored envelope"
            )
        return event

    def append(self, event: EventEnvelope) -> EventEnvelope:
        """Idempotently append one event and allocate its per-brain sequence."""
        stored, _inserted = self._append(event, expected_sequence=None)
        return stored

    @staticmethod
    def _validate_expected_sequence(expected_sequence: int) -> int:
        if isinstance(expected_sequence, bool) or not isinstance(
            expected_sequence, int
        ):
            raise TypeError("expected_sequence must be an integer")
        if expected_sequence < 1:
            raise ValueError("expected_sequence must be positive")
        return expected_sequence

    def get_event(self, event_id: str) -> EventEnvelope | None:
        """Return one integrity-checked event by its globally unique ID."""
        event_id = validate_id(event_id)
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT event_id, brain_id, sequence, body_fingerprint, "
                "envelope_fingerprint, envelope_json "
                "FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            return None if row is None else self._decode_event(row)

    def ensure_brain(self, brain_id: str) -> str:
        """Persist an empty brain foundation without accessing private state."""
        brain_id = validate_id(brain_id)
        with self._transaction(immediate=True):
            self._connection.execute(
                "INSERT INTO brains(brain_id, next_sequence) VALUES (?, 1) "
                "ON CONFLICT(brain_id) DO NOTHING",
                (brain_id,),
            )
            self._ensure_observability_row(brain_id)
        return brain_id

    def _ensure_observability_row(self, brain_id: str) -> None:
        self._connection.execute(
            "INSERT INTO brain_observability("
            "brain_id, trace_complete, semantic_complete, dropped_events, "
            "semantic_records, legacy_raw_only_records, semantic_gap_records) "
            "VALUES (?, 1, 1, 0, 0, 0, 0) "
            "ON CONFLICT(brain_id) DO NOTHING",
            (brain_id,),
        )

    def _record_identity_name_in_transaction(
        self,
        *,
        brain_id: str,
        name: str | None,
        source_event_id: str,
    ) -> None:
        if name is None:
            return
        self._connection.execute(
            "INSERT INTO identity_name_registry("
            "brain_id, normalized_name, display_name, source_event_id) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(brain_id) DO UPDATE SET "
            "normalized_name = excluded.normalized_name, "
            "display_name = excluded.display_name, "
            "source_event_id = excluded.source_event_id",
            (
                brain_id,
                self._normalized_identity_name(name),
                name,
                source_event_id,
            ),
        )

    def _current_identity_name_source_in_transaction(
        self,
        state: BrainState,
    ) -> EventEnvelope:
        name = state.identity.name
        if name is None:
            raise LedgerIntegrityError("unnamed identity has no naming source")
        source = self._last_identity_name_event_in_transaction(state.brain_id, name)
        row = self._connection.execute(
            "SELECT normalized_name, display_name, source_event_id "
            "FROM identity_name_registry WHERE brain_id = ?",
            (state.brain_id,),
        ).fetchone()
        if (
            row is None
            or row["normalized_name"] != self._normalized_identity_name(name)
            or row["display_name"] != name
            or row["source_event_id"] != source.event_id
        ):
            raise LedgerIntegrityError(
                "named identity registry lost its current source event"
            )
        return source

    def create_brain_foundation(
        self, brain_id: str, *, name: str | None
    ) -> EventEnvelope:
        """Atomically create a brain row and its sequence-one genesis event."""
        brain_id = validate_id(brain_id)
        event = new_event("brain.created", brain_id, brain_id, {"name": name})
        provisional = event.model_copy(update={"sequence": 1}).revalidated()
        reduce_state(BrainState.genesis(brain_id), provisional)
        with self._transaction(immediate=True):
            existing = self._connection.execute(
                "SELECT brain_id FROM brains WHERE brain_id = ?", (brain_id,)
            ).fetchone()
            if existing is not None:
                raise EventConflictError("brain foundation already exists")
            self._connection.execute(
                "INSERT INTO brains(brain_id, next_sequence) VALUES (?, 2)",
                (brain_id,),
            )
            self._ensure_observability_row(brain_id)
            self._connection.execute(
                "INSERT INTO events("
                "brain_id, sequence, event_id, body_fingerprint, "
                "envelope_fingerprint, envelope_json) VALUES (?, 1, ?, ?, ?, ?)",
                (
                    brain_id,
                    provisional.event_id,
                    provisional.body_fingerprint(),
                    provisional.envelope_fingerprint(),
                    provisional.canonical_json(),
                ),
            )
            self._record_identity_name_in_transaction(
                brain_id=brain_id,
                name=name,
                source_event_id=provisional.event_id,
            )
        return provisional

    def _validated_profile_rows_in_transaction(
        self,
    ) -> dict[str, tuple[BrainProfileV1, str]]:
        rows = self._connection.execute(
            "SELECT profile_key, profile_fingerprint, profile_json, brain_id "
            "FROM brain_profile ORDER BY profile_key"
        ).fetchall()
        persisted_profiles: dict[str, tuple[BrainProfileV1, str]] = {}
        for existing in rows:
            try:
                persisted = BrainProfileV1.model_validate_json(existing["profile_json"])
                brain_id = validate_id(existing["brain_id"])
            except Exception as error:
                raise LedgerIntegrityError(
                    "persisted stable brain profile is invalid"
                ) from error
            canonical_json = persisted.canonical_json()
            if (
                existing["profile_key"] != persisted.profile_key
                or existing["profile_json"] != canonical_json
                or existing["profile_fingerprint"] != persisted.fingerprint()
            ):
                raise LedgerIntegrityError(
                    "persisted stable brain profile integrity check failed"
                )
            brain = self._connection.execute(
                "SELECT next_sequence FROM brains WHERE brain_id = ?",
                (brain_id,),
            ).fetchone()
            foundation_row = self._connection.execute(
                "SELECT event_id, brain_id, sequence, body_fingerprint, "
                "envelope_fingerprint, envelope_json FROM events "
                "WHERE brain_id = ? AND sequence = 1",
                (brain_id,),
            ).fetchone()
            if brain is None or foundation_row is None:
                raise LedgerIntegrityError("stable brain profile foundation is missing")
            foundation = self._decode_event(foundation_row)
            if (
                foundation.event_type != "brain.created"
                or foundation.actor_id != brain_id
                or foundation.payload != {"name": persisted.name}
            ):
                raise LedgerIntegrityError(
                    "stable brain profile foundation does not match"
                )
            persisted_profiles[persisted.profile_key] = (persisted, brain_id)
        return persisted_profiles

    def resolve_brain_profile(
        self,
        profile: BrainProfileV1,
        *,
        new_brain_id: str | None = None,
    ) -> BrainResolveResult:
        """Atomically resolve or create one stable profile and foundation."""
        if not isinstance(profile, BrainProfileV1):
            raise TypeError("profile must be BrainProfileV1")
        profile = BrainProfileV1.model_validate(profile.model_dump(mode="python"))
        if new_brain_id is not None:
            new_brain_id = validate_id(new_brain_id)
        profile_json = profile.canonical_json()
        fingerprint = profile.fingerprint()
        with self._transaction(immediate=True):
            persisted_profiles = self._validated_profile_rows_in_transaction()

            resolved = persisted_profiles.get(profile.profile_key)
            if resolved is not None:
                persisted, brain_id = resolved
                if persisted != profile:
                    raise IdempotencyConflictError(
                        "stable brain profile key has a different immutable body"
                    )
                return BrainResolveResult(brain_id=brain_id, created=False)

            brain_id = new_id() if new_brain_id is None else new_brain_id
            event = new_event(
                "brain.created", brain_id, brain_id, {"name": profile.name}
            )
            foundation = event.model_copy(update={"sequence": 1}).revalidated()
            reduce_state(BrainState.genesis(brain_id), foundation)
            self._connection.execute(
                "INSERT INTO brains(brain_id, next_sequence) VALUES (?, 2)",
                (brain_id,),
            )
            self._ensure_observability_row(brain_id)
            self._connection.execute(
                "INSERT INTO events("
                "brain_id, sequence, event_id, body_fingerprint, "
                "envelope_fingerprint, envelope_json) VALUES (?, 1, ?, ?, ?, ?)",
                (
                    brain_id,
                    foundation.event_id,
                    foundation.body_fingerprint(),
                    foundation.envelope_fingerprint(),
                    foundation.canonical_json(),
                ),
            )
            self._record_identity_name_in_transaction(
                brain_id=brain_id,
                name=profile.name,
                source_event_id=foundation.event_id,
            )
            self._connection.execute(
                "INSERT INTO brain_profile("
                "profile_key, profile_fingerprint, profile_json, brain_id) "
                "VALUES (?, ?, ?, ?)",
                (profile.profile_key, fingerprint, profile_json, brain_id),
            )
            return BrainResolveResult(
                brain_id=brain_id,
                created=True,
                foundation=foundation,
            )

    def compensate_unpublished_brain_foundation(
        self,
        brain_id: str,
        *,
        foundation: EventEnvelope,
        profile: BrainProfileV1 | None,
    ) -> bool:
        """Delete only an exact, unobserved dynamic foundation after start fails."""
        brain_id = validate_id(brain_id)
        if not isinstance(foundation, EventEnvelope):
            raise TypeError("foundation must be EventEnvelope")
        foundation = foundation.revalidated()
        if (
            foundation.brain_id != brain_id
            or foundation.actor_id != brain_id
            or foundation.sequence != 1
            or foundation.event_type != "brain.created"
        ):
            raise ValueError("foundation does not match the compensated brain")
        if profile is not None:
            if not isinstance(profile, BrainProfileV1):
                raise TypeError("profile must be BrainProfileV1 or None")
            profile = BrainProfileV1.model_validate(profile.model_dump(mode="python"))
            if foundation.payload != {"name": profile.name}:
                raise ValueError(
                    "foundation payload does not match the compensated profile"
                )

        with self._transaction(immediate=True):
            if brain_id in self._startup_audited_final_states:
                return False
            brain = self._connection.execute(
                "SELECT next_sequence FROM brains WHERE brain_id = ?",
                (brain_id,),
            ).fetchone()
            event_rows = self._connection.execute(
                "SELECT event_id, brain_id, sequence, body_fingerprint, "
                "envelope_fingerprint, envelope_json FROM events "
                "WHERE brain_id = ? ORDER BY sequence ASC LIMIT 2",
                (brain_id,),
            ).fetchall()
            profile_rows = self._connection.execute(
                "SELECT profile_key, profile_fingerprint, profile_json, brain_id "
                "FROM brain_profile WHERE brain_id = ? "
                "ORDER BY profile_key LIMIT 2",
                (brain_id,),
            ).fetchall()
            has_snapshot = self._connection.execute(
                "SELECT 1 FROM snapshots WHERE brain_id = ? LIMIT 1",
                (brain_id,),
            ).fetchone()
            has_stream = self._connection.execute(
                "SELECT 1 FROM bridge_stream WHERE brain_id = ? LIMIT 1",
                (brain_id,),
            ).fetchone()
            has_bridge_record = self._connection.execute(
                "SELECT 1 FROM bridge_record AS record "
                "JOIN bridge_stream AS stream "
                "ON stream.bridge_instance_id = record.bridge_instance_id "
                "WHERE stream.brain_id = ? LIMIT 1",
                (brain_id,),
            ).fetchone()
            name_rows = self._connection.execute(
                "SELECT brain_id, normalized_name, display_name, source_event_id "
                "FROM identity_name_registry WHERE brain_id = ?",
                (brain_id,),
            ).fetchall()
            has_naming_lease = self._connection.execute(
                "SELECT 1 FROM identity_naming_lease WHERE brain_id = ? LIMIT 1",
                (brain_id,),
            ).fetchone()

            persisted_foundation = (
                self._decode_event(event_rows[0]) if len(event_rows) == 1 else None
            )
            exact_foundation = (
                persisted_foundation is not None
                and persisted_foundation == foundation
                and persisted_foundation.canonical_json() == foundation.canonical_json()
            )
            if profile is None:
                exact_profile = not profile_rows
            else:
                exact_profile = len(profile_rows) == 1 and (
                    profile_rows[0]["profile_key"] == profile.profile_key
                    and profile_rows[0]["profile_fingerprint"] == profile.fingerprint()
                    and profile_rows[0]["profile_json"] == profile.canonical_json()
                    and profile_rows[0]["brain_id"] == brain_id
                )
            foundation_name = foundation.payload.get("name")
            exact_name_registry = (
                not name_rows
                if foundation_name is None
                else len(name_rows) == 1
                and name_rows[0]["brain_id"] == brain_id
                and name_rows[0]["display_name"] == foundation_name
                and name_rows[0]["normalized_name"]
                == self._normalized_identity_name(foundation_name)
                and name_rows[0]["source_event_id"] == foundation.event_id
            )
            if (
                brain is None
                or int(brain["next_sequence"]) != 2
                or not exact_foundation
                or not exact_profile
                or has_snapshot is not None
                or has_stream is not None
                or has_bridge_record is not None
                or has_naming_lease is not None
                or not exact_name_registry
            ):
                return False

            if profile is not None:
                deleted_profile = self._connection.execute(
                    "DELETE FROM brain_profile WHERE profile_key = ? "
                    "AND profile_fingerprint = ? AND profile_json = ? "
                    "AND brain_id = ?",
                    (
                        profile.profile_key,
                        profile.fingerprint(),
                        profile.canonical_json(),
                        brain_id,
                    ),
                )
                if deleted_profile.rowcount != 1:
                    raise LedgerIntegrityError(
                        "exact dynamic profile compensation was lost"
                    )
            if foundation_name is not None:
                deleted_name = self._connection.execute(
                    "DELETE FROM identity_name_registry WHERE brain_id = ? "
                    "AND normalized_name = ? AND display_name = ? "
                    "AND source_event_id = ?",
                    (
                        brain_id,
                        self._normalized_identity_name(foundation_name),
                        foundation_name,
                        foundation.event_id,
                    ),
                )
                if deleted_name.rowcount != 1:
                    raise LedgerIntegrityError(
                        "exact dynamic identity-name compensation was lost"
                    )
            deleted_event = self._connection.execute(
                "DELETE FROM events WHERE brain_id = ? AND sequence = 1 "
                "AND event_id = ? AND body_fingerprint = ? "
                "AND envelope_fingerprint = ? AND envelope_json = ?",
                (
                    brain_id,
                    foundation.event_id,
                    foundation.body_fingerprint(),
                    foundation.envelope_fingerprint(),
                    foundation.canonical_json(),
                ),
            )
            if deleted_event.rowcount != 1:
                raise LedgerIntegrityError(
                    "exact dynamic foundation compensation was lost"
                )
            self._connection.execute(
                "DELETE FROM brain_observability WHERE brain_id = ?",
                (brain_id,),
            )
            deleted_brain = self._connection.execute(
                "DELETE FROM brains WHERE brain_id = ? AND next_sequence = 2",
                (brain_id,),
            )
            if deleted_brain.rowcount != 1:
                raise LedgerIntegrityError("exact dynamic brain compensation was lost")
        return True

    def list_brain_ids(self) -> list[str]:
        """Return every persisted brain ID in stable lexical order."""
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT brain_id FROM brains ORDER BY brain_id ASC"
            ).fetchall()
            return [validate_id(row["brain_id"]) for row in rows]

    @staticmethod
    def _identity_now(value: datetime) -> datetime:
        if not isinstance(value, datetime):
            raise TypeError("identity naming time must be a datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("identity naming time must be timezone-aware")
        return value.astimezone(UTC)

    def _identity_lease_row(self, lease_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT lease_id, brain_id, request_sequence, status, requested_at, "
            "expires_at, request_event_id, choice_fingerprint, choice_json, "
            "failure_code, terminal_event_id, terminal_at "
            "FROM identity_naming_lease WHERE lease_id = ?",
            (lease_id,),
        ).fetchone()
        if row is None:
            raise KeyError(lease_id)
        return row

    def identity_naming_status(
        self,
        lease_id: str,
    ) -> IdentityNamingLeaseStatusV1:
        lease_id = validate_id(lease_id)
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            return self._identity_status_from_row(self._identity_lease_row(lease_id))

    def identity_naming_brain_id(self, lease_id: str) -> str:
        return self.identity_naming_status(lease_id).brain_id

    @staticmethod
    def _identity_lease(status: IdentityNamingLeaseStatusV1) -> IdentityNamingLeaseV1:
        return IdentityNamingLeaseV1(
            lease_id=status.lease_id,
            brain_id=status.brain_id,
            state_sequence=status.state_sequence,
            expires_at=status.expires_at,
        )

    def _require_identity_head_in_transaction(self, state: BrainState) -> None:
        head = self._authoritative_brain_head_in_transaction(state.brain_id)
        if head != state.last_sequence:
            raise ExpectedSequenceError(
                "identity naming state does not match authoritative head"
            )

    def _insert_identity_event_batch_in_transaction(
        self,
        state: BrainState,
        events: tuple[EventEnvelope, ...],
    ) -> tuple[tuple[EventEnvelope, ...], BrainState]:
        if not events:
            raise ValueError("identity event batch cannot be empty")
        successor = state
        sequenced: list[EventEnvelope] = []
        for offset, event in enumerate(events, start=1):
            if event.brain_id != state.brain_id or event.sequence is not None:
                raise ValueError("identity event batch changed its brain or sequence")
            provisional = event.model_copy(
                update={"sequence": state.last_sequence + offset}
            ).revalidated()
            successor = reduce_state(successor, provisional)
            sequenced.append(provisional)
        next_sequence = successor.last_sequence + 1
        updated = self._connection.execute(
            "UPDATE brains SET next_sequence = ? "
            "WHERE brain_id = ? AND next_sequence = ?",
            (next_sequence, state.brain_id, state.last_sequence + 1),
        )
        if updated.rowcount != 1:
            raise ExpectedSequenceError(
                "identity naming sequence was lost before event insert"
            )
        for event in sequenced:
            self._connection.execute(
                "INSERT INTO events("
                "brain_id, sequence, event_id, body_fingerprint, "
                "envelope_fingerprint, envelope_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.brain_id,
                    event.sequence,
                    event.event_id,
                    event.body_fingerprint(),
                    event.envelope_fingerprint(),
                    event.canonical_json(),
                ),
            )
        return tuple(sequenced), successor

    def claim_identity_naming(
        self,
        *,
        expected_state: BrainState,
        actor_id: str,
        now: datetime,
    ) -> IdentityNamingClaimResult:
        expected_state = expected_state.revalidated()
        actor_id = validate_id(actor_id)
        if actor_id != expected_state.brain_id:
            raise ValueError("identity naming actor must be the brain self actor")
        now = self._identity_now(now)
        expires_at = now + timedelta(seconds=IDENTITY_NAMING_LEASE_SECONDS)
        lease_id = new_id()
        request = new_event(
            "cognition.requested",
            expected_state.brain_id,
            actor_id,
            {
                "schema_version": 1,
                "purpose": _IDENTITY_PURPOSE,
                "lease_id": lease_id,
                "requested_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
            },
            adapter_id=_IDENTITY_ADAPTER_ID,
        )
        with self._transaction(immediate=True):
            self._require_identity_head_in_transaction(expected_state)
            pending_row = self._connection.execute(
                "SELECT lease_id, brain_id, request_sequence, status, requested_at, "
                "expires_at, request_event_id, choice_fingerprint, choice_json, "
                "failure_code, terminal_event_id, terminal_at "
                "FROM identity_naming_lease "
                "WHERE brain_id = ? AND status = 'pending'",
                (expected_state.brain_id,),
            ).fetchone()
            if pending_row is not None:
                pending = self._identity_status_from_row(pending_row)
                if pending.expires_at > now and expected_state.identity.name is None:
                    return IdentityNamingClaimResult(lease=None, successor=None)
                if pending.expires_at <= now:
                    self._supersede_pending_identity_lease(
                        pending.lease_id,
                        now=now,
                        requested_at=pending.requested_at,
                        reason="expired",
                    )
                else:
                    source = self._current_identity_name_source_in_transaction(
                        expected_state
                    )
                    self._supersede_pending_identity_lease(
                        pending.lease_id,
                        now=now,
                        requested_at=pending.requested_at,
                        reason="identity_already_named",
                        terminal_event_id=source.event_id,
                    )
            if expected_state.identity.name is not None:
                return IdentityNamingClaimResult(lease=None, successor=None)
            events, successor = self._insert_identity_event_batch_in_transaction(
                expected_state,
                (request,),
            )
            request_event = events[0]
            self._connection.execute(
                "INSERT INTO identity_naming_lease("
                "lease_id, brain_id, request_sequence, status, requested_at, "
                "expires_at, request_event_id, choice_fingerprint, choice_json, "
                "failure_code, terminal_event_id, terminal_at) "
                "VALUES (?, ?, ?, 'pending', ?, ?, ?, NULL, NULL, NULL, NULL, NULL)",
                (
                    lease_id,
                    expected_state.brain_id,
                    request_event.sequence,
                    now.isoformat(),
                    expires_at.isoformat(),
                    request_event.event_id,
                ),
            )
            lease = IdentityNamingLeaseV1(
                lease_id=lease_id,
                brain_id=expected_state.brain_id,
                state_sequence=request_event.sequence,
                expires_at=expires_at,
            )
            return IdentityNamingClaimResult(lease=lease, successor=successor)

    def _supersede_pending_identity_lease(
        self,
        lease_id: str,
        *,
        now: datetime,
        requested_at: datetime,
        reason: str,
        terminal_event_id: str | None = None,
    ) -> None:
        if now < requested_at:
            raise ValueError("identity naming terminal time predates its request")
        if reason == "expired":
            if terminal_event_id is not None:
                raise ValueError("expired identity lease cannot cite a name event")
        elif reason == "identity_already_named":
            if terminal_event_id is None:
                raise ValueError("named identity lease requires its source event")
            terminal_event_id = validate_id(terminal_event_id)
        else:
            raise ValueError("identity naming supersession reason is invalid")
        updated = self._connection.execute(
            "UPDATE identity_naming_lease SET status = 'superseded', "
            "failure_code = ?, terminal_event_id = ?, terminal_at = ? "
            "WHERE lease_id = ? AND status = 'pending'",
            (reason, terminal_event_id, now.isoformat(), lease_id),
        )
        if updated.rowcount != 1:
            raise LedgerIntegrityError("identity naming lease supersession was lost")

    def complete_identity_naming(
        self,
        lease_id: str,
        choice: IdentityChoiceV1,
        *,
        expected_state: BrainState,
        actor_id: str,
        now: datetime,
    ) -> IdentityNamingTerminalResult:
        lease_id = validate_id(lease_id)
        if not isinstance(choice, IdentityChoiceV1):
            raise TypeError("choice must be IdentityChoiceV1")
        choice = IdentityChoiceV1.model_validate(
            choice.model_dump(mode="python"), strict=True
        )
        expected_state = expected_state.revalidated()
        actor_id = validate_id(actor_id)
        if actor_id != expected_state.brain_id:
            raise ValueError("identity naming actor must be the brain self actor")
        now = self._identity_now(now)
        choice_json = choice.canonical_json()
        choice_fingerprint = self._identity_choice_fingerprint(choice)
        with self._transaction(immediate=True):
            row = self._identity_lease_row(lease_id)
            status = self._identity_status_from_row(row)
            if status.brain_id != expected_state.brain_id:
                raise ValueError("identity naming lease changed brain identity")
            self._require_identity_head_in_transaction(expected_state)
            if status.status == "completed":
                if status.choice != choice:
                    raise IdempotencyConflictError(
                        "identity naming lease has a different completed choice"
                    )
                return IdentityNamingTerminalResult("completed", None)
            if status.status == "failed":
                if status.choice is not None and status.choice != choice:
                    raise IdempotencyConflictError(
                        "identity naming lease has a different failed choice"
                    )
                return IdentityNamingTerminalResult(
                    "failed" if status.choice == choice else "superseded",
                    None,
                )
            if status.status == "superseded":
                return IdentityNamingTerminalResult("superseded", None)
            if now < status.requested_at:
                raise ValueError("identity naming terminal time predates its request")
            if now >= status.expires_at:
                self._supersede_pending_identity_lease(
                    lease_id,
                    now=now,
                    requested_at=status.requested_at,
                    reason="expired",
                )
                return IdentityNamingTerminalResult("superseded", None)
            if expected_state.identity.name is not None:
                source = self._current_identity_name_source_in_transaction(
                    expected_state
                )
                self._supersede_pending_identity_lease(
                    lease_id,
                    now=now,
                    requested_at=status.requested_at,
                    reason="identity_already_named",
                    terminal_event_id=source.event_id,
                )
                return IdentityNamingTerminalResult("superseded", None)

            conflict = self._connection.execute(
                "SELECT brain_id FROM identity_name_registry "
                "WHERE normalized_name = ? AND brain_id != ? LIMIT 1",
                (
                    self._normalized_identity_name(choice.name),
                    expected_state.brain_id,
                ),
            ).fetchone()
            if conflict is not None:
                failed = new_event(
                    "cognition.failed",
                    expected_state.brain_id,
                    actor_id,
                    {
                        "schema_version": 1,
                        "purpose": _IDENTITY_PURPOSE,
                        "lease_id": lease_id,
                        "failure_code": "name_conflict",
                        "choice_fingerprint": choice_fingerprint,
                        "terminal_at": now.isoformat(),
                    },
                    adapter_id=_IDENTITY_ADAPTER_ID,
                )
                events, successor = self._insert_identity_event_batch_in_transaction(
                    expected_state,
                    (failed,),
                )
                updated = self._connection.execute(
                    "UPDATE identity_naming_lease SET status = 'failed', "
                    "choice_fingerprint = ?, choice_json = ?, "
                    "failure_code = 'name_conflict', terminal_event_id = ?, "
                    "terminal_at = ? WHERE lease_id = ? AND status = 'pending'",
                    (
                        choice_fingerprint,
                        choice_json,
                        events[0].event_id,
                        now.isoformat(),
                        lease_id,
                    ),
                )
                if updated.rowcount != 1:
                    raise LedgerIntegrityError(
                        "identity naming conflict result was not persisted"
                    )
                return IdentityNamingTerminalResult("failed", successor)

            completed = new_event(
                "cognition.completed",
                expected_state.brain_id,
                actor_id,
                {
                    "schema_version": 1,
                    "purpose": _IDENTITY_PURPOSE,
                    "lease_id": lease_id,
                    "choice_fingerprint": choice_fingerprint,
                    "structured": True,
                    "terminal_at": now.isoformat(),
                },
                adapter_id=_IDENTITY_ADAPTER_ID,
            )
            deliberated = new_event(
                "c1.deliberated",
                expected_state.brain_id,
                actor_id,
                {
                    "schema_version": 1,
                    "purpose": _IDENTITY_PURPOSE,
                    "lease_id": lease_id,
                    "name": choice.name,
                    "reason": choice.reason,
                    "source_event_id": completed.event_id,
                    "terminal_at": now.isoformat(),
                },
                adapter_id=_IDENTITY_ADAPTER_ID,
            )
            named = new_event(
                "identity.named",
                expected_state.brain_id,
                actor_id,
                {
                    "schema_version": 1,
                    "purpose": _IDENTITY_PURPOSE,
                    "lease_id": lease_id,
                    "name": choice.name,
                    "reason": choice.reason,
                    "source_event_id": deliberated.event_id,
                    "terminal_at": now.isoformat(),
                },
                adapter_id=_IDENTITY_ADAPTER_ID,
            )
            events, successor = self._insert_identity_event_batch_in_transaction(
                expected_state,
                (completed, deliberated, named),
            )
            self._record_identity_name_in_transaction(
                brain_id=expected_state.brain_id,
                name=choice.name,
                source_event_id=events[-1].event_id,
            )
            updated = self._connection.execute(
                "UPDATE identity_naming_lease SET status = 'completed', "
                "choice_fingerprint = ?, choice_json = ?, terminal_event_id = ?, "
                "terminal_at = ? WHERE lease_id = ? AND status = 'pending'",
                (
                    choice_fingerprint,
                    choice_json,
                    events[-1].event_id,
                    now.isoformat(),
                    lease_id,
                ),
            )
            if updated.rowcount != 1:
                raise LedgerIntegrityError(
                    "completed identity naming result was not persisted"
                )
            return IdentityNamingTerminalResult("completed", successor)

    def fail_identity_naming(
        self,
        lease_id: str,
        failure_code: str,
        *,
        expected_state: BrainState,
        actor_id: str,
        now: datetime,
    ) -> IdentityNamingTerminalResult:
        lease_id = validate_id(lease_id)
        if not isinstance(
            failure_code, str
        ) or not _IDENTITY_WORKER_FAILURE_CODE.fullmatch(failure_code):
            raise ValueError("identity naming failure code is invalid")
        expected_state = expected_state.revalidated()
        actor_id = validate_id(actor_id)
        if actor_id != expected_state.brain_id:
            raise ValueError("identity naming actor must be the brain self actor")
        now = self._identity_now(now)
        with self._transaction(immediate=True):
            status = self._identity_status_from_row(self._identity_lease_row(lease_id))
            if status.brain_id != expected_state.brain_id:
                raise ValueError("identity naming lease changed brain identity")
            self._require_identity_head_in_transaction(expected_state)
            if status.status == "failed":
                if status.failure_code != failure_code or status.choice is not None:
                    raise IdempotencyConflictError(
                        "identity naming lease has a different failure"
                    )
                return IdentityNamingTerminalResult("failed", None)
            if status.status in {"completed", "superseded"}:
                return IdentityNamingTerminalResult("superseded", None)
            if now < status.requested_at:
                raise ValueError("identity naming terminal time predates its request")
            if now >= status.expires_at:
                self._supersede_pending_identity_lease(
                    lease_id,
                    now=now,
                    requested_at=status.requested_at,
                    reason="expired",
                )
                return IdentityNamingTerminalResult("superseded", None)
            if expected_state.identity.name is not None:
                source = self._current_identity_name_source_in_transaction(
                    expected_state
                )
                self._supersede_pending_identity_lease(
                    lease_id,
                    now=now,
                    requested_at=status.requested_at,
                    reason="identity_already_named",
                    terminal_event_id=source.event_id,
                )
                return IdentityNamingTerminalResult("superseded", None)
            failed = new_event(
                "cognition.failed",
                expected_state.brain_id,
                actor_id,
                {
                    "schema_version": 1,
                    "purpose": _IDENTITY_PURPOSE,
                    "lease_id": lease_id,
                    "failure_code": failure_code,
                    "terminal_at": now.isoformat(),
                },
                adapter_id=_IDENTITY_ADAPTER_ID,
            )
            events, successor = self._insert_identity_event_batch_in_transaction(
                expected_state,
                (failed,),
            )
            updated = self._connection.execute(
                "UPDATE identity_naming_lease SET status = 'failed', "
                "failure_code = ?, terminal_event_id = ?, terminal_at = ? "
                "WHERE lease_id = ? AND status = 'pending'",
                (
                    failure_code,
                    events[0].event_id,
                    now.isoformat(),
                    lease_id,
                ),
            )
            if updated.rowcount != 1:
                raise LedgerIntegrityError("identity naming failure was not persisted")
            return IdentityNamingTerminalResult("failed", successor)

    def _authoritative_brain_head_in_transaction(self, brain_id: str) -> int | None:
        row = self._connection.execute(
            "SELECT next_sequence FROM brains WHERE brain_id = ?",
            (brain_id,),
        ).fetchone()
        if row is None:
            return None
        next_sequence = int(row["next_sequence"])
        if next_sequence < 1:
            raise LedgerIntegrityError(
                "authoritative brain sequence allocation is invalid"
            )
        # Startup replay proves this allocator matches the immutable event
        # tail.  The permanent mutation seal then makes this single-row value
        # the bounded live head without rescanning the append-only ledger.
        return next_sequence - 1

    def bootstrap_state(self, brain_id: str) -> BrainState:
        """Reuse startup replay only while it still matches the bounded DB head."""
        brain_id = validate_id(brain_id)
        with self._transaction(immediate=False):
            head = self._authoritative_brain_head_in_transaction(brain_id)
            audited = self._startup_audited_final_states.get(brain_id)
            if audited is not None and head == audited.last_sequence:
                return audited
        return self.replay(brain_id)

    @staticmethod
    def _stream_from_row(row: sqlite3.Row) -> BridgeStreamState:
        try:
            raw_disconnected_at = row["disconnected_at"]
            disconnected_at = raw_disconnected_at
            if isinstance(raw_disconnected_at, str):
                disconnected_at = datetime.fromisoformat(raw_disconnected_at)
                if (
                    disconnected_at.tzinfo is None
                    or disconnected_at.utcoffset() is None
                    or disconnected_at.astimezone(UTC).isoformat()
                    != raw_disconnected_at
                ):
                    raise ValueError("disconnected timestamp is not canonical UTC")
            raw_last_seen = row["last_seen"]
            last_seen = datetime.fromisoformat(raw_last_seen)
            if (
                last_seen.tzinfo is None
                or last_seen.utcoffset() is None
                or last_seen.astimezone(UTC).isoformat() != raw_last_seen
            ):
                raise ValueError("last_seen timestamp is not canonical UTC")
            return BridgeStreamState(
                bridge_instance_id=row["bridge_instance_id"],
                brain_id=row["brain_id"],
                server_actor_id=row["server_actor_id"],
                server_adapter_id=row["server_adapter_id"],
                next_capture_seq=row["next_capture_seq"],
                status=row["status"],
                connected_nonce=row["connected_nonce"],
                disconnected_reason=row["disconnected_reason"],
                disconnected_at=disconnected_at,
                last_seen=last_seen,
                closed_final_seq=row["closed_final_seq"],
            )
        except Exception as error:
            raise LedgerIntegrityError("persisted bridge stream is invalid") from error

    @staticmethod
    def _bridge_timestamp(last_seen: datetime | None = None) -> str:
        current = datetime.now(UTC)
        if last_seen is not None and last_seen > current:
            current = last_seen
        return current.astimezone(UTC).isoformat()

    @staticmethod
    def _validate_bridge_recovery_digest(row: sqlite3.Row) -> bytes:
        digest = row["recovery_token_digest"]
        if type(digest) is not bytes or len(digest) != 32:
            raise LedgerIntegrityError("persisted bridge recovery digest is invalid")
        return digest

    def _bridge_stream_row(self, bridge_instance_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT bridge_instance_id, brain_id, server_actor_id, "
            "server_adapter_id, recovery_token_digest, next_capture_seq, "
            "status, connected_nonce, disconnected_reason, disconnected_at, "
            "last_seen, closed_final_seq FROM bridge_stream "
            "WHERE bridge_instance_id = ?",
            (bridge_instance_id,),
        ).fetchone()
        if row is None:
            raise BridgeBindingError("bridge stream is not attached")
        self._validate_bridge_recovery_digest(row)
        return row

    def _validate_bridge_stream_history(
        self,
        stream: BridgeStreamState,
        *,
        historical_states: Mapping[int, BrainState] | None = None,
        final_state: BrainState | None = None,
        migrate_legacy_action_frames: bool = False,
        legacy_bridge_schema: bool = False,
    ) -> BrainState:
        semantic_columns = (
            ""
            if legacy_bridge_schema
            else ", semantic_status, semantic_complete, semantic_fingerprint, "
            "derived_event_count, derived_first_sequence, derived_last_sequence"
        )
        rows = self._connection.execute(
            "SELECT bridge_instance_id, first_capture_seq, last_capture_seq, "
            "record_kind, record_fingerprint, record_json, event_id, "
            "ledger_sequence, ack_json, accepted_at"
            + semantic_columns
            + " FROM bridge_record "
            "WHERE bridge_instance_id = ? ORDER BY first_capture_seq",
            (stream.bridge_instance_id,),
        ).fetchall()
        if historical_states is None:
            replay_targets = {
                int(row["ledger_sequence"])
                + (0 if legacy_bridge_schema else int(row["derived_event_count"]))
                for row in rows
            }
            if not legacy_bridge_schema:
                replay_targets.update(int(row["ledger_sequence"]) - 1 for row in rows)
            historical_states, replayed_final = (
                self._replay_target_states_in_transaction(
                    stream.brain_id,
                    replay_targets,
                )
            )
        else:
            replayed_final = final_state
            if replayed_final is None:
                raise ValueError("final_state is required with historical_states")
        if replayed_final.brain_id != stream.brain_id:
            raise LedgerIntegrityError(
                "bridge history replay brain does not match its stream"
            )
        expected_cursor = 1
        previous_ledger_sequence = 0
        expected_spans: list[HermesSpan] = []
        for row in rows:
            if row["first_capture_seq"] != expected_cursor:
                raise LedgerIntegrityError(
                    "persisted bridge record intervals are not contiguous"
                )
            if row["ledger_sequence"] <= previous_ledger_sequence:
                raise LedgerIntegrityError(
                    "persisted bridge event order is not strictly increasing"
                )
            try:
                if row["record_kind"] == "observation":
                    record: BridgeRecordV1 = validate_observation_json(
                        row["record_json"]
                    )
                elif row["record_kind"] == "gap":
                    record = BridgeGapV1.model_validate_json(row["record_json"])
                else:
                    raise ValueError("unknown bridge record kind")
            except Exception as error:
                raise LedgerIntegrityError(
                    "persisted bridge record history is invalid"
                ) from error
            row_last_sequence = int(row["ledger_sequence"]) + (
                0 if legacy_bridge_schema else int(row["derived_event_count"])
            )
            historical_state = historical_states.get(row_last_sequence)
            if historical_state is None:
                raise LedgerIntegrityError(
                    "bridge event sequence is absent from authoritative replay"
                )
            try:
                if legacy_bridge_schema:
                    self._decode_legacy_duplicate_bridge_record(
                        row,
                        requested=record,
                        requested_json=record.canonical_json(),
                        requested_fingerprint=record.fingerprint(),
                        historical_state=historical_state,
                        migrate_legacy_action_frame=migrate_legacy_action_frames,
                    )
                else:
                    ack = self._decode_duplicate_bridge_record(
                        row,
                        requested=record,
                        requested_json=record.canonical_json(),
                        requested_fingerprint=record.fingerprint(),
                        historical_state=historical_state,
                    )
                    if ack.semantic_status == "legacy_raw_only":
                        expected_legacy_fingerprint = self._legacy_semantic_fingerprint(
                            record_fingerprint=record.fingerprint(),
                            raw_event_id=ack.raw_event_id,
                            raw_event_sequence=ack.raw_event_sequence,
                            through_capture_seq=record.last_capture_seq,
                        )
                        if (
                            ack.semantic_fingerprint != expected_legacy_fingerprint
                            or ack.semantic_complete
                            or ack.derived_event_count != 0
                        ):
                            raise LedgerIntegrityError(
                                "legacy raw-only semantic evidence is not canonical"
                            )
                    else:
                        event_rows = self._connection.execute(
                            "SELECT event_id, brain_id, sequence, body_fingerprint, "
                            "envelope_fingerprint, envelope_json FROM events "
                            "WHERE brain_id = ? AND sequence BETWEEN ? AND ? "
                            "ORDER BY sequence",
                            (
                                stream.brain_id,
                                ack.raw_event_sequence,
                                ack.last_event_sequence,
                            ),
                        ).fetchall()
                        events = tuple(
                            self._decode_event(event_row) for event_row in event_rows
                        )
                        raw_event = (
                            events[0]
                            .model_copy(update={"sequence": None})
                            .revalidated()
                        )
                        capacity_gap: str | None = None
                        victim_indexes: list[int] = []
                        if (
                            isinstance(record, HermesObservationV1)
                            and record.hook in {"pre_tool_call", "pre_api_request"}
                            and len(expected_spans) >= MAX_HERMES_SPANS_PER_STREAM
                        ):
                            required = (
                                len(expected_spans) - MAX_HERMES_SPANS_PER_STREAM + 1
                            )
                            victim_indexes = sorted(
                                (
                                    index
                                    for index, span in enumerate(expected_spans)
                                    if span.closed_capture_seq is not None
                                ),
                                key=lambda index: (
                                    expected_spans[index].closed_capture_seq or 0,
                                    expected_spans[index].occurrence_capture_seq,
                                ),
                            )[:required]
                            if (
                                len(expected_spans) - len(victim_indexes)
                                >= MAX_HERMES_SPANS_PER_STREAM
                            ):
                                capacity_gap = "span_capacity_all_open"
                        matched_span: HermesSpan | None = None
                        correlation_gap: str | None = None
                        if isinstance(record, HermesObservationV1):
                            matched_span, correlation_gap = match_hermes_span(
                                record, tuple(expected_spans)
                            )
                        predecessor = historical_states.get(ack.raw_event_sequence - 1)
                        if predecessor is None:
                            raise LedgerIntegrityError(
                                "semantic batch has no predecessor state"
                            )
                        lifecycle_gap = self._semantic_lifecycle_gap(
                            predecessor,
                            record,
                            matched_span,
                        )
                        actual_derived = events[1:]
                        domain_capacity_gap = (
                            ack.semantic_status == "gap"
                            and len(actual_derived) == 1
                            and actual_derived[0].event_type == "semantic.gap"
                            and actual_derived[0].payload.get("reason")
                            == "semantic_domain_capacity"
                        )
                        forced_gap_reason = (
                            capacity_gap or correlation_gap or lifecycle_gap
                        )
                        if domain_capacity_gap:
                            candidate = build_semantic_plan(
                                stream,
                                record,
                                raw_event=raw_event,
                                matched_span=matched_span,
                                forced_gap_reason=forced_gap_reason,
                            )
                            candidate_state = predecessor
                            try:
                                for offset, event in enumerate(
                                    (raw_event, *candidate.derived_events)
                                ):
                                    provisional = event.model_copy(
                                        update={
                                            "sequence": (
                                                ack.raw_event_sequence + offset
                                            )
                                        }
                                    ).revalidated()
                                    candidate_state = reduce_state(
                                        candidate_state, provisional
                                    )
                            except DomainCapacityError:
                                forced_gap_reason = "semantic_domain_capacity"
                            else:
                                raise LedgerIntegrityError(
                                    "semantic capacity gap is not reproducible"
                                )
                        expected_plan = build_semantic_plan(
                            stream,
                            record,
                            raw_event=raw_event,
                            matched_span=matched_span,
                            forced_gap_reason=forced_gap_reason,
                        )
                        if (
                            expected_plan.semantic_status != ack.semantic_status
                            or expected_plan.semantic_complete
                            is not ack.semantic_complete
                            or expected_plan.fingerprint() != ack.semantic_fingerprint
                            or len(expected_plan.derived_events) != len(actual_derived)
                            or any(
                                expected.event_id != actual.event_id
                                or expected.canonical_json(exclude_sequence=True)
                                != actual.canonical_json(exclude_sequence=True)
                                for expected, actual in zip(
                                    expected_plan.derived_events,
                                    actual_derived,
                                    strict=True,
                                )
                            )
                        ):
                            raise LedgerIntegrityError(
                                "persisted derived events do not match semantic plan"
                            )
                        if expected_plan.span_close is not None:
                            for index in range(len(expected_spans) - 1, -1, -1):
                                span = expected_spans[index]
                                if (
                                    span == expected_plan.span_close
                                    and span.closed_capture_seq is None
                                ):
                                    expected_spans[index] = HermesSpan(
                                        **{
                                            **span.canonical_data(),
                                            "closed_capture_seq": (
                                                record.last_capture_seq
                                            ),
                                        }
                                    )
                                    break
                            else:
                                raise LedgerIntegrityError(
                                    "semantic history closed an absent span"
                                )
                        if expected_plan.span_open is not None:
                            for index in sorted(victim_indexes, reverse=True):
                                expected_spans.pop(index)
                            expected_spans.append(expected_plan.span_open)
            except IdempotencyConflictError as error:
                raise LedgerIntegrityError(
                    "persisted bridge history fingerprint does not match"
                ) from error
            expected_cursor = record.last_capture_seq + 1
            previous_ledger_sequence = row["ledger_sequence"]
        if stream.next_capture_seq != expected_cursor:
            raise LedgerIntegrityError(
                "persisted bridge cursor does not match contiguous history"
            )
        if not legacy_bridge_schema:
            persisted_span_rows = self._connection.execute(
                "SELECT bridge_instance_id, span_kind, external_id, "
                "occurrence_capture_seq, context_fingerprint, action_id, "
                "closed_capture_seq FROM hermes_span WHERE bridge_instance_id = ? "
                "ORDER BY span_kind, external_id, occurrence_capture_seq",
                (stream.bridge_instance_id,),
            ).fetchall()
            expected_span_values = sorted(
                (
                    span.bridge_instance_id,
                    span.span_kind,
                    span.external_id,
                    span.occurrence_capture_seq,
                    span.context_fingerprint,
                    span.action_id,
                    span.closed_capture_seq,
                )
                for span in expected_spans
            )
            persisted_span_values = [tuple(row) for row in persisted_span_rows]
            if persisted_span_values != expected_span_values:
                raise LedgerIntegrityError(
                    "bounded Hermes span cache does not match semantic history"
                )
        return replayed_final

    @staticmethod
    def _is_reserved_abandonment_gap(event: EventEnvelope) -> bool:
        if event.event_type != "trace.gap":
            return False
        cause_counts = event.payload.get("cause_counts")
        has_reserved_cause = (
            isinstance(cause_counts, (Mapping, list, tuple, str))
            and "abandoned_unknown" in cause_counts
        )
        return (
            "unknown_range" in event.payload
            or event.payload.get("exact") is False
            or has_reserved_cause
        )

    def _validate_abandonment_history_in_transaction(
        self, streams: Mapping[str, BridgeStreamState]
    ) -> None:
        last_record_sequences = {
            row["bridge_instance_id"]: int(row["last_record_sequence"])
            for row in self._connection.execute(
                "SELECT bridge_instance_id, MAX(ledger_sequence) "
                "AS last_record_sequence FROM bridge_record "
                "GROUP BY bridge_instance_id"
            ).fetchall()
        }
        matched_streams: set[str] = set()
        cursor = self._connection.execute(
            "SELECT event.event_id, event.brain_id, event.sequence, "
            "event.body_fingerprint, event.envelope_fingerprint, "
            "event.envelope_json, record.event_id "
            "AS linked_bridge_record_event_id FROM events AS event "
            "LEFT JOIN bridge_record AS record ON record.event_id = event.event_id "
            "ORDER BY event.brain_id, event.sequence"
        )
        while rows := cursor.fetchmany(512):
            for row in rows:
                event = self._decode_event(row)
                if not self._is_reserved_abandonment_gap(event):
                    continue
                bridge_instance_id = event.payload.get("bridge_instance_id")
                if not isinstance(bridge_instance_id, str):
                    raise LedgerIntegrityError(
                        "reserved abandonment gap has no valid bridge identity"
                    )
                stream = streams.get(bridge_instance_id)
                if stream is None or stream.status != "abandoned":
                    raise LedgerIntegrityError(
                        "reserved abandonment gap has no abandoned stream"
                    )
                expected_payload = {
                    "schema_version": 1,
                    "record_kind": "gap",
                    "bridge_instance_id": stream.bridge_instance_id,
                    "first_capture_seq": stream.next_capture_seq,
                    "last_capture_seq": None,
                    "dropped_count": None,
                    "cause_counts": {"abandoned_unknown": 1},
                    "exact": False,
                    "unknown_range": True,
                    "trace_complete": False,
                }
                if event.payload != expected_payload:
                    raise LedgerIntegrityError(
                        "reserved abandonment gap payload is not canonical"
                    )
                if (
                    event.brain_id != stream.brain_id
                    or event.actor_id != stream.server_actor_id
                    or event.adapter_id != stream.server_adapter_id
                    or event.session_id is not None
                    or event.turn_id is not None
                    or event.action_id is not None
                    or event.causation_id is not None
                    or event.correlation_id is not None
                ):
                    raise LedgerIntegrityError(
                        "reserved abandonment gap provenance does not match"
                    )
                if row["linked_bridge_record_event_id"] is not None:
                    raise LedgerIntegrityError(
                        "reserved abandonment gap cannot be a bridge record"
                    )
                if event.sequence <= last_record_sequences.get(bridge_instance_id, 0):
                    raise LedgerIntegrityError(
                        "reserved abandonment gap precedes bridge history"
                    )
                if bridge_instance_id in matched_streams:
                    raise LedgerIntegrityError(
                        "abandoned stream has duplicate unknown gaps"
                    )
                matched_streams.add(bridge_instance_id)

        abandoned_streams = {
            bridge_instance_id
            for bridge_instance_id, stream in streams.items()
            if stream.status == "abandoned"
        }
        if matched_streams != abandoned_streams:
            raise LedgerIntegrityError(
                "abandoned streams and unknown gaps are not one-to-one"
            )

    def _expected_observability_in_transaction(
        self,
        *,
        final_states: Mapping[str, BrainState],
        streams: Mapping[str, BridgeStreamState],
    ) -> dict[str, dict[str, int]]:
        expected = {
            brain_id: {
                "trace_complete": int(state.trace_complete),
                "semantic_complete": int(state.trace_complete),
                "dropped_events": 0,
                "semantic_records": 0,
                "legacy_raw_only_records": 0,
                "semantic_gap_records": 0,
            }
            for brain_id, state in final_states.items()
        }
        rows = self._connection.execute(
            "SELECT stream.brain_id, record.record_kind, record.record_json, "
            "record.semantic_status FROM bridge_record AS record "
            "JOIN bridge_stream AS stream ON stream.bridge_instance_id = "
            "record.bridge_instance_id ORDER BY stream.brain_id, "
            "record.first_capture_seq"
        ).fetchall()
        for row in rows:
            health = expected[row["brain_id"]]
            health["semantic_records"] += 1
            if row["semantic_status"] == "legacy_raw_only":
                health["legacy_raw_only_records"] += 1
                health["semantic_complete"] = 0
            if row["semantic_status"] == "gap":
                health["semantic_gap_records"] += 1
                health["semantic_complete"] = 0
            if row["record_kind"] == "gap":
                record = validate_bridge_record_json(row["record_json"])
                if not isinstance(record, BridgeGapV1):
                    raise LedgerIntegrityError(
                        "observability gap row is not a typed bridge gap"
                    )
                health["dropped_events"] += record.dropped_count
                if row["semantic_status"] != "gap":
                    health["semantic_gap_records"] += 1
                    health["semantic_complete"] = 0
        for stream in streams.values():
            if stream.status == "abandoned":
                health = expected[stream.brain_id]
                health["semantic_gap_records"] += 1
                health["semantic_complete"] = 0
        covered = self._bridge_batch_sequences_in_transaction()
        event_cursor = self._connection.execute(
            "SELECT event_id, brain_id, sequence, body_fingerprint, "
            "envelope_fingerprint, envelope_json FROM events "
            "ORDER BY brain_id, sequence"
        )
        while event_rows := event_cursor.fetchmany(512):
            for event_row in event_rows:
                event = self._decode_event(event_row)
                if (
                    event.event_type in {"semantic.gap", "trace.gap"}
                    and event.sequence not in covered.get(event.brain_id, set())
                    and not self._is_reserved_abandonment_gap(event)
                ):
                    health = expected[event.brain_id]
                    health["semantic_gap_records"] += 1
                    health["semantic_complete"] = 0
        for brain_id, health in expected.items():
            for field in (
                "dropped_events",
                "semantic_records",
                "legacy_raw_only_records",
                "semantic_gap_records",
            ):
                value = health[field]
                if (
                    type(value) is not int
                    or value < 0
                    or value > MAX_PERSISTED_OBSERVABILITY_COUNT
                ):
                    raise LedgerIntegrityError(
                        f"observability {field} exceeds persisted capacity for "
                        f"brain {brain_id}"
                    )
        return expected

    def _rebuild_observability_in_transaction(
        self, *, final_states: Mapping[str, BrainState]
    ) -> None:
        stream_rows = self._connection.execute(
            "SELECT bridge_instance_id, brain_id, server_actor_id, "
            "server_adapter_id, recovery_token_digest, next_capture_seq, "
            "status, connected_nonce, disconnected_reason, disconnected_at, "
            "last_seen, closed_final_seq FROM bridge_stream "
            "ORDER BY bridge_instance_id"
        ).fetchall()
        streams: dict[str, BridgeStreamState] = {}
        for row in stream_rows:
            self._validate_bridge_recovery_digest(row)
            stream = self._stream_from_row(row)
            streams[stream.bridge_instance_id] = stream
        expected = self._expected_observability_in_transaction(
            final_states=final_states,
            streams=streams,
        )
        self._connection.execute("DELETE FROM brain_observability")
        for brain_id in sorted(expected):
            health = expected[brain_id]
            self._connection.execute(
                "INSERT INTO brain_observability("
                "brain_id, trace_complete, semantic_complete, dropped_events, "
                "semantic_records, legacy_raw_only_records, semantic_gap_records) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    brain_id,
                    health["trace_complete"],
                    health["semantic_complete"],
                    health["dropped_events"],
                    health["semantic_records"],
                    health["legacy_raw_only_records"],
                    health["semantic_gap_records"],
                ),
            )

    def _validate_observability_in_transaction(
        self,
        *,
        final_states: Mapping[str, BrainState],
        streams: Mapping[str, BridgeStreamState],
    ) -> None:
        expected = self._expected_observability_in_transaction(
            final_states=final_states,
            streams=streams,
        )
        persisted_rows = self._connection.execute(
            "SELECT brain_id, trace_complete, semantic_complete, dropped_events, "
            "semantic_records, legacy_raw_only_records, semantic_gap_records "
            "FROM brain_observability ORDER BY brain_id"
        ).fetchall()
        persisted: dict[str, dict[str, int]] = {}
        for row in persisted_rows:
            persisted[str(row["brain_id"])] = {
                "trace_complete": self._persisted_observability_flag(
                    row, "trace_complete"
                ),
                "semantic_complete": self._persisted_observability_flag(
                    row, "semantic_complete"
                ),
                "dropped_events": self._persisted_observability_count(
                    row, "dropped_events"
                ),
                "semantic_records": self._persisted_observability_count(
                    row, "semantic_records"
                ),
                "legacy_raw_only_records": self._persisted_observability_count(
                    row, "legacy_raw_only_records"
                ),
                "semantic_gap_records": self._persisted_observability_count(
                    row, "semantic_gap_records"
                ),
            }
        if persisted != expected:
            raise LedgerIntegrityError(
                "persisted observability metadata does not match replay"
            )

    def _bridge_batch_sequences_in_transaction(self) -> dict[str, set[int]]:
        covered: dict[str, set[int]] = {}
        rows = self._connection.execute(
            "SELECT stream.brain_id, record.ledger_sequence, "
            "record.derived_event_count FROM bridge_record AS record "
            "JOIN bridge_stream AS stream ON stream.bridge_instance_id = "
            "record.bridge_instance_id ORDER BY stream.brain_id, "
            "record.ledger_sequence"
        ).fetchall()
        for row in rows:
            first = int(row["ledger_sequence"])
            last = first + int(row["derived_event_count"])
            covered.setdefault(str(row["brain_id"]), set()).update(
                range(first, last + 1)
            )
        return covered

    def _validate_semantic_frame_evidence_in_transaction(self) -> None:
        timeline: dict[str, list[tuple[int, str, object]]] = {}
        covered = self._bridge_batch_sequences_in_transaction()
        record_rows = self._connection.execute(
            "SELECT stream.brain_id, record.ledger_sequence, "
            "record.record_kind, record.record_json, record.semantic_status, "
            "record.ack_json FROM bridge_record AS record JOIN bridge_stream AS "
            "stream ON stream.bridge_instance_id = record.bridge_instance_id "
            "ORDER BY stream.brain_id, record.ledger_sequence"
        ).fetchall()
        for row in record_rows:
            timeline.setdefault(str(row["brain_id"]), []).append(
                (int(row["ledger_sequence"]), "record", row)
            )
        cursor = self._connection.execute(
            "SELECT event_id, brain_id, sequence, body_fingerprint, "
            "envelope_fingerprint, envelope_json FROM events "
            "ORDER BY brain_id, sequence"
        )
        while rows := cursor.fetchmany(512):
            for row in rows:
                event = self._decode_event(row)
                if self._is_reserved_abandonment_gap(event):
                    if event.sequence is None:
                        raise LedgerIntegrityError(
                            "abandonment gap is missing its sequence"
                        )
                    timeline.setdefault(event.brain_id, []).append(
                        (event.sequence, "abandonment", event)
                    )
                elif event.event_type in {
                    "semantic.gap",
                    "trace.gap",
                } and event.sequence not in covered.get(event.brain_id, set()):
                    if event.sequence is None:
                        raise LedgerIntegrityError("unbounded gap has no sequence")
                    timeline.setdefault(event.brain_id, []).append(
                        (event.sequence, "unbounded_gap", event)
                    )
        for brain_id, entries in timeline.items():
            health = {
                "schema_version": SEMANTIC_SCHEMA_VERSION,
                "semantic_records": 0,
                "legacy_raw_only_records": 0,
                "semantic_gap_records": 0,
                "dropped_events": 0,
            }
            for _, kind, value in sorted(entries, key=lambda item: item[0]):
                if kind in {"abandonment", "unbounded_gap"}:
                    health["semantic_gap_records"] += 1
                    continue
                row = value
                if not isinstance(row, sqlite3.Row):
                    raise LedgerIntegrityError("semantic frame timeline is invalid")
                health["semantic_records"] += 1
                if row["semantic_status"] == "legacy_raw_only":
                    health["legacy_raw_only_records"] += 1
                if row["semantic_status"] == "gap":
                    health["semantic_gap_records"] += 1
                if row["record_kind"] == "gap":
                    record = validate_bridge_record_json(row["record_json"])
                    if not isinstance(record, BridgeGapV1):
                        raise LedgerIntegrityError(
                            "semantic frame gap evidence is not typed"
                        )
                    health["dropped_events"] += record.dropped_count
                    if row["semantic_status"] != "gap":
                        health["semantic_gap_records"] += 1
                try:
                    ack = BridgeCommitAckV2.model_validate_json(row["ack_json"])
                except Exception as error:
                    raise LedgerIntegrityError(
                        "semantic frame acknowledgement is invalid"
                    ) from error
                expected_complete = (
                    ack.frame.trace_complete
                    and health["legacy_raw_only_records"] == 0
                    and health["semantic_gap_records"] == 0
                )
                if (
                    ack.frame.semantic_schema_version != SEMANTIC_SCHEMA_VERSION
                    or ack.frame.semantic_evidence.model_dump(mode="python") != health
                    or ack.frame.aggregate_semantic_complete is not expected_complete
                ):
                    raise LedgerIntegrityError(
                        f"semantic frame evidence is not truthful for brain {brain_id}"
                    )

    def _validate_bridge_tail_in_transaction(
        self, stream: BridgeStreamState
    ) -> BridgeRecordV1 | None:
        rows = self._connection.execute(
            "SELECT bridge_instance_id, first_capture_seq, last_capture_seq, "
            "record_kind, record_fingerprint, record_json, event_id, "
            "ledger_sequence, semantic_status, semantic_complete, "
            "semantic_fingerprint, derived_event_count, derived_first_sequence, "
            "derived_last_sequence, ack_json, accepted_at FROM bridge_record "
            "WHERE bridge_instance_id = ? "
            "ORDER BY first_capture_seq DESC LIMIT 2",
            (stream.bridge_instance_id,),
        ).fetchall()
        if not rows:
            if stream.next_capture_seq != 1:
                raise LedgerIntegrityError(
                    "persisted bridge cursor has no authoritative tail record"
                )
            return None

        decoded: list[BridgeRecordV1] = []
        for row in rows:
            try:
                record = validate_bridge_record_json(row["record_json"])
            except Exception as error:
                raise LedgerIntegrityError(
                    "persisted bridge tail record is invalid"
                ) from error
            self._decode_duplicate_bridge_record(
                row,
                requested=record,
                requested_json=record.canonical_json(),
                requested_fingerprint=record.fingerprint(),
            )
            decoded.append(record)

        latest = decoded[0]
        if latest.last_capture_seq + 1 != stream.next_capture_seq:
            raise LedgerIntegrityError(
                "persisted bridge cursor does not match its tail record"
            )
        if len(decoded) == 1:
            if latest.first_capture_seq != 1:
                raise LedgerIntegrityError(
                    "persisted bridge tail does not start at capture one"
                )
        else:
            previous = decoded[1]
            if previous.last_capture_seq + 1 != latest.first_capture_seq:
                raise LedgerIntegrityError(
                    "persisted bridge tail intervals are not contiguous"
                )
            if int(rows[1]["ledger_sequence"]) >= int(rows[0]["ledger_sequence"]):
                raise LedgerIntegrityError(
                    "persisted bridge event order is not strictly increasing"
                )
        return latest

    def _validate_bridge_profile_rows_in_transaction(
        self,
        *,
        historical_states: Mapping[str, Mapping[int, BrainState]],
        final_states: Mapping[str, BrainState],
        migrate_legacy_action_frames: bool,
        legacy_bridge_schema: bool,
    ) -> None:
        self._validated_profile_rows_in_transaction()
        rows = self._connection.execute(
            "SELECT bridge_instance_id, brain_id, server_actor_id, "
            "server_adapter_id, recovery_token_digest, next_capture_seq, "
            "status, connected_nonce, disconnected_reason, disconnected_at, "
            "last_seen, closed_final_seq FROM bridge_stream "
            "ORDER BY bridge_instance_id"
        ).fetchall()
        streams: dict[str, BridgeStreamState] = {}
        for row in rows:
            self._validate_bridge_recovery_digest(row)
            stream = self._stream_from_row(row)
            streams[stream.bridge_instance_id] = stream
            brain_history = historical_states.get(stream.brain_id)
            final_state = final_states.get(stream.brain_id)
            if brain_history is None or final_state is None:
                raise LedgerIntegrityError(
                    "bridge stream brain is absent from authoritative replay"
                )
            self._validate_bridge_stream_history(
                stream,
                historical_states=brain_history,
                final_state=final_state,
                migrate_legacy_action_frames=migrate_legacy_action_frames,
                legacy_bridge_schema=legacy_bridge_schema,
            )
        self._validate_abandonment_history_in_transaction(streams)
        if not legacy_bridge_schema:
            self._validate_semantic_frame_evidence_in_transaction()
            self._validate_observability_in_transaction(
                final_states=final_states,
                streams=streams,
            )

    def _validate_replay_and_snapshots_in_transaction(
        self, version: int
    ) -> tuple[dict[str, dict[int, BrainState]], dict[str, BrainState]]:
        brain_rows = self._connection.execute(
            "SELECT brain_id, next_sequence FROM brains ORDER BY brain_id"
        ).fetchall()
        brain_next = {
            validate_id(row["brain_id"]): int(row["next_sequence"])
            for row in brain_rows
        }
        target_sequences: dict[str, set[int]] = {
            brain_id: set() for brain_id in brain_next
        }
        snapshots = self._connection.execute(
            "SELECT brain_id, sequence, schema_version, fingerprint, state_json "
            "FROM snapshots ORDER BY brain_id, sequence"
        ).fetchall()
        for row in snapshots:
            brain_id = row["brain_id"]
            if brain_id not in brain_next:
                raise LedgerIntegrityError("snapshot references an unknown brain")
            if row["schema_version"] == STATE_SCHEMA_VERSION:
                target_sequences[brain_id].add(int(row["sequence"]))
        if version in _SQLITE_BRIDGE_SCHEMA_VERSIONS:
            sequence_expression = (
                "record.ledger_sequence"
                if version in {3, 4}
                else "record.ledger_sequence + record.derived_event_count"
            )
            bridge_targets = self._connection.execute(
                "SELECT stream.brain_id, "
                + sequence_expression
                + " AS ledger_sequence, record.ledger_sequence - 1 "
                "AS predecessor_sequence "
                "FROM bridge_record AS record JOIN bridge_stream AS stream "
                "ON stream.bridge_instance_id = record.bridge_instance_id "
                "ORDER BY stream.brain_id, record.ledger_sequence"
            ).fetchall()
            for row in bridge_targets:
                brain_id = row["brain_id"]
                if brain_id not in brain_next:
                    raise LedgerIntegrityError(
                        "bridge record references an unknown brain"
                    )
                target_sequences[brain_id].add(int(row["ledger_sequence"]))
                if version == SQLITE_SCHEMA_VERSION:
                    target_sequences[brain_id].add(int(row["predecessor_sequence"]))
        if version == SQLITE_SCHEMA_VERSION:
            identity_targets = self._connection.execute(
                "SELECT lease.brain_id, lease.request_sequence, lease.status, "
                "lease.failure_code, terminal.sequence AS terminal_sequence "
                "FROM identity_naming_lease AS lease "
                "LEFT JOIN events AS terminal "
                "ON terminal.event_id = lease.terminal_event_id "
                "ORDER BY lease.brain_id, lease.request_sequence"
            ).fetchall()
            for row in identity_targets:
                brain_id = row["brain_id"]
                if brain_id not in target_sequences:
                    raise LedgerIntegrityError(
                        "identity naming lease references an unknown brain"
                    )
                request_sequence = int(row["request_sequence"])
                target_sequences[brain_id].add(request_sequence - 1)
                if (
                    row["status"] == "superseded"
                    and row["failure_code"] == "identity_already_named"
                    and row["terminal_sequence"] is not None
                ):
                    target_sequences[brain_id].add(int(row["terminal_sequence"]))

        historical_states: dict[str, dict[int, BrainState]] = {}
        final_states: dict[str, BrainState] = {}
        for brain_id in sorted(brain_next):
            captured, replayed = self._replay_target_states_in_transaction(
                brain_id, target_sequences[brain_id]
            )
            if brain_next[brain_id] != replayed.last_sequence + 1:
                raise LedgerIntegrityError(
                    "brain sequence allocator does not match contiguous replay"
                )
            historical_states[brain_id] = captured
            final_states[brain_id] = replayed
            self._validate_snapshot_schemas_in_transaction(brain_id)
        for row in snapshots:
            brain_id = row["brain_id"]
            replayed = final_states[brain_id]
            if row["sequence"] > replayed.last_sequence:
                raise LedgerIntegrityError(
                    "snapshot sequence points beyond authoritative replay"
                )
            if row["schema_version"] in _LEGACY_STATE_SCHEMA_VERSIONS:
                if self._snapshot_fingerprint(row["state_json"]) != row["fingerprint"]:
                    raise LedgerIntegrityError(
                        "legacy snapshot fingerprint does not match"
                    )
                try:
                    legacy = json.loads(row["state_json"])
                except Exception as error:
                    raise LedgerIntegrityError(
                        "legacy snapshot JSON is invalid"
                    ) from error
                if not isinstance(legacy, dict):
                    raise LedgerIntegrityError("legacy snapshot must be one object")
                if (
                    legacy.get("schema_version") != row["schema_version"]
                    or legacy.get("brain_id") != brain_id
                    or legacy.get("last_sequence") != row["sequence"]
                ):
                    raise LedgerIntegrityError(
                        "legacy snapshot keys do not match its row"
                    )
                continue
            snapshot = self._decode_snapshot(row, brain_id)
            expected = historical_states[brain_id].get(int(row["sequence"]))
            if expected is None or snapshot != expected:
                raise LedgerIntegrityError(
                    "snapshot does not equal deterministic full replay"
                )
        return historical_states, final_states

    def attach_bridge_stream(
        self,
        bridge_instance_id: str,
        *,
        brain_id: str,
        server_actor_id: str,
        server_adapter_id: str,
        connected_nonce: str,
        recovery_token: str,
    ) -> BridgeStreamState:
        """Create or resume one server-provenance-bound bridge stream."""
        bridge_instance_id = validate_id(bridge_instance_id)
        brain_id = validate_id(brain_id)
        server_actor_id = validate_id(server_actor_id)
        if not isinstance(server_adapter_id, str) or not (
            1 <= len(server_adapter_id) <= 512 and server_adapter_id.strip()
        ):
            raise ValueError("server_adapter_id must be non-blank and bounded")
        if not isinstance(connected_nonce, str) or not (
            1 <= len(connected_nonce) <= 512 and connected_nonce.strip()
        ):
            raise ValueError("connected_nonce must be non-blank and bounded")
        recovery_token_digest = _bridge_recovery_token_digest(recovery_token)
        wall_now = self._bridge_timestamp()
        with self._transaction(immediate=True):
            brain = self._connection.execute(
                "SELECT brain_id FROM brains WHERE brain_id = ?", (brain_id,)
            ).fetchone()
            if brain is None:
                raise BridgeBindingError("bridge brain does not exist")
            existing = self._connection.execute(
                "SELECT bridge_instance_id, brain_id, server_actor_id, "
                "server_adapter_id, recovery_token_digest, next_capture_seq, "
                "status, connected_nonce, disconnected_reason, "
                "disconnected_at, last_seen, closed_final_seq FROM bridge_stream "
                "WHERE bridge_instance_id = ?",
                (bridge_instance_id,),
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    "INSERT INTO bridge_stream("
                    "bridge_instance_id, brain_id, server_actor_id, "
                    "server_adapter_id, recovery_token_digest, next_capture_seq, "
                    "status, connected_nonce, disconnected_reason, "
                    "disconnected_at, last_seen, closed_final_seq) "
                    "VALUES (?, ?, ?, ?, ?, 1, 'open', ?, NULL, NULL, ?, NULL)",
                    (
                        bridge_instance_id,
                        brain_id,
                        server_actor_id,
                        server_adapter_id,
                        recovery_token_digest,
                        connected_nonce,
                        wall_now,
                    ),
                )
            else:
                stream = self._stream_from_row(existing)
                persisted_now = self._bridge_timestamp(stream.last_seen)
                persisted_digest = self._validate_bridge_recovery_digest(existing)
                self._validate_bridge_tail_in_transaction(stream)
                recovery_matches = hmac.compare_digest(
                    persisted_digest,
                    recovery_token_digest,
                )
                if (
                    stream.brain_id != brain_id
                    or stream.server_actor_id != server_actor_id
                    or stream.server_adapter_id != server_adapter_id
                    or not recovery_matches
                ):
                    raise BridgeBindingError(
                        "bridge stream provenance does not match its binding"
                    )
                if stream.status == "clean_closed":
                    raise BridgeCleanClosedError(
                        "clean-closed bridge stream is not resumable"
                    )
                if stream.status == "abandoned":
                    raise BridgeAbandonedError(
                        "abandoned bridge stream is not resumable"
                    )
                if stream.connected_nonce not in {None, connected_nonce}:
                    raise BridgeBindingError(
                        "bridge stream is attached to another live connection"
                    )
                self._connection.execute(
                    "UPDATE bridge_stream SET connected_nonce = ?, "
                    "disconnected_reason = NULL, disconnected_at = NULL, "
                    "last_seen = ? "
                    "WHERE bridge_instance_id = ?",
                    (connected_nonce, persisted_now, bridge_instance_id),
                )
            return self._stream_from_row(self._bridge_stream_row(bridge_instance_id))

    def recover_bridge_close(
        self,
        bridge_instance_id: str,
        *,
        brain_id: str,
        server_actor_id: str,
        server_adapter_id: str,
        recovery_token: str,
        final_capture_seq: int,
    ) -> BridgeStreamState:
        """Read one exact terminal receipt using its stable recovery proof."""
        bridge_instance_id = validate_id(bridge_instance_id)
        brain_id = validate_id(brain_id)
        server_actor_id = validate_id(server_actor_id)
        if not isinstance(server_adapter_id, str) or not (
            1 <= len(server_adapter_id) <= 512 and server_adapter_id.strip()
        ):
            raise ValueError("server_adapter_id must be non-blank and bounded")
        recovery_token_digest = _bridge_recovery_token_digest(recovery_token)
        if isinstance(final_capture_seq, bool) or not isinstance(
            final_capture_seq, int
        ):
            raise TypeError("final_capture_seq must be an integer")
        if final_capture_seq < 0:
            raise ValueError("final_capture_seq cannot be negative")
        with self._transaction(immediate=False):
            row = self._bridge_stream_row(bridge_instance_id)
            stream = self._stream_from_row(row)
            recovery_matches = hmac.compare_digest(
                self._validate_bridge_recovery_digest(row),
                recovery_token_digest,
            )
            if (
                stream.brain_id != brain_id
                or stream.server_actor_id != server_actor_id
                or stream.server_adapter_id != server_adapter_id
                or not recovery_matches
            ):
                raise BridgeBindingError(
                    "bridge recovery proof or provenance does not match"
                )
            if stream.status != "clean_closed":
                raise BridgeClosedError("bridge stream has no clean-close receipt")
            if stream.closed_final_seq != final_capture_seq:
                raise CaptureSequenceError(
                    "bridge stream was closed at a different final capture"
                )
            return stream

    def disconnect_bridge_stream(
        self, bridge_instance_id: str, *, connected_nonce: str
    ) -> BridgeStreamState:
        """Mark one exact connection absent while retaining resumable state."""
        bridge_instance_id = validate_id(bridge_instance_id)
        with self._transaction(immediate=True):
            row = self._bridge_stream_row(bridge_instance_id)
            stream = self._stream_from_row(row)
            if stream.connected_nonce == connected_nonce:
                now = self._bridge_timestamp(stream.last_seen)
                self._connection.execute(
                    "UPDATE bridge_stream SET connected_nonce = NULL, "
                    "disconnected_reason = 'connection_eof', disconnected_at = ?, "
                    "last_seen = ? "
                    "WHERE bridge_instance_id = ?",
                    (now, now, bridge_instance_id),
                )
            return self._stream_from_row(self._bridge_stream_row(bridge_instance_id))

    def bridge_stream_state(self, bridge_instance_id: str) -> BridgeStreamState:
        bridge_instance_id = validate_id(bridge_instance_id)
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            return self._stream_from_row(self._bridge_stream_row(bridge_instance_id))

    def close_bridge_stream(
        self,
        bridge_instance_id: str,
        *,
        final_capture_seq: int,
        connected_nonce: str | None = None,
    ) -> BridgeStreamState:
        """Persist an exact clean close after every capture has representation."""
        bridge_instance_id = validate_id(bridge_instance_id)
        if isinstance(final_capture_seq, bool) or not isinstance(
            final_capture_seq, int
        ):
            raise TypeError("final_capture_seq must be an integer")
        if final_capture_seq < 0:
            raise ValueError("final_capture_seq cannot be negative")
        with self._transaction(immediate=True):
            row = self._bridge_stream_row(bridge_instance_id)
            stream = self._stream_from_row(row)
            if (
                connected_nonce is not None
                and stream.connected_nonce != connected_nonce
            ):
                raise BridgeBindingError(
                    "bridge stream is attached to another connection"
                )
            expected_final = stream.next_capture_seq - 1
            if stream.status == "clean_closed":
                if stream.closed_final_seq != final_capture_seq:
                    raise CaptureSequenceError(
                        "bridge stream was closed at a different final capture"
                    )
                return stream
            if stream.status != "open":
                raise BridgeClosedError("abandoned bridge stream cannot close")
            if final_capture_seq != expected_final:
                raise CaptureSequenceError(
                    "final_capture_seq does not match the contiguous cursor"
                )
            now = self._bridge_timestamp(stream.last_seen)
            self._connection.execute(
                "UPDATE bridge_stream SET status = 'clean_closed', "
                "connected_nonce = NULL, disconnected_reason = 'clean_close', "
                "disconnected_at = ?, closed_final_seq = ?, last_seen = ? "
                "WHERE bridge_instance_id = ?",
                (
                    now,
                    final_capture_seq,
                    now,
                    bridge_instance_id,
                ),
            )
            return self._stream_from_row(self._bridge_stream_row(bridge_instance_id))

    def abandon_bridge_stream(
        self,
        bridge_instance_id: str,
        *,
        expected_state: BrainState,
        last_seen_not_after: datetime | None = None,
    ) -> BridgeAbandonResult:
        """Atomically persist one explicitly unknown gap and abandon a stream."""
        bridge_instance_id = validate_id(bridge_instance_id)
        expected_state = expected_state.revalidated()
        with self._transaction(immediate=True):
            stream = self._stream_from_row(self._bridge_stream_row(bridge_instance_id))
            now = self._bridge_timestamp(stream.last_seen)
            if stream.brain_id != expected_state.brain_id:
                raise BridgeBindingError("bridge stream brain does not match engine")
            if stream.status == "abandoned":
                return BridgeAbandonResult(stream=stream, successor=None)
            if stream.status != "open":
                raise BridgeClosedError("clean bridge stream cannot be abandoned")
            if stream.connected_nonce is not None:
                raise BridgeBindingError("connected bridge stream cannot be abandoned")
            cutoff: str | None = None
            if last_seen_not_after is not None:
                if (
                    last_seen_not_after.tzinfo is None
                    or last_seen_not_after.utcoffset() is None
                ):
                    raise ValueError("abandonment cutoff must be timezone-aware")
                cutoff = last_seen_not_after.astimezone(UTC).isoformat()
                if stream.last_seen > last_seen_not_after.astimezone(UTC):
                    raise BridgeBindingError(
                        "bridge stream grace was refreshed before abandonment"
                    )
            expected_sequence = expected_state.last_sequence + 1
            brain_row = self._connection.execute(
                "SELECT next_sequence FROM brains WHERE brain_id = ?",
                (stream.brain_id,),
            ).fetchone()
            head = int(
                self._connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM events WHERE brain_id = ?",
                    (stream.brain_id,),
                ).fetchone()[0]
            )
            if (
                brain_row is None
                or int(brain_row["next_sequence"]) != expected_sequence
                or head != expected_state.last_sequence
            ):
                raise ExpectedSequenceError(
                    "abandonment engine state does not match authoritative head"
                )
            event = new_event(
                "trace.gap",
                stream.brain_id,
                stream.server_actor_id,
                {
                    "schema_version": 1,
                    "record_kind": "gap",
                    "bridge_instance_id": stream.bridge_instance_id,
                    "first_capture_seq": stream.next_capture_seq,
                    "last_capture_seq": None,
                    "dropped_count": None,
                    "cause_counts": {"abandoned_unknown": 1},
                    "exact": False,
                    "unknown_range": True,
                    "trace_complete": False,
                },
                adapter_id=stream.server_adapter_id,
            )
            provisional = event.model_copy(
                update={"sequence": expected_sequence}
            ).revalidated()
            successor = reduce_state(expected_state, provisional)
            sequence_update = self._connection.execute(
                "UPDATE brains SET next_sequence = ? "
                "WHERE brain_id = ? AND next_sequence = ?",
                (expected_sequence + 1, stream.brain_id, expected_sequence),
            )
            if sequence_update.rowcount != 1:
                raise ExpectedSequenceError(
                    "abandonment sequence was lost before event insert"
                )
            self._connection.execute(
                "INSERT INTO events("
                "brain_id, sequence, event_id, body_fingerprint, "
                "envelope_fingerprint, envelope_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    provisional.brain_id,
                    expected_sequence,
                    provisional.event_id,
                    provisional.body_fingerprint(),
                    provisional.envelope_fingerprint(),
                    provisional.canonical_json(),
                ),
            )
            if cutoff is None:
                status_update = self._connection.execute(
                    "UPDATE bridge_stream SET status = 'abandoned', "
                    "disconnected_reason = 'grace_abandonment', "
                    "disconnected_at = ?, last_seen = ? "
                    "WHERE bridge_instance_id = ? AND status = 'open' "
                    "AND connected_nonce IS NULL",
                    (now, now, bridge_instance_id),
                )
            else:
                status_update = self._connection.execute(
                    "UPDATE bridge_stream SET status = 'abandoned', "
                    "disconnected_reason = 'grace_abandonment', "
                    "disconnected_at = ?, last_seen = ? "
                    "WHERE bridge_instance_id = ? AND status = 'open' "
                    "AND connected_nonce IS NULL AND last_seen <= ?",
                    (now, now, bridge_instance_id, cutoff),
                )
            if status_update.rowcount != 1:
                raise BridgeBindingError(
                    "bridge stream changed before abandonment commit"
                )
            self._record_unbounded_gap_observability(stream.brain_id)
            abandoned = self._stream_from_row(
                self._bridge_stream_row(bridge_instance_id)
            )
        return BridgeAbandonResult(stream=abandoned, successor=successor)

    def list_abandonable_bridge_streams(
        self, *, last_seen_before: datetime
    ) -> list[tuple[str, str]]:
        """List disconnected open streams old enough for explicit abandonment."""
        if last_seen_before.tzinfo is None or last_seen_before.utcoffset() is None:
            raise ValueError("last_seen_before must be timezone-aware")
        cutoff = last_seen_before.astimezone(UTC).isoformat()
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT bridge_instance_id, brain_id FROM bridge_stream "
                "WHERE status = 'open' AND connected_nonce IS NULL "
                "AND last_seen <= ? ORDER BY bridge_instance_id",
                (cutoff,),
            ).fetchall()
            return [
                (validate_id(row["bridge_instance_id"]), validate_id(row["brain_id"]))
                for row in rows
            ]

    def recover_stale_bridge_connections(self) -> int:
        """Give every prior-process open stream one fresh restart grace."""
        with self._transaction(immediate=True):
            now = self._bridge_timestamp()
            updated = self._connection.execute(
                "UPDATE bridge_stream SET connected_nonce = NULL, "
                "disconnected_reason = 'daemon_restart', "
                "disconnected_at = CASE WHEN last_seen > ? THEN last_seen ELSE ? END, "
                "last_seen = CASE WHEN last_seen > ? THEN last_seen ELSE ? END "
                "WHERE status = 'open'",
                (now, now, now, now),
            )
            return updated.rowcount

    def refresh_daemon_restart_grace(self) -> int:
        """Refresh only still-disconnected restart streams before readiness."""
        with self._transaction(immediate=True):
            now = self._bridge_timestamp()
            updated = self._connection.execute(
                "UPDATE bridge_stream SET "
                "disconnected_at = CASE WHEN last_seen > ? THEN last_seen ELSE ? END, "
                "last_seen = CASE WHEN last_seen > ? THEN last_seen ELSE ? END "
                "WHERE status = 'open' AND connected_nonce IS NULL "
                "AND disconnected_reason = 'daemon_restart'",
                (now, now, now, now),
            )
            return updated.rowcount

    @staticmethod
    def _project_bridge_frame(
        state: BrainState,
        *,
        through_capture_seq: int,
        capture_coverage: Mapping[str, object] | None = None,
        aggregate_semantic_complete: bool | None = None,
        semantic_evidence: Mapping[str, object] | None = None,
        scheduler_sample: str = "not_sampled",
        stream_connected: bool = False,
    ) -> ConsciousnessFrameV3:
        limit = 4

        def bounded_mapping(values: Mapping[str, object]) -> list[dict[str, object]]:
            return [
                {"key": key, "value": values[key]} for key in sorted(values)[:limit]
            ]

        def json_nodes(value: object) -> int:
            pending = [value]
            count = 0
            while pending:
                item = pending.pop()
                count += 1
                if isinstance(item, Mapping):
                    count += len(item)
                    pending.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    pending.extend(item)
            return count

        def count(included: int, omitted: int) -> dict[str, int]:
            return {"included": included, "omitted": omitted}

        def model_nodes(value: object) -> int:
            return json_nodes(value.model_dump(mode="json"))  # type: ignore[union-attr]

        personality = state.personality
        personality_total = sum(
            len(getattr(personality, layer))
            for layer in ("traits", "adaptations", "narrative_ideal")
        )
        personality_included = sum(
            min(limit, len(getattr(personality, layer)))
            for layer in ("traits", "adaptations", "narrative_ideal")
        )
        energy_ordered = sorted(
            state.energy_records,
            key=lambda item: (-item.activation, item.action_id),
        )
        energy_items = energy_ordered[:limit]
        energy_omitted_items = energy_ordered[limit:]
        workspace_items = state.workspace.broadcast[:limit]
        workspace_omitted_items = state.workspace.broadcast[limit:]
        branch_items = state.thought_space[-limit:]
        branch_omitted_items = state.thought_space[:-limit]
        action_items = state.action_records[-limit:]
        action_omitted_items = state.action_records[:-limit]
        other_actors = tuple(
            item
            for item in state.identity.actors
            if item.actor_id != state.identity.self_actor_id
        )
        selected_other_actors = other_actors[-limit:]
        omitted_other_actors = other_actors[:-limit]
        memory_ordered = sorted(
            state.memories,
            key=lambda item: (-item.salience, item.memory_id),
        )
        memory_items = memory_ordered[:limit]
        memory_omitted_items = memory_ordered[limit:]
        ordered_capability_keys = sorted(state.capabilities)
        capability_keys = ordered_capability_keys[:limit]
        omitted_capability_keys = ordered_capability_keys[limit:]
        retained_world_records = sum(
            len(getattr(state.world, layer))
            for layer in ("observed", "believed", "simulated", "ideal")
        )
        projection_records_visited = (
            personality_total * 4
            + len(state.energy_records) * 3
            + len(state.workspace.broadcast) * 3
            + len(state.thought_space) * 3
            + len(state.action_records) * 8
            + len(state.identity.actors) * 4
            + len(state.identity.authorizations) * 2
            + len(state.memories) * 3
            + len(state.capabilities) * 3
            + retained_world_records * 3
            + len(state.cognition.reflections) * 2
            + len(state.raw_lifecycle_counts) * 2
        )
        if projection_records_visited > FRAME_PROJECTION_RECORD_BUDGET:
            raise LedgerIntegrityError(
                "bounded frame projection exceeded its fixed record budget"
            )

        def capability_type(value: object) -> str:
            if value is None:
                return "null"
            if isinstance(value, bool):
                return "boolean"
            if isinstance(value, (int, float)):
                return "number"
            if isinstance(value, str):
                return "string"
            if isinstance(value, Mapping):
                return "object"
            if isinstance(value, (list, tuple)):
                return "array"
            return "unknown"

        def action_evidence_ids(action: object) -> tuple[str, ...]:
            receipt = getattr(action, "receipt", None)
            evidence = receipt.get("effect_evidence") if receipt is not None else None
            raw_ids = (
                evidence.get("observation_ids")
                if isinstance(evidence, Mapping)
                else None
            )
            if not isinstance(raw_ids, (list, tuple)):
                return ()
            return tuple(item for item in raw_ids if isinstance(item, str))

        rd_actions: list[dict[str, object]] = []
        actual_actions: list[dict[str, object]] = []
        for action in action_items:
            evidence_ids = action_evidence_ids(action)
            canonical_outcome = (
                action.outcome.value if action.outcome is not None else None
            )
            canonical_receipt_status = canonical_outcome or (
                action.receipt.get("status") if action.receipt is not None else None
            )
            latest_receipt = (
                action.receipt_history[-1] if action.receipt_history else None
            )
            rd_actions.append(
                {
                    "action_id": action.action_id,
                    "phase": action.phase.value,
                    "rd_phase": action.rd_phase.value,
                    "prepared_branch_id": action.prepared_branch_id,
                    "execution_confirmed": action.execution_confirmed,
                    "effect_confirmed": action.effect_confirmed,
                    "last_transition_event_id": action.last_event_id,
                    "effect_evidence_ids": list(evidence_ids[:limit]),
                    "effect_evidence_ids_omitted": max(0, len(evidence_ids) - limit),
                }
            )
            history = {item.value for item in action.phase_history}
            blocked_observed = "blocked" in history
            dispatch_observed = "dispatched" in history
            projected_action: dict[str, object] = {
                "action_id": action.action_id,
                "dispatch_observed": dispatch_observed,
                "blocked": True if blocked_observed else None,
                "blocked_fact_available": blocked_observed,
                "receipt_observed": "receipt" in history,
                "receipt_status": canonical_receipt_status,
                "latest_receipt_status": (
                    None if latest_receipt is None else latest_receipt.status.value
                ),
                "latest_receipt_disposition": (
                    None if latest_receipt is None else latest_receipt.disposition.value
                ),
                "execution_confirmed": action.execution_confirmed,
                "effect_confirmed": action.effect_confirmed,
                "last_event_id": action.last_event_id,
            }
            if (
                action.receipt_corroboration_count > 0
                or action.receipt_conflict_count > 0
            ):
                projected_action.update(
                    {
                        "outcome": canonical_outcome,
                        "receipt_corroboration_count": (
                            action.receipt_corroboration_count
                        ),
                        "receipt_conflict_count": action.receipt_conflict_count,
                    }
                )
            actual_actions.append(projected_action)

        world_sections: dict[str, list[dict[str, object]]] = {}
        world_omissions: dict[str, dict[str, object]] = {}
        world_total = 0
        world_included = 0
        for layer in ("observed", "believed", "simulated", "ideal"):
            propositions = getattr(state.world, layer)
            world_counter = getattr(state.working_set, f"world_{layer}")
            selected = propositions[-limit:]
            world_total += world_counter.total
            world_included += len(selected)
            world_sections[layer] = [
                {
                    "proposition_id": item.proposition_id,
                    "layer": item.layer.value,
                    "confidence": item.confidence,
                    "grounded_observation": item.layer.value == "observed",
                    "source_event_id": item.source_event_id,
                    "source_actor_id": item.source_actor_id,
                    "action_id": item.action_id,
                }
                for item in selected
            ]
            omitted_propositions = propositions[:-limit]
            world_omissions[layer] = {
                **count(len(selected), world_counter.total - len(selected)),
                "fields": {
                    "content_json_nodes": count(
                        0,
                        sum(json_nodes(item.content) for item in selected),
                    ),
                    "omitted_record_json_nodes": count(
                        0,
                        sum(model_nodes(item) for item in omitted_propositions),
                    ),
                    "evicted_record_details": count(0, world_counter.evicted),
                },
            }

        unresolved = any(
            item.execution_confirmed is None or item.effect_confirmed is None
            for item in state.action_records
        )
        frame_semantic_evidence: Mapping[str, object] = semantic_evidence or {
            "schema_version": SEMANTIC_SCHEMA_VERSION,
            "semantic_records": 0,
            "legacy_raw_only_records": 0,
            "semantic_gap_records": 0,
            "dropped_events": 0,
        }
        if aggregate_semantic_complete is None:
            aggregate_semantic_complete = state.trace_complete
        return ConsciousnessFrameV3(
            brain_id=state.brain_id,
            state_sequence=state.last_sequence,
            through_capture_seq=through_capture_seq,
            logical_clock=state.logical_clock,
            trace_complete=state.trace_complete,
            runtime_health=state.runtime.health,
            c0_tick=state.runtime.tick_count,
            pc={
                "identity_name": state.identity.name,
                "naming_status": (
                    "named" if state.identity.name is not None else "unnamed"
                ),
                "traits": bounded_mapping(personality.traits),
                "adaptations": bounded_mapping(personality.adaptations),
                "narrative_ideal": bounded_mapping(personality.narrative_ideal),
            },
            energy={
                "record_count": state.working_set.energy_records.total,
                "items": [
                    {
                        "action_id": item.action_id,
                        "activation": item.activation,
                        "salience": item.salience,
                        "urgency": item.urgency,
                        "valence": item.valence,
                        "control": item.control,
                        "resources": item.resources,
                        "cost": item.cost,
                    }
                    for item in energy_items
                ],
            },
            st={
                "thought_count": state.working_set.thought_space.total,
                "workspace_cycle": state.workspace.cycle,
                "workspace": [
                    {
                        "candidate_id": item.candidate_id,
                        "specialist": item.specialist,
                        "score": item.score,
                        "source_ids": list(item.source_ids[:limit]),
                        "source_ids_omitted": max(0, len(item.source_ids) - limit),
                    }
                    for item in workspace_items
                ],
                "branches": [
                    {
                        "branch_id": item.branch_id,
                        "stance": item.stance,
                        "uncertainty": item.uncertainty,
                        "rd_phase": item.rd_phase.value,
                        "source_ids": list(item.source_ids[:limit]),
                        "source_ids_omitted": max(0, len(item.source_ids) - limit),
                    }
                    for item in branch_items
                ],
            },
            rd={
                "action_count": state.working_set.action_records.total,
                "unresolved_count": sum(
                    item.execution_confirmed is None or item.effect_confirmed is None
                    for item in state.action_records
                ),
                "actions": rd_actions,
            },
            a={"actions": actual_actions},
            world=world_sections,
            self_boundary={
                "self_actor_id": state.identity.self_actor_id,
                "self_name": state.identity.name,
                "self_kind": "self",
                "other_actor_ids": [item.actor_id for item in selected_other_actors],
                "authorization_count": (
                    state.working_set.provenance_authorizations.total
                ),
                "boundary_explicit": True,
            },
            memory={
                "selection": "most_salient_then_memory_id",
                "items": [
                    {
                        "memory_id": item.memory_id,
                        "salience": item.salience,
                        "source_ids": list(item.source_ids[:limit]),
                        "source_ids_omitted": max(0, len(item.source_ids) - limit),
                    }
                    for item in memory_items
                ],
            },
            capabilities={
                "items": [
                    {
                        "key": key,
                        "boolean_value": (
                            state.capabilities[key]
                            if isinstance(state.capabilities[key], bool)
                            else None
                        ),
                        "value_present": state.capabilities[key] is not None,
                        "value_type": capability_type(state.capabilities[key]),
                    }
                    for key in capability_keys
                ]
            },
            semantic_context={
                "available": False,
                "request_id": None,
                "turn_id": None,
                "reason": "task4_runtime_has_no_semantic_request_binding",
            },
            aggregate_semantic_complete=aggregate_semantic_complete,
            semantic_evidence=frame_semantic_evidence,
            unresolved_evidence=unresolved,
            capture_coverage=capture_coverage
            or {
                "policy_version": "no-capture-v1",
                "capture_coverage": "unobserved",
                "redacted_paths": 0,
                "truncated_paths": 0,
                "unsupported_paths": 0,
                "omitted_nodes": 0,
                "channels": {},
            },
            freshness=FrameFreshnessV1(
                projected_at_state_sequence=state.last_sequence,
                scheduler_tick=state.runtime.tick_count,
                scheduler_sample=scheduler_sample,
                stream_connection=("connected" if stream_connected else "disconnected"),
            ),
            omission_counts={
                "pc": {
                    "included": personality_included,
                    "omitted": personality_total - personality_included,
                    "fields": {
                        "rate_state_json_nodes": count(
                            0,
                            json_nodes(personality.rate_state.model_dump(mode="json")),
                        ),
                        "omitted_layer_value_json_nodes": count(
                            0,
                            sum(
                                json_nodes(value)
                                for layer in (
                                    "traits",
                                    "adaptations",
                                    "narrative_ideal",
                                )
                                for key, value in getattr(personality, layer).items()
                                if key
                                not in sorted(getattr(personality, layer))[:limit]
                            ),
                        ),
                    },
                },
                "energy": {
                    "included": len(energy_items),
                    "omitted": (
                        state.working_set.energy_records.total - len(energy_items)
                    ),
                    "fields": {
                        "deficits_json_nodes": count(
                            0,
                            sum(json_nodes(item.deficits) for item in energy_items),
                        ),
                        "arousal_values": count(0, len(energy_items)),
                        "personality_relevance_values": count(0, len(energy_items)),
                        "omitted_record_json_nodes": count(
                            0,
                            sum(model_nodes(item) for item in energy_omitted_items),
                        ),
                        "evicted_record_details": count(
                            0, state.working_set.energy_records.evicted
                        ),
                    },
                },
                "st": {
                    "included": len(workspace_items) + len(branch_items),
                    "omitted": (
                        len(state.workspace.broadcast)
                        - len(workspace_items)
                        + state.working_set.thought_space.total
                        - len(branch_items)
                    ),
                    "workspace": {
                        **count(len(workspace_items), len(workspace_omitted_items)),
                        "fields": {
                            "content_json_nodes": count(
                                0,
                                sum(
                                    json_nodes(item.content) for item in workspace_items
                                ),
                            ),
                            "cycle_values": count(0, len(workspace_items)),
                            "capacity_values": count(0, 1),
                            "source_id_items": count(
                                sum(
                                    min(limit, len(item.source_ids))
                                    for item in workspace_items
                                ),
                                sum(
                                    max(0, len(item.source_ids) - limit)
                                    for item in workspace_items
                                ),
                            ),
                            "omitted_record_json_nodes": count(
                                0,
                                sum(
                                    model_nodes(item)
                                    for item in workspace_omitted_items
                                ),
                            ),
                        },
                    },
                    "branches": {
                        **count(
                            len(branch_items),
                            state.working_set.thought_space.total - len(branch_items),
                        ),
                        "fields": {
                            "content_json_nodes": count(
                                0,
                                sum(json_nodes(item.content) for item in branch_items),
                            ),
                            "expected_consequence_json_nodes": count(
                                0,
                                sum(
                                    json_nodes(item.expected_consequences)
                                    for item in branch_items
                                ),
                            ),
                            "cognition_metadata_values": count(
                                0, 3 * len(branch_items)
                            ),
                            "source_id_items": count(
                                sum(
                                    min(limit, len(item.source_ids))
                                    for item in branch_items
                                ),
                                sum(
                                    max(0, len(item.source_ids) - limit)
                                    for item in branch_items
                                ),
                            ),
                            "omitted_record_json_nodes": count(
                                0,
                                sum(model_nodes(item) for item in branch_omitted_items),
                            ),
                            "evicted_record_details": count(
                                0, state.working_set.thought_space.evicted
                            ),
                        },
                    },
                },
                "rd": {
                    "included": len(action_items),
                    "omitted": (
                        state.working_set.action_records.total - len(action_items)
                    ),
                    "fields": {
                        "intent_json_nodes": count(
                            0,
                            sum(json_nodes(item.intent) for item in action_items),
                        ),
                        "phase_history_items": count(
                            0,
                            sum(len(item.phase_history) for item in action_items),
                        ),
                        "receipt_exact_json_nodes": count(
                            0,
                            sum(
                                json_nodes(item.receipt)
                                for item in action_items
                                if item.receipt is not None
                            ),
                        ),
                        "reconstruction_exact_json_nodes": count(
                            0,
                            sum(
                                json_nodes(item.reconstruction)
                                for item in action_items
                                if item.reconstruction is not None
                            ),
                        ),
                        "proposed_event_id_values": count(0, len(action_items)),
                        "effect_evidence_id_items": count(
                            sum(
                                min(limit, len(action_evidence_ids(item)))
                                for item in action_items
                            ),
                            sum(
                                max(
                                    0,
                                    len(action_evidence_ids(item)) - limit,
                                )
                                for item in action_items
                            ),
                        ),
                        "omitted_record_json_nodes": count(
                            0,
                            sum(model_nodes(item) for item in action_omitted_items),
                        ),
                        "evicted_record_details": count(
                            0, state.working_set.action_records.evicted
                        ),
                    },
                },
                "a": {
                    "included": len(action_items),
                    "omitted": (
                        state.working_set.action_records.total - len(action_items)
                    ),
                    "fields": {
                        "intent_json_nodes": count(
                            0,
                            sum(json_nodes(item.intent) for item in action_items),
                        ),
                        "phase_values": count(0, len(action_items)),
                        "phase_history_items": count(
                            0,
                            sum(len(item.phase_history) for item in action_items),
                        ),
                        "rd_phase_values": count(0, len(action_items)),
                        "prepared_branch_id_values": count(
                            0,
                            sum(
                                item.prepared_branch_id is not None
                                for item in action_items
                            ),
                        ),
                        "receipt_exact_json_nodes": count(
                            0,
                            sum(
                                json_nodes(item.receipt)
                                for item in action_items
                                if item.receipt is not None
                            ),
                        ),
                        "reconstruction_exact_json_nodes": count(
                            0,
                            sum(
                                json_nodes(item.reconstruction)
                                for item in action_items
                                if item.reconstruction is not None
                            ),
                        ),
                        "effect_evidence_id_items": count(
                            0,
                            sum(
                                len(action_evidence_ids(item)) for item in action_items
                            ),
                        ),
                        "proposed_event_id_values": count(0, len(action_items)),
                        "omitted_record_json_nodes": count(
                            0,
                            sum(model_nodes(item) for item in action_omitted_items),
                        ),
                        "evicted_record_details": count(
                            0, state.working_set.action_records.evicted
                        ),
                    },
                },
                "world": {
                    "included": world_included,
                    "omitted": world_total - world_included,
                    "layers": world_omissions,
                },
                "self_boundary": {
                    "included": len(selected_other_actors),
                    "omitted": len(omitted_other_actors),
                    "fields": {
                        "self_actor_metadata_json_nodes": count(
                            0,
                            json_nodes(
                                {
                                    "display_name": state.identity.actor(
                                        state.identity.self_actor_id
                                    ).display_name,
                                    "parent_actor_id": state.identity.actor(
                                        state.identity.self_actor_id
                                    ).parent_actor_id,
                                    "attributes": state.identity.actor(
                                        state.identity.self_actor_id
                                    ).attributes,
                                }
                            ),
                        ),
                        "other_actor_metadata_json_nodes": count(
                            0,
                            sum(
                                json_nodes(
                                    {
                                        "kind": item.kind.value,
                                        "display_name": item.display_name,
                                        "parent_actor_id": item.parent_actor_id,
                                        "attributes": item.attributes,
                                    }
                                )
                                for item in selected_other_actors
                            ),
                        ),
                        "authorization_record_json_nodes": count(
                            0,
                            sum(
                                model_nodes(item)
                                for item in state.identity.authorizations
                            ),
                        ),
                        "authorization_records": count(
                            0, len(state.identity.authorizations)
                        ),
                        "omitted_actor_record_json_nodes": count(
                            0,
                            sum(model_nodes(item) for item in omitted_other_actors),
                        ),
                    },
                },
                "memory": {
                    "included": len(memory_items),
                    "omitted": (state.working_set.memories.total - len(memory_items)),
                    "fields": {
                        "content_json_nodes": count(
                            0,
                            sum(json_nodes(item.content) for item in memory_items),
                        ),
                        "source_id_items": count(
                            sum(
                                min(limit, len(item.source_ids))
                                for item in memory_items
                            ),
                            sum(
                                max(0, len(item.source_ids) - limit)
                                for item in memory_items
                            ),
                        ),
                        "omitted_record_json_nodes": count(
                            0,
                            sum(model_nodes(item) for item in memory_omitted_items),
                        ),
                        "evicted_record_details": count(
                            0, state.working_set.memories.evicted
                        ),
                    },
                },
                "capabilities": {
                    "included": len(capability_keys),
                    "omitted": len(omitted_capability_keys),
                    "fields": {
                        "non_boolean_value_json_nodes": count(
                            0,
                            sum(
                                json_nodes(state.capabilities[key])
                                for key in capability_keys
                                if not isinstance(state.capabilities[key], bool)
                            ),
                        ),
                        "omitted_entry_json_nodes": count(
                            0,
                            sum(
                                json_nodes(
                                    {
                                        "key": key,
                                        "value": state.capabilities[key],
                                    }
                                )
                                for key in omitted_capability_keys
                            ),
                        ),
                    },
                },
                "runtime": {
                    "included": 2,
                    "omitted": 3 + (state.runtime.last_failure is not None),
                    "fields": {
                        "failure_count_values": count(0, 1),
                        "consecutive_failure_values": count(0, 1),
                        "last_elapsed_seconds_values": count(0, 1),
                        "last_failure_json_nodes": count(
                            0,
                            (
                                model_nodes(state.runtime.last_failure)
                                if state.runtime.last_failure is not None
                                else 0
                            ),
                        ),
                    },
                },
                "cognition": {
                    "included": 0,
                    "omitted": 1,
                    "fields": {
                        "state_json_nodes": count(
                            0,
                            json_nodes(state.cognition.model_dump(mode="json")),
                        ),
                        "reflection_records": count(
                            0,
                            state.working_set.cognition_reflections.total,
                        ),
                        "evicted_reflection_details": count(
                            0,
                            state.working_set.cognition_reflections.evicted,
                        ),
                    },
                },
                "raw_lifecycle_counts": {
                    "included": 0,
                    "omitted": state.working_set.raw_lifecycle_counts.total,
                    "fields": {
                        "entry_json_nodes": count(
                            0, json_nodes(state.raw_lifecycle_counts)
                        ),
                        "evicted_key_details": count(
                            0,
                            state.working_set.raw_lifecycle_counts.evicted,
                        ),
                        "evicted_event_counts": count(
                            0,
                            state.working_set.raw_lifecycle_events_evicted,
                        ),
                    },
                },
                "working_set": {
                    "projection_records_visited": projection_records_visited,
                    "projection_record_budget": (FRAME_PROJECTION_RECORD_BUDGET),
                    "ledger_events_scanned": 0,
                    "counters": state.working_set.model_dump(mode="json"),
                },
            },
        )

    @staticmethod
    def _as_legacy_frame_v2(frame: ConsciousnessFrameV3) -> ConsciousnessFrameV2:
        values = frame.model_dump(mode="python")
        values["schema_version"] = 2
        values.pop("semantic_schema_version")
        values.pop("aggregate_semantic_complete")
        values.pop("semantic_evidence")
        return ConsciousnessFrameV2.model_validate(values, strict=True)

    @staticmethod
    def _legacy_v3_action_frame(
        frame: ConsciousnessFrameV2,
        state: BrainState,
    ) -> ConsciousnessFrameV2:
        """Reconstruct the exact pre-outcome action projection for migration."""
        values = frame.model_dump(mode="json")
        projected_ids = {item["action_id"] for item in values["a"]["actions"]}
        failure_ids = {
            action.action_id
            for action in state.action_records
            if action.action_id in projected_ids
            and action.receipt is not None
            and action.receipt.get("status") == "failure"
        }
        for section in ("rd", "a"):
            for action in values[section]["actions"]:
                if action["action_id"] in failure_ids:
                    action["execution_confirmed"] = False

        retained_omitted = len(state.action_records) - len(projected_ids)
        legacy_node_delta = retained_omitted * 2
        for section in ("rd", "a"):
            node_count = values["omission_counts"][section]["fields"][
                "omitted_record_json_nodes"
            ]
            node_count["omitted"] -= legacy_node_delta
        return ConsciousnessFrameV2.model_validate(values)

    def project_bridge_frame(
        self,
        bridge_instance_id: str,
        *,
        expected_state: BrainState,
        connected_nonce: str | None = None,
        scheduler_sample: str,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> ConsciousnessFrameV3:
        """Project a live frame from canonical history and an exact engine state."""
        bridge_instance_id = validate_id(bridge_instance_id)
        if not isinstance(expected_state, BrainState):
            raise TypeError("expected_state must be BrainState")
        expected_state = expected_state.revalidated()
        if scheduler_sample not in {"running", "stopped"}:
            raise ValueError("scheduler_sample must be running or stopped")
        with self._transaction(immediate=False):
            stream = self._stream_from_row(self._bridge_stream_row(bridge_instance_id))
            if stream.brain_id != expected_state.brain_id:
                raise BridgeBindingError(
                    "bridge stream brain does not match projected state"
                )
            if (
                connected_nonce is not None
                and stream.connected_nonce != connected_nonce
            ):
                raise BridgeBindingError(
                    "bridge stream is attached to another connection"
                )
            head = self._authoritative_brain_head_in_transaction(stream.brain_id)
            if head is None or head != expected_state.last_sequence:
                raise LedgerIntegrityError(
                    "projected engine state does not match authoritative head"
                )
            latest = self._validate_bridge_tail_in_transaction(stream)
            coverage: Mapping[str, object] | None = None
            if latest is not None:
                coverage = self._record_capture_coverage(latest)
            aggregate_semantic_complete, semantic_evidence = (
                self._semantic_frame_evidence_in_transaction(stream.brain_id)
            )
            frame = self._project_bridge_frame(
                expected_state,
                through_capture_seq=stream.next_capture_seq - 1,
                capture_coverage=coverage,
                aggregate_semantic_complete=aggregate_semantic_complete,
                semantic_evidence=semantic_evidence,
                scheduler_sample=scheduler_sample,
                stream_connected=stream.connected_nonce is not None,
            )
            self._enforce_frame_limit(frame, max_frame_bytes=max_frame_bytes)
            return frame

    @staticmethod
    def _enforce_frame_limit(
        frame: ConsciousnessFrameV2 | ConsciousnessFrameV3, *, max_frame_bytes: int
    ) -> None:
        if isinstance(max_frame_bytes, bool) or not isinstance(max_frame_bytes, int):
            raise TypeError("max_frame_bytes must be an integer")
        if max_frame_bytes < 1:
            raise ValueError("max_frame_bytes must be positive")
        if len(frame.canonical_json().encode("utf-8")) > max_frame_bytes:
            raise FrameSizeError("consciousness frame exceeds negotiated byte limit")

    @staticmethod
    def _record_capture_coverage(
        record: BridgeRecordV1,
    ) -> Mapping[str, object]:
        if isinstance(record, HermesObservationV1):
            return record.coverage.model_dump(mode="json")
        return {
            "policy_version": "bridge-gap-v1",
            "capture_coverage": "partial",
            "redacted_paths": 0,
            "truncated_paths": 0,
            "unsupported_paths": 0,
            # A gap counts missing bridge records, not the unknown number of
            # JSON nodes those records would have contained.
            "omitted_nodes": 0,
            "channels": {
                "dropped_records": record.dropped_count,
                "omitted_nodes_known": False,
                "trace": "gap",
            },
        }

    @staticmethod
    def _normalize_bridge_record(record: BridgeRecordV1) -> BridgeRecordV1:
        if isinstance(record, HermesObservationV1):
            return validate_observation(record.model_dump(mode="python"))
        if isinstance(record, BridgeGapV1):
            return BridgeGapV1.model_validate(record.model_dump(mode="python"))
        raise TypeError("bridge accepts only HermesObservationV1 or BridgeGapV1")

    @staticmethod
    def _bridge_event(
        stream: BridgeStreamState, record: BridgeRecordV1
    ) -> EventEnvelope:
        return build_raw_event(stream, record)

    @staticmethod
    def _span_from_row(row: Mapping[str, object]) -> HermesSpan:
        return HermesSpan(
            bridge_instance_id=str(row["bridge_instance_id"]),
            span_kind=str(row["span_kind"]),  # type: ignore[arg-type]
            external_id=str(row["external_id"]),
            occurrence_capture_seq=int(row["occurrence_capture_seq"]),
            context_fingerprint=str(row["context_fingerprint"]),
            action_id=(None if row["action_id"] is None else str(row["action_id"])),
            closed_capture_seq=(
                None
                if row["closed_capture_seq"] is None
                else int(row["closed_capture_seq"])
            ),
        )

    def _matched_span_in_transaction(
        self, record: BridgeRecordV1
    ) -> tuple[HermesSpan | None, str | None]:
        if not isinstance(record, HermesObservationV1):
            return None, None
        if record.hook == "post_tool_call":
            span_kind = "tool"
            external_id = record.context.tool_call_id
        elif record.hook in {"post_api_request", "api_request_error"}:
            span_kind = "api"
            external_id = record.context.api_request_id
        else:
            return None, None
        rows = self._connection.execute(
            "SELECT bridge_instance_id, span_kind, external_id, "
            "occurrence_capture_seq, context_fingerprint, action_id, "
            "closed_capture_seq "
            "FROM hermes_span WHERE bridge_instance_id = ? AND span_kind = ? "
            "AND external_id = ? AND context_fingerprint = ? "
            "ORDER BY occurrence_capture_seq DESC LIMIT ?",
            (
                record.bridge_instance_id,
                span_kind,
                external_id,
                span_context_fingerprint(record),
                MAX_HERMES_SPANS_PER_STREAM + 1,
            ),
        ).fetchall()
        return match_hermes_span(
            record,
            tuple(self._span_from_row(row) for row in rows),
        )

    @staticmethod
    def _semantic_lifecycle_gap(
        state: BrainState,
        record: BridgeRecordV1,
        matched_span: HermesSpan | None,
    ) -> str | None:
        """Reject a tool/domain lifecycle mismatch before deriving action events."""
        if (
            not isinstance(record, HermesObservationV1)
            or record.hook != "post_tool_call"
            or matched_span is None
        ):
            return None
        late = matched_span.closed_capture_seq is not None
        if matched_span.action_id is None:
            return "late_action_unavailable" if late else "action_unavailable"
        action = state.actions.get(matched_span.action_id)
        if action is None:
            return "late_action_unavailable" if late else "action_unavailable"
        if (
            ActionPhase.BLOCKED in action.phase_history
            or action.execution_confirmed is False
        ):
            return "late_completion_after_blocked"
        if not late:
            return (
                None
                if action.phase is ActionPhase.PREPARED
                else "action_state_mismatch"
            )
        if action.phase not in {ActionPhase.RECEIPT, ActionPhase.RECONSTRUCTED}:
            return "late_action_state_mismatch"
        return None

    def _prepare_semantic_span_capacity_in_transaction(
        self, record: BridgeRecordV1
    ) -> tuple[str | None, tuple[tuple[str, str, int], ...]]:
        if not isinstance(record, HermesObservationV1) or record.hook not in {
            "pre_tool_call",
            "pre_api_request",
        }:
            return None, ()
        count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM hermes_span WHERE bridge_instance_id = ?",
                (record.bridge_instance_id,),
            ).fetchone()[0]
        )
        required = count - MAX_HERMES_SPANS_PER_STREAM + 1
        if required <= 0:
            return None, ()
        victims = self._connection.execute(
            "SELECT span_kind, external_id, occurrence_capture_seq "
            "FROM hermes_span WHERE bridge_instance_id = ? "
            "AND closed_capture_seq IS NOT NULL "
            "ORDER BY closed_capture_seq, occurrence_capture_seq LIMIT ?",
            (record.bridge_instance_id, required),
        ).fetchall()
        selected = tuple(
            (
                str(victim["span_kind"]),
                str(victim["external_id"]),
                int(victim["occurrence_capture_seq"]),
            )
            for victim in victims
        )
        return (
            (
                "span_capacity_all_open"
                if count - len(selected) >= MAX_HERMES_SPANS_PER_STREAM
                else None
            ),
            selected,
        )

    def _evict_semantic_span_victims_in_transaction(
        self,
        bridge_instance_id: str,
        victims: tuple[tuple[str, str, int], ...],
    ) -> None:
        for span_kind, external_id, occurrence_capture_seq in victims:
            deleted = self._connection.execute(
                "DELETE FROM hermes_span WHERE bridge_instance_id = ? "
                "AND span_kind = ? AND external_id = ? "
                "AND occurrence_capture_seq = ? AND closed_capture_seq IS NOT NULL",
                (
                    bridge_instance_id,
                    span_kind,
                    external_id,
                    occurrence_capture_seq,
                ),
            )
            if deleted.rowcount != 1:
                raise LedgerIntegrityError(
                    "selected closed Hermes span was lost before eviction"
                )

    def _apply_semantic_spans_in_transaction(
        self, plan: SemanticPlan, *, closing_capture_seq: int
    ) -> None:
        if plan.span_close is not None:
            span = plan.span_close
            updated = self._connection.execute(
                "UPDATE hermes_span SET closed_capture_seq = ? "
                "WHERE bridge_instance_id = ? AND span_kind = ? "
                "AND external_id = ? AND occurrence_capture_seq = ? "
                "AND context_fingerprint = ? AND closed_capture_seq IS NULL",
                (
                    closing_capture_seq,
                    span.bridge_instance_id,
                    span.span_kind,
                    span.external_id,
                    span.occurrence_capture_seq,
                    span.context_fingerprint,
                ),
            )
            if updated.rowcount != 1:
                raise LedgerIntegrityError("matched Hermes span was lost before close")
        if plan.span_open is not None:
            span = plan.span_open
            self._connection.execute(
                "INSERT INTO hermes_span("
                "bridge_instance_id, span_kind, external_id, "
                "occurrence_capture_seq, context_fingerprint, action_id, "
                "closed_capture_seq) VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (
                    span.bridge_instance_id,
                    span.span_kind,
                    span.external_id,
                    span.occurrence_capture_seq,
                    span.context_fingerprint,
                    span.action_id,
                ),
            )
        bridge_instance_id = (
            plan.span_open.bridge_instance_id
            if plan.span_open is not None
            else plan.span_close.bridge_instance_id
            if plan.span_close is not None
            else None
        )
        if bridge_instance_id is None:
            return
        count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM hermes_span WHERE bridge_instance_id = ?",
                (bridge_instance_id,),
            ).fetchone()[0]
        )
        excess = count - MAX_HERMES_SPANS_PER_STREAM
        if excess <= 0:
            return
        victims = self._connection.execute(
            "SELECT span_kind, external_id, occurrence_capture_seq "
            "FROM hermes_span WHERE bridge_instance_id = ? "
            "AND closed_capture_seq IS NOT NULL "
            "ORDER BY closed_capture_seq, occurrence_capture_seq LIMIT ?",
            (bridge_instance_id, excess),
        ).fetchall()
        for victim in victims:
            self._connection.execute(
                "DELETE FROM hermes_span WHERE bridge_instance_id = ? "
                "AND span_kind = ? AND external_id = ? "
                "AND occurrence_capture_seq = ?",
                (
                    bridge_instance_id,
                    victim["span_kind"],
                    victim["external_id"],
                    victim["occurrence_capture_seq"],
                ),
            )
        remaining = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM hermes_span WHERE bridge_instance_id = ?",
                (bridge_instance_id,),
            ).fetchone()[0]
        )
        if remaining > MAX_HERMES_SPANS_PER_STREAM:
            raise LedgerIntegrityError(
                "open Hermes spans exceed capacity; no open span was discarded"
            )

    def _decode_legacy_duplicate_bridge_record(
        self,
        row: sqlite3.Row,
        *,
        requested: BridgeRecordV1,
        requested_json: str,
        requested_fingerprint: str,
        historical_state: BrainState | None = None,
        event_row: Mapping[str, Any] | None = None,
        migrate_legacy_action_frame: bool = False,
    ) -> BridgeCommitAckV1:
        if (
            row["record_fingerprint"] != requested_fingerprint
            or row["record_json"] != requested_json
        ):
            raise IdempotencyConflictError(
                "bridge idempotency key has a different immutable body"
            )
        if (
            row["bridge_instance_id"] != requested.bridge_instance_id
            or row["first_capture_seq"] != requested.first_capture_seq
            or row["last_capture_seq"] != requested.last_capture_seq
            or row["record_kind"] != requested.record_kind
        ):
            raise LedgerIntegrityError(
                "persisted bridge record keys do not match its canonical body"
            )
        try:
            persisted = validate_bridge_record_json(row["record_json"])
            accepted_at = datetime.fromisoformat(row["accepted_at"])
            ack = BridgeCommitAckV1.model_validate_json(row["ack_json"])
        except Exception as error:
            raise LedgerIntegrityError(
                "persisted bridge record or acknowledgement is invalid"
            ) from error
        if (
            type(persisted) is not type(requested)
            or persisted != requested
            or persisted.canonical_json() != row["record_json"]
            or accepted_at.tzinfo is None
            or accepted_at.utcoffset() is None
            or accepted_at.astimezone(UTC).isoformat() != row["accepted_at"]
            or ack.canonical_json() != row["ack_json"]
        ):
            raise LedgerIntegrityError(
                "persisted bridge canonical data does not match its record"
            )
        stream = self._stream_from_row(
            self._bridge_stream_row(requested.bridge_instance_id)
        )
        if accepted_at.astimezone(UTC) > stream.last_seen:
            raise LedgerIntegrityError(
                "persisted bridge acceptance follows stream last_seen"
            )
        if stream.next_capture_seq <= requested.last_capture_seq:
            raise LedgerIntegrityError(
                "persisted bridge cursor does not cover its acknowledgement"
            )
        if event_row is None:
            event_row = self._connection.execute(
                "SELECT event_id, brain_id, sequence, body_fingerprint, "
                "envelope_fingerprint, envelope_json FROM events "
                "WHERE event_id = ?",
                (row["event_id"],),
            ).fetchone()
        if event_row is None:
            raise LedgerIntegrityError(
                "persisted bridge acknowledgement event is missing"
            )
        event = self._decode_event(event_row)
        if (
            event.sequence != row["ledger_sequence"]
            or event.brain_id != stream.brain_id
            or event.actor_id != stream.server_actor_id
            or event.adapter_id != stream.server_adapter_id
        ):
            raise LedgerIntegrityError(
                "persisted bridge event provenance does not match its stream"
            )
        if isinstance(persisted, HermesObservationV1):
            session_id = getattr(persisted.context, "session_id", None)
            turn_id = getattr(persisted.context, "turn_id", None)
            api_request_id = getattr(persisted.context, "api_request_id", None)
            event_semantics_match = (
                event.event_type == HOOK_EVENT_TYPES[persisted.hook]
                and event.payload == persisted.model_dump(mode="json")
                and event.wall_time == persisted.captured_at
                and event.monotonic_ns == persisted.captured_monotonic_ns
                and event.session_id == session_id
                and event.turn_id == turn_id
                and event.correlation_id == api_request_id
            )
        else:
            event_semantics_match = (
                event.event_type == "trace.gap"
                and event.payload
                == {
                    **persisted.model_dump(mode="json"),
                    "exact": True,
                    "trace_complete": False,
                }
            )
        if not event_semantics_match:
            raise LedgerIntegrityError(
                "persisted bridge event does not represent its canonical record"
            )
        if historical_state is not None and (
            historical_state.brain_id != stream.brain_id
            or historical_state.last_sequence != event.sequence
        ):
            raise LedgerIntegrityError(
                "historical bridge state does not match its event sequence"
            )
        expected_frame = None
        if historical_state is not None:
            projected = self._project_bridge_frame(
                historical_state,
                through_capture_seq=persisted.last_capture_seq,
                capture_coverage=self._record_capture_coverage(persisted),
                stream_connected=True,
            )
            expected_frame = self._as_legacy_frame_v2(projected)
        bounded_frame_relations_match = (
            ack.frame.brain_id == stream.brain_id
            and ack.frame.state_sequence == event.sequence
            and ack.frame.through_capture_seq == persisted.last_capture_seq
            and ack.frame.freshness.projected_at_state_sequence == event.sequence
            and ack.frame.freshness.scheduler_sample == "not_sampled"
            and ack.frame.freshness.stream_connection == "connected"
            and ack.frame.capture_coverage == self._record_capture_coverage(persisted)
        )
        frame_matches = expected_frame is None or ack.frame == expected_frame
        migrate_frame = False
        if (
            not frame_matches
            and migrate_legacy_action_frame
            and expected_frame is not None
            and historical_state is not None
        ):
            legacy_frame = self._legacy_v3_action_frame(
                expected_frame,
                historical_state,
            )
            frame_matches = ack.frame == legacy_frame
            migrate_frame = frame_matches
        if (
            ack.record_fingerprint != requested_fingerprint
            or ack.duplicate is not False
            or ack.event_id != event.event_id
            or ack.event_sequence != event.sequence
            or ack.through_capture_seq != persisted.last_capture_seq
            or not bounded_frame_relations_match
            or not frame_matches
        ):
            raise LedgerIntegrityError(
                "persisted bridge acknowledgement does not match replay"
            )
        if migrate_frame:
            if expected_frame is None:
                raise AssertionError("legacy frame migration requires replay")
            migrated = BridgeCommitAckV1(
                record_fingerprint=ack.record_fingerprint,
                duplicate=ack.duplicate,
                event_id=ack.event_id,
                event_sequence=ack.event_sequence,
                frame=expected_frame,
                through_capture_seq=ack.through_capture_seq,
            )
            migrated_json = migrated.canonical_json()
            updated = self._connection.execute(
                "UPDATE bridge_record SET ack_json = ? "
                "WHERE bridge_instance_id = ? AND first_capture_seq = ? "
                "AND ack_json = ?",
                (
                    migrated_json,
                    requested.bridge_instance_id,
                    requested.first_capture_seq,
                    row["ack_json"],
                ),
            )
            if updated.rowcount != 1:
                raise LedgerIntegrityError(
                    "legacy bridge acknowledgement migration lost its row"
                )
            ack = migrated
        return ack

    @staticmethod
    def _raw_bridge_event_matches(event: EventEnvelope, record: BridgeRecordV1) -> bool:
        if isinstance(record, HermesObservationV1):
            return (
                event.event_type == HOOK_EVENT_TYPES[record.hook]
                and event.payload == record.model_dump(mode="json")
                and event.wall_time == record.captured_at
                and event.monotonic_ns == record.captured_monotonic_ns
                and event.session_id == getattr(record.context, "session_id", None)
                and event.turn_id == getattr(record.context, "turn_id", None)
                and event.correlation_id
                == getattr(record.context, "api_request_id", None)
            )
        return event.event_type == "trace.gap" and event.payload == {
            **record.model_dump(mode="json"),
            "exact": True,
            "trace_complete": False,
        }

    def _decode_duplicate_bridge_record(
        self,
        row: Mapping[str, object],
        *,
        requested: BridgeRecordV1,
        requested_json: str,
        requested_fingerprint: str,
        historical_state: BrainState | None = None,
    ) -> BridgeCommitAckV2:
        if (
            row["record_fingerprint"] != requested_fingerprint
            or row["record_json"] != requested_json
        ):
            raise IdempotencyConflictError(
                "bridge idempotency key has a different immutable body"
            )
        if (
            row["bridge_instance_id"] != requested.bridge_instance_id
            or int(row["first_capture_seq"]) != requested.first_capture_seq
            or int(row["last_capture_seq"]) != requested.last_capture_seq
            or row["record_kind"] != requested.record_kind
        ):
            raise LedgerIntegrityError(
                "persisted bridge record keys do not match its canonical body"
            )
        try:
            persisted = validate_bridge_record_json(str(row["record_json"]))
            accepted_at = datetime.fromisoformat(str(row["accepted_at"]))
            ack = BridgeCommitAckV2.model_validate_json(str(row["ack_json"]))
        except Exception as error:
            raise LedgerIntegrityError(
                "persisted semantic bridge record or acknowledgement is invalid"
            ) from error
        raw_sequence = int(row["ledger_sequence"])
        derived_count = int(row["derived_event_count"])
        derived_first = row["derived_first_sequence"]
        derived_last = row["derived_last_sequence"]
        last_sequence = raw_sequence + derived_count
        if (
            type(persisted) is not type(requested)
            or persisted != requested
            or persisted.canonical_json() != row["record_json"]
            or accepted_at.tzinfo is None
            or accepted_at.utcoffset() is None
            or accepted_at.astimezone(UTC).isoformat() != row["accepted_at"]
            or ack.canonical_json() != row["ack_json"]
            or row["semantic_status"] != ack.semantic_status
            or bool(row["semantic_complete"]) is not ack.semantic_complete
            or row["semantic_fingerprint"] != ack.semantic_fingerprint
            or derived_count != ack.derived_event_count
            or (
                derived_count == 0
                and (derived_first is not None or derived_last is not None)
            )
            or (
                derived_count > 0
                and (
                    int(derived_first) != raw_sequence + 1
                    or int(derived_last) != last_sequence
                )
            )
        ):
            raise LedgerIntegrityError(
                "persisted semantic bridge canonical data does not match"
            )
        stream = self._stream_from_row(
            self._bridge_stream_row(requested.bridge_instance_id)
        )
        if (
            accepted_at.astimezone(UTC) > stream.last_seen
            or stream.next_capture_seq <= requested.last_capture_seq
        ):
            raise LedgerIntegrityError(
                "persisted semantic bridge cursor or acceptance is invalid"
            )
        event_rows = self._connection.execute(
            "SELECT event_id, brain_id, sequence, body_fingerprint, "
            "envelope_fingerprint, envelope_json FROM events "
            "WHERE brain_id = ? AND sequence BETWEEN ? AND ? ORDER BY sequence",
            (stream.brain_id, raw_sequence, last_sequence),
        ).fetchall()
        if len(event_rows) != derived_count + 1:
            raise LedgerIntegrityError("semantic bridge batch events are missing")
        events = tuple(self._decode_event(event_row) for event_row in event_rows)
        raw_event, *derived_events = events
        semantic_gap_shape_matches = True
        if ack.semantic_status == "gap":
            semantic_gap_shape_matches = (
                isinstance(persisted, BridgeGapV1) and not derived_events
            ) or (
                isinstance(persisted, HermesObservationV1)
                and len(derived_events) == 1
                and derived_events[0].event_type == "semantic.gap"
            )
        if (
            raw_event.event_id != row["event_id"]
            or raw_event.event_id != ack.raw_event_id
            or raw_event.sequence != raw_sequence
            or ack.raw_event_sequence != raw_sequence
            or ack.last_event_sequence != last_sequence
            or tuple(event.event_id for event in derived_events)
            != ack.derived_event_ids
            or not semantic_gap_shape_matches
            or any(
                event.brain_id != stream.brain_id
                or event.actor_id != stream.server_actor_id
                or event.adapter_id != stream.server_adapter_id
                for event in events
            )
            or not self._raw_bridge_event_matches(raw_event, persisted)
        ):
            raise LedgerIntegrityError(
                "persisted semantic bridge events do not match their record"
            )
        if historical_state is not None:
            expected_frame = self._project_bridge_frame(
                historical_state,
                through_capture_seq=persisted.last_capture_seq,
                capture_coverage=self._record_capture_coverage(persisted),
                aggregate_semantic_complete=(ack.frame.aggregate_semantic_complete),
                semantic_evidence=ack.frame.semantic_evidence.model_dump(mode="python"),
                stream_connected=True,
            )
            if (
                historical_state.brain_id != stream.brain_id
                or historical_state.last_sequence != last_sequence
                or ack.frame != expected_frame
            ):
                raise LedgerIntegrityError(
                    "persisted semantic acknowledgement does not match replay"
                )
        elif (
            ack.frame.brain_id != stream.brain_id
            or ack.frame.state_sequence != last_sequence
            or ack.frame.through_capture_seq != persisted.last_capture_seq
            or ack.frame.capture_coverage != self._record_capture_coverage(persisted)
        ):
            raise LedgerIntegrityError(
                "persisted semantic acknowledgement frame is not bound"
            )
        return ack

    def commit_bridge_record(
        self,
        bridge_instance_id: str,
        record: BridgeRecordV1,
        *,
        expected_state: BrainState,
        connected_nonce: str | None = None,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        max_ack_bytes: int = DEFAULT_MAX_ACK_BYTES,
    ) -> BridgeCommitResult:
        """Atomically commit raw, bounded semantic batch, cursor, frame and ACK."""
        bridge_instance_id = validate_id(bridge_instance_id)
        record = self._normalize_bridge_record(record)
        if record.bridge_instance_id != bridge_instance_id:
            raise BridgeBindingError("record bridge instance does not match binding")
        expected_state = expected_state.revalidated()
        record_json = record.canonical_json()
        fingerprint = record.fingerprint()
        first_capture_seq = record.first_capture_seq
        last_capture_seq = record.last_capture_seq
        with self._transaction(immediate=True):
            stream = self._stream_from_row(self._bridge_stream_row(bridge_instance_id))
            accepted_at = self._bridge_timestamp(stream.last_seen)
            if stream.brain_id != expected_state.brain_id:
                raise BridgeBindingError("bridge stream brain does not match engine")
            if stream.status != "open":
                raise BridgeClosedError("bridge stream is not open")
            if (
                connected_nonce is not None
                and stream.connected_nonce != connected_nonce
            ):
                raise BridgeBindingError(
                    "bridge stream is attached to another connection"
                )
            duplicate = self._connection.execute(
                "SELECT bridge_instance_id, first_capture_seq, last_capture_seq, "
                "record_kind, record_fingerprint, record_json, event_id, "
                "ledger_sequence, semantic_status, semantic_complete, "
                "semantic_fingerprint, derived_event_count, "
                "derived_first_sequence, derived_last_sequence, ack_json, "
                "accepted_at FROM bridge_record "
                "WHERE bridge_instance_id = ? AND first_capture_seq = ?",
                (bridge_instance_id, first_capture_seq),
            ).fetchone()
            if duplicate is not None:
                ack = self._decode_duplicate_bridge_record(
                    duplicate,
                    requested=record,
                    requested_json=record_json,
                    requested_fingerprint=fingerprint,
                )
                self._enforce_frame_limit(ack.frame, max_frame_bytes=max_frame_bytes)
                self._enforce_ack_limit(
                    ack.canonical_json(), max_ack_bytes=max_ack_bytes
                )
                return BridgeCommitResult(ack=ack, successor=None)

            if first_capture_seq != stream.next_capture_seq:
                if (
                    isinstance(record, HermesObservationV1)
                    and first_capture_seq > stream.next_capture_seq
                ):
                    raise CaptureGapRequiredError(
                        "later observation requires an exact persisted gap"
                    )
                raise CaptureSequenceError(
                    "bridge record does not begin at the capture cursor"
                )

            expected_sequence = expected_state.last_sequence + 1
            brain_row = self._connection.execute(
                "SELECT next_sequence FROM brains WHERE brain_id = ?",
                (stream.brain_id,),
            ).fetchone()
            head = int(
                self._connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM events WHERE brain_id = ?",
                    (stream.brain_id,),
                ).fetchone()[0]
            )
            if (
                brain_row is None
                or int(brain_row["next_sequence"]) != expected_sequence
                or head != expected_state.last_sequence
            ):
                raise ExpectedSequenceError(
                    "bridge engine state does not match the authoritative head"
                )

            raw_event = self._bridge_event(stream, record)
            capacity_gap, span_capacity_victims = (
                self._prepare_semantic_span_capacity_in_transaction(record)
            )
            matched_span, correlation_gap = self._matched_span_in_transaction(record)
            lifecycle_gap = self._semantic_lifecycle_gap(
                expected_state,
                record,
                matched_span,
            )
            plan = build_semantic_plan(
                stream,
                record,
                raw_event=raw_event,
                matched_span=matched_span,
                forced_gap_reason=capacity_gap or correlation_gap or lifecycle_gap,
            )

            def sequence_and_reduce(
                semantic_plan: SemanticPlan,
            ) -> tuple[tuple[EventEnvelope, ...], BrainState]:
                unsequenced = (raw_event, *semantic_plan.derived_events)
                provisionals = tuple(
                    event.model_copy(
                        update={"sequence": expected_sequence + offset}
                    ).revalidated()
                    for offset, event in enumerate(unsequenced)
                )
                reduced = expected_state
                for provisional in provisionals:
                    reduced = reduce_state(reduced, provisional)
                return provisionals, reduced

            try:
                provisional_events, successor = sequence_and_reduce(plan)
            except DomainCapacityError:
                if plan.semantic_status != "applied":
                    raise
                plan = build_semantic_plan(
                    stream,
                    record,
                    raw_event=raw_event,
                    matched_span=matched_span,
                    forced_gap_reason="semantic_domain_capacity",
                )
                provisional_events, successor = sequence_and_reduce(plan)
            raw_provisional = provisional_events[0]
            derived_provisionals = provisional_events[1:]
            last_event_sequence = provisional_events[-1].sequence
            if last_event_sequence is None:
                raise AssertionError("semantic batch sequence allocation failed")
            aggregate_semantic_complete, semantic_evidence = (
                self._semantic_frame_evidence_in_transaction(
                    stream.brain_id,
                    pending_record=record,
                    pending_plan=plan,
                )
            )
            frame = self._project_bridge_frame(
                successor,
                through_capture_seq=last_capture_seq,
                capture_coverage=self._record_capture_coverage(record),
                aggregate_semantic_complete=aggregate_semantic_complete,
                semantic_evidence=semantic_evidence,
                stream_connected=True,
            )
            self._enforce_frame_limit(frame, max_frame_bytes=max_frame_bytes)
            semantic_fingerprint = plan.fingerprint()
            ack = BridgeCommitAckV2(
                record_fingerprint=fingerprint,
                raw_event_id=raw_provisional.event_id,
                raw_event_sequence=expected_sequence,
                derived_event_ids=tuple(
                    event.event_id for event in derived_provisionals
                ),
                derived_event_count=len(derived_provisionals),
                last_event_sequence=last_event_sequence,
                semantic_status=plan.semantic_status,
                semantic_complete=plan.semantic_complete,
                semantic_fingerprint=semantic_fingerprint,
                frame=frame,
                through_capture_seq=last_capture_seq,
            )
            ack_json = ack.canonical_json()
            self._enforce_ack_limit(ack_json, max_ack_bytes=max_ack_bytes)
            updated = self._connection.execute(
                "UPDATE brains SET next_sequence = ? "
                "WHERE brain_id = ? AND next_sequence = ?",
                (last_event_sequence + 1, stream.brain_id, expected_sequence),
            )
            if updated.rowcount != 1:
                raise ExpectedSequenceError(
                    "bridge expected sequence was lost before event insert"
                )
            for provisional in provisional_events:
                self._connection.execute(
                    "INSERT INTO events("
                    "brain_id, sequence, event_id, body_fingerprint, "
                    "envelope_fingerprint, envelope_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        provisional.brain_id,
                        provisional.sequence,
                        provisional.event_id,
                        provisional.body_fingerprint(),
                        provisional.envelope_fingerprint(),
                        provisional.canonical_json(),
                    ),
                )
            cursor_update = self._connection.execute(
                "UPDATE bridge_stream SET next_capture_seq = ?, last_seen = ? "
                "WHERE bridge_instance_id = ? AND next_capture_seq = ?",
                (
                    last_capture_seq + 1,
                    accepted_at,
                    bridge_instance_id,
                    first_capture_seq,
                ),
            )
            if cursor_update.rowcount != 1:
                raise ExpectedSequenceError(
                    "bridge capture cursor was lost before record insert"
                )
            if plan.span_open is not None:
                self._evict_semantic_span_victims_in_transaction(
                    bridge_instance_id,
                    span_capacity_victims,
                )
            self._apply_semantic_spans_in_transaction(
                plan,
                closing_capture_seq=last_capture_seq,
            )
            self._record_semantic_observability(stream.brain_id, record, plan)
            derived_first_sequence = (
                None if not derived_provisionals else expected_sequence + 1
            )
            derived_last_sequence = (
                None if not derived_provisionals else last_event_sequence
            )
            self._connection.execute(
                "INSERT INTO bridge_record("
                "bridge_instance_id, first_capture_seq, last_capture_seq, "
                "record_kind, record_fingerprint, record_json, event_id, "
                "ledger_sequence, semantic_status, semantic_complete, "
                "semantic_fingerprint, derived_event_count, "
                "derived_first_sequence, derived_last_sequence, ack_json, "
                "accepted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    bridge_instance_id,
                    first_capture_seq,
                    last_capture_seq,
                    record.record_kind,
                    fingerprint,
                    record_json,
                    raw_provisional.event_id,
                    expected_sequence,
                    plan.semantic_status,
                    int(plan.semantic_complete),
                    semantic_fingerprint,
                    len(derived_provisionals),
                    derived_first_sequence,
                    derived_last_sequence,
                    ack_json,
                    accepted_at,
                ),
            )
        return BridgeCommitResult(ack=ack, successor=successor)

    @staticmethod
    def _enforce_ack_limit(ack_json: str, *, max_ack_bytes: int) -> None:
        if isinstance(max_ack_bytes, bool) or not isinstance(max_ack_bytes, int):
            raise TypeError("max_ack_bytes must be an integer")
        if max_ack_bytes < 1:
            raise ValueError("max_ack_bytes must be positive")
        if len(ack_json.encode("utf-8")) > max_ack_bytes:
            raise ResponseSizeError(
                "bridge acknowledgement exceeds the negotiated result budget"
            )

    def get_event_and_head(
        self, event_id: str, brain_id: str
    ) -> tuple[EventEnvelope | None, int]:
        """Read an event ID and one brain head at one SQLite snapshot."""
        event_id = validate_id(event_id)
        brain_id = validate_id(brain_id)
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT event.event_id, event.brain_id, event.sequence, "
                "event.body_fingerprint, event.envelope_fingerprint, "
                "event.envelope_json, "
                "COALESCE((SELECT MAX(sequence) FROM events "
                "WHERE brain_id = ?), 0) AS brain_head "
                "FROM (SELECT 1) AS singleton "
                "LEFT JOIN events AS event ON event.event_id = ?",
                (brain_id, event_id),
            ).fetchone()
            if row is None:
                raise LedgerIntegrityError("event/head read returned no row")
            event = None if row["event_id"] is None else self._decode_event(row)
            return event, int(row["brain_head"])

    def append_expected(
        self, event: EventEnvelope, *, expected_sequence: int
    ) -> tuple[EventEnvelope, bool]:
        """Append only at an exact sequence, returning ``(event, inserted)``.

        The comparison and insert share one ``BEGIN IMMEDIATE`` transaction.
        A raced exact next event is adopted only while it is still the head.
        """
        expected = self._validate_expected_sequence(expected_sequence)
        return self._append(event, expected_sequence=expected)

    def _append(
        self,
        event: EventEnvelope,
        *,
        expected_sequence: int | None,
    ) -> tuple[EventEnvelope, bool]:
        event = self._normalize_event(event)
        body_fingerprint = event.body_fingerprint()
        with self._transaction(immediate=True):
            head = None
            if expected_sequence is not None:
                head = int(
                    self._connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) FROM events "
                        "WHERE brain_id = ?",
                        (event.brain_id,),
                    ).fetchone()[0]
                )
            matches = self._connection.execute(
                "SELECT event_id, brain_id, sequence, body_fingerprint, "
                "envelope_fingerprint, envelope_json "
                "FROM events WHERE event_id = ? OR body_fingerprint = ?",
                (event.event_id, body_fingerprint),
            ).fetchall()
            if len(matches) > 1:
                raise LedgerIntegrityError(
                    f"event ID {event.event_id} resolves to multiple ledger rows"
                )
            existing = matches[0] if matches else None
            if existing is not None:
                stored = self._decode_event(existing)
                if existing[
                    "body_fingerprint"
                ] != body_fingerprint or stored.canonical_json(
                    exclude_sequence=True
                ) != event.canonical_json(exclude_sequence=True):
                    if expected_sequence is not None and head != expected_sequence - 1:
                        raise ExpectedSequenceError(
                            f"expected sequence {expected_sequence}, but target "
                            f"brain head is {head}"
                        )
                    raise EventConflictError(
                        f"event ID {event.event_id} already has a different body"
                    )
                if event.sequence is not None and event.sequence != stored.sequence:
                    raise EventConflictError(
                        f"event ID {event.event_id} retry sequence "
                        f"{event.sequence} does not match stored sequence "
                        f"{stored.sequence}"
                    )
                if expected_sequence is not None and (
                    stored.sequence != expected_sequence or head != expected_sequence
                ):
                    raise ExpectedSequenceError(
                        f"expected sequence {expected_sequence} for brain "
                        f"{event.brain_id}, but exact event is at sequence "
                        f"{stored.sequence} and current head is {head}"
                    )
                return stored, False

            if event.sequence is not None:
                raise ValueError(
                    "a new persistent event must not supply an allocated sequence"
                )
            self._connection.execute(
                "INSERT INTO brains(brain_id, next_sequence) VALUES (?, 1) "
                "ON CONFLICT(brain_id) DO NOTHING",
                (event.brain_id,),
            )
            self._ensure_observability_row(event.brain_id)
            sequence = int(
                self._connection.execute(
                    "SELECT next_sequence FROM brains WHERE brain_id = ?",
                    (event.brain_id,),
                ).fetchone()[0]
            )
            if expected_sequence is not None and sequence != expected_sequence:
                raise ExpectedSequenceError(
                    f"expected sequence {expected_sequence} for brain "
                    f"{event.brain_id}, but next sequence is {sequence}"
                )
            self._connection.execute(
                "UPDATE brains SET next_sequence = ? WHERE brain_id = ?",
                (sequence + 1, event.brain_id),
            )
            stored = EventEnvelope.model_validate(
                {**event.model_dump(mode="python"), "sequence": sequence}
            )
            self._connection.execute(
                "INSERT INTO events("
                "brain_id, sequence, event_id, body_fingerprint, "
                "envelope_fingerprint, envelope_json"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    stored.brain_id,
                    sequence,
                    stored.event_id,
                    body_fingerprint,
                    stored.envelope_fingerprint(),
                    stored.canonical_json(),
                ),
            )
            if stored.event_type in {"brain.created", "identity.named"}:
                self._record_identity_name_in_transaction(
                    brain_id=stored.brain_id,
                    name=stored.payload.get("name"),
                    source_event_id=stored.event_id,
                )
            if stored.event_type in {"semantic.gap", "trace.gap"}:
                self._record_unbounded_gap_observability(stored.brain_id)
            return stored, True

    @staticmethod
    def _validate_page(after_sequence: int, limit: int) -> None:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise TypeError("after_sequence must be an integer")
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")

    def list_events(
        self,
        brain_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[EventEnvelope]:
        """Read one ordered page strictly after a validated cursor."""
        brain_id = validate_id(brain_id)
        self._validate_page(after_sequence, limit)
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT event_id, brain_id, sequence, body_fingerprint, "
                "envelope_fingerprint, envelope_json "
                "FROM events WHERE brain_id = ? AND sequence > ? "
                "ORDER BY sequence ASC LIMIT ?",
                (brain_id, after_sequence, limit),
            ).fetchall()
            return [self._decode_event(row) for row in rows]

    def iter_events(
        self,
        brain_id: str,
        *,
        after_sequence: int = 0,
        page_size: int = 512,
    ) -> Iterator[EventEnvelope]:
        """Iterate every later event through explicit ordered pagination."""
        validate_id(brain_id)
        self._validate_page(after_sequence, page_size)
        cursor = after_sequence
        while page := self.list_events(
            brain_id, after_sequence=cursor, limit=page_size
        ):
            yield from page
            cursor = page[-1].sequence or cursor

    @staticmethod
    def _snapshot_fingerprint(state_json: str) -> str:
        return hashlib.sha256(state_json.encode("utf-8")).hexdigest()

    def _replay_target_states_in_transaction(
        self, brain_id: str, target_sequences: set[int]
    ) -> tuple[dict[int, BrainState], BrainState]:
        brain_id = validate_id(brain_id)
        if any(
            isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0
            for sequence in target_sequences
        ):
            raise LedgerIntegrityError("replay targets must be non-negative integers")
        state = BrainState.genesis(brain_id)
        captured: dict[int, BrainState] = {}
        if 0 in target_sequences:
            captured[0] = state
        cursor = self._connection.execute(
            "SELECT event_id, brain_id, sequence, body_fingerprint, "
            "envelope_fingerprint, envelope_json FROM events "
            "WHERE brain_id = ? ORDER BY sequence ASC",
            (brain_id,),
        )
        while rows := cursor.fetchmany(512):
            for row in rows:
                state = reduce_state(state, self._decode_event(row))
                if state.last_sequence in target_sequences:
                    captured[state.last_sequence] = state
        if set(captured) != target_sequences:
            raise LedgerIntegrityError(
                "replay target sequence is absent from contiguous history"
            )
        return captured, state

    def _full_replay_in_transaction(
        self, brain_id: str, *, through_sequence: int | None = None
    ) -> BrainState:
        state = BrainState.genesis(brain_id)
        parameters: tuple[Any, ...]
        where = "brain_id = ?"
        parameters = (brain_id,)
        if through_sequence is not None:
            where += " AND sequence <= ?"
            parameters = (brain_id, through_sequence)
        cursor = self._connection.execute(
            "SELECT event_id, brain_id, sequence, body_fingerprint, "
            "envelope_fingerprint, envelope_json "
            f"FROM events WHERE {where} ORDER BY sequence ASC",
            parameters,
        )
        while rows := cursor.fetchmany(512):
            for row in rows:
                state = reduce_state(state, self._decode_event(row))
        return state

    def _validate_snapshot_schemas_in_transaction(self, brain_id: str) -> None:
        rows = self._connection.execute(
            "SELECT DISTINCT schema_version FROM snapshots "
            "WHERE brain_id = ? ORDER BY schema_version",
            (brain_id,),
        ).fetchall()
        supported = {*_LEGACY_STATE_SCHEMA_VERSIONS, STATE_SCHEMA_VERSION}
        for row in rows:
            schema_version = row["schema_version"]
            if (
                isinstance(schema_version, bool)
                or not isinstance(schema_version, int)
                or schema_version not in supported
            ):
                raise SchemaVersionError(
                    f"unsupported snapshot schema {schema_version!r}"
                )

    def save_snapshot(self, state: BrainState) -> BrainState:
        """Save only a monotonic snapshot exactly equivalent to full replay."""
        if not isinstance(state, BrainState):
            raise TypeError("save_snapshot accepts only BrainState instances")
        state = state.revalidated()
        with self._transaction(immediate=True):
            brain = self._connection.execute(
                "SELECT 1 FROM brains WHERE brain_id = ?",
                (state.brain_id,),
            ).fetchone()
            if brain is None:
                raise KeyError(state.brain_id)
            self._validate_snapshot_schemas_in_transaction(state.brain_id)
            maximum_row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS maximum "
                "FROM events WHERE brain_id = ?",
                (state.brain_id,),
            ).fetchone()
            maximum = int(maximum_row["maximum"])
            if state.last_sequence > maximum:
                raise SnapshotConflictError(
                    f"snapshot sequence {state.last_sequence} points past ledger "
                    f"sequence {maximum}"
                )
            latest = self._connection.execute(
                "SELECT sequence, schema_version, fingerprint, state_json "
                "FROM snapshots "
                "WHERE brain_id = ? ORDER BY sequence DESC LIMIT 1",
                (state.brain_id,),
            ).fetchone()
            if latest is not None and latest["schema_version"] not in {
                *_LEGACY_STATE_SCHEMA_VERSIONS,
                STATE_SCHEMA_VERSION,
            }:
                raise SchemaVersionError(
                    f"unsupported snapshot schema {latest['schema_version']}"
                )
            if latest is not None and state.last_sequence < latest["sequence"]:
                raise SnapshotConflictError(
                    f"snapshot sequence {state.last_sequence} is older than "
                    f"saved sequence {latest['sequence']}"
                )

            expected = self._full_replay_in_transaction(
                state.brain_id, through_sequence=state.last_sequence
            )
            if expected.canonical_json() != state.canonical_json():
                raise SnapshotConflictError(
                    "snapshot does not equal deterministic full replay"
                )
            state_json = state.canonical_json()
            fingerprint = self._snapshot_fingerprint(state_json)
            if latest is not None and state.last_sequence == latest["sequence"]:
                if (
                    latest["schema_version"] == STATE_SCHEMA_VERSION
                    and latest["fingerprint"] != fingerprint
                ):
                    raise SnapshotConflictError(
                        "snapshot sequence already has a different state"
                    )
                if latest["schema_version"] == STATE_SCHEMA_VERSION:
                    return self._decode_snapshot(latest, state.brain_id)
                if latest["schema_version"] not in _LEGACY_STATE_SCHEMA_VERSIONS:
                    raise SchemaVersionError(
                        f"unsupported snapshot schema {latest['schema_version']}"
                    )
                self._connection.execute(
                    "UPDATE snapshots SET schema_version = ?, fingerprint = ?, "
                    "state_json = ? WHERE brain_id = ? AND sequence = ?",
                    (
                        STATE_SCHEMA_VERSION,
                        fingerprint,
                        state_json,
                        state.brain_id,
                        state.last_sequence,
                    ),
                )
                return state

            self._connection.execute(
                "INSERT INTO snapshots("
                "brain_id, sequence, schema_version, fingerprint, state_json"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    state.brain_id,
                    state.last_sequence,
                    STATE_SCHEMA_VERSION,
                    fingerprint,
                    state_json,
                ),
            )
            return state

    @staticmethod
    def _decode_snapshot(row: sqlite3.Row, brain_id: str) -> BrainState:
        if row["schema_version"] != STATE_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"unsupported snapshot schema {row['schema_version']}"
            )
        try:
            state = BrainState.model_validate_json(row["state_json"])
        except Exception as error:
            raise LedgerIntegrityError("persisted snapshot state is invalid") from error
        if state.brain_id != brain_id or state.last_sequence != row["sequence"]:
            raise LedgerIntegrityError("snapshot row keys do not match its state")
        canonical_json = state.canonical_json()
        if row["state_json"] != canonical_json:
            raise LedgerIntegrityError("persisted snapshot state is not canonical")
        fingerprint = SQLiteLedger._snapshot_fingerprint(canonical_json)
        if fingerprint != row["fingerprint"]:
            raise LedgerIntegrityError("snapshot fingerprint does not match its state")
        return state

    def _load_snapshot_in_transaction(self, brain_id: str) -> BrainState | None:
        row = self._connection.execute(
            "SELECT sequence, schema_version, fingerprint, state_json "
            "FROM snapshots WHERE brain_id = ? ORDER BY sequence DESC LIMIT 1",
            (brain_id,),
        ).fetchone()
        if row is None or row["schema_version"] in _LEGACY_STATE_SCHEMA_VERSIONS:
            return None
        return self._decode_snapshot(row, brain_id)

    def load_snapshot(self, brain_id: str) -> BrainState | None:
        """Load the latest compatible snapshot for a brain."""
        brain_id = validate_id(brain_id)
        with self._transaction(immediate=False):
            self._validate_snapshot_schemas_in_transaction(brain_id)
            return self._load_snapshot_in_transaction(brain_id)

    def replay(self, brain_id: str, *, use_snapshot: bool = True) -> BrainState:
        """Replay a consistent, untruncated ledger view into frozen state."""
        brain_id = validate_id(brain_id)
        if not isinstance(use_snapshot, bool):
            raise TypeError("use_snapshot must be a boolean")
        with self._transaction(immediate=False):
            self._validate_snapshot_schemas_in_transaction(brain_id)
            state = (
                self._load_snapshot_in_transaction(brain_id) if use_snapshot else None
            )
            if state is None:
                state = BrainState.genesis(brain_id)
            cursor = self._connection.execute(
                "SELECT event_id, brain_id, sequence, body_fingerprint, "
                "envelope_fingerprint, envelope_json "
                "FROM events WHERE brain_id = ? AND sequence > ? "
                "ORDER BY sequence ASC",
                (brain_id, state.last_sequence),
            )
            while rows := cursor.fetchmany(512):
                for row in rows:
                    state = reduce_state(state, self._decode_event(row))
            return state

    def close(self) -> None:
        """Close this connection; subsequent operations fail visibly."""
        self._assert_creator_process()
        with self._lock:
            if self._closed:
                if self._lease_registry is not None:
                    self._lease_registry.discard_resource(self)
                    self._lease_registry = None
                return
            if not self._connection_closed:
                try:
                    self._connection.close()
                except BaseException:
                    try:
                        _ = self._connection.in_transaction
                    except sqlite3.ProgrammingError:
                        self._connection_closed = True
                    self._closed = self._connection_closed
                    if self._closed and self._lease_registry is not None:
                        self._lease_registry.discard_resource(self)
                        self._lease_registry = None
                    raise
                else:
                    self._connection_closed = True
            self._closed = self._connection_closed
            if self._closed and self._lease_registry is not None:
                self._lease_registry.discard_resource(self)
                self._lease_registry = None

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["MAX_PAGE_SIZE", "SQLITE_SCHEMA_VERSION", "SQLiteLedger"]
