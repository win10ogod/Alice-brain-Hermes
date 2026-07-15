"""Append-only SQLite WAL event ledger with deterministic replay."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

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
from alice_brain_hermes.protocol.models import (
    HOOK_EVENT_TYPES,
    BrainProfileV1,
    BridgeCommitAckV1,
    BridgeGapV1,
    BridgeRecordV1,
    BridgeStreamState,
    ConsciousnessFrameV2,
    FrameFreshnessV1,
    HermesObservationV1,
    validate_bridge_record_json,
    validate_observation,
    validate_observation_json,
)
from alice_brain_hermes.runtime.lease import RetainedSQLiteFiles, RuntimeLease

SQLITE_SCHEMA_VERSION = 3
MAX_PAGE_SIZE = 10_000
DEFAULT_MAX_FRAME_BYTES = 65_536
DEFAULT_MAX_ACK_BYTES = 4_194_304

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


_CREATE_SCHEMA = """
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

CREATE TABLE brain_profile (
    profile_key TEXT PRIMARY KEY,
    profile_fingerprint TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    brain_id TEXT NOT NULL UNIQUE,
    FOREIGN KEY (brain_id) REFERENCES brains(brain_id)
) WITHOUT ROWID;
"""

_CREATE_BRIDGE_SCHEMA = """
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
CREATE TABLE brain_profile (
    profile_key TEXT PRIMARY KEY,
    profile_fingerprint TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    brain_id TEXT NOT NULL UNIQUE,
    FOREIGN KEY (brain_id) REFERENCES brains(brain_id)
) WITHOUT ROWID;
"""

_CREATE_V2_SCHEMA = _CREATE_SCHEMA.partition("\nCREATE TABLE bridge_stream")[0]


@dataclass(frozen=True, slots=True)
class BridgeCommitResult:
    ack: BridgeCommitAckV1
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
        r"^CREATE\s+(TABLE|INDEX)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b",
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
        retained_files: RetainedSQLiteFiles | None = None,
    ) -> None:
        self.path = path
        self._connection = connection
        self._retained_files = retained_files
        self._lease_registry: RuntimeLease | None = None
        self._creator_pid = os.getpid()
        self._lock = threading.RLock()
        self._closed = False
        self._connection_closed = False
        self._retained_files_closed = retained_files is None
        self._startup_audited_final_states: dict[str, BrainState] = {}
        self._mutation_seal_installed = False
        self._mutation_seal_poisoned = False
        self._mutation_seal_data_version: int | None = None
        self._authorized_transaction_thread: int | None = None

    @classmethod
    def open(cls, path: str | Path) -> SQLiteLedger:
        """Open or initialize a WAL ledger, rejecting unknown schemas."""
        database = Path(path)
        database.parent.mkdir(parents=True, exist_ok=True)
        return cls._open_database(database, logical_path=database)

    @classmethod
    def open_retained(
        cls,
        retained_files: RetainedSQLiteFiles,
        *,
        owner_sink: Callable[[SQLiteLedger], None] | None = None,
    ) -> SQLiteLedger:
        """Open SQLite through one pinned main-file descriptor."""

        if not isinstance(retained_files, RetainedSQLiteFiles):
            raise TypeError("retained_files must be RetainedSQLiteFiles")
        with retained_files.startup_operation():
            try:
                retained_files.verify()
            except BaseException as primary_error:
                traceback = primary_error.__traceback__
                try:
                    retained_files.close()
                except BaseException as cleanup_error:
                    raise primary_error.with_traceback(traceback) from cleanup_error
                raise
            return cls._open_database(
                retained_files.connection_path,
                logical_path=retained_files.logical_path,
                retained_files=retained_files,
                owner_sink=owner_sink,
            )

    def _adopt_retained_lifetime(
        self,
        retained_files: RetainedSQLiteFiles,
        *,
        owner_sink: Callable[[SQLiteLedger], None] | None = None,
    ) -> None:
        """Attach a factory-opened ledger to the same lease ownership graph."""
        self._assert_creator_process()
        if not isinstance(retained_files, RetainedSQLiteFiles):
            raise TypeError("retained_files must be RetainedSQLiteFiles")
        with retained_files.startup_operation(), self._lock:
            if (
                self._closed
                or self._retained_files is not None
                or self._lease_registry is not None
            ):
                raise RuntimeError("SQLite ledger ownership is already established")
            retained_files.verify(allow_missing_transient=True)
            try:
                retained_files.adopt_opening_connection(self._connection)
            except BaseException:
                retained_files.quarantine_unadopted_connection(self._connection)
                raise
            self._retained_files = retained_files
            self._retained_files_closed = False
            database_rows = self._connection.execute("PRAGMA database_list").fetchall()
            main_rows = [row for row in database_rows if row[1] == "main"]
            if len(main_rows) != 1:
                raise PermissionError("SQLite main connection is not unique")
            retained_files.verify_connection_path(main_rows[0][2])
            retained_files.verify(allow_missing_transient=True)
            if owner_sink is not None:
                owner_sink(self)
            self._lease_registry = retained_files._lease
            retained_files._lease._replace_retained_files(retained_files, self)
            retained_files.transfer_opening_connection(self._connection)

    @classmethod
    def _open_database(
        cls,
        database: Path,
        *,
        logical_path: Path,
        retained_files: RetainedSQLiteFiles | None = None,
        owner_sink: Callable[[SQLiteLedger], None] | None = None,
    ) -> SQLiteLedger:
        connection: Any | None = None
        restricted_connection: _RestrictedSQLiteConnection | None = None
        ledger: SQLiteLedger | None = None
        try:
            connection = sqlite3.connect(
                database,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
                cached_statements=0,
            )
            restricted_connection = _RestrictedSQLiteConnection(connection)
            if retained_files is not None:
                try:
                    retained_files.adopt_opening_connection(restricted_connection)
                except BaseException:
                    retained_files.quarantine_unadopted_connection(
                        restricted_connection
                    )
                    raise
            connection.row_factory = sqlite3.Row
            ledger = cls(
                logical_path,
                restricted_connection,
                retained_files=retained_files,
            )
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            startup_data_version = ledger._read_mutation_data_version()
            ledger._initialize_schema()
            connection.set_authorizer(ledger._sqlite_authorizer)
            ledger._install_mutation_seal(expected_data_version=startup_data_version)
            if retained_files is not None:
                database_rows = connection.execute("PRAGMA database_list").fetchall()
                main_rows = [row for row in database_rows if row[1] == "main"]
                if len(main_rows) != 1:
                    raise PermissionError("SQLite main connection is not unique")
                retained_files.verify_connection_path(main_rows[0][2])
                retained_files.verify(allow_missing_transient=True)
            # Publish the fully initialized ledger only at the final handoff.
            # From this point onward no startup path reacquires ledger._lock
            # while the retained lifecycle fence is held.
            if owner_sink is not None:
                owner_sink(ledger)
            if retained_files is not None:
                ledger._lease_registry = retained_files._lease
                retained_files._lease._replace_retained_files(retained_files, ledger)
                retained_files.transfer_opening_connection(restricted_connection)
        except BaseException as primary_error:
            traceback = primary_error.__traceback__
            try:
                if ledger is not None:
                    ledger.close()
                elif retained_files is not None:
                    retained_files.close()
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
                self._validate_schema_contract(2, validate_data=False)
                for statement in _statements(_CREATE_BRIDGE_SCHEMA):
                    self._connection.execute(statement)
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
            _CREATE_V2_SCHEMA if version == 2 else _CREATE_SCHEMA
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
        if version == SQLITE_SCHEMA_VERSION:
            try:
                self._validate_v3_rows_in_transaction(
                    historical_states=historical_states,
                    final_states=final_states,
                )
            except Exception as error:
                raise SchemaVersionError(
                    "SQLite v3 bridge or profile data integrity check failed"
                ) from error
        return final_states

    @property
    def schema_version(self) -> int:
        self._ensure_open()
        return SQLITE_SCHEMA_VERSION

    @property
    def foreign_keys_enabled(self) -> bool:
        self._assert_creator_process()
        with self._lock:
            self._ensure_open()
            return bool(self._connection.execute("PRAGMA foreign_keys").fetchone()[0])

    def _ensure_open(self) -> None:
        self._assert_creator_process()
        with self._lock:
            if self._closed:
                raise LedgerClosedError("ledger is closed")
            if self._retained_files is not None:
                try:
                    self._retained_files.verify(allow_missing_transient=True)
                except Exception as error:
                    self._mutation_seal_poisoned = True
                    raise LedgerIntegrityError(
                        "SQLite mutation seal detected a retained-file identity change"
                    ) from error
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
                    self._connection.commit()
                    if self._connection.in_transaction:
                        self._mutation_seal_poisoned = True
                        cleanup_failed = True
                        raise LedgerIntegrityError(
                            "SQLite mutation seal detected a transaction "
                            "remaining open after commit"
                        )
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
        return brain_id

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
            if (
                brain is None
                or int(brain["next_sequence"]) != 2
                or not exact_foundation
                or not exact_profile
                or has_snapshot is not None
                or has_stream is not None
                or has_bridge_record is not None
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
    ) -> BrainState:
        rows = self._connection.execute(
            "SELECT bridge_instance_id, first_capture_seq, last_capture_seq, "
            "record_kind, record_fingerprint, record_json, event_id, "
            "ledger_sequence, ack_json, accepted_at FROM bridge_record "
            "WHERE bridge_instance_id = ? ORDER BY first_capture_seq",
            (stream.bridge_instance_id,),
        ).fetchall()
        if historical_states is None:
            historical_states, replayed_final = (
                self._replay_target_states_in_transaction(
                    stream.brain_id,
                    {int(row["ledger_sequence"]) for row in rows},
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
            historical_state = historical_states.get(int(row["ledger_sequence"]))
            if historical_state is None:
                raise LedgerIntegrityError(
                    "bridge event sequence is absent from authoritative replay"
                )
            try:
                self._decode_duplicate_bridge_record(
                    row,
                    requested=record,
                    requested_json=record.canonical_json(),
                    requested_fingerprint=record.fingerprint(),
                    historical_state=historical_state,
                )
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
                if event.sequence <= last_record_sequences.get(
                    bridge_instance_id, 0
                ):
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

    def _validate_bridge_tail_in_transaction(
        self, stream: BridgeStreamState
    ) -> BridgeRecordV1 | None:
        rows = self._connection.execute(
            "SELECT record.bridge_instance_id, record.first_capture_seq, "
            "record.last_capture_seq, record.record_kind, "
            "record.record_fingerprint, record.record_json, record.event_id, "
            "record.ledger_sequence, record.ack_json, record.accepted_at, "
            "event.event_id AS linked_event_id, "
            "event.brain_id AS linked_brain_id, "
            "event.sequence AS linked_sequence, "
            "event.body_fingerprint AS linked_body_fingerprint, "
            "event.envelope_fingerprint AS linked_envelope_fingerprint, "
            "event.envelope_json AS linked_envelope_json "
            "FROM bridge_record AS record JOIN events AS event "
            "ON event.event_id = record.event_id "
            "WHERE record.bridge_instance_id = ? "
            "ORDER BY record.first_capture_seq DESC LIMIT 2",
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
                event_row={
                    "event_id": row["linked_event_id"],
                    "brain_id": row["linked_brain_id"],
                    "sequence": row["linked_sequence"],
                    "body_fingerprint": row["linked_body_fingerprint"],
                    "envelope_fingerprint": row["linked_envelope_fingerprint"],
                    "envelope_json": row["linked_envelope_json"],
                },
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

    def _validate_v3_rows_in_transaction(
        self,
        *,
        historical_states: Mapping[str, Mapping[int, BrainState]],
        final_states: Mapping[str, BrainState],
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
            )
        self._validate_abandonment_history_in_transaction(streams)

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
        if version == SQLITE_SCHEMA_VERSION:
            bridge_targets = self._connection.execute(
                "SELECT stream.brain_id, record.ledger_sequence "
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
            if row["schema_version"] in {1, 2}:
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
        now = self._bridge_timestamp()
        with self._transaction(immediate=True):
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
        now = self._bridge_timestamp()
        with self._transaction(immediate=True):
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
        scheduler_sample: str = "not_sampled",
        stream_connected: bool = False,
    ) -> ConsciousnessFrameV2:
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
            receipt_status = (
                action.receipt.get("status") if action.receipt is not None else None
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
            actual_actions.append(
                {
                    "action_id": action.action_id,
                    "dispatch_observed": "dispatched" in history,
                    "blocked": None,
                    "blocked_fact_available": False,
                    "receipt_observed": "receipt" in history,
                    "receipt_status": receipt_status,
                    "execution_confirmed": action.execution_confirmed,
                    "effect_confirmed": action.effect_confirmed,
                    "last_event_id": action.last_event_id,
                }
            )

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
        return ConsciousnessFrameV2(
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

    def project_bridge_frame(
        self,
        bridge_instance_id: str,
        *,
        expected_state: BrainState,
        connected_nonce: str | None = None,
        scheduler_sample: str,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> ConsciousnessFrameV2:
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
            frame = self._project_bridge_frame(
                expected_state,
                through_capture_seq=stream.next_capture_seq - 1,
                capture_coverage=coverage,
                scheduler_sample=scheduler_sample,
                stream_connected=stream.connected_nonce is not None,
            )
            self._enforce_frame_limit(frame, max_frame_bytes=max_frame_bytes)
            return frame

    @staticmethod
    def _enforce_frame_limit(
        frame: ConsciousnessFrameV2, *, max_frame_bytes: int
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
        if isinstance(record, HermesObservationV1):
            return new_event(
                HOOK_EVENT_TYPES[record.hook],
                stream.brain_id,
                stream.server_actor_id,
                record.model_dump(mode="json"),
                wall_time=record.captured_at,
                monotonic_ns=record.captured_monotonic_ns,
                adapter_id=stream.server_adapter_id,
                session_id=getattr(record.context, "session_id", None),
                turn_id=getattr(record.context, "turn_id", None),
                correlation_id=getattr(record.context, "api_request_id", None),
            )
        return new_event(
            "trace.gap",
            stream.brain_id,
            stream.server_actor_id,
            {
                **record.model_dump(mode="json"),
                "exact": True,
                "trace_complete": False,
            },
            adapter_id=stream.server_adapter_id,
        )

    def _decode_duplicate_bridge_record(
        self,
        row: sqlite3.Row,
        *,
        requested: BridgeRecordV1,
        requested_json: str,
        requested_fingerprint: str,
        historical_state: BrainState | None = None,
        event_row: Mapping[str, Any] | None = None,
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
        expected_frame = (
            self._project_bridge_frame(
                historical_state,
                through_capture_seq=persisted.last_capture_seq,
                capture_coverage=self._record_capture_coverage(persisted),
                stream_connected=True,
            )
            if historical_state is not None
            else None
        )
        bounded_frame_relations_match = (
            ack.frame.brain_id == stream.brain_id
            and ack.frame.state_sequence == event.sequence
            and ack.frame.through_capture_seq == persisted.last_capture_seq
            and ack.frame.freshness.projected_at_state_sequence == event.sequence
            and ack.frame.freshness.scheduler_sample == "not_sampled"
            and ack.frame.freshness.stream_connection == "connected"
            and ack.frame.capture_coverage == self._record_capture_coverage(persisted)
        )
        if (
            ack.record_fingerprint != requested_fingerprint
            or ack.duplicate is not False
            or ack.event_id != event.event_id
            or ack.event_sequence != event.sequence
            or ack.through_capture_seq != persisted.last_capture_seq
            or not bounded_frame_relations_match
            or (expected_frame is not None and ack.frame != expected_frame)
        ):
            raise LedgerIntegrityError(
                "persisted bridge acknowledgement does not match replay"
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
        """Atomically commit record, event, successor frame, cursor and ACK."""
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
                "ledger_sequence, ack_json, accepted_at FROM bridge_record "
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

            event = self._bridge_event(stream, record)
            provisional = event.model_copy(
                update={"sequence": expected_sequence}
            ).revalidated()
            successor = reduce_state(expected_state, provisional)
            frame = self._project_bridge_frame(
                successor,
                through_capture_seq=last_capture_seq,
                capture_coverage=self._record_capture_coverage(record),
                stream_connected=True,
            )
            self._enforce_frame_limit(frame, max_frame_bytes=max_frame_bytes)
            ack = BridgeCommitAckV1(
                record_fingerprint=fingerprint,
                event_id=provisional.event_id,
                event_sequence=expected_sequence,
                frame=frame,
                through_capture_seq=last_capture_seq,
            )
            ack_json = ack.canonical_json()
            self._enforce_ack_limit(ack_json, max_ack_bytes=max_ack_bytes)
            updated = self._connection.execute(
                "UPDATE brains SET next_sequence = ? "
                "WHERE brain_id = ? AND next_sequence = ?",
                (expected_sequence + 1, stream.brain_id, expected_sequence),
            )
            if updated.rowcount != 1:
                raise ExpectedSequenceError(
                    "bridge expected sequence was lost before event insert"
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
            self._connection.execute(
                "INSERT INTO bridge_record("
                "bridge_instance_id, first_capture_seq, last_capture_seq, "
                "record_kind, record_fingerprint, record_json, event_id, "
                "ledger_sequence, ack_json, accepted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    bridge_instance_id,
                    first_capture_seq,
                    last_capture_seq,
                    record.record_kind,
                    fingerprint,
                    record_json,
                    provisional.event_id,
                    expected_sequence,
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
        supported = {1, 2, STATE_SCHEMA_VERSION}
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
                1,
                2,
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
                if latest["schema_version"] not in {1, 2}:
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
        if row is None or row["schema_version"] in {1, 2}:
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
                    self._lease_registry._discard_retained_files(self)
                    self._lease_registry = None
                return
            if not self._connection_closed:
                try:
                    self._connection.close()
                except BaseException as primary_error:
                    try:
                        _ = self._connection.in_transaction
                    except sqlite3.ProgrammingError:
                        self._connection_closed = True
                        if self._retained_files is not None:
                            try:
                                self._retained_files.confirm_connection_closed(
                                    self._connection
                                )
                            except BaseException as cleanup_error:
                                raise primary_error from cleanup_error
                    self._closed = (
                        self._connection_closed and self._retained_files_closed
                    )
                    raise
                else:
                    self._connection_closed = True
                    if self._retained_files is not None:
                        self._retained_files.confirm_connection_closed(self._connection)
            if not self._retained_files_closed:
                if self._retained_files is None:
                    raise AssertionError("retained SQLite owner is missing")
                try:
                    self._retained_files.close()
                except BaseException:
                    self._retained_files_closed = self._retained_files.closed
                    self._closed = (
                        self._connection_closed and self._retained_files_closed
                    )
                    if self._closed and self._lease_registry is not None:
                        self._lease_registry._discard_retained_files(self)
                        self._lease_registry = None
                    raise
                else:
                    self._retained_files_closed = True
            self._closed = self._connection_closed and self._retained_files_closed
            if self._closed and self._lease_registry is not None:
                self._lease_registry._discard_retained_files(self)
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
