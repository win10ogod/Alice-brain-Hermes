"""Append-only SQLite WAL event ledger with deterministic replay."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from alice_brain_hermes.core.events import EventEnvelope
from alice_brain_hermes.core.reducer import reduce_state
from alice_brain_hermes.core.state import STATE_SCHEMA_VERSION, BrainState
from alice_brain_hermes.errors import (
    EventConflictError,
    ExpectedSequenceError,
    LedgerClosedError,
    LedgerIntegrityError,
    SchemaVersionError,
    SnapshotConflictError,
)
from alice_brain_hermes.ids import validate_id

SQLITE_SCHEMA_VERSION = 2
MAX_PAGE_SIZE = 10_000


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
"""


def _schema_statements() -> Iterator[str]:
    for statement in _CREATE_SCHEMA.split(";"):
        if stripped := statement.strip():
            yield stripped


class SQLiteLedger:
    """A thread-safe connection facade over a per-brain append-only ledger."""

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self._connection = connection
        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def open(cls, path: str | Path) -> SQLiteLedger:
        """Open or initialize a WAL ledger, rejecting unknown schemas."""
        database = Path(path)
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            database,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        ledger = cls(database, connection)
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            ledger._initialize_schema()
        except BaseException:
            connection.close()
            ledger._closed = True
            raise
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
                return
            if version != SQLITE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"unsupported SQLite schema version {version}; "
                    f"expected {SQLITE_SCHEMA_VERSION}"
                )
            try:
                row = self._connection.execute(
                    "SELECT value FROM schema_metadata WHERE key = ?",
                    ("schema_version",),
                ).fetchone()
            except sqlite3.DatabaseError as error:
                raise SchemaVersionError(
                    "SQLite schema metadata is missing or unreadable"
                ) from error
            if row is None or row["value"] != str(SQLITE_SCHEMA_VERSION):
                actual = None if row is None else row["value"]
                raise SchemaVersionError(
                    f"schema metadata version {actual!r} does not match "
                    f"{SQLITE_SCHEMA_VERSION}"
                )

    @property
    def schema_version(self) -> int:
        self._ensure_open()
        return SQLITE_SCHEMA_VERSION

    @property
    def foreign_keys_enabled(self) -> bool:
        with self._lock:
            self._ensure_open()
            return bool(self._connection.execute("PRAGMA foreign_keys").fetchone()[0])

    def _ensure_open(self) -> None:
        if self._closed:
            raise LedgerClosedError("ledger is closed")

    @contextmanager
    def _transaction(self, *, immediate: bool) -> Iterator[None]:
        with self._lock:
            self._ensure_open()
            begin = "BEGIN IMMEDIATE" if immediate else "BEGIN"
            self._connection.execute(begin)
            try:
                yield
                self._connection.commit()
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise

    @staticmethod
    def _normalize_event(event: EventEnvelope) -> EventEnvelope:
        if not isinstance(event, EventEnvelope):
            raise TypeError("append accepts only EventEnvelope instances")
        return event.revalidated()

    @staticmethod
    def _decode_event(row: sqlite3.Row) -> EventEnvelope:
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
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT event_id, brain_id, sequence, body_fingerprint, "
                "envelope_fingerprint, envelope_json "
                "FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            return None if row is None else self._decode_event(row)

    def append_expected(
        self, event: EventEnvelope, *, expected_sequence: int
    ) -> tuple[EventEnvelope, bool]:
        """Append only at an exact sequence, returning ``(event, inserted)``.

        The comparison and insert share one ``BEGIN IMMEDIATE`` transaction.
        An exact retry remains idempotent even after the ledger has advanced.
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
                if (
                    existing["body_fingerprint"] != body_fingerprint
                    or stored.canonical_json(exclude_sequence=True)
                    != event.canonical_json(exclude_sequence=True)
                ):
                    raise EventConflictError(
                        f"event ID {event.event_id} already has a different body"
                    )
                if event.sequence is not None and event.sequence != stored.sequence:
                    raise EventConflictError(
                        f"event ID {event.event_id} retry sequence "
                        f"{event.sequence} does not match stored sequence "
                        f"{stored.sequence}"
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

    def save_snapshot(self, state: BrainState) -> BrainState:
        """Save only a monotonic snapshot exactly equivalent to full replay."""
        if not isinstance(state, BrainState):
            raise TypeError("save_snapshot accepts only BrainState instances")
        state = state.revalidated()
        with self._transaction(immediate=True):
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
                if latest["schema_version"] == STATE_SCHEMA_VERSION and latest[
                    "fingerprint"
                ] != fingerprint:
                    raise SnapshotConflictError(
                        "snapshot sequence already has a different state"
                    )
                if latest["schema_version"] == STATE_SCHEMA_VERSION:
                    return self._decode_snapshot(latest, state.brain_id)
                if latest["schema_version"] != 1:
                    raise SchemaVersionError(
                        "unsupported snapshot schema "
                        f"{latest['schema_version']}"
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
        fingerprint = SQLiteLedger._snapshot_fingerprint(state.canonical_json())
        if fingerprint != row["fingerprint"]:
            raise LedgerIntegrityError("snapshot fingerprint does not match its state")
        return state

    def _load_snapshot_in_transaction(self, brain_id: str) -> BrainState | None:
        row = self._connection.execute(
            "SELECT sequence, schema_version, fingerprint, state_json "
            "FROM snapshots WHERE brain_id = ? ORDER BY sequence DESC LIMIT 1",
            (brain_id,),
        ).fetchone()
        if row is None or row["schema_version"] == 1:
            return None
        return self._decode_snapshot(row, brain_id)

    def load_snapshot(self, brain_id: str) -> BrainState | None:
        """Load the latest compatible snapshot for a brain."""
        brain_id = validate_id(brain_id)
        with self._lock:
            self._ensure_open()
            return self._load_snapshot_in_transaction(brain_id)

    def replay(self, brain_id: str, *, use_snapshot: bool = True) -> BrainState:
        """Replay a consistent, untruncated ledger view into frozen state."""
        brain_id = validate_id(brain_id)
        if not isinstance(use_snapshot, bool):
            raise TypeError("use_snapshot must be a boolean")
        with self._transaction(immediate=False):
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
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

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
