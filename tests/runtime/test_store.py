from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from alice_brain_hermes.core.events import EventEnvelope, new_event
from alice_brain_hermes.core.reducer import reduce_many
from alice_brain_hermes.core.state import STATE_SCHEMA_VERSION, BrainState
from alice_brain_hermes.errors import (
    EventConflictError,
    LedgerClosedError,
    LedgerIntegrityError,
    SchemaVersionError,
    SnapshotConflictError,
)
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.runtime.store import SQLITE_SCHEMA_VERSION, SQLiteLedger

BRAIN = new_id()
ACTOR = new_id()


def make_event(
    event_type: str = "observation.received",
    payload: dict[str, object] | None = None,
):
    return new_event(event_type, BRAIN, ACTOR, payload or {})


def test_same_event_is_idempotent_but_conflicting_body_is_rejected(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        original = make_event("brain.created", {"name": None})
        first = ledger.append(original)
        second = ledger.append(original)
        sequenced_copy = ledger.append(first)

        assert first.sequence == second.sequence == sequenced_copy.sequence == 1
        with pytest.raises(EventConflictError):
            ledger.append(original.model_copy(update={"payload": {"name": "changed"}}))

        after_conflict = ledger.append(make_event())
        assert after_conflict.sequence == 2


def test_expected_sequence_append_is_atomic_and_preserves_exact_retries(
    tmp_path: Path,
) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as first, SQLiteLedger.open(database) as second:
        original = make_event("clock.tick", {"elapsed_seconds": 1.0})
        stored, inserted = first.append_expected(original, expected_sequence=1)

        assert inserted is True
        assert stored.sequence == 1
        assert first.append_expected(original, expected_sequence=1) == (stored, False)

        intervening = second.append(make_event("opaque.event", {"writer": 2}))
        assert intervening.sequence == 2
        stale = make_event("clock.tick", {"elapsed_seconds": 2.0})
        with pytest.raises(EventConflictError, match="expected sequence"):
            first.append_expected(stale, expected_sequence=2)

        assert [item.event_id for item in first.list_events(BRAIN)] == [
            original.event_id,
            intervening.event_id,
        ]


def test_exact_retry_compares_canonical_body_even_if_fingerprints_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        EventEnvelope,
        "body_fingerprint",
        lambda _event: "0" * 64,
    )
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        original = make_event("clock.tick", {"elapsed_seconds": 1.0})
        ledger.append(original)
        conflicting = original.model_copy(
            update={"payload": {"elapsed_seconds": 2.0}}
        ).revalidated()

        with pytest.raises(EventConflictError, match="different body"):
            ledger.append(conflicting)


def test_sql_failure_rolls_back_sequence_allocation(tmp_path: Path) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        connection = sqlite3.connect(database)
        connection.execute(
            "CREATE TRIGGER reject_fixture BEFORE INSERT ON events "
            "BEGIN SELECT RAISE(ABORT, 'fixture rejection'); END"
        )
        connection.close()

        with pytest.raises(sqlite3.IntegrityError, match="fixture rejection"):
            ledger.append(make_event())

        connection = sqlite3.connect(database)
        connection.execute("DROP TRIGGER reject_fixture")
        connection.close()

        assert ledger.append(make_event()).sequence == 1


def test_commit_failure_rolls_back_and_connection_recovers(tmp_path: Path) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        connection = ledger._connection
        connection.executescript(
            "CREATE TABLE commit_parent(id INTEGER PRIMARY KEY);"
            "CREATE TABLE commit_child("
            "parent_id INTEGER NOT NULL REFERENCES commit_parent(id) "
            "DEFERRABLE INITIALLY DEFERRED);"
            "CREATE TRIGGER reject_commit AFTER INSERT ON events "
            "BEGIN INSERT INTO commit_child(parent_id) VALUES (999); END;"
        )

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            ledger.append(make_event())

        assert connection.in_transaction is False
        connection.execute("DROP TRIGGER reject_commit")
        connection.execute("DROP TABLE commit_child")
        connection.execute("DROP TABLE commit_parent")
        assert ledger.append(make_event()).sequence == 1


def test_idempotency_fingerprint_is_canonical_and_excludes_allocated_sequence(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        original = make_event(payload={"b": 2, "a": [1, 3]})
        first = ledger.append(original)
        reordered = original.model_copy(update={"payload": {"a": [1, 3], "b": 2}})

        assert ledger.append(reordered) == first
        assert ledger.append(original) == first
        assert ledger.append(first) == first

        mismatched_sequence = original.model_copy(update={"sequence": 999})
        with pytest.raises(EventConflictError, match="sequence"):
            ledger.append(mismatched_sequence)


def test_full_envelope_fingerprint_detects_stored_sequence_tampering(
    tmp_path: Path,
) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        stored = ledger.append(make_event())
        assert stored.sequence == 1

    connection = sqlite3.connect(database)
    try:
        envelope_json = connection.execute(
            "SELECT envelope_json FROM events WHERE event_id = ?",
            (stored.event_id,),
        ).fetchone()[0]
        tampered_json = envelope_json.replace('"sequence":1', '"sequence":99')
        assert tampered_json != envelope_json
        connection.execute(
            "UPDATE events SET sequence = 99, envelope_json = ? WHERE event_id = ?",
            (tampered_json, stored.event_id),
        )
        connection.commit()
    finally:
        connection.close()

    with (
        SQLiteLedger.open(database) as ledger,
        pytest.raises(LedgerIntegrityError, match="fingerprint"),
    ):
        ledger.list_events(BRAIN)


@pytest.mark.parametrize(
    "decode_path",
    ["append_retry", "list_events", "replay", "snapshot_validation"],
)
def test_relational_event_id_tampering_is_rejected_by_every_decode_path(
    tmp_path: Path, decode_path: str
) -> None:
    database = tmp_path / f"{decode_path}.db"
    original = make_event("brain.created", {"name": None})
    with SQLiteLedger.open(database) as ledger:
        stored = ledger.append(original)
        state = reduce_many(BrainState.genesis(BRAIN), [stored])

    tampered_event_id = new_id()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE events SET event_id = ? WHERE event_id = ?",
            (tampered_event_id, original.event_id),
        )
        connection.commit()
    finally:
        connection.close()

    with (
        SQLiteLedger.open(database) as ledger,
        pytest.raises(LedgerIntegrityError, match="row keys"),
    ):
        if decode_path == "append_retry":
            ledger.append(original)
        elif decode_path == "list_events":
            ledger.list_events(BRAIN)
        elif decode_path == "replay":
            ledger.replay(BRAIN)
        else:
            ledger.save_snapshot(state)

    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT sequence, event_id FROM events ORDER BY sequence"
        ).fetchall()
        assert rows == [(1, tampered_event_id)]
    finally:
        connection.close()


def test_concurrent_separate_connections_allocate_monotonic_per_brain_sequences(
    tmp_path: Path,
) -> None:
    database = tmp_path / "hermes.db"
    SQLiteLedger.open(database).close()

    def append_one(index: int) -> int:
        with SQLiteLedger.open(database) as ledger:
            stored = ledger.append(make_event(payload={"index": index}))
            assert stored.sequence is not None
            return stored.sequence

    with ThreadPoolExecutor(max_workers=8) as pool:
        sequences = list(pool.map(append_one, range(40)))

    assert sorted(sequences) == list(range(1, 41))
    with SQLiteLedger.open(database) as ledger:
        assert [item.sequence for item in ledger.list_events(BRAIN, limit=100)] == list(
            range(1, 41)
        )


def test_database_uses_wal_foreign_keys_and_explicit_schema_version(
    tmp_path: Path,
) -> None:
    database = tmp_path / "nested" / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        assert ledger.schema_version == SQLITE_SCHEMA_VERSION
        assert ledger.foreign_keys_enabled is True

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == SQLITE_SCHEMA_VERSION
        )
        assert connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()[0] == str(SQLITE_SCHEMA_VERSION)
    finally:
        connection.close()


def test_open_rejects_an_unknown_schema_without_overwriting_it(tmp_path: Path) -> None:
    database = tmp_path / "future.db"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 999")
    connection.close()

    with pytest.raises(SchemaVersionError, match="999"):
        SQLiteLedger.open(database)

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 999
    finally:
        connection.close()


def test_paginated_reads_are_ordered_and_replay_never_truncates(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        ledger.append(make_event("brain.created", {"name": None}))
        for index in range(1_025):
            ledger.append(make_event("opaque.event", {"index": index}))

        seen = []
        cursor = 0
        while page := ledger.list_events(BRAIN, after_sequence=cursor, limit=37):
            seen.extend(page)
            cursor = page[-1].sequence or 0

        assert [item.sequence for item in seen] == list(range(1, 1_027))
        state = ledger.replay(BRAIN)
        assert state.last_sequence == 1_026
        assert state.raw_lifecycle_counts["opaque.event"] == 1_025


@pytest.mark.parametrize(
    ("brain_id", "after_sequence", "limit"),
    [
        ("not-a-uuid", 0, 10),
        (BRAIN, -1, 10),
        (BRAIN, True, 10),
        (BRAIN, 0, 0),
        (BRAIN, 0, 10_001),
    ],
)
def test_list_events_validates_identifiers_cursors_and_limits(
    tmp_path: Path, brain_id: str, after_sequence: int, limit: int
) -> None:
    with (
        SQLiteLedger.open(tmp_path / "hermes.db") as ledger,
        pytest.raises((TypeError, ValueError)),
    ):
        ledger.list_events(brain_id, after_sequence=after_sequence, limit=limit)


def test_snapshot_roundtrip_and_snapshot_plus_tail_equals_full_replay(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        stored = [
            ledger.append(make_event("brain.created", {"name": None})),
            ledger.append(make_event("clock.tick", {"elapsed_seconds": 1.5})),
            ledger.append(make_event("opaque.event", {"stage": "before"})),
        ]
        at_snapshot = reduce_many(BrainState.genesis(BRAIN), stored)
        ledger.save_snapshot(at_snapshot)

        assert ledger.load_snapshot(BRAIN) == at_snapshot

        ledger.append(make_event("clock.tick", {"elapsed_seconds": 2.25}))
        ledger.append(make_event("opaque.event", {"stage": "after"}))

        assert ledger.replay(BRAIN) == ledger.replay(BRAIN, use_snapshot=False)
        assert ledger.replay(BRAIN).logical_clock == 3.75


def test_legacy_v1_rate_free_snapshot_is_a_cache_and_can_be_replaced(
    tmp_path: Path,
) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        ledger.append(
            make_event(
                "personality.revised",
                {"layer": "traits", "values": {"care": 0.05}},
            )
        )
        ledger.append(make_event("clock.tick", {"elapsed_seconds": 0.5}))
        ledger.append(
            make_event(
                "personality.revised",
                {"layer": "traits", "values": {"care": 0.075}},
            )
        )
        full = ledger.replay(BRAIN, use_snapshot=False)
        assert full.personality.rate_state.traits.available == pytest.approx(0.0)

        legacy = full.model_dump(mode="json")
        legacy["schema_version"] = 1
        del legacy["personality"]["rate_state"]
        legacy_json = json.dumps(
            legacy,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        ledger._connection.execute(
            "INSERT INTO snapshots("
            "brain_id, sequence, schema_version, fingerprint, state_json"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                BRAIN,
                full.last_sequence,
                1,
                hashlib.sha256(legacy_json.encode("utf-8")).hexdigest(),
                legacy_json,
            ),
        )

        assert STATE_SCHEMA_VERSION == 2
        assert ledger.load_snapshot(BRAIN) is None
        replayed = ledger.replay(BRAIN)
        assert replayed == full
        assert replayed.personality.rate_state.traits.available == pytest.approx(0.0)

        ledger.save_snapshot(replayed)
        row = ledger._connection.execute(
            "SELECT schema_version, state_json FROM snapshots "
            "WHERE brain_id = ? AND sequence = ?",
            (BRAIN, full.last_sequence),
        ).fetchone()
        assert row["schema_version"] == STATE_SCHEMA_VERSION
        assert '"rate_state"' in row["state_json"]
        assert ledger.load_snapshot(BRAIN) == full


@pytest.mark.parametrize(
    ("stored_value", "substituted_value"),
    [
        (True, 1),
        (1, 1.0),
        (1.0, True),
    ],
)
def test_snapshot_equivalence_is_recursively_type_sensitive(
    tmp_path: Path, stored_value: object, substituted_value: object
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        ledger.append(
            make_event(
                "capabilities.reported",
                {"capabilities": {"nested": {"value": stored_value}}},
            )
        )
        replayed = ledger.replay(BRAIN, use_snapshot=False)
        substituted = replayed.model_copy(
            update={
                "capabilities": {
                    "nested": {"value": substituted_value},
                }
            }
        )

        with pytest.raises(SnapshotConflictError, match="full replay"):
            ledger.save_snapshot(substituted)


def test_snapshot_sequence_is_monotonic_and_cannot_point_past_ledger(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        first = ledger.append(make_event("brain.created", {"name": None}))
        second = ledger.append(make_event("opaque.event"))
        state_one = reduce_many(BrainState.genesis(BRAIN), [first])
        state_two = reduce_many(BrainState.genesis(BRAIN), [first, second])

        ledger.save_snapshot(state_two)
        ledger.save_snapshot(state_two)  # exact same snapshot is idempotent
        with pytest.raises(SnapshotConflictError, match="older"):
            ledger.save_snapshot(state_one)

        future = state_two.model_copy(update={"last_sequence": 3})
        with pytest.raises(SnapshotConflictError, match="past"):
            ledger.save_snapshot(future)


def test_snapshot_schema_mismatch_is_visible(tmp_path: Path) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        stored = ledger.append(make_event("brain.created", {"name": None}))
        state = reduce_many(BrainState.genesis(BRAIN), [stored])
        ledger.save_snapshot(state)

    connection = sqlite3.connect(database)
    connection.execute("UPDATE snapshots SET schema_version = 999")
    connection.commit()
    connection.close()

    with (
        SQLiteLedger.open(database) as ledger,
        pytest.raises(SchemaVersionError, match="999"),
    ):
        ledger.load_snapshot(BRAIN)

    with (
        SQLiteLedger.open(database) as ledger,
        pytest.raises(SchemaVersionError, match="999"),
    ):
        ledger.save_snapshot(state)

    with SQLiteLedger.open(database) as ledger:
        tail = ledger.append(make_event("opaque.event", {"tail": True}))
        later = reduce_many(BrainState.genesis(BRAIN), [stored, tail])
        with pytest.raises(SchemaVersionError, match="999"):
            ledger.replay(BRAIN, use_snapshot=False)
        with pytest.raises(SchemaVersionError, match="999"):
            ledger.save_snapshot(later)


@pytest.mark.parametrize("newer_schema", [1, STATE_SCHEMA_VERSION])
@pytest.mark.parametrize("future_schema", [999, "future"])
def test_older_future_snapshot_schema_cannot_be_hidden_by_newer_cache(
    tmp_path: Path, newer_schema: int, future_schema: object
) -> None:
    database = tmp_path / f"{future_schema}-hidden-by-{newer_schema}.db"
    with SQLiteLedger.open(database) as ledger:
        first = ledger.append(make_event("brain.created", {"name": None}))
        second = ledger.append(make_event("opaque.event", {"tail": True}))
        state = reduce_many(BrainState.genesis(BRAIN), [first, second])
        ledger.save_snapshot(state)
        if newer_schema == 1:
            ledger._connection.execute(
                "UPDATE snapshots SET schema_version = 1 "
                "WHERE brain_id = ? AND sequence = ?",
                (BRAIN, state.last_sequence),
            )
        ledger._connection.execute(
            "INSERT INTO snapshots("
            "brain_id, sequence, schema_version, fingerprint, state_json"
            ") VALUES (?, ?, ?, ?, ?)",
            (BRAIN, 1, future_schema, "future", "{}"),
        )

        with pytest.raises(SchemaVersionError, match=str(future_schema)):
            ledger.load_snapshot(BRAIN)
        with pytest.raises(SchemaVersionError, match=str(future_schema)):
            ledger.replay(BRAIN)
        with pytest.raises(SchemaVersionError, match=str(future_schema)):
            ledger.replay(BRAIN, use_snapshot=False)
        with pytest.raises(SchemaVersionError, match=str(future_schema)):
            ledger.save_snapshot(state)


def test_replay_is_deterministic_and_unknown_events_are_raw_accounted(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        events = [
            make_event("brain.created", {"name": None}),
            make_event("clock.tick", {"elapsed_seconds": 1.0}),
            make_event("unknown.external.lifecycle", {"raw": True}),
        ]
        for item in events:
            ledger.append(item)

        assert ledger.replay(BRAIN) == ledger.replay(BRAIN)
        assert (
            ledger.replay(BRAIN).raw_lifecycle_counts["unknown.external.lifecycle"] == 1
        )


def test_closed_ledger_rejects_operations(tmp_path: Path) -> None:
    ledger = SQLiteLedger.open(tmp_path / "hermes.db")
    ledger.close()
    ledger.close()

    with pytest.raises(LedgerClosedError):
        ledger.append(make_event())
    with pytest.raises(LedgerClosedError):
        ledger.list_events(BRAIN)
    with pytest.raises(LedgerClosedError):
        ledger.replay(BRAIN)
