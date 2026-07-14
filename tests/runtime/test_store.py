from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.core.reducer import reduce_many
from alice_brain_hermes.core.state import BrainState
from alice_brain_hermes.errors import (
    EventConflictError,
    LedgerClosedError,
    SchemaVersionError,
    SnapshotConflictError,
)
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.runtime.store import SQLiteLedger

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


def test_idempotency_fingerprint_is_canonical_and_excludes_allocated_sequence(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        original = make_event(payload={"b": 2, "a": [1, 3]})
        first = ledger.append(original)
        reordered = original.model_copy(update={"payload": {"a": [1, 3], "b": 2}})
        replayed = original.model_copy(update={"sequence": 999})

        assert ledger.append(reordered) == first
        assert ledger.append(replayed) == first


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
        assert ledger.schema_version == 1
        assert ledger.foreign_keys_enabled is True

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
            == "1"
        )
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
