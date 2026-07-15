from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from alice_brain_hermes.core.events import EventEnvelope, new_event
from alice_brain_hermes.core.reducer import reduce_many
from alice_brain_hermes.core.state import STATE_SCHEMA_VERSION, BrainState
from alice_brain_hermes.errors import (
    DomainInvariantError,
    EventConflictError,
    LedgerClosedError,
    LedgerIntegrityError,
    SchemaVersionError,
    SnapshotConflictError,
)
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.models import BrainProfileV1
from alice_brain_hermes.runtime.store import SQLITE_SCHEMA_VERSION, SQLiteLedger

BRAIN = new_id()
ACTOR = new_id()


def make_event(
    event_type: str = "observation.received",
    payload: dict[str, object] | None = None,
):
    return new_event(event_type, BRAIN, ACTOR, payload or {})


def test_snapshot_requires_a_brain_while_bootstrap_preserves_genesis_creation(
    tmp_path: Path,
) -> None:
    brain_id = new_id()
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        with pytest.raises(KeyError):
            ledger.save_snapshot(BrainState.genesis(brain_id))
        assert ledger.bootstrap_state(brain_id) == BrainState.genesis(brain_id)
        assert (
            ledger._connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
            == 0
        )


def test_replay_rejects_a_legacy_event_with_a_second_self_actor(
    tmp_path: Path,
) -> None:
    brain_id = new_id()
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        ledger.ensure_brain(brain_id)
        ledger.append(
            new_event(
                "identity.actor_registered",
                brain_id,
                brain_id,
                {"actor_id": new_id(), "kind": "self"},
            )
        )

        with pytest.raises(DomainInvariantError, match="self actor"):
            ledger.replay(brain_id, use_snapshot=False)


def test_dynamic_compensation_validates_profile_payload_and_rolls_back_delete(
    tmp_path: Path,
) -> None:
    brain_id = new_id()
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        foundation = ledger.create_brain_foundation(brain_id, name="Alice")
        mismatched = BrainProfileV1(profile_key="compensation.profile", name="Mallory")
        with pytest.raises(ValueError, match="payload"):
            ledger.compensate_unpublished_brain_foundation(
                brain_id, foundation=foundation, profile=mismatched
            )

        with ledger._transaction(immediate=True):
            ledger._connection.execute(
                "CREATE TRIGGER reject_foundation_delete BEFORE DELETE ON events "
                "BEGIN SELECT RAISE(ABORT, 'reject compensation delete'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="reject compensation"):
            ledger.compensate_unpublished_brain_foundation(
                brain_id, foundation=foundation, profile=None
            )
        assert ledger.list_brain_ids() == [brain_id]
        assert ledger.list_events(brain_id) == [foundation]
        with ledger._transaction(immediate=True):
            ledger._connection.execute("DROP TRIGGER reject_foundation_delete")


def test_startup_audited_foundation_is_never_compensated(
    tmp_path: Path,
) -> None:
    database = tmp_path / "hermes.db"
    brain_id = new_id()
    with SQLiteLedger.open(database) as ledger:
        foundation = ledger.create_brain_foundation(brain_id, name=None)

    with SQLiteLedger.open(database) as restarted:
        assert (
            restarted.compensate_unpublished_brain_foundation(
                brain_id, foundation=foundation, profile=None
            )
            is False
        )
        assert restarted.list_brain_ids() == [brain_id]


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
    with SQLiteLedger.open(database) as first:
        original = make_event("clock.tick", {"elapsed_seconds": 1.0})
        stored, inserted = first.append_expected(original, expected_sequence=1)

        assert inserted is True
        assert stored.sequence == 1
        assert first.append_expected(original, expected_sequence=1) == (stored, False)

        intervening = first.append(make_event("opaque.event", {"writer": 2}))
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
        with ledger._transaction(immediate=True):
            ledger._connection.execute(
                "CREATE TRIGGER reject_fixture BEFORE INSERT ON events "
                "BEGIN SELECT RAISE(ABORT, 'fixture rejection'); END"
            )

        with pytest.raises(sqlite3.IntegrityError, match="fixture rejection"):
            ledger.append(make_event())

        with ledger._transaction(immediate=True):
            ledger._connection.execute("DROP TRIGGER reject_fixture")

        assert ledger.append(make_event()).sequence == 1


def test_commit_failure_rolls_back_and_connection_recovers(tmp_path: Path) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        connection = ledger._connection
        with ledger._transaction(immediate=True):
            connection.execute("CREATE TABLE commit_parent(id INTEGER PRIMARY KEY)")
            connection.execute(
                "CREATE TABLE commit_child("
                "parent_id INTEGER NOT NULL REFERENCES commit_parent(id) "
                "DEFERRABLE INITIALLY DEFERRED)"
            )
            connection.execute(
                "CREATE TRIGGER reject_commit AFTER INSERT ON events "
                "BEGIN INSERT INTO commit_child(parent_id) VALUES (999); END"
            )

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            ledger.append(make_event())

        assert connection.in_transaction is False
        with ledger._transaction(immediate=True):
            connection.execute("DROP TRIGGER reject_commit")
            connection.execute("DROP TABLE commit_child")
            connection.execute("DROP TABLE commit_parent")
        assert ledger.append(make_event()).sequence == 1


def test_ledger_connection_is_not_a_sqlite_backup_deserialize_or_extension_target(
    tmp_path: Path,
) -> None:
    database = tmp_path / "hermes.db"
    source_database = tmp_path / "replacement.db"
    with sqlite3.connect(source_database) as source:
        source.execute("CREATE TABLE replacement_marker(value TEXT NOT NULL)")
        source.execute("INSERT INTO replacement_marker(value) VALUES ('forbidden')")

    with SQLiteLedger.open(database) as ledger:
        connection = ledger._connection

        assert not isinstance(connection, sqlite3.Connection)
        for forbidden in (
            "backup",
            "deserialize",
            "enable_load_extension",
            "load_extension",
            "serialize",
            "set_authorizer",
        ):
            assert not hasattr(connection, forbidden)

        with (
            sqlite3.connect(source_database) as source,
            pytest.raises(TypeError),
        ):
            source.backup(connection)

        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'table' AND name = 'replacement_marker'"
            ).fetchone()[0]
            == 0
        )
        assert ledger.list_brain_ids() == []

        retained = ledger._retained_files
        assert retained is None or retained._opening_connection is None


class _RollbackFaultConnection:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        rollback_failure: BaseException | None,
    ) -> None:
        self._connection = connection
        self._rollback_failure = rollback_failure

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    def execute(self, statement: str, parameters=()):
        return self._connection.execute(statement, parameters)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        if self._rollback_failure is not None:
            raise self._rollback_failure

    def close(self) -> None:
        self._connection.close()


@pytest.mark.parametrize(
    "rollback_failure",
    [OSError("injected rollback failure"), None],
    ids=["rollback-raises", "transaction-remains-open"],
)
def test_failed_transaction_cleanup_preserves_primary_and_poison_seal(
    tmp_path: Path,
    rollback_failure: BaseException | None,
) -> None:
    database = tmp_path / "hermes.db"
    forbidden_brain_id = new_id()
    ledger = SQLiteLedger.open(database)
    raw_connection = ledger._connection
    ledger._connection = _RollbackFaultConnection(
        raw_connection,
        rollback_failure=rollback_failure,
    )
    try:
        with (
            pytest.raises(RuntimeError, match="injected transaction primary") as caught,
            ledger._transaction(immediate=True),
        ):
            ledger._connection.execute(
                "INSERT INTO brains(brain_id, next_sequence) VALUES (?, 1)",
                (forbidden_brain_id,),
            )
            raise RuntimeError("injected transaction primary")

        if rollback_failure is not None:
            assert caught.value.__cause__ is rollback_failure
        else:
            assert isinstance(caught.value.__cause__, LedgerIntegrityError)
            assert "transaction" in str(caught.value.__cause__)

        with pytest.raises(LedgerIntegrityError, match="mutation seal"):
            ledger.list_brain_ids()
    finally:
        ledger.close()

    with SQLiteLedger.open(database) as reopened:
        assert forbidden_brain_id not in reopened.list_brain_ids()


def test_post_rollback_seal_refresh_preserves_primary_and_chains_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = SQLiteLedger.open(tmp_path / "hermes.db")
    real_read = ledger._read_mutation_data_version
    read_calls = 0

    def fail_final_refresh() -> int:
        nonlocal read_calls
        read_calls += 1
        if read_calls == 3:
            raise OSError("injected final seal refresh failure")
        return real_read()

    monkeypatch.setattr(ledger, "_read_mutation_data_version", fail_final_refresh)
    try:
        with (
            pytest.raises(RuntimeError, match="injected body primary") as caught,
            ledger._transaction(immediate=True),
        ):
            raise RuntimeError("injected body primary")

        assert isinstance(caught.value.__cause__, LedgerIntegrityError)
        assert isinstance(caught.value.__cause__.__cause__, OSError)
        assert "final seal refresh failure" in str(caught.value.__cause__.__cause__)
        with pytest.raises(LedgerIntegrityError, match="mutation seal"):
            ledger.list_brain_ids()
    finally:
        ledger.close()


def test_external_event_commit_permanently_poison_mutation_seal(
    tmp_path: Path,
) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        stored = ledger.append(make_event("opaque.event"))
        with sqlite3.connect(database) as external:
            external.execute(
                "UPDATE events SET envelope_json = '{}' WHERE event_id = ?",
                (stored.event_id,),
            )

        for operation in (
            lambda: ledger.get_event(stored.event_id),
            lambda: ledger.list_events(BRAIN),
        ):
            with pytest.raises(LedgerIntegrityError, match="mutation seal"):
                operation()


def test_same_connection_rolled_back_dml_permanently_poison_mutation_seal(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        ledger.append(make_event("opaque.event"))
        ledger._connection.execute("BEGIN IMMEDIATE")
        ledger._connection.execute(
            "UPDATE brains SET next_sequence = next_sequence + 1 WHERE brain_id = ?",
            (BRAIN,),
        )
        ledger._connection.rollback()

        for operation in (
            lambda: ledger.list_events(BRAIN),
            lambda: ledger.append(make_event("opaque.event")),
        ):
            with pytest.raises(LedgerIntegrityError, match="mutation seal"):
                operation()


def test_cached_same_connection_dml_cannot_bypass_mutation_seal(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        ledger.ensure_brain(BRAIN)
        statement = "UPDATE brains SET next_sequence = next_sequence WHERE brain_id = ?"
        with ledger._transaction(immediate=True):
            ledger._connection.execute(statement, (BRAIN,))

        ledger._connection.execute(statement, (BRAIN,))

        with pytest.raises(LedgerIntegrityError, match="mutation seal"):
            ledger.list_events(BRAIN)


def test_external_commit_during_owned_transaction_poison_mutation_seal(
    tmp_path: Path,
) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(BRAIN)
        with (
            pytest.raises(LedgerIntegrityError, match="mutation seal"),
            ledger._transaction(immediate=False),
        ):
            ledger._connection.execute(
                "SELECT next_sequence FROM brains WHERE brain_id = ?", (BRAIN,)
            ).fetchone()
            with sqlite3.connect(database) as external:
                external.execute(
                    "INSERT INTO brains(brain_id, next_sequence) VALUES (?, 1)",
                    (new_id(),),
                )

        with pytest.raises(LedgerIntegrityError, match="mutation seal"):
            ledger.list_brain_ids()


def test_external_commit_between_entry_check_and_begin_cannot_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "hermes.db"
    ledger = SQLiteLedger.open(database)
    real_read = ledger._read_mutation_data_version
    injected = False

    def inject_after_entry_probe() -> int:
        nonlocal injected
        version = real_read()
        if not injected:
            injected = True
            with sqlite3.connect(database) as external:
                external.execute(
                    "INSERT INTO brains(brain_id, next_sequence) VALUES (?, 1)",
                    (new_id(),),
                )
        return version

    monkeypatch.setattr(ledger, "_read_mutation_data_version", inject_after_entry_probe)
    with pytest.raises(LedgerIntegrityError, match="mutation seal"):
        ledger.append(make_event())
    ledger.close()

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_external_commit_between_startup_audit_and_seal_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(BRAIN)

    real_install = SQLiteLedger._install_mutation_seal

    def race_before_seal(ledger: SQLiteLedger, *, expected_data_version: int) -> None:
        with sqlite3.connect(database) as external:
            external.execute(
                "INSERT INTO brains(brain_id, next_sequence) VALUES (?, 1)",
                (new_id(),),
            )
        real_install(ledger, expected_data_version=expected_data_version)

    monkeypatch.setattr(SQLiteLedger, "_install_mutation_seal", race_before_seal)

    with pytest.raises(LedgerIntegrityError, match="mutation seal"):
        SQLiteLedger.open(database)


def test_public_seal_check_waits_for_owned_transaction_lock(tmp_path: Path) -> None:
    ledger = SQLiteLedger.open(tmp_path / "hermes.db")
    transaction_started = threading.Event()
    release_transaction = threading.Event()
    schema_returned = threading.Event()
    thread_errors: list[BaseException] = []

    def hold_transaction() -> None:
        try:
            with ledger._transaction(immediate=True):
                transaction_started.set()
                assert release_transaction.wait(timeout=5.0)
        except BaseException as error:
            thread_errors.append(error)

    def read_schema() -> None:
        try:
            assert ledger.schema_version == SQLITE_SCHEMA_VERSION
            schema_returned.set()
        except BaseException as error:
            thread_errors.append(error)

    holder = threading.Thread(target=hold_transaction)
    reader = threading.Thread(target=read_schema)
    holder.start()
    assert transaction_started.wait(timeout=5.0)
    reader.start()
    returned_while_locked = schema_returned.wait(timeout=0.1)
    release_transaction.set()
    holder.join(timeout=5.0)
    reader.join(timeout=5.0)
    ledger.close()

    assert returned_while_locked is False
    assert not holder.is_alive()
    assert not reader.is_alive()
    assert thread_errors == []


def test_external_ddl_permanently_poison_mutation_seal(tmp_path: Path) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:
        with sqlite3.connect(database) as external:
            external.execute("CREATE TABLE unauthorized_fixture(value INTEGER)")

        for operation in (
            lambda: ledger.list_brain_ids(),
            lambda: ledger.ensure_brain(new_id()),
        ):
            with pytest.raises(LedgerIntegrityError, match="mutation seal"):
                operation()


def test_authorized_commit_and_rollback_refresh_mutation_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteLedger.open(tmp_path / "hermes.db") as ledger:
        brain_id = new_id()
        ledger.ensure_brain(brain_id)
        assert ledger.list_brain_ids() == [brain_id]

        real_bridge_stream_row = ledger._bridge_stream_row
        instance = new_id()
        ledger.attach_bridge_stream(
            instance,
            brain_id=brain_id,
            server_actor_id=brain_id,
            server_adapter_id="fixture-adapter",
            connected_nonce="fixture-connection",
            recovery_token="ab" * 32,
        )
        calls = 0

        def fail_after_authorized_update(bridge_instance_id: str):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("fixture failure after update")
            return real_bridge_stream_row(bridge_instance_id)

        monkeypatch.setattr(ledger, "_bridge_stream_row", fail_after_authorized_update)
        with pytest.raises(RuntimeError, match="fixture failure"):
            ledger.disconnect_bridge_stream(
                instance, connected_nonce="fixture-connection"
            )
        monkeypatch.setattr(ledger, "_bridge_stream_row", real_bridge_stream_row)

        assert ledger.bridge_stream_state(instance).connected_nonce == (
            "fixture-connection"
        )
        assert (
            ledger.append(new_event("opaque.event", brain_id, brain_id, {})).sequence
            == 1
        )


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
        connection.execute(
            "UPDATE brains SET next_sequence = 100 WHERE brain_id = ?",
            (BRAIN,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SchemaVersionError, match="replay or snapshot") as captured:
        SQLiteLedger.open(database)
    assert isinstance(captured.value.__cause__, LedgerIntegrityError)
    assert "fingerprint" in str(captured.value.__cause__)


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
        ledger.append(original)

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

    with pytest.raises(SchemaVersionError, match="replay or snapshot") as captured:
        SQLiteLedger.open(database)
    assert isinstance(captured.value.__cause__, LedgerIntegrityError)
    assert "row keys" in str(captured.value.__cause__)

    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT sequence, event_id FROM events ORDER BY sequence"
        ).fetchall()
        assert rows == [(1, tampered_event_id)]
    finally:
        connection.close()


def test_concurrent_calls_on_owned_connection_allocate_monotonic_sequences(
    tmp_path: Path,
) -> None:
    database = tmp_path / "hermes.db"
    with SQLiteLedger.open(database) as ledger:

        def append_one(index: int) -> int:
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


def test_existing_v4_reopen_rejects_tampered_sequence_allocator(
    tmp_path: Path,
) -> None:
    database = tmp_path / "allocator.db"
    brain_id = new_id()
    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(brain_id)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE brains SET next_sequence = 2 WHERE brain_id = ?",
            (brain_id,),
        )

    with pytest.raises(SchemaVersionError, match="sequence allocation"):
        SQLiteLedger.open(database)


@pytest.mark.parametrize(
    "extra_sql",
    [
        "CREATE TABLE unexpected_object(value INTEGER)",
        "CREATE INDEX unexpected_object ON brains(next_sequence)",
        "CREATE TRIGGER unexpected_object AFTER UPDATE ON brains BEGIN SELECT 1; END",
    ],
)
def test_existing_v4_reopen_rejects_every_extra_schema_object(
    tmp_path: Path, extra_sql: str
) -> None:
    database = tmp_path / "extra-schema.db"
    with SQLiteLedger.open(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute(extra_sql)

    with pytest.raises(SchemaVersionError, match="exact contract"):
        SQLiteLedger.open(database)


def test_existing_v4_reopen_replays_before_snapshot_and_rejects_false_cache(
    tmp_path: Path,
) -> None:
    database = tmp_path / "false-cache.db"
    with SQLiteLedger.open(database) as ledger:
        ledger.append(make_event("brain.created", {"name": None}))
        clock = ledger.append(make_event("clock.tick", {"elapsed_seconds": 1.0}))
        ledger.save_snapshot(ledger.replay(BRAIN, use_snapshot=False))

    changed = clock.model_copy(
        update={"payload": {"elapsed_seconds": 2.0}}
    ).revalidated()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE events SET body_fingerprint = ?, envelope_fingerprint = ?, "
            "envelope_json = ? WHERE event_id = ?",
            (
                changed.body_fingerprint(),
                changed.envelope_fingerprint(),
                changed.canonical_json(),
                changed.event_id,
            ),
        )

    with pytest.raises(SchemaVersionError, match="replay or snapshot"):
        SQLiteLedger.open(database)


def test_existing_v4_reopen_rejects_canonical_event_sequence_hole(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sequence-hole.db"
    with SQLiteLedger.open(database) as ledger:
        first = ledger.append(make_event("brain.created", {"name": None}))
        ledger.append(make_event("opaque.event", {"tail": True}))
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM events WHERE event_id = ?", (first.event_id,))

    with pytest.raises(SchemaVersionError, match="replay or snapshot"):
        SQLiteLedger.open(database)
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
        with ledger._transaction(immediate=True):
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

        assert STATE_SCHEMA_VERSION == 4
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


def test_v2_snapshot_is_fingerprint_checked_replay_only_and_replaced_by_v4(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-v2.db"
    with SQLiteLedger.open(database) as ledger:
        stored = ledger.append(make_event("future.lifecycle.event", {}))
        full = reduce_many(BrainState.genesis(BRAIN), [stored])
        legacy = full.model_dump(mode="json")
        legacy["schema_version"] = 2
        legacy.pop("working_set")
        legacy_json = json.dumps(
            legacy,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(legacy_json.encode("utf-8")).hexdigest()
        with ledger._transaction(immediate=True):
            ledger._connection.execute(
                "INSERT INTO snapshots("
                "brain_id, sequence, schema_version, fingerprint, state_json"
                ") VALUES (?, ?, 2, ?, ?)",
                (BRAIN, full.last_sequence, fingerprint, legacy_json),
            )

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE snapshots SET fingerprint = ? WHERE brain_id = ?",
            ("0" * 64, BRAIN),
        )
    with pytest.raises(SchemaVersionError, match="replay or snapshot"):
        SQLiteLedger.open(database)

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE snapshots SET fingerprint = ? WHERE brain_id = ?",
            (fingerprint, BRAIN),
        )
    with SQLiteLedger.open(database) as ledger:
        assert ledger.load_snapshot(BRAIN) is None
        assert ledger.replay(BRAIN) == full
        ledger.save_snapshot(full)
        row = ledger._connection.execute(
            "SELECT schema_version, state_json FROM snapshots "
            "WHERE brain_id = ? AND sequence = ?",
            (BRAIN, full.last_sequence),
        ).fetchone()
        assert row["schema_version"] == 4
        assert '"working_set"' in row["state_json"]


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

        future = reduce_many(
            state_two,
            [new_event("opaque.future", BRAIN, ACTOR, {}, sequence=3)],
        )
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

    with pytest.raises(SchemaVersionError, match="replay or snapshot") as captured:
        SQLiteLedger.open(database)
    assert isinstance(captured.value.__cause__, SchemaVersionError)
    assert "999" in str(captured.value.__cause__)


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
        with ledger._transaction(immediate=True):
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
