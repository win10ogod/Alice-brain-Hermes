from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alice_brain_hermes.core.events import EventEnvelope, new_event
from alice_brain_hermes.errors import (
    EventConflictError,
    IdempotencyConflictError,
    SchemaVersionError,
)
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.identity import IdentityChoiceV1
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.store import SQLITE_SCHEMA_VERSION, SQLiteLedger

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


def _engine(
    ledger: SQLiteLedger,
    *,
    name: str | None = None,
) -> ConsciousEngine:
    brain_id = new_id()
    ledger.create_brain_foundation(brain_id, name=name)
    return ConsciousEngine(ledger, brain_id, actor_id=brain_id)


def _choice(name: str = "Mira") -> IdentityChoiceV1:
    return IdentityChoiceV1(name=name, reason="Chosen for stable continuity")


def test_claim_is_durable_and_records_one_cognition_request(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)

        assert lease is not None
        assert lease.brain_id == engine.brain_id
        assert lease.state_sequence == 2
        assert lease.expires_at > NOW
        assert [event.event_type for event in ledger.list_events(engine.brain_id)] == [
            "brain.created",
            "cognition.requested",
        ]
        status = ledger.identity_naming_status(lease.lease_id)
        assert status.status == "pending"
        assert status.choice is None
        assert engine.state.last_sequence == 2

    with SQLiteLedger.open(database) as restarted:
        restarted_engine = ConsciousEngine(
            restarted,
            lease.brain_id,
            actor_id=lease.brain_id,
        )
        assert restarted_engine.claim_identity_naming(now=NOW) is None
        assert restarted.identity_naming_status(lease.lease_id).status == "pending"


def test_completion_is_atomic_causal_and_exactly_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "identity.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None

        # Continuous C0 may advance after the lease was issued. Completion is
        # guarded by lease/name state, not by a frozen global state sequence.
        engine.pulse(0.25)
        before = engine.state.last_sequence
        assert (
            engine.complete_identity_naming(
                lease.lease_id,
                _choice(),
                now=NOW + timedelta(seconds=1),
            )
            == "completed"
        )
        assert engine.state.last_sequence == before + 3
        assert engine.state.identity.name == "Mira"
        assert [
            event.event_type
            for event in ledger.list_events(
                engine.brain_id,
                after_sequence=before,
                limit=10,
            )
        ] == ["cognition.completed", "c1.deliberated", "identity.named"]

        status = ledger.identity_naming_status(lease.lease_id)
        assert status.status == "completed"
        assert status.choice == _choice()
        assert status.failure_code is None
        exact_head = engine.state.last_sequence
        assert (
            engine.complete_identity_naming(
                lease.lease_id,
                _choice(),
                now=NOW + timedelta(seconds=2),
            )
            == "completed"
        )
        assert engine.state.last_sequence == exact_head
        with pytest.raises(IdempotencyConflictError):
            engine.complete_identity_naming(
                lease.lease_id,
                _choice("Aster"),
                now=NOW + timedelta(seconds=2),
            )
        assert engine.state.last_sequence == exact_head

    with SQLiteLedger.open(database) as restarted:
        state = restarted.replay(lease.brain_id)
        assert state.identity.name == "Mira"
        assert restarted.identity_naming_status(lease.lease_id).status == "completed"


def test_expired_or_preempted_lease_is_superseded_without_late_mutation(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "identity.db") as ledger:
        engine = _engine(ledger)
        first = engine.claim_identity_naming(now=NOW)
        assert first is not None
        head = engine.state.last_sequence

        assert (
            engine.complete_identity_naming(
                first.lease_id,
                _choice(),
                now=first.expires_at + timedelta(microseconds=1),
            )
            == "superseded"
        )
        assert engine.state.last_sequence == head
        assert engine.state.identity.name is None
        assert ledger.identity_naming_status(first.lease_id).status == "superseded"

        second = engine.claim_identity_naming(
            now=first.expires_at + timedelta(seconds=1)
        )
        assert second is not None
        assert second.lease_id != first.lease_id
        assert (
            engine.complete_identity_naming(
                first.lease_id,
                _choice("Late"),
                now=first.expires_at + timedelta(seconds=2),
            )
            == "superseded"
        )
        assert engine.state.identity.name is None


def test_failure_is_causal_sanitized_and_idempotent(tmp_path: Path) -> None:
    with SQLiteLedger.open(tmp_path / "identity.db") as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None
        before = engine.state.last_sequence

        assert (
            engine.fail_identity_naming(
                lease.lease_id,
                "llm_error.TimeoutError",
                now=NOW + timedelta(seconds=1),
            )
            == "failed"
        )
        assert engine.state.last_sequence == before + 1
        failed_event = ledger.list_events(
            engine.brain_id,
            after_sequence=before,
            limit=1,
        )[0]
        assert failed_event.event_type == "cognition.failed"
        assert failed_event.payload["failure_code"] == "llm_error.TimeoutError"
        status = ledger.identity_naming_status(lease.lease_id)
        assert status.status == "failed"
        assert status.failure_code == "llm_error.TimeoutError"

        exact_head = engine.state.last_sequence
        assert (
            engine.fail_identity_naming(
                lease.lease_id,
                "llm_error.TimeoutError",
                now=NOW + timedelta(seconds=2),
            )
            == "failed"
        )
        assert engine.state.last_sequence == exact_head
        with pytest.raises(IdempotencyConflictError):
            engine.fail_identity_naming(
                lease.lease_id,
                "llm_error.ValueError",
                now=NOW + timedelta(seconds=2),
            )


def test_worker_failure_cannot_spoof_framework_terminal_codes(tmp_path: Path) -> None:
    with SQLiteLedger.open(tmp_path / "identity.db") as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None
        head = engine.state.last_sequence

        with pytest.raises(ValueError, match="failure code"):
            engine.fail_identity_naming(
                lease.lease_id,
                "name_conflict",
                now=NOW + timedelta(seconds=1),
            )

        assert engine.state.last_sequence == head
        assert ledger.identity_naming_status(lease.lease_id).status == "pending"


def test_name_conflict_is_visible_and_never_adds_a_suffix(tmp_path: Path) -> None:
    database = tmp_path / "identity.db"
    with SQLiteLedger.open(database) as ledger:
        named = _engine(ledger, name="Mira")
        unnamed = _engine(ledger)
        lease = unnamed.claim_identity_naming(now=NOW)
        assert lease is not None
        before = unnamed.state.last_sequence

        assert (
            unnamed.complete_identity_naming(
                lease.lease_id,
                _choice("Mira"),
                now=NOW + timedelta(seconds=1),
            )
            == "failed"
        )
        assert unnamed.state.identity.name is None
        assert unnamed.state.last_sequence == before + 1
        assert named.state.identity.name == "Mira"
        status = ledger.identity_naming_status(lease.lease_id)
        assert status.status == "failed"
        assert status.failure_code == "name_conflict"
        assert status.choice == _choice("Mira")
        assert ledger.list_events(unnamed.brain_id)[-1].event_type == "cognition.failed"
        assert "Mira-" not in ledger.list_events(unnamed.brain_id)[-1].canonical_json()

        retry = unnamed.claim_identity_naming(now=NOW + timedelta(seconds=2))
        assert retry is not None
        assert retry.lease_id != lease.lease_id

    with SQLiteLedger.open(database) as restarted:
        assert restarted.identity_naming_status(lease.lease_id).status == "failed"
        assert restarted.identity_naming_status(retry.lease_id).status == "pending"


def test_schema_v5_migrates_names_and_rejects_registry_tampering(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger, name="Mira")
        brain_id = engine.brain_id
        assert SQLITE_SCHEMA_VERSION == 6

    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX identity_naming_pending")
        connection.execute("DROP TABLE identity_naming_lease")
        connection.execute("DROP TABLE identity_name_registry")
        connection.execute("UPDATE schema_metadata SET value = '5'")
        connection.execute("PRAGMA user_version = 5")

    with SQLiteLedger.open(database) as migrated:
        assert migrated.schema_version == 6
        row = migrated._connection.execute(
            "SELECT display_name, normalized_name FROM identity_name_registry "
            "WHERE brain_id = ?",
            (brain_id,),
        ).fetchone()
        assert tuple(row) == ("Mira", "mira")

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE identity_name_registry SET display_name = 'Mallory' "
            "WHERE brain_id = ?",
            (brain_id,),
        )
    with pytest.raises(SchemaVersionError, match="identity"):
        SQLiteLedger.open(database)


def test_direct_identity_event_updates_registry_and_survives_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        event = new_event(
            "identity.named",
            engine.brain_id,
            engine.brain_id,
            {"name": "Mira"},
        )
        stored = ledger.append(event)
        assert ledger.append(event) == stored
        row = ledger._connection.execute(
            "SELECT display_name, source_event_id FROM identity_name_registry "
            "WHERE brain_id = ?",
            (engine.brain_id,),
        ).fetchone()
        assert tuple(row) == ("Mira", stored.event_id)

    with SQLiteLedger.open(database) as restarted:
        assert restarted.replay(engine.brain_id).identity.name == "Mira"


def test_restart_rejects_unbound_identity_lease_request_time(tmp_path: Path) -> None:
    database = tmp_path / "identity.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE identity_naming_lease SET requested_at = ? WHERE lease_id = ?",
            ((NOW + timedelta(seconds=1)).isoformat(), lease.lease_id),
        )

    with pytest.raises(SchemaVersionError, match="identity"):
        SQLiteLedger.open(database)


def test_restart_rejects_regenerated_broken_identity_causal_chain(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None
        assert (
            engine.complete_identity_naming(
                lease.lease_id,
                _choice(),
                now=NOW + timedelta(seconds=1),
            )
            == "completed"
        )

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT envelope_json FROM events WHERE brain_id = ? AND sequence = 4",
            (engine.brain_id,),
        ).fetchone()
        event = EventEnvelope.model_validate_json(row[0])
        tampered = event.model_copy(
            update={"payload": {**event.payload, "source_event_id": new_id()}}
        ).revalidated()
        connection.execute(
            "UPDATE events SET body_fingerprint = ?, envelope_fingerprint = ?, "
            "envelope_json = ? WHERE brain_id = ? AND sequence = 4",
            (
                tampered.body_fingerprint(),
                tampered.envelope_fingerprint(),
                tampered.canonical_json(),
                engine.brain_id,
            ),
        )

    with pytest.raises(SchemaVersionError, match="identity"):
        SQLiteLedger.open(database)


@pytest.mark.parametrize("terminal", ["complete", "fail"])
def test_pending_terminal_rejects_time_before_request_without_mutation(
    tmp_path: Path,
    terminal: str,
) -> None:
    database = tmp_path / f"identity-{terminal}.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None
        before_state = engine.state
        before_events = ledger.list_events(engine.brain_id)

        with pytest.raises(ValueError, match="predates"):
            if terminal == "complete":
                engine.complete_identity_naming(
                    lease.lease_id,
                    _choice(),
                    now=NOW - timedelta(microseconds=1),
                )
            else:
                engine.fail_identity_naming(
                    lease.lease_id,
                    "llm_error.TimeoutError",
                    now=NOW - timedelta(microseconds=1),
                )

        assert engine.state == before_state
        assert ledger.list_events(engine.brain_id) == before_events
        assert ledger.identity_naming_status(lease.lease_id).status == "pending"

    with SQLiteLedger.open(database) as restarted:
        assert restarted.replay(lease.brain_id) == before_state
        assert restarted.identity_naming_status(lease.lease_id).status == "pending"


@pytest.mark.parametrize("terminal", ["complete", "fail"])
def test_pre_request_time_cannot_supersede_a_named_pending_lease(
    tmp_path: Path,
    terminal: str,
) -> None:
    database = tmp_path / f"identity-named-{terminal}.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None
        engine.append(
            new_event(
                "identity.named",
                engine.brain_id,
                engine.brain_id,
                {"name": "Externally named"},
            )
        )
        before_state = engine.state
        before_events = ledger.list_events(engine.brain_id)

        with pytest.raises(ValueError, match="predates"):
            if terminal == "complete":
                engine.complete_identity_naming(
                    lease.lease_id,
                    _choice(),
                    now=NOW - timedelta(microseconds=1),
                )
            else:
                engine.fail_identity_naming(
                    lease.lease_id,
                    "llm_error.TimeoutError",
                    now=NOW - timedelta(microseconds=1),
                )

        assert engine.state == before_state
        assert ledger.list_events(engine.brain_id) == before_events
        assert ledger.identity_naming_status(lease.lease_id).status == "pending"

    with SQLiteLedger.open(database) as restarted:
        assert restarted.replay(lease.brain_id) == before_state
        assert restarted.identity_naming_status(lease.lease_id).status == "pending"


@pytest.mark.parametrize("terminal", ["completed", "failed", "superseded"])
def test_terminal_retry_rejects_a_stale_engine(
    tmp_path: Path,
    terminal: str,
) -> None:
    with SQLiteLedger.open(tmp_path / f"identity-stale-{terminal}.db") as ledger:
        writer = _engine(ledger)
        lease = writer.claim_identity_naming(now=NOW)
        assert lease is not None
        stale = ConsciousEngine(
            ledger,
            writer.brain_id,
            actor_id=writer.brain_id,
        )
        if terminal == "completed":
            assert (
                writer.complete_identity_naming(
                    lease.lease_id,
                    _choice(),
                    now=NOW + timedelta(seconds=1),
                )
                == "completed"
            )
            retry = lambda: stale.complete_identity_naming(  # noqa: E731
                lease.lease_id,
                _choice(),
                now=NOW + timedelta(seconds=2),
            )
        elif terminal == "failed":
            assert (
                writer.fail_identity_naming(
                    lease.lease_id,
                    "llm_error.TimeoutError",
                    now=NOW + timedelta(seconds=1),
                )
                == "failed"
            )
            retry = lambda: stale.fail_identity_naming(  # noqa: E731
                lease.lease_id,
                "llm_error.TimeoutError",
                now=NOW + timedelta(seconds=2),
            )
        else:
            assert (
                writer.complete_identity_naming(
                    lease.lease_id,
                    _choice(),
                    now=lease.expires_at,
                )
                == "superseded"
            )
            writer.pulse(0.25)
            retry = lambda: stale.complete_identity_naming(  # noqa: E731
                lease.lease_id,
                _choice(),
                now=lease.expires_at + timedelta(seconds=1),
            )

        stale_state = stale.state
        with pytest.raises(EventConflictError, match="divergence"):
            retry()
        assert stale.state == stale_state
        assert stale.is_stale is True


def test_restart_rejects_identity_events_orphaned_by_deleted_lease(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity-orphan.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None
        assert (
            engine.complete_identity_naming(
                lease.lease_id,
                _choice(),
                now=NOW + timedelta(seconds=1),
            )
            == "completed"
        )

    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM identity_naming_lease WHERE lease_id = ?",
            (lease.lease_id,),
        )

    with pytest.raises(SchemaVersionError, match="identity"):
        SQLiteLedger.open(database)


def test_restart_rejects_stray_reserved_identity_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity-stray.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None
        ledger.append(
            new_event(
                "cognition.failed",
                engine.brain_id,
                engine.brain_id,
                {
                    "schema_version": 1,
                    "purpose": "identity_self_naming",
                    "lease_id": lease.lease_id,
                    "failure_code": "llm_error.StrayError",
                    "terminal_at": (NOW + timedelta(seconds=1)).isoformat(),
                },
                adapter_id="alice-brain-hermes-identity-v1",
            )
        )

    with pytest.raises(SchemaVersionError, match="identity"):
        SQLiteLedger.open(database)


def test_restart_rejects_identity_causal_evidence_crossed_between_leases(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity-crossed-leases.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        first = engine.claim_identity_naming(now=NOW)
        assert first is not None
        assert (
            engine.fail_identity_naming(
                first.lease_id,
                "llm_error.TimeoutError",
                now=NOW + timedelta(seconds=1),
            )
            == "failed"
        )
        second = engine.claim_identity_naming(now=NOW + timedelta(seconds=2))
        assert second is not None
        assert (
            engine.complete_identity_naming(
                second.lease_id,
                _choice(),
                now=NOW + timedelta(seconds=3),
            )
            == "completed"
        )
        crossed_sequence = engine.state.last_sequence - 1

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT envelope_json FROM events WHERE brain_id = ? AND sequence = ?",
            (engine.brain_id, crossed_sequence),
        ).fetchone()
        event = EventEnvelope.model_validate_json(row[0])
        crossed = event.model_copy(
            update={"payload": {**event.payload, "lease_id": first.lease_id}}
        ).revalidated()
        connection.execute(
            "UPDATE events SET body_fingerprint = ?, envelope_fingerprint = ?, "
            "envelope_json = ? WHERE brain_id = ? AND sequence = ?",
            (
                crossed.body_fingerprint(),
                crossed.envelope_fingerprint(),
                crossed.canonical_json(),
                engine.brain_id,
                crossed_sequence,
            ),
        )

    with pytest.raises(SchemaVersionError, match="identity"):
        SQLiteLedger.open(database)


@pytest.mark.parametrize(
    ("event_type", "purpose", "adapter_id"),
    [
        ("cognition.completed", "ordinary_cognition", "alice-brain-hermes-identity-v1"),
        ("cognition.requested", "identity_self_naming", "ordinary-cognition-v1"),
        (
            "identity.unknown_evidence",
            "ordinary_cognition",
            "alice-brain-hermes-identity-v1",
        ),
    ],
)
def test_restart_rejects_half_reserved_or_unknown_identity_evidence(
    tmp_path: Path,
    event_type: str,
    purpose: str,
    adapter_id: str,
) -> None:
    database = tmp_path / f"identity-half-reserved-{event_type}.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        ledger.append(
            new_event(
                event_type,
                engine.brain_id,
                engine.brain_id,
                {"purpose": purpose, "evidence": "unbound"},
                adapter_id=adapter_id,
            )
        )

    with pytest.raises(SchemaVersionError, match="identity"):
        SQLiteLedger.open(database)


def test_restart_ignores_nonreserved_cognition_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity-general-cognition.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        ledger.append(
            new_event(
                "cognition.requested",
                engine.brain_id,
                engine.brain_id,
                {
                    "purpose": "ordinary_cognition",
                    "request": "ordinary adapter evidence",
                },
                adapter_id="ordinary-cognition-v1",
            )
        )
        ledger.append(
            new_event(
                "cognition.completed",
                engine.brain_id,
                engine.brain_id,
                {
                    "purpose": "ordinary_cognition",
                    "result": "ordinary evidence from another adapter",
                },
                adapter_id="another-cognition-v1",
            )
        )

    with SQLiteLedger.open(database) as restarted:
        assert restarted.replay(engine.brain_id).last_sequence == 3
