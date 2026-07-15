from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
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
from alice_brain_hermes.protocol.models import BrainProfileV1
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.store import (
    _CREATE_BRIDGE_RECORD_V4_SCHEMA,
    SQLITE_SCHEMA_VERSION,
    SQLiteLedger,
)

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


def test_identity_name_normalization_uses_a_stable_unicode_14_boundary() -> None:
    multilingual = "愛アリス한글😀\U0001fae0\U00030000"
    assert _choice(multilingual).name == multilingual

    with pytest.raises(ValueError, match=r"Unicode 14[.]0"):
        _choice("\U0001e030")


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


def test_foundation_names_are_globally_unique_after_nfkc_casefold_atomically(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity-foundation-unique.db"
    first_brain = new_id()
    second_brain = new_id()

    with SQLiteLedger.open(database) as ledger:

        def create(values: tuple[str, str]) -> str:
            brain_id, name = values
            try:
                ledger.create_brain_foundation(brain_id, name=name)
            except sqlite3.IntegrityError:
                return "conflict"
            return "created"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(
                pool.map(
                    create,
                    ((first_brain, "Mira"), (second_brain, "\uff2d\uff29\uff32\uff21")),
                )
            )

        assert sorted(outcomes) == ["conflict", "created"]
        rows = ledger._connection.execute(
            "SELECT brain_id, normalized_name FROM identity_name_registry"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["normalized_name"] == "mira"
        assert ledger.list_brain_ids() == [rows[0]["brain_id"]]
        assert len(ledger.list_events(rows[0]["brain_id"])) == 1
        rejected_brain = (
            second_brain if rows[0]["brain_id"] == first_brain else first_brain
        )
        assert rejected_brain not in ledger.list_brain_ids()

    with SQLiteLedger.open(database) as restarted:
        assert restarted.list_brain_ids() == [rows[0]["brain_id"]]


def test_profile_and_direct_identity_names_share_global_normalized_uniqueness(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity-entrypoint-unique.db"
    with SQLiteLedger.open(database) as ledger:
        first = ledger.resolve_brain_profile(
            BrainProfileV1(profile_key="profile.one", name="Mira")
        )
        before_brains = ledger.list_brain_ids()
        before_profiles = ledger._connection.execute(
            "SELECT profile_key, brain_id FROM brain_profile ORDER BY profile_key"
        ).fetchall()

        with pytest.raises(sqlite3.IntegrityError, match="normalized_name"):
            ledger.resolve_brain_profile(
                BrainProfileV1(profile_key="profile.two", name="mIRA")
            )

        assert ledger.list_brain_ids() == before_brains
        assert (
            ledger._connection.execute(
                "SELECT profile_key, brain_id FROM brain_profile ORDER BY profile_key"
            ).fetchall()
            == before_profiles
        )

        second = _engine(ledger)
        second_before = second.state
        second_events = ledger.list_events(second.brain_id)
        with pytest.raises(sqlite3.IntegrityError, match="normalized_name"):
            second.append(
                new_event(
                    "identity.named",
                    second.brain_id,
                    second.brain_id,
                    {"name": "\uff2d\uff49\uff52\uff41"},
                )
            )

        assert second.state == second_before
        assert ledger.list_events(second.brain_id) == second_events
        assert ledger.replay(first.brain_id).identity.name == "Mira"


def test_post_unicode14_name_is_rejected_without_any_entrypoint_mutation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity-unstable-unicode.db"
    unstable_name = "\U0001e030"
    with SQLiteLedger.open(database) as ledger:
        rejected_brain = new_id()
        with pytest.raises(ValueError, match=r"Unicode 14[.]0"):
            ledger.create_brain_foundation(rejected_brain, name=unstable_name)
        assert rejected_brain not in ledger.list_brain_ids()

        with pytest.raises(ValueError, match=r"Unicode 14[.]0"):
            ledger.resolve_brain_profile(
                BrainProfileV1(profile_key="unstable.profile", name=unstable_name)
            )
        assert ledger.list_brain_ids() == []
        assert (
            ledger._connection.execute("SELECT COUNT(*) FROM brain_profile").fetchone()[
                0
            ]
            == 0
        )

        engine = _engine(ledger)
        before_state = engine.state
        before_events = ledger.list_events(engine.brain_id)
        with pytest.raises(ValueError, match=r"Unicode 14[.]0"):
            engine.append(
                new_event(
                    "identity.named",
                    engine.brain_id,
                    engine.brain_id,
                    {"name": unstable_name},
                )
            )
        assert engine.state == before_state
        assert ledger.list_events(engine.brain_id) == before_events

        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None
        invalid_choice = _choice().model_copy(update={"name": unstable_name})
        before_state = engine.state
        before_events = ledger.list_events(engine.brain_id)
        with pytest.raises(ValueError, match=r"Unicode 14[.]0"):
            engine.complete_identity_naming(
                lease.lease_id,
                invalid_choice,
                now=NOW + timedelta(seconds=1),
            )
        assert engine.state == before_state
        assert ledger.list_events(engine.brain_id) == before_events
        assert ledger.identity_naming_status(lease.lease_id).status == "pending"


def test_legacy_identity_backfill_rejects_post_unicode14_name_explicitly(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity-v5-unstable-unicode.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger, name="Stable")

    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX identity_name_normalized")
        connection.execute("DROP INDEX identity_naming_pending")
        connection.execute("DROP TABLE identity_naming_lease")
        connection.execute("DROP TABLE identity_name_registry")
        row = connection.execute(
            "SELECT envelope_json FROM events WHERE brain_id = ? AND sequence = 1",
            (engine.brain_id,),
        ).fetchone()
        event = EventEnvelope.model_validate_json(row[0])
        unstable = event.model_copy(
            update={"payload": {"name": "\U0001e030"}}
        ).revalidated()
        connection.execute(
            "UPDATE events SET body_fingerprint = ?, envelope_fingerprint = ?, "
            "envelope_json = ? WHERE brain_id = ? AND sequence = 1",
            (
                unstable.body_fingerprint(),
                unstable.envelope_fingerprint(),
                unstable.canonical_json(),
                engine.brain_id,
            ),
        )
        connection.execute("UPDATE schema_metadata SET value = '5'")
        connection.execute("PRAGMA user_version = 5")

    with pytest.raises(SchemaVersionError, match=r"Unicode 14[.]0"):
        SQLiteLedger.open(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'identity_name_registry'"
            ).fetchone()
            is None
        )


@pytest.mark.parametrize("legacy_version", [2, 3, 4, 5])
def test_legacy_identity_backfill_rejects_normalized_name_collision_transactionally(
    tmp_path: Path,
    legacy_version: int,
) -> None:
    database = tmp_path / f"identity-v{legacy_version}-collision.db"
    with SQLiteLedger.open(database) as ledger:
        first = _engine(ledger, name="Mira")
        second = _engine(ledger, name="Aster")

    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX IF EXISTS identity_name_normalized")
        connection.execute("DROP INDEX identity_naming_pending")
        connection.execute("DROP TABLE identity_naming_lease")
        connection.execute("DROP TABLE identity_name_registry")
        if legacy_version == 2:
            connection.execute("DROP TABLE hermes_span")
            connection.execute("DROP TABLE brain_observability")
            connection.execute("DROP TABLE bridge_record")
            connection.execute("DROP TABLE bridge_stream")
            connection.execute("DROP TABLE brain_profile")
        elif legacy_version in {3, 4}:
            connection.execute("DROP TABLE hermes_span")
            connection.execute("DROP TABLE brain_observability")
            connection.execute("DROP TABLE bridge_record")
            connection.executescript(_CREATE_BRIDGE_RECORD_V4_SCHEMA)
        row = connection.execute(
            "SELECT envelope_json FROM events WHERE brain_id = ? AND sequence = 1",
            (second.brain_id,),
        ).fetchone()
        event = EventEnvelope.model_validate_json(row[0])
        collided = event.model_copy(
            update={"payload": {"name": "\uff2d\uff29\uff32\uff21"}}
        ).revalidated()
        connection.execute(
            "UPDATE events SET body_fingerprint = ?, envelope_fingerprint = ?, "
            "envelope_json = ? WHERE brain_id = ? AND sequence = 1",
            (
                collided.body_fingerprint(),
                collided.envelope_fingerprint(),
                collided.canonical_json(),
                second.brain_id,
            ),
        )
        connection.execute(
            "UPDATE schema_metadata SET value = ?",
            (str(legacy_version),),
        )
        connection.execute(f"PRAGMA user_version = {legacy_version}")

    with pytest.raises(SchemaVersionError, match="identity"):
        SQLiteLedger.open(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == legacy_version
        assert connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()[0] == str(legacy_version)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'identity_name_registry'"
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM events WHERE brain_id IN (?, ?)",
                (first.brain_id, second.brain_id),
            ).fetchone()[0]
            == 2
        )


def test_v6_restart_requires_the_normalized_name_unique_index(tmp_path: Path) -> None:
    database = tmp_path / "identity-v6-unique-index.db"
    with SQLiteLedger.open(database):
        pass

    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX IF EXISTS identity_name_normalized")

    with pytest.raises(SchemaVersionError, match="structure"):
        SQLiteLedger.open(database)


def test_restart_rejects_self_naming_request_after_identity_was_already_named(
    tmp_path: Path,
) -> None:
    database = tmp_path / "identity-request-precondition.db"
    requested_at = NOW
    expires_at = requested_at + timedelta(seconds=120)
    lease_id = new_id()
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger, name="Already named")
        request = ledger.append(
            new_event(
                "cognition.requested",
                engine.brain_id,
                engine.brain_id,
                {
                    "schema_version": 1,
                    "purpose": "identity_self_naming",
                    "lease_id": lease_id,
                    "requested_at": requested_at.isoformat(),
                    "expires_at": expires_at.isoformat(),
                },
                adapter_id="alice-brain-hermes-identity-v1",
            )
        )

    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO identity_naming_lease("
            "lease_id, brain_id, request_sequence, status, requested_at, "
            "expires_at, request_event_id, choice_fingerprint, choice_json, "
            "failure_code, terminal_event_id, terminal_at) "
            "VALUES (?, ?, ?, 'pending', ?, ?, ?, NULL, NULL, NULL, NULL, NULL)",
            (
                lease_id,
                engine.brain_id,
                request.sequence,
                requested_at.isoformat(),
                expires_at.isoformat(),
                request.event_id,
            ),
        )

    with pytest.raises(SchemaVersionError, match="identity"):
        SQLiteLedger.open(database)


@pytest.mark.parametrize("reason", ["expired", "identity_already_named"])
def test_restart_rejects_untrue_identity_supersession_reason(
    tmp_path: Path,
    reason: str,
) -> None:
    database = tmp_path / f"identity-untrue-{reason}.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE identity_naming_lease SET status = 'superseded', "
            "failure_code = ?, terminal_at = requested_at WHERE lease_id = ?",
            (reason, lease.lease_id),
        )

    with pytest.raises(SchemaVersionError, match="v6"):
        SQLiteLedger.open(database)


@pytest.mark.parametrize("operation", ["claim", "complete", "fail"])
def test_external_name_supersession_references_exact_name_event_and_restarts(
    tmp_path: Path,
    operation: str,
) -> None:
    database = tmp_path / f"identity-external-name-supersession-{operation}.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None
        named = engine.append(
            new_event(
                "identity.named",
                engine.brain_id,
                engine.brain_id,
                {"name": "Externally named"},
            )
        )

        if operation == "claim":
            assert engine.claim_identity_naming(now=NOW + timedelta(seconds=1)) is None
        elif operation == "complete":
            assert (
                engine.complete_identity_naming(
                    lease.lease_id,
                    _choice(),
                    now=NOW + timedelta(seconds=1),
                )
                == "superseded"
            )
        else:
            assert (
                engine.fail_identity_naming(
                    lease.lease_id,
                    "llm_error.TimeoutError",
                    now=NOW + timedelta(seconds=1),
                )
                == "superseded"
            )
        status = ledger.identity_naming_status(lease.lease_id)
        assert status.status == "superseded"
        assert status.failure_code == "identity_already_named"
        assert status.terminal_event_id == named.event_id

    with SQLiteLedger.open(database) as restarted:
        status = restarted.identity_naming_status(lease.lease_id)
        assert status.terminal_event_id == named.event_id
        assert restarted.replay(engine.brain_id).identity.name == "Externally named"


@pytest.mark.parametrize("spoof", ["request_event", "other_brain_name"])
def test_restart_rejects_spoofed_named_supersession_source(
    tmp_path: Path,
    spoof: str,
) -> None:
    database = tmp_path / f"identity-supersession-source-{spoof}.db"
    with SQLiteLedger.open(database) as ledger:
        engine = _engine(ledger)
        lease = engine.claim_identity_naming(now=NOW)
        assert lease is not None
        if spoof == "request_event":
            source_event_id = ledger.list_events(engine.brain_id)[-1].event_id
        else:
            other = _engine(ledger, name="Another identity")
            source_event_id = ledger.list_events(other.brain_id)[0].event_id

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE identity_naming_lease SET status = 'superseded', "
            "failure_code = 'identity_already_named', terminal_event_id = ?, "
            "terminal_at = ? WHERE lease_id = ?",
            (
                source_event_id,
                (NOW + timedelta(seconds=1)).isoformat(),
                lease.lease_id,
            ),
        )

    with pytest.raises(SchemaVersionError, match="identity"):
        SQLiteLedger.open(database)
