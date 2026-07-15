from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alice_brain_hermes.core.action import ActionOutcome, ActionPhase
from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict
from alice_brain_hermes.errors import IdempotencyConflictError, SchemaVersionError
from alice_brain_hermes.protocol.models import BridgeCommitAckV2, validate_observation
from alice_brain_hermes.runtime.store import SQLITE_SCHEMA_VERSION
from tests.runtime.test_bridge_store import make_engine
from tests.runtime.test_semantic_ingest import tool_observation


def test_fresh_store_exposes_v5_semantic_schema() -> None:
    assert SQLITE_SCHEMA_VERSION == 5


def test_pre_tool_commits_raw_and_complete_pc_e_st_rd_batch_atomically(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        record = tool_observation(instance, 1, hook="pre_tool_call")

        ack = engine.commit_bridge_record(instance, record)

        assert isinstance(ack, BridgeCommitAckV2)
        assert ack.raw_event_sequence == 1
        assert ack.derived_event_count == 5
        assert len(ack.derived_event_ids) == 5
        assert ack.last_event_sequence == 6
        assert ack.frame.state_sequence == 6
        assert ack.semantic_status == "applied"
        assert ack.semantic_complete is True
        assert engine.state.last_sequence == 6
        assert [event.event_type for event in ledger.list_events(engine.brain_id)] == [
            "hermes.observer.pre_tool_call",
            "action.proposed",
            "personality.control.sampled",
            "action.energy_assessed",
            "simulation.created",
            "action.prepared",
        ]
        [row] = ledger._connection.execute(
            "SELECT semantic_status, semantic_complete, semantic_fingerprint, "
            "derived_event_count, derived_first_sequence, derived_last_sequence "
            "FROM bridge_record"
        ).fetchall()
        assert tuple(row)[:2] == ("applied", 1)
        assert len(row["semantic_fingerprint"]) == 64
        assert tuple(row)[3:] == (5, 2, 6)
        [span] = ledger._connection.execute(
            "SELECT span_kind, external_id, occurrence_capture_seq, action_id, "
            "closed_capture_seq FROM hermes_span"
        ).fetchall()
        assert tuple(span) == (
            "tool",
            "tool-reused",
            1,
            engine.state.action_records[0].action_id,
            None,
        )


def test_matched_post_tool_closes_occurrence_and_commits_execution_outcome(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="pre_tool_call"),
        )

        ack = engine.commit_bridge_record(
            instance,
            tool_observation(instance, 2, hook="post_tool_call", status="error"),
        )

        assert ack.raw_event_sequence == 7
        assert ack.derived_event_count == 2
        assert ack.last_event_sequence == 9
        assert ack.frame.state_sequence == 9
        [action] = engine.state.action_records
        assert action.phase is ActionPhase.RECEIPT
        assert action.execution_confirmed is True
        assert action.outcome is ActionOutcome.FAILURE
        assert action.effect_confirmed is None
        [closed_capture_seq] = ledger._connection.execute(
            "SELECT closed_capture_seq FROM hermes_span"
        ).fetchone()
        assert closed_capture_seq == 2


def test_unmatched_post_tool_commits_raw_plus_semantic_gap(tmp_path: Path) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        ack = engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="post_tool_call"),
        )

        assert ack.semantic_status == "gap"
        assert ack.semantic_complete is False
        assert ack.derived_event_count == 1
        assert ack.last_event_sequence == 2
        assert engine.state.trace_complete is False
        assert [event.event_type for event in ledger.list_events(engine.brain_id)] == [
            "hermes.observer.post_tool_call",
            "semantic.gap",
        ]


def test_lost_ack_retry_returns_exact_ack_without_reapplying_span_or_batch(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    record = tool_observation(instance, 1, hook="pre_tool_call")
    with ledger:
        first = engine.commit_bridge_record(instance, record)
        retried = engine.commit_bridge_record(instance, record)

        assert retried.canonical_json() == first.canonical_json()
        assert retried.duplicate is False
        assert len(ledger.list_events(engine.brain_id)) == 6
        assert ledger._connection.execute(
            "SELECT COUNT(*) FROM hermes_span"
        ).fetchone()[0] == 1


def test_changed_retry_has_zero_event_cursor_span_or_metadata_mutation(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    record = tool_observation(instance, 1, hook="pre_tool_call")
    with ledger:
        engine.commit_bridge_record(instance, record)
        before = ledger.observability_snapshot(engine.brain_id)
        values = record.model_dump(mode="python")
        values["payload"]["args"] = {"command": "changed"}
        changed = validate_observation(values)

        with pytest.raises(IdempotencyConflictError):
            engine.commit_bridge_record(instance, changed)

        assert len(ledger.list_events(engine.brain_id)) == 6
        assert ledger.bridge_stream_state(instance).next_capture_seq == 2
        assert ledger._connection.execute(
            "SELECT COUNT(*) FROM hermes_span"
        ).fetchone()[0] == 1
        assert ledger.observability_snapshot(engine.brain_id) == before


def test_unexpected_sql_failure_rolls_back_raw_derived_span_cursor_and_metadata(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        before = ledger.observability_snapshot(engine.brain_id)
        with ledger._transaction(immediate=True):
            ledger._connection.execute(
                "CREATE TRIGGER reject_derived BEFORE INSERT ON events "
                "WHEN NEW.sequence > 1 BEGIN "
                "SELECT RAISE(ABORT, 'reject derived'); END"
            )

        with pytest.raises(sqlite3.IntegrityError, match="reject derived"):
            engine.commit_bridge_record(
                instance,
                tool_observation(instance, 1, hook="pre_tool_call"),
            )

        assert ledger.list_events(engine.brain_id) == []
        assert ledger.bridge_stream_state(instance).next_capture_seq == 1
        assert ledger._connection.execute(
            "SELECT COUNT(*) FROM hermes_span"
        ).fetchone()[0] == 0
        assert ledger.observability_snapshot(engine.brain_id) == before


def test_observability_snapshot_is_persisted_and_truthful_after_restart(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="post_tool_call"),
        )
        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
        before = ledger.observability_snapshot(engine.brain_id)
        aggregate = ledger.observability_snapshot()
        assert before.semantic_schema_version == 1
        assert before.sqlite_schema_version == 5
        assert before.trace_complete is False
        assert before.semantic_complete is False
        assert before.dropped_events == 0
        assert before.disconnected_open_bridges == 1
        assert aggregate == before.model_copy(update={"brain_id": None})

    from alice_brain_hermes.runtime.store import SQLiteLedger

    with SQLiteLedger.open(database) as reopened:
        assert reopened.observability_snapshot(engine.brain_id) == before


def test_startup_rejects_rehashed_derived_event_outside_canonical_semantic_plan(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="pre_tool_call"),
        )
    with sqlite3.connect(database) as connection:
        [encoded] = connection.execute(
            "SELECT envelope_json FROM events WHERE sequence = 2"
        ).fetchone()
        event = EventEnvelope.model_validate_json(encoded)
        payload = event.payload.model_dump() if hasattr(event.payload, "model_dump") else dict(event.payload)
        intent = dict(payload["intent"])
        intent["args_sha256"] = "c" * 64
        payload["intent"] = intent
        changed = event.model_copy(update={"payload": FrozenJsonDict(payload)}).revalidated()
        connection.execute(
            "UPDATE events SET body_fingerprint = ?, envelope_fingerprint = ?, "
            "envelope_json = ? WHERE sequence = 2",
            (
                changed.body_fingerprint(),
                changed.envelope_fingerprint(),
                changed.canonical_json(),
            ),
        )

    from alice_brain_hermes.runtime.store import SQLiteLedger

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)


def test_startup_rejects_observability_metadata_that_claims_false_green(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="post_tool_call"),
        )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE brain_observability SET trace_complete = 1, semantic_complete = 1"
        )

    from alice_brain_hermes.runtime.store import SQLiteLedger

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)


def test_startup_rejects_span_cache_that_lost_a_committed_open_occurrence(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    with ledger:
        record = tool_observation(instance, 1, hook="pre_tool_call")
        engine.commit_bridge_record(instance, record)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM hermes_span")

    from alice_brain_hermes.runtime.store import SQLiteLedger

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)
