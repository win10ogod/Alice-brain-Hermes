from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from alice_brain_hermes.core.action import ActionOutcome
from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.errors import SchemaVersionError
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.models import (
    BridgeCommitAckV1,
    BridgeCommitAckV2,
    BridgeGapV1,
    ConsciousnessFrameV2,
)
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.store import (
    _CREATE_BRIDGE_RECORD_V4_SCHEMA,
    SQLiteLedger,
)

RECOVERY_TOKEN = "ab" * 32


def test_v3_action_snapshot_is_replay_only_and_legacy_events_restore_outcome(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-v3-action.db"
    brain_id = new_id()

    def action_event(event_type: str, payload: dict[str, object]):
        return new_event(
            event_type,
            brain_id,
            brain_id,
            payload,
            action_id="action-1",
        )

    with SQLiteLedger.open(database) as ledger:
        for stored_event in (
            action_event(
                "action.proposed",
                {"action_id": "action-1", "intent": {"operation": "test"}},
            ),
            action_event("action.prepared", {"action_id": "action-1"}),
            action_event("action.dispatched", {"action_id": "action-1"}),
            action_event(
                "action.receipt",
                {"action_id": "action-1", "status": "failure"},
            ),
        ):
            ledger.append(stored_event)

        replayed = ledger.replay(brain_id, use_snapshot=False)
        legacy = replayed.model_dump(mode="json")
        legacy["schema_version"] = 3
        legacy_action = legacy["action_records"][0]
        legacy_action.pop("outcome")
        legacy_action["execution_confirmed"] = False
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
                    brain_id,
                    replayed.last_sequence,
                    3,
                    hashlib.sha256(legacy_json.encode("utf-8")).hexdigest(),
                    legacy_json,
                ),
            )

    with SQLiteLedger.open(database) as restarted:
        assert restarted.load_snapshot(brain_id) is None
        from_cache_path = restarted.replay(brain_id)
        from_full_replay = restarted.replay(brain_id, use_snapshot=False)

    assert from_cache_path == from_full_replay
    action = from_cache_path.actions["action-1"]
    assert action.execution_confirmed is True
    assert action.outcome is ActionOutcome.FAILURE


def legacy_action_ack_database(
    database: Path,
    *,
    sqlite_version: int | None,
    nonbridge_gap: str | None = None,
):
    brain_id = new_id()
    bridge_instance_id = new_id()
    gap = BridgeGapV1(
        bridge_instance_id=bridge_instance_id,
        first_capture_seq=1,
        last_capture_seq=1,
        dropped_count=1,
        cause_counts={"queue_full": 1},
    )

    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(brain_id)
        engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)
        ledger.attach_bridge_stream(
            bridge_instance_id,
            brain_id=brain_id,
            server_actor_id=brain_id,
            server_adapter_id="legacy-adapter",
            connected_nonce="legacy-daemon",
            recovery_token=RECOVERY_TOKEN,
        )
        for index in range(1, 6):
            action_id = f"action-{index}"
            status = "failure" if index == 5 else "success"
            for action_event in (
                new_event(
                    "action.proposed",
                    brain_id,
                    brain_id,
                    {"action_id": action_id, "intent": {"operation": "test"}},
                    action_id=action_id,
                ),
                new_event(
                    "action.prepared",
                    brain_id,
                    brain_id,
                    {"action_id": action_id},
                    action_id=action_id,
                ),
                new_event(
                    "action.dispatched",
                    brain_id,
                    brain_id,
                    {"action_id": action_id},
                    action_id=action_id,
                ),
                new_event(
                    "action.receipt",
                    brain_id,
                    brain_id,
                    {"action_id": action_id, "status": status},
                    action_id=action_id,
                ),
            ):
                engine.append(action_event)
        if nonbridge_gap == "before":
            engine.append(
                new_event(
                    "semantic.gap",
                    brain_id,
                    brain_id,
                    {"reason": "legacy-unbounded", "trace_complete": False},
                )
            )
        current_ack = engine.commit_bridge_record(bridge_instance_id, gap)
        if nonbridge_gap == "after":
            engine.append(
                new_event(
                    "semantic.gap",
                    brain_id,
                    brain_id,
                    {"reason": "legacy-unbounded", "trace_complete": False},
                )
            )

        legacy_frame = current_ack.frame.model_dump(mode="python")
        legacy_frame["schema_version"] = 2
        legacy_frame.pop("semantic_schema_version")
        legacy_frame.pop("aggregate_semantic_complete")
        legacy_frame.pop("semantic_evidence")
        if sqlite_version != 4:
            legacy_frame["rd"]["actions"][-1]["execution_confirmed"] = False
            legacy_frame["a"]["actions"][-1]["execution_confirmed"] = False
            for section in ("rd", "a"):
                legacy_frame["omission_counts"][section]["fields"][
                    "omitted_record_json_nodes"
                ]["omitted"] -= 2
        legacy_ack = BridgeCommitAckV1(
            record_fingerprint=current_ack.record_fingerprint,
            event_id=current_ack.raw_event_id,
            event_sequence=current_ack.raw_event_sequence,
            frame=ConsciousnessFrameV2.model_validate(legacy_frame, strict=True),
            through_capture_seq=current_ack.through_capture_seq,
        )
        legacy_ack_json = legacy_ack.canonical_json()

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE bridge_record SET ack_json = ? "
            "WHERE bridge_instance_id = ? AND first_capture_seq = 1",
            (legacy_ack_json, bridge_instance_id),
        )
        if sqlite_version is not None:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("DROP TABLE hermes_span")
            connection.execute("DROP TABLE brain_observability")
            connection.execute("DROP INDEX bridge_record_event")
            connection.execute("ALTER TABLE bridge_record RENAME TO bridge_record_v5")
            connection.executescript(_CREATE_BRIDGE_RECORD_V4_SCHEMA)
            connection.execute(
                "INSERT INTO bridge_record(bridge_instance_id, first_capture_seq, "
                "last_capture_seq, record_kind, record_fingerprint, record_json, "
                "event_id, ledger_sequence, ack_json, accepted_at) SELECT "
                "bridge_instance_id, first_capture_seq, last_capture_seq, "
                "record_kind, record_fingerprint, record_json, event_id, "
                "ledger_sequence, ack_json, accepted_at FROM bridge_record_v5"
            )
            connection.execute("DROP TABLE bridge_record_v5")
            connection.execute(
                "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'",
                (str(sqlite_version),),
            )
            connection.execute(f"PRAGMA user_version = {sqlite_version}")
    return brain_id, bridge_instance_id, gap, current_ack


def _downgrade_empty_database(database: Path, *, sqlite_version: int) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TABLE hermes_span")
        connection.execute("DROP TABLE brain_observability")
        if sqlite_version == 2:
            connection.execute("DROP TABLE bridge_record")
            connection.execute("DROP TABLE bridge_stream")
            connection.execute("DROP TABLE brain_profile")
        else:
            connection.execute("DROP INDEX bridge_record_event")
            connection.execute("ALTER TABLE bridge_record RENAME TO bridge_record_v5")
            connection.executescript(_CREATE_BRIDGE_RECORD_V4_SCHEMA)
            connection.execute("DROP TABLE bridge_record_v5")
        connection.execute(
            "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'",
            (str(sqlite_version),),
        )
        connection.execute(f"PRAGMA user_version = {sqlite_version}")


@pytest.mark.parametrize("event_type", ["trace.gap", "semantic.gap"])
def test_v2_gap_history_migrates_to_truthful_observability(
    tmp_path: Path,
    event_type: str,
) -> None:
    database = tmp_path / f"legacy-v2-{event_type}.db"
    brain_id = new_id()
    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(brain_id)
        ledger.append(
            new_event(
                event_type,
                brain_id,
                brain_id,
                {"reason": "legacy-unbounded", "trace_complete": False},
            )
        )
    _downgrade_empty_database(database, sqlite_version=2)

    with SQLiteLedger.open(database) as migrated:
        snapshot = migrated.observability_snapshot(brain_id)
        assert migrated.replay(brain_id).trace_complete is False
        assert snapshot.trace_complete is False
        assert snapshot.semantic_complete is False
        assert snapshot.semantic_records == 0
        assert snapshot.legacy_raw_only_records == 0
        assert snapshot.semantic_gap_records == 1
        assert snapshot.dropped_events == 0


def test_v4_unbounded_gap_without_bridge_rows_migrates_truthfully(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-v4-unbounded-gap.db"
    brain_id = new_id()
    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(brain_id)
        ledger.append(
            new_event(
                "semantic.gap",
                brain_id,
                brain_id,
                {"reason": "legacy-unbounded", "trace_complete": False},
            )
        )
    _downgrade_empty_database(database, sqlite_version=4)

    with SQLiteLedger.open(database) as migrated:
        snapshot = migrated.observability_snapshot(brain_id)
        assert migrated.replay(brain_id).trace_complete is False
        assert snapshot.trace_complete is False
        assert snapshot.semantic_complete is False
        assert snapshot.semantic_records == 0
        assert snapshot.semantic_gap_records == 1
        assert snapshot.total_bridges == 0


@pytest.mark.parametrize("gap_position", ["before", "after"])
def test_v4_mixed_bridge_and_unbounded_gaps_are_counted_once(
    tmp_path: Path,
    gap_position: str,
) -> None:
    database = tmp_path / f"legacy-v4-mixed-{gap_position}.db"
    brain_id, bridge_instance_id, _gap, _current = legacy_action_ack_database(
        database,
        sqlite_version=4,
        nonbridge_gap=gap_position,
    )

    with SQLiteLedger.open(database) as migrated:
        snapshot = migrated.observability_snapshot(brain_id)
        [row] = migrated._connection.execute(
            "SELECT ack_json FROM bridge_record WHERE bridge_instance_id = ?",
            (bridge_instance_id,),
        ).fetchall()
        ack = BridgeCommitAckV2.model_validate_json(row["ack_json"])

        assert snapshot.trace_complete is False
        assert snapshot.semantic_complete is False
        assert snapshot.semantic_records == 1
        assert snapshot.legacy_raw_only_records == 1
        assert snapshot.semantic_gap_records == 2
        assert snapshot.dropped_events == 1
        assert ack.frame.semantic_evidence.semantic_gap_records == (
            2 if gap_position == "before" else 1
        )
        assert ack.frame.semantic_evidence.dropped_events == 1


def test_v3_bridge_ack_is_migrated_after_failure_execution_semantics_change(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-v3-bridge-ack.db"
    brain_id, bridge_instance_id, gap, current_ack = legacy_action_ack_database(
        database,
        sqlite_version=3,
    )

    with SQLiteLedger.open(database) as restarted:
        assert restarted.schema_version == 5
        assert restarted._connection.execute("PRAGMA user_version").fetchone()[0] == 5
        [row] = restarted._connection.execute(
            "SELECT ack_json FROM bridge_record WHERE bridge_instance_id = ?",
            (bridge_instance_id,),
        ).fetchall()
        migrated_ack = BridgeCommitAckV2.model_validate_json(row["ack_json"])
        duplicate = ConsciousEngine(
            restarted,
            brain_id,
            actor_id=brain_id,
        ).commit_bridge_record(bridge_instance_id, gap)

    assert migrated_ack.semantic_status == "legacy_raw_only"
    assert migrated_ack.semantic_complete is False
    assert migrated_ack.derived_event_count == 0
    assert migrated_ack.raw_event_id == current_ack.raw_event_id
    assert migrated_ack.frame.aggregate_semantic_complete is False
    assert migrated_ack.frame.semantic_evidence.legacy_raw_only_records == 1
    assert migrated_ack.frame.semantic_evidence.semantic_gap_records == 1
    assert duplicate == migrated_ack


def test_v4_bridge_ack_migrates_to_explicit_legacy_raw_only_without_backfill(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-v4-bridge-ack.db"
    brain_id, bridge_instance_id, gap, current_ack = legacy_action_ack_database(
        database,
        sqlite_version=4,
    )

    with SQLiteLedger.open(database) as restarted:
        [row] = restarted._connection.execute(
            "SELECT ack_json FROM bridge_record WHERE bridge_instance_id = ?",
            (bridge_instance_id,),
        ).fetchall()
        migrated = BridgeCommitAckV2.model_validate_json(row["ack_json"])
        duplicate = ConsciousEngine(
            restarted, brain_id, actor_id=brain_id
        ).commit_bridge_record(bridge_instance_id, gap)

    assert migrated.semantic_status == "legacy_raw_only"
    assert migrated.semantic_complete is False
    assert migrated.derived_event_ids == ()
    assert migrated.raw_event_id == current_ack.raw_event_id
    assert migrated.frame.semantic_evidence.legacy_raw_only_records == 1
    assert duplicate == migrated


def test_current_database_rejects_ack_tampered_to_deterministic_legacy_shape(
    tmp_path: Path,
) -> None:
    database = tmp_path / "current-tampered-ack.db"
    legacy_action_ack_database(database, sqlite_version=None)

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)
