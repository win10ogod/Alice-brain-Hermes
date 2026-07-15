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
from alice_brain_hermes.protocol.models import BridgeCommitAckV1, BridgeGapV1
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.store import SQLiteLedger

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


def legacy_action_ack_database(database: Path, *, mark_sqlite_v3: bool):
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
        current_ack = engine.commit_bridge_record(bridge_instance_id, gap)

        legacy_ack = json.loads(current_ack.canonical_json())
        legacy_ack["frame"]["rd"]["actions"][-1]["execution_confirmed"] = False
        legacy_ack["frame"]["a"]["actions"][-1]["execution_confirmed"] = False
        for section in ("rd", "a"):
            legacy_ack["frame"]["omission_counts"][section]["fields"][
                "omitted_record_json_nodes"
            ]["omitted"] -= 2
        legacy_ack_json = json.dumps(
            legacy_ack,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        assert BridgeCommitAckV1.model_validate_json(legacy_ack_json)

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE bridge_record SET ack_json = ? "
            "WHERE bridge_instance_id = ? AND first_capture_seq = 1",
            (legacy_ack_json, bridge_instance_id),
        )
        if mark_sqlite_v3:
            connection.execute(
                "UPDATE schema_metadata SET value = '3' WHERE key = 'schema_version'"
            )
            connection.execute("PRAGMA user_version = 3")
    return brain_id, bridge_instance_id, gap, current_ack


def test_v3_bridge_ack_is_migrated_after_failure_execution_semantics_change(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-v3-bridge-ack.db"
    brain_id, bridge_instance_id, gap, current_ack = legacy_action_ack_database(
        database,
        mark_sqlite_v3=True,
    )

    with SQLiteLedger.open(database) as restarted:
        assert restarted.schema_version == 4
        assert restarted._connection.execute("PRAGMA user_version").fetchone()[0] == 4
        [row] = restarted._connection.execute(
            "SELECT ack_json FROM bridge_record WHERE bridge_instance_id = ?",
            (bridge_instance_id,),
        ).fetchall()
        migrated_ack = BridgeCommitAckV1.model_validate_json(row["ack_json"])
        duplicate = ConsciousEngine(
            restarted,
            brain_id,
            actor_id=brain_id,
        ).commit_bridge_record(bridge_instance_id, gap)

    assert migrated_ack == current_ack
    assert duplicate == current_ack


def test_current_database_rejects_ack_tampered_to_deterministic_legacy_shape(
    tmp_path: Path,
) -> None:
    database = tmp_path / "current-tampered-ack.db"
    legacy_action_ack_database(database, mark_sqlite_v3=False)

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)
