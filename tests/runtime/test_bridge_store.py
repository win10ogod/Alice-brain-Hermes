from __future__ import annotations

import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alice_brain_hermes import errors
from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.errors import (
    BridgeBindingError,
    BridgeClosedError,
    CaptureGapRequiredError,
    CaptureSequenceError,
    IdempotencyConflictError,
    LedgerIntegrityError,
    SchemaVersionError,
)
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.models import (
    BrainProfileV1,
    BridgeGapV1,
    BridgeStreamState,
    CoverageV1,
    HermesObservationV1,
    validate_observation,
)
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.scheduler import ContinuousScheduler
from alice_brain_hermes.runtime.snapshot import SnapshotWorker
from alice_brain_hermes.runtime.store import SQLiteLedger
from tests.protocol.test_models import HOOK_CASES

RECOVERY_TOKEN = "ab" * 32


def observation(instance: str, capture_seq: int) -> HermesObservationV1:
    return validate_observation(
        {
            "bridge_instance_id": instance,
            "capture_seq": capture_seq,
            "captured_at": datetime.now(UTC),
            "captured_monotonic_ns": capture_seq,
            "hook": "post_api_request",
            "context": {
                "session_id": "session-1",
                "task_id": "task-1",
                "turn_id": "turn-1",
                "api_request_id": f"request-{capture_seq}",
            },
            "payload": {
                "platform": "test",
                "model": "model-y",
                "provider": "provider-x",
                "base_url": None,
                "api_mode": "streaming",
                "api_call_count": 1,
                "api_duration": 0.25,
                "started_at": "start",
                "ended_at": "end",
                "finish_reason": "stop",
                "message_count": 1,
                "response_model": "model-y",
                "response": {},
                "usage": {},
                "assistant_message": {},
                "assistant_content_chars": 0,
                "assistant_tool_call_count": 1,
                "extensions": {
                    "stream": True,
                    "reasoning": {"effort": "high"},
                    "multimodal": [{"type": "image", "id": "image-1"}],
                    "tool_calls": [{"name": "lookup", "arguments": {"q": "full"}}],
                },
            },
            "coverage": CoverageV1(
                policy_version="copy-v1",
                capture_coverage="host_sanitized",
            ),
        }
    )


def make_engine(tmp_path: Path):
    ledger = SQLiteLedger.open(tmp_path / "runtime.db")
    brain_id = new_id()
    actor_id = brain_id
    instance = new_id()
    ledger.ensure_brain(brain_id)
    engine = ConsciousEngine(ledger, brain_id, actor_id=actor_id)
    ledger.attach_bridge_stream(
        instance,
        brain_id=brain_id,
        server_actor_id=actor_id,
        server_adapter_id="alice-brain-hermes-observer-v1",
        connected_nonce="daemon-nonce",
        recovery_token=RECOVERY_TOKEN,
    )
    return ledger, engine, instance


def test_reconnected_stream_rejects_stale_connection_commit_atomically(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
        ledger.attach_bridge_stream(
            instance,
            brain_id=engine.brain_id,
            server_actor_id=engine.actor_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            connected_nonce="new-daemon-nonce",
            recovery_token=RECOVERY_TOKEN,
        )

        with pytest.raises(BridgeBindingError, match="another connection"):
            engine.commit_bridge_record(
                instance,
                observation(instance, 1),
                connected_nonce="daemon-nonce",
            )

        assert ledger.list_events(engine.brain_id) == []
        assert ledger.bridge_stream_state(instance).next_capture_seq == 1
        accepted = engine.commit_bridge_record(
            instance,
            observation(instance, 1),
            connected_nonce="new-daemon-nonce",
        )
        assert accepted.event_sequence == 1


def test_reconnect_then_disconnect_refreshes_abandonment_grace(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    cutoff = datetime.now(UTC) - timedelta(hours=1)
    stale = cutoff - timedelta(hours=1)
    with ledger:
        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
        with ledger._transaction(immediate=True):
            ledger._connection.execute(
                "UPDATE bridge_stream SET last_seen = ?, disconnected_at = ? "
                "WHERE bridge_instance_id = ?",
                (stale.isoformat(), stale.isoformat(), instance),
            )
        assert ledger.list_abandonable_bridge_streams(last_seen_before=cutoff) == [
            (instance, engine.brain_id)
        ]

        ledger.attach_bridge_stream(
            instance,
            brain_id=engine.brain_id,
            server_actor_id=engine.actor_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            connected_nonce="new-daemon-nonce",
            recovery_token=RECOVERY_TOKEN,
        )
        ledger.disconnect_bridge_stream(
            instance,
            connected_nonce="new-daemon-nonce",
        )

        with pytest.raises(BridgeBindingError, match="grace was refreshed"):
            engine.abandon_bridge_stream(
                instance,
                last_seen_not_after=cutoff,
            )

        stream = ledger.bridge_stream_state(instance)
        assert stream.status == "open"
        assert stream.connected_nonce is None
        assert ledger.list_events(engine.brain_id) == []


def test_bridge_timestamps_never_regress_when_wall_clock_moves_backward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    baseline = ledger.bridge_stream_state(instance).last_seen
    regressed_wall = baseline - timedelta(days=1)

    class RegressedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return regressed_wall.replace(tzinfo=None)
            return regressed_wall.astimezone(tz)

    from alice_brain_hermes.runtime import store as store_module

    monkeypatch.setattr(store_module, "datetime", RegressedDateTime)
    with ledger:
        disconnected = ledger.disconnect_bridge_stream(
            instance, connected_nonce="daemon-nonce"
        )
        assert disconnected.last_seen == baseline
        assert disconnected.disconnected_at == baseline

        resumed = ledger.attach_bridge_stream(
            instance,
            brain_id=engine.brain_id,
            server_actor_id=engine.actor_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            connected_nonce="resumed-connection",
            recovery_token=RECOVERY_TOKEN,
        )
        assert resumed.last_seen == baseline

        engine.commit_bridge_record(
            instance,
            observation(instance, 1),
            connected_nonce="resumed-connection",
        )
        [accepted_at] = ledger._connection.execute(
            "SELECT accepted_at FROM bridge_record "
            "WHERE bridge_instance_id = ? AND first_capture_seq = 1",
            (instance,),
        ).fetchone()
        assert accepted_at == baseline.isoformat()

        closed = ledger.close_bridge_stream(
            instance,
            final_capture_seq=1,
            connected_nonce="resumed-connection",
        )
        assert closed.last_seen == baseline
        assert closed.disconnected_at == baseline


def test_restart_grace_refresh_never_regresses_persisted_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _engine, instance = make_engine(tmp_path)
    database = ledger.path
    baseline = ledger.bridge_stream_state(instance).last_seen
    regressed_wall = baseline - timedelta(days=1)

    class RegressedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return regressed_wall.replace(tzinfo=None)
            return regressed_wall.astimezone(tz)

    from alice_brain_hermes.runtime import store as store_module

    monkeypatch.setattr(store_module, "datetime", RegressedDateTime)
    with ledger:
        assert ledger.recover_stale_bridge_connections() == 1
        restarted = ledger.bridge_stream_state(instance)
        assert restarted.last_seen == baseline
        assert restarted.disconnected_at == baseline
        assert ledger.refresh_daemon_restart_grace() == 1
        assert ledger.bridge_stream_state(instance).last_seen == baseline

    with SQLiteLedger.open(database) as reopened:
        assert reopened.refresh_daemon_restart_grace() == 1
        refreshed = reopened.bridge_stream_state(instance)
        assert refreshed.last_seen == baseline
        assert refreshed.disconnected_at == baseline


def test_restart_grace_timestamp_is_generated_inside_write_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, _engine, _instance = make_engine(tmp_path)
    assert ledger.recover_stale_bridge_connections() == 1
    original = ledger._bridge_timestamp
    transaction_states: list[bool] = []

    def timestamp(last_seen=None):
        transaction_states.append(ledger._connection.in_transaction)
        return original(last_seen)

    monkeypatch.setattr(ledger, "_bridge_timestamp", timestamp)

    with ledger:
        assert ledger.refresh_daemon_restart_grace() == 1

    assert transaction_states == [True]


@pytest.mark.parametrize(
    "updates",
    [
        {
            "status": "open",
            "connected_nonce": None,
            "disconnected_reason": None,
            "disconnected_at": None,
        },
        {
            "status": "open",
            "connected_nonce": None,
            "disconnected_reason": "clean_close",
            "disconnected_at": datetime.now(UTC),
        },
        {"status": "open", "closed_final_seq": 0},
        {
            "status": "clean_closed",
            "connected_nonce": None,
            "disconnected_reason": "clean_close",
            "disconnected_at": datetime.now(UTC),
            "closed_final_seq": None,
        },
        {
            "status": "clean_closed",
            "connected_nonce": None,
            "disconnected_reason": "clean_close",
            "disconnected_at": datetime.now(UTC),
            "closed_final_seq": 1,
        },
        {
            "status": "abandoned",
            "connected_nonce": None,
            "disconnected_reason": "connection_eof",
            "disconnected_at": datetime.now(UTC),
        },
        {
            "status": "abandoned",
            "connected_nonce": None,
            "disconnected_reason": "grace_abandonment",
            "disconnected_at": datetime.now(UTC),
            "closed_final_seq": 0,
        },
    ],
)
def test_bridge_stream_model_rejects_impossible_status_combinations(
    updates: dict[str, object],
) -> None:
    values = {
        "bridge_instance_id": new_id(),
        "brain_id": new_id(),
        "server_actor_id": new_id(),
        "server_adapter_id": "adapter",
        "next_capture_seq": 1,
        "status": "open",
        "connected_nonce": "connection",
        "disconnected_reason": None,
        "disconnected_at": None,
        "last_seen": datetime.now(UTC),
        "closed_final_seq": None,
    }
    values.update(updates)

    with pytest.raises(ValueError):
        BridgeStreamState(**values)


def test_atomic_bridge_commit_persists_typed_event_frame_cursor_and_ack(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        record = observation(instance, 1)
        ack = engine.commit_bridge_record(instance, record)

        assert ack.record_fingerprint == record.fingerprint()
        assert ack.raw_event_sequence == 1
        assert ack.last_event_sequence == ack.frame.state_sequence == 2
        assert ack.derived_event_count == 1
        assert ack.semantic_status == "gap"
        assert ack.through_capture_seq == 1
        assert ack.frame.through_capture_seq == 1
        assert ack.frame.brain_id == engine.brain_id
        assert ack.frame.capture_coverage["capture_coverage"] == "host_sanitized"
        assert ack.frame.freshness.stream_connection == "connected"
        assert ack.frame.freshness.scheduler_sample == "not_sampled"
        assert ack.frame.freshness.scheduler_tick == 0
        assert ack.duplicate is False
        stored, semantic_gap = ledger.list_events(engine.brain_id)
        assert stored.event_type == "hermes.observer.post_api_request"
        assert stored.actor_id == engine.actor_id
        assert stored.adapter_id == "alice-brain-hermes-observer-v1"
        assert stored.payload["payload"]["provider"] == "provider-x"
        assert stored.payload["payload"]["extensions"]["reasoning"] == {
            "effort": "high"
        }
        assert semantic_gap.event_type == "semantic.gap"
        assert ledger.bridge_stream_state(instance).next_capture_seq == 2

        with pytest.raises(TypeError):
            ack.frame.capture_coverage["capture_coverage"] = "full"


def test_bridge_commit_snapshot_and_scheduler_serialize_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brain_id = new_id()
    instance = new_id()
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        ledger.ensure_brain(brain_id)
        worker = SnapshotWorker(ledger, interval_events=2)
        engine = ConsciousEngine(
            ledger,
            brain_id,
            actor_id=brain_id,
            on_state_published=worker.notify,
        )
        worker.register(engine, known_snapshot_sequence=0)
        ledger.attach_bridge_stream(
            instance,
            brain_id=brain_id,
            server_actor_id=brain_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            connected_nonce="daemon-nonce",
            recovery_token=RECOVERY_TOKEN,
        )
        entered = threading.Event()
        release = threading.Event()
        scheduler_done = threading.Event()
        real_checkpoint = ledger.checkpoint_current_state

        def blocking_checkpoint(state):
            entered.set()
            assert release.wait(2.0)
            return real_checkpoint(state)

        monkeypatch.setattr(ledger, "checkpoint_current_state", blocking_checkpoint)
        worker.start()
        try:
            ack = engine.commit_bridge_record(instance, observation(instance, 1))
            assert ack.last_event_sequence == 2
            assert entered.wait(2.0)

            scheduler = ContinuousScheduler(
                engine,
                interval_seconds=60.0,
                monotonic=lambda: 1.0,
            )
            scheduler_thread = threading.Thread(
                target=lambda: (scheduler.step(), scheduler_done.set())
            )
            scheduler_thread.start()
            assert scheduler_done.wait(0.05) is False

            release.set()
            scheduler_thread.join(2.0)
            assert scheduler_done.is_set()
            worker.wait_idle(timeout=2.0)
            assert engine.state == ledger.replay(brain_id, use_snapshot=False)
            assert ledger.load_snapshot(brain_id) == engine.state
        finally:
            release.set()
            worker.stop(timeout=2.0)


def test_public_frame_projection_reports_every_omitted_state_collection(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        engine.append(
            new_event(
                "action.proposed",
                engine.brain_id,
                engine.actor_id,
                {"action_id": "action-1", "intent": {"kind": "test"}},
            )
        )
        engine.append(
            new_event(
                "action.energy_assessed",
                engine.brain_id,
                engine.actor_id,
                {
                    "action_id": "action-1",
                    "deficits": {"certainty": 0.5},
                    "salience": 0.5,
                    "urgency": 0.5,
                    "valence": 0.0,
                    "arousal": 0.0,
                    "control": 0.5,
                    "resources": 0.5,
                    "cost": 0.5,
                    "personality_relevance": 0.5,
                },
            )
        )
        engine.append(
            new_event(
                "simulation.created",
                engine.brain_id,
                engine.actor_id,
                {
                    "branch_id": "branch-1",
                    "proposition_id": "proposition-1",
                    "content": {"action": "test"},
                },
            )
        )
        engine.append(
            new_event(
                "personality.revised",
                engine.brain_id,
                engine.actor_id,
                {"layer": "traits", "values": {"care": 0.01}},
            )
        )
        engine.append(
            new_event(
                "action.prepared",
                engine.brain_id,
                engine.actor_id,
                {"action_id": "action-1", "branch_id": "branch-1"},
            )
        )
        engine.append(
            new_event(
                "action.dispatched",
                engine.brain_id,
                engine.actor_id,
                {"action_id": "action-1"},
            )
        )
        engine.append(
            new_event(
                "action.receipt",
                engine.brain_id,
                engine.actor_id,
                {
                    "action_id": "action-1",
                    "status": "success",
                    "effect_evidence": {
                        "kind": "linked_observation",
                        "observation_ids": [f"effect-{index}" for index in range(6)],
                    },
                    "observations": [
                        {
                            "proposition_id": f"effect-{index}",
                            "content": {"index": index},
                        }
                        for index in range(6)
                    ],
                },
            )
        )
        engine.append(
            new_event(
                "observation.recorded",
                engine.brain_id,
                engine.actor_id,
                {"proposition_id": "observed-1", "content": {"fact": True}},
            )
        )
        other_actor = new_id()
        engine.append(
            new_event(
                "identity.actor_registered",
                engine.brain_id,
                engine.actor_id,
                {
                    "actor_id": other_actor,
                    "kind": "human",
                    "display_name": "Operator",
                    "parent_actor_id": engine.actor_id,
                    "attributes": {"team": "red"},
                },
            )
        )
        engine.append(
            new_event(
                "identity.provenance_authorized",
                engine.brain_id,
                engine.actor_id,
                {"actor_id": other_actor, "adapter_id": "trusted-adapter"},
            )
        )
        engine.append(
            new_event(
                "workspace.broadcast",
                engine.brain_id,
                engine.actor_id,
                {
                    "cycle": 1,
                    "candidates": [
                        {
                            "candidate_id": "candidate-1",
                            "specialist": "drives",
                            "score": 0.8,
                            "content": {"signal": "bounded"},
                            "source_ids": ["action-1"],
                            "cycle": 1,
                        }
                    ],
                },
            )
        )
        engine.append(
            new_event(
                "memory.recorded",
                engine.brain_id,
                engine.actor_id,
                {
                    "memory_id": "memory-1",
                    "content": {"fact": "test"},
                    "source_ids": [f"source-{index}" for index in range(6)],
                },
            )
        )
        engine.append(
            new_event(
                "capabilities.reported",
                engine.brain_id,
                engine.actor_id,
                {
                    "capabilities": {
                        "shell": True,
                        "limits": {"mode": "full", "tokens": 8_192},
                    }
                },
            )
        )
        engine.append(
            new_event(
                "runtime.failure",
                engine.brain_id,
                engine.actor_id,
                {
                    "error_type": "InjectedFailure",
                    "message": "bounded failure",
                    "phase": "test.projection",
                },
            )
        )

        frame = ledger.project_bridge_frame(
            instance,
            expected_state=engine.state,
            scheduler_sample="running",
        )

        assert frame.schema_version == 3
        assert frame.semantic_schema_version == 1
        assert frame.aggregate_semantic_complete is True
        assert frame.pc["traits"][0] == {"key": "care", "value": 0.01}
        assert frame.energy["items"][0]["action_id"] == "action-1"
        assert frame.st["workspace"][0]["candidate_id"] == "candidate-1"
        assert frame.st["branches"][0]["stance"] == "simulate"
        assert frame.rd["actions"][0]["phase"] == "receipt"
        assert frame.a["actions"][0]["dispatch_observed"] is True
        assert frame.a["actions"][0]["receipt_status"] == "success"
        assert frame.a["actions"][0]["blocked"] is None
        assert frame.a["actions"][0]["blocked_fact_available"] is False
        assert "outcome" not in frame.a["actions"][0]
        assert "receipt_conflict_count" not in frame.a["actions"][0]
        assert any(
            item["proposition_id"] == "observed-1" for item in frame.world["observed"]
        )
        assert frame.self_boundary["self_actor_id"] == engine.actor_id
        assert frame.self_boundary["other_actor_ids"] == (other_actor,)
        assert frame.memory["items"][0]["memory_id"] == "memory-1"
        shell = next(
            item for item in frame.capabilities["items"] if item["key"] == "shell"
        )
        assert shell == {
            "key": "shell",
            "boolean_value": True,
            "value_present": True,
            "value_type": "boolean",
        }
        assert frame.semantic_context["available"] is False
        omissions = frame.omission_counts
        assert omissions["energy"]["included"] == 1
        assert omissions["energy"]["omitted"] == 0
        assert omissions["energy"]["fields"]["deficits_json_nodes"]["omitted"] > 0
        assert omissions["energy"]["fields"]["arousal_values"]["omitted"] == 1
        assert (
            omissions["energy"]["fields"]["personality_relevance_values"]["omitted"]
            == 1
        )
        assert (
            omissions["st"]["workspace"]["fields"]["content_json_nodes"]["omitted"] > 0
        )
        assert omissions["st"]["workspace"]["fields"]["cycle_values"]["omitted"] == 1
        assert (
            omissions["st"]["branches"]["fields"]["content_json_nodes"]["omitted"] > 0
        )
        assert omissions["rd"]["fields"]["intent_json_nodes"]["omitted"] > 0
        assert omissions["rd"]["fields"]["receipt_exact_json_nodes"]["omitted"] > 0
        assert omissions["rd"]["fields"]["effect_evidence_id_items"] == {
            "included": 4,
            "omitted": 2,
        }
        assert omissions["a"]["fields"]["intent_json_nodes"]["omitted"] > 0
        assert omissions["a"]["fields"]["effect_evidence_id_items"] == {
            "included": 0,
            "omitted": 6,
        }
        assert (
            omissions["world"]["layers"]["observed"]["fields"]["content_json_nodes"][
                "omitted"
            ]
            > 0
        )
        assert (
            omissions["self_boundary"]["fields"]["other_actor_metadata_json_nodes"][
                "omitted"
            ]
            > 0
        )
        assert omissions["self_boundary"]["fields"]["authorization_records"] == {
            "included": 0,
            "omitted": 1,
        }
        assert omissions["memory"]["included"] == 1
        assert omissions["memory"]["omitted"] == 0
        assert omissions["memory"]["fields"]["content_json_nodes"]["omitted"] > 0
        assert omissions["memory"]["fields"]["source_id_items"] == {
            "included": 4,
            "omitted": 2,
        }
        assert (
            omissions["capabilities"]["fields"]["non_boolean_value_json_nodes"][
                "omitted"
            ]
            > 0
        )
        assert omissions["runtime"]["fields"]["last_failure_json_nodes"]["omitted"] > 0
        assert omissions["cognition"]["fields"]["state_json_nodes"]["omitted"] > 0
        assert omissions["raw_lifecycle_counts"]["omitted"] > 0


def test_rd_projection_labels_each_action_event_as_a_transition_not_outcome(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    transitions = [
        (
            "action.proposed",
            {"action_id": "action-1", "intent": {"kind": "test"}},
            "proposed",
        ),
        ("action.prepared", {"action_id": "action-1"}, "prepared"),
        ("action.dispatched", {"action_id": "action-1"}, "dispatched"),
        (
            "action.receipt",
            {"action_id": "action-1", "status": "success"},
            "receipt",
        ),
        (
            "action.reconstructed",
            {"action_id": "action-1", "summary": "observed result"},
            "reconstructed",
        ),
    ]

    with ledger:
        for event_type, payload, expected_phase in transitions:
            stored = engine.append(
                new_event(
                    event_type,
                    engine.brain_id,
                    engine.actor_id,
                    payload,
                )
            )
            frame = ledger.project_bridge_frame(
                instance,
                expected_state=engine.state,
                scheduler_sample="running",
            )
            [projected] = frame.rd["actions"]
            assert projected["phase"] == expected_phase
            assert projected["last_transition_event_id"] == stored.event_id
            assert "outcome_event_id" not in projected


def test_action_projection_preserves_canonical_outcome_and_reports_conflicts(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    action_id = "conflicting-receipt-action"
    events = (
        (
            "action.proposed",
            {"action_id": action_id, "intent": {"kind": "test"}},
        ),
        ("action.prepared", {"action_id": action_id}),
        ("action.dispatched", {"action_id": action_id}),
        (
            "action.receipt",
            {
                "action_id": action_id,
                "status": "success",
                "source_status": "ok",
            },
        ),
        ("action.reconstructed", {"action_id": action_id}),
        (
            "action.receipt",
            {
                "action_id": action_id,
                "status": "failure",
                "source_status": "error",
                "late": True,
            },
        ),
    )

    with ledger:
        for event_type, payload in events:
            engine.append(
                new_event(
                    event_type,
                    engine.brain_id,
                    engine.actor_id,
                    payload,
                    action_id=action_id,
                )
            )

        frame = ledger.project_bridge_frame(
            instance,
            expected_state=engine.state,
            scheduler_sample="running",
        )

    [projected] = frame.a["actions"]
    assert projected["receipt_status"] == "success"
    assert projected["outcome"] == "success"
    assert projected["receipt_conflict_count"] == 1
    assert projected["receipt_corroboration_count"] == 0


def test_blocked_action_projection_reports_observed_non_dispatch(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    action_id = "blocked-action"

    with ledger:
        for event_type, payload in (
            (
                "action.proposed",
                {"action_id": action_id, "intent": {"kind": "test"}},
            ),
            ("action.prepared", {"action_id": action_id}),
            (
                "action.blocked",
                {"action_id": action_id, "source_status": "blocked"},
            ),
        ):
            engine.append(
                new_event(
                    event_type,
                    engine.brain_id,
                    engine.actor_id,
                    payload,
                    action_id=action_id,
                )
            )

        frame = ledger.project_bridge_frame(
            instance,
            expected_state=engine.state,
            scheduler_sample="running",
        )

    [projected] = frame.a["actions"]
    assert projected["dispatch_observed"] is False
    assert projected["blocked"] is True
    assert projected["blocked_fact_available"] is True
    assert projected["execution_confirmed"] is False
    assert "outcome" not in projected


def test_public_frame_projection_rejects_tampered_latest_capture_record(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        engine.commit_bridge_record(instance, observation(instance, 1))
        ledger._connection.execute(
            "UPDATE bridge_record SET record_json = "
            "replace(record_json, 'host_sanitized', 'full')"
        )

        with pytest.raises(LedgerIntegrityError):
            ledger.project_bridge_frame(
                instance,
                expected_state=engine.state,
                scheduler_sample="running",
            )


def test_external_bridge_commit_permanently_poison_mutation_seal(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    with ledger:
        engine.commit_bridge_record(instance, observation(instance, 1))
        with sqlite3.connect(database) as external:
            external.execute(
                "UPDATE bridge_stream SET next_capture_seq = 1 "
                "WHERE bridge_instance_id = ?",
                (instance,),
            )

        for operation in (
            lambda: ledger.bridge_stream_state(instance),
            lambda: engine.commit_bridge_record(instance, observation(instance, 1)),
        ):
            with pytest.raises(LedgerIntegrityError, match="mutation seal"):
                operation()


def test_lost_ack_retry_is_exact_after_later_c0_state(tmp_path: Path) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        record = observation(instance, 1)
        first = engine.commit_bridge_record(instance, record)
        engine.pulse(0.25)
        assert engine.state.last_sequence > first.frame.state_sequence

        retried = engine.commit_bridge_record(instance, record)

        assert retried == first
        assert retried.frame.state_sequence == first.last_event_sequence
        assert (
            len(
                [
                    event
                    for event in ledger.list_events(engine.brain_id)
                    if event.event_type == "hermes.observer.post_api_request"
                ]
            )
            == 1
        )


def test_bridge_hot_paths_never_replay_or_scan_full_stream_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        record = observation(instance, 1)
        first = engine.commit_bridge_record(instance, record)
        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")

        def forbidden(*_args, **_kwargs):
            raise AssertionError("hot path attempted full history replay")

        monkeypatch.setattr(ledger, "_full_replay_in_transaction", forbidden)
        monkeypatch.setattr(ledger, "_replay_target_states_in_transaction", forbidden)
        monkeypatch.setattr(ledger, "_validate_bridge_stream_history", forbidden)

        resumed = ledger.attach_bridge_stream(
            instance,
            brain_id=engine.brain_id,
            server_actor_id=engine.actor_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            connected_nonce="resumed-connection",
            recovery_token=RECOVERY_TOKEN,
        )
        projected = ledger.project_bridge_frame(
            instance,
            expected_state=engine.state,
            connected_nonce="resumed-connection",
            scheduler_sample="running",
        )
        duplicate = engine.commit_bridge_record(
            instance,
            record,
            connected_nonce="resumed-connection",
        )

        assert resumed.next_capture_seq == 2
        assert projected.state_sequence == engine.state.last_sequence
        assert duplicate.canonical_json() == first.canonical_json()


@pytest.mark.parametrize(
    ("hook", "context", "payload", "required_payload_field"),
    HOOK_CASES,
    ids=[case[0] for case in HOOK_CASES],
)
def test_lost_ack_retry_is_exact_for_every_hook_context_shape(
    tmp_path: Path,
    hook: str,
    context: dict[str, object],
    payload: dict[str, object],
    required_payload_field: str,
) -> None:
    del required_payload_field
    ledger, engine, instance = make_engine(tmp_path)
    record = validate_observation(
        {
            "bridge_instance_id": instance,
            "capture_seq": 1,
            "captured_at": datetime.now(UTC),
            "captured_monotonic_ns": 1,
            "hook": hook,
            "context": context,
            "payload": {**payload, "extensions": {}},
            "coverage": CoverageV1(
                policy_version="copy-v1",
                capture_coverage="host_sanitized",
            ),
        }
    )

    with ledger:
        first = engine.commit_bridge_record(instance, record)
        retried = engine.commit_bridge_record(instance, record)

        assert retried == first
        assert retried.event_sequence == 1
        assert len(ledger.list_events(engine.brain_id)) == (
            1 + first.derived_event_count
        )


def test_lost_ack_retry_for_gap_uses_strict_shared_record_validator(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    gap = BridgeGapV1(
        bridge_instance_id=instance,
        first_capture_seq=1,
        last_capture_seq=2,
        dropped_count=2,
        cause_counts={"queue_full": 2},
    )

    with ledger:
        first = engine.commit_bridge_record(instance, gap)
        retried = engine.commit_bridge_record(instance, gap)

        assert retried == first
        assert len(ledger.list_events(engine.brain_id)) == 1


def test_changed_retry_conflicts_without_mutating_ledger(tmp_path: Path) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        record = observation(instance, 1)
        accepted = engine.commit_bridge_record(instance, record)
        changed_values = record.model_dump(mode="python")
        changed_values["payload"]["provider"] = "changed"
        changed = validate_observation(changed_values)

        with pytest.raises(IdempotencyConflictError):
            engine.commit_bridge_record(instance, changed)

        assert engine.state.last_sequence == accepted.last_event_sequence
        assert len(ledger.list_events(engine.brain_id)) == (
            1 + accepted.derived_event_count
        )
        assert ledger.bridge_stream_state(instance).next_capture_seq == 2


def test_out_of_order_observation_requires_exact_gap(tmp_path: Path) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        with pytest.raises(CaptureGapRequiredError):
            engine.commit_bridge_record(instance, observation(instance, 3))

        gap = BridgeGapV1(
            bridge_instance_id=instance,
            first_capture_seq=1,
            last_capture_seq=2,
            dropped_count=2,
            cause_counts={"queue_full": 1, "shape_failed": 1},
        )
        gap_ack = engine.commit_bridge_record(instance, gap)
        observation_ack = engine.commit_bridge_record(
            instance, observation(instance, 3)
        )

        assert gap_ack.through_capture_seq == 2
        assert gap_ack.frame.trace_complete is False
        assert gap_ack.frame.capture_coverage["omitted_nodes"] == 0
        assert gap_ack.frame.capture_coverage["channels"] == {
            "dropped_records": 2,
            "omitted_nodes_known": False,
            "trace": "gap",
        }
        assert observation_ack.through_capture_seq == 3
        assert [event.event_type for event in ledger.list_events(engine.brain_id)] == [
            "trace.gap",
            "hermes.observer.post_api_request",
            "semantic.gap",
        ]


def test_bridge_close_is_exact_idempotent_and_prevents_more_commits(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        engine.commit_bridge_record(instance, observation(instance, 1))
        first = ledger.close_bridge_stream(instance, final_capture_seq=1)
        second = ledger.close_bridge_stream(instance, final_capture_seq=1)

        assert first == second
        assert first.status == "clean_closed"
        with pytest.raises(BridgeClosedError):
            engine.commit_bridge_record(instance, observation(instance, 2))
        with pytest.raises(errors.BridgeCleanClosedError):
            ledger.attach_bridge_stream(
                instance,
                brain_id=engine.brain_id,
                server_actor_id=engine.actor_id,
                server_adapter_id="alice-brain-hermes-observer-v1",
                connected_nonce="new-connection",
                recovery_token=RECOVERY_TOKEN,
            )


def test_bridge_recovery_proof_is_digest_only_and_required_for_open_resume(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        [stored_digest] = ledger._connection.execute(
            "SELECT recovery_token_digest FROM bridge_stream "
            "WHERE bridge_instance_id = ?",
            (instance,),
        ).fetchone()
        assert stored_digest == hashlib.sha256(RECOVERY_TOKEN.encode("ascii")).digest()
        assert RECOVERY_TOKEN.encode("ascii") not in stored_digest

        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
        with pytest.raises(BridgeBindingError, match="provenance"):
            ledger.attach_bridge_stream(
                instance,
                brain_id=engine.brain_id,
                server_actor_id=engine.actor_id,
                server_adapter_id="alice-brain-hermes-observer-v1",
                connected_nonce="replacement-connection",
                recovery_token="cd" * 32,
            )

        resumed = ledger.attach_bridge_stream(
            instance,
            brain_id=engine.brain_id,
            server_actor_id=engine.actor_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            connected_nonce="replacement-connection",
            recovery_token=RECOVERY_TOKEN,
        )
        assert resumed.status == "open"
        assert resumed.connected_nonce == "replacement-connection"


def test_bridge_close_recovery_returns_immutable_terminal_receipt(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        closed = ledger.close_bridge_stream(
            instance,
            final_capture_seq=0,
            connected_nonce="daemon-nonce",
        )
        recovered = ledger.recover_bridge_close(
            instance,
            brain_id=engine.brain_id,
            server_actor_id=engine.actor_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            recovery_token=RECOVERY_TOKEN,
            final_capture_seq=0,
        )

        assert recovered == closed
        assert ledger.bridge_stream_state(instance) == closed
        with pytest.raises(CaptureSequenceError, match="different final capture"):
            ledger.recover_bridge_close(
                instance,
                brain_id=engine.brain_id,
                server_actor_id=engine.actor_id,
                server_adapter_id="alice-brain-hermes-observer-v1",
                recovery_token=RECOVERY_TOKEN,
                final_capture_seq=1,
            )


def test_bridge_close_recovery_rejects_open_and_abandoned_streams(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        recovery = {
            "brain_id": engine.brain_id,
            "server_actor_id": engine.actor_id,
            "server_adapter_id": "alice-brain-hermes-observer-v1",
            "recovery_token": RECOVERY_TOKEN,
            "final_capture_seq": 0,
        }
        with pytest.raises(BridgeClosedError, match="no clean-close receipt"):
            ledger.recover_bridge_close(instance, **recovery)

        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
        engine.abandon_bridge_stream(instance)
        with pytest.raises(BridgeClosedError, match="no clean-close receipt"):
            ledger.recover_bridge_close(instance, **recovery)


def test_bridge_sql_failure_rolls_back_event_cursor_record_and_engine_state(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        with ledger._transaction(immediate=True):
            ledger._connection.execute(
                "CREATE TRIGGER reject_bridge_record BEFORE INSERT ON bridge_record "
                "BEGIN SELECT RAISE(ABORT, 'fixture bridge rejection'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="fixture bridge rejection"):
            engine.commit_bridge_record(instance, observation(instance, 1))

        assert engine.state.last_sequence == 0
        assert ledger.list_events(engine.brain_id) == []
        assert ledger.bridge_stream_state(instance).next_capture_seq == 1


def test_schema_v2_migrates_transactionally_without_losing_brains(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy.db"
    brain_id = new_id()
    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(brain_id)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TABLE hermes_span")
        connection.execute("DROP TABLE brain_observability")
        connection.execute("DROP TABLE bridge_record")
        connection.execute("DROP TABLE bridge_stream")
        connection.execute("DROP TABLE brain_profile")
        connection.execute("DROP INDEX energy_assessment_pending")
        connection.execute("DROP TABLE energy_assessment_lease")
        connection.execute("DROP INDEX identity_naming_pending")
        connection.execute("DROP TABLE identity_naming_lease")
        connection.execute("DROP TABLE identity_name_registry")
        connection.execute(
            "UPDATE schema_metadata SET value = '2' WHERE key = 'schema_version'"
        )
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
    finally:
        connection.close()

    with SQLiteLedger.open(database) as migrated:
        assert migrated.list_brain_ids() == [brain_id]
        tables = {
            row[0]
            for row in migrated._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"bridge_stream", "bridge_record"}.issubset(tables)
        recovery_column = next(
            row
            for row in migrated._connection.execute("PRAGMA table_info(bridge_stream)")
            if row[1] == "recovery_token_digest"
        )
        assert recovery_column[2] == "BLOB"
        assert recovery_column[3] == 1


def test_malformed_v2_schema_rolls_back_the_whole_migration(tmp_path: Path) -> None:
    database = tmp_path / "malformed-legacy.db"
    with SQLiteLedger.open(database):
        pass
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TABLE hermes_span")
        connection.execute("DROP TABLE brain_observability")
        connection.execute("DROP TABLE bridge_record")
        connection.execute("DROP TABLE bridge_stream")
        connection.execute("DROP TABLE brain_profile")
        connection.execute("DROP INDEX energy_assessment_pending")
        connection.execute("DROP TABLE energy_assessment_lease")
        connection.execute("DROP INDEX identity_naming_pending")
        connection.execute("DROP TABLE identity_naming_lease")
        connection.execute("DROP TABLE identity_name_registry")
        connection.execute("DROP INDEX events_event_id")
        connection.execute(
            "UPDATE schema_metadata SET value = '2' WHERE key = 'schema_version'"
        )
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SchemaVersionError, match="v2 structure"):
        SQLiteLedger.open(database)

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'bridge_stream'"
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
            == "2"
        )
    finally:
        connection.close()


def test_existing_v7_rejects_extra_schema_metadata_rows(tmp_path: Path) -> None:
    database = tmp_path / "extra-v7-metadata.db"
    with SQLiteLedger.open(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO schema_metadata(key, value) VALUES ('unexpected', '4')"
        )

    with pytest.raises(SchemaVersionError, match="metadata"):
        SQLiteLedger.open(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT key, value FROM schema_metadata ORDER BY key"
        ).fetchall() == [("schema_version", "7"), ("unexpected", "4")]


def test_v2_extra_schema_metadata_row_fails_without_partial_migration(
    tmp_path: Path,
) -> None:
    database = tmp_path / "extra-v2-metadata.db"
    with SQLiteLedger.open(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE hermes_span")
        connection.execute("DROP TABLE brain_observability")
        connection.execute("DROP TABLE bridge_record")
        connection.execute("DROP TABLE bridge_stream")
        connection.execute("DROP TABLE brain_profile")
        connection.execute("DROP INDEX energy_assessment_pending")
        connection.execute("DROP TABLE energy_assessment_lease")
        connection.execute("DROP INDEX identity_naming_pending")
        connection.execute("DROP TABLE identity_naming_lease")
        connection.execute("DROP TABLE identity_name_registry")
        connection.execute(
            "UPDATE schema_metadata SET value = '2' WHERE key = 'schema_version'"
        )
        connection.execute(
            "INSERT INTO schema_metadata(key, value) VALUES ('unexpected', '2')"
        )
        connection.execute("PRAGMA user_version = 2")

    with pytest.raises(SchemaVersionError, match="metadata"):
        SQLiteLedger.open(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'bridge_stream'"
            ).fetchone()
            is None
        )
        assert connection.execute(
            "SELECT key, value FROM schema_metadata ORDER BY key"
        ).fetchall() == [("schema_version", "2"), ("unexpected", "2")]


def test_existing_v4_reopen_rejects_unused_profile_tampering(
    tmp_path: Path,
) -> None:
    database = tmp_path / "profile-tamper.db"
    with SQLiteLedger.open(database) as ledger:
        ledger.resolve_brain_profile(
            BrainProfileV1(profile_key="hermes.default", name="Alice")
        )
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE brain_profile SET profile_fingerprint = 'tampered'")

    with pytest.raises(SchemaVersionError, match="profile data integrity"):
        SQLiteLedger.open(database)


def test_recovery_digest_database_constraint_rejects_null(tmp_path: Path) -> None:
    ledger, _engine, _instance = make_engine(tmp_path)
    database = ledger.path
    ledger.close()

    with (
        sqlite3.connect(database) as connection,
        pytest.raises(sqlite3.IntegrityError, match="NOT NULL"),
    ):
        connection.execute("UPDATE bridge_stream SET recovery_token_digest = NULL")


@pytest.mark.parametrize(
    "tampered_digest",
    ["x" * 32, sqlite3.Binary(b"x" * 31), sqlite3.Binary(b"x" * 33)],
    ids=["text", "31-bytes", "33-bytes"],
)
def test_existing_v4_reopen_rejects_invalid_recovery_digest_storage(
    tmp_path: Path,
    tampered_digest: object,
) -> None:
    ledger, _engine, _instance = make_engine(tmp_path)
    database = ledger.path
    ledger.close()
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE bridge_stream SET recovery_token_digest = ?",
            (tampered_digest,),
        )

    with pytest.raises(SchemaVersionError, match="integrity"):
        SQLiteLedger.open(database)


@pytest.mark.parametrize(
    "tamper_sql",
    [
        "UPDATE bridge_record SET record_json = "
        "replace(record_json, 'host_sanitized', 'full')",
        "UPDATE bridge_record SET accepted_at = '2000-01-01T01:00:00+01:00'",
        "UPDATE bridge_stream SET last_seen = 'not-a-time'",
        "UPDATE bridge_stream SET status = 'clean_closed', "
        "connected_nonce = NULL, disconnected_reason = 'clean_close', "
        "disconnected_at = last_seen, closed_final_seq = NULL",
    ],
)
def test_existing_v4_reopen_rejects_unattached_bridge_row_tampering(
    tmp_path: Path, tamper_sql: str
) -> None:
    database = tmp_path / "bridge-tamper.db"
    ledger = SQLiteLedger.open(database)
    brain_id = new_id()
    instance = new_id()
    ledger.ensure_brain(brain_id)
    engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)
    ledger.attach_bridge_stream(
        instance,
        brain_id=brain_id,
        server_actor_id=brain_id,
        server_adapter_id="alice-brain-hermes-observer-v1",
        connected_nonce="prior-process",
        recovery_token=RECOVERY_TOKEN,
    )
    engine.commit_bridge_record(instance, observation(instance, 1))
    ledger.close()
    with sqlite3.connect(database) as connection:
        connection.execute(tamper_sql)

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)


def test_v2_migration_with_extra_object_rolls_back_without_partial_v4(
    tmp_path: Path,
) -> None:
    database = tmp_path / "extra-v2.db"
    with SQLiteLedger.open(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE hermes_span")
        connection.execute("DROP TABLE brain_observability")
        connection.execute("DROP TABLE bridge_record")
        connection.execute("DROP TABLE bridge_stream")
        connection.execute("DROP TABLE brain_profile")
        connection.execute("DROP INDEX energy_assessment_pending")
        connection.execute("DROP TABLE energy_assessment_lease")
        connection.execute("DROP INDEX identity_naming_pending")
        connection.execute("DROP TABLE identity_naming_lease")
        connection.execute("DROP TABLE identity_name_registry")
        connection.execute("CREATE TABLE unexpected_object(value INTEGER)")
        connection.execute(
            "UPDATE schema_metadata SET value = '2' WHERE key = 'schema_version'"
        )
        connection.execute("PRAGMA user_version = 2")

    with pytest.raises(SchemaVersionError, match="v2 structure"):
        SQLiteLedger.open(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'bridge_stream'"
            ).fetchone()
            is None
        )


@pytest.mark.parametrize("audit_path", ["attach", "reopen"])
def test_bridge_history_requires_strictly_increasing_ledger_sequence(
    tmp_path: Path, audit_path: str
) -> None:
    ledger, engine, instance = make_engine(tmp_path / audit_path)
    database = ledger.path
    engine.commit_bridge_record(instance, observation(instance, 1))
    engine.commit_bridge_record(instance, observation(instance, 2))
    with ledger._transaction(immediate=True):
        ledger._connection.execute(
            "UPDATE bridge_record SET ledger_sequence = 1, "
            "derived_first_sequence = 2, derived_last_sequence = 2 "
            "WHERE bridge_instance_id = ? AND first_capture_seq = 2",
            (instance,),
        )

    if audit_path == "attach":
        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
        with pytest.raises(LedgerIntegrityError):
            ledger.attach_bridge_stream(
                instance,
                brain_id=engine.brain_id,
                server_actor_id=engine.actor_id,
                server_adapter_id="alice-brain-hermes-observer-v1",
                connected_nonce="reconnect",
                recovery_token=RECOVERY_TOKEN,
            )
        ledger.close()
    else:
        ledger.close()
        with pytest.raises(SchemaVersionError, match="bridge or profile"):
            SQLiteLedger.open(database)


def test_startup_integrity_audit_reduces_long_history_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    for capture_seq in range(1, 21):
        engine.commit_bridge_record(instance, observation(instance, capture_seq))
        if capture_seq % 5 == 0:
            ledger.save_snapshot(engine.state)
    event_count = len(ledger.list_events(engine.brain_id, limit=100))
    ledger.close()

    from alice_brain_hermes.runtime import store as store_module

    real_reduce = store_module.reduce_state
    calls = 0

    def counted_reduce(state, event):
        nonlocal calls
        calls += 1
        return real_reduce(state, event)

    monkeypatch.setattr(store_module, "reduce_state", counted_reduce)

    with SQLiteLedger.open(database):
        pass

    assert calls == event_count


def test_engine_bootstrap_reuses_startup_audited_final_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "runtime.db"
    brain_id = new_id()
    actor_id = brain_id
    with SQLiteLedger.open(database) as ledger:
        for sequence in range(20):
            ledger.append(
                new_event(
                    "opaque.event",
                    brain_id,
                    actor_id,
                    {"sequence": sequence},
                )
            )

    with SQLiteLedger.open(database) as ledger:

        def forbidden(*_args, **_kwargs):
            raise AssertionError("engine repeated startup replay")

        monkeypatch.setattr(ledger, "replay", forbidden)
        engine = ConsciousEngine(ledger, brain_id, actor_id=actor_id)

        assert engine.state.last_sequence == 20


def test_engine_bootstrap_replays_when_startup_audit_falls_behind_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "runtime.db"
    brain_id = new_id()
    actor_id = brain_id
    with SQLiteLedger.open(database) as ledger:
        ledger.append(new_event("opaque.event", brain_id, actor_id, {"before": True}))

    with SQLiteLedger.open(database) as ledger:
        ledger.append(new_event("opaque.event", brain_id, actor_id, {"after": True}))
        real_replay = ledger.replay
        replay_calls = 0

        def counted_replay(target_brain_id: str):
            nonlocal replay_calls
            replay_calls += 1
            return real_replay(target_brain_id)

        monkeypatch.setattr(ledger, "replay", counted_replay)
        state = ledger.bootstrap_state(brain_id)

        assert replay_calls == 1
        assert state.last_sequence == 2


@pytest.mark.parametrize(
    "tamper_sql",
    [
        "UPDATE bridge_record SET last_capture_seq = 2",
        "UPDATE bridge_record SET record_kind = 'gap'",
        "UPDATE bridge_record SET ack_json = "
        "replace(ack_json, '\"through_capture_seq\":1', "
        "'\"through_capture_seq\":2')",
        "UPDATE bridge_stream SET next_capture_seq = 1",
    ],
)
def test_lost_ack_retry_rejects_tampered_persisted_relations(
    tmp_path: Path, tamper_sql: str
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        record = observation(instance, 1)
        engine.commit_bridge_record(instance, record)
        ledger._connection.execute(tamper_sql)

        with pytest.raises(LedgerIntegrityError):
            engine.commit_bridge_record(instance, record)


def test_gap_cause_counts_are_deeply_immutable_and_fingerprint_stable() -> None:
    causes = {"queue_full": 2}
    gap = BridgeGapV1(
        bridge_instance_id=new_id(),
        first_capture_seq=1,
        last_capture_seq=2,
        dropped_count=2,
        cause_counts=causes,
    )
    fingerprint = gap.fingerprint()

    causes["queue_full"] = 999
    with pytest.raises(TypeError):
        gap.cause_counts["queue_full"] = 3
    with pytest.raises(TypeError, match="immutable"):
        gap.cause_counts._data = {"queue_full": 999}

    assert gap.cause_counts == {"queue_full": 2}
    assert gap.fingerprint() == fingerprint


def test_frame_and_ack_frozen_json_backing_slots_cannot_be_rebound(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        acknowledgement = engine.commit_bridge_record(
            instance, observation(instance, 1)
        )
        ack_json = acknowledgement.canonical_json()
        frame_json = acknowledgement.frame.canonical_json()

        for frozen in (
            acknowledgement.frame.pc,
            acknowledgement.frame.capture_coverage,
            acknowledgement.frame.omission_counts,
        ):
            with pytest.raises(TypeError, match="immutable"):
                frozen._data = {"rebound": True}

        assert acknowledgement.canonical_json() == ack_json
        assert acknowledgement.frame.canonical_json() == frame_json


def test_concurrent_streams_share_one_engine_serialization_boundary(
    tmp_path: Path,
) -> None:
    ledger = SQLiteLedger.open(tmp_path / "runtime.db")
    brain_id = new_id()
    ledger.ensure_brain(brain_id)
    engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)
    instances = [new_id() for _ in range(24)]
    for index, instance in enumerate(instances):
        ledger.attach_bridge_stream(
            instance,
            brain_id=brain_id,
            server_actor_id=brain_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            connected_nonce=f"connection-{index}",
            recovery_token=hashlib.sha256(
                f"stream-{index}".encode("ascii")
            ).hexdigest(),
        )

    with ledger, ThreadPoolExecutor(max_workers=8) as pool:
        acknowledgements = list(
            pool.map(
                lambda instance: engine.commit_bridge_record(
                    instance, observation(instance, 1)
                ),
                instances,
            )
        )

        assert sorted(ack.raw_event_sequence for ack in acknowledgements) == list(
            range(1, 2 * len(instances), 2)
        )
        assert [event.sequence for event in ledger.list_events(brain_id)] == list(
            range(1, 2 * len(instances) + 1)
        )


def test_disconnect_is_resumable_with_same_cursor_and_rotated_connection(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        engine.commit_bridge_record(instance, observation(instance, 1))
        disconnected = ledger.disconnect_bridge_stream(
            instance, connected_nonce="daemon-nonce"
        )
        assert disconnected.connected_nonce is None

        resumed = ledger.attach_bridge_stream(
            instance,
            brain_id=engine.brain_id,
            server_actor_id=engine.actor_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            connected_nonce="new-connection",
            recovery_token=RECOVERY_TOKEN,
        )
        assert resumed.next_capture_seq == 2
        assert resumed.connected_nonce == "new-connection"


def test_grace_abandonment_atomically_appends_unknown_gap_and_is_not_resumable(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        engine.commit_bridge_record(instance, observation(instance, 1))
        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")

        abandoned = engine.abandon_bridge_stream(instance)

        assert abandoned.status == "abandoned"
        assert engine.state.trace_complete is False
        assert ledger.list_events(engine.brain_id)[-1].event_type == "trace.gap"
        payload = ledger.list_events(engine.brain_id)[-1].payload
        assert payload["exact"] is False
        assert payload["unknown_range"] is True
        assert payload["first_capture_seq"] == 2
        assert payload["last_capture_seq"] is None
        with pytest.raises(errors.BridgeAbandonedError):
            ledger.attach_bridge_stream(
                instance,
                brain_id=engine.brain_id,
                server_actor_id=engine.actor_id,
                server_adapter_id="alice-brain-hermes-observer-v1",
                connected_nonce="new-connection",
                recovery_token=RECOVERY_TOKEN,
            )


def test_reopen_rejects_abandoned_stream_without_its_unknown_gap(
    tmp_path: Path,
) -> None:
    ledger, _engine, instance = make_engine(tmp_path)
    database = ledger.path
    ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
    ledger.close()

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE bridge_stream SET status = 'abandoned', "
            "disconnected_reason = 'grace_abandonment' "
            "WHERE bridge_instance_id = ?",
            (instance,),
        )

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)


def test_reopen_rejects_orphaned_unknown_abandonment_gap(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
    engine.abandon_bridge_stream(instance)
    ledger.close()

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE bridge_stream SET status = 'open', "
            "disconnected_reason = 'connection_eof' "
            "WHERE bridge_instance_id = ?",
            (instance,),
        )

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)


def test_reopen_rejects_unknown_abandonment_gap_provenance_mismatch(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
    engine.abandon_bridge_stream(instance)
    ledger.close()

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE bridge_stream SET server_adapter_id = 'tampered-adapter' "
            "WHERE bridge_instance_id = ?",
            (instance,),
        )

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)


@pytest.mark.parametrize(
    "payload",
    [
        {"unknown_range": False},
        {"exact": False},
        {"cause_counts": {"abandoned_unknown": 2}},
        {"cause_counts": ["abandoned_unknown"]},
        {"cause_counts": "abandoned_unknown"},
    ],
    ids=[
        "unknown-range-key",
        "inexact-marker",
        "abandoned-cause",
        "abandoned-cause-list",
        "abandoned-cause-string",
    ],
)
def test_reopen_rejects_malformed_reserved_abandonment_gap(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    ledger, engine, _instance = make_engine(tmp_path)
    database = ledger.path
    ledger.append(new_event("trace.gap", engine.brain_id, engine.actor_id, payload))
    ledger.close()

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)


def test_reopen_keeps_exact_client_gap_disjoint_from_valid_abandonment(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    engine.commit_bridge_record(
        instance,
        BridgeGapV1(
            bridge_instance_id=instance,
            first_capture_seq=1,
            last_capture_seq=1,
            dropped_count=1,
            cause_counts={"queue_full": 1},
        ),
    )
    ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
    engine.abandon_bridge_stream(instance)
    ledger.close()

    with SQLiteLedger.open(database) as reopened:
        assert reopened.bridge_stream_state(instance).status == "abandoned"


def test_reopen_rejects_duplicate_unknown_abandonment_gap(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
    engine.abandon_bridge_stream(instance)
    stream = ledger.bridge_stream_state(instance)
    engine.append(
        new_event(
            "trace.gap",
            stream.brain_id,
            stream.server_actor_id,
            {
                "schema_version": 1,
                "record_kind": "gap",
                "bridge_instance_id": instance,
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
    )
    ledger.close()

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)


def test_reopen_rejects_unknown_abandonment_gap_before_later_bridge_record(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    stream = ledger.bridge_stream_state(instance)
    engine.append(
        new_event(
            "trace.gap",
            stream.brain_id,
            stream.server_actor_id,
            {
                "schema_version": 1,
                "record_kind": "gap",
                "bridge_instance_id": instance,
                "first_capture_seq": 2,
                "last_capture_seq": None,
                "dropped_count": None,
                "cause_counts": {"abandoned_unknown": 1},
                "exact": False,
                "unknown_range": True,
                "trace_complete": False,
            },
            adapter_id=stream.server_adapter_id,
        )
    )
    engine.commit_bridge_record(
        instance,
        BridgeGapV1(
            bridge_instance_id=instance,
            first_capture_seq=1,
            last_capture_seq=1,
            dropped_count=1,
            cause_counts={"queue_full": 1},
        ),
    )
    ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
    with ledger._transaction(immediate=True):
        ledger._connection.execute(
            "UPDATE bridge_stream SET status = 'abandoned', "
            "disconnected_reason = 'grace_abandonment' "
            "WHERE bridge_instance_id = ?",
            (instance,),
        )
    ledger.close()

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)


def test_abandonment_refuses_a_stream_that_resumed_before_grace(tmp_path: Path) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
        ledger.attach_bridge_stream(
            instance,
            brain_id=engine.brain_id,
            server_actor_id=engine.actor_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            connected_nonce="resumed",
            recovery_token=RECOVERY_TOKEN,
        )

        with pytest.raises(BridgeBindingError, match="connected"):
            engine.abandon_bridge_stream(instance)

        assert ledger.bridge_stream_state(instance).status == "open"
        assert engine.state.trace_complete is True


@pytest.mark.parametrize(
    "tamper_sql",
    [
        "UPDATE bridge_stream SET next_capture_seq = 1",
        "UPDATE bridge_stream SET next_capture_seq = 3",
        "DELETE FROM bridge_record",
        "UPDATE bridge_record SET last_capture_seq = 2",
    ],
)
def test_reconnect_rejects_false_cursor_holes_and_overlaps(
    tmp_path: Path, tamper_sql: str
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        engine.commit_bridge_record(instance, observation(instance, 1))
        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
        ledger._connection.execute(tamper_sql)

        with pytest.raises(LedgerIntegrityError):
            ledger.attach_bridge_stream(
                instance,
                brain_id=engine.brain_id,
                server_actor_id=engine.actor_id,
                server_adapter_id="alice-brain-hermes-observer-v1",
                connected_nonce="reconnect",
                recovery_token=RECOVERY_TOKEN,
            )


def test_reconnect_rejects_bridge_record_event_relation_tampering(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        engine.commit_bridge_record(instance, observation(instance, 1))
        ledger.disconnect_bridge_stream(instance, connected_nonce="daemon-nonce")
        unrelated = ledger.append(
            new_event("opaque.event", engine.brain_id, engine.actor_id, {})
        )
        ledger._connection.execute(
            "UPDATE bridge_record SET event_id = ?, ledger_sequence = ?, "
            "derived_first_sequence = ?, derived_last_sequence = ?",
            (
                unrelated.event_id,
                unrelated.sequence,
                unrelated.sequence + 1,
                unrelated.sequence + 1,
            ),
        )

        with pytest.raises(LedgerIntegrityError):
            ledger.attach_bridge_stream(
                instance,
                brain_id=engine.brain_id,
                server_actor_id=engine.actor_id,
                server_adapter_id="alice-brain-hermes-observer-v1",
                connected_nonce="reconnect",
                recovery_token=RECOVERY_TOKEN,
            )
