from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import alice_brain_hermes.runtime.store as store_module
from alice_brain_hermes.core.action import ActionOutcome, ActionPhase, RDPhase
from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict
from alice_brain_hermes.core.workspace import derive_candidates
from alice_brain_hermes.errors import (
    DomainCapacityError,
    IdempotencyConflictError,
    SchemaVersionError,
)
from alice_brain_hermes.protocol.models import (
    BridgeCommitAckV2,
    BridgeGapV1,
    validate_observation,
)
from alice_brain_hermes.runtime.store import SQLITE_SCHEMA_VERSION
from tests.protocol.test_models import HOOK_CASES
from tests.runtime.test_bridge_store import make_engine
from tests.runtime.test_semantic_ingest import generic_observation, tool_observation


def test_fresh_store_exposes_v7_energy_identity_and_semantic_schema() -> None:
    assert SQLITE_SCHEMA_VERSION == 7


def test_empty_store_observability_is_complete_zero_evidence_not_a_gap(
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.runtime.store import SQLiteLedger

    with SQLiteLedger.open(tmp_path / "empty-observability.db") as ledger:
        snapshot = ledger.observability_snapshot()

    assert snapshot.brain_count == 0
    assert snapshot.trace_complete is True
    assert snapshot.semantic_complete is True
    assert snapshot.dropped_events == 0
    assert snapshot.semantic_records == 0
    assert snapshot.semantic_gap_records == 0
    assert snapshot.total_bridges == 0


def test_global_observability_aggregates_large_brain_counts_exactly(
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.ids import new_id
    from alice_brain_hermes.runtime.engine import ConsciousEngine
    from alice_brain_hermes.runtime.store import SQLiteLedger

    database = tmp_path / "large-global-observability.db"
    per_brain = 2**62
    brain_ids: list[str] = []
    with SQLiteLedger.open(database) as ledger:
        for index in range(2):
            brain_id = new_id()
            bridge_instance_id = new_id()
            brain_ids.append(brain_id)
            ledger.ensure_brain(brain_id)
            engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)
            ledger.attach_bridge_stream(
                bridge_instance_id,
                brain_id=brain_id,
                server_actor_id=brain_id,
                server_adapter_id="alice-brain-hermes-observer-v1",
                connected_nonce=f"daemon-{index}",
                recovery_token="ab" * 32,
            )
            engine.commit_bridge_record(
                bridge_instance_id,
                BridgeGapV1(
                    bridge_instance_id=bridge_instance_id,
                    first_capture_seq=1,
                    last_capture_seq=per_brain,
                    dropped_count=per_brain,
                    cause_counts={"queue_full": per_brain},
                ),
            )

        aggregate = ledger.observability_snapshot()
        assert aggregate.dropped_events == per_brain * 2
        assert aggregate.dropped_events > 2**63 - 1
        assert aggregate.brain_count == 2
        assert aggregate.semantic_records == 2
        assert aggregate.semantic_gap_records == 2
        for brain_id in brain_ids:
            assert ledger.observability_snapshot(brain_id).dropped_events == per_brain

    with SQLiteLedger.open(database) as reopened:
        assert reopened.observability_snapshot() == aggregate


def test_per_brain_observability_capacity_fails_atomically_and_restart_safe(
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.ids import new_id
    from alice_brain_hermes.runtime.engine import ConsciousEngine
    from alice_brain_hermes.runtime.store import SQLiteLedger

    database = tmp_path / "per-brain-observability-capacity.db"
    maximum_capture_sequence = 2**63 - 2
    brain_id = new_id()
    first_bridge = new_id()
    second_bridge = new_id()
    first_gap = BridgeGapV1(
        bridge_instance_id=first_bridge,
        first_capture_seq=1,
        last_capture_seq=maximum_capture_sequence,
        dropped_count=maximum_capture_sequence,
        cause_counts={"queue_full": maximum_capture_sequence},
    )
    overflow_gap = BridgeGapV1(
        bridge_instance_id=second_bridge,
        first_capture_seq=1,
        last_capture_seq=2,
        dropped_count=2,
        cause_counts={"queue_full": 2},
    )
    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(brain_id)
        engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)
        for index, bridge_instance_id in enumerate((first_bridge, second_bridge)):
            ledger.attach_bridge_stream(
                bridge_instance_id,
                brain_id=brain_id,
                server_actor_id=brain_id,
                server_adapter_id="alice-brain-hermes-observer-v1",
                connected_nonce=f"daemon-{index}",
                recovery_token="ab" * 32,
            )
        accepted = engine.commit_bridge_record(first_bridge, first_gap)
        before = (
            engine.state,
            ledger.observability_snapshot(brain_id),
            ledger.bridge_stream_state(second_bridge),
            ledger._connection.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            ledger._connection.execute("SELECT COUNT(*) FROM bridge_record").fetchone()[
                0
            ],
        )

        with pytest.raises(DomainCapacityError, match="observability"):
            engine.commit_bridge_record(second_bridge, overflow_gap)

        assert (
            engine.state,
            ledger.observability_snapshot(brain_id),
            ledger.bridge_stream_state(second_bridge),
            ledger._connection.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            ledger._connection.execute("SELECT COUNT(*) FROM bridge_record").fetchone()[
                0
            ],
        ) == before
        assert engine.commit_bridge_record(first_bridge, first_gap) == accepted
        assert ledger.bridge_stream_state(first_bridge).next_capture_seq == 2**63 - 1
        [stored_count, storage_type] = ledger._connection.execute(
            "SELECT dropped_events, typeof(dropped_events) "
            "FROM brain_observability WHERE brain_id = ?",
            (brain_id,),
        ).fetchone()
        assert stored_count == maximum_capture_sequence
        assert storage_type == "integer"

    with SQLiteLedger.open(database) as reopened:
        assert reopened.observability_snapshot(brain_id).dropped_events == (
            maximum_capture_sequence
        )
        assert reopened.bridge_stream_state(second_bridge).next_capture_seq == 1


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
        assert ack.frame.aggregate_semantic_complete is True
        assert ack.frame.semantic_schema_version == 1
        assert ack.frame.semantic_evidence.semantic_records == 1
        assert ack.frame.semantic_evidence.semantic_gap_records == 0
        assert engine.state.last_sequence == 6
        assert [event.event_type for event in ledger.list_events(engine.brain_id)] == [
            "hermes.observer.pre_tool_call",
            "action.proposed",
            "personality.control.sampled",
            "action.energy_requested",
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
        lease = engine.claim_energy_assessment()
        assert lease is not None
        assert lease.assessment_input["st"]["tool_name"] == "shell"
        assert lease.assessment_input["st"]["args"] == {
            "command": "TOP SECRET RAW ARGUMENT"
        }
        assert lease.assessment_input["st"]["middleware_trace"] == {
            "private": "TRACE SECRET"
        }


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
        assert ack.derived_event_count == 3
        assert ack.last_event_sequence == 10
        assert ack.frame.state_sequence == 10
        [action] = engine.state.action_records
        assert action.phase is ActionPhase.RECONSTRUCTED
        assert action.rd_phase is RDPhase.RECONSTRUCT
        assert action.execution_confirmed is True
        assert action.outcome is ActionOutcome.FAILURE
        assert action.effect_confirmed is None
        assert action.receipt is not None
        assert action.receipt["status"] == "failure"
        assert action.reconstruction_history[-1].after_receipt_event_id == (
            action.receipt_history[-1].event_id
        )
        assert action.reconstruction_history[-1].payload["assessment"] == (
            "execution_failed"
        )
        [projected] = ack.frame.a["actions"]
        assert projected["dispatch_observed"] is True
        assert projected["execution_confirmed"] is True
        assert projected["receipt_status"] == "failure"
        assert ack.frame.rd["retained_action_count"] == 1
        assert ack.frame.rd["projected_action_count"] == 1
        assert ack.frame.rd["lifecycle_incomplete_count"] == 0
        assert ack.frame.rd["execution_unknown_count"] == 0
        assert ack.frame.rd["effect_unknown_count"] == 1
        assert ack.frame.rd["phase_counts"] == {
            "blocked": 0,
            "dispatched": 0,
            "prepared": 0,
            "proposed": 0,
            "receipt": 0,
            "reconstructed": 1,
        }
        assert ack.frame.rd["outcome_counts"] == {
            "failure": 1,
            "success": 0,
            "unknown": 0,
        }
        [closed_capture_seq] = ledger._connection.execute(
            "SELECT closed_capture_seq FROM hermes_span"
        ).fetchone()
        assert closed_capture_seq == 2


def test_56_completed_tools_are_retained_but_only_four_are_projected(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    latest = None
    with ledger:
        for index in range(56):
            pre_capture_seq = index * 2 + 1
            tool_call_id = f"tool-{index:02d}"
            engine.commit_bridge_record(
                instance,
                tool_observation(
                    instance,
                    pre_capture_seq,
                    hook="pre_tool_call",
                    tool_call_id=tool_call_id,
                ),
            )
            latest = engine.commit_bridge_record(
                instance,
                tool_observation(
                    instance,
                    pre_capture_seq + 1,
                    hook="post_tool_call",
                    status="error",
                    tool_call_id=tool_call_id,
                ),
            )

        assert latest is not None
        frame = latest.frame
        assert frame.rd["action_count"] == 56
        assert frame.rd["retained_action_count"] == 56
        assert frame.rd["projected_action_count"] == 4
        assert frame.rd["lifecycle_incomplete_count"] == 0
        assert frame.rd["execution_unknown_count"] == 0
        assert frame.rd["effect_unknown_count"] == 56
        assert frame.rd["unresolved_count"] == 56
        assert frame.rd["phase_counts"] == {
            "blocked": 0,
            "dispatched": 0,
            "prepared": 0,
            "proposed": 0,
            "receipt": 0,
            "reconstructed": 56,
        }
        assert frame.rd["outcome_counts"] == {
            "failure": 56,
            "success": 0,
            "unknown": 0,
        }
        assert len(frame.rd["actions"]) == 4
        assert len(frame.a["actions"]) == 4
        assert [item["action_id"] for item in frame.rd["actions"]] == [
            action.action_id for action in engine.state.action_records[-4:]
        ]
        assert all(item["phase"] == "reconstructed" for item in frame.rd["actions"])
        assert all(item["dispatch_observed"] is True for item in frame.a["actions"])
        assert all(
            item["execution_confirmed"] is True for item in frame.a["actions"]
        )
        assert all(item["receipt_status"] == "failure" for item in frame.a["actions"])
        assert frame.omission_counts["rd"]["included"] == 4
        assert frame.omission_counts["rd"]["omitted"] == 52
        incomplete = next(
            candidate
            for candidate in derive_candidates(engine.state)
            if candidate.specialist == "incomplete_action"
        )
        assert incomplete.content["action_ids"] == ()
        assert incomplete.score == 0.15


def test_contradictory_post_source_semantics_commit_raw_plus_one_gap(
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
            tool_observation(
                instance,
                2,
                hook="post_tool_call",
                status="ok",
                error_type="ImpossibleOkError",
            ),
        )

        assert ack.semantic_status == "gap"
        assert ack.derived_event_count == 1
        assert ledger.list_events(engine.brain_id)[-1].event_type == "semantic.gap"
        [action] = engine.state.action_records
        assert action.phase is ActionPhase.PREPARED
        assert action.outcome is None


def test_timeout_preserves_true_host_error_type_through_typed_receipt(
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
            tool_observation(
                instance,
                2,
                hook="post_tool_call",
                status="timeout",
                error_type="TimeoutError",
            ),
        )

        assert ack.semantic_status == "applied"
        [action] = engine.state.action_records
        assert action.phase is ActionPhase.RECONSTRUCTED
        assert action.execution_confirmed is None
        assert action.outcome is None
        assert action.receipt_history[-1].status.value == "unknown"
        assert action.receipt_history[-1].source_status == "timeout"
        assert action.receipt_history[-1].source_error_type == "TimeoutError"


@pytest.mark.parametrize("status", ["error", "timeout", "blocked"])
def test_overlong_host_error_type_commits_raw_plus_gap_and_retries_exactly(
    tmp_path: Path,
    status: str,
) -> None:
    from alice_brain_hermes.runtime.engine import ConsciousEngine
    from alice_brain_hermes.runtime.store import SQLiteLedger

    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    brain_id = engine.brain_id
    record = tool_observation(
        instance,
        2,
        hook="post_tool_call",
        status=status,
        error_type="X" * 161,
    )
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="pre_tool_call"),
        )

        accepted = engine.commit_bridge_record(instance, record)
        duplicate = engine.commit_bridge_record(instance, record)

        assert duplicate.canonical_json() == accepted.canonical_json()
        assert accepted.semantic_status == "gap"
        assert accepted.derived_event_count == 1
        assert accepted.frame.semantic_evidence.semantic_gap_records == 1
        assert [
            event.event_type
            for event in ledger.list_events(brain_id)
            if event.sequence is not None
            and event.sequence >= accepted.raw_event_sequence
        ] == ["hermes.observer.post_tool_call", "semantic.gap"]
        assert ledger.list_events(brain_id)[-1].payload["reason"] == (
            "invalid_post_tool_error_type"
        )
        [action] = engine.state.action_records
        assert action.phase is ActionPhase.PREPARED
        assert action.execution_confirmed is None
        assert action.outcome is None
        assert action.receipt is None
        before = (
            len(ledger.list_events(brain_id)),
            ledger.bridge_stream_state(instance),
            ledger.observability_snapshot(brain_id),
        )

        changed_values = record.model_dump(mode="python")
        changed_values["payload"]["error_type"] = "Y" * 161
        changed = validate_observation(changed_values)
        with pytest.raises(IdempotencyConflictError):
            engine.commit_bridge_record(instance, changed)
        assert (
            len(ledger.list_events(brain_id)),
            ledger.bridge_stream_state(instance),
            ledger.observability_snapshot(brain_id),
        ) == before

    with SQLiteLedger.open(database) as reopened:
        restarted = ConsciousEngine(reopened, brain_id, actor_id=brain_id)
        duplicate_after_restart = restarted.commit_bridge_record(instance, record)

        assert duplicate_after_restart.canonical_json() == accepted.canonical_json()
        [action] = restarted.state.action_records
        assert action.phase is ActionPhase.PREPARED
        assert action.receipt is None


@pytest.mark.parametrize(
    ("status", "error_type", "reason"),
    [
        pytest.param(
            "unexpected",
            None,
            "unknown_post_tool_status",
            id="unknown-status",
        ),
        pytest.param(
            "ok",
            "ImpossibleOkError",
            "ok_with_error_type",
            id="ok-with-error-type",
        ),
        pytest.param(
            "timeout",
            "thread_missing_result",
            "misattributed_thread_missing_result",
            id="timeout-reserved-error-type",
        ),
        pytest.param(
            "cancelled",
            "thread_missing_result",
            "misattributed_thread_missing_result",
            id="cancelled-reserved-error-type",
        ),
        pytest.param(
            "blocked",
            "thread_missing_result",
            "misattributed_thread_missing_result",
            id="blocked-reserved-error-type",
        ),
        pytest.param(
            "error",
            {"kind": "ExternalError"},
            "invalid_post_tool_error_type",
            id="non-string-error-type",
        ),
        pytest.param(
            "error",
            "   ",
            "invalid_post_tool_error_type",
            id="blank-error-type",
        ),
        pytest.param(
            "error",
            "X" * 161,
            "invalid_post_tool_error_type",
            id="overlong-error-type",
        ),
    ],
)
def test_matched_terminal_mapper_gap_closes_span_without_later_action_claims(
    tmp_path: Path,
    status: str,
    error_type: object,
    reason: str,
) -> None:
    from alice_brain_hermes.runtime.engine import ConsciousEngine
    from alice_brain_hermes.runtime.store import SQLiteLedger

    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    brain_id = engine.brain_id
    invalid_record = tool_observation(
        instance,
        2,
        hook="post_tool_call",
        status=status,
        error_type=error_type,
    )
    later_record = tool_observation(
        instance,
        3,
        hook="post_tool_call",
        status="ok",
    )
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="pre_tool_call"),
        )

        accepted = engine.commit_bridge_record(instance, invalid_record)
        duplicate = engine.commit_bridge_record(instance, invalid_record)

        assert duplicate.canonical_json() == accepted.canonical_json()
        assert accepted.semantic_status == "gap"
        assert accepted.derived_event_count == 1
        assert ledger.list_events(brain_id)[-1].payload["reason"] == reason
        [closed_capture_seq] = ledger._connection.execute(
            "SELECT closed_capture_seq FROM hermes_span"
        ).fetchone()
        assert closed_capture_seq == 2
        [action] = engine.state.action_records
        assert action.phase is ActionPhase.PREPARED
        assert ActionPhase.DISPATCHED not in action.phase_history
        assert ActionPhase.RECEIPT not in action.phase_history
        assert action.execution_confirmed is None
        assert action.outcome is None
        assert action.effect_confirmed is None
        assert action.receipt is None
        assert action.receipt_history == ()
        snapshot = ledger.observability_snapshot(brain_id)
        assert snapshot.semantic_records == 2
        assert snapshot.semantic_gap_records == 1
        assert snapshot.trace_complete is False
        assert snapshot.semantic_complete is False
        before_conflict = (
            engine.state,
            tuple(ledger.list_events(brain_id)),
            ledger.bridge_stream_state(instance),
            snapshot,
            tuple(ledger._connection.execute("SELECT * FROM hermes_span").fetchall()),
        )

        changed_values = invalid_record.model_dump(mode="python")
        changed_values["payload"]["result"] = {"body_conflict": True}
        with pytest.raises(IdempotencyConflictError):
            engine.commit_bridge_record(
                instance,
                validate_observation(changed_values),
            )
        assert (
            engine.state,
            tuple(ledger.list_events(brain_id)),
            ledger.bridge_stream_state(instance),
            ledger.observability_snapshot(brain_id),
            tuple(ledger._connection.execute("SELECT * FROM hermes_span").fetchall()),
        ) == before_conflict

    with SQLiteLedger.open(database) as reopened:
        restarted = ConsciousEngine(reopened, brain_id, actor_id=brain_id)
        duplicate_after_restart = restarted.commit_bridge_record(
            instance, invalid_record
        )
        late = restarted.commit_bridge_record(instance, later_record)
        late_duplicate = restarted.commit_bridge_record(instance, later_record)

        assert duplicate_after_restart.canonical_json() == accepted.canonical_json()
        assert late_duplicate.canonical_json() == late.canonical_json()
        assert late.semantic_status == "gap"
        assert late.derived_event_count == 1
        assert reopened.list_events(brain_id)[-1].payload["reason"] == (
            "late_action_state_mismatch"
        )
        assert reopened.bridge_stream_state(instance).next_capture_seq == 4
        [closed_capture_seq] = reopened._connection.execute(
            "SELECT closed_capture_seq FROM hermes_span"
        ).fetchone()
        assert closed_capture_seq == 2
        [action] = restarted.state.action_records
        assert action.phase is ActionPhase.PREPARED
        assert ActionPhase.DISPATCHED not in action.phase_history
        assert ActionPhase.RECEIPT not in action.phase_history
        assert action.execution_confirmed is None
        assert action.outcome is None
        assert action.effect_confirmed is None
        assert action.receipt is None
        assert action.receipt_history == ()
        assert not any(
            event.event_type in {"action.dispatched", "action.receipt"}
            for event in reopened.list_events(brain_id)
        )
        snapshot = reopened.observability_snapshot(brain_id)
        assert snapshot.semantic_records == 3
        assert snapshot.semantic_gap_records == 2
        assert snapshot.trace_complete is False
        assert snapshot.semantic_complete is False

    with SQLiteLedger.open(database) as reopened_again:
        restarted_again = ConsciousEngine(
            reopened_again,
            brain_id,
            actor_id=brain_id,
        )
        duplicate_after_second_restart = restarted_again.commit_bridge_record(
            instance, later_record
        )

        assert duplicate_after_second_restart.canonical_json() == late.canonical_json()
        [action] = restarted_again.state.action_records
        assert action.phase is ActionPhase.PREPARED
        assert ActionPhase.DISPATCHED not in action.phase_history
        assert ActionPhase.RECEIPT not in action.phase_history
        assert action.outcome is None
        assert action.effect_confirmed is None
        [closed_capture_seq] = reopened_again._connection.execute(
            "SELECT closed_capture_seq FROM hermes_span"
        ).fetchone()
        assert closed_capture_seq == 2


def test_late_receipt_matches_latest_closed_occurrence_without_redispatch(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="pre_tool_call"),
        )
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 2, hook="post_tool_call", status="timeout"),
        )

        ack = engine.commit_bridge_record(
            instance,
            tool_observation(instance, 3, hook="post_tool_call", status="ok"),
        )

        assert ack.semantic_status == "applied"
        assert ack.derived_event_count == 2
        assert [
            event.event_type
            for event in ledger.list_events(engine.brain_id)
            if event.sequence is not None and event.sequence >= ack.raw_event_sequence
        ] == [
            "hermes.observer.post_tool_call",
            "action.receipt",
            "action.reconstructed",
        ]
        [action] = engine.state.action_records
        assert action.phase is ActionPhase.RECONSTRUCTED
        assert action.outcome is ActionOutcome.SUCCESS
        assert action.receipt_history[-1].late is True
        assert len(action.receipt_history) == 2
        assert len(action.reconstruction_history) == 2
        assert action.reconstruction_history[-1].after_receipt_event_id == (
            action.receipt_history[-1].event_id
        )

    from alice_brain_hermes.runtime.store import SQLiteLedger

    with SQLiteLedger.open(database) as reopened:
        assert reopened.observability_snapshot(engine.brain_id).semantic_complete


@pytest.mark.parametrize("late_status", ["ok", "error", "timeout", "cancelled"])
def test_late_completion_after_blocked_is_gap_without_receipt_or_redispatch(
    tmp_path: Path,
    late_status: str,
) -> None:
    from alice_brain_hermes.runtime.engine import ConsciousEngine
    from alice_brain_hermes.runtime.store import SQLiteLedger

    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    brain_id = engine.brain_id
    late_record = tool_observation(
        instance,
        3,
        hook="post_tool_call",
        status=late_status,
    )
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="pre_tool_call"),
        )
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 2, hook="post_tool_call", status="blocked"),
        )
        late = engine.commit_bridge_record(instance, late_record)
        duplicate = engine.commit_bridge_record(instance, late_record)

        assert duplicate.canonical_json() == late.canonical_json()
        assert late.semantic_status == "gap"
        assert late.derived_event_count == 1
        assert [
            event.event_type
            for event in ledger.list_events(brain_id)
            if event.sequence is not None and event.sequence >= late.raw_event_sequence
        ] == ["hermes.observer.post_tool_call", "semantic.gap"]
        assert ledger.list_events(brain_id)[-1].payload["reason"] == (
            "late_completion_after_blocked"
        )
        [action] = engine.state.action_records
        assert action.phase is ActionPhase.RECONSTRUCTED
        assert ActionPhase.DISPATCHED not in action.phase_history
        assert ActionPhase.RECEIPT not in action.phase_history
        assert action.execution_confirmed is False
        assert action.outcome is None
        assert action.effect_confirmed is None
        assert len(action.reconstruction_history) == 1
        assert action.reconstruction_history[-1].after_receipt_event_id is None
        assert action.reconstruction_history[-1].payload["assessment"] == (
            "dispatch_prevented"
        )
        snapshot = ledger.observability_snapshot(brain_id)
        assert ledger.bridge_stream_state(instance).next_capture_seq == 4
        [closed_capture_seq] = ledger._connection.execute(
            "SELECT closed_capture_seq FROM hermes_span"
        ).fetchone()
        assert closed_capture_seq == 2
        assert snapshot.semantic_records == 3
        assert snapshot.semantic_gap_records == 1
        assert snapshot.semantic_complete is False
        before = (
            len(ledger.list_events(brain_id)),
            ledger.bridge_stream_state(instance),
            ledger.observability_snapshot(brain_id),
        )

        changed_values = late_record.model_dump(mode="python")
        changed_values["payload"]["result"] = {"changed": True}
        changed = validate_observation(changed_values)
        with pytest.raises(IdempotencyConflictError):
            engine.commit_bridge_record(instance, changed)
        assert (
            len(ledger.list_events(brain_id)),
            ledger.bridge_stream_state(instance),
            ledger.observability_snapshot(brain_id),
        ) == before

    with SQLiteLedger.open(database) as reopened:
        restarted = ConsciousEngine(reopened, brain_id, actor_id=brain_id)
        duplicate_after_restart = restarted.commit_bridge_record(instance, late_record)

        assert duplicate_after_restart.canonical_json() == late.canonical_json()
        [action] = restarted.state.action_records
        assert action.phase is ActionPhase.RECONSTRUCTED
        assert ActionPhase.DISPATCHED not in action.phase_history
        assert ActionPhase.RECEIPT not in action.phase_history


@pytest.mark.parametrize("reconstruct_before_post", [False, True])
@pytest.mark.parametrize("post_status", ["ok", "error", "timeout", "cancelled"])
def test_completion_after_direct_blocked_open_span_is_gap_and_restart_safe(
    tmp_path: Path,
    post_status: str,
    reconstruct_before_post: bool,
) -> None:
    from alice_brain_hermes.core.events import new_event
    from alice_brain_hermes.runtime.engine import ConsciousEngine
    from alice_brain_hermes.runtime.store import SQLiteLedger

    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    brain_id = engine.brain_id
    post_record = tool_observation(
        instance,
        2,
        hook="post_tool_call",
        status=post_status,
    )
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="pre_tool_call"),
        )
        [prepared_action] = engine.state.action_records
        engine.append(
            new_event(
                "action.blocked",
                brain_id,
                brain_id,
                {"action_id": prepared_action.action_id},
                action_id=prepared_action.action_id,
            )
        )
        if reconstruct_before_post:
            engine.append(
                new_event(
                    "action.reconstructed",
                    brain_id,
                    brain_id,
                    {
                        "action_id": prepared_action.action_id,
                        "analysis": "blocked action reconstruction",
                    },
                    action_id=prepared_action.action_id,
                )
            )

        accepted = engine.commit_bridge_record(instance, post_record)
        duplicate = engine.commit_bridge_record(instance, post_record)

        assert duplicate.canonical_json() == accepted.canonical_json()
        assert accepted.semantic_status == "gap"
        assert accepted.derived_event_count == 1
        assert [
            event.event_type
            for event in ledger.list_events(brain_id)
            if event.sequence is not None
            and event.sequence >= accepted.raw_event_sequence
        ] == ["hermes.observer.post_tool_call", "semantic.gap"]
        assert ledger.list_events(brain_id)[-1].payload["reason"] == (
            "late_completion_after_blocked"
        )
        [action] = engine.state.action_records
        assert action.phase is (
            ActionPhase.RECONSTRUCTED
            if reconstruct_before_post
            else ActionPhase.BLOCKED
        )
        assert ActionPhase.DISPATCHED not in action.phase_history
        assert ActionPhase.RECEIPT not in action.phase_history
        assert action.execution_confirmed is False
        assert action.outcome is None
        assert action.effect_confirmed is None
        snapshot = ledger.observability_snapshot(brain_id)
        assert ledger.bridge_stream_state(instance).next_capture_seq == 3
        [closed_capture_seq] = ledger._connection.execute(
            "SELECT closed_capture_seq FROM hermes_span"
        ).fetchone()
        assert closed_capture_seq == 2
        assert snapshot.semantic_records == 2
        assert snapshot.semantic_gap_records == 1
        assert snapshot.semantic_complete is False
        before = (
            len(ledger.list_events(brain_id)),
            ledger.bridge_stream_state(instance),
            snapshot,
        )

        changed_values = post_record.model_dump(mode="python")
        changed_values["payload"]["result"] = {"changed": True}
        changed = validate_observation(changed_values)
        with pytest.raises(IdempotencyConflictError):
            engine.commit_bridge_record(instance, changed)
        assert (
            len(ledger.list_events(brain_id)),
            ledger.bridge_stream_state(instance),
            ledger.observability_snapshot(brain_id),
        ) == before

    with SQLiteLedger.open(database) as reopened:
        restarted = ConsciousEngine(reopened, brain_id, actor_id=brain_id)
        duplicate_after_restart = restarted.commit_bridge_record(instance, post_record)

        assert duplicate_after_restart.canonical_json() == accepted.canonical_json()
        [action] = restarted.state.action_records
        assert action.phase is (
            ActionPhase.RECONSTRUCTED
            if reconstruct_before_post
            else ActionPhase.BLOCKED
        )
        assert ActionPhase.DISPATCHED not in action.phase_history
        assert ActionPhase.RECEIPT not in action.phase_history
        [closed_capture_seq] = reopened._connection.execute(
            "SELECT closed_capture_seq FROM hermes_span"
        ).fetchone()
        assert closed_capture_seq == 2


def test_late_conflict_keeps_canonical_outcome_and_projects_typed_disposition(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="pre_tool_call"),
        )
        engine.commit_bridge_record(
            instance,
            tool_observation(instance, 2, hook="post_tool_call", status="ok"),
        )

        late_record = tool_observation(
            instance, 3, hook="post_tool_call", status="error"
        )
        late = engine.commit_bridge_record(instance, late_record)
        event_count = len(ledger.list_events(engine.brain_id))
        duplicate = engine.commit_bridge_record(instance, late_record)

        [action] = engine.state.action_records
        assert duplicate.canonical_json() == late.canonical_json()
        assert len(ledger.list_events(engine.brain_id)) == event_count
        assert action.phase is ActionPhase.RECONSTRUCTED
        assert action.outcome is ActionOutcome.SUCCESS
        assert action.receipt is not None
        assert action.receipt["status"] == "success"
        assert action.receipt_history[-1].disposition.value == "conflict"
        assert len(action.receipt_history) == 2
        assert len(action.reconstruction_history) == 2
        assert action.reconstruction_history[-1].after_receipt_event_id == (
            action.receipt_history[-1].event_id
        )
        [projected] = late.frame.a["actions"]
        assert projected["receipt_status"] == "success"
        assert projected["outcome"] == "success"
        assert projected["latest_receipt_status"] == "failure"
        assert projected["latest_receipt_disposition"] == "conflict"
        assert projected["receipt_conflict_count"] == 1


def test_all_open_span_capacity_commits_raw_plus_explicit_gap_without_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_module, "MAX_HERMES_SPANS_PER_STREAM", 2)
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        for capture_seq in (1, 2):
            ack = engine.commit_bridge_record(
                instance,
                tool_observation(
                    instance,
                    capture_seq,
                    hook="pre_tool_call",
                    tool_call_id=f"tool-{capture_seq}",
                ),
            )
            assert ack.semantic_status == "applied"

        capped = engine.commit_bridge_record(
            instance,
            tool_observation(instance, 3, hook="pre_tool_call", tool_call_id="tool-3"),
        )

        assert capped.semantic_status == "gap"
        assert capped.semantic_complete is False
        assert capped.derived_event_count == 1
        assert (
            ledger._connection.execute(
                "SELECT COUNT(*) FROM hermes_span WHERE closed_capture_seq IS NULL"
            ).fetchone()[0]
            == 2
        )
        last = ledger.list_events(engine.brain_id)[-1]
        assert last.event_type == "semantic.gap"
        assert last.payload["reason"] == "span_capacity_all_open"


def test_real_all_open_api_span_cap_survives_restart_without_open_eviction(
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.runtime.store import (
        MAX_HERMES_SPANS_PER_STREAM,
        SQLiteLedger,
    )

    _, base_context, payload, _ = next(
        case for case in HOOK_CASES if case[0] == "pre_api_request"
    )
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    with ledger:
        for capture_seq in range(1, MAX_HERMES_SPANS_PER_STREAM + 1):
            context = {
                **base_context,
                "api_request_id": f"api-{capture_seq}",
            }
            ack = engine.commit_bridge_record(
                instance,
                generic_observation(
                    instance,
                    capture_seq,
                    "pre_api_request",
                    context,
                    payload,
                ),
            )
            assert ack.semantic_status == "applied"
        capped_seq = MAX_HERMES_SPANS_PER_STREAM + 1
        capped = engine.commit_bridge_record(
            instance,
            generic_observation(
                instance,
                capped_seq,
                "pre_api_request",
                {**base_context, "api_request_id": f"api-{capped_seq}"},
                payload,
            ),
        )
        assert capped.semantic_status == "gap"
        assert (
            ledger._connection.execute(
                "SELECT COUNT(*) FROM hermes_span WHERE closed_capture_seq IS NULL"
            ).fetchone()[0]
            == MAX_HERMES_SPANS_PER_STREAM
        )

    with SQLiteLedger.open(database) as reopened:
        assert (
            reopened._connection.execute(
                "SELECT COUNT(*) FROM hermes_span WHERE closed_capture_seq IS NULL"
            ).fetchone()[0]
            == MAX_HERMES_SPANS_PER_STREAM
        )


def test_semantic_domain_capacity_gap_does_not_evict_closed_late_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alice_brain_hermes.core.events import new_event
    from alice_brain_hermes.core.limits import MAX_ACTION_RECORDS
    from alice_brain_hermes.runtime.engine import ConsciousEngine
    from alice_brain_hermes.runtime.store import SQLiteLedger

    monkeypatch.setattr(store_module, "MAX_HERMES_SPANS_PER_STREAM", 1)
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    brain_id = engine.brain_id
    old_pre = tool_observation(
        instance,
        1,
        hook="pre_tool_call",
        tool_call_id="old-tool",
    )
    old_post = tool_observation(
        instance,
        2,
        hook="post_tool_call",
        tool_call_id="old-tool",
        status="ok",
    )
    rejected_pre = tool_observation(
        instance,
        3,
        hook="pre_tool_call",
        tool_call_id="new-tool",
    )
    with ledger:
        engine.commit_bridge_record(instance, old_pre)
        engine.commit_bridge_record(instance, old_post)
        for index in range(MAX_ACTION_RECORDS):
            action_id = f"capacity-{index}"
            engine.append(
                new_event(
                    "action.proposed",
                    brain_id,
                    brain_id,
                    {"action_id": action_id, "intent": {}},
                    action_id=action_id,
                )
            )
        before_span = tuple(
            tuple(row)
            for row in ledger._connection.execute(
                "SELECT external_id, occurrence_capture_seq, closed_capture_seq "
                "FROM hermes_span"
            )
        )
        assert before_span == (("old-tool", 1, 2),)

        gap = engine.commit_bridge_record(instance, rejected_pre)
        duplicate = engine.commit_bridge_record(instance, rejected_pre)

        assert gap.semantic_status == "gap"
        assert duplicate == gap
        assert (
            ledger.list_events(brain_id, after_sequence=gap.raw_event_sequence)[
                0
            ].payload["reason"]
            == "semantic_domain_capacity"
        )
        assert (
            tuple(
                tuple(row)
                for row in ledger._connection.execute(
                    "SELECT external_id, occurrence_capture_seq, closed_capture_seq "
                    "FROM hermes_span"
                )
            )
            == before_span
        )
        before_conflict = (
            engine.state,
            ledger.observability_snapshot(brain_id),
            ledger.bridge_stream_state(instance),
            before_span,
        )
        changed_values = rejected_pre.model_dump(mode="python")
        changed_values["payload"]["args"] = {"changed": True}
        with pytest.raises(IdempotencyConflictError):
            engine.commit_bridge_record(
                instance,
                validate_observation(changed_values),
            )
        assert (
            engine.state,
            ledger.observability_snapshot(brain_id),
            ledger.bridge_stream_state(instance),
            tuple(
                tuple(row)
                for row in ledger._connection.execute(
                    "SELECT external_id, occurrence_capture_seq, closed_capture_seq "
                    "FROM hermes_span"
                )
            ),
        ) == before_conflict

    with SQLiteLedger.open(database) as reopened:
        restarted = ConsciousEngine(reopened, brain_id, actor_id=brain_id)
        assert restarted.commit_bridge_record(instance, rejected_pre) == gap
        late = restarted.commit_bridge_record(
            instance,
            tool_observation(
                instance,
                4,
                hook="post_tool_call",
                tool_call_id="old-tool",
                status="error",
            ),
        )
        assert late.semantic_status == "gap"
        assert late.derived_event_count == 1
        assert (
            reopened.list_events(brain_id, after_sequence=late.raw_event_sequence)[
                0
            ].payload["reason"]
            == "late_action_unavailable"
        )
        assert all(
            action.intent.get("capture_seq") != 1
            for action in restarted.state.action_records
        )
        assert (
            tuple(
                tuple(row)
                for row in reopened._connection.execute(
                    "SELECT external_id, occurrence_capture_seq, closed_capture_seq "
                    "FROM hermes_span"
                )
            )
            == before_span
        )
        assert reopened.observability_snapshot(brain_id).semantic_gap_records == 2


def test_successful_span_open_evicts_only_the_oldest_closed_victim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alice_brain_hermes.runtime.store import SQLiteLedger

    monkeypatch.setattr(store_module, "MAX_HERMES_SPANS_PER_STREAM", 2)
    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    with ledger:
        for capture_seq, tool_id in ((1, "oldest"), (3, "newer")):
            engine.commit_bridge_record(
                instance,
                tool_observation(
                    instance,
                    capture_seq,
                    hook="pre_tool_call",
                    tool_call_id=tool_id,
                ),
            )
            engine.commit_bridge_record(
                instance,
                tool_observation(
                    instance,
                    capture_seq + 1,
                    hook="post_tool_call",
                    tool_call_id=tool_id,
                ),
            )

        opened = engine.commit_bridge_record(
            instance,
            tool_observation(
                instance,
                5,
                hook="pre_tool_call",
                tool_call_id="replacement",
            ),
        )
        assert opened.semantic_status == "applied"
        assert [
            tuple(row)
            for row in ledger._connection.execute(
                "SELECT external_id, closed_capture_seq FROM hermes_span "
                "ORDER BY occurrence_capture_seq"
            )
        ] == [("newer", 4), ("replacement", None)]

    with SQLiteLedger.open(database) as reopened:
        assert [
            tuple(row)
            for row in reopened._connection.execute(
                "SELECT external_id, closed_capture_seq FROM hermes_span "
                "ORDER BY occurrence_capture_seq"
            )
        ] == [("newer", 4), ("replacement", None)]


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
        assert ack.frame.aggregate_semantic_complete is False
        assert ack.frame.semantic_evidence.semantic_gap_records == 1
        assert engine.state.trace_complete is False
        assert [event.event_type for event in ledger.list_events(engine.brain_id)] == [
            "hermes.observer.post_tool_call",
            "semantic.gap",
        ]


def test_unmatched_terminal_does_not_close_an_unrelated_open_span(
    tmp_path: Path,
) -> None:
    ledger, engine, instance = make_engine(tmp_path)
    with ledger:
        engine.commit_bridge_record(
            instance,
            tool_observation(
                instance,
                1,
                hook="pre_tool_call",
                tool_call_id="expected-tool",
            ),
        )

        unmatched = engine.commit_bridge_record(
            instance,
            tool_observation(
                instance,
                2,
                hook="post_tool_call",
                tool_call_id="unrelated-tool",
            ),
        )

        assert unmatched.semantic_status == "gap"
        [span] = ledger._connection.execute(
            "SELECT external_id, closed_capture_seq FROM hermes_span"
        ).fetchall()
        assert tuple(span) == ("expected-tool", None)

        matched = engine.commit_bridge_record(
            instance,
            tool_observation(
                instance,
                3,
                hook="post_tool_call",
                tool_call_id="expected-tool",
            ),
        )

        assert matched.semantic_status == "applied"
        [span] = ledger._connection.execute(
            "SELECT external_id, closed_capture_seq FROM hermes_span"
        ).fetchall()
        assert tuple(span) == ("expected-tool", 3)
        [action] = engine.state.action_records
        assert action.phase is ActionPhase.RECONSTRUCTED
        assert action.execution_confirmed is True
        assert action.outcome is ActionOutcome.SUCCESS


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
        assert (
            ledger._connection.execute("SELECT COUNT(*) FROM hermes_span").fetchone()[0]
            == 1
        )


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
        assert (
            ledger._connection.execute("SELECT COUNT(*) FROM hermes_span").fetchone()[0]
            == 1
        )
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
        assert (
            ledger._connection.execute("SELECT COUNT(*) FROM hermes_span").fetchone()[0]
            == 0
        )
        assert ledger.observability_snapshot(engine.brain_id) == before


def test_known_derived_domain_capacity_commits_raw_plus_semantic_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alice_brain_hermes.errors import DomainCapacityError

    ledger, engine, instance = make_engine(tmp_path)
    original_reduce = store_module.reduce_state

    def capacity_on_proposal(state, event):
        if event.event_type == "action.proposed":
            raise DomainCapacityError("fixture active action capacity")
        return original_reduce(state, event)

    monkeypatch.setattr(store_module, "reduce_state", capacity_on_proposal)
    with ledger:
        ack = engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="pre_tool_call"),
        )

        assert ack.semantic_status == "gap"
        assert ack.derived_event_count == 1
        assert [event.event_type for event in ledger.list_events(engine.brain_id)] == [
            "hermes.observer.pre_tool_call",
            "semantic.gap",
        ]
        assert (
            ledger._connection.execute("SELECT COUNT(*) FROM hermes_span").fetchone()[0]
            == 0
        )


def test_real_action_capacity_gap_is_reproducible_by_startup_audit(
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.core.events import new_event
    from alice_brain_hermes.core.limits import MAX_ACTION_RECORDS
    from alice_brain_hermes.runtime.store import SQLiteLedger

    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    with ledger:
        for index in range(MAX_ACTION_RECORDS):
            action_id = f"active-{index}"
            engine.append(
                new_event(
                    "action.proposed",
                    engine.brain_id,
                    engine.actor_id,
                    {"action_id": action_id, "intent": {}},
                    action_id=action_id,
                )
            )
        ack = engine.commit_bridge_record(
            instance,
            tool_observation(instance, 1, hook="pre_tool_call"),
        )
        assert ack.semantic_status == "gap"
        assert ack.derived_event_count == 1

    with SQLiteLedger.open(database) as reopened:
        snapshot = reopened.observability_snapshot(engine.brain_id)
        assert snapshot.semantic_complete is False
        assert snapshot.semantic_gap_records == 1


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
        assert before.sqlite_schema_version == 7
        assert before.trace_complete is False
        assert before.semantic_complete is False
        assert before.dropped_events == 0
        assert before.disconnected_open_bridges == 1
        assert aggregate == before.model_copy(update={"brain_id": None})

    from alice_brain_hermes.runtime.store import SQLiteLedger

    with SQLiteLedger.open(database) as reopened:
        assert reopened.observability_snapshot(engine.brain_id) == before


def test_non_bridge_gap_observability_is_truthful_without_replay_after_restart(
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.core.events import new_event
    from alice_brain_hermes.ids import new_id
    from alice_brain_hermes.runtime.store import SQLiteLedger

    database = tmp_path / "non-bridge-gap.db"
    brain_id = new_id()
    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(brain_id)
        ledger.append(
            new_event(
                "semantic.gap",
                brain_id,
                brain_id,
                {"reason": "external_semantic_gap", "trace_complete": False},
            )
        )
        before = ledger.observability_snapshot(brain_id)
        assert before.trace_complete is False
        assert before.semantic_complete is False
        assert before.semantic_gap_records == 1

    with SQLiteLedger.open(database) as reopened:
        assert reopened.observability_snapshot(brain_id) == before


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
        payload = (
            event.payload.model_dump()
            if hasattr(event.payload, "model_dump")
            else dict(event.payload)
        )
        intent = dict(payload["intent"])
        intent["args_sha256"] = "c" * 64
        payload["intent"] = intent
        changed = event.model_copy(
            update={"payload": FrozenJsonDict(payload)}
        ).revalidated()
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


def test_startup_rejects_fractional_real_observability_counter_even_if_int_cast_matches(
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.runtime.store import SQLiteLedger

    ledger, engine, instance = make_engine(tmp_path)
    database = ledger.path
    dropped = 10
    with ledger:
        engine.commit_bridge_record(
            instance,
            BridgeGapV1(
                bridge_instance_id=instance,
                first_capture_seq=1,
                last_capture_seq=dropped,
                dropped_count=dropped,
                cause_counts={"queue_full": dropped},
            ),
        )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE brain_observability SET dropped_events = "
            "CAST(dropped_events AS REAL) + 0.5"
        )
        [value, storage_type] = connection.execute(
            "SELECT dropped_events, typeof(dropped_events) FROM brain_observability"
        ).fetchone()
        assert int(value) == dropped
        assert storage_type == "real"

    with pytest.raises(SchemaVersionError, match="bridge or profile"):
        SQLiteLedger.open(database)


def test_startup_rejects_recanonicalized_false_frame_semantic_evidence(
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
        [encoded] = connection.execute("SELECT ack_json FROM bridge_record").fetchone()
        ack = BridgeCommitAckV2.model_validate_json(encoded)
        frame_values = ack.frame.model_dump(mode="python")
        frame_values["aggregate_semantic_complete"] = False
        frame_values["semantic_evidence"]["semantic_gap_records"] = 1
        ack_values = ack.model_dump(mode="python")
        ack_values["frame"] = ack.frame.__class__.model_validate(
            frame_values, strict=True
        )
        changed = BridgeCommitAckV2.model_validate(ack_values, strict=True)
        connection.execute(
            "UPDATE bridge_record SET ack_json = ?",
            (changed.canonical_json(),),
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
