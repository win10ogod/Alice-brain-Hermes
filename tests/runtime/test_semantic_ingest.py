from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alice_brain_hermes.core.action import ActionPhase
from alice_brain_hermes.core.reducer import reduce_many
from alice_brain_hermes.core.state import BrainState
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.models import (
    BridgeGapV1,
    BridgeStreamState,
    CoverageV1,
    HermesObservationV1,
    validate_observation,
)
from alice_brain_hermes.runtime.semantic_ingest import (
    HermesSpan,
    build_raw_event,
    build_semantic_plan,
    span_context_fingerprint,
)
from tests.protocol.test_models import HOOK_CASES


def stream(instance: str) -> BridgeStreamState:
    now = datetime.now(UTC)
    brain_id = new_id()
    return BridgeStreamState(
        bridge_instance_id=instance,
        brain_id=brain_id,
        server_actor_id=brain_id,
        server_adapter_id="alice-brain-hermes-observer-v1",
        next_capture_seq=1,
        status="open",
        connected_nonce="nonce",
        last_seen=now,
    )


def tool_observation(
    instance: str,
    capture_seq: int,
    *,
    hook: str,
    status: str = "ok",
    tool_call_id: str = "tool-reused",
) -> HermesObservationV1:
    payload: dict[str, object] = {
        "tool_name": "shell",
        "args": {"command": "TOP SECRET RAW ARGUMENT"},
        "middleware_trace": {"private": "TRACE SECRET"},
        "extensions": {},
    }
    if hook == "post_tool_call":
        payload.update(
            {
                "result": {"stdout": "TOP SECRET RAW RESULT"},
                "duration_ms": 12.5,
                "status": status,
                "error_type": "SyntheticError" if status == "error" else None,
                "error_message": "TOP SECRET ERROR" if status == "error" else None,
            }
        )
    return validate_observation(
        {
            "bridge_instance_id": instance,
            "capture_seq": capture_seq,
            "captured_at": datetime.now(UTC),
            "captured_monotonic_ns": capture_seq,
            "hook": hook,
            "context": {
                "session_id": "session",
                "task_id": "task",
                "turn_id": "turn",
                "api_request_id": "api-reused",
                "tool_call_id": tool_call_id,
            },
            "payload": payload,
            "coverage": CoverageV1(
                policy_version="copy-v1",
                capture_coverage="host_sanitized",
            ),
        }
    )


def generic_observation(
    instance: str,
    capture_seq: int,
    hook: str,
    context: dict[str, object],
    payload: dict[str, object],
) -> HermesObservationV1:
    return validate_observation(
        {
            "bridge_instance_id": instance,
            "capture_seq": capture_seq,
            "captured_at": datetime.now(UTC),
            "captured_monotonic_ns": capture_seq,
            "hook": hook,
            "context": context,
            "payload": {**payload, "extensions": {}},
            "coverage": CoverageV1(
                policy_version="copy-v1",
                capture_coverage="host_sanitized",
            ),
        }
    )


def test_pre_tool_builds_complete_pc_e_st_rd_chain_without_copying_raw_values() -> None:
    instance = new_id()
    source = stream(instance)
    record = tool_observation(instance, 7, hook="pre_tool_call")
    raw = build_raw_event(source, record)

    first = build_semantic_plan(source, record, raw_event=raw)
    second = build_semantic_plan(source, record, raw_event=raw)

    assert [event.event_type for event in first.derived_events] == [
        "action.proposed",
        "personality.control.sampled",
        "action.energy_assessed",
        "simulation.created",
        "action.prepared",
    ]
    assert first.semantic_status == "applied"
    assert first.semantic_complete is True
    assert first.fingerprint() == second.fingerprint()
    assert [event.event_id for event in first.derived_events] == [
        event.event_id for event in second.derived_events
    ]
    assert first.span_open is not None
    assert first.span_open.occurrence_capture_seq == 7
    encoded = "\n".join(event.canonical_json() for event in first.derived_events)
    for secret in ("TOP SECRET RAW ARGUMENT", "TRACE SECRET"):
        assert secret not in encoded
    assert "args_sha256" in encoded
    assert "middleware_trace_sha256" in encoded


@pytest.mark.parametrize(
    ("status", "event_types", "execution", "outcome"),
    [
        ("ok", ["action.dispatched", "action.receipt"], True, "success"),
        ("error", ["action.dispatched", "action.receipt"], True, "failure"),
        ("timeout", ["action.dispatched", "action.receipt"], None, None),
        ("cancelled", ["action.dispatched", "action.receipt"], None, None),
        ("error+thread_missing_result", ["action.dispatched", "action.receipt"], None, None),
        ("blocked", ["action.blocked"], False, None),
    ],
)
def test_matched_post_tool_maps_execution_separately_from_outcome(
    status: str,
    event_types: list[str],
    execution: bool | None,
    outcome: str | None,
) -> None:
    instance = new_id()
    source = stream(instance)
    record = tool_observation(instance, 9, hook="post_tool_call", status=status)
    matched = HermesSpan(
        bridge_instance_id=instance,
        span_kind="tool",
        external_id="tool-reused",
        occurrence_capture_seq=7,
        context_fingerprint=span_context_fingerprint(record),
        action_id="hermes-action-7",
    )
    raw = build_raw_event(source, record)

    plan = build_semantic_plan(source, record, raw_event=raw, matched_span=matched)

    assert [event.event_type for event in plan.derived_events] == event_types
    terminal = plan.derived_events[-1]
    assert terminal.payload["execution_confirmed"] is execution
    assert terminal.payload["outcome"] == outcome
    assert terminal.payload.get("effect_confirmed") is None
    assert plan.span_close == matched
    encoded = "\n".join(event.canonical_json() for event in plan.derived_events)
    assert "TOP SECRET RAW RESULT" not in encoded
    assert "TOP SECRET ERROR" not in encoded
    assert "result_sha256" in encoded


def test_unmatched_post_tool_is_raw_plus_semantic_gap_not_fabricated_action() -> None:
    instance = new_id()
    source = stream(instance)
    record = tool_observation(instance, 1, hook="post_tool_call")
    raw = build_raw_event(source, record)

    plan = build_semantic_plan(source, record, raw_event=raw)

    assert plan.semantic_status == "gap"
    assert plan.semantic_complete is False
    assert [event.event_type for event in plan.derived_events] == ["semantic.gap"]
    assert plan.derived_events[0].payload["reason"] == "unmatched_post_tool"
    assert "action_id" not in plan.derived_events[0].payload
    state = reduce_many(
        BrainState.genesis(source.brain_id),
        (raw, *plan.derived_events),
    )
    assert state.trace_complete is False


def test_pre_and_matched_blocked_post_reduce_to_non_executed_action() -> None:
    instance = new_id()
    source = stream(instance)
    pre = tool_observation(instance, 1, hook="pre_tool_call")
    pre_raw = build_raw_event(source, pre)
    pre_plan = build_semantic_plan(source, pre, raw_event=pre_raw)
    assert pre_plan.span_open is not None
    post = tool_observation(instance, 2, hook="post_tool_call", status="blocked")
    post_raw = build_raw_event(source, post)
    post_plan = build_semantic_plan(
        source,
        post,
        raw_event=post_raw,
        matched_span=pre_plan.span_open,
    )

    state = reduce_many(
        BrainState.genesis(source.brain_id),
        (
            pre_raw,
            *pre_plan.derived_events,
            post_raw,
            *post_plan.derived_events,
        ),
    )

    [action] = state.action_records
    assert action.phase is ActionPhase.BLOCKED
    assert action.execution_confirmed is False
    assert action.outcome is None


@pytest.mark.parametrize(
    ("hook", "context", "payload", "_required"),
    [case for case in HOOK_CASES if case[0] not in {"pre_tool_call", "post_tool_call"}],
    ids=[case[0] for case in HOOK_CASES if case[0] not in {"pre_tool_call", "post_tool_call"}],
)
def test_every_non_tool_hook_has_one_attributed_semantic_event(
    hook: str,
    context: dict[str, object],
    payload: dict[str, object],
    _required: str,
) -> None:
    instance = new_id()
    source = stream(instance)
    record = generic_observation(instance, 1, hook, context, payload)
    raw = build_raw_event(source, record)
    matched = (
        HermesSpan(
            bridge_instance_id=instance,
            span_kind="api",
            external_id=record.context.api_request_id,
            occurrence_capture_seq=1,
            context_fingerprint=span_context_fingerprint(record),
        )
        if hook in {"post_api_request", "api_request_error"}
        else None
    )

    plan = build_semantic_plan(
        source,
        record,
        raw_event=raw,
        matched_span=matched,
    )

    assert plan.semantic_status == "applied"
    assert plan.semantic_complete is True
    assert len(plan.derived_events) == 1
    assert plan.derived_events[0].event_type.startswith("semantic.")
    assert plan.derived_events[0].event_type != "observation.recorded"
    assert plan.derived_events[0].payload["raw_payload_sha256"]


def test_exact_bridge_gap_is_explicitly_incomplete_without_derived_backfill() -> None:
    instance = new_id()
    source = stream(instance)
    record = BridgeGapV1(
        bridge_instance_id=instance,
        first_capture_seq=1,
        last_capture_seq=3,
        dropped_count=3,
        cause_counts={"queue_full": 3},
    )
    raw = build_raw_event(source, record)

    plan = build_semantic_plan(source, record, raw_event=raw)

    assert raw.event_type == "trace.gap"
    assert plan.semantic_status == "gap"
    assert plan.semantic_complete is False
    assert plan.derived_events == ()


def test_reused_external_id_uses_capture_occurrence_not_one_ambiguous_span() -> None:
    instance = new_id()
    source = stream(instance)
    first_record = tool_observation(instance, 3, hook="pre_tool_call")
    second_record = tool_observation(instance, 8, hook="pre_tool_call")

    first = build_semantic_plan(
        source,
        first_record,
        raw_event=build_raw_event(source, first_record),
    )
    second = build_semantic_plan(
        source,
        second_record,
        raw_event=build_raw_event(source, second_record),
    )

    assert first.span_open is not None and second.span_open is not None
    assert first.span_open.external_id == second.span_open.external_id
    assert first.span_open.occurrence_capture_seq == 3
    assert second.span_open.occurrence_capture_seq == 8
    assert first.span_open.action_id != second.span_open.action_id
