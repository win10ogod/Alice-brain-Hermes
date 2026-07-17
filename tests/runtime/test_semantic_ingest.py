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
    match_hermes_span,
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
    error_type: object = None,
    tool_call_id: str = "tool-reused",
) -> HermesObservationV1:
    payload: dict[str, object] = {
        "tool_name": "shell",
        "args": {"command": "TOP SECRET RAW ARGUMENT"},
        "middleware_trace": {"private": "TRACE SECRET"},
        "extensions": {},
    }
    if hook == "post_tool_call":
        if error_type is None and status == "error":
            error_type = "SyntheticError"
        payload.update(
            {
                "result": {"stdout": "TOP SECRET RAW RESULT"},
                "duration_ms": 12.5,
                "status": status,
                "error_type": error_type,
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


def test_pre_tool_builds_host_energy_request_without_copying_raw_values() -> None:
    instance = new_id()
    source = stream(instance)
    record = tool_observation(instance, 7, hook="pre_tool_call")
    raw = build_raw_event(source, record)

    first = build_semantic_plan(source, record, raw_event=raw)
    second = build_semantic_plan(source, record, raw_event=raw)

    assert [event.event_type for event in first.derived_events] == [
        "action.proposed",
        "personality.control.sampled",
        "action.energy_requested",
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
    energy = first.derived_events[2]
    assert energy.payload == {
        "action_id": energy.action_id,
        "assessment_source": "hermes_host_llm",
        "prompt_version": "alice-energy-v1",
    }


@pytest.mark.parametrize(
    ("status", "event_types", "execution", "outcome", "assessment"),
    [
        (
            "ok",
            ["action.dispatched", "action.receipt", "action.reconstructed"],
            True,
            "success",
            "execution_succeeded",
        ),
        (
            "error",
            ["action.dispatched", "action.receipt", "action.reconstructed"],
            True,
            "failure",
            "execution_failed",
        ),
        (
            "timeout",
            ["action.dispatched", "action.receipt", "action.reconstructed"],
            None,
            None,
            "execution_unknown",
        ),
        (
            "cancelled",
            ["action.dispatched", "action.receipt", "action.reconstructed"],
            None,
            None,
            "execution_unknown",
        ),
        (
            "blocked",
            ["action.blocked", "action.reconstructed"],
            False,
            None,
            "dispatch_prevented",
        ),
    ],
)
def test_matched_post_tool_maps_execution_separately_from_outcome(
    status: str,
    event_types: list[str],
    execution: bool | None,
    outcome: str | None,
    assessment: str,
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
    repeated = build_semantic_plan(
        source, record, raw_event=raw, matched_span=matched
    )

    assert [event.event_type for event in plan.derived_events] == event_types
    terminal = plan.derived_events[-2]
    reconstruction = plan.derived_events[-1]
    assert terminal.payload["execution_confirmed"] is execution
    assert terminal.payload["outcome"] == outcome
    assert terminal.payload.get("effect_confirmed") is None
    assert terminal.payload["source_status"] == status
    assert terminal.payload["source_error_type"] == (
        "SyntheticError" if status == "error" else None
    )
    assert dict(reconstruction.payload) == {
        "action_id": "hermes-action-7",
        "assessment": assessment,
    }
    assert reconstruction.action_id == "hermes-action-7"
    assert reconstruction.causation_id == terminal.event_id
    assert plan.fingerprint() == repeated.fingerprint()
    assert [event.event_id for event in plan.derived_events] == [
        event.event_id for event in repeated.derived_events
    ]
    assert plan.span_close == matched
    encoded = "\n".join(event.canonical_json() for event in plan.derived_events)
    assert "TOP SECRET RAW RESULT" not in encoded
    assert "TOP SECRET ERROR" not in encoded
    assert "result_sha256" in encoded


def test_thread_missing_result_is_real_error_shape_with_unknown_execution() -> None:
    instance = new_id()
    source = stream(instance)
    record = tool_observation(
        instance,
        9,
        hook="post_tool_call",
        status="error",
        error_type="thread_missing_result",
    )
    matched = HermesSpan(
        bridge_instance_id=instance,
        span_kind="tool",
        external_id="tool-reused",
        occurrence_capture_seq=7,
        context_fingerprint=span_context_fingerprint(record),
        action_id="hermes-action-7",
    )

    plan = build_semantic_plan(
        source,
        record,
        raw_event=build_raw_event(source, record),
        matched_span=matched,
    )

    receipt = plan.derived_events[-2]
    assert receipt.payload["status"] == "unknown"
    assert receipt.payload["execution_confirmed"] is None
    assert receipt.payload["outcome"] is None
    assert receipt.payload["source_status"] == "error"
    assert receipt.payload["source_error_type"] == "thread_missing_result"


@pytest.mark.parametrize("status", ["success", "failure", "OK", " error "])
def test_non_host_post_tool_status_is_one_semantic_gap(status: str) -> None:
    instance = new_id()
    source = stream(instance)
    record = tool_observation(
        instance, 9, hook="post_tool_call", status=status, error_type=None
    )
    matched = HermesSpan(
        bridge_instance_id=instance,
        span_kind="tool",
        external_id="tool-reused",
        occurrence_capture_seq=7,
        context_fingerprint=span_context_fingerprint(record),
        action_id="hermes-action-7",
    )

    plan = build_semantic_plan(
        source,
        record,
        raw_event=build_raw_event(source, record),
        matched_span=matched,
    )

    assert plan.semantic_status == "gap"
    assert [event.event_type for event in plan.derived_events] == ["semantic.gap"]


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        ("ok", "ImpossibleOkError"),
        ("timeout", "thread_missing_result"),
        ("cancelled", "thread_missing_result"),
        ("blocked", "thread_missing_result"),
    ],
)
def test_contradictory_source_error_type_is_one_semantic_gap(
    status: str, error_type: str
) -> None:
    instance = new_id()
    source = stream(instance)
    record = tool_observation(
        instance,
        9,
        hook="post_tool_call",
        status=status,
        error_type=error_type,
    )
    matched = HermesSpan(
        bridge_instance_id=instance,
        span_kind="tool",
        external_id="tool-reused",
        occurrence_capture_seq=7,
        context_fingerprint=span_context_fingerprint(record),
        action_id="hermes-action-7",
    )

    plan = build_semantic_plan(
        source,
        record,
        raw_event=build_raw_event(source, record),
        matched_span=matched,
    )

    assert plan.semantic_status == "gap"
    assert [event.event_type for event in plan.derived_events] == ["semantic.gap"]


@pytest.mark.parametrize(
    ("status", "error_type", "event_type"),
    [
        ("timeout", "TimeoutError", "action.receipt"),
        ("cancelled", "CancelledError", "action.receipt"),
        ("blocked", "PolicyBlocked", "action.blocked"),
    ],
)
def test_true_host_non_error_status_preserves_nonreserved_error_type(
    status: str, error_type: str, event_type: str
) -> None:
    instance = new_id()
    source = stream(instance)
    record = tool_observation(
        instance,
        9,
        hook="post_tool_call",
        status=status,
        error_type=error_type,
    )
    matched = HermesSpan(
        bridge_instance_id=instance,
        span_kind="tool",
        external_id="tool-reused",
        occurrence_capture_seq=7,
        context_fingerprint=span_context_fingerprint(record),
        action_id="hermes-action-7",
    )

    plan = build_semantic_plan(
        source,
        record,
        raw_event=build_raw_event(source, record),
        matched_span=matched,
    )

    terminal = plan.derived_events[-2]
    assert terminal.event_type == event_type
    assert terminal.payload["source_status"] == status
    assert terminal.payload["source_error_type"] == error_type


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
    assert action.phase is ActionPhase.RECONSTRUCTED
    assert ActionPhase.BLOCKED in action.phase_history
    assert ActionPhase.DISPATCHED not in action.phase_history
    assert action.execution_confirmed is False
    assert action.outcome is None
    assert action.effect_confirmed is None
    assert action.reconstruction_history[-1].after_receipt_event_id is None
    assert action.reconstruction_history[-1].payload["assessment"] == (
        "dispatch_prevented"
    )


def test_late_tool_receipt_uses_closed_occurrence_without_redispatch() -> None:
    instance = new_id()
    source = stream(instance)
    pre = tool_observation(instance, 1, hook="pre_tool_call")
    pre_raw = build_raw_event(source, pre)
    pre_plan = build_semantic_plan(source, pre, raw_event=pre_raw)
    assert pre_plan.span_open is not None
    first_post = tool_observation(instance, 2, hook="post_tool_call", status="timeout")
    first_raw = build_raw_event(source, first_post)
    first_plan = build_semantic_plan(
        source,
        first_post,
        raw_event=first_raw,
        matched_span=pre_plan.span_open,
    )
    closed = HermesSpan(
        **{**pre_plan.span_open.canonical_data(), "closed_capture_seq": 2}
    )
    late_post = tool_observation(instance, 3, hook="post_tool_call", status="ok")
    late_raw = build_raw_event(source, late_post)

    matched, reason = match_hermes_span(late_post, (closed,))
    late_plan = build_semantic_plan(
        source,
        late_post,
        raw_event=late_raw,
        matched_span=matched,
        forced_gap_reason=reason,
    )

    assert [event.event_type for event in late_plan.derived_events] == [
        "action.receipt",
        "action.reconstructed",
    ]
    assert late_plan.derived_events[0].payload["late"] is True
    assert late_plan.derived_events[1].causation_id == (
        late_plan.derived_events[0].event_id
    )
    assert late_plan.span_close is None
    state = reduce_many(
        BrainState.genesis(source.brain_id),
        (
            pre_raw,
            *pre_plan.derived_events,
            first_raw,
            *first_plan.derived_events,
            late_raw,
            *late_plan.derived_events,
        ),
    )
    [action] = state.action_records
    assert action.phase is ActionPhase.RECONSTRUCTED
    assert action.outcome is not None and action.outcome.value == "success"
    assert len(action.receipt_history) == 2
    assert len(action.reconstruction_history) == 2
    assert action.reconstruction_history[-1].after_receipt_event_id == (
        action.receipt_history[-1].event_id
    )


def test_two_indistinguishable_open_occurrences_are_explicitly_ambiguous() -> None:
    instance = new_id()
    source = stream(instance)
    completion = tool_observation(instance, 9, hook="post_tool_call")
    context_fingerprint = span_context_fingerprint(completion)
    candidates = tuple(
        HermesSpan(
            bridge_instance_id=instance,
            span_kind="tool",
            external_id="tool-reused",
            occurrence_capture_seq=occurrence,
            context_fingerprint=context_fingerprint,
            action_id=f"action-{occurrence}",
        )
        for occurrence in (3, 8)
    )

    matched, reason = match_hermes_span(completion, candidates)
    plan = build_semantic_plan(
        source,
        completion,
        raw_event=build_raw_event(source, completion),
        matched_span=matched,
        forced_gap_reason=reason,
    )

    assert matched is None
    assert reason == "ambiguous_open_span"
    assert plan.semantic_status == "gap"
    assert [event.event_type for event in plan.derived_events] == ["semantic.gap"]


def test_open_and_closed_reused_occurrences_are_explicitly_ambiguous() -> None:
    instance = new_id()
    source = stream(instance)
    completion = tool_observation(instance, 9, hook="post_tool_call")
    context_fingerprint = span_context_fingerprint(completion)
    closed = HermesSpan(
        bridge_instance_id=instance,
        span_kind="tool",
        external_id="tool-reused",
        occurrence_capture_seq=2,
        context_fingerprint=context_fingerprint,
        action_id="action-closed",
        closed_capture_seq=3,
    )
    opened = HermesSpan(
        bridge_instance_id=instance,
        span_kind="tool",
        external_id="tool-reused",
        occurrence_capture_seq=8,
        context_fingerprint=context_fingerprint,
        action_id="action-open",
    )

    matched, reason = match_hermes_span(completion, (closed, opened))
    plan = build_semantic_plan(
        source,
        completion,
        raw_event=build_raw_event(source, completion),
        matched_span=matched,
        forced_gap_reason=reason,
    )

    assert matched is None
    assert reason == "ambiguous_open_closed_span"
    assert plan.semantic_status == "gap"
    assert [event.event_type for event in plan.derived_events] == ["semantic.gap"]


@pytest.mark.parametrize(
    ("hook", "context", "payload", "_required"),
    [case for case in HOOK_CASES if case[0] not in {"pre_tool_call", "post_tool_call"}],
    ids=[
        case[0]
        for case in HOOK_CASES
        if case[0] not in {"pre_tool_call", "post_tool_call"}
    ],
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
    expected_count = 2 if hook == "subagent_start" else 1
    assert len(plan.derived_events) == expected_count
    assert plan.derived_events[-1].event_type.startswith("semantic.")
    assert plan.derived_events[-1].event_type != "observation.recorded"
    assert plan.derived_events[-1].payload["raw_payload_sha256"]
    if hook == "subagent_start":
        registration = plan.derived_events[0]
        assert registration.event_type == "identity.actor_registered"
        assert registration.payload["kind"] == "external_agent"
        assert registration.payload["parent_actor_id"] == source.brain_id
        assert registration.payload["actor_id"] != source.brain_id
        assert "child_goal" not in registration.canonical_json()


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
