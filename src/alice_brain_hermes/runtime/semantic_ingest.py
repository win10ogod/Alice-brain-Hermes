"""Pure, bounded Hermes raw-to-semantic ingestion plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from alice_brain_hermes.core.action import MAX_ACTION_SOURCE_ERROR_TYPE_LENGTH
from alice_brain_hermes.core.events import EventEnvelope, new_event, thaw_json
from alice_brain_hermes.ids import validate_id
from alice_brain_hermes.protocol.models import (
    HOOK_EVENT_TYPES,
    BridgeGapV1,
    BridgeRecordV1,
    BridgeStreamState,
    HermesObservationV1,
    SemanticStatus,
)

MAX_DERIVED_EVENTS_PER_RECORD = 8
SpanKind = Literal["tool", "api"]


def _canonical_json(value: object) -> str:
    return json.dumps(
        thaw_json(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _derived_id(raw_event_id: str, role: str) -> str:
    digest = bytearray(
        hashlib.sha256(f"{raw_event_id}\x00{role}".encode()).digest()[:16]
    )
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return str(UUID(bytes=bytes(digest)))


def span_context_fingerprint(record: HermesObservationV1) -> str:
    """Fingerprint exact hook correlation context without retaining its values."""
    if not isinstance(record, HermesObservationV1):
        raise TypeError("span context requires one Hermes observation")
    return _json_fingerprint(record.context.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class HermesSpan:
    bridge_instance_id: str
    span_kind: SpanKind
    external_id: str
    occurrence_capture_seq: int
    context_fingerprint: str
    action_id: str | None = None
    closed_capture_seq: int | None = None

    def __post_init__(self) -> None:
        validate_id(self.bridge_instance_id)
        if self.span_kind not in {"tool", "api"}:
            raise ValueError("Hermes span kind must be tool or api")
        if (
            not isinstance(self.external_id, str)
            or not 1 <= len(self.external_id) <= 512
        ):
            raise ValueError("Hermes span external ID must be non-blank and bounded")
        if (
            isinstance(self.occurrence_capture_seq, bool)
            or not isinstance(self.occurrence_capture_seq, int)
            or self.occurrence_capture_seq < 1
        ):
            raise ValueError("Hermes span occurrence must be a capture sequence")
        if (
            not isinstance(self.context_fingerprint, str)
            or len(self.context_fingerprint) != 64
            or any(item not in "0123456789abcdef" for item in self.context_fingerprint)
        ):
            raise ValueError("Hermes span context fingerprint must be SHA-256")
        if self.action_id is not None and (
            not isinstance(self.action_id, str)
            or not self.action_id.strip()
            or len(self.action_id) > 512
        ):
            raise ValueError("Hermes span action ID must be non-blank and bounded")
        if self.closed_capture_seq is not None and (
            isinstance(self.closed_capture_seq, bool)
            or not isinstance(self.closed_capture_seq, int)
            or self.closed_capture_seq < self.occurrence_capture_seq
        ):
            raise ValueError("Hermes span close must follow its occurrence")

    def canonical_data(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "bridge_instance_id": self.bridge_instance_id,
            "closed_capture_seq": self.closed_capture_seq,
            "context_fingerprint": self.context_fingerprint,
            "external_id": self.external_id,
            "occurrence_capture_seq": self.occurrence_capture_seq,
            "span_kind": self.span_kind,
        }


def match_hermes_span(
    record: HermesObservationV1,
    candidates: tuple[HermesSpan, ...],
) -> tuple[HermesSpan | None, str | None]:
    """Correlate a completion against one bounded occurrence cache."""
    if record.hook == "post_tool_call":
        span_kind: SpanKind = "tool"
        external_id = record.context.tool_call_id
    elif record.hook in {"post_api_request", "api_request_error"}:
        span_kind = "api"
        external_id = record.context.api_request_id
    else:
        return None, None
    context_fingerprint = span_context_fingerprint(record)
    eligible = tuple(
        span
        for span in candidates
        if span.bridge_instance_id == record.bridge_instance_id
        and span.span_kind == span_kind
        and span.external_id == external_id
        and span.context_fingerprint == context_fingerprint
        and span.occurrence_capture_seq < record.capture_seq
    )
    open_spans = tuple(span for span in eligible if span.closed_capture_seq is None)
    closed_spans = tuple(
        span
        for span in eligible
        if span.closed_capture_seq is not None
        and span.closed_capture_seq < record.capture_seq
    )
    if len(open_spans) > 1:
        return None, "ambiguous_open_span"
    if len(open_spans) == 1:
        if closed_spans:
            return None, "ambiguous_open_closed_span"
        return open_spans[0], None
    if not closed_spans:
        return None, None
    return max(
        closed_spans,
        key=lambda span: (
            span.closed_capture_seq or 0,
            span.occurrence_capture_seq,
        ),
    ), None


@dataclass(frozen=True, slots=True)
class SemanticPlan:
    semantic_status: SemanticStatus
    semantic_complete: bool
    derived_events: tuple[EventEnvelope, ...]
    span_open: HermesSpan | None = None
    span_close: HermesSpan | None = None

    def __post_init__(self) -> None:
        if len(self.derived_events) > MAX_DERIVED_EVENTS_PER_RECORD:
            raise ValueError("semantic plan exceeds the derived event limit")
        expected_complete = self.semantic_status in {"applied", "not_applicable"}
        if self.semantic_complete is not expected_complete:
            raise ValueError("semantic plan completeness does not match status")
        if self.semantic_status == "applied" and not self.derived_events:
            raise ValueError("applied semantic plan requires derived evidence")
        if self.semantic_status == "gap" and len(self.derived_events) not in {0, 1}:
            raise ValueError("semantic gap plan contains zero or one gap event")
        if self.semantic_status in {"not_applicable", "legacy_raw_only"} and (
            self.derived_events
        ):
            raise ValueError("raw-only semantic statuses cannot carry derived events")

    def fingerprint(self) -> str:
        value = {
            "derived_events": [
                event.canonical_data(exclude_sequence=True)
                for event in self.derived_events
            ],
            "semantic_complete": self.semantic_complete,
            "semantic_status": self.semantic_status,
            "span_close": (
                None if self.span_close is None else self.span_close.canonical_data()
            ),
            "span_open": (
                None if self.span_open is None else self.span_open.canonical_data()
            ),
        }
        return _json_fingerprint(value)


def _observation_provenance(record: HermesObservationV1) -> dict[str, str | None]:
    return {
        "session_id": getattr(record.context, "session_id", None),
        "turn_id": getattr(record.context, "turn_id", None),
        "correlation_id": getattr(record.context, "api_request_id", None),
    }


def build_raw_event(stream: BridgeStreamState, record: BridgeRecordV1) -> EventEnvelope:
    if record.bridge_instance_id != stream.bridge_instance_id:
        raise ValueError("bridge record does not match semantic stream")
    if isinstance(record, HermesObservationV1):
        return new_event(
            HOOK_EVENT_TYPES[record.hook],
            stream.brain_id,
            stream.server_actor_id,
            record.model_dump(mode="json"),
            wall_time=record.captured_at,
            monotonic_ns=record.captured_monotonic_ns,
            adapter_id=stream.server_adapter_id,
            **_observation_provenance(record),
        )
    if not isinstance(record, BridgeGapV1):
        raise TypeError("semantic ingestion requires a typed bridge record")
    return new_event(
        "trace.gap",
        stream.brain_id,
        stream.server_actor_id,
        {
            **record.model_dump(mode="json"),
            "exact": True,
            "trace_complete": False,
        },
        adapter_id=stream.server_adapter_id,
    )


def _derived_event(
    stream: BridgeStreamState,
    record: HermesObservationV1,
    raw_event: EventEnvelope,
    *,
    role: str,
    event_type: str,
    payload: dict[str, object],
    action_id: str | None = None,
    causation_id: str | None = None,
) -> EventEnvelope:
    return new_event(
        event_type,
        stream.brain_id,
        stream.server_actor_id,
        payload,
        event_id=_derived_id(raw_event.event_id, role),
        wall_time=raw_event.wall_time,
        monotonic_ns=raw_event.monotonic_ns,
        adapter_id=stream.server_adapter_id,
        action_id=action_id,
        causation_id=raw_event.event_id if causation_id is None else causation_id,
        **_observation_provenance(record),
    )


def _action_id(raw_event: EventEnvelope, capture_seq: int) -> str:
    digest = hashlib.sha256(
        f"{raw_event.event_id}\x00tool\x00{capture_seq}".encode("ascii")
    ).hexdigest()
    return f"hermes-action-{digest}"


def _pre_tool_plan(
    stream: BridgeStreamState,
    record: HermesObservationV1,
    raw_event: EventEnvelope,
) -> SemanticPlan:
    action_id = _action_id(raw_event, record.capture_seq)
    branch_id = f"hermes-branch-{hashlib.sha256(action_id.encode('ascii')).hexdigest()}"
    args_sha256 = _json_fingerprint(record.payload.args)
    middleware_sha256 = _json_fingerprint(record.payload.middleware_trace)
    intent = {
        "args_sha256": args_sha256,
        "capture_seq": record.capture_seq,
        "kind": "hermes.tool_call",
        "middleware_trace_sha256": middleware_sha256,
        "raw_event_id": raw_event.event_id,
        "tool_name": record.payload.tool_name,
    }
    proposed = _derived_event(
        stream,
        record,
        raw_event,
        role="pc.action_proposed",
        event_type="action.proposed",
        payload={"action_id": action_id, "intent": intent},
        action_id=action_id,
    )
    pc = _derived_event(
        stream,
        record,
        raw_event,
        role="pc.control_sampled",
        event_type="personality.control.sampled",
        payload={
            "action_id": action_id,
            "algorithm_version": "hermes-semantic-v1",
            "args_sha256": args_sha256,
            "sample": "commit_predecessor_personality",
        },
        action_id=action_id,
        causation_id=proposed.event_id,
    )
    energy_request = _derived_event(
        stream,
        record,
        raw_event,
        role="energy.requested",
        event_type="action.energy_requested",
        payload={
            "action_id": action_id,
            "assessment_source": "hermes_host_llm",
            "prompt_version": "alice-energy-v1",
        },
        action_id=action_id,
        causation_id=pc.event_id,
    )
    simulated = _derived_event(
        stream,
        record,
        raw_event,
        role="thought.simulated",
        event_type="simulation.created",
        payload={
            "algorithm_version": "hermes-semantic-v1",
            "branch_id": branch_id,
            "cognition_mode": "event",
            "config_version": "hermes-semantic-v1",
            "content": {
                "args_sha256": args_sha256,
                "kind": "hermes.tool_counterfactual",
                "tool_name": record.payload.tool_name,
            },
            "expected_consequences": [
                {"kind": "execution_success", "requires_confirmation": True},
                {"kind": "execution_failure", "requires_confirmation": True},
            ],
            "proposition_id": branch_id,
            "source_ids": [raw_event.event_id],
            "stance": "simulate",
            "uncertainty": 0.5,
        },
        action_id=action_id,
        causation_id=energy_request.event_id,
    )
    prepared = _derived_event(
        stream,
        record,
        raw_event,
        role="decision.prepared",
        event_type="action.prepared",
        payload={"action_id": action_id, "branch_id": branch_id},
        action_id=action_id,
        causation_id=simulated.event_id,
    )
    span = HermesSpan(
        bridge_instance_id=record.bridge_instance_id,
        span_kind="tool",
        external_id=record.context.tool_call_id,
        occurrence_capture_seq=record.capture_seq,
        context_fingerprint=span_context_fingerprint(record),
        action_id=action_id,
    )
    return SemanticPlan(
        semantic_status="applied",
        semantic_complete=True,
        derived_events=(proposed, pc, energy_request, simulated, prepared),
        span_open=span,
    )


def _semantic_gap(
    stream: BridgeStreamState,
    record: HermesObservationV1,
    raw_event: EventEnvelope,
    *,
    reason: str,
    span_close: HermesSpan | None = None,
) -> SemanticPlan:
    event = _derived_event(
        stream,
        record,
        raw_event,
        role=f"semantic_gap.{reason}",
        event_type="semantic.gap",
        payload={
            "capture_seq": record.capture_seq,
            "context_sha256": span_context_fingerprint(record),
            "hook": record.hook,
            "raw_payload_sha256": _json_fingerprint(
                record.payload.model_dump(mode="json")
            ),
            "reason": reason,
            "trace_complete": False,
        },
    )
    return SemanticPlan(
        semantic_status="gap",
        semantic_complete=False,
        derived_events=(event,),
        span_close=span_close,
    )


def _validate_matched_span(
    record: HermesObservationV1,
    span: HermesSpan,
    *,
    span_kind: SpanKind,
    external_id: str,
) -> None:
    if (
        span.bridge_instance_id != record.bridge_instance_id
        or span.span_kind != span_kind
        or span.external_id != external_id
        or span.context_fingerprint != span_context_fingerprint(record)
    ):
        raise ValueError("matched Hermes span does not bind the observation context")


def _post_tool_plan(
    stream: BridgeStreamState,
    record: HermesObservationV1,
    raw_event: EventEnvelope,
    matched_span: HermesSpan | None,
) -> SemanticPlan:
    if matched_span is None:
        return _semantic_gap(stream, record, raw_event, reason="unmatched_post_tool")
    _validate_matched_span(
        record,
        matched_span,
        span_kind="tool",
        external_id=record.context.tool_call_id,
    )
    if matched_span.action_id is None:
        raise ValueError("matched tool span is missing its action identity")
    action_id = matched_span.action_id

    def matched_terminal_gap(reason: str) -> SemanticPlan:
        return _semantic_gap(
            stream,
            record,
            raw_event,
            reason=reason,
            span_close=(
                matched_span if matched_span.closed_capture_seq is None else None
            ),
        )

    raw_status = record.payload.status
    raw_error_type = record.payload.error_type
    if raw_error_type is not None and (
        type(raw_error_type) is not str
        or not raw_error_type.strip()
        or len(raw_error_type) > MAX_ACTION_SOURCE_ERROR_TYPE_LENGTH
    ):
        return matched_terminal_gap("invalid_post_tool_error_type")
    if raw_status == "ok" and raw_error_type is not None:
        return matched_terminal_gap("ok_with_error_type")
    if raw_status != "error" and raw_error_type == "thread_missing_result":
        return matched_terminal_gap("misattributed_thread_missing_result")
    if raw_status == "ok":
        receipt_status, execution, outcome = "success", True, "success"
    elif raw_status == "error" and raw_error_type == "thread_missing_result":
        receipt_status, execution, outcome = "unknown", None, None
    elif raw_status == "error":
        receipt_status, execution, outcome = "failure", True, "failure"
    elif raw_status in {"timeout", "cancelled"}:
        receipt_status, execution, outcome = "unknown", None, None
    elif raw_status == "blocked":
        receipt_status, execution, outcome = "blocked", False, None
    else:
        return matched_terminal_gap("unknown_post_tool_status")
    result_sha256 = _json_fingerprint(record.payload.result)
    error_type_sha256 = _json_fingerprint(record.payload.error_type)
    error_message_sha256 = _json_fingerprint(record.payload.error_message)
    late = matched_span.closed_capture_seq is not None
    if late and raw_status == "blocked":
        return matched_terminal_gap("late_blocked_status")
    evidence = {
        "action_id": action_id,
        "duration_ms": record.payload.duration_ms,
        "effect_confirmed": None,
        "error_message_sha256": error_message_sha256,
        "error_type_sha256": error_type_sha256,
        "execution_confirmed": execution,
        "late": late,
        "outcome": outcome,
        "result_sha256": result_sha256,
        "source_error_type": raw_error_type,
        "source_status": raw_status,
        "status": receipt_status,
    }
    if raw_status == "blocked":
        blocked = _derived_event(
            stream,
            record,
            raw_event,
            role="action.blocked",
            event_type="action.blocked",
            payload=evidence,
            action_id=action_id,
        )
        events = (blocked,)
    elif late:
        receipt = _derived_event(
            stream,
            record,
            raw_event,
            role="action.receipt.late",
            event_type="action.receipt",
            payload=evidence,
            action_id=action_id,
        )
        events = (receipt,)
    else:
        dispatched = _derived_event(
            stream,
            record,
            raw_event,
            role="action.dispatched",
            event_type="action.dispatched",
            payload={"action_id": action_id},
            action_id=action_id,
        )
        receipt = _derived_event(
            stream,
            record,
            raw_event,
            role="action.receipt",
            event_type="action.receipt",
            payload=evidence,
            action_id=action_id,
            causation_id=dispatched.event_id,
        )
        events = (dispatched, receipt)
    terminal = events[-1]
    assessment = (
        "dispatch_prevented"
        if raw_status == "blocked"
        else (
            "execution_succeeded"
            if outcome == "success"
            else "execution_failed" if outcome == "failure" else "execution_unknown"
        )
    )
    reconstructed = _derived_event(
        stream,
        record,
        raw_event,
        role="action.reconstructed",
        event_type="action.reconstructed",
        payload={"action_id": action_id, "assessment": assessment},
        action_id=action_id,
        causation_id=terminal.event_id,
    )
    return SemanticPlan(
        semantic_status="applied",
        semantic_complete=True,
        derived_events=(*events, reconstructed),
        span_close=None if late else matched_span,
    )


_ATTRIBUTED_EVENT_TYPES = {
    "on_session_start": "semantic.session.attributed",
    "on_session_end": "semantic.session.attributed",
    "on_session_finalize": "semantic.session.attributed",
    "on_session_reset": "semantic.session.attributed",
    "pre_llm_call": "semantic.llm.attributed",
    "post_llm_call": "semantic.llm.attributed",
    "pre_api_request": "semantic.api.attributed",
    "post_api_request": "semantic.api.attributed",
    "api_request_error": "semantic.api.attributed",
    "pre_approval_request": "semantic.approval.attributed",
    "post_approval_response": "semantic.approval.attributed",
    "subagent_start": "semantic.subagent.attributed",
    "subagent_stop": "semantic.subagent.attributed",
    "pre_verify": "semantic.verification.attributed",
}


def _attributed_plan(
    stream: BridgeStreamState,
    record: HermesObservationV1,
    raw_event: EventEnvelope,
    matched_span: HermesSpan | None,
) -> SemanticPlan:
    event_type = _ATTRIBUTED_EVENT_TYPES.get(record.hook)
    if event_type is None:
        return _semantic_gap(
            stream, record, raw_event, reason="unsupported_semantic_hook"
        )
    span_open: HermesSpan | None = None
    span_close: HermesSpan | None = None
    occurrence: int | None = None
    if record.hook == "pre_api_request":
        span_open = HermesSpan(
            bridge_instance_id=record.bridge_instance_id,
            span_kind="api",
            external_id=record.context.api_request_id,
            occurrence_capture_seq=record.capture_seq,
            context_fingerprint=span_context_fingerprint(record),
        )
        occurrence = record.capture_seq
    elif record.hook in {"post_api_request", "api_request_error"}:
        if matched_span is None:
            return _semantic_gap(
                stream, record, raw_event, reason="unmatched_api_completion"
            )
        _validate_matched_span(
            record,
            matched_span,
            span_kind="api",
            external_id=record.context.api_request_id,
        )
        span_close = (
            None if matched_span.closed_capture_seq is not None else matched_span
        )
        occurrence = matched_span.occurrence_capture_seq
    preceding: tuple[EventEnvelope, ...] = ()
    if record.hook == "subagent_start":
        child_actor_id = _derived_id(
            record.bridge_instance_id,
            f"subagent.child.{record.context.child_session_id}",
        )
        registration = _derived_event(
            stream,
            record,
            raw_event,
            role="subagent.actor_registered",
            event_type="identity.actor_registered",
            payload={
                "actor_id": child_actor_id,
                "attributes": {
                    "child_session_id_sha256": _json_fingerprint(
                        record.context.child_session_id
                    ),
                    "child_subagent_id_sha256": _json_fingerprint(
                        record.payload.child_subagent_id
                    ),
                    "source": "hermes_observer_v1",
                },
                "kind": "external_agent",
                "parent_actor_id": stream.brain_id,
            },
        )
        preceding = (registration,)
    event = _derived_event(
        stream,
        record,
        raw_event,
        role=f"attributed.{record.hook}",
        event_type=event_type,
        payload={
            "capture_seq": record.capture_seq,
            "context_sha256": span_context_fingerprint(record),
            "hook": record.hook,
            "late": bool(
                matched_span is not None and matched_span.closed_capture_seq is not None
            ),
            "occurrence_capture_seq": occurrence,
            "raw_payload_sha256": _json_fingerprint(
                record.payload.model_dump(mode="json")
            ),
            "source": "hermes_observer_v1",
        },
        causation_id=(raw_event.event_id if not preceding else preceding[-1].event_id),
    )
    return SemanticPlan(
        semantic_status="applied",
        semantic_complete=True,
        derived_events=(*preceding, event),
        span_open=span_open,
        span_close=span_close,
    )


def build_semantic_plan(
    stream: BridgeStreamState,
    record: BridgeRecordV1,
    *,
    raw_event: EventEnvelope,
    matched_span: HermesSpan | None = None,
    forced_gap_reason: str | None = None,
) -> SemanticPlan:
    if record.bridge_instance_id != stream.bridge_instance_id:
        raise ValueError("bridge record does not match semantic stream")
    if (
        raw_event.brain_id != stream.brain_id
        or raw_event.actor_id != stream.server_actor_id
        or raw_event.adapter_id != stream.server_adapter_id
        or raw_event.sequence is not None
    ):
        raise ValueError("raw event does not bind the semantic stream")
    if isinstance(record, BridgeGapV1):
        if raw_event.event_type != "trace.gap":
            raise ValueError("gap record requires one raw trace gap")
        return SemanticPlan(
            semantic_status="gap",
            semantic_complete=False,
            derived_events=(),
        )
    if not isinstance(record, HermesObservationV1):
        raise TypeError("semantic ingestion requires a typed bridge record")
    if raw_event.event_type != HOOK_EVENT_TYPES[record.hook]:
        raise ValueError("raw event does not represent the observation hook")
    if forced_gap_reason is not None:
        if not forced_gap_reason.strip() or len(forced_gap_reason) > 160:
            raise ValueError("forced semantic gap reason must be non-blank and bounded")
        return _semantic_gap(
            stream,
            record,
            raw_event,
            reason=forced_gap_reason,
            span_close=(
                matched_span
                if matched_span is not None and matched_span.closed_capture_seq is None
                else None
            ),
        )
    if record.hook == "pre_tool_call":
        if matched_span is not None:
            raise ValueError("pre-tool observation cannot close a span")
        return _pre_tool_plan(stream, record, raw_event)
    if record.hook == "post_tool_call":
        return _post_tool_plan(stream, record, raw_event, matched_span)
    return _attributed_plan(stream, record, raw_event, matched_span)


__all__ = [
    "MAX_DERIVED_EVENTS_PER_RECORD",
    "HermesSpan",
    "SemanticPlan",
    "build_raw_event",
    "build_semantic_plan",
    "match_hermes_span",
    "span_context_fingerprint",
]
