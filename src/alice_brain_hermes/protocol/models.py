"""Strict versioned wire and bridge record models."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Any, ClassVar, Literal, TypeAlias, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
    field_validator,
    model_validator,
)

from alice_brain_hermes.core.events import FrozenJsonDict, thaw_json
from alice_brain_hermes.ids import validate_id

PROTOCOL_VERSION = 2
SERVER_ADAPTER_ID = "alice-brain-hermes-observer-v1"
OBSERVER_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
GAP_SCHEMA_VERSION = 1
FRAME_SCHEMA_VERSION = 3
MAX_PROTOCOL_BYTES = 4_194_304
MAX_PROTOCOL_DEPTH = 128
MAX_PROTOCOL_NODES = 100_000
TASK6_MAX_DETACHED_RECORD_BYTES = 262_144
MAX_BRIDGE_RECORD_BYTES = TASK6_MAX_DETACHED_RECORD_BYTES
MAX_BRIDGE_RECORD_DEPTH = 8
MAX_BRIDGE_RECORD_NODES = 2_048
MAX_BRIDGE_CONTAINER_ITEMS = 128
MAX_BRIDGE_KEY_BYTES = 256
MAX_BRIDGE_STRING_BYTES = 16_384
MAX_COVERAGE_CHANNELS = 64
MIN_BRIDGE_INTEGER = -(2**63)
MAX_BRIDGE_INTEGER = 2**63 - 1
# ``bridge_stream.next_capture_seq`` stores the successor cursor in SQLite.
# Reserve the final signed-int64 value so every accepted capture has one exact,
# persistable successor instead of failing after wire validation.
MAX_CAPTURE_SEQUENCE = MAX_BRIDGE_INTEGER - 1

GapCause: TypeAlias = Literal[
    "queue_full",
    "detach_failed",
    "serialization_failed",
    "shutdown",
    "fork_reset",
    "shape_failed",
    "validation_failed",
    "invalid_source_schema",
    "record_too_large",
    "record_too_deep",
    "record_too_complex",
    "unsupported_value",
    "capture_failed",
    "callback_internal",
    "transport_failed",
    "daemon_unavailable",
    "backpressure",
    "other",
]
GAP_CAUSES = frozenset(get_args(GapCause))


def _maximum_bridge_commit_envelope_bytes() -> int:
    envelope = {
        "jsonrpc": "2.0",
        "id": "\x00" * 128,
        "method": "bridge.commit",
        "params": {"binding": "0" * 36, "record": {}},
        "auth": "0" * 64,
    }
    encoded = json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return len(encoded) - len(b"{}")


MAX_BRIDGE_COMMIT_ENVELOPE_BYTES = _maximum_bridge_commit_envelope_bytes()

HermesHook: TypeAlias = Literal[
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "pre_tool_call",
    "post_tool_call",
    "pre_approval_request",
    "post_approval_response",
    "subagent_start",
    "subagent_stop",
    "pre_verify",
]

HOOK_EVENT_TYPES: dict[str, str] = {
    hook: f"hermes.observer.{hook}"
    for hook in (
        "on_session_start",
        "on_session_end",
        "on_session_finalize",
        "on_session_reset",
        "pre_llm_call",
        "post_llm_call",
        "pre_api_request",
        "post_api_request",
        "api_request_error",
        "pre_tool_call",
        "post_tool_call",
        "pre_approval_request",
        "post_approval_response",
        "subagent_start",
        "subagent_stop",
        "pre_verify",
    )
}


class _StrictModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


IdentifierText: TypeAlias = Annotated[str, Field(min_length=1, max_length=512)]
# Hermes observer identifiers are host correlation values, not Alice UUIDs.  The
# 0.18.x call sites intentionally use ``""`` or ``None`` when a surface has no
# value (for example TUI finalization and pre-tool calls outside an API turn).
# Keeping that absence is more accurate than fabricating an identifier or
# turning a legal host observation into a trace gap.
HostIdentifierText: TypeAlias = Annotated[str, Field(max_length=512)] | None
WireText: TypeAlias = Annotated[str, Field(max_length=16_384)]
NonNegativeInteger: TypeAlias = Annotated[int, Field(ge=0)]
NonNegativeNumber: TypeAlias = Annotated[int | float, Field(ge=0)]


def _freeze_observation_json(value: Any) -> Any:
    return FrozenJsonDict({"value": value})["value"]


FrozenJsonValue: TypeAlias = Annotated[
    Any,
    BeforeValidator(_freeze_observation_json),
    PlainSerializer(thaw_json, return_type=Any),
]


class ObservationContextV1(_StrictModel):
    """Base for hook-specific, exact identifier contexts."""


class SessionContextV1(ObservationContextV1):
    session_id: HostIdentifierText


class SessionTurnContextV1(SessionContextV1):
    task_id: HostIdentifierText
    turn_id: HostIdentifierText


class SessionEndContextV1(SessionContextV1):
    task_id: HostIdentifierText = None
    turn_id: HostIdentifierText = None
    api_request_id: HostIdentifierText = None


class PreLlmContextV1(SessionTurnContextV1):
    sender_id: HostIdentifierText


class ApiContextV1(SessionTurnContextV1):
    api_request_id: HostIdentifierText


class ToolContextV1(ApiContextV1):
    tool_call_id: HostIdentifierText


class ApprovalContextV1(ObservationContextV1):
    turn_id: HostIdentifierText
    tool_call_id: HostIdentifierText


class SubagentContextV1(ObservationContextV1):
    parent_session_id: HostIdentifierText
    child_session_id: HostIdentifierText


class CoverageV1(_StrictModel):
    """Explicit capture coverage; omissions are never represented as complete."""

    policy_version: str = Field(min_length=1, max_length=160)
    capture_coverage: Literal[
        "full", "host_sanitized", "redacted", "partial", "unobserved"
    ]
    redacted_paths: int = Field(default=0, ge=0)
    truncated_paths: int = Field(default=0, ge=0)
    unsupported_paths: int = Field(default=0, ge=0)
    omitted_nodes: int = Field(default=0, ge=0)
    channels: FrozenJsonDict = Field(default_factory=FrozenJsonDict)

    @field_validator("channels")
    @classmethod
    def _bounded_channels(cls, value: FrozenJsonDict) -> FrozenJsonDict:
        if len(value) > MAX_COVERAGE_CHANNELS:
            raise ValueError("coverage channel samples exceed the fixed limit")
        return value

    @model_validator(mode="after")
    def _full_has_no_recorded_omissions(self) -> CoverageV1:
        if self.capture_coverage == "full" and any(
            (
                self.redacted_paths,
                self.truncated_paths,
                self.unsupported_paths,
                self.omitted_nodes,
            )
        ):
            raise ValueError("full capture coverage cannot contain recorded omissions")
        return self


def _bounded_tree(value: Any) -> None:
    inspected = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    pending: list[tuple[Any, int]] = [(inspected, 1)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        if isinstance(item, BaseModel):
            item = item.model_dump(mode="json")
        nodes += 1
        if nodes > MAX_BRIDGE_RECORD_NODES:
            raise ValueError("bridge record exceeds the node limit")
        if depth > MAX_BRIDGE_RECORD_DEPTH:
            raise ValueError("bridge record exceeds the depth limit")
        if isinstance(item, Mapping):
            if len(item) > MAX_BRIDGE_CONTAINER_ITEMS:
                raise ValueError("bridge record mapping exceeds the item limit")
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("bridge record keys must be strings")
                nodes += 1
                if nodes > MAX_BRIDGE_RECORD_NODES:
                    raise ValueError("bridge record exceeds the node limit")
                if len(key.encode("utf-8", errors="strict")) > MAX_BRIDGE_KEY_BYTES:
                    raise ValueError("bridge record key exceeds the byte limit")
                pending.append((child, depth + 1))
        elif isinstance(item, (list, tuple)):
            if len(item) > MAX_BRIDGE_CONTAINER_ITEMS:
                raise ValueError("bridge record sequence exceeds the item limit")
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            if len(item.encode("utf-8", errors="strict")) > MAX_BRIDGE_STRING_BYTES:
                raise ValueError("bridge record string exceeds the byte limit")
        elif isinstance(item, datetime):
            if len(item.isoformat().encode("utf-8")) > MAX_BRIDGE_STRING_BYTES:
                raise ValueError("bridge record timestamp exceeds the byte limit")
        elif isinstance(item, bool) or item is None:
            continue
        elif isinstance(item, int):
            if not MIN_BRIDGE_INTEGER <= item <= MAX_BRIDGE_INTEGER:
                raise ValueError("bridge record integer exceeds int64")
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("bridge record numbers must be finite")
        else:
            raise ValueError("bridge record contains a non-JSON value")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("bridge record JSON contains duplicate keys")
        value[key] = item
    return value


def _bounded_json_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > 19:
        raise ValueError("bridge record integer exceeds int64")
    parsed = int(value)
    if not MIN_BRIDGE_INTEGER <= parsed <= MAX_BRIDGE_INTEGER:
        raise ValueError("bridge record integer exceeds int64")
    return parsed


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("bridge record numbers must be finite")
    return parsed


def _reject_json_constant(_value: str) -> None:
    raise ValueError("bridge record numbers must be finite")


def _prevalidate_record_json(value: str | bytes) -> bytes:
    if type(value) is str:
        if len(value) > MAX_BRIDGE_RECORD_BYTES:
            raise ValueError("bridge record JSON exceeds the byte limit")
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ValueError("bridge record JSON is not valid UTF-8") from error
    elif type(value) is bytes:
        encoded = value
    else:
        raise TypeError("bridge record JSON must be exact str or bytes")
    if len(encoded) > MAX_BRIDGE_RECORD_BYTES:
        raise ValueError("bridge record JSON exceeds the byte limit")
    try:
        decoded = encoded.decode("utf-8", errors="strict")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_unique_json_object,
            parse_int=_bounded_json_integer,
            parse_float=_finite_json_float,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, OverflowError, RecursionError) as error:
        raise ValueError("bridge record JSON is invalid") from error
    try:
        _bounded_tree(parsed)
    except UnicodeError as error:
        raise ValueError("bridge record JSON is not valid UTF-8") from error
    return encoded


class ObservationPayloadV1(_StrictModel):
    extensions: FrozenJsonDict = Field(default_factory=FrozenJsonDict)


class SessionStartPayloadV1(ObservationPayloadV1):
    model: WireText
    platform: WireText


class SessionEndPayloadV1(SessionStartPayloadV1):
    model: WireText | None = None
    completed: bool
    interrupted: bool
    reason: WireText | None = None


class SessionBoundaryPayloadV1(ObservationPayloadV1):
    platform: WireText
    reason: WireText | None = None
    old_session_id: HostIdentifierText = None
    new_session_id: HostIdentifierText = None


class PreLlmPayloadV1(ObservationPayloadV1):
    user_message: FrozenJsonValue
    conversation_history: FrozenJsonValue
    is_first_turn: bool
    model: WireText
    platform: WireText


class PostLlmPayloadV1(ObservationPayloadV1):
    user_message: FrozenJsonValue
    assistant_response: FrozenJsonValue
    conversation_history: FrozenJsonValue
    model: WireText
    platform: WireText


class PreApiPayloadV1(ObservationPayloadV1):
    user_message: FrozenJsonValue
    conversation_history: FrozenJsonValue
    platform: WireText
    model: WireText
    provider: WireText
    base_url: FrozenJsonValue
    api_mode: WireText
    api_call_count: NonNegativeInteger
    request_messages: FrozenJsonValue
    message_count: NonNegativeInteger
    tool_count: NonNegativeInteger
    approx_input_tokens: NonNegativeInteger
    request_char_count: NonNegativeInteger
    max_tokens: FrozenJsonValue
    started_at: FrozenJsonValue
    middleware_trace: FrozenJsonValue
    request: FrozenJsonValue


class PostApiPayloadV1(ObservationPayloadV1):
    platform: WireText
    model: WireText
    provider: WireText
    base_url: FrozenJsonValue
    api_mode: WireText
    api_call_count: NonNegativeInteger
    api_duration: NonNegativeNumber
    started_at: FrozenJsonValue
    ended_at: FrozenJsonValue
    finish_reason: FrozenJsonValue
    message_count: NonNegativeInteger
    response_model: FrozenJsonValue
    response: FrozenJsonValue
    usage: FrozenJsonValue
    assistant_message: FrozenJsonValue
    assistant_content_chars: NonNegativeInteger
    assistant_tool_call_count: NonNegativeInteger


class ApiErrorPayloadV1(ObservationPayloadV1):
    platform: WireText
    model: WireText
    provider: WireText
    base_url: FrozenJsonValue
    api_mode: WireText
    api_call_count: NonNegativeInteger
    api_duration: NonNegativeNumber
    started_at: FrozenJsonValue
    ended_at: FrozenJsonValue
    status_code: FrozenJsonValue
    retry_count: NonNegativeInteger | None = None
    max_retries: NonNegativeInteger | None = None
    retryable: bool | None = None
    reason: WireText | None = None
    error: FrozenJsonValue
    request: FrozenJsonValue


class PreToolPayloadV1(ObservationPayloadV1):
    tool_name: WireText
    args: FrozenJsonValue
    middleware_trace: FrozenJsonValue


class PostToolPayloadV1(PreToolPayloadV1):
    result: FrozenJsonValue
    duration_ms: NonNegativeNumber
    status: WireText
    error_type: FrozenJsonValue
    error_message: FrozenJsonValue


class ApprovalPayloadV1(ObservationPayloadV1):
    command: WireText
    description: WireText
    pattern_key: WireText
    pattern_keys: FrozenJsonValue
    session_key: WireText
    surface: WireText


class ApprovalResponsePayloadV1(ApprovalPayloadV1):
    choice: WireText
    decided_by: WireText | None = None


class SubagentStartPayloadV1(ObservationPayloadV1):
    parent_turn_id: HostIdentifierText
    parent_subagent_id: FrozenJsonValue
    child_subagent_id: FrozenJsonValue
    child_role: WireText | None = None
    child_goal: FrozenJsonValue


class SubagentStopPayloadV1(ObservationPayloadV1):
    parent_turn_id: HostIdentifierText
    child_role: WireText | None = None
    child_summary: FrozenJsonValue
    child_status: WireText | None = None
    duration_ms: NonNegativeNumber


class PreVerifyPayloadV1(ObservationPayloadV1):
    platform: WireText
    model: WireText
    coding: bool
    attempt: NonNegativeInteger
    final_response: FrozenJsonValue
    changed_paths: FrozenJsonValue


class HermesObservationV1(_StrictModel):
    """Abstract immutable envelope for the closed Hermes observer union."""

    schema_version: Literal[RECORD_SCHEMA_VERSION] = RECORD_SCHEMA_VERSION
    record_kind: Literal["observation"] = "observation"
    bridge_instance_id: str
    capture_seq: int = Field(ge=1, le=MAX_CAPTURE_SEQUENCE)
    captured_at: datetime
    captured_monotonic_ns: int = Field(ge=0)
    source_schema_version: Literal["hermes.observer.v1"] = "hermes.observer.v1"
    hook: HermesHook
    context: ObservationContextV1
    payload: ObservationPayloadV1
    coverage: CoverageV1

    @field_validator("bridge_instance_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("captured_at")
    @classmethod
    def _aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _bounded(self) -> HermesObservationV1:
        if type(self) is HermesObservationV1:
            raise ValueError("observation must use one hook-specific variant")
        _validate_record_size(self)
        return self

    @property
    def first_capture_seq(self) -> int:
        return self.capture_seq

    @property
    def last_capture_seq(self) -> int:
        return self.capture_seq

    def canonical_json(self) -> str:
        return _canonical_model_json(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class SessionStartObservationV1(HermesObservationV1):
    hook: Literal["on_session_start"]
    context: SessionContextV1
    payload: SessionStartPayloadV1


class SessionEndObservationV1(HermesObservationV1):
    hook: Literal["on_session_end"]
    context: SessionEndContextV1
    payload: SessionEndPayloadV1


class SessionFinalizeObservationV1(HermesObservationV1):
    hook: Literal["on_session_finalize"]
    context: SessionContextV1
    payload: SessionBoundaryPayloadV1


class SessionResetObservationV1(HermesObservationV1):
    hook: Literal["on_session_reset"]
    context: SessionContextV1
    payload: SessionBoundaryPayloadV1


class PreLlmObservationV1(HermesObservationV1):
    hook: Literal["pre_llm_call"]
    context: PreLlmContextV1
    payload: PreLlmPayloadV1


class PostLlmObservationV1(HermesObservationV1):
    hook: Literal["post_llm_call"]
    context: SessionTurnContextV1
    payload: PostLlmPayloadV1


class PreApiObservationV1(HermesObservationV1):
    hook: Literal["pre_api_request"]
    context: ApiContextV1
    payload: PreApiPayloadV1


class PostApiObservationV1(HermesObservationV1):
    hook: Literal["post_api_request"]
    context: ApiContextV1
    payload: PostApiPayloadV1


class ApiErrorObservationV1(HermesObservationV1):
    hook: Literal["api_request_error"]
    context: ApiContextV1
    payload: ApiErrorPayloadV1


class PreToolObservationV1(HermesObservationV1):
    hook: Literal["pre_tool_call"]
    context: ToolContextV1
    payload: PreToolPayloadV1


class PostToolObservationV1(HermesObservationV1):
    hook: Literal["post_tool_call"]
    context: ToolContextV1
    payload: PostToolPayloadV1


class PreApprovalObservationV1(HermesObservationV1):
    hook: Literal["pre_approval_request"]
    context: ApprovalContextV1
    payload: ApprovalPayloadV1


class PostApprovalObservationV1(HermesObservationV1):
    hook: Literal["post_approval_response"]
    context: ApprovalContextV1
    payload: ApprovalResponsePayloadV1


class SubagentStartObservationV1(HermesObservationV1):
    hook: Literal["subagent_start"]
    context: SubagentContextV1
    payload: SubagentStartPayloadV1


class SubagentStopObservationV1(HermesObservationV1):
    hook: Literal["subagent_stop"]
    context: SubagentContextV1
    payload: SubagentStopPayloadV1


class PreVerifyObservationV1(HermesObservationV1):
    hook: Literal["pre_verify"]
    context: SessionContextV1
    payload: PreVerifyPayloadV1


_DiscriminatedHermesObservationV1: TypeAlias = Annotated[
    SessionStartObservationV1
    | SessionEndObservationV1
    | SessionFinalizeObservationV1
    | SessionResetObservationV1
    | PreLlmObservationV1
    | PostLlmObservationV1
    | PreApiObservationV1
    | PostApiObservationV1
    | ApiErrorObservationV1
    | PreToolObservationV1
    | PostToolObservationV1
    | PreApprovalObservationV1
    | PostApprovalObservationV1
    | SubagentStartObservationV1
    | SubagentStopObservationV1
    | PreVerifyObservationV1,
    Field(discriminator="hook"),
]
HermesObservationRecordV1: TypeAlias = _DiscriminatedHermesObservationV1


class BridgeGapV1(_StrictModel):
    """One exact, finite interval of observations known to have been dropped."""

    schema_version: Literal[GAP_SCHEMA_VERSION] = GAP_SCHEMA_VERSION
    record_kind: Literal["gap"] = "gap"
    bridge_instance_id: str
    first_capture_seq: int = Field(ge=1, le=MAX_CAPTURE_SEQUENCE)
    last_capture_seq: int = Field(ge=1, le=MAX_CAPTURE_SEQUENCE)
    dropped_count: int = Field(ge=1, le=MAX_CAPTURE_SEQUENCE)
    cause_counts: FrozenJsonDict

    @field_validator("bridge_instance_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("cause_counts", mode="before")
    @classmethod
    def _causes(cls, value: Any) -> FrozenJsonDict:
        if not isinstance(value, Mapping) or not 1 <= len(value) <= 16:
            raise ValueError("gap cause_counts must be a bounded object")
        for cause, count in value.items():
            if cause not in GAP_CAUSES:
                raise ValueError("gap causes must use the fixed cause enum")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("gap cause counts must be positive integers")
            if count > MAX_CAPTURE_SEQUENCE:
                raise ValueError("gap cause counts exceed the capture sequence bound")
        return FrozenJsonDict(dict(value))

    @model_validator(mode="after")
    def _interval(self) -> BridgeGapV1:
        expected = self.last_capture_seq - self.first_capture_seq + 1
        if expected <= 0 or self.dropped_count != expected:
            raise ValueError("gap interval must exactly match dropped_count")
        if sum(self.cause_counts.values()) != self.dropped_count:
            raise ValueError("gap cause counts must sum to dropped_count")
        _validate_record_size(self)
        return self

    def canonical_json(self) -> str:
        return _canonical_model_json(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


BridgeRecordV1: TypeAlias = HermesObservationRecordV1 | BridgeGapV1
_OBSERVATION_ADAPTER = TypeAdapter(HermesObservationRecordV1)
_BRIDGE_RECORD_ADAPTER = TypeAdapter(BridgeRecordV1)


def validate_observation(value: object) -> HermesObservationV1:
    _bounded_tree(value)
    return _OBSERVATION_ADAPTER.validate_python(value, strict=True)


def validate_observation_json(value: str | bytes) -> HermesObservationV1:
    encoded = _prevalidate_record_json(value)
    return _OBSERVATION_ADAPTER.validate_json(encoded, strict=True)


def validate_bridge_record(value: object) -> BridgeRecordV1:
    _bounded_tree(value)
    return _BRIDGE_RECORD_ADAPTER.validate_python(value, strict=True)


def validate_bridge_record_json(value: str | bytes) -> BridgeRecordV1:
    encoded = _prevalidate_record_json(value)
    return _BRIDGE_RECORD_ADAPTER.validate_json(encoded, strict=True)


def validate_bridge_record_tree(value: object) -> None:
    """Apply fixed copier limits before any hook-specific model validation."""
    _bounded_tree(value)


class FrameFreshnessV1(_StrictModel):
    projected_at_state_sequence: int = Field(ge=0)
    scheduler_tick: int = Field(ge=0)
    scheduler_sample: Literal["running", "stopped", "not_sampled"]
    stream_connection: Literal["connected", "disconnected"]


class _ConsciousnessFrameBase(_StrictModel):
    """Fields shared by persisted legacy and current structural projections."""

    brain_id: str
    state_sequence: int = Field(ge=0)
    through_capture_seq: int = Field(ge=0, le=MAX_CAPTURE_SEQUENCE)
    logical_clock: float = Field(ge=0.0)
    trace_complete: bool
    runtime_health: Literal["healthy", "degraded"]
    c0_tick: int = Field(ge=0)
    pc: FrozenJsonDict
    energy: FrozenJsonDict
    st: FrozenJsonDict
    rd: FrozenJsonDict
    a: FrozenJsonDict
    world: FrozenJsonDict
    self_boundary: FrozenJsonDict
    memory: FrozenJsonDict
    capabilities: FrozenJsonDict
    semantic_context: FrozenJsonDict
    unresolved_evidence: bool
    capture_coverage: FrozenJsonDict
    freshness: FrameFreshnessV1
    omission_counts: FrozenJsonDict = Field(default_factory=FrozenJsonDict)

    @field_validator("brain_id")
    @classmethod
    def _brain_id(cls, value: str) -> str:
        return validate_id(value)

    def canonical_json(self) -> str:
        return _canonical_model_json(self)


class ConsciousnessFrameV2(_ConsciousnessFrameBase):
    """Legacy raw-only frame retained solely for verified SQLite migration."""

    schema_version: Literal[2] = 2


class FrameSemanticEvidenceV1(_StrictModel):
    """Cumulative persisted semantic health at one frame boundary."""

    schema_version: Literal[1] = 1
    semantic_records: int = Field(ge=0)
    legacy_raw_only_records: int = Field(ge=0)
    semantic_gap_records: int = Field(ge=0)
    dropped_events: int = Field(ge=0)

    @model_validator(mode="after")
    def _counts_are_possible(self) -> FrameSemanticEvidenceV1:
        if self.legacy_raw_only_records > self.semantic_records:
            raise ValueError("semantic frame evidence counts are inconsistent")
        if self.dropped_events and not self.semantic_gap_records:
            raise ValueError("dropped events require explicit semantic gap evidence")
        return self


class ConsciousnessFrameV3(_ConsciousnessFrameBase):
    """Bounded projection through the final event of one semantic batch."""

    schema_version: Literal[FRAME_SCHEMA_VERSION] = FRAME_SCHEMA_VERSION
    semantic_schema_version: Literal[1] = 1
    aggregate_semantic_complete: bool
    semantic_evidence: FrameSemanticEvidenceV1

    @model_validator(mode="after")
    def _aggregate_semantic_evidence_is_truthful(self) -> ConsciousnessFrameV3:
        expected = (
            self.trace_complete
            and self.semantic_evidence.legacy_raw_only_records == 0
            and self.semantic_evidence.semantic_gap_records == 0
        )
        if self.aggregate_semantic_complete is not expected:
            raise ValueError(
                "aggregate semantic completeness must match cumulative evidence"
            )
        return self


class BridgeCommitAckV1(_StrictModel):
    """Canonical persisted acknowledgement for one accepted bridge record."""

    schema_version: Literal[1] = 1
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    duplicate: bool = False
    event_id: str
    event_sequence: int = Field(ge=1)
    frame: ConsciousnessFrameV2
    through_capture_seq: int = Field(ge=1, le=MAX_CAPTURE_SEQUENCE)

    @field_validator("event_id")
    @classmethod
    def _event_id(cls, value: str) -> str:
        return validate_id(value)

    def canonical_json(self) -> str:
        return _canonical_model_json(self)


SemanticStatus: TypeAlias = Literal[
    "applied",
    "not_applicable",
    "gap",
    "legacy_raw_only",
]


class BridgeCommitAckV2(_StrictModel):
    """Canonical acknowledgement for one raw-plus-derived atomic batch."""

    schema_version: Literal[2] = 2
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    duplicate: Literal[False] = False
    raw_event_id: str
    raw_event_sequence: int = Field(ge=1)
    derived_event_ids: tuple[str, ...] = Field(default=(), max_length=8)
    derived_event_count: int = Field(ge=0, le=8)
    last_event_sequence: int = Field(ge=1)
    semantic_status: SemanticStatus
    semantic_complete: bool
    semantic_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame: ConsciousnessFrameV3
    through_capture_seq: int = Field(ge=1, le=MAX_CAPTURE_SEQUENCE)

    @field_validator("raw_event_id")
    @classmethod
    def _raw_event_id(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("derived_event_ids")
    @classmethod
    def _derived_event_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_id(item) for item in value)
        if len(validated) != len(set(validated)):
            raise ValueError("derived event IDs must be unique")
        return validated

    @model_validator(mode="after")
    def _semantic_batch(self) -> BridgeCommitAckV2:
        if self.derived_event_count != len(self.derived_event_ids):
            raise ValueError("derived event count must match derived event IDs")
        if self.raw_event_id in self.derived_event_ids:
            raise ValueError("raw event ID cannot also be a derived event ID")
        if self.last_event_sequence != (
            self.raw_event_sequence + self.derived_event_count
        ):
            raise ValueError("semantic batch event sequences must be contiguous")
        expected_complete = self.semantic_status in {"applied", "not_applicable"}
        if self.semantic_complete is not expected_complete:
            raise ValueError("semantic completeness must match semantic status")
        if self.semantic_status == "applied" and self.derived_event_count == 0:
            raise ValueError("applied semantic batches require a derived event")
        if self.semantic_status in {"not_applicable", "legacy_raw_only"} and (
            self.derived_event_count != 0
        ):
            raise ValueError(
                "not-applicable and legacy batches cannot claim derived events"
            )
        if self.semantic_status == "gap" and self.derived_event_count not in {0, 1}:
            raise ValueError("semantic gaps contain zero or one derived gap event")
        if (
            self.frame.state_sequence != self.last_event_sequence
            or self.frame.freshness.projected_at_state_sequence
            != self.last_event_sequence
            or self.frame.through_capture_seq != self.through_capture_seq
        ):
            raise ValueError("frame must bind the semantic batch terminal sequence")
        return self

    @property
    def event_id(self) -> str:
        """Compatibility accessor: the bridge record's event is the raw event."""
        return self.raw_event_id

    @property
    def event_sequence(self) -> int:
        """Compatibility accessor for the raw event's allocated sequence."""
        return self.raw_event_sequence

    def canonical_json(self) -> str:
        return _canonical_model_json(self)


class BridgeStreamState(_StrictModel):
    bridge_instance_id: str
    brain_id: str
    server_actor_id: str
    server_adapter_id: str = Field(min_length=1, max_length=512)
    next_capture_seq: int = Field(ge=1, le=MAX_BRIDGE_INTEGER)
    status: Literal["open", "clean_closed", "abandoned"]
    connected_nonce: str | None = Field(default=None, min_length=1, max_length=512)
    disconnected_reason: (
        Literal["connection_eof", "daemon_restart", "clean_close", "grace_abandonment"]
        | None
    ) = None
    disconnected_at: datetime | None = None
    last_seen: datetime
    closed_final_seq: int | None = Field(
        default=None,
        ge=0,
        le=MAX_CAPTURE_SEQUENCE,
    )

    @field_validator("bridge_instance_id", "brain_id", "server_actor_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("disconnected_at", "last_seen")
    @classmethod
    def _aware_time(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("disconnected_at must be timezone-aware")
            return value.astimezone(UTC)
        return None

    @model_validator(mode="after")
    def _connection_freshness(self) -> BridgeStreamState:
        if (self.disconnected_reason is None) != (self.disconnected_at is None):
            raise ValueError("disconnect reason and timestamp must appear together")
        if self.disconnected_at is not None and self.disconnected_at > self.last_seen:
            raise ValueError("disconnect timestamp cannot follow last_seen")
        if self.status == "open":
            if self.closed_final_seq is not None:
                raise ValueError("open stream cannot carry a closed cursor")
            if self.connected_nonce is not None:
                if self.disconnected_reason is not None:
                    raise ValueError(
                        "connected stream cannot carry disconnected provenance"
                    )
            elif self.disconnected_reason not in {
                "connection_eof",
                "daemon_restart",
            }:
                raise ValueError(
                    "disconnected open stream requires resumable provenance"
                )
        elif self.status == "clean_closed":
            if (
                self.connected_nonce is not None
                or self.disconnected_reason != "clean_close"
                or self.closed_final_seq != self.next_capture_seq - 1
            ):
                raise ValueError("clean-closed stream state is inconsistent")
        elif (
            self.connected_nonce is not None
            or self.disconnected_reason != "grace_abandonment"
            or self.closed_final_seq is not None
        ):
            raise ValueError("abandoned stream state is inconsistent")
        return self


class ObservabilitySnapshotV1(_StrictModel):
    """Bounded persisted bridge and semantic-health projection."""

    schema_version: Literal[1] = 1
    semantic_schema_version: Literal[1] = 1
    sqlite_schema_version: Literal[6] = 6
    brain_id: str | None = None
    brain_count: int = Field(ge=0)
    trace_complete: bool
    semantic_complete: bool
    dropped_events: int = Field(ge=0)
    semantic_records: int = Field(ge=0)
    legacy_raw_only_records: int = Field(ge=0)
    semantic_gap_records: int = Field(ge=0)
    total_bridges: int = Field(ge=0)
    connected_open_bridges: int = Field(ge=0)
    disconnected_open_bridges: int = Field(ge=0)
    clean_closed_bridges: int = Field(ge=0)
    abandoned_bridges: int = Field(ge=0)

    @field_validator("brain_id")
    @classmethod
    def _optional_brain_id(cls, value: str | None) -> str | None:
        return None if value is None else validate_id(value)

    @model_validator(mode="after")
    def _bridge_partition(self) -> ObservabilitySnapshotV1:
        if self.total_bridges != sum(
            (
                self.connected_open_bridges,
                self.disconnected_open_bridges,
                self.clean_closed_bridges,
                self.abandoned_bridges,
            )
        ):
            raise ValueError("observability bridge counts must form an exact partition")
        return self


class LoopbackEndpointV1(_StrictModel):
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(ge=1, le=65_535)


class DaemonDiscoveryV1(_StrictModel):
    schema_version: Literal[1] = 1
    pid: int = Field(ge=1)
    process_marker: str = Field(
        min_length=43,
        max_length=128,
        pattern=(
            r"^linux:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}:[1-9][0-9]{0,31}$"
        ),
    )
    instance_nonce: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )
    endpoint: LoopbackEndpointV1
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    package_version: str = Field(default="0.1.0", min_length=1, max_length=64)
    credential_ref: str = Field(
        min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.-]+$"
    )

    @model_validator(mode="after")
    def _credential_matches_nonce(self) -> DaemonDiscoveryV1:
        if self.credential_ref != f"credential-{self.instance_nonce}.key":
            raise ValueError("credential reference must be nonce-specific")
        return self

    def canonical_json(self) -> str:
        return _canonical_model_json(self)


class ProtocolLimitsV1(_StrictModel):
    max_request_bytes: int = Field(
        default=(TASK6_MAX_DETACHED_RECORD_BYTES + MAX_BRIDGE_COMMIT_ENVELOPE_BYTES),
        ge=4_096,
        le=MAX_PROTOCOL_BYTES,
    )
    max_response_bytes: int = Field(default=1_048_576, ge=4_096, le=MAX_PROTOCOL_BYTES)
    max_frame_bytes: int = Field(default=65_536, ge=4_096, le=1_048_576)
    max_record_bytes: int = Field(
        default=TASK6_MAX_DETACHED_RECORD_BYTES,
        ge=4_096,
        le=TASK6_MAX_DETACHED_RECORD_BYTES,
    )
    max_depth: int = Field(default=32, ge=8, le=MAX_PROTOCOL_DEPTH)
    max_nodes: int = Field(default=20_000, ge=256, le=MAX_PROTOCOL_NODES)
    max_concurrent_connections: int = Field(default=64, ge=1, le=1_024)
    unauthenticated_idle_timeout_ms: int = Field(default=5_000, ge=100, le=60_000)

    @model_validator(mode="after")
    def _request_contains_declared_record(self) -> ProtocolLimitsV1:
        required = self.max_record_bytes + MAX_BRIDGE_COMMIT_ENVELOPE_BYTES
        if self.max_request_bytes < required:
            raise ValueError(
                "max_request_bytes cannot contain max_record_bytes and the "
                "worst-case bridge.commit envelope"
            )
        return self


def copy_protocol_limits(value: object) -> ProtocolLimitsV1:
    """Return a fresh strict limits object; only None selects defaults."""
    if value is None:
        return ProtocolLimitsV1()
    if type(value) is not ProtocolLimitsV1:
        raise TypeError("limits must be an exact ProtocolLimitsV1 instance or None")
    return ProtocolLimitsV1.model_validate(
        value.model_dump(mode="python"),
        strict=True,
    )


class CapabilityProfileV1(_StrictModel):
    observer_schema_version: Literal[OBSERVER_SCHEMA_VERSION] = OBSERVER_SCHEMA_VERSION
    record_schema_version: Literal[RECORD_SCHEMA_VERSION] = RECORD_SCHEMA_VERSION
    gap_schema_version: Literal[GAP_SCHEMA_VERSION] = GAP_SCHEMA_VERSION
    frame_schema_version: Literal[FRAME_SCHEMA_VERSION] = FRAME_SCHEMA_VERSION
    limits: ProtocolLimitsV1 = Field(default_factory=ProtocolLimitsV1)
    continuous_runtime: Literal[True] = True
    typed_bridge: Literal[True] = True
    bridge_close_recovery: Literal["opaque_token_v1"] = "opaque_token_v1"
    arbitrary_event_append: Literal[False] = False
    provider_model_metadata: Literal[True] = True
    tool_metadata: Literal[True] = True
    multimodal_metadata: Literal[True] = True
    reasoning_metadata: Literal[True] = True
    chunk_capture: Literal["unobserved"] = "unobserved"
    reasoning_capture: Literal["unobserved"] = "unobserved"

    @classmethod
    def validate_wire(cls, value: object) -> CapabilityProfileV1:
        if not isinstance(value, dict) or set(value) != set(cls.model_fields):
            raise ValueError("capability profile fields are not exact")
        limits = value.get("limits")
        if not isinstance(limits, dict) or set(limits) != set(
            ProtocolLimitsV1.model_fields
        ):
            raise ValueError("protocol limit fields are not exact")
        return cls.model_validate(value, strict=True)


class InitializeResultV1(_StrictModel):
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    capabilities: CapabilityProfileV1
    server_adapter_id: Literal["alice-brain-hermes-observer-v1"] = SERVER_ADAPTER_ID
    instance_nonce: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )


class BrainProfileV1(_StrictModel):
    """Stable server-resolved profile key, not a client-owned brain identity."""

    schema_version: Literal[1] = 1
    profile_key: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    name: str | None = Field(default=None, min_length=1, max_length=160)

    @field_validator("name")
    @classmethod
    def _nonblank_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("brain profile name must be non-blank or null")
        return value

    def canonical_json(self) -> str:
        return _canonical_model_json(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _canonical_model_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_record_size(model: BaseModel) -> None:
    data = model.model_dump(mode="python")
    _bounded_tree(data)
    encoded = _canonical_model_json(model).encode("utf-8")
    if len(encoded) > MAX_BRIDGE_RECORD_BYTES:
        raise ValueError("bridge record exceeds the byte limit")


__all__ = [
    "FRAME_SCHEMA_VERSION",
    "GAP_SCHEMA_VERSION",
    "HOOK_EVENT_TYPES",
    "MAX_BRIDGE_COMMIT_ENVELOPE_BYTES",
    "MAX_BRIDGE_RECORD_BYTES",
    "MAX_CAPTURE_SEQUENCE",
    "OBSERVER_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "RECORD_SCHEMA_VERSION",
    "SERVER_ADAPTER_ID",
    "TASK6_MAX_DETACHED_RECORD_BYTES",
    "BrainProfileV1",
    "BridgeCommitAckV1",
    "BridgeCommitAckV2",
    "BridgeGapV1",
    "BridgeRecordV1",
    "BridgeStreamState",
    "CapabilityProfileV1",
    "ConsciousnessFrameV2",
    "ConsciousnessFrameV3",
    "CoverageV1",
    "DaemonDiscoveryV1",
    "FrameFreshnessV1",
    "FrameSemanticEvidenceV1",
    "HermesHook",
    "HermesObservationRecordV1",
    "HermesObservationV1",
    "InitializeResultV1",
    "LoopbackEndpointV1",
    "ObservabilitySnapshotV1",
    "ObservationContextV1",
    "ProtocolLimitsV1",
    "SemanticStatus",
    "copy_protocol_limits",
    "validate_bridge_record",
    "validate_bridge_record_json",
    "validate_bridge_record_tree",
    "validate_observation",
    "validate_observation_json",
]
