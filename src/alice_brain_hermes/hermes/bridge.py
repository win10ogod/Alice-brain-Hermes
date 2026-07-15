"""Bounded synchronous capture bridge with one transport-owning worker."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import ValidationError

from alice_brain_hermes.errors import DaemonClientError, DaemonRpcError
from alice_brain_hermes.ids import new_id, validate_id
from alice_brain_hermes.projections import AtomicProjectionCache, BridgeHealth
from alice_brain_hermes.protocol.models import (
    GAP_CAUSES,
    MAX_BRIDGE_INTEGER,
    MAX_BRIDGE_STRING_BYTES,
    MAX_CAPTURE_SEQUENCE,
    MIN_BRIDGE_INTEGER,
    BrainProfileV1,
    BridgeCommitAckV2,
    BridgeGapV1,
    BridgeRecordV1,
    BridgeStreamState,
    ConsciousnessFrameV3,
    HermesObservationV1,
    validate_observation,
)

SOURCE_SCHEMA_VERSION = "hermes.observer.v1"
DEFAULT_QUEUE_CAPACITY = 256
DEFAULT_RECONNECT_DELAY_SECONDS = 0.25
DEFAULT_CONNECT_TIMEOUT_SECONDS = 1.0
DEFAULT_FRAME_REFRESH_SECONDS = 1.0
_WORKER_CONTROL_POLL_SECONDS = 0.05
MAX_COPY_DEPTH = 6
MAX_COPY_NODES = 1_024
MAX_COPY_CONTAINER_ITEMS = 64
_TOKEN = re.compile(r"^[0-9a-f]{64}$")

ClientFactory = Callable[..., Any]
_BootstrapCaptureDisposition = Literal["accepted", "late_after_close"]
ProfileFactory = Callable[[], BrainProfileV1]


def _default_brain_profile() -> BrainProfileV1:
    return BrainProfileV1(profile_key="hermes.default", name=None)


@dataclass(slots=True)
class _DetachStats:
    nodes: int = 0
    redacted_paths: int = 0
    truncated_paths: int = 0
    unsupported_paths: int = 0
    omitted_nodes: int = 0


@dataclass(slots=True)
class _GapSpan:
    first_capture_seq: int
    last_capture_seq: int
    cause_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _BootstrapReservationReceipt:
    first_capture_seq: int
    last_capture_seq: int
    hook: str | None
    detached_kwargs: Mapping[str, object] | None
    gap_cause_counts: tuple[tuple[str, int], ...] | None
    copy_stats: tuple[tuple[str, int], ...]
    disposition: _BootstrapCaptureDisposition

    def matches(self, other: _BootstrapReservationReceipt) -> bool:
        return (
            self.first_capture_seq == other.first_capture_seq
            and self.last_capture_seq == other.last_capture_seq
            and self.hook == other.hook
            and self.detached_kwargs == other.detached_kwargs
            and self.gap_cause_counts == other.gap_cause_counts
            and self.copy_stats == other.copy_stats
        )


def default_runtime_home() -> Path:
    """Resolve the project-owned home without creating or inspecting it."""

    configured = os.environ.get("ALICE_BRAIN_HERMES_HOME")
    if configured:
        return Path(configured)
    home = os.environ.get("HOME", "~")
    return Path(home) / ".alice-brain-hermes"


def _truncate_text(value: str, stats: _DetachStats) -> str:
    bounded_prefix = value[: MAX_BRIDGE_STRING_BYTES + 1]
    try:
        encoded = bounded_prefix.encode("utf-8", errors="strict")
    except UnicodeError:
        stats.unsupported_paths += 1
        return ""
    if (
        len(value) <= MAX_BRIDGE_STRING_BYTES
        and len(encoded) <= MAX_BRIDGE_STRING_BYTES
    ):
        return value
    stats.truncated_paths += 1
    return encoded[:MAX_BRIDGE_STRING_BYTES].decode("utf-8", errors="ignore")


def _detach_json(
    value: object,
    stats: _DetachStats,
    *,
    depth: int = 1,
) -> object:
    stats.nodes += 1
    if stats.nodes > MAX_COPY_NODES:
        stats.omitted_nodes += 1
        return {"$omitted": "node_budget"}
    if depth > MAX_COPY_DEPTH:
        stats.omitted_nodes += 1
        return {"$omitted": "depth_budget"}
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        return _truncate_text(value, stats)
    if type(value) is int:
        if MIN_BRIDGE_INTEGER <= value <= MAX_BRIDGE_INTEGER:
            return value
        stats.unsupported_paths += 1
        return {"$unsupported": "integer_out_of_range"}
    if type(value) is float:
        if math.isfinite(value):
            return value
        stats.unsupported_paths += 1
        return {"$unsupported": "non_finite_number"}
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            stats.unsupported_paths += 1
            return {"$unsupported": "naive_datetime"}
        return value.astimezone(UTC).isoformat()
    if type(value) is dict:
        result: dict[str, object] = {}
        item_count = len(value)
        iterator = iter(value.items())
        items = []
        for _index in range(min(item_count, MAX_COPY_CONTAINER_ITEMS)):
            items.append(next(iterator))
        if item_count > MAX_COPY_CONTAINER_ITEMS:
            stats.truncated_paths += 1
            stats.omitted_nodes += item_count - MAX_COPY_CONTAINER_ITEMS
        for key, child in items:
            if type(key) is not str:
                stats.unsupported_paths += 1
                continue
            detached_key = _truncate_text(key, stats)
            result[detached_key] = _detach_json(child, stats, depth=depth + 1)
        return result
    if type(value) in {list, tuple}:
        items = value
        if len(items) > MAX_COPY_CONTAINER_ITEMS:
            stats.truncated_paths += 1
            stats.omitted_nodes += len(items) - MAX_COPY_CONTAINER_ITEMS
        return [
            _detach_json(item, stats, depth=depth + 1)
            for item in items[:MAX_COPY_CONTAINER_ITEMS]
        ]
    # OpenAI and provider response objects commonly expose an already materialized
    # instance dictionary.  Reading it directly avoids calling properties,
    # serializers, provider code, or ``str(obj)`` on the callback thread.
    try:
        attributes = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        attributes = None
    if type(attributes) is dict:
        stats.unsupported_paths += 1
        return {
            "$object_type": type(value).__name__[:160],
            "fields": _detach_json(attributes, stats, depth=depth + 1),
        }
    stats.unsupported_paths += 1
    return {"$unsupported_type": type(value).__name__[:160]}


def _host_identifier(value: object, stats: _DetachStats) -> str | None:
    if value is None:
        return None
    if type(value) is str:
        return _truncate_text(value, stats)[:512]
    stats.unsupported_paths += 1
    return None


def _text(value: object, stats: _DetachStats) -> str:
    if value is None:
        return ""
    if type(value) is str:
        return _truncate_text(value, stats)
    stats.unsupported_paths += 1
    return ""


def _optional_text(value: object, stats: _DetachStats) -> str | None:
    return None if value is None else _text(value, stats)


def _boolean(value: object, stats: _DetachStats, *, optional: bool = False) -> Any:
    if type(value) is bool:
        return value
    if optional and value is None:
        return None
    stats.unsupported_paths += 1
    return False


def _nonnegative_integer(
    value: object,
    stats: _DetachStats,
    *,
    optional: bool = False,
) -> int | None:
    if type(value) is int and 0 <= value <= MAX_BRIDGE_INTEGER:
        return value
    if optional and value is None:
        return None
    stats.unsupported_paths += 1
    return 0


def _nonnegative_number(value: object, stats: _DetachStats) -> int | float:
    if (
        type(value) in {int, float}
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    ):
        return value
    stats.unsupported_paths += 1
    return 0


_CONTEXT_KEYS: dict[str, tuple[str, ...]] = {
    "on_session_start": ("session_id",),
    "on_session_end": ("session_id", "task_id", "turn_id", "api_request_id"),
    "on_session_finalize": ("session_id",),
    "on_session_reset": ("session_id",),
    "pre_llm_call": ("session_id", "task_id", "turn_id", "sender_id"),
    "post_llm_call": ("session_id", "task_id", "turn_id"),
    "pre_api_request": ("session_id", "task_id", "turn_id", "api_request_id"),
    "post_api_request": ("session_id", "task_id", "turn_id", "api_request_id"),
    "api_request_error": ("session_id", "task_id", "turn_id", "api_request_id"),
    "pre_tool_call": (
        "session_id",
        "task_id",
        "turn_id",
        "api_request_id",
        "tool_call_id",
    ),
    "post_tool_call": (
        "session_id",
        "task_id",
        "turn_id",
        "api_request_id",
        "tool_call_id",
    ),
    "pre_approval_request": ("turn_id", "tool_call_id"),
    "post_approval_response": ("turn_id", "tool_call_id"),
    "subagent_start": ("parent_session_id", "child_session_id"),
    "subagent_stop": ("parent_session_id", "child_session_id"),
    "pre_verify": ("session_id",),
}


_PAYLOAD_KINDS: dict[str, dict[str, str]] = {
    "on_session_start": {"model": "text", "platform": "text"},
    "on_session_end": {
        "model": "optional_text",
        "platform": "text",
        "completed": "bool",
        "interrupted": "bool",
        "reason": "optional_text",
    },
    "on_session_finalize": {
        "platform": "text",
        "reason": "optional_text",
        "old_session_id": "identifier",
        "new_session_id": "identifier",
    },
    "on_session_reset": {
        "platform": "text",
        "reason": "optional_text",
        "old_session_id": "identifier",
        "new_session_id": "identifier",
    },
    "pre_llm_call": {
        "user_message": "json",
        "conversation_history": "json",
        "is_first_turn": "bool",
        "model": "text",
        "platform": "text",
    },
    "post_llm_call": {
        "user_message": "json",
        "assistant_response": "json",
        "conversation_history": "json",
        "model": "text",
        "platform": "text",
    },
    "pre_api_request": {
        "user_message": "json",
        "conversation_history": "json",
        "platform": "text",
        "model": "text",
        "provider": "text",
        "base_url": "json",
        "api_mode": "text",
        "api_call_count": "int",
        "request_messages": "json",
        "message_count": "int",
        "tool_count": "int",
        "approx_input_tokens": "int",
        "request_char_count": "int",
        "max_tokens": "json",
        "started_at": "json",
        "middleware_trace": "json",
        "request": "json",
    },
    "post_api_request": {
        "platform": "text",
        "model": "text",
        "provider": "text",
        "base_url": "json",
        "api_mode": "text",
        "api_call_count": "int",
        "api_duration": "number",
        "started_at": "json",
        "ended_at": "json",
        "finish_reason": "json",
        "message_count": "int",
        "response_model": "json",
        "response": "json",
        "usage": "json",
        "assistant_message": "json",
        "assistant_content_chars": "int",
        "assistant_tool_call_count": "int",
    },
    "api_request_error": {
        "platform": "text",
        "model": "text",
        "provider": "text",
        "base_url": "json",
        "api_mode": "text",
        "api_call_count": "int",
        "api_duration": "number",
        "started_at": "json",
        "ended_at": "json",
        "status_code": "json",
        "retry_count": "optional_int",
        "max_retries": "optional_int",
        "retryable": "optional_bool",
        "reason": "optional_text",
        "error": "json",
        "request": "json",
    },
    "pre_tool_call": {
        "tool_name": "text",
        "args": "json",
        "middleware_trace": "json",
    },
    "post_tool_call": {
        "tool_name": "text",
        "args": "json",
        "middleware_trace": "json",
        "result": "json",
        "duration_ms": "number",
        "status": "text",
        "error_type": "json",
        "error_message": "json",
    },
    "pre_approval_request": {
        "command": "text",
        "description": "text",
        "pattern_key": "text",
        "pattern_keys": "json",
        "session_key": "text",
        "surface": "text",
    },
    "post_approval_response": {
        "command": "text",
        "description": "text",
        "pattern_key": "text",
        "pattern_keys": "json",
        "session_key": "text",
        "surface": "text",
        "choice": "text",
        "decided_by": "optional_text",
    },
    "subagent_start": {
        "parent_turn_id": "identifier",
        "parent_subagent_id": "json",
        "child_subagent_id": "json",
        "child_role": "optional_text",
        "child_goal": "json",
    },
    "subagent_stop": {
        "parent_turn_id": "identifier",
        "child_role": "optional_text",
        "child_summary": "json",
        "child_status": "optional_text",
        "duration_ms": "number",
    },
    "pre_verify": {
        "platform": "text",
        "model": "text",
        "coding": "bool",
        "attempt": "int",
        "final_response": "json",
        "changed_paths": "json",
    },
}


def _shape_value(kind: str, value: object, stats: _DetachStats) -> object:
    if kind == "json":
        return _detach_json(value, stats)
    if kind == "text":
        return _text(value, stats)
    if kind == "optional_text":
        return _optional_text(value, stats)
    if kind == "identifier":
        return _host_identifier(value, stats)
    if kind == "bool":
        return _boolean(value, stats)
    if kind == "optional_bool":
        return _boolean(value, stats, optional=True)
    if kind == "int":
        return _nonnegative_integer(value, stats)
    if kind == "optional_int":
        return _nonnegative_integer(value, stats, optional=True)
    if kind == "number":
        return _nonnegative_number(value, stats)
    raise AssertionError(f"unknown shape kind {kind!r}")


class HookBridge:
    """Capture-only callback surface plus one private transport worker."""

    def __init__(
        self,
        runtime_home: str | Path,
        *,
        bridge_instance_id: str | None = None,
        recovery_token: str | None = None,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
        reconnect_delay_seconds: float = DEFAULT_RECONNECT_DELAY_SECONDS,
        connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        frame_refresh_seconds: float = DEFAULT_FRAME_REFRESH_SECONDS,
        client_factory: ClientFactory | None = None,
        profile_factory: ProfileFactory | None = None,
        start_worker_on_capture: bool = True,
        context_sink: Callable[[str | None], None] | None = None,
    ) -> None:
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int):
            raise TypeError("queue_capacity must be an exact int")
        if not 1 <= queue_capacity <= 65_536:
            raise ValueError("queue_capacity must be between 1 and 65536")
        for name, value in (
            ("reconnect_delay_seconds", reconnect_delay_seconds),
            ("connect_timeout_seconds", connect_timeout_seconds),
            ("frame_refresh_seconds", frame_refresh_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 < float(value) <= 60
            ):
                raise ValueError(f"{name} must be finite, positive, and at most 60")
        if type(start_worker_on_capture) is not bool:
            raise TypeError("start_worker_on_capture must be an exact bool")
        if profile_factory is not None and not callable(profile_factory):
            raise TypeError("profile_factory must be callable")
        active_id = bridge_instance_id or new_id()
        self._bridge_instance_id = validate_id(active_id)
        active_token = recovery_token or secrets.token_hex(32)
        if type(active_token) is not str or _TOKEN.fullmatch(active_token) is None:
            raise ValueError("recovery_token must be exactly 256 bits of lowercase hex")
        self._recovery_token = active_token
        self.runtime_home = Path(runtime_home)
        self.queue: queue.Queue[HermesObservationV1] = queue.Queue(
            maxsize=queue_capacity
        )
        self.projections = AtomicProjectionCache(context_sink=context_sink)
        self._client_factory = client_factory or self._default_client_factory
        self._profile_factory = profile_factory or _default_brain_profile
        self._reconnect_delay_seconds = float(reconnect_delay_seconds)
        self._connect_timeout_seconds = float(connect_timeout_seconds)
        self._frame_refresh_seconds = float(frame_refresh_seconds)
        self._start_worker_on_capture = start_worker_on_capture
        self._capture_lock = threading.Lock()
        self._health_lock = threading.Lock()
        self._worker_lifecycle_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._wake_lock = threading.Lock()
        self._next_capture_seq = 1
        self._gap_spans: deque[_GapSpan] = deque()
        self._health = BridgeHealth()
        self._emergency_trace_incomplete = False
        self._emergency_connection_disconnected = False
        self._emergency_last_error: str | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._retained: BridgeRecordV1 | None = None
        self._queue_head: HermesObservationV1 | None = None
        self._client: Any | None = None
        self._binding: str | None = None
        self._brain_id: str | None = None
        self._brain_state_sequence_floor = 0
        self._last_ack: BridgeCommitAckV2 | None = None
        self._attached_next_capture_seq: int | None = None
        self._retained_attempted = False
        self._exact_retry_pending = False
        self._stop_requested = False
        self._wake_requested = False
        self._close_requested = False
        self._close_sealed = False
        self._close_final_capture_seq: int | None = None
        self._ever_had_capture = False
        self._last_frame_refresh_monotonic = 0.0
        self._last_bootstrap_reservation: _BootstrapReservationReceipt | None = None

    @staticmethod
    def _default_client_factory(runtime_home: Path, **kwargs: object) -> Any:
        # Import and all discovery/socket work are worker-only.  Registration and
        # module import remain inert.
        from alice_brain_hermes.protocol.client import DaemonClient

        return DaemonClient.connect(runtime_home, **kwargs)

    @property
    def bridge_instance_id(self) -> str:
        return self._bridge_instance_id

    @property
    def recovery_token(self) -> str:
        return self._recovery_token

    @property
    def health(self) -> BridgeHealth:
        health = self._health
        worker_started = self.worker_started
        if (
            not self._emergency_trace_incomplete
            and health.worker_started is worker_started
        ):
            return health
        return BridgeHealth(
            connection=(
                "disconnected"
                if self._emergency_connection_disconnected
                else health.connection
            ),
            trace_complete=False,
            dropped_events=health.dropped_events,
            pending_gap_ranges=health.pending_gap_ranges,
            last_capture_seq=health.last_capture_seq,
            through_capture_seq=health.through_capture_seq,
            worker_started=worker_started,
            last_error=self._emergency_last_error or health.last_error,
            abandoned_streams=health.abandoned_streams,
            abandoned_local_records=health.abandoned_local_records,
            ambiguous_records=health.ambiguous_records,
            late_after_close=health.late_after_close,
            last_abandonment=health.last_abandonment,
            capabilities=health.capabilities,
        )

    def _mark_emergency_failure(self, error: BaseException) -> None:
        self._emergency_trace_incomplete = True
        self._emergency_connection_disconnected = True
        try:
            self._emergency_last_error = type(error).__name__[:160]
        except BaseException:
            self._emergency_last_error = "callback_internal"

    @property
    def worker_started(self) -> bool:
        worker = self._worker
        if worker is None:
            return False
        try:
            return worker.is_alive()
        except BaseException as error:
            self._mark_emergency_failure(error)
            return False

    @property
    def retained_record(self) -> BridgeRecordV1 | None:
        return self._retained

    @property
    def last_ack(self) -> BridgeCommitAckV2 | None:
        return self._last_ack

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def close_requested(self) -> bool:
        return self._close_requested

    @property
    def close_sealed(self) -> bool:
        return self._close_sealed

    def capture(self, hook: str, kwargs: dict[str, Any]) -> None:
        """Reserve one sequence, detach a bounded copy, and never raise."""

        try:
            produced = self._capture_atomic(hook, kwargs)
        except BaseException as error:
            self._mark_emergency_failure(error)
            return
        if produced:
            self._notify_worker()

    def _notify_worker(self, *, force_start: bool = False) -> None:
        try:
            with self._wake_lock:
                self._wake_requested = True
        except BaseException as error:
            self._mark_emergency_failure(error)
        try:
            self._wake_event.set()
        except BaseException as error:
            self._mark_emergency_failure(error)
        finally:
            if force_start or self._start_worker_on_capture:
                try:
                    self.start_worker()
                except BaseException as error:
                    self._mark_emergency_failure(error)

    def _consume_worker_wake(self) -> bool:
        try:
            with self._wake_lock:
                if not self._wake_requested:
                    return False
                self._wake_requested = False
                return True
        except BaseException as error:
            self._mark_emergency_failure(error)
            return False

    def _capture_atomic(self, hook: str, kwargs: dict[str, Any]) -> bool:
        with self._capture_lock, self._health_lock:
            if self._close_sealed:
                updated_health = replace(
                    self._health,
                    trace_complete=False,
                    dropped_events=self._health.dropped_events + 1,
                    late_after_close=self._health.late_after_close + 1,
                    last_error="capture_after_close_seal",
                )
                self._health = updated_health
                return False
            capture_seq = self._next_capture_seq
            if capture_seq > MAX_CAPTURE_SEQUENCE:
                self._health = replace(
                    self._health,
                    trace_complete=False,
                    last_error="capture_sequence_capacity_exhausted",
                )
                return False
            next_capture_seq = capture_seq + 1
            if kwargs.get("telemetry_schema_version") != SOURCE_SCHEMA_VERSION:
                prepared_gap = self._prepare_gap_publication_locked(
                    capture_seq,
                    capture_seq,
                    {"invalid_source_schema": 1},
                    health_last_capture_seq=capture_seq,
                )
                self._publish_gap_locked(*prepared_gap)
            else:
                try:
                    record = self._shape_observation(hook, capture_seq, kwargs)
                except BaseException:
                    prepared_gap = self._prepare_gap_publication_locked(
                        capture_seq,
                        capture_seq,
                        {"callback_internal": 1},
                        health_last_capture_seq=capture_seq,
                    )
                    self._publish_gap_locked(*prepared_gap)
                else:
                    self._publish_observation_or_gap_locked(
                        record,
                        capture_seq=capture_seq,
                    )
            self._next_capture_seq = next_capture_seq
            self._ever_had_capture = True
            return True

    def capture_reserved(
        self,
        *,
        hook: str | None,
        detached_kwargs: Mapping[str, object] | None,
        first_capture_seq: int,
        last_capture_seq: int,
        gap_cause_counts: Mapping[str, int] | None,
        copy_stats: Mapping[str, int],
    ) -> _BootstrapCaptureDisposition:
        """Accept one exact bootstrap reservation without renumbering it."""

        for name, value in (
            ("first_capture_seq", first_capture_seq),
            ("last_capture_seq", last_capture_seq),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive exact int")
            if value > MAX_CAPTURE_SEQUENCE:
                raise ValueError("capture sequence capacity is exhausted")
        if last_capture_seq < first_capture_seq:
            raise ValueError("reserved capture interval is reversed")
        validated_stats: dict[str, int] = {}
        for name in (
            "redacted_paths",
            "truncated_paths",
            "unsupported_paths",
            "omitted_nodes",
        ):
            value = copy_stats.get(name, 0)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= MAX_BRIDGE_INTEGER
            ):
                raise ValueError("bootstrap copy_stats are invalid")
            validated_stats[name] = value
        capture_count = last_capture_seq - first_capture_seq + 1
        if gap_cause_counts is None:
            if (
                type(hook) is not str
                or detached_kwargs is None
                or first_capture_seq != last_capture_seq
            ):
                raise ValueError("observation reservation is invalid")
        else:
            if hook is not None or detached_kwargs is not None:
                raise ValueError("gap reservation cannot contain an observation")
            if not 1 <= len(gap_cause_counts) <= 16:
                raise ValueError("gap causes are invalid")
            if any(
                cause not in GAP_CAUSES
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                for cause, count in gap_cause_counts.items()
            ):
                raise ValueError("gap cause counts are invalid")
            if sum(gap_cause_counts.values()) != capture_count:
                raise ValueError("gap cause counts do not match the interval")

        canonical_gap_causes = (
            None
            if gap_cause_counts is None
            else tuple(sorted(gap_cause_counts.items()))
        )
        canonical_copy_stats = tuple(
            (name, validated_stats[name])
            for name in (
                "redacted_paths",
                "truncated_paths",
                "unsupported_paths",
                "omitted_nodes",
            )
        )
        receipt = _BootstrapReservationReceipt(
            first_capture_seq=first_capture_seq,
            last_capture_seq=last_capture_seq,
            hook=hook,
            detached_kwargs=detached_kwargs,
            gap_cause_counts=canonical_gap_causes,
            copy_stats=canonical_copy_stats,
            disposition="accepted",
        )

        with self._capture_lock:
            if first_capture_seq != self._next_capture_seq:
                previous = self._last_bootstrap_reservation
                if (
                    previous is not None
                    and first_capture_seq == previous.first_capture_seq
                    and last_capture_seq == previous.last_capture_seq
                    and self._next_capture_seq == previous.last_capture_seq + 1
                ):
                    if previous.matches(receipt):
                        return previous.disposition
                    raise ValueError("bootstrap reservation changed after acceptance")
                raise ValueError("bootstrap capture sequence is not contiguous")
            if self._close_sealed:
                # The clean-close cursor is already immutable, but bootstrap
                # reservations still need an input cursor so each late capture
                # is accounted exactly once instead of blocking handoff.
                with self._health_lock:
                    updated_health = replace(
                        self._health,
                        trace_complete=False,
                        dropped_events=self._health.dropped_events + capture_count,
                        late_after_close=(
                            self._health.late_after_close + capture_count
                        ),
                        last_error="capture_after_close_seal",
                    )
                    next_capture_seq = last_capture_seq + 1
                    terminal_receipt = _BootstrapReservationReceipt(
                        first_capture_seq=receipt.first_capture_seq,
                        last_capture_seq=receipt.last_capture_seq,
                        hook=receipt.hook,
                        detached_kwargs=receipt.detached_kwargs,
                        gap_cause_counts=receipt.gap_cause_counts,
                        copy_stats=receipt.copy_stats,
                        disposition="late_after_close",
                    )
                    self._health = updated_health
                    self._next_capture_seq = next_capture_seq
                    self._last_bootstrap_reservation = terminal_receipt
                return "late_after_close"
            next_capture_seq = last_capture_seq + 1
            with self._health_lock:
                if gap_cause_counts is not None:
                    prepared_gap = self._prepare_gap_publication_locked(
                        first_capture_seq,
                        last_capture_seq,
                        dict(gap_cause_counts),
                        health_last_capture_seq=last_capture_seq,
                    )
                    self._publish_gap_locked(*prepared_gap)
                else:
                    try:
                        stats = _DetachStats(**validated_stats)
                        if hook is None or detached_kwargs is None:
                            raise ValueError("observation reservation is incomplete")
                        record = self._shape_observation(
                            hook,
                            first_capture_seq,
                            detached_kwargs,
                            stats=stats,
                        )
                    except BaseException:
                        prepared_gap = self._prepare_gap_publication_locked(
                            first_capture_seq,
                            last_capture_seq,
                            {"callback_internal": capture_count},
                            health_last_capture_seq=last_capture_seq,
                        )
                        self._publish_gap_locked(*prepared_gap)
                    else:
                        self._publish_observation_or_gap_locked(
                            record,
                            capture_seq=last_capture_seq,
                        )
                self._next_capture_seq = next_capture_seq
                self._ever_had_capture = True
                self._last_bootstrap_reservation = receipt
        self._notify_worker()
        return "accepted"

    def _publish_observation_or_gap_locked(
        self,
        record: HermesObservationV1,
        *,
        capture_seq: int,
    ) -> None:
        observation_health = replace(
            self._health,
            last_capture_seq=capture_seq,
        )
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            prepared_gap = self._prepare_gap_publication_locked(
                capture_seq,
                capture_seq,
                {"queue_full": 1},
                health_last_capture_seq=capture_seq,
            )
            self._publish_gap_locked(*prepared_gap)
        except BaseException as error:
            with self.queue.mutex:
                inserted = any(item is record for item in self.queue.queue)
            if inserted:
                self._health = observation_health
                self._mark_emergency_failure(error)
            else:
                prepared_gap = self._prepare_gap_publication_locked(
                    capture_seq,
                    capture_seq,
                    {"callback_internal": 1},
                    health_last_capture_seq=capture_seq,
                )
                self._publish_gap_locked(*prepared_gap)
        else:
            self._health = observation_health

    def _prepare_gap_publication_locked(
        self,
        first_capture_seq: int,
        last_capture_seq: int,
        cause_counts: dict[str, int],
        *,
        health_last_capture_seq: int,
    ) -> tuple[_GapSpan, bool, BridgeHealth]:
        replace_last = bool(
            self._gap_spans
            and self._gap_spans[-1].last_capture_seq + 1 == first_capture_seq
        )
        if replace_last:
            previous = self._gap_spans[-1]
            merged_causes = dict(previous.cause_counts)
            for cause, count in cause_counts.items():
                merged_causes[cause] = merged_causes.get(cause, 0) + count
            span = _GapSpan(
                previous.first_capture_seq,
                last_capture_seq,
                merged_causes,
            )
        else:
            span = _GapSpan(
                first_capture_seq,
                last_capture_seq,
                dict(cause_counts),
            )
        claimed_gap = 1 if isinstance(self._retained, BridgeGapV1) else 0
        pending_gap_ranges = (
            len(self._gap_spans) + claimed_gap + (0 if replace_last else 1)
        )
        dropped_count = last_capture_seq - first_capture_seq + 1
        updated_health = replace(
            self._health,
            trace_complete=False,
            dropped_events=self._health.dropped_events + dropped_count,
            pending_gap_ranges=pending_gap_ranges,
            last_capture_seq=health_last_capture_seq,
        )
        return span, replace_last, updated_health

    def _publish_gap_locked(
        self,
        span: _GapSpan,
        replace_last: bool,
        updated_health: BridgeHealth,
    ) -> None:
        if replace_last:
            self._gap_spans[-1] = span
        else:
            self._gap_spans.append(span)
        self._health = updated_health

    def _shape_observation(
        self,
        hook: str,
        capture_seq: int,
        kwargs: Mapping[str, object],
        *,
        stats: _DetachStats | None = None,
    ) -> HermesObservationV1:
        if hook not in _CONTEXT_KEYS or hook not in _PAYLOAD_KINDS:
            raise ValueError("unsupported Hermes hook")
        expected = set(_CONTEXT_KEYS[hook]) | set(_PAYLOAD_KINDS[hook])
        if not expected.intersection(kwargs):
            raise ValueError("hook payload contains no recognized host fields")
        stats = stats or _DetachStats()
        context = {
            key: _host_identifier(kwargs.get(key), stats) for key in _CONTEXT_KEYS[hook]
        }
        payload = {
            key: _shape_value(kind, kwargs.get(key), stats)
            for key, kind in _PAYLOAD_KINDS[hook].items()
        }
        extension_source: dict[str, object] = {}
        item_count = len(kwargs)
        scan_limit = len(expected) + MAX_COPY_CONTAINER_ITEMS + 1
        iterator = iter(kwargs.items())
        for _index in range(min(item_count, scan_limit)):
            key, value = next(iterator)
            if key in expected or key == "telemetry_schema_version":
                continue
            if len(extension_source) >= MAX_COPY_CONTAINER_ITEMS:
                stats.truncated_paths += 1
                stats.omitted_nodes += 1
                break
            extension_source[key] = value
        if item_count > scan_limit:
            stats.truncated_paths += 1
            stats.omitted_nodes += item_count - scan_limit
        payload["extensions"] = _detach_json(extension_source, stats)
        omissions = (
            stats.redacted_paths
            + stats.truncated_paths
            + stats.unsupported_paths
            + stats.omitted_nodes
        )
        coverage = {
            "policy_version": "alice-brain-hermes.copy.v1",
            "capture_coverage": "full" if omissions == 0 else "partial",
            "redacted_paths": stats.redacted_paths,
            "truncated_paths": stats.truncated_paths,
            "unsupported_paths": stats.unsupported_paths,
            "omitted_nodes": stats.omitted_nodes,
            "channels": {
                "hook": "observed",
                "chunk_capture": "unobserved",
                "reasoning_capture": "unobserved",
            },
        }
        raw = {
            "schema_version": 1,
            "record_kind": "observation",
            "bridge_instance_id": self._bridge_instance_id,
            "capture_seq": capture_seq,
            "captured_at": datetime.now(UTC),
            "captured_monotonic_ns": time.monotonic_ns(),
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "hook": hook,
            "context": context,
            "payload": payload,
            "coverage": coverage,
        }
        return validate_observation(raw)

    def pending_gaps(self) -> tuple[BridgeGapV1, ...]:
        with self._capture_lock:
            claimed = (
                (self._retained,) if isinstance(self._retained, BridgeGapV1) else ()
            )
            return claimed + tuple(
                self._gap_from_span(span) for span in self._gap_spans
            )

    def _gap_from_span(self, span: _GapSpan) -> BridgeGapV1:
        return BridgeGapV1(
            bridge_instance_id=self._bridge_instance_id,
            first_capture_seq=span.first_capture_seq,
            last_capture_seq=span.last_capture_seq,
            dropped_count=span.last_capture_seq - span.first_capture_seq + 1,
            cause_counts=dict(span.cause_counts),
        )

    def _claim_first_gap(self) -> BridgeGapV1 | None:
        with self._capture_lock:
            if not self._gap_spans:
                return None
            span = self._gap_spans[0]
            gap = self._gap_from_span(span)
            self._gap_spans.popleft()
            return gap

    def start_worker(self) -> None:
        with self._worker_lifecycle_lock:
            self._start_worker_under_lifecycle_lock()

    def _start_worker_under_lifecycle_lock(self) -> None:
        with self._worker_lock:
            existing = self._worker
            if existing is not None:
                try:
                    if existing.is_alive():
                        return
                except BaseException as error:
                    self._mark_emergency_failure(error)
                    return
                self._worker = None
            try:
                if self._stop_requested:
                    if self._close_sealed:
                        return
                    self._stop_event.clear()
                    self._stop_requested = False
                with self._health_lock:
                    previous_health = self._health
                    updated_health = replace(self._health, worker_started=True)
                    worker = threading.Thread(
                        target=self._run_worker,
                        name="alice-brain-hermes-bridge",
                        daemon=True,
                    )
                    self._worker = worker
                    try:
                        worker.start()
                    except BaseException as start_error:
                        try:
                            started = worker.is_alive()
                        except BaseException:
                            # The start result is unknown.  Retaining ownership is
                            # the only safe choice: clearing the pointer could let
                            # a second worker race a thread that did spawn.
                            self._health = updated_health
                            self._mark_emergency_failure(start_error)
                            return
                        if not started:
                            self._worker = None
                            self._health = previous_health
                            raise
                        self._health = updated_health
                        self._mark_emergency_failure(start_error)
                        return
                    self._health = updated_health
            except BaseException as error:
                self._mark_emergency_failure(error)
                return

    def _worker_alive_strict(self) -> bool:
        with self._worker_lock:
            worker = self._worker
        if worker is None:
            return False
        return worker.is_alive()

    def _worker_stop_requested(self) -> bool:
        if self._stop_requested:
            return True
        try:
            return self._stop_event.is_set()
        except BaseException as error:
            # Production stop paths publish the non-throwing latch first.  A
            # hostile Event probe therefore degrades health without making an
            # otherwise live worker inert.
            self._mark_emergency_failure(error)
            return False

    def _run_worker(self) -> None:
        current_worker = self._worker
        if current_worker is None:
            self._mark_emergency_failure(
                RuntimeError("bridge worker started without a published identity")
            )
            return
        try:
            while True:
                try:
                    if self._worker_stop_requested():
                        return
                    self._consume_worker_wake()
                    try:
                        record = self._select_next_record()
                    except BaseException as error:
                        self._mark_emergency_failure(error)
                        self._disconnect_client(error=self._error_label(error))
                        self._wait_reconnect()
                        continue
                    if record is None:
                        if self._close_requested and self._seal_close_if_ready():
                            if self._client is None and not self._connect():
                                self._wait_reconnect()
                                continue
                            if self._close_stream():
                                return
                        elif self._client is None and self._ever_had_capture:
                            if not self._connect():
                                self._wait_reconnect()
                                continue
                        elif (
                            self._client is not None
                            and time.monotonic() - self._last_frame_refresh_monotonic
                            >= self._frame_refresh_seconds
                            and not self._refresh_frame()
                        ):
                            self._wait_reconnect()
                            continue
                        self._wait_for_worker(self._reconnect_delay_seconds)
                        continue
                    if self._client is None and not self._connect():
                        self._wait_reconnect()
                        continue
                    if not self._commit_retained(record):
                        self._wait_reconnect()
                except BaseException as error:
                    self._mark_emergency_failure(error)
                    try:
                        self._disconnect_client(error=self._error_label(error))
                    except BaseException as disconnect_error:
                        self._mark_emergency_failure(disconnect_error)
                    self._wait_reconnect()
        finally:
            try:
                self._disconnect_client()
            except BaseException as error:
                self._mark_emergency_failure(error)
            self._publish_worker_exit(current_worker)

    def _publish_worker_exit(self, worker: threading.Thread) -> None:
        try:
            with self._worker_lock:
                if self._worker is worker:
                    self._worker = None
                with self._health_lock:
                    self._health = replace(self._health, worker_started=False)
        except BaseException as error:
            self._mark_emergency_failure(error)

    def _select_next_record(self) -> BridgeRecordV1 | None:
        if self._retained is not None:
            return self._retained
        if self._queue_head is None:
            try:
                self._queue_head = self.queue.get_nowait()
            except queue.Empty:
                self._queue_head = None
        with self._capture_lock:
            gap_first = (
                self._gap_spans[0].first_capture_seq if self._gap_spans else None
            )
        if gap_first is not None and (
            self._queue_head is None or gap_first < self._queue_head.capture_seq
        ):
            self._retained = self._claim_first_gap()
        elif self._queue_head is not None:
            self._retained = self._queue_head
            self._queue_head = None
        return self._retained

    def _validate_connected_frame(
        self,
        frame: ConsciousnessFrameV3,
        *,
        brain_id: str,
        expected_through_capture_seq: int | None = None,
        minimum_state_sequence: int | None = None,
        permit_historical_state: bool = False,
    ) -> bool:
        if frame.brain_id != brain_id:
            raise DaemonClientError("state frame changed the stable brain identity")
        if (
            expected_through_capture_seq is not None
            and frame.through_capture_seq != expected_through_capture_seq
        ):
            raise DaemonClientError("state frame cursor does not match attach")
        if (
            minimum_state_sequence is not None
            and frame.state_sequence < minimum_state_sequence
        ):
            raise DaemonClientError("state frame predates brain.resolve")
        if (
            frame.state_sequence < self._brain_state_sequence_floor
            and not permit_historical_state
        ):
            raise DaemonClientError("state frame regressed the stable brain")
        if frame.freshness.projected_at_state_sequence != frame.state_sequence:
            raise DaemonClientError("state frame freshness sequence is invalid")
        if frame.freshness.stream_connection != "connected":
            raise DaemonClientError("state frame is not connected")
        current = self.projections.frame
        regressed = current is not None and (
            frame.state_sequence < current.state_sequence
            or frame.through_capture_seq < current.through_capture_seq
        )
        if regressed and not permit_historical_state:
            raise DaemonClientError("state frame regressed")
        return regressed

    def _connect(self) -> bool:
        client: Any | None = None
        resolved_brain_id: str | None = None
        try:
            client = self._client_factory(
                self.runtime_home,
                initialize=True,
                timeout_seconds=self._connect_timeout_seconds,
            )
            profile = self._profile_factory()
            if type(profile) is not BrainProfileV1:
                raise TypeError("profile_factory must return an exact BrainProfileV1")
            resolved = client.call(
                "brain.resolve",
                {"profile": profile.model_dump(mode="json")},
            )
            if set(resolved) != {"brain_id", "state_sequence", "created"}:
                raise DaemonClientError("brain.resolve result fields are invalid")
            brain_id = validate_id(resolved["brain_id"])  # type: ignore[arg-type]
            resolved_brain_id = brain_id
            if self._brain_id is not None and brain_id != self._brain_id:
                raise DaemonClientError(
                    "brain.resolve changed the stable brain identity"
                )
            if (
                isinstance(resolved["state_sequence"], bool)
                or not isinstance(resolved["state_sequence"], int)
                or not 0 <= resolved["state_sequence"] <= MAX_BRIDGE_INTEGER
                or type(resolved["created"]) is not bool
            ):
                raise DaemonClientError("brain.resolve result is invalid")
            attached = client.call(
                "brain.attach",
                {
                    "brain_id": brain_id,
                    "bridge_instance_id": self._bridge_instance_id,
                    "recovery_token": self._recovery_token,
                },
            )
            if set(attached) != {"binding", "brain_id", "next_capture_seq"}:
                raise DaemonClientError("brain.attach result fields are invalid")
            binding = validate_id(attached["binding"])  # type: ignore[arg-type]
            if attached["brain_id"] != brain_id:
                raise DaemonClientError("brain.attach changed brain identity")
            next_capture_seq = attached["next_capture_seq"]
            if (
                isinstance(next_capture_seq, bool)
                or not isinstance(next_capture_seq, int)
                or not 1 <= next_capture_seq <= MAX_BRIDGE_INTEGER
            ):
                raise DaemonClientError("brain.attach cursor is invalid")
            local_last = self._health.last_capture_seq
            if next_capture_seq > local_last + 1:
                raise DaemonClientError(
                    "brain.attach cursor is beyond the locally reserved history"
                )
            retained = self._retained
            exact_retry = False
            if retained is not None:
                if next_capture_seq == retained.first_capture_seq:
                    # A successful attach proves the earlier transport attempt
                    # did not advance this cursor.
                    self._retained_attempted = False
                elif (
                    self._retained_attempted
                    and next_capture_seq == retained.last_capture_seq + 1
                ):
                    exact_retry = True
                else:
                    raise DaemonClientError(
                        "brain.attach cursor does not bind the retained record"
                    )
            elif next_capture_seq != local_last + 1:
                raise DaemonClientError(
                    "brain.attach cursor does not match local committed history"
                )
            frame_result = client.call("state.get", {"binding": binding})
            frame = ConsciousnessFrameV3.model_validate(frame_result, strict=True)
            self._validate_connected_frame(
                frame,
                brain_id=brain_id,
                expected_through_capture_seq=next_capture_seq - 1,
                minimum_state_sequence=resolved["state_sequence"],
            )
            self.projections.publish_frame(frame)
            self._brain_state_sequence_floor = max(
                self._brain_state_sequence_floor,
                frame.state_sequence,
            )
            self._last_frame_refresh_monotonic = time.monotonic()
            self._client = client
            self._binding = binding
            self._brain_id = brain_id
            self._attached_next_capture_seq = next_capture_seq
            self._exact_retry_pending = exact_retry
            self._publish_connection("connected", frame=frame)
            return True
        except DaemonRpcError as error:
            if (
                error.code == "bridge_clean_closed"
                and self._close_requested
                and client is not None
                and resolved_brain_id is not None
                and self._recover_clean_close(client, resolved_brain_id)
            ):
                return False
            if client is not None:
                self._close_client(client)
            failure: BaseException = error
            if error.code == "bridge_abandoned":
                rotation_error = self._try_rotate_abandoned_stream()
                if rotation_error is not None:
                    failure = rotation_error
            self._publish_connection("disconnected", error=self._error_label(failure))
            return False
        except (
            ValidationError,
            ValueError,
            TypeError,
            DaemonClientError,
            OSError,
        ) as error:
            if client is not None:
                self._close_client(client)
            self._publish_connection("disconnected", error=self._error_label(error))
            return False
        except BaseException as error:
            if client is not None:
                self._close_client(client)
            self._publish_connection("disconnected", error=self._error_label(error))
            return False

    def _commit_retained(self, record: BridgeRecordV1) -> bool:
        client = self._client
        binding = self._binding
        if client is None or binding is None:
            return False
        try:
            self._retained_attempted = True
            result = client.call(
                "bridge.commit",
                {"binding": binding, "record": record.model_dump(mode="json")},
            )
            if (
                type(result) is not dict
                or type(result.get("derived_event_ids")) is not list
            ):
                raise DaemonClientError(
                    "bridge acknowledgement is not an exact JSON object"
                )
            ack_values = dict(result)
            ack_values["derived_event_ids"] = tuple(result["derived_event_ids"])
            ack = BridgeCommitAckV2.model_validate(ack_values, strict=True)
            if (
                ack.duplicate is not False
                or ack.record_fingerprint != record.fingerprint()
                or ack.through_capture_seq != record.last_capture_seq
                or ack.frame.through_capture_seq != record.last_capture_seq
                or ack.frame.brain_id != self._brain_id
                or ack.last_event_sequence != ack.frame.state_sequence
                or ack.frame.freshness.projected_at_state_sequence
                != ack.last_event_sequence
                or ack.frame.freshness.stream_connection != "connected"
                or ack.frame.freshness.scheduler_sample != "not_sampled"
            ):
                raise DaemonClientError(
                    "bridge acknowledgement changed record identity"
                )
            if self._brain_id is None:
                raise DaemonClientError("bridge acknowledgement has no brain binding")
            historical = self._validate_connected_frame(
                ack.frame,
                brain_id=self._brain_id,
                expected_through_capture_seq=record.last_capture_seq,
                permit_historical_state=self._exact_retry_pending,
            )
            current = self.projections.frame
            if not historical:
                self.projections.publish_frame(ack.frame)
                self._brain_state_sequence_floor = max(
                    self._brain_state_sequence_floor,
                    ack.frame.state_sequence,
                )
                effective_frame = ack.frame
            elif current is not None:
                effective_frame = current
            else:
                raise DaemonClientError("historical ACK has no current projection")
            self._last_frame_refresh_monotonic = time.monotonic()
            was_gap = isinstance(record, BridgeGapV1)
            attached_next_capture_seq = record.last_capture_seq + 1
            if was_gap:
                with self._capture_lock, self._health_lock:
                    updated_health = replace(
                        self._health,
                        pending_gap_ranges=len(self._gap_spans),
                    )
                    self._last_ack = ack
                    self._retained = None
                    self._retained_attempted = False
                    self._exact_retry_pending = False
                    self._attached_next_capture_seq = attached_next_capture_seq
                    self._health = updated_health
            else:
                self._last_ack = ack
                self._retained = None
                self._retained_attempted = False
                self._exact_retry_pending = False
                self._attached_next_capture_seq = attached_next_capture_seq
            self._publish_connection("connected", frame=effective_frame)
            return True
        except DaemonRpcError as error:
            failure: BaseException = error
            if error.code == "bridge_abandoned":
                rotation_error = self._try_rotate_abandoned_stream()
                if rotation_error is not None:
                    failure = rotation_error
            self._disconnect_client(error=self._error_label(failure))
            return False
        except (
            ValidationError,
            ValueError,
            TypeError,
            DaemonClientError,
            OSError,
        ) as error:
            self._disconnect_client(error=self._error_label(error))
            return False
        except BaseException as error:
            self._disconnect_client(error=self._error_label(error))
            return False

    def _refresh_frame(self) -> bool:
        client = self._client
        binding = self._binding
        brain_id = self._brain_id
        if client is None or binding is None or brain_id is None:
            return False
        try:
            result = client.call("state.get", {"binding": binding})
            frame = ConsciousnessFrameV3.model_validate(result, strict=True)
            self._validate_connected_frame(frame, brain_id=brain_id)
            self.projections.publish_frame(frame)
            self._brain_state_sequence_floor = max(
                self._brain_state_sequence_floor,
                frame.state_sequence,
            )
            self._last_frame_refresh_monotonic = time.monotonic()
            self._publish_connection("connected", frame=frame)
            return True
        except DaemonRpcError as error:
            failure: BaseException = error
            if error.code == "bridge_abandoned":
                rotation_error = self._try_rotate_abandoned_stream()
                if rotation_error is not None:
                    failure = rotation_error
            self._disconnect_client(error=self._error_label(failure))
            return False
        except (
            ValidationError,
            ValueError,
            TypeError,
            DaemonClientError,
            OSError,
        ) as error:
            self._disconnect_client(error=self._error_label(error))
            return False
        except BaseException as error:
            self._disconnect_client(error=self._error_label(error))
            return False

    def _try_rotate_abandoned_stream(self) -> BaseException | None:
        try:
            self._rotate_abandoned_stream()
        except BaseException as error:
            self._mark_emergency_failure(error)
            return error
        return None

    def _rotate_abandoned_stream(self) -> None:
        with self._capture_lock:
            old_bridge_instance_id = self._bridge_instance_id
            retained = self._retained
            records: list[BridgeRecordV1] = []
            if retained is not None:
                records.append(retained)
            if self._queue_head is not None:
                records.append(self._queue_head)
            with self.queue.mutex:
                records.extend(self.queue.queue)
            gap_capture_count = sum(
                span.last_capture_seq - span.first_capture_seq + 1
                for span in self._gap_spans
            )
            record_capture_count = sum(
                record.last_capture_seq - record.first_capture_seq + 1
                for record in records
            )
            abandoned_capture_count = gap_capture_count + record_capture_count
            ambiguous_capture_count = (
                retained.last_capture_seq - retained.first_capture_seq + 1
                if retained is not None and self._retained_attempted
                else 0
            )
            unsent_capture_count = abandoned_capture_count - ambiguous_capture_count
            newly_lost_observations = sum(
                record.last_capture_seq - record.first_capture_seq + 1
                for record in records
                if not isinstance(record, BridgeGapV1)
            )
            audit = MappingProxyType(
                {
                    "bridge_instance_id": old_bridge_instance_id,
                    "abandoned_capture_count": abandoned_capture_count,
                    "unsent_capture_count": unsent_capture_count,
                    "ambiguous_capture_count": ambiguous_capture_count,
                    "exact_replay_permitted": False,
                    "daemon_accounting": "unknown_capture_range_trace_gap",
                }
            )
            replacement_bridge_instance_id = new_id()
            replacement_recovery_token = secrets.token_hex(32)
            with self._health_lock:
                updated_health = replace(
                    self._health,
                    trace_complete=False,
                    dropped_events=(
                        self._health.dropped_events + newly_lost_observations
                    ),
                    pending_gap_ranges=0,
                    last_capture_seq=0,
                    through_capture_seq=0,
                    last_error="bridge_abandoned",
                    abandoned_streams=self._health.abandoned_streams + 1,
                    abandoned_local_records=(
                        self._health.abandoned_local_records + abandoned_capture_count
                    ),
                    ambiguous_records=(
                        self._health.ambiguous_records + ambiguous_capture_count
                    ),
                    last_abandonment=audit,
                )
            self.projections.clear()
            with self.queue.mutex:
                self.queue.queue.clear()
            self._bridge_instance_id = replacement_bridge_instance_id
            self._recovery_token = replacement_recovery_token
            self._next_capture_seq = 1
            self._gap_spans.clear()
            self._queue_head = None
            self._retained = None
            self._retained_attempted = False
            self._exact_retry_pending = False
            self._attached_next_capture_seq = None
            self._close_sealed = False
            self._close_final_capture_seq = None
            self._last_ack = None
            self._last_bootstrap_reservation = None
            with self._health_lock:
                self._health = updated_health

    def _publish_connection(
        self,
        connection: str,
        *,
        frame: ConsciousnessFrameV3 | None = None,
        error: str | None = None,
    ) -> None:
        try:
            with self._health_lock:
                through = (
                    frame.through_capture_seq
                    if frame is not None
                    else self._health.through_capture_seq
                )
                trace_complete = self._health.trace_complete
                if frame is not None:
                    trace_complete = trace_complete and frame.trace_complete
                updated_health = replace(
                    self._health,
                    connection=connection,
                    trace_complete=trace_complete,
                    through_capture_seq=through,
                    last_error=error,
                )
                self._health = updated_health
                self._emergency_connection_disconnected = False
        except BaseException as health_error:
            self._mark_emergency_failure(health_error)

    @staticmethod
    def _error_label(error: BaseException) -> str:
        try:
            if isinstance(error, DaemonRpcError):
                return error.code[:160]
            return type(error).__name__[:160]
        except BaseException:
            return "callback_internal"

    @staticmethod
    def _close_client(client: Any) -> None:
        with suppress(BaseException):
            client.close()

    def _disconnect_client(self, *, error: str | None = None) -> None:
        client = self._client
        self._client = None
        self._binding = None
        self._attached_next_capture_seq = None
        self._exact_retry_pending = False
        if client is not None:
            self._close_client(client)
        self._publish_connection("disconnected", error=error)

    def _wait_reconnect(self) -> None:
        self._wait_for_worker(self._reconnect_delay_seconds)

    def _wait_for_worker(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if self._worker_stop_requested() or self._consume_worker_wake():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            wait_seconds = min(remaining, _WORKER_CONTROL_POLL_SECONDS)
            failed = False
            started = time.monotonic()
            try:
                self._wake_event.wait(wait_seconds)
            except BaseException as error:
                failed = True
                self._mark_emergency_failure(error)
            try:
                self._wake_event.clear()
            except BaseException as error:
                failed = True
                self._mark_emergency_failure(error)
            if failed:
                elapsed = time.monotonic() - started
                with suppress(BaseException):
                    time.sleep(max(0.0, wait_seconds - elapsed))

    def request_clean_close(self) -> None:
        """Request a typed stream close; this never stops the daemon."""

        with self._capture_lock:
            self._close_requested = True
        self._notify_worker(force_start=True)

    def _seal_close_if_ready(self) -> bool:
        with self._capture_lock:
            if self._close_sealed:
                return True
            if (
                self._gap_spans
                or self._retained is not None
                or self._queue_head is not None
                or not self.queue.empty()
            ):
                return False
            self._close_final_capture_seq = self._next_capture_seq - 1
            self._close_sealed = True
            return True

    def _close_stream(self) -> bool:
        if self._client is None or self._binding is None:
            return False
        final = self._close_final_capture_seq
        if final is None:
            return False
        try:
            result = self._client.call(
                "bridge.close",
                {"binding": self._binding, "final_capture_seq": final},
            )
            stream = BridgeStreamState.model_validate_json(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                strict=True,
            )
            if (
                stream.bridge_instance_id != self._bridge_instance_id
                or stream.brain_id != self._brain_id
                or stream.status != "clean_closed"
                or stream.closed_final_seq != final
            ):
                raise DaemonClientError("bridge.close result is invalid")
            self._stop_requested = True
            self._stop_event.set()
            return True
        except (
            ValidationError,
            ValueError,
            TypeError,
            DaemonClientError,
            OSError,
        ) as error:
            self._disconnect_client(error=self._error_label(error))
            return False
        except BaseException as error:
            self._disconnect_client(error=self._error_label(error))
            return False

    def _recover_clean_close(self, client: Any, brain_id: str) -> bool:
        # A fresh connection is required because attach returned a typed terminal
        # state.  Recovery is intentionally attempted by the worker only.
        final = self._close_final_capture_seq
        if final is None:
            self._close_client(client)
            return False
        try:
            result = client.call(
                "bridge.close.recover",
                {
                    "brain_id": brain_id,
                    "bridge_instance_id": self._bridge_instance_id,
                    "recovery_token": self._recovery_token,
                    "final_capture_seq": final,
                },
            )
            stream = BridgeStreamState.model_validate_json(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                strict=True,
            )
            if (
                stream.bridge_instance_id != self._bridge_instance_id
                or stream.brain_id != brain_id
                or stream.status != "clean_closed"
                or stream.closed_final_seq != final
            ):
                raise DaemonClientError("bridge.close.recover result is invalid")
            self._close_client(client)
            self._stop_requested = True
            self._stop_event.set()
            self._publish_connection("disconnected")
            return True
        except BaseException as error:
            self._close_client(client)
            self._publish_connection("disconnected", error=self._error_label(error))
            return False

    def stop_worker_for_test(self) -> None:
        """Bounded test cleanup; it does not issue daemon.shutdown or stream close."""

        with self._worker_lifecycle_lock:
            with self._worker_lock:
                self._stop_requested = True
                try:
                    self._stop_event.set()
                except BaseException as error:
                    self._mark_emergency_failure(error)
                try:
                    with self._wake_lock:
                        self._wake_requested = True
                except BaseException as error:
                    self._mark_emergency_failure(error)
                try:
                    self._wake_event.set()
                except BaseException as error:
                    self._mark_emergency_failure(error)
                worker = self._worker
            if worker is None:
                return
            if worker is threading.current_thread():
                error = RuntimeError("bridge worker cannot join itself")
                self._mark_emergency_failure(error)
                raise error
            try:
                worker.join(timeout=2.0)
            except BaseException as error:
                self._mark_emergency_failure(error)
            try:
                alive = worker.is_alive()
            except BaseException as error:
                self._mark_emergency_failure(error)
                raise RuntimeError("bridge worker liveness is unknown") from error
            if alive:
                error = RuntimeError("bridge worker did not stop within the test bound")
                self._mark_emergency_failure(error)
                raise error
            with self._worker_lock:
                if self._worker is worker:
                    self._worker = None
            try:
                with self._health_lock:
                    self._health = replace(self._health, worker_started=False)
            except BaseException as error:
                self._mark_emergency_failure(error)


__all__ = [
    "DEFAULT_FRAME_REFRESH_SECONDS",
    "DEFAULT_QUEUE_CAPACITY",
    "SOURCE_SCHEMA_VERSION",
    "HookBridge",
    "default_runtime_home",
]
