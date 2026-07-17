"""Inert Hermes Agent plugin registration boundary."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, replace
from importlib import import_module, metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

APPROVED_HOOKS = (
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


SUPPORTED_HERMES = ">=0.18,<0.19"
PLUGIN_SKILL_NAME = "operating-alice-brain-hermes"
PLUGIN_SKILL_DESCRIPTION = (
    "Use when starting, observing, diagnosing, verifying, or tracing an "
    "Alice-brain-Hermes plugin runtime."
)
_SUPPORTED_HERMES_SPECIFIER = SpecifierSet(SUPPORTED_HERMES)
_REGISTRATION_STATE_ATTRIBUTE = "_alice_brain_hermes_registration_v1"
_REGISTRATION_LOCK = threading.RLock()

_BOOTSTRAP_QUEUE_CAPACITY = 256
_BOOTSTRAP_MAX_DEPTH = 6
_BOOTSTRAP_MAX_NODES = 1_024
_BOOTSTRAP_MAX_ITEMS = 64
_BOOTSTRAP_MAX_STRING_BYTES = 16_384
# The runtime persists the successor cursor in a signed SQLite INTEGER.  The
# final int64 value is therefore reserved for ``next_capture_seq``.  Keep this
# stdlib-only registration boundary inert instead of importing runtime models.
_BOOTSTRAP_MAX_CAPTURE_SEQUENCE = 2**63 - 2
_ENERGY_HEALTH_REPORT_INTERVAL_SECONDS = 1.0
_EMPTY_BOOTSTRAP_COPY_STATS: MappingProxyType[str, int] = MappingProxyType({})
_HOST_VALUE_UNRESOLVED = object()


class _HermesHostAccess:
    """Worker-only, once-resolved access to one Hermes plugin context."""

    def __init__(self, context: Any) -> None:
        self._context = context
        self._lock = threading.Lock()
        self._brain_profile: Any = _HOST_VALUE_UNRESOLVED
        self._llm: Any = _HOST_VALUE_UNRESOLVED

    def brain_profile(self) -> Any:
        """Resolve and cache the project profile only on an operational worker."""

        with self._lock:
            if self._brain_profile is _HOST_VALUE_UNRESOLVED:
                profile_name = self._context.profile_name
                from alice_brain_hermes.hermes.identity_client import (
                    hermes_brain_profile,
                )

                self._brain_profile = hermes_brain_profile(profile_name)
            return self._brain_profile

    def llm(self) -> Any:
        """Resolve and cache Hermes' host-owned LLM with its existing defaults."""

        with self._lock:
            if self._llm is _HOST_VALUE_UNRESOLVED:
                self._llm = self._context.llm
            return self._llm


@dataclass(frozen=True, slots=True)
class _BootstrapHealth:
    trace_complete: bool = True
    dropped_events: int = 0
    pending_records: int = 0
    pending_gap_ranges: int = 0
    late_after_close: int = 0
    worker_started: bool = False
    degraded: bool = False
    last_error: str | None = None
    worker_error: str | None = None
    energy_worker_started: bool = False
    energy_terminal_intent_pending: bool = False
    energy_worker_error: str | None = None
    registration_attempts: int = 0
    registration_failures: int = 0
    registration_complete: bool = False
    registered_hook_count: int = 0
    missing_hooks: tuple[str, ...] = APPROVED_HOOKS
    registration_error: str | None = None


@dataclass(frozen=True, slots=True)
class _DispatchReservation:
    """Identity proving which sequence belongs to one public callback."""

    buffer: _BootstrapCaptureBuffer
    capture_seq: int
    fallback_gap: _BootstrapCapture


@dataclass(slots=True)
class _DispatchAttempt:
    """Thread-local callback attempt; nested callbacks restore their parent."""

    reservation: _DispatchReservation | None = None
    notify_buffer: _BootstrapCaptureBuffer | None = None


_DISPATCH_STATE = threading.local()


def _active_dispatch_attempt() -> _DispatchAttempt | None:
    attempt = getattr(_DISPATCH_STATE, "attempt", None)
    return attempt if isinstance(attempt, _DispatchAttempt) else None


@dataclass(frozen=True, slots=True)
class _BootstrapCapture:
    capture_seq: int
    last_capture_seq: int
    hook: str | None
    detached_kwargs: MappingProxyType[str, object] | None
    copy_stats: MappingProxyType[str, int]
    gap_cause_counts: MappingProxyType[str, int] | None = None

    @property
    def capture_count(self) -> int:
        return self.last_capture_seq - self.capture_seq + 1

    @property
    def gap_cause(self) -> str | None:
        """Compatibility diagnostic for a single-cause interval."""

        causes = self.gap_cause_counts
        if causes is None or len(causes) != 1:
            return None
        return next(iter(causes))

    @property
    def is_gap(self) -> bool:
        return self.gap_cause_counts is not None


@dataclass(slots=True)
class _BootstrapCopyStats:
    nodes: int = 0
    truncated_paths: int = 0
    unsupported_paths: int = 0
    omitted_nodes: int = 0

    def frozen(self) -> MappingProxyType[str, int]:
        return MappingProxyType(
            {
                "truncated_paths": self.truncated_paths,
                "unsupported_paths": self.unsupported_paths,
                "omitted_nodes": self.omitted_nodes,
            }
        )


def _copy_bootstrap_text(value: str, stats: _BootstrapCopyStats) -> str:
    prefix = value[: _BOOTSTRAP_MAX_STRING_BYTES + 1]
    try:
        encoded = prefix.encode("utf-8", errors="strict")
    except UnicodeError:
        stats.unsupported_paths += 1
        return ""
    if (
        len(value) <= _BOOTSTRAP_MAX_STRING_BYTES
        and len(encoded) <= _BOOTSTRAP_MAX_STRING_BYTES
    ):
        return value
    stats.truncated_paths += 1
    return encoded[:_BOOTSTRAP_MAX_STRING_BYTES].decode("utf-8", errors="ignore")


def _copy_bootstrap_value(
    value: object,
    stats: _BootstrapCopyStats,
    *,
    depth: int = 1,
) -> object:
    stats.nodes += 1
    if stats.nodes > _BOOTSTRAP_MAX_NODES:
        stats.omitted_nodes += 1
        return {"$omitted": "node_budget"}
    if depth > _BOOTSTRAP_MAX_DEPTH:
        stats.omitted_nodes += 1
        return {"$omitted": "depth_budget"}
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        return _copy_bootstrap_text(value, stats)
    if type(value) is int:
        if -(2**63) <= value <= 2**63 - 1:
            return value
        stats.unsupported_paths += 1
        return {"$unsupported": "integer_out_of_range"}
    if type(value) is float:
        if math.isfinite(value):
            return value
        stats.unsupported_paths += 1
        return {"$unsupported": "non_finite_number"}
    if type(value) is dict:
        result: dict[str, object] = {}
        item_count = len(value)
        iterator = iter(value.items())
        for _index in range(min(item_count, _BOOTSTRAP_MAX_ITEMS)):
            key, child = next(iterator)
            if type(key) is not str:
                stats.unsupported_paths += 1
                continue
            result[_copy_bootstrap_text(key, stats)] = _copy_bootstrap_value(
                child,
                stats,
                depth=depth + 1,
            )
        if item_count > _BOOTSTRAP_MAX_ITEMS:
            stats.truncated_paths += 1
            stats.omitted_nodes += item_count - _BOOTSTRAP_MAX_ITEMS
        return result
    if type(value) in {list, tuple}:
        item_count = len(value)
        if item_count > _BOOTSTRAP_MAX_ITEMS:
            stats.truncated_paths += 1
            stats.omitted_nodes += item_count - _BOOTSTRAP_MAX_ITEMS
        return [
            _copy_bootstrap_value(child, stats, depth=depth + 1)
            for child in value[:_BOOTSTRAP_MAX_ITEMS]
        ]
    try:
        attributes = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        attributes = None
    if type(attributes) is dict:
        stats.unsupported_paths += 1
        return {
            "$object_type": type(value).__name__[:160],
            "fields": _copy_bootstrap_value(
                attributes,
                stats,
                depth=depth + 1,
            ),
        }
    stats.unsupported_paths += 1
    return {"$unsupported_type": type(value).__name__[:160]}


class _BootstrapCaptureBuffer:
    """Stdlib-only pre-runtime capture boundary used by registered callbacks."""

    def __init__(
        self,
        *,
        queue_capacity: int = _BOOTSTRAP_QUEUE_CAPACITY,
        start_worker_on_capture: bool = True,
    ) -> None:
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int):
            raise TypeError("queue_capacity must be an exact int")
        if not 1 <= queue_capacity <= 65_536:
            raise ValueError("queue_capacity must be between 1 and 65536")
        if type(start_worker_on_capture) is not bool:
            raise TypeError("start_worker_on_capture must be an exact bool")
        self._queue: queue.Queue[_BootstrapCapture] = queue.Queue(queue_capacity)
        self._overflow: deque[_BootstrapCapture] = deque()
        self._capture_lock = threading.RLock()
        self._worker_lifecycle_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._wake = threading.Event()
        self._next_capture_seq = 1
        self._queue_head: _BootstrapCapture | None = None
        self._worker_retained: _BootstrapCapture | None = None
        self._worker: threading.Thread | None = None
        self._transport_bridge: Any | None = None
        self._stop_event = threading.Event()
        self._stop_requested = False
        self._host_context: Any | None = None
        self._host_access: _HermesHostAccess | None = None
        self._host_profile_factory: Any | None = None
        self._host_llm_factory: Any | None = None
        self._identity_worker: Any | None = None
        self._energy_worker: Any | None = None
        self._start_worker_on_capture = start_worker_on_capture
        self._health = _BootstrapHealth()
        self._handed_off_dropped_events = 0
        self._health_reconciliation_required = False
        self._emergency_trace_incomplete = False
        self._emergency_degraded = False
        self._emergency_last_error: str | None = None
        self._context: str | None = None

    @property
    def health(self) -> _BootstrapHealth:
        with self._capture_lock:
            health = self._health
            worker_started = self.worker_started
            energy_error = health.energy_worker_error
            energy_pending = health.energy_terminal_intent_pending
            energy_diagnostic = energy_error or (
                "energy_terminal_intent_pending" if energy_pending else None
            )
            if (
                not self._health_reconciliation_required
                and not self._emergency_trace_incomplete
                and not self._emergency_degraded
                and energy_diagnostic is None
                and health.worker_started is worker_started
            ):
                return health
            if self._health_reconciliation_required:
                pending_gap_ranges, pending_dropped_events = (
                    self._pending_gap_metrics_locked()
                )
                dropped_events = (
                    self._handed_off_dropped_events + pending_dropped_events
                )
            else:
                pending_gap_ranges = health.pending_gap_ranges
                dropped_events = health.dropped_events
            return _BootstrapHealth(
                trace_complete=(
                    health.trace_complete
                    and not self._health_reconciliation_required
                    and not self._emergency_trace_incomplete
                ),
                dropped_events=dropped_events,
                pending_records=health.pending_records,
                pending_gap_ranges=pending_gap_ranges,
                late_after_close=health.late_after_close,
                worker_started=worker_started,
                degraded=(
                    health.degraded
                    or self._health_reconciliation_required
                    or self._emergency_degraded
                    or energy_diagnostic is not None
                ),
                last_error=(
                    self._emergency_last_error
                    or health.last_error
                    or energy_diagnostic
                    or "health_reconciliation_required"
                ),
                worker_error=health.worker_error or energy_diagnostic,
                energy_worker_started=health.energy_worker_started,
                energy_terminal_intent_pending=energy_pending,
                energy_worker_error=energy_error,
                registration_attempts=health.registration_attempts,
                registration_failures=health.registration_failures,
                registration_complete=health.registration_complete,
                registered_hook_count=health.registered_hook_count,
                missing_hooks=health.missing_hooks,
                registration_error=health.registration_error,
            )

    def read_context(self) -> str | None:
        return self._context

    def publish_context(self, context: str | None) -> None:
        self._context = context if type(context) is str and context else None

    def bind_host_context(self, context: Any) -> None:
        """Retain the opaque context without reading any lazy host property."""

        with self._worker_lock:
            if self._host_context is not context:
                self._host_access = None
                self._host_profile_factory = None
                self._host_llm_factory = None
            self._host_context = context

    def host_context_for_worker(self) -> Any:
        """Return the context only to the operational bootstrap worker."""

        context = self.bound_host_context_for_worker()
        if context is None:
            raise RuntimeError("Hermes host context is not registered")
        return context

    def bound_host_context_for_worker(self) -> Any | None:
        """Return a bound context, or ``None`` for direct bridge test seams."""

        with self._worker_lock:
            return self._host_context

    def host_factories_for_worker(self) -> tuple[Any, Any, Any] | None:
        """Return one process-owned access object and its exact bound factories."""

        with self._worker_lock:
            context = self._host_context
            if context is None:
                return None
            access = self._host_access
            if access is None:
                access = _HermesHostAccess(context)
                self._host_access = access
                self._host_profile_factory = access.brain_profile
                self._host_llm_factory = access.llm
            profile_factory = self._host_profile_factory
            llm_factory = self._host_llm_factory
            if not callable(profile_factory) or not callable(llm_factory):
                raise RuntimeError("Hermes host factories are unavailable")
            return access, profile_factory, llm_factory

    @property
    def identity_worker_for_test(self) -> Any | None:
        return self.identity_worker_for_worker()

    def identity_worker_for_worker(self) -> Any | None:
        with self._worker_lock:
            return self._identity_worker

    def _adopt_identity_worker(self, worker: Any) -> None:
        with self._worker_lock:
            current = self._identity_worker
            if current is not None and current is not worker:
                raise RuntimeError("bootstrap identity worker ownership changed")
            self._identity_worker = worker

    def _stop_identity_worker(
        self,
        worker: Any | None = None,
        *,
        release: bool = True,
    ) -> None:
        with self._worker_lock:
            owned = self._identity_worker
        if owned is None:
            return
        if worker is not None and owned is not worker:
            raise RuntimeError("bootstrap identity worker identity changed")
        try:
            stop = getattr(owned, "stop_for_test", None)
            if not callable(stop):
                raise RuntimeError("identity worker has no bounded stop operation")
            stop()
            strict_probe = getattr(owned, "_worker_alive_strict", None)
            if not callable(strict_probe):
                raise RuntimeError("identity worker has no strict liveness probe")
            try:
                alive = strict_probe()
            except BaseException as error:
                raise RuntimeError("identity worker liveness is unknown") from error
            if type(alive) is not bool:
                raise RuntimeError("identity worker liveness is invalid")
            if alive:
                raise RuntimeError("identity worker did not stop")
            if not release:
                return
            try:
                terminal_intent_pending = owned.terminal_intent_pending
            except BaseException as error:
                raise RuntimeError(
                    "identity worker terminal intent state is unknown"
                ) from error
            if type(terminal_intent_pending) is not bool:
                raise RuntimeError("identity worker terminal intent state is invalid")
            if terminal_intent_pending:
                raise RuntimeError("identity worker terminal intent remains pending")
        except BaseException as error:
            self.mark_worker_degraded(error.__cause__ or error)
            raise
        with self._worker_lock:
            if self._identity_worker is owned:
                self._identity_worker = None

    @property
    def energy_worker_for_test(self) -> Any | None:
        return self.energy_worker_for_worker()

    def energy_worker_for_worker(self) -> Any | None:
        with self._worker_lock:
            return self._energy_worker

    def _adopt_energy_worker(self, worker: Any) -> None:
        with self._worker_lock:
            current = self._energy_worker
            if current is not None and current is not worker:
                raise RuntimeError("bootstrap energy worker ownership changed")
            self._energy_worker = worker

    def _stop_energy_worker(
        self,
        worker: Any | None = None,
        *,
        release: bool = True,
    ) -> None:
        with self._worker_lock:
            owned = self._energy_worker
        if owned is None:
            return
        if worker is not None and owned is not worker:
            raise RuntimeError("bootstrap energy worker identity changed")
        try:
            stop = getattr(owned, "stop_for_test", None)
            if not callable(stop):
                raise RuntimeError("energy worker has no bounded stop operation")
            stop()
            strict_probe = getattr(owned, "_worker_alive_strict", None)
            if not callable(strict_probe):
                raise RuntimeError("energy worker has no strict liveness probe")
            try:
                alive = strict_probe()
            except BaseException as error:
                raise RuntimeError("energy worker liveness is unknown") from error
            if type(alive) is not bool:
                raise RuntimeError("energy worker liveness is invalid")
            if alive:
                raise RuntimeError("energy worker did not stop")
            try:
                terminal_intent_pending = owned.terminal_intent_pending
            except BaseException as error:
                raise RuntimeError(
                    "energy worker terminal intent state is unknown"
                ) from error
            if type(terminal_intent_pending) is not bool:
                raise RuntimeError("energy worker terminal intent state is invalid")
            try:
                internal_error = owned.last_internal_error_type
            except BaseException as error:
                raise RuntimeError("energy worker diagnostic is unknown") from error
            if internal_error is not None and not isinstance(internal_error, str):
                raise RuntimeError("energy worker diagnostic is invalid")
            self.publish_energy_worker_diagnostics(
                worker_started=False,
                terminal_intent_pending=terminal_intent_pending,
                error_type=internal_error,
            )
            if not release:
                return
        except BaseException as error:
            cause = error.__cause__ or error
            try:
                self.publish_energy_worker_diagnostics(
                    worker_started=False,
                    terminal_intent_pending=False,
                    error_type=type(cause).__name__[:160],
                )
            except BaseException as diagnostic_error:
                self.mark_worker_degraded(diagnostic_error)
            raise
        if terminal_intent_pending:
            error = RuntimeError("energy worker terminal intent remains pending")
            raise error
        with self._worker_lock:
            if self._energy_worker is owned:
                self._energy_worker = None

    def _stop_owned_children(self) -> None:
        errors: list[BaseException] = []
        for cleanup in (
            self._stop_identity_worker,
            self._stop_energy_worker,
            self._stop_transport_bridge,
        ):
            try:
                cleanup()
            except BaseException as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(
                "bootstrap child cleanup failed",
                errors,
            )

    def capture(self, hook: str, kwargs: dict[str, Any]) -> None:
        with self._capture_lock:
            if self._next_capture_seq > _BOOTSTRAP_MAX_CAPTURE_SEQUENCE:
                self._mark_capture_capacity_exhausted_locked()
                return
            reservation = self._reserve_locked()
            capture_seq = reservation.capture_seq
            try:
                if kwargs.get("telemetry_schema_version") != "hermes.observer.v1":
                    self._record_gap_locked(
                        capture_seq,
                        "invalid_source_schema",
                        fallback_gap=reservation.fallback_gap,
                    )
                else:
                    stats = _BootstrapCopyStats()
                    detached: dict[str, object] = {}
                    item_count = len(kwargs)
                    iterator = iter(kwargs.items())
                    for _index in range(min(item_count, _BOOTSTRAP_MAX_ITEMS)):
                        key, value = next(iterator)
                        if type(key) is not str:
                            stats.unsupported_paths += 1
                            continue
                        detached[_copy_bootstrap_text(key, stats)] = (
                            _copy_bootstrap_value(value, stats)
                        )
                    if item_count > _BOOTSTRAP_MAX_ITEMS:
                        stats.truncated_paths += 1
                        stats.omitted_nodes += item_count - _BOOTSTRAP_MAX_ITEMS
                    capture = _BootstrapCapture(
                        capture_seq=capture_seq,
                        last_capture_seq=capture_seq,
                        hook=hook,
                        detached_kwargs=MappingProxyType(detached),
                        copy_stats=stats.frozen(),
                    )
                    try:
                        self._queue.put_nowait(capture)
                    except queue.Full:
                        self._record_gap_locked(
                            capture_seq,
                            "queue_full",
                            fallback_gap=reservation.fallback_gap,
                        )
                    except BaseException:
                        self._record_gap_locked(
                            capture_seq,
                            "callback_internal",
                            fallback_gap=reservation.fallback_gap,
                        )
            except BaseException:
                self._record_gap_locked(
                    capture_seq,
                    "callback_internal",
                    fallback_gap=reservation.fallback_gap,
                )
        self._defer_or_notify_worker()

    def _reserve_locked(self) -> _DispatchReservation:
        if self._next_capture_seq > _BOOTSTRAP_MAX_CAPTURE_SEQUENCE:
            self._mark_capture_capacity_exhausted_locked()
            raise OverflowError("bootstrap capture sequence capacity is exhausted")
        attempt = _active_dispatch_attempt()
        if attempt is not None and attempt.reservation is not None:
            raise RuntimeError("one Hermes callback may reserve only one capture")
        capture_seq = self._next_capture_seq
        fallback_gap = self._new_gap(capture_seq, "callback_internal")
        reservation = _DispatchReservation(self, capture_seq, fallback_gap)
        next_capture_seq = capture_seq + 1
        updated_health = replace(
            self._health,
            pending_records=self._health.pending_records + 1,
        )
        if attempt is not None:
            attempt.reservation = reservation
        self._next_capture_seq = next_capture_seq
        self._health = updated_health
        if attempt is not None:
            attempt.notify_buffer = self
        return reservation

    def _mark_capture_capacity_exhausted_locked(self) -> None:
        """Publish an exact fail-closed latch without reserving a false record."""

        self._emergency_trace_incomplete = True
        self._emergency_degraded = True
        self._emergency_last_error = "capture_sequence_capacity_exhausted"

    def record_dispatch_failure(
        self,
        reservation: _DispatchReservation | None = None,
    ) -> None:
        """Retain one exact gap, reusing this callback's reservation if present."""

        try:
            with self._capture_lock:
                if reservation is None:
                    reservation = self._reserve_locked()
                elif reservation.buffer is not self:
                    raise RuntimeError("dispatch reservation belongs to another buffer")
                self._record_gap_locked(
                    reservation.capture_seq,
                    "callback_internal",
                    fallback_gap=reservation.fallback_gap,
                )
        except BaseException as error:
            self.mark_worker_degraded(error)
            return
        self._defer_or_notify_worker()

    def _defer_or_notify_worker(self) -> None:
        attempt = _active_dispatch_attempt()
        if (
            attempt is not None
            and attempt.reservation is not None
            and attempt.reservation.buffer is self
        ):
            attempt.notify_buffer = self
            return
        self._notify_worker()

    def _replace_pending_with_gap_locked(
        self,
        gap: _BootstrapCapture,
    ) -> str:
        """Return ``replaced``, ``gap``, or ``missing`` for one reserved sequence."""

        capture_seq = gap.capture_seq

        def replacement_state(existing: _BootstrapCapture) -> str | None:
            if not existing.capture_seq <= capture_seq <= existing.last_capture_seq:
                return None
            if existing.is_gap:
                return "gap"
            if (
                existing.capture_seq != capture_seq
                or existing.last_capture_seq != capture_seq
            ):
                raise RuntimeError("observation capture range is not singular")
            return "replaced"

        for attribute in ("_worker_retained", "_queue_head"):
            existing = getattr(self, attribute)
            if existing is None:
                continue
            state = replacement_state(existing)
            if state == "replaced":
                setattr(self, attribute, gap)
                return state
            if state == "gap":
                return state

        with self._queue.mutex:
            for index, existing in enumerate(self._queue.queue):
                state = replacement_state(existing)
                if state == "replaced":
                    self._queue.queue[index] = gap
                    return state
                if state == "gap":
                    return state

        for existing in self._overflow:
            state = replacement_state(existing)
            if state is not None:
                return state
        return "missing"

    def _retain_new_gap_locked(self, gap: _BootstrapCapture) -> int:
        """Retain a new gap and return its contribution to gap-range count."""

        try:
            self._queue.put_nowait(gap)
            return 1
        except BaseException:
            if self._replace_pending_with_gap_locked(gap) == "gap":
                # A hostile put implementation may insert and then raise.
                return 1
        if (
            self._overflow
            and self._overflow[-1].last_capture_seq + 1 == gap.capture_seq
        ):
            try:
                previous = self._overflow[-1]
                causes = dict(previous.gap_cause_counts or {})
                for cause, count in (gap.gap_cause_counts or {}).items():
                    causes[cause] = causes.get(cause, 0) + count
                merged = replace(
                    previous,
                    last_capture_seq=gap.last_capture_seq,
                    gap_cause_counts=MappingProxyType(causes),
                )
            except BaseException:
                # The exact fallback is already constructed for this sequence.
                # Retaining a second adjacent range is safer than losing it
                # merely because the optional merge allocation failed.
                self._overflow.append(gap)
                return 1
            self._overflow[-1] = merged
            return 0
        self._overflow.append(gap)
        return 1

    @staticmethod
    def _new_gap(capture_seq: int, cause: str) -> _BootstrapCapture:
        return _BootstrapCapture(
            capture_seq=capture_seq,
            last_capture_seq=capture_seq,
            hook=None,
            detached_kwargs=None,
            copy_stats=_EMPTY_BOOTSTRAP_COPY_STATS,
            gap_cause_counts=MappingProxyType({cause: 1}),
        )

    def _record_gap_locked(
        self,
        capture_seq: int,
        cause: str,
        *,
        fallback_gap: _BootstrapCapture | None = None,
    ) -> None:
        try:
            gap = self._new_gap(capture_seq, cause)
        except BaseException:
            if fallback_gap is None or fallback_gap.capture_seq != capture_seq:
                raise
            gap = fallback_gap
        self._health_reconciliation_required = True
        pending_state = self._replace_pending_with_gap_locked(gap)
        if pending_state == "missing":
            self._retain_new_gap_locked(gap)
        self._reconcile_gap_health_locked()

    def _pending_gap_metrics_locked(
        self,
        *,
        exclude: _BootstrapCapture | None = None,
    ) -> tuple[int, int]:
        items = [self._worker_retained, self._queue_head]
        with self._queue.mutex:
            items.extend(self._queue.queue)
        items.extend(self._overflow)
        pending_gap_ranges = 0
        pending_dropped_events = 0
        for item in items:
            if item is None or item is exclude or not item.is_gap:
                continue
            pending_gap_ranges += 1
            pending_dropped_events += item.capture_count
        return pending_gap_ranges, pending_dropped_events

    def _reconcile_gap_health_locked(self) -> None:
        pending_gap_ranges, pending_dropped_events = self._pending_gap_metrics_locked()
        dropped_events = self._handed_off_dropped_events + pending_dropped_events
        self._health = replace(
            self._health,
            trace_complete=(
                self._health.trace_complete
                and dropped_events == 0
                and self._health.registration_failures == 0
            ),
            dropped_events=dropped_events,
            pending_gap_ranges=pending_gap_ranges,
        )
        self._health_reconciliation_required = False

    def _notify_worker(self) -> None:
        try:
            self._wake.set()
        except BaseException as error:
            self.mark_worker_degraded(error)
        if self._start_worker_on_capture:
            self._start_worker()

    def _start_worker(self) -> None:
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
                    self.mark_unrepresented_callback(error)
                    return
                self._worker = None
            try:
                if self._stop_requested:
                    self._stop_event.clear()
                    self._stop_requested = False
                with self._capture_lock:
                    previous_health = self._health
                    updated_health = replace(self._health, worker_started=True)
                    worker = threading.Thread(
                        target=_bootstrap_worker_entry,
                        args=(self,),
                        name="alice-brain-hermes-bootstrap",
                        daemon=True,
                    )
                    self._worker = worker
                    try:
                        worker.start()
                    except BaseException as start_error:
                        try:
                            started = worker.is_alive()
                        except BaseException:
                            # The start result is unknown.  Keep the ownership
                            # pointer so a possibly live worker cannot be joined
                            # by a duplicate bootstrap worker.
                            self._health = updated_health
                            self.mark_unrepresented_callback(start_error)
                            self.mark_worker_degraded(start_error)
                            return
                        if not started:
                            self._worker = None
                            self._health = previous_health
                            raise
                        self._health = updated_health
                        self.mark_unrepresented_callback(start_error)
                        self.mark_worker_degraded(start_error)
                        return
                    self._health = updated_health
            except BaseException as error:
                with self._capture_lock:
                    self.mark_unrepresented_callback(error)
                self.mark_worker_degraded(error)
                return

    @property
    def worker_started(self) -> bool:
        worker = self._worker
        if worker is None:
            return False
        try:
            return worker.is_alive()
        except BaseException as error:
            self.mark_unrepresented_callback(error)
            return False

    def next_for_worker(self) -> _BootstrapCapture | None:
        with self._capture_lock:
            if self._worker_retained is not None:
                return self._worker_retained
            if self._queue_head is None:
                try:
                    self._queue_head = self._queue.get_nowait()
                except queue.Empty:
                    self._queue_head = None
            overflow = self._overflow[0] if self._overflow else None
            if overflow is not None and (
                self._queue_head is None
                or overflow.capture_seq < self._queue_head.capture_seq
            ):
                self._worker_retained = self._overflow.popleft()
            elif self._queue_head is not None:
                self._worker_retained = self._queue_head
                self._queue_head = None
            return self._worker_retained

    def mark_handed_off(self, capture: _BootstrapCapture) -> None:
        with self._capture_lock:
            if self._worker_retained is not capture:
                raise RuntimeError("bootstrap handoff identity changed")
            handed_off_dropped_events = self._handed_off_dropped_events
            if capture.is_gap:
                handed_off_dropped_events += capture.capture_count
            pending_gap_ranges, pending_dropped_events = (
                self._pending_gap_metrics_locked(exclude=capture)
            )
            dropped_events = handed_off_dropped_events + pending_dropped_events
            updated_health = replace(
                self._health,
                trace_complete=(
                    self._health.trace_complete
                    and dropped_events == 0
                    and self._health.registration_failures == 0
                ),
                dropped_events=dropped_events,
                pending_records=self._health.pending_records - capture.capture_count,
                pending_gap_ranges=pending_gap_ranges,
                degraded=self._health.registration_failures > 0,
                last_error=self._health.registration_error,
                worker_error=None,
            )
            self._handed_off_dropped_events = handed_off_dropped_events
            self._worker_retained = None
            self._health = updated_health
            self._health_reconciliation_required = False

    def mark_late_after_close(self, capture: _BootstrapCapture) -> None:
        with self._capture_lock:
            if self._worker_retained is not capture:
                raise RuntimeError("bootstrap terminal disposition identity changed")
            terminal_dropped_events = (
                self._handed_off_dropped_events + capture.capture_count
            )
            pending_gap_ranges, pending_dropped_events = (
                self._pending_gap_metrics_locked(exclude=capture)
            )
            updated_health = replace(
                self._health,
                trace_complete=False,
                dropped_events=terminal_dropped_events + pending_dropped_events,
                pending_records=self._health.pending_records - capture.capture_count,
                pending_gap_ranges=pending_gap_ranges,
                late_after_close=(
                    self._health.late_after_close + capture.capture_count
                ),
                degraded=True,
                last_error="capture_after_close_seal",
                worker_error=None,
            )
            self._handed_off_dropped_events = terminal_dropped_events
            self._worker_retained = None
            self._health = updated_health
            self._health_reconciliation_required = False

    def mark_worker_degraded(self, error: BaseException) -> None:
        try:
            error_name = type(error).__name__[:160]
        except BaseException:
            error_name = "callback_internal"
        try:
            with self._capture_lock:
                self._health = replace(
                    self._health,
                    degraded=True,
                    last_error=error_name,
                    worker_error=error_name,
                )
        except BaseException:
            self.mark_unrepresented_callback(error)

    def publish_energy_worker_diagnostics(
        self,
        *,
        worker_started: bool,
        terminal_intent_pending: bool,
        error_type: str | None,
    ) -> None:
        """Publish source-specific energy state that capture success cannot erase."""

        if type(worker_started) is not bool:
            raise TypeError("energy worker started state must be an exact bool")
        if type(terminal_intent_pending) is not bool:
            raise TypeError("energy terminal intent state must be an exact bool")
        if error_type is not None and (
            not isinstance(error_type, str)
            or not error_type
            or len(error_type) > 160
            or any(
                not (character.isascii() and (character.isalnum() or character == "_"))
                for character in error_type
            )
        ):
            raise ValueError("energy worker error type is invalid")
        with self._capture_lock:
            self._health = replace(
                self._health,
                energy_worker_started=worker_started,
                energy_terminal_intent_pending=terminal_intent_pending,
                energy_worker_error=error_type,
            )

    def mark_unrepresented_callback(self, error: BaseException) -> None:
        """Publish an allocation-free conservative latch for an unseen callback."""

        self._emergency_trace_incomplete = True
        self._emergency_degraded = True
        try:
            self._emergency_last_error = type(error).__name__[:160]
        except BaseException:
            self._emergency_last_error = "callback_internal"

    def record_registration_success(self) -> None:
        """Record one complete context without erasing earlier partial installs."""

        with self._capture_lock:
            attempts = self._health.registration_attempts + 1
            if self._health.registration_failures:
                self._health = replace(
                    self._health,
                    registration_attempts=attempts,
                )
                return
            self._health = replace(
                self._health,
                registration_attempts=attempts,
                registration_complete=True,
                registered_hook_count=len(APPROVED_HOOKS),
                missing_hooks=(),
                registration_error=None,
            )

    def record_registration_failure(
        self,
        registered_hook_count: int,
        error: BaseException,
    ) -> None:
        """Persist bounded append-only hook coverage; this is not a trace gap."""

        if not 0 <= registered_hook_count <= len(APPROVED_HOOKS):
            raise ValueError("registered hook count is outside the approved range")
        error_name = type(error).__name__[:160]
        current_missing = APPROVED_HOOKS[registered_hook_count:]
        with self._capture_lock:
            health = self._health
            if health.registration_failures:
                confirmed_count = min(
                    health.registered_hook_count,
                    registered_hook_count,
                )
                missing_set = set(health.missing_hooks)
                missing_set.update(current_missing)
                missing_hooks = tuple(
                    hook for hook in APPROVED_HOOKS if hook in missing_set
                )
            else:
                confirmed_count = registered_hook_count
                missing_hooks = current_missing
            self._health = replace(
                health,
                trace_complete=False,
                degraded=True,
                last_error=health.worker_error or error_name,
                registration_attempts=health.registration_attempts + 1,
                registration_failures=health.registration_failures + 1,
                registration_complete=False,
                registered_hook_count=confirmed_count,
                missing_hooks=missing_hooks,
                registration_error=error_name,
            )

    def wait(self, timeout: float) -> None:
        failed = False
        try:
            self._wake.wait(timeout)
        except BaseException as error:
            failed = True
            self.mark_worker_degraded(error)
        try:
            self._wake.clear()
        except BaseException as error:
            failed = True
            self.mark_worker_degraded(error)
        if failed:
            with suppress(BaseException):
                time.sleep(min(timeout, 0.05))

    def worker_stop_requested(self) -> bool:
        if self._stop_requested:
            return True
        try:
            return self._stop_event.is_set()
        except BaseException as error:
            self.mark_worker_degraded(error)
            return False

    def _publish_worker_exit(self, worker: threading.Thread) -> None:
        try:
            with self._worker_lock:
                if self._worker is worker:
                    self._worker = None
                with self._capture_lock:
                    self._health = replace(self._health, worker_started=False)
        except BaseException as error:
            self.mark_unrepresented_callback(error)

    def _adopt_transport_bridge(self, bridge: Any) -> None:
        with self._worker_lock:
            current = self._transport_bridge
            if current is not None and current is not bridge:
                raise RuntimeError("bootstrap transport bridge ownership changed")
            self._transport_bridge = bridge

    def _stop_transport_bridge(self, bridge: Any | None = None) -> None:
        with self._worker_lock:
            owned = self._transport_bridge
        if owned is None:
            return
        if bridge is not None and owned is not bridge:
            raise RuntimeError("bootstrap transport bridge identity changed")

        try:
            stop = getattr(owned, "stop_worker_for_test", None)
            if not callable(stop):
                raise RuntimeError("bootstrap transport bridge has no callable stop")
            stop()
            strict_probe = getattr(owned, "_worker_alive_strict", None)
            if not callable(strict_probe):
                raise RuntimeError(
                    "bootstrap transport bridge has no strict liveness probe"
                )
            try:
                alive = strict_probe()
            except BaseException as error:
                raise RuntimeError(
                    "bootstrap transport bridge liveness is unknown"
                ) from error
            if alive:
                raise RuntimeError("bootstrap transport bridge did not stop")
        except BaseException as error:
            self.mark_worker_degraded(error.__cause__ or error)
            raise
        with self._worker_lock:
            if self._transport_bridge is owned:
                self._transport_bridge = None

    def stop_worker_for_test(self) -> None:
        with self._worker_lifecycle_lock:
            with self._worker_lock:
                self._stop_requested = True
                try:
                    self._stop_event.set()
                except BaseException as error:
                    self.mark_worker_degraded(error)
                try:
                    self._wake.set()
                except BaseException as error:
                    self.mark_worker_degraded(error)
                worker = self._worker
            if worker is not None:
                if worker is threading.current_thread():
                    error = RuntimeError("bootstrap worker cannot join itself")
                    self.mark_worker_degraded(error)
                    raise error
                try:
                    worker.join(timeout=4.0)
                except BaseException as error:
                    self.mark_worker_degraded(error)
                try:
                    alive = worker.is_alive()
                except BaseException as error:
                    self.mark_worker_degraded(error)
                    raise RuntimeError(
                        "bootstrap worker liveness is unknown"
                    ) from error
                if alive:
                    error = RuntimeError(
                        "bootstrap worker did not stop within the test bound"
                    )
                    self.mark_worker_degraded(error)
                    raise error
                with self._worker_lock:
                    if self._worker is worker:
                        self._worker = None
                try:
                    with self._capture_lock:
                        self._health = replace(
                            self._health,
                            worker_started=False,
                        )
                except BaseException as error:
                    self.mark_worker_degraded(error)
            self._stop_owned_children()

    def pending_for_test(self) -> tuple[_BootstrapCapture, ...]:
        with self._capture_lock, self._queue.mutex:
            queued = list(self._queue.queue)
            items = [
                item
                for item in (self._worker_retained, self._queue_head)
                if item is not None
            ]
            items.extend(queued)
            items.extend(self._overflow)
            return tuple(sorted(items, key=lambda item: item.capture_seq))


def _configure_identity_worker(
    buffer: _BootstrapCaptureBuffer,
    *,
    runtime_home: Any,
    profile_factory: Any,
    llm_factory: Any,
) -> None:
    """Create the optional naming worker without resolving either host value."""

    from os import environ

    from alice_brain_hermes.hermes.identity import (
        IdentityLlmMode,
        IdentityNamingWorker,
        read_identity_llm_mode,
    )

    mode = read_identity_llm_mode(environ)
    if mode is IdentityLlmMode.OFF:
        return

    from alice_brain_hermes.hermes.identity_client import (
        DaemonIdentityNamingLeasePort,
    )

    port = DaemonIdentityNamingLeasePort(
        runtime_home,
        profile_factory=profile_factory,
    )
    worker = IdentityNamingWorker(
        mode=mode,
        lease_port=port,
        llm_factory=llm_factory,
    )
    adopt = getattr(buffer, "_adopt_identity_worker", None)
    if not callable(adopt):
        raise RuntimeError("bootstrap cannot own the identity naming worker")
    adopt(worker)
    # Once adopted, this exact object owns any RAM terminal intent it may
    # create. A prelaunch failure, ambiguous start, or fast fatal exit must
    # never make bootstrap replace or clear it; the monitor restarts it in place.
    worker.start()


def _ensure_identity_worker_running(buffer: _BootstrapCaptureBuffer) -> bool:
    """Restart the same dead worker while preserving ambiguous ownership."""

    worker = buffer.identity_worker_for_worker()
    if worker is None:
        return True
    try:
        if worker.worker_started:
            return True
    except BaseException as error:
        buffer.mark_worker_degraded(error)
        return False
    try:
        worker.start()
    except BaseException as error:
        # IdentityNamingWorker retains an alive or unprobeable thread owner.
        # Never replace that object: its RAM terminal intent is authoritative.
        buffer.mark_worker_degraded(error)
        return False
    try:
        if worker.worker_started:
            return True
    except BaseException as error:
        buffer.mark_worker_degraded(error)
        return False
    buffer.mark_worker_degraded(
        RuntimeError("Hermes identity naming worker did not restart")
    )
    return False


def _configure_energy_worker(
    buffer: _BootstrapCaptureBuffer,
    *,
    runtime_home: Any,
    profile_factory: Any,
    llm_factory: Any,
) -> None:
    """Create the independent energy worker without resolving host values."""

    from alice_brain_hermes.hermes.energy import EnergyAssessmentWorker
    from alice_brain_hermes.hermes.energy_client import (
        DaemonEnergyAssessmentLeasePort,
    )

    port = DaemonEnergyAssessmentLeasePort(
        runtime_home,
        profile_factory=profile_factory,
    )
    worker = EnergyAssessmentWorker(
        lease_port=port,
        llm_factory=llm_factory,
    )
    adopt = getattr(buffer, "_adopt_energy_worker", None)
    if not callable(adopt):
        raise RuntimeError("bootstrap cannot own the energy assessment worker")
    adopt(worker)
    # The adopted object retains any terminal intent across daemon ACK loss.
    # It must be restarted in place, never replaced by another host LLM call.
    worker.start()


def _ensure_energy_worker_running(buffer: _BootstrapCaptureBuffer) -> bool:
    """Restart one exact energy owner and surface its sanitized diagnostic."""

    worker = buffer.energy_worker_for_worker()
    if worker is None:
        return True

    def publish_running_diagnostics() -> None:
        internal_error = worker.last_internal_error_type
        terminal_intent_pending = worker.terminal_intent_pending
        buffer.publish_energy_worker_diagnostics(
            worker_started=True,
            terminal_intent_pending=terminal_intent_pending,
            error_type=internal_error,
        )

    def publish_failure(error: BaseException) -> None:
        try:
            buffer.publish_energy_worker_diagnostics(
                worker_started=False,
                terminal_intent_pending=False,
                error_type=type(error).__name__[:160],
            )
        except BaseException as diagnostic_error:
            buffer.mark_worker_degraded(diagnostic_error)

    try:
        if worker.worker_started:
            publish_running_diagnostics()
            return True
    except BaseException as error:
        publish_failure(error)
        return False
    try:
        worker.start()
    except BaseException as error:
        publish_failure(error)
        return False
    try:
        if worker.worker_started:
            publish_running_diagnostics()
            return True
    except BaseException as error:
        publish_failure(error)
        return False
    buffer.publish_energy_worker_diagnostics(
        worker_started=False,
        terminal_intent_pending=False,
        error_type="RuntimeError",
    )
    return False


def _bootstrap_worker_entry(buffer: _BootstrapCaptureBuffer) -> None:
    worker = buffer._worker
    if worker is None:
        buffer.mark_unrepresented_callback(
            RuntimeError("bootstrap worker started without a published identity")
        )
        return
    try:
        _bootstrap_worker_main(buffer)
    finally:
        identity_cleanup = getattr(buffer, "_stop_identity_worker", None)
        energy_cleanup = getattr(buffer, "_stop_energy_worker", None)
        transport_cleanup = getattr(buffer, "_stop_transport_bridge", None)
        cleanups = (
            (
                identity_cleanup,
                {"release": False},
                True,
            ),
            (
                energy_cleanup,
                {"release": False},
                False,
            ),
            (transport_cleanup, {}, True),
        )
        for cleanup, kwargs, publish_generic_error in cleanups:
            if not callable(cleanup):
                continue
            try:
                cleanup(**kwargs)
            except BaseException as error:
                if publish_generic_error:
                    buffer.mark_worker_degraded(error.__cause__ or error)
        buffer._publish_worker_exit(worker)


def _bootstrap_worker_main(buffer: _BootstrapCaptureBuffer) -> None:
    bridge: Any | None = None
    runtime_home: Any = None
    hook_bridge_factory: Any = None
    shared_profile_factory: Any | None = None
    shared_llm_factory: Any | None = None
    identity_configured = False
    identity_retry_after = 0.0
    energy_configured = False
    energy_retry_after = 0.0
    energy_health_port: Any | None = None
    energy_health_retry_after = 0.0
    energy_health_next_report = 0.0
    last_energy_health_report: tuple[bool, bool, str | None] | None = None
    prelude_ready = False
    while True:
        try:
            stop_probe = getattr(buffer, "worker_stop_requested", None)
            if stop_probe is not None and stop_probe():
                return
            if not prelude_ready:
                from alice_brain_hermes.hermes.bridge import (
                    HookBridge,
                    default_runtime_home,
                )

                adopted_bridge = getattr(buffer, "_transport_bridge", None)
                candidate_runtime_home = getattr(
                    adopted_bridge,
                    "runtime_home",
                    None,
                )
                if candidate_runtime_home is None:
                    candidate_runtime_home = default_runtime_home()
                host_factories = None
                host_factory_access = getattr(
                    buffer,
                    "host_factories_for_worker",
                    None,
                )
                if callable(host_factory_access):
                    host_factories = host_factory_access()
                candidate_profile_factory = None
                candidate_llm_factory = None
                if host_factories is not None:
                    (
                        _host_access,
                        candidate_profile_factory,
                        candidate_llm_factory,
                    ) = host_factories
                identity_owner = getattr(
                    buffer,
                    "identity_worker_for_worker",
                    None,
                )
                candidate_identity_configured = (
                    callable(identity_owner) and identity_owner() is not None
                )
                energy_owner = getattr(
                    buffer,
                    "energy_worker_for_worker",
                    None,
                )
                candidate_energy_configured = (
                    callable(energy_owner) and energy_owner() is not None
                )
                bridge = adopted_bridge
                runtime_home = candidate_runtime_home
                hook_bridge_factory = HookBridge
                shared_profile_factory = candidate_profile_factory
                shared_llm_factory = candidate_llm_factory
                identity_configured = candidate_identity_configured
                energy_configured = candidate_energy_configured
                prelude_ready = True
            terminal = False
            if bridge is None:
                candidate: Any | None = None
                try:
                    bridge_arguments: dict[str, Any] = {
                        "start_worker_on_capture": False,
                        "context_sink": buffer.publish_context,
                    }
                    if shared_profile_factory is not None:
                        bridge_arguments["profile_factory"] = shared_profile_factory
                    candidate = hook_bridge_factory(
                        runtime_home,
                        **bridge_arguments,
                    )
                    adopt = getattr(buffer, "_adopt_transport_bridge", None)
                    if adopt is not None:
                        adopt(candidate)
                    candidate.start_worker()
                    if not candidate.worker_started:
                        raise RuntimeError("Hermes bridge worker did not start")
                    bridge = candidate
                except BaseException as error:
                    if candidate is not None:
                        stop_transport = getattr(
                            buffer,
                            "_stop_transport_bridge",
                            None,
                        )
                        if stop_transport is not None:
                            try:
                                stop_transport(candidate)
                            except BaseException as cleanup_error:
                                buffer.mark_worker_degraded(cleanup_error)
                    bridge = getattr(buffer, "_transport_bridge", None)
                    buffer.mark_worker_degraded(error)
                    buffer.wait(0.1)
                    continue
            else:
                terminal = getattr(bridge, "close_sealed", False)
                if type(terminal) is not bool:
                    raise RuntimeError("Hermes bridge terminal state is invalid")
                if not terminal and not bridge.worker_started:
                    try:
                        bridge.start_worker()
                        if not bridge.worker_started:
                            raise RuntimeError("Hermes bridge worker did not restart")
                    except BaseException as error:
                        buffer.mark_worker_degraded(error)
                        buffer.wait(0.1)
                        continue
            current_time = time.monotonic()
            if (
                shared_llm_factory is not None
                and energy_health_port is None
                and current_time >= energy_health_retry_after
            ):
                try:
                    from alice_brain_hermes.hermes.energy_client import (
                        DaemonEnergyWorkerHealthPort,
                    )

                    energy_health_port = DaemonEnergyWorkerHealthPort(runtime_home)
                except BaseException as error:
                    buffer.mark_worker_degraded(error)
                    energy_health_retry_after = current_time + 0.1
            if (
                shared_llm_factory is not None
                and not identity_configured
                and current_time >= identity_retry_after
            ):
                try:
                    _configure_identity_worker(
                        buffer,
                        runtime_home=runtime_home,
                        profile_factory=shared_profile_factory,
                        llm_factory=shared_llm_factory,
                    )
                except BaseException as error:
                    buffer.mark_worker_degraded(error)
                    identity_configured = (
                        buffer.identity_worker_for_worker() is not None
                    )
                    identity_retry_after = current_time + 0.1
                else:
                    # Includes the explicit, permanent operator opt-out.
                    identity_configured = True
            if (
                identity_configured
                and current_time >= identity_retry_after
                and not _ensure_identity_worker_running(buffer)
            ):
                identity_retry_after = current_time + 0.1
            if (
                shared_llm_factory is not None
                and not energy_configured
                and current_time >= energy_retry_after
            ):
                try:
                    _configure_energy_worker(
                        buffer,
                        runtime_home=runtime_home,
                        profile_factory=shared_profile_factory,
                        llm_factory=shared_llm_factory,
                    )
                except BaseException as error:
                    try:
                        buffer.publish_energy_worker_diagnostics(
                            worker_started=False,
                            terminal_intent_pending=False,
                            error_type=type(error).__name__[:160],
                        )
                    except BaseException as diagnostic_error:
                        buffer.mark_worker_degraded(diagnostic_error)
                    energy_configured = buffer.energy_worker_for_worker() is not None
                    energy_retry_after = current_time + 0.1
                else:
                    energy_configured = True
            if (
                energy_configured
                and current_time >= energy_retry_after
                and not _ensure_energy_worker_running(buffer)
            ):
                energy_retry_after = current_time + 0.1
            if energy_health_port is not None:
                health = buffer.health
                energy_diagnostics = (
                    health.energy_worker_started,
                    health.energy_terminal_intent_pending,
                    health.energy_worker_error,
                )
                if (
                    energy_diagnostics != last_energy_health_report
                    or current_time >= energy_health_next_report
                ):
                    try:
                        accepted = energy_health_port.report(
                            worker_started=energy_diagnostics[0],
                            terminal_intent_pending=energy_diagnostics[1],
                            error_type=energy_diagnostics[2],
                        )
                        if accepted is not True:
                            raise RuntimeError(
                                "daemon rejected the energy worker health report"
                            )
                    except BaseException:
                        # The daemon will expose this transport break as
                        # unreported/stale. Do not overwrite the underlying
                        # worker/capture diagnostic with a reporter symptom.
                        energy_health_next_report = current_time + 0.1
                    else:
                        last_energy_health_report = energy_diagnostics
                        energy_health_next_report = (
                            current_time + _ENERGY_HEALTH_REPORT_INTERVAL_SECONDS
                        )
            capture = buffer.next_for_worker()
            if capture is None:
                buffer.publish_context(bridge.projections.read_context())
                buffer.wait(0.05)
                continue
            try:
                disposition = bridge.capture_reserved(
                    hook=capture.hook,
                    detached_kwargs=capture.detached_kwargs,
                    first_capture_seq=capture.capture_seq,
                    last_capture_seq=capture.last_capture_seq,
                    gap_cause_counts=capture.gap_cause_counts,
                    copy_stats=capture.copy_stats,
                )
            except BaseException as error:
                buffer.mark_worker_degraded(error)
                buffer.wait(0.1)
                continue
            try:
                if disposition == "late_after_close":
                    buffer.mark_late_after_close(capture)
                elif disposition == "accepted":
                    buffer.mark_handed_off(capture)
                else:
                    raise RuntimeError("Hermes bridge disposition is invalid")
            except BaseException as error:
                buffer.mark_worker_degraded(error)
                buffer.wait(0.1)
                continue
            try:
                buffer.publish_context(bridge.projections.read_context())
            except BaseException as error:
                buffer.mark_worker_degraded(error)
        except BaseException as error:
            buffer.mark_worker_degraded(error)
            buffer.wait(0.1)


_BOOTSTRAP = _BootstrapCaptureBuffer()


def _capture_hook(hook: str, kwargs: dict[str, Any]) -> str | None:
    _BOOTSTRAP.capture(hook, kwargs)
    if hook == "pre_llm_call":
        return _BOOTSTRAP.read_context()
    return None


def _lazy_dispatch(hook: str, kwargs: dict[str, Any]) -> str | None:
    """Compatibility seam retained for host registration contract tests."""

    return _capture_hook(hook, kwargs)


def _dispatch_attempt(hook: str, kwargs: dict[str, Any]) -> str | None:
    missing = object()
    previous = getattr(_DISPATCH_STATE, "attempt", missing)
    attempt = _DispatchAttempt()
    dispatch_buffer = _BOOTSTRAP
    lock_acquired = False
    try:
        _DISPATCH_STATE.attempt = attempt
        dispatch_buffer._capture_lock.acquire()
        lock_acquired = True
        return _lazy_dispatch(hook, kwargs)
    except BaseException:
        # Registered callbacks are observers and must never interrupt Hermes.
        reservation = attempt.reservation
        buffer = reservation.buffer if reservation is not None else _BOOTSTRAP
        try:
            buffer.record_dispatch_failure(reservation)
            if hook == "pre_llm_call":
                context = buffer.read_context()
                return context if type(context) is str and context else None
        except BaseException:
            pass
        return None
    finally:
        cleanup_error: BaseException | None = None
        try:
            if previous is missing:
                with suppress(AttributeError):
                    del _DISPATCH_STATE.attempt
            else:
                _DISPATCH_STATE.attempt = previous
        except BaseException as error:
            cleanup_error = error
        try:
            if lock_acquired:
                dispatch_buffer._capture_lock.release()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        try:
            notify_buffer = attempt.notify_buffer
        except BaseException as error:
            notify_buffer = None
            if cleanup_error is None:
                cleanup_error = error
        if notify_buffer is not None:
            try:
                if isinstance(previous, _DispatchAttempt):
                    previous.notify_buffer = notify_buffer
                else:
                    notify_buffer._notify_worker()
            except BaseException as error:
                with suppress(BaseException):
                    notify_buffer.mark_worker_degraded(error)
        if cleanup_error is not None:
            raise cleanup_error


def _safe_dispatch(hook: str, kwargs: dict[str, Any]) -> str | None:
    """Contain every callback failure, including prologue and cleanup failures."""

    try:
        return _dispatch_attempt(hook, kwargs)
    except BaseException as error:
        buffer = _BOOTSTRAP
        with suppress(BaseException):
            buffer.mark_unrepresented_callback(error)
        if hook == "pre_llm_call":
            with suppress(BaseException):
                context = buffer.read_context()
                return context if type(context) is str and context else None
        return None


def on_session_start(**kwargs: Any) -> None:
    _safe_dispatch("on_session_start", kwargs)
    return None


def on_session_end(**kwargs: Any) -> None:
    _safe_dispatch("on_session_end", kwargs)
    return None


def on_session_finalize(**kwargs: Any) -> None:
    _safe_dispatch("on_session_finalize", kwargs)
    return None


def on_session_reset(**kwargs: Any) -> None:
    _safe_dispatch("on_session_reset", kwargs)
    return None


def pre_llm_call(**kwargs: Any) -> str | None:
    return _safe_dispatch("pre_llm_call", kwargs)


def post_llm_call(**kwargs: Any) -> None:
    _safe_dispatch("post_llm_call", kwargs)
    return None


def pre_api_request(**kwargs: Any) -> None:
    _safe_dispatch("pre_api_request", kwargs)
    return None


def post_api_request(**kwargs: Any) -> None:
    _safe_dispatch("post_api_request", kwargs)
    return None


def api_request_error(**kwargs: Any) -> None:
    _safe_dispatch("api_request_error", kwargs)
    return None


def pre_tool_call(**kwargs: Any) -> None:
    _safe_dispatch("pre_tool_call", kwargs)
    return None


def post_tool_call(**kwargs: Any) -> None:
    _safe_dispatch("post_tool_call", kwargs)
    return None


def pre_approval_request(**kwargs: Any) -> None:
    _safe_dispatch("pre_approval_request", kwargs)
    return None


def post_approval_response(**kwargs: Any) -> None:
    _safe_dispatch("post_approval_response", kwargs)
    return None


def subagent_start(**kwargs: Any) -> None:
    _safe_dispatch("subagent_start", kwargs)
    return None


def subagent_stop(**kwargs: Any) -> None:
    _safe_dispatch("subagent_stop", kwargs)
    return None


def pre_verify(**kwargs: Any) -> None:
    _safe_dispatch("pre_verify", kwargs)
    return None


HOOK_CALLBACKS = MappingProxyType(
    {
        "on_session_start": on_session_start,
        "on_session_end": on_session_end,
        "on_session_finalize": on_session_finalize,
        "on_session_reset": on_session_reset,
        "pre_llm_call": pre_llm_call,
        "post_llm_call": post_llm_call,
        "pre_api_request": pre_api_request,
        "post_api_request": post_api_request,
        "api_request_error": api_request_error,
        "pre_tool_call": pre_tool_call,
        "post_tool_call": post_tool_call,
        "pre_approval_request": pre_approval_request,
        "post_approval_response": post_approval_response,
        "subagent_start": subagent_start,
        "subagent_stop": subagent_stop,
        "pre_verify": pre_verify,
    }
)


def _parse_hermes_version(version: object, *, source: str) -> Version:
    if not isinstance(version, str):
        raise RuntimeError(f"Hermes {source} version is invalid")
    try:
        return Version(version)
    except InvalidVersion as error:
        raise RuntimeError(f"Hermes {source} version {version!r} is invalid") from error


def resolve_hermes_version() -> str:
    """Resolve and cross-check the installed Hermes Agent host version."""

    distribution_version: str | None = None
    try:
        distribution_version = metadata.version("hermes-agent")
    except metadata.PackageNotFoundError:
        pass
    except Exception as error:
        raise RuntimeError(
            "Hermes Agent distribution version is unavailable"
        ) from error

    module_version: str | None = None
    try:
        module = import_module("hermes_cli")
    except ModuleNotFoundError as error:
        if error.name != "hermes_cli":
            raise RuntimeError("Hermes Agent module version is unavailable") from error
    except Exception as error:
        raise RuntimeError("Hermes Agent module version is unavailable") from error
    else:
        raw_module_version = getattr(module, "__version__", None)
        if raw_module_version is not None and not isinstance(raw_module_version, str):
            raise RuntimeError("Hermes module version is invalid")
        module_version = raw_module_version

    parsed_distribution = (
        _parse_hermes_version(distribution_version, source="distribution")
        if distribution_version is not None
        else None
    )
    parsed_module = (
        _parse_hermes_version(module_version, source="module")
        if module_version is not None
        else None
    )

    if parsed_distribution is not None and parsed_module is not None:
        if parsed_distribution != parsed_module:
            raise RuntimeError(
                "Hermes Agent distribution/module version mismatch: "
                f"{distribution_version!r} != {module_version!r}"
            )
        if distribution_version is None:
            raise RuntimeError("Hermes Agent distribution version is unavailable")
        return distribution_version
    if parsed_distribution is not None:
        if distribution_version is None:
            raise RuntimeError("Hermes Agent distribution version is unavailable")
        return distribution_version
    if parsed_module is not None:
        if module_version is None:
            raise RuntimeError("Hermes Agent module version is unavailable")
        return module_version
    raise RuntimeError("Hermes Agent is not installed or has no version metadata")


def require_supported_hermes(version: str | None) -> str:
    """Fail visibly unless *version* is in the verified Hermes release line."""

    if version is None:
        raise RuntimeError("Hermes Agent version is invalid")
    parsed = _parse_hermes_version(version, source="Agent")
    if parsed not in _SUPPORTED_HERMES_SPECIFIER:
        raise RuntimeError(
            f"Hermes Agent version {version!r} is unsupported; "
            f"required {SUPPORTED_HERMES}"
        )
    return version


def setup_alice_brain_cli(parser: Any) -> None:
    """Attach the shared runtime CLI lazily when Hermes builds its parser."""

    from alice_brain_hermes.hermes.cli import setup_alice_brain_cli as setup

    setup(parser)


def handle_alice_brain_cli(args: Any) -> int:
    """Dispatch the Hermes command through the shared in-process handler."""

    from alice_brain_hermes.hermes.cli import handle_alice_brain_cli as handle

    return handle(args)


def _plugin_skill_path() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    relative_path = Path("skills") / PLUGIN_SKILL_NAME / "SKILL.md"
    candidates = (
        package_root / relative_path,
        package_root.parents[1] / relative_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Alice-brain-Hermes skill is missing from {candidates[0]} and "
        f"{candidates[1]}"
    )


def register(ctx: Any) -> None:
    """Register the inert Task 5 seam with a supported Hermes context."""

    with _REGISTRATION_LOCK:
        state = getattr(ctx, _REGISTRATION_STATE_ATTRIBUTE, None)
        if state == "registered":
            return
        if state == "registering":
            raise RuntimeError("Alice-brain-Hermes registration is re-entrant")
        if state == "failed":
            raise RuntimeError(
                "Alice-brain-Hermes registration previously failed for this context"
            )
        if state is not None:
            raise RuntimeError("Alice-brain-Hermes registration state is invalid")

        require_supported_hermes(resolve_hermes_version())
        register_hook = getattr(ctx, "register_hook", None)
        register_cli_command = getattr(ctx, "register_cli_command", None)
        register_skill = getattr(ctx, "register_skill", None)
        if (
            not callable(register_hook)
            or not callable(register_cli_command)
            or not callable(register_skill)
        ):
            raise RuntimeError("Hermes plugin context lacks registration callables")
        skill_path = _plugin_skill_path()

        setattr(ctx, _REGISTRATION_STATE_ATTRIBUTE, "registering")
        registered_hook_count = 0
        try:
            _BOOTSTRAP.bind_host_context(ctx)
            for hook_name in APPROVED_HOOKS:
                register_hook(hook_name, HOOK_CALLBACKS[hook_name])
                registered_hook_count += 1
            register_cli_command(
                name="alice-brain",
                help="Inspect and control the Alice-brain-Hermes runtime",
                setup_fn=setup_alice_brain_cli,
                handler_fn=handle_alice_brain_cli,
                description="Alice-brain-Hermes consciousness runtime commands",
            )
            register_skill(
                PLUGIN_SKILL_NAME,
                skill_path,
                PLUGIN_SKILL_DESCRIPTION,
            )
            setattr(ctx, _REGISTRATION_STATE_ATTRIBUTE, "registered")
        except BaseException as registration_error:
            # Coverage diagnostics must not mask the host registration error.
            with suppress(BaseException):
                _BOOTSTRAP.record_registration_failure(
                    registered_hook_count,
                    registration_error,
                )
            try:
                setattr(ctx, _REGISTRATION_STATE_ATTRIBUTE, "failed")
            except BaseException as state_error:
                raise registration_error from state_error
            raise
        else:
            # Registration remains valid even if health sampling is unavailable.
            with suppress(BaseException):
                _BOOTSTRAP.record_registration_success()
