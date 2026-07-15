"""Inert Hermes Agent plugin registration boundary."""

from __future__ import annotations

import math
import queue
import threading
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, replace
from importlib import import_module, metadata
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
_SUPPORTED_HERMES_SPECIFIER = SpecifierSet(SUPPORTED_HERMES)
_REGISTRATION_STATE_ATTRIBUTE = "_alice_brain_hermes_registration_v1"
_REGISTRATION_LOCK = threading.RLock()

_BOOTSTRAP_QUEUE_CAPACITY = 256
_BOOTSTRAP_MAX_DEPTH = 6
_BOOTSTRAP_MAX_NODES = 1_024
_BOOTSTRAP_MAX_ITEMS = 64
_BOOTSTRAP_MAX_STRING_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class _BootstrapHealth:
    trace_complete: bool = True
    dropped_events: int = 0
    pending_records: int = 0
    pending_gap_ranges: int = 0
    worker_started: bool = False
    degraded: bool = False
    last_error: str | None = None
    worker_error: str | None = None
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
        self._worker_lock = threading.Lock()
        self._wake = threading.Event()
        self._next_capture_seq = 1
        self._queue_head: _BootstrapCapture | None = None
        self._worker_retained: _BootstrapCapture | None = None
        self._worker: threading.Thread | None = None
        self._start_worker_on_capture = start_worker_on_capture
        self._health = _BootstrapHealth()
        self._handed_off_dropped_events = 0
        self._context: str | None = None

    @property
    def health(self) -> _BootstrapHealth:
        return self._health

    def read_context(self) -> str | None:
        return self._context

    def publish_context(self, context: str | None) -> None:
        self._context = context if type(context) is str and context else None

    def capture(self, hook: str, kwargs: dict[str, Any]) -> None:
        with self._capture_lock:
            reservation = self._reserve_locked()
            capture_seq = reservation.capture_seq
            try:
                if kwargs.get("telemetry_schema_version") != "hermes.observer.v1":
                    self._record_gap_locked(
                        capture_seq,
                        "invalid_source_schema",
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
                        self._record_gap_locked(capture_seq, "queue_full")
                    except BaseException:
                        self._record_gap_locked(capture_seq, "callback_internal")
            except BaseException:
                self._record_gap_locked(capture_seq, "callback_internal")
        self._defer_or_notify_worker()

    def _reserve_locked(self) -> _DispatchReservation:
        attempt = _active_dispatch_attempt()
        if attempt is not None and attempt.reservation is not None:
            raise RuntimeError("one Hermes callback may reserve only one capture")
        capture_seq = self._next_capture_seq
        updated_health = replace(
            self._health,
            pending_records=self._health.pending_records + 1,
        )
        self._next_capture_seq = capture_seq + 1
        self._health = updated_health
        reservation = _DispatchReservation(self, capture_seq)
        if attempt is not None:
            attempt.reservation = reservation
            attempt.notify_buffer = self
        return reservation

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
            previous = self._overflow[-1]
            causes = dict(previous.gap_cause_counts or {})
            for cause, count in (gap.gap_cause_counts or {}).items():
                causes[cause] = causes.get(cause, 0) + count
            self._overflow[-1] = replace(
                previous,
                last_capture_seq=gap.last_capture_seq,
                gap_cause_counts=MappingProxyType(causes),
            )
            return 0
        self._overflow.append(gap)
        return 1

    def _record_gap_locked(self, capture_seq: int, cause: str) -> None:
        gap = _BootstrapCapture(
            capture_seq=capture_seq,
            last_capture_seq=capture_seq,
            hook=None,
            detached_kwargs=None,
            copy_stats=MappingProxyType({}),
            gap_cause_counts=MappingProxyType({cause: 1}),
        )
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

    def _notify_worker(self) -> None:
        try:
            self._wake.set()
        except BaseException as error:
            self.mark_worker_degraded(error)
        if self._start_worker_on_capture:
            self._start_worker()

    def _start_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None:
                return
            try:
                worker = threading.Thread(
                    target=_bootstrap_worker_main,
                    args=(self,),
                    name="alice-brain-hermes-bootstrap",
                    daemon=True,
                )
                self._worker = worker
                worker.start()
            except BaseException as error:
                self._worker = None
                self.mark_worker_degraded(error)
                return
            with self._capture_lock:
                self._health = replace(self._health, worker_started=True)

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

    def mark_worker_degraded(self, error: BaseException) -> None:
        error_name = type(error).__name__[:160]
        with self._capture_lock:
            self._health = replace(
                self._health,
                degraded=True,
                last_error=error_name,
                worker_error=error_name,
            )

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
        self._wake.wait(timeout)
        self._wake.clear()

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


def _bootstrap_worker_main(buffer: _BootstrapCaptureBuffer) -> None:
    bridge: Any | None = None
    while True:
        if bridge is None:
            try:
                from alice_brain_hermes.hermes.bridge import (
                    HookBridge,
                    default_runtime_home,
                )

                bridge = HookBridge(
                    default_runtime_home(),
                    start_worker_on_capture=False,
                    context_sink=buffer.publish_context,
                )
                bridge.start_worker()
            except Exception as error:
                buffer.mark_worker_degraded(error)
                buffer.wait(0.1)
                continue
        capture = buffer.next_for_worker()
        if capture is None:
            buffer.publish_context(bridge.projections.read_context())
            buffer.wait(0.05)
            continue
        try:
            bridge.capture_reserved(
                hook=capture.hook,
                detached_kwargs=capture.detached_kwargs,
                first_capture_seq=capture.capture_seq,
                last_capture_seq=capture.last_capture_seq,
                gap_cause_counts=capture.gap_cause_counts,
                copy_stats=capture.copy_stats,
            )
        except Exception as error:
            buffer.mark_worker_degraded(error)
            buffer.wait(0.1)
            continue
        buffer.mark_handed_off(capture)
        buffer.publish_context(bridge.projections.read_context())


_BOOTSTRAP = _BootstrapCaptureBuffer()


def _capture_hook(hook: str, kwargs: dict[str, Any]) -> str | None:
    _BOOTSTRAP.capture(hook, kwargs)
    if hook == "pre_llm_call":
        return _BOOTSTRAP.read_context()
    return None


def _lazy_dispatch(hook: str, kwargs: dict[str, Any]) -> str | None:
    """Compatibility seam retained for host registration contract tests."""

    return _capture_hook(hook, kwargs)


def _safe_dispatch(hook: str, kwargs: dict[str, Any]) -> str | None:
    missing = object()
    previous = getattr(_DISPATCH_STATE, "attempt", missing)
    attempt = _DispatchAttempt()
    _DISPATCH_STATE.attempt = attempt
    dispatch_buffer = _BOOTSTRAP
    lock_acquired = False
    try:
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
        if previous is missing:
            with suppress(AttributeError):
                del _DISPATCH_STATE.attempt
        else:
            _DISPATCH_STATE.attempt = previous
        if lock_acquired:
            dispatch_buffer._capture_lock.release()
        notify_buffer = attempt.notify_buffer
        if notify_buffer is not None:
            if isinstance(previous, _DispatchAttempt):
                previous.notify_buffer = notify_buffer
            else:
                try:
                    notify_buffer._notify_worker()
                except BaseException as error:
                    with suppress(BaseException):
                        notify_buffer.mark_worker_degraded(error)


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
        if not callable(register_hook) or not callable(register_cli_command):
            raise RuntimeError("Hermes plugin context lacks registration callables")

        setattr(ctx, _REGISTRATION_STATE_ATTRIBUTE, "registering")
        registered_hook_count = 0
        try:
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
