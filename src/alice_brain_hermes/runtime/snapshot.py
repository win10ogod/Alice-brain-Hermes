"""Portable, coalescing automatic snapshot worker."""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from alice_brain_hermes.errors import SchedulerShutdownError
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.store import SQLiteLedger

DEFAULT_SNAPSHOT_INTERVAL_EVENTS = 1_024


def validate_snapshot_interval(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("snapshot interval_events must be a positive integer")
    return value


def _validated_timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError("snapshot worker timeout must be finite and non-negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class SnapshotWorkerHealth:
    status: str
    running: bool
    pending_brain_count: int
    snapshot_count: int
    latest_sequence: int
    last_error_type: str | None


class SnapshotWorker:
    """Own one bounded pending checkpoint per registered brain."""

    def __init__(
        self,
        ledger: SQLiteLedger,
        *,
        interval_events: int = DEFAULT_SNAPSHOT_INTERVAL_EVENTS,
        fatal_error_sink: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.ledger = ledger
        self.interval_events = validate_snapshot_interval(interval_events)
        self._fatal_error_sink = fatal_error_sink
        self._creator_pid = os.getpid()
        self._condition = threading.Condition(threading.Lock())
        self._engines: dict[str, ConsciousEngine] = {}
        self._observed_heads: dict[str, int] = {}
        self._last_snapshot_sequences: dict[str, int] = {}
        self._next_attempt_sequences: dict[str, int] = {}
        self._pending: set[str] = set()
        self._processing: str | None = None
        self._stopping = False
        self._snapshot_count = 0
        self._latest_sequence = 0
        self._errors_by_brain: dict[str, str] = {}
        self._thread: threading.Thread | None = None

    @staticmethod
    def _error_type(error: BaseException) -> str:
        return type(error).__name__[:160]

    @staticmethod
    def _is_transient_io_error(error: BaseException) -> bool:
        return isinstance(error, (OSError, sqlite3.OperationalError))

    def _record_error(self, brain_id: str, error: BaseException) -> None:
        self._errors_by_brain.pop(brain_id, None)
        self._errors_by_brain[brain_id] = self._error_type(error)

    def _clear_error(self, brain_id: str) -> None:
        self._errors_by_brain.pop(brain_id, None)

    def _assert_creator_process(self) -> None:
        if os.getpid() != self._creator_pid:
            raise PermissionError("snapshot worker belongs to another process")

    def register(
        self,
        engine: ConsciousEngine,
        *,
        known_snapshot_sequence: int | None = None,
    ) -> None:
        self._assert_creator_process()
        if known_snapshot_sequence is not None and (
            isinstance(known_snapshot_sequence, bool)
            or not isinstance(known_snapshot_sequence, int)
            or known_snapshot_sequence < 0
        ):
            raise ValueError("known snapshot sequence must be non-negative")
        state_sequence = engine.state.last_sequence
        with self._condition:
            existing = self._engines.get(engine.brain_id)
            if existing is not None and existing is not engine:
                raise RuntimeError("a different engine is already registered")
            self._engines[engine.brain_id] = engine
            self._observed_heads[engine.brain_id] = max(
                self._observed_heads.get(engine.brain_id, 0),
                state_sequence,
            )
            if known_snapshot_sequence is not None:
                self._last_snapshot_sequences[engine.brain_id] = (
                    known_snapshot_sequence
                )
                self._next_attempt_sequences[engine.brain_id] = (
                    known_snapshot_sequence + self.interval_events
                )
            next_attempt = self._next_attempt_sequences.get(engine.brain_id)
            if next_attempt is None or (
                self._observed_heads[engine.brain_id] >= next_attempt
            ):
                self._pending.add(engine.brain_id)
            self._condition.notify_all()

    def notify(self, brain_id: str, sequence: int) -> None:
        """Record one committed publication without performing I/O."""
        with self._condition:
            self._observed_heads[brain_id] = max(
                self._observed_heads.get(brain_id, 0), sequence
            )
            if brain_id not in self._engines:
                return
            next_attempt = self._next_attempt_sequences.get(brain_id)
            if next_attempt is None or sequence >= next_attempt:
                self._pending.add(brain_id)
                self._condition.notify_all()

    def _process(self, brain_id: str) -> None:
        engine = self._engines[brain_id]
        latest = self._last_snapshot_sequences.get(brain_id)
        if latest is None:
            snapshot = self.ledger.load_snapshot(brain_id)
            latest = 0 if snapshot is None else snapshot.last_sequence
            self._last_snapshot_sequences[brain_id] = latest
            with self._condition:
                self._clear_error(brain_id)
        with self._condition:
            observed = self._observed_heads.get(brain_id)
        if observed is None:
            observed = engine.state.last_sequence
        if observed - latest < self.interval_events:
            with self._condition:
                self._next_attempt_sequences[brain_id] = latest + self.interval_events
            return
        attempted_sequence = engine.state.last_sequence
        try:
            saved = engine.checkpoint_current_state()
        except Exception as error:
            with self._condition:
                self._record_error(brain_id, error)
                self._next_attempt_sequences[brain_id] = (
                    attempted_sequence + self.interval_events
                )
            if (
                not self._is_transient_io_error(error)
                and self._fatal_error_sink is not None
            ):
                self._fatal_error_sink(error)
            return
        with self._condition:
            self._last_snapshot_sequences[brain_id] = saved.last_sequence
            self._next_attempt_sequences[brain_id] = (
                saved.last_sequence + self.interval_events
            )
            self._snapshot_count += 1
            self._latest_sequence = max(self._latest_sequence, saved.last_sequence)
            self._clear_error(brain_id)

    def _record_unhandled_failure(
        self,
        brain_id: str,
        error: BaseException,
        *,
        fatal: bool,
    ) -> None:
        with self._condition:
            observed = self._observed_heads.get(brain_id, 0)
            self._record_error(brain_id, error)
            self._next_attempt_sequences[brain_id] = (
                observed + self.interval_events
            )
        if fatal and self._fatal_error_sink is not None:
            self._fatal_error_sink(error)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping and not self._pending:
                    return
                brain_id = self._pending.pop()
                self._processing = brain_id
            try:
                try:
                    self._process(brain_id)
                except Exception as error:
                    self._record_unhandled_failure(
                        brain_id,
                        error,
                        fatal=not self._is_transient_io_error(error),
                    )
            finally:
                with self._condition:
                    self._processing = None
                    observed = self._observed_heads.get(brain_id, 0)
                    next_attempt = self._next_attempt_sequences.get(brain_id)
                    if next_attempt is not None and observed >= next_attempt:
                        self._pending.add(brain_id)
                    self._condition.notify_all()

    def start(self) -> None:
        self._assert_creator_process()
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run,
                name="alice-brain-hermes-snapshots",
                daemon=True,
            )
            self._thread.start()

    def wait_idle(self, *, timeout: float = 5.0) -> None:
        self._assert_creator_process()
        deadline = time.monotonic() + _validated_timeout(timeout)
        with self._condition:
            while self._pending or self._processing is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("snapshot worker did not become idle")
                self._condition.wait(remaining)

    def stop(self, *, timeout: float = 5.0) -> None:
        self._assert_creator_process()
        timeout = _validated_timeout(timeout)
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
            thread = self._thread
        if thread is None:
            return
        if thread is threading.current_thread():
            raise SchedulerShutdownError("cannot join the current snapshot worker")
        thread.join(timeout)
        if thread.is_alive():
            raise SchedulerShutdownError(
                "snapshot worker did not exit before the shutdown deadline"
            )

    @property
    def health(self) -> SnapshotWorkerHealth:
        self._assert_creator_process()
        with self._condition:
            running = self._thread is not None and self._thread.is_alive()
            error_brain_id = next(reversed(self._errors_by_brain), None)
            last_error_type = (
                None
                if error_brain_id is None
                else self._errors_by_brain[error_brain_id]
            )
            return SnapshotWorkerHealth(
                status=(
                    "degraded"
                    if last_error_type is not None or not running
                    else "healthy"
                ),
                running=running,
                pending_brain_count=len(self._pending)
                + (self._processing is not None),
                snapshot_count=self._snapshot_count,
                latest_sequence=self._latest_sequence,
                last_error_type=last_error_type,
            )


__all__ = [
    "DEFAULT_SNAPSHOT_INTERVAL_EVENTS",
    "SnapshotWorker",
    "SnapshotWorkerHealth",
    "validate_snapshot_interval",
]
