"""Real-time continuous C0 scheduler with visible failure recovery."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from alice_brain_hermes.errors import SchedulerShutdownError
from alice_brain_hermes.runtime.engine import ConsciousEngine


def _validated_shutdown_timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError("scheduler timeout must be finite and non-negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class SchedulerHealth:
    status: str
    failure_event_persisted: bool
    last_error_type: str | None
    running: bool


class ContinuousScheduler:
    """One delayed wake is one tick carrying actual monotonic elapsed time."""

    def __init__(
        self,
        engine: ConsciousEngine,
        *,
        interval_seconds: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be finite and positive")
        self.engine = engine
        self.interval_seconds = float(interval_seconds)
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._interruptible_sleep = sleeper is time.sleep
        self._last_wake = float(monotonic())
        if not math.isfinite(self._last_wake):
            raise ValueError("monotonic clock must return a finite value")
        self._volatile_degraded = False
        self._failure_event_persisted = True
        self._last_error_type: str | None = None
        self._creator_pid = os.getpid()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _assert_creator_process(self) -> None:
        if os.getpid() != self._creator_pid:
            raise PermissionError("continuous scheduler belongs to another process")

    @property
    def health(self) -> SchedulerHealth:
        self._assert_creator_process()
        degraded = (
            self._volatile_degraded or self.engine.state.runtime.health == "degraded"
        )
        return SchedulerHealth(
            status="degraded" if degraded else "healthy",
            failure_event_persisted=self._failure_event_persisted,
            last_error_type=self._last_error_type,
            running=self._thread is not None and self._thread.is_alive(),
        )

    def step(self) -> bool:
        """Attempt exactly one tick and retain enough state for a later recovery."""
        self._assert_creator_process()
        try:
            sample = self._monotonic()
            if isinstance(sample, bool) or not isinstance(sample, (int, float)):
                raise RuntimeError("monotonic clock returned a non-numeric sample")
            now = float(sample)
            elapsed = now - self._last_wake
            if not math.isfinite(now) or not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError(
                    "monotonic clock moved backwards or became non-finite"
                )
            self._last_wake = now
            was_degraded = self.health.status == "degraded"
            self.engine.pulse(elapsed)
            if was_degraded:
                self.engine.record_recovered()
            self._volatile_degraded = False
            self._failure_event_persisted = True
            return True
        except Exception as error:
            self._last_error_type = type(error).__name__
            self._volatile_degraded = True
            try:
                self.engine.record_failure(error, phase="c0.pulse")
            except Exception as persistence_error:
                self._last_error_type = type(persistence_error).__name__
                self._failure_event_persisted = False
            else:
                self._failure_event_persisted = True
                self._volatile_degraded = False
            return False

    def run(self, *, max_ticks: int | None = None) -> None:
        """Run until stopped, or for an explicit number of testable ticks."""
        self._assert_creator_process()
        if max_ticks is not None and (
            isinstance(max_ticks, bool)
            or not isinstance(max_ticks, int)
            or max_ticks < 0
        ):
            raise ValueError("max_ticks must be a non-negative integer or None")
        completed = 0
        while not self._stop.is_set() and (max_ticks is None or completed < max_ticks):
            if self._interruptible_sleep:
                if self._stop.wait(self.interval_seconds):
                    break
            else:
                self._sleeper(self.interval_seconds)
                if self._stop.is_set():
                    break
            self.step()
            completed += 1

    def start(self) -> None:
        """Start continuous off-turn ticking in one owned background thread."""
        self._assert_creator_process()
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run,
            name=f"alice-brain-hermes-c0-{self.engine.brain_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Request termination and prove the writer thread exited."""
        self._assert_creator_process()
        timeout = _validated_shutdown_timeout(timeout)
        self._stop.set()
        self.join(timeout=timeout)

    def join(self, *, timeout: float = 5.0) -> None:
        """Join the owned writer or fail before its ledger can be closed."""
        self._assert_creator_process()
        timeout = _validated_shutdown_timeout(timeout)
        thread = self._thread
        if thread is None:
            return
        if thread is threading.current_thread():
            raise SchedulerShutdownError("cannot join the current scheduler thread")
        thread.join(timeout)
        if thread.is_alive():
            raise SchedulerShutdownError(
                "scheduler writer did not exit before the shutdown deadline"
            )


C0Scheduler = ContinuousScheduler

__all__ = ["C0Scheduler", "ContinuousScheduler", "SchedulerHealth"]
