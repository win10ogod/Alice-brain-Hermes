"""Append-first event-sourced consciousness-engineering runtime."""

from __future__ import annotations

import math
import os
import re
import sqlite3
import threading
from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from alice_brain_hermes.core.cognition import LocalCognitionPort, result_payload
from alice_brain_hermes.core.events import EventEnvelope, new_event
from alice_brain_hermes.core.reducer import reduce_state
from alice_brain_hermes.core.state import BrainState
from alice_brain_hermes.core.workspace import WorkspaceCoordinator
from alice_brain_hermes.errors import EventConflictError, ExpectedSequenceError
from alice_brain_hermes.ids import validate_id
from alice_brain_hermes.protocol.models import (
    BridgeCommitAckV2,
    BridgeRecordV1,
    BridgeStreamState,
    ConsciousnessFrameV3,
)
from alice_brain_hermes.runtime.store import BridgeAbandonResult, BridgeCommitResult


class EventLedger(Protocol):
    def append(self, event: EventEnvelope) -> EventEnvelope: ...

    def append_expected(
        self, event: EventEnvelope, *, expected_sequence: int
    ) -> tuple[EventEnvelope, bool]: ...

    def get_event(self, event_id: str) -> EventEnvelope | None: ...

    def get_event_and_head(
        self, event_id: str, brain_id: str
    ) -> tuple[EventEnvelope | None, int]: ...

    def replay(self, brain_id: str) -> BrainState: ...

    def commit_bridge_record(
        self,
        bridge_instance_id: str,
        record: BridgeRecordV1,
        *,
        expected_state: BrainState,
        connected_nonce: str | None = None,
        max_frame_bytes: int = 65_536,
        max_ack_bytes: int = 4_194_304,
    ) -> BridgeCommitResult: ...

    def abandon_bridge_stream(
        self,
        bridge_instance_id: str,
        *,
        expected_state: BrainState,
        last_seen_not_after: datetime | None = None,
    ) -> BridgeAbandonResult: ...

    def project_bridge_frame(
        self,
        bridge_instance_id: str,
        *,
        expected_state: BrainState,
        connected_nonce: str | None = None,
        scheduler_sample: str,
        max_frame_bytes: int = 65_536,
    ) -> ConsciousnessFrameV3: ...


class Coordinator(Protocol):
    def propose(self, state: BrainState): ...


_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")


def sanitized_failure(error: Exception, *, phase: str) -> dict[str, str]:
    """Return bounded single-line failure evidence, never a traceback."""
    message = _CONTROL_CHARACTERS.sub(" ", str(error)).strip()
    if not message:
        message = "no error message"
    return {
        "error_type": type(error).__name__[:160],
        "message": message[:512],
        "phase": phase[:160] or "runtime",
    }


class ConsciousEngine:
    """Own one replay-derived state; ledger append precedes every reduction."""

    def __init__(
        self,
        ledger: EventLedger,
        brain_id: str,
        *,
        actor_id: str,
        cognition: LocalCognitionPort | None = None,
        coordinator: Coordinator | None = None,
    ) -> None:
        self.ledger = ledger
        self.brain_id = validate_id(brain_id)
        self.actor_id = validate_id(actor_id)
        self.cognition = cognition or LocalCognitionPort()
        self.coordinator = coordinator
        self._default_coordinator = WorkspaceCoordinator()
        self._creator_pid = os.getpid()
        self._lock = threading.RLock()
        bootstrap_state = getattr(ledger, "bootstrap_state", None)
        self._state = (
            bootstrap_state(brain_id)
            if callable(bootstrap_state)
            else ledger.replay(brain_id)
        )
        self._diverged = False

    def _assert_creator_process(self) -> None:
        if os.getpid() != self._creator_pid:
            raise PermissionError("conscious engine belongs to another process")

    @property
    def state(self) -> BrainState:
        self._assert_creator_process()
        with self._lock:
            return self._state

    @property
    def is_stale(self) -> bool:
        """Whether this engine must be reconstructed from authoritative replay."""
        self._assert_creator_process()
        with self._lock:
            return self._diverged

    def append(self, event: EventEnvelope) -> EventEnvelope:
        """Validate next state, persist, then publish the authoritative state."""
        self._assert_creator_process()
        if event.brain_id != self.brain_id:
            raise ValueError("event brain does not match engine brain")
        if event.sequence is not None:
            raise ValueError("engine rejects presequenced client events")
        with self._lock:
            if self._diverged:
                raise EventConflictError(
                    "engine sequence divergence requires a replayed restart"
                )
            existing, head = self.ledger.get_event_and_head(
                event.event_id, self.brain_id
            )
            if head != self._state.last_sequence:
                self._diverged = True
                raise EventConflictError(
                    "engine sequence divergence: authoritative ledger head "
                    f"{head} does not match replayed state "
                    f"{self._state.last_sequence}; restart from ledger replay "
                    "is required"
                )
            if existing is not None:
                if (
                    existing.body_fingerprint() != event.body_fingerprint()
                    or existing.canonical_json(exclude_sequence=True)
                    != event.canonical_json(exclude_sequence=True)
                ):
                    raise EventConflictError(
                        f"event ID {event.event_id} already has a different body"
                    )
                if (
                    existing.sequence is None
                    or existing.sequence > self._state.last_sequence
                ):
                    self._diverged = True
                    raise EventConflictError(
                        "engine sequence divergence: exact event exists beyond "
                        "the replayed state; restart from ledger replay is required"
                    )
                return existing

            expected_sequence = self._state.last_sequence + 1
            provisional = event.model_copy(
                update={"sequence": expected_sequence}
            ).revalidated()
            successor = reduce_state(self._state, provisional)
            try:
                stored, _inserted = self.ledger.append_expected(
                    event, expected_sequence=expected_sequence
                )
            except ExpectedSequenceError as error:
                self._diverged = True
                raise EventConflictError(
                    "engine sequence divergence: expected sequence was lost; "
                    "restart from ledger replay is required"
                ) from error
            if stored != provisional:
                self._diverged = True
                raise EventConflictError(
                    "engine sequence divergence: stored envelope does not match "
                    f"provisionally validated sequence {provisional.sequence}; "
                    "restart from ledger replay is required"
                )
            self._state = successor
            return stored

    def commit_bridge_record(
        self,
        bridge_instance_id: str,
        record: BridgeRecordV1,
        *,
        connected_nonce: str | None = None,
        max_frame_bytes: int = 65_536,
        max_ack_bytes: int = 4_194_304,
    ) -> BridgeCommitAckV2:
        """Commit one typed bridge record and publish only its committed successor."""
        self._assert_creator_process()
        with self._lock:
            if self._diverged:
                raise EventConflictError(
                    "engine sequence divergence requires a replayed restart"
                )
            try:
                result = self.ledger.commit_bridge_record(
                    bridge_instance_id,
                    record,
                    expected_state=self._state,
                    connected_nonce=connected_nonce,
                    max_frame_bytes=max_frame_bytes,
                    max_ack_bytes=max_ack_bytes,
                )
            except ExpectedSequenceError as error:
                self._diverged = True
                raise EventConflictError(
                    "engine sequence divergence during bridge commit; "
                    "restart from ledger replay is required"
                ) from error
            except sqlite3.DatabaseError:
                self._diverged = True
                raise
            if result.successor is not None:
                if result.successor.last_sequence != result.ack.last_event_sequence:
                    self._diverged = True
                    raise EventConflictError(
                        "bridge publication does not match committed sequence"
                    )
                self._state = result.successor
            return result.ack

    def abandon_bridge_stream(
        self,
        bridge_instance_id: str,
        *,
        last_seen_not_after: datetime | None = None,
    ) -> BridgeStreamState:
        """Publish one committed unknown-gap successor after grace expiry."""
        self._assert_creator_process()
        with self._lock:
            if self._diverged:
                raise EventConflictError(
                    "engine sequence divergence requires a replayed restart"
                )
            try:
                result = self.ledger.abandon_bridge_stream(
                    bridge_instance_id,
                    expected_state=self._state,
                    last_seen_not_after=last_seen_not_after,
                )
            except ExpectedSequenceError as error:
                self._diverged = True
                raise EventConflictError(
                    "engine sequence divergence during bridge abandonment; "
                    "restart from ledger replay is required"
                ) from error
            except sqlite3.DatabaseError:
                self._diverged = True
                raise
            if result.successor is not None:
                self._state = result.successor
            return result.stream

    def project_bridge_frame(
        self,
        bridge_instance_id: str,
        *,
        connected_nonce: str | None = None,
        scheduler_sample: str,
        max_frame_bytes: int = 65_536,
    ) -> ConsciousnessFrameV3:
        """Return one integrity-checked live frame under engine serialization."""
        self._assert_creator_process()
        with self._lock:
            if self._diverged:
                raise EventConflictError(
                    "engine sequence divergence requires a replayed restart"
                )
            return self.ledger.project_bridge_frame(
                bridge_instance_id,
                expected_state=self._state,
                connected_nonce=connected_nonce,
                scheduler_sample=scheduler_sample,
                max_frame_bytes=max_frame_bytes,
            )

    def _event(self, event_type: str, payload: Mapping[str, object]) -> EventEnvelope:
        return new_event(event_type, self.brain_id, self.actor_id, payload)

    def pulse(self, elapsed_seconds: float) -> BrainState:
        """Advance one genuine C0 cycle without an agent turn or provider."""
        self._assert_creator_process()
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be finite and non-negative")
        self.append(
            self._event("clock.tick", {"elapsed_seconds": float(elapsed_seconds)})
        )

        coordinator = self.coordinator or self._default_coordinator
        candidates = tuple(coordinator.propose(self.state))
        cycle = self.state.workspace.cycle + 1
        self.append(
            self._event(
                "workspace.broadcast",
                {
                    "cycle": cycle,
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                },
            )
        )

        broadcast = self.state.workspace.broadcast
        structured_content = {
            "cycle": self.state.workspace.cycle,
            "broadcast": [item.model_dump(mode="json") for item in broadcast],
        }
        result = self.cognition.reflect(
            structured_content,
            source_ids=tuple(item.candidate_id for item in broadcast),
        )
        self.append(self._event("cognition.reflected", result_payload(result)))
        return self.state

    def record_failure(self, error: Exception, *, phase: str) -> EventEnvelope:
        self._assert_creator_process()
        return self.append(
            self._event("runtime.failure", sanitized_failure(error, phase=phase))
        )

    def record_recovered(self) -> EventEnvelope:
        self._assert_creator_process()
        return self.append(self._event("runtime.recovered", {"status": "healthy"}))


__all__ = ["ConsciousEngine", "EventLedger", "sanitized_failure"]
