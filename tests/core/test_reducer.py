from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from alice_brain_hermes.core.events import EventEnvelope, new_event
from alice_brain_hermes.core.reducer import reduce_many, reduce_state
from alice_brain_hermes.core.state import BrainState
from alice_brain_hermes.errors import DomainInvariantError
from alice_brain_hermes.ids import new_id

BRAIN = new_id()
ACTOR = new_id()


def event(event_type: str, payload: dict[str, object], sequence: int | None = None):
    return new_event(
        event_type,
        BRAIN,
        ACTOR,
        payload,
        sequence=sequence,
        wall_time=datetime(2026, 7, 14, tzinfo=UTC),
        monotonic_ns=100 + (sequence or 0),
    )


def test_event_envelope_is_strict_deeply_frozen_and_canonical() -> None:
    envelope = event(
        "observation.received",
        {"z": [1, {"nested": True}], "a": "first"},
    )

    with pytest.raises(ValidationError):
        envelope.event_type = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        envelope.payload["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        envelope.payload["z"][1]["nested"] = False  # type: ignore[index]

    assert envelope.canonical_json(exclude_sequence=True) == event(
        "observation.received",
        {"a": "first", "z": [1, {"nested": True}]},
    ).model_copy(
        update={
            "event_id": envelope.event_id,
            "wall_time": envelope.wall_time,
            "monotonic_ns": envelope.monotonic_ns,
        }
    ).canonical_json(exclude_sequence=True)


@pytest.mark.parametrize(
    "payload",
    [
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": {1: "non-string key"}},
        {"bad": object()},
    ],
)
def test_event_rejects_non_json_payloads(payload: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError, ValidationError)):
        event("observation.received", payload)


def test_direct_envelope_construction_is_strict() -> None:
    values = {
        "schema_version": 1,
        "event_id": new_id(),
        "event_type": "brain.created",
        "brain_id": BRAIN,
        "sequence": None,
        "wall_time": datetime(2026, 7, 14, tzinfo=UTC),
        "monotonic_ns": 1,
        "actor_id": ACTOR,
        "adapter_id": None,
        "session_id": None,
        "turn_id": None,
        "action_id": None,
        "causation_id": None,
        "correlation_id": None,
        "payload": {},
    }
    assert EventEnvelope.model_validate(values).brain_id == BRAIN
    with pytest.raises(ValidationError):
        EventEnvelope.model_validate({**values, "monotonic_ns": "1"})
    with pytest.raises(ValidationError):
        EventEnvelope.model_validate({**values, "extra": True})


def test_external_provenance_ids_remain_exact_bounded_strings() -> None:
    envelope = new_event(
        "hermes.session.started",
        BRAIN,
        ACTOR,
        {},
        adapter_id="hermes-0.18.2",
        session_id="session/external-1",
        turn_id="turn-1",
        action_id="a1",
        correlation_id="provider-attempt-1",
    )

    assert envelope.session_id == "session/external-1"
    assert hash(envelope) == hash(envelope)
    with pytest.raises(ValidationError):
        envelope.model_copy(update={"session_id": " "}).revalidated()


def test_brain_state_starts_unnamed_with_explicit_foundation() -> None:
    state = BrainState.genesis(BRAIN)

    assert state.name is None
    assert state.capabilities == {}
    assert state.logical_clock == 0.0
    assert state.trace_complete is True
    assert state.raw_lifecycle_counts == {}
    assert state.last_sequence == 0


def test_reducer_tracks_clock_capabilities_trace_and_unknown_raw_events() -> None:
    state = reduce_many(
        BrainState.genesis(BRAIN),
        [
            event("brain.created", {"name": None}, 1),
            event("clock.tick", {"elapsed_seconds": 1.25}, 2),
            event(
                "capabilities.reported",
                {"capabilities": {"chunk_capture": "unobserved"}},
                3,
            ),
            event("future.lifecycle.event", {"opaque": True}, 4),
            event("trace.gap", {"reason": "queue_overflow"}, 5),
        ],
    )

    assert state.logical_clock == 1.25
    assert state.capabilities == {"chunk_capture": "unobserved"}
    assert state.trace_complete is False
    assert state.raw_lifecycle_counts == {
        "brain.created": 1,
        "capabilities.reported": 1,
        "clock.tick": 1,
        "future.lifecycle.event": 1,
        "trace.gap": 1,
    }
    assert state.last_sequence == 5


def test_reduce_state_is_pure_and_deterministic() -> None:
    initial = BrainState.genesis(BRAIN)
    tick = event("clock.tick", {"elapsed_seconds": 2.0}, 1)

    first = reduce_state(initial, tick)
    second = reduce_state(initial, tick)

    assert first == second
    assert initial.logical_clock == 0.0
    assert initial.raw_lifecycle_counts == {}


def test_reducer_rejects_brain_mismatch_sequence_gap_and_bad_clock() -> None:
    state = BrainState.genesis(BRAIN)
    other_brain_event = new_event(
        "clock.tick", new_id(), ACTOR, {"elapsed_seconds": 1.0}, sequence=1
    )
    with pytest.raises(DomainInvariantError, match="brain"):
        reduce_state(state, other_brain_event)
    with pytest.raises(DomainInvariantError, match="sequence"):
        reduce_state(state, event("clock.tick", {"elapsed_seconds": 1.0}, 2))
    with pytest.raises(DomainInvariantError, match="elapsed_seconds"):
        reduce_state(state, event("clock.tick", {"elapsed_seconds": -1.0}, 1))
