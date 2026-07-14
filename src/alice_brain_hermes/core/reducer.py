"""Pure foundation reducer; the sole mutation path for brain state."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict, thaw_json
from alice_brain_hermes.core.state import BrainState, state_from_data
from alice_brain_hermes.errors import DomainInvariantError, SchemaVersionError


def _state_values(state: BrainState) -> dict[str, Any]:
    return state.model_dump(mode="python")


def _payload_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DomainInvariantError(f"{field} must be a JSON object")
    return value


def reduce_state(state: BrainState, event: EventEnvelope) -> BrainState:
    """Return the deterministic successor state for one immutable event."""
    state = state.revalidated()
    event = event.revalidated()
    if event.schema_version != 1:
        raise SchemaVersionError(f"unsupported event schema {event.schema_version}")
    if event.brain_id != state.brain_id:
        raise DomainInvariantError(
            f"event brain {event.brain_id!r} does not match state brain "
            f"{state.brain_id!r}"
        )
    if event.sequence is not None and event.sequence != state.last_sequence + 1:
        raise DomainInvariantError(
            f"event sequence {event.sequence} does not follow {state.last_sequence}"
        )

    values = _state_values(state)
    counts = dict(state.raw_lifecycle_counts.items())
    counts[event.event_type] = counts.get(event.event_type, 0) + 1
    values["raw_lifecycle_counts"] = dict(sorted(counts.items()))
    if event.sequence is not None:
        values["last_sequence"] = event.sequence

    if event.event_type in {"brain.created", "identity.named"}:
        name = event.payload.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise DomainInvariantError("name must be null or a non-blank string")
        values["name"] = name
    elif event.event_type == "clock.tick":
        elapsed = event.payload.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise DomainInvariantError(
                "clock.tick elapsed_seconds must be a finite non-negative number"
            )
        logical_clock = state.logical_clock + float(elapsed)
        if not math.isfinite(logical_clock):
            raise DomainInvariantError("clock.tick logical clock overflow")
        values["logical_clock"] = logical_clock
    elif event.event_type == "capabilities.reported":
        capabilities = _payload_mapping(
            event.payload.get("capabilities"), field="capabilities"
        )
        values["capabilities"] = thaw_json(FrozenJsonDict(capabilities))
    elif event.event_type == "trace.gap":
        values["trace_complete"] = False

    return state_from_data(values)


def reduce_many(state: BrainState, events: Iterable[EventEnvelope]) -> BrainState:
    """Reduce an ordered iterable without any hidden truncation."""
    current = state
    for event in events:
        current = reduce_state(current, event)
    return current


__all__ = ["reduce_many", "reduce_state"]
