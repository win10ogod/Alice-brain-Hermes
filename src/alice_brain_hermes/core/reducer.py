"""Pure foundation reducer; the sole mutation path for brain state."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from alice_brain_hermes.core.action import (
    RDPhase,
    ThoughtBranch,
    action_id_from_event,
    reduce_actions,
)
from alice_brain_hermes.core.cognition import cognition_result_from_payload
from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict, thaw_json
from alice_brain_hermes.core.identity import reduce_identity
from alice_brain_hermes.core.personality import (
    advance_personality_clock,
    energy_from_event,
    reduce_personality,
    upsert_energy,
)
from alice_brain_hermes.core.state import (
    BrainState,
    RuntimeFailure,
    RuntimeState,
    state_from_data,
)
from alice_brain_hermes.core.workspace import (
    MemoryRecord,
    WorkspaceCandidate,
    WorkspaceState,
    choose_broadcast,
)
from alice_brain_hermes.core.world import reduce_world
from alice_brain_hermes.errors import DomainInvariantError, SchemaVersionError


def _state_values(state: BrainState) -> dict[str, Any]:
    return state.model_dump(mode="python")


def _payload_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DomainInvariantError(f"{field} must be a JSON object")
    return value


def _workspace_candidate(value: Any) -> WorkspaceCandidate:
    item = _payload_mapping(value, field="workspace candidate")
    try:
        return WorkspaceCandidate(
            candidate_id=item["candidate_id"],
            specialist=item["specialist"],
            score=float(item["score"]),
            content=item["content"],
            source_ids=tuple(item.get("source_ids", ())),
            cycle=int(item["cycle"]),
        )
    except Exception as error:
        raise DomainInvariantError("invalid workspace candidate") from error


def _reduce_workspace(state: BrainState, event: EventEnvelope) -> WorkspaceState:
    if event.event_type != "workspace.broadcast":
        return state.workspace
    cycle = event.payload.get("cycle")
    if isinstance(cycle, bool) or not isinstance(cycle, int):
        raise DomainInvariantError("workspace cycle must be an integer")
    if cycle != state.workspace.cycle + 1:
        raise DomainInvariantError("workspace cycle must advance exactly once")
    raw_candidates = event.payload.get("candidates")
    if not isinstance(raw_candidates, (list, tuple)):
        raise DomainInvariantError("workspace candidates must be an array")
    candidates = tuple(_workspace_candidate(item) for item in raw_candidates)
    if any(item.cycle != cycle for item in candidates):
        raise DomainInvariantError("candidate cycle must match workspace cycle")
    selected = choose_broadcast(candidates, capacity=state.workspace.capacity)
    if len(selected) != len(candidates) or selected != candidates:
        raise DomainInvariantError(
            "workspace event must already be bounded, deduplicated and ranked"
        )
    return WorkspaceState(
        capacity=state.workspace.capacity,
        cycle=cycle,
        broadcast=selected,
    )


def _reduce_memory(state: BrainState, event: EventEnvelope) -> tuple[MemoryRecord, ...]:
    if event.event_type != "memory.recorded":
        return state.memories
    try:
        memory = MemoryRecord(
            memory_id=event.payload["memory_id"],
            content=event.payload["content"],
            salience=float(event.payload.get("salience", 0.5)),
            source_ids=tuple(event.payload.get("source_ids", ())),
        )
    except Exception as error:
        raise DomainInvariantError("invalid memory record") from error
    retained = tuple(
        item for item in state.memories if item.memory_id != memory.memory_id
    )
    return (*retained, memory)[-256:]


def _runtime_failure(event: EventEnvelope) -> RuntimeFailure:
    try:
        return RuntimeFailure(
            error_type=event.payload["error_type"],
            message=event.payload["message"],
            phase=event.payload["phase"],
        )
    except Exception as error:
        raise DomainInvariantError("invalid runtime failure") from error


def _simulation_branch(event: EventEnvelope) -> ThoughtBranch:
    proposition_id = event.payload.get("proposition_id")
    if not isinstance(proposition_id, str) or not proposition_id.strip():
        raise DomainInvariantError("simulation requires proposition_id")
    consequences = event.payload.get(
        "expected_consequences",
        ({"kind": "counterfactual", "requires_confirmation": True},),
    )
    if not isinstance(consequences, (list, tuple)):
        raise DomainInvariantError("simulation consequences must be an array")
    try:
        return ThoughtBranch(
            branch_id=event.payload.get("branch_id", proposition_id),
            stance=event.payload.get("stance", "simulate"),
            content=event.payload["content"],
            expected_consequences=tuple(
                FrozenJsonDict(item) for item in consequences
            ),
            uncertainty=float(event.payload.get("uncertainty", 0.5)),
            source_ids=tuple(event.payload.get("source_ids", ())),
            rd_phase=RDPhase.SIMULATE,
            cognition_mode=event.payload.get("cognition_mode", "event"),
            algorithm_version=event.payload.get("algorithm_version", "event-v1"),
            config_version=event.payload.get("config_version", "event-v1"),
        )
    except Exception as error:
        raise DomainInvariantError("invalid simulation branch") from error


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

    trusted_provenance = state.identity.is_authorized(
        event.actor_id, event.adapter_id
    )
    identity = reduce_identity(state.identity, event)
    values["identity"] = identity
    values["personality"] = reduce_personality(
        state.personality,
        event,
        logical_clock=state.logical_clock,
    )

    actions, grounded_receipt = reduce_actions(
        state.action_records,
        event,
        trusted_provenance=trusted_provenance,
    )
    values["action_records"] = actions
    values["world"] = reduce_world(
        state.world,
        event,
        trusted_provenance=trusted_provenance,
        grounded_receipt=grounded_receipt,
    )
    values["workspace"] = _reduce_workspace(state, event)
    values["memories"] = _reduce_memory(state, event)

    if event.event_type in {"brain.created", "identity.named"}:
        name = event.payload.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise DomainInvariantError("name must be null or a non-blank string")
        values["name"] = name
        if event.event_type == "brain.created" and identity.name != name:
            values["identity"] = identity.model_copy(update={"name": name})
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
        values["personality"] = advance_personality_clock(
            state.personality, logical_clock
        )
        values["runtime"] = RuntimeState(
            **{
                **state.runtime.model_dump(mode="python"),
                "tick_count": state.runtime.tick_count + 1,
                "last_elapsed_seconds": float(elapsed),
            }
        )
    elif event.event_type == "capabilities.reported":
        capabilities = _payload_mapping(
            event.payload.get("capabilities"), field="capabilities"
        )
        values["capabilities"] = thaw_json(FrozenJsonDict(capabilities))
    elif event.event_type == "trace.gap":
        values["trace_complete"] = False
    elif event.event_type == "action.energy_assessed":
        action_id = action_id_from_event(event)
        if action_id not in state.actions:
            raise DomainInvariantError("energy assessment requires an existing action")
        energy = energy_from_event(event)
        values["energy_records"] = upsert_energy(state.energy_records, energy)
    elif event.event_type == "simulation.created":
        branch = _simulation_branch(event)
        retained = tuple(
            item for item in state.thought_space if item.branch_id != branch.branch_id
        )
        values["thought_space"] = (*retained, branch)[-256:]
    elif event.event_type == "cognition.reflected":
        try:
            result = cognition_result_from_payload(event.payload)
        except Exception as error:
            raise DomainInvariantError("invalid local cognition result") from error
        if result.cognition_mode != "local" or result.provider_used is not False:
            raise DomainInvariantError("local cognition cannot claim provider use")
        values["cognition"] = state.cognition.model_copy(
            update={"reflections": (*state.cognition.reflections, result)[-128:]}
        )
        branches = tuple(
            ThoughtBranch(
                branch_id=item.branch_id,
                stance=item.stance,
                content=item.content,
                expected_consequences=item.expected_consequences,
                uncertainty=result.uncertainty,
                source_ids=result.source_ids,
                rd_phase=RDPhase.SIMULATE,
                cognition_mode=result.cognition_mode,
                algorithm_version=result.algorithm_version,
                config_version=result.config_version,
            )
            for item in result.alternatives
        )
        existing = {
            item.branch_id: item for item in (*state.thought_space, *branches)
        }
        values["thought_space"] = tuple(existing.values())[-256:]
    elif event.event_type == "runtime.failure":
        failure = _runtime_failure(event)
        values["runtime"] = RuntimeState(
            health="degraded",
            tick_count=state.runtime.tick_count,
            failure_count=state.runtime.failure_count + 1,
            consecutive_failures=state.runtime.consecutive_failures + 1,
            last_elapsed_seconds=state.runtime.last_elapsed_seconds,
            last_failure=failure,
        )
    elif event.event_type == "runtime.recovered":
        values["runtime"] = RuntimeState(
            health="healthy",
            tick_count=state.runtime.tick_count,
            failure_count=state.runtime.failure_count,
            consecutive_failures=0,
            last_elapsed_seconds=state.runtime.last_elapsed_seconds,
            last_failure=state.runtime.last_failure,
        )

    return state_from_data(values)


def reduce_many(state: BrainState, events: Iterable[EventEnvelope]) -> BrainState:
    """Reduce an ordered iterable without any hidden truncation."""
    current = state
    for event in events:
        current = reduce_state(current, event)
    return current


__all__ = ["reduce_many", "reduce_state"]
