"""Pure foundation reducer; the sole mutation path for brain state."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from alice_brain_hermes.core.action import (
    ActionPhase,
    RDPhase,
    ThoughtBranch,
    action_id_from_event,
    reduce_actions,
)
from alice_brain_hermes.core.cognition import cognition_result_from_payload
from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict, thaw_json
from alice_brain_hermes.core.identity import reduce_identity
from alice_brain_hermes.core.limits import (
    MAX_ACTION_RECORDS,
    MAX_CAPABILITIES,
    MAX_COGNITION_REFLECTIONS,
    MAX_ENERGY_RECORDS,
    MAX_MEMORY_RECORDS,
    MAX_RAW_LIFECYCLE_KEYS,
    MAX_THOUGHT_BRANCHES,
)
from alice_brain_hermes.core.personality import (
    ENERGY_DIMENSIONS,
    advance_personality_clock,
    energy_from_event,
    reduce_personality,
    upsert_energy,
)
from alice_brain_hermes.core.state import (
    BrainState,
    RuntimeFailure,
    RuntimeState,
    WorkingSetCounter,
    WorkingSetState,
    state_from_data,
)
from alice_brain_hermes.core.workspace import (
    MemoryRecord,
    WorkspaceCandidate,
    WorkspaceState,
    choose_broadcast,
)
from alice_brain_hermes.core.world import reduce_world
from alice_brain_hermes.errors import (
    DomainCapacityError,
    DomainInvariantError,
    SchemaVersionError,
)


def _state_values(state: BrainState) -> dict[str, Any]:
    return state.model_dump(mode="python")


def _advance_working_set(
    working_set: WorkingSetState,
    name: str,
    *,
    admitted: int = 0,
    evicted: int = 0,
) -> WorkingSetState:
    counter = getattr(working_set, name)
    return working_set.model_copy(
        update={
            name: WorkingSetCounter(
                total=counter.total + admitted,
                evicted=counter.evicted + evicted,
            )
        }
    )


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
    return (*retained, memory)[-MAX_MEMORY_RECORDS:]


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
            expected_consequences=tuple(FrozenJsonDict(item) for item in consequences),
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
    working_set = state.working_set.model_copy(
        update={"reduced_event_count": state.working_set.reduced_event_count + 1}
    )
    counts = dict(state.raw_lifecycle_counts.items())
    if event.event_type not in counts:
        raw_evicted_events = working_set.raw_lifecycle_events_evicted
        raw_key_evicted = 0
        if len(counts) >= MAX_RAW_LIFECYCLE_KEYS:
            evicted_key = sorted(counts)[0]
            raw_evicted_events += counts.pop(evicted_key)
            raw_key_evicted = 1
        working_set = _advance_working_set(
            working_set,
            "raw_lifecycle_counts",
            admitted=1,
            evicted=raw_key_evicted,
        ).model_copy(update={"raw_lifecycle_events_evicted": raw_evicted_events})
    counts[event.event_type] = counts.get(event.event_type, 0) + 1
    values["raw_lifecycle_counts"] = dict(sorted(counts.items()))
    if event.sequence is not None:
        values["last_sequence"] = event.sequence

    trusted_provenance = state.identity.is_authorized(event.actor_id, event.adapter_id)
    identity = reduce_identity(state.identity, event)
    values["identity"] = identity
    if len(identity.actors) > len(state.identity.actors):
        working_set = _advance_working_set(working_set, "identity_actors", admitted=1)
    if len(identity.authorizations) > len(state.identity.authorizations):
        working_set = _advance_working_set(
            working_set, "provenance_authorizations", admitted=1
        )
    personality = reduce_personality(
        state.personality,
        event,
        logical_clock=state.logical_clock,
    )
    values["personality"] = personality
    for layer in ("traits", "adaptations", "narrative_ideal"):
        admitted = len(getattr(personality, layer)) - len(
            getattr(state.personality, layer)
        )
        if admitted:
            working_set = _advance_working_set(
                working_set, f"personality_{layer}", admitted=admitted
            )

    previous_action_ids = {item.action_id for item in state.action_records}
    actions, grounded_receipt = reduce_actions(
        state.action_records,
        event,
        trusted_provenance=trusted_provenance,
    )
    admitted_actions = sum(
        item.action_id not in previous_action_ids for item in actions
    )
    evicted_action_id: str | None = None
    if len(actions) > MAX_ACTION_RECORDS:
        terminal_index = next(
            (
                index
                for index, item in enumerate(actions)
                if item.phase in {ActionPhase.BLOCKED, ActionPhase.RECONSTRUCTED}
            ),
            None,
        )
        if terminal_index is None:
            raise DomainCapacityError(
                "active action capacity is full; proposal was not applied"
            )
        evicted_action_id = actions[terminal_index].action_id
        actions = actions[:terminal_index] + actions[terminal_index + 1 :]
    if admitted_actions:
        working_set = _advance_working_set(
            working_set,
            "action_records",
            admitted=admitted_actions,
            evicted=int(evicted_action_id is not None),
        )
    values["action_records"] = actions
    if evicted_action_id is not None:
        retained_energy = tuple(
            item for item in state.energy_records if item.action_id != evicted_action_id
        )
        if len(retained_energy) != len(state.energy_records):
            values["energy_records"] = retained_energy
            working_set = _advance_working_set(working_set, "energy_records", evicted=1)

    world = reduce_world(
        state.world,
        event,
        trusted_provenance=trusted_provenance,
        grounded_receipt=grounded_receipt,
    )
    values["world"] = world
    for layer in ("observed", "believed", "simulated", "ideal"):
        previous = getattr(state.world, layer)
        current = getattr(world, layer)
        previous_ids = {item.proposition_id for item in previous}
        current_ids = {item.proposition_id for item in current}
        retained_existing = len(previous_ids & current_ids)
        admitted = len(current) - retained_existing
        evicted = len(previous) + admitted - len(current)
        if admitted or evicted:
            working_set = _advance_working_set(
                working_set,
                f"world_{layer}",
                admitted=admitted,
                evicted=evicted,
            )
    values["workspace"] = _reduce_workspace(state, event)
    memories = _reduce_memory(state, event)
    values["memories"] = memories
    if event.event_type == "memory.recorded":
        memory_id = event.payload.get("memory_id")
        admitted_memory = int(
            isinstance(memory_id, str)
            and all(item.memory_id != memory_id for item in state.memories)
        )
        memory_evicted = max(0, len(state.memories) + admitted_memory - len(memories))
        working_set = _advance_working_set(
            working_set,
            "memories",
            admitted=admitted_memory,
            evicted=memory_evicted,
        )

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
        if len(capabilities) > MAX_CAPABILITIES:
            raise DomainCapacityError(
                "capabilities exceed the bounded working-set capacity"
            )
        values["capabilities"] = thaw_json(FrozenJsonDict(capabilities))
        working_set = working_set.model_copy(
            update={"capabilities": WorkingSetCounter(total=len(capabilities))}
        )
    elif event.event_type in {"semantic.gap", "trace.gap"}:
        values["trace_complete"] = False
    elif (
        event.event_type == "action.energy_requested"
        and event.payload.get("reassessment_reason") == "legacy_neutral_default"
    ):
        action_id = action_id_from_event(event)
        legacy = state.energies.get(action_id)
        if (
            legacy is None
            or legacy.assessment_source is not None
            or legacy.assessment_summary is not None
            or legacy.provenance
            or legacy.evidence_basis
            or legacy.unknown_dimensions != ENERGY_DIMENSIONS
            or legacy.deficits
            or legacy.salience != 0.5
            or legacy.urgency != 0.5
            or legacy.valence != 0.0
            or legacy.arousal != 0.0
            or legacy.control != 0.5
            or legacy.resources != 0.5
            or legacy.cost != 0.5
            or legacy.personality_relevance != 0.5
        ):
            raise DomainInvariantError(
                "legacy energy reassessment requires the exact neutral vector"
            )
        retained_energy = tuple(
            item for item in state.energy_records if item.action_id != action_id
        )
        if len(retained_energy) == len(state.energy_records):
            raise DomainInvariantError(
                "legacy energy reassessment requires its prior vector"
            )
        values["energy_records"] = retained_energy
        working_set = _advance_working_set(
            working_set,
            "energy_records",
            evicted=1,
        )
    elif event.event_type == "action.energy_assessed":
        action_id = action_id_from_event(event)
        if action_id not in state.actions:
            raise DomainInvariantError("energy assessment requires an existing action")
        energy = energy_from_event(event)
        energy_source = state.energy_records
        energies = upsert_energy(energy_source, energy)
        admitted_energy = int(
            all(item.action_id != action_id for item in energy_source)
        )
        energy_evicted = 0
        if len(energies) > MAX_ENERGY_RECORDS:
            energies = energies[-MAX_ENERGY_RECORDS:]
            energy_evicted = 1
        values["energy_records"] = energies
        working_set = _advance_working_set(
            working_set,
            "energy_records",
            admitted=admitted_energy,
            evicted=energy_evicted,
        )
    elif event.event_type == "simulation.created":
        branch = _simulation_branch(event)
        admitted_branch = int(
            all(item.branch_id != branch.branch_id for item in state.thought_space)
        )
        retained = tuple(
            item for item in state.thought_space if item.branch_id != branch.branch_id
        )
        branches = (*retained, branch)[-MAX_THOUGHT_BRANCHES:]
        values["thought_space"] = branches
        working_set = _advance_working_set(
            working_set,
            "thought_space",
            admitted=admitted_branch,
            evicted=max(0, len(state.thought_space) + admitted_branch - len(branches)),
        )
    elif event.event_type == "cognition.reflected":
        try:
            result = cognition_result_from_payload(event.payload)
        except Exception as error:
            raise DomainInvariantError("invalid local cognition result") from error
        if result.cognition_mode != "local" or result.provider_used is not False:
            raise DomainInvariantError("local cognition cannot claim provider use")
        values["cognition"] = state.cognition.model_copy(
            update={
                "reflections": (
                    *state.cognition.reflections,
                    result,
                )[-MAX_COGNITION_REFLECTIONS:]
            }
        )
        working_set = _advance_working_set(
            working_set,
            "cognition_reflections",
            admitted=1,
            evicted=int(len(state.cognition.reflections) >= MAX_COGNITION_REFLECTIONS),
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
        old_branch_ids = {item.branch_id for item in state.thought_space}
        existing = {item.branch_id: item for item in (*state.thought_space, *branches)}
        new_branch_count = sum(
            item.branch_id not in old_branch_ids for item in branches
        )
        bounded_branches = tuple(existing.values())[-MAX_THOUGHT_BRANCHES:]
        values["thought_space"] = bounded_branches
        working_set = _advance_working_set(
            working_set,
            "thought_space",
            admitted=new_branch_count,
            evicted=max(
                0,
                len(state.thought_space) + new_branch_count - len(bounded_branches),
            ),
        )
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

    values["working_set"] = working_set
    return state_from_data(values)


def reduce_many(state: BrainState, events: Iterable[EventEnvelope]) -> BrainState:
    """Reduce an ordered iterable without any hidden truncation."""
    current = state
    for event in events:
        current = reduce_state(current, event)
    return current


__all__ = ["reduce_many", "reduce_state"]
