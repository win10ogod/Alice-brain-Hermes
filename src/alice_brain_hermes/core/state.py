"""Frozen replay-derived foundation state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alice_brain_hermes.core.action import ActionRecord, ThoughtBranch
from alice_brain_hermes.core.cognition import CognitionState
from alice_brain_hermes.core.events import FrozenJsonDict
from alice_brain_hermes.core.identity import IdentityState
from alice_brain_hermes.core.limits import (
    FRAME_PROJECTION_RECORD_BUDGET,
    MAX_ACTION_RECORDS,
    MAX_CAPABILITIES,
    MAX_COGNITION_REFLECTIONS,
    MAX_ENERGY_RECORDS,
    MAX_IDENTITY_ACTORS,
    MAX_MEMORY_RECORDS,
    MAX_PERSONALITY_VALUES_PER_LAYER,
    MAX_PROVENANCE_AUTHORIZATIONS,
    MAX_RAW_LIFECYCLE_KEYS,
    MAX_THOUGHT_BRANCHES,
    MAX_WORKSPACE_BROADCAST,
    MAX_WORLD_PROPOSITIONS_PER_LAYER,
)
from alice_brain_hermes.core.personality import EnergyVector, PersonalityControl
from alice_brain_hermes.core.workspace import MemoryRecord, WorkspaceState
from alice_brain_hermes.core.world import WorldModel
from alice_brain_hermes.ids import validate_id

STATE_SCHEMA_VERSION = 3


class WorkingSetCounter(BaseModel):
    """Replay-visible admission and historical-detail eviction totals."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    total: int = Field(default=0, ge=0)
    evicted: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _evictions_do_not_exceed_total(self) -> WorkingSetCounter:
        if self.evicted > self.total:
            raise ValueError("working-set evictions cannot exceed admissions")
        return self


class WorkingSetState(BaseModel):
    """Canonical counters for every history-accumulating state collection."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    action_records: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    energy_records: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    thought_space: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    memories: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    cognition_reflections: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    world_observed: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    world_believed: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    world_simulated: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    world_ideal: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    identity_actors: WorkingSetCounter = Field(
        default_factory=lambda: WorkingSetCounter(total=1)
    )
    provenance_authorizations: WorkingSetCounter = Field(
        default_factory=WorkingSetCounter
    )
    personality_traits: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    personality_adaptations: WorkingSetCounter = Field(
        default_factory=WorkingSetCounter
    )
    personality_narrative_ideal: WorkingSetCounter = Field(
        default_factory=WorkingSetCounter
    )
    raw_lifecycle_counts: WorkingSetCounter = Field(default_factory=WorkingSetCounter)
    raw_lifecycle_events_evicted: int = Field(default=0, ge=0)
    reduced_event_count: int = Field(default=0, ge=0)
    capabilities: WorkingSetCounter = Field(default_factory=WorkingSetCounter)


class RuntimeFailure(BaseModel):
    """Sanitized, bounded failure evidence persisted by the scheduler."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    error_type: str = Field(min_length=1, max_length=160)
    message: str = Field(min_length=1, max_length=512)
    phase: str = Field(min_length=1, max_length=160)


class RuntimeState(BaseModel):
    """Replay-derived continuous-runtime health, not an in-memory assertion."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    health: Literal["healthy", "degraded"] = "healthy"
    tick_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    last_elapsed_seconds: float = Field(default=0.0, ge=0.0)
    last_failure: RuntimeFailure | None = None


class BrainState(BaseModel):
    """Deterministic state produced only by event reduction."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    schema_version: Literal[STATE_SCHEMA_VERSION] = STATE_SCHEMA_VERSION
    brain_id: str
    name: str | None = None
    identity: IdentityState
    personality: PersonalityControl = Field(default_factory=PersonalityControl)
    energy_records: tuple[EnergyVector, ...] = ()
    thought_space: tuple[ThoughtBranch, ...] = ()
    action_records: tuple[ActionRecord, ...] = ()
    world: WorldModel = Field(default_factory=WorldModel)
    workspace: WorkspaceState = Field(default_factory=WorkspaceState)
    memories: tuple[MemoryRecord, ...] = ()
    cognition: CognitionState = Field(default_factory=CognitionState)
    runtime: RuntimeState = Field(default_factory=RuntimeState)
    capabilities: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    working_set: WorkingSetState
    logical_clock: float = Field(default=0.0, ge=0.0)
    trace_complete: bool = True
    raw_lifecycle_counts: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    last_sequence: int = Field(default=0, ge=0)

    @field_validator("brain_id")
    @classmethod
    def _validate_brain_id(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("name must be non-blank when present")
        return value

    @field_validator("raw_lifecycle_counts")
    @classmethod
    def _validate_raw_counts(cls, value: FrozenJsonDict) -> FrozenJsonDict:
        for event_type, count in value.items():
            if not event_type or isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    "raw lifecycle counts must map event types to integers"
                )
            if count < 0:
                raise ValueError("raw lifecycle counts cannot be negative")
        return value

    @field_validator(
        "energy_records",
        "thought_space",
        "action_records",
        "memories",
        mode="before",
    )
    @classmethod
    def _json_arrays_to_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="before")
    @classmethod
    def _supply_identity_for_legacy_foundation(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "identity" not in value:
            brain_id = value.get("brain_id")
            if isinstance(brain_id, str):
                identity = IdentityState.genesis(brain_id).model_copy(
                    update={"name": value.get("name")}
                )
                return {**value, "identity": identity}
        return value

    @model_validator(mode="after")
    def _identity_matches_brain(self) -> BrainState:
        if self.identity.self_actor_id != self.brain_id:
            raise ValueError("identity self_actor_id must equal brain_id")
        if self.identity.name != self.name:
            raise ValueError("foundation and identity names must match")
        for layer in ("traits", "adaptations", "narrative_ideal"):
            bucket = getattr(self.personality.rate_state, layer)
            if bucket.logical_clock != self.logical_clock:
                raise ValueError(
                    "personality rate state logical clock must match brain "
                    "logical clock"
                )
        action_ids = [item.action_id for item in self.action_records]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action records must have unique action IDs")
        energy_ids = [item.action_id for item in self.energy_records]
        if len(energy_ids) != len(set(energy_ids)):
            raise ValueError("energy records must have unique action IDs")
        if not set(energy_ids).issubset(action_ids):
            raise ValueError("energy records must reference retained action IDs")

        bounded_collections = (
            ("action records", len(self.action_records), MAX_ACTION_RECORDS),
            ("energy records", len(self.energy_records), MAX_ENERGY_RECORDS),
            ("thought branches", len(self.thought_space), MAX_THOUGHT_BRANCHES),
            ("memories", len(self.memories), MAX_MEMORY_RECORDS),
            (
                "cognition reflections",
                len(self.cognition.reflections),
                MAX_COGNITION_REFLECTIONS,
            ),
            ("identity actors", len(self.identity.actors), MAX_IDENTITY_ACTORS),
            (
                "provenance authorizations",
                len(self.identity.authorizations),
                MAX_PROVENANCE_AUTHORIZATIONS,
            ),
            ("capabilities", len(self.capabilities), MAX_CAPABILITIES),
            (
                "raw lifecycle keys",
                len(self.raw_lifecycle_counts),
                MAX_RAW_LIFECYCLE_KEYS,
            ),
        )
        for label, current, capacity in bounded_collections:
            if current > capacity:
                raise ValueError(f"{label} exceed the bounded working-set capacity")
        for layer in ("traits", "adaptations", "narrative_ideal"):
            if len(getattr(self.personality, layer)) > (
                MAX_PERSONALITY_VALUES_PER_LAYER
            ):
                raise ValueError(
                    f"personality {layer} exceeds the bounded working-set capacity"
                )
        for layer in ("observed", "believed", "simulated", "ideal"):
            if len(getattr(self.world, layer)) > (MAX_WORLD_PROPOSITIONS_PER_LAYER):
                raise ValueError(
                    f"world {layer} exceeds the bounded working-set capacity"
                )
        if len(self.workspace.broadcast) > self.workspace.capacity:
            raise ValueError("workspace broadcast exceeds its declared capacity")

        exact_counters = (
            ("action_records", len(self.action_records)),
            ("energy_records", len(self.energy_records)),
            ("thought_space", len(self.thought_space)),
            ("memories", len(self.memories)),
            ("cognition_reflections", len(self.cognition.reflections)),
            ("world_observed", len(self.world.observed)),
            ("world_believed", len(self.world.believed)),
            ("world_simulated", len(self.world.simulated)),
            ("world_ideal", len(self.world.ideal)),
            ("identity_actors", len(self.identity.actors)),
            (
                "provenance_authorizations",
                len(self.identity.authorizations),
            ),
            ("personality_traits", len(self.personality.traits)),
            ("personality_adaptations", len(self.personality.adaptations)),
            (
                "personality_narrative_ideal",
                len(self.personality.narrative_ideal),
            ),
            ("raw_lifecycle_counts", len(self.raw_lifecycle_counts)),
            ("capabilities", len(self.capabilities)),
        )
        for name, retained in exact_counters:
            counter = getattr(self.working_set, name)
            if counter.total - counter.evicted != retained:
                raise ValueError(
                    f"working-set {name} counter does not match retained state"
                )
        for name in (
            "identity_actors",
            "provenance_authorizations",
            "personality_traits",
            "personality_adaptations",
            "personality_narrative_ideal",
            "capabilities",
        ):
            if getattr(self.working_set, name).evicted != 0:
                raise ValueError(f"working-set {name} cannot claim a silent eviction")
        retained_raw_events = sum(self.raw_lifecycle_counts.values())
        if (
            retained_raw_events + self.working_set.raw_lifecycle_events_evicted
            != self.working_set.reduced_event_count
        ):
            raise ValueError("raw lifecycle event counters do not match reduced events")
        if self.last_sequence > self.working_set.reduced_event_count:
            raise ValueError("persisted sequence exceeds the reduced event count")
        return self

    @classmethod
    def genesis(cls, brain_id: str) -> BrainState:
        """Create the unnamed, capability-neutral foundation for one brain."""
        return cls(
            brain_id=brain_id,
            identity=IdentityState.genesis(brain_id),
            working_set=WorkingSetState(),
        )

    @property
    def actions(self) -> Mapping[str, ActionRecord]:
        return MappingProxyType({item.action_id: item for item in self.action_records})

    @property
    def energies(self) -> Mapping[str, EnergyVector]:
        return MappingProxyType({item.action_id: item for item in self.energy_records})

    def canonical_json(self) -> str:
        """Return a stable snapshot representation."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BrainState):
            try:
                return self.canonical_json() == other.canonical_json()
            except (TypeError, ValueError):
                return False
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.canonical_json())

    def revalidated(self) -> BrainState:
        """Revalidate values created through Pydantic's unchecked model_copy."""
        return BrainState.model_validate(self.model_dump(mode="python"))


def state_from_data(values: dict[str, Any]) -> BrainState:
    """Build a validated state from reducer-owned ordinary values."""
    return BrainState.model_validate(values)


__all__ = [
    "FRAME_PROJECTION_RECORD_BUDGET",
    "MAX_ACTION_RECORDS",
    "MAX_CAPABILITIES",
    "MAX_COGNITION_REFLECTIONS",
    "MAX_ENERGY_RECORDS",
    "MAX_IDENTITY_ACTORS",
    "MAX_MEMORY_RECORDS",
    "MAX_PERSONALITY_VALUES_PER_LAYER",
    "MAX_PROVENANCE_AUTHORIZATIONS",
    "MAX_RAW_LIFECYCLE_KEYS",
    "MAX_THOUGHT_BRANCHES",
    "MAX_WORKSPACE_BROADCAST",
    "MAX_WORLD_PROPOSITIONS_PER_LAYER",
    "STATE_SCHEMA_VERSION",
    "BrainState",
    "RuntimeFailure",
    "RuntimeState",
    "WorkingSetCounter",
    "WorkingSetState",
    "state_from_data",
]
