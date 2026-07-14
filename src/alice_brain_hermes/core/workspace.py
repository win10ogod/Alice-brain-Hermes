"""Bounded recurrent global workspace and deterministic specialists."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alice_brain_hermes.core.events import FrozenJsonDict

if TYPE_CHECKING:
    from alice_brain_hermes.core.state import BrainState

SpecialistName = Literal[
    "prediction_error",
    "drives",
    "incomplete_action",
    "memory",
    "self_world_conflict",
]
DEFAULT_WORKSPACE_CAPACITY = 4


class WorkspaceCandidate(BaseModel):
    """One specialist bid for limited recurrent broadcast capacity."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    candidate_id: str = Field(min_length=1, max_length=256)
    specialist: SpecialistName
    score: float = Field(ge=0.0, le=1.0)
    content: FrozenJsonDict
    source_ids: tuple[str, ...] = ()
    cycle: int = Field(ge=1)

    @field_validator("source_ids")
    @classmethod
    def _bounded_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 256 or any(not item.strip() for item in value):
            raise ValueError("workspace source IDs must be non-blank and bounded")
        return value

    @field_validator("source_ids", mode="before")
    @classmethod
    def _json_source_ids(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class MemoryRecord(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    memory_id: str = Field(min_length=1, max_length=256)
    content: FrozenJsonDict
    salience: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    source_ids: tuple[str, ...] = ()

    @field_validator("source_ids", mode="before")
    @classmethod
    def _json_source_ids(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class WorkspaceState(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    capacity: int = Field(default=DEFAULT_WORKSPACE_CAPACITY, ge=1, le=64)
    cycle: int = Field(default=0, ge=0)
    broadcast: tuple[WorkspaceCandidate, ...] = ()

    @field_validator("broadcast", mode="before")
    @classmethod
    def _json_broadcast(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


def choose_broadcast(
    candidates: Iterable[WorkspaceCandidate], *, capacity: int
) -> tuple[WorkspaceCandidate, ...]:
    """Deduplicate by structured content, then score and tie-break stably."""
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("workspace capacity must be an integer")
    if not 1 <= capacity <= 64:
        raise ValueError("workspace capacity must be between 1 and 64")
    best_by_content: dict[str, WorkspaceCandidate] = {}
    for candidate in candidates:
        if not isinstance(candidate, WorkspaceCandidate):
            raise TypeError("workspace candidates must be typed")
        fingerprint = candidate.content.canonical_json()
        previous = best_by_content.get(fingerprint)
        rank = (-candidate.score, candidate.specialist, candidate.candidate_id)
        previous_rank = (
            (-previous.score, previous.specialist, previous.candidate_id)
            if previous is not None
            else None
        )
        if previous_rank is None or rank < previous_rank:
            best_by_content[fingerprint] = candidate
    ordered = sorted(
        best_by_content.values(),
        key=lambda item: (-item.score, item.specialist, item.candidate_id),
    )
    return tuple(ordered[:capacity])


def _candidate_id(cycle: int, specialist: str, content: FrozenJsonDict) -> str:
    body = f"{cycle}:{specialist}:{content.canonical_json()}".encode()
    return f"ws-{hashlib.sha256(body).hexdigest()[:24]}"


def _candidate(
    cycle: int,
    specialist: SpecialistName,
    score: float,
    content: dict[str, Any],
    source_ids: tuple[str, ...] = (),
) -> WorkspaceCandidate:
    frozen = FrozenJsonDict(content)
    return WorkspaceCandidate(
        candidate_id=_candidate_id(cycle, specialist, frozen),
        specialist=specialist,
        score=round(min(1.0, max(0.0, score)), 12),
        content=frozen,
        source_ids=tuple(sorted(set(source_ids))),
        cycle=cycle,
    )


def _prediction_error(state: BrainState, cycle: int) -> WorkspaceCandidate:
    observed = {item.proposition_id: item for item in state.world.observed}
    conflicts = [
        item.proposition_id
        for item in state.world.believed
        if item.proposition_id in observed
        and item.content != observed[item.proposition_id].content
    ]
    return _candidate(
        cycle,
        "prediction_error",
        0.2 + min(0.7, len(conflicts) * 0.2),
        {"conflicting_proposition_ids": sorted(conflicts)},
        tuple(sorted(conflicts)),
    )


def _drives(state: BrainState, cycle: int) -> WorkspaceCandidate:
    ranked = sorted(
        state.energies.values(), key=lambda item: (-item.activation, item.action_id)
    )
    top = ranked[0] if ranked else None
    return _candidate(
        cycle,
        "drives",
        0.15 if top is None else top.activation,
        {
            "action_id": None if top is None else top.action_id,
            "activation": 0.0 if top is None else top.activation,
        },
        () if top is None else (top.action_id,),
    )


def _incomplete_action(state: BrainState, cycle: int) -> WorkspaceCandidate:
    incomplete = sorted(
        action.action_id
        for action in state.actions.values()
        if action.phase.value != "reconstructed"
    )
    return _candidate(
        cycle,
        "incomplete_action",
        0.15 + min(0.75, len(incomplete) * 0.15),
        {"action_ids": incomplete},
        tuple(incomplete),
    )


def _memory(state: BrainState, cycle: int) -> WorkspaceCandidate:
    recent = state.memories[-1] if state.memories else None
    prior_ids = tuple(item.candidate_id for item in state.workspace.broadcast)
    score = 0.15 if recent is None else max(0.15, recent.salience)
    if prior_ids:
        score = min(1.0, score + 0.1)
    return _candidate(
        cycle,
        "memory",
        score,
        {
            "memory_id": None if recent is None else recent.memory_id,
            "prior_broadcast_ids": list(prior_ids),
        },
        (*(() if recent is None else (recent.memory_id,)), *prior_ids),
    )


def _self_world_conflict(state: BrainState, cycle: int) -> WorkspaceCandidate:
    ideal_keys = set(state.personality.narrative_ideal.keys())
    observed_keys = {
        key for item in state.world.observed for key in item.content
    }
    unmatched = sorted(ideal_keys - observed_keys)
    return _candidate(
        cycle,
        "self_world_conflict",
        0.15 + min(0.75, len(unmatched) * 0.15),
        {"unmatched_narrative_keys": unmatched},
        tuple(unmatched),
    )


def derive_candidates(state: BrainState) -> tuple[WorkspaceCandidate, ...]:
    """Run all five specialists; prior broadcast recurs into the next cycle."""
    cycle = state.workspace.cycle + 1
    return (
        _prediction_error(state, cycle),
        _drives(state, cycle),
        _incomplete_action(state, cycle),
        _memory(state, cycle),
        _self_world_conflict(state, cycle),
    )


class WorkspaceCoordinator:
    """Pure coordinator that proposes; the engine must append before reducing."""

    def propose(self, state: BrainState) -> tuple[WorkspaceCandidate, ...]:
        return choose_broadcast(
            derive_candidates(state), capacity=state.workspace.capacity
        )


__all__ = [
    "DEFAULT_WORKSPACE_CAPACITY",
    "MemoryRecord",
    "SpecialistName",
    "WorkspaceCandidate",
    "WorkspaceCoordinator",
    "WorkspaceState",
    "choose_broadcast",
    "derive_candidates",
]
