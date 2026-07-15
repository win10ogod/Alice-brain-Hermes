"""Typed ST/RD/A models and legal action transitions."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict
from alice_brain_hermes.core.limits import MAX_WORLD_PROPOSITIONS_PER_LAYER
from alice_brain_hermes.errors import DomainCapacityError, DomainInvariantError


class RDPhase(StrEnum):
    SIMULATE = "simulate"
    PREPARE = "prepare"
    RECONSTRUCT = "reconstruct"


class ActionPhase(StrEnum):
    PROPOSED = "proposed"
    PREPARED = "prepared"
    DISPATCHED = "dispatched"
    RECEIPT = "receipt"
    RECONSTRUCTED = "reconstructed"


class ThoughtBranch(BaseModel):
    """An isolated counterfactual ST branch; never an observed fact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    branch_id: str = Field(min_length=1, max_length=256)
    stance: str = Field(min_length=1, max_length=64)
    content: FrozenJsonDict
    expected_consequences: tuple[FrozenJsonDict, ...]
    uncertainty: float = Field(ge=0.0, le=1.0)
    source_ids: tuple[str, ...] = ()
    rd_phase: RDPhase = RDPhase.SIMULATE
    cognition_mode: str = Field(min_length=1, max_length=64)
    algorithm_version: str = Field(min_length=1, max_length=64)
    config_version: str = Field(min_length=1, max_length=64)

    @field_validator("source_ids")
    @classmethod
    def _bounded_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 256 or any(not item.strip() for item in value):
            raise ValueError("source_ids must be non-blank and bounded")
        return value

    @field_validator("expected_consequences", "source_ids", mode="before")
    @classmethod
    def _json_arrays_to_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class ActionRecord(BaseModel):
    """A with explicit proposal, preparation, dispatch, receipt and reflection."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    action_id: str = Field(min_length=1, max_length=512)
    intent: FrozenJsonDict
    phase: ActionPhase
    phase_history: tuple[ActionPhase, ...]
    rd_phase: RDPhase
    prepared_branch_id: str | None = Field(default=None, max_length=256)
    execution_confirmed: bool | None = None
    effect_confirmed: bool | None = None
    receipt: FrozenJsonDict | None = None
    reconstruction: FrozenJsonDict | None = None
    proposed_event_id: str
    last_event_id: str

    @field_validator("phase_history", mode="before")
    @classmethod
    def _json_phase_history(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


def action_id_from_event(event: EventEnvelope) -> str:
    payload_id = event.payload.get("action_id")
    if not isinstance(payload_id, str) or not payload_id.strip():
        raise DomainInvariantError("action event requires a non-blank action_id")
    if event.action_id is not None and event.action_id != payload_id:
        raise DomainInvariantError("envelope/payload action_id mismatch")
    return payload_id


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DomainInvariantError(f"{field} must be an object")
    return value


def _replace(
    actions: tuple[ActionRecord, ...], action: ActionRecord
) -> tuple[ActionRecord, ...]:
    retained = tuple(item for item in actions if item.action_id != action.action_id)
    return (*retained, action)


def _find(actions: tuple[ActionRecord, ...], action_id: str) -> ActionRecord:
    for action in actions:
        if action.action_id == action_id:
            return action
    raise DomainInvariantError(f"action {action_id!r} must be proposed first")


def _transition(
    action: ActionRecord,
    event: EventEnvelope,
    *,
    required: ActionPhase,
    phase: ActionPhase,
    rd_phase: RDPhase,
    updates: dict[str, Any] | None = None,
) -> ActionRecord:
    if action.phase is not required:
        raise DomainInvariantError(
            f"action must be {required.value} before {phase.value}"
        )
    values = {
        "phase": phase,
        "phase_history": (*action.phase_history, phase),
        "rd_phase": rd_phase,
        "last_event_id": event.event_id,
        **(updates or {}),
    }
    try:
        return ActionRecord.model_validate(
            {**action.model_dump(mode="python"), **values}
        )
    except Exception as error:
        raise DomainInvariantError("invalid action transition") from error


def _receipt_grounded_ids(event: EventEnvelope, *, trusted: bool) -> frozenset[str]:
    evidence = event.payload.get("effect_evidence")
    if evidence is None:
        return frozenset()
    if not isinstance(evidence, Mapping):
        raise DomainInvariantError("receipt effect evidence must be an object")
    if evidence.get("kind") != "linked_observation":
        raise DomainInvariantError("receipt effect evidence kind is unsupported")
    evidence_ids = evidence.get("observation_ids")
    if not isinstance(evidence_ids, (list, tuple)) or not evidence_ids:
        raise DomainInvariantError("receipt effect evidence requires observation IDs")
    if any(not isinstance(item, str) or not item.strip() for item in evidence_ids):
        raise DomainInvariantError(
            "receipt effect evidence observation IDs must be non-blank strings"
        )
    if len(evidence_ids) != len(set(evidence_ids)):
        raise DomainInvariantError(
            "receipt effect evidence observation IDs must be unique"
        )
    if len(evidence_ids) > MAX_WORLD_PROPOSITIONS_PER_LAYER:
        raise DomainCapacityError(
            "receipt observation capacity is bounded; receipt was not applied"
        )

    observations = event.payload.get("observations")
    if not isinstance(observations, (list, tuple)):
        raise DomainInvariantError("receipt observations must be an array")
    observations_by_id: dict[str, Mapping[str, Any]] = {}
    for item in observations:
        if not isinstance(item, Mapping):
            raise DomainInvariantError("receipt observation must be an object")
        proposition_id = item.get("proposition_id")
        if not isinstance(proposition_id, str) or not proposition_id.strip():
            raise DomainInvariantError(
                "receipt observation requires a non-blank proposition ID"
            )
        if proposition_id in observations_by_id:
            raise DomainInvariantError(
                f"receipt observation ID {proposition_id!r} is ambiguous"
            )
        observations_by_id[proposition_id] = item

    linked_ids = frozenset(evidence_ids)
    if not linked_ids.issubset(observations_by_id):
        raise DomainInvariantError(
            "receipt effect evidence does not match supplied observations"
        )
    for proposition_id in linked_ids:
        if not isinstance(observations_by_id[proposition_id].get("content"), Mapping):
            raise DomainInvariantError(
                "linked receipt observation content must be an object"
            )
    return linked_ids if trusted else frozenset()


def reduce_actions(
    actions: tuple[ActionRecord, ...],
    event: EventEnvelope,
    *,
    trusted_provenance: bool,
) -> tuple[tuple[ActionRecord, ...], frozenset[str]]:
    """Return new actions and exact receipt observation IDs grounded as effects."""
    lifecycle_events = {
        "action.proposed",
        "action.prepared",
        "action.dispatched",
        "action.receipt",
        "action.reconstructed",
    }
    if event.event_type not in lifecycle_events:
        return actions, frozenset()

    action_id = action_id_from_event(event)
    if event.event_type == "action.proposed":
        if any(item.action_id == action_id for item in actions):
            raise DomainInvariantError(f"action {action_id!r} is already proposed")
        intent = _mapping(event.payload.get("intent"), field="intent")
        try:
            action = ActionRecord(
                action_id=action_id,
                intent=intent,
                phase=ActionPhase.PROPOSED,
                phase_history=(ActionPhase.PROPOSED,),
                rd_phase=RDPhase.SIMULATE,
                proposed_event_id=event.event_id,
                last_event_id=event.event_id,
            )
        except Exception as error:
            raise DomainInvariantError("invalid action proposal") from error
        return (*actions, action), frozenset()

    action = _find(actions, action_id)
    grounded_ids: frozenset[str] = frozenset()
    if event.event_type == "action.prepared":
        branch_id = event.payload.get("branch_id")
        if branch_id is not None and (
            not isinstance(branch_id, str) or not branch_id.strip()
        ):
            raise DomainInvariantError("prepared branch_id must be non-blank")
        action = _transition(
            action,
            event,
            required=ActionPhase.PROPOSED,
            phase=ActionPhase.PREPARED,
            rd_phase=RDPhase.PREPARE,
            updates={"prepared_branch_id": branch_id},
        )
    elif event.event_type == "action.dispatched":
        action = _transition(
            action,
            event,
            required=ActionPhase.PREPARED,
            phase=ActionPhase.DISPATCHED,
            rd_phase=RDPhase.PREPARE,
        )
    elif event.event_type == "action.receipt":
        status = event.payload.get("status")
        if status not in {"success", "failure", "unknown"}:
            raise DomainInvariantError(
                "receipt status must be success, failure or unknown"
            )
        execution = (
            True if status == "success" else False if status == "failure" else None
        )
        grounded_ids = _receipt_grounded_ids(event, trusted=trusted_provenance)
        action = _transition(
            action,
            event,
            required=ActionPhase.DISPATCHED,
            phase=ActionPhase.RECEIPT,
            rd_phase=RDPhase.PREPARE,
            updates={
                "execution_confirmed": execution,
                "effect_confirmed": True if grounded_ids else None,
                "receipt": FrozenJsonDict(event.payload),
            },
        )
    elif event.event_type == "action.reconstructed":
        outcome = _mapping(event.payload, field="reconstruction")
        action = _transition(
            action,
            event,
            required=ActionPhase.RECEIPT,
            phase=ActionPhase.RECONSTRUCTED,
            rd_phase=RDPhase.RECONSTRUCT,
            updates={"reconstruction": FrozenJsonDict(outcome)},
        )
    else:
        return actions, frozenset()
    return _replace(actions, action), grounded_ids


__all__ = [
    "ActionPhase",
    "ActionRecord",
    "RDPhase",
    "ThoughtBranch",
    "action_id_from_event",
    "reduce_actions",
]
