"""Typed ST/RD/A models and legal action transitions."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict
from alice_brain_hermes.errors import DomainInvariantError


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


def _receipt_grounded(event: EventEnvelope, *, trusted: bool) -> bool:
    if not trusted:
        return False
    evidence = event.payload.get("effect_evidence")
    observations = event.payload.get("observations")
    if not isinstance(evidence, Mapping) or not isinstance(observations, (list, tuple)):
        return False
    if evidence.get("kind") != "linked_observation":
        return False
    evidence_ids = evidence.get("observation_ids")
    if not isinstance(evidence_ids, (list, tuple)) or not evidence_ids:
        return False
    linked_ids = {
        item.get("proposition_id")
        for item in observations
        if isinstance(item, Mapping)
        and isinstance(item.get("proposition_id"), str)
        and isinstance(item.get("content"), Mapping)
    }
    return all(isinstance(item, str) and item in linked_ids for item in evidence_ids)


def reduce_actions(
    actions: tuple[ActionRecord, ...],
    event: EventEnvelope,
    *,
    trusted_provenance: bool,
) -> tuple[tuple[ActionRecord, ...], bool]:
    """Return new actions and whether this receipt carries grounded effects."""
    lifecycle_events = {
        "action.proposed",
        "action.prepared",
        "action.dispatched",
        "action.receipt",
        "action.reconstructed",
    }
    if event.event_type not in lifecycle_events:
        return actions, False

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
        return (*actions, action), False

    action = _find(actions, action_id)
    grounded = False
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
        grounded = _receipt_grounded(event, trusted=trusted_provenance)
        action = _transition(
            action,
            event,
            required=ActionPhase.DISPATCHED,
            phase=ActionPhase.RECEIPT,
            rd_phase=RDPhase.PREPARE,
            updates={
                "execution_confirmed": execution,
                "effect_confirmed": grounded,
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
        return actions, False
    return _replace(actions, action), grounded


__all__ = [
    "ActionPhase",
    "ActionRecord",
    "RDPhase",
    "ThoughtBranch",
    "action_id_from_event",
    "reduce_actions",
]
