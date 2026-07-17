"""Typed ST/RD/A models and legal action transitions."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    BLOCKED = "blocked"
    DISPATCHED = "dispatched"
    RECEIPT = "receipt"
    RECONSTRUCTED = "reconstructed"


class ActionOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class ActionReceiptStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class ActionReceiptDisposition(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    RESOLUTION = "resolution"
    CORROBORATION = "corroboration"
    CONFLICT = "conflict"


class EnergyAssessmentStatus(StrEnum):
    UNREQUESTED = "unrequested"
    PENDING = "pending"
    ASSESSED = "assessed"
    FAILED = "failed"


MAX_ACTION_RECEIPT_HISTORY = 16
MAX_ACTION_RECONSTRUCTION_HISTORY = 16
MAX_ACTION_SOURCE_ERROR_TYPE_LENGTH = 160
_HERMES_SOURCE_STATUSES = frozenset({"ok", "error", "timeout", "cancelled", "blocked"})
_THREAD_MISSING_RESULT = "thread_missing_result"


def _validate_source_error_type(source_error_type: object) -> None:
    if source_error_type is None:
        return
    if (
        type(source_error_type) is not str
        or not source_error_type.strip()
        or len(source_error_type) > MAX_ACTION_SOURCE_ERROR_TYPE_LENGTH
    ):
        raise ValueError(
            "receipt source semantics require a bounded non-blank error type"
        )


def _validate_receipt_source_semantics(
    status: ActionReceiptStatus,
    source_status: object,
    source_error_type: object,
) -> None:
    """Validate the exact Hermes post-tool status to domain receipt mapping."""
    if source_status is None:
        if source_error_type is not None:
            raise ValueError(
                "receipt source semantics require a source status for error type"
            )
        # Historical and non-Hermes events predate attributed source fields.
        return
    if type(source_status) is not str or source_status not in _HERMES_SOURCE_STATUSES:
        raise ValueError("receipt source semantics use an unsupported source status")
    _validate_source_error_type(source_error_type)
    if source_status == "ok" and source_error_type is not None:
        raise ValueError(
            "receipt source semantics do not allow an error type for ok status"
        )
    if source_status != "error" and source_error_type == _THREAD_MISSING_RESULT:
        raise ValueError(
            "receipt source semantics reserve thread_missing_result for error status"
        )
    if source_status == "blocked":
        raise ValueError("receipt source semantics require an action.blocked event")

    expected = {
        "ok": ActionReceiptStatus.SUCCESS,
        "timeout": ActionReceiptStatus.UNKNOWN,
        "cancelled": ActionReceiptStatus.UNKNOWN,
    }.get(source_status)
    if source_status == "error":
        expected = (
            ActionReceiptStatus.UNKNOWN
            if source_error_type == _THREAD_MISSING_RESULT
            else ActionReceiptStatus.FAILURE
        )
    if status is not expected:
        raise ValueError("receipt source semantics contradict normalized status")


def _validate_blocked_source_semantics(payload: Mapping[str, Any]) -> None:
    source_status = payload.get("source_status")
    source_error_type = payload.get("source_error_type")
    if source_status is not None and (
        type(source_status) is not str or source_status != "blocked"
    ):
        raise DomainInvariantError(
            "blocked source status must be exact 'blocked' when attributed"
        )
    if source_status is None and source_error_type is not None:
        raise DomainInvariantError(
            "blocked error type requires attributed source status"
        )
    try:
        _validate_source_error_type(source_error_type)
    except ValueError as error:
        raise DomainInvariantError("blocked source error type is invalid") from error
    if source_error_type == _THREAD_MISSING_RESULT:
        raise DomainInvariantError(
            "blocked source status cannot use thread_missing_result"
        )


class ActionReceiptRecord(BaseModel):
    """One typed execution receipt with its exact attributed source status."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    event_id: str = Field(min_length=1, max_length=512)
    status: ActionReceiptStatus
    disposition: ActionReceiptDisposition
    source_status: str | None = Field(default=None, min_length=1, max_length=160)
    source_error_type: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ACTION_SOURCE_ERROR_TYPE_LENGTH,
    )
    late: bool = False
    execution_confirmed: bool | None = None
    outcome: ActionOutcome | None = None
    effect_observation_ids: tuple[str, ...] = ()
    payload: FrozenJsonDict

    @field_validator("effect_observation_ids", mode="before")
    @classmethod
    def _json_observation_ids(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("effect_observation_ids")
    @classmethod
    def _bounded_observation_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            len(value) > MAX_WORLD_PROPOSITIONS_PER_LAYER
            or len(value) != len(set(value))
            or any(not item.strip() for item in value)
        ):
            raise ValueError(
                "receipt effect observation IDs must be unique, non-blank and bounded"
            )
        return value

    @model_validator(mode="after")
    def _status_matches_execution_claims(self) -> ActionReceiptRecord:
        _validate_receipt_source_semantics(
            self.status,
            self.source_status,
            self.source_error_type,
        )
        if self.payload.get("status") != self.status.value:
            raise ValueError("receipt payload status does not match typed status")
        if self.payload.get("source_status") != self.source_status:
            raise ValueError(
                "receipt payload source status does not match typed source"
            )
        if self.payload.get("source_error_type") != self.source_error_type:
            raise ValueError("receipt payload error type does not match typed source")
        if self.payload.get("late", False) is not self.late:
            raise ValueError("receipt payload late flag does not match typed receipt")
        if self.status is ActionReceiptStatus.UNKNOWN:
            if self.execution_confirmed is not None or self.outcome is not None:
                raise ValueError("unknown receipt cannot confirm execution or outcome")
            if self.disposition is not ActionReceiptDisposition.PENDING:
                raise ValueError("unknown receipt disposition must remain pending")
            return self
        if self.execution_confirmed is not True:
            raise ValueError("confirmed receipt must confirm execution")
        if self.outcome is not ActionOutcome(self.status.value):
            raise ValueError("confirmed receipt outcome must match its status")
        if self.disposition is ActionReceiptDisposition.PENDING:
            raise ValueError("confirmed receipt disposition cannot remain pending")
        return self


class ActionReconstructionRecord(BaseModel):
    """One typed RD reconstruction linked to the receipt it interprets."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    event_id: str = Field(min_length=1, max_length=512)
    after_receipt_event_id: str | None = Field(default=None, max_length=512)
    payload: FrozenJsonDict


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
    outcome: ActionOutcome | None = None
    effect_confirmed: bool | None = None
    receipt: FrozenJsonDict | None = None
    receipt_history: tuple[ActionReceiptRecord, ...] = ()
    receipt_history_evicted: int = Field(default=0, ge=0)
    receipt_corroboration_count: int = Field(default=0, ge=0)
    receipt_conflict_count: int = Field(default=0, ge=0)
    reconstruction: FrozenJsonDict | None = None
    reconstruction_history: tuple[ActionReconstructionRecord, ...] = ()
    reconstruction_history_evicted: int = Field(default=0, ge=0)
    energy_assessment_status: EnergyAssessmentStatus = (
        EnergyAssessmentStatus.UNREQUESTED
    )
    energy_request_event_id: str | None = None
    energy_assessment_event_id: str | None = None
    energy_failure_code: str | None = Field(default=None, max_length=160)
    proposed_event_id: str
    last_event_id: str

    @field_validator(
        "phase_history",
        "receipt_history",
        "reconstruction_history",
        mode="before",
    )
    @classmethod
    def _json_histories(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("receipt_history")
    @classmethod
    def _bounded_receipt_history(
        cls, value: tuple[ActionReceiptRecord, ...]
    ) -> tuple[ActionReceiptRecord, ...]:
        if len(value) > MAX_ACTION_RECEIPT_HISTORY:
            raise ValueError("action receipt history exceeds its fixed bound")
        return value

    @field_validator("reconstruction_history")
    @classmethod
    def _bounded_reconstruction_history(
        cls, value: tuple[ActionReconstructionRecord, ...]
    ) -> tuple[ActionReconstructionRecord, ...]:
        if len(value) > MAX_ACTION_RECONSTRUCTION_HISTORY:
            raise ValueError("action reconstruction history exceeds its fixed bound")
        return value

    @model_validator(mode="after")
    def _execution_claims_match_lifecycle(self) -> ActionRecord:
        was_blocked = ActionPhase.BLOCKED in self.phase_history
        was_dispatched = ActionPhase.DISPATCHED in self.phase_history
        was_receipted = ActionPhase.RECEIPT in self.phase_history
        if was_blocked and was_dispatched:
            raise ValueError("blocked action cannot claim dispatch")
        if was_blocked and (
            self.execution_confirmed is not False
            or self.outcome is not None
            or self.effect_confirmed is not None
            or self.receipt is not None
            or self.receipt_history
        ):
            raise ValueError("blocked action cannot claim execution or effect")
        if self.outcome is not None and self.execution_confirmed is not True:
            raise ValueError("action outcome requires confirmed execution")
        if self.execution_confirmed is True and self.outcome is None:
            raise ValueError("action execution confirmation requires outcome")
        receipt_claimed = (
            was_receipted
            or self.execution_confirmed is True
            or self.outcome is not None
            or self.effect_confirmed is not None
            or self.receipt is not None
            or bool(self.receipt_history)
        )
        if receipt_claimed and not was_dispatched:
            raise ValueError("action receipt requires observed dispatch")
        if (self.receipt is not None or self.receipt_history) and not was_receipted:
            raise ValueError("action receipt evidence requires receipt phase")
        if self.execution_confirmed is False and not was_blocked:
            raise ValueError("non-execution claim requires a blocked action")
        if self.receipt is not None:
            canonical_status = self.receipt.get("status")
            if canonical_status not in {"success", "failure", "unknown"}:
                raise ValueError("canonical action receipt status is invalid")
            try:
                _validate_receipt_source_semantics(
                    ActionReceiptStatus(canonical_status),
                    self.receipt.get("source_status"),
                    self.receipt.get("source_error_type"),
                )
            except ValueError as error:
                raise ValueError(
                    "canonical receipt source semantics are inconsistent"
                ) from error
            if canonical_status == "unknown":
                if self.execution_confirmed is not None or self.outcome is not None:
                    raise ValueError("unknown canonical receipt cannot confirm outcome")
            elif (
                self.execution_confirmed is not True
                or self.outcome is not ActionOutcome(canonical_status)
            ):
                raise ValueError(
                    "canonical receipt must match confirmed action outcome"
                )
        if self.energy_assessment_status is EnergyAssessmentStatus.UNREQUESTED:
            if any(
                value is not None
                for value in (
                    self.energy_request_event_id,
                    self.energy_assessment_event_id,
                    self.energy_failure_code,
                )
            ):
                raise ValueError("unrequested action energy has terminal evidence")
        elif self.energy_assessment_status is EnergyAssessmentStatus.PENDING:
            if (
                self.energy_request_event_id is None
                or self.energy_assessment_event_id is not None
                or self.energy_failure_code is not None
            ):
                raise ValueError("pending action energy evidence is inconsistent")
        elif self.energy_assessment_status is EnergyAssessmentStatus.ASSESSED:
            if self.energy_assessment_event_id is None or self.energy_failure_code:
                raise ValueError("assessed action energy evidence is inconsistent")
        elif (
            self.energy_assessment_status is EnergyAssessmentStatus.FAILED
            and (
                self.energy_request_event_id is None
                or self.energy_assessment_event_id is not None
                or self.energy_failure_code is None
            )
        ):
            raise ValueError("failed action energy evidence is inconsistent")
        return self


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
        "phase_history": (
            action.phase_history
            if phase in action.phase_history
            else (*action.phase_history, phase)
        ),
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


def _receipt_disposition(
    action: ActionRecord, status: ActionReceiptStatus
) -> ActionReceiptDisposition:
    if status is ActionReceiptStatus.UNKNOWN:
        return ActionReceiptDisposition.PENDING
    incoming = ActionOutcome(status.value)
    if action.outcome is None:
        if action.phase is ActionPhase.DISPATCHED:
            return ActionReceiptDisposition.CONFIRMED
        return ActionReceiptDisposition.RESOLUTION
    if action.outcome is incoming:
        return ActionReceiptDisposition.CORROBORATION
    return ActionReceiptDisposition.CONFLICT


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
        "action.blocked",
        "action.dispatched",
        "action.receipt",
        "action.reconstructed",
        "action.energy_requested",
        "action.energy_assessed",
        "action.energy_assessment_failed",
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
    if event.event_type == "action.energy_requested":
        if action.energy_assessment_status is not EnergyAssessmentStatus.UNREQUESTED:
            raise DomainInvariantError("action energy assessment is already requested")
        action = ActionRecord.model_validate(
            {
                **action.model_dump(mode="python"),
                "energy_assessment_status": EnergyAssessmentStatus.PENDING,
                "energy_request_event_id": event.event_id,
            }
        )
    elif event.event_type == "action.energy_assessed":
        if action.energy_assessment_status not in {
            EnergyAssessmentStatus.UNREQUESTED,
            EnergyAssessmentStatus.PENDING,
        }:
            raise DomainInvariantError("action energy assessment is already terminal")
        action = ActionRecord.model_validate(
            {
                **action.model_dump(mode="python"),
                "energy_assessment_status": EnergyAssessmentStatus.ASSESSED,
                "energy_assessment_event_id": event.event_id,
                "energy_failure_code": None,
            }
        )
    elif event.event_type == "action.energy_assessment_failed":
        failure_code = event.payload.get("failure_code")
        if (
            action.energy_assessment_status is not EnergyAssessmentStatus.PENDING
            or not isinstance(failure_code, str)
            or not failure_code.strip()
            or len(failure_code) > 160
        ):
            raise DomainInvariantError("action energy failure evidence is invalid")
        action = ActionRecord.model_validate(
            {
                **action.model_dump(mode="python"),
                "energy_assessment_status": EnergyAssessmentStatus.FAILED,
                "energy_failure_code": failure_code,
            }
        )
    elif event.event_type == "action.prepared":
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
    elif event.event_type == "action.blocked":
        _validate_blocked_source_semantics(event.payload)
        action = _transition(
            action,
            event,
            required=ActionPhase.PREPARED,
            phase=ActionPhase.BLOCKED,
            rd_phase=RDPhase.PREPARE,
            updates={"execution_confirmed": False},
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
        receipt_status = ActionReceiptStatus(status)
        try:
            _validate_receipt_source_semantics(
                receipt_status,
                event.payload.get("source_status"),
                event.payload.get("source_error_type"),
            )
        except ValueError as error:
            raise DomainInvariantError(
                "receipt source semantics are inconsistent"
            ) from error
        execution = True if status in {"success", "failure"} else None
        incoming_outcome = ActionOutcome(status) if execution is True else None
        late = event.payload.get("late", False)
        if not isinstance(late, bool):
            raise DomainInvariantError("receipt late flag must be a boolean")
        if action.phase is ActionPhase.DISPATCHED:
            if late:
                raise DomainInvariantError("initial receipt cannot be marked late")
        elif action.phase in {ActionPhase.RECEIPT, ActionPhase.RECONSTRUCTED}:
            if not late:
                raise DomainInvariantError("additional receipt must be marked late")
        else:
            raise DomainInvariantError("action must be dispatched before receipt")
        disposition = _receipt_disposition(action, receipt_status)
        grounded_ids = _receipt_grounded_ids(event, trusted=trusted_provenance)
        try:
            receipt_record = ActionReceiptRecord(
                event_id=event.event_id,
                status=receipt_status,
                disposition=disposition,
                source_status=event.payload.get("source_status"),
                source_error_type=event.payload.get("source_error_type"),
                late=late,
                execution_confirmed=execution,
                outcome=incoming_outcome,
                effect_observation_ids=tuple(sorted(grounded_ids)),
                payload=event.payload,
            )
        except Exception as error:
            raise DomainInvariantError("invalid action receipt") from error
        updates = {
            "execution_confirmed": (
                True if execution is True else action.execution_confirmed
            ),
            "outcome": (incoming_outcome if action.outcome is None else action.outcome),
            "effect_confirmed": (
                True
                if grounded_ids or action.effect_confirmed is True
                else action.effect_confirmed
            ),
            "receipt": (
                FrozenJsonDict(event.payload)
                if action.receipt is None
                or disposition is ActionReceiptDisposition.RESOLUTION
                else action.receipt
            ),
            "receipt_history": (
                *action.receipt_history,
                receipt_record,
            )[-MAX_ACTION_RECEIPT_HISTORY:],
            "receipt_history_evicted": (
                action.receipt_history_evicted
                + int(len(action.receipt_history) >= MAX_ACTION_RECEIPT_HISTORY)
            ),
            "receipt_corroboration_count": (
                action.receipt_corroboration_count
                + int(disposition is ActionReceiptDisposition.CORROBORATION)
            ),
            "receipt_conflict_count": (
                action.receipt_conflict_count
                + int(disposition is ActionReceiptDisposition.CONFLICT)
            ),
        }
        if action.phase is ActionPhase.RECEIPT:
            action = ActionRecord.model_validate(
                {
                    **action.model_dump(mode="python"),
                    **updates,
                    "last_event_id": event.event_id,
                }
            )
        else:
            action = _transition(
                action,
                event,
                required=action.phase,
                phase=ActionPhase.RECEIPT,
                rd_phase=RDPhase.PREPARE,
                updates=updates,
            )
    elif event.event_type == "action.reconstructed":
        reconstruction = _mapping(event.payload, field="reconstruction")
        if action.phase not in {ActionPhase.BLOCKED, ActionPhase.RECEIPT}:
            raise DomainInvariantError(
                "action must have a blocked outcome or receipt before reconstruction"
            )
        try:
            reconstruction_record = ActionReconstructionRecord(
                event_id=event.event_id,
                after_receipt_event_id=(
                    action.receipt_history[-1].event_id
                    if action.receipt_history
                    else None
                ),
                payload=FrozenJsonDict(reconstruction),
            )
        except Exception as error:
            raise DomainInvariantError("invalid action reconstruction") from error
        action = _transition(
            action,
            event,
            required=action.phase,
            phase=ActionPhase.RECONSTRUCTED,
            rd_phase=RDPhase.RECONSTRUCT,
            updates={
                "reconstruction": FrozenJsonDict(reconstruction),
                "reconstruction_history": (
                    *action.reconstruction_history,
                    reconstruction_record,
                )[-MAX_ACTION_RECONSTRUCTION_HISTORY:],
                "reconstruction_history_evicted": (
                    action.reconstruction_history_evicted
                    + int(
                        len(action.reconstruction_history)
                        >= MAX_ACTION_RECONSTRUCTION_HISTORY
                    )
                ),
            },
        )
    else:
        return actions, frozenset()
    return _replace(actions, action), grounded_ids


__all__ = [
    "MAX_ACTION_RECEIPT_HISTORY",
    "MAX_ACTION_RECONSTRUCTION_HISTORY",
    "MAX_ACTION_SOURCE_ERROR_TYPE_LENGTH",
    "ActionOutcome",
    "ActionPhase",
    "ActionReceiptDisposition",
    "ActionReceiptRecord",
    "ActionReceiptStatus",
    "ActionReconstructionRecord",
    "ActionRecord",
    "EnergyAssessmentStatus",
    "RDPhase",
    "ThoughtBranch",
    "action_id_from_event",
    "reduce_actions",
]
