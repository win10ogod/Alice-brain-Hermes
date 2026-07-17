"""Strict wire models for Hermes-hosted action energy assessment."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alice_brain_hermes.core.events import FrozenJsonDict, thaw_json
from alice_brain_hermes.core.personality import ENERGY_DIMENSIONS
from alice_brain_hermes.ids import validate_id

ENERGY_ASSESSMENT_LEASE_SECONDS = 120
MAX_ENERGY_ASSESSMENT_INPUT_BYTES = 262_144


class _StrictEnergyModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class EnergyAssessmentChoiceV1(_StrictEnergyModel):
    """One host-LLM energy vector with an explicit evidence partition."""

    schema_version: Literal[1] = 1
    deficits: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    salience: float = Field(ge=0.0, le=1.0)
    urgency: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=-1.0, le=1.0)
    control: float = Field(ge=0.0, le=1.0)
    resources: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0, le=1.0)
    personality_relevance: float = Field(ge=0.0, le=1.0)
    evidence_basis: FrozenJsonDict
    unknown_dimensions: tuple[str, ...] = ()
    summary: str = Field(min_length=1, max_length=512)

    @field_validator("unknown_dimensions", mode="before")
    @classmethod
    def _json_unknown_dimensions(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("deficits")
    @classmethod
    def _bounded_deficits(cls, value: FrozenJsonDict) -> FrozenJsonDict:
        if len(value) > 16:
            raise ValueError("energy deficits exceed their fixed bound")
        for key, amount in value.items():
            if (
                not isinstance(key, str)
                or not key.strip()
                or len(key) > 160
                or isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(float(amount))
                or not 0.0 <= float(amount) <= 1.0
            ):
                raise ValueError("energy deficits must be bounded numeric evidence")
        return value

    @field_validator("evidence_basis")
    @classmethod
    def _bounded_evidence(cls, value: FrozenJsonDict) -> FrozenJsonDict:
        if len(value) > len(ENERGY_DIMENSIONS):
            raise ValueError("energy evidence exceeds its fixed dimension bound")
        for dimension, basis in value.items():
            if dimension not in ENERGY_DIMENSIONS:
                raise ValueError("unknown evidenced energy dimension")
            if (
                not isinstance(basis, str)
                or not basis.strip()
                or basis != basis.strip()
                or len(basis) > 512
                or len(basis.encode("utf-8")) > 2_048
            ):
                raise ValueError("energy evidence basis must be exact and bounded")
        return value

    @model_validator(mode="after")
    def _evidence_partition(self) -> EnergyAssessmentChoiceV1:
        evidenced = frozenset(self.evidence_basis)
        unknown = frozenset(self.unknown_dimensions)
        known = frozenset(ENERGY_DIMENSIONS)
        if len(unknown) != len(self.unknown_dimensions) or not unknown <= known:
            raise ValueError("unknown energy dimensions are invalid")
        if evidenced & unknown:
            raise ValueError("energy dimensions cannot be both evidenced and unknown")
        if evidenced | unknown != known:
            raise ValueError("energy evidence must classify every dimension")
        return self


class EnergyAssessmentUsageV1(_StrictEnergyModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)


class EnergyAssessmentProvenanceV1(_StrictEnergyModel):
    """Exact host completion identity and usage retained with the vector."""

    schema_version: Literal[1] = 1
    agent_id: str = Field(min_length=1, max_length=512)
    audit: FrozenJsonDict
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1, max_length=512)
    prompt_version: Literal["alice-energy-v1"] = "alice-energy-v1"
    provider: str = Field(min_length=1, max_length=512)
    usage: EnergyAssessmentUsageV1

    @field_validator("audit")
    @classmethod
    def _bounded_audit(cls, value: FrozenJsonDict) -> FrozenJsonDict:
        if len(value) > 16 or any(
            not isinstance(key, str)
            or not isinstance(item, str)
            or len(key) > 160
            or len(item) > 512
            for key, item in value.items()
        ):
            raise ValueError("energy host audit evidence is invalid")
        return value


class EnergyAssessmentLeaseV1(_StrictEnergyModel):
    """A daemon-owned assessment job bound to one persisted action request."""

    schema_version: Literal[1] = 1
    lease_id: str
    brain_id: str
    action_id: str
    request_event_id: str
    state_sequence: int = Field(ge=1)
    expires_at: datetime
    assessment_input: FrozenJsonDict

    @field_validator("lease_id", "brain_id", "request_event_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("action_id")
    @classmethod
    def _action_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip() or len(value) > 512:
            raise ValueError("energy assessment action_id is invalid")
        return value

    @field_validator("expires_at")
    @classmethod
    def _aware_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("energy assessment lease expiry must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("assessment_input")
    @classmethod
    def _bounded_input(cls, value: FrozenJsonDict) -> FrozenJsonDict:
        encoded = json.dumps(
            thaw_json(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > MAX_ENERGY_ASSESSMENT_INPUT_BYTES:
            raise ValueError("energy assessment input exceeds its byte bound")
        return value


__all__ = [
    "ENERGY_ASSESSMENT_LEASE_SECONDS",
    "ENERGY_DIMENSIONS",
    "MAX_ENERGY_ASSESSMENT_INPUT_BYTES",
    "EnergyAssessmentChoiceV1",
    "EnergyAssessmentLeaseV1",
    "EnergyAssessmentProvenanceV1",
    "EnergyAssessmentUsageV1",
]
