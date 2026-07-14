"""Three-layer personality control and action-indexed energy vectors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict
from alice_brain_hermes.errors import DomainInvariantError

PersonalityLayer = Literal["traits", "adaptations", "narrative_ideal"]
_MAX_DELTAS: dict[str, float] = {
    "traits": 0.05,
    "adaptations": 0.20,
    "narrative_ideal": 0.10,
}


def _validate_numeric_map(
    values: FrozenJsonDict, *, lower: float, upper: float, label: str
) -> FrozenJsonDict:
    for key, value in values.items():
        if (
            not key.strip()
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not lower <= float(value) <= upper
        ):
            raise ValueError(
                f"{label} must map names to finite values in [{lower}, {upper}]"
            )
    return values


class PersonalityControl(BaseModel):
    """PC: slow traits, contextual adaptations, and narrative/ideal self."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    traits: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    adaptations: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    narrative_ideal: FrozenJsonDict = Field(default_factory=FrozenJsonDict)

    @field_validator("traits", "adaptations", "narrative_ideal")
    @classmethod
    def _validate_layers(cls, value: FrozenJsonDict) -> FrozenJsonDict:
        return _validate_numeric_map(
            value, lower=-1.0, upper=1.0, label="personality layers"
        )

    def revalidated(self) -> PersonalityControl:
        return PersonalityControl.model_validate(self.model_dump(mode="python"))


class EnergyVector(BaseModel):
    """E: bounded action-indexed activation evidence, never a dispatcher."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    action_id: str = Field(min_length=1, max_length=512)
    deficits: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    salience: float = Field(ge=0.0, le=1.0)
    urgency: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=-1.0, le=1.0)
    control: float = Field(ge=0.0, le=1.0)
    resources: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0, le=1.0)
    personality_relevance: float = Field(ge=0.0, le=1.0)

    @field_validator("deficits")
    @classmethod
    def _validate_deficits(cls, value: FrozenJsonDict) -> FrozenJsonDict:
        return _validate_numeric_map(
            value, lower=0.0, upper=1.0, label="energy deficits"
        )

    @property
    def activation(self) -> float:
        deficit = (
            sum(float(value) for value in self.deficits.values()) / len(self.deficits)
            if self.deficits
            else 0.0
        )
        raw = (
            0.18 * deficit
            + 0.15 * self.salience
            + 0.15 * self.urgency
            + 0.10 * abs(self.arousal)
            + 0.10 * self.control
            + 0.10 * self.resources
            - 0.10 * self.cost
            + 0.12 * self.personality_relevance
            + 0.05 * ((self.valence + 1.0) / 2.0)
        )
        return round(min(1.0, max(0.0, raw)), 12)


def _numeric_values(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise DomainInvariantError("personality values must be an object")
    return payload


def reduce_personality(
    personality: PersonalityControl, event: EventEnvelope
) -> PersonalityControl:
    if event.event_type != "personality.revised":
        return personality
    layer = event.payload.get("layer")
    if layer not in _MAX_DELTAS:
        raise DomainInvariantError("unknown personality layer")
    updates = _numeric_values(event.payload.get("values"))
    current = getattr(personality, layer)
    merged = dict(current.items())
    maximum = _MAX_DELTAS[layer]
    for key, raw_value in updates.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise DomainInvariantError("personality values must be numeric")
        value = float(raw_value)
        before = float(current.get(key, 0.0))
        if not math.isfinite(value) or abs(value - before) > maximum + 1e-12:
            raise DomainInvariantError(
                f"{layer} revision exceeds its bounded rate {maximum}"
            )
        merged[key] = value
    try:
        revised = personality.model_copy(update={layer: FrozenJsonDict(merged)})
        return revised.revalidated()
    except Exception as error:
        raise DomainInvariantError("invalid personality revision") from error


def energy_from_event(event: EventEnvelope) -> EnergyVector:
    try:
        return EnergyVector(
            action_id=event.payload["action_id"],
            deficits=event.payload.get("deficits", {}),
            salience=float(event.payload["salience"]),
            urgency=float(event.payload["urgency"]),
            valence=float(event.payload["valence"]),
            arousal=float(event.payload["arousal"]),
            control=float(event.payload["control"]),
            resources=float(event.payload["resources"]),
            cost=float(event.payload["cost"]),
            personality_relevance=float(event.payload["personality_relevance"]),
        )
    except Exception as error:
        raise DomainInvariantError("invalid action energy vector") from error


def upsert_energy(
    energies: tuple[EnergyVector, ...], energy: EnergyVector
) -> tuple[EnergyVector, ...]:
    retained = tuple(item for item in energies if item.action_id != energy.action_id)
    return (*retained, energy)


__all__ = [
    "EnergyVector",
    "PersonalityControl",
    "PersonalityLayer",
    "energy_from_event",
    "reduce_personality",
    "upsert_energy",
]
