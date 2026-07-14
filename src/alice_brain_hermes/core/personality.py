"""Three-layer personality control and action-indexed energy vectors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict
from alice_brain_hermes.errors import DomainInvariantError

PersonalityLayer = Literal["traits", "adaptations", "narrative_ideal"]
_RATE_POLICIES: dict[str, tuple[float, float]] = {
    # (capacity, tokens refilled per logical second)
    "traits": (0.05, 0.05),
    "adaptations": (0.20, 0.20),
    "narrative_ideal": (0.10, 0.10),
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


class LayerRateState(BaseModel):
    """Persisted logical-time token bucket for one personality layer."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    logical_clock: float = Field(default=0.0, ge=0.0)
    available: float = Field(ge=0.0)
    capacity: float = Field(gt=0.0)
    refill_rate: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _available_does_not_exceed_capacity(self) -> LayerRateState:
        if self.available > self.capacity:
            raise ValueError("personality rate availability exceeds capacity")
        return self

    def advanced_to(self, logical_clock: float) -> LayerRateState:
        """Refill deterministically up to a finite non-decreasing clock."""
        if (
            isinstance(logical_clock, bool)
            or not isinstance(logical_clock, (int, float))
            or not math.isfinite(float(logical_clock))
        ):
            raise DomainInvariantError("personality rate logical clock must be finite")
        target = float(logical_clock)
        if target < self.logical_clock:
            raise DomainInvariantError(
                "personality rate logical clock cannot move backwards"
            )
        elapsed = Decimal(str(target)) - Decimal(str(self.logical_clock))
        available = min(
            Decimal(str(self.capacity)),
            Decimal(str(self.available))
            + elapsed * Decimal(str(self.refill_rate)),
        )
        return self.model_copy(
            update={
                "logical_clock": target,
                "available": float(available),
            }
        ).revalidated()

    def consumed(self, amount: Decimal) -> LayerRateState:
        """Consume a finite cumulative-change amount from this bucket."""
        if not amount.is_finite() or amount < 0:
            raise DomainInvariantError("personality rate cost must be finite")
        available = Decimal(str(self.available))
        if amount > available:
            raise DomainInvariantError(
                "personality revision exceeds bounded rate cumulative layer budget"
            )
        return self.model_copy(
            update={"available": float(available - amount)}
        ).revalidated()

    def revalidated(self) -> LayerRateState:
        return LayerRateState.model_validate(self.model_dump(mode="python"))


def _new_layer_rate_state(layer: PersonalityLayer) -> LayerRateState:
    capacity, refill_rate = _RATE_POLICIES[layer]
    return LayerRateState(
        available=capacity,
        capacity=capacity,
        refill_rate=refill_rate,
    )


class PersonalityRateState(BaseModel):
    """Typed persisted buckets for all three independent PC layers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    traits: LayerRateState = Field(
        default_factory=lambda: _new_layer_rate_state("traits")
    )
    adaptations: LayerRateState = Field(
        default_factory=lambda: _new_layer_rate_state("adaptations")
    )
    narrative_ideal: LayerRateState = Field(
        default_factory=lambda: _new_layer_rate_state("narrative_ideal")
    )

    @model_validator(mode="after")
    def _policies_are_fixed(self) -> PersonalityRateState:
        for layer, (capacity, refill_rate) in _RATE_POLICIES.items():
            bucket = getattr(self, layer)
            if (
                bucket.capacity != capacity
                or bucket.refill_rate != refill_rate
            ):
                raise ValueError(
                    f"{layer} personality rate policy does not match runtime policy"
                )
        return self

    def advanced_to(self, logical_clock: float) -> PersonalityRateState:
        return PersonalityRateState(
            traits=self.traits.advanced_to(logical_clock),
            adaptations=self.adaptations.advanced_to(logical_clock),
            narrative_ideal=self.narrative_ideal.advanced_to(logical_clock),
        )


class PersonalityControl(BaseModel):
    """PC: slow traits, contextual adaptations, and narrative/ideal self."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    traits: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    adaptations: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    narrative_ideal: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    rate_state: PersonalityRateState = Field(default_factory=PersonalityRateState)

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


def advance_personality_clock(
    personality: PersonalityControl, logical_clock: float
) -> PersonalityControl:
    """Advance and persist every layer's token bucket on a clock event."""
    try:
        return personality.model_copy(
            update={"rate_state": personality.rate_state.advanced_to(logical_clock)}
        ).revalidated()
    except DomainInvariantError:
        raise
    except Exception as error:
        raise DomainInvariantError("invalid personality rate state") from error


def reduce_personality(
    personality: PersonalityControl,
    event: EventEnvelope,
    *,
    logical_clock: float,
) -> PersonalityControl:
    if event.event_type != "personality.revised":
        return personality
    layer = event.payload.get("layer")
    if layer not in _RATE_POLICIES:
        raise DomainInvariantError("unknown personality layer")
    updates = _numeric_values(event.payload.get("values"))
    current = getattr(personality, layer)
    merged = dict(current.items())
    try:
        bucket = getattr(personality.rate_state, layer).advanced_to(logical_clock)
    except DomainInvariantError:
        raise
    except Exception as error:
        raise DomainInvariantError("invalid personality rate state") from error
    cumulative_change = Decimal(0)
    for key, raw_value in updates.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise DomainInvariantError("personality values must be numeric")
        value = float(raw_value)
        before = Decimal(str(current.get(key, 0.0)))
        if not math.isfinite(value):
            raise DomainInvariantError("personality values must be finite")
        cumulative_change += abs(Decimal(str(raw_value)) - before)
        merged[key] = value
    try:
        revised_bucket = bucket.consumed(cumulative_change)
        revised_rates = personality.rate_state.model_copy(
            update={layer: revised_bucket}
        )
        revised = personality.model_copy(
            update={layer: FrozenJsonDict(merged), "rate_state": revised_rates}
        )
        return revised.revalidated()
    except DomainInvariantError:
        raise
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
    "LayerRateState",
    "PersonalityControl",
    "PersonalityLayer",
    "PersonalityRateState",
    "advance_personality_clock",
    "energy_from_event",
    "reduce_personality",
    "upsert_energy",
]
