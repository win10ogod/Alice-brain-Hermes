"""Four-layer world model with conservative observation grounding."""

from __future__ import annotations

from collections.abc import Collection
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict
from alice_brain_hermes.errors import DomainInvariantError


class WorldLayer(StrEnum):
    OBSERVED = "observed"
    BELIEVED = "believed"
    SIMULATED = "simulated"
    IDEAL = "ideal"


class WorldProposition(BaseModel):
    """A proposition whose epistemic layer is fixed by its event type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    proposition_id: str = Field(min_length=1, max_length=256)
    content: FrozenJsonDict
    layer: WorldLayer
    source_event_id: str
    source_actor_id: str
    action_id: str | None = Field(default=None, min_length=1, max_length=512)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("proposition_id")
    @classmethod
    def _bounded_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("proposition_id must be non-blank")
        return value


class WorldModel(BaseModel):
    """Observed, believed, simulated and ideal contents never alias."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    observed: tuple[WorldProposition, ...] = ()
    believed: tuple[WorldProposition, ...] = ()
    simulated: tuple[WorldProposition, ...] = ()
    ideal: tuple[WorldProposition, ...] = ()

    @field_validator("observed", "believed", "simulated", "ideal", mode="before")
    @classmethod
    def _json_arrays_to_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    def revalidated(self) -> WorldModel:
        return WorldModel.model_validate(self.model_dump(mode="python"))


_EVENT_LAYERS = {
    "observation.recorded": WorldLayer.OBSERVED,
    "belief.updated": WorldLayer.BELIEVED,
    "simulation.created": WorldLayer.SIMULATED,
    "ideal.updated": WorldLayer.IDEAL,
}


def _proposition(
    event: EventEnvelope,
    item: Any,
    layer: WorldLayer,
    *,
    action_id: str | None = None,
) -> WorldProposition:
    if not isinstance(item, dict) and not hasattr(item, "get"):
        raise DomainInvariantError("world proposition must be an object")
    try:
        return WorldProposition(
            proposition_id=item["proposition_id"],
            content=item["content"],
            layer=layer,
            source_event_id=event.event_id,
            source_actor_id=event.actor_id,
            action_id=action_id,
            confidence=item.get("confidence", 1.0),
        )
    except Exception as error:
        raise DomainInvariantError("invalid world proposition") from error


def _upsert(
    items: tuple[WorldProposition, ...], proposition: WorldProposition
) -> tuple[WorldProposition, ...]:
    retained = tuple(
        item for item in items if item.proposition_id != proposition.proposition_id
    )
    return (*retained, proposition)


def reduce_world(
    world: WorldModel,
    event: EventEnvelope,
    *,
    trusted_provenance: bool,
    grounded_receipt: Collection[str],
) -> WorldModel:
    """Apply only event-type-fixed layer transitions."""
    layer = _EVENT_LAYERS.get(event.event_type)
    if layer is not None:
        if layer is WorldLayer.OBSERVED and not trusted_provenance:
            return world
        proposition = _proposition(event, event.payload, layer)
        field = layer.value
        return world.model_copy(
            update={field: _upsert(getattr(world, field), proposition)}
        ).revalidated()

    if event.event_type == "action.receipt" and grounded_receipt:
        observations = event.payload.get("observations", ())
        if not isinstance(observations, (list, tuple)):
            raise DomainInvariantError("receipt observations must be an array")
        observed = world.observed
        action_id = event.payload.get("action_id")
        for item in observations:
            proposition_id = (
                item.get("proposition_id") if hasattr(item, "get") else None
            )
            if proposition_id not in grounded_receipt:
                continue
            proposition = _proposition(
                event,
                item,
                WorldLayer.OBSERVED,
                action_id=action_id if isinstance(action_id, str) else None,
            )
            observed = _upsert(observed, proposition)
        return world.model_copy(update={"observed": observed}).revalidated()

    return world


__all__ = [
    "WorldLayer",
    "WorldModel",
    "WorldProposition",
    "reduce_world",
]
