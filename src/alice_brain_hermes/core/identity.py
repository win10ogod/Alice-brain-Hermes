"""Typed self/other identity boundaries and provenance authorization."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alice_brain_hermes.core.events import EventEnvelope, FrozenJsonDict
from alice_brain_hermes.core.limits import (
    MAX_IDENTITY_ACTORS,
    MAX_PROVENANCE_AUTHORIZATIONS,
)
from alice_brain_hermes.errors import DomainCapacityError, DomainInvariantError
from alice_brain_hermes.ids import validate_id


class ActorKind(StrEnum):
    SELF = "self"
    HUMAN = "human"
    EXTERNAL_AGENT = "external_agent"
    TOOL_ADAPTER = "tool_adapter"
    SYSTEM = "system"


class ActorRecord(BaseModel):
    """One attributed actor; external actors never merge with the self actor."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    actor_id: str
    kind: ActorKind
    display_name: str | None = Field(default=None, max_length=160)
    parent_actor_id: str | None = None
    attributes: FrozenJsonDict = Field(default_factory=FrozenJsonDict)

    @field_validator("actor_id", "parent_actor_id")
    @classmethod
    def _validate_actor_ids(cls, value: str | None) -> str | None:
        return None if value is None else validate_id(value)

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("display_name must be non-blank when present")
        return value


class ProvenanceAuthorization(BaseModel):
    """Reducer-visible authority for one exact actor/adapter pair."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    actor_id: str
    adapter_id: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("actor_id")
    @classmethod
    def _validate_actor_id(cls, value: str) -> str:
        return validate_id(value)


class IdentityState(BaseModel):
    """Finite identity state with an immutable self/world actor boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True, validate_default=True
    )

    self_actor_id: str
    name: str | None = Field(default=None, max_length=160)
    actors: tuple[ActorRecord, ...]
    authorizations: tuple[ProvenanceAuthorization, ...] = ()

    @field_validator("self_actor_id")
    @classmethod
    def _validate_self_actor_id(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("identity name must be non-blank when present")
        return value

    @field_validator("actors", "authorizations", mode="before")
    @classmethod
    def _json_arrays_to_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @classmethod
    def genesis(cls, brain_id: str) -> IdentityState:
        validate_id(brain_id)
        return cls(
            self_actor_id=brain_id,
            actors=(ActorRecord(actor_id=brain_id, kind=ActorKind.SELF),),
        )

    def actor(self, actor_id: str) -> ActorRecord:
        validate_id(actor_id)
        for actor in self.actors:
            if actor.actor_id == actor_id:
                return actor
        raise KeyError(actor_id)

    def is_authorized(self, actor_id: str, adapter_id: str | None) -> bool:
        """Trust self, or an exact authorization already present in state."""
        if actor_id == self.self_actor_id:
            return True
        return any(
            item.actor_id == actor_id and item.adapter_id == adapter_id
            for item in self.authorizations
        )

    def revalidated(self) -> IdentityState:
        return IdentityState.model_validate(self.model_dump(mode="python"))


def _replace_actor(
    actors: tuple[ActorRecord, ...], actor: ActorRecord
) -> tuple[ActorRecord, ...]:
    retained = tuple(item for item in actors if item.actor_id != actor.actor_id)
    return (*retained, actor)


def reduce_identity(identity: IdentityState, event: EventEnvelope) -> IdentityState:
    """Reduce identity events without consulting mutable external configuration."""
    if event.event_type == "identity.named":
        name = event.payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise DomainInvariantError("identity name must be a non-blank string")
        return identity.model_copy(update={"name": name}).revalidated()

    if event.event_type == "identity.actor_registered":
        try:
            actor = ActorRecord(
                actor_id=event.payload["actor_id"],
                kind=ActorKind(event.payload["kind"]),
                display_name=event.payload.get("display_name"),
                parent_actor_id=event.payload.get("parent_actor_id"),
                attributes=event.payload.get("attributes", {}),
            )
        except Exception as error:
            raise DomainInvariantError("invalid actor registration") from error
        if (
            actor.actor_id == identity.self_actor_id
            and actor.kind is not ActorKind.SELF
        ):
            raise DomainInvariantError("the self actor kind cannot be replaced")
        if actor.parent_actor_id == actor.actor_id:
            raise DomainInvariantError("an actor cannot be its own parent")
        if (
            all(item.actor_id != actor.actor_id for item in identity.actors)
            and len(identity.actors) >= MAX_IDENTITY_ACTORS
        ):
            raise DomainCapacityError(
                "identity actor capacity is full; registration was not applied"
            )
        return identity.model_copy(
            update={"actors": _replace_actor(identity.actors, actor)}
        ).revalidated()

    if event.event_type == "identity.provenance_authorized":
        if event.actor_id != identity.self_actor_id:
            raise DomainInvariantError(
                "only self authority can authorize trusted provenance"
            )
        try:
            authorization = ProvenanceAuthorization(
                actor_id=event.payload["actor_id"],
                adapter_id=event.payload.get("adapter_id"),
            )
        except Exception as error:
            raise DomainInvariantError("invalid provenance authorization") from error
        if authorization in identity.authorizations:
            return identity
        if len(identity.authorizations) >= MAX_PROVENANCE_AUTHORIZATIONS:
            raise DomainCapacityError(
                "provenance authorization capacity is full; authorization "
                "was not applied"
            )
        return identity.model_copy(
            update={"authorizations": (*identity.authorizations, authorization)}
        ).revalidated()

    return identity


__all__ = [
    "ActorKind",
    "ActorRecord",
    "IdentityState",
    "ProvenanceAuthorization",
    "reduce_identity",
]
