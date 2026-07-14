"""Strict immutable event envelopes and canonical JSON support."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import core_schema

from alice_brain_hermes.ids import new_id, validate_id

EVENT_SCHEMA_VERSION = 1
_OPTIONAL_PROVENANCE_FIELDS = (
    "adapter_id",
    "session_id",
    "turn_id",
    "action_id",
    "causation_id",
    "correlation_id",
)


def _freeze_json(value: Any, *, location: str = "payload") -> Any:
    if isinstance(value, FrozenJsonDict):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} keys must be strings")
            frozen[key] = _freeze_json(item, location=f"{location}.{key}")
        return FrozenJsonDict(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} floating-point values must be finite")
        return value
    raise TypeError(f"{location} contains a non-JSON value: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return ordinary JSON containers for an immutable JSON value."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


def _canonical_json_value(value: Any) -> str:
    frozen = _freeze_json(value)
    return json.dumps(
        thaw_json(frozen),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class FrozenJsonDict(Mapping[str, Any]):
    """A recursively immutable mapping whose contents are valid JSON values."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        source = {} if values is None else values
        data: dict[str, Any] = {}
        keys = tuple(source)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("JSON object keys must be strings")
        for key in sorted(keys):
            value = source[key]
            data[key] = _freeze_json(value, location=f"payload.{key}")
        self._data = MappingProxyType(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            try:
                return self.canonical_json() == _canonical_json_value(other)
            except (TypeError, ValueError):
                return False
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.canonical_json())

    def canonical_json(self) -> str:
        """Serialize with JSON type distinctions preserved recursively."""
        return _canonical_json_value(self)

    @classmethod
    def _validate(cls, value: Any) -> FrozenJsonDict:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("payload must be a JSON object")
        return cls(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                thaw_json,
                return_schema=core_schema.dict_schema(
                    core_schema.str_schema(), core_schema.any_schema()
                ),
            ),
        )


class EventEnvelope(BaseModel):
    """The sole strict, immutable input accepted by the persistent ledger."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    schema_version: Literal[EVENT_SCHEMA_VERSION] = EVENT_SCHEMA_VERSION
    event_id: str
    event_type: str = Field(
        min_length=1, max_length=160, pattern=r"^[a-z][a-z0-9_.-]*$"
    )
    brain_id: str
    sequence: int | None = Field(default=None, ge=1)
    wall_time: datetime
    monotonic_ns: int = Field(ge=0)
    actor_id: str
    adapter_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    action_id: str | None = None
    causation_id: str | None = None
    correlation_id: str | None = None
    payload: FrozenJsonDict

    @field_validator("event_id", "brain_id", "actor_id")
    @classmethod
    def _validate_required_id(cls, value: str) -> str:
        return validate_id(value)

    @field_validator(*_OPTIONAL_PROVENANCE_FIELDS)
    @classmethod
    def _validate_optional_provenance(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip() or len(value) > 512:
            raise ValueError("provenance identifiers must be non-blank and bounded")
        return value

    @field_validator("wall_time")
    @classmethod
    def _canonical_wall_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("wall_time must be timezone-aware")
        return value.astimezone(UTC)

    def canonical_data(self, *, exclude_sequence: bool = False) -> dict[str, Any]:
        """Return a stable JSON-compatible representation of this envelope."""
        data = self.model_dump(mode="json")
        if exclude_sequence:
            data.pop("sequence", None)
        return data

    def canonical_json(self, *, exclude_sequence: bool = False) -> str:
        """Serialize with stable ordering and no non-standard JSON values."""
        return json.dumps(
            self.canonical_data(exclude_sequence=exclude_sequence),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def body_fingerprint(self) -> str:
        """Hash the complete immutable body, excluding ledger allocation."""
        body = self.canonical_json(exclude_sequence=True).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    def envelope_fingerprint(self) -> str:
        """Hash the full stored envelope, including its allocated sequence."""
        envelope = self.canonical_json().encode("utf-8")
        return hashlib.sha256(envelope).hexdigest()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EventEnvelope):
            try:
                return self.canonical_json() == other.canonical_json()
            except (TypeError, ValueError):
                return False
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.canonical_json())

    def revalidated(self) -> EventEnvelope:
        """Revalidate even values created through Pydantic's unchecked model_copy."""
        return EventEnvelope.model_validate(self.model_dump(mode="python"))


def new_event(
    event_type: str,
    brain_id: str,
    actor_id: str,
    payload: Mapping[str, Any],
    *,
    event_id: str | None = None,
    sequence: int | None = None,
    wall_time: datetime | None = None,
    monotonic_ns: int | None = None,
    adapter_id: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    action_id: str | None = None,
    causation_id: str | None = None,
    correlation_id: str | None = None,
) -> EventEnvelope:
    """Create a canonical event without allocating a persistent sequence."""
    return EventEnvelope(
        schema_version=EVENT_SCHEMA_VERSION,
        event_id=new_id() if event_id is None else event_id,
        event_type=event_type,
        brain_id=brain_id,
        sequence=sequence,
        wall_time=datetime.now(UTC) if wall_time is None else wall_time,
        monotonic_ns=time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
        actor_id=actor_id,
        adapter_id=adapter_id,
        session_id=session_id,
        turn_id=turn_id,
        action_id=action_id,
        causation_id=causation_id,
        correlation_id=correlation_id,
        payload=payload,
    )


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "EventEnvelope",
    "FrozenJsonDict",
    "new_event",
    "thaw_json",
]
