"""Frozen replay-derived foundation state."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alice_brain_hermes.core.events import FrozenJsonDict
from alice_brain_hermes.ids import validate_id

STATE_SCHEMA_VERSION = 1


class BrainState(BaseModel):
    """Deterministic state produced only by event reduction."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    schema_version: Literal[STATE_SCHEMA_VERSION] = STATE_SCHEMA_VERSION
    brain_id: str
    name: str | None = None
    capabilities: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    logical_clock: float = Field(default=0.0, ge=0.0)
    trace_complete: bool = True
    raw_lifecycle_counts: FrozenJsonDict = Field(default_factory=FrozenJsonDict)
    last_sequence: int = Field(default=0, ge=0)

    @field_validator("brain_id")
    @classmethod
    def _validate_brain_id(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("name must be non-blank when present")
        return value

    @field_validator("raw_lifecycle_counts")
    @classmethod
    def _validate_raw_counts(cls, value: FrozenJsonDict) -> FrozenJsonDict:
        for event_type, count in value.items():
            if not event_type or isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    "raw lifecycle counts must map event types to integers"
                )
            if count < 0:
                raise ValueError("raw lifecycle counts cannot be negative")
        return value

    @classmethod
    def genesis(cls, brain_id: str) -> BrainState:
        """Create the unnamed, capability-neutral foundation for one brain."""
        return cls(brain_id=brain_id)

    def canonical_json(self) -> str:
        """Return a stable snapshot representation."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def revalidated(self) -> BrainState:
        """Revalidate values created through Pydantic's unchecked model_copy."""
        return BrainState.model_validate(self.model_dump(mode="python"))


def state_from_data(values: dict[str, Any]) -> BrainState:
    """Build a validated state from reducer-owned ordinary values."""
    return BrainState.model_validate(values)


__all__ = ["STATE_SCHEMA_VERSION", "BrainState", "state_from_data"]
