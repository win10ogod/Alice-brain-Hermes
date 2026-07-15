"""Strict wire models for optional agent-selected identity naming."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alice_brain_hermes.ids import validate_id


class _StrictIdentityModel(BaseModel):
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


class IdentityChoiceV1(_StrictIdentityModel):
    """One exact self-designated name; the framework never supplies a fallback."""

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=160)
    reason: str = Field(min_length=1, max_length=512)

    @field_validator("name", "reason")
    @classmethod
    def _exact_nonblank_text(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("identity choice text must be exact and non-blank")
        return value


class IdentityNamingLeaseV1(_StrictIdentityModel):
    """A bounded daemon lease tied to one still-unnamed state sequence."""

    schema_version: Literal[1] = 1
    lease_id: str
    brain_id: str
    state_sequence: int = Field(ge=1)
    expires_at: datetime

    @field_validator("lease_id", "brain_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("expires_at")
    @classmethod
    def _aware_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("identity naming lease expiry must be timezone-aware")
        return value.astimezone(UTC)


__all__ = ["IdentityChoiceV1", "IdentityNamingLeaseV1"]
