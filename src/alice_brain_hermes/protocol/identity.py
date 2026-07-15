"""Strict wire models for optional agent-selected identity naming."""

from __future__ import annotations

import json
import unicodedata
from datetime import UTC, datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alice_brain_hermes.ids import validate_id

IDENTITY_NAME_MAX_CODEPOINTS = 160
IDENTITY_NAME_MAX_UTF8_BYTES = 512
IDENTITY_REASON_MAX_CODEPOINTS = 512
IDENTITY_REASON_MAX_UTF8_BYTES = 2_048


def _is_noncharacter(value: str) -> bool:
    codepoint = ord(value)
    return 0xFDD0 <= codepoint <= 0xFDEF or (codepoint & 0xFFFF) in {
        0xFFFE,
        0xFFFF,
    }


def _validate_choice_text(
    value: str,
    *,
    field: str,
    max_utf8_bytes: int,
    require_nfkc: bool,
) -> str:
    if not value.strip() or value != value.strip():
        raise ValueError(f"identity {field} must be exact and non-blank")
    if require_nfkc and unicodedata.normalize("NFKC", value) != value:
        raise ValueError("identity name must already be in NFKC form")
    if any(
        unicodedata.category(character) in {"Cc", "Cs"} or _is_noncharacter(character)
        for character in value
    ):
        raise ValueError(f"identity {field} contains a forbidden codepoint")
    if len(value.encode("utf-8", errors="strict")) > max_utf8_bytes:
        raise ValueError(f"identity {field} exceeds its UTF-8 byte bound")
    return value


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
    name: str = Field(min_length=1, max_length=IDENTITY_NAME_MAX_CODEPOINTS)
    reason: str = Field(min_length=1, max_length=IDENTITY_REASON_MAX_CODEPOINTS)

    @field_validator("name")
    @classmethod
    def _exact_name(cls, value: str) -> str:
        return _validate_choice_text(
            value,
            field="name",
            max_utf8_bytes=IDENTITY_NAME_MAX_UTF8_BYTES,
            require_nfkc=True,
        )

    @field_validator("reason")
    @classmethod
    def _exact_reason(cls, value: str) -> str:
        return _validate_choice_text(
            value,
            field="reason",
            max_utf8_bytes=IDENTITY_REASON_MAX_UTF8_BYTES,
            require_nfkc=False,
        )


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


__all__ = [
    "IDENTITY_NAME_MAX_CODEPOINTS",
    "IDENTITY_NAME_MAX_UTF8_BYTES",
    "IDENTITY_REASON_MAX_CODEPOINTS",
    "IDENTITY_REASON_MAX_UTF8_BYTES",
    "IdentityChoiceV1",
    "IdentityNamingLeaseV1",
]
