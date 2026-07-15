"""Strict wire models for optional agent-selected identity naming."""

from __future__ import annotations

import json
import unicodedata
from datetime import UTC, datetime, timedelta
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alice_brain_hermes.ids import validate_id

IDENTITY_NAME_MAX_CODEPOINTS = 160
IDENTITY_NAME_MAX_UTF8_BYTES = 512
IDENTITY_REASON_MAX_CODEPOINTS = 512
IDENTITY_REASON_MAX_UTF8_BYTES = 2_048
IDENTITY_NAMING_LEASE_SECONDS = 120
IDENTITY_NORMALIZATION_UNICODE_VERSION = "14.0.0"

# Python 3.11 exposes Unicode 14.0 while Python 3.12/3.13 expose newer UCDs.
# NFKC and casefold mappings are stable for codepoints assigned by Unicode 14.
# These are the codepoints first assigned in Unicode 15.0/15.1. Keeping one
# explicit minimum-assigned boundary preserves common multilingual text and
# emoji already present in Unicode 14 without making persisted keys depend on
# the interpreter used to open the database.
_POST_UNICODE_14_RANGES = (
    (0x0CF3, 0x0CF3),
    (0x0ECE, 0x0ECE),
    (0x2FFC, 0x2FFF),
    (0x31EF, 0x31EF),
    (0x10EFD, 0x10EFF),
    (0x1123F, 0x11241),
    (0x11B00, 0x11B09),
    (0x11F00, 0x11F10),
    (0x11F12, 0x11F3A),
    (0x11F3E, 0x11F59),
    (0x1342F, 0x1342F),
    (0x13439, 0x13455),
    (0x1B132, 0x1B132),
    (0x1B155, 0x1B155),
    (0x1D2C0, 0x1D2D3),
    (0x1DF25, 0x1DF2A),
    (0x1E030, 0x1E06D),
    (0x1E08F, 0x1E08F),
    (0x1E4D0, 0x1E4F9),
    (0x1F6DC, 0x1F6DC),
    (0x1F774, 0x1F776),
    (0x1F77B, 0x1F77F),
    (0x1F7D9, 0x1F7D9),
    (0x1FA75, 0x1FA77),
    (0x1FA87, 0x1FA88),
    (0x1FAAD, 0x1FAAF),
    (0x1FABB, 0x1FABD),
    (0x1FABF, 0x1FABF),
    (0x1FACE, 0x1FACF),
    (0x1FADA, 0x1FADB),
    (0x1FAE8, 0x1FAE8),
    (0x1FAF7, 0x1FAF8),
    (0x2B739, 0x2B739),
    (0x2EBF0, 0x2EE5D),
    (0x31350, 0x323AF),
)


class IdentityNameNormalizationError(ValueError):
    """A name cannot have one stable key on every supported interpreter."""


def identity_name_normalization_key(value: str) -> str:
    """Return the version-stable Unicode-14 NFKC+casefold identity key."""

    if not isinstance(value, str):
        raise TypeError("identity name must be a string")
    for character in value:
        codepoint = ord(character)
        if unicodedata.category(character) == "Cn" or any(
            lower <= codepoint <= upper for lower, upper in _POST_UNICODE_14_RANGES
        ):
            raise IdentityNameNormalizationError(
                "identity name contains a codepoint outside the stable "
                "Unicode 14.0 normalization boundary"
            )
    return unicodedata.normalize("NFKC", value).casefold()


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
        identity_name_normalization_key(value)
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


class IdentityNamingLeaseStatusV1(_StrictIdentityModel):
    """Durable audit state for one daemon-owned naming lease."""

    schema_version: Literal[1] = 1
    lease_id: str
    brain_id: str
    state_sequence: int = Field(ge=1)
    status: Literal["pending", "completed", "failed", "superseded"]
    requested_at: datetime
    expires_at: datetime
    choice: IdentityChoiceV1 | None = None
    failure_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    terminal_event_id: str | None = None
    terminal_at: datetime | None = None

    @field_validator("lease_id", "brain_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("terminal_event_id")
    @classmethod
    def _optional_event_id(cls, value: str | None) -> str | None:
        return None if value is None else validate_id(value)

    @field_validator("requested_at", "expires_at", "terminal_at")
    @classmethod
    def _aware_times(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("identity naming status times must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _terminal_shape(self) -> IdentityNamingLeaseStatusV1:
        if self.expires_at != self.requested_at + timedelta(
            seconds=IDENTITY_NAMING_LEASE_SECONDS
        ):
            raise ValueError("identity naming lease duration is invalid")
        if self.status == "pending":
            if any(
                value is not None
                for value in (
                    self.choice,
                    self.failure_code,
                    self.terminal_event_id,
                    self.terminal_at,
                )
            ):
                raise ValueError("pending identity naming lease is already terminal")
        elif self.status == "completed":
            if (
                self.choice is None
                or self.failure_code is not None
                or self.terminal_event_id is None
                or self.terminal_at is None
            ):
                raise ValueError("completed identity naming lease is incomplete")
        elif self.status == "failed":
            if (
                self.failure_code is None
                or self.terminal_event_id is None
                or self.terminal_at is None
            ):
                raise ValueError("failed identity naming lease lacks failure evidence")
        elif (
            self.choice is not None
            or self.failure_code not in {"expired", "identity_already_named"}
            or self.terminal_at is None
            or (self.failure_code == "expired" and self.terminal_event_id is not None)
            or (
                self.failure_code == "identity_already_named"
                and self.terminal_event_id is None
            )
        ):
            raise ValueError("superseded identity naming lease is inconsistent")
        if self.terminal_at is not None and self.terminal_at < self.requested_at:
            raise ValueError("identity naming terminal time predates its request")
        if (
            self.status in {"completed", "failed"}
            and self.terminal_at is not None
            and self.terminal_at >= self.expires_at
        ):
            raise ValueError("identity naming terminal time exceeds its live lease")
        if self.status == "superseded" and self.terminal_at is not None:
            if self.failure_code == "expired" and self.terminal_at < self.expires_at:
                raise ValueError("expired identity naming lease was still live")
            if (
                self.failure_code == "identity_already_named"
                and self.terminal_at >= self.expires_at
            ):
                raise ValueError("named identity lease outlived its expiry")
        return self


__all__ = [
    "IDENTITY_NAME_MAX_CODEPOINTS",
    "IDENTITY_NAME_MAX_UTF8_BYTES",
    "IDENTITY_NAMING_LEASE_SECONDS",
    "IDENTITY_NORMALIZATION_UNICODE_VERSION",
    "IDENTITY_REASON_MAX_CODEPOINTS",
    "IDENTITY_REASON_MAX_UTF8_BYTES",
    "IdentityChoiceV1",
    "IdentityNameNormalizationError",
    "IdentityNamingLeaseStatusV1",
    "IdentityNamingLeaseV1",
    "identity_name_normalization_key",
]
