"""Canonical identifiers shared by Alice-brain-Hermes components."""

from __future__ import annotations

from uuid import RFC_4122, UUID, uuid4


def new_id() -> str:
    """Return a new canonical RFC 4122 UUID version 4 string."""
    return str(uuid4())


def validate_id(value: str) -> str:
    """Return *value* if it is a canonical RFC 4122 UUID4 string."""
    if not isinstance(value, str):
        raise ValueError("identifier must be a string")

    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("identifier must be a canonical UUID4 string") from error

    if str(parsed) != value or parsed.version != 4 or parsed.variant != RFC_4122:
        raise ValueError("identifier must be a canonical UUID4 string")
    return value
