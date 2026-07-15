"""Strict, bounded read models for white-box runtime diagnostics."""

from __future__ import annotations

import json
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alice_brain_hermes.core.events import EventEnvelope
from alice_brain_hermes.core.identity import ActorRecord, ProvenanceAuthorization
from alice_brain_hermes.errors import ResponseSizeError
from alice_brain_hermes.ids import validate_id

TRACE_MAX_PAGE_SIZE = 1_000


class _StrictDiagnosticModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        allow_inf_nan=False,
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


class IdentitySnapshotV1(_StrictDiagnosticModel):
    """One replay-derived identity; no name or actor is inferred by the RPC."""

    schema_version: Literal[1] = 1
    brain_id: str
    self_actor_id: str
    name: str | None = Field(default=None, max_length=160)
    state_sequence: int = Field(ge=1)
    actors: tuple[ActorRecord, ...]
    authorizations: tuple[ProvenanceAuthorization, ...]

    @field_validator("brain_id", "self_actor_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_id(value)

    @field_validator("actors", "authorizations", mode="before")
    @classmethod
    def _wire_arrays_to_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _self_boundary(self) -> IdentitySnapshotV1:
        if self.brain_id != self.self_actor_id:
            raise ValueError("identity snapshot changed the self boundary")
        self_actors = tuple(
            actor for actor in self.actors if actor.actor_id == self.self_actor_id
        )
        if len(self_actors) != 1 or self_actors[0].kind.value != "self":
            raise ValueError("identity snapshot lacks one exact self actor")
        return self


class TracePageV1(_StrictDiagnosticModel):
    """Ordered trace page with explicit count and byte truncation evidence."""

    schema_version: Literal[1] = 1
    brain_id: str
    after_sequence: int = Field(ge=0)
    requested_limit: int = Field(ge=1, le=TRACE_MAX_PAGE_SIZE)
    returned_count: int = Field(ge=0, le=TRACE_MAX_PAGE_SIZE)
    next_after_sequence: int = Field(ge=0)
    has_more: bool
    byte_limited: bool
    blocked_event_sequence: int | None = Field(default=None, ge=1)
    events: tuple[EventEnvelope, ...]

    @field_validator("brain_id")
    @classmethod
    def _brain_id(cls, value: str) -> str:
        return validate_id(value)

    @model_validator(mode="after")
    def _cursor_truth(self) -> TracePageV1:
        if self.returned_count != len(self.events):
            raise ValueError("trace returned_count does not match events")
        if len(self.events) > self.requested_limit:
            raise ValueError("trace page exceeds requested_limit")
        sequences = tuple(event.sequence for event in self.events)
        if any(sequence is None for sequence in sequences):
            raise ValueError("trace pages require sequenced events")
        exact_sequences = tuple(int(sequence) for sequence in sequences)
        expected_sequences = tuple(
            range(
                self.after_sequence + 1,
                self.after_sequence + len(exact_sequences) + 1,
            )
        )
        if exact_sequences != expected_sequences:
            raise ValueError("trace events are not contiguous after the cursor")
        if any(event.brain_id != self.brain_id for event in self.events):
            raise ValueError("trace page mixes brain identities")
        if any(sequence <= self.after_sequence for sequence in exact_sequences):
            raise ValueError("trace page contains an event before its cursor")
        expected_cursor = (
            exact_sequences[-1] if exact_sequences else self.after_sequence
        )
        if self.next_after_sequence != expected_cursor:
            raise ValueError("trace next cursor does not match returned events")
        if self.byte_limited:
            if not self.has_more or self.blocked_event_sequence is None:
                raise ValueError("byte-limited trace page lacks blocked evidence")
            if self.blocked_event_sequence <= self.next_after_sequence:
                raise ValueError("blocked trace event does not follow the cursor")
            if self.blocked_event_sequence != self.next_after_sequence + 1:
                raise ValueError("blocked trace event is not the next event")
        elif self.blocked_event_sequence is not None:
            raise ValueError("count-limited trace page claims a blocked event")
        if not self.has_more and self.byte_limited:
            raise ValueError("byte-limited trace page must report more data")
        return self


def build_trace_page(
    events: list[EventEnvelope],
    *,
    brain_id: str,
    after_sequence: int,
    requested_limit: int,
    max_result_bytes: int,
) -> TracePageV1:
    """Fit an ordered lookahead page without skipping an oversized event."""

    validate_id(brain_id)
    if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
        raise TypeError("after_sequence must be an exact integer")
    if after_sequence < 0:
        raise ValueError("after_sequence cannot be negative")
    if isinstance(requested_limit, bool) or not isinstance(requested_limit, int):
        raise TypeError("requested_limit must be an exact integer")
    if not 1 <= requested_limit <= TRACE_MAX_PAGE_SIZE:
        raise ValueError("requested_limit is outside the trace page bound")
    if isinstance(max_result_bytes, bool) or not isinstance(max_result_bytes, int):
        raise TypeError("max_result_bytes must be an exact integer")
    if max_result_bytes < 1:
        raise ValueError("max_result_bytes must be positive")
    if len(events) > requested_limit + 1:
        raise ValueError("trace lookahead exceeds its bounded query")

    selected = list(events[:requested_limit])
    count_has_more = len(events) > requested_limit
    byte_limited = False
    while True:
        blocked_sequence = None
        if byte_limited:
            blocked = events[len(selected)]
            if blocked.sequence is None:
                raise ValueError("blocked trace event has no sequence")
            blocked_sequence = blocked.sequence
        page = TracePageV1(
            brain_id=brain_id,
            after_sequence=after_sequence,
            requested_limit=requested_limit,
            returned_count=len(selected),
            next_after_sequence=(selected[-1].sequence if selected else after_sequence),
            has_more=count_has_more or byte_limited,
            byte_limited=byte_limited,
            blocked_event_sequence=blocked_sequence,
            events=tuple(selected),
        )
        if len(page.canonical_json().encode("utf-8")) <= max_result_bytes:
            return page
        if not selected:
            raise ResponseSizeError(
                "empty trace page exceeds the negotiated result byte limit"
            )
        selected.pop()
        byte_limited = True


__all__ = [
    "TRACE_MAX_PAGE_SIZE",
    "IdentitySnapshotV1",
    "TracePageV1",
    "build_trace_page",
]
