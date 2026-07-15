"""Strict, bounded read models for white-box runtime diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    state_sequence: int = Field(ge=0)
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
        self_id_records = tuple(
            actor for actor in self.actors if actor.actor_id == self.self_actor_id
        )
        self_kind_records = tuple(
            actor for actor in self.actors if actor.kind.value == "self"
        )
        if (
            len(self_id_records) != 1
            or len(self_kind_records) != 1
            or self_id_records[0] != self_kind_records[0]
        ):
            raise ValueError("identity snapshot lacks one exact self actor")
        return self


class TracePageV1(_StrictDiagnosticModel):
    """Ordered trace page with explicit response-limit evidence.

    Limit flags identify every negotiated bound exceeded by the caller's
    untruncated requested page. ``blocked_event_sequence`` is always the first
    omitted event, so a client can retry without advancing past it.
    """

    schema_version: Literal[1] = 1
    brain_id: str
    after_sequence: int = Field(ge=0)
    requested_limit: int = Field(ge=1, le=TRACE_MAX_PAGE_SIZE)
    returned_count: int = Field(ge=0, le=TRACE_MAX_PAGE_SIZE)
    next_after_sequence: int = Field(ge=0)
    has_more: bool
    byte_limited: bool
    node_limited: bool = False
    depth_limited: bool = False
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
        response_limited = self.byte_limited or self.node_limited or self.depth_limited
        if response_limited:
            if not self.has_more or self.blocked_event_sequence is None:
                raise ValueError("response-limited trace page lacks blocked evidence")
            if self.blocked_event_sequence <= self.next_after_sequence:
                raise ValueError("blocked trace event does not follow the cursor")
            if self.blocked_event_sequence != self.next_after_sequence + 1:
                raise ValueError("blocked trace event is not the next event")
        elif self.blocked_event_sequence is not None:
            raise ValueError("count-limited trace page claims a blocked event")
        if (
            self.has_more
            and not response_limited
            and self.returned_count != self.requested_limit
        ):
            raise ValueError("count-limited trace page is not full")
        return self


def _tree_shape(value: object) -> tuple[int, int]:
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    maximum_depth = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        maximum_depth = max(maximum_depth, depth)
        if isinstance(item, Mapping):
            pending.extend((key, depth + 1) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend((child, depth + 1) for child in item)
    return nodes, maximum_depth


def _page_limit_violations(
    page: TracePageV1,
    *,
    max_result_bytes: int,
    max_result_nodes: int,
    max_result_depth: int,
) -> tuple[bool, bool, bool]:
    payload = page.model_dump(mode="json")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    nodes, depth = _tree_shape(payload)
    return (
        len(encoded) > max_result_bytes,
        nodes > max_result_nodes,
        depth > max_result_depth,
    )


def build_trace_page(
    events: list[EventEnvelope],
    *,
    brain_id: str,
    after_sequence: int,
    requested_limit: int,
    max_result_bytes: int,
    max_result_nodes: int,
    max_result_depth: int,
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
    if isinstance(max_result_nodes, bool) or not isinstance(max_result_nodes, int):
        raise TypeError("max_result_nodes must be an exact integer")
    if max_result_nodes < 1:
        raise ValueError("max_result_nodes must be positive")
    if isinstance(max_result_depth, bool) or not isinstance(max_result_depth, int):
        raise TypeError("max_result_depth must be an exact integer")
    if max_result_depth < 1:
        raise ValueError("max_result_depth must be positive")
    if len(events) > requested_limit + 1:
        raise ValueError("trace lookahead exceeds its bounded query")

    count_has_more = len(events) > requested_limit
    maximum_count = min(len(events), requested_limit)

    def page_for(
        selected_count: int,
        limit_violations: tuple[bool, bool, bool],
    ) -> TracePageV1:
        blocked_sequence = None
        response_limited = any(limit_violations)
        if response_limited:
            blocked = events[selected_count]
            if blocked.sequence is None:
                raise ValueError("blocked trace event has no sequence")
            blocked_sequence = blocked.sequence
        selected = events[:selected_count]
        return TracePageV1(
            brain_id=brain_id,
            after_sequence=after_sequence,
            requested_limit=requested_limit,
            returned_count=selected_count,
            next_after_sequence=(selected[-1].sequence if selected else after_sequence),
            has_more=count_has_more or response_limited,
            byte_limited=limit_violations[0],
            node_limited=limit_violations[1],
            depth_limited=limit_violations[2],
            blocked_event_sequence=blocked_sequence,
            events=tuple(selected),
        )

    unrestricted = page_for(maximum_count, (False, False, False))
    limit_violations = _page_limit_violations(
        unrestricted,
        max_result_bytes=max_result_bytes,
        max_result_nodes=max_result_nodes,
        max_result_depth=max_result_depth,
    )
    if not any(limit_violations):
        return unrestricted

    lower = 0
    upper = maximum_count - 1
    fitted: TracePageV1 | None = None
    while lower <= upper:
        candidate_count = (lower + upper) // 2
        candidate = page_for(candidate_count, limit_violations)
        candidate_violations = _page_limit_violations(
            candidate,
            max_result_bytes=max_result_bytes,
            max_result_nodes=max_result_nodes,
            max_result_depth=max_result_depth,
        )
        if not any(candidate_violations):
            fitted = candidate
            lower = candidate_count + 1
        else:
            upper = candidate_count - 1
    if fitted is None:
        raise ResponseSizeError("empty trace page exceeds negotiated response limits")
    return fitted


__all__ = [
    "TRACE_MAX_PAGE_SIZE",
    "IdentitySnapshotV1",
    "TracePageV1",
    "build_trace_page",
]
