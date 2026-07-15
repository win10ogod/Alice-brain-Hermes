"""Atomic, bounded projections shared by synchronous hooks and the worker."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from alice_brain_hermes.core.events import thaw_json
from alice_brain_hermes.protocol.models import ConsciousnessFrameV3

MAX_EPHEMERAL_CONTEXT_BYTES = 16_384


def _truncate_utf8(value: str, maximum: int) -> str:
    encoded = value.encode("utf-8", errors="strict")
    if len(encoded) <= maximum:
        return value
    return encoded[:maximum].decode("utf-8", errors="ignore")


@dataclass(frozen=True, slots=True)
class BridgeHealth:
    """One immutable health sample; reads are a single reference load."""

    connection: str = "disconnected"
    trace_complete: bool = True
    dropped_events: int = 0
    pending_gap_ranges: int = 0
    last_capture_seq: int = 0
    through_capture_seq: int = 0
    worker_started: bool = False
    last_error: str | None = None
    abandoned_streams: int = 0
    abandoned_local_records: int = 0
    ambiguous_records: int = 0
    late_after_close: int = 0
    last_abandonment: Mapping[str, object] | None = None
    capabilities: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType(
            {
                "chunk_capture": "unobserved",
                "reasoning_capture": "unobserved",
            }
        )
    )


class AtomicProjectionCache:
    """Lock-free reader surface with a single worker-side publisher."""

    def __init__(
        self,
        *,
        context_sink: Callable[[str | None], None] | None = None,
    ) -> None:
        if context_sink is not None and not callable(context_sink):
            raise TypeError("context_sink must be callable or None")
        self._frame: ConsciousnessFrameV3 | None = None
        self._context: str | None = None
        self._context_sink = context_sink

    @property
    def frame(self) -> ConsciousnessFrameV3 | None:
        return self._frame

    def read_context(self) -> str | None:
        return self._context

    def publish_context(self, value: str | None) -> None:
        if value is not None and type(value) is not str:
            raise TypeError("ephemeral context must be an exact str or None")
        self._context = (
            None if not value else _truncate_utf8(value, MAX_EPHEMERAL_CONTEXT_BYTES)
        )
        self._publish_context_sink()

    def publish_frame(self, frame: ConsciousnessFrameV3) -> None:
        if type(frame) is not ConsciousnessFrameV3:
            raise TypeError("frame must be an exact ConsciousnessFrameV3")
        context = self._compact_context(frame)
        # Publish the derived string first and frame second.  Hook readers only
        # consume the string; diagnostic readers that see the new frame also see
        # its corresponding context under CPython's object reference semantics.
        self._context = context
        self._frame = frame
        self._publish_context_sink()

    def clear(self) -> None:
        """Clear a stream-scoped frame before binding a replacement stream."""

        self._context = None
        self._frame = None
        self._publish_context_sink()

    def _publish_context_sink(self) -> None:
        sink = self._context_sink
        if sink is None:
            return
        try:
            sink(self._context)
        except Exception:
            # Projection publication must not make transport progress depend on
            # an optional integration cache.
            return

    @staticmethod
    def _compact_context(frame: ConsciousnessFrameV3) -> str | None:
        body = {
            "alice_brain": {
                "schema_version": frame.schema_version,
                "brain_id": frame.brain_id,
                "state_sequence": frame.state_sequence,
                "through_capture_seq": frame.through_capture_seq,
                "trace_complete": frame.trace_complete,
                "runtime_health": frame.runtime_health,
                "pc": thaw_json(frame.pc),
                "energy": thaw_json(frame.energy),
                "st": thaw_json(frame.st),
                "rd": thaw_json(frame.rd),
                "a": thaw_json(frame.a),
                "semantic_context": thaw_json(frame.semantic_context),
                "aggregate_semantic_complete": frame.aggregate_semantic_complete,
                "semantic_evidence": frame.semantic_evidence.model_dump(mode="json"),
                "unresolved_evidence": frame.unresolved_evidence,
                "capabilities": thaw_json(frame.capabilities),
                "omission_counts": thaw_json(frame.omission_counts),
            }
        }
        rendered = json.dumps(
            body,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(rendered.encode("utf-8")) <= MAX_EPHEMERAL_CONTEXT_BYTES:
            return rendered or None
        fallback = {
            "alice_brain": {
                "schema_version": frame.schema_version,
                "brain_id": frame.brain_id,
                "state_sequence": frame.state_sequence,
                "through_capture_seq": frame.through_capture_seq,
                "trace_complete": frame.trace_complete,
                "runtime_health": frame.runtime_health,
                "aggregate_semantic_complete": frame.aggregate_semantic_complete,
                "semantic_evidence": frame.semantic_evidence.model_dump(mode="json"),
                "unresolved_evidence": frame.unresolved_evidence,
                "capabilities": {
                    "chunk_capture": (
                        frame.capabilities.get("chunk_capture")
                        if frame.capabilities.get("chunk_capture")
                        in {"observed", "unobserved"}
                        else "unobserved"
                    ),
                    "reasoning_capture": (
                        frame.capabilities.get("reasoning_capture")
                        if frame.capabilities.get("reasoning_capture")
                        in {"observed", "unobserved"}
                        else "unobserved"
                    ),
                },
                "projection_truncated": True,
            }
        }
        bounded = json.dumps(
            fallback,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(bounded.encode("utf-8")) > MAX_EPHEMERAL_CONTEXT_BYTES:
            raise AssertionError(
                "fixed critical projection envelope exceeded its bound"
            )
        return bounded


__all__ = [
    "MAX_EPHEMERAL_CONTEXT_BYTES",
    "AtomicProjectionCache",
    "BridgeHealth",
]
