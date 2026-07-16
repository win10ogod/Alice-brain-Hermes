"""Strict bounded daemon status models derived from persisted evidence."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alice_brain_hermes.ids import validate_id
from alice_brain_hermes.protocol.models import (
    FRAME_SCHEMA_VERSION,
    GAP_SCHEMA_VERSION,
    OBSERVER_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    RECORD_SCHEMA_VERSION,
)

UNOBSERVED_HERMES_FIELDS = (
    "chunk_capture",
    "reasoning_capture",
    "unregistered_host_state",
)


class _StrictStatusModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class SchedulerHealthSummaryV1(_StrictStatusModel):
    """Exact aggregate of the daemon's continuous scheduler registry."""

    status: Literal["healthy", "degraded"]
    fail_stopped: bool
    brain_count: int = Field(ge=0)
    engine_count: int = Field(ge=0)
    scheduler_count: int = Field(ge=0)
    running_scheduler_count: int = Field(ge=0)
    degraded_brain_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _status_matches_counts(self) -> SchedulerHealthSummaryV1:
        if self.running_scheduler_count > self.scheduler_count:
            raise ValueError("running scheduler count exceeds scheduler count")
        if self.degraded_brain_count > self.brain_count:
            raise ValueError("degraded brain count exceeds brain count")
        healthy = (
            not self.fail_stopped
            and self.engine_count == self.brain_count
            and self.scheduler_count == self.brain_count
            and self.running_scheduler_count == self.brain_count
            and self.degraded_brain_count == 0
        )
        if (self.status == "healthy") is not healthy:
            raise ValueError("scheduler health label does not match its counts")
        return self


class BridgeConnectionSummaryV1(_StrictStatusModel):
    """Exact persisted partition of all Hermes bridge streams."""

    state: Literal["never_connected", "connected", "idle", "degraded"]
    total_bridges: int = Field(ge=0)
    connected_open_bridges: int = Field(ge=0)
    disconnected_open_bridges: int = Field(ge=0)
    clean_closed_bridges: int = Field(ge=0)
    abandoned_bridges: int = Field(ge=0)

    @model_validator(mode="after")
    def _state_matches_partition(self) -> BridgeConnectionSummaryV1:
        partition = (
            self.connected_open_bridges
            + self.disconnected_open_bridges
            + self.clean_closed_bridges
            + self.abandoned_bridges
        )
        if self.total_bridges != partition:
            raise ValueError("bridge connection counts are not an exact partition")
        if self.disconnected_open_bridges:
            expected = "degraded"
        elif self.connected_open_bridges:
            expected = "connected"
        elif self.total_bridges:
            expected = "idle"
        else:
            expected = "never_connected"
        if self.state != expected:
            raise ValueError("bridge connection state does not match its partition")
        return self


class SemanticEvidenceSummaryV1(_StrictStatusModel):
    """Bounded counts of persisted semantic derivation evidence."""

    semantic_records: int = Field(default=0, ge=0)
    legacy_raw_only_records: int = Field(default=0, ge=0)
    semantic_gap_records: int = Field(default=0, ge=0)


class RuntimeSchemaVersionsV1(_StrictStatusModel):
    """Wire and persistence schema versions used by this runtime snapshot."""

    protocol: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    observer: Literal[OBSERVER_SCHEMA_VERSION] = OBSERVER_SCHEMA_VERSION
    record: Literal[RECORD_SCHEMA_VERSION] = RECORD_SCHEMA_VERSION
    gap: Literal[GAP_SCHEMA_VERSION] = GAP_SCHEMA_VERSION
    frame: Literal[FRAME_SCHEMA_VERSION] = FRAME_SCHEMA_VERSION
    semantic: Literal[1] = 1
    sqlite: int = Field(ge=1)


class DaemonRuntimeStatusV1(_StrictStatusModel):
    """One white-box status projection; no field is inferred from trust."""

    schema_version: Literal[1] = 1
    runtime_mode: Literal["continuous_daemon"] = "continuous_daemon"
    cognition_mode: Literal["local"] = "local"
    continuous_runtime: Literal[True] = True
    brain_ids: tuple[str, ...]
    engine_count: int = Field(ge=0)
    scheduler_count: int = Field(ge=0)
    runtime_ready: bool
    scheduler_health: SchedulerHealthSummaryV1
    bridge_connection: BridgeConnectionSummaryV1
    trace_complete: bool
    semantic_complete: bool
    dropped_events: int = Field(ge=0)
    semantic_evidence: SemanticEvidenceSummaryV1
    unobserved_hermes_fields: tuple[
        Literal["chunk_capture"],
        Literal["reasoning_capture"],
        Literal["unregistered_host_state"],
    ] = UNOBSERVED_HERMES_FIELDS
    schema_versions: RuntimeSchemaVersionsV1

    @field_validator("brain_ids")
    @classmethod
    def _exact_brain_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_id(item) for item in value)
        if len(set(validated)) != len(validated):
            raise ValueError("runtime brain ids must be unique")
        if validated != tuple(sorted(validated)):
            raise ValueError("runtime brain ids must be sorted")
        return validated

    @model_validator(mode="after")
    def _evidence_is_consistent(self) -> DaemonRuntimeStatusV1:
        scheduler = self.scheduler_health
        if scheduler.brain_count != len(self.brain_ids):
            raise ValueError("runtime brain count does not match its ids")
        if self.engine_count != scheduler.engine_count:
            raise ValueError("runtime engine count conflicts with scheduler health")
        if self.scheduler_count != scheduler.scheduler_count:
            raise ValueError("runtime scheduler count conflicts with scheduler health")
        writers_ready = (
            scheduler.engine_count == scheduler.brain_count
            and scheduler.scheduler_count == scheduler.brain_count
            and scheduler.running_scheduler_count == scheduler.brain_count
        )
        if self.runtime_ready and not writers_ready:
            raise ValueError("runtime readiness conflicts with writer counts")
        if self.semantic_complete and not self.trace_complete:
            raise ValueError("semantic completeness requires trace completeness")
        evidence = self.semantic_evidence
        if self.dropped_events and (
            self.trace_complete
            or self.semantic_complete
            or evidence.semantic_gap_records == 0
        ):
            raise ValueError("dropped events require visible incomplete gap evidence")
        if evidence.semantic_gap_records and self.semantic_complete:
            raise ValueError("semantic gaps conflict with semantic completeness")
        if evidence.legacy_raw_only_records and self.semantic_complete:
            raise ValueError("legacy raw-only evidence is not semantically complete")
        return self


__all__ = [
    "UNOBSERVED_HERMES_FIELDS",
    "BridgeConnectionSummaryV1",
    "DaemonRuntimeStatusV1",
    "RuntimeSchemaVersionsV1",
    "SchedulerHealthSummaryV1",
    "SemanticEvidenceSummaryV1",
]
