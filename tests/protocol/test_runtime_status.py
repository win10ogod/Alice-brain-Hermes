from __future__ import annotations

import pytest
from pydantic import ValidationError

from alice_brain_hermes.errors import LedgerIntegrityError
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.status import (
    BridgeConnectionSummaryV1,
    DaemonRuntimeStatusV1,
    EnergyWorkerHealthV1,
    EnergyWorkerReportV1,
    RuntimeSchemaVersionsV1,
    SchedulerHealthSummaryV1,
    SemanticEvidenceSummaryV1,
)


def test_energy_worker_wire_models_cannot_claim_false_health() -> None:
    reporter_id = "00000000-0000-1000-8000-000000000001"
    with pytest.raises(ValidationError):
        EnergyWorkerReportV1(
            reporter_id=reporter_id,
            report_sequence=1,
            worker_started=True,
            terminal_intent_pending=False,
        )

    unreported = EnergyWorkerHealthV1.unreported(stale_after_ms=5_000)
    assert unreported.status == "unreported"
    with pytest.raises(ValidationError):
        EnergyWorkerHealthV1(
            status="healthy",
            worker_started=False,
            terminal_intent_pending=False,
            reporter_id=None,
            report_sequence=0,
            last_report_age_ms=None,
            stale_after_ms=5_000,
        )
    with pytest.raises(ValidationError):
        EnergyWorkerHealthV1(
            status="healthy",
            worker_started=True,
            terminal_intent_pending=False,
            reporter_id=new_id(),
            report_sequence=1,
            last_report_age_ms=5_000,
            stale_after_ms=5_000,
        )
    with pytest.raises(ValidationError):
        EnergyWorkerHealthV1(
            status="healthy",
            worker_started=True,
            terminal_intent_pending=True,
            reporter_id=new_id(),
            report_sequence=1,
            last_report_age_ms=0,
            stale_after_ms=5_000,
        )


def _scheduler(**changes: object) -> SchedulerHealthSummaryV1:
    values: dict[str, object] = {
        "status": "healthy",
        "fail_stopped": False,
        "brain_count": 0,
        "engine_count": 0,
        "scheduler_count": 0,
        "running_scheduler_count": 0,
        "degraded_brain_count": 0,
    }
    values.update(changes)
    return SchedulerHealthSummaryV1.model_validate(values, strict=True)


def _bridge(**changes: object) -> BridgeConnectionSummaryV1:
    values: dict[str, object] = {
        "state": "never_connected",
        "total_bridges": 0,
        "connected_open_bridges": 0,
        "disconnected_open_bridges": 0,
        "clean_closed_bridges": 0,
        "abandoned_bridges": 0,
    }
    values.update(changes)
    return BridgeConnectionSummaryV1.model_validate(values, strict=True)


def _energy() -> EnergyWorkerHealthV1:
    return EnergyWorkerHealthV1.unreported(stale_after_ms=5_000)


def test_fresh_runtime_status_is_complete_zero_evidence() -> None:
    status = DaemonRuntimeStatusV1(
        brain_ids=(),
        engine_count=0,
        scheduler_count=0,
        runtime_ready=True,
        scheduler_health=_scheduler(),
        bridge_connection=_bridge(),
        trace_complete=True,
        semantic_complete=True,
        dropped_events=0,
        semantic_evidence=SemanticEvidenceSummaryV1(),
        energy_worker_health=_energy(),
        unobserved_hermes_fields=(
            "chunk_capture",
            "reasoning_capture",
        ),
        schema_versions=RuntimeSchemaVersionsV1(sqlite=5),
    )

    assert status.runtime_mode == "continuous_daemon"
    assert status.cognition_mode == "local"
    assert status.continuous_runtime is True
    assert status.bridge_connection.state == "never_connected"
    assert status.trace_complete is True
    assert status.semantic_complete is True
    assert status.dropped_events == 0
    assert status.host_state_scope == "registered_hook_payloads_only"


@pytest.mark.parametrize(
    "values",
    [
        {"status": "healthy", "brain_count": 1},
        {
            "status": "degraded",
            "fail_stopped": False,
            "brain_count": 0,
            "engine_count": 0,
            "scheduler_count": 0,
            "running_scheduler_count": 0,
            "degraded_brain_count": 0,
        },
    ],
)
def test_scheduler_summary_cannot_claim_false_health(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _scheduler(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"state": "connected", "total_bridges": 0},
        {
            "state": "idle",
            "total_bridges": 1,
            "clean_closed_bridges": 0,
        },
        {
            "state": "connected",
            "total_bridges": 1,
            "connected_open_bridges": 0,
            "disconnected_open_bridges": 1,
        },
    ],
)
def test_bridge_summary_state_is_derived_from_exact_partition(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _bridge(**values)


def test_runtime_status_rejects_false_completeness_and_count_collisions() -> None:
    base = {
        "brain_ids": (),
        "engine_count": 0,
        "scheduler_count": 0,
        "runtime_ready": True,
        "scheduler_health": _scheduler(),
        "bridge_connection": _bridge(),
        "trace_complete": True,
        "semantic_complete": True,
        "dropped_events": 0,
        "semantic_evidence": SemanticEvidenceSummaryV1(),
        "energy_worker_health": _energy(),
        "unobserved_hermes_fields": (
            "chunk_capture",
            "reasoning_capture",
        ),
        "schema_versions": RuntimeSchemaVersionsV1(sqlite=5),
    }

    with pytest.raises(ValidationError):
        DaemonRuntimeStatusV1(**{**base, "dropped_events": 1})
    with pytest.raises(ValidationError):
        DaemonRuntimeStatusV1(
            **{
                **base,
                "brain_ids": ("00000000-0000-4000-8000-000000000001",),
            }
        )
    gated = DaemonRuntimeStatusV1(
        **{
            **base,
            "runtime_ready": False,
        }
    )
    assert gated.runtime_ready is False


def test_running_but_degraded_scheduler_is_ready_without_claiming_health() -> None:
    scheduler = _scheduler(
        status="degraded",
        fail_stopped=True,
    )

    status = DaemonRuntimeStatusV1(
        brain_ids=(),
        engine_count=0,
        scheduler_count=0,
        runtime_ready=True,
        scheduler_health=scheduler,
        bridge_connection=_bridge(),
        trace_complete=True,
        semantic_complete=True,
        dropped_events=0,
        semantic_evidence=SemanticEvidenceSummaryV1(),
        energy_worker_health=_energy(),
        schema_versions=RuntimeSchemaVersionsV1(sqlite=5),
    )

    assert status.runtime_ready is True
    assert status.scheduler_health.status == "degraded"


def test_runtime_status_requires_explicit_energy_worker_evidence_on_wire() -> None:
    status = DaemonRuntimeStatusV1(
        brain_ids=(),
        engine_count=0,
        scheduler_count=0,
        runtime_ready=True,
        scheduler_health=_scheduler(),
        bridge_connection=_bridge(),
        trace_complete=True,
        semantic_complete=True,
        dropped_events=0,
        semantic_evidence=SemanticEvidenceSummaryV1(),
        energy_worker_health=_energy(),
        schema_versions=RuntimeSchemaVersionsV1(sqlite=5),
    )
    wire = status.model_dump(mode="json")
    wire.pop("energy_worker_health")

    with pytest.raises(ValidationError):
        DaemonRuntimeStatusV1.model_validate(wire, strict=True)


def test_runtime_status_rejects_observability_brain_coverage_mismatch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.runtime.daemon import HermesDaemonRuntime

    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    try:
        snapshot = runtime.ledger.observability_snapshot()
        monkeypatch.setattr(
            runtime.ledger,
            "observability_snapshot",
            lambda: snapshot.model_copy(update={"brain_count": 1}),
        )

        with pytest.raises(LedgerIntegrityError, match="observability coverage"):
            runtime.status_snapshot()
    finally:
        runtime.close()
