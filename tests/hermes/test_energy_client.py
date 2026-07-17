from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alice_brain_hermes.errors import DaemonClientError
from alice_brain_hermes.hermes.energy_client import DaemonEnergyAssessmentLeasePort
from alice_brain_hermes.hermes.identity_client import hermes_brain_profile
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.energy import (
    ENERGY_DIMENSIONS,
    EnergyAssessmentChoiceV1,
    EnergyAssessmentLeaseV1,
)
from alice_brain_hermes.protocol.models import BrainProfileV1


class FakeClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
        self.calls.append((method, params))
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def choice() -> EnergyAssessmentChoiceV1:
    return EnergyAssessmentChoiceV1(
        deficits={"completion": 0.8},
        salience=0.9,
        urgency=0.7,
        valence=0.2,
        arousal=0.4,
        control=0.8,
        resources=0.6,
        cost=0.3,
        personality_relevance=0.75,
        evidence_basis={
            dimension: f"Observable host basis for {dimension}."
            for dimension in ENERGY_DIMENSIONS
        },
        unknown_dimensions=(),
        summary="Host-assessed action energy.",
    )


def provenance() -> dict[str, object]:
    return {
        "agent_id": "default",
        "audit": {
            "plugin_id": "alice-brain",
            "profile": "",
            "purpose": "alice_energy_assessment",
            "schema_name": "alice_energy_assessment_v1",
        },
        "input_sha256": "a" * 64,
        "model": "host-model",
        "prompt_version": "alice-energy-v1",
        "provider": "host-provider",
        "usage": {
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": None,
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
    }


def test_port_uses_fresh_authenticated_clients_and_strict_wire_models(
    tmp_path: Path,
) -> None:
    brain_id = new_id()
    lease = EnergyAssessmentLeaseV1(
        lease_id=new_id(),
        brain_id=brain_id,
        action_id="action-energy-client",
        request_event_id=new_id(),
        state_sequence=7,
        expires_at=datetime.now(UTC) + timedelta(seconds=30),
        assessment_input={
            "schema_version": 1,
            "pc": {"traits": {"deliberation": 0.7}},
            "st": {"tool_name": "shell"},
        },
    )
    clients = [
        FakeClient(
            [
                {"brain_id": brain_id, "state_sequence": 4, "created": False},
                {"lease": lease.model_dump(mode="json")},
            ]
        ),
        FakeClient([{"status": "completed"}]),
        FakeClient([{"status": "failed"}]),
    ]
    constructed: list[tuple[Path, dict[str, object]]] = []
    profile_reads = 0
    profile = hermes_brain_profile("default")

    def profile_factory() -> BrainProfileV1:
        nonlocal profile_reads
        profile_reads += 1
        return profile

    def client_factory(home: Path, **kwargs: object) -> FakeClient:
        constructed.append((home, kwargs))
        return clients[len(constructed) - 1]

    port = DaemonEnergyAssessmentLeasePort(
        tmp_path,
        profile_factory=profile_factory,
        client_factory=client_factory,
    )
    assert profile_reads == 0
    assert constructed == []

    assert port.claim() == lease
    assessed = choice()
    host_provenance = provenance()
    assert port.complete(lease.lease_id, assessed, host_provenance) == "completed"
    assert port.fail(lease.lease_id, "llm_error.TimeoutError") == "failed"

    assert profile_reads == 1
    assert len(constructed) == 3
    assert all(client.closed for client in clients)
    assert clients[0].calls == [
        ("brain.resolve", {"profile": profile.model_dump(mode="json")}),
        ("energy.assessment.claim", {"brain_id": brain_id}),
    ]
    assert clients[1].calls == [
        (
            "energy.assessment.complete",
            {
                "lease_id": lease.lease_id,
                "choice": assessed.model_dump(mode="json"),
                "provenance": {
                    **host_provenance,
                    "schema_version": 1,
                },
            },
        )
    ]
    assert clients[2].calls == [
        (
            "energy.assessment.fail",
            {
                "lease_id": lease.lease_id,
                "failure_code": "llm_error.TimeoutError",
            },
        )
    ]
    assert all(
        kwargs == {"initialize": True, "timeout_seconds": 3.0}
        for _home, kwargs in constructed
    )


@pytest.mark.parametrize(
    "claim_result",
    [
        {"lease": "invalid"},
        {"lease": None, "extra": True},
        {"lease": {"schema_version": 1}},
    ],
)
def test_port_rejects_malformed_claim_results(
    tmp_path: Path,
    claim_result: dict[str, object],
) -> None:
    brain_id = new_id()
    client = FakeClient(
        [
            {"brain_id": brain_id, "state_sequence": 1, "created": False},
            claim_result,
        ]
    )
    port = DaemonEnergyAssessmentLeasePort(
        tmp_path,
        profile_factory=lambda: hermes_brain_profile("default"),
        client_factory=lambda *_args, **_kwargs: client,
    )

    with pytest.raises(DaemonClientError, match="energy assessment"):
        port.claim()
    assert client.closed is True


def test_port_rejects_non_strict_provenance_before_connecting(tmp_path: Path) -> None:
    client_reads = 0

    def client_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal client_reads
        client_reads += 1
        return object()

    port = DaemonEnergyAssessmentLeasePort(
        tmp_path,
        profile_factory=lambda: hermes_brain_profile("default"),
        client_factory=client_factory,
    )
    invalid = {**provenance(), "usage": {**provenance()["usage"], "extra": 1}}

    with pytest.raises(ValueError):
        port.complete(new_id(), choice(), invalid)
    assert client_reads == 0
