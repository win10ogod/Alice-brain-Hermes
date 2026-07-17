from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from alice_brain_hermes.hermes.energy import (
    EnergyAssessmentWorker,
    EnergyRunResult,
)
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.energy import (
    EnergyAssessmentChoiceV1,
    EnergyAssessmentLeaseV1,
)


def lease() -> EnergyAssessmentLeaseV1:
    return EnergyAssessmentLeaseV1(
        lease_id=new_id(),
        brain_id=new_id(),
        action_id="hermes-action-" + "a" * 64,
        request_event_id=new_id(),
        state_sequence=7,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
        assessment_input={
            "pc": {"traits": {"deliberation": 0.7}},
            "st": {"tool_name": "shell", "args": {"command": "pytest"}},
        },
    )


class LeasePort:
    def __init__(self, active: EnergyAssessmentLeaseV1 | None) -> None:
        self.active = active
        self.claim_count = 0
        self.completed: list[
            tuple[str, EnergyAssessmentChoiceV1, dict[str, object]]
        ] = []
        self.failed: list[tuple[str, str]] = []
        self.complete_status = "completed"
        self.failure_status = "failed"

    def claim(self) -> EnergyAssessmentLeaseV1 | None:
        self.claim_count += 1
        return self.active

    def complete(
        self,
        lease_id: str,
        choice: EnergyAssessmentChoiceV1,
        provenance: dict[str, object],
    ) -> str:
        self.completed.append((lease_id, choice, provenance))
        return self.complete_status

    def fail(self, lease_id: str, failure_code: str) -> str:
        self.failed.append((lease_id, failure_code))
        return self.failure_status


class StructuredLlm:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, object]] = []

    def complete_structured(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            parsed=self.parsed,
            content_type="json",
            provider="host-provider",
            model="host-model",
            agent_id="default",
            usage=SimpleNamespace(
                input_tokens=101,
                output_tokens=37,
                total_tokens=138,
                cache_read_tokens=5,
                cache_write_tokens=0,
                cost_usd=0.0125,
            ),
            audit={
                "plugin_id": "alice-brain",
                "purpose": "alice_energy_assessment",
                "profile": "",
                "schema_name": "alice_energy_assessment_v1",
            },
        )


def valid_choice() -> dict[str, object]:
    return {
        "deficits": {"completion": 0.8},
        "salience": 0.9,
        "urgency": 0.7,
        "valence": 0.2,
        "arousal": 0.4,
        "control": 0.8,
        "resources": 0.6,
        "cost": 0.3,
        "personality_relevance": 0.75,
        "evidence_basis": {
            "deficits": "Completion pressure is explicit in the active task.",
            "salience": "The pending verification is directly relevant.",
            "urgency": "The action unblocks the active task.",
            "valence": "A successful verification has positive task value.",
            "arousal": "The action requires moderate focused attention.",
            "control": "The tool invocation is directly controllable.",
            "resources": "The required local test runner is available.",
            "cost": "The full test suite has a measurable runtime cost.",
            "personality_relevance": "Deliberation is present in PC evidence.",
        },
        "unknown_dimensions": [],
        "summary": "High-value verification with bounded cost.",
    }


def test_worker_uses_only_hermes_host_defaults_and_submits_exact_evidence() -> None:
    active = lease()
    port = LeasePort(active)
    llm = StructuredLlm(valid_choice())
    worker = EnergyAssessmentWorker(lease_port=port, llm_factory=lambda: llm)

    assert worker.run_once() is EnergyRunResult.COMPLETED

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert set(call) == {
        "input",
        "instructions",
        "json_schema",
        "purpose",
        "schema_name",
    }
    assert not {
        "agent_id",
        "max_tokens",
        "model",
        "profile",
        "provider",
        "system_prompt",
        "temperature",
        "timeout",
    }.intersection(call)
    assert port.failed == []
    [(lease_id, choice, provenance)] = port.completed
    assert lease_id == active.lease_id
    assert choice.salience == 0.9
    assert choice.unknown_dimensions == ()
    assert provenance == {
        "agent_id": "default",
        "audit": {
            "plugin_id": "alice-brain",
            "profile": "",
            "purpose": "alice_energy_assessment",
            "schema_name": "alice_energy_assessment_v1",
        },
        "input_sha256": provenance["input_sha256"],
        "model": "host-model",
        "prompt_version": "alice-energy-v1",
        "provider": "host-provider",
        "usage": {
            "cache_read_tokens": 5,
            "cache_write_tokens": 0,
            "cost_usd": 0.0125,
            "input_tokens": 101,
            "output_tokens": 37,
            "total_tokens": 138,
        },
    }
    assert isinstance(provenance["input_sha256"], str)
    assert len(provenance["input_sha256"]) == 64


@pytest.mark.parametrize(
    "parsed",
    [
        None,
        {},
        {**valid_choice(), "salience": 1.1},
        {**valid_choice(), "extra": True},
        {**valid_choice(), "unknown_dimensions": ["salience"]},
        {
            **valid_choice(),
            "evidence_basis": {
                key: value
                for key, value in valid_choice()["evidence_basis"].items()
                if key != "cost"
            },
            "unknown_dimensions": [],
        },
    ],
)
def test_invalid_structured_energy_fails_without_neutral_fallback(
    parsed: object,
) -> None:
    active = lease()
    port = LeasePort(active)
    worker = EnergyAssessmentWorker(
        lease_port=port,
        llm_factory=lambda: StructuredLlm(parsed),
    )

    assert worker.run_once() is EnergyRunResult.FAILED
    assert port.completed == []
    assert port.failed == [(active.lease_id, "invalid_structured_assessment")]


def test_provider_failure_records_type_without_fabricating_energy() -> None:
    active = lease()
    port = LeasePort(active)

    class FailedLlm:
        def complete_structured(self, **_kwargs: object) -> object:
            raise RuntimeError("secret provider response")

    worker = EnergyAssessmentWorker(lease_port=port, llm_factory=FailedLlm)

    assert worker.run_once() is EnergyRunResult.FAILED
    assert port.completed == []
    assert port.failed == [(active.lease_id, "llm_error.RuntimeError")]


def test_transient_submit_retries_same_choice_without_second_llm_call() -> None:
    active = lease()

    class TransientPort(LeasePort):
        def claim(self) -> EnergyAssessmentLeaseV1 | None:
            self.claim_count += 1
            return active if self.claim_count == 1 else None

        def complete(
            self,
            lease_id: str,
            choice: EnergyAssessmentChoiceV1,
            provenance: dict[str, object],
        ) -> str:
            self.completed.append((lease_id, choice, provenance))
            if len(self.completed) == 1:
                raise MemoryError("transient daemon acknowledgement loss")
            return "completed"

    port = TransientPort(active)
    llm = StructuredLlm(valid_choice())
    worker = EnergyAssessmentWorker(lease_port=port, llm_factory=lambda: llm)

    with pytest.raises(MemoryError, match="acknowledgement"):
        worker.run_once()

    assert worker.run_once() is EnergyRunResult.COMPLETED
    assert port.claim_count == 1
    assert len(llm.calls) == 1
    assert port.completed == [port.completed[0], port.completed[0]]


def test_no_pending_lease_never_resolves_host_llm() -> None:
    port = LeasePort(None)
    worker = EnergyAssessmentWorker(
        lease_port=port,
        llm_factory=lambda: (_ for _ in ()).throw(
            AssertionError("host LLM resolved without an energy lease")
        ),
    )

    assert worker.run_once() is EnergyRunResult.IDLE
    assert port.claim_count == 1
