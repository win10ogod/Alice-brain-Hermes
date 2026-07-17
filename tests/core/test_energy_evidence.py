from __future__ import annotations

import pytest
from pydantic import ValidationError

from alice_brain_hermes.core.action import EnergyAssessmentStatus
from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.core.personality import ENERGY_DIMENSIONS, EnergyVector
from alice_brain_hermes.core.reducer import reduce_many
from alice_brain_hermes.core.state import BrainState
from alice_brain_hermes.ids import new_id


def energy_values() -> dict[str, object]:
    return {
        "action_id": "action-1",
        "deficits": {},
        "salience": 0.8,
        "urgency": 0.0,
        "valence": 0.0,
        "arousal": 0.0,
        "control": 0.0,
        "resources": 0.0,
        "cost": 0.0,
        "personality_relevance": 0.0,
    }


def test_v4_energy_defaults_mark_every_dimension_unknown() -> None:
    energy = EnergyVector.model_validate(energy_values())

    assert energy.evidence_basis == {}
    assert energy.unknown_dimensions == ENERGY_DIMENSIONS
    assert energy.activation > 0.0


@pytest.mark.parametrize(
    ("evidence_basis", "unknown_dimensions", "message"),
    [
        (
            {"salience": "hermes.pre_tool_call"},
            ENERGY_DIMENSIONS,
            "both evidenced and unknown",
        ),
        (
            {"salience": "hermes.pre_tool_call"},
            tuple(
                item
                for item in ENERGY_DIMENSIONS
                if item not in {"salience", "urgency"}
            ),
            "classify every dimension",
        ),
        ({}, (*ENERGY_DIMENSIONS[:-1], "imaginary"), "unknown energy dimension"),
        (
            {"salience": ""},
            tuple(item for item in ENERGY_DIMENSIONS if item != "salience"),
            "non-blank bounded source",
        ),
    ],
)
def test_energy_evidence_partition_is_exact_and_bounded(
    evidence_basis: dict[str, object],
    unknown_dimensions: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        EnergyVector.model_validate(
            {
                **energy_values(),
                "evidence_basis": evidence_basis,
                "unknown_dimensions": unknown_dimensions,
            }
        )


def test_energy_reducer_preserves_explicit_basis_and_neutral_priors() -> None:
    brain_id = new_id()
    values = energy_values()
    unknown = tuple(item for item in ENERGY_DIMENSIONS if item != "salience")
    state = reduce_many(
        BrainState.genesis(brain_id),
        [
            new_event(
                "action.proposed",
                brain_id,
                brain_id,
                {"action_id": "action-1", "intent": {}},
                action_id="action-1",
            ),
            new_event(
                "action.energy_assessed",
                brain_id,
                brain_id,
                {
                    **values,
                    "evidence_basis": {"salience": "hermes.pre_tool_call"},
                    "unknown_dimensions": unknown,
                },
                action_id="action-1",
            ),
        ],
    )

    energy = state.energies["action-1"]
    assert energy.evidence_basis == {"salience": "hermes.pre_tool_call"}
    assert energy.unknown_dimensions == unknown
    assert "urgency" in energy.unknown_dimensions


def test_host_energy_request_and_assessment_are_explicit_action_evidence() -> None:
    brain_id = new_id()
    request_event = new_event(
        "action.energy_requested",
        brain_id,
        brain_id,
        {
            "action_id": "action-1",
            "assessment_source": "hermes_host_llm",
            "prompt_version": "alice-energy-v1",
        },
        action_id="action-1",
    )
    assessed_event = new_event(
        "action.energy_assessed",
        brain_id,
        brain_id,
        {
            **energy_values(),
            "evidence_basis": {
                dimension: f"Observed evidence for {dimension}."
                for dimension in ENERGY_DIMENSIONS
            },
            "unknown_dimensions": [],
            "assessment_source": "hermes_host_llm",
            "assessment_summary": "Host-assessed task activation.",
            "provenance": {
                "input_sha256": "a" * 64,
                "model": "host-model",
                "prompt_version": "alice-energy-v1",
                "provider": "host-provider",
            },
        },
        action_id="action-1",
    )
    state = reduce_many(
        BrainState.genesis(brain_id),
        [
            new_event(
                "action.proposed",
                brain_id,
                brain_id,
                {"action_id": "action-1", "intent": {}},
                action_id="action-1",
            ),
            request_event,
            assessed_event,
        ],
    )

    action = state.actions["action-1"]
    energy = state.energies["action-1"]
    assert action.energy_assessment_status is EnergyAssessmentStatus.ASSESSED
    assert action.energy_request_event_id == request_event.event_id
    assert action.energy_assessment_event_id == assessed_event.event_id
    assert action.energy_failure_code is None
    assert energy.assessment_source == "hermes_host_llm"
    assert energy.assessment_summary == "Host-assessed task activation."
    assert energy.provenance["model"] == "host-model"


def test_failed_host_energy_is_visible_without_creating_a_vector() -> None:
    brain_id = new_id()
    state = reduce_many(
        BrainState.genesis(brain_id),
        [
            new_event(
                "action.proposed",
                brain_id,
                brain_id,
                {"action_id": "action-1", "intent": {}},
                action_id="action-1",
            ),
            new_event(
                "action.energy_requested",
                brain_id,
                brain_id,
                {
                    "action_id": "action-1",
                    "assessment_source": "hermes_host_llm",
                    "prompt_version": "alice-energy-v1",
                },
                action_id="action-1",
            ),
            new_event(
                "action.energy_assessment_failed",
                brain_id,
                brain_id,
                {"action_id": "action-1", "failure_code": "llm_error.RuntimeError"},
                action_id="action-1",
            ),
        ],
    )

    action = state.actions["action-1"]
    assert action.energy_assessment_status is EnergyAssessmentStatus.FAILED
    assert action.energy_failure_code == "llm_error.RuntimeError"
    assert state.energy_records == ()
