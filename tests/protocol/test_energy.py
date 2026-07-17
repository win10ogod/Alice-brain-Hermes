from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.energy import (
    ENERGY_DIMENSIONS,
    EnergyAssessmentChoiceV1,
    EnergyAssessmentLeaseV1,
)


def evidence() -> dict[str, str]:
    return {
        dimension: f"Observed basis for {dimension}."
        for dimension in ENERGY_DIMENSIONS
    }


def test_energy_choice_requires_an_exact_evidence_unknown_partition() -> None:
    choice = EnergyAssessmentChoiceV1(
        deficits={"completion": 0.5},
        salience=0.8,
        urgency=0.7,
        valence=0.1,
        arousal=0.2,
        control=0.9,
        resources=0.6,
        cost=0.4,
        personality_relevance=0.75,
        evidence_basis=evidence(),
        unknown_dimensions=(),
        summary="Evidence-grounded action energy.",
    )

    assert set(choice.evidence_basis) == set(ENERGY_DIMENSIONS)
    assert choice.unknown_dimensions == ()


def test_energy_choice_rejects_overlap_or_unclassified_dimensions() -> None:
    with pytest.raises(ValidationError, match="both evidenced and unknown"):
        EnergyAssessmentChoiceV1(
            deficits={},
            salience=0.5,
            urgency=0.5,
            valence=0.0,
            arousal=0.0,
            control=0.5,
            resources=0.5,
            cost=0.5,
            personality_relevance=0.5,
            evidence_basis=evidence(),
            unknown_dimensions=("salience",),
            summary="invalid overlap",
        )


def test_energy_lease_binds_exact_action_and_bounded_input() -> None:
    lease = EnergyAssessmentLeaseV1(
        lease_id=new_id(),
        brain_id=new_id(),
        action_id="hermes-action-" + "a" * 64,
        request_event_id=new_id(),
        state_sequence=3,
        expires_at=datetime.now(UTC) + timedelta(seconds=120),
        assessment_input={"pc": {}, "st": {"tool_name": "shell"}},
    )

    assert lease.assessment_input["st"] == {"tool_name": "shell"}
