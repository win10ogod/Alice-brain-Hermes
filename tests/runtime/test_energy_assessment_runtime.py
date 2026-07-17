from __future__ import annotations

import sqlite3
from pathlib import Path

from alice_brain_hermes.core.action import EnergyAssessmentStatus
from alice_brain_hermes.core.events import new_event, thaw_json
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.energy import (
    ENERGY_DIMENSIONS,
    EnergyAssessmentChoiceV1,
)
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.store import SQLiteLedger


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


def requested_engine(tmp_path: Path) -> tuple[SQLiteLedger, ConsciousEngine, str]:
    brain_id = new_id()
    action_id = "action-energy-runtime"
    ledger = SQLiteLedger.open(tmp_path / "runtime.db")
    engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)
    engine.append(
        new_event(
            "action.proposed",
            brain_id,
            brain_id,
            {
                "action_id": action_id,
                "intent": {
                    "kind": "hermes.tool_call",
                    "tool_name": "shell",
                    "args": {"command": "pytest -q"},
                },
            },
            action_id=action_id,
        )
    )
    engine.append(
        new_event(
            "action.energy_requested",
            brain_id,
            brain_id,
            {
                "action_id": action_id,
                "assessment_source": "hermes_host_llm",
                "prompt_version": "alice-energy-v1",
            },
            action_id=action_id,
        )
    )
    return ledger, engine, action_id


def provenance(input_sha256: str) -> dict[str, object]:
    return {
        "agent_id": "default",
        "audit": {
            "plugin_id": "alice-brain",
            "profile": "",
            "purpose": "alice_energy_assessment",
            "schema_name": "alice_energy_assessment_v1",
        },
        "input_sha256": input_sha256,
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


def test_energy_lease_claim_and_completion_persist_host_vector(
    tmp_path: Path,
) -> None:
    ledger, engine, action_id = requested_engine(tmp_path)
    try:
        lease = engine.claim_energy_assessment()
        assert lease is not None
        assert lease.action_id == action_id
        assert lease.assessment_input["st"]["tool_name"] == "shell"
        assert engine.claim_energy_assessment() is None
        input_sha256 = ledger._energy_fingerprint(
            ledger._canonical_json_value(thaw_json(lease.assessment_input))
        )
        host_provenance = provenance(input_sha256)

        assert (
            engine.complete_energy_assessment(
                lease.lease_id, choice(), host_provenance
            )
            == "completed"
        )
        action = engine.state.actions[action_id]
        vector = engine.state.energies[action_id]
        assert action.energy_assessment_status is EnergyAssessmentStatus.ASSESSED
        assert vector.salience == 0.9
        assert vector.assessment_source == "hermes_host_llm"
        assert vector.provenance["model"] == "host-model"
        assert engine.complete_energy_assessment(
            lease.lease_id, choice(), host_provenance
        ) == "completed"
        assert engine.claim_energy_assessment() is None
        assert ledger.replay(engine.brain_id) == engine.state
    finally:
        ledger.close()


def test_energy_failure_is_terminal_visible_and_never_creates_default_vector(
    tmp_path: Path,
) -> None:
    ledger, engine, action_id = requested_engine(tmp_path)
    try:
        lease = engine.claim_energy_assessment()
        assert lease is not None
        assert engine.fail_energy_assessment(
            lease.lease_id, "llm_error.RuntimeError"
        ) == "failed"
        action = engine.state.actions[action_id]
        assert action.energy_assessment_status is EnergyAssessmentStatus.FAILED
        assert action.energy_failure_code == "llm_error.RuntimeError"
        assert action_id not in engine.state.energies
        assert engine.claim_energy_assessment() is None
        assert ledger.replay(engine.brain_id) == engine.state
    finally:
        ledger.close()


def test_v6_legacy_neutral_energy_is_requeued_for_hermes_host_llm(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-neutral-energy.db"
    brain_id = new_id()
    action_id = "legacy-hermes-action"
    with SQLiteLedger.open(database) as ledger:
        engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)
        engine.append(
            new_event(
                "action.proposed",
                brain_id,
                brain_id,
                {
                    "action_id": action_id,
                    "intent": {"kind": "hermes.tool_call", "tool_name": "shell"},
                },
                action_id=action_id,
            )
        )
        legacy = new_event(
            "action.energy_assessed",
            brain_id,
            brain_id,
            {
                "action_id": action_id,
                "arousal": 0.0,
                "control": 0.5,
                "cost": 0.5,
                "deficits": {},
                "evidence_basis": {},
                "personality_relevance": 0.5,
                "resources": 0.5,
                "salience": 0.5,
                "unknown_dimensions": list(ENERGY_DIMENSIONS),
                "urgency": 0.5,
                "valence": 0.0,
            },
            action_id=action_id,
        )
        engine.append(legacy)

    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX energy_assessment_pending")
        connection.execute("DROP TABLE energy_assessment_lease")
        connection.execute(
            "UPDATE schema_metadata SET value = '6' WHERE key = 'schema_version'"
        )
        connection.execute("PRAGMA user_version = 6")

    with SQLiteLedger.open(database) as migrated:
        state = migrated.replay(brain_id)
        action = state.actions[action_id]
        assert action.energy_assessment_status is EnergyAssessmentStatus.PENDING
        assert action.energy_assessment_event_id is None
        assert action.energy_request_event_id is not None
        assert action_id not in state.energies
        migration = migrated.get_event(action.energy_request_event_id)
        assert migration is not None
        assert migration.adapter_id == "alice-brain-hermes-energy-migration-v1"
        assert migration.payload["reassessment_reason"] == "legacy_neutral_default"

        engine = ConsciousEngine(migrated, brain_id, actor_id=brain_id)
        lease = engine.claim_energy_assessment()
        assert lease is not None
        assert lease.action_id == action_id


def test_v6_receipt_terminal_action_is_reconstructed_during_upgrade(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-receipt-action.db"
    brain_id = new_id()
    action_id = "legacy-receipt-action"
    with SQLiteLedger.open(database) as ledger:
        engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)
        for event_type, payload in (
            (
                "action.proposed",
                {"action_id": action_id, "intent": {"kind": "hermes.tool_call"}},
            ),
            ("action.prepared", {"action_id": action_id, "branch_id": None}),
            ("action.dispatched", {"action_id": action_id}),
            (
                "action.receipt",
                {
                    "action_id": action_id,
                    "status": "success",
                    "effect_observation_ids": [],
                },
            ),
        ):
            engine.append(
                new_event(
                    event_type,
                    brain_id,
                    brain_id,
                    payload,
                    action_id=action_id,
                )
            )

    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX energy_assessment_pending")
        connection.execute("DROP TABLE energy_assessment_lease")
        connection.execute(
            "UPDATE schema_metadata SET value = '6' WHERE key = 'schema_version'"
        )
        connection.execute("PRAGMA user_version = 6")

    with SQLiteLedger.open(database) as migrated:
        state = migrated.replay(brain_id)
        action = state.actions[action_id]
        assert action.phase.value == "reconstructed"
        assert action.reconstruction == {
            "action_id": action_id,
            "assessment": "execution_succeeded",
        }
        migration = migrated.get_event(action.last_event_id)
        assert migration is not None
        assert migration.adapter_id == "alice-brain-hermes-action-migration-v1"
