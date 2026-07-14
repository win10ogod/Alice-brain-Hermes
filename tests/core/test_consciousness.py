from __future__ import annotations

from datetime import UTC, datetime
from decimal import localcontext
from uuid import UUID

import pytest
from pydantic import ValidationError

from alice_brain_hermes.core.action import ActionPhase, RDPhase
from alice_brain_hermes.core.cognition import LocalCognitionPort, result_payload
from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.core.personality import EnergyVector
from alice_brain_hermes.core.reducer import reduce_many, reduce_state
from alice_brain_hermes.core.state import BrainState
from alice_brain_hermes.core.workspace import (
    WorkspaceCandidate,
    choose_broadcast,
    derive_candidates,
)
from alice_brain_hermes.errors import DomainInvariantError
from alice_brain_hermes.ids import new_id

BRAIN = new_id()
TOOL_ACTOR = new_id()
OTHER_ACTOR = new_id()
WALL_TIME = datetime(2026, 7, 14, tzinfo=UTC)


def event(
    event_type: str,
    payload: dict[str, object],
    sequence: int | None = None,
    *,
    actor_id: str = BRAIN,
    adapter_id: str | None = None,
    action_id: str | None = None,
):
    return new_event(
        event_type,
        BRAIN,
        actor_id,
        payload,
        sequence=sequence,
        wall_time=WALL_TIME,
        monotonic_ns=10,
        adapter_id=adapter_id,
        action_id=action_id,
    )


def authorized_tool_event(sequence: int | None = None):
    return event(
        "identity.provenance_authorized",
        {"actor_id": TOOL_ACTOR, "adapter_id": "hermes-tool"},
        sequence,
    )


def action_events(*, trusted_receipt: bool = True):
    receipt_actor = TOOL_ACTOR if trusted_receipt else OTHER_ACTOR
    receipt_adapter = "hermes-tool"
    return [
        event(
            "action.proposed",
            {"action_id": "a1", "intent": {"operation": "open-door"}},
        ),
        event(
            "action.energy_assessed",
            {
                "action_id": "a1",
                "deficits": {"access": 0.8},
                "salience": 0.9,
                "urgency": 0.6,
                "valence": 0.2,
                "arousal": 0.5,
                "control": 0.9,
                "resources": 0.8,
                "cost": 0.2,
                "personality_relevance": 0.7,
            },
        ),
        event("action.prepared", {"action_id": "a1", "branch_id": "b1"}),
        event("action.dispatched", {"action_id": "a1"}),
        event(
            "action.receipt",
            {
                "action_id": "a1",
                "status": "success",
                "trusted": True,
                "effect_confirmed": True,
                "effect_evidence": {
                    "kind": "linked_observation",
                    "observation_ids": ["o1"],
                },
                "observations": [
                    {
                        "proposition_id": "o1",
                        "content": {"door": "open"},
                    }
                ],
            },
            actor_id=receipt_actor,
            adapter_id=receipt_adapter,
            action_id="a1",
        ),
        event("action.reconstructed", {"action_id": "a1", "outcome": "done"}),
    ]


def test_genesis_state_is_deeply_immutable_and_has_explicit_layers() -> None:
    state = BrainState.genesis(BRAIN)

    assert state.identity.self_actor_id == BRAIN
    assert state.personality.traits == {}
    assert state.world.observed == ()
    assert state.world.believed == ()
    assert state.world.simulated == ()
    assert state.world.ideal == ()
    assert state.actions == {}
    assert state.energies == {}
    assert state.cognition.cognition_mode == "local"
    with pytest.raises((TypeError, ValidationError)):
        state.world.observed += ()


def test_identity_keeps_self_and_external_actor_attribution_separate() -> None:
    state = reduce_state(
        BrainState.genesis(BRAIN),
        event(
            "identity.actor_registered",
            {
                "actor_id": OTHER_ACTOR,
                "kind": "external_agent",
                "display_name": "Hermes child",
            },
        ),
    )

    assert state.identity.self_actor_id == BRAIN
    assert state.identity.actor(OTHER_ACTOR).kind.value == "external_agent"
    assert state.identity.actor(OTHER_ACTOR).actor_id != state.identity.self_actor_id


def test_only_self_authority_can_add_trusted_provenance() -> None:
    forged = event(
        "identity.provenance_authorized",
        {"actor_id": TOOL_ACTOR, "adapter_id": "hermes-tool"},
        actor_id=OTHER_ACTOR,
    )

    with pytest.raises(DomainInvariantError, match="self authority"):
        reduce_state(BrainState.genesis(BRAIN), forged)


def test_pc_has_three_layers_and_enforces_bounded_revision_rates() -> None:
    state = reduce_many(
        BrainState.genesis(BRAIN),
        [
            event(
                "personality.revised",
                {"layer": "traits", "values": {"care": 0.04}},
            ),
            event(
                "personality.revised",
                {"layer": "adaptations", "values": {"caution": 0.2}},
            ),
            event(
                "personality.revised",
                {
                    "layer": "narrative_ideal",
                    "values": {"reliable": 0.1},
                },
            ),
        ],
    )

    assert state.personality.traits == {"care": 0.04}
    assert state.personality.adaptations == {"caution": 0.2}
    assert state.personality.narrative_ideal == {"reliable": 0.1}
    with pytest.raises(DomainInvariantError, match="bounded rate"):
        reduce_state(
            state,
            event(
                "personality.revised",
                {"layer": "traits", "values": {"care": 0.2}},
            ),
        )


def test_pc_same_clock_revisions_share_one_cumulative_layer_budget() -> None:
    state = BrainState.genesis(BRAIN)
    initial = state.personality.rate_state.traits
    assert initial.capacity == pytest.approx(0.05)
    assert initial.refill_rate == pytest.approx(0.05)
    assert initial.available == pytest.approx(initial.capacity)

    state = reduce_state(
        state,
        event(
            "personality.revised",
            {"layer": "traits", "values": {"care": 0.03, "honesty": 0.02}},
        ),
    )

    assert state.personality.rate_state.traits.available == pytest.approx(0.0)
    before = state
    with pytest.raises(DomainInvariantError, match=r"cumulative.*budget"):
        reduce_state(
            state,
            event(
                "personality.revised",
                {"layer": "traits", "values": {"care": -0.02}},
            ),
        )
    assert state is before
    assert state.personality.traits == {"care": 0.03, "honesty": 0.02}
    with pytest.raises(DomainInvariantError, match=r"cumulative.*budget"):
        reduce_state(
            state,
            event(
                "personality.revised",
                {"layer": "traits", "values": {"care": 0.0300000000001}},
            ),
        )


def test_pc_cumulative_change_is_independent_of_ambient_decimal_precision() -> None:
    tiny_updates = {f"z-tiny-{index:04d}": 1e-30 for index in range(1_000)}
    values = {"a-big": 0.05, **tiny_updates}
    initial = BrainState.genesis(BRAIN)

    with localcontext() as context:
        context.prec = 6
        with pytest.raises(DomainInvariantError, match=r"cumulative.*budget"):
            reduce_state(
                initial,
                event(
                    "personality.revised",
                    {"layer": "traits", "values": values},
                ),
            )

    assert initial.personality.traits == {}
    assert initial.personality.rate_state.traits.available == pytest.approx(0.05)


def test_pc_accepted_revision_is_order_and_ambient_precision_independent() -> None:
    tiny_updates = {f"z-tiny-{index:04d}": 1e-30 for index in range(1_000)}
    first_values = {"a-big": 0.049, **tiny_updates}
    second_values = dict(reversed(tuple(first_values.items())))

    with localcontext() as context:
        context.prec = 6
        first = reduce_state(
            BrainState.genesis(BRAIN),
            event(
                "personality.revised",
                {"layer": "traits", "values": first_values},
            ),
        )
    with localcontext() as context:
        context.prec = 50
        second = reduce_state(
            BrainState.genesis(BRAIN),
            event(
                "personality.revised",
                {"layer": "traits", "values": second_values},
            ),
        )

    assert first == second
    assert first.personality.rate_state.traits.available == pytest.approx(0.001)


def test_pc_budget_refills_by_elapsed_logical_clock_and_rejects_bad_rate_state(
) -> None:
    state = reduce_state(
        BrainState.genesis(BRAIN),
        event(
            "personality.revised",
            {"layer": "traits", "values": {"care": 0.05}},
        ),
    )
    state = reduce_state(
        state,
        event("clock.tick", {"elapsed_seconds": 0.5}),
    )

    bucket = state.personality.rate_state.traits
    assert bucket.logical_clock == pytest.approx(0.5)
    assert bucket.available == pytest.approx(0.025)
    state = reduce_state(
        state,
        event(
            "personality.revised",
            {"layer": "traits", "values": {"care": 0.075}},
        ),
    )
    assert state.personality.rate_state.traits.available == pytest.approx(0.0)

    raw = state.model_dump(mode="python")
    raw["personality"]["rate_state"]["traits"]["logical_clock"] = 1.5
    with pytest.raises(ValidationError, match="logical clock"):
        BrainState.model_validate(raw)

    raw = state.model_dump(mode="python")
    raw["personality"]["rate_state"]["traits"]["refill_rate"] = float("inf")
    with pytest.raises(ValidationError):
        BrainState.model_validate(raw)

    raw = state.model_dump(mode="python")
    raw["personality"]["rate_state"]["traits"]["refill_rate"] += 5e-13
    with pytest.raises(ValidationError, match="runtime policy"):
        BrainState.model_validate(raw)

    raw = state.model_dump(mode="python")
    raw["personality"]["rate_state"]["traits"]["logical_clock"] += 5e-13
    with pytest.raises(ValidationError, match="logical clock"):
        BrainState.model_validate(raw)


def test_energy_is_action_indexed_bounded_and_cannot_dispatch() -> None:
    proposed, assessed, *_ = action_events()
    state = reduce_many(BrainState.genesis(BRAIN), [proposed, assessed])

    energy = state.energies["a1"]
    assert energy.action_id == "a1"
    assert 0.0 <= energy.activation <= 1.0
    assert state.actions["a1"].phase is ActionPhase.PROPOSED
    with pytest.raises(ValidationError):
        EnergyVector.model_validate(
            {
                **energy.model_dump(mode="python"),
                "urgency": float("inf"),
            }
        )
    with pytest.raises(DomainInvariantError, match="existing action"):
        reduce_state(
            BrainState.genesis(BRAIN),
            assessed,
        )


def test_world_event_types_have_fixed_isolated_target_layers() -> None:
    state = reduce_many(
        BrainState.genesis(BRAIN),
        [
            event(
                "belief.updated",
                {"proposition_id": "b1", "content": {"weather": "rain"}},
            ),
            event(
                "simulation.created",
                {"proposition_id": "s1", "content": {"door": "open"}},
            ),
            event(
                "ideal.updated",
                {"proposition_id": "i1", "content": {"door": "closed"}},
            ),
        ],
    )

    assert {item.proposition_id for item in state.world.believed} == {"b1"}
    assert {item.proposition_id for item in state.world.simulated} == {"s1"}
    assert {item.proposition_id for item in state.world.ideal} == {"i1"}
    assert state.world.observed == ()


def test_simulation_and_forged_trust_cannot_change_observed() -> None:
    state = reduce_state(
        BrainState.genesis(BRAIN),
        event(
            "simulation.created",
            {"proposition_id": "s1", "content": {"door": "open"}},
        ),
    )
    forged = event(
        "action.receipt",
        {
            "action_id": "a1",
            "status": "success",
            "trusted": True,
            "effect_confirmed": True,
            "effect_evidence": {
                "kind": "linked_observation",
                "observation_ids": ["o1"],
            },
            "observations": [
                {"proposition_id": "o1", "content": {"door": "open"}}
            ],
        },
        actor_id=OTHER_ACTOR,
        adapter_id="hermes-tool",
        action_id="a1",
    )
    state = reduce_many(state, action_events(trusted_receipt=False)[:-2])
    state = reduce_state(state, forged)

    assert state.world.observed == ()
    assert {item.proposition_id for item in state.world.simulated} == {"s1"}
    assert state.actions["a1"].execution_confirmed is True
    assert state.actions["a1"].effect_confirmed is None


def test_authorized_receipt_with_linked_evidence_can_ground_observed() -> None:
    state = reduce_many(
        BrainState.genesis(BRAIN),
        [authorized_tool_event(), *action_events()],
    )

    action = state.actions["a1"]
    assert action.phase is ActionPhase.RECONSTRUCTED
    assert action.rd_phase is RDPhase.RECONSTRUCT
    assert action.execution_confirmed is True
    assert action.effect_confirmed is True
    assert {item.proposition_id for item in state.world.observed} == {"o1"}


def test_status_success_without_effect_evidence_only_confirms_execution() -> None:
    state = reduce_many(
        BrainState.genesis(BRAIN),
        [authorized_tool_event(), *action_events()[:4]],
    )
    receipt = event(
        "action.receipt",
        {"action_id": "a1", "status": "success"},
        actor_id=TOOL_ACTOR,
        adapter_id="hermes-tool",
        action_id="a1",
    )

    state = reduce_state(state, receipt)

    assert state.actions["a1"].execution_confirmed is True
    assert state.actions["a1"].effect_confirmed is None
    assert state.world.observed == ()


def test_receipt_promotes_only_exact_linked_observation_ids() -> None:
    state = reduce_many(
        BrainState.genesis(BRAIN),
        [authorized_tool_event(), *action_events()[:4]],
    )
    receipt = event(
        "action.receipt",
        {
            "action_id": "a1",
            "status": "success",
            "effect_evidence": {
                "kind": "linked_observation",
                "observation_ids": ["o-linked"],
            },
            "observations": [
                {
                    "proposition_id": "o-linked",
                    "content": {"door": "open"},
                },
                {
                    "proposition_id": "o-bystander",
                    "content": {"forged": "not linked"},
                },
            ],
        },
        actor_id=TOOL_ACTOR,
        adapter_id="hermes-tool",
        action_id="a1",
    )

    state = reduce_state(state, receipt)

    assert state.actions["a1"].effect_confirmed is True
    assert {item.proposition_id for item in state.world.observed} == {"o-linked"}


@pytest.mark.parametrize(
    ("prefix", "illegal", "message"),
    [
        ([], "action.prepared", "proposed"),
        ([], "action.dispatched", "proposed"),
        ([], "action.receipt", "proposed"),
        ([0], "action.dispatched", "prepared"),
        ([0], "action.receipt", "dispatched"),
        ([0, 2], "action.reconstructed", "receipt"),
    ],
)
def test_every_action_phase_requires_its_legal_predecessor(
    prefix: list[int], illegal: str, message: str
) -> None:
    legal = action_events()
    state = reduce_many(BrainState.genesis(BRAIN), [legal[index] for index in prefix])
    payload: dict[str, object] = {"action_id": "a1"}
    if illegal == "action.receipt":
        payload["status"] = "success"

    with pytest.raises(DomainInvariantError, match=message):
        reduce_state(state, event(illegal, payload, action_id="a1"))


def test_envelope_and_payload_action_ids_must_match() -> None:
    proposed = event(
        "action.proposed",
        {"action_id": "a1", "intent": {}},
        action_id="a2",
    )

    with pytest.raises(DomainInvariantError, match="action_id mismatch"):
        reduce_state(BrainState.genesis(BRAIN), proposed)


def test_repeated_or_late_action_transitions_are_rejected() -> None:
    legal = action_events()
    proposed = reduce_state(BrainState.genesis(BRAIN), legal[0])
    prepared = reduce_state(proposed, legal[2])
    dispatched = reduce_state(prepared, legal[3])
    receipted = reduce_state(dispatched, legal[4])
    reconstructed = reduce_state(receipted, legal[5])

    cases = [
        (proposed, legal[0]),
        (prepared, legal[2]),
        (dispatched, legal[3]),
        (receipted, legal[4]),
        (reconstructed, legal[5]),
    ]
    for state, repeated in cases:
        with pytest.raises(DomainInvariantError):
            reduce_state(state, repeated)


def test_local_cognition_is_deterministic_honest_and_side_effect_free() -> None:
    port = LocalCognitionPort()
    content = {"goal": "open door", "constraint": "avoid damage"}

    first = port.reflect(content, source_ids=("workspace-2", "workspace-1"))
    second = port.reflect(content, source_ids=("workspace-1", "workspace-2"))

    assert first == second
    assert first.cognition_mode == "local"
    assert first.provider_used is False
    assert first.uncertainty_basis == "deterministic_heuristic"
    assert first.calibrated is False
    assert [item.stance for item in first.alternatives] == [
        "proceed",
        "defer",
        "seek_observation",
    ]
    assert all(item.expected_consequences for item in first.alternatives)
    assert 0.0 <= first.uncertainty <= 1.0
    assert BrainState.genesis(BRAIN).actions == {}


def test_legacy_local_cognition_event_gets_explicit_uncertainty_labels() -> None:
    payload = result_payload(LocalCognitionPort().reflect({"legacy": True}))
    payload.pop("uncertainty_basis", None)
    payload.pop("calibrated", None)

    state = reduce_state(
        BrainState.genesis(BRAIN),
        event("cognition.reflected", payload),
    )

    restored = state.cognition.reflections[-1]
    assert restored.uncertainty_basis == "deterministic_heuristic"
    assert restored.calibrated is False


def candidate(
    candidate_id: str,
    specialist: str,
    score: float,
    content: dict[str, object] | None = None,
) -> WorkspaceCandidate:
    return WorkspaceCandidate(
        candidate_id=candidate_id,
        specialist=specialist,
        score=score,
        content=content or {"id": candidate_id},
        source_ids=(candidate_id,),
        cycle=1,
    )


def test_workspace_is_bounded_deduplicated_and_has_stable_tie_breaks() -> None:
    selected = choose_broadcast(
        [
            candidate("z", "memory", 0.7, {"same": True}),
            candidate("a", "drives", 0.7),
            candidate("b", "prediction_error", 0.9),
            candidate("duplicate", "drives", 0.8, {"same": True}),
        ],
        capacity=2,
    )

    assert [item.candidate_id for item in selected] == ["b", "duplicate"]
    assert len({item.content.canonical_json() for item in selected}) == 2


def test_broadcast_from_cycle_n_changes_specialists_in_cycle_n_plus_one() -> None:
    initial = BrainState.genesis(BRAIN)
    before = derive_candidates(initial)
    state = reduce_state(
        initial,
        event(
            "workspace.broadcast",
            {
                "cycle": 1,
                "candidates": [
                    {
                        "candidate_id": "seed",
                        "specialist": "memory",
                        "score": 0.8,
                        "content": {"topic": "door"},
                        "source_ids": ["seed"],
                        "cycle": 1,
                    }
                ],
            },
        ),
    )
    after = derive_candidates(state)

    assert before != after
    assert any("seed" in item.source_ids for item in after)
    assert all(item.cycle == 2 for item in after)


def test_each_required_specialist_bids_and_energy_intervention_changes_drive() -> None:
    proposed, assessed, *_ = action_events()
    state = reduce_many(BrainState.genesis(BRAIN), [proposed, assessed])
    before = derive_candidates(state)
    stronger = assessed.model_copy(
        update={
            "payload": {
                **assessed.payload,
                "urgency": 1.0,
                "salience": 1.0,
                "cost": 0.0,
            }
        }
    ).revalidated()
    after = derive_candidates(reduce_state(state, stronger))

    assert {item.specialist for item in before} == {
        "prediction_error",
        "drives",
        "incomplete_action",
        "memory",
        "self_world_conflict",
    }
    before_drive = next(item for item in before if item.specialist == "drives")
    after_drive = next(item for item in after if item.specialist == "drives")
    assert after_drive.score > before_drive.score


def test_reducing_the_same_sequenced_history_has_exact_replay_parity() -> None:
    raw = [authorized_tool_event(1)]
    raw.extend(
        item.model_copy(update={"sequence": index})
        for index, item in enumerate(action_events(), start=2)
    )

    first = reduce_many(BrainState.genesis(BRAIN), raw)
    second = reduce_many(BrainState.genesis(BRAIN), raw)

    assert first == second
    assert first.canonical_json() == second.canonical_json()
    assert UUID(first.identity.self_actor_id).version == 4
