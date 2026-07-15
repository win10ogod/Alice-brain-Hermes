from __future__ import annotations

from datetime import UTC, datetime

from alice_brain_hermes.core.action import ActionOutcome, ActionPhase
from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.core.reducer import reduce_many
from alice_brain_hermes.core.state import BrainState
from alice_brain_hermes.ids import new_id

BRAIN = new_id()
OTHER_ACTOR = new_id()
WALL_TIME = datetime(2026, 7, 15, tzinfo=UTC)


def action_event(event_type: str, payload: dict[str, object]):
    return new_event(
        event_type,
        BRAIN,
        BRAIN,
        payload,
        wall_time=WALL_TIME,
        monotonic_ns=10,
        action_id="action-1",
    )


def dispatched_state() -> BrainState:
    return reduce_many(
        BrainState.genesis(BRAIN),
        [
            action_event(
                "action.proposed",
                {"action_id": "action-1", "intent": {"operation": "test"}},
            ),
            action_event("action.prepared", {"action_id": "action-1"}),
            action_event("action.dispatched", {"action_id": "action-1"}),
        ],
    )


def prepared_state() -> BrainState:
    return reduce_many(
        BrainState.genesis(BRAIN),
        [
            action_event(
                "action.proposed",
                {"action_id": "action-1", "intent": {"operation": "test"}},
            ),
            action_event("action.prepared", {"action_id": "action-1"}),
        ],
    )


def state_with_receipt(status: str) -> BrainState:
    return reduce_many(
        dispatched_state(),
        [
            action_event(
                "action.receipt",
                {"action_id": "action-1", "status": status},
            )
        ],
    )


def test_failure_receipt_confirms_execution_and_records_failure_outcome() -> None:
    state = state_with_receipt("failure")

    action = state.actions["action-1"]
    assert action.phase is ActionPhase.RECEIPT
    assert action.execution_confirmed is True
    assert action.outcome is ActionOutcome.FAILURE


def test_success_receipt_confirms_execution_and_records_success_outcome() -> None:
    state = state_with_receipt("success")
    action = state.actions["action-1"]

    assert action.execution_confirmed is True
    assert action.outcome is ActionOutcome.SUCCESS
    assert action.effect_confirmed is None
    assert state.world.observed == ()


def test_unknown_receipt_keeps_execution_and_outcome_unknown() -> None:
    action = state_with_receipt("unknown").actions["action-1"]

    assert action.execution_confirmed is None
    assert action.outcome is None


def test_blocked_action_records_non_dispatch_without_fabricating_outcome_or_effect() -> None:
    state = reduce_many(
        prepared_state(),
        [
            action_event(
                "action.blocked",
                {
                    "action_id": "action-1",
                    "status": "blocked",
                    "execution_confirmed": False,
                    "outcome": None,
                    "effect_confirmed": None,
                },
            )
        ],
    )

    action = state.actions["action-1"]
    assert action.phase is ActionPhase.BLOCKED
    assert action.execution_confirmed is False
    assert action.outcome is None
    assert action.effect_confirmed is None
    assert state.world.observed == ()


def receipt_with_evidence(*, actor_id: str, status: str = "success"):
    return new_event(
        "action.receipt",
        BRAIN,
        actor_id,
        {
            "action_id": "action-1",
            "status": status,
            "effect_evidence": {
                "kind": "linked_observation",
                "observation_ids": ["observation-1"],
            },
            "observations": [
                {
                    "proposition_id": "observation-1",
                    "content": {"door": "open"},
                }
            ],
        },
        wall_time=WALL_TIME,
        monotonic_ns=11,
        adapter_id="untrusted-adapter",
        action_id="action-1",
    )


def test_untrusted_linked_observation_cannot_confirm_world_effect() -> None:
    state = reduce_many(
        dispatched_state(),
        [receipt_with_evidence(actor_id=OTHER_ACTOR)],
    )

    action = state.actions["action-1"]
    assert action.execution_confirmed is True
    assert action.outcome is ActionOutcome.SUCCESS
    assert action.effect_confirmed is None
    assert state.world.observed == ()


def test_trusted_linked_observation_grounds_effect_independently_of_outcome() -> None:
    state = reduce_many(
        dispatched_state(),
        [receipt_with_evidence(actor_id=BRAIN, status="failure")],
    )

    action = state.actions["action-1"]
    assert action.execution_confirmed is True
    assert action.outcome is ActionOutcome.FAILURE
    assert action.effect_confirmed is True
    assert [item.proposition_id for item in state.world.observed] == ["observation-1"]
