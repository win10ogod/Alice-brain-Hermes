from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from alice_brain_hermes.core.action import (
    MAX_ACTION_RECEIPT_HISTORY,
    MAX_ACTION_RECONSTRUCTION_HISTORY,
    ActionOutcome,
    ActionPhase,
    ActionReceiptDisposition,
    ActionReceiptStatus,
    ActionReconstructionRecord,
)
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


def test_blocked_action_never_claims_dispatch_or_execution() -> None:
    state = reduce_many(
        prepared_state(),
        [
            action_event(
                "action.blocked",
                {
                    "action_id": "action-1",
                    "reason": "guardrail",
                    "source_status": "blocked",
                },
            )
        ],
    )

    action = state.actions["action-1"]
    assert action.phase is ActionPhase.BLOCKED
    assert ActionPhase.DISPATCHED not in action.phase_history
    assert action.execution_confirmed is False
    assert action.outcome is None
    assert action.effect_confirmed is None


def test_blocked_action_record_rejects_executed_or_dispatched_claims() -> None:
    blocked = reduce_many(
        prepared_state(),
        [action_event("action.blocked", {"action_id": "action-1"})],
    ).actions["action-1"]
    raw = blocked.model_dump(mode="python")
    raw["execution_confirmed"] = True
    raw["phase_history"] = (*blocked.phase_history, ActionPhase.DISPATCHED)

    with pytest.raises(ValidationError, match="blocked action cannot claim dispatch"):
        type(blocked).model_validate(raw)


@pytest.mark.parametrize(
    ("status", "source_status", "source_error_type", "execution", "outcome"),
    [
        ("success", "ok", None, True, ActionOutcome.SUCCESS),
        ("failure", "error", None, True, ActionOutcome.FAILURE),
        ("unknown", "error", "thread_missing_result", None, None),
        ("unknown", "timeout", None, None, None),
        ("unknown", "cancelled", None, None, None),
    ],
)
def test_typed_receipt_history_preserves_exact_post_tool_semantics(
    status: str,
    source_status: str,
    source_error_type: str | None,
    execution: bool | None,
    outcome: ActionOutcome | None,
) -> None:
    payload: dict[str, object] = {
        "action_id": "action-1",
        "status": status,
        "source_status": source_status,
    }
    if source_error_type is not None:
        payload["source_error_type"] = source_error_type

    action = reduce_many(
        dispatched_state(),
        [action_event("action.receipt", payload)],
    ).actions["action-1"]

    receipt = action.receipt_history[-1]
    assert receipt.status is ActionReceiptStatus(status)
    assert receipt.source_status == source_status
    assert receipt.source_error_type == source_error_type
    assert receipt.execution_confirmed is execution
    assert receipt.outcome is outcome
    assert action.execution_confirmed is execution
    assert action.outcome is outcome


def test_typed_receipt_rejects_inconsistent_execution_claims() -> None:
    receipt = state_with_receipt("success").actions["action-1"].receipt_history[0]
    raw = receipt.model_dump(mode="python")
    raw["execution_confirmed"] = None

    with pytest.raises(
        ValidationError, match="confirmed receipt must confirm execution"
    ):
        type(receipt).model_validate(raw)


def test_late_receipt_resolves_unknown_without_a_second_dispatch() -> None:
    reconstructed = reduce_many(
        state_with_receipt("unknown"),
        [action_event("action.reconstructed", {"action_id": "action-1"})],
    )

    action = reduce_many(
        reconstructed,
        [
            action_event(
                "action.receipt",
                {
                    "action_id": "action-1",
                    "status": "success",
                    "source_status": "ok",
                    "late": True,
                },
            )
        ],
    ).actions["action-1"]

    assert action.phase is ActionPhase.RECEIPT
    assert action.phase_history.count(ActionPhase.DISPATCHED) == 1
    assert action.execution_confirmed is True
    assert action.outcome is ActionOutcome.SUCCESS
    assert [item.disposition for item in action.receipt_history] == [
        ActionReceiptDisposition.PENDING,
        ActionReceiptDisposition.RESOLUTION,
    ]


def test_corroborating_and_conflicting_receipts_are_audited_without_overwrite() -> None:
    first = reduce_many(
        state_with_receipt("success"),
        [action_event("action.reconstructed", {"action_id": "action-1"})],
    )
    corroborated = reduce_many(
        first,
        [
            action_event(
                "action.receipt",
                {
                    "action_id": "action-1",
                    "status": "success",
                    "source_status": "ok",
                    "late": True,
                },
            ),
            action_event("action.reconstructed", {"action_id": "action-1"}),
        ],
    )
    action = reduce_many(
        corroborated,
        [
            action_event(
                "action.receipt",
                {
                    "action_id": "action-1",
                    "status": "failure",
                    "source_status": "error",
                    "late": True,
                },
            )
        ],
    ).actions["action-1"]

    assert action.execution_confirmed is True
    assert action.outcome is ActionOutcome.SUCCESS
    assert [item.disposition for item in action.receipt_history] == [
        ActionReceiptDisposition.CONFIRMED,
        ActionReceiptDisposition.CORROBORATION,
        ActionReceiptDisposition.CONFLICT,
    ]
    assert action.receipt_history[-1].outcome is ActionOutcome.FAILURE
    assert action.receipt_corroboration_count == 1
    assert action.receipt_conflict_count == 1


def test_blocked_action_can_be_reconstructed_with_typed_history() -> None:
    blocked = reduce_many(
        prepared_state(),
        [
            action_event(
                "action.blocked",
                {
                    "action_id": "action-1",
                    "reason": "guardrail",
                    "source_status": "blocked",
                },
            )
        ],
    )

    action = reduce_many(
        blocked,
        [
            action_event(
                "action.reconstructed",
                {
                    "action_id": "action-1",
                    "assessment": "dispatch prevented",
                },
            )
        ],
    ).actions["action-1"]

    assert action.phase is ActionPhase.RECONSTRUCTED
    assert ActionPhase.BLOCKED in action.phase_history
    assert ActionPhase.DISPATCHED not in action.phase_history
    assert action.execution_confirmed is False
    assert action.outcome is None
    assert len(action.reconstruction_history) == 1
    assert isinstance(action.reconstruction_history[0], ActionReconstructionRecord)
    assert action.reconstruction_history[0].after_receipt_event_id is None
    assert action.reconstruction_history[0].payload["assessment"] == (
        "dispatch prevented"
    )


def test_receipt_and_reconstruction_histories_have_visible_fixed_bounds() -> None:
    state = dispatched_state()
    total = MAX_ACTION_RECEIPT_HISTORY + 3
    for index in range(total):
        state = reduce_many(
            state,
            [
                action_event(
                    "action.receipt",
                    {
                        "action_id": "action-1",
                        "status": "unknown",
                        "source_status": "timeout",
                        "source_error_type": f"attempt-{index}",
                        "late": index > 0,
                    },
                ),
                action_event(
                    "action.reconstructed",
                    {"action_id": "action-1", "cycle": index},
                ),
            ],
        )

    action = state.actions["action-1"]
    assert len(action.receipt_history) == MAX_ACTION_RECEIPT_HISTORY
    assert len(action.reconstruction_history) == MAX_ACTION_RECONSTRUCTION_HISTORY
    assert action.receipt_history_evicted == total - MAX_ACTION_RECEIPT_HISTORY
    assert action.reconstruction_history_evicted == (
        total - MAX_ACTION_RECONSTRUCTION_HISTORY
    )
    assert action.receipt_history[0].source_error_type == "attempt-3"
    assert action.reconstruction_history[0].payload["cycle"] == 3
    assert action.phase_history == (
        ActionPhase.PROPOSED,
        ActionPhase.PREPARED,
        ActionPhase.DISPATCHED,
        ActionPhase.RECEIPT,
        ActionPhase.RECONSTRUCTED,
    )


def test_v4_action_payload_replays_with_history_defaults() -> None:
    original = reduce_many(
        state_with_receipt("failure"),
        [action_event("action.reconstructed", {"action_id": "action-1"})],
    ).actions["action-1"]
    legacy = original.model_dump(mode="python")
    for field in (
        "receipt_history",
        "receipt_history_evicted",
        "receipt_corroboration_count",
        "receipt_conflict_count",
        "reconstruction_history",
        "reconstruction_history_evicted",
    ):
        legacy.pop(field)

    replayed = type(original).model_validate(legacy)

    assert replayed.execution_confirmed is True
    assert replayed.outcome is ActionOutcome.FAILURE
    assert replayed.receipt == original.receipt
    assert replayed.reconstruction == original.reconstruction
    assert replayed.receipt_history == ()
    assert replayed.reconstruction_history == ()


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
    assert action.receipt_history[-1].effect_observation_ids == ()
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
    assert action.receipt_history[-1].effect_observation_ids == ("observation-1",)
    assert [item.proposition_id for item in state.world.observed] == ["observation-1"]
