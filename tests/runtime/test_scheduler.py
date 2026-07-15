from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from alice_brain_hermes.core.events import EventEnvelope, new_event
from alice_brain_hermes.core.state import BrainState
from alice_brain_hermes.core.workspace import derive_candidates
from alice_brain_hermes.errors import DomainInvariantError, EventConflictError
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.scheduler import ContinuousScheduler
from alice_brain_hermes.runtime.store import SQLiteLedger

BRAIN = new_id()
ACTOR = BRAIN


class AppendProbeLedger:
    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []
        self.on_append: Callable[[], None] | None = None
        self.fail_next = False

    def replay(self, brain_id: str) -> BrainState:
        return BrainState.genesis(brain_id)

    def append(self, event: EventEnvelope) -> EventEnvelope:
        if self.fail_next:
            self.fail_next = False
            raise OSError("fixture append failed")
        if self.on_append is not None:
            self.on_append()
        stored = event.model_copy(
            update={"sequence": len(self.events) + 1}
        ).revalidated()
        self.events.append(stored)
        return stored

    def get_event(self, event_id: str) -> EventEnvelope | None:
        return next((item for item in self.events if item.event_id == event_id), None)

    def get_event_and_head(
        self, event_id: str, brain_id: str
    ) -> tuple[EventEnvelope | None, int]:
        return self.get_event(event_id), len(
            [item for item in self.events if item.brain_id == brain_id]
        )

    def append_expected(
        self, event: EventEnvelope, *, expected_sequence: int
    ) -> tuple[EventEnvelope, bool]:
        existing = self.get_event(event.event_id)
        if existing is not None:
            if existing.body_fingerprint() != event.body_fingerprint():
                raise EventConflictError("event ID already has a different body")
            return existing, False
        if expected_sequence != len(self.events) + 1:
            raise EventConflictError("expected sequence does not match")
        return self.append(event), True


class FakeClock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


@pytest.mark.parametrize("timeout", [True, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("operation", ["stop", "join"])
def test_scheduler_shutdown_timeout_must_be_finite_nonnegative(
    timeout: float, operation: str
) -> None:
    engine = ConsciousEngine(AppendProbeLedger(), BRAIN, actor_id=ACTOR)
    scheduler = ContinuousScheduler(engine, monotonic=lambda: 0.0)

    with pytest.raises(ValueError, match="timeout"):
        getattr(scheduler, operation)(timeout=timeout)

    assert scheduler._stop.is_set() is False


def test_engine_appends_before_reducing_and_append_failure_leaves_state() -> None:
    ledger = AppendProbeLedger()
    engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
    state_seen_during_append: list[BrainState] = []
    ledger.on_append = lambda: state_seen_during_append.append(engine.state)
    created = new_event("identity.named", BRAIN, ACTOR, {"name": "Aster"})

    engine.append(created)

    assert state_seen_during_append[0].name is None
    assert engine.state.name == "Aster"
    before = engine.state
    ledger.fail_next = True
    with pytest.raises(OSError, match="append failed"):
        engine.append(new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 1.0}))
    assert engine.state is before


def test_engine_rejects_domain_failure_before_append() -> None:
    ledger = AppendProbeLedger()
    engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)

    with pytest.raises(DomainInvariantError):
        engine.append(
            new_event(
                "action.receipt",
                BRAIN,
                ACTOR,
                {"action_id": "missing", "status": "success"},
                action_id="missing",
            )
        )

    assert ledger.events == []
    assert engine.state.last_sequence == 0


def test_invalid_event_leaves_sqlite_clean_then_valid_append_and_restart_work(
    tmp_path: Path,
) -> None:
    database = tmp_path / "brain.db"
    with SQLiteLedger.open(database) as ledger:
        engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        with pytest.raises(DomainInvariantError):
            engine.append(
                new_event(
                    "action.receipt",
                    BRAIN,
                    ACTOR,
                    {"action_id": "missing", "status": "success"},
                    action_id="missing",
                )
            )

        assert ledger.list_events(BRAIN) == []
        assert engine.state.last_sequence == 0
        stored = engine.append(
            new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 1.25})
        )
        assert stored.sequence == 1
        expected = engine.state

    with SQLiteLedger.open(database) as ledger:
        restarted = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        assert restarted.state == expected
        assert restarted.state.logical_clock == pytest.approx(1.25)


def test_engine_rejects_presequenced_client_event_without_writing() -> None:
    ledger = AppendProbeLedger()
    engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)

    with pytest.raises(ValueError, match="presequenced"):
        engine.append(
            new_event(
                "clock.tick",
                BRAIN,
                ACTOR,
                {"elapsed_seconds": 1.0},
                sequence=1,
            )
        )

    assert ledger.events == []
    assert engine.state.last_sequence == 0


def test_engine_rejects_stale_conflicting_action_atomically_until_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "brain.db"
    with SQLiteLedger.open(database) as engine_ledger:
        engine = ConsciousEngine(engine_ledger, BRAIN, actor_id=ACTOR)
        concurrent = new_event(
            "action.proposed",
            BRAIN,
            ACTOR,
            {"action_id": "a1", "intent": {"writer": "concurrent"}},
            action_id="a1",
        )
        engine_ledger.append(concurrent)

        with pytest.raises(EventConflictError, match="sequence divergence"):
            engine.append(
                new_event(
                    "action.proposed",
                    BRAIN,
                    ACTOR,
                    {"action_id": "a1", "intent": {"writer": "stale"}},
                    action_id="a1",
                )
            )
        assert engine.state.last_sequence == 0
        assert engine.is_stale is True
        assert [item.event_id for item in engine_ledger.list_events(BRAIN)] == [
            concurrent.event_id
        ]

        with pytest.raises(EventConflictError, match="restart"):
            engine.append(
                new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 4.0})
            )
        assert len(engine_ledger.list_events(BRAIN)) == 1

    with SQLiteLedger.open(database) as ledger:
        restarted = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        assert restarted.state.last_sequence == 1
        assert restarted.state.actions["a1"].intent == {"writer": "concurrent"}


def test_engine_uses_one_authoritative_head_view_before_event_semantics(
    tmp_path: Path,
) -> None:
    database = tmp_path / "brain.db"
    with SQLiteLedger.open(database) as engine_ledger:
        engine = ConsciousEngine(engine_ledger, BRAIN, actor_id=ACTOR)
        future = new_event(
            "action.proposed",
            BRAIN,
            ACTOR,
            {"action_id": "future", "intent": {"writer": "concurrent"}},
            action_id="future",
        )
        engine_ledger.append(future)

        changed_body = future.model_copy(
            update={
                "payload": {
                    "action_id": "future",
                    "intent": {"writer": "stale-engine"},
                }
            }
        ).revalidated()
        with pytest.raises(EventConflictError, match="sequence divergence"):
            engine.append(changed_body)

        assert engine.is_stale is True
        assert engine.state.last_sequence == 0

    followup_database = tmp_path / "followup.db"
    with SQLiteLedger.open(followup_database) as engine_ledger:
        engine = ConsciousEngine(engine_ledger, BRAIN, actor_id=ACTOR)
        engine_ledger.append(
            new_event(
                "action.proposed",
                BRAIN,
                ACTOR,
                {"action_id": "precursor", "intent": {}},
                action_id="precursor",
            )
        )

        with pytest.raises(EventConflictError, match="sequence divergence"):
            engine.append(
                new_event(
                    "action.prepared",
                    BRAIN,
                    ACTOR,
                    {"action_id": "precursor", "branch_id": "b1"},
                    action_id="precursor",
                )
            )

        assert engine.is_stale is True


def test_engine_rejects_old_exact_retry_when_an_unseen_tail_exists(
    tmp_path: Path,
) -> None:
    database = tmp_path / "brain.db"
    with SQLiteLedger.open(database) as engine_ledger:
        engine = ConsciousEngine(engine_ledger, BRAIN, actor_id=ACTOR)
        original = new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 1.0})
        stored = engine.append(original)
        engine_ledger.append(new_event("opaque.tail", BRAIN, ACTOR, {}))

        with pytest.raises(EventConflictError, match="sequence divergence"):
            engine.append(original)

        assert stored.sequence == 1
        assert engine.state.last_sequence == 1
        assert engine.is_stale is True


@pytest.mark.parametrize("with_tail", [False, True])
def test_engine_get_miss_race_adopts_exact_next_event_only_without_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_tail: bool
) -> None:
    database = tmp_path / f"race-{with_tail}.db"
    with SQLiteLedger.open(database) as engine_ledger:
        engine = ConsciousEngine(engine_ledger, BRAIN, actor_id=ACTOR)
        target = new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 1.0})
        original_read = engine_ledger.get_event_and_head
        raced = False

        def read_then_race(
            event_id: str, brain_id: str
        ) -> tuple[EventEnvelope | None, int]:
            nonlocal raced
            result = original_read(event_id, brain_id)
            if not raced:
                raced = True
                engine_ledger.append(target)
                if with_tail:
                    engine_ledger.append(new_event("opaque.tail", BRAIN, ACTOR, {}))
            return result

        monkeypatch.setattr(engine_ledger, "get_event_and_head", read_then_race)

        if with_tail:
            with pytest.raises(EventConflictError, match="sequence divergence"):
                engine.append(target)
            assert engine.state.last_sequence == 0
            assert engine.is_stale is True
        else:
            stored = engine.append(target)
            assert stored.sequence == 1
            assert engine.state.last_sequence == 1
            assert engine.state.logical_clock == pytest.approx(1.0)
            assert engine.is_stale is False


def test_engine_get_miss_race_marks_changed_body_in_target_brain_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "changed-body-target-race.db"
    with SQLiteLedger.open(database) as engine_ledger:
        engine = ConsciousEngine(engine_ledger, BRAIN, actor_id=ACTOR)
        target = new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 1.0})
        conflicting = target.model_copy(
            update={"payload": {"elapsed_seconds": 2.0}}
        ).revalidated()
        original_read = engine_ledger.get_event_and_head
        raced = False

        def read_then_race(
            event_id: str, brain_id: str
        ) -> tuple[EventEnvelope | None, int]:
            nonlocal raced
            result = original_read(event_id, brain_id)
            if not raced:
                raced = True
                engine_ledger.append(conflicting)
            return result

        monkeypatch.setattr(engine_ledger, "get_event_and_head", read_then_race)

        with pytest.raises(EventConflictError, match="sequence divergence"):
            engine.append(target)

        assert engine.state.last_sequence == 0
        assert engine.is_stale is True
        assert engine_ledger.list_events(BRAIN) == [
            conflicting.model_copy(update={"sequence": 1}).revalidated()
        ]


def test_engine_get_miss_race_with_changed_body_in_other_brain_is_not_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "changed-body-other-brain-race.db"
    other_brain = new_id()
    with SQLiteLedger.open(database) as engine_ledger:
        engine = ConsciousEngine(engine_ledger, BRAIN, actor_id=ACTOR)
        target = new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 1.0})
        conflicting = target.model_copy(
            update={
                "brain_id": other_brain,
                "payload": {"elapsed_seconds": 2.0},
            }
        ).revalidated()
        original_read = engine_ledger.get_event_and_head
        raced = False

        def read_then_race(
            event_id: str, brain_id: str
        ) -> tuple[EventEnvelope | None, int]:
            nonlocal raced
            result = original_read(event_id, brain_id)
            if not raced:
                raced = True
                engine_ledger.append(conflicting)
            return result

        monkeypatch.setattr(engine_ledger, "get_event_and_head", read_then_race)

        with pytest.raises(EventConflictError, match="different body"):
            engine.append(target)

        assert engine.state.last_sequence == 0
        assert engine.is_stale is False
        assert engine_ledger.list_events(BRAIN) == []
        assert len(engine_ledger.list_events(other_brain)) == 1


def test_engine_exact_retries_return_original_without_reduction_or_state_change(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "brain.db") as ledger:
        engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        tick = new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 1.0})
        stored_tick = engine.append(tick)
        after_tick = engine.state

        assert engine.append(tick) == stored_tick
        assert engine.state is after_tick

        proposal = new_event(
            "action.proposed",
            BRAIN,
            ACTOR,
            {"action_id": "retry-action", "intent": {"operation": "inspect"}},
            action_id="retry-action",
        )
        stored_proposal = engine.append(proposal)
        after_proposal = engine.state

        assert engine.append(proposal) == stored_proposal
        assert engine.state is after_proposal
        later = engine.append(
            new_event(
                "action.prepared",
                BRAIN,
                ACTOR,
                {"action_id": "retry-action", "branch_id": "b1"},
                action_id="retry-action",
            )
        )
        assert later.sequence == 3
        assert engine.is_stale is False


def test_pc_rate_budget_replays_identically_after_restart(tmp_path: Path) -> None:
    database = tmp_path / "brain.db"
    with SQLiteLedger.open(database) as ledger:
        engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        engine.append(
            new_event(
                "personality.revised",
                BRAIN,
                ACTOR,
                {"layer": "traits", "values": {"care": 0.05}},
            )
        )
        engine.append(new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 0.5}))
        engine.append(
            new_event(
                "personality.revised",
                BRAIN,
                ACTOR,
                {"layer": "traits", "values": {"care": 0.075}},
            )
        )
        expected = engine.state
        assert expected.personality.rate_state.traits.available == pytest.approx(0.0)

    with SQLiteLedger.open(database) as ledger:
        restarted = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        assert restarted.state == expected
        assert restarted.state.personality.rate_state == expected.personality.rate_state


def test_energy_activation_and_c0_drive_are_stable_across_canonical_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "brain.db"
    deficit_items = [
        item
        for index in range(50)
        for item in ((f"a-{index:02d}", 0.1), (f"z-{index:02d}", 0.9))
    ]
    deficits = dict(deficit_items)
    with SQLiteLedger.open(database) as ledger:
        engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        engine.append(
            new_event(
                "action.proposed",
                BRAIN,
                ACTOR,
                {"action_id": "stable-energy", "intent": {}},
                action_id="stable-energy",
            )
        )
        engine.append(
            new_event(
                "action.energy_assessed",
                BRAIN,
                ACTOR,
                {
                    "action_id": "stable-energy",
                    "deficits": deficits,
                    "salience": 0.0,
                    "urgency": 0.0,
                    "valence": 0.0,
                    "arousal": 0.0,
                    "control": 0.0,
                    "resources": 0.0,
                    "cost": 5.49999e-11,
                    "personality_relevance": 0.0,
                },
                action_id="stable-energy",
            )
        )
        live_activation = engine.state.energies["stable-energy"].activation
        live_drive = next(
            item
            for item in derive_candidates(engine.state)
            if item.specialist == "drives"
        ).score

    with SQLiteLedger.open(database) as ledger:
        restarted = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        replayed_activation = restarted.state.energies["stable-energy"].activation
        replayed_drive = next(
            item
            for item in derive_candidates(restarted.state)
            if item.specialist == "drives"
        ).score

    assert live_activation == replayed_activation == 0.114999999995
    assert live_drive == replayed_drive == 0.114999999995


def test_off_turn_pulse_records_clock_workspace_and_local_cognition(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "brain.db") as ledger:
        engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)

        engine.pulse(2.5)

        event_types = [item.event_type for item in ledger.list_events(BRAIN, limit=50)]
        assert event_types[0] == "clock.tick"
        assert "workspace.broadcast" in event_types
        assert "cognition.reflected" in event_types
        assert engine.state.logical_clock == 2.5
        assert engine.state.cognition.reflections
        assert engine.state.cognition.cognition_mode == "local"
        reflection = engine.state.cognition.reflections[-1]
        assert reflection.uncertainty_basis == "deterministic_heuristic"
        assert reflection.calibrated is False
        cognition_event = next(
            item
            for item in ledger.list_events(BRAIN, limit=50)
            if item.event_type == "cognition.reflected"
        )
        assert cognition_event.payload["uncertainty_basis"] == (
            "deterministic_heuristic"
        )
        assert cognition_event.payload["calibrated"] is False


def test_scheduler_uses_real_elapsed_and_never_synthesizes_catchup_ticks(
    tmp_path: Path,
) -> None:
    clock = FakeClock(100.0, 107.75, 130.0)
    with SQLiteLedger.open(tmp_path / "brain.db") as ledger:
        engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        scheduler = ContinuousScheduler(
            engine,
            interval_seconds=1.0,
            monotonic=clock,
            sleeper=lambda _: None,
        )

        assert scheduler.step() is True
        assert scheduler.step() is True

        ticks = [
            item
            for item in ledger.list_events(BRAIN, limit=100)
            if item.event_type == "clock.tick"
        ]
        assert len(ticks) == 2
        assert [item.payload["elapsed_seconds"] for item in ticks] == [7.75, 22.25]
        assert engine.state.logical_clock == 30.0


def test_invalid_clock_samples_do_not_terminate_or_corrupt_valid_baseline(
    tmp_path: Path,
) -> None:
    class ScriptedClock:
        def __init__(self, *values: object) -> None:
            self.values = iter(values)

        def __call__(self) -> object:
            value = next(self.values)
            if isinstance(value, BaseException):
                raise value
            return value

    clock = ScriptedClock(
        100.0,
        90.0,
        float("nan"),
        float("inf"),
        "not-a-number",
        RuntimeError("clock unavailable"),
        101.0,
    )
    with SQLiteLedger.open(tmp_path / "brain.db") as ledger:
        engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        scheduler = ContinuousScheduler(
            engine,
            interval_seconds=1.0,
            monotonic=clock,  # type: ignore[arg-type]
            sleeper=lambda _: None,
        )

        assert [scheduler.step() for _ in range(5)] == [False] * 5
        assert scheduler.step() is True

        ticks = [
            item
            for item in ledger.list_events(BRAIN, limit=100)
            if item.event_type == "clock.tick"
        ]
        assert [item.payload["elapsed_seconds"] for item in ticks] == [1.0]
        assert engine.state.logical_clock == pytest.approx(1.0)
        assert engine.state.runtime.failure_count == 5
        assert engine.state.runtime.health == "healthy"


def test_invalid_energy_types_and_ranges_leave_ledger_clean(tmp_path: Path) -> None:
    base_payload: dict[str, object] = {
        "action_id": "energy-action",
        "deficits": {"need": 0.2},
        "salience": 0.5,
        "urgency": 0.5,
        "valence": 0.0,
        "arousal": 0.0,
        "control": 0.5,
        "resources": 0.5,
        "cost": 0.2,
        "personality_relevance": 0.4,
    }
    invalid_values: tuple[object, ...] = (
        True,
        "0.5",
        float("nan"),
        float("inf"),
        -0.01,
        1.01,
    )
    with SQLiteLedger.open(tmp_path / "brain.db") as ledger:
        engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        engine.append(
            new_event(
                "action.proposed",
                BRAIN,
                ACTOR,
                {"action_id": "energy-action", "intent": {}},
                action_id="energy-action",
            )
        )
        baseline = engine.state
        count = len(ledger.list_events(BRAIN))

        for invalid in invalid_values:
            candidate = new_event(
                "action.energy_assessed",
                BRAIN,
                ACTOR,
                base_payload,
                action_id="energy-action",
            ).model_copy(update={"payload": {**base_payload, "urgency": invalid}})
            with pytest.raises((DomainInvariantError, ValueError)):
                engine.append(candidate)
            assert engine.state is baseline
            assert len(ledger.list_events(BRAIN)) == count


def test_ambiguous_or_mismatched_receipt_evidence_is_not_persisted(
    tmp_path: Path,
) -> None:
    with SQLiteLedger.open(tmp_path / "brain.db") as ledger:
        engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        action_id = "receipt-action"
        for event_type, payload in (
            ("action.proposed", {"action_id": action_id, "intent": {}}),
            ("action.prepared", {"action_id": action_id, "branch_id": "b1"}),
            ("action.dispatched", {"action_id": action_id}),
        ):
            engine.append(
                new_event(
                    event_type,
                    BRAIN,
                    ACTOR,
                    payload,
                    action_id=action_id,
                )
            )
        baseline = engine.state
        count = len(ledger.list_events(BRAIN))
        invalid_payloads = (
            {
                "effect_evidence": {
                    "kind": "linked_observation",
                    "observation_ids": ["o1", "o1"],
                },
                "observations": [{"proposition_id": "o1", "content": {"ok": True}}],
            },
            {
                "effect_evidence": {
                    "kind": "linked_observation",
                    "observation_ids": ["o1"],
                },
                "observations": [
                    {"proposition_id": "o1", "content": {"version": 1}},
                    {"proposition_id": "o1", "content": {"version": 2}},
                ],
            },
            {
                "effect_evidence": {
                    "kind": "linked_observation",
                    "observation_ids": ["missing"],
                },
                "observations": [{"proposition_id": "o1", "content": {"ok": True}}],
            },
            {
                "effect_evidence": {
                    "kind": "linked_observation",
                    "observation_ids": ["o1"],
                },
                "observations": [
                    {
                        "proposition_id": "o1",
                        "content": {"ok": True},
                        "confidence": "0.9",
                    }
                ],
            },
        )

        for extra in invalid_payloads:
            receipt = new_event(
                "action.receipt",
                BRAIN,
                ACTOR,
                {"action_id": action_id, "status": "success", **extra},
                action_id=action_id,
            )
            with pytest.raises(
                DomainInvariantError, match=r"evidence|observation|proposition"
            ):
                engine.append(receipt)
            assert engine.state is baseline
            assert len(ledger.list_events(BRAIN)) == count


def test_scheduler_records_sanitized_failure_and_continues_future_ticks(
    tmp_path: Path,
) -> None:
    class RaisingCoordinator:
        def propose(self, state: BrainState):
            raise RuntimeError("coordinator exploded\nsecret details")

    clock = FakeClock(0.0, 1.0, 2.0)
    with SQLiteLedger.open(tmp_path / "brain.db") as ledger:
        engine = ConsciousEngine(
            ledger,
            BRAIN,
            actor_id=ACTOR,
            coordinator=RaisingCoordinator(),
        )
        scheduler = ContinuousScheduler(
            engine,
            interval_seconds=1.0,
            monotonic=clock,
            sleeper=lambda _: None,
        )

        assert scheduler.step() is False
        assert scheduler.health.status == "degraded"
        failure = engine.state.runtime.last_failure
        assert failure is not None
        assert failure.error_type == "RuntimeError"
        assert "\n" not in failure.message

        engine.coordinator = None
        assert scheduler.step() is True
        assert scheduler.health.status == "healthy"
        assert engine.state.runtime.failure_count == 1
        assert engine.state.runtime.tick_count == 2


def test_failure_append_failure_degrades_volatile_health_without_recursion() -> None:
    class AlwaysFailLedger(AppendProbeLedger):
        def append(self, event: EventEnvelope) -> EventEnvelope:
            raise OSError("database unavailable")

    ledger = AlwaysFailLedger()
    engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
    scheduler = ContinuousScheduler(
        engine,
        interval_seconds=1.0,
        monotonic=FakeClock(0.0, 1.0),
        sleeper=lambda _: None,
    )

    assert scheduler.step() is False
    assert scheduler.health.status == "degraded"
    assert scheduler.health.failure_event_persisted is False
    assert scheduler.health.last_error_type == "OSError"
    assert engine.state == BrainState.genesis(BRAIN)


def test_run_advances_c0_without_turns_or_provider_calls(tmp_path: Path) -> None:
    clock = FakeClock(0.0, 1.0, 2.0, 3.0)
    sleeps: list[float] = []
    with SQLiteLedger.open(tmp_path / "brain.db") as ledger:
        engine = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        scheduler = ContinuousScheduler(
            engine,
            interval_seconds=1.0,
            monotonic=clock,
            sleeper=sleeps.append,
        )

        scheduler.run(max_ticks=3)

        assert sleeps == [1.0, 1.0, 1.0]
        assert engine.state.runtime.tick_count == 3
        assert engine.state.logical_clock == 3.0
        assert engine.state.cognition.cognition_mode == "local"
        assert engine.state.capabilities.get("provider_fallback") is None


def test_restart_replays_persisted_derived_events_without_rerunning_algorithms(
    tmp_path: Path,
) -> None:
    database = tmp_path / "brain.db"
    with SQLiteLedger.open(database) as ledger:
        first = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        first.pulse(1.25)
        snapshot = first.state
        count = len(ledger.list_events(BRAIN, limit=100))

    class MustNotRun:
        def propose(self, state: BrainState):
            raise AssertionError("replay reran current coordinator")

    with SQLiteLedger.open(database) as ledger:
        restarted = ConsciousEngine(
            ledger,
            BRAIN,
            actor_id=ACTOR,
            coordinator=MustNotRun(),
        )

        assert restarted.state == snapshot
        assert len(ledger.list_events(BRAIN, limit=100)) == count
