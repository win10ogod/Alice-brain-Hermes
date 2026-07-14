from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from alice_brain_hermes.core.events import EventEnvelope, new_event
from alice_brain_hermes.core.state import BrainState
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


class FakeClock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


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


def test_engine_surfaces_sequence_divergence_and_fails_closed_until_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "brain.db"
    with SQLiteLedger.open(database) as engine_ledger:
        engine = ConsciousEngine(engine_ledger, BRAIN, actor_id=ACTOR)
        with SQLiteLedger.open(database) as concurrent_ledger:
            concurrent_ledger.append(
                new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 1.0})
            )

        with pytest.raises(EventConflictError, match="sequence divergence"):
            engine.append(
                new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 2.0})
            )
        assert engine.state.last_sequence == 0
        assert len(engine_ledger.list_events(BRAIN)) == 2

        with pytest.raises(EventConflictError, match="restart"):
            engine.append(
                new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 4.0})
            )
        assert len(engine_ledger.list_events(BRAIN)) == 2

    with SQLiteLedger.open(database) as ledger:
        restarted = ConsciousEngine(ledger, BRAIN, actor_id=ACTOR)
        assert restarted.state.last_sequence == 2
        assert restarted.state.logical_clock == pytest.approx(3.0)


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
        engine.append(
            new_event("clock.tick", BRAIN, ACTOR, {"elapsed_seconds": 0.5})
        )
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
