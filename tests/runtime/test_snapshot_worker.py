from __future__ import annotations

import threading
from pathlib import Path

from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.errors import DomainInvariantError, LedgerIntegrityError
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.snapshot import SnapshotWorker
from alice_brain_hermes.runtime.store import SQLiteLedger


def test_snapshot_worker_waits_for_the_per_brain_event_threshold(
    tmp_path: Path,
) -> None:
    brain_id = new_id()
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        worker = SnapshotWorker(ledger, interval_events=4)
        engine = ConsciousEngine(
            ledger,
            brain_id,
            actor_id=brain_id,
            on_state_published=worker.notify,
        )
        worker.register(engine)
        worker.start()
        try:
            for index in range(3):
                engine.append(
                    new_event("opaque.event", brain_id, brain_id, {"index": index})
                )
            worker.wait_idle(timeout=2.0)
            assert ledger.load_snapshot(brain_id) is None

            engine.append(
                new_event("opaque.event", brain_id, brain_id, {"index": 3})
            )
            worker.wait_idle(timeout=2.0)
            snapshot = ledger.load_snapshot(brain_id)
            assert snapshot is not None
            assert snapshot.last_sequence == 4
        finally:
            worker.stop(timeout=2.0)


def test_snapshot_worker_coalesces_hot_publications_to_one_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brain_id = new_id()
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        worker = SnapshotWorker(ledger, interval_events=4)
        engine = ConsciousEngine(
            ledger,
            brain_id,
            actor_id=brain_id,
            on_state_published=worker.notify,
        )
        worker.register(engine)
        calls = 0
        real_checkpoint = ledger.checkpoint_current_state

        def counted_checkpoint(state):
            nonlocal calls
            calls += 1
            return real_checkpoint(state)

        monkeypatch.setattr(ledger, "checkpoint_current_state", counted_checkpoint)
        for index in range(20):
            engine.append(
                new_event("opaque.event", brain_id, brain_id, {"index": index})
            )

        worker.start()
        try:
            worker.wait_idle(timeout=2.0)
            assert calls == 1
            assert worker.health.snapshot_count == 1
            assert ledger.load_snapshot(brain_id).last_sequence == 20  # type: ignore[union-attr]
        finally:
            worker.stop(timeout=2.0)


def test_snapshot_failure_does_not_fail_events_and_retries_next_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brain_id = new_id()
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        worker = SnapshotWorker(ledger, interval_events=4)
        engine = ConsciousEngine(
            ledger,
            brain_id,
            actor_id=brain_id,
            on_state_published=worker.notify,
        )
        worker.register(engine)
        attempts: list[int] = []
        real_checkpoint = ledger.checkpoint_current_state

        def fail_checkpoint(state):
            attempts.append(state.last_sequence)
            if len(attempts) == 1:
                raise OSError("injected snapshot I/O failure")
            return real_checkpoint(state)

        monkeypatch.setattr(ledger, "checkpoint_current_state", fail_checkpoint)
        worker.start()
        try:
            for index in range(4):
                engine.append(
                    new_event("opaque.event", brain_id, brain_id, {"index": index})
                )
            worker.wait_idle(timeout=2.0)
            assert attempts == [4]
            assert worker.health.status == "degraded"

            for index in range(4, 7):
                engine.append(
                    new_event("opaque.event", brain_id, brain_id, {"index": index})
                )
            worker.wait_idle(timeout=2.0)
            assert attempts == [4]

            engine.append(
                new_event("opaque.event", brain_id, brain_id, {"index": 7})
            )
            worker.wait_idle(timeout=2.0)
            assert attempts == [4, 8]
            assert engine.state.last_sequence == 8
            assert len(ledger.list_events(brain_id)) == 8
            assert ledger.load_snapshot(brain_id) == engine.state
            assert worker.health.status == "healthy"
            assert worker.health.last_error_type is None
        finally:
            worker.stop(timeout=2.0)


def test_snapshot_worker_marks_head_mismatch_stale_and_reports_fatal(
    tmp_path: Path,
) -> None:
    brain_id = new_id()
    fatal_errors: list[BaseException] = []
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        worker = SnapshotWorker(
            ledger,
            interval_events=1,
            fatal_error_sink=fatal_errors.append,
        )
        engine = ConsciousEngine(
            ledger,
            brain_id,
            actor_id=brain_id,
            on_state_published=worker.notify,
        )
        worker.register(engine)
        engine.append(new_event("opaque.event", brain_id, brain_id, {"owner": True}))
        ledger.append(new_event("opaque.event", brain_id, brain_id, {"tail": True}))

        worker.start()
        try:
            worker.wait_idle(timeout=2.0)
            assert engine.is_stale is True
            assert ledger.load_snapshot(brain_id) is None
            assert len(fatal_errors) == 1
            assert "current ledger head" in str(fatal_errors[0])
        finally:
            worker.stop(timeout=2.0)


def test_snapshot_worker_never_holds_its_condition_while_waiting_for_engine(
    tmp_path: Path,
) -> None:
    brain_id = new_id()
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        worker = SnapshotWorker(ledger, interval_events=1)
        engine = ConsciousEngine(
            ledger,
            brain_id,
            actor_id=brain_id,
            on_state_published=worker.notify,
        )
        worker.register(engine)
        worker.start()
        writer_done = threading.Event()
        try:
            writer = threading.Thread(
                target=lambda: (
                    engine.append(
                        new_event("opaque.event", brain_id, brain_id, {"safe": True})
                    ),
                    writer_done.set(),
                )
            )
            writer.start()
            writer.join(2.0)
            assert writer_done.is_set()
            worker.wait_idle(timeout=2.0)
        finally:
            worker.stop(timeout=2.0)


def test_snapshot_worker_reports_integrity_failure_from_initial_cache_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brain_id = new_id()
    fatal_errors: list[BaseException] = []
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        engine = ConsciousEngine(ledger, brain_id, actor_id=brain_id)
        worker = SnapshotWorker(
            ledger,
            interval_events=1,
            fatal_error_sink=fatal_errors.append,
        )
        worker.register(engine)

        def fail_load(_brain_id: str):
            raise LedgerIntegrityError("injected initial snapshot integrity failure")

        monkeypatch.setattr(ledger, "load_snapshot", fail_load)
        worker.start()
        try:
            worker.wait_idle(timeout=2.0)
            assert len(fatal_errors) == 1
            assert worker.health.running is True
            assert worker.health.last_error_type == "LedgerIntegrityError"
        finally:
            worker.stop(timeout=2.0)


def test_other_brain_success_does_not_clear_outstanding_snapshot_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_id = new_id()
    second_id = new_id()
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        worker = SnapshotWorker(ledger, interval_events=1)
        first = ConsciousEngine(
            ledger,
            first_id,
            actor_id=first_id,
            on_state_published=worker.notify,
        )
        second = ConsciousEngine(
            ledger,
            second_id,
            actor_id=second_id,
            on_state_published=worker.notify,
        )
        worker.register(first, known_snapshot_sequence=0)
        worker.register(second, known_snapshot_sequence=0)
        real_checkpoint = ledger.checkpoint_current_state

        def fail_first(state):
            if state.brain_id == first_id:
                raise OSError("first brain snapshot unavailable")
            return real_checkpoint(state)

        monkeypatch.setattr(ledger, "checkpoint_current_state", fail_first)
        worker.start()
        try:
            first.append(new_event("opaque.event", first_id, first_id, {}))
            worker.wait_idle(timeout=2.0)
            assert worker.health.last_error_type == "OSError"

            second.append(new_event("opaque.event", second_id, second_id, {}))
            worker.wait_idle(timeout=2.0)
            assert worker.health.status == "degraded"
            assert worker.health.last_error_type == "OSError"
        finally:
            worker.stop(timeout=2.0)


def test_reducer_failure_marks_engine_stale_and_reports_fatal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brain_id = new_id()
    fatal_errors: list[BaseException] = []
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        worker = SnapshotWorker(
            ledger,
            interval_events=1,
            fatal_error_sink=fatal_errors.append,
        )
        engine = ConsciousEngine(
            ledger,
            brain_id,
            actor_id=brain_id,
            on_state_published=worker.notify,
        )
        worker.register(engine, known_snapshot_sequence=0)
        engine.append(new_event("opaque.event", brain_id, brain_id, {}))

        from alice_brain_hermes.runtime import store as store_module

        def fail_reduce(_state, _event):
            raise DomainInvariantError("injected deterministic replay failure")

        monkeypatch.setattr(store_module, "reduce_state", fail_reduce)
        worker.start()
        try:
            worker.wait_idle(timeout=2.0)
            assert engine.is_stale is True
            assert len(fatal_errors) == 1
            assert isinstance(fatal_errors[0], LedgerIntegrityError)
            assert ledger.load_snapshot(brain_id) is None
        finally:
            worker.stop(timeout=2.0)


def test_snapshot_error_type_is_bounded_for_strict_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brain_id = new_id()
    long_error = type("X" * 200, (OSError,), {})
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        worker = SnapshotWorker(ledger, interval_events=1)
        engine = ConsciousEngine(
            ledger,
            brain_id,
            actor_id=brain_id,
            on_state_published=worker.notify,
        )
        worker.register(engine, known_snapshot_sequence=0)

        def fail_checkpoint(_state):
            raise long_error("injected")

        monkeypatch.setattr(
            ledger,
            "checkpoint_current_state",
            fail_checkpoint,
        )
        worker.start()
        try:
            engine.append(new_event("opaque.event", brain_id, brain_id, {}))
            worker.wait_idle(timeout=2.0)
            assert worker.health.last_error_type == "X" * 160
        finally:
            worker.stop(timeout=2.0)
