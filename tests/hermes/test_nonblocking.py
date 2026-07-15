from __future__ import annotations

import json
import os
import select
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from alice_brain_hermes.errors import DaemonClientError, DaemonRpcError
from alice_brain_hermes.hermes.bridge import HookBridge
from alice_brain_hermes.hermes.hooks import HermesHooks
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.projections import MAX_EPHEMERAL_CONTEXT_BYTES
from alice_brain_hermes.protocol.client import DaemonClient
from alice_brain_hermes.protocol.models import (
    BridgeCommitAckV1,
    BridgeGapV1,
    validate_bridge_record_json,
)


def _session_start(hooks: HermesHooks, session_id: str = "session") -> None:
    assert hooks.on_session_start(
        telemetry_schema_version="hermes.observer.v1",
        session_id=session_id,
        model="model",
        platform="cli",
    ) is None


def _wait_until(predicate: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def test_callback_thread_never_touches_transport_sqlite_or_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_thread = threading.get_ident()
    transport_threads: list[int] = []

    def blocked_connect(*_args: Any, **_kwargs: Any) -> Any:
        transport_threads.append(threading.get_ident())
        raise DaemonClientError("daemon is not running")

    def callback_io(label: str) -> Any:
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            if threading.get_ident() == callback_thread:
                raise AssertionError(f"callback touched {label}")
            raise DaemonClientError("worker probe blocked")

        return fail

    monkeypatch.setattr(socket, "create_connection", callback_io("socket"))
    monkeypatch.setattr(sqlite3, "connect", callback_io("sqlite"))
    monkeypatch.setattr(subprocess, "Popen", callback_io("process"))
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        client_factory=blocked_connect,
    )
    hooks = HermesHooks(bridge)

    started = time.perf_counter()
    _session_start(hooks)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05
    _wait_until(lambda: bool(transport_threads))
    assert callback_thread not in transport_threads
    assert bridge.worker_started is True
    bridge.stop_worker_for_test()


def test_missing_daemon_does_not_auto_start_or_create_runtime_home(
    tmp_path: Path,
) -> None:
    runtime_home = tmp_path / "must-not-be-created"
    attempts = 0

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise DaemonClientError("no discovery")

    bridge = HookBridge(
        runtime_home,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        client_factory=unavailable,
    )
    hooks = HermesHooks(bridge)

    _session_start(hooks)
    _wait_until(lambda: attempts > 0)

    assert not runtime_home.exists()
    assert bridge.health.connection == "disconnected"
    assert bridge.queue.qsize() == 0
    assert bridge.retained_record is not None
    assert bridge.pending_gaps() == ()
    bridge.stop_worker_for_test()


class _FakeClient:
    def __init__(self, calls: list[tuple[str, dict[str, object]]]) -> None:
        self.calls = calls
        self.closed = False
        self._binding = new_id()
        self._brain_id = new_id()
        self._next_capture_seq = 1
        self._through_capture_seq = 0

    def call(
        self,
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        value = dict(params or {})
        self.calls.append((method, value))
        if method == "brain.resolve":
            return {"brain_id": self._brain_id, "state_sequence": 0, "created": True}
        if method == "brain.attach":
            return {
                "binding": self._binding,
                "brain_id": self._brain_id,
                "next_capture_seq": self._next_capture_seq,
            }
        if method == "bridge.commit":
            record = value["record"]
            assert isinstance(record, dict)
            validated = validate_bridge_record_json(
                json.dumps(record, separators=(",", ":"), sort_keys=True)
            )
            through = int(
                record.get("capture_seq", record.get("last_capture_seq", 0))
            )
            self._through_capture_seq = through
            self._next_capture_seq = through + 1
            frame = _frame(self._brain_id, through)
            freshness = frame["freshness"]
            assert isinstance(freshness, dict)
            freshness["scheduler_sample"] = "not_sampled"
            return {
                "schema_version": 1,
                "record_fingerprint": validated.fingerprint(),
                "duplicate": False,
                "event_id": new_id(),
                "event_sequence": through,
                "frame": frame,
                "through_capture_seq": through,
            }
        if method == "state.get":
            return _frame(self._brain_id, self._through_capture_seq)
        raise AssertionError(method)

    def close(self) -> None:
        self.closed = True


def _frame(brain_id: str, through: int) -> dict[str, object]:
    return {
        "schema_version": 2,
        "brain_id": brain_id,
        "state_sequence": through,
        "through_capture_seq": through,
        "logical_clock": float(through),
        "trace_complete": through == 0,
        "runtime_health": "healthy" if through == 0 else "degraded",
        "c0_tick": 0,
        "pc": {},
        "energy": {},
        "st": {},
        "rd": {},
        "a": {},
        "world": {},
        "self_boundary": {},
        "memory": {},
        "capabilities": {
            "chunk_capture": "unobserved",
            "reasoning_capture": "unobserved",
        },
        "semantic_context": {},
        "unresolved_evidence": False,
        "capture_coverage": {},
        "freshness": {
            "projected_at_state_sequence": through,
            "scheduler_tick": 0,
            "scheduler_sample": "running",
            "stream_connection": "connected",
        },
        "omission_counts": {},
    }


def test_worker_commits_exact_gap_before_later_record(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    client = _FakeClient(calls)
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        queue_capacity=1,
        reconnect_delay_seconds=0.01,
        client_factory=lambda *_args, **_kwargs: client,
        start_worker_on_capture=False,
    )
    hooks = HermesHooks(bridge)

    _session_start(hooks, "first")
    _session_start(hooks, "dropped")
    bridge.start_worker()
    _wait_until(bridge.queue.empty)
    _session_start(hooks, "later")
    _wait_until(
        lambda: len([call for call in calls if call[0] == "bridge.commit"]) >= 3
    )
    bridge.stop_worker_for_test()

    commits = [
        params["record"] for method, params in calls if method == "bridge.commit"
    ]
    assert [record["record_kind"] for record in commits] == [
        "observation",
        "gap",
        "observation",
    ]
    assert commits[0]["capture_seq"] == 1
    assert commits[1]["first_capture_seq"] == 2
    assert commits[1]["last_capture_seq"] == 2
    assert commits[1]["cause_counts"] == {"queue_full": 1}
    assert commits[2]["capture_seq"] == 3
    assert len([call for call in calls if call[0] == "state.get"]) == 1


def test_reserved_bootstrap_handoff_preserves_cursor_and_multicause_gap(
    tmp_path: Path,
) -> None:
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        start_worker_on_capture=False,
    )
    bridge.capture_reserved(
        hook="on_session_start",
        detached_kwargs={
            "telemetry_schema_version": "hermes.observer.v1",
            "session_id": "session",
            "model": "model",
            "platform": "cli",
        },
        first_capture_seq=1,
        last_capture_seq=1,
        gap_cause_counts=None,
        copy_stats={"truncated_paths": 1},
    )
    bridge.capture_reserved(
        hook=None,
        detached_kwargs=None,
        first_capture_seq=2,
        last_capture_seq=3,
        gap_cause_counts={
            "callback_internal": 1,
            "invalid_source_schema": 1,
        },
        copy_stats={},
    )

    observation = bridge.queue.get_nowait()
    assert observation.capture_seq == 1
    assert observation.coverage.capture_coverage == "partial"
    (gap,) = bridge.pending_gaps()
    assert (gap.first_capture_seq, gap.last_capture_seq) == (2, 3)
    assert dict(gap.cause_counts) == {
        "callback_internal": 1,
        "invalid_source_schema": 1,
    }
    assert bridge.health.last_capture_seq == 3
    assert bridge.health.dropped_events == 2


def test_transport_outage_retains_head_without_fabricating_gap(
    tmp_path: Path,
) -> None:
    attempts = 0
    calls: list[tuple[str, dict[str, object]]] = []
    working = _FakeClient(calls)

    def connect(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DaemonClientError("temporary outage")
        return working

    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        client_factory=connect,
        start_worker_on_capture=False,
    )
    hooks = HermesHooks(bridge)
    _session_start(hooks)
    bridge.start_worker()

    _wait_until(lambda: any(method == "bridge.commit" for method, _ in calls))
    bridge.stop_worker_for_test()

    commits = [
        params["record"] for method, params in calls if method == "bridge.commit"
    ]
    assert len(commits) == 1
    assert commits[0]["capture_seq"] == 1
    assert bridge.pending_gaps() == ()
    assert bridge.health.dropped_events == 0


def test_lost_ack_retries_identical_record_and_accepts_duplicate_false(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    first = _FakeClient(calls)
    second = _FakeClient(calls)
    second._brain_id = first._brain_id
    persisted_ack: dict[str, object] | None = None

    original_first_call = first.call
    original_second_call = second.call

    def first_call(
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        nonlocal persisted_ack
        if method == "bridge.commit":
            persisted_ack = original_first_call(method, params)
            second._next_capture_seq = first._next_capture_seq
            second._through_capture_seq = first._through_capture_seq
            raise DaemonClientError("ack lost after persistence")
        return original_first_call(method, params)

    def second_call(
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if method == "bridge.commit":
            calls.append((method, dict(params or {})))
            assert persisted_ack is not None
            return persisted_ack
        result = original_second_call(method, params)
        if method == "state.get":
            result["state_sequence"] = 2
            result["logical_clock"] = 2.0
            freshness = result["freshness"]
            assert isinstance(freshness, dict)
            freshness["projected_at_state_sequence"] = 2
        return result

    first.call = first_call  # type: ignore[method-assign]
    second.call = second_call  # type: ignore[method-assign]
    clients = iter((first, second))
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        client_factory=lambda *_args, **_kwargs: next(clients),
        start_worker_on_capture=False,
    )
    hooks = HermesHooks(bridge)
    _session_start(hooks)
    bridge.start_worker()

    _wait_until(
        lambda: len([call for call in calls if call[0] == "bridge.commit"]) == 2
    )
    _wait_until(lambda: bridge.retained_record is None)
    bridge.stop_worker_for_test()

    sent = [params["record"] for method, params in calls if method == "bridge.commit"]
    assert sent[0] == sent[1]
    attachments = [
        params for method, params in calls if method == "brain.attach"
    ]
    assert len(attachments) == 2
    assert attachments[0]["bridge_instance_id"] == attachments[1][
        "bridge_instance_id"
    ]
    assert attachments[0]["recovery_token"] == attachments[1]["recovery_token"]
    assert bridge.retained_record is None
    assert isinstance(bridge.last_ack, BridgeCommitAckV1)
    assert bridge.last_ack.duplicate is False
    assert bridge.projections.frame is not None
    assert bridge.projections.frame.state_sequence == 2


def test_recovery_token_is_stable_until_typed_abandonment(tmp_path: Path) -> None:
    attempts: list[tuple[str, str]] = []

    class AbandonOnce(_FakeClient):
        def call(
            self,
            method: str,
            params: dict[str, object] | None = None,
        ) -> dict[str, object]:
            if method == "brain.attach":
                assert params is not None
                attempts.append(
                    (
                        str(params["bridge_instance_id"]),
                        str(params["recovery_token"]),
                    )
                )
                if len(attempts) == 1:
                    raise DaemonRpcError("bridge_abandoned", "abandoned", {})
            return super().call(method, params)

    calls: list[tuple[str, dict[str, object]]] = []
    first_id = new_id()
    client = AbandonOnce(calls)
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=first_id,
        reconnect_delay_seconds=0.01,
        client_factory=lambda *_args, **_kwargs: client,
        start_worker_on_capture=False,
    )
    first_token = bridge.recovery_token
    hooks = HermesHooks(bridge)
    _session_start(hooks)
    bridge.start_worker()

    _wait_until(lambda: len(attempts) >= 2)
    bridge.stop_worker_for_test()

    assert attempts[0] == (first_id, first_token)
    assert attempts[1][0] != first_id
    assert attempts[1][1] != first_token
    assert len(attempts[1][1]) == 64
    assert bridge.bridge_instance_id == attempts[1][0]
    assert bridge.recovery_token == attempts[1][1]
    assert bridge.health.abandoned_streams == 1
    assert bridge.health.abandoned_local_records == 1
    assert bridge.health.ambiguous_records == 0
    assert bridge.health.dropped_events == 1
    assert bridge.health.pending_gap_ranges == 0
    assert bridge.health.trace_complete is False
    assert bridge.health.last_abandonment is not None
    assert bridge.health.last_abandonment["exact_replay_permitted"] is False
    assert not any(method == "bridge.commit" for method, _params in calls)


def test_typed_abandonment_accounts_possibly_acked_record_without_replay(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    first = _FakeClient(calls)
    second = _FakeClient(calls)
    third = _FakeClient(calls)
    second._brain_id = first._brain_id
    third._brain_id = first._brain_id
    first_call = first.call
    second_call = second.call

    def lose_commit_ack(
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if method == "bridge.commit":
            first_call(method, params)
            raise DaemonClientError("ACK lost after persistence")
        return first_call(method, params)

    def report_abandoned(
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if method == "brain.attach":
            calls.append((method, dict(params or {})))
            raise DaemonRpcError("bridge_abandoned", "abandoned", {})
        return second_call(method, params)

    first.call = lose_commit_ack  # type: ignore[method-assign]
    second.call = report_abandoned  # type: ignore[method-assign]
    clients = iter((first, second, third))
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        client_factory=lambda *_args, **_kwargs: next(clients),
    )
    hooks = HermesHooks(bridge)
    _session_start(hooks)

    _wait_until(lambda: bridge.health.abandoned_streams == 1)
    bridge.stop_worker_for_test()

    commits = [params for method, params in calls if method == "bridge.commit"]
    assert len(commits) == 1
    assert bridge.health.ambiguous_records == 1
    assert bridge.health.abandoned_local_records == 1
    assert bridge.health.dropped_events == 1
    assert bridge.health.trace_complete is False
    assert bridge.health.last_abandonment is not None
    assert bridge.health.last_abandonment["ambiguous_capture_count"] == 1


def test_abandonment_rebind_clears_stream_cursor_without_regressing_brain(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class AbandonOnRefresh(_FakeClient):
        state_reads = 0

        def call(
            self,
            method: str,
            params: dict[str, object] | None = None,
        ) -> dict[str, object]:
            if method == "state.get":
                self.state_reads += 1
                if self.state_reads > 1:
                    self.calls.append((method, dict(params or {})))
                    raise DaemonRpcError("bridge_abandoned", "abandoned", {})
            return super().call(method, params)

    class CurrentBrainClient(_FakeClient):
        def call(
            self,
            method: str,
            params: dict[str, object] | None = None,
        ) -> dict[str, object]:
            if method == "brain.resolve":
                self.calls.append((method, dict(params or {})))
                return {
                    "brain_id": self._brain_id,
                    "state_sequence": 1,
                    "created": False,
                }
            result = super().call(method, params)
            if method == "state.get":
                result["state_sequence"] = 1
                result["logical_clock"] = 1.0
                freshness = result["freshness"]
                assert isinstance(freshness, dict)
                freshness["projected_at_state_sequence"] = 1
            return result

    first = AbandonOnRefresh(calls)
    regressed = _FakeClient(calls)
    current = CurrentBrainClient(calls)
    regressed._brain_id = first._brain_id
    current._brain_id = first._brain_id
    clients = iter((first, regressed, current))
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        frame_refresh_seconds=0.02,
        client_factory=lambda *_args, **_kwargs: next(clients),
    )
    hooks = HermesHooks(bridge)
    _session_start(hooks)

    _wait_until(
        lambda: bridge.health.abandoned_streams == 1
        and bridge.health.connection == "connected"
        and bridge.projections.frame is not None
        and bridge.projections.frame.through_capture_seq == 0
    )
    assert regressed.closed is True
    assert bridge.projections.frame is not None
    assert bridge.projections.frame.brain_id == first._brain_id
    assert bridge.projections.frame.state_sequence == 1
    assert bridge.health.abandoned_local_records == 0
    assert bridge.health.trace_complete is False
    bridge.stop_worker_for_test()


def test_session_boundaries_do_not_close_stream_or_stop_daemon(
    tmp_path: Path,
) -> None:
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        start_worker_on_capture=False,
    )
    hooks = HermesHooks(bridge)

    for name in ("on_session_end", "on_session_finalize", "on_session_reset"):
        payload: dict[str, object]
        if name == "on_session_end":
            payload = {
                "session_id": "session",
                "task_id": "task",
                "turn_id": "turn",
                "completed": True,
                "interrupted": False,
                "model": "model",
                "platform": "cli",
            }
        else:
            payload = {
                "session_id": "session",
                "platform": "cli",
                "reason": "boundary",
            }
        assert getattr(hooks, name)(
            telemetry_schema_version="hermes.observer.v1", **payload
        ) is None

    assert bridge.stop_requested is False
    assert bridge.close_requested is False


def test_health_explicitly_reports_unobserved_chunk_and_reasoning(
    tmp_path: Path,
) -> None:
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        start_worker_on_capture=False,
    )

    assert bridge.health.capabilities == {
        "chunk_capture": "unobserved",
        "reasoning_capture": "unobserved",
    }
    assert bridge.health.trace_complete is True
    assert not isinstance(bridge.retained_record, BridgeGapV1)


def test_gap_only_first_capture_bootstraps_worker_and_reaches_transport(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    client = _FakeClient(calls)
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        client_factory=lambda *_args, **_kwargs: client,
    )
    hooks = HermesHooks(bridge)

    hooks.on_session_start(
        telemetry_schema_version="wrong",
        session_id="session",
        model="model",
        platform="cli",
    )

    _wait_until(lambda: any(method == "bridge.commit" for method, _ in calls))
    bridge.stop_worker_for_test()
    (committed,) = [
        params["record"] for method, params in calls if method == "bridge.commit"
    ]
    assert committed["record_kind"] == "gap"
    assert committed["cause_counts"] == {"invalid_source_schema": 1}


def test_gap_retained_during_outage_remains_pending_in_health(
    tmp_path: Path,
) -> None:
    attempts = 0

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise DaemonClientError("outage")

    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        client_factory=unavailable,
    )
    hooks = HermesHooks(bridge)
    hooks.on_session_start(
        telemetry_schema_version="wrong",
        session_id="session",
        model="model",
        platform="cli",
    )

    _wait_until(
        lambda: attempts > 0
        and isinstance(bridge.retained_record, BridgeGapV1)
    )
    assert bridge.health.pending_gap_ranges == 1
    assert len(bridge.pending_gaps()) == 1
    bridge.stop_worker_for_test()


def test_idle_worker_refreshes_atomic_frame_from_daemon(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    client = _FakeClient(calls)
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        frame_refresh_seconds=0.02,
        client_factory=lambda *_args, **_kwargs: client,
    )
    hooks = HermesHooks(bridge)
    _session_start(hooks)

    _wait_until(
        lambda: len([call for call in calls if call[0] == "state.get"]) >= 2
    )
    assert bridge.projections.frame is not None
    assert bridge.projections.frame.through_capture_seq == 1
    bridge.stop_worker_for_test()


@pytest.mark.parametrize(
    "mutation",
    [
        "brain_id",
        "through_capture_seq",
        "projected_state_sequence",
        "stream_connection",
    ],
)
def test_initial_state_frame_must_match_resolve_attach_and_freshness(
    tmp_path: Path,
    mutation: str,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class MaliciousStateClient(_FakeClient):
        def call(
            self,
            method: str,
            params: dict[str, object] | None = None,
        ) -> dict[str, object]:
            result = super().call(method, params)
            if method != "state.get":
                return result
            if mutation == "brain_id":
                result["brain_id"] = new_id()
            elif mutation == "through_capture_seq":
                result["through_capture_seq"] = 1
            elif mutation == "projected_state_sequence":
                freshness = result["freshness"]
                assert isinstance(freshness, dict)
                freshness["projected_at_state_sequence"] = 1
            elif mutation == "stream_connection":
                freshness = result["freshness"]
                assert isinstance(freshness, dict)
                freshness["stream_connection"] = "disconnected"
            return result

    client = MaliciousStateClient(calls)
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        frame_refresh_seconds=60,
        client_factory=lambda *_args, **_kwargs: client,
    )
    hooks = HermesHooks(bridge)
    _session_start(hooks)

    _wait_until(lambda: any(method == "state.get" for method, _ in calls))
    time.sleep(0.03)
    assert bridge.last_ack is None
    assert bridge.retained_record is not None
    assert bridge.health.connection == "disconnected"
    bridge.stop_worker_for_test()


@pytest.mark.parametrize(
    "mutation",
    [
        "brain_id",
        "frame_cursor",
        "event_sequence",
        "freshness_sequence",
        "scheduler_sample",
        "duplicate",
    ],
)
def test_commit_ack_relations_are_strictly_bound(
    tmp_path: Path,
    mutation: str,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class MaliciousAckClient(_FakeClient):
        def call(
            self,
            method: str,
            params: dict[str, object] | None = None,
        ) -> dict[str, object]:
            result = super().call(method, params)
            if method != "bridge.commit":
                return result
            frame = result["frame"]
            assert isinstance(frame, dict)
            if mutation == "brain_id":
                frame["brain_id"] = new_id()
            elif mutation == "frame_cursor":
                frame["through_capture_seq"] = 0
            elif mutation == "event_sequence":
                result["event_sequence"] = 2
            elif mutation == "freshness_sequence":
                freshness = frame["freshness"]
                assert isinstance(freshness, dict)
                freshness["projected_at_state_sequence"] = 2
            elif mutation == "scheduler_sample":
                freshness = frame["freshness"]
                assert isinstance(freshness, dict)
                freshness["scheduler_sample"] = "running"
            elif mutation == "duplicate":
                result["duplicate"] = True
            return result

    client = MaliciousAckClient(calls)
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.05,
        frame_refresh_seconds=60,
        client_factory=lambda *_args, **_kwargs: client,
    )
    hooks = HermesHooks(bridge)
    _session_start(hooks)

    _wait_until(lambda: any(method == "bridge.commit" for method, _ in calls))
    time.sleep(0.02)
    assert bridge.last_ack is None
    assert bridge.retained_record is not None
    bridge.stop_worker_for_test()


def _closed_stream(
    bridge: HookBridge,
    client: _FakeClient,
    final_capture_seq: int,
) -> dict[str, object]:
    timestamp = datetime.now(UTC).isoformat()
    return {
        "bridge_instance_id": bridge.bridge_instance_id,
        "brain_id": client._brain_id,
        "server_actor_id": new_id(),
        "server_adapter_id": "alice-brain-hermes-observer-v1",
        "next_capture_seq": final_capture_seq + 1,
        "status": "clean_closed",
        "connected_nonce": None,
        "disconnected_reason": "clean_close",
        "disconnected_at": timestamp,
        "last_seen": timestamp,
        "closed_final_seq": final_capture_seq,
    }


def test_capture_after_close_request_before_seal_extends_final_cursor(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    release_connect = threading.Event()

    class ClosingClient(_FakeClient):
        def call(
            self,
            method: str,
            params: dict[str, object] | None = None,
        ) -> dict[str, object]:
            if method == "bridge.close":
                assert params is not None
                self.calls.append((method, dict(params)))
                return _closed_stream(bridge, self, int(params["final_capture_seq"]))
            return super().call(method, params)

    client = ClosingClient(calls)

    def connect(*_args: Any, **_kwargs: Any) -> ClosingClient:
        assert release_connect.wait(2)
        return client

    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        client_factory=connect,
        start_worker_on_capture=False,
    )
    hooks = HermesHooks(bridge)
    _session_start(hooks, "first")
    bridge.request_clean_close()
    _session_start(hooks, "second")
    release_connect.set()

    _wait_until(lambda: any(method == "bridge.close" for method, _ in calls))
    commits = [
        params["record"] for method, params in calls if method == "bridge.commit"
    ]
    close_params = next(params for method, params in calls if method == "bridge.close")
    assert [record["capture_seq"] for record in commits] == [1, 2]
    assert close_params["final_capture_seq"] == 2
    _wait_until(lambda: bridge._stop_event.is_set())
    assert client.closed is True
    bridge.stop_worker_for_test()


def test_capture_after_close_seal_is_accounted_without_cursor_race(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    close_entered = threading.Event()
    release_close = threading.Event()

    class BlockingCloseClient(_FakeClient):
        def call(
            self,
            method: str,
            params: dict[str, object] | None = None,
        ) -> dict[str, object]:
            if method == "bridge.close":
                assert params is not None
                self.calls.append((method, dict(params)))
                close_entered.set()
                assert release_close.wait(2)
                return _closed_stream(bridge, self, int(params["final_capture_seq"]))
            return super().call(method, params)

    client = BlockingCloseClient(calls)
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        client_factory=lambda *_args, **_kwargs: client,
    )
    hooks = HermesHooks(bridge)
    _session_start(hooks, "first")
    bridge.request_clean_close()
    assert close_entered.wait(2)
    _session_start(hooks, "too-late")
    release_close.set()

    _wait_until(lambda: client.closed)
    commits = [
        params["record"] for method, params in calls if method == "bridge.commit"
    ]
    assert [record["capture_seq"] for record in commits] == [1]
    assert bridge.health.late_after_close == 1
    assert bridge.health.dropped_events == 1
    assert bridge.health.trace_complete is False
    assert bridge._stop_event.is_set() is True
    bridge.stop_worker_for_test()


def test_lost_close_ack_uses_authenticated_recovery_without_daemon_stop(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    first = _FakeClient(calls)
    second = _FakeClient(calls)
    second._brain_id = first._brain_id
    second._next_capture_seq = 2
    second._through_capture_seq = 1

    original_first = first.call
    original_second = second.call

    def first_call(
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if method == "bridge.close":
            calls.append((method, dict(params or {})))
            raise DaemonClientError("close ACK lost after persistence")
        return original_first(method, params)

    def second_call(
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if method == "brain.attach":
            calls.append((method, dict(params or {})))
            raise DaemonRpcError("bridge_clean_closed", "closed", {})
        if method == "bridge.close.recover":
            assert params is not None
            calls.append((method, dict(params)))
            return _closed_stream(bridge, second, int(params["final_capture_seq"]))
        return original_second(method, params)

    first.call = first_call  # type: ignore[method-assign]
    second.call = second_call  # type: ignore[method-assign]
    clients = iter((first, second))
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        reconnect_delay_seconds=0.01,
        client_factory=lambda *_args, **_kwargs: next(clients),
    )
    hooks = HermesHooks(bridge)
    _session_start(hooks)
    bridge.request_clean_close()

    _wait_until(
        lambda: any(method == "bridge.close.recover" for method, _ in calls)
    )
    recovery = next(
        params for method, params in calls if method == "bridge.close.recover"
    )
    assert recovery == {
        "brain_id": first._brain_id,
        "bridge_instance_id": bridge.bridge_instance_id,
        "recovery_token": bridge.recovery_token,
        "final_capture_seq": 1,
    }
    assert not any(method == "daemon.shutdown" for method, _ in calls)
    _wait_until(lambda: bridge._stop_event.is_set())
    assert second.closed is True
    bridge.stop_worker_for_test()


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX readiness fd")
def test_real_daemon_commit_projection_and_clean_close(tmp_path: Path) -> None:
    home = tmp_path / "real-daemon-runtime"
    home.mkdir(mode=0o700)
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "alice_brain_hermes.runtime.daemon",
            "--runtime-home",
            str(home),
            "--readiness-fd",
            str(write_fd),
            "--scheduler-interval",
            "0.02",
            "--abandonment-grace",
            "30",
        ],
        pass_fds=(write_fd,),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.close(write_fd)
    bridge: HookBridge | None = None
    try:
        ready, _, _ = select.select([read_fd], [], [], 10)
        assert ready
        body = bytearray()
        while b"\n" not in body:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            body.extend(chunk)
        assert json.loads(bytes(body))["ready"] is True

        bridge = HookBridge(home, reconnect_delay_seconds=0.01)
        hooks = HermesHooks(bridge)
        _session_start(hooks, "real-daemon-session")
        _wait_until(lambda: bridge.last_ack is not None, timeout=10)
        assert bridge.last_ack is not None
        assert bridge.last_ack.through_capture_seq == 1
        assert bridge.last_ack.frame.freshness.scheduler_sample == "not_sampled"
        assert bridge.health.through_capture_seq == 1
        assert bridge.projections.read_context()

        bridge.request_clean_close()
        _wait_until(lambda: bridge._stop_event.is_set(), timeout=10)
        bridge.stop_worker_for_test()

        client = DaemonClient.connect(home)
        client.shutdown()
        client.close()
        assert process.wait(timeout=10) == 0
        stdout, stderr = process.communicate()
        assert stdout == b""
        assert stderr == b""
    finally:
        os.close(read_fd)
        if bridge is not None:
            bridge.stop_worker_for_test()
        if process.poll() is None:
            try:
                cleanup = DaemonClient.connect(home)
                cleanup.shutdown()
                cleanup.close()
            except BaseException:
                process.kill()
            process.wait(timeout=10)


def test_oversized_projection_falls_back_to_valid_bounded_json(
    tmp_path: Path,
) -> None:
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        start_worker_on_capture=False,
    )
    brain_id = new_id()
    raw = _frame(brain_id, 0)
    raw["semantic_context"] = {"large": "x" * 100_000}
    raw["capabilities"] = {
        "chunk_capture": "x" * 100_000,
        "reasoning_capture": "x" * 100_000,
        **{f"extra-{index}": "x" * 1_000 for index in range(128)},
    }
    raw["omission_counts"] = {"large": "x" * 100_000}
    from alice_brain_hermes.protocol.models import ConsciousnessFrameV2

    bridge.projections.publish_frame(
        ConsciousnessFrameV2.model_validate(raw, strict=True)
    )

    context = bridge.projections.read_context()
    assert type(context) is str
    assert len(context.encode("utf-8")) <= MAX_EPHEMERAL_CONTEXT_BYTES
    decoded = json.loads(context)
    assert decoded["alice_brain"]["projection_truncated"] is True
    assert decoded["alice_brain"]["brain_id"] == brain_id
    assert decoded["alice_brain"]["trace_complete"] is True


def test_projection_thaws_nested_immutable_frame_values(tmp_path: Path) -> None:
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        start_worker_on_capture=False,
    )
    brain_id = new_id()
    raw = _frame(brain_id, 0)
    raw["pc"] = {"traits": {"care": {"value": 0.75}}}
    raw["semantic_context"] = {"nested": {"items": [1, {"two": 2}]}}
    from alice_brain_hermes.protocol.models import ConsciousnessFrameV2

    bridge.projections.publish_frame(
        ConsciousnessFrameV2.model_validate(raw, strict=True)
    )

    context = bridge.projections.read_context()
    assert type(context) is str
    decoded = json.loads(context)["alice_brain"]
    assert decoded["pc"] == {"traits": {"care": {"value": 0.75}}}
    assert decoded["semantic_context"] == {
        "nested": {"items": [1, {"two": 2}]}
    }
