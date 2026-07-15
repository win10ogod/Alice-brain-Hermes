from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from alice_brain_hermes.hermes.bridge import HookBridge
from alice_brain_hermes.hermes.hooks import HermesHooks
from alice_brain_hermes.protocol.models import (
    MAX_BRIDGE_INTEGER,
    MAX_CAPTURE_SEQUENCE,
)

LAST_PERSISTABLE_CAPTURE_SEQUENCE = MAX_CAPTURE_SEQUENCE


def _session_start(hooks: HermesHooks, *, session_id: str) -> None:
    assert (
        hooks.on_session_start(
            telemetry_schema_version="hermes.observer.v1",
            session_id=session_id,
            model="model",
            platform="cli",
        )
        is None
    )


def _bootstrap_reservation(sequence: int) -> dict[str, object]:
    return {
        "hook": "on_session_start",
        "detached_kwargs": MappingProxyType(
            {
                "telemetry_schema_version": "hermes.observer.v1",
                "session_id": "capacity",
                "model": "model",
                "platform": "cli",
            }
        ),
        "first_capture_seq": sequence,
        "last_capture_seq": sequence,
        "gap_cause_counts": None,
        "copy_stats": MappingProxyType({}),
    }


def test_direct_bridge_accepts_last_persistable_capture_then_fails_closed(
    tmp_path: Path,
) -> None:
    bridge = HookBridge(tmp_path, start_worker_on_capture=False)
    bridge._next_capture_seq = LAST_PERSISTABLE_CAPTURE_SEQUENCE  # type: ignore[attr-defined]
    hooks = HermesHooks(bridge)

    _session_start(hooks, session_id="last-persistable")
    accepted = bridge.queue.get_nowait()

    assert accepted.capture_seq == LAST_PERSISTABLE_CAPTURE_SEQUENCE
    assert bridge._next_capture_seq == MAX_BRIDGE_INTEGER  # type: ignore[attr-defined]

    _session_start(hooks, session_id="must-not-wrap")

    assert bridge.queue.empty()
    assert bridge._next_capture_seq == MAX_BRIDGE_INTEGER  # type: ignore[attr-defined]
    assert bridge.health.last_capture_seq == LAST_PERSISTABLE_CAPTURE_SEQUENCE
    assert bridge.health.trace_complete is False
    assert bridge.health.last_error == "capture_sequence_capacity_exhausted"


def test_reserved_bridge_rejects_unpersistable_capture_without_advancing(
    tmp_path: Path,
) -> None:
    bridge = HookBridge(tmp_path, start_worker_on_capture=False)
    bridge._next_capture_seq = LAST_PERSISTABLE_CAPTURE_SEQUENCE  # type: ignore[attr-defined]

    bridge.capture_reserved(**_bootstrap_reservation(LAST_PERSISTABLE_CAPTURE_SEQUENCE))
    accepted = bridge.queue.get_nowait()

    assert accepted.capture_seq == LAST_PERSISTABLE_CAPTURE_SEQUENCE
    assert bridge._next_capture_seq == MAX_BRIDGE_INTEGER  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="capture sequence capacity is exhausted"):
        bridge.capture_reserved(**_bootstrap_reservation(MAX_BRIDGE_INTEGER))

    assert bridge.queue.empty()
    assert bridge.retained_record is None
    assert bridge._next_capture_seq == MAX_BRIDGE_INTEGER  # type: ignore[attr-defined]
    assert bridge.health.last_capture_seq == LAST_PERSISTABLE_CAPTURE_SEQUENCE


def test_bootstrap_capacity_exhaustion_is_visible_without_unhandoffable_record() -> (
    None
):
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    bootstrap._next_capture_seq = (  # type: ignore[attr-defined]
        LAST_PERSISTABLE_CAPTURE_SEQUENCE
    )

    bootstrap.capture(
        "on_session_start",
        {
            "telemetry_schema_version": "hermes.observer.v1",
            "session_id": "last-persistable",
        },
    )
    [accepted] = bootstrap.pending_for_test()

    assert accepted.capture_seq == LAST_PERSISTABLE_CAPTURE_SEQUENCE
    assert bootstrap._next_capture_seq == MAX_BRIDGE_INTEGER  # type: ignore[attr-defined]

    bootstrap.capture(
        "on_session_start",
        {
            "telemetry_schema_version": "hermes.observer.v1",
            "session_id": "must-not-wrap",
        },
    )

    assert bootstrap.pending_for_test() == (accepted,)
    assert bootstrap._next_capture_seq == MAX_BRIDGE_INTEGER  # type: ignore[attr-defined]
    assert bootstrap.health.pending_records == 1
    assert bootstrap.health.trace_complete is False
    assert bootstrap.health.degraded is True
    assert bootstrap.health.last_error == "capture_sequence_capacity_exhausted"


def test_bridge_accepts_an_attached_exhausted_successor_but_no_later_capture(
    tmp_path: Path,
) -> None:
    from tests.hermes.test_nonblocking import _FakeClient

    calls: list[tuple[str, dict[str, object]]] = []
    client = _FakeClient(calls)
    client._next_capture_seq = MAX_BRIDGE_INTEGER
    client._through_capture_seq = LAST_PERSISTABLE_CAPTURE_SEQUENCE
    bridge = HookBridge(
        tmp_path,
        client_factory=lambda *_args, **_kwargs: client,
        start_worker_on_capture=False,
    )
    bridge._next_capture_seq = LAST_PERSISTABLE_CAPTURE_SEQUENCE  # type: ignore[attr-defined]
    bridge.capture_reserved(**_bootstrap_reservation(LAST_PERSISTABLE_CAPTURE_SEQUENCE))
    bridge.queue.get_nowait()

    assert bridge._connect() is True  # type: ignore[attr-defined]
    assert bridge._attached_next_capture_seq == MAX_BRIDGE_INTEGER  # type: ignore[attr-defined]

    _session_start(HermesHooks(bridge), session_id="must-not-wrap-after-attach")

    assert bridge.queue.empty()
    assert bridge._next_capture_seq == MAX_BRIDGE_INTEGER  # type: ignore[attr-defined]
    assert bridge.health.last_error == "capture_sequence_capacity_exhausted"
    assert any(method == "brain.attach" for method, _params in calls)
    bridge._disconnect_client()  # type: ignore[attr-defined]
