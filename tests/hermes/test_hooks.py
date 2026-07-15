from __future__ import annotations

import queue
from datetime import UTC, datetime
from typing import Any

import pytest

from alice_brain_hermes.hermes.bridge import HookBridge
from alice_brain_hermes.hermes.hooks import HermesHooks
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.models import (
    BridgeGapV1,
    HermesObservationV1,
)


def _host_payloads() -> dict[str, dict[str, Any]]:
    common_turn = {
        "session_id": "session",
        "task_id": "task",
        "turn_id": "turn",
    }
    common_api = {**common_turn, "api_request_id": "api-request"}
    common_tool = {**common_api, "tool_call_id": "tool-call"}
    return {
        "on_session_start": {
            "session_id": "session",
            "model": "model",
            "platform": "cli",
        },
        "on_session_end": {
            **common_api,
            "completed": True,
            "interrupted": False,
            "model": "model",
            "platform": "cli",
            "reason": None,
        },
        "on_session_finalize": {
            "session_id": "session",
            "platform": "cli",
            "reason": "shutdown",
        },
        "on_session_reset": {
            "session_id": "new-session",
            "old_session_id": "old-session",
            "new_session_id": "new-session",
            "platform": "cli",
            "reason": "new_session",
        },
        "pre_llm_call": {
            **common_turn,
            "sender_id": "sender",
            "user_message": "hello",
            "conversation_history": [{"role": "user", "content": "hello"}],
            "is_first_turn": True,
            "model": "model",
            "platform": "cli",
        },
        "post_llm_call": {
            **common_turn,
            "user_message": "hello",
            "assistant_response": "hi",
            "conversation_history": [],
            "model": "model",
            "platform": "cli",
        },
        "pre_api_request": {
            **common_api,
            "user_message": "hello",
            "conversation_history": [],
            "platform": "cli",
            "model": "model",
            "provider": "provider",
            "base_url": None,
            "api_mode": "chat",
            "api_call_count": 0,
            "request_messages": [],
            "message_count": 0,
            "tool_count": 0,
            "approx_input_tokens": 0,
            "request_char_count": 0,
            "max_tokens": None,
            "started_at": 1.0,
            "middleware_trace": [],
            "request": {},
        },
        "post_api_request": {
            **common_api,
            "platform": "cli",
            "model": "model",
            "provider": "provider",
            "base_url": None,
            "api_mode": "chat",
            "api_call_count": 0,
            "api_duration": 0.1,
            "started_at": 1.0,
            "ended_at": 1.1,
            "finish_reason": "stop",
            "message_count": 1,
            "response_model": "model",
            "response": {},
            "usage": {},
            "assistant_message": {"role": "assistant", "content": "hi"},
            "assistant_content_chars": 2,
            "assistant_tool_call_count": 0,
        },
        "api_request_error": {
            **common_api,
            "platform": "cli",
            "model": "model",
            "provider": "provider",
            "base_url": None,
            "api_mode": "chat",
            "api_call_count": 0,
            "api_duration": 0.1,
            "started_at": 1.0,
            "ended_at": 1.1,
            "status_code": None,
            "retry_count": 0,
            "max_retries": 0,
            "retryable": False,
            "reason": "provider_error",
            "error": {"type": "Error", "message": "failed"},
            "request": {},
        },
        "pre_tool_call": {
            **common_tool,
            "tool_name": "terminal",
            "args": {"command": "true"},
            "middleware_trace": [],
        },
        "post_tool_call": {
            **common_tool,
            "tool_name": "terminal",
            "args": {"command": "true"},
            "middleware_trace": [],
            "result": "ok",
            "duration_ms": 1,
            "status": "ok",
            "error_type": None,
            "error_message": None,
        },
        "pre_approval_request": {
            "turn_id": "turn",
            "tool_call_id": "tool-call",
            "command": "rm file",
            "description": "remove file",
            "pattern_key": "rm",
            "pattern_keys": ["rm"],
            "session_key": "session",
            "surface": "cli",
        },
        "post_approval_response": {
            "turn_id": "turn",
            "tool_call_id": "tool-call",
            "command": "rm file",
            "description": "remove file",
            "pattern_key": "rm",
            "pattern_keys": ["rm"],
            "session_key": "session",
            "surface": "cli",
            "choice": "deny",
            "decided_by": None,
        },
        "subagent_start": {
            "parent_session_id": "parent-session",
            "parent_turn_id": "parent-turn",
            "parent_subagent_id": None,
            "child_session_id": "child-session",
            "child_subagent_id": None,
            "child_role": "worker",
            "child_goal": "check",
        },
        "subagent_stop": {
            "parent_session_id": "parent-session",
            "parent_turn_id": "parent-turn",
            "child_session_id": "child-session",
            "child_role": "worker",
            "child_summary": "done",
            "child_status": "completed",
            "duration_ms": 10,
        },
        "pre_verify": {
            "session_id": "session",
            "platform": "cli",
            "model": "model",
            "coding": True,
            "attempt": 0,
            "final_response": "done",
            "changed_paths": ["a.py"],
        },
    }


@pytest.fixture
def bridge(tmp_path: Any) -> HookBridge:
    return HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        queue_capacity=64,
        start_worker_on_capture=False,
    )


def test_exact_hermes_018_hook_payloads_become_typed_observations(
    bridge: HookBridge,
) -> None:
    hooks = HermesHooks(bridge)
    payloads = _host_payloads()

    for hook_name, payload in payloads.items():
        result = getattr(hooks, hook_name)(
            telemetry_schema_version="hermes.observer.v1",
            **payload,
        )
        assert result is None

    records = [bridge.queue.get_nowait() for _ in payloads]
    assert [record.hook for record in records] == list(payloads)
    assert all(isinstance(record, HermesObservationV1) for record in records)
    assert [record.capture_seq for record in records] == list(
        range(1, len(payloads) + 1)
    )
    pre_tool = records[9]
    assert pre_tool.payload.args == {"command": "true"}
    assert pre_tool.context.api_request_id == "api-request"


def test_pre_llm_is_the_only_hook_that_can_return_plain_cached_context(
    bridge: HookBridge,
) -> None:
    hooks = HermesHooks(bridge)
    bridge.projections.publish_context("bounded cached self-context")
    payloads = _host_payloads()

    for hook_name, payload in payloads.items():
        result = getattr(hooks, hook_name)(
            telemetry_schema_version="hermes.observer.v1",
            **payload,
        )
        if hook_name == "pre_llm_call":
            assert type(result) is str
            assert result == "bounded cached self-context"
        else:
            assert result is None


def test_legal_host_empty_identifiers_are_observed_not_downgraded_to_gap(
    bridge: HookBridge,
) -> None:
    hooks = HermesHooks(bridge)

    assert (
        hooks.on_session_finalize(
            telemetry_schema_version="hermes.observer.v1",
            session_id=None,
            platform="gateway",
            reason="shutdown",
        )
        is None
    )
    assert (
        hooks.subagent_start(
            telemetry_schema_version="hermes.observer.v1",
            parent_session_id=None,
            parent_turn_id="",
            parent_subagent_id=None,
            child_session_id=None,
            child_subagent_id=None,
            child_role="worker",
            child_goal="goal",
        )
        is None
    )

    first = bridge.queue.get_nowait()
    second = bridge.queue.get_nowait()
    assert isinstance(first, HermesObservationV1)
    assert isinstance(second, HermesObservationV1)
    assert first.context.session_id is None
    assert second.context.parent_session_id is None
    assert second.context.child_session_id is None
    assert bridge.pending_gaps() == ()


def test_invalid_source_schema_reserves_exact_capture_sequence(
    bridge: HookBridge,
) -> None:
    hooks = HermesHooks(bridge)

    hooks.on_session_start(
        telemetry_schema_version=2,
        session_id="session",
        model="model",
        platform="cli",
    )

    gaps = bridge.pending_gaps()
    assert len(gaps) == 1
    assert isinstance(gaps[0], BridgeGapV1)
    assert gaps[0].first_capture_seq == 1
    assert gaps[0].last_capture_seq == 1
    assert dict(gaps[0].cause_counts) == {"invalid_source_schema": 1}
    assert bridge.health.trace_complete is False


def test_full_queue_records_queue_full_without_blocking(
    tmp_path: Any,
) -> None:
    bridge = HookBridge(
        tmp_path,
        bridge_instance_id=new_id(),
        queue_capacity=1,
        start_worker_on_capture=False,
    )
    hooks = HermesHooks(bridge)
    payload = _host_payloads()["on_session_start"]

    hooks.on_session_start(telemetry_schema_version="hermes.observer.v1", **payload)
    hooks.on_session_start(telemetry_schema_version="hermes.observer.v1", **payload)

    queued = bridge.queue.get_nowait()
    assert queued.capture_seq == 1
    (gap,) = bridge.pending_gaps()
    assert (gap.first_capture_seq, gap.last_capture_seq) == (2, 2)
    assert dict(gap.cause_counts) == {"queue_full": 1}
    assert bridge.health.dropped_events == 1


def test_callback_internal_failure_is_reported_and_never_raised(
    bridge: HookBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks = HermesHooks(bridge)

    def broken_shape(*_args: Any, **_kwargs: Any) -> HermesObservationV1:
        raise RuntimeError("shape exploded")

    monkeypatch.setattr(bridge, "_shape_observation", broken_shape)

    assert (
        hooks.on_session_start(
            telemetry_schema_version="hermes.observer.v1",
            session_id="session",
            model="model",
            platform="cli",
        )
        is None
    )
    (gap,) = bridge.pending_gaps()
    assert dict(gap.cause_counts) == {"callback_internal": 1}


@pytest.mark.parametrize(
    "failure_type",
    [RuntimeError, KeyboardInterrupt, MemoryError],
    ids=["exception", "keyboard-interrupt", "memory-error"],
)
def test_callback_facade_contains_every_capture_baseexception(
    bridge: HookBridge,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    hooks = HermesHooks(bridge)

    def fail_capture(*_args: object, **_kwargs: object) -> None:
        raise failure_type("capture boundary failed")

    monkeypatch.setattr(bridge, "capture", fail_capture)

    for hook_name, payload in _host_payloads().items():
        assert (
            getattr(hooks, hook_name)(
                telemetry_schema_version="hermes.observer.v1",
                **payload,
            )
            is None
        )

    assert bridge.health.trace_complete is False
    assert bridge.health.last_error == failure_type.__name__


def test_pre_llm_cache_read_baseexception_never_escapes(
    bridge: HookBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks = HermesHooks(bridge)

    def fail_cache_read() -> None:
        raise MemoryError("projection cache read failed")

    monkeypatch.setattr(bridge.projections, "read_context", fail_cache_read)

    assert (
        hooks.pre_llm_call(
            telemetry_schema_version="hermes.observer.v1",
            **_host_payloads()["pre_llm_call"],
        )
        is None
    )
    assert bridge.queue.qsize() == 1
    assert bridge.health.trace_complete is False
    assert bridge.health.last_error == "MemoryError"


def test_capture_metadata_is_utc_and_monotonic(bridge: HookBridge) -> None:
    hooks = HermesHooks(bridge)
    payload = _host_payloads()["on_session_start"]

    hooks.on_session_start(telemetry_schema_version="hermes.observer.v1", **payload)
    hooks.on_session_start(telemetry_schema_version="hermes.observer.v1", **payload)

    first = bridge.queue.get_nowait()
    second = bridge.queue.get_nowait()
    assert isinstance(first.captured_at, datetime)
    assert first.captured_at.tzinfo == UTC
    assert second.captured_monotonic_ns >= first.captured_monotonic_ns


def test_gap_causes_never_invent_transport_or_daemon_loss(
    bridge: HookBridge,
) -> None:
    hooks = HermesHooks(bridge)
    hooks.on_session_start(telemetry_schema_version="1")

    causes = {cause for gap in bridge.pending_gaps() for cause in gap.cause_counts}
    assert causes == {"invalid_source_schema"}
    assert not causes.intersection(
        {"transport_failed", "daemon_unavailable", "backpressure"}
    )


def test_queue_is_exactly_bounded(bridge: HookBridge) -> None:
    assert isinstance(bridge.queue, queue.Queue)
    assert bridge.queue.maxsize == 64
