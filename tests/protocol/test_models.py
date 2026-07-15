from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol import models as models_module
from alice_brain_hermes.protocol.models import (
    GAP_CAUSES,
    MAX_BRIDGE_RECORD_BYTES,
    BrainProfileV1,
    BridgeGapV1,
    CapabilityProfileV1,
    CoverageV1,
    ProtocolLimitsV1,
    validate_bridge_record_json,
    validate_observation,
    validate_observation_json,
)


def test_semantic_batch_uses_versioned_protocol_and_frame_contract() -> None:
    assert models_module.PROTOCOL_VERSION == 2
    assert models_module.FRAME_SCHEMA_VERSION == 3


def _empty_frame_v3(*, brain_id: str, state_sequence: int) -> object:
    return models_module.ConsciousnessFrameV3(
        brain_id=brain_id,
        state_sequence=state_sequence,
        through_capture_seq=1,
        logical_clock=0.0,
        trace_complete=True,
        runtime_health="healthy",
        c0_tick=0,
        pc={},
        energy={},
        st={},
        rd={},
        a={},
        world={},
        self_boundary={},
        memory={},
        capabilities={},
        semantic_context={},
        aggregate_semantic_complete=True,
        semantic_evidence={
            "semantic_records": 1,
            "legacy_raw_only_records": 0,
            "semantic_gap_records": 0,
            "dropped_events": 0,
        },
        unresolved_evidence=False,
        capture_coverage={},
        freshness={
            "projected_at_state_sequence": state_sequence,
            "scheduler_tick": 0,
            "scheduler_sample": "not_sampled",
            "stream_connection": "connected",
        },
    )


def test_ack_v2_binds_raw_and_bounded_contiguous_derived_batch() -> None:
    brain_id = new_id()
    raw_event_id = new_id()
    derived_ids = (new_id(), new_id())
    ack = models_module.BridgeCommitAckV2(
        record_fingerprint="a" * 64,
        raw_event_id=raw_event_id,
        raw_event_sequence=4,
        derived_event_ids=derived_ids,
        derived_event_count=2,
        last_event_sequence=6,
        semantic_status="applied",
        semantic_complete=True,
        semantic_fingerprint="b" * 64,
        frame=_empty_frame_v3(brain_id=brain_id, state_sequence=6),
        through_capture_seq=1,
    )

    assert ack.schema_version == 2
    assert ack.event_id == raw_event_id
    assert ack.event_sequence == 4
    assert ack.frame.state_sequence == ack.last_event_sequence
    assert ack.frame.semantic_schema_version == 1
    assert ack.frame.aggregate_semantic_complete is True


def test_ack_v2_rejects_raw_event_id_reused_as_derived_identity() -> None:
    raw_event_id = new_id()
    with pytest.raises(ValidationError, match="raw event ID"):
        models_module.BridgeCommitAckV2(
            record_fingerprint="a" * 64,
            raw_event_id=raw_event_id,
            raw_event_sequence=4,
            derived_event_ids=(raw_event_id,),
            derived_event_count=1,
            last_event_sequence=5,
            semantic_status="applied",
            semantic_complete=True,
            semantic_fingerprint="b" * 64,
            frame=_empty_frame_v3(brain_id=new_id(), state_sequence=5),
            through_capture_seq=1,
        )


def test_ack_v2_rejects_arbitrary_multi_event_semantic_gap() -> None:
    with pytest.raises(ValidationError, match="zero or one"):
        models_module.BridgeCommitAckV2(
            record_fingerprint="a" * 64,
            raw_event_id=new_id(),
            raw_event_sequence=4,
            derived_event_ids=(new_id(), new_id()),
            derived_event_count=2,
            last_event_sequence=6,
            semantic_status="gap",
            semantic_complete=False,
            semantic_fingerprint="b" * 64,
            frame=_empty_frame_v3(brain_id=new_id(), state_sequence=6),
            through_capture_seq=1,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"derived_event_count": 1}, "derived event count"),
        ({"last_event_sequence": 5}, "contiguous"),
        ({"semantic_complete": False}, "semantic completeness"),
    ],
)
def test_ack_v2_rejects_inconsistent_semantic_batch(
    updates: dict[str, object], message: str
) -> None:
    values = {
        "record_fingerprint": "a" * 64,
        "raw_event_id": new_id(),
        "raw_event_sequence": 4,
        "derived_event_ids": (new_id(), new_id()),
        "derived_event_count": 2,
        "last_event_sequence": 6,
        "semantic_status": "applied",
        "semantic_complete": True,
        "semantic_fingerprint": "b" * 64,
        "frame": _empty_frame_v3(brain_id=new_id(), state_sequence=6),
        "through_capture_seq": 1,
    }
    values.update(updates)
    with pytest.raises(ValidationError, match=message):
        models_module.BridgeCommitAckV2(**values)


def _coverage() -> dict[str, object]:
    return CoverageV1(
        policy_version="copy-v1",
        capture_coverage="host_sanitized",
    ).model_dump(mode="json")


def _case(
    hook: str,
    context: dict[str, object],
    payload: dict[str, object],
    required_payload_field: str,
) -> tuple[str, dict[str, object], dict[str, object], str]:
    return hook, context, payload, required_payload_field


HOOK_CASES = [
    _case(
        "on_session_start",
        {"session_id": "session"},
        {"model": "model", "platform": "cli"},
        "model",
    ),
    _case(
        "on_session_end",
        {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "api_request_id": "request",
        },
        {
            "model": "model",
            "platform": "cli",
            "completed": True,
            "interrupted": False,
            "reason": "done",
        },
        "completed",
    ),
    _case(
        "on_session_finalize",
        {"session_id": "session"},
        {
            "platform": "cli",
            "reason": "finalize",
            "old_session_id": "old-session",
            "new_session_id": "new-session",
        },
        "platform",
    ),
    _case(
        "on_session_reset",
        {"session_id": "session"},
        {
            "platform": "cli",
            "reason": "reset",
            "old_session_id": "old-session",
            "new_session_id": "new-session",
        },
        "platform",
    ),
    _case(
        "pre_llm_call",
        {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "sender_id": "sender",
        },
        {
            "user_message": {"content": "hello"},
            "conversation_history": [{"role": "user"}],
            "is_first_turn": True,
            "model": "model",
            "platform": "cli",
        },
        "user_message",
    ),
    _case(
        "post_llm_call",
        {"session_id": "session", "task_id": "task", "turn_id": "turn"},
        {
            "user_message": {"content": "hello"},
            "assistant_response": {"content": "hi"},
            "conversation_history": [],
            "model": "model",
            "platform": "cli",
        },
        "assistant_response",
    ),
    _case(
        "pre_api_request",
        {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "api_request_id": "request",
        },
        {
            "user_message": "hello",
            "conversation_history": [],
            "platform": "cli",
            "model": "model",
            "provider": "provider",
            "base_url": "https://example.invalid",
            "api_mode": "stream",
            "api_call_count": 1,
            "request_messages": [],
            "message_count": 0,
            "tool_count": 0,
            "approx_input_tokens": 0,
            "request_char_count": 0,
            "max_tokens": None,
            "started_at": "start",
            "middleware_trace": [],
            "request": {},
        },
        "request",
    ),
    _case(
        "post_api_request",
        {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "api_request_id": "request",
        },
        {
            "platform": "cli",
            "model": "model",
            "provider": "provider",
            "base_url": None,
            "api_mode": "stream",
            "api_call_count": 1,
            "api_duration": 0.1,
            "started_at": "start",
            "ended_at": "end",
            "finish_reason": "stop",
            "message_count": 1,
            "response_model": "model",
            "response": {},
            "usage": {},
            "assistant_message": {},
            "assistant_content_chars": 0,
            "assistant_tool_call_count": 0,
        },
        "response",
    ),
    _case(
        "api_request_error",
        {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "api_request_id": "request",
        },
        {
            "platform": "cli",
            "model": "model",
            "provider": "provider",
            "base_url": None,
            "api_mode": "stream",
            "api_call_count": 1,
            "api_duration": 0.1,
            "started_at": "start",
            "ended_at": "end",
            "status_code": 500,
            "retry_count": 1,
            "max_retries": 3,
            "retryable": True,
            "reason": "server error",
            "error": {"type": "HTTPError"},
            "request": {},
        },
        "error",
    ),
    _case(
        "pre_tool_call",
        {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "api_request_id": "request",
            "tool_call_id": "tool-call",
        },
        {"tool_name": "shell", "args": {}, "middleware_trace": []},
        "args",
    ),
    _case(
        "post_tool_call",
        {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "api_request_id": "request",
            "tool_call_id": "tool-call",
        },
        {
            "tool_name": "shell",
            "args": {},
            "middleware_trace": [],
            "result": {"stdout": "ok"},
            "duration_ms": 1.0,
            "status": "success",
            "error_type": None,
            "error_message": None,
        },
        "result",
    ),
    _case(
        "pre_approval_request",
        {"turn_id": "turn", "tool_call_id": "tool-call"},
        {
            "command": "rm file",
            "description": "remove a file",
            "pattern_key": "rm",
            "pattern_keys": ["rm"],
            "session_key": "session",
            "surface": "cli",
        },
        "command",
    ),
    _case(
        "post_approval_response",
        {"turn_id": "turn", "tool_call_id": "tool-call"},
        {
            "command": "rm file",
            "description": "remove a file",
            "pattern_key": "rm",
            "pattern_keys": ["rm"],
            "session_key": "session",
            "surface": "cli",
            "choice": "allow",
            "decided_by": "user",
        },
        "choice",
    ),
    _case(
        "subagent_start",
        {
            "parent_session_id": "parent-session",
            "child_session_id": "child-session",
        },
        {
            "parent_turn_id": "turn",
            "parent_subagent_id": None,
            "child_subagent_id": "child-agent",
            "child_role": "researcher",
            "child_goal": "research",
        },
        "child_role",
    ),
    _case(
        "subagent_stop",
        {
            "parent_session_id": "parent-session",
            "child_session_id": "child-session",
        },
        {
            "parent_turn_id": "turn",
            "child_role": "researcher",
            "child_summary": "done",
            "child_status": "completed",
            "duration_ms": 1.0,
        },
        "child_status",
    ),
    _case(
        "pre_verify",
        {"session_id": "session"},
        {
            "platform": "cli",
            "model": "model",
            "coding": True,
            "attempt": 1,
            "final_response": "done",
            "changed_paths": ["file.py"],
        },
        "final_response",
    ),
]


def _observation(
    hook: str,
    context: dict[str, object],
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_kind": "observation",
        "bridge_instance_id": new_id(),
        "capture_seq": 1,
        "captured_at": datetime.now(UTC),
        "captured_monotonic_ns": 1,
        "source_schema_version": "hermes.observer.v1",
        "hook": hook,
        "context": context,
        "payload": {**payload, "extensions": {"raw": {"preserved": True}}},
        "coverage": _coverage(),
    }


@pytest.mark.parametrize(
    ("hook", "context", "payload", "required_payload_field"),
    HOOK_CASES,
    ids=[case[0] for case in HOOK_CASES],
)
def test_closed_hook_union_accepts_exact_variant_and_round_trips_json(
    hook: str,
    context: dict[str, object],
    payload: dict[str, object],
    required_payload_field: str,
) -> None:
    record = validate_observation(_observation(hook, context, payload))

    restored = validate_observation_json(record.canonical_json())

    assert restored == record
    assert restored.hook == hook
    assert restored.payload.extensions["raw"] == {"preserved": True}


@pytest.mark.parametrize(
    ("hook", "context", "payload", "required_payload_field"),
    HOOK_CASES,
    ids=[case[0] for case in HOOK_CASES],
)
def test_closed_hook_union_rejects_missing_required_and_extra_fields(
    hook: str,
    context: dict[str, object],
    payload: dict[str, object],
    required_payload_field: str,
) -> None:
    missing = _observation(hook, context, payload)
    del missing["payload"][required_payload_field]
    extra_payload = _observation(hook, context, payload)
    extra_payload["payload"]["undeclared"] = True
    extra_context = _observation(hook, context, payload)
    extra_context["context"]["undeclared"] = True

    for invalid in (missing, extra_payload, extra_context):
        with pytest.raises(ValidationError):
            validate_observation(invalid)


@pytest.mark.parametrize(
    ("hook", "context", "payload", "required_payload_field"),
    HOOK_CASES,
    ids=[case[0] for case in HOOK_CASES],
)
def test_closed_hook_union_rejects_payload_from_a_different_hook(
    hook: str,
    context: dict[str, object],
    payload: dict[str, object],
    required_payload_field: str,
) -> None:
    del required_payload_field
    invalid = _observation(hook, context, payload)
    current_index = next(
        index for index, case in enumerate(HOOK_CASES) if case[0] == hook
    )
    next_index = (current_index + 1) % len(HOOK_CASES)
    invalid["hook"] = HOOK_CASES[next_index][0]

    with pytest.raises(ValidationError):
        validate_observation(deepcopy(invalid))


@pytest.mark.parametrize(
    ("name", "hostile_value"),
    [
        ("multibyte-string", "界" * 5_462),
        ("129-sequence", list(range(129))),
        ("int64-overflow", 2**63),
        ("depth-nine", {"a": {"b": {"c": {"d": {"e": "x"}}}}}),
        (
            "node-2049",
            [{f"k{index}": index for index in range(8)} for _ in range(128)],
        ),
    ],
)
def test_observation_rejects_hostile_nested_values_at_fixed_copier_bounds(
    name: str, hostile_value: object
) -> None:
    del name
    record = _observation(
        "pre_llm_call",
        {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "sender_id": "sender",
        },
        {
            "user_message": "hello",
            "conversation_history": [],
            "is_first_turn": True,
            "model": "model",
            "platform": "cli",
        },
    )
    record["payload"]["extensions"] = {"hostile": hostile_value}

    with pytest.raises(ValueError, match=r"bridge record"):
        validate_observation(record)


def test_observation_rejects_multibyte_key_and_129_mapping() -> None:
    base = _observation(
        "on_session_start",
        {"session_id": "session"},
        {"model": "model", "platform": "cli"},
    )
    hostile_extensions = [
        {"界" * 86: True},
        {f"key-{index}": index for index in range(129)},
    ]

    for extensions in hostile_extensions:
        record = deepcopy(base)
        record["payload"]["extensions"] = extensions
        with pytest.raises(ValueError, match=r"bridge record"):
            validate_observation(record)


def test_record_json_limits_run_before_pydantic_or_frozen_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = validate_observation(
        _observation(
            "pre_llm_call",
            {
                "session_id": "session",
                "task_id": "task",
                "turn_id": "turn",
                "sender_id": "sender",
            },
            {
                "user_message": "hello",
                "conversation_history": [],
                "is_first_turn": True,
                "model": "model",
                "platform": "cli",
            },
        )
    )
    canonical = record.canonical_json().encode("utf-8")

    def with_extensions(value: object) -> bytes:
        data = record.model_dump(mode="json")
        data["payload"]["extensions"] = {"hostile": value}
        return json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    deep: object = "leaf"
    for _ in range(5):
        deep = {"child": deep}
    node_overflow = [{f"key-{item}": item for item in range(8)} for _ in range(128)]
    hostile = {
        "oversized": b" " * (MAX_BRIDGE_RECORD_BYTES + 1),
        "deep": with_extensions(deep),
        "node-overflow": with_extensions(node_overflow),
        "duplicate": canonical.replace(
            b'"schema_version":1',
            b'"schema_version":1,"schema_version":1',
            1,
        ),
        "nonfinite": canonical.replace(
            b'"captured_monotonic_ns":1',
            b'"captured_monotonic_ns":NaN',
            1,
        ),
        "lone-surrogate": canonical.replace(
            b'"model":"model"', b'"model":"\\ud800"', 1
        ),
        "invalid-utf8": b"\xff",
    }

    class PydanticMustNotRun:
        called = False

        def validate_json(self, *_args: object, **_kwargs: object) -> object:
            self.called = True
            raise AssertionError("Pydantic ran before fixed JSON limits")

    for helper, adapter_name in (
        (validate_observation_json, "_OBSERVATION_ADAPTER"),
        (validate_bridge_record_json, "_BRIDGE_RECORD_ADAPTER"),
    ):
        for name, payload in hostile.items():
            probe = PydanticMustNotRun()
            with monkeypatch.context() as scoped:
                scoped.setattr(models_module, adapter_name, probe)
                with pytest.raises(ValueError, match="bridge record"):
                    helper(payload)
            assert probe.called is False, name


def test_record_json_byte_ceiling_accepts_the_exact_utf8_boundary() -> None:
    record = validate_observation(
        _observation(
            "on_session_start",
            {"session_id": "boundary"},
            {"model": "界", "platform": "cli"},
        )
    )
    canonical = record.canonical_json().encode("utf-8")
    at_limit = canonical + b" " * (MAX_BRIDGE_RECORD_BYTES - len(canonical))

    assert len(at_limit) == MAX_BRIDGE_RECORD_BYTES
    assert validate_observation_json(at_limit) == record
    assert validate_bridge_record_json(at_limit) == record
    for helper in (validate_observation_json, validate_bridge_record_json):
        with pytest.raises(ValueError, match="byte limit"):
            helper(at_limit + b" ")


def test_coverage_samples_and_negotiated_record_bytes_have_fixed_ceilings() -> None:
    with pytest.raises(ValidationError, match="coverage channel"):
        CoverageV1(
            policy_version="copy-v1",
            capture_coverage="partial",
            channels={f"sample-{index}": True for index in range(65)},
        )
    with pytest.raises(ValidationError, match="max_record_bytes"):
        ProtocolLimitsV1(max_record_bytes=MAX_BRIDGE_RECORD_BYTES + 1)


@pytest.mark.parametrize(
    "counter",
    [
        "redacted_paths",
        "truncated_paths",
        "unsupported_paths",
        "omitted_nodes",
    ],
)
def test_full_coverage_rejects_any_recorded_omission(counter: str) -> None:
    with pytest.raises(ValidationError, match="full capture coverage"):
        CoverageV1(
            policy_version="copy-v1",
            capture_coverage="full",
            **{counter: 1},
        )


def test_capabilities_report_unobserved_streams_and_transport_limits() -> None:
    capabilities = CapabilityProfileV1().model_dump(mode="json")

    assert capabilities["chunk_capture"] == "unobserved"
    assert capabilities["reasoning_capture"] == "unobserved"
    assert capabilities["bridge_close_recovery"] == "opaque_token_v1"
    assert capabilities["limits"]["max_response_bytes"] == 1_048_576
    assert (
        capabilities["limits"]["max_response_bytes"]
        <= models_module.MAX_PROTOCOL_BYTES
    )
    assert capabilities["limits"]["max_request_bytes"] == (
        models_module.TASK6_MAX_DETACHED_RECORD_BYTES
        + models_module.MAX_BRIDGE_COMMIT_ENVELOPE_BYTES
    )
    assert capabilities["limits"]["max_record_bytes"] == (
        models_module.TASK6_MAX_DETACHED_RECORD_BYTES
    )
    assert capabilities["limits"]["max_frame_bytes"] == 65_536
    assert capabilities["limits"]["max_concurrent_connections"] == 64
    assert capabilities["limits"]["unauthenticated_idle_timeout_ms"] == 5_000


@pytest.mark.parametrize("cause", ["invalid_source_schema", "callback_internal"])
def test_new_gap_contract_causes_are_accepted(cause: str) -> None:
    valid = BridgeGapV1(
        bridge_instance_id=new_id(),
        first_capture_seq=1,
        last_capture_seq=1,
        dropped_count=1,
        cause_counts={cause: 1},
    )
    assert valid.cause_counts == {cause: 1}


def test_gap_causes_are_closed_and_remain_bounded_to_sixteen_kinds() -> None:
    with pytest.raises(ValidationError, match="fixed cause enum"):
        BridgeGapV1(
            bridge_instance_id=new_id(),
            first_capture_seq=1,
            last_capture_seq=1,
            dropped_count=1,
            cause_counts={"raw_exception_name": 1},
        )

    sorted_causes = sorted(GAP_CAUSES)
    assert {"invalid_source_schema", "callback_internal"} <= GAP_CAUSES
    assert len(sorted_causes) == 18
    sixteen_causes = {cause: 1 for cause in sorted_causes[:16]}
    valid = BridgeGapV1(
        bridge_instance_id=new_id(),
        first_capture_seq=1,
        last_capture_seq=16,
        dropped_count=16,
        cause_counts=sixteen_causes,
    )
    assert len(valid.cause_counts) == 16
    with pytest.raises(ValidationError, match=r"bounded|fixed cause enum"):
        BridgeGapV1(
            bridge_instance_id=new_id(),
            first_capture_seq=1,
            last_capture_seq=17,
            dropped_count=17,
            cause_counts={cause: 1 for cause in sorted_causes[:17]},
        )


@pytest.mark.parametrize("name", [" ", "\t\r\n"])
def test_brain_profile_rejects_whitespace_only_names(name: str) -> None:
    with pytest.raises(ValidationError, match="non-blank"):
        BrainProfileV1(profile_key="hermes.default", name=name)


def test_brain_profile_preserves_null_nonblank_whitespace_and_name_boundary() -> None:
    assert BrainProfileV1(profile_key="null", name=None).name is None
    spaced = "  Alice  "
    assert BrainProfileV1(profile_key="spaced", name=spaced).name == spaced
    boundary = "x" * 160
    assert BrainProfileV1(profile_key="boundary", name=boundary).name == boundary
    with pytest.raises(ValidationError):
        BrainProfileV1(profile_key="too-long", name="x" * 161)
