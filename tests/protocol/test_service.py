from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.models import (
    MAX_BRIDGE_COMMIT_ENVELOPE_BYTES,
    TASK6_MAX_DETACHED_RECORD_BYTES,
    BrainProfileV1,
    CapabilityProfileV1,
    ConsciousnessFrameV2,
    CoverageV1,
    ProtocolLimitsV1,
)
from alice_brain_hermes.protocol.service import ProtocolService
from alice_brain_hermes.runtime.daemon import HermesDaemonRuntime
from alice_brain_hermes.runtime.store import SQLiteLedger

TOKEN = "a" * 64
RECOVERY_TOKEN = "ab" * 32


def bridge_attach_params(
    brain_id: str,
    bridge_instance_id: str,
    *,
    recovery_token: str = RECOVERY_TOKEN,
) -> dict[str, object]:
    return {
        "brain_id": brain_id,
        "bridge_instance_id": bridge_instance_id,
        "recovery_token": recovery_token,
    }


def request(
    request_id: int,
    method: str,
    params: dict[str, object] | None = None,
    *,
    token: object = TOKEN,
) -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
            "auth": token,
        },
        separators=(",", ":"),
    ).encode()


def decode(response: bytes) -> dict[str, object]:
    return json.loads(response)


def nested_object(depth: int) -> dict[str, object]:
    value: object = "leaf"
    for _ in range(depth):
        value = {"child": value}
    return {"root": value}


def pre_llm_observation(
    instance: str,
    capture_seq: int,
    *,
    extensions: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_kind": "observation",
        "bridge_instance_id": instance,
        "capture_seq": capture_seq,
        "captured_at": datetime.now(UTC).isoformat(),
        "captured_monotonic_ns": capture_seq,
        "source_schema_version": "hermes.observer.v1",
        "hook": "pre_llm_call",
        "context": {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "sender_id": "sender",
        },
        "payload": {
            "user_message": "hello",
            "conversation_history": [],
            "is_first_turn": True,
            "model": "model-y",
            "platform": "test",
            "extensions": extensions or {},
        },
        "coverage": CoverageV1(
            policy_version="copy-v1", capture_coverage="full"
        ).model_dump(mode="json"),
    }


@pytest.fixture
def service(tmp_path: Path):
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    service = ProtocolService(
        runtime,
        credential=TOKEN,
        instance_nonce=runtime.lease.instance_nonce,
    )
    try:
        yield service
    finally:
        runtime.close()


def initialize(connection, request_id: int = 1) -> dict[str, object]:
    return decode(
        connection.handle_frame(
            request(
                request_id,
                "initialize",
                {
                    "protocol_version": 1,
                    "capabilities": CapabilityProfileV1().model_dump(mode="json"),
                },
            )
        )
    )


def test_closed_connection_rejects_late_worker_dispatch_without_mutation(
    service: ProtocolService,
) -> None:
    connection = service.new_connection()
    assert "result" in initialize(connection)
    connection.close()

    response = decode(connection.handle_frame(request(2, "brain.create")))

    assert response["error"]["code"] == "connection_closed"
    assert service.runtime.ledger.list_brain_ids() == []


def test_close_waits_for_inflight_dispatch_before_disconnect_fence(
    service: ProtocolService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = service.new_connection()
    assert "result" in initialize(connection)
    entered = threading.Event()
    release = threading.Event()
    close_done = threading.Event()
    real_create = service.runtime.create_brain

    def blocked_create(*, name=None):
        entered.set()
        assert release.wait(timeout=2.0)
        return real_create(name=name)

    monkeypatch.setattr(service.runtime, "create_brain", blocked_create)
    responses: list[dict[str, object]] = []

    worker = threading.Thread(
        target=lambda: responses.append(
            decode(connection.handle_frame(request(2, "brain.create")))
        )
    )
    closer = threading.Thread(target=lambda: (connection.close(), close_done.set()))
    worker.start()
    assert entered.wait(timeout=2.0)
    closer.start()
    assert close_done.wait(timeout=0.05) is False
    release.set()
    worker.join(timeout=2.0)
    closer.join(timeout=2.0)

    assert worker.is_alive() is False
    assert closer.is_alive() is False
    assert "result" in responses[0]
    assert close_done.is_set()
    late = decode(connection.handle_frame(request(3, "brain.create")))
    assert late["error"]["code"] == "connection_closed"


@pytest.mark.parametrize("invalid_limits", [False, {}, object()])
def test_service_rejects_non_exact_limits_without_default_fallback(
    service: ProtocolService, invalid_limits: object
) -> None:
    with pytest.raises(TypeError, match="limits"):
        ProtocolService(
            service.runtime,
            credential=TOKEN,
            instance_nonce=service.instance_nonce,
            limits=invalid_limits,  # type: ignore[arg-type]
        )


def test_service_capabilities_and_enforcement_share_validated_limits_copy(
    service: ProtocolService,
) -> None:
    requested = ProtocolLimitsV1(max_depth=8)
    bounded = ProtocolService(
        service.runtime,
        credential=TOKEN,
        instance_nonce=service.instance_nonce,
        limits=requested,
    )

    assert bounded.limits == requested
    assert bounded.limits is not requested
    assert bounded.capabilities.limits is bounded.limits
    response = decode(
        bounded.new_connection().handle_frame(
            request(1, "health", nested_object(requested.max_depth))
        )
    )
    assert response["error"]["code"] == "invalid_request"


def test_authentication_precedes_method_and_initialization_disclosure(service) -> None:
    connection = service.new_connection()
    missing = decode(connection.handle_frame(request(1, "secret.unknown", token=None)))
    wrong = decode(connection.handle_frame(request(1, "secret.unknown", token="x")))

    assert missing == wrong
    assert missing["error"]["code"] == "unauthorized"
    assert "method" not in json.dumps(missing)


def test_authentication_hashes_one_fixed_size_candidate_path(
    service, monkeypatch
) -> None:
    from alice_brain_hermes.protocol import service as service_module

    real_sha256 = service_module.hashlib.sha256
    lengths: list[int] = []

    def measured_sha256(payload: bytes):
        lengths.append(len(payload))
        return real_sha256(payload)

    monkeypatch.setattr(service_module.hashlib, "sha256", measured_sha256)
    connection = service.new_connection()
    responses = [
        decode(connection.handle_frame(request(1, "health", token=None))),
        decode(connection.handle_frame(request(2, "health", token="x"))),
        decode(connection.handle_frame(request(3, "health", token="x" * 10_000))),
        decode(connection.handle_frame(request(4, "health", token="b" * 64))),
    ]

    assert all(item["error"]["code"] == "unauthorized" for item in responses)
    assert lengths == [64, 64, 64, 64]


@pytest.mark.parametrize(
    ("id_present", "request_id"),
    [
        (False, None),
        (True, None),
        (True, True),
        (True, 1.25),
        (True, 2**63),
        (True, "x" * 129),
    ],
    ids=["missing", "null", "boolean", "fraction", "huge", "long-string"],
)
@pytest.mark.parametrize(
    ("auth_present", "token"),
    [(False, None), (True, "b" * 64)],
    ids=["missing-token", "wrong-token"],
)
def test_authentication_precedes_request_id_validation(
    service,
    id_present: bool,
    request_id: object,
    auth_present: bool,
    token: object,
) -> None:
    body: dict[str, object] = {
        "jsonrpc": "2.0",
        "method": "health",
        "params": {},
    }
    if id_present:
        body["id"] = request_id
    if auth_present:
        body["auth"] = token

    response = decode(
        service.new_connection().handle_frame(
            json.dumps(body, separators=(",", ":")).encode()
        )
    )

    assert response["id"] is None
    assert response["error"]["code"] == "unauthorized"


def test_authorized_request_id_is_validated_after_authentication(service) -> None:
    body = {
        "jsonrpc": "2.0",
        "id": True,
        "method": "health",
        "params": {},
        "auth": TOKEN,
    }

    response = decode(
        service.new_connection().handle_frame(
            json.dumps(body, separators=(",", ":")).encode()
        )
    )

    assert response["error"]["code"] == "invalid_request"


def test_health_requires_auth_but_not_initialize(service) -> None:
    connection = service.new_connection()

    response = decode(connection.handle_frame(request(1, "health")))

    assert response["result"]["instance_nonce"] == service.instance_nonce
    assert response["result"]["runtime_ready"] is True


def test_each_connection_has_isolated_exact_initialization_state(service) -> None:
    first = service.new_connection()
    second = service.new_connection()

    assert (
        decode(first.handle_frame(request(1, "daemon.status")))["error"]["code"]
        == "not_initialized"
    )
    assert "result" in initialize(first)
    assert (
        decode(first.handle_frame(request(2, "initialize")))["error"]["code"]
        == "already_initialized"
    )
    assert (
        decode(second.handle_frame(request(1, "daemon.status")))["error"]["code"]
        == "not_initialized"
    )


def test_capability_mismatch_never_downgrades_or_initializes(service) -> None:
    connection = service.new_connection()
    incompatible = CapabilityProfileV1().model_dump(mode="json")
    incompatible["frame_schema_version"] = 999

    rejected = decode(
        connection.handle_frame(
            request(
                1,
                "initialize",
                {"protocol_version": 1, "capabilities": incompatible},
            )
        )
    )

    assert rejected["error"]["code"] == "capability_mismatch"
    assert (
        decode(connection.handle_frame(request(2, "daemon.status")))["error"]["code"]
        == "not_initialized"
    )


@pytest.mark.parametrize("version", [True, 1.0], ids=["boolean", "float"])
def test_initialize_requires_an_exact_integer_protocol_version(
    service: ProtocolService, version: object
) -> None:
    connection = service.new_connection()
    capabilities = CapabilityProfileV1().model_dump(mode="json")

    rejected = decode(
        connection.handle_frame(
            request(
                1,
                "initialize",
                {
                    "protocol_version": version,
                    "capabilities": capabilities,
                },
            )
        )
    )

    assert rejected["error"]["code"] == "protocol_mismatch"
    assert (
        decode(connection.handle_frame(request(2, "daemon.status")))["error"]["code"]
        == "not_initialized"
    )


def test_typed_bridge_uses_opaque_binding_and_server_provenance(service) -> None:
    connection = service.new_connection()
    initialize(connection)
    brain = decode(connection.handle_frame(request(2, "brain.create", {"name": None})))[
        "result"
    ]
    instance = new_id()
    attached = decode(
        connection.handle_frame(
            request(
                3,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], instance),
            )
        )
    )["result"]
    record = {
        "schema_version": 1,
        "record_kind": "observation",
        "bridge_instance_id": instance,
        "capture_seq": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "captured_monotonic_ns": 1,
        "source_schema_version": "hermes.observer.v1",
        "hook": "pre_tool_call",
        "context": {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "api_request_id": "request",
            "tool_call_id": "tool-call",
        },
        "payload": {
            "tool_name": "shell",
            "args": {"command": "full command"},
            "middleware_trace": [],
            "extensions": {"provider_options": {"reasoning": "high"}},
        },
        "coverage": CoverageV1(
            policy_version="copy-v1", capture_coverage="full"
        ).model_dump(mode="json"),
    }

    committed = decode(
        connection.handle_frame(
            request(
                4,
                "bridge.commit",
                {"binding": attached["binding"], "record": record},
            )
        )
    )

    assert committed["result"]["through_capture_seq"] == 1
    stored = service.runtime.ledger.list_events(brain["brain_id"])[-1]
    assert stored.actor_id == brain["brain_id"]
    assert stored.adapter_id == service.server_adapter_id
    assert stored.payload["payload"]["args"] == {"command": "full command"}

    untrusted = {**record, "capture_seq": 2, "actor_id": new_id()}
    rejected = decode(
        connection.handle_frame(
            request(
                5,
                "bridge.commit",
                {"binding": attached["binding"], "record": untrusted},
            )
        )
    )
    assert rejected["error"]["code"] == "invalid_params"


def test_bridge_close_recovery_is_cross_connection_and_fail_closed(service) -> None:
    first_connection = service.new_connection()
    initialize(first_connection)
    brain = decode(
        first_connection.handle_frame(request(2, "brain.create", {"name": None}))
    )["result"]
    instance = new_id()
    binding = decode(
        first_connection.handle_frame(
            request(
                3,
                "brain.attach",
                {
                    "brain_id": brain["brain_id"],
                    "bridge_instance_id": instance,
                    "recovery_token": RECOVERY_TOKEN,
                },
            )
        )
    )["result"]["binding"]
    closed = decode(
        first_connection.handle_frame(
            request(
                4,
                "bridge.close",
                {"binding": binding, "final_capture_seq": 0},
            )
        )
    )["result"]

    recovered_connection = service.new_connection()
    initialize(recovered_connection)

    def recover(
        request_id: int,
        *,
        brain_id: str = brain["brain_id"],
        bridge_instance_id: str = instance,
        recovery_token: str = RECOVERY_TOKEN,
        final_capture_seq: int = 0,
    ) -> dict[str, object]:
        return decode(
            recovered_connection.handle_frame(
                request(
                    request_id,
                    "bridge.close.recover",
                    {
                        "brain_id": brain_id,
                        "bridge_instance_id": bridge_instance_id,
                        "recovery_token": recovery_token,
                        "final_capture_seq": final_capture_seq,
                    },
                )
            )
        )

    recovered = recover(2)
    other_brain = decode(
        recovered_connection.handle_frame(request(3, "brain.create", {"name": None}))
    )["result"]
    wrong_token_and_sequence = recover(
        4,
        recovery_token="cd" * 32,
        final_capture_seq=1,
    )
    wrong_brain = recover(5, brain_id=other_brain["brain_id"])
    wrong_bridge = recover(6, bridge_instance_id=new_id())
    wrong_sequence = recover(7, final_capture_seq=1)

    assert recovered["result"] == closed
    assert wrong_token_and_sequence["error"]["code"] == "invalid_binding"
    assert wrong_brain["error"]["code"] == "invalid_binding"
    assert wrong_bridge["error"]["code"] == "invalid_binding"
    assert wrong_sequence["error"]["code"] == "capture_sequence_error"
    assert (
        decode(
            recovered_connection.handle_frame(
                request(8, "state.get", {"binding": binding})
            )
        )["error"]["code"]
        == "invalid_binding"
    )
    assert (
        service.runtime.ledger.bridge_stream_state(instance).model_dump(mode="json")
        == closed
    )


def test_attach_reports_clean_closed_and_abandoned_as_distinct_typed_states(
    service: ProtocolService,
) -> None:
    connection = service.new_connection()
    initialize(connection)
    brain = decode(connection.handle_frame(request(2, "brain.create", {"name": None})))[
        "result"
    ]

    clean_instance = new_id()
    clean_binding = decode(
        connection.handle_frame(
            request(
                3,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], clean_instance),
            )
        )
    )["result"]["binding"]
    assert "result" in decode(
        connection.handle_frame(
            request(
                4,
                "bridge.close",
                {"binding": clean_binding, "final_capture_seq": 0},
            )
        )
    )

    clean_retry = service.new_connection()
    initialize(clean_retry)
    clean_error = decode(
        clean_retry.handle_frame(
            request(
                2,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], clean_instance),
            )
        )
    )["error"]

    abandoned_instance = new_id()
    abandoned_connection = service.new_connection()
    initialize(abandoned_connection)
    assert "result" in decode(
        abandoned_connection.handle_frame(
            request(
                2,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], abandoned_instance),
            )
        )
    )
    abandoned_connection.close()
    service.runtime.engine(brain["brain_id"]).abandon_bridge_stream(abandoned_instance)
    abandoned_retry = service.new_connection()
    initialize(abandoned_retry)
    abandoned_error = decode(
        abandoned_retry.handle_frame(
            request(
                2,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], abandoned_instance),
            )
        )
    )["error"]

    assert clean_error == {
        "code": "bridge_clean_closed",
        "message": "bridge stream was cleanly closed",
        "data": {"status": "clean_closed"},
    }
    assert abandoned_error == {
        "code": "bridge_abandoned",
        "message": "bridge stream was abandoned",
        "data": {"status": "abandoned"},
    }


def test_brain_resolve_reuses_one_server_persisted_stable_profile(service) -> None:
    first = service.new_connection()
    second = service.new_connection()
    initialize(first)
    initialize(second)
    profile = BrainProfileV1(profile_key="hermes.default", name="Alice").model_dump(
        mode="json"
    )

    one = decode(first.handle_frame(request(2, "brain.resolve", {"profile": profile})))
    two = decode(second.handle_frame(request(2, "brain.resolve", {"profile": profile})))

    assert one["result"]["brain_id"] == two["result"]["brain_id"]
    assert one["result"]["created"] is True
    assert two["result"]["created"] is False


def test_brain_resolve_rejects_whitespace_name_and_accepts_null(service) -> None:
    connection = service.new_connection()
    initialize(connection)

    rejected = decode(
        connection.handle_frame(
            request(
                2,
                "brain.resolve",
                {
                    "profile": {
                        "schema_version": 1,
                        "profile_key": "whitespace.invalid",
                        "name": " \t ",
                    }
                },
            )
        )
    )
    accepted = decode(
        connection.handle_frame(
            request(
                3,
                "brain.resolve",
                {
                    "profile": {
                        "schema_version": 1,
                        "profile_key": "null.valid",
                        "name": None,
                    }
                },
            )
        )
    )

    assert rejected["error"]["code"] == "invalid_params"
    assert accepted["result"]["created"] is True


def test_public_brain_name_bound_matches_identity_domain(service) -> None:
    with pytest.raises(ValueError):
        BrainProfileV1(profile_key="too-long", name="x" * 161)
    assert BrainProfileV1(profile_key="boundary", name="x" * 160).name == ("x" * 160)

    connection = service.new_connection()
    initialize(connection)
    rejected = decode(
        connection.handle_frame(request(2, "brain.create", {"name": "x" * 161}))
    )
    accepted = decode(
        connection.handle_frame(request(3, "brain.create", {"name": "x" * 160}))
    )

    assert rejected["error"]["code"] == "invalid_params"
    assert accepted["result"]["state_sequence"] == 1


def test_state_get_preserves_latest_capture_coverage_and_samples_scheduler(
    service,
) -> None:
    connection = service.new_connection()
    initialize(connection)
    brain = decode(
        connection.handle_frame(request(2, "brain.create", {"name": "Alice"}))
    )["result"]
    instance = new_id()
    binding = decode(
        connection.handle_frame(
            request(
                3,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], instance),
            )
        )
    )["result"]["binding"]
    record = {
        "schema_version": 1,
        "record_kind": "observation",
        "bridge_instance_id": instance,
        "capture_seq": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "captured_monotonic_ns": 1,
        "source_schema_version": "hermes.observer.v1",
        "hook": "post_api_request",
        "context": {
            "session_id": "session",
            "task_id": "task",
            "turn_id": "turn",
            "api_request_id": "request",
        },
        "payload": {
            "platform": "test",
            "model": "model-y",
            "provider": "provider-x",
            "base_url": None,
            "api_mode": "streaming",
            "api_call_count": 1,
            "api_duration": 0.1,
            "started_at": "start",
            "ended_at": "end",
            "finish_reason": "stop",
            "message_count": 1,
            "response_model": "model-y",
            "response": {},
            "usage": {},
            "assistant_message": {},
            "assistant_content_chars": 0,
            "assistant_tool_call_count": 0,
        },
        "coverage": CoverageV1(
            policy_version="host-copy-v1",
            capture_coverage="host_sanitized",
            redacted_paths=2,
        ).model_dump(mode="json"),
    }
    committed = decode(
        connection.handle_frame(
            request(4, "bridge.commit", {"binding": binding, "record": record})
        )
    )["result"]

    running = decode(
        connection.handle_frame(request(5, "state.get", {"binding": binding}))
    )["result"]

    assert running["through_capture_seq"] == 1
    assert running["capture_coverage"] == committed["frame"]["capture_coverage"]
    assert running["freshness"]["stream_connection"] == "connected"
    assert running["freshness"]["scheduler_sample"] == "running"

    service.runtime.scheduler(brain["brain_id"]).stop()
    stopped = decode(
        connection.handle_frame(request(6, "state.get", {"binding": binding}))
    )["result"]
    assert stopped["capture_coverage"] == running["capture_coverage"]
    assert stopped["freshness"]["scheduler_sample"] == "stopped"


@pytest.mark.parametrize(
    "frame",
    [
        b"[]",
        b'{"jsonrpc":"2.0","id":1,"id":2,"method":"health","params":{},"auth":"'
        + TOKEN.encode()
        + b'"}',
        b'{"jsonrpc":"2.0","id":1,"method":"health","params":{"x":NaN},"auth":"'
        + TOKEN.encode()
        + b'"}',
        b"\xff",
    ],
)
def test_strict_json_rejects_batches_duplicates_nonfinite_and_invalid_utf8(
    service, frame: bytes
) -> None:
    response = decode(service.new_connection().handle_frame(frame))
    assert response["error"]["code"] == "invalid_request"


def test_request_byte_and_depth_limits_are_stable_and_recoverable(service) -> None:
    connection = service.new_connection()
    oversized = b"{" + b"x" * service.limits.max_request_bytes + b"}"
    deep: object = None
    for _ in range(service.limits.max_depth + 1):
        deep = {"x": deep}

    assert (
        decode(connection.handle_frame(oversized))["error"]["code"]
        == "request_too_large"
    )
    assert (
        decode(connection.handle_frame(request(2, "health", {"deep": deep})))["error"][
            "code"
        ]
        == "invalid_request"
    )
    assert "result" in decode(connection.handle_frame(request(3, "health")))


def test_parser_recursion_and_extreme_ids_are_invalid_and_recoverable(service) -> None:
    connection = service.new_connection()
    recursive = b"[" * 2_000 + b"0" + b"]" * 2_000
    huge_id = (
        b'{"jsonrpc":"2.0","id":'
        + b"9" * 200
        + b',"method":"health","params":{},"auth":"'
        + TOKEN.encode()
        + b'"}'
    )
    lone_surrogate = (
        b'{"jsonrpc":"2.0","id":1,"method":"health","params":'
        b'{"value":"\\ud800"},"auth":"' + TOKEN.encode() + b'"}'
    )

    for frame in (recursive, huge_id, lone_surrogate):
        assert decode(connection.handle_frame(frame))["error"]["code"] == (
            "invalid_request"
        )
        assert "result" in decode(connection.handle_frame(request(1, "health")))


@pytest.mark.parametrize(
    "bad_result",
    [
        {"value": float("nan")},
        {"value": object()},
        {"value": "\ud800"},
    ],
)
def test_service_result_serialization_faults_are_generic_internal_errors(
    service, monkeypatch, bad_result: dict[str, object]
) -> None:
    connection = service.new_connection()
    monkeypatch.setattr(
        connection, "_dispatch", lambda _method, _params, _budget: bad_result
    )

    response = decode(connection.handle_frame(request(1, "health")))

    assert response["error"]["code"] == "internal_error"
    assert response["error"]["message"] == "internal request failure"
    assert "object at" not in json.dumps(response)
    assert "surrogate" not in json.dumps(response)


def test_oversized_service_result_uses_bounded_protocol_fault(
    service, monkeypatch
) -> None:
    connection = service.new_connection()
    monkeypatch.setattr(
        connection,
        "_dispatch",
        lambda _method, _params, _budget: {
            "value": "x" * service.limits.max_response_bytes
        },
    )

    response = decode(connection.handle_frame(request(1, "health")))

    assert response["error"]["code"] == "response_too_large"
    assert len(json.dumps(response).encode()) < service.limits.max_response_bytes


def test_internal_dispatch_failure_never_discloses_exception_text(
    service, monkeypatch
) -> None:
    connection = service.new_connection()

    def fail(_method, _params, _budget):
        raise RuntimeError("sensitive-provider-token")

    monkeypatch.setattr(connection, "_dispatch", fail)
    response = decode(connection.handle_frame(request(1, "health")))

    assert response["error"]["code"] == "internal_error"
    assert "sensitive-provider-token" not in json.dumps(response)


def test_negotiated_frame_limit_rejects_before_bridge_commit_without_writes(
    service,
) -> None:
    limits = ProtocolLimitsV1(max_frame_bytes=4_096)
    bounded = ProtocolService(
        service.runtime,
        credential=TOKEN,
        instance_nonce=service.instance_nonce,
        limits=limits,
    )
    connection = bounded.new_connection()
    initialized = decode(
        connection.handle_frame(
            request(
                1,
                "initialize",
                {
                    "protocol_version": 1,
                    "capabilities": CapabilityProfileV1(limits=limits).model_dump(
                        mode="json"
                    ),
                },
            )
        )
    )
    assert "result" in initialized
    brain = decode(connection.handle_frame(request(2, "brain.create", {"name": None})))[
        "result"
    ]
    engine = service.runtime.engine(brain["brain_id"])
    for index in range(4):
        action_id = f"action-{index}-" + "x" * 500
        engine.append(
            new_event(
                "action.proposed",
                engine.brain_id,
                engine.actor_id,
                {"action_id": action_id, "intent": {}},
            )
        )
    instance = new_id()
    binding = decode(
        connection.handle_frame(
            request(
                3,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], instance),
            )
        )
    )["result"]["binding"]
    record = pre_llm_observation(instance, 1)
    events_before = service.runtime.ledger.list_events(engine.brain_id, limit=100)

    rejected = decode(
        connection.handle_frame(
            request(4, "bridge.commit", {"binding": binding, "record": record})
        )
    )

    assert rejected["error"]["code"] == "frame_too_large"
    assert service.runtime.ledger.list_events(engine.brain_id, limit=100) == (
        events_before
    )
    assert service.runtime.ledger.bridge_stream_state(instance).next_capture_seq == 1
    assert (
        service.runtime.ledger._connection.execute(
            "SELECT COUNT(*) FROM bridge_record"
        ).fetchone()[0]
        == 0
    )


def test_full_success_response_limit_rejects_before_bridge_commit(
    service, monkeypatch: pytest.MonkeyPatch
) -> None:
    limits = ProtocolLimitsV1(
        max_response_bytes=4_096,
        max_frame_bytes=20_000,
    )
    bounded = ProtocolService(
        service.runtime,
        credential=TOKEN,
        instance_nonce=service.instance_nonce,
        limits=limits,
    )
    connection = bounded.new_connection()
    assert "result" in decode(
        connection.handle_frame(
            request(
                1,
                "initialize",
                {
                    "protocol_version": 1,
                    "capabilities": CapabilityProfileV1(limits=limits).model_dump(
                        mode="json"
                    ),
                },
            )
        )
    )
    brain = decode(connection.handle_frame(request(2, "brain.create", {"name": None})))[
        "result"
    ]
    instance = new_id()
    binding = decode(
        connection.handle_frame(
            request(
                3,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], instance),
            )
        )
    )["result"]["binding"]
    original_projection = SQLiteLedger._project_bridge_frame

    def large_projection(*args, **kwargs) -> ConsciousnessFrameV2:
        frame = original_projection(*args, **kwargs)
        values = frame.model_dump(mode="python")
        values["semantic_context"] = {
            "available": False,
            "request_id": None,
            "turn_id": None,
            "reason": "x" * 5_000,
        }
        return ConsciousnessFrameV2.model_validate(values)

    monkeypatch.setattr(
        SQLiteLedger, "_project_bridge_frame", staticmethod(large_projection)
    )
    engine = service.runtime.engine(brain["brain_id"])
    events_before = service.runtime.ledger.list_events(engine.brain_id, limit=100)
    record = pre_llm_observation(instance, 1)

    rejected = decode(
        connection.handle_frame(
            request(4, "bridge.commit", {"binding": binding, "record": record})
        )
    )

    assert rejected["error"]["code"] == "response_too_large"
    assert service.runtime.ledger.list_events(engine.brain_id, limit=100) == (
        events_before
    )
    assert service.runtime.ledger.bridge_stream_state(instance).next_capture_seq == 1
    assert (
        service.runtime.ledger._connection.execute(
            "SELECT COUNT(*) FROM bridge_record"
        ).fetchone()[0]
        == 0
    )

    connection.close()
    generous_limits = limits.model_copy(update={"max_response_bytes": 20_000})
    generous = ProtocolService(
        service.runtime,
        credential=TOKEN,
        instance_nonce=service.instance_nonce,
        limits=generous_limits,
    )
    retry = generous.new_connection()
    assert "result" in decode(
        retry.handle_frame(
            request(
                5,
                "initialize",
                {
                    "protocol_version": 1,
                    "capabilities": CapabilityProfileV1(
                        limits=generous_limits
                    ).model_dump(mode="json"),
                },
            )
        )
    )
    retry_binding = decode(
        retry.handle_frame(
            request(
                6,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], instance),
            )
        )
    )["result"]["binding"]
    accepted_wire = retry.handle_frame(
        request(
            7,
            "bridge.commit",
            {"binding": retry_binding, "record": record},
        )
    )
    assert "result" in decode(accepted_wire)
    assert len(accepted_wire) <= generous_limits.max_response_bytes
    assert service.runtime.ledger.bridge_stream_state(instance).next_capture_seq == 2


def test_ack_larger_than_legacy_64k_persists_when_negotiated_wire_fits(
    service, monkeypatch: pytest.MonkeyPatch
) -> None:
    limits = ProtocolLimitsV1(
        max_response_bytes=200_000,
        max_frame_bytes=100_000,
    )
    bounded = ProtocolService(
        service.runtime,
        credential=TOKEN,
        instance_nonce=service.instance_nonce,
        limits=limits,
    )
    connection = bounded.new_connection()
    assert "result" in decode(
        connection.handle_frame(
            request(
                1,
                "initialize",
                {
                    "protocol_version": 1,
                    "capabilities": CapabilityProfileV1(limits=limits).model_dump(
                        mode="json"
                    ),
                },
            )
        )
    )
    brain = decode(connection.handle_frame(request(2, "brain.create", {"name": None})))[
        "result"
    ]
    instance = new_id()
    binding = decode(
        connection.handle_frame(
            request(
                3,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], instance),
            )
        )
    )["result"]["binding"]
    original_projection = SQLiteLedger._project_bridge_frame

    def large_projection(*args, **kwargs) -> ConsciousnessFrameV2:
        frame = original_projection(*args, **kwargs)
        values = frame.model_dump(mode="python")
        values["semantic_context"] = {
            "available": False,
            "request_id": None,
            "turn_id": None,
            "reason": "x" * 70_000,
        }
        return ConsciousnessFrameV2.model_validate(values)

    monkeypatch.setattr(
        SQLiteLedger, "_project_bridge_frame", staticmethod(large_projection)
    )
    record = pre_llm_observation(instance, 1)

    encoded_response = connection.handle_frame(
        request(4, "bridge.commit", {"binding": binding, "record": record})
    )
    committed = decode(encoded_response)["result"]

    assert (
        65_536
        < len(json.dumps(committed, separators=(",", ":"), sort_keys=True).encode())
        < limits.max_response_bytes
    )
    assert len(encoded_response) <= limits.max_response_bytes
    assert committed["frame"]["semantic_context"]["reason"] == "x" * 70_000
    assert service.runtime.ledger.bridge_stream_state(instance).next_capture_seq == 2


@pytest.mark.parametrize(
    ("limits", "extensions"),
    [
        (
            ProtocolLimitsV1(
                max_request_bytes=400_000,
            ),
            {"blob": "x" * 16_000},
        ),
        (ProtocolLimitsV1(max_depth=64), nested_object(3)),
        (
            ProtocolLimitsV1(max_nodes=30_000),
            {"items": [0] * 100},
        ),
    ],
    ids=["bytes", "depth", "nodes"],
)
def test_bridge_record_has_no_hidden_limits_below_negotiated_profile(
    service,
    limits: ProtocolLimitsV1,
    extensions: dict[str, object],
) -> None:
    bounded = ProtocolService(
        service.runtime,
        credential=TOKEN,
        instance_nonce=service.instance_nonce,
        limits=limits,
    )
    connection = bounded.new_connection()
    assert "result" in decode(
        connection.handle_frame(
            request(
                1,
                "initialize",
                {
                    "protocol_version": 1,
                    "capabilities": CapabilityProfileV1(limits=limits).model_dump(
                        mode="json"
                    ),
                },
            )
        )
    )
    brain = decode(connection.handle_frame(request(2, "brain.create", {"name": None})))[
        "result"
    ]
    instance = new_id()
    binding = decode(
        connection.handle_frame(
            request(
                3,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], instance),
            )
        )
    )["result"]["binding"]
    record = pre_llm_observation(instance, 1, extensions=extensions)

    committed = decode(
        connection.handle_frame(
            request(4, "bridge.commit", {"binding": binding, "record": record})
        )
    )

    assert "result" in committed
    assert committed["result"]["through_capture_seq"] == 1


def test_protocol_limits_fit_task6_record_inside_complete_commit_envelope() -> None:
    limits = ProtocolLimitsV1()

    assert limits.max_record_bytes == TASK6_MAX_DETACHED_RECORD_BYTES
    assert limits.max_request_bytes == (
        limits.max_record_bytes + MAX_BRIDGE_COMMIT_ENVELOPE_BYTES
    )
    with pytest.raises(ValueError, match="max_request_bytes"):
        ProtocolLimitsV1(
            max_request_bytes=(
                TASK6_MAX_DETACHED_RECORD_BYTES + MAX_BRIDGE_COMMIT_ENVELOPE_BYTES - 1
            )
        )


def test_near_boundary_complete_commit_succeeds_and_record_plus_one_is_atomic(
    service,
) -> None:
    limits = ProtocolLimitsV1()
    bounded = ProtocolService(
        service.runtime,
        credential=TOKEN,
        instance_nonce=service.instance_nonce,
        limits=limits,
    )
    connection = bounded.new_connection()
    assert "result" in decode(
        connection.handle_frame(
            request(
                1,
                "initialize",
                {
                    "protocol_version": 1,
                    "capabilities": CapabilityProfileV1(limits=limits).model_dump(
                        mode="json"
                    ),
                },
            )
        )
    )
    brain = decode(connection.handle_frame(request(2, "brain.create", {"name": None})))[
        "result"
    ]
    instance = new_id()
    binding = decode(
        connection.handle_frame(
            request(
                3,
                "brain.attach",
                bridge_attach_params(brain["brain_id"], instance),
            )
        )
    )["result"]["binding"]
    record = pre_llm_observation(instance, 1, extensions={"padding": []})
    encoded_record = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    padding = limits.max_record_bytes - len(encoded_record)
    assert 0 < padding <= 16 * 16_384
    chunks = ["" for _ in range(16)]
    # Empty strings already contribute their quotes and commas. Account for
    # that structural growth before distributing only payload bytes.
    record["payload"]["extensions"]["padding"] = chunks
    empty_chunks_size = len(
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    payload_bytes = limits.max_record_bytes - empty_chunks_size
    assert 0 < payload_bytes <= len(chunks) * 16_384
    for index in range(len(chunks)):
        chunk_size = min(16_384, payload_bytes)
        chunks[index] = "x" * chunk_size
        payload_bytes -= chunk_size
    assert payload_bytes == 0
    exact_record = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert len(exact_record) == limits.max_record_bytes
    exact_request = request(4, "bridge.commit", {"binding": binding, "record": record})
    assert len(exact_request) <= limits.max_request_bytes

    accepted = decode(connection.handle_frame(exact_request))

    assert accepted["result"]["through_capture_seq"] == 1
    events_after_accept = service.runtime.ledger.list_events(
        brain["brain_id"], limit=100
    )
    record["capture_seq"] = 2
    padding_chunks = record["payload"]["extensions"]["padding"]
    expandable = next(
        index for index, chunk in enumerate(padding_chunks) if len(chunk) < 16_384
    )
    padding_chunks[expandable] += "x"
    oversized_request = request(
        5, "bridge.commit", {"binding": binding, "record": record}
    )
    assert len(oversized_request) <= limits.max_request_bytes

    rejected = decode(connection.handle_frame(oversized_request))

    assert rejected["error"]["code"] == "record_too_large"
    assert (
        service.runtime.ledger.list_events(brain["brain_id"], limit=100)
        == events_after_accept
    )
    assert service.runtime.ledger.bridge_stream_state(instance).next_capture_seq == 2
