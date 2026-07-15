from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.models import (
    PROTOCOL_VERSION,
    ProtocolLimitsV1,
)
from alice_brain_hermes.protocol.service import ProtocolService
from alice_brain_hermes.runtime.daemon import HermesDaemonRuntime

TOKEN = "d" * 64


def _request(
    request_id: int,
    method: str,
    params: dict[str, object] | None = None,
) -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
            "auth": TOKEN,
        },
        separators=(",", ":"),
    ).encode()


def _decode(response: bytes) -> dict[str, object]:
    return json.loads(response)


@pytest.fixture
def service(tmp_path: Path):
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    protocol = ProtocolService(
        runtime,
        credential=TOKEN,
        instance_nonce=runtime.lease.instance_nonce,
    )
    try:
        yield protocol
    finally:
        runtime.close()


def _initialized(service: ProtocolService):
    connection = service.new_connection()
    response = _decode(
        connection.handle_frame(
            _request(
                1,
                "initialize",
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "capabilities": service.capabilities.model_dump(mode="json"),
                },
            )
        )
    )
    assert "result" in response
    return connection


def _create_brain(connection, request_id: int, name: str | None = None) -> str:
    response = _decode(
        connection.handle_frame(
            _request(request_id, "brain.create", {"name": name})
        )
    )
    return response["result"]["brain_id"]


def test_identity_get_reports_no_brain_without_fabricating_identity(
    service: ProtocolService,
) -> None:
    connection = _initialized(service)

    response = _decode(connection.handle_frame(_request(2, "identity.get")))

    assert response["error"]["code"] == "not_found"


def test_identity_get_auto_selects_exactly_one_brain_and_is_typed(
    service: ProtocolService,
) -> None:
    connection = _initialized(service)
    brain_id = _create_brain(connection, 2, "Mira")

    response = _decode(connection.handle_frame(_request(3, "identity.get")))

    assert response["result"] == {
        "schema_version": 1,
        "brain_id": brain_id,
        "self_actor_id": brain_id,
        "name": "Mira",
        "state_sequence": 1,
        "actors": [
            {
                "actor_id": brain_id,
                "kind": "self",
                "display_name": None,
                "parent_actor_id": None,
                "attributes": {},
            }
        ],
        "authorizations": [],
    }


def test_identity_get_requires_selection_for_multiple_brains(
    service: ProtocolService,
) -> None:
    connection = _initialized(service)
    first = _create_brain(connection, 2, "First")
    second = _create_brain(connection, 3, "Second")

    ambiguous = _decode(connection.handle_frame(_request(4, "identity.get")))
    selected = _decode(
        connection.handle_frame(
            _request(5, "identity.get", {"brain_id": second})
        )
    )
    missing = _decode(
        connection.handle_frame(
            _request(6, "identity.get", {"brain_id": new_id()})
        )
    )

    assert ambiguous["error"]["code"] == "brain_id_required"
    assert ambiguous["error"]["data"] == {"brain_count": 2}
    assert selected["result"]["brain_id"] == second
    assert selected["result"]["brain_id"] != first
    assert missing["error"]["code"] == "not_found"


def test_trace_list_is_ordered_cursor_paged_and_truthful(
    service: ProtocolService,
) -> None:
    connection = _initialized(service)
    brain_id = _create_brain(connection, 2, None)
    engine = service.runtime.engine(brain_id)
    for index in range(3):
        engine.append(
            new_event(
                "capabilities.reported",
                brain_id,
                brain_id,
                {"capabilities": {f"capability-{index}": "observed"}},
            )
        )

    first = _decode(
        connection.handle_frame(
            _request(3, "trace.list", {"brain_id": brain_id, "limit": 2})
        )
    )["result"]
    second = _decode(
        connection.handle_frame(
            _request(
                4,
                "trace.list",
                {
                    "brain_id": brain_id,
                    "after_sequence": first["next_after_sequence"],
                    "limit": 3,
                },
            )
        )
    )["result"]

    assert first["schema_version"] == 1
    assert first["brain_id"] == brain_id
    assert first["after_sequence"] == 0
    assert first["requested_limit"] == 2
    assert first["returned_count"] == 2
    assert [event["sequence"] for event in first["events"]] == [1, 2]
    assert first["next_after_sequence"] == 2
    assert first["has_more"] is True
    assert first["byte_limited"] is False
    assert first["blocked_event_sequence"] is None

    assert [event["sequence"] for event in second["events"]] == [3, 4]
    assert second["next_after_sequence"] == 4
    assert second["returned_count"] == 2
    assert second["has_more"] is False
    assert second["byte_limited"] is False
    assert second["blocked_event_sequence"] is None


@pytest.mark.parametrize(
    ("params", "code"),
    [
        ({"after_sequence": True}, "invalid_params"),
        ({"after_sequence": -1}, "invalid_params"),
        ({"limit": True}, "invalid_params"),
        ({"limit": 0}, "invalid_params"),
        ({"limit": 1001}, "invalid_params"),
        ({"unknown": 1}, "invalid_params"),
    ],
)
def test_trace_list_rejects_ambiguous_or_unbounded_params(
    service: ProtocolService,
    params: dict[str, object],
    code: str,
) -> None:
    connection = _initialized(service)
    _create_brain(connection, 2, None)

    response = _decode(connection.handle_frame(_request(3, "trace.list", params)))

    assert response["error"]["code"] == code


def test_trace_list_stops_before_response_budget_without_skipping_event(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    limits = ProtocolLimitsV1(max_response_bytes=4_096)
    service = ProtocolService(
        runtime,
        credential=TOKEN,
        instance_nonce=runtime.lease.instance_nonce,
        limits=limits,
    )
    try:
        connection = _initialized(service)
        brain_id = _create_brain(connection, 2, None)
        engine = runtime.engine(brain_id)
        for index in range(12):
            engine.append(
                new_event(
                    "capabilities.reported",
                    brain_id,
                    brain_id,
                    {
                        "capabilities": {
                            f"key-{index}": "x" * 512,
                        }
                    },
                )
            )

        wire = connection.handle_frame(
            _request(3, "trace.list", {"brain_id": brain_id, "limit": 100})
        )
        page = _decode(wire)["result"]

        assert len(wire) <= limits.max_response_bytes
        assert 0 < page["returned_count"] < 13
        assert page["byte_limited"] is True
        assert page["has_more"] is True
        assert page["blocked_event_sequence"] == page["next_after_sequence"] + 1
        assert [event["sequence"] for event in page["events"]] == list(
            range(1, page["next_after_sequence"] + 1)
        )
    finally:
        runtime.close()


def test_trace_list_reports_a_single_unreturnable_event_without_advancing_cursor(
    service: ProtocolService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _initialized(service)
    brain_id = _create_brain(connection, 2, None)
    event = new_event(
        "capabilities.reported",
        brain_id,
        brain_id,
        {"capabilities": {"large": "x" * 20_000}},
    ).model_copy(update={"sequence": 1}).revalidated()

    monkeypatch.setattr(
        service.runtime.ledger,
        "list_events",
        lambda *_args, **_kwargs: [event],
    )
    original_limits = service.limits
    service.limits = original_limits.model_copy(update={"max_response_bytes": 4_096})
    try:
        response = _decode(
            connection.handle_frame(
                _request(3, "trace.list", {"brain_id": brain_id, "limit": 1})
            )
        )
    finally:
        service.limits = original_limits

    page = response["result"]
    assert page["events"] == []
    assert page["returned_count"] == 0
    assert page["next_after_sequence"] == 0
    assert page["has_more"] is True
    assert page["byte_limited"] is True
    assert page["blocked_event_sequence"] == 1
