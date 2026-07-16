from __future__ import annotations

import io
import json
import os
import socket
from pathlib import Path

import pytest

from alice_brain_hermes.errors import DaemonClientError, DaemonRpcError
from alice_brain_hermes.protocol.client import DaemonClient
from alice_brain_hermes.protocol.models import (
    PROTOCOL_VERSION,
    SERVER_ADAPTER_ID,
    CapabilityProfileV1,
    DaemonDiscoveryV2,
    LoopbackEndpointV1,
    ProtocolLimitsV1,
)
from alice_brain_hermes.runtime.process_marker import current_process_marker


def discovery() -> DaemonDiscoveryV2:
    return DaemonDiscoveryV2(
        pid=os.getpid(),
        process_marker=current_process_marker(),
        instance_nonce="test-nonce",
        launch_nonce="test-launch",
        endpoint=LoopbackEndpointV1(port=1),
        credential_ref="credential-test-nonce.key",
    )


class FakeSocket:
    def __init__(
        self,
        *,
        family: int = socket.AF_INET,
        local: tuple[str, int] = ("127.0.0.1", 43210),
        peer: tuple[str, int] = ("127.0.0.1", 1),
    ) -> None:
        self.closed = False
        self.family = family
        self.local = local
        self.peer = peer

    def makefile(self, _mode: str):
        return io.BytesIO()

    def settimeout(self, _timeout: float) -> None:
        return None

    def getsockname(self) -> tuple[str, int]:
        return self.local

    def getpeername(self) -> tuple[str, int]:
        return self.peer

    def close(self) -> None:
        self.closed = True


class CleanupFaultSocket(FakeSocket):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.stream_close_calls = 0
        self.socket_close_calls = 0

    def makefile(self, _mode: str):
        owner = self

        class FailOnceCloseStream(io.BytesIO):
            def close(self) -> None:
                owner.stream_close_calls += 1
                super().close()
                if owner.stream_close_calls == 1:
                    raise OSError("injected stream close failure")

        return FailOnceCloseStream()

    def close(self) -> None:
        self.socket_close_calls += 1
        self.closed = True
        if self.socket_close_calls == 1:
            raise OSError("injected socket close failure")


def rotated_discovery(nonce: str, port: int) -> DaemonDiscoveryV2:
    return DaemonDiscoveryV2(
        pid=os.getpid(),
        process_marker=current_process_marker(),
        instance_nonce=nonce,
        launch_nonce="test-launch",
        endpoint=LoopbackEndpointV1(port=port),
        credential_ref=f"credential-{nonce}.key",
    )


def discovery_health(record: DaemonDiscoveryV2) -> dict[str, object]:
    return {
        "pid": record.pid,
        "instance_nonce": record.instance_nonce,
        "launch_nonce": record.launch_nonce,
        "process_marker": record.process_marker,
        "protocol_version": record.protocol_version,
        "package_version": record.package_version,
    }


@pytest.mark.parametrize(
    "timeout_seconds",
    [True, False, 0, -1, float("nan"), float("inf"), float("-inf"), 301],
    ids=[
        "true",
        "false",
        "zero",
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "over-bound",
    ],
)
def test_connect_rejects_invalid_timeout_before_discovery_or_socket_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    timeout_seconds: object,
) -> None:
    discovery_calls = 0
    socket_calls = 0

    def forbidden_discovery(_home):
        nonlocal discovery_calls
        discovery_calls += 1
        raise AssertionError("invalid timeout must fail before discovery")

    def forbidden_socket(*_args, **_kwargs):
        nonlocal socket_calls
        socket_calls += 1
        raise AssertionError("invalid timeout must fail before socket work")

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential",
        forbidden_discovery,
    )
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection",
        forbidden_socket,
    )

    with pytest.raises(ValueError, match="timeout_seconds"):
        DaemonClient.connect(tmp_path, timeout_seconds=timeout_seconds)  # type: ignore[arg-type]

    assert discovery_calls == 0
    assert socket_calls == 0


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("limits", False),
        ("limits", {}),
        ("limits", object()),
        ("initialize", 0),
        ("initialize", 1),
    ],
    ids=[
        "false-limits",
        "empty-mapping-limits",
        "object-limits",
        "zero-initialize",
        "one-initialize",
    ],
)
def test_connect_rejects_non_exact_options_before_discovery_or_socket_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argument: str,
    value: object,
) -> None:
    discovery_calls = 0
    socket_calls = 0

    def forbidden_discovery(_home):
        nonlocal discovery_calls
        discovery_calls += 1
        raise AssertionError("invalid options must fail before discovery")

    def forbidden_socket(*_args, **_kwargs):
        nonlocal socket_calls
        socket_calls += 1
        raise AssertionError("invalid options must fail before socket work")

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential",
        forbidden_discovery,
    )
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection",
        forbidden_socket,
    )

    with pytest.raises(TypeError, match=argument):
        DaemonClient.connect(tmp_path, **{argument: value})  # type: ignore[arg-type]

    assert discovery_calls == 0
    assert socket_calls == 0


@pytest.mark.parametrize("nesting", [40, 2_000])
def test_client_rejects_deep_or_parser_recursive_response_and_poison_closes(
    nesting: int,
) -> None:
    client_socket, server_socket = socket.socketpair()
    limits = ProtocolLimitsV1()
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=limits,
    )
    invalid = (
        b'{"jsonrpc":"2.0","id":1,"result":{"nested":'
        + b"[" * nesting
        + b"0"
        + b"]" * nesting
        + b"}}\n"
    )
    server_socket.sendall(invalid)
    try:
        with pytest.raises(DaemonClientError, match="invalid"):
            client.health()
        assert client._closed is True
        with pytest.raises(DaemonClientError, match="closed"):
            client.health()
    finally:
        client.close()
        server_socket.close()


def test_client_generates_unique_256_bit_bridge_recovery_tokens() -> None:
    first = DaemonClient.new_bridge_recovery_token()
    second = DaemonClient.new_bridge_recovery_token()

    assert len(first) == len(second) == 64
    assert set(first + second) <= set("0123456789abcdef")
    assert first != second


@pytest.mark.parametrize("invalid_params", [False, [], "", 0])
def test_call_rejects_falsy_non_dict_params_before_send_and_remains_usable(
    invalid_params: object,
) -> None:
    class RecordingSocket(FakeSocket):
        def __init__(self) -> None:
            super().__init__()
            self.sent: list[bytes] = []
            self.stream = io.BytesIO(
                b'{"jsonrpc":"2.0","id":1,"result":{"status":"ready"}}\n'
            )

        def makefile(self, _mode: str):
            return self.stream

        def sendall(self, payload: bytes) -> None:
            self.sent.append(payload)

    connection = RecordingSocket()
    client = DaemonClient(
        connection,  # type: ignore[arg-type]
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=ProtocolLimitsV1(),
    )
    try:
        with pytest.raises(TypeError, match="params"):
            client.call("health", invalid_params)  # type: ignore[arg-type]

        assert connection.sent == []
        assert client.health() == {"status": "ready"}
        assert len(connection.sent) == 1
    finally:
        client.close()


def test_client_oversized_physical_line_poison_closes_before_tail_reuse() -> None:
    client_socket, server_socket = socket.socketpair()
    limits = ProtocolLimitsV1(max_response_bytes=4_096)
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=limits,
    )
    forged_tail = b'{"jsonrpc":"2.0","id":2,"result":{"forged":true}}'
    server_socket.sendall(b"x" * (limits.max_response_bytes + 1) + forged_tail + b"\n")
    try:
        with pytest.raises(DaemonClientError, match=r"incomplete|oversized"):
            client.health()

        assert client._closed is True
        with pytest.raises(DaemonClientError, match="closed"):
            client.health()
    finally:
        client.close()
        server_socket.close()


@pytest.mark.parametrize(
    "response",
    [
        b'{"jsonrpc":"2.0","id":1,"result":{},"result":{}}\n',
        b'{"jsonrpc":"2.0","id":1,"result":{"number":NaN}}\n',
        b'{"jsonrpc":"2.0","id":1,"result":{"number":Infinity}}\n',
        b'{"jsonrpc":"2.0","id":1,"result":{"text":"\\ud800"}}\n',
        b'{"jsonrpc":"2.0","id":1,"result":{"text":"\xed\xa0\x80"}}\n',
    ],
    ids=[
        "duplicate-response-key",
        "nan",
        "infinity",
        "escaped-lone-surrogate",
        "lone-surrogate-utf8",
    ],
)
def test_client_rejects_hostile_json_responses_and_poison_closes(
    response: bytes,
) -> None:
    client_socket, server_socket = socket.socketpair()
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=ProtocolLimitsV1(max_response_bytes=4_096),
    )
    server_socket.sendall(response)
    try:
        with pytest.raises(DaemonClientError):
            client.health()

        assert client._closed is True
        with pytest.raises(DaemonClientError, match="closed"):
            client.health()
    finally:
        client.close()
        server_socket.close()


def test_client_rejects_response_node_overflow_and_poison_closes() -> None:
    client_socket, server_socket = socket.socketpair()
    limits = ProtocolLimitsV1(
        max_response_bytes=4_096,
        max_nodes=256,
    )
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=limits,
    )
    response = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"items": [0] * 248},
        },
        separators=(",", ":"),
    ).encode()
    assert len(response) < limits.max_response_bytes
    server_socket.sendall(response + b"\n")
    try:
        with pytest.raises(DaemonClientError, match="invalid"):
            client.health()

        assert client._closed is True
        with pytest.raises(DaemonClientError, match="closed"):
            client.health()
    finally:
        client.close()
        server_socket.close()


def _success_response_with_exact_size(size: int) -> tuple[bytes, str]:
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"padding": ""},
    }
    empty = json.dumps(response, separators=(",", ":")).encode()
    padding = "x" * (size - len(empty))
    response["result"] = {"padding": padding}
    encoded = json.dumps(response, separators=(",", ":")).encode()
    assert len(encoded) == size
    return encoded, padding


def test_client_accepts_response_at_exact_negotiated_byte_limit() -> None:
    client_socket, server_socket = socket.socketpair()
    limits = ProtocolLimitsV1(max_response_bytes=4_096)
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=limits,
    )
    response, padding = _success_response_with_exact_size(limits.max_response_bytes)
    server_socket.sendall(
        response + b'\n{"jsonrpc":"2.0","id":2,"result":{"status":"ready"}}\n'
    )
    try:
        assert client.health() == {"padding": padding}
        assert client._closed is False
        assert client.health() == {"status": "ready"}
    finally:
        client.close()
        server_socket.close()


def test_client_rejects_response_one_byte_over_limit_and_poison_closes() -> None:
    client_socket, server_socket = socket.socketpair()
    limits = ProtocolLimitsV1(max_response_bytes=4_096)
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=limits,
    )
    response, _padding = _success_response_with_exact_size(
        limits.max_response_bytes + 1
    )
    server_socket.sendall(response + b"\n")
    try:
        with pytest.raises(DaemonClientError, match="byte limit"):
            client.health()

        assert client._closed is True
        with pytest.raises(DaemonClientError, match="closed"):
            client.health()
    finally:
        client.close()
        server_socket.close()


@pytest.mark.parametrize(
    "response",
    [
        b'{"jsonrpc":"2.0","id":1,"result":{},"error":'
        b'{"code":"failure","message":"failed","data":{}}}\n',
        b'{"jsonrpc":"2.0","id":1}\n',
        b'{"jsonrpc":"2.0","id":1,"result":{},"extra":true}\n',
        b'{"jsonrpc":"2.0","id":1,"error":{"code":"failure","message":"failed"}}\n',
        b'{"jsonrpc":"2.0","id":1,"error":'
        b'{"code":"failure","message":"failed","data":{},"extra":true}}\n',
        b'{"jsonrpc":"2.0","id":1,"error":'
        b'{"code":"failure","message":"failed","data":{}},"extra":true}\n',
        b'{"jsonrpc":"2.0","id":2,"result":{}}\n',
    ],
    ids=[
        "result-and-error",
        "neither-result-nor-error",
        "extra-success-envelope-key",
        "missing-error-key",
        "extra-error-key",
        "extra-error-envelope-key",
        "wrong-response-id",
    ],
)
def test_client_rejects_non_exact_response_envelopes(response: bytes) -> None:
    client_socket, server_socket = socket.socketpair()
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=ProtocolLimitsV1(),
    )
    server_socket.sendall(response)
    try:
        with pytest.raises(DaemonClientError, match=r"response|error"):
            client.health()
        assert client._closed is True
    finally:
        client.close()
        server_socket.close()


def test_client_accepts_exact_success_envelope() -> None:
    client_socket, server_socket = socket.socketpair()
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=ProtocolLimitsV1(),
    )
    server_socket.sendall(b'{"jsonrpc":"2.0","id":1,"result":{"status":"ready"}}\n')
    try:
        assert client.health() == {"status": "ready"}
    finally:
        client.close()
        server_socket.close()


def test_client_preserves_exact_rpc_error_fields() -> None:
    client_socket, server_socket = socket.socketpair()
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=ProtocolLimitsV1(),
    )
    server_socket.sendall(
        b'{"jsonrpc":"2.0","id":1,"error":'
        b'{"code":"specific_failure","message":"exact message",'
        b'"data":{"reason":"exact-data"}}}\n'
    )
    try:
        with pytest.raises(DaemonRpcError) as captured:
            client.health()
        assert captured.value.code == "specific_failure"
        assert captured.value.message == "exact message"
        assert captured.value.data == {"reason": "exact-data"}
    finally:
        client.close()
        server_socket.close()


@pytest.mark.parametrize(
    "error",
    [
        {"code": "failure", "message": "failed", "data": "scalar"},
        {"code": "", "message": "failed", "data": {}},
        {"code": "failure", "message": "", "data": {}},
        {"code": "x" * 161, "message": "failed", "data": {}},
        {"code": "failure", "message": "x" * 1_025, "data": {}},
    ],
    ids=[
        "scalar-data",
        "blank-code",
        "blank-message",
        "oversized-code",
        "oversized-message",
    ],
)
def test_client_rejects_invalid_error_schema_and_poison_closes(
    error: dict[str, object],
) -> None:
    client_socket, server_socket = socket.socketpair()
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=ProtocolLimitsV1(),
    )
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": error,
    }
    server_socket.sendall(json.dumps(response, separators=(",", ":")).encode() + b"\n")
    try:
        with pytest.raises(DaemonClientError, match="error response"):
            client.health()
        assert client._closed is True
        with pytest.raises(DaemonClientError, match="closed"):
            client.health()
    finally:
        client.close()
        server_socket.close()


def test_client_preserves_nonce_blind_unauthorized_error() -> None:
    client_socket, server_socket = socket.socketpair()
    client = DaemonClient(
        client_socket,
        discovery(),
        "a" * 64,
        timeout_seconds=1.0,
        limits=ProtocolLimitsV1(),
    )
    server_socket.sendall(
        b'{"jsonrpc":"2.0","id":null,"error":'
        b'{"code":"unauthorized","message":"authentication failed",'
        b'"data":{}}}\n'
    )
    try:
        with pytest.raises(DaemonRpcError) as captured:
            client.health()
        assert captured.value.code == "unauthorized"
        assert captured.value.message == "authentication failed"
        assert captured.value.data == {}
    finally:
        client.close()
        server_socket.close()


def test_connect_same_token_unauthorized_is_sent_once_and_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = rotated_discovery("nonce-one", 1)
    loads = 0
    health_tokens: list[str] = []

    def load(_home):
        nonlocal loads
        loads += 1
        return record, "a" * 64

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential", load
    )
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection",
        lambda address, **_kwargs: FakeSocket(peer=address),
    )

    def unauthorized(client: DaemonClient):
        health_tokens.append(client._credential)
        raise DaemonRpcError("unauthorized", "authentication failed", {})

    monkeypatch.setattr(DaemonClient, "health", unauthorized)

    with pytest.raises(DaemonRpcError) as captured:
        DaemonClient.connect(tmp_path, initialize=False)

    assert captured.value.code == "unauthorized"
    assert health_tokens == ["a" * 64]
    assert loads == 2  # initial load plus one bounded rotation proof reread


def test_connect_rejects_stale_process_marker_before_socket_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    current = rotated_discovery("nonce-one", 1)
    prefix, ticks = current.process_marker.rsplit(":", 1)
    stale = current.model_copy(update={"process_marker": f"{prefix}:{int(ticks) + 1}"})
    socket_calls = 0

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential",
        lambda _home: (stale, "a" * 64),
    )

    def forbidden_socket(*_args, **_kwargs):
        nonlocal socket_calls
        socket_calls += 1
        raise AssertionError("stale process marker must fail before socket creation")

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection",
        forbidden_socket,
    )

    with pytest.raises(DaemonClientError):
        DaemonClient.connect(tmp_path, initialize=False)

    assert socket_calls == 0


def test_connect_retries_once_only_after_nonce_credential_rotation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = [
        (rotated_discovery("nonce-one", 1), "a" * 64),
        (rotated_discovery("nonce-two", 2), "b" * 64),
    ]
    health_tokens: list[str] = []

    def load(_home):
        return records.pop(0) if len(records) > 1 else records[0]

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential", load
    )
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection",
        lambda address, **_kwargs: FakeSocket(peer=address),
    )

    def health(client: DaemonClient):
        health_tokens.append(client._credential)
        if client._credential == "a" * 64:
            raise DaemonRpcError("unauthorized", "authentication failed", {})
        return discovery_health(client.discovery)

    monkeypatch.setattr(DaemonClient, "health", health)

    client = DaemonClient.connect(tmp_path, initialize=False)
    try:
        assert client.discovery.instance_nonce == "nonce-two"
        assert health_tokens == ["a" * 64, "b" * 64]
    finally:
        client.close()


def test_connect_cleanup_fault_cannot_bypass_one_rotation_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = [
        (rotated_discovery("nonce-one", 1), "a" * 64),
        (rotated_discovery("nonce-two", 2), "b" * 64),
    ]
    first_socket = CleanupFaultSocket(peer=("127.0.0.1", 1))
    second_socket = FakeSocket(peer=("127.0.0.1", 2))
    connections = [first_socket, second_socket]
    health_tokens: list[str] = []

    def load(_home):
        return records.pop(0) if len(records) > 1 else records[0]

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential", load
    )
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection",
        lambda *_args, **_kwargs: connections.pop(0),
    )

    def health(client: DaemonClient):
        health_tokens.append(client._credential)
        if client._credential == "a" * 64:
            raise DaemonRpcError("unauthorized", "authentication failed", {})
        return discovery_health(client.discovery)

    monkeypatch.setattr(DaemonClient, "health", health)

    client = DaemonClient.connect(tmp_path, initialize=False)
    try:
        assert health_tokens == ["a" * 64, "b" * 64]
        assert first_socket.stream_close_calls == 1
        assert first_socket.socket_close_calls == 1
        assert first_socket.closed is True
    finally:
        client.close()


def test_connect_cleanup_fault_preserves_original_handshake_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = rotated_discovery("nonce-one", 1)
    connection = CleanupFaultSocket(peer=("127.0.0.1", 1))
    loads = 0

    def load(_home):
        nonlocal loads
        loads += 1
        return record, "a" * 64

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential", load
    )
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection",
        lambda *_args, **_kwargs: connection,
    )
    monkeypatch.setattr(
        DaemonClient,
        "health",
        lambda _client: (_ for _ in ()).throw(
            DaemonRpcError("unauthorized", "authentication failed", {})
        ),
    )

    with pytest.raises(DaemonRpcError) as captured:
        DaemonClient.connect(tmp_path, initialize=False)

    assert captured.value.code == "unauthorized"
    assert loads == 2
    assert connection.stream_close_calls == 1
    assert connection.socket_close_calls == 1


def test_connect_rejects_changed_token_without_nonce_rotation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = rotated_discovery("nonce-one", 1)
    credentials = ["a" * 64, "b" * 64]
    health_tokens: list[str] = []
    socket_calls = 0

    def load(_home):
        token = credentials.pop(0) if len(credentials) > 1 else credentials[0]
        return record, token

    def connect(address, **_kwargs):
        nonlocal socket_calls
        socket_calls += 1
        return FakeSocket(peer=address)

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential", load
    )
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection", connect
    )

    def unauthorized(client: DaemonClient):
        health_tokens.append(client._credential)
        raise DaemonRpcError("unauthorized", "authentication failed", {})

    monkeypatch.setattr(DaemonClient, "health", unauthorized)

    with pytest.raises(DaemonRpcError) as captured:
        DaemonClient.connect(tmp_path, initialize=False)

    assert captured.value.code == "unauthorized"
    assert health_tokens == ["a" * 64]
    assert socket_calls == 1


def test_connect_capability_mismatch_is_not_retried_or_erased(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = rotated_discovery("nonce-one", 1)
    loads = 0
    initialize_calls = 0

    def load(_home):
        nonlocal loads
        loads += 1
        return record, "a" * 64

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential", load
    )
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection",
        lambda address, **_kwargs: FakeSocket(peer=address),
    )
    monkeypatch.setattr(
        DaemonClient,
        "health",
        lambda client: discovery_health(client.discovery),
    )

    def mismatch(_client: DaemonClient, _method: str, _params=None):
        nonlocal initialize_calls
        initialize_calls += 1
        raise DaemonRpcError(
            "capability_mismatch", "exact capability profile is required", {}
        )

    monkeypatch.setattr(DaemonClient, "call", mismatch)

    with pytest.raises(DaemonRpcError) as captured:
        DaemonClient.connect(tmp_path)

    assert captured.value.code == "capability_mismatch"
    assert initialize_calls == 1
    assert loads == 1


@pytest.mark.parametrize(
    "initialize_result",
    [
        {},
        {
            "protocol_version": 0,
            "capabilities": CapabilityProfileV1().model_dump(mode="json"),
            "server_adapter_id": SERVER_ADAPTER_ID,
            "instance_nonce": "nonce-one",
        },
        {
            "protocol_version": PROTOCOL_VERSION,
            "capabilities": {},
            "server_adapter_id": SERVER_ADAPTER_ID,
            "instance_nonce": "nonce-one",
        },
        {
            "protocol_version": PROTOCOL_VERSION,
            "capabilities": CapabilityProfileV1().model_dump(mode="json"),
            "server_adapter_id": "downgraded",
            "instance_nonce": "nonce-one",
        },
        {
            "protocol_version": PROTOCOL_VERSION,
            "capabilities": CapabilityProfileV1().model_dump(mode="json"),
            "server_adapter_id": SERVER_ADAPTER_ID,
            "instance_nonce": "other-nonce",
        },
        {
            "protocol_version": PROTOCOL_VERSION,
            "capabilities": CapabilityProfileV1().model_dump(mode="json"),
            "server_adapter_id": SERVER_ADAPTER_ID,
            "instance_nonce": "nonce-one",
            "extra": True,
        },
    ],
    ids=[
        "empty",
        "protocol-downgrade",
        "capability-downgrade",
        "adapter-downgrade",
        "nonce-mismatch",
        "extra-field",
    ],
)
def test_connect_rejects_non_exact_initialize_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    initialize_result: dict[str, object],
) -> None:
    record = rotated_discovery("nonce-one", 1)
    connection = FakeSocket(peer=("127.0.0.1", 1))
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential",
        lambda _home: (record, "a" * 64),
    )
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection",
        lambda *_args, **_kwargs: connection,
    )
    monkeypatch.setattr(
        DaemonClient,
        "health",
        lambda client: discovery_health(client.discovery),
    )
    monkeypatch.setattr(
        DaemonClient,
        "call",
        lambda _client, _method, _params=None: initialize_result,
    )

    with pytest.raises(DaemonClientError):
        DaemonClient.connect(tmp_path)

    assert connection.closed is True


@pytest.mark.parametrize(
    "connection",
    [
        FakeSocket(family=socket.AF_INET6),
        FakeSocket(local=("0.0.0.0", 43210)),
        FakeSocket(peer=("192.0.2.10", 1)),
        FakeSocket(peer=("127.0.0.1", 2)),
    ],
    ids=["non-ipv4", "non-loopback-local", "non-loopback-peer", "wrong-port"],
)
def test_connect_rejects_unproven_socket_endpoints_before_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    connection: FakeSocket,
) -> None:
    record = rotated_discovery("nonce-one", 1)
    health_calls = 0

    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.load_discovery_and_credential",
        lambda _home: (record, "a" * 64),
    )
    monkeypatch.setattr(
        "alice_brain_hermes.protocol.client.socket.create_connection",
        lambda *_args, **_kwargs: connection,
    )

    def health(_client: DaemonClient) -> dict[str, object]:
        nonlocal health_calls
        health_calls += 1
        return discovery_health(record)

    monkeypatch.setattr(DaemonClient, "health", health)

    with pytest.raises(DaemonClientError):
        DaemonClient.connect(tmp_path, initialize=False)

    assert connection.closed is True
    assert health_calls == 0
