"""Secure bounded client for the private loopback daemon."""

from __future__ import annotations

import json
import math
import secrets
import socket
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import ValidationError

from alice_brain_hermes.errors import DaemonClientError, DaemonRpcError
from alice_brain_hermes.protocol.models import (
    PROTOCOL_VERSION,
    CapabilityProfileV1,
    DaemonDiscoveryV2,
    InitializeResultV1,
    ProtocolLimitsV1,
    copy_protocol_limits,
)
from alice_brain_hermes.runtime.discovery import load_discovery_and_credential
from alice_brain_hermes.runtime.process_marker import verify_process_marker

_MAX_TIMEOUT_SECONDS = 300.0


def _bounded_timeout_seconds(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= _MAX_TIMEOUT_SECONDS
    ):
        raise ValueError(
            "timeout_seconds must be finite, positive, and at most "
            f"{_MAX_TIMEOUT_SECONDS:g}"
        )
    return float(value)


def _close_without_masking(resource: object) -> None:
    """Best-effort error-path close that preserves the primary fault."""
    with suppress(BaseException):
        resource.close()  # type: ignore[attr-defined]


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise DaemonClientError("daemon response contains duplicate keys")
        result[key] = value
    return result


def _finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise DaemonClientError("daemon response contains a non-finite number")
    return number


def _validate_response_tree(value: object, limits: ProtocolLimitsV1) -> None:
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > limits.max_nodes or depth > limits.max_depth:
            raise ValueError("daemon response structure exceeds limits")
        if isinstance(item, str):
            item.encode("utf-8", errors="strict")
        elif isinstance(item, Mapping):
            pending.extend((key, depth + 1) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)


def _ipv4_endpoint(value: object) -> tuple[str, int] | None:
    if not isinstance(value, tuple) or len(value) < 2:
        return None
    host, port = value[:2]
    if (
        not isinstance(host, str)
        or isinstance(port, bool)
        or not isinstance(port, int)
        or not 1 <= port <= 65_535
    ):
        return None
    return host, port


def _validate_connected_loopback(
    connection: socket.socket, discovery: DaemonDiscoveryV2
) -> None:
    try:
        family = connection.family
        local = _ipv4_endpoint(connection.getsockname())
        peer = _ipv4_endpoint(connection.getpeername())
    except OSError as error:
        raise DaemonClientError("daemon socket endpoints cannot be proven") from error
    expected_peer = (discovery.endpoint.host, discovery.endpoint.port)
    if (
        family != socket.AF_INET
        or local is None
        or local[0] != "127.0.0.1"
        or peer != expected_peer
    ):
        raise DaemonClientError("daemon socket is not exact IPv4 loopback")


class DaemonClient:
    """One finite-timeout authenticated connection with isolated initialize state."""

    def __init__(
        self,
        connection: socket.socket,
        discovery: DaemonDiscoveryV2,
        credential: str,
        *,
        timeout_seconds: float,
        limits: ProtocolLimitsV1,
    ) -> None:
        self._socket = connection
        self._stream = connection.makefile("rb")
        self.discovery = discovery
        self._credential = credential
        self.timeout_seconds = timeout_seconds
        self.limits = limits
        self._next_id = 1
        self._closed = False

    def __repr__(self) -> str:
        return (
            "DaemonClient(instance_nonce="
            f"{self.discovery.instance_nonce!r}, closed={self._closed!r})"
        )

    @staticmethod
    def new_bridge_recovery_token() -> str:
        """Create one opaque proof for attach and later terminal recovery."""
        return secrets.token_hex(32)

    @classmethod
    def connect(
        cls,
        runtime_home: str | Path,
        *,
        initialize: bool = True,
        timeout_seconds: float = 3.0,
        limits: ProtocolLimitsV1 | None = None,
    ) -> Self:
        timeout_seconds = _bounded_timeout_seconds(timeout_seconds)
        if type(initialize) is not bool:
            raise TypeError("initialize must be an exact bool")
        active_limits = copy_protocol_limits(limits)
        try:
            discovery, credential = load_discovery_and_credential(runtime_home)
        except (OSError, ValueError) as error:
            raise DaemonClientError("daemon discovery load failed") from error
        for attempt in range(2):
            client: Self | None = None
            try:
                verify_process_marker(discovery.pid, discovery.process_marker)
                connection = socket.create_connection(
                    (discovery.endpoint.host, discovery.endpoint.port),
                    timeout=timeout_seconds,
                )
                try:
                    connection.settimeout(timeout_seconds)
                    _validate_connected_loopback(connection, discovery)
                    client = cls(
                        connection,
                        discovery,
                        credential,
                        timeout_seconds=timeout_seconds,
                        limits=active_limits,
                    )
                except BaseException:
                    _close_without_masking(connection)
                    raise
                health = client.health()
                if (
                    health.get("pid") != discovery.pid
                    or health.get("instance_nonce") != discovery.instance_nonce
                    or health.get("process_marker") != discovery.process_marker
                    or health.get("launch_nonce") != discovery.launch_nonce
                    or health.get("protocol_version") != discovery.protocol_version
                    or health.get("package_version") != discovery.package_version
                ):
                    raise DaemonClientError("daemon discovery identity changed")
                if initialize:
                    expected_capabilities = CapabilityProfileV1(limits=active_limits)
                    result = client.call(
                        "initialize",
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "capabilities": expected_capabilities.model_dump(
                                mode="json"
                            ),
                        },
                    )
                    try:
                        if not isinstance(result, dict):
                            raise ValueError("initialize result is not an object")
                        CapabilityProfileV1.validate_wire(result.get("capabilities"))
                        initialized = InitializeResultV1.model_validate(
                            result, strict=True
                        )
                    except (ValidationError, ValueError) as error:
                        raise DaemonClientError(
                            "daemon initialize result is invalid"
                        ) from error
                    if (
                        initialized.capabilities != expected_capabilities
                        or initialized.instance_nonce != discovery.instance_nonce
                    ):
                        raise DaemonClientError("daemon initialize negotiation changed")
                return client
            except DaemonRpcError as error:
                if client is not None:
                    _close_without_masking(client)
                if error.code != "unauthorized" or attempt == 1:
                    raise
                try:
                    reread = load_discovery_and_credential(runtime_home)
                except (OSError, ValueError) as reread_error:
                    raise error from reread_error
                next_discovery, next_credential = reread
                if (
                    next_discovery.instance_nonce == discovery.instance_nonce
                    or next_credential == credential
                ):
                    raise error
                discovery, credential = next_discovery, next_credential
            except (OSError, ValueError, DaemonClientError) as error:
                if client is not None:
                    _close_without_masking(client)
                if attempt == 1:
                    raise DaemonClientError(
                        "daemon connection failed after one rotated discovery"
                    ) from error
                try:
                    reread = load_discovery_and_credential(runtime_home)
                except (OSError, ValueError) as reread_error:
                    raise DaemonClientError(
                        "daemon connection failed and discovery reread failed"
                    ) from reread_error
                next_discovery, next_credential = reread
                if (
                    next_discovery.instance_nonce == discovery.instance_nonce
                    or next_credential == credential
                ):
                    raise DaemonClientError(
                        "daemon connection failed without discovery rotation"
                    ) from error
                discovery, credential = next_discovery, next_credential
        raise AssertionError("bounded daemon connection attempts exhausted")

    def call(
        self, method: str, params: dict[str, object] | None = None
    ) -> dict[str, object]:
        if self._closed:
            raise DaemonClientError("daemon client is closed")
        if not isinstance(method, str) or not method:
            raise ValueError("method must be non-blank")
        if params is None:
            request_params: dict[str, object] = {}
        elif type(params) is dict:
            request_params = dict(params)
        else:
            raise TypeError("params must be an exact dict or None")
        request_id = self._next_id
        self._next_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": request_params,
            "auth": self._credential,
        }
        encoded = json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > self.limits.max_request_bytes:
            raise DaemonClientError("request exceeds the negotiated byte limit")
        try:
            self._socket.sendall(encoded + b"\n")
            response = self._stream.readline(self.limits.max_response_bytes + 2)
        except OSError as error:
            self._poison()
            raise DaemonClientError("daemon transport failed") from error
        if not response or not response.endswith(b"\n"):
            self._poison()
            raise DaemonClientError("daemon response is incomplete or oversized")
        if len(response) - 1 > self.limits.max_response_bytes:
            self._poison()
            raise DaemonClientError("daemon response exceeds the byte limit")
        try:
            return self._decode_response(response[:-1], request_id)
        except DaemonRpcError:
            raise
        except DaemonClientError:
            self._poison()
            raise

    def _decode_response(self, payload: bytes, request_id: int) -> dict[str, object]:
        try:
            body = json.loads(
                payload.decode("utf-8", errors="strict"),
                object_pairs_hook=_pairs,
                parse_float=_finite,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    DaemonClientError("daemon response contains non-finite data")
                ),
            )
            _validate_response_tree(body, self.limits)
        except (
            UnicodeError,
            ValueError,
            OverflowError,
            RecursionError,
            json.JSONDecodeError,
        ) as error:
            raise DaemonClientError("daemon response is invalid") from error
        if not isinstance(body, dict):
            raise DaemonClientError("daemon response envelope is invalid")
        if body.get("jsonrpc") != "2.0":
            raise DaemonClientError("daemon response version is invalid")
        envelope_keys = set(body)
        success_keys = {"jsonrpc", "id", "result"}
        error_keys = {"jsonrpc", "id", "error"}
        if envelope_keys not in (success_keys, error_keys):
            raise DaemonClientError("daemon response envelope is invalid")
        if envelope_keys == error_keys:
            error = body["error"]
            if not isinstance(error, dict) or set(error) != {
                "code",
                "message",
                "data",
            }:
                raise DaemonClientError("daemon error response is invalid")
            code = error["code"]
            message = error["message"]
            data = error["data"]
            if (
                not isinstance(code, str)
                or not 1 <= len(code) <= 160
                or not code.strip()
                or not isinstance(message, str)
                or not 1 <= len(message) <= 1_024
                or not message.strip()
                or (data is not None and not isinstance(data, dict))
            ):
                raise DaemonClientError("daemon error response is invalid")
            response_id = body["id"]
            id_matches = (
                type(response_id) is type(request_id) and response_id == request_id
            )
            if not id_matches and not (code == "unauthorized" and response_id is None):
                raise DaemonClientError("daemon response id does not match request")
            raise DaemonRpcError(code, message, data)
        response_id = body["id"]
        if type(response_id) is not type(request_id) or response_id != request_id:
            raise DaemonClientError("daemon response id does not match request")
        result = body["result"]
        if not isinstance(result, dict):
            raise DaemonClientError("daemon result must be an object")
        return result

    def health(self) -> dict[str, object]:
        return self.call("health", {})

    def shutdown(self) -> dict[str, object]:
        return self.call("daemon.shutdown", {})

    def _poison(self) -> None:
        try:
            self.close()
        except BaseException:
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        finally:
            self._socket.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["DaemonClient"]
