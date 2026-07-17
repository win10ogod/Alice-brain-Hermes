"""Per-connection authenticated strict JSON-RPC service."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import sqlite3
import threading
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from alice_brain_hermes import __version__
from alice_brain_hermes.errors import (
    BridgeAbandonedError,
    BridgeBindingError,
    BridgeCleanClosedError,
    BridgeClosedError,
    CaptureGapRequiredError,
    CaptureSequenceError,
    DomainCapacityError,
    EventConflictError,
    FrameSizeError,
    IdempotencyConflictError,
    LedgerIntegrityError,
    ResponseSizeError,
    RuntimeOwnedError,
)
from alice_brain_hermes.ids import new_id, validate_id
from alice_brain_hermes.protocol.diagnostics import (
    TRACE_MAX_PAGE_SIZE,
    IdentitySnapshotV1,
    build_trace_page,
)
from alice_brain_hermes.protocol.energy import (
    EnergyAssessmentChoiceV1,
    EnergyAssessmentProvenanceV1,
)
from alice_brain_hermes.protocol.identity import IdentityChoiceV1
from alice_brain_hermes.protocol.models import (
    PROTOCOL_VERSION,
    SERVER_ADAPTER_ID,
    BrainProfileV1,
    BridgeRecordV1,
    CapabilityProfileV1,
    ProtocolLimitsV1,
    copy_protocol_limits,
    validate_bridge_record_tree,
)
from alice_brain_hermes.protocol.status import EnergyWorkerReportV1
from alice_brain_hermes.runtime.daemon import HermesDaemonRuntime

_BRIDGE_RECORD_ADAPTER = TypeAdapter(BridgeRecordV1)
_ALLOWED_REQUEST_KEYS = {"jsonrpc", "id", "method", "params", "auth"}
_PRE_SERVE_READ_METHODS = frozenset(
    {"daemon.status", "snapshot.status", "identity.get", "trace.list", "state.get"}
)


class _DuplicateKey(ValueError):
    pass


class ProtocolFault(Exception):
    def __init__(
        self, code: str, message: str, data: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(code)
        self.code = code
        self.message = message
        self.data = dict(data or {})


@dataclass(frozen=True, slots=True)
class _Binding:
    binding: str
    brain_id: str
    bridge_instance_id: str


@dataclass(frozen=True, slots=True)
class _ResultBudget:
    bytes: int
    nodes: int
    depth: int


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite number")
    return parsed


def _validate_tree(value: object, limits: ProtocolLimitsV1) -> None:
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > limits.max_nodes or depth > limits.max_depth:
            raise ValueError("JSON structure exceeds protocol limits")
        if isinstance(item, str):
            item.encode("utf-8", errors="strict")
        elif isinstance(item, Mapping):
            pending.extend((key, depth + 1) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)


def _request_id(value: object) -> int | str:
    if isinstance(value, bool):
        raise ValueError("request id cannot be boolean")
    if isinstance(value, int):
        if not -(2**63) <= value <= 2**63 - 1:
            raise ValueError("integer request id is out of range")
        return value
    if isinstance(value, str) and 1 <= len(value) <= 128:
        return value
    raise ValueError("request id must be a bounded string or integer")


def _success_result_budget(
    request_id: int | str,
    *,
    max_response_bytes: int,
    max_response_nodes: int,
    max_response_depth: int,
) -> _ResultBudget:
    empty = json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "result": {}},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _ResultBudget(
        bytes=max_response_bytes - (len(empty) - len(b"{}")),
        # A success envelope contributes its root, three keys, and the
        # jsonrpc/id values outside the result subtree.
        nodes=max_response_nodes - 6,
        # The result object is nested one level below the response root.
        depth=max_response_depth - 1,
    )


class ProtocolService:
    """Shared stateless authority; socket-local state lives in each connection."""

    server_adapter_id = SERVER_ADAPTER_ID

    def __init__(
        self,
        runtime: HermesDaemonRuntime,
        *,
        credential: str,
        instance_nonce: str,
        limits: ProtocolLimitsV1 | None = None,
    ) -> None:
        if not isinstance(credential, str) or len(credential) != 64:
            raise ValueError("credential must be one 256-bit hex token")
        if any(character not in "0123456789abcdef" for character in credential):
            raise ValueError("credential must be lowercase hexadecimal")
        self.runtime = runtime
        self.instance_nonce = instance_nonce
        self.limits = copy_protocol_limits(limits)
        self.capabilities = CapabilityProfileV1(limits=self.limits)
        self._credential_digest = hashlib.sha256(credential.encode("ascii")).digest()
        self._shutting_down = False

    def new_connection(self) -> ProtocolConnection:
        return ProtocolConnection(self)

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    def begin_shutdown(self) -> None:
        self._shutting_down = True

    @property
    def mutations_enabled(self) -> bool:
        return self.runtime.daemon_serving_active

    def authenticated(self, candidate: object) -> bool:
        if (
            isinstance(candidate, str)
            and len(candidate) == 64
            and all(character in "0123456789abcdef" for character in candidate)
        ):
            supplied = candidate.encode("ascii")
        else:
            supplied = b"\x00" * 64
        digest = hashlib.sha256(supplied).digest()
        return hmac.compare_digest(digest, self._credential_digest)


class ProtocolConnection:
    """One authenticated/initialized session with private opaque bindings."""

    def __init__(self, service: ProtocolService) -> None:
        self.service = service
        self.connection_nonce = new_id()
        self.initialized = False
        self.authenticated = False
        self._bindings: dict[str, _Binding] = {}
        self._shutdown_requested = False
        self._closed = False
        self._lifecycle = threading.RLock()

    def _error(
        self,
        request_id: int | str | None,
        code: str,
        message: str,
        data: Mapping[str, object] | None = None,
    ) -> bytes:
        return self._encode(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message, "data": dict(data or {})},
            },
            request_id=request_id,
        )

    def _success(self, request_id: int | str, result: Mapping[str, object]) -> bytes:
        try:
            return self._encode(
                {"jsonrpc": "2.0", "id": request_id, "result": dict(result)},
                request_id=request_id,
            )
        except Exception:
            if self.service.runtime.fail_stopped:
                raise
            return self._error(
                request_id,
                "internal_error",
                "internal request failure",
                {"incident_id": new_id()},
            )

    def _encode(
        self, body: Mapping[str, object], *, request_id: int | str | None
    ) -> bytes:
        encoded = json.dumps(
            body,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) <= self.service.limits.max_response_bytes:
            return encoded
        fallback = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": "response_too_large",
                "message": "response exceeds the negotiated byte limit",
                "data": {},
            },
        }
        return json.dumps(fallback, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )

    def handle_frame(self, frame: bytes) -> bytes:
        with self._lifecycle:
            if self._closed:
                return self._error(
                    None,
                    "connection_closed",
                    "connection is closed",
                )
            return self._handle_frame_locked(frame)

    def _handle_frame_locked(self, frame: bytes) -> bytes:
        request_id: int | str | None = None
        if not isinstance(frame, bytes):
            raise TypeError("protocol frames must be bytes")
        if len(frame) > self.service.limits.max_request_bytes:
            return self._error(
                None,
                "request_too_large",
                "request exceeds the protocol byte limit",
            )
        try:
            text = frame.decode("utf-8", errors="strict")
            parsed = json.loads(
                text,
                object_pairs_hook=_pairs,
                parse_float=_finite_float,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("non-finite number")
                ),
            )
            _validate_tree(parsed, self.service.limits)
            if not isinstance(parsed, dict):
                raise ValueError("request must be one object")
        except (
            UnicodeError,
            ValueError,
            TypeError,
            OverflowError,
            RecursionError,
            json.JSONDecodeError,
        ):
            return self._error(None, "invalid_request", "request is invalid")

        if not self.service.authenticated(parsed.get("auth")):
            return self._error(None, "unauthorized", "authentication failed")
        self.authenticated = True
        try:
            request_id = _request_id(parsed.get("id"))
        except ValueError:
            return self._error(None, "invalid_request", "request is invalid")
        try:
            if set(parsed) != _ALLOWED_REQUEST_KEYS:
                raise ProtocolFault("invalid_request", "request envelope is invalid")
            if parsed.get("jsonrpc") != "2.0":
                raise ProtocolFault("invalid_request", "request envelope is invalid")
            method = parsed.get("method")
            params = parsed.get("params")
            if not isinstance(method, str) or not 1 <= len(method) <= 160:
                raise ProtocolFault("invalid_request", "request method is invalid")
            if not isinstance(params, dict):
                raise ProtocolFault("invalid_params", "named params are required")
            result = self._dispatch(
                method,
                params,
                _success_result_budget(
                    request_id,
                    max_response_bytes=self.service.limits.max_response_bytes,
                    max_response_nodes=self.service.limits.max_nodes,
                    max_response_depth=self.service.limits.max_depth,
                ),
            )
            return self._success(request_id, result)
        except ProtocolFault as error:
            return self._error(request_id, error.code, error.message, error.data)
        except ValidationError:
            return self._error(
                request_id, "invalid_params", "request params are invalid"
            )
        except IdempotencyConflictError:
            return self._error(
                request_id,
                "idempotency_conflict",
                "idempotency key conflicts",
            )
        except CaptureGapRequiredError:
            return self._error(
                request_id,
                "capture_gap_required",
                "an exact capture gap is required",
            )
        except CaptureSequenceError:
            return self._error(
                request_id,
                "capture_sequence_error",
                "capture sequence does not match the cursor",
            )
        except BridgeCleanClosedError:
            return self._error(
                request_id,
                "bridge_clean_closed",
                "bridge stream was cleanly closed",
                {"status": "clean_closed"},
            )
        except BridgeAbandonedError:
            return self._error(
                request_id,
                "bridge_abandoned",
                "bridge stream was abandoned",
                {"status": "abandoned"},
            )
        except BridgeClosedError:
            return self._error(request_id, "bridge_closed", "bridge stream is not open")
        except BridgeBindingError:
            return self._error(
                request_id, "invalid_binding", "bridge binding is invalid"
            )
        except FrameSizeError:
            return self._error(
                request_id,
                "frame_too_large",
                "consciousness frame exceeds negotiated byte limit",
            )
        except ResponseSizeError:
            return self._error(
                request_id,
                "response_too_large",
                "response exceeds the negotiated byte limit",
            )
        except DomainCapacityError:
            return self._error(
                request_id,
                "capacity_exhausted",
                "bounded runtime capacity is exhausted",
            )
        except KeyError:
            return self._error(
                request_id, "not_found", "requested object was not found"
            )
        except (TypeError, ValueError):
            return self._error(
                request_id, "invalid_params", "request params are invalid"
            )
        except (EventConflictError, LedgerIntegrityError, sqlite3.DatabaseError):
            return self._error(
                request_id,
                "runtime_conflict",
                "runtime state requires authoritative replay",
            )
        except Exception:
            if self.service.runtime.fail_stopped:
                raise
            return self._error(
                request_id,
                "internal_error",
                "internal request failure",
                {"incident_id": new_id()},
            )

    @staticmethod
    def _only(params: Mapping[str, object], allowed: set[str]) -> None:
        if set(params) - allowed:
            raise ProtocolFault("invalid_params", "request params are invalid")

    def _binding(self, value: object) -> _Binding:
        if not isinstance(value, str):
            raise ProtocolFault("invalid_binding", "bridge binding is invalid")
        try:
            return self._bindings[value]
        except KeyError:
            raise ProtocolFault(
                "invalid_binding", "bridge binding is invalid"
            ) from None

    def _selected_brain_id(self, value: object) -> str:
        brain_ids = self.service.runtime.brain_ids
        if value is None:
            if not brain_ids:
                raise ProtocolFault(
                    "not_found",
                    "no persisted brain identity is available",
                    {"brain_count": 0},
                )
            if len(brain_ids) != 1:
                raise ProtocolFault(
                    "brain_id_required",
                    "brain_id is required when multiple brains exist",
                    {"brain_count": len(brain_ids)},
                )
            return brain_ids[0]
        brain_id = validate_id(value)  # type: ignore[arg-type]
        if brain_id not in brain_ids:
            raise ProtocolFault(
                "not_found",
                "requested brain identity was not found",
                {"brain_id": brain_id},
            )
        return brain_id

    def _dispatch(
        self,
        method: str,
        params: Mapping[str, object],
        result_budget: _ResultBudget,
    ) -> Mapping[str, object]:
        if method == "health":
            self._only(params, set())
            readiness = self.service.runtime.readiness_snapshot()
            return {
                "pid": os.getpid(),
                "instance_nonce": self.service.instance_nonce,
                "launch_nonce": self.service.runtime.lease.launch_nonce,
                "process_marker": self.service.runtime.lease.process_marker,
                "shutting_down": self.service.shutting_down,
                "protocol_version": PROTOCOL_VERSION,
                "package_version": __version__,
                **readiness,
            }
        if method == "initialize":
            if self.initialized:
                raise ProtocolFault(
                    "already_initialized", "connection is already initialized"
                )
            self._only(params, {"protocol_version", "capabilities"})
            protocol_version = params.get("protocol_version")
            if (
                type(protocol_version) is not int
                or protocol_version != PROTOCOL_VERSION
            ):
                raise ProtocolFault(
                    "protocol_mismatch", "exact protocol version is required"
                )
            capabilities = params.get("capabilities")
            if not isinstance(capabilities, dict):
                raise ProtocolFault(
                    "capability_mismatch", "exact capability profile is required"
                )
            try:
                requested = CapabilityProfileV1.validate_wire(capabilities)
            except (ValidationError, ValueError):
                raise ProtocolFault(
                    "capability_mismatch", "exact capability profile is required"
                ) from None
            if requested != self.service.capabilities:
                raise ProtocolFault(
                    "capability_mismatch", "exact capability profile is required"
                )
            self.initialized = True
            return {
                "protocol_version": PROTOCOL_VERSION,
                "capabilities": self.service.capabilities.model_dump(mode="json"),
                "server_adapter_id": self.service.server_adapter_id,
                "instance_nonce": self.service.instance_nonce,
            }
        if not self.initialized:
            raise ProtocolFault("not_initialized", "connection must initialize first")
        if self.service.shutting_down:
            raise ProtocolFault("shutting_down", "daemon is shutting down")
        if not self.service.mutations_enabled and method not in _PRE_SERVE_READ_METHODS:
            raise ProtocolFault("not_ready", "daemon is not ready for mutation")
        if method == "daemon.status":
            self._only(params, set())
            return self.service.runtime.status_snapshot().model_dump(mode="json")
        if method == "snapshot.status":
            self._only(params, set())
            return self.service.runtime.snapshot_status().model_dump(mode="json")
        if method == "identity.get":
            self._only(params, {"brain_id"})
            brain_id = self._selected_brain_id(params.get("brain_id"))
            state = self.service.runtime.engine(brain_id).state
            snapshot = IdentitySnapshotV1(
                brain_id=brain_id,
                self_actor_id=state.identity.self_actor_id,
                name=state.identity.name,
                state_sequence=state.last_sequence,
                actors=state.identity.actors,
                authorizations=state.identity.authorizations,
            )
            return snapshot.model_dump(mode="json")
        if method == "trace.list":
            self._only(params, {"brain_id", "after_sequence", "limit"})
            brain_id = self._selected_brain_id(params.get("brain_id"))
            after_sequence = params.get("after_sequence", 0)
            limit = params.get("limit", 100)
            if (
                isinstance(after_sequence, bool)
                or not isinstance(after_sequence, int)
                or after_sequence < 0
                or isinstance(limit, bool)
                or not isinstance(limit, int)
                or not 1 <= limit <= TRACE_MAX_PAGE_SIZE
            ):
                raise ProtocolFault("invalid_params", "request params are invalid")
            events = self.service.runtime.ledger.list_events(
                brain_id,
                after_sequence=after_sequence,
                limit=limit + 1,
            )
            page = build_trace_page(
                events,
                brain_id=brain_id,
                after_sequence=after_sequence,
                requested_limit=limit,
                max_result_bytes=result_budget.bytes,
                max_result_nodes=result_budget.nodes,
                max_result_depth=result_budget.depth,
            )
            return page.model_dump(mode="json")
        if method == "brain.create":
            self._only(params, {"name"})
            name = params.get("name")
            if name is not None and (
                not isinstance(name, str) or not name.strip() or len(name) > 160
            ):
                raise ProtocolFault("invalid_params", "brain name is invalid")
            engine = self.service.runtime.create_brain(name=name)
            return {
                "brain_id": engine.brain_id,
                "state_sequence": engine.state.last_sequence,
            }
        if method == "brain.resolve":
            self._only(params, {"profile"})
            profile_data = params.get("profile")
            if not isinstance(profile_data, dict):
                raise ProtocolFault("invalid_params", "brain profile is invalid")
            profile = BrainProfileV1.model_validate(profile_data, strict=True)
            engine, created = self.service.runtime.resolve_brain(profile)
            return {
                "brain_id": engine.brain_id,
                "state_sequence": engine.state.last_sequence,
                "created": created,
            }
        if method == "identity.naming.claim":
            self._only(params, {"brain_id"})
            brain_id = validate_id(params.get("brain_id"))  # type: ignore[arg-type]
            lease = self.service.runtime.claim_identity_naming(brain_id)
            return {"lease": (None if lease is None else lease.model_dump(mode="json"))}
        if method == "identity.naming.complete":
            self._only(params, {"lease_id", "choice"})
            lease_id = validate_id(params.get("lease_id"))  # type: ignore[arg-type]
            choice_data = params.get("choice")
            if not isinstance(choice_data, dict):
                raise ProtocolFault("invalid_params", "identity choice is invalid")
            choice = IdentityChoiceV1.model_validate(choice_data, strict=True)
            status = self.service.runtime.complete_identity_naming(lease_id, choice)
            return {"status": status}
        if method == "identity.naming.fail":
            self._only(params, {"lease_id", "failure_code"})
            lease_id = validate_id(params.get("lease_id"))  # type: ignore[arg-type]
            failure_code = params.get("failure_code")
            if not isinstance(failure_code, str):
                raise ProtocolFault(
                    "invalid_params", "identity failure code is invalid"
                )
            status = self.service.runtime.fail_identity_naming(
                lease_id,
                failure_code,
            )
            return {"status": status}
        if method == "energy.assessment.claim":
            self._only(params, {"brain_id"})
            brain_id = validate_id(params.get("brain_id"))  # type: ignore[arg-type]
            lease = self.service.runtime.claim_energy_assessment(brain_id)
            return {"lease": (None if lease is None else lease.model_dump(mode="json"))}
        if method == "energy.worker.report":
            if set(params) != {
                "schema_version",
                "reporter_id",
                "report_sequence",
                "worker_started",
                "terminal_intent_pending",
                "last_error_type",
            }:
                raise ProtocolFault("invalid_params", "request params are invalid")
            report = EnergyWorkerReportV1.model_validate(params, strict=True)
            try:
                accepted = self.service.runtime.report_energy_worker(report)
            except RuntimeOwnedError:
                raise ProtocolFault(
                    "energy_worker_owned",
                    "energy worker heartbeat has a fresh owner",
                ) from None
            return {"accepted": accepted}
        if method == "energy.assessment.complete":
            self._only(params, {"lease_id", "choice", "provenance"})
            lease_id = validate_id(params.get("lease_id"))  # type: ignore[arg-type]
            choice_data = params.get("choice")
            provenance_data = params.get("provenance")
            if not isinstance(choice_data, dict) or not isinstance(
                provenance_data, dict
            ):
                raise ProtocolFault(
                    "invalid_params", "energy assessment evidence is invalid"
                )
            choice = EnergyAssessmentChoiceV1.model_validate(choice_data, strict=True)
            provenance = EnergyAssessmentProvenanceV1.model_validate(
                provenance_data, strict=True
            )
            status = self.service.runtime.complete_energy_assessment(
                lease_id,
                choice,
                provenance,
            )
            return {"status": status}
        if method == "energy.assessment.fail":
            self._only(params, {"lease_id", "failure_code"})
            lease_id = validate_id(params.get("lease_id"))  # type: ignore[arg-type]
            failure_code = params.get("failure_code")
            if not isinstance(failure_code, str):
                raise ProtocolFault(
                    "invalid_params", "energy assessment failure code is invalid"
                )
            status = self.service.runtime.fail_energy_assessment(
                lease_id,
                failure_code,
            )
            return {"status": status}
        if method == "brain.attach":
            self._only(
                params,
                {"brain_id", "bridge_instance_id", "recovery_token"},
            )
            brain_id = validate_id(params.get("brain_id"))  # type: ignore[arg-type]
            bridge_instance_id = validate_id(
                params.get("bridge_instance_id")  # type: ignore[arg-type]
            )
            engine = self.service.runtime.engine(brain_id)
            stream = self.service.runtime.ledger.attach_bridge_stream(
                bridge_instance_id,
                brain_id=brain_id,
                server_actor_id=engine.actor_id,
                server_adapter_id=self.service.server_adapter_id,
                connected_nonce=self.connection_nonce,
                recovery_token=params.get("recovery_token"),  # type: ignore[arg-type]
            )
            binding_id = new_id()
            self._bindings[binding_id] = _Binding(
                binding=binding_id,
                brain_id=brain_id,
                bridge_instance_id=bridge_instance_id,
            )
            return {
                "binding": binding_id,
                "brain_id": brain_id,
                "next_capture_seq": stream.next_capture_seq,
            }
        if method == "state.get":
            self._only(params, {"binding"})
            binding = self._binding(params.get("binding"))
            engine = self.service.runtime.engine(binding.brain_id)
            scheduler = self.service.runtime.scheduler(binding.brain_id)
            frame = engine.project_bridge_frame(
                binding.bridge_instance_id,
                connected_nonce=self.connection_nonce,
                scheduler_sample=("running" if scheduler.health.running else "stopped"),
                max_frame_bytes=self.service.limits.max_frame_bytes,
            )
            return frame.model_dump(mode="json")
        if method == "bridge.commit":
            self._only(params, {"binding", "record"})
            binding = self._binding(params.get("binding"))
            raw_record = params.get("record")
            encoded = json.dumps(
                raw_record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if len(encoded) > self.service.limits.max_record_bytes:
                raise ProtocolFault(
                    "record_too_large",
                    "bridge record exceeds the negotiated byte limit",
                )
            validate_bridge_record_tree(raw_record)
            record = _BRIDGE_RECORD_ADAPTER.validate_json(encoded, strict=True)
            if record.bridge_instance_id != binding.bridge_instance_id:
                raise ProtocolFault("invalid_binding", "bridge binding is invalid")
            ack = self.service.runtime.engine(binding.brain_id).commit_bridge_record(
                binding.bridge_instance_id,
                record,
                connected_nonce=self.connection_nonce,
                max_frame_bytes=self.service.limits.max_frame_bytes,
                max_ack_bytes=result_budget.bytes,
            )
            return ack.model_dump(mode="json")
        if method == "bridge.close":
            self._only(params, {"binding", "final_capture_seq"})
            binding = self._binding(params.get("binding"))
            final_capture_seq = params.get("final_capture_seq")
            stream = self.service.runtime.ledger.close_bridge_stream(
                binding.bridge_instance_id,
                final_capture_seq=final_capture_seq,  # type: ignore[arg-type]
                connected_nonce=self.connection_nonce,
            )
            return stream.model_dump(mode="json")
        if method == "bridge.close.recover":
            self._only(
                params,
                {
                    "brain_id",
                    "bridge_instance_id",
                    "recovery_token",
                    "final_capture_seq",
                },
            )
            brain_id = validate_id(params.get("brain_id"))  # type: ignore[arg-type]
            bridge_instance_id = validate_id(
                params.get("bridge_instance_id")  # type: ignore[arg-type]
            )
            try:
                engine = self.service.runtime.engine(brain_id)
            except KeyError:
                raise BridgeBindingError(
                    "bridge recovery provenance does not match"
                ) from None
            stream = self.service.runtime.ledger.recover_bridge_close(
                bridge_instance_id,
                brain_id=brain_id,
                server_actor_id=engine.actor_id,
                server_adapter_id=self.service.server_adapter_id,
                recovery_token=params.get("recovery_token"),  # type: ignore[arg-type]
                final_capture_seq=params.get("final_capture_seq"),  # type: ignore[arg-type]
            )
            return stream.model_dump(mode="json")
        if method == "daemon.shutdown":
            self._only(params, set())
            self._shutdown_requested = True
            return {"accepted": True, "instance_nonce": self.service.instance_nonce}
        raise ProtocolFault("method_not_found", "method is not available")

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def close(self) -> None:
        with self._lifecycle:
            if self._closed:
                return
            self._closed = True
            seen: set[str] = set()
            for binding in self._bindings.values():
                if binding.bridge_instance_id in seen:
                    continue
                seen.add(binding.bridge_instance_id)
                with suppress(BridgeBindingError, BridgeClosedError):
                    self.service.runtime.ledger.disconnect_bridge_stream(
                        binding.bridge_instance_id,
                        connected_nonce=self.connection_nonce,
                    )


__all__ = ["ProtocolConnection", "ProtocolService"]
