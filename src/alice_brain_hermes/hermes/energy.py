"""Lease-bound energy assessment through Hermes' host-owned LLM."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from alice_brain_hermes.core.events import thaw_json
from alice_brain_hermes.protocol.energy import (
    ENERGY_DIMENSIONS,
    EnergyAssessmentChoiceV1,
    EnergyAssessmentLeaseV1,
)

ENERGY_PROMPT_VERSION = "alice-energy-v1"
_ENERGY_THREAD_NAME = "alice-brain-hermes-energy"
_FAILURE_TYPE_PATTERN = re.compile(r"[^A-Za-z0-9_]")

_ENERGY_INSTRUCTIONS = (
    "Assess the supplied action using Alice's PC/E/ST/RD engineering model. "
    "Return exactly one JSON object matching the schema. Every numeric value "
    "must be your calibrated assessment from the supplied evidence. Classify "
    "each dimension as evidenced or unknown; never use hidden defaults. "
    "Evidence text must cite only observable fields in the supplied input."
)


def _energy_choice_schema() -> dict[str, object]:
    numeric_01 = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    numeric_signed = {"type": "number", "minimum": -1.0, "maximum": 1.0}
    properties: dict[str, object] = {
        "deficits": {
            "type": "object",
            "additionalProperties": numeric_01,
            "maxProperties": 16,
        },
        "salience": numeric_01,
        "urgency": numeric_01,
        "valence": numeric_signed,
        "arousal": numeric_signed,
        "control": numeric_01,
        "resources": numeric_01,
        "cost": numeric_01,
        "personality_relevance": numeric_01,
        "evidence_basis": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                dimension: {"type": "string", "minLength": 1, "maxLength": 512}
                for dimension in ENERGY_DIMENSIONS
            },
        },
        "unknown_dimensions": {
            "type": "array",
            "items": {"type": "string", "enum": list(ENERGY_DIMENSIONS)},
            "uniqueItems": True,
            "maxItems": len(ENERGY_DIMENSIONS),
        },
        "summary": {"type": "string", "minLength": 1, "maxLength": 512},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


class EnergyRunResult(StrEnum):
    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class EnergyAssessmentLeasePort(Protocol):
    def claim(self) -> EnergyAssessmentLeaseV1 | None: ...

    def complete(
        self,
        lease_id: str,
        choice: EnergyAssessmentChoiceV1,
        provenance: dict[str, object],
    ) -> Literal["completed", "failed", "superseded"]: ...

    def fail(
        self, lease_id: str, failure_code: str
    ) -> Literal["failed", "superseded"]: ...


class StructuredEnergyLlm(Protocol):
    def complete_structured(self, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class _EnergyTerminalIntent:
    operation: Literal["complete", "fail"]
    lease_id: str
    choice: EnergyAssessmentChoiceV1 | None = None
    provenance: dict[str, object] | None = None
    failure_code: str | None = None

    def submit(self, port: EnergyAssessmentLeasePort) -> EnergyRunResult:
        if self.operation == "complete":
            if self.choice is None or self.provenance is None:
                raise RuntimeError("energy completion intent is invalid")
            status = port.complete(self.lease_id, self.choice, self.provenance)
        else:
            if self.failure_code is None:
                raise RuntimeError("energy failure intent is invalid")
            status = port.fail(self.lease_id, self.failure_code)
        if status not in {"completed", "failed", "superseded"}:
            raise RuntimeError("energy lease port returned an invalid status")
        return EnergyRunResult(status)


def _sanitized_error_type(error: Exception) -> str:
    name = _FAILURE_TYPE_PATTERN.sub("_", type(error).__name__)[:80]
    return name or "Exception"


def _bounded_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"host energy {field} is invalid")
    return value


def _usage_provenance(value: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    ):
        item = getattr(value, field, 0)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError("host energy usage is invalid")
        result[field] = item
    cost = getattr(value, "cost_usd", None)
    if cost is not None and (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or cost < 0
    ):
        raise ValueError("host energy cost is invalid")
    result["cost_usd"] = None if cost is None else float(cost)
    return result


def _result_provenance(response: object, *, input_sha256: str) -> dict[str, object]:
    audit = getattr(response, "audit", None)
    if not isinstance(audit, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or len(key) > 160
        or len(value) > 512
        for key, value in audit.items()
    ):
        raise ValueError("host energy audit evidence is invalid")
    return {
        "agent_id": _bounded_text(getattr(response, "agent_id", None), field="agent"),
        "audit": dict(sorted(audit.items())),
        "input_sha256": input_sha256,
        "model": _bounded_text(getattr(response, "model", None), field="model"),
        "prompt_version": ENERGY_PROMPT_VERSION,
        "provider": _bounded_text(
            getattr(response, "provider", None), field="provider"
        ),
        "usage": _usage_provenance(getattr(response, "usage", None)),
    }


class EnergyAssessmentWorker:
    """Claim durable action jobs and use only Hermes' host LLM surface."""

    def __init__(
        self,
        *,
        lease_port: EnergyAssessmentLeasePort,
        llm_factory: Callable[[], StructuredEnergyLlm],
        poll_interval_seconds: float = 0.25,
    ) -> None:
        if not callable(llm_factory):
            raise TypeError("llm_factory must be callable")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(float(poll_interval_seconds))
            or poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be finite and positive")
        self._lease_port = lease_port
        self._llm_factory = llm_factory
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._lifecycle_lock = threading.Lock()
        self._iteration_lock = threading.Lock()
        self._stop = threading.Event()
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._terminal_intent: _EnergyTerminalIntent | None = None
        self._last_internal_error_type: str | None = None

    @property
    def worker_started(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def terminal_intent_pending(self) -> bool:
        with self._iteration_lock:
            return self._terminal_intent is not None

    @property
    def last_internal_error_type(self) -> str | None:
        with self._lifecycle_lock:
            return self._last_internal_error_type

    def _submit_terminal_intent(self) -> EnergyRunResult:
        intent = self._terminal_intent
        if intent is None:
            raise RuntimeError("energy terminal intent is missing")
        result = intent.submit(self._lease_port)
        self._terminal_intent = None
        return result

    def _fail(self, lease: EnergyAssessmentLeaseV1, code: str) -> EnergyRunResult:
        self._terminal_intent = _EnergyTerminalIntent(
            operation="fail", lease_id=lease.lease_id, failure_code=code
        )
        return self._submit_terminal_intent()

    def _run_assessment(self, lease: EnergyAssessmentLeaseV1) -> EnergyRunResult:
        assessment_json = json.dumps(
            thaw_json(lease.assessment_input),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        input_sha256 = hashlib.sha256(assessment_json.encode("utf-8")).hexdigest()
        try:
            response = self._llm_factory().complete_structured(
                input=[{"type": "text", "text": assessment_json}],
                instructions=_ENERGY_INSTRUCTIONS,
                json_schema=_energy_choice_schema(),
                purpose="alice_energy_assessment",
                schema_name="alice_energy_assessment_v1",
            )
        except Exception as error:
            return self._fail(lease, f"llm_error.{_sanitized_error_type(error)}")
        choice: EnergyAssessmentChoiceV1 | None = None
        provenance: dict[str, object] | None = None
        try:
            if getattr(response, "content_type", None) != "json":
                raise ValueError("structured result is not JSON")
            parsed = getattr(response, "parsed", None)
            expected = set(EnergyAssessmentChoiceV1.model_fields) - {"schema_version"}
            if type(parsed) is not dict or set(parsed) != expected:
                raise ValueError("structured energy result has an invalid shape")
            choice = EnergyAssessmentChoiceV1.model_validate(parsed, strict=True)
            provenance = _result_provenance(response, input_sha256=input_sha256)
        except Exception:
            pass
        if choice is None or provenance is None:
            return self._fail(lease, "invalid_structured_assessment")
        self._terminal_intent = _EnergyTerminalIntent(
            operation="complete",
            lease_id=lease.lease_id,
            choice=choice,
            provenance=provenance,
        )
        return self._submit_terminal_intent()

    def run_once(self) -> EnergyRunResult:
        with self._iteration_lock:
            if self._terminal_intent is not None:
                return self._submit_terminal_intent()
            lease = self._lease_port.claim()
            if lease is None:
                return EnergyRunResult.IDLE
            return self._run_assessment(lease)

    def _run(self) -> None:
        current = threading.current_thread()
        try:
            while not self._stop_requested:
                try:
                    result = self.run_once()
                except Exception as error:
                    with self._lifecycle_lock:
                        self._last_internal_error_type = _sanitized_error_type(error)
                else:
                    if result is not EnergyRunResult.IDLE:
                        with self._lifecycle_lock:
                            self._last_internal_error_type = None
                deadline = time.monotonic() + self._poll_interval_seconds
                while not self._stop_requested:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._stop.wait(min(remaining, 0.05))
        finally:
            with self._lifecycle_lock:
                if self._thread is current:
                    self._thread = None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_requested = False
            self._stop.clear()
            worker = threading.Thread(
                target=self._run,
                name=_ENERGY_THREAD_NAME,
                daemon=True,
            )
            self._thread = worker
            try:
                worker.start()
            except BaseException:
                if not worker.is_alive() and self._thread is worker:
                    self._thread = None
                raise

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._lifecycle_lock:
            worker = self._thread
            self._stop_requested = True
            self._stop.set()
        if worker is None:
            return
        if worker is threading.current_thread():
            raise RuntimeError("energy worker cannot join its own thread")
        worker.join(timeout)
        if worker.is_alive():
            raise RuntimeError("energy worker did not stop before timeout")
        with self._lifecycle_lock:
            if self._thread is worker:
                self._thread = None


__all__ = [
    "ENERGY_PROMPT_VERSION",
    "EnergyAssessmentLeasePort",
    "EnergyAssessmentWorker",
    "EnergyRunResult",
    "StructuredEnergyLlm",
]
