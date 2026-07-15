"""Optional, lease-bound identity naming outside Hermes hook callbacks."""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Literal, Protocol

from alice_brain_hermes.protocol.identity import (
    IDENTITY_NAME_MAX_CODEPOINTS,
    IDENTITY_REASON_MAX_CODEPOINTS,
    IdentityChoiceV1,
    IdentityNamingLeaseV1,
)

IDENTITY_LLM_MODE_ENV = "ALICE_BRAIN_HERMES_IDENTITY_LLM_MODE"
_IDENTITY_THREAD_NAME = "alice-brain-hermes-identity"
_FAILURE_TYPE_PATTERN = re.compile(r"[^A-Za-z0-9_]")

_IDENTITY_INSTRUCTIONS = (
    "Select an optional stable display name for your own currently unnamed "
    "runtime identity. Return exactly one JSON object matching the supplied "
    "schema. Make only this naming choice and do not assert unrelated facts."
)


def _identity_choice_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": IDENTITY_NAME_MAX_CODEPOINTS,
            },
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": IDENTITY_REASON_MAX_CODEPOINTS,
            },
        },
        "required": ["name", "reason"],
    }


def _identity_input() -> list[dict[str, str]]:
    return [
        {
            "type": "text",
            "text": (
                "Provide one self-selected name and a brief reason for selecting it."
            ),
        }
    ]


class IdentityLlmMode(StrEnum):
    """Explicit operator policy for optional host-LLM identity naming."""

    OFF = "off"
    NAME_WHEN_UNNAMED = "name_when_unnamed"


class NamingRunResult(StrEnum):
    """One bounded identity-worker iteration outcome."""

    DISABLED = "disabled"
    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class IdentityNamingLeasePort(Protocol):
    """Daemon-owned lease operations; completion is compare-and-swap guarded."""

    def claim(self) -> IdentityNamingLeaseV1 | None: ...

    def complete(
        self,
        lease_id: str,
        choice: IdentityChoiceV1,
    ) -> Literal["completed", "failed", "superseded"]: ...

    def fail(
        self,
        lease_id: str,
        failure_code: str,
    ) -> Literal["failed", "superseded"]: ...


class StructuredIdentityLlm(Protocol):
    """Minimal host-owned LLM surface used by the separate naming worker."""

    def complete_structured(self, **kwargs: object) -> object: ...


def read_identity_llm_mode(
    environ: Mapping[str, str],
) -> IdentityLlmMode:
    """Read an exact opt-in value; absence safely keeps identity naming off."""

    raw = environ.get(IDENTITY_LLM_MODE_ENV)
    if raw is None:
        return IdentityLlmMode.OFF
    try:
        return IdentityLlmMode(raw)
    except ValueError as error:
        raise ValueError(
            f"{IDENTITY_LLM_MODE_ENV} must be exactly 'off' or 'name_when_unnamed'"
        ) from error


def _sanitized_error_type(error: Exception) -> str:
    name = _FAILURE_TYPE_PATTERN.sub("_", type(error).__name__)[:80]
    return name or "Exception"


class IdentityNamingWorker:
    """Run optional self-naming on a thread independent of bridge delivery."""

    def __init__(
        self,
        *,
        mode: IdentityLlmMode,
        lease_port: IdentityNamingLeasePort,
        llm_factory: Callable[[], StructuredIdentityLlm],
        poll_interval_seconds: float = 1.0,
    ) -> None:
        if not isinstance(mode, IdentityLlmMode):
            raise TypeError("mode must be an IdentityLlmMode")
        if not callable(llm_factory):
            raise TypeError("llm_factory must be callable")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(float(poll_interval_seconds))
            or poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be finite and positive")

        self._mode = mode
        self._lease_port = lease_port
        self._llm_factory = llm_factory
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._stop = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._iteration_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_internal_error_type: str | None = None

    @property
    def thread_name(self) -> str:
        return _IDENTITY_THREAD_NAME

    @property
    def worker_started(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def last_internal_error_type(self) -> str | None:
        """Expose a sanitized background-loop failure without leaking content."""

        with self._lifecycle_lock:
            return self._last_internal_error_type

    def _record_failure(
        self,
        lease: IdentityNamingLeaseV1,
        failure_code: str,
    ) -> NamingRunResult:
        status = self._lease_port.fail(lease.lease_id, failure_code)
        if status == "failed":
            return NamingRunResult.FAILED
        if status == "superseded":
            return NamingRunResult.SUPERSEDED
        raise RuntimeError("identity lease port returned an invalid failure status")

    def _run_leased_choice(
        self,
        lease: IdentityNamingLeaseV1,
    ) -> NamingRunResult:
        try:
            llm = self._llm_factory()
            response = llm.complete_structured(
                input=_identity_input(),
                instructions=_IDENTITY_INSTRUCTIONS,
                json_schema=_identity_choice_schema(),
                purpose="identity_self_naming",
                schema_name="identity_choice_v1",
            )
        except Exception as error:
            return self._record_failure(
                lease,
                f"llm_error.{_sanitized_error_type(error)}",
            )

        choice: IdentityChoiceV1 | None = None
        try:
            if getattr(response, "content_type", None) != "json":
                raise ValueError("structured result is not JSON")
            parsed = getattr(response, "parsed", None)
            if type(parsed) is not dict or set(parsed) != {"name", "reason"}:
                raise ValueError("structured result has an invalid shape")
            choice = IdentityChoiceV1.model_validate(parsed, strict=True)
        except Exception:
            pass
        if choice is None:
            return self._record_failure(lease, "invalid_structured_choice")

        status = self._lease_port.complete(lease.lease_id, choice)
        if status == "completed":
            return NamingRunResult.COMPLETED
        if status == "failed":
            return NamingRunResult.FAILED
        if status == "superseded":
            return NamingRunResult.SUPERSEDED
        raise RuntimeError("identity lease port returned an invalid completion status")

    def run_once(self) -> NamingRunResult:
        """Claim at most one lease and make at most one host-LLM request."""

        with self._iteration_lock:
            if self._mode is IdentityLlmMode.OFF:
                return NamingRunResult.DISABLED
            lease = self._lease_port.claim()
            if lease is None:
                return NamingRunResult.IDLE
            return self._run_leased_choice(lease)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as error:
                # The daemon lease/store remains authoritative. A transient worker
                # failure must not terminate this independent polling thread.
                error_type = _sanitized_error_type(error)
                with self._lifecycle_lock:
                    self._last_internal_error_type = error_type
            else:
                with self._lifecycle_lock:
                    self._last_internal_error_type = None
            if self._stop.wait(self._poll_interval_seconds):
                return

    def start(self) -> None:
        """Start one daemon thread without coupling it to bridge delivery."""

        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            thread = threading.Thread(
                target=self._run,
                name=_IDENTITY_THREAD_NAME,
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop_for_test(self, *, timeout: float = 5.0) -> None:
        """Stop and join the owned thread; production wiring owns final shutdown."""

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive")
        with self._lifecycle_lock:
            thread = self._thread
            self._stop.set()
        if thread is None:
            return
        if thread is threading.current_thread():
            raise RuntimeError("identity worker cannot join its own thread")
        thread.join(float(timeout))
        if thread.is_alive():
            raise RuntimeError("identity worker did not stop before timeout")
        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None


__all__ = [
    "IDENTITY_LLM_MODE_ENV",
    "IdentityLlmMode",
    "IdentityNamingLeasePort",
    "IdentityNamingWorker",
    "NamingRunResult",
    "StructuredIdentityLlm",
    "read_identity_llm_mode",
]
