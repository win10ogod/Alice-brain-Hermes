from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from alice_brain_hermes.hermes.identity import (
    IDENTITY_LLM_MODE_ENV,
    IdentityLlmMode,
    IdentityNamingWorker,
    NamingRunResult,
    read_identity_llm_mode,
)
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.identity import (
    IdentityChoiceV1,
    IdentityNamingLeaseV1,
)


def lease() -> IdentityNamingLeaseV1:
    return IdentityNamingLeaseV1(
        lease_id=new_id(),
        brain_id=new_id(),
        state_sequence=1,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )


class LeasePort:
    def __init__(self, active: IdentityNamingLeaseV1 | None) -> None:
        self.active = active
        self.claim_count = 0
        self.completed: list[tuple[str, IdentityChoiceV1]] = []
        self.failed: list[tuple[str, str]] = []
        self.complete_status = "completed"
        self.failure_status = "failed"

    def claim(self) -> IdentityNamingLeaseV1 | None:
        self.claim_count += 1
        return self.active

    def complete(
        self,
        lease_id: str,
        choice: IdentityChoiceV1,
    ) -> str:
        self.completed.append((lease_id, choice))
        return self.complete_status

    def fail(self, lease_id: str, failure_code: str) -> str:
        self.failed.append((lease_id, failure_code))
        return self.failure_status


class StructuredLlm:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, object]] = []

    def complete_structured(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=self.parsed, content_type="json")


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, IdentityLlmMode.OFF),
        ({IDENTITY_LLM_MODE_ENV: "off"}, IdentityLlmMode.OFF),
        (
            {IDENTITY_LLM_MODE_ENV: "name_when_unnamed"},
            IdentityLlmMode.NAME_WHEN_UNNAMED,
        ),
    ],
)
def test_identity_mode_is_explicit_and_defaults_off(
    environment: dict[str, str], expected: IdentityLlmMode
) -> None:
    assert read_identity_llm_mode(environment) is expected


@pytest.mark.parametrize("value", ["", "ON", "always", " name_when_unnamed "])
def test_identity_mode_rejects_unknown_or_implicitly_normalized_values(
    value: str,
) -> None:
    with pytest.raises(ValueError, match=IDENTITY_LLM_MODE_ENV):
        read_identity_llm_mode({IDENTITY_LLM_MODE_ENV: value})


def test_off_mode_never_claims_lease_or_reads_lazy_llm() -> None:
    port = LeasePort(lease())
    llm_reads = 0

    def llm_factory() -> object:
        nonlocal llm_reads
        llm_reads += 1
        return object()

    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.OFF,
        lease_port=port,
        llm_factory=llm_factory,
    )

    assert worker.run_once() is NamingRunResult.DISABLED
    assert port.claim_count == 0
    assert llm_reads == 0


def test_no_lease_never_reads_lazy_llm() -> None:
    port = LeasePort(None)
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: (_ for _ in ()).throw(
            AssertionError("LLM was read without an unnamed lease")
        ),
    )

    assert worker.run_once() is NamingRunResult.IDLE
    assert port.claim_count == 1


def test_worker_uses_host_defaults_and_commits_only_exact_structured_choice() -> None:
    active = lease()
    port = LeasePort(active)
    llm = StructuredLlm({"name": "Mira", "reason": "A self-selected name."})
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: llm,
    )

    assert worker.run_once() is NamingRunResult.COMPLETED

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert set(call) == {
        "input",
        "instructions",
        "json_schema",
        "purpose",
        "schema_name",
    }
    assert not {
        "agent_id",
        "max_tokens",
        "model",
        "profile",
        "provider",
        "system_prompt",
        "temperature",
        "timeout",
    }.intersection(call)
    rendered_input = repr(call["input"]).casefold()
    rendered_prompt = f"{rendered_input} {call['instructions']!r}".casefold()
    assert "conversation" not in rendered_prompt
    assert "trust" not in rendered_prompt
    assert "jailbreak" not in rendered_prompt
    assert "alice" not in rendered_prompt
    assert port.completed == [
        (
            active.lease_id,
            IdentityChoiceV1(name="Mira", reason="A self-selected name."),
        )
    ]
    assert port.failed == []


@pytest.mark.parametrize(
    "parsed",
    [
        None,
        "Mira",
        {"name": "Mira"},
        {"name": "Mira", "reason": "why", "extra": True},
        {"name": " Alice ", "reason": "why"},
        {"name": "", "reason": "why"},
        {"name": "Mira", "reason": ""},
    ],
)
def test_invalid_structured_choice_is_failed_without_default_name(
    parsed: object,
) -> None:
    active = lease()
    port = LeasePort(active)
    llm = StructuredLlm(parsed)
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: llm,
    )

    assert worker.run_once() is NamingRunResult.FAILED
    assert port.completed == []
    assert port.failed == [(active.lease_id, "invalid_structured_choice")]


def test_provider_failure_records_only_sanitized_type_not_message() -> None:
    active = lease()
    port = LeasePort(active)

    class FailedLlm:
        def complete_structured(self, **_kwargs: object) -> object:
            raise RuntimeError("secret provider response and raw user text")

    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=FailedLlm,
    )

    assert worker.run_once() is NamingRunResult.FAILED
    assert port.completed == []
    assert port.failed == [(active.lease_id, "llm_error.RuntimeError")]


def test_provider_value_error_is_not_misreported_as_a_bad_choice() -> None:
    active = lease()
    port = LeasePort(active)

    class FailedLlm:
        def complete_structured(self, **_kwargs: object) -> object:
            raise ValueError("secret provider response")

    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=FailedLlm,
    )

    assert worker.run_once() is NamingRunResult.FAILED
    assert port.completed == []
    assert port.failed == [(active.lease_id, "llm_error.ValueError")]


def test_non_json_structured_result_is_an_invalid_choice() -> None:
    active = lease()
    port = LeasePort(active)
    llm = StructuredLlm({"name": "Mira", "reason": "chosen"})

    def complete_structured(**kwargs: object) -> object:
        llm.calls.append(kwargs)
        return SimpleNamespace(parsed=llm.parsed, content_type="text")

    llm.complete_structured = complete_structured  # type: ignore[method-assign]
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: llm,
    )

    assert worker.run_once() is NamingRunResult.FAILED
    assert port.completed == []
    assert port.failed == [(active.lease_id, "invalid_structured_choice")]


def test_late_completion_is_reported_as_superseded_without_retry_or_suffix() -> None:
    active = lease()
    port = LeasePort(active)
    port.complete_status = "superseded"
    llm = StructuredLlm({"name": "Mira", "reason": "chosen"})
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: llm,
    )

    assert worker.run_once() is NamingRunResult.SUPERSEDED
    assert len(port.completed) == 1
    assert port.completed[0][1].name == "Mira"
    assert len(llm.calls) == 1


def test_background_worker_uses_its_own_named_thread() -> None:
    port = LeasePort(None)
    claimed = threading.Event()
    original_claim = port.claim

    def observed_claim() -> IdentityNamingLeaseV1 | None:
        claimed.set()
        return original_claim()

    port.claim = observed_claim  # type: ignore[method-assign]
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: object(),
        poll_interval_seconds=0.01,
    )

    worker.start()
    try:
        assert claimed.wait(timeout=1.0)
        assert worker.thread_name == "alice-brain-hermes-identity"
        assert worker.worker_started is True
    finally:
        worker.stop_for_test()

    assert worker.worker_started is False
