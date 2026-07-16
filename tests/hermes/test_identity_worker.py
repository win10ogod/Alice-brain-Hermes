from __future__ import annotations

import threading
import time
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
        {"name": "Å", "reason": "not already normalized"},
        {"name": "\U0001e030", "reason": "outside Unicode 14"},
        {"name": "Mi\x00ra", "reason": "contains a control"},
        {"name": "Mira\ufdd0", "reason": "contains a noncharacter"},
        {"name": "😀" * 129, "reason": "exceeds the UTF-8 byte bound"},
        {"name": "Mira", "reason": "line one\nline two"},
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


def test_identity_choice_accepts_an_exact_normalized_utf8_boundary() -> None:
    choice = IdentityChoiceV1(name="😀" * 128, reason="chosen")

    assert choice.name == "😀" * 128
    assert len(choice.name.encode("utf-8")) == 512


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


def test_transient_completion_failure_retries_the_exact_terminal_intent() -> None:
    active = lease()

    class TransientCompletionPort(LeasePort):
        def claim(self) -> IdentityNamingLeaseV1 | None:
            self.claim_count += 1
            return active if self.claim_count == 1 else None

        def complete(
            self,
            lease_id: str,
            choice: IdentityChoiceV1,
        ) -> str:
            self.completed.append((lease_id, choice))
            if len(self.completed) == 1:
                raise MemoryError("transient terminal commit failure")
            return "completed"

    port = TransientCompletionPort(active)
    llm = StructuredLlm({"name": "Mira", "reason": "chosen"})
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: llm,
    )

    with pytest.raises(MemoryError, match="terminal commit"):
        worker.run_once()

    assert worker.run_once() is NamingRunResult.COMPLETED
    assert port.claim_count == 1
    assert len(llm.calls) == 1
    assert port.completed == [port.completed[0], port.completed[0]]


def test_transient_failure_commit_retries_without_reclaiming_or_recalling_llm() -> None:
    active = lease()

    class TransientFailurePort(LeasePort):
        def claim(self) -> IdentityNamingLeaseV1 | None:
            self.claim_count += 1
            return active if self.claim_count == 1 else None

        def fail(self, lease_id: str, failure_code: str) -> str:
            self.failed.append((lease_id, failure_code))
            if len(self.failed) == 1:
                raise MemoryError("transient failure-evidence commit failure")
            return "failed"

    class FailedLlm:
        def __init__(self) -> None:
            self.calls = 0

        def complete_structured(self, **_kwargs: object) -> object:
            self.calls += 1
            raise RuntimeError("provider failed")

    port = TransientFailurePort(active)
    llm = FailedLlm()
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: llm,
    )

    with pytest.raises(MemoryError, match="failure-evidence commit"):
        worker.run_once()

    assert worker.run_once() is NamingRunResult.FAILED
    assert port.claim_count == 1
    assert llm.calls == 1
    assert port.failed == [port.failed[0], port.failed[0]]


def test_loop_guard_uses_a_bool_latch_not_the_fallible_wake_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = LeasePort(None)
    claimed = threading.Event()

    def observed_claim() -> IdentityNamingLeaseV1 | None:
        claimed.set()
        return None

    port.claim = observed_claim  # type: ignore[method-assign]
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: object(),
        poll_interval_seconds=0.01,
    )

    def fail_is_set() -> bool:
        raise MemoryError("loop guard must not call Event.is_set")

    worker._stop.is_set = fail_is_set  # type: ignore[method-assign]
    monkeypatch.setattr(threading, "excepthook", lambda _args: None)

    worker.start()
    try:
        assert claimed.wait(timeout=1.0)
        assert worker.worker_started is True
    finally:
        worker.stop_for_test()


def test_background_wait_failure_keeps_thread_and_health_error_alive() -> None:
    port = LeasePort(None)
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: object(),
        poll_interval_seconds=0.01,
    )
    first_wait_failed = threading.Event()
    second_iteration_completed = threading.Event()
    original_wait = worker._stop.wait
    wait_calls = 0

    def transient_wait(timeout: float | None = None) -> bool:
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            first_wait_failed.set()
            raise MemoryError("transient poll wait failure")
        if port.claim_count >= 2:
            second_iteration_completed.set()
        return original_wait(timeout)

    worker._stop.wait = transient_wait  # type: ignore[method-assign]

    worker.start()
    try:
        assert first_wait_failed.wait(timeout=1.0)
        assert second_iteration_completed.wait(timeout=1.0)
        assert worker.worker_started is True
        assert port.claim_count >= 2
        assert worker.last_internal_error_type == "MemoryError"
    finally:
        worker.stop_for_test()


def test_stop_request_survives_a_wake_event_memory_error() -> None:
    port = LeasePort(None)
    claimed = threading.Event()

    def observed_claim() -> IdentityNamingLeaseV1 | None:
        claimed.set()
        return None

    port.claim = observed_claim  # type: ignore[method-assign]
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: object(),
        poll_interval_seconds=0.01,
    )

    def fail_set() -> None:
        raise MemoryError("transient wake signal failure")

    worker.start()
    assert claimed.wait(timeout=1.0)
    worker._stop.set = fail_set  # type: ignore[method-assign]

    worker.stop_for_test(timeout=1.0)

    assert worker.worker_started is False


def test_unexpected_thread_exit_clears_pointer_and_same_worker_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FatalWorkerExit(BaseException):
        pass

    port = LeasePort(None)
    recovered_claim = threading.Event()
    claims = 0

    def fatal_then_recover() -> IdentityNamingLeaseV1 | None:
        nonlocal claims
        claims += 1
        if claims == 1:
            raise FatalWorkerExit()
        recovered_claim.set()
        return None

    port.claim = fatal_then_recover  # type: ignore[method-assign]
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: object(),
        poll_interval_seconds=0.01,
    )
    failed_thread = threading.Event()
    monkeypatch.setattr(threading, "excepthook", lambda _args: failed_thread.set())

    worker.start()
    assert failed_thread.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while worker.worker_started and time.monotonic() < deadline:
        time.sleep(0.001)
    assert worker.worker_started is False
    assert worker._thread is None

    worker.start()
    try:
        assert recovered_claim.wait(timeout=1.0)
        assert worker.worker_started is True
    finally:
        worker.stop_for_test()


def test_ambiguous_worker_started_probe_retains_owner_and_blocks_second_start() -> None:
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=LeasePort(None),
        llm_factory=lambda: object(),
    )

    class AmbiguousOwner:
        def is_alive(self) -> bool:
            raise MemoryError("thread ownership probe failed")

    owner = AmbiguousOwner()
    worker._thread = owner  # type: ignore[assignment]

    with pytest.raises(MemoryError, match="ownership probe"):
        _ = worker.worker_started
    assert worker._thread is owner

    with pytest.raises(MemoryError, match="ownership probe"):
        worker.start()
    assert worker._thread is owner


def test_prelaunch_thread_start_failure_clears_known_dead_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StartBeforeLaunchFailure(BaseException):
        pass

    class NeverStartedThread:
        join_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            raise StartBeforeLaunchFailure()

        def is_alive(self) -> bool:
            return False

        def join(self, _timeout: float) -> None:
            type(self).join_calls += 1

    monkeypatch.setattr(threading, "Thread", NeverStartedThread)
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=LeasePort(None),
        llm_factory=lambda: object(),
    )

    with pytest.raises(StartBeforeLaunchFailure):
        worker.start()

    assert worker._thread is None
    assert worker.worker_started is False
    worker.stop_for_test()
    assert NeverStartedThread.join_calls == 0


def test_postlaunch_thread_start_failure_retains_live_owner_and_blocks_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StartAfterLaunchFailure(BaseException):
        pass

    class LiveThread:
        instances = 0

        def __init__(self, **_kwargs: object) -> None:
            type(self).instances += 1

        def start(self) -> None:
            raise StartAfterLaunchFailure()

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(threading, "Thread", LiveThread)
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=LeasePort(None),
        llm_factory=lambda: object(),
    )

    with pytest.raises(StartAfterLaunchFailure):
        worker.start()
    owner = worker._thread

    assert owner is not None
    assert worker.worker_started is True
    worker.start()
    assert worker._thread is owner
    assert LiveThread.instances == 1


@pytest.mark.parametrize("failure_stage", ["join", "post_join_probe"])
def test_stop_probe_failure_retains_owner_and_prevents_a_second_worker(
    failure_stage: str,
) -> None:
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=LeasePort(None),
        llm_factory=lambda: object(),
    )

    class AmbiguousOwner:
        def __init__(self) -> None:
            self.join_calls = 0
            self.probe_calls = 0

        def join(self, _timeout: float) -> None:
            self.join_calls += 1
            if failure_stage == "join":
                raise MemoryError("thread join failed")

        def is_alive(self) -> bool:
            self.probe_calls += 1
            if failure_stage == "post_join_probe":
                raise MemoryError("post-join ownership probe failed")
            return True

    owner = AmbiguousOwner()
    worker._thread = owner  # type: ignore[assignment]

    with pytest.raises(MemoryError):
        worker.stop_for_test(timeout=1.0)
    assert worker._thread is owner

    if failure_stage == "join":
        worker.start()
    else:
        with pytest.raises(MemoryError, match="ownership probe"):
            worker.start()
    assert worker._thread is owner


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
