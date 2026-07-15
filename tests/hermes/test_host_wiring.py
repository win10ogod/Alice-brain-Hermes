from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alice_brain_hermes.errors import DaemonClientError
from alice_brain_hermes.hermes.bridge import HookBridge
from alice_brain_hermes.hermes.identity import (
    IdentityLlmMode,
    IdentityNamingWorker,
)
from alice_brain_hermes.hermes.identity_client import (
    DaemonIdentityNamingLeasePort,
    hermes_brain_profile,
)
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.identity import (
    IdentityChoiceV1,
    IdentityNamingLeaseV1,
)
from alice_brain_hermes.protocol.models import BrainProfileV1


class _HostContext:
    def __init__(self, *, profile_name: str = "default") -> None:
        self._profile_name = profile_name
        self.profile_reads = 0
        self.llm_reads = 0
        self.host_llm = object()

    @property
    def profile_name(self) -> str:
        self.profile_reads += 1
        return self._profile_name

    @property
    def llm(self) -> object:
        self.llm_reads += 1
        return self.host_llm


def test_host_access_resolves_one_profile_once_across_threads() -> None:
    from alice_brain_hermes.hermes import registration

    context = _HostContext(profile_name="research")
    access = registration._HermesHostAccess(context)  # type: ignore[attr-defined]
    callers = 24
    barrier = threading.Barrier(callers)

    def read_profile() -> BrainProfileV1:
        barrier.wait()
        return access.brain_profile()

    with ThreadPoolExecutor(max_workers=callers) as executor:
        profiles = list(executor.map(lambda _index: read_profile(), range(callers)))

    assert context.profile_reads == 1
    assert all(profile is profiles[0] for profile in profiles)
    assert profiles[0] == hermes_brain_profile("research")
    assert context.llm_reads == 0


def test_host_access_resolves_host_llm_once_without_overrides() -> None:
    from alice_brain_hermes.hermes import registration

    context = _HostContext()
    access = registration._HermesHostAccess(context)  # type: ignore[attr-defined]

    assert access.llm() is context.host_llm
    assert access.llm() is context.host_llm
    assert context.llm_reads == 1
    assert context.profile_reads == 0


def test_register_binds_context_without_reading_profile_or_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)
    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")

    class Context(_HostContext):
        def __init__(self) -> None:
            super().__init__()
            self.hooks: list[tuple[str, object]] = []
            self.cli: list[dict[str, object]] = []

        def register_hook(self, hook: str, callback: object) -> None:
            self.hooks.append((hook, callback))

        def register_cli_command(self, **kwargs: object) -> None:
            self.cli.append(kwargs)

    context = Context()
    registration.register(context)

    assert bootstrap.host_context_for_worker() is context
    assert context.profile_reads == 0
    assert context.llm_reads == 0


def test_bridge_uses_injected_profile_factory_only_on_connect(tmp_path: Path) -> None:
    expected = hermes_brain_profile("research")
    profile_reads = 0
    resolve_params: list[dict[str, object]] = []

    def profile_factory() -> BrainProfileV1:
        nonlocal profile_reads
        profile_reads += 1
        return expected

    class Client:
        def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
            assert method == "brain.resolve"
            resolve_params.append(params)
            raise DaemonClientError("stop after observing profile")

        @staticmethod
        def close() -> None:
            return None

    bridge = HookBridge(
        tmp_path,
        profile_factory=profile_factory,
        client_factory=lambda *_args, **_kwargs: Client(),
        start_worker_on_capture=False,
    )

    assert profile_reads == 0
    assert bridge._connect() is False  # type: ignore[attr-defined]
    assert profile_reads == 1
    assert resolve_params == [{"profile": expected.model_dump(mode="json")}]


class _StopProbe(BaseException):
    pass


class _FakeProjection:
    @staticmethod
    def read_context() -> None:
        return None


def test_unbound_direct_buffer_keeps_the_default_profile_test_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import bridge as bridge_module
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bridge_arguments: dict[str, object] = {}
    wait_calls = 0

    class Bridge:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            bridge_arguments.update(kwargs)
            self.projections = _FakeProjection()
            self.worker_started = False

        def start_worker(self) -> None:
            self.worker_started = True

    def stop_from_wait(_timeout: float) -> None:
        nonlocal wait_calls
        wait_calls += 1
        raise _StopProbe()

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "wait", stop_from_wait)
    monkeypatch.setattr(
        bootstrap,
        "next_for_worker",
        lambda: (_ for _ in ()).throw(_StopProbe()),
    )

    with pytest.raises(_StopProbe):
        registration._bootstrap_worker_main(bootstrap)  # type: ignore[attr-defined]

    # Task6 contains worker faults and backs off before the next iteration.
    assert wait_calls == 1
    assert "profile_factory" not in bridge_arguments


def test_bootstrap_off_mode_shares_profile_but_never_builds_identity_or_reads_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import bridge as bridge_module
    from alice_brain_hermes.hermes import identity as identity_module
    from alice_brain_hermes.hermes import identity_client as identity_client_module
    from alice_brain_hermes.hermes import registration

    monkeypatch.delenv(identity_module.IDENTITY_LLM_MODE_ENV, raising=False)
    context = _HostContext(profile_name="research")
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap.bind_host_context(context)
    captured: dict[str, object] = {}

    class Bridge:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            captured.update(kwargs)
            self.projections = _FakeProjection()
            self.worker_started = False

        def start_worker(self) -> None:
            self.worker_started = True

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(
        identity_client_module,
        "DaemonIdentityNamingLeasePort",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("off mode constructed an identity lease port")
        ),
    )
    monkeypatch.setattr(
        identity_module,
        "IdentityNamingWorker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("off mode constructed an identity worker")
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "next_for_worker",
        lambda: (_ for _ in ()).throw(_StopProbe()),
    )

    with pytest.raises(_StopProbe):
        registration._bootstrap_worker_main(bootstrap)  # type: ignore[attr-defined]

    profile_factory = captured["profile_factory"]
    assert callable(profile_factory)
    assert profile_factory() is profile_factory()
    assert context.profile_reads == 1
    assert context.llm_reads == 0
    assert bootstrap.identity_worker_for_test is None


def test_bootstrap_opt_in_gives_bridge_and_identity_one_factory_and_stops_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import bridge as bridge_module
    from alice_brain_hermes.hermes import identity as identity_module
    from alice_brain_hermes.hermes import identity_client as identity_client_module
    from alice_brain_hermes.hermes import registration

    monkeypatch.setenv(
        identity_module.IDENTITY_LLM_MODE_ENV,
        identity_module.IdentityLlmMode.NAME_WHEN_UNNAMED.value,
    )
    context = _HostContext(profile_name="research")
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap.bind_host_context(context)
    bridge_arguments: dict[str, object] = {}
    port_arguments: dict[str, object] = {}
    worker_arguments: dict[str, object] = {}
    workers: list[Any] = []

    class Bridge:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            bridge_arguments.update(kwargs)
            self.projections = _FakeProjection()
            self.worker_started = False

        def start_worker(self) -> None:
            self.worker_started = True

    class Port:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            port_arguments.update(kwargs)

    class Worker:
        def __init__(self, **kwargs: object) -> None:
            worker_arguments.update(kwargs)
            self.started = False
            self.stopped = False
            workers.append(self)

        def start(self) -> None:
            self.started = True

        def stop_for_test(self) -> None:
            self.stopped = True

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(identity_client_module, "DaemonIdentityNamingLeasePort", Port)
    monkeypatch.setattr(identity_module, "IdentityNamingWorker", Worker)
    monkeypatch.setattr(
        bootstrap,
        "next_for_worker",
        lambda: (_ for _ in ()).throw(_StopProbe()),
    )

    with pytest.raises(_StopProbe):
        registration._bootstrap_worker_main(bootstrap)  # type: ignore[attr-defined]

    assert len(workers) == 1
    assert workers[0].started is True
    assert workers[0].stopped is True
    assert bootstrap.identity_worker_for_test is None
    assert worker_arguments["mode"] is identity_module.IdentityLlmMode.NAME_WHEN_UNNAMED
    assert worker_arguments["lease_port"] is not None
    bridge_profile_factory = bridge_arguments["profile_factory"]
    port_profile_factory = port_arguments["profile_factory"]
    assert bridge_profile_factory is port_profile_factory
    assert bridge_profile_factory() is port_profile_factory()
    assert context.profile_reads == 1
    llm_factory = worker_arguments["llm_factory"]
    assert llm_factory() is llm_factory()
    assert context.llm_reads == 1
    assert not {
        "agent_id",
        "max_tokens",
        "model",
        "profile",
        "provider",
        "temperature",
        "timeout",
    }.intersection(worker_arguments)


def test_bootstrap_restarts_the_same_owned_identity_worker_after_fatal_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import bridge as bridge_module
    from alice_brain_hermes.hermes import identity as identity_module
    from alice_brain_hermes.hermes import identity_client as identity_client_module
    from alice_brain_hermes.hermes import registration

    monkeypatch.setenv(
        identity_module.IDENTITY_LLM_MODE_ENV,
        identity_module.IdentityLlmMode.NAME_WHEN_UNNAMED.value,
    )
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap.bind_host_context(_HostContext())
    workers: list[Any] = []
    next_calls = 0

    class Bridge:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.projections = _FakeProjection()
            self.worker_started = False

        def start_worker(self) -> None:
            self.worker_started = True

    class Worker:
        def __init__(self, **_kwargs: object) -> None:
            self.start_calls = 0
            self.alive = False
            self.stop_calls = 0
            workers.append(self)

        @property
        def worker_started(self) -> bool:
            return self.alive

        def start(self) -> None:
            self.start_calls += 1
            self.alive = True

        def stop_for_test(self) -> None:
            self.stop_calls += 1
            self.alive = False

    def next_record() -> None:
        nonlocal next_calls
        next_calls += 1
        if next_calls == 1:
            workers[0].alive = False
            return None
        raise _StopProbe()

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(
        identity_client_module,
        "DaemonIdentityNamingLeasePort",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(identity_module, "IdentityNamingWorker", Worker)
    monkeypatch.setattr(bootstrap, "next_for_worker", next_record)
    monkeypatch.setattr(bootstrap, "wait", lambda _timeout: None)

    with pytest.raises(_StopProbe):
        registration._bootstrap_worker_main(bootstrap)  # type: ignore[attr-defined]

    assert len(workers) == 1
    assert workers[0].start_calls == 2
    assert workers[0].stop_calls == 1
    assert bootstrap.identity_worker_for_test is None


def test_bootstrap_retries_prelaunch_failure_on_the_same_adopted_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import bridge as bridge_module
    from alice_brain_hermes.hermes import identity as identity_module
    from alice_brain_hermes.hermes import identity_client as identity_client_module
    from alice_brain_hermes.hermes import registration

    monkeypatch.setenv(
        identity_module.IDENTITY_LLM_MODE_ENV,
        identity_module.IdentityLlmMode.NAME_WHEN_UNNAMED.value,
    )
    context = _HostContext()
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap.bind_host_context(context)
    workers: list[Any] = []
    next_calls = 0

    class Bridge:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.projections = _FakeProjection()
            self.worker_started = False

        def start_worker(self) -> None:
            self.worker_started = True

    class Worker:
        def __init__(self, **_kwargs: object) -> None:
            self.start_calls = 0
            self.alive = False
            self.stop_calls = 0
            workers.append(self)

        @property
        def worker_started(self) -> bool:
            return self.alive

        def start(self) -> None:
            self.start_calls += 1
            if self.start_calls == 1:
                raise RuntimeError("definite prelaunch failure")
            self.alive = True

        def stop_for_test(self) -> None:
            self.stop_calls += 1
            self.alive = False

    def next_record() -> None:
        nonlocal next_calls
        next_calls += 1
        if workers and workers[0].start_calls >= 2:
            raise _StopProbe()
        if next_calls >= 10:
            raise _StopProbe()
        return None

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(
        identity_client_module,
        "DaemonIdentityNamingLeasePort",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(identity_module, "IdentityNamingWorker", Worker)
    monkeypatch.setattr(bootstrap, "next_for_worker", next_record)
    monkeypatch.setattr(bootstrap, "wait", lambda timeout: time.sleep(timeout))

    with pytest.raises(_StopProbe):
        registration._bootstrap_worker_main(bootstrap)  # type: ignore[attr-defined]

    assert len(workers) == 1
    assert workers[0].start_calls == 2
    assert workers[0].stop_calls == 1
    assert context.llm_reads == 0
    assert bootstrap.identity_worker_for_test is None


def test_identity_monitor_retains_ambiguous_owner_without_duplicate_start() -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )

    class Worker:
        def __init__(self) -> None:
            self.start_calls = 0
            self.stopped = False

        @property
        def worker_started(self) -> bool:
            raise MemoryError("identity ownership probe is ambiguous")

        def start(self) -> None:
            self.start_calls += 1

        def stop_for_test(self) -> None:
            self.stopped = True

    worker = Worker()
    bootstrap._adopt_identity_worker(worker)  # type: ignore[attr-defined]

    assert registration._ensure_identity_worker_running(bootstrap) is False  # type: ignore[attr-defined]
    assert bootstrap.identity_worker_for_test is worker
    assert worker.start_calls == 0
    assert bootstrap.health.degraded is True
    assert bootstrap.health.worker_error == "MemoryError"
    bootstrap._stop_identity_worker()  # type: ignore[attr-defined]
    assert worker.stopped is True


def test_actual_identity_worker_restart_replays_exact_ram_terminal_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    class FatalWorkerExit(BaseException):
        pass

    active = IdentityNamingLeaseV1(
        lease_id=new_id(),
        brain_id=new_id(),
        state_sequence=1,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )

    class Port:
        def __init__(self) -> None:
            self.claim_count = 0
            self.claims_before_terminal_ack = 0
            self.completed: list[tuple[str, IdentityChoiceV1]] = []
            self.first_terminal_attempt = threading.Event()
            self.terminal_completed = threading.Event()

        def claim(self) -> IdentityNamingLeaseV1 | None:
            self.claim_count += 1
            if not self.terminal_completed.is_set():
                self.claims_before_terminal_ack += 1
            return active if self.claim_count == 1 else None

        def complete(self, lease_id: str, choice: IdentityChoiceV1) -> str:
            self.completed.append((lease_id, choice))
            if len(self.completed) == 1:
                self.first_terminal_attempt.set()
                raise FatalWorkerExit()
            self.terminal_completed.set()
            return "completed"

        @staticmethod
        def fail(_lease_id: str, _failure_code: str) -> str:
            raise AssertionError("valid choice must not use failure terminal")

    class Llm:
        def __init__(self) -> None:
            self.calls = 0

        def complete_structured(self, **_kwargs: object) -> object:
            self.calls += 1
            return SimpleNamespace(
                content_type="json",
                parsed={"name": "Mira", "reason": "chosen"},
            )

    port = Port()
    llm = Llm()
    worker = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: llm,
        poll_interval_seconds=0.01,
    )
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap._adopt_identity_worker(worker)  # type: ignore[attr-defined]
    monkeypatch.setattr(threading, "excepthook", lambda _args: None)

    worker.start()
    assert port.first_terminal_attempt.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while worker.worker_started and time.monotonic() < deadline:
        time.sleep(0.001)
    assert worker.worker_started is False

    assert registration._ensure_identity_worker_running(bootstrap) is True  # type: ignore[attr-defined]
    assert port.terminal_completed.wait(timeout=1.0)
    bootstrap._stop_identity_worker()  # type: ignore[attr-defined]

    assert port.claims_before_terminal_ack == 1
    assert llm.calls == 1
    assert len(port.completed) == 2
    assert port.completed[0] == port.completed[1]
    assert bootstrap.identity_worker_for_test is None


def test_configure_retains_quick_fatal_actual_worker_terminal_intent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import identity as identity_module
    from alice_brain_hermes.hermes import identity_client as identity_client_module
    from alice_brain_hermes.hermes import registration

    class FatalWorkerExit(BaseException):
        pass

    active = IdentityNamingLeaseV1(
        lease_id=new_id(),
        brain_id=new_id(),
        state_sequence=1,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )

    class Port:
        def __init__(self) -> None:
            self.claim_count = 0
            self.claims_before_terminal_ack = 0
            self.completed: list[tuple[str, IdentityChoiceV1]] = []
            self.first_terminal_attempt = threading.Event()
            self.terminal_completed = threading.Event()

        def claim(self) -> IdentityNamingLeaseV1 | None:
            self.claim_count += 1
            if not self.terminal_completed.is_set():
                self.claims_before_terminal_ack += 1
            return active if self.claim_count == 1 else None

        def complete(self, lease_id: str, choice: IdentityChoiceV1) -> str:
            self.completed.append((lease_id, choice))
            if len(self.completed) == 1:
                self.first_terminal_attempt.set()
                raise FatalWorkerExit()
            self.terminal_completed.set()
            return "completed"

        @staticmethod
        def fail(_lease_id: str, _failure_code: str) -> str:
            raise AssertionError("valid choice must not use failure terminal")

    class Llm:
        def __init__(self) -> None:
            self.calls = 0

        def complete_structured(self, **_kwargs: object) -> object:
            self.calls += 1
            return SimpleNamespace(
                content_type="json",
                parsed={"name": "Mira", "reason": "chosen"},
            )

    port = Port()

    class WaitForFirstFatalWorker(IdentityNamingWorker):
        @property
        def worker_started(self) -> bool:
            assert port.first_terminal_attempt.wait(timeout=1.0)
            deadline = time.monotonic() + 1.0
            while super().worker_started and time.monotonic() < deadline:
                time.sleep(0.001)
            return super().worker_started

    llm = Llm()
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    monkeypatch.setenv(
        identity_module.IDENTITY_LLM_MODE_ENV,
        identity_module.IdentityLlmMode.NAME_WHEN_UNNAMED.value,
    )
    monkeypatch.setattr(
        identity_module,
        "IdentityNamingWorker",
        WaitForFirstFatalWorker,
    )
    monkeypatch.setattr(threading, "excepthook", lambda _args: None)
    monkeypatch.setattr(
        identity_client_module,
        "DaemonIdentityNamingLeasePort",
        lambda *_args, **_kwargs: port,
    )

    registration._configure_identity_worker(  # type: ignore[attr-defined]
        bootstrap,
        runtime_home=tmp_path,
        profile_factory=lambda: hermes_brain_profile("default"),
        llm_factory=lambda: llm,
    )
    retained = bootstrap.identity_worker_for_test

    assert retained is not None
    assert retained.worker_started is False
    assert registration._ensure_identity_worker_running(bootstrap) is True  # type: ignore[attr-defined]
    assert port.terminal_completed.wait(timeout=1.0)
    bootstrap._stop_identity_worker()  # type: ignore[attr-defined]

    assert port.claims_before_terminal_ack == 1
    assert llm.calls == 1
    assert len(port.completed) == 2
    assert port.completed[0] == port.completed[1]
    assert bootstrap.identity_worker_for_test is None


def test_missing_daemon_does_not_create_runtime_home(tmp_path: Path) -> None:
    runtime_home = tmp_path / "must-remain-absent"
    profile = hermes_brain_profile("default")
    bridge = HookBridge(runtime_home, profile_factory=lambda: profile)
    port = DaemonIdentityNamingLeasePort(
        runtime_home,
        profile_factory=lambda: profile,
    )

    assert bridge._connect() is False  # type: ignore[attr-defined]
    with pytest.raises(DaemonClientError):
        port.claim()
    assert not runtime_home.exists()


def test_real_hermes_0182_context_exposes_expected_lazy_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host = pytest.importorskip(
        "hermes_cli.plugins",
        reason="real Hermes contract requires the local 0.18.2 checkout",
    )
    hermes = pytest.importorskip("hermes_cli")
    if getattr(hermes, "__version__", None) != "0.18.2":
        pytest.skip("real contract is pinned to Hermes Agent 0.18.2")

    from alice_brain_hermes.hermes import registration

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "Hermes Home"))
    manifest = host.PluginManifest(
        name="alice-brain",
        key="alice-brain",
        source="entrypoint",
    )
    context = host.PluginContext(manifest, SimpleNamespace())
    access = registration._HermesHostAccess(context)  # type: ignore[attr-defined]

    assert context._llm is None
    assert isinstance(access.brain_profile(), BrainProfileV1)
    assert context._llm is None
    llm = access.llm()
    assert llm is context._llm
    assert callable(llm.complete_structured)
