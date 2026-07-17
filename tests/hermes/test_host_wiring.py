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
            self.skills: list[tuple[str, Path, str]] = []

        def register_hook(self, hook: str, callback: object) -> None:
            self.hooks.append((hook, callback))

        def register_cli_command(self, **kwargs: object) -> None:
            self.cli.append(kwargs)

        def register_skill(
            self,
            name: str,
            path: Path,
            description: str = "",
        ) -> None:
            self.skills.append((name, path, description))

    context = Context()
    registration.register(context)

    assert bootstrap.host_context_for_worker() is context
    assert context.profile_reads == 0
    assert context.llm_reads == 0
    assert len(context.skills) == 1
    assert context.skills[0][1].is_file()


def test_host_factory_cache_changes_only_with_context_identity() -> None:
    from alice_brain_hermes.hermes import registration

    first_context = _HostContext(profile_name="research")
    second_context = _HostContext(profile_name="default")
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )

    bootstrap.bind_host_context(first_context)
    first = bootstrap.host_factories_for_worker()
    bootstrap.bind_host_context(first_context)
    rebound = bootstrap.host_factories_for_worker()
    bootstrap.bind_host_context(second_context)
    replaced = bootstrap.host_factories_for_worker()

    assert first is not None
    assert rebound is not None
    assert replaced is not None
    assert all(left is right for left, right in zip(first, rebound, strict=True))
    assert all(left is not right for left, right in zip(first, replaced, strict=True))
    assert first_context.profile_reads == 0
    assert first_context.llm_reads == 0
    assert second_context.profile_reads == 0
    assert second_context.llm_reads == 0


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


class _FakeProjection:
    @staticmethod
    def read_context() -> None:
        return None


def _stop_bootstrap_on_next_record(bootstrap: Any) -> Any:
    def stop_on_next_record() -> None:
        bootstrap._stop_requested = True
        return None

    return stop_on_next_record


def _run_bootstrap_entry_in_current_thread(
    registration: Any,
    bootstrap: Any,
) -> None:
    bootstrap._worker = threading.current_thread()
    registration._bootstrap_worker_entry(bootstrap)
    assert bootstrap._worker is None


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

    def observe_wait(_timeout: float) -> None:
        nonlocal wait_calls
        wait_calls += 1

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "wait", observe_wait)
    monkeypatch.setattr(
        bootstrap,
        "next_for_worker",
        _stop_bootstrap_on_next_record(bootstrap),
    )

    registration._bootstrap_worker_main(bootstrap)  # type: ignore[attr-defined]

    # The stop latch is observed after one normal idle wait.
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
            self.close_sealed = False

        def start_worker(self) -> None:
            self.worker_started = True

        def stop_worker_for_test(self) -> None:
            self.worker_started = False

        def _worker_alive_strict(self) -> bool:
            return self.worker_started

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
        _stop_bootstrap_on_next_record(bootstrap),
    )

    _run_bootstrap_entry_in_current_thread(registration, bootstrap)

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
            self.close_sealed = False

        def start_worker(self) -> None:
            self.worker_started = True

        def stop_worker_for_test(self) -> None:
            self.worker_started = False

        def _worker_alive_strict(self) -> bool:
            return self.worker_started

    class Port:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            port_arguments.update(kwargs)

    class Worker:
        def __init__(self, **kwargs: object) -> None:
            worker_arguments.update(kwargs)
            self.started = False
            self.stopped = False
            self.start_calls = 0
            workers.append(self)

        def start(self) -> None:
            self.start_calls += 1
            self.started = True

        @property
        def worker_started(self) -> bool:
            return self.started

        def stop_for_test(self) -> None:
            self.stopped = True
            self.started = False

        def _worker_alive_strict(self) -> bool:
            return self.started

        @property
        def terminal_intent_pending(self) -> bool:
            return False

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(identity_client_module, "DaemonIdentityNamingLeasePort", Port)
    monkeypatch.setattr(identity_module, "IdentityNamingWorker", Worker)
    monkeypatch.setattr(
        bootstrap,
        "next_for_worker",
        _stop_bootstrap_on_next_record(bootstrap),
    )

    _run_bootstrap_entry_in_current_thread(registration, bootstrap)

    assert len(workers) == 1
    assert workers[0].start_calls == 1
    assert workers[0].stopped is True
    assert bootstrap.identity_worker_for_test is workers[0]
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
    bootstrap.stop_worker_for_test()
    assert bootstrap.identity_worker_for_test is None


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
            self.close_sealed = False

        def start_worker(self) -> None:
            self.worker_started = True

        def stop_worker_for_test(self) -> None:
            self.worker_started = False

        def _worker_alive_strict(self) -> bool:
            return self.worker_started

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

        def _worker_alive_strict(self) -> bool:
            return self.alive

        @property
        def terminal_intent_pending(self) -> bool:
            return False

    def next_record() -> None:
        nonlocal next_calls
        next_calls += 1
        if next_calls == 1:
            workers[0].alive = False
            return None
        bootstrap._stop_requested = True
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
    monkeypatch.setattr(bootstrap, "wait", lambda _timeout: None)

    _run_bootstrap_entry_in_current_thread(registration, bootstrap)

    assert len(workers) == 1
    assert workers[0].start_calls == 2
    assert workers[0].stop_calls == 1
    assert bootstrap.identity_worker_for_test is workers[0]
    bootstrap.stop_worker_for_test()
    assert workers[0].stop_calls == 2
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
            self.close_sealed = False

        def start_worker(self) -> None:
            self.worker_started = True

        def stop_worker_for_test(self) -> None:
            self.worker_started = False

        def _worker_alive_strict(self) -> bool:
            return self.worker_started

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

        def _worker_alive_strict(self) -> bool:
            return self.alive

        @property
        def terminal_intent_pending(self) -> bool:
            return False

    def next_record() -> None:
        nonlocal next_calls
        next_calls += 1
        if workers and workers[0].start_calls >= 2:
            bootstrap._stop_requested = True
            return None
        if next_calls >= 10:
            bootstrap._stop_requested = True
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

    _run_bootstrap_entry_in_current_thread(registration, bootstrap)

    assert len(workers) == 1
    assert workers[0].start_calls == 2
    assert workers[0].stop_calls == 1
    assert context.llm_reads == 0
    assert bootstrap.identity_worker_for_test is workers[0]
    bootstrap.stop_worker_for_test()
    assert workers[0].stop_calls == 2
    assert bootstrap.identity_worker_for_test is None


def test_bootstrap_reuses_process_host_factories_across_outer_generations(
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
    bridge_profile_factories: list[Any] = []
    port_profile_factories: list[Any] = []
    worker_llm_factories: list[Any] = []
    resolved_profiles: list[object] = []
    resolved_llms: list[object] = []
    workers: list[Any] = []

    class Bridge:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            profile_factory = kwargs["profile_factory"]
            bridge_profile_factories.append(profile_factory)
            resolved_profiles.append(profile_factory())
            self.projections = _FakeProjection()
            self.worker_started = False
            self.close_sealed = False

        def start_worker(self) -> None:
            self.worker_started = True

        def stop_worker_for_test(self) -> None:
            self.worker_started = False

        def _worker_alive_strict(self) -> bool:
            return self.worker_started

    class Port:
        def __init__(self, _runtime_home: object, **kwargs: object) -> None:
            profile_factory = kwargs["profile_factory"]
            port_profile_factories.append(profile_factory)
            resolved_profiles.append(profile_factory())

    class Worker:
        def __init__(self, **kwargs: object) -> None:
            llm_factory = kwargs["llm_factory"]
            worker_llm_factories.append(llm_factory)
            resolved_llms.append(llm_factory())
            self.alive = False
            self.start_calls = 0
            workers.append(self)

        @property
        def worker_started(self) -> bool:
            return self.alive

        def start(self) -> None:
            self.start_calls += 1
            self.alive = True

        def stop_for_test(self) -> None:
            self.alive = False

        def _worker_alive_strict(self) -> bool:
            return self.alive

        @property
        def terminal_intent_pending(self) -> bool:
            return False

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(identity_client_module, "DaemonIdentityNamingLeasePort", Port)
    monkeypatch.setattr(identity_module, "IdentityNamingWorker", Worker)
    monkeypatch.setattr(bootstrap, "wait", lambda _timeout: None)
    monkeypatch.setattr(
        bootstrap,
        "next_for_worker",
        _stop_bootstrap_on_next_record(bootstrap),
    )

    for _generation in range(2):
        bootstrap._stop_requested = False
        bootstrap._stop_event.clear()
        _run_bootstrap_entry_in_current_thread(registration, bootstrap)

    assert len(workers) == 1
    assert workers[0].start_calls == 2
    assert workers[0].alive is False
    assert bootstrap.identity_worker_for_test is workers[0]
    assert bridge_profile_factories[0] is bridge_profile_factories[1]
    assert bridge_profile_factories[0] is port_profile_factories[0]
    assert len(port_profile_factories) == 1
    assert len(worker_llm_factories) == 1
    assert all(profile is resolved_profiles[0] for profile in resolved_profiles)
    assert len(resolved_profiles) == 3
    assert resolved_llms == [context.host_llm]
    assert context.profile_reads == 1
    assert context.llm_reads == 1
    bootstrap.stop_worker_for_test()
    assert bootstrap.identity_worker_for_test is None


def test_bootstrap_existing_transport_defines_identity_runtime_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import bridge as bridge_module
    from alice_brain_hermes.hermes import identity as identity_module
    from alice_brain_hermes.hermes import identity_client as identity_client_module
    from alice_brain_hermes.hermes import registration

    monkeypatch.setenv(
        identity_module.IDENTITY_LLM_MODE_ENV,
        identity_module.IdentityLlmMode.NAME_WHEN_UNNAMED.value,
    )
    runtime_home = object()
    port_homes: list[object] = []
    workers: list[Any] = []
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap.bind_host_context(_HostContext())

    class ExistingBridge:
        def __init__(self) -> None:
            self.runtime_home = runtime_home
            self.projections = _FakeProjection()
            self.worker_started = True
            self.close_sealed = False

        def start_worker(self) -> None:
            self.worker_started = True

        def stop_worker_for_test(self) -> None:
            self.worker_started = False

        def _worker_alive_strict(self) -> bool:
            return self.worker_started

    class Port:
        def __init__(self, candidate_home: object, **_kwargs: object) -> None:
            port_homes.append(candidate_home)

    class Worker:
        def __init__(self, **_kwargs: object) -> None:
            self.alive = False
            workers.append(self)

        @property
        def worker_started(self) -> bool:
            return self.alive

        def start(self) -> None:
            self.alive = True

        def stop_for_test(self) -> None:
            self.alive = False

        def _worker_alive_strict(self) -> bool:
            return self.alive

        @property
        def terminal_intent_pending(self) -> bool:
            return False

    existing = ExistingBridge()
    bootstrap._adopt_transport_bridge(existing)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        bridge_module,
        "default_runtime_home",
        lambda: (_ for _ in ()).throw(
            AssertionError("existing transport home was ignored")
        ),
    )
    monkeypatch.setattr(identity_client_module, "DaemonIdentityNamingLeasePort", Port)
    monkeypatch.setattr(identity_module, "IdentityNamingWorker", Worker)
    monkeypatch.setattr(
        bootstrap,
        "next_for_worker",
        _stop_bootstrap_on_next_record(bootstrap),
    )

    _run_bootstrap_entry_in_current_thread(registration, bootstrap)

    assert port_homes == [runtime_home]
    assert bootstrap.identity_worker_for_test is workers[0]
    bootstrap.stop_worker_for_test()
    assert bootstrap.identity_worker_for_test is None


def test_external_stop_retains_unknown_identity_but_clears_transport() -> None:
    from alice_brain_hermes.hermes import registration

    identity_stop_calls = 0
    transport_stop_calls = 0

    class UnknownIdentity:
        def stop_for_test(self) -> None:
            nonlocal identity_stop_calls
            identity_stop_calls += 1

        @staticmethod
        def _worker_alive_strict() -> bool:
            raise MemoryError("identity liveness unavailable")

        @property
        def terminal_intent_pending(self) -> bool:
            raise AssertionError("pending state is unsafe to inspect")

    class StoppedTransport:
        alive = True

        def stop_worker_for_test(self) -> None:
            nonlocal transport_stop_calls
            transport_stop_calls += 1
            self.alive = False

        def _worker_alive_strict(self) -> bool:
            return self.alive

    identity = UnknownIdentity()
    transport = StoppedTransport()
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap._adopt_identity_worker(identity)  # type: ignore[attr-defined]
    bootstrap._adopt_transport_bridge(transport)  # type: ignore[attr-defined]

    with pytest.raises(
        RuntimeError,
        match="identity worker liveness is unknown",
    ) as raised:
        bootstrap.stop_worker_for_test()

    assert identity_stop_calls == 1
    assert transport_stop_calls == 1
    assert isinstance(raised.value.__cause__, MemoryError)
    assert bootstrap.identity_worker_for_test is identity
    assert bootstrap._transport_bridge is None  # type: ignore[attr-defined]
    assert bootstrap.health.degraded is True
    assert bootstrap.health.worker_error == "MemoryError"


def test_bootstrap_external_stop_aggregates_double_unknown_children() -> None:
    from alice_brain_hermes.hermes import registration

    stop_calls: list[str] = []

    class UnknownIdentity:
        def stop_for_test(self) -> None:
            stop_calls.append("identity")

        @staticmethod
        def _worker_alive_strict() -> bool:
            raise MemoryError("identity liveness unavailable")

        @property
        def terminal_intent_pending(self) -> bool:
            raise AssertionError("pending state is unsafe to inspect")

    class UnknownTransport:
        def stop_worker_for_test(self) -> None:
            stop_calls.append("transport")

        @staticmethod
        def _worker_alive_strict() -> bool:
            raise KeyboardInterrupt("transport liveness unavailable")

    identity = UnknownIdentity()
    transport = UnknownTransport()
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap._adopt_identity_worker(identity)  # type: ignore[attr-defined]
    bootstrap._adopt_transport_bridge(transport)  # type: ignore[attr-defined]

    with pytest.raises(BaseExceptionGroup) as raised:
        bootstrap.stop_worker_for_test()

    assert stop_calls == ["identity", "transport"]
    assert len(raised.value.exceptions) == 2
    assert {str(error) for error in raised.value.exceptions} == {
        "identity worker liveness is unknown",
        "bootstrap transport bridge liveness is unknown",
    }
    assert {type(error.__cause__) for error in raised.value.exceptions} == {
        MemoryError,
        KeyboardInterrupt,
    }
    assert bootstrap.identity_worker_for_test is identity
    assert bootstrap._transport_bridge is transport  # type: ignore[attr-defined]
    assert bootstrap.health.degraded is True
    assert bootstrap.health.worker_error == "KeyboardInterrupt"


def test_outer_unknown_liveness_does_not_touch_either_child() -> None:
    from alice_brain_hermes.hermes import registration

    join_timeouts: list[float] = []
    child_stop_calls: list[str] = []

    class UnknownOuter:
        @staticmethod
        def is_alive() -> bool:
            raise MemoryError("outer liveness unavailable")

        @staticmethod
        def join(timeout: float) -> None:
            join_timeouts.append(timeout)

    class Identity:
        def stop_for_test(self) -> None:
            child_stop_calls.append("identity")

        @staticmethod
        def _worker_alive_strict() -> bool:
            return False

        @property
        def terminal_intent_pending(self) -> bool:
            return False

    class Transport:
        def stop_worker_for_test(self) -> None:
            child_stop_calls.append("transport")

        @staticmethod
        def _worker_alive_strict() -> bool:
            return False

    outer = UnknownOuter()
    identity = Identity()
    transport = Transport()
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap._worker = outer  # type: ignore[attr-defined,assignment]
    bootstrap._adopt_identity_worker(identity)  # type: ignore[attr-defined]
    bootstrap._adopt_transport_bridge(transport)  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="liveness is unknown") as raised:
        bootstrap.stop_worker_for_test()

    assert isinstance(raised.value.__cause__, MemoryError)
    assert join_timeouts == [4.0]
    assert child_stop_calls == []
    assert bootstrap._worker is outer  # type: ignore[attr-defined]
    assert bootstrap.identity_worker_for_test is identity
    assert bootstrap._transport_bridge is transport  # type: ignore[attr-defined]
    assert bootstrap.health.degraded is True
    assert bootstrap.health.worker_error == "MemoryError"


def test_entry_identity_unknown_still_stops_transport_and_publishes_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    cleanup_calls: list[str] = []

    class FatalEntryExit(BaseException):
        pass

    class UnknownIdentity:
        def stop_for_test(self) -> None:
            cleanup_calls.append("identity")

        @staticmethod
        def _worker_alive_strict() -> bool:
            raise MemoryError("identity liveness unavailable")

    class Transport:
        alive = True

        def stop_worker_for_test(self) -> None:
            cleanup_calls.append("transport")
            self.alive = False

        def _worker_alive_strict(self) -> bool:
            return self.alive

    identity = UnknownIdentity()
    transport = Transport()
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap._worker = threading.current_thread()  # type: ignore[attr-defined]
    bootstrap._adopt_identity_worker(identity)  # type: ignore[attr-defined]
    bootstrap._adopt_transport_bridge(transport)  # type: ignore[attr-defined]

    def fatal_main(_buffer: object) -> None:
        raise FatalEntryExit()

    monkeypatch.setattr(registration, "_bootstrap_worker_main", fatal_main)

    with pytest.raises(FatalEntryExit):
        registration._bootstrap_worker_entry(bootstrap)  # type: ignore[attr-defined]

    assert cleanup_calls == ["identity", "transport"]
    assert bootstrap._worker is None  # type: ignore[attr-defined]
    assert bootstrap.identity_worker_for_test is identity
    assert bootstrap._transport_bridge is None  # type: ignore[attr-defined]
    assert bootstrap.health.worker_started is False
    assert bootstrap.health.degraded is True
    assert bootstrap.health.worker_error == "MemoryError"


def test_pending_identity_terminal_intent_survives_disposal_and_replays_next_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import bridge as bridge_module
    from alice_brain_hermes.hermes import identity as identity_module
    from alice_brain_hermes.hermes import registration

    active = IdentityNamingLeaseV1(
        lease_id=new_id(),
        brain_id=new_id(),
        state_sequence=1,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )

    class Port:
        def __init__(self) -> None:
            self.claim_count = 0
            self.completed: list[tuple[str, IdentityChoiceV1]] = []
            self.terminal_completed = threading.Event()

        def claim(self) -> IdentityNamingLeaseV1 | None:
            self.claim_count += 1
            return active if self.claim_count == 1 else None

        def complete(self, lease_id: str, choice: IdentityChoiceV1) -> str:
            self.completed.append((lease_id, choice))
            if len(self.completed) == 1:
                raise DaemonClientError("terminal ACK unavailable")
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

    class StoppedTransport:
        alive = False

        def stop_worker_for_test(self) -> None:
            self.alive = False

        def _worker_alive_strict(self) -> bool:
            return self.alive

    class Bridge(StoppedTransport):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.alive = False
            self.close_sealed = False
            self.projections = _FakeProjection()

        @property
        def worker_started(self) -> bool:
            return self.alive

        def start_worker(self) -> None:
            self.alive = True

    port = Port()
    llm = Llm()
    identity = IdentityNamingWorker(
        mode=IdentityLlmMode.NAME_WHEN_UNNAMED,
        lease_port=port,
        llm_factory=lambda: llm,
        poll_interval_seconds=0.01,
    )
    with pytest.raises(DaemonClientError, match="terminal ACK unavailable"):
        identity.run_once()

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap._adopt_identity_worker(identity)  # type: ignore[attr-defined]
    bootstrap._adopt_transport_bridge(StoppedTransport())  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="terminal intent remains pending"):
        bootstrap.stop_worker_for_test()

    assert bootstrap.identity_worker_for_test is identity
    assert bootstrap._transport_bridge is None  # type: ignore[attr-defined]

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(
        identity_module,
        "IdentityNamingWorker",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("pending identity owner was replaced")
        ),
    )

    terminal_replay_observed: list[bool] = []

    def stop_after_terminal_replay() -> None:
        terminal_replay_observed.append(port.terminal_completed.wait(timeout=1.0))
        bootstrap._stop_requested = True
        return None

    bootstrap._stop_requested = False
    bootstrap._stop_event.clear()
    monkeypatch.setattr(bootstrap, "next_for_worker", stop_after_terminal_replay)
    _run_bootstrap_entry_in_current_thread(registration, bootstrap)

    assert port.claim_count == 1
    assert llm.calls == 1
    assert terminal_replay_observed == [True]
    assert len(port.completed) == 2
    assert port.completed[0] == port.completed[1]
    assert bootstrap.identity_worker_for_test is identity
    assert identity.worker_started is False
    bootstrap.stop_worker_for_test()
    assert bootstrap.identity_worker_for_test is None


def test_bootstrap_contains_default_home_prelude_fault_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import bridge as bridge_module
    from alice_brain_hermes.hermes import registration

    home_calls = 0
    bridge_instances: list[Any] = []
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )

    def default_home() -> Path:
        nonlocal home_calls
        home_calls += 1
        if home_calls == 1:
            raise MemoryError("default home prelude failed")
        return tmp_path

    class Bridge:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.projections = _FakeProjection()
            self.worker_started = False
            self.close_sealed = False
            bridge_instances.append(self)

        def start_worker(self) -> None:
            self.worker_started = True

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", default_home)
    monkeypatch.setattr(bootstrap, "wait", lambda _timeout: None)
    monkeypatch.setattr(
        bootstrap,
        "next_for_worker",
        _stop_bootstrap_on_next_record(bootstrap),
    )

    registration._bootstrap_worker_main(bootstrap)  # type: ignore[attr-defined]

    assert home_calls == 2
    assert len(bridge_instances) == 1
    assert bootstrap.health.degraded is True
    assert bootstrap.health.worker_error == "MemoryError"


def test_bootstrap_contains_host_binding_prelude_fault_without_rebuilding_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import bridge as bridge_module
    from alice_brain_hermes.hermes import registration

    context = _HostContext(profile_name="research")
    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap.bind_host_context(context)
    original_factories = bootstrap.host_factories_for_worker
    observed_factories: list[tuple[object, object, object] | None] = []

    def fail_after_first_binding() -> tuple[object, object, object] | None:
        factories = original_factories()
        observed_factories.append(factories)
        if len(observed_factories) == 1:
            raise MemoryError("host binding prelude failed")
        return factories

    class Bridge:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.projections = _FakeProjection()
            self.worker_started = False
            self.close_sealed = False

        def start_worker(self) -> None:
            self.worker_started = True

    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(
        bootstrap,
        "host_factories_for_worker",
        fail_after_first_binding,
    )
    monkeypatch.setattr(bootstrap, "wait", lambda _timeout: None)
    monkeypatch.setattr(
        bootstrap,
        "next_for_worker",
        _stop_bootstrap_on_next_record(bootstrap),
    )

    registration._bootstrap_worker_main(bootstrap)  # type: ignore[attr-defined]

    assert len(observed_factories) == 2
    first = observed_factories[0]
    second = observed_factories[1]
    assert first is not None
    assert second is not None
    assert all(left is right for left, right in zip(first, second, strict=True))
    assert context.profile_reads == 0
    assert context.llm_reads == 0
    assert bootstrap.health.degraded is True
    assert bootstrap.health.worker_error == "MemoryError"


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

        def _worker_alive_strict(self) -> bool:
            return not self.stopped

        @property
        def terminal_intent_pending(self) -> bool:
            return False

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
    import hermes_cli as hermes
    from hermes_cli import plugins as host

    assert hermes.__version__ == "0.18.2"

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
