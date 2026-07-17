from __future__ import annotations

import threading
from pathlib import Path

import pytest


class Worker:
    def __init__(self, **kwargs: object) -> None:
        self.arguments = kwargs
        self.alive = False
        self.start_calls = 0
        self.stop_calls = 0

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

    @property
    def last_internal_error_type(self) -> None:
        return None


def test_bootstrap_background_wires_energy_when_identity_is_opted_out(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import bridge as bridge_module
    from alice_brain_hermes.hermes import energy as energy_module
    from alice_brain_hermes.hermes import energy_client as energy_client_module
    from alice_brain_hermes.hermes import identity as identity_module
    from alice_brain_hermes.hermes import identity_client as identity_client_module
    from alice_brain_hermes.hermes import registration

    class Context:
        profile_name = "default"
        llm = object()

    bridge_arguments: dict[str, object] = {}
    port_arguments: dict[str, object] = {}
    health_reports: list[dict[str, object]] = []
    workers: list[Worker] = []

    class Projection:
        @staticmethod
        def read_context() -> None:
            return None

    class Bridge:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            bridge_arguments.update(kwargs)
            self.projections = Projection()
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
            port_arguments.update(kwargs)

    class EnergyWorker(Worker):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            workers.append(self)

    class HealthPort:
        def __init__(self, runtime_home: object) -> None:
            assert runtime_home == tmp_path

        def report(self, **kwargs: object) -> bool:
            health_reports.append(kwargs)
            if len(health_reports) == 1:
                raise ConnectionError("first heartbeat transport failed")
            return True

    monkeypatch.setenv(
        identity_module.IDENTITY_LLM_MODE_ENV,
        identity_module.IdentityLlmMode.OFF.value,
    )
    monkeypatch.setattr(bridge_module, "HookBridge", Bridge)
    monkeypatch.setattr(bridge_module, "default_runtime_home", lambda: tmp_path)
    monkeypatch.setattr(
        identity_client_module,
        "DaemonIdentityNamingLeasePort",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity opt-out constructed a lease port")
        ),
    )
    monkeypatch.setattr(
        identity_module,
        "IdentityNamingWorker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity opt-out constructed a worker")
        ),
    )
    monkeypatch.setattr(energy_client_module, "DaemonEnergyAssessmentLeasePort", Port)
    monkeypatch.setattr(
        energy_client_module,
        "DaemonEnergyWorkerHealthPort",
        HealthPort,
    )
    monkeypatch.setattr(energy_module, "EnergyAssessmentWorker", EnergyWorker)

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap.bind_host_context(Context())

    def stop_on_next_record() -> None:
        if len(health_reports) >= 2:
            bootstrap._stop_requested = True
        return None

    monkeypatch.setattr(bootstrap, "next_for_worker", stop_on_next_record)
    bootstrap._worker = threading.current_thread()  # type: ignore[attr-defined]
    registration._bootstrap_worker_entry(bootstrap)  # type: ignore[attr-defined]

    assert len(workers) == 1
    worker = workers[0]
    assert worker.start_calls == 1
    assert worker.stop_calls == 1
    assert bootstrap.identity_worker_for_test is None
    assert bootstrap.energy_worker_for_test is worker
    assert bridge_arguments["profile_factory"] is port_arguments["profile_factory"]
    assert worker.arguments["llm_factory"]() is Context.llm
    assert set(worker.arguments) == {"lease_port", "llm_factory"}
    assert health_reports == [
        {
            "worker_started": True,
            "terminal_intent_pending": False,
            "error_type": None,
        },
        {
            "worker_started": True,
            "terminal_intent_pending": False,
            "error_type": None,
        },
    ]
    assert not {
        "agent_id",
        "max_tokens",
        "model",
        "profile",
        "provider",
        "temperature",
        "timeout",
    }.intersection(worker.arguments)

    bootstrap.stop_worker_for_test()
    assert worker.stop_calls == 2
    assert bootstrap.energy_worker_for_test is None


def test_energy_and_identity_workers_share_exact_host_factories_but_not_owners(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from alice_brain_hermes.hermes import energy as energy_module
    from alice_brain_hermes.hermes import energy_client as energy_client_module
    from alice_brain_hermes.hermes import identity as identity_module
    from alice_brain_hermes.hermes import identity_client as identity_client_module
    from alice_brain_hermes.hermes import registration

    class Context:
        profile_name = "default"
        llm = object()

    ports: dict[str, dict[str, object]] = {}
    workers: dict[str, Worker] = {}

    class Port:
        def __init__(self, _runtime_home: object, **kwargs: object) -> None:
            ports[type(self).__name__] = kwargs

    class IdentityPort(Port):
        pass

    class EnergyPort(Port):
        pass

    class IdentityWorker(Worker):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            workers["identity"] = self

    class EnergyWorker(Worker):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            workers["energy"] = self

    monkeypatch.setenv(
        identity_module.IDENTITY_LLM_MODE_ENV,
        identity_module.IdentityLlmMode.NAME_WHEN_UNNAMED.value,
    )
    monkeypatch.setattr(
        identity_client_module,
        "DaemonIdentityNamingLeasePort",
        IdentityPort,
    )
    monkeypatch.setattr(
        energy_client_module,
        "DaemonEnergyAssessmentLeasePort",
        EnergyPort,
    )
    monkeypatch.setattr(identity_module, "IdentityNamingWorker", IdentityWorker)
    monkeypatch.setattr(energy_module, "EnergyAssessmentWorker", EnergyWorker)

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap.bind_host_context(Context())
    factories = bootstrap.host_factories_for_worker()
    assert factories is not None
    _access, profile_factory, llm_factory = factories

    registration._configure_identity_worker(  # type: ignore[attr-defined]
        bootstrap,
        runtime_home=tmp_path,
        profile_factory=profile_factory,
        llm_factory=llm_factory,
    )
    registration._configure_energy_worker(  # type: ignore[attr-defined]
        bootstrap,
        runtime_home=tmp_path,
        profile_factory=profile_factory,
        llm_factory=llm_factory,
    )

    identity = workers["identity"]
    energy = workers["energy"]
    assert identity is not energy
    assert bootstrap.identity_worker_for_test is identity
    assert bootstrap.energy_worker_for_test is energy
    assert identity.arguments["llm_factory"] is energy.arguments["llm_factory"]
    assert identity.arguments["llm_factory"] is llm_factory
    assert ports["IdentityPort"]["profile_factory"] is profile_factory
    assert ports["EnergyPort"]["profile_factory"] is profile_factory
    assert identity.start_calls == 1
    assert energy.start_calls == 1
    assert set(energy.arguments) == {"lease_port", "llm_factory"}

    bootstrap._stop_owned_children()  # type: ignore[attr-defined]
    assert identity.stop_calls == 1
    assert energy.stop_calls == 1
    assert bootstrap.identity_worker_for_test is None
    assert bootstrap.energy_worker_for_test is None


def test_energy_worker_monitor_restarts_same_owned_worker() -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    worker = Worker()
    bootstrap._adopt_energy_worker(worker)  # type: ignore[attr-defined]

    assert registration._ensure_energy_worker_running(bootstrap) is True  # type: ignore[attr-defined]
    assert worker.start_calls == 1
    assert bootstrap.energy_worker_for_test is worker
    bootstrap._stop_energy_worker()  # type: ignore[attr-defined]
    assert bootstrap.energy_worker_for_test is None


def test_energy_cleanup_retains_ambiguous_owner_and_publishes_diagnostic() -> None:
    from alice_brain_hermes.hermes import registration

    class AmbiguousWorker(Worker):
        def _worker_alive_strict(self) -> bool:
            raise MemoryError("energy liveness unavailable")

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    worker = AmbiguousWorker()
    bootstrap._adopt_energy_worker(worker)  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="energy worker liveness is unknown"):
        bootstrap._stop_energy_worker()  # type: ignore[attr-defined]

    assert bootstrap.energy_worker_for_test is worker
    assert bootstrap.health.degraded is True
    assert bootstrap.health.worker_error == "MemoryError"


def test_energy_worker_diagnostics_surface_sanitized_internal_error() -> None:
    from alice_brain_hermes.hermes import registration

    class FailedWorker(Worker):
        @property
        def last_internal_error_type(self) -> str:
            return "DaemonClientError"

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    worker = FailedWorker()
    worker.alive = True
    bootstrap._adopt_energy_worker(worker)  # type: ignore[attr-defined]

    assert registration._ensure_energy_worker_running(bootstrap) is True  # type: ignore[attr-defined]
    assert bootstrap.health.degraded is True
    assert bootstrap.health.worker_error == "DaemonClientError"

    worker.alive = False
    bootstrap._stop_energy_worker()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("internal_error", "terminal_pending", "expected_error"),
    [
        ("DaemonClientError", False, "DaemonClientError"),
        (None, True, "energy_terminal_intent_pending"),
    ],
)
def test_energy_diagnostic_is_not_overwritten_by_successful_capture_handoff(
    internal_error: str | None,
    terminal_pending: bool,
    expected_error: str,
) -> None:
    from alice_brain_hermes.hermes import registration

    class DiagnosticWorker(Worker):
        @property
        def last_internal_error_type(self) -> str | None:
            return internal_error

        @property
        def terminal_intent_pending(self) -> bool:
            return terminal_pending

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    worker = DiagnosticWorker()
    worker.alive = True
    bootstrap._adopt_energy_worker(worker)  # type: ignore[attr-defined]
    bootstrap.capture(
        "on_session_start",
        {
            "telemetry_schema_version": "hermes.observer.v1",
            "session_id": "session",
        },
    )
    capture = bootstrap.next_for_worker()
    assert capture is not None

    assert registration._ensure_energy_worker_running(bootstrap) is True  # type: ignore[attr-defined]
    bootstrap.mark_handed_off(capture)

    assert bootstrap.health.degraded is True
    assert bootstrap.health.worker_error == expected_error
    assert bootstrap.health.energy_terminal_intent_pending is terminal_pending
    assert bootstrap.health.energy_worker_error == internal_error

    worker.alive = False
    if terminal_pending:
        with pytest.raises(RuntimeError, match="terminal intent remains pending"):
            bootstrap._stop_energy_worker()  # type: ignore[attr-defined]
    else:
        bootstrap._stop_energy_worker()  # type: ignore[attr-defined]


def test_owned_children_stop_workers_before_transport() -> None:
    from alice_brain_hermes.hermes import registration

    stopped: list[str] = []

    class OrderedWorker(Worker):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def stop_for_test(self) -> None:
            stopped.append(self.name)
            super().stop_for_test()

    class Transport:
        alive = True

        def stop_worker_for_test(self) -> None:
            stopped.append("transport")
            self.alive = False

        def _worker_alive_strict(self) -> bool:
            return self.alive

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    bootstrap._adopt_identity_worker(OrderedWorker("identity"))  # type: ignore[attr-defined]
    bootstrap._adopt_energy_worker(OrderedWorker("energy"))  # type: ignore[attr-defined]
    bootstrap._adopt_transport_bridge(Transport())  # type: ignore[attr-defined]

    bootstrap._stop_owned_children()  # type: ignore[attr-defined]

    assert stopped == ["identity", "energy", "transport"]


def test_only_same_energy_worker_resolution_clears_its_diagnostic() -> None:
    from alice_brain_hermes.hermes import registration

    class MutableWorker(Worker):
        error: str | None = "DaemonClientError"
        pending = True

        @property
        def last_internal_error_type(self) -> str | None:
            return self.error

        @property
        def terminal_intent_pending(self) -> bool:
            return self.pending

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        start_worker_on_capture=False
    )
    worker = MutableWorker()
    worker.alive = True
    bootstrap._adopt_energy_worker(worker)  # type: ignore[attr-defined]

    assert registration._ensure_energy_worker_running(bootstrap) is True  # type: ignore[attr-defined]
    assert bootstrap.health.degraded is True
    assert bootstrap.health.energy_worker_error == "DaemonClientError"
    assert bootstrap.health.energy_terminal_intent_pending is True

    worker.error = None
    worker.pending = False
    assert registration._ensure_energy_worker_running(bootstrap) is True  # type: ignore[attr-defined]
    assert bootstrap.health.degraded is False
    assert bootstrap.health.energy_worker_error is None
    assert bootstrap.health.energy_terminal_intent_pending is False

    worker.alive = False
    bootstrap._stop_energy_worker()  # type: ignore[attr-defined]
