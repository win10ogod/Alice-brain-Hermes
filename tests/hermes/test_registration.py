from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import compat32
from importlib import metadata
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any
from zipfile import ZipFile

import pytest
import yaml
from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_ROOT = PROJECT_ROOT / "integration" / "alice-brain"
ENTRY_POINT_GROUP = "hermes_agent.plugins"
PLUGIN_NAME = "alice-brain"


def _alice_entry_point() -> metadata.EntryPoint:
    matches = [
        entry_point
        for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP)
        if entry_point.name == PLUGIN_NAME
        and entry_point.dist is not None
        and entry_point.dist.name == "alice-brain-hermes"
    ]
    assert len(matches) == 1
    return matches[0]


def _load_directory_shim() -> ModuleType:
    init_file = INTEGRATION_ROOT / "__init__.py"
    module_name = "task5_test_directory_plugin"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[str(INTEGRATION_ROOT)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _manifest() -> dict[str, object]:
    with (INTEGRATION_ROOT / "plugin.yaml").open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    assert isinstance(loaded, dict)
    return loaded


def test_pip_entry_point_loads_module_with_sync_register() -> None:
    entry_point = _alice_entry_point()
    module = entry_point.load()

    assert isinstance(module, ModuleType)
    assert module.__name__ == "alice_brain_hermes.hermes_plugin"
    assert callable(module.register)
    assert not inspect.iscoroutinefunction(module.register)
    assert module.__all__ == ["register"]


def test_manifest_has_exact_schema_and_dual_hook_lists() -> None:
    from alice_brain_hermes.hermes.registration import APPROVED_HOOKS

    manifest = _manifest()

    assert manifest == {
        "manifest_version": 1,
        "name": "alice-brain",
        "version": "0.1.0",
        "description": "Independent Alice-brain-Hermes consciousness runtime plugin",
        "author": "Alice-brain-Hermes",
        "kind": "standalone",
        "hooks": list(APPROVED_HOOKS),
        "provides_hooks": list(APPROVED_HOOKS),
    }


def test_directory_shim_reexports_the_same_register() -> None:
    pip_module = _alice_entry_point().load()
    directory_module = _load_directory_shim()

    assert directory_module.register is pip_module.register
    assert directory_module.__all__ == ["register"]


def test_manifest_and_registered_hook_constant_cannot_drift() -> None:
    from alice_brain_hermes.hermes.registration import APPROVED_HOOKS

    manifest = _manifest()

    assert tuple(manifest["hooks"]) == APPROVED_HOOKS
    assert tuple(manifest["provides_hooks"]) == APPROVED_HOOKS


class RecordingContext:
    def __init__(self) -> None:
        self.hooks: list[tuple[str, object]] = []
        self.cli_calls: list[dict[str, object]] = []

    @property
    def llm(self) -> object:
        raise AssertionError("registration must not access ctx.llm")

    def register_hook(self, hook_name: str, callback: object) -> None:
        self.hooks.append((hook_name, callback))

    def register_cli_command(self, **kwargs: object) -> None:
        self.cli_calls.append(kwargs)


@pytest.mark.parametrize(
    ("version", "accepted"),
    [
        ("0.17.9", False),
        ("0.18.0rc1", False),
        ("0.18.2", True),
        ("0.19.0", False),
        (None, False),
        ("not-a-version", False),
    ],
)
def test_version_gate_accepts_only_018_release_line(
    version: str | None,
    accepted: bool,
) -> None:
    from alice_brain_hermes.hermes.registration import require_supported_hermes

    if accepted:
        assert require_supported_hermes(version) == version
    else:
        with pytest.raises(RuntimeError, match="Hermes"):
            require_supported_hermes(version)


def test_resolve_version_uses_module_fallback_when_metadata_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    def metadata_absent(distribution: str) -> str:
        assert distribution == "hermes-agent"
        raise metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(registration.metadata, "version", metadata_absent)
    monkeypatch.setattr(
        registration,
        "import_module",
        lambda name: (
            SimpleNamespace(__version__="0.18.2") if name == "hermes_cli" else None
        ),
    )

    assert registration.resolve_hermes_version() == "0.18.2"


def test_resolve_version_rejects_metadata_module_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    monkeypatch.setattr(registration.metadata, "version", lambda _name: "0.18.2")
    monkeypatch.setattr(
        registration,
        "import_module",
        lambda _name: SimpleNamespace(__version__="0.18.1"),
    )

    with pytest.raises(RuntimeError, match="mismatch"):
        registration.resolve_hermes_version()


def test_resolve_version_rejects_missing_or_invalid_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    def metadata_absent(distribution: str) -> str:
        raise metadata.PackageNotFoundError(distribution)

    def module_absent(name: str) -> object:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(registration.metadata, "version", metadata_absent)
    monkeypatch.setattr(registration, "import_module", module_absent)
    with pytest.raises(RuntimeError, match="not installed"):
        registration.resolve_hermes_version()

    monkeypatch.setattr(
        registration,
        "import_module",
        lambda _name: SimpleNamespace(__version__="invalid"),
    )
    with pytest.raises(RuntimeError, match="invalid"):
        registration.resolve_hermes_version()


def test_register_adds_exact_hooks_and_cli_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")
    context = RecordingContext()

    registration.register(context)

    assert [name for name, _callback in context.hooks] == list(
        registration.APPROVED_HOOKS
    )
    assert context.cli_calls == [
        {
            "name": "alice-brain",
            "help": "Inspect and control the Alice-brain-Hermes runtime",
            "setup_fn": registration.setup_alice_brain_cli,
            "handler_fn": registration.handle_alice_brain_cli,
            "description": "Alice-brain-Hermes consciousness runtime commands",
        }
    ]


def test_callbacks_are_named_sync_immutable_and_have_exact_return_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    assert isinstance(registration.HOOK_CALLBACKS, MappingProxyType)
    assert tuple(registration.HOOK_CALLBACKS) == registration.APPROVED_HOOKS

    dispatched: list[tuple[str, dict[str, object]]] = []

    def inert_dispatch(hook: str, kwargs: dict[str, object]) -> str | None:
        dispatched.append((hook, kwargs))
        return "cached context" if hook == "pre_llm_call" else None

    monkeypatch.setattr(registration, "_lazy_dispatch", inert_dispatch)
    hostile = object()
    for name, callback in registration.HOOK_CALLBACKS.items():
        signature = inspect.signature(callback)
        assert callback.__name__ == name
        assert not inspect.iscoroutinefunction(callback)
        assert list(signature.parameters) == ["kwargs"]
        assert signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
        assert signature.parameters["kwargs"].annotation in {Any, "Any"}
        if name == "pre_llm_call":
            assert signature.return_annotation == "str | None"
            assert callback(payload=hostile) == "cached context"
        else:
            assert signature.return_annotation in {None, "None"}
            assert callback(payload=hostile) is None

    assert [hook for hook, _kwargs in dispatched] == list(registration.APPROVED_HOOKS)

    with pytest.raises(TypeError):
        registration.HOOK_CALLBACKS["extra"] = lambda **_kwargs: None  # type: ignore[index]


def test_public_callbacks_fail_open_when_lazy_bootstrap_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=1,
        start_worker_on_capture=False,
    )
    bootstrap.publish_context("cached context")
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)

    def failed_bootstrap(_hook: str, _kwargs: dict[str, object]) -> str | None:
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(registration, "_lazy_dispatch", failed_bootstrap)

    for name, callback in registration.HOOK_CALLBACKS.items():
        result = callback(telemetry_schema_version="hermes.observer.v1")
        if name == "pre_llm_call":
            assert result == "cached context"
        else:
            assert result is None

    first, merged = bootstrap.pending_for_test()
    assert (first.capture_seq, first.last_capture_seq) == (1, 1)
    assert (merged.capture_seq, merged.last_capture_seq) == (2, 16)
    assert first.gap_cause == "callback_internal"
    assert merged.gap_cause == "callback_internal"
    assert dict(merged.gap_cause_counts or {}) == {"callback_internal": 15}
    assert bootstrap.health.dropped_events == 16
    assert bootstrap.health.pending_records == 16
    assert bootstrap.health.pending_gap_ranges == 2


def test_packaging_is_a_direct_bounded_runtime_dependency() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]

    assert "packaging>=24,<27" in dependencies


def _run_inert_registration_probe() -> dict[str, Any]:
    script = textwrap.dedent(
        """
        import asyncio
        import builtins
        import importlib.metadata
        import json
        import socket
        import sqlite3
        import subprocess
        import sys
        import threading
        import types

        forbidden_prefixes = (
            "alice_brain_hermes.runtime",
            "alice_brain_hermes.protocol",
            "alice_brain_hermes.hermes.bridge",
            "alice_brain_hermes.projections",
        )
        operations = []

        def forbidden(label):
            def fail(*args, **kwargs):
                operations.append(label)
                raise AssertionError(label)
            return fail

        real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith(forbidden_prefixes):
                return forbidden("import:" + name)()
            return real_import(name, *args, **kwargs)

        sqlite3.connect = forbidden("sqlite3.connect")
        socket.socket = forbidden("socket.socket")
        socket.create_connection = forbidden("socket.create_connection")
        subprocess.Popen = forbidden("subprocess.Popen")
        asyncio.create_task = forbidden("asyncio.create_task")
        threading.Thread.start = forbidden("thread.start")
        threading.Thread.join = forbidden("thread.join")
        builtins.__import__ = guarded_import

        host = types.ModuleType("hermes_cli")
        host.__version__ = "0.18.2"
        sys.modules["hermes_cli"] = host

        class Context:
            def __init__(self):
                self.hooks = []
                self.cli = []
            @property
            def llm(self):
                return forbidden("ctx.llm")()
            def register_hook(self, name, callback):
                self.hooks.append((name, callback))
            def register_cli_command(self, **kwargs):
                self.cli.append(kwargs)

        entry_points = [
            item
            for item in importlib.metadata.entry_points(
                group="hermes_agent.plugins"
            )
            if item.name == "alice-brain"
            and item.dist is not None
            and item.dist.name == "alice-brain-hermes"
        ]
        module = entry_points[0].load()
        context = Context()
        module.register(context)
        result = {
            "hook_count": len(context.hooks),
            "cli_count": len(context.cli),
            "operations": operations,
            "operational_modules": sorted(
                name for name in sys.modules
                if name.startswith(forbidden_prefixes)
            ),
        }
        print(json.dumps(result, sort_keys=True))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _run_first_callback_purity_probe() -> dict[str, Any]:
    script = textwrap.dedent(
        """
        import builtins
        import importlib.metadata
        import json
        import os
        import secrets
        import socket
        import sqlite3
        import subprocess
        import sys
        import threading
        import types
        import uuid

        callback_thread = threading.get_ident()
        operations = []
        forbidden_prefixes = (
            "alice_brain_hermes.runtime",
            "alice_brain_hermes.protocol",
            "alice_brain_hermes.hermes.bridge",
            "alice_brain_hermes.hermes.hooks",
            "alice_brain_hermes.projections",
        )

        def on_callback():
            return threading.get_ident() == callback_thread

        def forbidden(label):
            def fail(*args, **kwargs):
                if on_callback():
                    operations.append(label)
                    raise AssertionError(label)
                raise RuntimeError("worker intentionally blocked: " + label)
            return fail

        real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if on_callback() and name.startswith(forbidden_prefixes):
                operations.append("import:" + name)
                raise AssertionError("import:" + name)
            return real_import(name, *args, **kwargs)

        real_start = threading.Thread.start
        def observed_start(self):
            operations.append("thread.start")
            return None

        builtins.__import__ = guarded_import
        secrets.token_hex = forbidden("secrets.token_hex")
        uuid.uuid4 = forbidden("uuid.uuid4")
        os.urandom = forbidden("os.urandom")
        socket.socket = forbidden("socket.socket")
        socket.create_connection = forbidden("socket.create_connection")
        sqlite3.connect = forbidden("sqlite3.connect")
        subprocess.Popen = forbidden("subprocess.Popen")
        threading.Thread.start = observed_start

        host = types.ModuleType("hermes_cli")
        host.__version__ = "0.18.2"
        sys.modules["hermes_cli"] = host

        class Context:
            def __init__(self):
                self.hooks = {}
            def register_hook(self, name, callback):
                self.hooks[name] = callback
            def register_cli_command(self, **kwargs):
                pass

        matches = [
            item for item in importlib.metadata.entry_points(
                group="hermes_agent.plugins"
            )
            if item.name == "alice-brain"
            and item.dist is not None
            and item.dist.name == "alice-brain-hermes"
        ]
        module = matches[0].load()
        context = Context()
        module.register(context)
        result = context.hooks["on_session_start"](
            telemetry_schema_version="hermes.observer.v1",
            session_id="session",
            model="model",
            platform="cli",
        )
        print(json.dumps({
            "result": result,
            "operations": operations,
            "operational_modules": sorted(
                name for name in sys.modules
                if name.startswith(forbidden_prefixes)
            ),
        }, sort_keys=True))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_register_does_not_import_operational_modules() -> None:
    result = _run_inert_registration_probe()

    assert result["hook_count"] == 16
    assert result["cli_count"] == 1
    assert result["operational_modules"] == []


def test_register_does_not_touch_io_provider_or_threads() -> None:
    result = _run_inert_registration_probe()

    assert result["hook_count"] == 16
    assert result["cli_count"] == 1
    assert result["operations"] == []


def test_first_callback_only_starts_worker_without_operational_import_or_entropy() -> (
    None
):
    result = _run_first_callback_purity_probe()

    assert result == {
        "result": None,
        "operations": ["thread.start"],
        "operational_modules": [],
    }


def test_bootstrap_shape_failure_reserves_exact_callback_internal_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)
    monkeypatch.setattr(
        registration,
        "_copy_bootstrap_value",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("shape")),
    )

    assert (
        registration.on_session_start(
            telemetry_schema_version="hermes.observer.v1",
            session_id="session",
            model="model",
            platform="cli",
        )
        is None
    )

    (capture,) = bootstrap.pending_for_test()
    assert capture.capture_seq == 1
    assert capture.gap_cause == "callback_internal"
    health = bootstrap.health
    assert health.trace_complete is False
    assert health.dropped_events == 1
    assert health.pending_records == 1


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("copy failed"), KeyboardInterrupt(), MemoryError()],
    ids=["exception", "keyboard-interrupt", "memory-error"],
)
def test_post_reservation_copy_failure_retains_one_gap_at_the_reserved_sequence(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)

    def fail_copy(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(registration, "_copy_bootstrap_value", fail_copy)

    assert (
        registration.on_session_start(
            telemetry_schema_version="hermes.observer.v1",
            session_id="session",
        )
        is None
    )

    (retained,) = bootstrap.pending_for_test()
    assert (retained.capture_seq, retained.last_capture_seq) == (1, 1)
    assert retained.gap_cause == "callback_internal"
    assert bootstrap.health.dropped_events == 1
    assert bootstrap.health.pending_records == 1


@pytest.mark.parametrize(
    ("failure", "after_insert"),
    [
        (RuntimeError("put failed"), False),
        (KeyboardInterrupt(), False),
        (MemoryError(), False),
        (RuntimeError("put failed after insert"), True),
        (KeyboardInterrupt(), True),
        (MemoryError(), True),
    ],
    ids=[
        "exception-before-insert",
        "keyboard-interrupt-before-insert",
        "memory-error-before-insert",
        "exception-after-insert",
        "keyboard-interrupt-after-insert",
        "memory-error-after-insert",
    ],
)
def test_post_reservation_queue_failure_replaces_the_same_sequence_with_a_gap(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    after_insert: bool,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)
    original_put = bootstrap._queue.put_nowait  # type: ignore[attr-defined]
    calls = 0

    def fail_put(item: object) -> None:
        nonlocal calls
        calls += 1
        if after_insert and calls == 1:
            original_put(item)
        raise failure

    monkeypatch.setattr(bootstrap._queue, "put_nowait", fail_put)  # type: ignore[attr-defined]

    assert (
        registration.on_session_start(
            telemetry_schema_version="hermes.observer.v1",
            session_id="session",
        )
        is None
    )

    (retained,) = bootstrap.pending_for_test()
    assert (retained.capture_seq, retained.last_capture_seq) == (1, 1)
    assert retained.gap_cause == "callback_internal"
    assert bootstrap.health.dropped_events == 1
    assert bootstrap.health.pending_records == 1
    assert bootstrap.health.pending_gap_ranges == 1


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("after reservation"), KeyboardInterrupt(), MemoryError()],
    ids=["exception", "keyboard-interrupt", "memory-error"],
)
def test_lazy_dispatch_failure_after_capture_converts_that_reservation_only(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)

    def capture_then_fail(hook: str, kwargs: dict[str, object]) -> str | None:
        bootstrap.capture(hook, kwargs)
        raise failure

    monkeypatch.setattr(registration, "_lazy_dispatch", capture_then_fail)

    assert (
        registration.on_session_start(
            telemetry_schema_version="hermes.observer.v1",
            session_id="session",
        )
        is None
    )

    (retained,) = bootstrap.pending_for_test()
    assert (retained.capture_seq, retained.last_capture_seq) == (1, 1)
    assert retained.gap_cause == "callback_internal"
    assert bootstrap.health.dropped_events == 1
    assert bootstrap.health.pending_records == 1

    bootstrap.capture(
        "on_session_end",
        {
            "telemetry_schema_version": "hermes.observer.v1",
            "session_id": "next-callback",
        },
    )
    first, second = bootstrap.pending_for_test()
    assert first is retained
    assert second.capture_seq == 2
    assert second.gap_cause is None


def test_worker_cannot_observe_a_reservation_before_its_callback_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)
    observation_staged = threading.Event()
    release_callback = threading.Event()
    worker_finished = threading.Event()
    worker_result: list[object] = []

    def capture_then_fail(hook: str, kwargs: dict[str, object]) -> str | None:
        bootstrap.capture(hook, kwargs)
        observation_staged.set()
        assert release_callback.wait(2)
        raise RuntimeError("post-capture failure")

    monkeypatch.setattr(registration, "_lazy_dispatch", capture_then_fail)

    callback = threading.Thread(
        target=registration.on_session_start,
        kwargs={
            "telemetry_schema_version": "hermes.observer.v1",
            "session_id": "session",
        },
    )

    def read_for_worker() -> None:
        worker_result.append(bootstrap.next_for_worker())
        worker_finished.set()

    worker = threading.Thread(target=read_for_worker)
    callback.start()
    assert observation_staged.wait(2)
    worker.start()
    assert worker_finished.wait(0.2) is False
    release_callback.set()
    callback.join(2)
    worker.join(2)

    assert not callback.is_alive()
    assert not worker.is_alive()
    (retained,) = worker_result
    assert retained is not None
    assert retained.capture_seq == 1
    assert retained.gap_cause == "callback_internal"


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("cache failed"), KeyboardInterrupt(), MemoryError()],
    ids=["exception", "keyboard-interrupt", "memory-error"],
)
def test_pre_llm_cache_failure_converts_its_existing_reservation_to_one_gap(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)

    def fail_read() -> str | None:
        raise failure

    monkeypatch.setattr(bootstrap, "read_context", fail_read)

    assert (
        registration.pre_llm_call(
            telemetry_schema_version="hermes.observer.v1",
            session_id="session",
        )
        is None
    )

    (retained,) = bootstrap.pending_for_test()
    assert (retained.capture_seq, retained.last_capture_seq) == (1, 1)
    assert retained.gap_cause == "callback_internal"
    assert bootstrap.health.dropped_events == 1
    assert bootstrap.health.pending_records == 1


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("notify failed"), KeyboardInterrupt(), MemoryError()],
    ids=["exception", "keyboard-interrupt", "memory-error"],
)
def test_worker_notification_failure_never_escapes_the_public_callback(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)

    def fail_notification() -> None:
        raise failure

    monkeypatch.setattr(bootstrap, "_notify_worker", fail_notification)

    assert (
        registration.on_session_start(
            telemetry_schema_version="hermes.observer.v1",
            session_id="session",
        )
        is None
    )

    (retained,) = bootstrap.pending_for_test()
    assert retained.capture_seq == 1
    assert retained.gap_cause is None
    assert bootstrap.health.degraded is True
    assert bootstrap.health.last_error == type(failure).__name__


def test_worker_degradation_and_capture_gap_health_updates_cannot_overwrite_each_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    original_replace = registration.replace
    degradation_snapshot_taken = threading.Event()
    release_degradation = threading.Event()
    capture_finished = threading.Event()

    def pause_degradation(instance: object, **changes: object) -> object:
        if changes.get("degraded") is True and "last_error" in changes:
            degradation_snapshot_taken.set()
            assert release_degradation.wait(2)
        return original_replace(instance, **changes)

    monkeypatch.setattr(registration, "replace", pause_degradation)
    degradation = threading.Thread(
        target=bootstrap.mark_worker_degraded,
        args=(RuntimeError("worker"),),
    )

    def capture_gap() -> None:
        bootstrap.capture(
            "on_session_start",
            {"telemetry_schema_version": "invalid"},
        )
        capture_finished.set()

    capture = threading.Thread(target=capture_gap)
    degradation.start()
    assert degradation_snapshot_taken.wait(2)
    capture.start()
    capture_finished.wait(0.2)
    release_degradation.set()
    degradation.join(2)
    capture.join(2)

    assert not degradation.is_alive()
    assert not capture.is_alive()
    assert bootstrap.health.degraded is True
    assert bootstrap.health.last_error == "RuntimeError"
    assert bootstrap.health.trace_complete is False
    assert bootstrap.health.dropped_events == 1
    assert bootstrap.health.pending_records == 1


def test_worker_start_failure_and_capture_gap_use_the_same_health_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    original_replace = registration.replace
    real_thread = threading.Thread
    degradation_snapshot_taken = threading.Event()
    release_degradation = threading.Event()
    capture_finished = threading.Event()

    def pause_degradation(instance: object, **changes: object) -> object:
        if changes.get("degraded") is True and "last_error" in changes:
            degradation_snapshot_taken.set()
            assert release_degradation.wait(2)
        return original_replace(instance, **changes)

    class FailedWorker:
        def start(self) -> None:
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(registration, "replace", pause_degradation)
    monkeypatch.setattr(
        registration.threading,
        "Thread",
        lambda **_kwargs: FailedWorker(),
    )
    starter = real_thread(target=bootstrap._start_worker)  # type: ignore[attr-defined]

    def capture_gap() -> None:
        bootstrap.capture(
            "on_session_start",
            {"telemetry_schema_version": "invalid"},
        )
        capture_finished.set()

    capture = real_thread(target=capture_gap)
    starter.start()
    assert degradation_snapshot_taken.wait(2)
    capture.start()
    capture_finished.wait(0.2)
    release_degradation.set()
    starter.join(2)
    capture.join(2)

    assert not starter.is_alive()
    assert not capture.is_alive()
    assert bootstrap.health.worker_started is False
    assert bootstrap.health.degraded is True
    assert bootstrap.health.last_error == "RuntimeError"
    assert bootstrap.health.trace_complete is False
    assert bootstrap.health.dropped_events == 1
    assert bootstrap.health.pending_records == 1


def test_bootstrap_copy_cost_is_bounded_for_hostile_nested_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    import tracemalloc

    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)
    huge_string = "大" * 2_000_000
    huge_mapping = {str(index): huge_string for index in range(100_000)}

    tracemalloc.start()
    started = time.perf_counter()
    assert (
        registration.pre_tool_call(
            telemetry_schema_version="hermes.observer.v1",
            tool_name="terminal",
            args=huge_mapping,
            task_id="task",
            session_id="session",
            tool_call_id="tool",
            turn_id="turn",
            api_request_id="request",
            middleware_trace=[],
        )
        is None
    )
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert elapsed < 0.15
    assert peak < 2_000_000
    (capture,) = bootstrap.pending_for_test()
    assert capture.capture_seq == 1
    assert capture.gap_cause is None
    assert capture.detached_kwargs is not None


def test_bootstrap_overflow_merges_alternating_gap_causes_into_one_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=1,
        start_worker_on_capture=False,
    )
    original_copy = registration._copy_bootstrap_value  # type: ignore[attr-defined]

    def sometimes_fail(value: object, *args: object, **kwargs: object) -> object:
        if value == "explode":
            raise RuntimeError("hostile copier value")
        return original_copy(value, *args, **kwargs)

    monkeypatch.setattr(registration, "_copy_bootstrap_value", sometimes_fail)
    bootstrap.capture(
        "on_session_start",
        {
            "telemetry_schema_version": "hermes.observer.v1",
            "session_id": "queued",
        },
    )
    for index in range(200):
        if index % 2 == 0:
            payload = {"telemetry_schema_version": "invalid"}
        else:
            payload = {
                "telemetry_schema_version": "hermes.observer.v1",
                "payload": "explode",
            }
        bootstrap.capture("on_session_start", payload)

    queued, overflow = bootstrap.pending_for_test()
    assert queued.capture_seq == 1
    assert overflow.capture_seq == 2
    assert overflow.last_capture_seq == 201
    assert overflow.gap_cause is None
    assert dict(overflow.gap_cause_counts or {}) == {
        "callback_internal": 100,
        "invalid_source_schema": 100,
    }
    assert bootstrap.health.pending_gap_ranges == 1
    assert bootstrap.health.pending_records == 201


def test_same_context_registers_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from alice_brain_hermes.hermes import registration

    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")
    context = RecordingContext()
    registration.register(context)

    def unexpected_resolution() -> str:
        raise AssertionError("registered context must return before version resolution")

    monkeypatch.setattr(registration, "resolve_hermes_version", unexpected_resolution)
    registration.register(context)

    assert len(context.hooks) == 16
    assert len(context.cli_calls) == 1
    assert context._alice_brain_hermes_registration_v1 == "registered"


def test_concurrent_same_context_registers_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    caller_count = 12
    barrier = threading.Barrier(caller_count)
    resolution_count = 0
    resolution_lock = threading.Lock()

    def resolve() -> str:
        nonlocal resolution_count
        with resolution_lock:
            resolution_count += 1
        return "0.18.2"

    monkeypatch.setattr(registration, "resolve_hermes_version", resolve)
    context = RecordingContext()

    def call_register() -> None:
        barrier.wait()
        registration.register(context)

    with ThreadPoolExecutor(max_workers=caller_count) as executor:
        futures = [executor.submit(call_register) for _ in range(caller_count)]
        for future in futures:
            future.result()

    assert resolution_count == 1
    assert [name for name, _callback in context.hooks] == list(
        registration.APPROVED_HOOKS
    )
    assert len(context.cli_calls) == 1


def test_reentrant_registration_fails_visibly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")

    class ReentrantContext(RecordingContext):
        attempted = False

        def register_hook(self, hook_name: str, callback: object) -> None:
            if not self.attempted:
                self.attempted = True
                registration.register(self)
            super().register_hook(hook_name, callback)

    context = ReentrantContext()
    with pytest.raises(RuntimeError, match="re-entrant"):
        registration.register(context)

    assert context.hooks == []
    assert context.cli_calls == []
    assert context._alice_brain_hermes_registration_v1 == "failed"


class EighthHookFailureContext(RecordingContext):
    def __init__(self) -> None:
        super().__init__()
        self.hook_attempts = 0

    def register_hook(self, hook_name: str, callback: object) -> None:
        self.hook_attempts += 1
        if self.hook_attempts == 8:
            raise ValueError("eighth hook rejected")
        super().register_hook(hook_name, callback)


def test_partial_registration_failure_poisons_only_that_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")
    context = EighthHookFailureContext()

    with pytest.raises(ValueError, match="eighth hook"):
        registration.register(context)
    snapshot = (list(context.hooks), list(context.cli_calls), context.hook_attempts)

    with pytest.raises(RuntimeError, match="previously failed"):
        registration.register(context)

    assert (context.hooks, context.cli_calls, context.hook_attempts) == snapshot
    assert context._alice_brain_hermes_registration_v1 == "failed"


def test_partial_registration_failure_reports_bounded_hook_coverage_without_a_fake_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)
    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")
    context = EighthHookFailureContext()

    with pytest.raises(ValueError, match="eighth hook"):
        registration.register(context)

    health = bootstrap.health
    assert health.registration_attempts == 1
    assert health.registration_failures == 1
    assert health.registration_complete is False
    assert health.registered_hook_count == 7
    assert health.missing_hooks == registration.APPROVED_HOOKS[7:]
    assert health.degraded is True
    assert health.trace_complete is False
    assert health.dropped_events == 0
    assert health.pending_records == 0
    assert bootstrap.pending_for_test() == ()

    with pytest.raises(RuntimeError, match="previously failed"):
        registration.register(context)
    assert bootstrap.health == health

    first_active_callback = context.hooks[0][1]
    assert callable(first_active_callback)
    assert (
        first_active_callback(
            telemetry_schema_version="hermes.observer.v1",
            session_id="still-active",
        )
        is None
    )
    (retained,) = bootstrap.pending_for_test()
    assert retained.hook == registration.APPROVED_HOOKS[0]
    assert retained.gap_cause is None
    assert bootstrap.health.registration_complete is False
    assert bootstrap.health.trace_complete is False
    assert bootstrap.health.degraded is True
    assert bootstrap.health.dropped_events == 0

    retained_for_worker = bootstrap.next_for_worker()
    assert retained_for_worker is retained
    bootstrap.mark_handed_off(retained)
    assert bootstrap.health.registration_complete is False
    assert bootstrap.health.trace_complete is False
    assert bootstrap.health.degraded is True
    assert bootstrap.health.last_error == "ValueError"


def test_first_hook_registration_failure_reports_zero_confirmed_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)
    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")

    class FirstHookFailureContext(RecordingContext):
        def register_hook(self, hook_name: str, callback: object) -> None:
            raise RuntimeError("first hook rejected")

    with pytest.raises(RuntimeError, match="first hook"):
        registration.register(FirstHookFailureContext())

    assert bootstrap.health.registration_complete is False
    assert bootstrap.health.registered_hook_count == 0
    assert bootstrap.health.missing_hooks == registration.APPROVED_HOOKS
    assert bootstrap.health.dropped_events == 0
    assert bootstrap.pending_for_test() == ()


def test_complete_registration_reports_all_hook_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)
    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")
    context = RecordingContext()

    registration.register(context)

    health = bootstrap.health
    assert health.registration_attempts == 1
    assert health.registration_failures == 0
    assert health.registration_complete is True
    assert health.registered_hook_count == len(registration.APPROVED_HOOKS)
    assert health.missing_hooks == ()
    assert health.degraded is False
    assert health.trace_complete is True
    assert health.dropped_events == 0
    assert bootstrap.pending_for_test() == ()

    registration.register(context)
    assert bootstrap.health == health


def test_later_complete_context_does_not_erase_append_only_partial_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    bootstrap = registration._BootstrapCaptureBuffer(  # type: ignore[attr-defined]
        queue_capacity=4,
        start_worker_on_capture=False,
    )
    monkeypatch.setattr(registration, "_BOOTSTRAP", bootstrap)
    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")
    partial = EighthHookFailureContext()
    with pytest.raises(ValueError, match="eighth hook"):
        registration.register(partial)

    registration.register(RecordingContext())

    health = bootstrap.health
    assert health.registration_attempts == 2
    assert health.registration_failures == 1
    assert health.registration_complete is False
    assert health.registered_hook_count == 7
    assert health.missing_hooks == registration.APPROVED_HOOKS[7:]
    assert health.degraded is True
    assert health.trace_complete is False
    assert health.dropped_events == 0


def test_fresh_context_registers_after_previous_context_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")
    failed = EighthHookFailureContext()
    with pytest.raises(ValueError, match="eighth hook"):
        registration.register(failed)

    fresh = RecordingContext()
    registration.register(fresh)

    assert len(fresh.hooks) == 16
    assert len(fresh.cli_calls) == 1
    assert fresh._alice_brain_hermes_registration_v1 == "registered"
    assert failed._alice_brain_hermes_registration_v1 == "failed"


def test_context_is_validated_before_registration_state_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")
    context = SimpleNamespace(register_hook=lambda *_args: None)

    with pytest.raises(RuntimeError, match="callables"):
        registration.register(context)

    assert not hasattr(context, "_alice_brain_hermes_registration_v1")


def test_host_version_is_validated_before_registration_state_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.19.0")
    context = RecordingContext()

    with pytest.raises(RuntimeError, match="unsupported"):
        registration.register(context)

    assert context.hooks == []
    assert context.cli_calls == []
    assert not hasattr(context, "_alice_brain_hermes_registration_v1")


def test_final_state_transition_failure_marks_context_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")

    class FinalTransitionFailureContext(RecordingContext):
        def __setattr__(self, name: str, value: object) -> None:
            if name == "_alice_brain_hermes_registration_v1" and value == "registered":
                raise ValueError("registered transition rejected")
            super().__setattr__(name, value)

    context = FinalTransitionFailureContext()
    with pytest.raises(ValueError, match="registered transition"):
        registration.register(context)

    assert len(context.hooks) == 16
    assert len(context.cli_calls) == 1
    assert context._alice_brain_hermes_registration_v1 == "failed"


def test_failed_state_transition_does_not_mask_registration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alice_brain_hermes.hermes import registration

    monkeypatch.setattr(registration, "resolve_hermes_version", lambda: "0.18.2")

    class FailedTransitionFailureContext(RecordingContext):
        def __setattr__(self, name: str, value: object) -> None:
            if name == "_alice_brain_hermes_registration_v1" and value == "failed":
                raise RuntimeError("failed transition rejected")
            super().__setattr__(name, value)

        def register_hook(self, hook_name: str, callback: object) -> None:
            raise ValueError("primary registration failure")

    context = FailedTransitionFailureContext()
    with pytest.raises(ValueError, match="primary registration failure") as captured:
        registration.register(context)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "failed transition rejected"


def _real_host_plugins() -> ModuleType:
    return pytest.importorskip(
        "hermes_cli.plugins",
        reason="real Hermes integration requires the local hermes-agent checkout",
    )


def _write_enabled_config(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["alice-brain"]}}),
        encoding="utf-8",
    )


def _assert_real_manager_surface(manager: object, host: ModuleType) -> None:
    from alice_brain_hermes.hermes.registration import APPROVED_HOOKS

    loaded = manager._plugins["alice-brain"]  # type: ignore[attr-defined]
    assert loaded.enabled is True
    assert loaded.error is None
    assert loaded.hooks_registered == list(APPROVED_HOOKS)
    assert tuple(manager._hooks) == APPROVED_HOOKS  # type: ignore[attr-defined]
    assert all(
        len(manager._hooks[name]) == 1  # type: ignore[attr-defined]
        for name in APPROVED_HOOKS
    )
    assert set(manager._cli_commands) == {"alice-brain"}  # type: ignore[attr-defined]
    assert manager._cli_commands["alice-brain"] == {  # type: ignore[attr-defined]
        "name": "alice-brain",
        "help": "Inspect and control the Alice-brain-Hermes runtime",
        "setup_fn": loaded.module.register.__globals__["setup_alice_brain_cli"],
        "handler_fn": loaded.module.register.__globals__["handle_alice_brain_cli"],
        "description": "Alice-brain-Hermes consciousness runtime commands",
        "plugin": "alice-brain",
    }
    for hook_name in APPROVED_HOOKS:
        assert manager.invoke_hook(hook_name, payload=object()) == []  # type: ignore[attr-defined]
    assert host.__name__ == "hermes_cli.plugins"


def test_disabled_entrypoint_is_discovered_but_not_imported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host = _real_host_plugins()
    home = tmp_path / "Hermes Home 未啟用"
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = host.PluginManager()
    monkeypatch.setattr(manager, "_scan_directory", lambda *_args, **_kwargs: [])
    load_attempts: list[str] = []

    def reject_load(manifest: object) -> ModuleType:
        load_attempts.append(manifest.name)
        raise AssertionError("disabled entry point was imported")

    monkeypatch.setattr(manager, "_load_entrypoint_module", reject_load)
    manager.discover_and_load()

    loaded = manager._plugins["alice-brain"]
    assert loaded.enabled is False
    assert loaded.module is None
    assert load_attempts == []
    assert manager._hooks == {}
    assert manager._cli_commands == {}


def test_enabled_entrypoint_loads_exact_hooks_and_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host = _real_host_plugins()
    home = tmp_path / "Hermes Home 入口點"
    _write_enabled_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = host.PluginManager()
    monkeypatch.setattr(manager, "_scan_directory", lambda *_args, **_kwargs: [])
    operational_before = set(sys.modules)

    manager.discover_and_load()

    operational_after_discovery = {
        name
        for name in set(sys.modules) - operational_before
        if name.startswith(
            (
                "alice_brain_hermes.runtime",
                "alice_brain_hermes.protocol",
                "alice_brain_hermes.hermes.bridge",
                "alice_brain_hermes.projections",
            )
        )
    }
    assert operational_after_discovery == set()
    # Surface validation invokes every callback.  The first callback may start
    # the bootstrap worker; operational imports are then worker-owned rather
    # than registration/discovery side effects.
    _assert_real_manager_surface(manager, host)


def test_enabled_directory_plugin_loads_exact_hooks_and_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host = _real_host_plugins()
    home = tmp_path / "Hermes Home 目錄插件"
    _write_enabled_config(home)
    plugin_directory = home / "plugins" / "alice-brain"
    shutil.copytree(INTEGRATION_ROOT, plugin_directory)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(host, "get_bundled_plugins_dir", lambda: tmp_path / "none")
    manager = host.PluginManager()
    monkeypatch.setattr(manager, "_scan_entry_points", lambda: [])
    operational_before = set(sys.modules)

    manager.discover_and_load()

    loaded = manager._plugins["alice-brain"]
    assert loaded.manifest.source == "user"
    assert loaded.manifest.provides_hooks == list(
        __import__(
            "alice_brain_hermes.hermes.registration",
            fromlist=["APPROVED_HOOKS"],
        ).APPROVED_HOOKS
    )
    operational_after_discovery = {
        name
        for name in set(sys.modules) - operational_before
        if name.startswith(
            (
                "alice_brain_hermes.runtime",
                "alice_brain_hermes.protocol",
                "alice_brain_hermes.hermes.bridge",
                "alice_brain_hermes.projections",
            )
        )
    }
    assert operational_after_discovery == set()
    # Surface validation invokes every callback.  The first callback may start
    # the bootstrap worker; operational imports are then worker-owned rather
    # than registration/discovery side effects.
    _assert_real_manager_surface(manager, host)


def test_entrypoint_and_directory_paths_are_tested_in_isolation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host = _real_host_plugins()

    entry_home = tmp_path / "entry-only"
    _write_enabled_config(entry_home)
    monkeypatch.setenv("HERMES_HOME", str(entry_home))
    entry_manager = host.PluginManager()
    monkeypatch.setattr(
        entry_manager,
        "_scan_directory",
        lambda *_args, **_kwargs: [],
    )
    entry_manager.discover_and_load()

    directory_home = tmp_path / "directory-only"
    _write_enabled_config(directory_home)
    shutil.copytree(
        INTEGRATION_ROOT,
        directory_home / "plugins" / "alice-brain",
    )
    monkeypatch.setenv("HERMES_HOME", str(directory_home))
    monkeypatch.setattr(host, "get_bundled_plugins_dir", lambda: tmp_path / "none")
    directory_manager = host.PluginManager()
    monkeypatch.setattr(directory_manager, "_scan_entry_points", lambda: [])
    directory_manager.discover_and_load()

    assert entry_manager._plugins["alice-brain"].manifest.source == "entrypoint"
    assert directory_manager._plugins["alice-brain"].manifest.source == "user"
    _assert_real_manager_surface(entry_manager, host)
    _assert_real_manager_surface(directory_manager, host)


def test_fresh_context_registers_after_force_discovery_clears_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host = _real_host_plugins()
    home = tmp_path / "Hermes Home force"
    _write_enabled_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = host.PluginManager()
    monkeypatch.setattr(manager, "_scan_directory", lambda *_args, **_kwargs: [])
    manager.discover_and_load()
    first_callbacks = {
        name: manager._hooks[name][0]
        for name in manager._plugins["alice-brain"].hooks_registered
    }

    manager.discover_and_load(force=True)

    _assert_real_manager_surface(manager, host)
    assert {
        name: manager._hooks[name][0]
        for name in manager._plugins["alice-brain"].hooks_registered
    } == first_callbacks


def _run_real_hermes_cli(
    tmp_path: Path,
    *arguments: str,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    pytest.importorskip(
        "hermes_cli.main",
        reason="real Hermes CLI requires the local hermes-agent checkout",
    )
    home = tmp_path / "Hermes Home CLI 測試"
    alice_home = tmp_path / "Alice runtime 不應出現"
    _write_enabled_config(home)
    environment = os.environ.copy()
    environment["HERMES_HOME"] = str(home)
    environment["ALICE_BRAIN_HERMES_HOME"] = str(alice_home)
    completed = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *arguments],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed, alice_home


def test_bare_hermes_help_does_not_need_plugin_discovery(tmp_path: Path) -> None:
    completed, alice_home = _run_real_hermes_cli(tmp_path, "--help")

    assert completed.returncode == 0
    assert "alice-brain" not in completed.stdout
    assert "Traceback" not in completed.stdout + completed.stderr
    assert not alice_home.exists()


def test_enabled_hermes_alice_brain_help_uses_lazy_cli(tmp_path: Path) -> None:
    help_result, alice_home = _run_real_hermes_cli(
        tmp_path,
        "alice-brain",
        "--help",
    )
    handler_result, handler_alice_home = _run_real_hermes_cli(
        tmp_path,
        "alice-brain",
    )

    assert help_result.returncode == 0
    assert handler_result.returncode == 0
    for result in (help_result, handler_result):
        combined = result.stdout + result.stderr
        assert "alice-brain" in result.stdout
        assert "Alice-brain-Hermes consciousness runtime commands" in result.stdout
        assert "Traceback" not in combined
    assert not alice_home.exists()
    assert not handler_alice_home.exists()


def test_enabled_hermes_alice_brain_propagates_machine_failure_exit(
    tmp_path: Path,
) -> None:
    completed, alice_home = _run_real_hermes_cli(
        tmp_path,
        "alice-brain",
        "identity",
    )

    assert completed.returncode == 3
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert payload["code"] == "daemon_not_running"
    assert payload["ok"] is False
    assert not alice_home.exists()


def test_lazy_cli_handler_prints_stored_parser_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from alice_brain_hermes.hermes.registration import (
        handle_alice_brain_cli,
        setup_alice_brain_cli,
    )

    parser = argparse.ArgumentParser(
        prog="hermes alice-brain",
        description="Alice-brain-Hermes consciousness runtime commands",
    )
    setup_alice_brain_cli(parser)
    args = parser.parse_args([])

    assert handle_alice_brain_cli(args) == 0
    output = capsys.readouterr().out
    assert "usage: hermes alice-brain" in output
    assert "Alice-brain-Hermes consciousness runtime commands" in output


@pytest.fixture(scope="module")
def task5_release_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    output_directory = tmp_path_factory.mktemp("task5-release-artifacts")
    completed = subprocess.run(
        ["uv", "build", "--out-dir", str(output_directory)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    wheels = list(output_directory.glob("*.whl"))
    source_distributions = list(output_directory.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(source_distributions) == 1
    return wheels[0], source_distributions[0]


def test_wheel_contains_entrypoint_and_package_modules(
    task5_release_artifacts: tuple[Path, Path],
) -> None:
    wheel, _source_distribution = task5_release_artifacts

    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        entry_points_names = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        assert "alice_brain_hermes/hermes_plugin.py" in names
        assert "alice_brain_hermes/hermes/__init__.py" in names
        assert "alice_brain_hermes/hermes/registration.py" in names
        assert len(entry_points_names) == 1
        entry_points_text = archive.read(entry_points_names[0]).decode("utf-8")
        assert "[hermes_agent.plugins]" in entry_points_text
        assert "alice-brain = alice_brain_hermes.hermes_plugin" in entry_points_text


def test_wheel_entrypoint_loads_outside_checkout(
    task5_release_artifacts: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    wheel, _source_distribution = task5_release_artifacts
    environment = tmp_path / "wheel-only-environment"
    create_result = subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert create_result.returncode == 0, create_result.stderr
    python_executable = (
        environment / "Scripts" / "python.exe"
        if os.name == "nt"
        else environment / "bin" / "python"
    )
    install_result = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_executable),
            str(wheel),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert install_result.returncode == 0, install_result.stderr
    probe = textwrap.dedent(
        """
        import inspect
        from importlib import metadata
        from types import ModuleType

        matches = [
            item
            for item in metadata.entry_points(group="hermes_agent.plugins")
            if item.name == "alice-brain"
            and item.dist is not None
            and item.dist.name == "alice-brain-hermes"
        ]
        assert len(matches) == 1
        module = matches[0].load()
        assert isinstance(module, ModuleType)
        assert module.__name__ == "alice_brain_hermes.hermes_plugin"
        assert callable(module.register)
        assert not inspect.iscoroutinefunction(module.register)
        """
    )
    probe_result = subprocess.run(
        [str(python_executable), "-I", "-c", probe],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe_result.returncode == 0, probe_result.stderr


def test_sdist_contains_directory_integration_artifact(
    task5_release_artifacts: tuple[Path, Path],
) -> None:
    _wheel, source_distribution = task5_release_artifacts

    with tarfile.open(source_distribution, mode="r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
        assert any(
            name.endswith("/integration/alice-brain/plugin.yaml") for name in names
        )
        assert any(
            name.endswith("/integration/alice-brain/__init__.py") for name in names
        )


def test_wheel_has_no_separate_alice_brain_dependency(
    task5_release_artifacts: tuple[Path, Path],
) -> None:
    wheel, _source_distribution = task5_release_artifacts

    with ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_names) == 1
        message = BytesParser(policy=compat32).parsebytes(
            archive.read(metadata_names[0])
        )
        requirements = [
            Requirement(value) for value in message.get_all("Requires-Dist", [])
        ]

    normalized_names = {
        requirement.name.lower().replace("_", "-") for requirement in requirements
    }
    assert "packaging" in normalized_names
    assert "alice-brain" not in normalized_names
