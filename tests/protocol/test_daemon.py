from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Never

import psutil
import pytest

from alice_brain_hermes.errors import DaemonRpcError
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.client import DaemonClient
from alice_brain_hermes.protocol.models import (
    PROTOCOL_VERSION,
    BrainProfileV1,
    CapabilityProfileV1,
    CoverageV1,
    ProtocolLimitsV1,
)
from alice_brain_hermes.runtime.discovery import load_discovery_and_credential
from alice_brain_hermes.runtime.process_marker import read_process_marker
from alice_brain_hermes.runtime.store import SQLiteLedger
from alice_brain_hermes.runtime.supervisor import (
    DmonAdapter,
    DmonIdentityError,
    DmonProcessHint,
)

RECOVERY_TOKEN = "ab" * 32


@dataclass
class _ManagedProtocolLaunch:
    process: subprocess.Popen[bytes]
    adapter: DmonAdapter
    supervisor_hint: DmonProcessHint | None = None
    descendant_hints: list[DmonProcessHint] = field(default_factory=list)
    daemon_hint: DmonProcessHint | None = None
    captured_stdout: bytes | None = None
    captured_stderr: bytes | None = None
    promoted_from_quarantine: bool = False
    cleaned: bool = False

    def exact_hints(self) -> tuple[DmonProcessHint, ...]:
        ordered = (
            self.supervisor_hint,
            *self.descendant_hints,
            self.daemon_hint,
        )
        unique: list[DmonProcessHint] = []
        for hint in ordered:
            if hint is not None and hint not in unique:
                unique.append(hint)
        return tuple(unique)

    def remember_descendant(self, hint: DmonProcessHint) -> None:
        if hint != self.supervisor_hint and hint not in self.descendant_hints:
            self.descendant_hints.append(hint)

    def settle_exact_handle_exit(self) -> None:
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as error:
            raise DmonIdentityError(
                "protocol launch cleanup could not prove supervisor exit"
            ) from error
        self.captured_stdout, self.captured_stderr = self.process.communicate()
        self.cleaned = True

    def cleanup(self) -> None:
        if self.cleaned:
            return
        if self.supervisor_hint is None:
            if self.process.poll() is not None:
                self.settle_exact_handle_exit()
                return
            try:
                self.supervisor_hint = _bootstrap_supervisor_hint(self.process)
            except DmonIdentityError:
                if self.process.poll() is not None:
                    self.settle_exact_handle_exit()
                    return
                raise
            self.promoted_from_quarantine = True
            _capture_supervisor_descendant_hints(self)
        for hint in self.exact_hints():
            self.adapter.terminate_exact(
                hint,
                timeout_seconds=5.0,
            )
        self.settle_exact_handle_exit()


_ACTIVE_PROTOCOL_LAUNCHES: list[_ManagedProtocolLaunch] = []


def _discard_cleaned_protocol_launches() -> None:
    _ACTIVE_PROTOCOL_LAUNCHES[:] = [
        tracked for tracked in _ACTIVE_PROTOCOL_LAUNCHES if not tracked.cleaned
    ]


def _cleanup_registered_protocol_launches() -> None:
    errors: list[BaseException] = []
    for tracked in reversed(_ACTIVE_PROTOCOL_LAUNCHES):
        try:
            tracked.cleanup()
        except BaseException as error:
            errors.append(error)
    _discard_cleaned_protocol_launches()
    if errors:
        for extra in errors[1:]:
            errors[0].add_note(f"additional cleanup failure: {extra!r}")
        raise errors[0]


def _prepare_protocol_launch_registry_for_setup() -> None:
    if _ACTIVE_PROTOCOL_LAUNCHES:
        _cleanup_registered_protocol_launches()
    if _ACTIVE_PROTOCOL_LAUNCHES:
        raise RuntimeError("protocol launch cleanup remains unproven")


@pytest.fixture(autouse=True)
def _cleanup_protocol_launches_after_test():
    _prepare_protocol_launch_registry_for_setup()
    try:
        yield
    finally:
        _cleanup_registered_protocol_launches()


def _bootstrap_supervisor_hint(
    process: subprocess.Popen[bytes],
) -> DmonProcessHint:
    if process.poll() is not None:
        raise DmonIdentityError(
            "protocol supervisor exited before marker capture",
            cleanup_unproven=True,
        )
    try:
        marker = read_process_marker(process.pid)
    except (OSError, ValueError) as error:
        raise DmonIdentityError(
            "protocol supervisor marker capture is unproven",
            cleanup_unproven=True,
        ) from error
    if process.poll() is not None:
        raise DmonIdentityError(
            "protocol supervisor exited during marker capture",
            cleanup_unproven=True,
        )
    return DmonProcessHint(pid=process.pid, process_marker=marker)


def _actual_python_image() -> str:
    try:
        executable = psutil.Process().exe()
    except (OSError, psutil.Error) as error:
        raise DmonIdentityError(
            "actual Python process image is not verifiable",
            cleanup_unproven=True,
        ) from error
    if not isinstance(executable, str) or not executable:
        raise DmonIdentityError(
            "actual Python process image is not verifiable",
            cleanup_unproven=True,
        )
    path = Path(executable)
    if not path.is_file():
        raise DmonIdentityError(
            "actual Python process image is not a regular file",
            cleanup_unproven=True,
        )
    return str(path)


def _raw_launch_environment() -> dict[str, str]:
    environment = dict(os.environ)
    current_directory = os.getcwd()
    environment["PYTHONPATH"] = os.pathsep.join(
        entry if entry else current_directory for entry in sys.path
    )
    return environment


def _settle_or_quarantine_bootstrap_failure(
    process: subprocess.Popen[bytes],
    adapter: DmonAdapter,
    error: DmonIdentityError,
) -> Never:
    if process.poll() is None:
        _ACTIVE_PROTOCOL_LAUNCHES.append(
            _ManagedProtocolLaunch(
                process=process,
                adapter=adapter,
            )
        )
        raise error
    try:
        return_code = process.wait(timeout=5.0)
        stdout, stderr = process.communicate()
    except (OSError, subprocess.SubprocessError) as settle_error:
        raise DmonIdentityError(
            "unhinted protocol supervisor exit could not be settled",
            cleanup_unproven=True,
        ) from settle_error
    pytest.fail(
        f"daemon exited before readiness: {return_code}; "
        f"stdout={stdout!r}; stderr={stderr!r}"
    )


def _capture_supervisor_descendant_hints(
    tracked: _ManagedProtocolLaunch,
) -> None:
    root_hint = tracked.supervisor_hint
    if root_hint is None:
        return
    try:
        if read_process_marker(root_hint.pid) != root_hint.process_marker:
            return
        root = psutil.Process(root_hint.pid)
        frontier = [(root, root_hint)]
        candidates: list[DmonProcessHint] = []
        visited = {root_hint.pid}
        while frontier:
            parent, parent_hint = frontier.pop(0)
            if read_process_marker(parent.pid) != parent_hint.process_marker:
                continue
            children = parent.children(recursive=False)
            if read_process_marker(parent.pid) != parent_hint.process_marker:
                continue
            for child in children:
                if child.pid in visited:
                    continue
                marker = read_process_marker(child.pid)
                if child.ppid() != parent.pid:
                    continue
                if read_process_marker(child.pid) != marker:
                    continue
                hint = DmonProcessHint(pid=child.pid, process_marker=marker)
                visited.add(child.pid)
                candidates.append(hint)
                frontier.append((child, hint))
        if read_process_marker(root_hint.pid) != root_hint.process_marker:
            return
        for hint in candidates:
            if read_process_marker(hint.pid) == hint.process_marker:
                tracked.remember_descendant(hint)
    except (OSError, ValueError, psutil.Error):
        return


def _related_process_hint(
    *,
    pid: int,
    process_marker: str,
    supervisor_hint: DmonProcessHint,
) -> DmonProcessHint | None:
    try:
        current_marker = read_process_marker(pid)
        if pid == supervisor_hint.pid:
            return (
                supervisor_hint
                if current_marker == process_marker == supervisor_hint.process_marker
                else None
            )
        process = psutil.Process(pid)
        parent_pid = process.ppid()
        parent_marker = read_process_marker(supervisor_hint.pid)
        final_marker = read_process_marker(pid)
    except (OSError, ValueError, psutil.Error):
        return None
    if (
        parent_pid != supervisor_hint.pid
        or parent_marker != supervisor_hint.process_marker
        or current_marker != process_marker
        or final_marker != process_marker
    ):
        return None
    return DmonProcessHint(pid=pid, process_marker=process_marker)


def _matches_launched_process(
    *,
    pid: int,
    process_marker: str,
    supervisor_hint: DmonProcessHint,
) -> bool:
    return (
        _related_process_hint(
            pid=pid,
            process_marker=process_marker,
            supervisor_hint=supervisor_hint,
        )
        is not None
    )


def bridge_attach_params(
    brain_id: str,
    bridge_instance_id: str,
) -> dict[str, object]:
    return {
        "brain_id": brain_id,
        "bridge_instance_id": bridge_instance_id,
        "recovery_token": RECOVERY_TOKEN,
    }


def pre_llm_observation(bridge_instance_id: str, capture_seq: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_kind": "observation",
        "bridge_instance_id": bridge_instance_id,
        "capture_seq": capture_seq,
        "captured_at": datetime.now(UTC).isoformat(),
        "captured_monotonic_ns": capture_seq,
        "source_schema_version": "hermes.observer.v1",
        "hook": "pre_llm_call",
        "context": {
            "session_id": "restart-session",
            "task_id": "restart-task",
            "turn_id": f"restart-turn-{capture_seq}",
            "sender_id": "restart-sender",
        },
        "payload": {
            "user_message": f"capture-{capture_seq}",
            "conversation_history": [],
            "is_first_turn": capture_seq == 1,
            "model": "restart-model",
            "platform": "process-test",
            "extensions": {},
        },
        "coverage": CoverageV1(
            policy_version="copy-v1", capture_coverage="full"
        ).model_dump(mode="json"),
    }


def launch(
    home: Path,
    *,
    expect_ready: bool = True,
    abandonment_grace_seconds: float = 30.0,
    readiness_timeout_seconds: float = 10.0,
):
    launch_nonce = new_id()
    executable = _actual_python_image()
    command = (
        executable,
        "-m",
        "alice_brain_hermes.runtime.daemon",
        "--runtime-home",
        str(home),
        "--launch-nonce",
        launch_nonce,
        "--scheduler-interval",
        "0.02",
        "--abandonment-grace",
        str(abandonment_grace_seconds),
    )
    adapter = DmonAdapter.create(home, command, launch_nonce=launch_nonce)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_raw_launch_environment(),
    )
    try:
        supervisor_hint = _bootstrap_supervisor_hint(process)
    except DmonIdentityError as error:
        _settle_or_quarantine_bootstrap_failure(process, adapter, error)
    tracked = _ManagedProtocolLaunch(
        process=process,
        adapter=adapter,
        supervisor_hint=supervisor_hint,
    )
    _ACTIVE_PROTOCOL_LAUNCHES.append(tracked)
    _capture_supervisor_descendant_hints(tracked)
    deadline = time.monotonic() + readiness_timeout_seconds
    if not expect_ready:
        while process.poll() is None and time.monotonic() < deadline:
            _capture_supervisor_descendant_hints(tracked)
            time.sleep(0.01)
        return_code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        assert return_code != 0
        tracked.cleanup()
        _discard_cleaned_protocol_launches()
        return process, {"ready": False, "code": "runtime_owned"}
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        _capture_supervisor_descendant_hints(tracked)
        if process.poll() is not None:
            tracked.cleanup()
            _discard_cleaned_protocol_launches()
            pytest.fail(
                f"daemon exited before readiness: {process.returncode}; "
                f"stdout={tracked.captured_stdout!r}; "
                f"stderr={tracked.captured_stderr!r}"
            )
        client: DaemonClient | None = None
        try:
            client = DaemonClient.connect(home, timeout_seconds=0.25)
            health = client.health()
            status = client.call("daemon.status", {})
            discovery = client.discovery
            daemon_hint = _related_process_hint(
                pid=discovery.pid,
                process_marker=discovery.process_marker,
                supervisor_hint=supervisor_hint,
            )
            authenticated_identity = (
                discovery.launch_nonce == launch_nonce
                and health.get("pid") == discovery.pid
                and health.get("process_marker") == discovery.process_marker
                and health.get("instance_nonce") == discovery.instance_nonce
                and health.get("launch_nonce") == discovery.launch_nonce
            )
            if daemon_hint is not None and authenticated_identity:
                tracked.daemon_hint = daemon_hint
                tracked.remember_descendant(daemon_hint)
            exact = (
                _matches_launched_process(
                    pid=discovery.pid,
                    process_marker=discovery.process_marker,
                    supervisor_hint=supervisor_hint,
                )
                and authenticated_identity
                and health.get("runtime_ready") is True
                and status.get("runtime_ready") is True
            )
            if exact:
                return process, {
                    "ready": True,
                    "instance_nonce": discovery.instance_nonce,
                    "launch_nonce": discovery.launch_nonce,
                    "pid": discovery.pid,
                    "process_marker": discovery.process_marker,
                }
        except BaseException as error:
            last_error = error
        finally:
            if client is not None:
                client.close()
        time.sleep(0.02)
    tracked.cleanup()
    _discard_cleaned_protocol_launches()
    pytest.fail(
        f"daemon authenticated readiness timed out: {last_error!r}; "
        f"stdout={tracked.captured_stdout!r}; "
        f"stderr={tracked.captured_stderr!r}"
    )


def stop_process(process: subprocess.Popen[bytes], client: DaemonClient) -> None:
    client.shutdown()
    client.close()
    assert process.wait(timeout=10.0) == 0
    stdout, stderr = process.communicate()
    assert stdout == b""
    assert stderr == b""


def _force_process_death(
    process: subprocess.Popen[bytes],
) -> tuple[DmonProcessHint, DmonProcessHint]:
    matches = [
        tracked for tracked in _ACTIVE_PROTOCOL_LAUNCHES if tracked.process is process
    ]
    if len(matches) != 1:
        raise DmonIdentityError("forced death requires one exact managed launch")
    tracked = matches[0]
    _capture_supervisor_descendant_hints(tracked)
    if tracked.supervisor_hint is None or tracked.daemon_hint is None:
        raise DmonIdentityError(
            "forced death requires captured supervisor and daemon identities",
            cleanup_unproven=True,
        )
    supervisor_hint = tracked.supervisor_hint
    daemon_hint = tracked.daemon_hint
    tracked.cleanup()
    return supervisor_hint, daemon_hint


def _assert_exact_process_gone(hint: DmonProcessHint) -> None:
    try:
        current = read_process_marker(hint.pid)
    except PermissionError:
        assert not psutil.pid_exists(hint.pid)
        return
    assert current != hint.process_marker


def test_launch_match_rejects_a_reused_supervisor_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor_hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "read_process_marker",
        lambda _pid: "psutil-create-time-us:999000000",
    )

    assert not _matches_launched_process(
        pid=supervisor_hint.pid,
        process_marker=supervisor_hint.process_marker,
        supervisor_hint=supervisor_hint,
    )


def test_bootstrap_hint_rejects_exit_during_marker_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitedDuringCapture:
        pid = 4321

        def __init__(self) -> None:
            self.poll_results = iter((None, 17))
            self.poll_calls = 0

        def poll(self) -> int | None:
            self.poll_calls += 1
            return next(self.poll_results)

    process = ExitedDuringCapture()
    monkeypatch.setattr(
        sys.modules[__name__],
        "read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )

    with pytest.raises(DmonIdentityError, match="marker capture") as failure:
        _bootstrap_supervisor_hint(process)

    assert failure.value.cleanup_unproven is True
    assert process.poll_calls == 2


def test_launch_reports_controlled_early_exit_and_clears_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = subprocess.Popen

    def controlled_exit(
        _command: tuple[str, ...],
        *,
        stdout: int,
        stderr: int,
        env: dict[str, str],
    ) -> subprocess.Popen[bytes]:
        return real_popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys,time; time.sleep(0.2); "
                    "sys.stdout.write('early-out'); sys.stdout.flush(); "
                    "sys.stderr.write('early-err'); sys.stderr.flush(); "
                    "raise SystemExit(23)"
                ),
            ],
            stdout=stdout,
            stderr=stderr,
            env=env,
        )

    monkeypatch.setattr(subprocess, "Popen", controlled_exit)
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    try:
        with pytest.raises(
            pytest.fail.Exception, match="daemon exited before readiness"
        ) as failure:
            launch(home)

        message = str(failure.value)
        assert "stdout=b'early-out'" in message
        assert "stderr=b'early-err'" in message
        assert _ACTIVE_PROTOCOL_LAUNCHES == []
    finally:
        _ACTIVE_PROTOCOL_LAUNCHES.clear()


def test_launch_bootstrap_exit_never_registers_unhinted_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitedDuringBootstrap:
        pid = 4321
        returncode: int | None = None

        def __init__(self) -> None:
            self.poll_results = iter((None, 17, 17))
            self.wait_calls = 0
            self.communicate_calls = 0

        def poll(self) -> int | None:
            result = next(self.poll_results)
            if result is not None:
                self.returncode = result
            return result

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            self.wait_calls += 1
            self.returncode = 17
            return 17

        def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            return b"bootstrap-out", b"bootstrap-err"

    process = ExitedDuringBootstrap()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        sys.modules[__name__],
        "read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    try:
        with pytest.raises(
            pytest.fail.Exception, match="daemon exited before readiness"
        ) as failure:
            launch(home)

        assert "stdout=b'bootstrap-out'" in str(failure.value)
        assert "stderr=b'bootstrap-err'" in str(failure.value)
        assert process.wait_calls == 1
        assert process.communicate_calls == 1
        assert _ACTIVE_PROTOCOL_LAUNCHES == []
    finally:
        _ACTIVE_PROTOCOL_LAUNCHES.clear()


def test_live_bootstrap_quarantine_promotes_and_cleans_on_marker_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LiveUnhintedHandle:
        pid = 4321
        returncode: int | None = None

        def __init__(self) -> None:
            self.alive = True
            self.poll_calls = 0
            self.wait_calls = 0
            self.communicate_calls = 0

        def poll(self) -> int | None:
            self.poll_calls += 1
            return None if self.alive else 17

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            self.wait_calls += 1
            self.returncode = 17
            return 17

        def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            return b"quarantine-out", b"quarantine-err"

    class Adapter:
        def __init__(self, process: LiveUnhintedHandle) -> None:
            self.process = process
            self.terminated: list[DmonProcessHint] = []

        def terminate_exact(
            self,
            hint: DmonProcessHint,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds > 0
            self.terminated.append(hint)
            self.process.alive = False

    process = LiveUnhintedHandle()
    adapter = Adapter(process)
    marker = "psutil-create-time-us:123000000"
    marker_readable = False

    def read_marker(_pid: int) -> str:
        if not marker_readable:
            raise PermissionError("unreadable")
        return marker

    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        DmonAdapter,
        "create",
        staticmethod(lambda *_args, **_kwargs: adapter),
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "read_process_marker",
        read_marker,
    )
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    try:
        with pytest.raises(DmonIdentityError, match="marker capture") as failure:
            launch(home)

        assert failure.value.cleanup_unproven is True
        assert len(_ACTIVE_PROTOCOL_LAUNCHES) == 1
        quarantined = _ACTIVE_PROTOCOL_LAUNCHES[0]
        assert quarantined.process is process
        assert quarantined.supervisor_hint is None
        assert quarantined.cleaned is False

        marker_readable = True
        _cleanup_registered_protocol_launches()

        expected = DmonProcessHint(pid=process.pid, process_marker=marker)
        assert quarantined.supervisor_hint == expected
        assert adapter.terminated == [expected]
        assert process.wait_calls == 1
        assert process.communicate_calls == 1
        assert _ACTIVE_PROTOCOL_LAUNCHES == []
    finally:
        process.alive = False
        _ACTIVE_PROTOCOL_LAUNCHES.clear()


def test_quarantine_promotion_cleans_exact_descendant_when_root_exits_in_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    child_hint = DmonProcessHint(
        pid=8765,
        process_marker="psutil-create-time-us:456000000",
    )

    class ExactHandle:
        pid = root_hint.pid

        def __init__(self) -> None:
            self.alive = True
            self.wait_calls = 0
            self.communicate_calls = 0

        def poll(self) -> int | None:
            return None if self.alive else 17

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            self.wait_calls += 1
            return 17

        def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            return b"capture-out", b"capture-err"

    class Adapter:
        def __init__(self) -> None:
            self.terminated: list[DmonProcessHint] = []

        def terminate_exact(
            self,
            hint: DmonProcessHint,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds > 0
            self.terminated.append(hint)

    process = ExactHandle()
    adapter = Adapter()
    marker_readable = False

    def read_marker(_pid: int) -> str:
        if not marker_readable:
            raise PermissionError("unreadable")
        return root_hint.process_marker

    def capture_and_exit(tracked: _ManagedProtocolLaunch) -> None:
        tracked.remember_descendant(child_hint)
        process.alive = False

    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        DmonAdapter,
        "create",
        staticmethod(lambda *_args, **_kwargs: adapter),
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "read_process_marker",
        read_marker,
    )
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    try:
        with pytest.raises(DmonIdentityError, match="marker capture"):
            launch(home)
        quarantined = _ACTIVE_PROTOCOL_LAUNCHES[0]
        monkeypatch.setattr(
            sys.modules[__name__],
            "_capture_supervisor_descendant_hints",
            capture_and_exit,
        )

        marker_readable = True
        _cleanup_registered_protocol_launches()

        assert quarantined.supervisor_hint == root_hint
        assert adapter.terminated == [root_hint, child_hint]
        assert process.wait_calls == 1
        assert process.communicate_calls == 1
        assert quarantined.promoted_from_quarantine is True
        assert quarantined.cleaned is True
        assert _ACTIVE_PROTOCOL_LAUNCHES == []
    finally:
        process.alive = False
        _ACTIVE_PROTOCOL_LAUNCHES.clear()


def test_quarantine_promotion_settles_root_exit_without_capture_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )

    class ExactHandle:
        pid = root_hint.pid

        def __init__(self) -> None:
            self.alive = True
            self.wait_calls = 0

        def poll(self) -> int | None:
            return None if self.alive else 17

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            self.wait_calls += 1
            return 17

        def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    class Adapter:
        def __init__(self) -> None:
            self.terminated: list[DmonProcessHint] = []

        def terminate_exact(
            self,
            hint: DmonProcessHint,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds > 0
            self.terminated.append(hint)

    process = ExactHandle()
    adapter = Adapter()
    tracked = _ManagedProtocolLaunch(process=process, adapter=adapter)
    _ACTIVE_PROTOCOL_LAUNCHES.append(tracked)
    monkeypatch.setattr(
        sys.modules[__name__],
        "read_process_marker",
        lambda _pid: root_hint.process_marker,
    )

    def capture_nothing_and_exit(_tracked: _ManagedProtocolLaunch) -> None:
        process.alive = False

    monkeypatch.setattr(
        sys.modules[__name__],
        "_capture_supervisor_descendant_hints",
        capture_nothing_and_exit,
    )
    try:
        _cleanup_registered_protocol_launches()

        assert tracked.supervisor_hint == root_hint
        assert tracked.descendant_hints == []
        assert tracked.promoted_from_quarantine is True
        assert adapter.terminated == [root_hint]
        assert process.wait_calls == 1
        assert tracked.cleaned is True
        assert _ACTIVE_PROTOCOL_LAUNCHES == []
    finally:
        process.alive = False
        _ACTIVE_PROTOCOL_LAUNCHES.clear()


def test_quarantine_promotion_survives_transient_wait_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )

    class ExactHandle:
        pid = root_hint.pid

        def __init__(self) -> None:
            self.alive = True
            self.wait_calls = 0
            self.communicate_calls = 0

        def poll(self) -> int | None:
            return None if self.alive else 17

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("direct-daemon", timeout)
            return 17

        def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            return b"retry-out", b"retry-err"

    class Adapter:
        def __init__(self, process: ExactHandle) -> None:
            self.process = process
            self.terminated: list[DmonProcessHint] = []

        def terminate_exact(
            self,
            hint: DmonProcessHint,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds > 0
            self.terminated.append(hint)
            self.process.alive = False

    process = ExactHandle()
    adapter = Adapter(process)
    tracked = _ManagedProtocolLaunch(process=process, adapter=adapter)
    _ACTIVE_PROTOCOL_LAUNCHES.append(tracked)
    monkeypatch.setattr(
        sys.modules[__name__],
        "read_process_marker",
        lambda _pid: root_hint.process_marker,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_capture_supervisor_descendant_hints",
        lambda _tracked: None,
    )
    try:
        with pytest.raises(DmonIdentityError, match="could not prove"):
            _cleanup_registered_protocol_launches()

        assert tracked.supervisor_hint == root_hint
        assert tracked.promoted_from_quarantine is True
        assert tracked.cleaned is False
        assert [tracked] == _ACTIVE_PROTOCOL_LAUNCHES

        _cleanup_registered_protocol_launches()

        assert adapter.terminated == [root_hint, root_hint]
        assert process.wait_calls == 2
        assert process.communicate_calls == 1
        assert tracked.cleaned is True
        assert _ACTIVE_PROTOCOL_LAUNCHES == []
    finally:
        process.alive = False
        _ACTIVE_PROTOCOL_LAUNCHES.clear()


def test_persistent_unreadable_quarantine_blocks_next_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LiveUnhintedHandle:
        pid = 4321

        def __init__(self) -> None:
            self.alive = True
            self.wait_calls = 0
            self.communicate_calls = 0

        def poll(self) -> int | None:
            return None if self.alive else 17

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            self.wait_calls += 1
            return 17

        def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            return b"", b""

    class Adapter:
        def __init__(self) -> None:
            self.terminate_calls = 0

        def terminate_exact(
            self,
            _hint: DmonProcessHint,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds > 0
            self.terminate_calls += 1

    process = LiveUnhintedHandle()
    adapter = Adapter()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        DmonAdapter,
        "create",
        staticmethod(lambda *_args, **_kwargs: adapter),
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "read_process_marker",
        lambda _pid: (_ for _ in ()).throw(PermissionError("still unreadable")),
    )
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    try:
        with pytest.raises(DmonIdentityError, match="marker capture"):
            launch(home)
        quarantined = _ACTIVE_PROTOCOL_LAUNCHES[0]

        with pytest.raises(DmonIdentityError, match="marker capture") as blocked:
            _prepare_protocol_launch_registry_for_setup()

        assert blocked.value.cleanup_unproven is True
        assert [quarantined] == _ACTIVE_PROTOCOL_LAUNCHES
        assert quarantined.cleaned is False
        assert adapter.terminate_calls == 0

        process.alive = False
        _cleanup_registered_protocol_launches()
        assert process.wait_calls == 1
        assert process.communicate_calls == 1
        assert _ACTIVE_PROTOCOL_LAUNCHES == []
    finally:
        process.alive = False
        _ACTIVE_PROTOCOL_LAUNCHES.clear()


def test_readiness_tracking_captures_exact_descendant_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    child_hint = DmonProcessHint(
        pid=8765,
        process_marker="psutil-create-time-us:456000000",
    )

    class Process:
        def __init__(self, pid: int, parent_pid: int, children: list[Process]) -> None:
            self.pid = pid
            self._parent_pid = parent_pid
            self._children = children

        def ppid(self) -> int:
            return self._parent_pid

        def children(self, *, recursive: bool) -> list[Process]:
            assert recursive is False
            return self._children

    child = Process(child_hint.pid, root_hint.pid, [])
    root = Process(root_hint.pid, 1, [child])
    processes = {root.pid: root, child.pid: child}
    markers = {
        root_hint.pid: root_hint.process_marker,
        child_hint.pid: child_hint.process_marker,
    }
    monkeypatch.setattr(psutil, "Process", lambda pid: processes[pid])
    monkeypatch.setattr(
        sys.modules[__name__],
        "read_process_marker",
        lambda pid: markers[pid],
    )
    tracked = _ManagedProtocolLaunch(
        process=object(),
        adapter=object(),
        supervisor_hint=root_hint,
    )

    _capture_supervisor_descendant_hints(tracked)

    assert tracked.descendant_hints == [child_hint]


def test_root_gone_direct_identity_settles_exact_hints_without_daemon() -> None:
    root_hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    child_hint = DmonProcessHint(
        pid=8765,
        process_marker="psutil-create-time-us:456000000",
    )

    class Adapter:
        def __init__(self) -> None:
            self.terminated: list[DmonProcessHint] = []

        def terminate_exact(
            self,
            hint: DmonProcessHint,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds > 0
            self.terminated.append(hint)

    class ExitedHandle:
        pid = root_hint.pid

        def poll(self) -> int:
            return 17

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            return 17

        def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    adapter = Adapter()
    tracked = _ManagedProtocolLaunch(
        process=ExitedHandle(),
        adapter=adapter,
        supervisor_hint=root_hint,
    )
    _ACTIVE_PROTOCOL_LAUNCHES.append(tracked)
    tracked.descendant_hints.append(child_hint)
    try:
        _cleanup_registered_protocol_launches()

        assert tracked.cleaned is True
        assert tracked not in _ACTIVE_PROTOCOL_LAUNCHES
        assert adapter.terminated == [root_hint, child_hint]
    finally:
        tracked.cleaned = True
        if tracked in _ACTIVE_PROTOCOL_LAUNCHES:
            _ACTIVE_PROTOCOL_LAUNCHES.remove(tracked)


def test_failed_registered_cleanup_is_retained_and_retried() -> None:
    root_hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )

    class RetryAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def terminate_exact(
            self,
            _hint: DmonProcessHint,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds > 0
            self.calls += 1
            if self.calls == 1:
                raise DmonIdentityError("transient exact cleanup failure")

    class ExactHandle:
        pid = root_hint.pid

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            return 0

        def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    adapter = RetryAdapter()
    tracked = _ManagedProtocolLaunch(
        process=ExactHandle(),
        adapter=adapter,
        supervisor_hint=root_hint,
    )
    _ACTIVE_PROTOCOL_LAUNCHES.append(tracked)
    try:
        with pytest.raises(DmonIdentityError, match="transient"):
            _cleanup_registered_protocol_launches()
        assert tracked.cleaned is False
        assert tracked in _ACTIVE_PROTOCOL_LAUNCHES

        _cleanup_registered_protocol_launches()

        assert adapter.calls == 2
        assert tracked.cleaned is True
        assert tracked not in _ACTIVE_PROTOCOL_LAUNCHES
    finally:
        tracked.cleaned = True
        if tracked in _ACTIVE_PROTOCOL_LAUNCHES:
            _ACTIVE_PROTOCOL_LAUNCHES.remove(tracked)


def test_registered_launch_cleanup_stops_tree_after_test_body_error(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    _process, _readiness = launch(home)
    tracked = _ACTIVE_PROTOCOL_LAUNCHES[-1]
    assert tracked.daemon_hint is not None

    try:
        raise RuntimeError("simulated test body failure")
    except RuntimeError:
        _cleanup_registered_protocol_launches()

    _assert_exact_process_gone(tracked.supervisor_hint)
    _assert_exact_process_gone(tracked.daemon_hint)


def test_launch_readiness_timeout_stops_direct_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_matches_launched_process",
        lambda **_kwargs: False,
    )
    discarded: list[_ManagedProtocolLaunch] = []
    real_discard = _discard_cleaned_protocol_launches

    def capture_then_discard() -> None:
        discarded.extend(_ACTIVE_PROTOCOL_LAUNCHES)
        real_discard()

    monkeypatch.setattr(
        sys.modules[__name__],
        "_discard_cleaned_protocol_launches",
        capture_then_discard,
    )

    with pytest.raises(pytest.fail.Exception, match="readiness timed out"):
        launch(home, readiness_timeout_seconds=3.0)

    tracked = discarded[-1]
    assert tracked.daemon_hint is not None
    _assert_exact_process_gone(tracked.supervisor_hint)
    _assert_exact_process_gone(tracked.daemon_hint)
    assert _ACTIVE_PROTOCOL_LAUNCHES == []


def test_deadline_exit_without_daemon_hint_settles_direct_root_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )

    class ExitAfterFinalPoll:
        pid = root_hint.pid
        returncode: int | None = None

        def __init__(self) -> None:
            self.poll_results = iter((None, None, None, 17))
            self.wait_calls = 0
            self.communicate_calls = 0

        def poll(self) -> int | None:
            result = next(self.poll_results)
            if result is not None:
                self.returncode = result
            return result

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            self.wait_calls += 1
            self.returncode = 17
            return 17

        def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                return b"deadline-out", b"deadline-err"
            return b"", b""

    class Adapter:
        def __init__(self) -> None:
            self.terminated: list[DmonProcessHint] = []

        def terminate_exact(
            self,
            hint: DmonProcessHint,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds > 0
            self.terminated.append(hint)

    process = ExitAfterFinalPoll()
    adapter = Adapter()
    monotonic_values = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        DmonAdapter,
        "create",
        staticmethod(lambda *_args, **_kwargs: adapter),
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "read_process_marker",
        lambda _pid: root_hint.process_marker,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_capture_supervisor_descendant_hints",
        lambda _tracked: None,
    )
    monkeypatch.setattr(time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def unavailable(*_args, **_kwargs):
        raise OSError("not ready")

    monkeypatch.setattr(DaemonClient, "connect", staticmethod(unavailable))
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    try:
        with pytest.raises(
            pytest.fail.Exception, match="readiness timed out"
        ) as failure:
            launch(home, readiness_timeout_seconds=1.0)

        message = str(failure.value)
        assert "stdout=b'deadline-out'" in message
        assert "stderr=b'deadline-err'" in message
        assert adapter.terminated == [root_hint]
        assert process.wait_calls == 1
        assert process.communicate_calls == 1
        assert _ACTIVE_PROTOCOL_LAUNCHES == []
    finally:
        _ACTIVE_PROTOCOL_LAUNCHES.clear()


def test_launch_cleanup_propagates_ambiguous_identity_without_broad_signal() -> None:
    class AmbiguousAdapter:
        def terminate_exact(
            self,
            _hint: DmonProcessHint,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds > 0
            raise DmonIdentityError("identity is ambiguous")

    class ExactPopenHandle:
        pid = 4321

        def __init__(self) -> None:
            self.wait_calls = 0

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            self.wait_calls += 1
            return 0

    process = ExactPopenHandle()
    tracked = _ManagedProtocolLaunch(
        process=process,
        adapter=AmbiguousAdapter(),
        supervisor_hint=DmonProcessHint(
            pid=process.pid,
            process_marker="psutil-create-time-us:123000000",
        ),
    )

    with pytest.raises(DmonIdentityError, match="ambiguous"):
        tracked.cleanup()

    assert process.wait_calls == 0


def test_subprocess_survives_launcher_return_and_explicit_shutdown_stops_it(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    process, readiness = launch(home)
    first = DaemonClient.connect(home)
    assert first._socket.family == socket.AF_INET
    assert first._socket.getsockname()[0] == "127.0.0.1"
    assert first._socket.getpeername() == (
        first.discovery.endpoint.host,
        first.discovery.endpoint.port,
    )
    health = first.health()
    assert health["instance_nonce"] == readiness["instance_nonce"]
    brain = first.call("brain.create", {"name": "Alice"})
    first.close()

    second = DaemonClient.connect(home)
    assert second.health()["instance_nonce"] == readiness["instance_nonce"]
    status = second.call("daemon.status", {})
    assert brain["brain_id"] in status["brain_ids"]
    stop_process(process, second)


def test_launch_root_is_direct_cpython_daemon_identity(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    process, readiness = launch(home)

    assert readiness["pid"] == process.pid
    assert readiness["process_marker"] == read_process_marker(process.pid)

    client = DaemonClient.connect(home)
    stop_process(process, client)


def test_fragmented_coalesced_oversized_and_invalid_frames_are_recoverable(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    process, _readiness = launch(home)
    record, token = load_discovery_and_credential(home)
    connection = socket.create_connection(
        (record.endpoint.host, record.endpoint.port), timeout=3.0
    )
    connection.settimeout(3.0)
    health = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "health",
            "params": {},
            "auth": token,
        },
        separators=(",", ":"),
    ).encode()
    stream = connection.makefile("rb")
    connection.sendall(health[:7])
    connection.sendall(health[7:] + b"\n")
    assert (
        json.loads(stream.readline())["result"]["instance_nonce"]
        == record.instance_nonce
    )

    initialize = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {
                "protocol_version": PROTOCOL_VERSION,
                "capabilities": CapabilityProfileV1().model_dump(mode="json"),
            },
            "auth": token,
        },
        separators=(",", ":"),
    ).encode()
    status = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "daemon.status",
            "params": {},
            "auth": token,
        },
        separators=(",", ":"),
    ).encode()
    connection.sendall(initialize + b"\n" + status + b"\n")
    assert "result" in json.loads(stream.readline())
    assert "result" in json.loads(stream.readline())

    oversized = b"x" * (ProtocolLimitsV1().max_request_bytes + 1)
    connection.sendall(oversized + b"\n" + status + b"\n")
    assert json.loads(stream.readline())["error"]["code"] == "request_too_large"
    assert "result" in json.loads(stream.readline())
    connection.sendall(b"\xff\n" + status + b"\n")
    assert json.loads(stream.readline())["error"]["code"] == "invalid_request"
    assert "result" in json.loads(stream.readline())
    stream.close()
    connection.close()

    client = DaemonClient.connect(home)
    stop_process(process, client)


def test_second_process_is_excluded_without_changing_live_discovery(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    first_process, first_ready = launch(home)
    second_process, second_ready = launch(home, expect_ready=False)

    assert second_ready["code"] == "runtime_owned"
    assert second_process.wait(timeout=5.0) != 0
    second_process.communicate()
    current, _token = load_discovery_and_credential(home)
    assert current.instance_nonce == first_ready["instance_nonce"]

    client = DaemonClient.connect(home)
    stop_process(first_process, client)


def test_forced_restart_rotates_nonce_and_replays_brains(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    first_process, first_ready = launch(home)
    old_discovery, old_token = load_discovery_and_credential(home)
    old_credential = home / old_discovery.credential_ref
    first_client = DaemonClient.connect(home)
    brain = first_client.call("brain.create", {"name": None})
    first_client.close()
    supervisor_hint, daemon_hint = _force_process_death(first_process)
    _assert_exact_process_gone(supervisor_hint)
    _assert_exact_process_gone(daemon_hint)

    second_process, second_ready = launch(home)
    assert second_ready["instance_nonce"] != first_ready["instance_nonce"]
    current_discovery, current_token = load_discovery_and_credential(home)
    assert current_discovery.instance_nonce == second_ready["instance_nonce"]
    assert current_token != old_token
    assert not old_credential.exists()
    assert list(home.glob("credential-*.key")) == [
        home / current_discovery.credential_ref
    ]
    second_client = DaemonClient.connect(home)
    assert brain["brain_id"] in second_client.call("daemon.status", {})["brain_ids"]
    stop_process(second_process, second_client)


def test_forced_death_recovers_connected_stream_for_exact_resume(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    first_process, first_ready = launch(home, abandonment_grace_seconds=30.0)
    first_client = DaemonClient.connect(home)
    brain = first_client.call("brain.create", {"name": None})
    bridge_instance_id = new_id()
    first_binding = first_client.call(
        "brain.attach",
        {
            "brain_id": brain["brain_id"],
            "bridge_instance_id": bridge_instance_id,
            "recovery_token": RECOVERY_TOKEN,
        },
    )
    first_ack = first_client.call(
        "bridge.commit",
        {
            "binding": first_binding["binding"],
            "record": pre_llm_observation(bridge_instance_id, 1),
        },
    )
    assert first_ack["through_capture_seq"] == 1

    supervisor_hint, daemon_hint = _force_process_death(first_process)
    _assert_exact_process_gone(supervisor_hint)
    _assert_exact_process_gone(daemon_hint)
    first_client.close()

    second_process, second_ready = launch(home, abandonment_grace_seconds=30.0)
    assert second_ready["instance_nonce"] != first_ready["instance_nonce"]
    resumed = DaemonClient.connect(home)
    binding = resumed.call(
        "brain.attach",
        {
            "brain_id": brain["brain_id"],
            "bridge_instance_id": bridge_instance_id,
            "recovery_token": RECOVERY_TOKEN,
        },
    )
    assert binding["next_capture_seq"] == 2
    second_ack = resumed.call(
        "bridge.commit",
        {
            "binding": binding["binding"],
            "record": pre_llm_observation(bridge_instance_id, 2),
        },
    )
    assert second_ack["through_capture_seq"] == 2
    assert second_ack["raw_event_sequence"] > first_ack["last_event_sequence"]
    closed = resumed.call(
        "bridge.close",
        {"binding": binding["binding"], "final_capture_seq": 2},
    )
    assert closed["status"] == "clean_closed"
    stop_process(second_process, resumed)
    with SQLiteLedger.open(home / "runtime.db") as ledger:
        stream = ledger.bridge_stream_state(bridge_instance_id)
        events = [
            event
            for event in ledger.list_events(brain["brain_id"])
            if event.event_type == "hermes.observer.pre_llm_call"
        ]
        assert stream.next_capture_seq == 3
        assert stream.closed_final_seq == 2
        assert [event.payload["capture_seq"] for event in events] == [1, 2]


def test_concurrent_daemon_clients_resolve_one_profile_once_across_restart(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    process, _readiness = launch(home)
    profile = BrainProfileV1(profile_key="hermes.default", name="Alice")
    profile_data = profile.model_dump(mode="json")

    def resolve(_index: int) -> dict[str, object]:
        client = DaemonClient.connect(home)
        try:
            return client.call("brain.resolve", {"profile": profile_data})
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=12) as pool:
        resolved = list(pool.map(resolve, range(24)))

    brain_ids = {item["brain_id"] for item in resolved}
    assert len(brain_ids) == 1
    assert sum(item["created"] is True for item in resolved) == 1
    [brain_id] = brain_ids
    admin = DaemonClient.connect(home)
    assert admin.call("daemon.status", {})["brain_ids"] == [brain_id]
    stop_process(process, admin)

    restarted_process, _readiness = launch(home)
    restarted = DaemonClient.connect(home)
    again = restarted.call("brain.resolve", {"profile": profile_data})
    assert again["brain_id"] == brain_id
    assert again["created"] is False
    assert restarted.call("daemon.status", {})["brain_ids"] == [brain_id]
    stop_process(restarted_process, restarted)


def test_c0_continues_after_all_clients_disconnect(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    process, _ready = launch(home)
    first = DaemonClient.connect(home)
    brain = first.call("brain.create", {"name": None})
    first.close()

    second = DaemonClient.connect(home)
    binding = second.call(
        "brain.attach",
        bridge_attach_params(brain["brain_id"], new_id()),
    )
    deadline = time.monotonic() + 5.0
    while True:
        frame = second.call("state.get", {"binding": binding["binding"]})
        if frame["c0_tick"] >= 1:
            break
        assert time.monotonic() < deadline

    stop_process(process, second)


def test_unclean_eof_is_abandoned_after_finite_grace_with_unknown_gap(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    process, _ready = launch(home, abandonment_grace_seconds=0.12)
    client = DaemonClient.connect(home)
    brain = client.call("brain.create", {"name": None})
    instance = new_id()
    client.call(
        "brain.attach",
        bridge_attach_params(brain["brain_id"], instance),
    )
    client.close()

    deadline = time.monotonic() + 5.0
    status = "open"
    while status == "open" and time.monotonic() < deadline:
        with closing(sqlite3.connect(home / "runtime.db")) as connection:
            row = connection.execute(
                "SELECT status FROM bridge_stream WHERE bridge_instance_id = ?",
                (instance,),
            ).fetchone()
        assert row is not None
        status = row[0]
        if status == "open":
            time.sleep(0.02)

    assert status == "abandoned"
    admin = DaemonClient.connect(home)
    with pytest.raises(DaemonRpcError) as abandoned_attach:
        admin.call(
            "brain.attach",
            bridge_attach_params(brain["brain_id"], instance),
        )
    assert abandoned_attach.value.code == "bridge_abandoned"
    assert abandoned_attach.value.data == {"status": "abandoned"}
    stop_process(process, admin)
    with SQLiteLedger.open(home / "runtime.db") as ledger:
        stream = ledger.bridge_stream_state(instance)
        event = next(
            item
            for item in ledger.list_events(brain["brain_id"])
            if item.event_type == "trace.gap"
        )
        assert stream.disconnected_reason == "grace_abandonment"
        assert event.event_type == "trace.gap"
        assert event.payload["unknown_range"] is True
        assert event.payload["exact"] is False


def test_reconnect_before_grace_remains_open_and_can_cleanly_close(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    process, _ready = launch(home, abandonment_grace_seconds=0.5)
    first = DaemonClient.connect(home)
    brain = first.call("brain.create", {"name": None})
    instance = new_id()
    first.call(
        "brain.attach",
        bridge_attach_params(brain["brain_id"], instance),
    )
    first.close()
    time.sleep(0.05)

    resumed = DaemonClient.connect(home)
    binding = resumed.call(
        "brain.attach",
        bridge_attach_params(brain["brain_id"], instance),
    )
    time.sleep(0.65)
    closed = resumed.call(
        "bridge.close",
        {"binding": binding["binding"], "final_capture_seq": 0},
    )

    assert closed["status"] == "clean_closed"
    assert closed["disconnected_reason"] == "clean_close"
    stop_process(process, resumed)


def test_clean_close_is_never_reclassified_as_grace_abandonment(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    process, _ready = launch(home, abandonment_grace_seconds=0.1)
    client = DaemonClient.connect(home)
    brain = client.call("brain.create", {"name": None})
    instance = new_id()
    binding = client.call(
        "brain.attach",
        bridge_attach_params(brain["brain_id"], instance),
    )
    closed = client.call(
        "bridge.close",
        {"binding": binding["binding"], "final_capture_seq": 0},
    )
    assert closed["disconnected_reason"] == "clean_close"
    client.close()
    time.sleep(0.3)

    admin = DaemonClient.connect(home)
    with pytest.raises(DaemonRpcError) as clean_attach:
        admin.call(
            "brain.attach",
            bridge_attach_params(brain["brain_id"], instance),
        )
    assert clean_attach.value.code == "bridge_clean_closed"
    assert clean_attach.value.data == {"status": "clean_closed"}
    stop_process(process, admin)
    with SQLiteLedger.open(home / "runtime.db") as ledger:
        assert ledger.bridge_stream_state(instance).status == "clean_closed"
        assert all(
            event.event_type != "trace.gap"
            for event in ledger.list_events(brain["brain_id"])
        )


def test_clean_close_receipt_recovers_after_daemon_restart(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    first_process, first_ready = launch(home)
    first = DaemonClient.connect(home)
    brain = first.call("brain.create", {"name": None})
    instance = new_id()
    binding = first.call(
        "brain.attach",
        bridge_attach_params(brain["brain_id"], instance),
    )
    closed = first.call(
        "bridge.close",
        {"binding": binding["binding"], "final_capture_seq": 0},
    )
    stop_process(first_process, first)

    second_process, second_ready = launch(home)
    assert second_ready["instance_nonce"] != first_ready["instance_nonce"]
    recovered_client = DaemonClient.connect(home)
    recovered = recovered_client.call(
        "bridge.close.recover",
        {
            "brain_id": brain["brain_id"],
            "bridge_instance_id": instance,
            "recovery_token": RECOVERY_TOKEN,
            "final_capture_seq": 0,
        },
    )

    assert recovered == closed
    stop_process(second_process, recovered_client)


def test_restart_grants_fresh_grace_after_long_clean_daemon_downtime(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    restart_grace = 0.5
    first_process, _ready = launch(
        home,
        abandonment_grace_seconds=restart_grace,
    )
    first = DaemonClient.connect(home)
    brain = first.call("brain.create", {"name": None})
    instance = new_id()
    first.call(
        "brain.attach",
        bridge_attach_params(brain["brain_id"], instance),
    )
    stop_process(first_process, first)
    time.sleep(restart_grace + 0.2)

    second_process, _ready = launch(
        home,
        abandonment_grace_seconds=restart_grace,
    )
    resumed = DaemonClient.connect(home)
    binding = resumed.call(
        "brain.attach",
        bridge_attach_params(brain["brain_id"], instance),
    )
    closed = resumed.call(
        "bridge.close",
        {"binding": binding["binding"], "final_capture_seq": 0},
    )

    assert closed["status"] == "clean_closed"
    stop_process(second_process, resumed)
