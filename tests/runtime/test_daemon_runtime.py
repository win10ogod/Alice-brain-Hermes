from __future__ import annotations

import asyncio
import fcntl
import gc
import json
import os
import socket
import subprocess
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import pytest

from alice_brain_hermes.core.events import new_event
from alice_brain_hermes.errors import (
    DaemonClientError,
    DaemonRpcError,
    EventConflictError,
    LedgerIntegrityError,
    RuntimeOwnedError,
    SchedulerShutdownError,
)
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.protocol.client import DaemonClient
from alice_brain_hermes.protocol.models import (
    BrainProfileV1,
    DaemonDiscoveryV1,
    LoopbackEndpointV1,
)
from alice_brain_hermes.runtime.daemon import (
    HermesDaemonRuntime,
    PrivateDaemonServer,
    _main,
    _ReadinessSignal,
    _run_daemon,
    _run_private_daemon_loop,
    _write_readiness,
)
from alice_brain_hermes.runtime.discovery import (
    create_credential,
    publish_discovery,
)
from alice_brain_hermes.runtime.engine import ConsciousEngine
from alice_brain_hermes.runtime.lease import RuntimeLease
from alice_brain_hermes.runtime.process_marker import current_process_marker
from alice_brain_hermes.runtime.scheduler import (
    ContinuousScheduler,
    SchedulerHealth,
)
from alice_brain_hermes.runtime.store import SQLiteLedger

RECOVERY_TOKEN = "ab" * 32


def test_runtime_acquires_lease_before_opening_sqlite(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    observed: list[bool] = []

    def ledger_factory(path: Path) -> SQLiteLedger:
        observed.append(path.parent == Path("/proc/self/fd") and path.name.isdecimal())
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)
        credentials = list(home.glob("credential-*.key"))
        assert len(credentials) == 1
        assert credentials[0].stat().st_mode & 0o077 == 0
        return SQLiteLedger.open(path)

    runtime = HermesDaemonRuntime.open(
        home, ledger_factory=ledger_factory, scheduler_interval_seconds=60.0
    )
    try:
        assert observed == [True]
        with pytest.raises(
            PermissionError, match="retained SQLite resources must close"
        ):
            runtime.lease.release()
        assert runtime.ledger._connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("failure_stage", ["write", "fsync", "authority"])
def test_lease_acquire_failure_after_lock_preserves_primary_and_releases_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    real_write = os.write
    real_fsync = os.fsync
    real_assert_authority = RuntimeLease.assert_authority
    real_close = os.close
    failed_descriptor: int | None = None
    stage_injected = False
    close_injected = False

    def fail_write(descriptor: int, payload: bytes) -> int:
        nonlocal failed_descriptor, stage_injected
        if failure_stage == "write" and not stage_injected:
            stage_injected = True
            failed_descriptor = descriptor
            raise OSError("injected write failure")
        return real_write(descriptor, payload)

    def fail_fsync(descriptor: int) -> None:
        nonlocal failed_descriptor, stage_injected
        if failure_stage == "fsync" and not stage_injected:
            stage_injected = True
            failed_descriptor = descriptor
            raise OSError("injected fsync failure")
        real_fsync(descriptor)

    def fail_authority(lease: RuntimeLease) -> Path:
        nonlocal failed_descriptor, stage_injected
        if failure_stage == "authority" and not stage_injected:
            stage_injected = True
            failed_descriptor = lease._descriptor
            raise PermissionError("injected authority failure")
        return real_assert_authority(lease)

    def close_then_fail(descriptor: int) -> None:
        nonlocal close_injected
        if descriptor == failed_descriptor and not close_injected:
            close_injected = True
            real_close(descriptor)
            raise OSError("injected cleanup close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "write", fail_write)
    monkeypatch.setattr(os, "fsync", fail_fsync)
    monkeypatch.setattr(RuntimeLease, "assert_authority", fail_authority)
    monkeypatch.setattr(os, "close", close_then_fail)

    with pytest.raises(
        (OSError, PermissionError), match=f"injected {failure_stage} failure"
    ):
        RuntimeLease.acquire(home)

    assert stage_injected is True
    assert close_injected is True
    with RuntimeLease.acquire(home):
        pass


def test_lease_acquire_unlock_failure_retains_retryable_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    real_flock = fcntl.flock
    real_close = os.close
    locked_descriptor: int | None = None
    unlock_failed = False
    unlock_succeeded = False
    close_before_unlock = False

    def tracked_flock(descriptor: int, operation: int) -> None:
        nonlocal locked_descriptor, unlock_failed, unlock_succeeded
        if operation & fcntl.LOCK_EX:
            locked_descriptor = descriptor
        if operation == fcntl.LOCK_UN and not unlock_failed:
            unlock_failed = True
            raise OSError("injected unlock failure")
        real_flock(descriptor, operation)
        if operation == fcntl.LOCK_UN:
            unlock_succeeded = True

    def guarded_close(descriptor: int) -> None:
        nonlocal close_before_unlock
        if descriptor == locked_descriptor and not unlock_succeeded:
            close_before_unlock = True
        real_close(descriptor)

    monkeypatch.setattr(fcntl, "flock", tracked_flock)
    monkeypatch.setattr(os, "close", guarded_close)
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.lease.current_process_marker",
        lambda: (_ for _ in ()).throw(RuntimeError("injected marker failure")),
    )

    baseline = RuntimeLease.failed_acquire_count()
    with pytest.raises(RuntimeOwnedError, match="cleanup") as captured:
        RuntimeLease.acquire(home)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "marker failure" in str(captured.value.__cause__)
    assert unlock_failed is True
    assert close_before_unlock is False
    assert RuntimeLease.failed_acquire_count() == baseline + 1

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.lease.current_process_marker",
        current_process_marker,
    )
    with RuntimeLease.acquire(home):
        pass
    assert RuntimeLease.failed_acquire_count() == baseline


@pytest.mark.parametrize("operation", ["validate-home", "reopen-home"])
def test_directory_walk_close_error_never_recloses_reused_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    lease = RuntimeLease.acquire(home) if operation == "reopen-home" else None
    real_close = os.close
    replacement_descriptor: int | None = None
    injected = False

    def close_then_reuse_and_fail(descriptor: int) -> None:
        nonlocal injected, replacement_descriptor
        if not injected:
            injected = True
            real_close(descriptor)
            replacement_descriptor = os.open(os.devnull, os.O_RDONLY)
            assert replacement_descriptor == descriptor
            raise OSError("injected directory parent close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", close_then_reuse_and_fail)
    try:
        if operation == "validate-home":
            with pytest.raises(OSError, match="directory parent close failure"):
                RuntimeLease.acquire(home)
        else:
            assert lease is not None
            with pytest.raises(PermissionError) as captured:
                lease.assert_authority()
            assert captured.value.__cause__ is not None
            assert "directory parent close failure" in str(captured.value.__cause__)

        assert injected is True
        assert replacement_descriptor is not None
        os.fstat(replacement_descriptor)
    finally:
        monkeypatch.setattr(os, "close", real_close)
        if lease is not None:
            lease.release()
        if replacement_descriptor is not None:
            real_close(replacement_descriptor)


def test_sqlite_open_uses_retained_home_and_never_mutates_replacement_tree(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    retained = tmp_path / "retained-runtime"
    replacement = tmp_path / "replacement-runtime"
    replacement.mkdir(mode=0o700)

    def swap_home_before_sqlite(path: Path) -> SQLiteLedger:
        assert str(path).startswith("/proc/self/fd/")
        assert path.name.isdecimal()
        home.rename(retained)
        home.symlink_to(replacement, target_is_directory=True)
        return SQLiteLedger.open(path)

    baseline = HermesDaemonRuntime.failed_owner_count()
    with pytest.raises(PermissionError, match="authority"):
        HermesDaemonRuntime.open(
            home,
            ledger_factory=swap_home_before_sqlite,
            scheduler_interval_seconds=60.0,
        )

    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    assert list(replacement.iterdir()) == []
    assert not (replacement / "runtime.db").exists()
    assert not (replacement / "runtime.db-wal").exists()
    assert not (replacement / "runtime.db-shm").exists()
    assert (retained / "runtime.db").is_file()

    home.unlink()
    retained.rename(home)
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    reopened = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    reopened.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux retained-home and abstract path-guard contract",
)
def test_runtime_and_retained_ledger_fail_closed_after_home_replacement(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    displaced = tmp_path / "displaced-runtime"
    replacement = tmp_path / "replacement-runtime"
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    unexpected_second: HermesDaemonRuntime | None = None
    forbidden_brain_id = new_id()

    home.rename(displaced)
    home.mkdir(mode=0o700)
    try:
        with pytest.raises(RuntimeOwnedError, match="already owned"):
            unexpected_second = HermesDaemonRuntime.open(
                home,
                scheduler_interval_seconds=60.0,
            )
        with pytest.raises(PermissionError, match="authority"):
            runtime.create_brain(name="must-not-persist")
        with pytest.raises(
            LedgerIntegrityError, match="mutation seal"
        ) as ledger_failure:
            runtime.ledger.ensure_brain(forbidden_brain_id)
        assert isinstance(ledger_failure.value.__cause__, PermissionError)
        assert "authority" in str(ledger_failure.value.__cause__)
    finally:
        if unexpected_second is not None:
            unexpected_second.close()
        home.rename(replacement)
        displaced.rename(home)
        runtime.close()

    with HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0) as recovered:
        assert recovered.brain_ids == ()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux retained-file identity contract",
)
@pytest.mark.parametrize(
    "candidate",
    ["runtime.db", "runtime.db-wal", "runtime.db-shm"],
)
@pytest.mark.parametrize("mutation", ["path-loss", "replacement"])
def test_live_retained_sqlite_path_mutation_permanently_poison_seal(
    tmp_path: Path,
    candidate: str,
    mutation: str,
) -> None:
    home = tmp_path / "runtime"
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    original = home / candidate
    displaced = tmp_path / f"displaced-{candidate}"
    original.rename(displaced)
    if mutation == "replacement":
        original.write_bytes(b"")
        original.chmod(0o600)

    try:
        with pytest.raises(LedgerIntegrityError, match="mutation seal") as caught:
            runtime.ledger.list_brain_ids()
        assert isinstance(caught.value.__cause__, PermissionError)
    finally:
        if original.exists():
            original.unlink()
        displaced.rename(original)

    try:
        with pytest.raises(LedgerIntegrityError, match="mutation seal"):
            runtime.ledger.list_brain_ids()
    finally:
        runtime.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not hasattr(os, "fork"),
    reason="Linux fork/open-file-description contract",
)
def test_fork_child_cannot_write_close_or_release_inherited_runtime(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    credential_path = runtime.credential.path
    forbidden_brain_ids = (new_id(), new_id())
    engine_brain_id = new_id()
    runtime.ledger.ensure_brain(engine_brain_id)
    inherited_engine = ConsciousEngine(
        runtime.ledger,
        engine_brain_id,
        actor_id=engine_brain_id,
    )
    inherited_scheduler = ContinuousScheduler(
        inherited_engine,
        interval_seconds=60.0,
    )
    child_event = new_event(
        "clock.tick",
        engine_brain_id,
        engine_brain_id,
        {"elapsed_seconds": 1.0},
    )
    read_descriptor, write_descriptor = os.pipe()

    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions are reported over the pipe
        os.close(read_descriptor)
        outcomes: list[str] = []
        actions = (
            lambda: runtime.create_brain(name="child-must-not-persist"),
            lambda: runtime.ledger.ensure_brain(forbidden_brain_ids[0]),
            lambda: runtime.engine_count,
            lambda: runtime.scheduler_count,
            lambda: runtime.closed,
            lambda: inherited_engine.state,
            lambda: inherited_engine.append(child_event),
            lambda: inherited_scheduler.health,
            lambda: inherited_scheduler.stop(timeout=0.0),
            runtime.ledger.close,
            runtime.close,
            runtime.lease.release,
        )
        for action in actions:
            try:
                action()
            except PermissionError:
                outcomes.append("denied")
            except BaseException as error:
                outcomes.append(type(error).__name__)
            else:
                outcomes.append("allowed")
        os.write(write_descriptor, ",".join(outcomes).encode("ascii"))
        os.close(write_descriptor)
        os._exit(0)

    os.close(write_descriptor)
    try:
        child_result = os.read(read_descriptor, 4_096).decode("ascii")
    finally:
        os.close(read_descriptor)
    waited_pid, status = os.waitpid(child_pid, 0)
    assert waited_pid == child_pid
    assert os.waitstatus_to_exitcode(status) == 0
    assert child_result == (
        "denied,denied,denied,denied,denied,denied,denied,denied,"
        "denied,denied,denied,denied"
    )
    assert credential_path.is_file()
    assert runtime.brain_ids == (engine_brain_id,)
    assert inherited_engine.state.last_sequence == 0

    competitor = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from alice_brain_hermes.errors import RuntimeOwnedError; "
                "from alice_brain_hermes.runtime.lease import RuntimeLease; "
                "home=sys.argv[1]; "
                "\ntry: lease=RuntimeLease.acquire(home)"
                "\nexcept RuntimeOwnedError: raise SystemExit(23)"
                "\nelse: lease.release(); raise SystemExit(0)"
            ),
            os.fspath(home),
        ],
        check=False,
        close_fds=True,
        capture_output=True,
        text=True,
    )
    assert competitor.returncode == 23, competitor.stderr

    runtime.close()
    with HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0) as recovered:
        assert recovered.brain_ids == (engine_brain_id,)


@pytest.mark.parametrize(
    "candidate",
    ["runtime.db", "runtime.db-journal", "runtime.db-wal", "runtime.db-shm"],
)
def test_sqlite_symlink_is_rejected_without_mutating_target(
    tmp_path: Path,
    candidate: str,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    outside = tmp_path / "outside.db"
    outside.write_bytes(b"")
    outside.chmod(0o600)
    (home / candidate).symlink_to(outside)
    opened: list[HermesDaemonRuntime] = []

    try:
        with pytest.raises(PermissionError, match="SQLite"):
            opened.append(
                HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
            )
    finally:
        for runtime in opened:
            runtime.close()

    assert outside.read_bytes() == b""
    assert (home / candidate).is_symlink()


@pytest.mark.parametrize(
    "candidate",
    ["runtime.db", "runtime.db-journal", "runtime.db-wal", "runtime.db-shm"],
)
def test_sqlite_hardlink_is_rejected_without_mutating_target(
    tmp_path: Path,
    candidate: str,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    outside = tmp_path / "outside.db"
    outside.write_bytes(b"")
    outside.chmod(0o600)
    os.link(outside, home / candidate)

    with pytest.raises(PermissionError, match="SQLite"):
        HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

    assert outside.read_bytes() == b""
    assert outside.stat().st_nlink == 2


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_scheduler_interval_fails_before_runtime_home_mutation(
    tmp_path: Path,
    interval: float,
) -> None:
    home = tmp_path / "runtime"

    with pytest.raises(ValueError, match="scheduler_interval_seconds"):
        HermesDaemonRuntime.open(home, scheduler_interval_seconds=interval)

    assert not home.exists()


@pytest.mark.parametrize("grace", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_abandonment_grace_fails_before_runtime_home_mutation(
    tmp_path: Path,
    grace: float,
) -> None:
    home = tmp_path / "runtime"

    with pytest.raises(ValueError, match="abandonment_grace_seconds"):
        asyncio.run(
            _run_daemon(
                home,
                readiness=_ReadinessSignal(None),
                scheduler_interval_seconds=60.0,
                abandonment_grace_seconds=grace,
            )
        )

    assert not home.exists()


def test_corrupt_stale_discovery_fails_closed_without_unproven_deletion(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    discovery = home / "daemon.json"
    discovery.write_bytes(b"not-json")
    discovery.chmod(0o600)
    unproven = home / "credential-unproven.key"
    unproven.write_text("a" * 64, encoding="ascii")
    unproven.chmod(0o600)

    with pytest.raises(ValueError, match="discovery"):
        HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

    assert discovery.read_bytes() == b"not-json"
    assert unproven.read_text(encoding="ascii") == "a" * 64
    assert list(home.glob("credential-*.key")) == [unproven]
    with RuntimeLease.acquire(home):
        pass


def test_startup_failure_removes_only_proven_stale_and_current_credentials(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    with RuntimeLease.acquire(home) as stale_lease:
        stale = create_credential(stale_lease)
        publish_discovery(
            stale_lease,
            DaemonDiscoveryV1(
                pid=os.getpid(),
                process_marker=current_process_marker(),
                instance_nonce=stale_lease.instance_nonce,
                endpoint=LoopbackEndpointV1(port=43210),
                credential_ref=stale.path.name,
            ),
        )
    unproven = home / "credential-unproven.key"
    unproven.write_text("b" * 64, encoding="ascii")
    unproven.chmod(0o600)

    def fail_ledger(_path: Path) -> SQLiteLedger:
        raise RuntimeError("injected ledger failure")

    with pytest.raises(RuntimeError, match="injected ledger failure"):
        HermesDaemonRuntime.open(
            home,
            ledger_factory=fail_ledger,
            scheduler_interval_seconds=60.0,
        )

    assert not stale.path.exists()
    assert not (home / "daemon.json").exists()
    assert list(home.glob("credential-*.key")) == [unproven]
    with RuntimeLease.acquire(home):
        pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX hardlink contract")
def test_runtime_lease_rejects_hardlinked_persistent_lock(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    lock = home / "daemon.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    os.link(lock, tmp_path / "lock-hardlink")

    with pytest.raises(PermissionError, match="hardlink"):
        RuntimeLease.acquire(home)

    assert lock.exists()


def test_runtime_lease_write_loops_until_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    real_write = os.write

    def short_write(descriptor: int, payload: bytes) -> int:
        return real_write(descriptor, payload[: max(1, len(payload) // 3)])

    monkeypatch.setattr("alice_brain_hermes.runtime.lease.os.write", short_write)
    with RuntimeLease.acquire(home) as lease:
        body = json.loads((home / "daemon.lock").read_bytes())
        assert body["instance_nonce"] == lease.instance_nonce


def test_authenticated_readiness_rejects_scheduler_without_live_writer(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    database = home / "runtime.db"
    with SQLiteLedger.open(database) as ledger:
        ledger.ensure_brain(new_id())

    class NoopScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        @property
        def health(self) -> SchedulerHealth:
            return SchedulerHealth(
                status="healthy",
                failure_event_persisted=True,
                last_error_type=None,
                running=False,
            )

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=NoopScheduler,
        scheduler_interval_seconds=60.0,
    )
    server = PrivateDaemonServer(runtime)

    async def scenario() -> None:
        record = await server.start()
        try:
            with pytest.raises(RuntimeError, match="readiness"):
                await server.prove_readiness(record)
        finally:
            assert server._server is not None
            server._server.close()
            await server._server.wait_closed()

    try:
        asyncio.run(scenario())
    finally:
        runtime.close()


def test_attach_and_close_recover_after_committed_responses_are_dropped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    real_send = server._send
    instance = new_id()
    brain_id: str | None = None
    dropped_attach_wire: bytes | None = None
    dropped_close_wire: bytes | None = None
    recovered_wire: bytes | None = None
    attach_was_committed = False
    close_was_committed = False

    async def drop_committed_receipts(writer, response: bytes) -> bool:
        nonlocal attach_was_committed, close_was_committed
        nonlocal dropped_attach_wire, dropped_close_wire, recovered_wire
        body = json.loads(response)
        result = body.get("result")
        is_target_attach = (
            isinstance(result, dict)
            and result.get("brain_id") == brain_id
            and set(result) == {"binding", "brain_id", "next_capture_seq"}
        )
        is_target_receipt = (
            isinstance(result, dict)
            and result.get("bridge_instance_id") == instance
            and result.get("status") == "clean_closed"
        )
        if is_target_attach and dropped_attach_wire is None:
            attach_was_committed = (
                runtime.ledger.bridge_stream_state(instance).status == "open"
            )
            dropped_attach_wire = response
            writer.close()
            await writer.wait_closed()
            return False
        if is_target_receipt and dropped_close_wire is None:
            close_was_committed = (
                runtime.ledger.bridge_stream_state(instance).status == "clean_closed"
            )
            dropped_close_wire = response
            writer.close()
            await writer.wait_closed()
            return False
        if is_target_receipt:
            recovered_wire = response
        return await real_send(writer, response)

    monkeypatch.setattr(server, "_send", drop_committed_receipts)

    async def scenario() -> None:
        nonlocal brain_id
        first: DaemonClient | None = None
        attached_client: DaemonClient | None = None
        recovered_client: DaemonClient | None = None
        await server.start()
        try:
            first = await asyncio.to_thread(DaemonClient.connect, home)
            recovery_token = first.new_bridge_recovery_token()
            brain = await asyncio.to_thread(
                first.call,
                "brain.create",
                {"name": None},
            )
            brain_id = brain["brain_id"]  # type: ignore[assignment]
            attach_params = {
                "brain_id": brain_id,
                "bridge_instance_id": instance,
                "recovery_token": recovery_token,
            }
            with pytest.raises(DaemonClientError, match=r"incomplete|transport"):
                await asyncio.to_thread(
                    first.call,
                    "brain.attach",
                    attach_params,
                )
            with pytest.raises(DaemonClientError, match="closed"):
                first.health()
            for _attempt in range(100):
                stream = runtime.ledger.bridge_stream_state(instance)
                if stream.connected_nonce is None:
                    break
                await asyncio.sleep(0.01)
            assert stream.connected_nonce is None

            attached_client = await asyncio.to_thread(DaemonClient.connect, home)
            with pytest.raises(DaemonRpcError) as wrong_open_proof:
                await asyncio.to_thread(
                    attached_client.call,
                    "brain.attach",
                    {
                        **attach_params,
                        "recovery_token": attached_client.new_bridge_recovery_token(),
                    },
                )
            assert wrong_open_proof.value.code == "invalid_binding"
            binding = await asyncio.to_thread(
                attached_client.call,
                "brain.attach",
                attach_params,
            )

            with pytest.raises(DaemonClientError, match=r"incomplete|transport"):
                await asyncio.to_thread(
                    attached_client.call,
                    "bridge.close",
                    {"binding": binding["binding"], "final_capture_seq": 0},
                )
            with pytest.raises(DaemonClientError, match="closed"):
                attached_client.health()

            recovered_client = await asyncio.to_thread(
                DaemonClient.connect,
                home,
            )
            recovered = await asyncio.to_thread(
                recovered_client.call,
                "bridge.close.recover",
                {
                    "brain_id": brain["brain_id"],
                    "bridge_instance_id": instance,
                    "recovery_token": recovery_token,
                    "final_capture_seq": 0,
                },
            )

            assert attach_was_committed is True
            assert dropped_attach_wire is not None
            assert close_was_committed is True
            assert dropped_close_wire is not None
            assert recovered_wire is not None
            dropped_result = json.loads(dropped_close_wire)["result"]
            recovered_result = json.loads(recovered_wire)["result"]
            assert recovered == dropped_result == recovered_result
            assert json.dumps(
                dropped_result, separators=(",", ":"), sort_keys=True
            ) == json.dumps(recovered_result, separators=(",", ":"), sort_keys=True)

            with pytest.raises(DaemonRpcError) as wrong_proof:
                await asyncio.to_thread(
                    recovered_client.call,
                    "bridge.close.recover",
                    {
                        "brain_id": brain["brain_id"],
                        "bridge_instance_id": instance,
                        "recovery_token": recovered_client.new_bridge_recovery_token(),
                        "final_capture_seq": 0,
                    },
                )
            assert wrong_proof.value.code == "invalid_binding"
        finally:
            if first is not None:
                first.close()
            if attached_client is not None:
                attached_client.close()
            if recovered_client is not None:
                recovered_client.close()
            assert server._server is not None
            server._server.close()
            await server._server.wait_closed()
            if server._handlers:
                await asyncio.gather(*tuple(server._handlers))

    try:
        asyncio.run(scenario())
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("family", "sockname", "peername"),
    [
        (socket.AF_UNIX, ("127.0.0.1", 40000), ("127.0.0.1", 50000)),
        (socket.AF_INET, ("0.0.0.0", 40000), ("127.0.0.1", 50000)),
        (socket.AF_INET, ("127.0.0.1", 40000), ("192.0.2.10", 50000)),
    ],
    ids=["non-ipv4", "non-loopback-bound", "non-loopback-peer"],
)
def test_server_rejects_unproven_socket_before_protocol_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: int,
    sockname: tuple[str, int],
    peername: tuple[str, int],
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)

    class FakeAcceptedSocket:
        def __init__(self) -> None:
            self.family = family

    class UnreadReader:
        read_called = False

        async def read(self, _maximum: int) -> bytes:
            self.read_called = True
            return b""

    class RejectedWriter:
        closed = False
        waited = False

        def get_extra_info(self, name: str):
            return {
                "socket": FakeAcceptedSocket(),
                "sockname": sockname,
                "peername": peername,
            }.get(name)

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            self.waited = True

    reader = UnreadReader()
    writer = RejectedWriter()
    protocol_calls = 0

    def forbidden_protocol_state():
        nonlocal protocol_calls
        protocol_calls += 1
        raise AssertionError("protocol state must not be constructed")

    monkeypatch.setattr(server.service, "new_connection", forbidden_protocol_state)
    try:
        asyncio.run(server._handle_client(reader, writer))
        assert writer.closed is True
        assert writer.waited is True
        assert reader.read_called is False
        assert protocol_calls == 0
        assert server.service.shutting_down is False
        assert server._writers == set()
        assert server._handlers == set()
    finally:
        runtime.close()


def test_unauthenticated_idle_client_is_closed_at_absolute_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    server._unauthenticated_idle_timeout_seconds = 0.02

    class FakeAcceptedSocket:
        family = socket.AF_INET

    class IdleReader:
        read_started = False

        async def read(self, _maximum: int) -> bytes:
            self.read_started = True
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class TrackedWriter:
        closed = False
        waited = False

        def get_extra_info(self, name: str):
            return {
                "socket": FakeAcceptedSocket(),
                "sockname": ("127.0.0.1", 40000),
                "peername": ("127.0.0.1", 50000),
            }.get(name)

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            self.waited = True

    class UnauthenticatedConnection:
        authenticated = False
        shutdown_requested = False

        def close(self) -> None:
            return None

    reader = IdleReader()
    writer = TrackedWriter()
    monkeypatch.setattr(server.service, "new_connection", UnauthenticatedConnection)
    try:
        began = time.monotonic()
        asyncio.run(server._handle_client(reader, writer))

        assert time.monotonic() - began < 0.5
        assert reader.read_started is True
        assert writer.closed is True
        assert writer.waited is True
        assert server._active_connection_count == 0
        assert server._writers == set()
        assert server._handlers == set()
        assert server.service.shutting_down is False
    finally:
        runtime.close()


def test_unauthenticated_response_drain_cannot_outlive_auth_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    server._unauthenticated_idle_timeout_seconds = 0.1

    class FakeAcceptedSocket:
        family = socket.AF_INET

    class OneFrameReader:
        sent = False

        async def read(self, _maximum: int) -> bytes:
            if not self.sent:
                self.sent = True
                return b"{}\n"
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class BlockedWriter:
        closed = False
        waited = False
        writes = 0

        def get_extra_info(self, name: str):
            return {
                "socket": FakeAcceptedSocket(),
                "sockname": ("127.0.0.1", 40000),
                "peername": ("127.0.0.1", 50000),
            }.get(name)

        def write(self, _data: bytes) -> None:
            self.writes += 1

        async def drain(self) -> None:
            await asyncio.Event().wait()

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            self.waited = True

    class UnauthenticatedConnection:
        authenticated = False
        shutdown_requested = False

        def handle_frame(self, _frame: bytes) -> bytes:
            return b'{"error":"unauthorized"}'

        def close(self) -> None:
            return None

    writer = BlockedWriter()
    monkeypatch.setattr(server.service, "new_connection", UnauthenticatedConnection)

    async def scenario() -> None:
        # Keep the assertion focused on blocked drain rather than first-use
        # executor thread startup consuming the intentionally small deadline.
        await asyncio.to_thread(lambda: None)
        await server._handle_client(OneFrameReader(), writer)

    try:
        began = time.monotonic()
        asyncio.run(scenario())

        assert time.monotonic() - began < 0.5
        assert writer.writes == 1
        assert writer.closed is True
        assert writer.waited is True
        assert server._active_connection_count == 0
        assert server._writers == set()
        assert server._handlers == set()
        assert server.service.shutting_down is False
    finally:
        runtime.close()


def test_connection_admission_limit_rejects_before_protocol_or_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    server._max_concurrent_connections = 1

    class FakeAcceptedSocket:
        family = socket.AF_INET

    class TrackedWriter:
        def __init__(self, local_port: int) -> None:
            self.local_port = local_port
            self.closed = False
            self.waited = False

        def get_extra_info(self, name: str):
            return {
                "socket": FakeAcceptedSocket(),
                "sockname": ("127.0.0.1", self.local_port),
                "peername": ("127.0.0.1", self.local_port + 1),
            }.get(name)

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            self.waited = True

    class AuthenticatedConnection:
        authenticated = True
        shutdown_requested = False

        def close(self) -> None:
            return None

    protocol_calls = 0

    def connection_factory() -> AuthenticatedConnection:
        nonlocal protocol_calls
        protocol_calls += 1
        return AuthenticatedConnection()

    monkeypatch.setattr(server.service, "new_connection", connection_factory)

    async def scenario() -> tuple[object, object]:
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        class FirstReader:
            async def read(self, _maximum: int) -> bytes:
                first_started.set()
                await release_first.wait()
                return b""

        class ForbiddenReader:
            read_called = False

            async def read(self, _maximum: int) -> bytes:
                self.read_called = True
                return b""

        first_writer = TrackedWriter(40000)
        rejected_writer = TrackedWriter(41000)
        rejected_reader = ForbiddenReader()
        first = asyncio.create_task(server._handle_client(FirstReader(), first_writer))
        await first_started.wait()
        assert server._active_connection_count == 1

        await server._handle_client(rejected_reader, rejected_writer)
        assert rejected_reader.read_called is False
        assert server._active_connection_count == 1

        release_first.set()
        await first
        return rejected_reader, rejected_writer

    try:
        rejected_reader, rejected_writer = asyncio.run(scenario())

        assert protocol_calls == 1
        assert rejected_reader.read_called is False
        assert rejected_writer.closed is True
        assert rejected_writer.waited is True
        assert server._active_connection_count == 0
        assert server._writers == set()
        assert server._handlers == set()
        assert server.service.shutting_down is False
    finally:
        runtime.close()


@pytest.mark.parametrize("failure_stage", ["new_connection", "connection_close"])
def test_client_handler_failure_always_closes_socket_and_registry_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)

    class FakeAcceptedSocket:
        family = socket.AF_INET

    class EofReader:
        async def read(self, _maximum: int) -> bytes:
            return b""

    class TrackedWriter:
        closed = False
        waited = False

        def get_extra_info(self, name: str):
            return {
                "socket": FakeAcceptedSocket(),
                "sockname": ("127.0.0.1", 40000),
                "peername": ("127.0.0.1", 50000),
            }.get(name)

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            self.waited = True

    class FailingCloseConnection:
        shutdown_requested = False

        def close(self) -> None:
            raise RuntimeError("injected disconnect persistence failure")

    if failure_stage == "new_connection":
        monkeypatch.setattr(
            server.service,
            "new_connection",
            lambda: (_ for _ in ()).throw(
                RuntimeError("injected connection construction failure")
            ),
        )
    else:
        monkeypatch.setattr(
            server.service,
            "new_connection",
            lambda: FailingCloseConnection(),
        )
    writer = TrackedWriter()
    try:
        asyncio.run(server._handle_client(EofReader(), writer))

        assert writer.closed is True
        assert writer.waited is True
        assert server._writers == set()
        assert server._handlers == set()
        assert isinstance(server._maintenance_error, RuntimeError)
        assert server.service.shutting_down is True
        assert server._shutdown.is_set()
    finally:
        runtime.close()


def test_invalid_server_constructor_unwinds_open_runtime_for_reuse(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)

    with pytest.raises(ValueError, match="abandonment_grace"):
        asyncio.run(
            _run_daemon(
                home,
                readiness=_ReadinessSignal(None),
                scheduler_interval_seconds=60.0,
                abandonment_grace_seconds=0.0,
            )
        )

    with RuntimeLease.acquire(home):
        pass


def test_run_daemon_does_not_override_transport_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    captured: list[PrivateDaemonServer] = []
    baseline = HermesDaemonRuntime.failed_owner_count()

    async def quarantine(
        server: PrivateDaemonServer, *, readiness: _ReadinessSignal
    ) -> None:
        del readiness
        captured.append(server)
        HermesDaemonRuntime._retain_failed_owner(server)
        raise SchedulerShutdownError("listener shutdown unproven")

    monkeypatch.setattr(PrivateDaemonServer, "run", quarantine)

    with pytest.raises(SchedulerShutdownError, match="listener shutdown"):
        asyncio.run(
            _run_daemon(
                home,
                readiness=_ReadinessSignal(None),
                scheduler_interval_seconds=60.0,
                abandonment_grace_seconds=30.0,
            )
        )

    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)
    captured[0]._transport_quiesced = True
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline


def test_writer_wait_failure_quarantines_runtime_even_after_close_requested(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    baseline = HermesDaemonRuntime.failed_owner_count()

    class UnprovenWriter:
        closed = False

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            raise RuntimeError("injected wait failure")

    writer = UnprovenWriter()
    server._writers.add(writer)  # type: ignore[arg-type]

    failure = asyncio.run(server._cleanup_after_run())

    assert isinstance(failure, RuntimeError)
    assert writer.closed is True
    assert server._transport_quiesced is False
    assert runtime.closed is False
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    server._writers.clear()
    server._transport_quiesced = True
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline


def test_writer_wait_that_never_returns_is_bounded_and_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    baseline = HermesDaemonRuntime.failed_owner_count()
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.daemon._SHUTDOWN_DRAIN_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )

    class StuckWriter:
        closed = False

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            await asyncio.Future()

    writer = StuckWriter()
    server._writers.add(writer)  # type: ignore[arg-type]

    async def cleanup_with_outer_bound() -> BaseException | None:
        return await asyncio.wait_for(server._cleanup_after_run(), timeout=0.2)

    try:
        failure = asyncio.run(cleanup_with_outer_bound())
        assert isinstance(failure, SchedulerShutdownError)
        assert "writer" in str(failure)
        assert writer.closed is True
        assert writer in server._writers
        assert server._transport_quiesced is False
        assert runtime.closed is False
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)
    finally:
        server._writers.clear()
        server._transport_quiesced = True
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            HermesDaemonRuntime.retry_failed_cleanup(home)
        elif not runtime.closed:
            runtime.close()


def test_stuck_writers_share_one_shutdown_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    baseline = HermesDaemonRuntime.failed_owner_count()
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.daemon._SHUTDOWN_DRAIN_TIMEOUT_SECONDS",
        0.02,
    )

    class StuckWriter:
        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            await asyncio.Future()

    writers = [StuckWriter() for _ in range(40)]
    server._writers.update(writers)  # type: ignore[arg-type]

    async def cleanup_with_global_bound() -> BaseException | None:
        return await asyncio.wait_for(server._cleanup_after_run(), timeout=0.5)

    try:
        failure = asyncio.run(cleanup_with_global_bound())
        assert isinstance(failure, SchedulerShutdownError)
        assert set(writers).issubset(server._writers)
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    finally:
        server._writers.clear()
        server._transport_quiesced = True
        if HermesDaemonRuntime.failed_owner_count() > baseline:
            HermesDaemonRuntime.retry_failed_cleanup(home)
        elif not runtime.closed:
            runtime.close()


def test_listener_wait_is_bounded_and_async_retry_recovers_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    baseline = HermesDaemonRuntime.failed_owner_count()
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.daemon._SHUTDOWN_DRAIN_TIMEOUT_SECONDS",
        0.01,
    )

    class RecoverableListener:
        close_calls = 0
        ready = False

        def close(self) -> None:
            self.close_calls += 1

        async def wait_closed(self) -> None:
            if not self.ready:
                await asyncio.Future()

    listener = RecoverableListener()
    server._server = listener  # type: ignore[assignment]

    async def cleanup_with_outer_bound() -> BaseException | None:
        return await asyncio.wait_for(server._cleanup_after_run(), timeout=0.2)

    failure = asyncio.run(cleanup_with_outer_bound())
    assert isinstance(failure, SchedulerShutdownError)
    assert "listener" in str(failure)
    assert listener.close_calls == 1
    assert server._transport_quiesced is False
    assert runtime.closed is False
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    listener.ready = True
    assert asyncio.run(HermesDaemonRuntime.retry_failed_cleanup_async(home)) is True
    assert listener.close_calls == 2
    assert server._transport_quiesced is True
    assert runtime.closed is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    with RuntimeLease.acquire(home):
        pass


def test_handler_wait_failure_retains_writer_until_async_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    baseline = HermesDaemonRuntime.failed_owner_count()

    class FakeAcceptedSocket:
        family = socket.AF_INET

    class EofReader:
        async def read(self, _maximum: int) -> bytes:
            return b""

    class RecoverableWriter:
        closed = False
        wait_calls = 0
        fail_wait = True

        def get_extra_info(self, name: str):
            return {
                "socket": FakeAcceptedSocket(),
                "sockname": ("127.0.0.1", 40000),
                "peername": ("127.0.0.1", 50000),
            }.get(name)

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            self.wait_calls += 1
            if self.fail_wait:
                raise RuntimeError("injected handler wait failure")

    class Connection:
        shutdown_requested = False

        def close(self) -> None:
            return None

    writer = RecoverableWriter()
    monkeypatch.setattr(server.service, "new_connection", Connection)

    asyncio.run(server._handle_client(EofReader(), writer))

    assert writer in server._writers
    assert server._handlers == set()
    assert isinstance(server._maintenance_error, RuntimeError)
    assert runtime.closed is False

    failure = asyncio.run(server._cleanup_after_run())
    assert isinstance(failure, RuntimeError)
    assert writer in server._writers
    assert server._transport_quiesced is False
    assert runtime.closed is False
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    writer.fail_wait = False
    assert asyncio.run(server._cleanup_after_run()) is None
    assert writer not in server._writers
    assert server._transport_quiesced is True
    assert runtime.closed is True
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is False
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    with RuntimeLease.acquire(home):
        pass


def test_transport_wrapper_preserves_runtime_cleanup_failure_during_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    baseline = HermesDaemonRuntime.failed_owner_count()

    class RecoverableWriter:
        fail = True

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            if self.fail:
                raise RuntimeError("writer-primary")

    writer = RecoverableWriter()
    server._writers.add(writer)  # type: ignore[arg-type]
    first = asyncio.run(server._cleanup_after_run())
    assert isinstance(first, RuntimeError)
    assert str(first) == "writer-primary"
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1

    real_stop = runtime._stop_all_schedulers
    fail_runtime_once = True

    def stop_schedulers() -> None:
        nonlocal fail_runtime_once
        if fail_runtime_once:
            fail_runtime_once = False
            raise RuntimeError("scheduler-primary")
        real_stop()

    monkeypatch.setattr(runtime, "_stop_all_schedulers", stop_schedulers)
    writer.fail = False

    with pytest.raises(RuntimeError, match="scheduler-primary"):
        asyncio.run(HermesDaemonRuntime.retry_failed_cleanup_async(home))
    assert runtime.closed is False
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1

    assert asyncio.run(HermesDaemonRuntime.retry_failed_cleanup_async(home)) is True
    assert runtime.closed is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline


def test_transport_wrapper_promotes_its_exact_retained_runtime_owner(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    baseline = HermesDaemonRuntime.failed_owner_count()

    HermesDaemonRuntime._retain_failed_owner(runtime)
    HermesDaemonRuntime._retain_failed_owner(server)
    with HermesDaemonRuntime._failed_owner_lock:
        assert HermesDaemonRuntime._failed_owners[home] is server

    server._transport_quiesced = True
    server.close()
    assert runtime.closed is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline


def test_async_cleanup_discards_exact_owner_reinserted_by_admitted_refail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    baseline = HermesDaemonRuntime.failed_owner_count()
    close_entered = threading.Event()
    real_close = runtime.close

    def observed_close() -> None:
        close_entered.set()
        real_close()

    monkeypatch.setattr(runtime, "close", observed_close)
    HermesDaemonRuntime._retain_failed_owner(server)
    runtime._begin_operation()

    async def scenario() -> None:
        retry = asyncio.create_task(
            HermesDaemonRuntime.retry_failed_cleanup_async(home)
        )
        assert await asyncio.to_thread(close_entered.wait, 2.0)
        with HermesDaemonRuntime._failed_owner_lock:
            assert home not in HermesDaemonRuntime._failed_owners
        try:
            runtime._mark_fail_stopped(
                RuntimeError("injected admitted operation refail")
            )
            with HermesDaemonRuntime._failed_owner_lock:
                assert HermesDaemonRuntime._failed_owners[home] is server
        finally:
            runtime._end_operation()
        assert await asyncio.wait_for(retry, timeout=2.0) is True

    asyncio.run(scenario())

    assert server.closed is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    with RuntimeLease.acquire(home):
        pass


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_daemon_run_invokes_uninterruptible_cleanup_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    record = DaemonDiscoveryV1(
        pid=os.getpid(),
        process_marker=runtime.lease.process_marker,
        instance_nonce=runtime.lease.instance_nonce,
        endpoint=LoopbackEndpointV1(port=1),
        credential_ref=runtime.credential.path.name,
    )
    cleanup_calls = 0

    async def maintenance() -> None:
        server._maintenance_ready.set()
        await server._maintenance_enabled.wait()
        server._maintenance_first_pass.set()

    async def start() -> DaemonDiscoveryV1:
        return record

    async def prove_readiness(_record: DaemonDiscoveryV1) -> None:
        return None

    async def cleanup() -> BaseException | None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_fails:
            return RuntimeError("injected cleanup failure")
        return None

    monkeypatch.setattr(server, "_maintain_abandoned_streams", maintenance)
    monkeypatch.setattr(server, "start", start)
    monkeypatch.setattr(server, "prove_readiness", prove_readiness)
    monkeypatch.setattr(server, "_cleanup_uninterruptibly", cleanup)
    server._shutdown.set()

    try:
        if cleanup_fails:
            with pytest.raises(RuntimeError, match="cleanup failure"):
                asyncio.run(server.run(readiness=_ReadinessSignal(None)))
        else:
            asyncio.run(server.run(readiness=_ReadinessSignal(None)))
        assert cleanup_calls == 1
    finally:
        runtime.close()


def test_daemon_refreshes_restart_grace_after_first_pass_before_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    server = PrivateDaemonServer(runtime)
    record = DaemonDiscoveryV1(
        pid=os.getpid(),
        process_marker=runtime.lease.process_marker,
        instance_nonce=runtime.lease.instance_nonce,
        endpoint=LoopbackEndpointV1(port=1),
        credential_ref=runtime.credential.path.name,
    )
    events: list[str] = []
    readiness = _ReadinessSignal(None)

    async def maintenance() -> None:
        server._maintenance_ready.set()
        await server._maintenance_enabled.wait()
        events.append("first_pass")
        server._maintenance_first_pass.set()

    async def start() -> DaemonDiscoveryV1:
        return record

    async def prove_readiness(_record: DaemonDiscoveryV1) -> None:
        return None

    async def cleanup() -> BaseException | None:
        return None

    def refresh() -> int:
        events.append("refresh")
        return 0

    def ready(_body: dict[str, object]) -> None:
        events.append("ready")

    monkeypatch.setattr(server, "_maintain_abandoned_streams", maintenance)
    monkeypatch.setattr(server, "start", start)
    monkeypatch.setattr(server, "prove_readiness", prove_readiness)
    monkeypatch.setattr(server, "_cleanup_uninterruptibly", cleanup)
    monkeypatch.setattr(runtime.ledger, "refresh_daemon_restart_grace", refresh)
    monkeypatch.setattr(readiness, "write_once", ready)
    server._shutdown.set()

    try:
        asyncio.run(server.run(readiness=readiness))
        assert events == ["refresh", "first_pass", "refresh", "ready"]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "failure_stage",
    ["bind", "publish", "discovery_readback", "readiness_write"],
)
def test_daemon_startup_failure_unwinds_listener_writer_files_and_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    brain_id = new_id()
    with SQLiteLedger.open(home / "runtime.db") as ledger:
        ledger.ensure_brain(brain_id)
    schedulers: list[TrackingScheduler] = []

    class TrackingScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine
            self.running = False
            self.stop_calls = 0
            schedulers.append(self)

        def start(self) -> None:
            self.running = True

        def stop(self) -> None:
            self.stop_calls += 1
            self.running = False

        @property
        def health(self) -> SchedulerHealth:
            return SchedulerHealth(
                status="healthy",
                failure_event_persisted=True,
                last_error_type=None,
                running=self.running,
            )

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=TrackingScheduler,
        scheduler_interval_seconds=60.0,
    )
    server = PrivateDaemonServer(runtime)
    readiness = _ReadinessSignal(None)
    if failure_stage == "bind":

        async def fail_bind(*_args, **_kwargs):
            raise OSError("injected bind failure")

        monkeypatch.setattr(asyncio, "start_server", fail_bind)
    elif failure_stage == "publish":
        monkeypatch.setattr(
            "alice_brain_hermes.runtime.daemon.publish_discovery",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected publish failure")
            ),
        )
    elif failure_stage == "discovery_readback":

        async def fail_readback(_record: DaemonDiscoveryV1) -> None:
            raise OSError("injected discovery readback failure")

        monkeypatch.setattr(server, "prove_readiness", fail_readback)
    else:
        monkeypatch.setattr(
            readiness,
            "write_once",
            lambda _body: (_ for _ in ()).throw(
                OSError("injected readiness write failure")
            ),
        )

    with pytest.raises(OSError, match="injected"):
        asyncio.run(server.run(readiness=readiness))

    assert runtime.closed is True
    assert schedulers[0].running is False
    assert schedulers[0].stop_calls == 1
    assert server._writers == set()
    assert server._handlers == set()
    if server._server is not None:
        assert server._server.is_serving() is False
    assert not (home / "daemon.json").exists()
    assert list(home.glob("credential-*.key")) == []
    with RuntimeLease.acquire(home):
        pass
    reopened = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    reopened.close()


def test_readiness_writer_handles_short_writes_and_closes_on_zero_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = os.write
    read_fd, write_fd = os.pipe()

    def short_write(descriptor: int, payload: bytes) -> int:
        return real_write(descriptor, payload[: max(1, len(payload) // 3)])

    monkeypatch.setattr("alice_brain_hermes.runtime.daemon.os.write", short_write)
    _write_readiness(write_fd, {"ready": True})
    assert json.loads(os.read(read_fd, 4_096)) == {"ready": True}
    os.close(read_fd)

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.daemon.os.write",
        lambda _descriptor, _payload: 0,
    )
    with pytest.raises(OSError, match="no progress"):
        _write_readiness(write_fd, {"ready": True})
    with pytest.raises(OSError):
        os.fstat(write_fd)
    os.close(read_fd)


def test_late_daemon_failure_cannot_reuse_consumed_readiness_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_fd, write_fd = os.pipe()
    unrelated_read_fd, unrelated_write_fd = os.pipe()

    def late_failure(
        _runtime_home: str | Path,
        *,
        readiness_fd: int | None,
        scheduler_interval_seconds: float,
        abandonment_grace_seconds: float,
        _readiness_signal: _ReadinessSignal | None = None,
    ) -> None:
        assert readiness_fd == write_fd
        assert scheduler_interval_seconds == 60.0
        assert abandonment_grace_seconds == 30.0
        assert _readiness_signal is not None
        assert _readiness_signal.write_once({"ready": True}) is True
        os.dup2(unrelated_write_fd, write_fd)
        raise RuntimeError("late shutdown failure")

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.daemon.run_private_daemon", late_failure
    )
    try:
        assert (
            _main(
                [
                    "--runtime-home",
                    str(tmp_path),
                    "--readiness-fd",
                    str(write_fd),
                    "--scheduler-interval",
                    "60.0",
                ]
            )
            == 1
        )
        assert json.loads(os.read(read_fd, 4_096)) == {"ready": True}
        os.fstat(write_fd)
        os.set_blocking(unrelated_read_fd, False)
        with pytest.raises(BlockingIOError):
            os.read(unrelated_read_fd, 1)
    finally:
        for descriptor in (
            read_fd,
            write_fd,
            unrelated_read_fd,
            unrelated_write_fd,
        ):
            with suppress(OSError):
                os.close(descriptor)


def test_fail_stop_loop_does_not_join_a_stuck_executor_worker() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_worker() -> None:
        started.set()
        release.wait()

    async def fail_after_worker_starts() -> None:
        worker = asyncio.create_task(asyncio.to_thread(blocked_worker))
        while not started.is_set():
            await asyncio.sleep(0)
        # Retain the task until cancellation proves that loop teardown does not
        # wait for the underlying, non-cooperative worker thread.
        assert not worker.done()
        raise SchedulerShutdownError("transport shutdown unproven")

    began = time.monotonic()
    try:
        with pytest.raises(SchedulerShutdownError, match="transport shutdown unproven"):
            _run_private_daemon_loop(fail_after_worker_starts())
        assert time.monotonic() - began < 0.5
    finally:
        release.set()


def test_main_hard_exits_on_unproven_transport_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_fd, write_fd = os.pipe()
    exit_codes: list[int] = []

    class HardExit(BaseException):
        pass

    def fail_stop(*_args, **_kwargs) -> None:
        raise SchedulerShutdownError("transport shutdown unproven")

    def hard_exit(code: int) -> None:
        exit_codes.append(code)
        raise HardExit

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.daemon.run_private_daemon", fail_stop
    )
    monkeypatch.setattr("alice_brain_hermes.runtime.daemon.os._exit", hard_exit)
    try:
        with pytest.raises(HardExit):
            _main(
                [
                    "--runtime-home",
                    str(tmp_path),
                    "--readiness-fd",
                    str(write_fd),
                ]
            )
        assert exit_codes == [3]
        assert json.loads(os.read(read_fd, 4_096)) == {
            "ready": False,
            "code": "shutdown_unproven",
        }
    finally:
        for descriptor in (read_fd, write_fd):
            with suppress(OSError):
                os.close(descriptor)


def test_concurrent_engine_attach_uses_one_once_cell_and_scheduler(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    brain_id = new_id()
    runtime.ledger.ensure_brain(brain_id)
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            engines = list(pool.map(lambda _index: runtime.engine(brain_id), range(64)))

        assert all(engine is engines[0] for engine in engines)
        assert runtime.engine_count == 1
        assert runtime.scheduler_count == 1
        assert runtime.scheduler(brain_id).health.running is True
    finally:
        runtime.close()


def test_startup_replays_every_persisted_brain_before_runtime_is_ready(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    database = home / "runtime.db"
    brains = [new_id(), new_id()]
    with SQLiteLedger.open(database) as ledger:
        for brain_id in brains:
            ledger.ensure_brain(brain_id)

    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    try:
        assert runtime.brain_ids == tuple(sorted(brains))
        assert runtime.engine_count == runtime.scheduler_count == 2
        assert all(runtime.scheduler(item).health.running for item in brains)
    finally:
        runtime.close()


def test_runtime_close_releases_lease_only_after_all_writers_exit(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    runtime.create_brain(name=None)

    runtime.close()

    assert runtime.closed is True
    with RuntimeLease.acquire(home):
        pass


def test_brain_foundation_commits_before_scheduler_can_write(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)

    class EagerScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine
            self.interval_seconds = interval_seconds
            self.started = False

        def start(self) -> None:
            self.engine.pulse(0.01)
            self.started = True

        def stop(self) -> None:
            self.started = False

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=EagerScheduler,
        scheduler_interval_seconds=60.0,
    )
    try:
        engine = runtime.create_brain(name="Alice")
        events = runtime.ledger.list_events(engine.brain_id)

        assert events[0].event_type == "brain.created"
        assert events[0].sequence == 1
        assert engine.state.name == "Alice"
    finally:
        runtime.close()


def test_failed_foundation_insert_leaves_no_empty_brain_row(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    with runtime.ledger._transaction(immediate=True):
        runtime.ledger._connection.execute(
            "CREATE TRIGGER reject_foundation BEFORE INSERT ON events "
            "BEGIN SELECT RAISE(ABORT, 'reject foundation'); END"
        )
    try:
        with pytest.raises(Exception, match="reject foundation"):
            runtime.create_brain(name=None)

        assert runtime.ledger.list_brain_ids() == []
        assert runtime.engine_count == runtime.scheduler_count == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("operation", ["create", "first-resolve"])
def test_scheduler_start_failure_compensates_unpublished_foundation(
    tmp_path: Path, operation: str
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    stop_calls = 0

    class StartFailureScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine

        def start(self) -> None:
            raise RuntimeError("injected dynamic scheduler start failure")

        def stop(self) -> None:
            nonlocal stop_calls
            stop_calls += 1

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=StartFailureScheduler,
        scheduler_interval_seconds=60.0,
    )
    profile = BrainProfileV1(profile_key="compensation.test", name=None)
    try:
        with pytest.raises(RuntimeError, match="dynamic scheduler start"):
            if operation == "create":
                runtime.create_brain(name=None)
            else:
                runtime.resolve_brain(profile)

        assert stop_calls == 1
        assert runtime.ledger.list_brain_ids() == []
        assert (
            runtime.ledger._connection.execute(
                "SELECT COUNT(*) FROM events"
            ).fetchone()[0]
            == 0
        )
        assert (
            runtime.ledger._connection.execute(
                "SELECT COUNT(*) FROM brain_profile"
            ).fetchone()[0]
            == 0
        )
        assert runtime.engine_count == runtime.scheduler_count == 0
        assert runtime.readiness_snapshot()["runtime_ready"] is True
    finally:
        runtime.close()


def test_scheduler_write_before_start_failure_forces_retained_fail_stop(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)

    class DirtyStartScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine

        def start(self) -> None:
            self.engine.pulse(0.01)
            raise RuntimeError("injected dirty scheduler start failure")

        def stop(self) -> None:
            return None

    baseline = HermesDaemonRuntime.failed_owner_count()
    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=DirtyStartScheduler,
        scheduler_interval_seconds=60.0,
    )
    try:
        with pytest.raises(RuntimeError, match="dirty scheduler start"):
            runtime.create_brain(name=None)

        assert len(runtime.ledger.list_brain_ids()) == 1
        [brain_id] = runtime.ledger.list_brain_ids()
        assert len(runtime.ledger.list_events(brain_id)) > 1
        assert runtime.fail_stopped is True
        assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
        with pytest.raises(RuntimeError, match="closing"):
            runtime.create_brain(name=None)
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(home)
        runtime.close()
        assert HermesDaemonRuntime.failed_owner_count() == baseline
        assert runtime.closed is True
        with HermesDaemonRuntime.open(
            home, scheduler_interval_seconds=60.0
        ) as restarted:
            assert restarted.brain_ids == (brain_id,)
            assert restarted.ledger.list_events(brain_id)[-1].sequence > 1
    finally:
        if not runtime.closed:
            runtime.close()


def test_inflight_engine_cannot_publish_after_another_creation_fail_stops(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    first_entered = threading.Event()
    second_entered = threading.Event()
    allow_second_publication = threading.Event()
    construction_count = 0
    construction_lock = threading.Lock()

    class RacingScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            nonlocal construction_count
            self.engine = engine
            with construction_lock:
                construction_count += 1
                self.index = construction_count

        def start(self) -> None:
            if self.index == 1:
                first_entered.set()
                assert second_entered.wait(2.0)
                self.engine.pulse(0.01)
                raise RuntimeError("injected racing dirty start failure")
            second_entered.set()
            assert allow_second_publication.wait(2.0)

        def stop(self) -> None:
            return None

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=RacingScheduler,
        scheduler_interval_seconds=60.0,
    )
    successes: list[ConsciousEngine] = []
    failures: list[BaseException] = []

    def create() -> None:
        try:
            successes.append(runtime.create_brain(name=None))
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=create)
    second = threading.Thread(target=create)
    first.start()
    assert first_entered.wait(2.0)
    second.start()
    assert second_entered.wait(2.0)
    first.join(2.0)
    try:
        assert first.is_alive() is False
        assert runtime.fail_stopped is True
        allow_second_publication.set()
        second.join(2.0)
        assert second.is_alive() is False
        assert successes == []
        assert len(failures) == 2
        assert runtime.engine_count == runtime.scheduler_count == 0
        assert len(runtime.ledger.list_brain_ids()) == 2
    finally:
        allow_second_publication.set()
        first.join(2.0)
        second.join(2.0)
        runtime.close()


def test_snapshot_before_start_failure_blocks_compensation_and_replays(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)

    class SnapshotThenFailScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine

        def start(self) -> None:
            self.engine.ledger.save_snapshot(self.engine.state)
            raise RuntimeError("injected post-snapshot start failure")

        def stop(self) -> None:
            return None

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=SnapshotThenFailScheduler,
        scheduler_interval_seconds=60.0,
    )
    try:
        with pytest.raises(RuntimeError, match="post-snapshot"):
            runtime.create_brain(name="Alice")

        [brain_id] = runtime.ledger.list_brain_ids()
        assert runtime.fail_stopped is True
        assert (
            runtime.ledger._connection.execute(
                "SELECT COUNT(*) FROM snapshots WHERE brain_id = ?", (brain_id,)
            ).fetchone()[0]
            == 1
        )
        runtime.close()
        with HermesDaemonRuntime.open(
            home, scheduler_interval_seconds=60.0
        ) as restarted:
            assert restarted.engine(brain_id).state.name == "Alice"
    finally:
        if not runtime.closed:
            runtime.close()


def test_late_snapshot_and_stale_engine_write_reject_compensated_brain(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    captured: list[ConsciousEngine] = []

    class CaptureThenFailScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine
            captured.append(engine)

        def start(self) -> None:
            raise RuntimeError("injected clean start failure")

        def stop(self) -> None:
            return None

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=CaptureThenFailScheduler,
        scheduler_interval_seconds=60.0,
    )
    try:
        with pytest.raises(RuntimeError, match="clean start failure"):
            runtime.create_brain(name=None)

        assert runtime.ledger.list_brain_ids() == []
        stale_state = captured[0].state
        with pytest.raises(KeyError):
            runtime.ledger.save_snapshot(stale_state)
        assert runtime.ledger.bootstrap_state(stale_state.brain_id).last_sequence == 0
        with pytest.raises(EventConflictError, match="sequence divergence"):
            captured[0].append(
                new_event(
                    "clock.tick",
                    stale_state.brain_id,
                    stale_state.brain_id,
                    {"elapsed_seconds": 1.0},
                )
            )
        assert runtime.ledger.list_brain_ids() == []
        assert (
            runtime.ledger._connection.execute(
                "SELECT COUNT(*) FROM snapshots"
            ).fetchone()[0]
            == 0
        )
    finally:
        runtime.close()


def test_dynamic_start_stop_failure_never_attempts_compensation(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)

    class UnprovenStopScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine
            self.stop_calls = 0

        def start(self) -> None:
            raise RuntimeError("injected partial dynamic start")

        def stop(self) -> None:
            self.stop_calls += 1
            if self.stop_calls <= 2:
                raise SchedulerShutdownError("dynamic writer still alive")

    baseline = HermesDaemonRuntime.failed_owner_count()
    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=UnprovenStopScheduler,
        scheduler_interval_seconds=60.0,
    )
    with pytest.raises(SchedulerShutdownError, match="writer still alive"):
        runtime.create_brain(name=None)

    assert len(runtime.ledger.list_brain_ids()) == 1
    assert runtime.scheduler_count == 1
    assert runtime.fail_stopped is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    with pytest.raises(SchedulerShutdownError, match="writer still alive"):
        runtime.close()
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline


def test_unproven_dynamic_stop_retains_last_runtime_reference_for_retry(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)

    class RetainedStopScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine
            self.stop_calls = 0

        def start(self) -> None:
            raise RuntimeError("injected retained partial start")

        def stop(self) -> None:
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise SchedulerShutdownError("retained writer still alive")

    baseline = HermesDaemonRuntime.failed_owner_count()
    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=RetainedStopScheduler,
        scheduler_interval_seconds=60.0,
    )
    runtime_ref = weakref.ref(runtime)
    with pytest.raises(SchedulerShutdownError, match="writer still alive"):
        runtime.create_brain(name=None)
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1

    del runtime
    gc.collect()
    assert runtime_ref() is not None
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    assert runtime_ref() is None or runtime_ref().closed is True


def test_dirty_fail_stop_retains_last_runtime_reference_for_retry(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)

    class RetainedDirtyScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine

        def start(self) -> None:
            self.engine.pulse(0.01)
            raise RuntimeError("injected retained dirty start")

        def stop(self) -> None:
            return None

    baseline = HermesDaemonRuntime.failed_owner_count()
    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=RetainedDirtyScheduler,
        scheduler_interval_seconds=60.0,
    )
    runtime_ref = weakref.ref(runtime)
    with pytest.raises(RuntimeError, match="retained dirty start"):
        runtime.create_brain(name=None)
    [brain_id] = runtime.ledger.list_brain_ids()
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1

    del runtime
    gc.collect()
    assert runtime_ref() is not None
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    with HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0) as restarted:
        assert restarted.brain_ids == (brain_id,)


def test_live_server_fail_stop_retains_outer_owner_until_transport_drains(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)

    class StopFailsOnceScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine
            self.stop_calls = 0

        def start(self) -> None:
            raise RuntimeError("injected live-server partial start")

        def stop(self) -> None:
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise SchedulerShutdownError("live-server writer still alive")

    baseline = HermesDaemonRuntime.failed_owner_count()
    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=StopFailsOnceScheduler,
        scheduler_interval_seconds=60.0,
    )
    server = PrivateDaemonServer(runtime)

    async def scenario() -> None:
        await server.start()
        assert server._server is not None
        client = await asyncio.to_thread(DaemonClient.connect, home)
        try:
            with pytest.raises(DaemonClientError) as caught:
                await asyncio.to_thread(client.call, "brain.create", {"name": None})
            assert not isinstance(caught.value, DaemonRpcError)
        finally:
            client.close()

        with HermesDaemonRuntime._failed_owner_lock:
            assert HermesDaemonRuntime._failed_owners[home] is server
        assert server._server.is_serving() is True
        with pytest.raises(SchedulerShutdownError, match="transport authority"):
            await asyncio.to_thread(HermesDaemonRuntime.retry_failed_cleanup, home)
        assert runtime.closed is False
        assert server._server.is_serving() is True
        assert await HermesDaemonRuntime.retry_failed_cleanup_async(home) is True

    asyncio.run(scenario())

    assert server._transport_quiesced is True
    assert runtime.closed is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline


def test_server_fail_stop_closes_transport_and_run_raises(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)

    class DirtyStartScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine

        def start(self) -> None:
            self.engine.pulse(0.01)
            raise RuntimeError("injected server dirty start failure")

        def stop(self) -> None:
            return None

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=DirtyStartScheduler,
        scheduler_interval_seconds=60.0,
    )
    server = PrivateDaemonServer(runtime)

    async def scenario() -> None:
        run_task = asyncio.create_task(server.run(readiness=_ReadinessSignal(None)))
        client: DaemonClient | None = None
        try:
            await asyncio.wait_for(
                server._maintenance_periodic_enabled.wait(), timeout=2.0
            )
            client = await asyncio.to_thread(DaemonClient.connect, home)
            with pytest.raises(DaemonClientError) as caught:
                await asyncio.to_thread(client.call, "brain.create", {"name": None})
            assert not isinstance(caught.value, DaemonRpcError)
            with pytest.raises(RuntimeError, match="compensation"):
                await asyncio.wait_for(run_task, timeout=2.0)
        finally:
            if client is not None:
                client.close()
            if not run_task.done():
                server.service.begin_shutdown()
                server._shutdown.set()
                with suppress(BaseException):
                    await run_task

    asyncio.run(scenario())

    assert runtime.closed is True
    assert not (home / "daemon.json").exists()
    with HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0) as restarted:
        assert len(restarted.brain_ids) == 1


def test_dynamic_foundation_reservation_blocks_engine_publication_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    scheduler_count = 0

    class CountingScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            nonlocal scheduler_count
            self.engine = engine
            scheduler_count += 1

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=CountingScheduler,
        scheduler_interval_seconds=60.0,
    )
    foundation_persisted = threading.Event()
    allow_publication = threading.Event()
    real_create = runtime.ledger.create_brain_foundation

    def blocked_create(brain_id: str, *, name: str | None):
        foundation = real_create(brain_id, name=name)
        foundation_persisted.set()
        assert allow_publication.wait(2.0)
        return foundation

    monkeypatch.setattr(runtime.ledger, "create_brain_foundation", blocked_create)
    created: list[ConsciousEngine] = []
    observed: list[ConsciousEngine] = []
    creator = threading.Thread(
        target=lambda: created.append(runtime.create_brain(name=None))
    )
    creator.start()
    assert foundation_persisted.wait(2.0)
    [brain_id] = runtime.ledger.list_brain_ids()
    observer = threading.Thread(
        target=lambda: observed.append(runtime.engine(brain_id))
    )
    observer.start()
    try:
        observer.join(0.05)
        assert observer.is_alive() is True
        allow_publication.set()
        creator.join(2.0)
        observer.join(2.0)
        assert creator.is_alive() is observer.is_alive() is False
        assert created == observed
        assert scheduler_count == 1
    finally:
        allow_publication.set()
        creator.join(2.0)
        observer.join(2.0)
        runtime.close()


def test_close_waits_for_inflight_engine_publication_before_snapshot(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    start_entered = threading.Event()
    allow_start = threading.Event()
    scheduler_stopped = threading.Event()

    class BlockingScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine

        def start(self) -> None:
            start_entered.set()
            assert allow_start.wait(2.0)

        def stop(self) -> None:
            scheduler_stopped.set()

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=BlockingScheduler,
        scheduler_interval_seconds=60.0,
    )
    brain_id = new_id()
    runtime.ledger.ensure_brain(brain_id)
    engine_done = threading.Event()
    close_done = threading.Event()

    def attach() -> None:
        runtime.engine(brain_id)
        engine_done.set()

    def close() -> None:
        runtime.close()
        close_done.set()

    creator = threading.Thread(target=attach)
    creator.start()
    assert start_entered.wait(2.0)
    closer = threading.Thread(target=close)
    closer.start()
    assert close_done.wait(0.05) is False

    allow_start.set()
    creator.join(2.0)
    closer.join(2.0)

    assert engine_done.is_set()
    assert close_done.is_set()
    assert scheduler_stopped.is_set()
    assert runtime.closed is True


def test_stable_profile_resolve_is_atomic_concurrent_and_restart_safe(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    profile = BrainProfileV1(profile_key="hermes.default", name="Alice")
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            resolved = list(
                pool.map(lambda _index: runtime.resolve_brain(profile), range(64))
            )
        brain_ids = {item[0].brain_id for item in resolved}
        assert len(brain_ids) == 1
        assert sum(item[1] for item in resolved) == 1
        [brain_id] = brain_ids
        assert runtime.ledger.list_events(brain_id)[0].event_type == "brain.created"
        assert runtime.ledger.list_brain_ids() == [brain_id]
    finally:
        runtime.close()

    restarted = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    try:
        engine, created = restarted.resolve_brain(profile)
        assert engine.brain_id == brain_id
        assert created is False
    finally:
        restarted.close()


def test_first_profile_foundation_reservation_blocks_resolve_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    scheduler_count = 0

    class CountingScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            nonlocal scheduler_count
            self.engine = engine
            scheduler_count += 1

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=CountingScheduler,
        scheduler_interval_seconds=60.0,
    )
    profile = BrainProfileV1(profile_key="reservation.profile", name="Alice")
    profile_persisted = threading.Event()
    allow_receipt = threading.Event()
    real_resolve = runtime.ledger.resolve_brain_profile

    def blocked_resolve(requested: BrainProfileV1, *, new_brain_id: str | None = None):
        result = real_resolve(requested, new_brain_id=new_brain_id)
        if result.created:
            profile_persisted.set()
            assert allow_receipt.wait(2.0)
        return result

    monkeypatch.setattr(runtime.ledger, "resolve_brain_profile", blocked_resolve)
    created: list[tuple[ConsciousEngine, bool]] = []
    observed: list[tuple[ConsciousEngine, bool]] = []
    creator = threading.Thread(
        target=lambda: created.append(runtime.resolve_brain(profile))
    )
    observer = threading.Thread(
        target=lambda: observed.append(runtime.resolve_brain(profile))
    )
    creator.start()
    assert profile_persisted.wait(2.0)
    observer.start()
    try:
        observer.join(0.05)
        assert observer.is_alive() is True
        allow_receipt.set()
        creator.join(2.0)
        observer.join(2.0)
        assert creator.is_alive() is observer.is_alive() is False
        assert created[0][0] is observed[0][0]
        assert {created[0][1], observed[0][1]} == {False, True}
        assert scheduler_count == 1
    finally:
        allow_receipt.set()
        creator.join(2.0)
        observer.join(2.0)
        runtime.close()


def test_concurrent_first_profile_start_failure_uses_one_cell_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    profile = BrainProfileV1(profile_key="failure.profile", name=None)
    start_entered = threading.Event()
    allow_failure = threading.Event()
    all_resolved = threading.Event()
    scheduler_starts = 0
    resolve_calls = 0
    worker_count = 8

    class FailOnceScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine

        def start(self) -> None:
            nonlocal scheduler_starts
            scheduler_starts += 1
            if scheduler_starts == 1:
                start_entered.set()
                assert allow_failure.wait(2.0)
                raise RuntimeError("injected shared profile start failure")

        def stop(self) -> None:
            return None

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=FailOnceScheduler,
        scheduler_interval_seconds=60.0,
    )
    real_resolve = runtime.ledger.resolve_brain_profile

    def counted_resolve(requested: BrainProfileV1, *, new_brain_id: str | None = None):
        nonlocal resolve_calls
        result = real_resolve(requested, new_brain_id=new_brain_id)
        resolve_calls += 1
        if resolve_calls == worker_count:
            all_resolved.set()
        return result

    monkeypatch.setattr(runtime.ledger, "resolve_brain_profile", counted_resolve)
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(runtime.resolve_brain, profile)
                for _index in range(worker_count)
            ]
            assert start_entered.wait(2.0)
            assert all_resolved.wait(2.0)
            allow_failure.set()
            failures = [future.exception(timeout=2.0) for future in futures]

        assert all(
            isinstance(error, RuntimeError)
            and "shared profile start failure" in str(error)
            for error in failures
        )
        assert scheduler_starts == 1
        assert runtime.ledger.list_brain_ids() == []
        assert runtime.fail_stopped is False

        engine, created = runtime.resolve_brain(profile)
        assert created is True
        assert engine.brain_id in runtime.ledger.list_brain_ids()
        assert scheduler_starts == 2
    finally:
        allow_failure.set()
        runtime.close()


def test_failed_profile_mapping_insert_rolls_back_foundation(tmp_path: Path) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    with runtime.ledger._transaction(immediate=True):
        runtime.ledger._connection.execute(
            "CREATE TRIGGER reject_profile BEFORE INSERT ON brain_profile "
            "BEGIN SELECT RAISE(ABORT, 'reject profile'); END"
        )
    try:
        with pytest.raises(Exception, match="reject profile"):
            runtime.resolve_brain(
                BrainProfileV1(profile_key="hermes.default", name=None)
            )
        assert runtime.ledger.list_brain_ids() == []
        assert runtime.engine_count == 0
    finally:
        runtime.close()


def test_partially_started_scheduler_is_stopped_when_start_raises(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    stop_called = threading.Event()

    class PartialScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine

        def start(self) -> None:
            raise RuntimeError("partial start")

        def stop(self) -> None:
            stop_called.set()

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=PartialScheduler,
        scheduler_interval_seconds=60.0,
    )
    brain_id = new_id()
    runtime.ledger.ensure_brain(brain_id)
    try:
        with pytest.raises(RuntimeError, match="partial start"):
            runtime.engine(brain_id)
        assert stop_called.is_set()
        assert runtime.engine_count == runtime.scheduler_count == 0
    finally:
        runtime.close()


def test_scheduler_cleanup_failure_retains_ledger_and_lease_until_retry(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)

    class RetryableCleanupScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine
            self.stop_calls = 0

        def start(self) -> None:
            raise RuntimeError("partial start")

        def stop(self) -> None:
            self.stop_calls += 1
            if self.stop_calls <= 2:
                raise SchedulerShutdownError("writer still alive")

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=RetryableCleanupScheduler,
        scheduler_interval_seconds=60.0,
    )
    brain_id = new_id()
    runtime.ledger.ensure_brain(brain_id)

    with pytest.raises(SchedulerShutdownError, match="writer still alive"):
        runtime.engine(brain_id)
    assert runtime.scheduler_count == 1
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    with pytest.raises(SchedulerShutdownError, match="writer still alive"):
        runtime.close()
    assert runtime.ledger.list_brain_ids() == [brain_id]
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    runtime.close()
    assert runtime.closed is True
    with RuntimeLease.acquire(home):
        pass


def test_runtime_close_attempts_every_scheduler_and_retries_only_unproven(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    schedulers: list[MultiStopScheduler] = []

    class MultiStopScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine
            self.stop_calls = 0
            self.fail_once = not schedulers
            schedulers.append(self)

        def start(self) -> None:
            return None

        def stop(self) -> None:
            self.stop_calls += 1
            if self.fail_once and self.stop_calls == 1:
                raise SchedulerShutdownError("first writer still alive")

        @property
        def health(self) -> SchedulerHealth:
            return SchedulerHealth(
                status="healthy",
                failure_event_persisted=True,
                last_error_type=None,
                running=True,
            )

    runtime = HermesDaemonRuntime.open(
        home,
        scheduler_factory=MultiStopScheduler,
        scheduler_interval_seconds=60.0,
    )
    runtime.create_brain(name=None)
    runtime.create_brain(name=None)

    with pytest.raises(SchedulerShutdownError, match="first writer"):
        runtime.close()

    assert [scheduler.stop_calls for scheduler in schedulers] == [1, 1]
    assert runtime.ledger.list_brain_ids()
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    runtime.close()
    assert [scheduler.stop_calls for scheduler in schedulers] == [2, 1]
    with RuntimeLease.acquire(home):
        pass


@pytest.mark.parametrize(
    "failure_stage", ["ledger", "discovery", "credential", "lease"]
)
def test_runtime_cleanup_stage_failure_quarantines_until_exact_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    runtime.create_brain(name=None)
    baseline = HermesDaemonRuntime.failed_owner_count()
    calls = 0

    if failure_stage == "ledger":
        real_cleanup = runtime.ledger.close

        def cleanup() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected ledger cleanup failure")
            real_cleanup()

        monkeypatch.setattr(runtime.ledger, "close", cleanup)
    elif failure_stage == "discovery":
        from alice_brain_hermes.runtime import daemon as daemon_module

        real_cleanup = daemon_module.cleanup_discovery

        def cleanup(authority: RuntimeLease) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected discovery cleanup failure")
            real_cleanup(authority)

        monkeypatch.setattr(daemon_module, "cleanup_discovery", cleanup)
    elif failure_stage == "credential":
        from alice_brain_hermes.runtime import daemon as daemon_module

        real_cleanup = daemon_module.cleanup_credential

        def cleanup(authority: RuntimeLease, credential) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected credential cleanup failure")
            real_cleanup(authority, credential)

        monkeypatch.setattr(daemon_module, "cleanup_credential", cleanup)
    else:
        real_cleanup = runtime.lease.release

        def cleanup() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected lease cleanup failure")
            real_cleanup()

        monkeypatch.setattr(runtime.lease, "release", cleanup)

    with pytest.raises(OSError, match=failure_stage):
        runtime.close()

    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)
    with pytest.raises(RuntimeOwnedError):
        HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)

    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    reopened = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    reopened.close()


def test_partial_start_cleanup_failure_is_quarantined_until_explicit_retry(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    brain_id = new_id()
    with SQLiteLedger.open(home / "runtime.db") as ledger:
        ledger.ensure_brain(brain_id)
    scheduler_refs: list[weakref.ReferenceType[object]] = []

    class PartialStartScheduler:
        def __init__(self, engine, *, interval_seconds: float) -> None:
            self.engine = engine
            self.stop_calls = 0
            scheduler_refs.append(weakref.ref(self))

        def start(self) -> None:
            raise RuntimeError("injected partial scheduler start")

        def stop(self) -> None:
            self.stop_calls += 1
            if self.stop_calls <= 2:
                raise SchedulerShutdownError("injected writer still alive")

    baseline = HermesDaemonRuntime.failed_owner_count()
    with pytest.raises(SchedulerShutdownError, match="writer still alive"):
        HermesDaemonRuntime.open(
            home,
            scheduler_factory=PartialStartScheduler,
            scheduler_interval_seconds=60.0,
        )

    gc.collect()
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    assert scheduler_refs[0]() is not None
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is False
    with RuntimeLease.acquire(home):
        pass


def test_preruntime_cleanup_retry_removes_exact_quarantined_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    real_unlink = RuntimeLease.unlink_home_file
    unlink_calls = 0

    def fail_once(lease: RuntimeLease, name: str, *, missing_ok: bool = False) -> None:
        nonlocal unlink_calls
        unlink_calls += 1
        if unlink_calls == 1:
            raise OSError("injected credential cleanup failure")
        real_unlink(lease, name, missing_ok=missing_ok)

    def fail_ledger(_path: Path) -> SQLiteLedger:
        raise RuntimeError("injected ledger startup failure")

    monkeypatch.setattr(RuntimeLease, "unlink_home_file", fail_once)
    baseline = HermesDaemonRuntime.failed_owner_count()

    with pytest.raises(RuntimeError, match="ledger startup failure") as raised:
        HermesDaemonRuntime.open(
            home,
            ledger_factory=fail_ledger,
            scheduler_interval_seconds=60.0,
        )
    assert isinstance(raised.value.__cause__, OSError)
    assert "credential cleanup failure" in str(raised.value.__cause__)

    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is False
    with RuntimeLease.acquire(home):
        pass


def test_preruntime_cleanup_quarantines_swapped_credential_content(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    baseline = HermesDaemonRuntime.failed_owner_count()
    swapped_path: Path | None = None

    def swap_then_fail(_path: Path) -> SQLiteLedger:
        nonlocal swapped_path
        [swapped_path] = list(home.glob("credential-*.key"))
        swapped_path.write_text("f" * 64, encoding="ascii")
        swapped_path.chmod(0o600)
        raise RuntimeError("injected ledger startup failure")

    with pytest.raises(RuntimeError, match="ledger startup failure") as raised:
        HermesDaemonRuntime.open(
            home,
            ledger_factory=swap_then_fail,
            scheduler_interval_seconds=60.0,
        )
    assert isinstance(raised.value.__cause__, PermissionError)
    assert "content" in str(raised.value.__cause__)

    assert swapped_path is not None
    assert swapped_path.read_text(encoding="ascii") == "f" * 64
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    with HermesDaemonRuntime._failed_owner_lock:
        owner = HermesDaemonRuntime._failed_owners[home]
    expected = owner.credential  # type: ignore[attr-defined]
    assert expected is not None
    swapped_path.write_text(expected.token, encoding="ascii")
    swapped_path.chmod(0o600)
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline


def test_unlock_failure_quarantines_live_lease_until_exact_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    baseline = HermesDaemonRuntime.failed_owner_count()
    real_flock = fcntl.flock
    unlock_calls = 0

    def fail_explicit_unlock(descriptor: int, operation: int) -> None:
        nonlocal unlock_calls
        if operation & fcntl.LOCK_UN:
            unlock_calls += 1
            if unlock_calls == 1:
                raise OSError("injected explicit unlock failure")
        real_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", fail_explicit_unlock)

    with pytest.raises(OSError, match="explicit unlock failure"):
        runtime.close()

    assert runtime.closed is False
    assert runtime.lease.released is False
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert unlock_calls == 2
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    with RuntimeLease.acquire(home):
        pass


def test_post_unlock_close_error_is_terminal_and_never_reuses_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    baseline = HermesDaemonRuntime.failed_owner_count()
    lock_descriptor = runtime.lease._descriptor
    real_close = os.close
    injected = False

    def close_then_fail(descriptor: int) -> None:
        nonlocal injected
        if descriptor == lock_descriptor and not injected:
            injected = True
            real_close(descriptor)
            raise OSError("injected post-unlock close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", close_then_fail)

    with pytest.raises(OSError, match="post-unlock close failure"):
        runtime.close()

    assert injected is True
    assert runtime.closed is True
    assert runtime.lease.released is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    with RuntimeLease.acquire(home):
        pass

    reused: list[int] = []
    try:
        for _ in range(64):
            descriptor = os.open(os.devnull, os.O_RDONLY)
            reused.append(descriptor)
            if descriptor == lock_descriptor:
                break
        assert lock_descriptor in reused

        runtime.close()
        os.fstat(lock_descriptor)
    finally:
        for descriptor in reused:
            real_close(descriptor)


def test_runtime_cleanup_quarantines_same_length_swapped_credential_and_discovery(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    record = DaemonDiscoveryV1(
        pid=os.getpid(),
        process_marker=runtime.lease.process_marker,
        instance_nonce=runtime.lease.instance_nonce,
        endpoint=LoopbackEndpointV1(port=1),
        credential_ref=runtime.credential.path.name,
    )
    discovery_path = publish_discovery(runtime.lease, record)
    original_discovery = discovery_path.read_bytes()
    expected_token = runtime.credential.token
    replacement_prefix = "0" if expected_token[0] != "0" else "1"
    swapped_token = replacement_prefix + expected_token[1:]
    runtime.credential.path.write_text(swapped_token, encoding="ascii")
    runtime.credential.path.chmod(0o600)
    baseline = HermesDaemonRuntime.failed_owner_count()

    with pytest.raises(PermissionError, match="content"):
        runtime.close()

    assert runtime.credential.path.read_text(encoding="ascii") == swapped_token
    assert discovery_path.read_bytes() == original_discovery
    assert HermesDaemonRuntime.failed_owner_count() == baseline + 1
    with pytest.raises(RuntimeOwnedError):
        RuntimeLease.acquire(home)

    runtime.credential.path.write_text(expected_token, encoding="ascii")
    runtime.credential.path.chmod(0o600)
    assert HermesDaemonRuntime.retry_failed_cleanup(home) is True
    assert HermesDaemonRuntime.failed_owner_count() == baseline
    assert not runtime.credential.path.exists()
    assert not discovery_path.exists()
    with RuntimeLease.acquire(home):
        pass


def test_new_lease_recovers_prior_process_connection_markers_conservatively(
    tmp_path: Path,
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    first = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    brain_id = new_id()
    instance = new_id()
    first.ledger.ensure_brain(brain_id)
    first.engine(brain_id)
    first.ledger.attach_bridge_stream(
        instance,
        brain_id=brain_id,
        server_actor_id=brain_id,
        server_adapter_id="alice-brain-hermes-observer-v1",
        connected_nonce="dead-process-connection",
        recovery_token=RECOVERY_TOKEN,
    )
    first.close()

    restarted = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    try:
        stream = restarted.ledger.bridge_stream_state(instance)
        assert stream.connected_nonce is None
        assert stream.disconnected_reason == "daemon_restart"
        assert stream.disconnected_at is not None
        resumed = restarted.ledger.attach_bridge_stream(
            instance,
            brain_id=brain_id,
            server_actor_id=brain_id,
            server_adapter_id="alice-brain-hermes-observer-v1",
            connected_nonce="new-process-connection",
            recovery_token=RECOVERY_TOKEN,
        )
        assert resumed.next_capture_seq == 1
        assert resumed.disconnected_reason is None
    finally:
        restarted.close()


@pytest.mark.parametrize(
    "tamper_sql",
    [
        "UPDATE brain_profile SET profile_fingerprint = '0'",
        "UPDATE brain_profile SET profile_json = "
        "replace(profile_json, 'Alice', 'Mallory')",
        "UPDATE brain_profile SET profile_key = 'tampered.key'",
    ],
)
def test_profile_row_tampering_is_integrity_failure_not_body_conflict(
    tmp_path: Path, tamper_sql: str
) -> None:
    home = tmp_path / "runtime"
    home.mkdir(mode=0o700)
    profile = BrainProfileV1(profile_key="hermes.default", name="Alice")
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    try:
        runtime.resolve_brain(profile)
        runtime.ledger._connection.execute(tamper_sql)

        with pytest.raises(LedgerIntegrityError):
            runtime.resolve_brain(profile)
    finally:
        runtime.close()


def test_failed_owner_retry_claim_never_deletes_a_newer_owner(
    tmp_path: Path,
) -> None:
    home = (tmp_path / "runtime").absolute()
    old_released = threading.Event()
    allow_old_return = threading.Event()

    class Owner:
        def __init__(self, *, pause_after_release: bool) -> None:
            self.runtime_home = home
            self.closed = False
            self.pause_after_release = pause_after_release

        def close(self) -> None:
            self.closed = True
            if self.pause_after_release:
                old_released.set()
                assert allow_old_return.wait(timeout=5.0)

    old = Owner(pause_after_release=True)
    newer = Owner(pause_after_release=False)
    failures: list[BaseException] = []
    results: list[bool] = []
    HermesDaemonRuntime._retain_failed_owner(old)

    def retry() -> None:
        try:
            results.append(HermesDaemonRuntime.retry_failed_cleanup(home))
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=retry)
    worker.start()
    assert old_released.wait(timeout=5.0)
    try:
        HermesDaemonRuntime._retain_failed_owner(newer)
    finally:
        allow_old_return.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()

    try:
        assert failures == []
        assert results == [True]
        with HermesDaemonRuntime._failed_owner_lock:
            assert HermesDaemonRuntime._failed_owners.get(home) is newer
    finally:
        with HermesDaemonRuntime._failed_owner_lock:
            if HermesDaemonRuntime._failed_owners.get(home) is newer:
                del HermesDaemonRuntime._failed_owners[home]


def test_runtime_direct_close_claims_failed_entry_before_native_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = (tmp_path / "runtime").absolute()
    runtime = HermesDaemonRuntime.open(home, scheduler_interval_seconds=60.0)
    real_release = runtime.lease.release
    native_released = threading.Event()
    allow_close_return = threading.Event()
    failures: list[BaseException] = []

    class NewerOwner:
        runtime_home = home
        closed = False

        def close(self) -> None:
            self.closed = True

    newer = NewerOwner()

    def release_then_pause() -> None:
        real_release()
        native_released.set()
        assert allow_close_return.wait(timeout=5.0)

    def close() -> None:
        try:
            runtime.close()
        except BaseException as error:
            failures.append(error)

    HermesDaemonRuntime._retain_failed_owner(runtime)
    monkeypatch.setattr(runtime.lease, "release", release_then_pause)
    worker = threading.Thread(target=close)
    worker.start()
    assert native_released.wait(timeout=5.0)
    try:
        HermesDaemonRuntime._retain_failed_owner(newer)
    finally:
        allow_close_return.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()

    try:
        assert failures == []
        assert runtime.closed is True
        with HermesDaemonRuntime._failed_owner_lock:
            assert HermesDaemonRuntime._failed_owners.get(home) is newer
    finally:
        with HermesDaemonRuntime._failed_owner_lock:
            if HermesDaemonRuntime._failed_owners.get(home) is newer:
                del HermesDaemonRuntime._failed_owners[home]


def test_async_failed_owner_retry_defers_cancellation_until_close_finishes(
    tmp_path: Path,
) -> None:
    home = (tmp_path / "runtime").absolute()
    close_entered = threading.Event()
    allow_close = threading.Event()

    class Owner:
        runtime_home = home

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            close_entered.set()
            assert allow_close.wait(timeout=5.0)
            self.closed = True

    owner = Owner()
    HermesDaemonRuntime._retain_failed_owner(owner)

    async def scenario() -> None:
        retry = asyncio.create_task(
            HermesDaemonRuntime.retry_failed_cleanup_async(home)
        )
        assert await asyncio.to_thread(close_entered.wait, 5.0)
        retry.cancel()
        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await retry

    try:
        asyncio.run(scenario())
        assert owner.closed is True
        with HermesDaemonRuntime._failed_owner_lock:
            assert home not in HermesDaemonRuntime._failed_owners
    finally:
        allow_close.set()
        with HermesDaemonRuntime._failed_owner_lock:
            if HermesDaemonRuntime._failed_owners.get(home) is owner:
                del HermesDaemonRuntime._failed_owners[home]


def test_async_failed_owner_retry_restores_owner_on_internal_cancellation(
    tmp_path: Path,
) -> None:
    home = (tmp_path / "runtime").absolute()

    class Owner:
        runtime_home = home
        closed = False

        def close(self) -> None:
            raise asyncio.CancelledError

    owner = Owner()
    HermesDaemonRuntime._retain_failed_owner(owner)
    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(HermesDaemonRuntime.retry_failed_cleanup_async(home))
        with HermesDaemonRuntime._failed_owner_lock:
            assert HermesDaemonRuntime._failed_owners.get(home) is owner
    finally:
        with HermesDaemonRuntime._failed_owner_lock:
            if HermesDaemonRuntime._failed_owners.get(home) is owner:
                del HermesDaemonRuntime._failed_owners[home]
