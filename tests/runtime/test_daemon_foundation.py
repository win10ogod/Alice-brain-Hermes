from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from alice_brain_hermes.errors import RuntimeOwnedError
from alice_brain_hermes.ids import new_id
from alice_brain_hermes.runtime.lease import RuntimeLease
from alice_brain_hermes.runtime.process_marker import (
    _parse_linux_start_ticks,
    read_process_marker,
)
from alice_brain_hermes.runtime.scheduler import ContinuousScheduler
from alice_brain_hermes.runtime.store import SQLiteLedger


def test_list_brain_ids_is_public_sorted_and_complete(tmp_path: Path) -> None:
    first = new_id()
    second = new_id()
    with SQLiteLedger.open(tmp_path / "runtime.db") as ledger:
        ledger.ensure_brain(second)
        ledger.ensure_brain(first)

        assert ledger.list_brain_ids() == sorted((first, second))


def test_linux_process_marker_parser_handles_closing_parenthesis_in_comm() -> None:
    fields_after_comm = ["S", *(["0"] * 18), "123456"]
    payload = f"321 (worker ) name) {' '.join(fields_after_comm)}"

    assert _parse_linux_start_ticks(payload, expected_pid=321) == 123456


def test_process_marker_rejects_unstable_double_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = os.getpid()
    boot = b"12345678-1234-1234-1234-123456789abc\n"

    def process_stat(ticks: int) -> bytes:
        fields = ["S", *(["0"] * 18), str(ticks)]
        return f"{pid} (worker) {' '.join(fields)}".encode()

    responses = iter((boot, process_stat(100), boot, process_stat(101)))
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.process_marker._read_bounded_nofollow",
        lambda _path, *, maximum: next(responses),
    )

    with pytest.raises(PermissionError, match="changed"):
        read_process_marker(pid)


def test_scheduler_join_must_prove_writer_exit() -> None:
    scheduler = object.__new__(ContinuousScheduler)
    scheduler._creator_pid = os.getpid()
    scheduler._stop = threading.Event()
    scheduler._thread = threading.current_thread()

    with pytest.raises(RuntimeError, match="current scheduler thread"):
        scheduler.join(timeout=0.01)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode/lock contract")
def test_runtime_lease_is_private_held_and_second_owner_fails(tmp_path: Path) -> None:
    runtime_home = tmp_path / "home"
    runtime_home.mkdir(mode=0o700)

    with RuntimeLease.acquire(runtime_home) as first:
        assert first.path.name == "daemon.lock"
        assert first.instance_nonce
        assert first.path.stat().st_mode & 0o077 == 0
        with pytest.raises(RuntimeOwnedError):
            RuntimeLease.acquire(runtime_home)

    with RuntimeLease.acquire(runtime_home) as recovered:
        assert recovered.instance_nonce != first.instance_nonce


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux abstract AF_UNIX path-guard contract",
)
def test_runtime_path_guard_survives_home_rename_and_replacement(
    tmp_path: Path,
) -> None:
    runtime_home = tmp_path / "home"
    runtime_home.mkdir(mode=0o700)
    displaced_home = tmp_path / "displaced-home"
    replacement_home = tmp_path / "replacement-home"
    unexpected_second: RuntimeLease | None = None
    lease = RuntimeLease.acquire(runtime_home)

    runtime_home.rename(displaced_home)
    runtime_home.mkdir(mode=0o700)
    try:
        with pytest.raises(RuntimeOwnedError, match="already owned"):
            unexpected_second = RuntimeLease.acquire(runtime_home)

        with pytest.raises(PermissionError, match="authority"):
            lease.open_home_file(
                "must-not-exist",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        assert not (displaced_home / "must-not-exist").exists()
        assert not (runtime_home / "must-not-exist").exists()
    finally:
        if unexpected_second is not None:
            unexpected_second.release()
        runtime_home.rename(replacement_home)
        displaced_home.rename(runtime_home)
        lease.release()

    with RuntimeLease.acquire(runtime_home):
        pass


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not hasattr(os, "fork"),
    reason="Linux fork/open-file-description contract",
)
def test_fork_child_cannot_release_parent_lease_or_adopt_its_authority(
    tmp_path: Path,
) -> None:
    runtime_home = tmp_path / "home"
    runtime_home.mkdir(mode=0o700)
    lease = RuntimeLease.acquire(runtime_home)
    read_descriptor, write_descriptor = os.pipe()
    child_release_descriptor, parent_release_descriptor = os.pipe()

    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - assertions are reported over the pipe
        os.close(read_descriptor)
        os.close(parent_release_descriptor)
        outcomes: list[str] = []
        for action in (
            lease.release,
            lambda: lease.open_home_file(
                "child-must-not-write",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            ),
        ):
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
        os.read(child_release_descriptor, 1)
        os.close(child_release_descriptor)
        os._exit(0)

    os.close(write_descriptor)
    os.close(child_release_descriptor)
    try:
        child_result = os.read(read_descriptor, 1_024).decode("ascii")
    finally:
        os.close(read_descriptor)
    try:
        assert child_result == "denied,denied"
        assert not (runtime_home / "child-must-not-write").exists()

        competitor = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from alice_brain_hermes.errors import RuntimeOwnedError; "
                    "from alice_brain_hermes.runtime.lease import RuntimeLease; "
                    "home=sys.argv[1]; "
                    "\ntry: lease=RuntimeLease.acquire(home)"
                    "\nexcept RuntimeOwnedError: raise SystemExit(23)"
                    "\nelse: lease.release(); raise SystemExit(0)"
                ),
                os.fspath(runtime_home),
            ],
            check=False,
            close_fds=True,
            capture_output=True,
            text=True,
        )
        assert competitor.returncode == 23, competitor.stderr

        lease.release()
        recovered = subprocess.run(
            competitor.args,
            check=False,
            close_fds=True,
            capture_output=True,
            text=True,
        )
        assert recovered.returncode == 0, recovered.stderr
    finally:
        if not lease.released:
            lease.release()
        os.write(parent_release_descriptor, b"x")
        os.close(parent_release_descriptor)
        waited_pid, status = os.waitpid(child_pid, 0)
    assert waited_pid == child_pid
    assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_runtime_lease_rejects_non_private_home(tmp_path: Path) -> None:
    runtime_home = tmp_path / "home"
    runtime_home.mkdir(mode=0o755)

    with pytest.raises(PermissionError, match="0700"):
        RuntimeLease.acquire(runtime_home)


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow contract")
def test_runtime_home_symlink_ancestor_is_rejected_before_creation(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    link = tmp_path / "runtime-link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(PermissionError, match="symlink"):
        RuntimeLease.acquire(link / "new-home")

    assert not (outside / "new-home").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX filesystem contract")
@pytest.mark.parametrize("filesystem_type", ["nfs", None])
def test_runtime_home_rejects_unreliable_or_unverifiable_filesystem_before_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filesystem_type: str | None,
) -> None:
    runtime_home = tmp_path / "not-created"
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.lease._filesystem_type_for_descriptor",
        lambda _descriptor: filesystem_type,
    )

    with pytest.raises(PermissionError, match="filesystem"):
        RuntimeLease.acquire(runtime_home)

    assert not runtime_home.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mkdir race contract")
def test_runtime_home_fileexists_race_never_chmods_unowned_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_home = tmp_path / "raced-home"
    real_mkdir = os.mkdir

    def race_mkdir(
        path: str,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        real_mkdir(path, mode=0o755, dir_fd=dir_fd)
        os.chmod(path, 0o755, dir_fd=dir_fd, follow_symlinks=False)
        raise FileExistsError(path)

    monkeypatch.setattr("alice_brain_hermes.runtime.lease.os.mkdir", race_mkdir)

    with pytest.raises(PermissionError, match="0700"):
        RuntimeLease.acquire(runtime_home)

    assert stat.S_IMODE(runtime_home.stat().st_mode) == 0o755
    assert not (runtime_home / "daemon.lock").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX openat contract")
def test_runtime_lease_ancestor_swap_never_mutates_replacement_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir(mode=0o700)
    retained = tmp_path / "retained"
    real_open = os.open
    swapped = False

    def swap_before_lock_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "daemon.lock" and dir_fd is not None and not swapped:
            swapped = True
            runtime_home.rename(retained)
            runtime_home.mkdir(mode=0o700)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.lease.os.open", swap_before_lock_open
    )

    with pytest.raises(PermissionError, match="authority"):
        RuntimeLease.acquire(runtime_home)

    assert swapped is True
    assert not (runtime_home / "daemon.lock").exists()
    assert (retained / "daemon.lock").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow contract")
def test_live_lease_rejects_symlinked_ancestor_before_sensitive_mutation(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    home = parent / "runtime"
    home.mkdir(mode=0o700, parents=True)
    displaced = tmp_path / "displaced"
    lease = RuntimeLease.acquire(home)
    try:
        parent.rename(displaced)
        parent.symlink_to(displaced, target_is_directory=True)

        with pytest.raises(PermissionError, match="authority"):
            lease.open_home_file(
                "probe",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )

        assert not (displaced / "runtime" / "probe").exists()
    finally:
        lease.release()


@pytest.mark.skipif(os.name == "nt", reason="POSIX umask contract")
def test_restrictive_umask_still_creates_exact_private_home_and_lock(
    tmp_path: Path,
) -> None:
    runtime_home = tmp_path / "runtime"
    previous = os.umask(0o777)
    try:
        lease = RuntimeLease.acquire(runtime_home)
    finally:
        os.umask(previous)
    try:
        assert stat.S_IMODE(runtime_home.stat().st_mode) == 0o700
        assert stat.S_IMODE((runtime_home / "daemon.lock").stat().st_mode) == 0o600
    finally:
        lease.release()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_existing_insecure_lock_is_rejected_without_metadata_repair(
    tmp_path: Path,
) -> None:
    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir(mode=0o700)
    lock = runtime_home / "daemon.lock"
    lock.write_bytes(b"untrusted")
    lock.chmod(0o666)

    with pytest.raises(PermissionError, match="0600"):
        RuntimeLease.acquire(runtime_home)

    assert stat.S_IMODE(lock.stat().st_mode) == 0o666
    assert lock.read_bytes() == b"untrusted"
