from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from dmon.types import DmonMeta

from alice_brain_hermes.runtime.supervisor import (
    DmonAdapter,
    DmonCoordinationTimeout,
    DmonIdentityError,
    DmonMetaHint,
    DmonProcessHint,
    DmonStartCoordinator,
)


class _Dirs:
    def __init__(self, root: Path) -> None:
        self.user_state_path = os.fspath(root / "state")
        self.user_log_path = os.fspath(root / "log")


@pytest.mark.parametrize(
    "command",
    ["python -m daemon", b"python", [], [sys.executable, 1]],
)
def test_adapter_requires_nonempty_string_sequence(
    tmp_path: Path,
    command: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        DmonAdapter.create(tmp_path / "runtime", command)  # type: ignore[arg-type]


def test_adapter_builds_unique_private_platformdirs_config_with_fresh_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    command = [
        sys.executable,
        "-m",
        "alice_brain_hermes.runtime.daemon",
        "--runtime-home",
        os.fspath(tmp_path / "runtime"),
        "--launch-nonce",
        "launch-one",
    ]
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        command,
        launch_nonce="launch-one",
    )
    captured: list[object] = []

    def start_single(config) -> int:
        captured.append(config)
        assert isinstance(config.cmd, list)
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            env=dict(config.env),
            override_env=config.override_env,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=4321,
            shell=False,
            create_time=123.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        config.cmd.append("mutated-by-dmon")
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda pid: (
            "psutil-create-time-us:123000000"
            if pid == 4321
            else pytest.fail("unexpected pid")
        ),
    )

    hint = adapter.start()

    assert hint == DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    assert len(captured) == 1
    config = captured[0]
    assert config.cmd[:-1] == command
    assert command[-1] == "launch-one"
    assert config.cmd is not command
    assert config.task.endswith("-launch-one")
    assert Path(config.meta_path).parent.parent == (
        tmp_path / "private" / "state" / "dmon"
    )
    assert Path(config.log_path).parent.parent == (
        tmp_path / "private" / "state" / "dmon"
    )
    assert Path(config.log_path).parent == Path(config.meta_path).parent
    assert Path(config.meta_path).parent.name == adapter.task
    assert Path(config.log_path).parent.name == adapter.task
    assert Path(config.cwd) == Path(config.meta_path).parent
    assert config.env == {}
    assert config.override_env is False
    assert adapter.last_dmon_output == ""


def test_adapter_accepts_only_equivalent_executable_canonicalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    canonical = os.fspath(tmp_path / "canonical-python")
    command = ["python-alias", "-m", "example", "--value", "exact"]
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        command,
        launch_nonce="windows-canonical",
    )

    def start_single(config) -> int:
        config.cmd[0] = canonical
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=4321,
            shell=False,
            create_time=123.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.shutil.which",
        lambda value: canonical if value == "python-alias" else None,
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )

    assert adapter.start().pid == 4321
    assert command == ["python-alias", "-m", "example", "--value", "exact"]


def test_adapter_refuses_precreated_launch_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_symlink: Callable[[Path, Path, bool], bool],
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce="precreated",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    adapter.meta_path.parent.parent.mkdir(parents=True)
    make_symlink(adapter.meta_path.parent, outside, True)
    called = False

    def start_single(_config) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )

    with pytest.raises(DmonIdentityError, match="launch directory"):
        adapter.start()
    assert called is False
    assert list(outside.iterdir()) == []


def test_adapter_rejects_unbounded_metadata_before_json_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce="oversized-meta",
    )

    def start_single(config) -> int:
        Path(config.meta_path).write_text(
            json.dumps({"padding": "x" * 70_000}),
            encoding="utf-8",
        )
        Path(config.log_path).touch()
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )

    with pytest.raises(DmonIdentityError, match="byte limit"):
        adapter.start()


def test_adapter_rejects_symlink_log_created_during_dmon_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_symlink: Callable[[Path, Path, bool], bool],
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce="log-symlink",
    )
    outside = tmp_path / "outside.log"
    outside.write_bytes(b"unchanged")

    def start_single(config) -> int:
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=4321,
            shell=False,
            create_time=123.0,
        ).dump(config.meta_path)
        log_path = Path(config.log_path)
        make_symlink(log_path, outside, False)
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )

    with pytest.raises(DmonIdentityError, match="log path"):
        adapter.start()
    assert outside.read_bytes() == b"unchanged"


def test_post_launch_failure_never_signals_child_from_wrong_command_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "expected"],
        launch_nonce="wrong-command",
    )

    def start_single(config) -> int:
        DmonMeta(
            task=config.task,
            cmd=[sys.executable, "-m", "wrong"],
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=4321,
            shell=False,
            create_time=123.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.psutil.Process",
        lambda _pid: pytest.fail("unowned metadata PID must never be signalled"),
    )

    with pytest.raises(DmonIdentityError) as failure:
        adapter.start()

    assert failure.value.cleanup_unproven is True
    assert failure.value.meta_hint == DmonMetaHint(pid=4321, create_time=123.0)
    assert adapter.meta_path.exists()
    assert adapter.log_path.exists()


def test_portable_start_coordinator_is_exclusive_and_reacquirable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    first = DmonStartCoordinator.create(tmp_path / "runtime")
    second = DmonStartCoordinator.create(tmp_path / "runtime")
    first.acquire(timeout_seconds=0.1)
    try:
        with pytest.raises(DmonCoordinationTimeout):
            second.acquire(timeout_seconds=0.05)
    finally:
        first.release()

    second.acquire(timeout_seconds=0.1)
    second.release()


def test_parent_guard_prevents_recovery_from_signalling_inflight_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    home = tmp_path / "runtime"

    def command(nonce: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "alice_brain_hermes.runtime.daemon",
            "--runtime-home",
            os.fspath(home),
            "--launch-nonce",
            nonce,
        ]

    metadata_written = threading.Event()
    release_start = threading.Event()
    first = DmonAdapter.create(home, command("first"), launch_nonce="first")
    starts: list[str] = []

    def start_single(config) -> int:
        starts.append(config.task)
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=4321,
            shell=False,
            create_time=123.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        metadata_written.set()
        assert release_start.wait(timeout=5.0)
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.psutil.Process",
        lambda _pid: pytest.fail("in-flight parent-owned child must not be signalled"),
    )
    outcome: list[DmonProcessHint | BaseException] = []

    def launch_first() -> None:
        try:
            outcome.append(first.start())
        except BaseException as error:
            outcome.append(error)

    worker = threading.Thread(target=launch_first)
    worker.start()
    assert metadata_written.wait(timeout=5.0)
    second = DmonAdapter.create(home, command("second"), launch_nonce="second")
    try:
        with pytest.raises(DmonIdentityError, match="parent-managed") as blocked:
            second.start()
        assert blocked.value.cleanup_unproven is True
        assert starts == [first.task]
        assert first.meta_path.exists()
        assert first.log_path.exists()
    finally:
        release_start.set()
        worker.join(timeout=5.0)
        first.release_parent_guard()
    assert worker.is_alive() is False
    assert outcome == [
        DmonProcessHint(
            pid=4321,
            process_marker="psutil-create-time-us:123000000",
        )
    ]


def test_post_launch_marker_ambiguity_persists_hint_and_blocks_next_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    starts: list[str] = []

    def start_single(config) -> int:
        starts.append(config.task)
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=4321,
            shell=False,
            create_time=123.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )

    def unreadable(_pid: int) -> str:
        raise PermissionError("injected marker denial")

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker", unreadable
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.psutil.pid_exists", lambda _pid: True
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.psutil.Process",
        lambda _pid: pytest.fail("ambiguous PID must never be signalled"),
    )
    first = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example", "--launch-nonce", "first"],
        launch_nonce="first",
    )

    with pytest.raises(DmonIdentityError) as failure:
        first.start()
    assert failure.value.cleanup_unproven is True
    assert failure.value.meta_hint == DmonMetaHint(pid=4321, create_time=123.0)

    second = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example", "--launch-nonce", "second"],
        launch_nonce="second",
    )
    with pytest.raises(DmonIdentityError, match="prior dmon launch") as blocked:
        second.start()
    assert blocked.value.cleanup_unproven is True
    assert starts == [first.task]


def test_prior_exact_product_launch_is_terminated_and_both_attempt_trees_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    home = tmp_path / "runtime"

    def command(nonce: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "alice_brain_hermes.runtime.daemon",
            "--runtime-home",
            os.fspath(home),
            "--launch-nonce",
            nonce,
        ]

    old = DmonAdapter.create(home, command("old"), launch_nonce="old")
    old.meta_path.parent.mkdir(parents=True)
    DmonMeta(
        task=old.task,
        cmd=command("old"),
        cwd=os.fspath(old.meta_path.parent),
        log_path=os.fspath(old.log_path),
        meta_path=os.fspath(old.meta_path),
        pid=4321,
        shell=False,
        create_time=123.0,
    ).dump(old.meta_path)
    old.log_path.write_text("old log", encoding="utf-8")
    signals: list[str] = []

    class Process:
        def __init__(self, pid: int) -> None:
            assert pid == 4321
            self.pid = pid

        def create_time(self) -> float:
            return 123.0

        def children(self, *, recursive: bool) -> list[Process]:
            assert recursive is True
            return []

        def terminate(self) -> None:
            signals.append("terminate")

        def wait(self, timeout: float) -> None:
            assert timeout > 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )
    monkeypatch.setattr("alice_brain_hermes.runtime.supervisor.psutil.Process", Process)

    def start_single(config) -> int:
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=5678,
            shell=False,
            create_time=456.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )

    current = DmonAdapter.create(home, command("new"), launch_nonce="new")
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda pid: (
            "psutil-create-time-us:123000000"
            if pid == 4321
            else "psutil-create-time-us:456000000"
        ),
    )
    assert current.start().pid == 5678
    assert signals == ["terminate"]
    assert not old.meta_path.parent.exists()
    assert not old.log_path.parent.exists()


def test_prior_gone_product_launch_recovers_partial_legacy_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    home = tmp_path / "runtime"

    def command(nonce: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "alice_brain_hermes.runtime.daemon",
            "--runtime-home",
            os.fspath(home),
            "--launch-nonce",
            nonce,
        ]

    old = DmonAdapter.create(home, command("old"), launch_nonce="old")
    legacy_log_path = (
        tmp_path
        / "private"
        / "log"
        / "dmon"
        / old.task
        / f"{old.task}.log"
    )
    old.meta_path.parent.mkdir(parents=True)
    DmonMeta(
        task=old.task,
        cmd=command("old"),
        cwd=os.fspath(old.meta_path.parent),
        log_path=os.fspath(legacy_log_path),
        meta_path=os.fspath(old.meta_path),
        pid=4321,
        shell=False,
        create_time=123.0,
    ).dump(old.meta_path)

    def read_marker(pid: int) -> str:
        if pid == 4321:
            raise ProcessLookupError(pid)
        assert pid == 5678
        return "psutil-create-time-us:456000000"

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        read_marker,
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.psutil.pid_exists",
        lambda pid: False if pid == 4321 else pytest.fail("unexpected pid"),
    )

    def start_single(config) -> int:
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=5678,
            shell=False,
            create_time=456.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single",
        start_single,
    )

    current = DmonAdapter.create(home, command("new"), launch_nonce="new")
    assert current.start().pid == 5678
    assert not old.meta_path.parent.exists()
    assert not legacy_log_path.parent.exists()


def test_prior_recovery_restores_interrupted_shared_cleanup_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    home = tmp_path / "runtime"

    def command(nonce: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "alice_brain_hermes.runtime.daemon",
            "--runtime-home",
            os.fspath(home),
            "--launch-nonce",
            nonce,
        ]

    old = DmonAdapter.create(home, command("old"), launch_nonce="old")
    old.meta_path.parent.mkdir(parents=True)
    DmonMeta(
        task=old.task,
        cmd=command("old"),
        cwd=os.fspath(old.meta_path.parent),
        log_path=os.fspath(old.log_path),
        meta_path=os.fspath(old.meta_path),
        pid=4321,
        shell=False,
        create_time=123.0,
    ).dump(old.meta_path)
    old.log_path.write_text("old log", encoding="utf-8")
    quarantine = old.meta_path.parent.with_name(
        f".{old.task}.cleanup-{'a' * 32}"
    )
    os.replace(old.meta_path.parent, quarantine)

    def read_marker(pid: int) -> str:
        if pid == 4321:
            raise ProcessLookupError(pid)
        assert pid == 5678
        return "psutil-create-time-us:456000000"

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        read_marker,
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.psutil.pid_exists",
        lambda pid: False if pid == 4321 else pytest.fail("unexpected pid"),
    )

    def start_single(config) -> int:
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=5678,
            shell=False,
            create_time=456.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single",
        start_single,
    )

    current = DmonAdapter.create(home, command("new"), launch_nonce="new")
    assert current.start().pid == 5678
    assert not quarantine.exists()
    assert not old.meta_path.parent.exists()


def test_interrupted_legacy_log_cleanup_quarantine_is_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    home = tmp_path / "runtime"
    adapter = DmonAdapter.create(
        home,
        [sys.executable, "-m", "example"],
        launch_nonce="legacy-interrupted",
    )
    legacy_root = tmp_path / "private" / "log" / "dmon"
    quarantine = legacy_root / (
        f".{adapter.task}.cleanup-{'b' * 32}"
    )
    quarantine.mkdir(parents=True)
    quarantined_log = quarantine / f"{adapter.task}.log"
    quarantined_log.write_text("legacy log", encoding="utf-8")

    adapter._restore_interrupted_cleanup_directories(legacy_root)

    restored = legacy_root / adapter.task / quarantined_log.name
    assert restored.read_text(encoding="utf-8") == "legacy log"
    assert not quarantine.exists()


def test_prior_metadata_with_non_product_argv_never_signals_and_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    home = tmp_path / "runtime"
    old = DmonAdapter.create(
        home,
        [sys.executable, "-m", "attacker"],
        launch_nonce="old",
    )
    old.meta_path.parent.mkdir(parents=True)
    DmonMeta(
        task=old.task,
        cmd=[sys.executable, "-m", "attacker"],
        cwd=os.fspath(old.meta_path.parent),
        log_path=os.fspath(old.log_path),
        meta_path=os.fspath(old.meta_path),
        pid=4321,
        shell=False,
        create_time=123.0,
    ).dump(old.meta_path)
    old.log_path.touch()
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.psutil.Process",
        lambda _pid: pytest.fail("unowned metadata must never be signalled"),
    )
    current = DmonAdapter.create(
        home,
        [
            sys.executable,
            "-m",
            "alice_brain_hermes.runtime.daemon",
            "--runtime-home",
            os.fspath(home),
            "--launch-nonce",
            "new",
        ],
        launch_nonce="new",
    )

    with pytest.raises(DmonIdentityError, match="prior dmon launch") as blocked:
        current.start()
    assert blocked.value.cleanup_unproven is True


def test_metadata_read_binds_open_file_identity_and_applies_private_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce="file-identity",
    )
    chmod_calls: list[tuple[Path, int]] = []
    fstat_calls: list[int] = []
    real_chmod = Path.chmod
    real_fstat = os.fstat

    def chmod(path: Path, mode: int, *args, **kwargs) -> None:
        chmod_calls.append((path, mode))
        real_chmod(path, mode, *args, **kwargs)

    def fstat(descriptor: int):
        fstat_calls.append(descriptor)
        return real_fstat(descriptor)

    def start_single(config) -> int:
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=4321,
            shell=False,
            create_time=123.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        return 0

    monkeypatch.setattr(Path, "chmod", chmod)
    monkeypatch.setattr("alice_brain_hermes.runtime.supervisor.os.fstat", fstat)
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )

    adapter.start()

    assert fstat_calls
    assert (adapter.meta_path.parent, 0o700) in chmod_calls
    assert (adapter.log_path.parent, 0o700) in chmod_calls
    assert (adapter.meta_path, 0o600) in chmod_calls
    assert (adapter.log_path, 0o600) in chmod_calls


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("log_path", "outside.log"),
        ("cwd", "outside-cwd"),
        ("env", {"INJECTED": "1"}),
        ("override_env", True),
        ("shell", True),
        ("log_rotate", True),
        ("pid", 9999),
        ("create_time", 999.0),
    ],
)
def test_remove_meta_hint_rejects_any_tampered_launch_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce=f"tampered-{field}",
    )

    def start_single(config) -> int:
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=4321,
            shell=False,
            create_time=123.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )
    hint = adapter.start()
    meta = DmonMeta.load(adapter.meta_path)
    assert meta is not None
    if field in {"log_path", "cwd"}:
        value = os.fspath(tmp_path / str(value))
    setattr(meta, field, value)
    meta.dump(adapter.meta_path)

    assert adapter.remove_meta_hint(hint) is False
    assert adapter.meta_path.exists()
    assert adapter.log_path.exists()


def test_authenticated_cleanup_removes_fully_validated_launch_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce="authenticated-cleanup",
    )

    def start_single(config) -> int:
        DmonMeta(
            task=config.task,
            cmd=list(config.cmd),
            cwd=config.cwd,
            log_path=config.log_path,
            meta_path=config.meta_path,
            pid=4321,
            shell=False,
            create_time=123.0,
        ).dump(config.meta_path)
        Path(config.log_path).touch()
        return 0

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", start_single
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )
    adapter.start()
    adapter.release_parent_guard()

    hint = adapter.current_process_hint()
    assert adapter.remove_meta_hint(hint) is True
    assert not adapter.meta_path.exists()
    assert not adapter.log_path.exists()
    assert not adapter.meta_path.parent.exists()
    assert not adapter.log_path.parent.exists()


def test_authenticated_cleanup_migrates_and_removes_legacy_split_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    home = tmp_path / "runtime"
    command = [
        sys.executable,
        "-m",
        "alice_brain_hermes.runtime.daemon",
        "--runtime-home",
        os.fspath(home),
        "--launch-nonce",
        "legacy-layout",
    ]
    adapter = DmonAdapter.create(
        home,
        command,
        launch_nonce="legacy-layout",
    )
    legacy_log_path = (
        tmp_path
        / "private"
        / "log"
        / "dmon"
        / adapter.task
        / f"{adapter.task}.log"
    )
    adapter.meta_path.parent.mkdir(parents=True)
    legacy_log_path.parent.mkdir(parents=True)
    DmonMeta(
        task=adapter.task,
        cmd=command,
        cwd=os.fspath(adapter.meta_path.parent),
        log_path=os.fspath(legacy_log_path),
        meta_path=os.fspath(adapter.meta_path),
        pid=4321,
        shell=False,
        create_time=123.0,
    ).dump(adapter.meta_path)
    legacy_log_path.write_text("legacy log", encoding="utf-8")
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: "psutil-create-time-us:123000000",
    )

    hint = adapter.current_process_hint()

    assert hint == DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    assert adapter.remove_meta_hint(hint) is True
    assert not adapter.meta_path.parent.exists()
    assert not legacy_log_path.parent.exists()


def test_attempt_cleanup_never_deletes_files_from_replaced_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_directory = tmp_path / "state" / "attempt"
    attempt_directory.mkdir(parents=True)
    meta_path = attempt_directory / "attempt.meta.json"
    log_path = attempt_directory / "attempt.log"
    meta_path.write_text("owned-meta", encoding="utf-8")
    log_path.write_text("owned-log", encoding="utf-8")
    identity = attempt_directory.stat(follow_symlinks=False)

    displaced = tmp_path / "displaced-owned-attempt"
    victim = tmp_path / "victim-attempt"
    victim.mkdir()
    (victim / meta_path.name).write_text("victim-meta", encoding="utf-8")
    (victim / log_path.name).write_text("victim-log", encoding="utf-8")
    real_replace = os.replace
    swapped = False

    def replace(source: Path, destination: Path) -> None:
        nonlocal swapped
        if Path(source) == attempt_directory and not swapped:
            swapped = True
            attempt_directory.rename(displaced)
            victim.rename(attempt_directory)
        real_replace(source, destination)

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.os.replace",
        replace,
    )

    assert (
        DmonAdapter._remove_attempt_paths(
            meta_path,
            log_path,
            meta_directory_identity=identity,
            log_directory_identity=identity,
        )
        is False
    )
    assert swapped is True
    assert (attempt_directory / meta_path.name).read_text(
        encoding="utf-8"
    ) == "victim-meta"
    assert (attempt_directory / log_path.name).read_text(
        encoding="utf-8"
    ) == "victim-log"
    assert (displaced / meta_path.name).read_text(encoding="utf-8") == "owned-meta"
    assert (displaced / log_path.name).read_text(encoding="utf-8") == "owned-log"


def test_attempt_cleanup_keeps_both_evidence_files_if_directory_is_dirty(
    tmp_path: Path,
) -> None:
    attempt_directory = tmp_path / "state" / "attempt"
    attempt_directory.mkdir(parents=True)
    meta_path = attempt_directory / "attempt.meta.json"
    log_path = attempt_directory / "attempt.log"
    extra_log = attempt_directory / "unexpected.log"
    meta_path.write_text("owned-meta", encoding="utf-8")
    log_path.write_text("owned-log", encoding="utf-8")
    extra_log.write_text("unexpected", encoding="utf-8")

    assert (
        DmonAdapter._remove_attempt_paths(
            meta_path,
            log_path,
            meta_directory_identity=attempt_directory.stat(follow_symlinks=False),
            log_directory_identity=attempt_directory.stat(follow_symlinks=False),
        )
        is False
    )
    assert meta_path.read_text(encoding="utf-8") == "owned-meta"
    assert log_path.read_text(encoding="utf-8") == "owned-log"
    assert extra_log.read_text(encoding="utf-8") == "unexpected"


def test_attempt_cleanup_retries_authenticated_metadata_after_partial_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_directory = tmp_path / "state" / "attempt"
    attempt_directory.mkdir(parents=True)
    meta_path = attempt_directory / "attempt.meta.json"
    log_path = attempt_directory / "attempt.log"
    meta_path.write_text("owned-meta", encoding="utf-8")
    log_path.write_text("owned-log", encoding="utf-8")
    identity = attempt_directory.stat(follow_symlinks=False)
    calls: list[tuple[str, ...]] = []

    def partial_then_complete(tasks) -> bool:
        directory, expected, names = tasks[0]
        assert os.path.samestat(expected, identity)
        calls.append(names)
        if names == (log_path.name, meta_path.name):
            (directory / log_path.name).unlink()
            return False
        assert names == (meta_path.name,)
        (directory / meta_path.name).unlink()
        return False

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor._anchored_unlink",
        partial_then_complete,
    )

    assert DmonAdapter._remove_attempt_paths(
        meta_path,
        log_path,
        meta_directory_identity=identity,
        log_directory_identity=identity,
    )
    assert calls == [
        (log_path.name, meta_path.name),
        (meta_path.name,),
    ]
    assert not attempt_directory.exists()


def test_dmon_output_capture_is_bounded_and_redacted_log_tail_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    home = tmp_path / "secret-home"
    nonce = "a" * 64
    adapter = DmonAdapter.create(
        home,
        [sys.executable, "-m", "example", "--launch-nonce", nonce],
        launch_nonce=nonce,
    )

    def fail_start(_config) -> int:
        print("x" * 20_000, file=sys.stderr)
        return 1

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.start_single", fail_start
    )
    with pytest.raises(DmonIdentityError, match="start failed"):
        adapter.start()
    assert len(adapter.last_dmon_output.encode("utf-8")) <= 4_096

    adapter.log_path.parent.mkdir(parents=True, exist_ok=True)
    adapter.log_path.write_text(
        "z" * 10_000 + os.fspath(home) + " " + nonce,
        encoding="utf-8",
    )
    tail = adapter.redacted_log_tail()
    assert len(tail.encode("utf-8")) <= 4_096
    assert os.fspath(home) not in tail
    assert nonce not in tail
    assert "<runtime-home>" in tail
    assert "<redacted>" in tail


def test_exact_cleanup_rechecks_process_create_time_before_terminate_and_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce="launch-cleanup",
    )
    hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    create_time_reads: list[int] = []

    class Process:
        def __init__(self, pid: int) -> None:
            assert pid == hint.pid
            self.pid = pid
            self.terminate_calls = 0
            self.kill_calls = 0
            processes.append(self)

        def create_time(self) -> float:
            create_time_reads.append(self.pid)
            return 123.0

        def children(self, *, recursive: bool) -> list[Process]:
            assert recursive is True
            return []

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

        def wait(self, timeout: float) -> None:
            waits.append(timeout)
            if len(waits) == 1:
                import psutil

                raise psutil.TimeoutExpired(timeout, pid=self.pid)

    processes: list[Process] = []
    waits: list[float] = []
    monkeypatch.setattr("alice_brain_hermes.runtime.supervisor.psutil.Process", Process)

    adapter.terminate_exact(hint, timeout_seconds=1.0)

    assert create_time_reads == [4321, 4321, 4321, 4321, 4321]
    assert len(processes) == 2
    assert processes[0].terminate_calls == 1
    assert processes[0].kill_calls == 0
    assert processes[1].terminate_calls == 0
    assert processes[1].kill_calls == 1


def test_exact_cleanup_terminates_authenticated_descendant_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce="launch-tree-cleanup",
    )

    class Process:
        def __init__(self, pid: int, created: float) -> None:
            self.pid = pid
            self.created = created
            self.terminate_calls = 0
            self.wait_calls = 0
            self.descendants: list[Process] = []

        def create_time(self) -> float:
            return self.created

        def children(self, *, recursive: bool) -> list[Process]:
            assert recursive is True
            return list(self.descendants)

        def terminate(self) -> None:
            self.terminate_calls += 1

        def wait(self, timeout: float) -> None:
            assert timeout > 0
            self.wait_calls += 1

    parent = Process(4321, 123.0)
    child = Process(8765, 456.0)
    parent.descendants.append(child)
    processes = {parent.pid: parent, child.pid: child}
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.psutil.Process",
        lambda pid: processes[pid],
    )

    adapter.terminate_exact(
        DmonProcessHint(
            pid=parent.pid,
            process_marker="psutil-create-time-us:123000000",
        ),
        timeout_seconds=1.0,
    )

    assert parent.terminate_calls == 1
    assert child.terminate_calls == 1
    assert parent.wait_calls == 1
    assert child.wait_calls == 1


def test_exact_cleanup_never_signals_reused_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce="launch-reused",
    )
    hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    class Process:
        pid = 4321

        def __init__(self, _pid: int) -> None:
            self.terminate_calls = 0

        def create_time(self) -> float:
            return 999.0

        def children(self, *, recursive: bool) -> list[Process]:
            assert recursive is True
            return []

        def terminate(self) -> None:
            self.terminate_calls += 1

    process = Process(4321)
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.psutil.Process", lambda _pid: process
    )

    adapter.terminate_exact(hint, timeout_seconds=1.0)
    assert process.terminate_calls == 0


def test_exact_cleanup_rechecks_process_create_time_immediately_before_terminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce="reuse-before-terminate",
    )
    hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )

    class Process:
        pid = 4321

        def __init__(self, _pid: int) -> None:
            self.create_time_reads = 0
            self.terminate_calls = 0

        def create_time(self) -> float:
            self.create_time_reads += 1
            return 123.0 if self.create_time_reads == 1 else 999.0

        def children(self, *, recursive: bool) -> list[Process]:
            assert recursive is True
            return []

        def terminate(self) -> None:
            self.terminate_calls += 1

    process = Process(4321)
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: hint.process_marker,
    )
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.psutil.Process", lambda _pid: process
    )

    adapter.terminate_exact(hint, timeout_seconds=1.0)

    assert process.create_time_reads == 2
    assert process.terminate_calls == 0


def test_exact_cleanup_rechecks_process_create_time_immediately_before_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.PlatformDirs",
        lambda *_args, **_kwargs: _Dirs(tmp_path / "private"),
    )
    adapter = DmonAdapter.create(
        tmp_path / "runtime",
        [sys.executable, "-m", "example"],
        launch_nonce="reuse-before-kill",
    )
    hint = DmonProcessHint(
        pid=4321,
        process_marker="psutil-create-time-us:123000000",
    )
    instances: list[object] = []

    class Process:
        pid = 4321

        def __init__(self, _pid: int) -> None:
            self.create_time_reads = 0
            self.terminate_calls = 0
            self.kill_calls = 0
            instances.append(self)

        def create_time(self) -> float:
            self.create_time_reads += 1
            if len(instances) == 2 and self.create_time_reads == 2:
                return 999.0
            return 123.0

        def children(self, *, recursive: bool) -> list[Process]:
            assert recursive is True
            return []

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

        def wait(self, timeout: float) -> None:
            raise __import__("psutil").TimeoutExpired(timeout, pid=self.pid)

    monkeypatch.setattr(
        "alice_brain_hermes.runtime.supervisor.read_process_marker",
        lambda _pid: hint.process_marker,
    )
    monkeypatch.setattr("alice_brain_hermes.runtime.supervisor.psutil.Process", Process)

    adapter.terminate_exact(hint, timeout_seconds=1.0)

    assert len(instances) == 2
    first, second = instances
    assert first.terminate_calls == 1
    assert second.create_time_reads == 2
    assert second.kill_calls == 0
