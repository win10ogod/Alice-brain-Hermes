from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def test_daemon_cli_process_hard_exits_with_a_stuck_executor_worker(
    tmp_path: Path,
) -> None:
    program = r"""
import asyncio
import sys
import threading

from alice_brain_hermes.errors import SchedulerShutdownError
from alice_brain_hermes.runtime import daemon

started = threading.Event()
never = threading.Event()

def blocked_writer():
    started.set()
    never.wait()

async def unproven_shutdown(*_args, **_kwargs):
    worker = asyncio.create_task(asyncio.to_thread(blocked_writer))
    while not started.is_set():
        await asyncio.sleep(0)
    assert not worker.done()
    raise SchedulerShutdownError("injected unproven writer")

daemon._run_daemon = unproven_shutdown
raise SystemExit(daemon._main(["--runtime-home", sys.argv[1]]))
"""
    began = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "runtime")],
        check=False,
        close_fds=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )

    assert completed.returncode == 3, completed.stderr
    assert time.monotonic() - began < 1.5


def test_daemon_cli_process_hard_exits_with_a_stuck_real_handler(
    tmp_path: Path,
) -> None:
    program = r"""
import asyncio
import sys
import threading
from types import SimpleNamespace

from alice_brain_hermes.errors import SchedulerShutdownError
from alice_brain_hermes.runtime import daemon

started = threading.Event()
never = threading.Event()

class Connection:
    shutdown_requested = False
    authenticated = False

    def handle_frame(self, _frame):
        started.set()
        never.wait()

    def close(self):
        never.wait()

class Service:
    limits = SimpleNamespace(max_request_bytes=1024)
    shutting_down = False

    def new_connection(self):
        return Connection()

    def begin_shutdown(self):
        self.shutting_down = True

class Reader:
    first = True

    async def read(self, _maximum):
        if self.first:
            self.first = False
            return b'{}\n'
        await asyncio.Event().wait()

class Writer:
    def write(self, _data):
        pass

    async def drain(self):
        pass

    def close(self):
        pass

    async def wait_closed(self):
        pass

async def unproven_handler(*_args, **_kwargs):
    server = object.__new__(daemon.PrivateDaemonServer)
    server.service = Service()
    server._handlers = set()
    server._writers = set()
    server._maintenance_error = None
    server._shutdown = asyncio.Event()
    server._max_concurrent_connections = 1
    server._unauthenticated_idle_timeout_seconds = 1.0
    server._active_connection_count = 0
    daemon._is_private_ipv4_stream = lambda _writer: True
    asyncio.create_task(server._handle_client(Reader(), Writer()))
    while not started.is_set():
        await asyncio.sleep(0)
    raise SchedulerShutdownError('real handler shutdown is unproven')

daemon._run_daemon = unproven_handler
raise SystemExit(daemon._main(['--runtime-home', sys.argv[1]]))
"""
    began = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "runtime")],
        check=False,
        close_fds=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )

    assert completed.returncode == 3, completed.stderr
    assert time.monotonic() - began < 1.5
