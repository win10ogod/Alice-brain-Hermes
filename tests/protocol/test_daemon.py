from __future__ import annotations

import json
import os
import select
import socket
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

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
from alice_brain_hermes.runtime.store import SQLiteLedger

RECOVERY_TOKEN = "ab" * 32


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
):
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "alice_brain_hermes.runtime.daemon",
            "--runtime-home",
            str(home),
            "--readiness-fd",
            str(write_fd),
            "--scheduler-interval",
            "0.02",
            "--abandonment-grace",
            str(abandonment_grace_seconds),
        ],
        pass_fds=(write_fd,),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.close(write_fd)
    ready, _, _ = select.select([read_fd], [], [], 10.0)
    assert ready, "daemon readiness pipe timed out"
    body = bytearray()
    while b"\n" not in body:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        body.extend(chunk)
    os.close(read_fd)
    readiness = json.loads(bytes(body))
    assert readiness["ready"] is expect_ready
    return process, readiness


def stop_process(process: subprocess.Popen[bytes], client: DaemonClient) -> None:
    client.shutdown()
    client.close()
    assert process.wait(timeout=10.0) == 0
    stdout, stderr = process.communicate()
    assert stdout == b""
    assert stderr == b""


def test_subprocess_readiness_eof_does_not_stop_and_explicit_shutdown_does(
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
    first_process.kill()
    assert first_process.wait(timeout=5.0) != 0
    first_process.communicate()

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

    first_process.kill()
    assert first_process.wait(timeout=5.0) != 0
    first_process.communicate()
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
    first_process, _ready = launch(home, abandonment_grace_seconds=0.1)
    first = DaemonClient.connect(home)
    brain = first.call("brain.create", {"name": None})
    instance = new_id()
    first.call(
        "brain.attach",
        bridge_attach_params(brain["brain_id"], instance),
    )
    stop_process(first_process, first)
    time.sleep(0.3)

    second_process, _ready = launch(home, abandonment_grace_seconds=0.1)
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
