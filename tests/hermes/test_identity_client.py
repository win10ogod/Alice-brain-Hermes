from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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


class FakeClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
        self.calls.append((method, params))
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def test_profile_mapping_is_stable_bounded_and_does_not_expose_name() -> None:
    assert hermes_brain_profile("default").profile_key == "hermes.default"
    expected = hashlib.sha256("工作".encode()).hexdigest()
    profile = hermes_brain_profile("工作")
    assert profile.profile_key == f"hermes.profile.{expected}"
    assert profile.name is None
    with pytest.raises(ValueError):
        hermes_brain_profile("bad\nprofile")


def test_port_reads_profile_only_when_claiming_and_uses_fresh_clients(
    tmp_path: Path,
) -> None:
    brain_id = new_id()
    lease = IdentityNamingLeaseV1(
        lease_id=new_id(),
        brain_id=brain_id,
        state_sequence=2,
        expires_at=datetime.now(UTC) + timedelta(seconds=30),
    )
    clients = [
        FakeClient(
            [
                {"brain_id": brain_id, "state_sequence": 1, "created": True},
                {"lease": lease.model_dump(mode="json")},
            ]
        ),
        FakeClient([{"status": "completed"}]),
        FakeClient([{"status": "failed"}]),
    ]
    constructed: list[tuple[Path, dict[str, object]]] = []
    profile_reads = 0

    profile = hermes_brain_profile("default")

    def profile_factory() -> BrainProfileV1:
        nonlocal profile_reads
        profile_reads += 1
        return profile

    def client_factory(home: Path, **kwargs: object) -> FakeClient:
        constructed.append((home, kwargs))
        return clients[len(constructed) - 1]

    port = DaemonIdentityNamingLeasePort(
        tmp_path,
        profile_factory=profile_factory,
        client_factory=client_factory,
    )
    assert profile_reads == 0
    assert constructed == []

    assert port.claim() == lease
    assert profile_reads == 1
    choice = IdentityChoiceV1(name="Mira", reason="Chosen")
    assert port.complete(lease.lease_id, choice) == "completed"
    assert port.fail(lease.lease_id, "llm_error.TimeoutError") == "failed"

    assert len(constructed) == 3
    assert all(client.closed for client in clients)
    assert clients[0].calls[0][0] == "brain.resolve"
    assert clients[0].calls[0][1] == {"profile": profile.model_dump(mode="json")}
    assert clients[0].calls[1] == (
        "identity.naming.claim",
        {"brain_id": brain_id},
    )
    assert clients[1].calls == [
        (
            "identity.naming.complete",
            {
                "lease_id": lease.lease_id,
                "choice": choice.model_dump(mode="json"),
            },
        )
    ]
    assert clients[2].calls == [
        (
            "identity.naming.fail",
            {
                "lease_id": lease.lease_id,
                "failure_code": "llm_error.TimeoutError",
            },
        )
    ]
    assert all(
        kwargs == {"initialize": True, "timeout_seconds": 3.0}
        for _home, kwargs in constructed
    )


def test_port_rejects_a_second_host_name_mapping_boundary(tmp_path: Path) -> None:
    client_reads = 0

    def client_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal client_reads
        client_reads += 1
        return object()

    port = DaemonIdentityNamingLeasePort(
        tmp_path,
        profile_factory=lambda: "default",  # type: ignore[return-value]
        client_factory=client_factory,
    )

    with pytest.raises(TypeError, match="BrainProfileV1"):
        port.claim()
    assert client_reads == 0
