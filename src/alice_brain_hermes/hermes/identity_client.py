"""Dedicated daemon client for optional Hermes self-naming leases."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from alice_brain_hermes.errors import DaemonClientError
from alice_brain_hermes.ids import validate_id
from alice_brain_hermes.protocol.client import DaemonClient
from alice_brain_hermes.protocol.identity import (
    IdentityChoiceV1,
    IdentityNamingLeaseV1,
)
from alice_brain_hermes.protocol.models import BrainProfileV1

ClientFactory = Callable[..., Any]
ProfileFactory = Callable[[], BrainProfileV1]


def hermes_brain_profile(profile_name: str) -> BrainProfileV1:
    """Map one exact host profile to a stable project-owned profile key."""

    if not isinstance(profile_name, str):
        raise TypeError("Hermes profile name must be a string")
    if (
        not profile_name
        or profile_name != profile_name.strip()
        or len(profile_name) > 256
        or len(profile_name.encode("utf-8", errors="strict")) > 1_024
        or any(
            unicodedata.category(character) in {"Cc", "Cs"}
            for character in profile_name
        )
    ):
        raise ValueError("Hermes profile name is invalid")
    if profile_name == "default":
        key = "hermes.default"
    else:
        digest = hashlib.sha256(profile_name.encode("utf-8")).hexdigest()
        key = f"hermes.profile.{digest}"
    return BrainProfileV1(profile_key=key, name=None)


class DaemonIdentityNamingLeasePort:
    """Use a fresh authenticated client, never the bridge transport client."""

    def __init__(
        self,
        runtime_home: str | Path,
        *,
        profile_factory: ProfileFactory,
        client_factory: ClientFactory = DaemonClient.connect,
        timeout_seconds: float = 3.0,
    ) -> None:
        if not callable(profile_factory):
            raise TypeError("profile_factory must be callable")
        if not callable(client_factory):
            raise TypeError("client_factory must be callable")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or not 0 < float(timeout_seconds) <= 300
        ):
            raise ValueError("timeout_seconds must be finite and between 0 and 300")
        self._runtime_home = Path(runtime_home)
        self._profile_factory = profile_factory
        self._client_factory = client_factory
        self._timeout_seconds = float(timeout_seconds)

    def _client(self) -> Any:
        return self._client_factory(
            self._runtime_home,
            initialize=True,
            timeout_seconds=self._timeout_seconds,
        )

    @staticmethod
    def _close(client: Any) -> None:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def claim(self) -> IdentityNamingLeaseV1 | None:
        profile = self._profile_factory()
        if type(profile) is not BrainProfileV1:
            raise TypeError("profile_factory must return an exact BrainProfileV1")
        client = self._client()
        try:
            resolved = client.call(
                "brain.resolve",
                {"profile": profile.model_dump(mode="json")},
            )
            if set(resolved) != {"brain_id", "state_sequence", "created"}:
                raise DaemonClientError("brain.resolve result fields are invalid")
            brain_id = validate_id(resolved["brain_id"])
            if (
                isinstance(resolved["state_sequence"], bool)
                or not isinstance(resolved["state_sequence"], int)
                or resolved["state_sequence"] < 0
                or type(resolved["created"]) is not bool
            ):
                raise DaemonClientError("brain.resolve result is invalid")
            result = client.call(
                "identity.naming.claim",
                {"brain_id": brain_id},
            )
            if set(result) != {"lease"}:
                raise DaemonClientError("identity naming claim fields are invalid")
            lease_data = result["lease"]
            if lease_data is None:
                return None
            if not isinstance(lease_data, dict):
                raise DaemonClientError("identity naming lease is invalid")
            try:
                lease = IdentityNamingLeaseV1.model_validate_json(
                    json.dumps(
                        lease_data,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    strict=True,
                )
            except ValidationError as error:
                raise DaemonClientError("identity naming lease is invalid") from error
            if lease.brain_id != brain_id:
                raise DaemonClientError("identity naming lease changed brain identity")
            return lease
        finally:
            self._close(client)

    def complete(self, lease_id: str, choice: IdentityChoiceV1) -> str:
        lease_id = validate_id(lease_id)
        if not isinstance(choice, IdentityChoiceV1):
            raise TypeError("choice must be IdentityChoiceV1")
        client = self._client()
        try:
            result = client.call(
                "identity.naming.complete",
                {
                    "lease_id": lease_id,
                    "choice": choice.model_dump(mode="json"),
                },
            )
            return self._terminal_status(result)
        finally:
            self._close(client)

    def fail(self, lease_id: str, failure_code: str) -> str:
        lease_id = validate_id(lease_id)
        if not isinstance(failure_code, str):
            raise TypeError("failure_code must be a string")
        client = self._client()
        try:
            result = client.call(
                "identity.naming.fail",
                {"lease_id": lease_id, "failure_code": failure_code},
            )
            return self._terminal_status(result)
        finally:
            self._close(client)

    @staticmethod
    def _terminal_status(result: object) -> str:
        if not isinstance(result, dict) or set(result) != {"status"}:
            raise DaemonClientError("identity naming result fields are invalid")
        status = result["status"]
        if status not in {"completed", "failed", "superseded"}:
            raise DaemonClientError("identity naming result status is invalid")
        return status


__all__ = ["DaemonIdentityNamingLeasePort", "hermes_brain_profile"]
