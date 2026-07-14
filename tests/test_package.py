from __future__ import annotations

from importlib.metadata import entry_points, metadata, requires
from uuid import UUID, uuid1

import pytest

from alice_brain_hermes import __version__
from alice_brain_hermes.ids import new_id, validate_id


def test_package_identity_and_uuid() -> None:
    value = new_id()

    assert __version__ == "0.1.0"
    assert UUID(value).version == 4
    assert validate_id(value) == value


def test_new_ids_are_distinct_canonical_uuid4_values() -> None:
    first = new_id()
    second = new_id()

    assert first != second
    assert first == str(UUID(first))
    assert second == str(UUID(second))


@pytest.mark.parametrize(
    "value",
    [
        "not-an-id",
        "{12345678-1234-4234-8234-123456789abc}",
        "12345678123442348234123456789abc",
        str(uuid1()),
    ],
)
def test_validate_id_rejects_noncanonical_or_non_uuid4_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_id(value)


def test_distribution_metadata_exposes_module_entry_points() -> None:
    console = {item.name: item.value for item in entry_points(group="console_scripts")}
    plugins = {
        item.name: item.value
        for item in entry_points(group="hermes_agent.plugins")
        if item.dist is not None and item.dist.name == "alice-brain-hermes"
    }

    assert console["alice-brain-hermes"] == "alice_brain_hermes.cli:main"
    assert plugins["alice-brain"] == "alice_brain_hermes.hermes_plugin"
    assert set(metadata("alice-brain-hermes")["Requires-Python"].split(",")) == {
        ">=3.11",
        "<3.14",
    }


def test_no_alice_brain_distribution_dependency() -> None:
    normalized = {
        item.split(";", 1)[0].strip().lower().replace("_", "-")
        for item in (requires("alice-brain-hermes") or [])
    }

    assert not any(
        item.startswith("alice-brain") and not item.startswith("alice-brain-hermes")
        for item in normalized
    )
