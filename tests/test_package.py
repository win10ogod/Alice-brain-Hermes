from __future__ import annotations

import subprocess
import tarfile
import tomllib
from importlib.metadata import entry_points, metadata, requires
from pathlib import Path
from uuid import UUID, uuid1
from zipfile import ZipFile

import pytest

from alice_brain_hermes import __version__
from alice_brain_hermes.ids import new_id, validate_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MIT_LICENSE = """MIT License

Copyright (c) 2026 Alice-brain-Hermes contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


@pytest.fixture(scope="module")
def release_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    output_directory = tmp_path_factory.mktemp("release-artifacts")
    subprocess.run(
        ["uv", "build", "--out-dir", str(output_directory)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    wheels = list(output_directory.glob("*.whl"))
    source_distributions = list(output_directory.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(source_distributions) == 1
    return wheels[0], source_distributions[0]


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
    distribution_metadata = metadata("alice-brain-hermes")

    assert console["alice-brain-hermes"] == "alice_brain_hermes.cli:main"
    assert plugins["alice-brain"] == "alice_brain_hermes.hermes_plugin"
    assert set(distribution_metadata["Requires-Python"].split(",")) == {
        ">=3.11",
        "<3.14",
    }
    assert distribution_metadata["License-Expression"] == "MIT"


def test_no_alice_brain_distribution_dependency() -> None:
    normalized = {
        item.split(";", 1)[0].strip().lower().replace("_", "-")
        for item in (requires("alice-brain-hermes") or [])
    }

    assert not any(
        item.startswith("alice-brain") and not item.startswith("alice-brain-hermes")
        for item in normalized
    )


def test_release_uses_canonical_mit_license() -> None:
    assert (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8") == (
        CANONICAL_MIT_LICENSE
    )


def test_project_declares_pep639_mit_license() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]


def test_readme_documents_mit_license() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "[MIT License](LICENSE)" in readme


def test_wheel_contains_license_and_spdx_metadata(
    release_artifacts: tuple[Path, Path],
) -> None:
    wheel, _ = release_artifacts
    with ZipFile(wheel) as archive:
        license_members = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/licenses/LICENSE")
        ]
        metadata_members = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]

        assert len(license_members) == 1
        assert archive.read(license_members[0]).decode("utf-8") == CANONICAL_MIT_LICENSE
        assert len(metadata_members) == 1
        metadata_text = archive.read(metadata_members[0]).decode("utf-8")
        assert "License-Expression: MIT\n" in metadata_text


def test_source_distribution_contains_license(
    release_artifacts: tuple[Path, Path],
) -> None:
    _, source_distribution = release_artifacts
    with tarfile.open(source_distribution, mode="r:gz") as archive:
        license_members = [
            member
            for member in archive.getmembers()
            if member.name.endswith("/LICENSE")
        ]

        assert len(license_members) == 1
        extracted = archive.extractfile(license_members[0])
        assert extracted is not None
        assert extracted.read().decode("utf-8") == CANONICAL_MIT_LICENSE
