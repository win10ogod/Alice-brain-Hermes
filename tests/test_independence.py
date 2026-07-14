from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.check_independence import AuditViolation, audit_project, audit_wheel


def _write_source(root: Path, source: str) -> None:
    path = root / "src" / "fixture.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")


def _write_wheel(
    path: Path,
    *,
    dependencies: tuple[str, ...] = (),
    source: str = "from alice_brain_hermes import __version__\n",
) -> None:
    requires_dist = "".join(f"Requires-Dist: {item}\n" for item in dependencies)
    metadata_text = (
        f"Metadata-Version: 2.4\nName: fixture\nVersion: 0.1.0\n{requires_dist}\n"
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("fixture/__init__.py", source)
        archive.writestr("fixture-0.1.0.dist-info/METADATA", metadata_text)


def test_project_audit_allows_hermes_namespace(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        "from alice_brain_hermes.ids import new_id\n"
        "import alice_brain_hermes.runtime\n",
    )

    audit_project(tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        "import alice_brain\n",
        "import alice_brain.runtime\n",
        "from alice_brain import runtime\n",
        "from alice_brain.runtime import client\n",
    ],
)
def test_project_audit_rejects_exact_alice_brain_import_root(
    tmp_path: Path, source: str
) -> None:
    _write_source(tmp_path, source)

    with pytest.raises(AuditViolation, match="forbidden import root"):
        audit_project(tmp_path)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('CHECKOUT = "../Alice-brain"\n', "sibling Alice-brain checkout path"),
        ('HOME_VARIABLE = "ALICE_BRAIN_HOME"\n', "Alice-brain environment variable"),
        ('SOCKET = "alice-brain.sock"\n', "daemon service"),
        (
            'COMMAND = ["alice-brain", "daemon", "status"]\n',
            "daemon service",
        ),
    ],
)
def test_project_audit_rejects_explicit_external_runtime_references(
    tmp_path: Path, source: str, message: str
) -> None:
    _write_source(tmp_path, source)

    with pytest.raises(AuditViolation, match=message):
        audit_project(tmp_path)


def test_project_audit_rejects_git_submodules(tmp_path: Path) -> None:
    (tmp_path / ".gitmodules").write_text(
        '[submodule "Alice-brain"]\n\tpath = Alice-brain\n', encoding="utf-8"
    )

    with pytest.raises(AuditViolation, match="git submodules"):
        audit_project(tmp_path)


def test_project_audit_checks_pyproject_for_sibling_paths(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "0.1.0"\n'
        'dependencies = ["fixture @ file://../Alice-brain"]\n',
        encoding="utf-8",
    )

    with pytest.raises(AuditViolation, match="sibling Alice-brain checkout path"):
        audit_project(tmp_path)


def test_project_audit_does_not_exempt_the_auditor_source(tmp_path: Path) -> None:
    audit_source = tmp_path / "scripts" / "check_independence.py"
    audit_source.parent.mkdir(parents=True)
    audit_source.write_text('SOCKET = "alice-brain.sock"\n', encoding="utf-8")

    with pytest.raises(AuditViolation, match="daemon service"):
        audit_project(tmp_path)


def test_project_audit_ignores_only_directories_inside_the_root(
    tmp_path: Path,
) -> None:
    project = tmp_path / "tests" / "fixture-project"
    _write_source(project, "import alice_brain\n")

    with pytest.raises(AuditViolation, match="forbidden import root"):
        audit_project(project)


@pytest.mark.parametrize("dependency", ["alice-brain>=1", "alice_brain ~= 2.0"])
def test_wheel_audit_rejects_pep503_normalized_alice_brain_dependency(
    tmp_path: Path, dependency: str
) -> None:
    wheel = tmp_path / "fixture-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, dependencies=(dependency,))

    with pytest.raises(AuditViolation, match="forbidden distribution dependency"):
        audit_wheel(wheel)


def test_wheel_audit_allows_hermes_name_and_namespace(tmp_path: Path) -> None:
    wheel = tmp_path / "fixture-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, dependencies=("alice-brain-hermes>=0.1",))

    audit_wheel(wheel)


def test_wheel_audit_checks_packaged_python_ast(tmp_path: Path) -> None:
    wheel = tmp_path / "fixture-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, source="from alice_brain.runtime import DaemonClient\n")

    with pytest.raises(AuditViolation, match="forbidden import root"):
        audit_wheel(wheel)
