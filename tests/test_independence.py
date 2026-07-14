from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.check_independence import AuditViolation, audit_project, audit_wheel, main


def _write_source(root: Path, source: str) -> None:
    path = root / "src" / "fixture.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")


def _write_wheel(
    path: Path,
    *,
    name: str = "alice-brain-hermes",
    version: str = "0.1.0",
    dependencies: tuple[str, ...] = (),
    source: str = "from alice_brain_hermes import __version__\n",
    entry_points: str | None = None,
) -> None:
    requires_dist = "".join(f"Requires-Dist: {item}\n" for item in dependencies)
    metadata_text = (
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
        f"{requires_dist}\n"
    )
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("fixture/__init__.py", source)
        archive.writestr(f"{dist_info}/METADATA", metadata_text)
        if entry_points is not None:
            archive.writestr(f"{dist_info}/entry_points.txt", entry_points)


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
    "source",
    [
        'import importlib\nimportlib.import_module("alice_brain")\n',
        'import importlib as loader\nloader.import_module("alice_brain.runtime")\n',
        'from importlib import import_module\nimport_module("alice_brain")\n',
        (
            'from importlib import import_module as load_module\n'
            'load_module("alice_brain.runtime")\n'
        ),
        'import builtins\nbuiltins.__import__("alice_brain")\n',
        'import builtins as runtime\nruntime.__import__("alice_brain.runtime")\n',
        'from builtins import __import__ as load\nload("alice_brain")\n',
        '__import__("alice_" + "brain")\n',
        (
            'import importlib as loader\n'
            'TARGET = "alice_" + "brain"\n'
            'loader.import_module(TARGET)\n'
        ),
        'from . import alice_brain as runtime\n',
        'from .. import alice_brain\n',
    ],
)
def test_project_audit_rejects_statically_resolvable_import_bypasses(
    tmp_path: Path, source: str
) -> None:
    _write_source(tmp_path, source)

    with pytest.raises(AuditViolation, match="forbidden import root"):
        audit_project(tmp_path)


def test_project_audit_allows_dynamic_hermes_namespace(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        'import importlib as loader\n'
        'loader.import_module("alice_brain_hermes.runtime")\n',
    )

    audit_project(tmp_path)


@pytest.mark.parametrize("suffix", [".toml", ".yaml", ".ini"])
def test_project_audit_rejects_exact_root_executable_config_values(
    tmp_path: Path, suffix: str
) -> None:
    config = tmp_path / f"plugin{suffix}"
    config.write_text('handler = "alice_brain.cli:main"\n', encoding="utf-8")

    with pytest.raises(AuditViolation, match="forbidden executable import root"):
        audit_project(tmp_path)


def test_project_audit_allows_hermes_executable_config_value(tmp_path: Path) -> None:
    (tmp_path / "plugin.toml").write_text(
        '[project.entry-points."hermes_agent.plugins"]\n'
        'alice-brain = "alice_brain_hermes.hermes_plugin"\n',
        encoding="utf-8",
    )

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


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('HOME = "~/.alice-brain"\n', "sibling Alice-brain checkout path"),
        ('CHECKOUT = "../alice_brain"\n', "sibling Alice-brain checkout path"),
        ('STATE = "/tmp/_alice_brain/state"\n', "sibling Alice-brain checkout path"),
        (
            'HOME = "~/" + ".alice-" + "brain"\n',
            "sibling Alice-brain checkout path",
        ),
        (
            'HOME_VARIABLE = "ALICE_" + "BRAIN_HOME"\n',
            "Alice-brain environment variable",
        ),
        (
            'COMMAND = ["uv", "run", "alice-brain", "daemon"]\n',
            "daemon service",
        ),
        (
            'BINARY = "alice-" + "brain"\n'
            'COMMAND = ["uv", "run", BINARY, "daemon"]\n',
            "daemon service",
        ),
    ],
)
def test_project_audit_rejects_hidden_folded_and_wrapped_runtime_references(
    tmp_path: Path, source: str, message: str
) -> None:
    _write_source(tmp_path, source)

    with pytest.raises(AuditViolation, match=message):
        audit_project(tmp_path)


def test_project_audit_allows_hermes_state_and_wrapped_command(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        'STATE = "~/.alice-brain-hermes"\n'
        'COMMAND = ["uv", "run", "alice-brain-hermes", "daemon"]\n',
    )

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


@pytest.mark.parametrize("nested_name", ["tests", "docs", "dist"])
def test_project_audit_never_ignores_nested_shipping_source_directories(
    tmp_path: Path, nested_name: str
) -> None:
    source = (
        tmp_path
        / "src"
        / "alice_brain_hermes"
        / nested_name
        / "forbidden_bridge.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("import alice_brain\n", encoding="utf-8")

    with pytest.raises(AuditViolation, match="forbidden import root"):
        audit_project(tmp_path)


@pytest.mark.parametrize("root_name", ["tests", "docs", "dist"])
def test_project_audit_ignores_root_only_nonshipping_directories(
    tmp_path: Path, root_name: str
) -> None:
    source = tmp_path / root_name / "negative_fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("import alice_brain\n", encoding="utf-8")

    audit_project(tmp_path)


@pytest.mark.parametrize(
    "dependency",
    [
        "alice-brain>=1",
        "alice_brain ~= 2.0",
        "alice.brain>=3",
        'alice.brain[daemon]>=3; python_version >= "3.11"',
        "alice_brain[cli] @ https://example.invalid/alice_brain-1.0.whl",
        'alice-brain @ file:///tmp/alice-brain; os_name == "posix"',
    ],
)
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


def test_wheel_audit_checks_packaged_entry_points(tmp_path: Path) -> None:
    wheel = tmp_path / "alice_brain_hermes-0.1.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        entry_points="[console_scripts]\nlegacy = alice_brain.cli:main\n",
    )

    with pytest.raises(AuditViolation, match="forbidden executable import root"):
        audit_wheel(wheel)


def test_wheel_audit_allows_hermes_entry_points(tmp_path: Path) -> None:
    wheel = tmp_path / "alice_brain_hermes-0.1.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        entry_points=(
            "[console_scripts]\n"
            "alice-brain-hermes = alice_brain_hermes.cli:main\n"
        ),
    )

    audit_wheel(wheel)


def test_wheel_audit_rejects_unrelated_distribution(tmp_path: Path) -> None:
    wheel = tmp_path / "fixture-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="fixture")

    with pytest.raises(AuditViolation, match="expected distribution Name"):
        audit_wheel(wheel)


def test_release_audit_requires_a_wheel_unless_project_only(
    tmp_path: Path,
) -> None:
    assert main(["--root", str(tmp_path)]) == 1
    assert main(["--root", str(tmp_path), "--project-only"]) == 0


def test_release_audit_requires_project_version_match(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "alice-brain-hermes"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    wheel = tmp_path / "alice_brain_hermes-9.9.9-py3-none-any.whl"
    _write_wheel(wheel, version="9.9.9")

    assert main(["--root", str(tmp_path), str(wheel)]) == 1
