#!/usr/bin/env python3
"""Audit source trees and wheels for dependencies on the separate project."""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from collections.abc import Iterable, Sequence
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
from zipfile import BadZipFile, ZipFile

FORBIDDEN_IMPORT_ROOT = "alice" + "_brain"
FORBIDDEN_DISTRIBUTION = "alice" + "-brain"
FORBIDDEN_HOME = "ALICE" + "_BRAIN_HOME"

_DISTRIBUTION_NAME = re.compile(r"^\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_PATH_COMPONENT = re.compile(
    r'''(?<=[\\/])alice-brain(?=$|[\\/"'\s,\]\)}])''',
    flags=re.IGNORECASE,
)
_HOME_TOKEN = re.compile(rf"(?<![A-Z0-9_]){re.escape(FORBIDDEN_HOME)}(?![A-Z0-9_])")
_DAEMON_SERVICE = re.compile(
    r"(?<![A-Za-z0-9_-])alice-brain(?:\.sock|://|[-_]daemon|\s+daemon|/daemon)",
    flags=re.IGNORECASE,
)
_SCANNED_CONFIG_SUFFIXES = {".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"}
_IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".superpowers",
    ".venv",
    "dist",
    "docs",
    "tests",
}


class AuditViolation(RuntimeError):
    """Raised when an independence contract is violated."""


def _canonicalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str | None:
    match = _DISTRIBUTION_NAME.match(requirement)
    if match is None:
        return None
    return _canonicalize_distribution(match.group(1))


def _dependency_violations(requirements: Iterable[str], location: str) -> list[str]:
    violations: list[str] = []
    for requirement in requirements:
        if _requirement_name(requirement) == FORBIDDEN_DISTRIBUTION:
            violations.append(
                f"{location}: forbidden distribution dependency {requirement!r}"
            )
    return violations


def _import_violations(tree: ast.AST, location: str) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.append(node.args[0].value)

        for name in names:
            if name.split(".", 1)[0] == FORBIDDEN_IMPORT_ROOT:
                line = getattr(node, "lineno", "?")
                violations.append(
                    f"{location}:{line}: forbidden import root "
                    f"{FORBIDDEN_IMPORT_ROOT!r}"
                )
    return violations


def _literal_values(tree: ast.AST) -> Iterable[tuple[int | str, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield getattr(node, "lineno", "?"), node.value


def _external_reference_violations(
    values: Iterable[tuple[int | str, str]], location: str
) -> list[str]:
    violations: list[str] = []
    for line, value in values:
        prefix = f"{location}:{line}"
        if _PATH_COMPONENT.search(value):
            violations.append(f"{prefix}: sibling Alice-brain checkout path")
        if _HOME_TOKEN.search(value):
            violations.append(f"{prefix}: Alice-brain environment variable")
        if _DAEMON_SERVICE.search(value):
            violations.append(f"{prefix}: forbidden external daemon service")
    return violations


def _command_sequence_violations(tree: ast.AST, location: str) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        items = [
            item.value
            for item in node.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
        if len(items) < 2:
            continue
        if items[0].lower() == FORBIDDEN_DISTRIBUTION and any(
            item.lower() == "daemon" for item in items[1:]
        ):
            line = getattr(node, "lineno", "?")
            violations.append(f"{location}:{line}: forbidden external daemon service")
    return violations


def _python_violations(source: str, location: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=location)
    except SyntaxError as error:
        raise AuditViolation(
            f"{location}: cannot audit invalid Python: {error}"
        ) from error

    violations = _import_violations(tree, location)
    violations.extend(_external_reference_violations(_literal_values(tree), location))
    violations.extend(_command_sequence_violations(tree, location))
    return violations


def _config_violations(text: str, location: str) -> list[str]:
    values = (
        (line_number, line) for line_number, line in enumerate(text.splitlines(), 1)
    )
    return _external_reference_violations(values, location)


def _project_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        relative_directories = path.relative_to(root).parts[:-1]
        if not path.is_file() or any(
            part in _IGNORED_DIRECTORIES for part in relative_directories
        ):
            continue
        if path.suffix == ".py" or path.suffix in _SCANNED_CONFIG_SUFFIXES:
            yield path


def _declared_requirements(pyproject: Path) -> Iterable[str]:
    if not pyproject.exists():
        return ()
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    build_system = data.get("build-system", {})
    dependency_groups = data.get("dependency-groups", {})

    requirements: list[str] = list(project.get("dependencies", ()))
    for group in project.get("optional-dependencies", {}).values():
        requirements.extend(item for item in group if isinstance(item, str))
    for group in dependency_groups.values():
        requirements.extend(item for item in group if isinstance(item, str))
    requirements.extend(
        item for item in build_system.get("requires", ()) if isinstance(item, str)
    )
    return requirements


def _raise_violations(violations: Sequence[str]) -> None:
    if violations:
        raise AuditViolation("independence audit failed:\n- " + "\n- ".join(violations))


def audit_project(root: Path) -> None:
    """Audit shipping source and configuration beneath *root*."""
    root = root.resolve()
    violations: list[str] = []
    if (root / ".gitmodules").exists():
        violations.append(f"{root / '.gitmodules'}: git submodules are forbidden")

    violations.extend(
        _dependency_violations(
            _declared_requirements(root / "pyproject.toml"),
            str(root / "pyproject.toml"),
        )
    )
    for path in _project_files(root):
        text = path.read_text(encoding="utf-8")
        location = str(path.relative_to(root))
        if path.suffix == ".py":
            violations.extend(_python_violations(text, location))
        else:
            violations.extend(_config_violations(text, location))

    _raise_violations(violations)


def audit_wheel(path: Path) -> None:
    """Audit dependency metadata and packaged source in a wheel."""
    violations: list[str] = []
    try:
        with ZipFile(path) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                violations.append(
                    f"{path}: expected exactly one wheel METADATA file, found "
                    f"{len(metadata_names)}"
                )
            else:
                message = BytesParser(policy=compat32).parsebytes(
                    archive.read(metadata_names[0])
                )
                violations.extend(
                    _dependency_violations(
                        message.get_all("Requires-Dist", []),
                        f"{path}!{metadata_names[0]}",
                    )
                )

            for name in archive.namelist():
                if name.endswith(".py"):
                    source = archive.read(name).decode("utf-8")
                    violations.extend(_python_violations(source, f"{path}!{name}"))
                elif Path(name).suffix in _SCANNED_CONFIG_SUFFIXES:
                    text = archive.read(name).decode("utf-8")
                    violations.extend(_config_violations(text, f"{path}!{name}"))
    except (BadZipFile, OSError, UnicodeDecodeError) as error:
        raise AuditViolation(f"{path}: cannot audit wheel: {error}") from error

    _raise_violations(violations)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wheels",
        nargs="*",
        type=Path,
        help="built wheel files whose metadata and packaged source must be audited",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="project root to audit (default: current directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        audit_project(args.root)
        for wheel in args.wheels:
            audit_wheel(wheel)
    except AuditViolation as error:
        print(error, file=sys.stderr)
        return 1

    print(
        f"independence audit passed: project={args.root.resolve()} "
        f"wheels={len(args.wheels)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
