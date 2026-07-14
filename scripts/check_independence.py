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

FORBIDDEN_IMPORT_ROOT = "_".join(("alice", "brain"))
FORBIDDEN_DISTRIBUTION = "-".join(("alice", "brain"))
FORBIDDEN_HOME = "_".join(("ALICE", "BRAIN", "HOME"))
TARGET_DISTRIBUTION = "alice-brain-hermes"

_DISTRIBUTION_NAME = re.compile(r"^\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_PATH_COMPONENT = re.compile(
    r'''[\\/](?:[._]?alice[-_]brain)(?=$|[\\/"'\s,\]\)}])''',
    flags=re.IGNORECASE,
)
_HOME_TOKEN = re.compile(rf"(?<![A-Z0-9_]){re.escape(FORBIDDEN_HOME)}(?![A-Z0-9_])")
_EXECUTABLE_IMPORT_ROOT = re.compile(
    rf"(?<![A-Za-z0-9_]){re.escape(FORBIDDEN_IMPORT_ROOT)}(?![A-Za-z0-9_])"
)
_DAEMON_SERVICE = re.compile(
    r"(?<![A-Za-z0-9_-])alice-brain(?:\.sock|://|[-_]daemon|\s+daemon|/daemon)",
    flags=re.IGNORECASE,
)
_SCANNED_CONFIG_SUFFIXES = {".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"}
_IGNORED_DIRECTORIES_ANYWHERE = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
_IGNORED_ROOT_DIRECTORIES = {
    ".superpowers",
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


def _static_string(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left, bindings)
        right = _static_string(node.right, bindings)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                part = _static_string(value.value, bindings)
            else:
                part = _static_string(value, bindings)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    return None


def _static_bindings(tree: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    body = getattr(tree, "body", ())
    for statement in body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
        if isinstance(target, ast.Name) and value is not None:
            resolved = _static_string(value, bindings)
            if resolved is None:
                bindings.pop(target.id, None)
            else:
                bindings[target.id] = resolved
    return bindings


def _import_aliases(
    tree: ast.AST,
) -> tuple[set[str], set[str], set[str]]:
    importlib_modules: set[str] = set()
    builtins_modules: set[str] = set()
    import_functions = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", 1)[0]
                if alias.name == "importlib":
                    importlib_modules.add(bound_name)
                elif alias.name == "builtins":
                    builtins_modules.add(bound_name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_functions.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
            for alias in node.names:
                if alias.name == "__import__":
                    import_functions.add(alias.asname or alias.name)
    return importlib_modules, builtins_modules, import_functions


def _import_violations(
    tree: ast.AST, location: str, bindings: dict[str, str]
) -> list[str]:
    violations: list[str] = []
    importlib_modules, builtins_modules, import_functions = _import_aliases(tree)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
            elif node.level:
                names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            is_import_call = False
            if isinstance(node.func, ast.Name):
                is_import_call = node.func.id in import_functions
            elif isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                owner = node.func.value.id
                is_import_call = (
                    node.func.attr == "import_module" and owner in importlib_modules
                ) or (node.func.attr == "__import__" and owner in builtins_modules)
            if is_import_call:
                resolved = _static_string(node.args[0], bindings)
                if resolved is not None:
                    names.append(resolved)

        for name in names:
            if name.split(".", 1)[0] == FORBIDDEN_IMPORT_ROOT:
                line = getattr(node, "lineno", "?")
                violations.append(
                    f"{location}:{line}: forbidden import root "
                    f"{FORBIDDEN_IMPORT_ROOT!r}"
                )
    return violations


def _literal_values(
    tree: ast.AST, bindings: dict[str, str]
) -> Iterable[tuple[int | str, str]]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr, ast.Name)):
            value = _static_string(node, bindings)
            if value is not None:
                yield getattr(node, "lineno", "?"), value


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


def _command_sequence_violations(
    tree: ast.AST, location: str, bindings: dict[str, str]
) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        items = [_static_string(item, bindings) for item in node.elts]
        if len(items) < 2:
            continue
        normalized = [
            item.strip().lower() if item is not None else None for item in items
        ]
        for index, item in enumerate(normalized):
            if item not in {FORBIDDEN_DISTRIBUTION, FORBIDDEN_IMPORT_ROOT}:
                continue
            if any(later == "daemon" for later in normalized[index + 1 :]):
                line = getattr(node, "lineno", "?")
                violations.append(
                    f"{location}:{line}: forbidden external daemon service"
                )
                break
    return violations


def _python_violations(source: str, location: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=location)
    except SyntaxError as error:
        raise AuditViolation(
            f"{location}: cannot audit invalid Python: {error}"
        ) from error

    bindings = _static_bindings(tree)
    violations = _import_violations(tree, location, bindings)
    violations.extend(
        _external_reference_violations(_literal_values(tree, bindings), location)
    )
    violations.extend(_command_sequence_violations(tree, location, bindings))
    return violations


def _config_violations(text: str, location: str) -> list[str]:
    values = list(enumerate(text.splitlines(), 1))
    violations = _external_reference_violations(values, location)
    for line_number, line in values:
        if _EXECUTABLE_IMPORT_ROOT.search(line):
            violations.append(
                f"{location}:{line_number}: forbidden executable import root "
                f"{FORBIDDEN_IMPORT_ROOT!r}"
            )
    return violations


def _project_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        relative_directories = path.relative_to(root).parts[:-1]
        root_directory = relative_directories[0] if relative_directories else None
        if not path.is_file():
            continue
        if root_directory in _IGNORED_ROOT_DIRECTORIES or any(
            part in _IGNORED_DIRECTORIES_ANYWHERE for part in relative_directories
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


def audit_wheel(
    path: Path,
    *,
    expected_name: str = TARGET_DISTRIBUTION,
    expected_version: str | None = None,
) -> None:
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
                actual_name = message.get("Name")
                if (
                    actual_name is None
                    or _canonicalize_distribution(actual_name)
                    != _canonicalize_distribution(expected_name)
                ):
                    violations.append(
                        f"{path}!{metadata_names[0]}: expected distribution Name "
                        f"{expected_name!r}, found {actual_name!r}"
                    )
                actual_version = message.get("Version")
                if expected_version is not None and actual_version != expected_version:
                    violations.append(
                        f"{path}!{metadata_names[0]}: expected Version "
                        f"{expected_version!r}, found {actual_version!r}"
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
                elif (
                    Path(name).suffix in _SCANNED_CONFIG_SUFFIXES
                    or name.endswith(".dist-info/entry_points.txt")
                ):
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
    parser.add_argument(
        "--project-only",
        action="store_true",
        help="audit only project source; release mode otherwise requires wheel files",
    )
    return parser


def _project_version(pyproject: Path) -> str | None:
    if not pyproject.exists():
        return None
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    return version if isinstance(version, str) else None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        audit_project(args.root)
        if args.project_only:
            if args.wheels:
                raise AuditViolation("--project-only does not accept wheel files")
        else:
            if not args.wheels:
                raise AuditViolation(
                    "release wheel audit requires at least one wheel; "
                    "use --project-only for a source-only audit"
                )
            expected_version = _project_version(args.root / "pyproject.toml")
            for wheel in args.wheels:
                audit_wheel(wheel, expected_version=expected_version)
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
