#!/usr/bin/env python3
"""Audit source trees and wheels for dependencies on the separate project."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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


_IMPORT_FUNCTION = "import_function"
_IMPORTLIB_MODULE = "importlib_module"
_BUILTINS_MODULE = "builtins_module"
_PATH_CONSTRUCTOR = "path_constructor"
_PATHLIB_MODULE = "pathlib_module"
_OS_MODULE = "os_module"
_OS_PATH_MODULE = "os_path_module"
_PATH_JOIN = "path_join"
_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


@dataclass
class _StaticState:
    strings: dict[str, str] = field(default_factory=dict)
    symbols: dict[str, str] = field(
        default_factory=lambda: {"__import__": _IMPORT_FUNCTION}
    )

    def clone(self) -> _StaticState:
        return _StaticState(strings=self.strings.copy(), symbols=self.symbols.copy())

    def discard(self, name: str) -> None:
        self.strings.pop(name, None)
        self.symbols.pop(name, None)


def _resolve_symbol(node: ast.AST, state: _StaticState) -> str | None:
    if isinstance(node, ast.Name):
        return state.symbols.get(node.id)
    if not isinstance(node, ast.Attribute):
        return None

    owner = _resolve_symbol(node.value, state)
    if owner == _IMPORTLIB_MODULE and node.attr == "import_module":
        return _IMPORT_FUNCTION
    if owner == _BUILTINS_MODULE and node.attr == "__import__":
        return _IMPORT_FUNCTION
    if owner == _PATHLIB_MODULE and node.attr == "Path":
        return _PATH_CONSTRUCTOR
    if owner == _OS_MODULE and node.attr == "path":
        return _OS_PATH_MODULE
    if owner == _OS_PATH_MODULE and node.attr == "join":
        return _PATH_JOIN
    return None


def _join_static_path(parts: Sequence[str]) -> str:
    result = parts[0]
    for part in parts[1:]:
        if not result or result.endswith(("/", "\\")):
            result += part.lstrip("/\\")
        else:
            result += "/" + part.lstrip("/\\")
    return result


def _static_string(node: ast.AST, state: _StaticState) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return state.strings.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left, state)
        right = _static_string(node.right, state)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _static_string(node.left, state)
        right = _static_string(node.right, state)
        if left is not None and right is not None:
            return _join_static_path((left, right))
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                part = _static_string(value.value, state)
            else:
                part = _static_string(value, state)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    if isinstance(node, ast.Call):
        symbol = _resolve_symbol(node.func, state)
        if symbol == _PATH_CONSTRUCTOR and len(node.args) == 1 and not node.keywords:
            return _static_string(node.args[0], state)
        if symbol == _PATH_JOIN and node.args and not node.keywords:
            parts = [_static_string(argument, state) for argument in node.args]
            if all(part is not None for part in parts):
                return _join_static_path([part for part in parts if part is not None])
    return None


def _bound_import_symbol(alias: ast.alias) -> tuple[str, str | None]:
    if alias.asname is not None:
        bound_name = alias.asname
        symbol = {
            "importlib": _IMPORTLIB_MODULE,
            "builtins": _BUILTINS_MODULE,
            "pathlib": _PATHLIB_MODULE,
            "os": _OS_MODULE,
            "os.path": _OS_PATH_MODULE,
        }.get(alias.name)
        return bound_name, symbol

    root = alias.name.split(".", 1)[0]
    symbol = {
        "importlib": _IMPORTLIB_MODULE,
        "builtins": _BUILTINS_MODULE,
        "pathlib": _PATHLIB_MODULE,
        "os": _OS_MODULE,
    }.get(root)
    return root, symbol


def _bind_import(statement: ast.Import | ast.ImportFrom, state: _StaticState) -> None:
    if isinstance(statement, ast.Import):
        for alias in statement.names:
            name, symbol = _bound_import_symbol(alias)
            state.discard(name)
            if symbol is not None:
                state.symbols[name] = symbol
        return

    module_symbols = {
        ("importlib", "import_module"): _IMPORT_FUNCTION,
        ("builtins", "__import__"): _IMPORT_FUNCTION,
        ("pathlib", "Path"): _PATH_CONSTRUCTOR,
        ("os", "path"): _OS_PATH_MODULE,
        ("os.path", "join"): _PATH_JOIN,
    }
    for alias in statement.names:
        name = alias.asname or alias.name
        state.discard(name)
        symbol = module_symbols.get((statement.module, alias.name))
        if symbol is not None:
            state.symbols[name] = symbol


def _assignment_value(
    statement: ast.stmt,
) -> tuple[ast.Name | None, ast.expr | None]:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        return (target if isinstance(target, ast.Name) else None), statement.value
    if isinstance(statement, ast.AnnAssign):
        target = statement.target
        return (target if isinstance(target, ast.Name) else None), statement.value
    return None, None


def _bind_statement(statement: ast.stmt, state: _StaticState) -> None:
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        _bind_import(statement, state)
        return

    target, value = _assignment_value(statement)
    if target is not None:
        state.discard(target.id)
        if value is None:
            return
        string = _static_string(value, state)
        symbol = _resolve_symbol(value, state)
        if string is not None:
            state.strings[target.id] = string
        elif symbol is not None:
            state.symbols[target.id] = symbol
        return

    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        state.discard(statement.name)


def _parameter_names(scope: ast.AST) -> set[str]:
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return set()
    arguments = scope.args
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


def _scope_body(scope: ast.AST) -> Sequence[ast.stmt]:
    body_scopes = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if isinstance(scope, body_scopes):
        return scope.body
    return ()


def _scope_states(tree: ast.AST) -> dict[int, _StaticState]:
    states: dict[int, _StaticState] = {}

    def build(scope: ast.AST, inherited: _StaticState) -> None:
        state = inherited.clone()
        for name in _parameter_names(scope):
            state.discard(name)
        for statement in _scope_body(scope):
            _bind_statement(statement, state)

        def attach(node: ast.AST) -> None:
            states[id(node)] = state
            for child in ast.iter_child_nodes(node):
                if child is not scope and isinstance(child, _SCOPES):
                    build(child, state)
                else:
                    attach(child)

        attach(scope)

    build(tree, _StaticState())
    return states


def _import_violations(
    tree: ast.AST, location: str, states: dict[int, _StaticState]
) -> list[str]:
    violations: list[str] = []
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
            state = states[id(node)]
            if _resolve_symbol(node.func, state) == _IMPORT_FUNCTION:
                resolved = _static_string(node.args[0], state)
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
    tree: ast.AST, states: dict[int, _StaticState]
) -> Iterable[tuple[int | str, str]]:
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Constant, ast.BinOp, ast.JoinedStr, ast.Name, ast.Call)
        ):
            value = _static_string(node, states[id(node)])
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
    tree: ast.AST, location: str, states: dict[int, _StaticState]
) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        items = [_static_string(item, states[id(item)]) for item in node.elts]
        violation = _string_command_violation(
            items, location, getattr(node, "lineno", "?")
        )
        if violation is not None:
            violations.append(violation)
    return violations


def _python_violations(source: str, location: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=location)
    except SyntaxError as error:
        raise AuditViolation(
            f"{location}: cannot audit invalid Python: {error}"
        ) from error

    states = _scope_states(tree)
    violations = _import_violations(tree, location, states)
    violations.extend(
        _external_reference_violations(_literal_values(tree, states), location)
    )
    violations.extend(_command_sequence_violations(tree, location, states))
    return violations


def _nested_command_sequences(value: object) -> Iterable[Sequence[str]]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _nested_command_sequences(child)
    elif isinstance(value, list):
        if len(value) >= 2 and all(isinstance(item, str) for item in value):
            yield value
        for child in value:
            yield from _nested_command_sequences(child)


def _parsed_config_sequences(text: str, location: str) -> Iterable[Sequence[str]]:
    member = location.rsplit("!", 1)[-1]
    suffix = Path(member).suffix.lower()
    parsed: object | None = None
    try:
        if suffix == ".toml":
            parsed = tomllib.loads(text)
        elif suffix == ".json":
            parsed = json.loads(text)
    except (tomllib.TOMLDecodeError, json.JSONDecodeError):
        parsed = None

    if parsed is not None:
        yield from _nested_command_sequences(parsed)


def _string_command_violation(
    items: Sequence[str | None], location: str, line: int | str
) -> str | None:
    normalized = [item.strip().lower() if item is not None else None for item in items]
    for index, item in enumerate(normalized):
        if item not in {FORBIDDEN_DISTRIBUTION, FORBIDDEN_IMPORT_ROOT}:
            continue
        if any(later == "daemon" for later in normalized[index + 1 :]):
            return f"{location}:{line}: forbidden external daemon service"
    return None


def _config_violations(text: str, location: str) -> list[str]:
    values = list(enumerate(text.splitlines(), 1))
    violations = _external_reference_violations(values, location)
    for sequence in _parsed_config_sequences(text, location):
        violation = _string_command_violation(sequence, location, "?")
        if violation is not None:
            violations.append(violation)
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


def _is_ignored_project_path(relative: Path) -> bool:
    directories = relative.parts if relative.suffix == "" else relative.parts[:-1]
    root_directory = directories[0] if directories else None
    return root_directory in _IGNORED_ROOT_DIRECTORIES or any(
        part in _IGNORED_DIRECTORIES_ANYWHERE for part in directories
    )


def _project_package_path_violations(root: Path) -> list[str]:
    violations: list[str] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        relative = path.relative_to(root)
        if _is_ignored_project_path(relative):
            continue
        if any(part.casefold() == FORBIDDEN_IMPORT_ROOT for part in relative.parts):
            violations.append(
                f"{relative}: forbidden package path component "
                f"{FORBIDDEN_IMPORT_ROOT!r}"
            )
    return violations


def _wheel_package_path_violation(path: Path, member: str) -> str | None:
    if any(
        part.casefold() == FORBIDDEN_IMPORT_ROOT
        for part in Path(member).parts
    ):
        return (
            f"{path}!{member}: forbidden package path component "
            f"{FORBIDDEN_IMPORT_ROOT!r}"
        )
    return None


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
    violations.extend(_project_package_path_violations(root))

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
                path_violation = _wheel_package_path_violation(path, name)
                if path_violation is not None:
                    violations.append(path_violation)
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
            if expected_version is None:
                raise AuditViolation(
                    "release wheel audit requires a static [project].version"
                )
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
