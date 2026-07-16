# ruff: noqa: E501
from __future__ import annotations

import ast
import hashlib
import json
import re
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
TEST_ROOT = PROJECT_ROOT / "tests"
# These lexical rules intentionally reject references in comments and docstrings too,
# so removed platform implementations cannot linger as copy-paste source templates.
FORBIDDEN_LINE_PATTERNS = (
    ("procfs path", re.compile(r"/proc(?:/|\b)")),
    (
        "platform-specific process marker literal",
        re.compile(r"(?i)\b(?:linux|darwin|macos|windows|win32):"),
    ),
    ("renameat2", re.compile(r"\brenameat2\b")),
    ("direct fcntl", re.compile(r"\bfcntl\b")),
    ("dir_fd", re.compile(r"\bdir_fd\b")),
    ("pass_fds", re.compile(r"\bpass_fds\b")),
    ("start_new_session", re.compile(r"\bstart_new_session\b")),
    ("Windows creationflags", re.compile(r"\bcreationflags\b")),
    (
        "platform-only readiness fd/handle",
        re.compile(r"\breadiness[-_](?:fd|handle)s?\b", re.IGNORECASE),
    ),
    ("Windows STARTUPINFO", re.compile(r"\bSTARTUPINFO\b", re.IGNORECASE)),
    ("Windows handle_list", re.compile(r"\bhandle_list\b", re.IGNORECASE)),
)
PLATFORM_ONLY_STDLIB_MODULES = frozenset(
    {
        "grp",
        "msvcrt",
        "nt",
        "posix",
        "pty",
        "pwd",
        "resource",
        "syslog",
        "termios",
        "winreg",
        "winsound",
    }
)
PLATFORM_NAMES = frozenset(
    {
        "darwin",
        "linux",
        "macos",
        "nt",
        "osx",
        "posix",
        "pty",
        "syslog",
        "unix",
        "win32",
        "windows",
        "winsound",
    }
)
RUNTIME_IMPORT_NAMES = frozenset({"readiness", "runtime"})
IGNORED_AST_FIELDS = frozenset({"type_params"})
FORK_ATTRIBUTES = frozenset({"fork", "register_at_fork"})
MODE_ATTRIBUTES = frozenset({"chmod", "fchmod"})
DESCRIPTOR_ATTRIBUTES = frozenset(
    {"close", "fsync", "ftruncate", "lseek", "open", "read", "write"}
)
STAT_SEMANTIC_ATTRIBUTES = frozenset(
    {"st_dev", "st_gid", "st_ino", "st_mode", "st_nlink", "st_uid"}
)
STAT_MODE_FUNCTIONS = frozenset(
    {
        "S_IFMT",
        "S_IMODE",
        "S_ISBLK",
        "S_ISCHR",
        "S_ISDIR",
        "S_ISFIFO",
        "S_ISLNK",
        "S_ISREG",
        "S_ISSOCK",
    }
)
RETAINED_SQLITE_NAMES = frozenset(
    {"RetainedSQLiteFiles", "open_retained", "retain_sqlite_files"}
)
PLATFORM_SKIP_NAMES = frozenset({"skip", "skipif", "xfail"})
NATIVE_TEST_MODULES = PLATFORM_ONLY_STDLIB_MODULES | {"fcntl"}
EXPECTED_PROJECT_DEPENDENCIES = (
    "packaging>=24,<27",
    "platformdirs>=4.9,<4.10",
    "portalocker>=3.2,<3.3",
    "pydantic>=2.12,<3",
    "psutil==7.2.2",
    "python-dmon==0.3.0",
    "PyYAML>=6,<7",
)


@dataclass(frozen=True, order=True)
class SourceViolation:
    path: str
    rule: str
    identity: str
    line_number: int
    source: str

    def render(self) -> str:
        return f"{self.path}:{self.line_number}: {self.rule}: {self.source}"


# Exact occurrence-count ratchet: additions and removals both fail until Tasks 2-3
# update this inventory. The final portable implementation must make it empty.
def _parse_source_policy_baseline(
    snapshot: str,
) -> Counter[tuple[str, str, str]]:
    baseline: Counter[tuple[str, str, str]] = Counter()
    for line in snapshot.strip().splitlines():
        path, rule, identity, count = line.split("|")
        baseline[(path, rule, identity)] = int(count)
    return baseline


KNOWN_SOURCE_POLICY_VIOLATIONS = _parse_source_policy_baseline(
    ""
)
# Exact legacy-test debt. Task 4 removes these entries as portable replacements
# land; this ratchet must be empty before the native collection gate can pass.
KNOWN_TEST_POLICY_VIOLATIONS = _parse_source_policy_baseline(
    ""
)


def _active_python_sources() -> tuple[Path, ...]:
    return tuple(sorted(SOURCE_ROOT.rglob("*.py")))


def _statement_at(tree: ast.AST, line_number: int, column: int) -> ast.stmt | None:
    candidates: list[ast.stmt] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt) or not hasattr(node, "end_lineno"):
            continue
        end_line = node.end_lineno
        end_column = node.end_col_offset
        if end_line is None or end_column is None:
            continue
        if not node.lineno <= line_number <= end_line:
            continue
        if line_number == node.lineno and column < node.col_offset:
            continue
        if line_number == end_line and column >= end_column:
            continue
        candidates.append(node)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda node: (
            (node.end_lineno or node.lineno) - node.lineno,
            -node.col_offset,
        ),
    )


def _normalized_ast(value: object) -> object:
    if isinstance(value, ast.AST):
        return [
            type(value).__name__,
            [
                [field_name, _normalized_ast(field_value)]
                for field_name, field_value in ast.iter_fields(value)
                if field_name not in IGNORED_AST_FIELDS
            ],
        ]
    if isinstance(value, list):
        return [_normalized_ast(item) for item in value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if value is Ellipsis:
        return ["ellipsis"]
    if isinstance(value, complex):
        return ["complex", repr(value)]
    return value


def _parent_links(
    tree: ast.AST,
) -> dict[ast.AST, tuple[ast.AST, str, int | None]]:
    links: dict[ast.AST, tuple[ast.AST, str, int | None]] = {}
    for parent in ast.walk(tree):
        for field_name, value in ast.iter_fields(parent):
            if isinstance(value, ast.AST):
                links[value] = (parent, field_name, None)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        links[child] = (parent, field_name, index)
    return links


def _enclosing_statement(
    anchor: ast.AST,
    links: dict[ast.AST, tuple[ast.AST, str, int | None]],
) -> ast.stmt | None:
    current = anchor
    while not isinstance(current, ast.stmt):
        link = links.get(current)
        if link is None:
            return None
        current = link[0]
    return current


def _control_context(parent: ast.AST, field_name: str) -> object | None:
    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [type(parent).__name__, parent.name, field_name]
    if isinstance(parent, ast.ClassDef):
        return ["ClassDef", parent.name, field_name]
    if isinstance(parent, ast.If):
        return ["If", field_name, _normalized_ast(parent.test)]
    if isinstance(parent, (ast.For, ast.AsyncFor)):
        return [
            type(parent).__name__,
            field_name,
            _normalized_ast(parent.target),
            _normalized_ast(parent.iter),
        ]
    if isinstance(parent, ast.While):
        return ["While", field_name, _normalized_ast(parent.test)]
    if isinstance(parent, (ast.With, ast.AsyncWith)):
        return [type(parent).__name__, field_name, _normalized_ast(parent.items)]
    if isinstance(parent, (ast.Try, ast.TryStar)):
        return [type(parent).__name__, field_name]
    if isinstance(parent, ast.ExceptHandler):
        return [
            "ExceptHandler",
            field_name,
            _normalized_ast(parent.type),
            parent.name,
        ]
    if isinstance(parent, ast.Match):
        return ["Match", field_name, _normalized_ast(parent.subject)]
    if isinstance(parent, ast.match_case):
        return [
            "match_case",
            field_name,
            _normalized_ast(parent.pattern),
            _normalized_ast(parent.guard),
        ]
    return None


def _enclosing_control_context(
    statement: ast.stmt,
    links: dict[ast.AST, tuple[ast.AST, str, int | None]],
) -> list[object]:
    contexts: list[object] = []
    current: ast.AST = statement
    while (link := links.get(current)) is not None:
        parent, field_name, _ = link
        descriptor = _control_context(parent, field_name)
        if descriptor is not None:
            contexts.append(descriptor)
        current = parent
    contexts.reverse()
    return contexts


def _source_identity(
    tree: ast.AST,
    source: str,
    line_number: int,
    column: int,
    *,
    anchor: ast.AST | None = None,
) -> str:
    links = _parent_links(tree)
    statement = (
        _enclosing_statement(anchor, links)
        if anchor is not None
        else _statement_at(tree, line_number, column)
    )
    normalized: object
    if statement is None:
        normalized = ["SourceLine", source.splitlines()[line_number - 1].strip()]
    else:
        normalized = [
            "SourceOccurrence",
            _enclosing_control_context(statement, links),
            _normalized_ast(statement),
        ]
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _platform_accesses(tree: ast.AST) -> tuple[tuple[ast.AST, str], ...]:
    module_aliases: dict[str, str] = {}
    violations: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"os", "sys"}:
                    module_aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module in {"os", "sys"}:
            for alias in node.names:
                if node.module == "sys" and alias.name in {"platform", "*"}:
                    violations.append((alias, "sys.platform"))
                elif node.module == "os" and alias.name in {"name", "*"}:
                    violations.append((alias, "os.name"))
                if node.module == "os" and alias.name in {"getuid", "geteuid", "*"}:
                    violations.append((alias, "os.getuid/os.geteuid"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        module = module_aliases.get(node.value.id)
        if module == "sys" and node.attr == "platform":
            violations.append((node, "sys.platform"))
        elif module == "os" and node.attr == "name":
            violations.append((node, "os.name"))
        elif module == "os" and node.attr in {"getuid", "geteuid"}:
            violations.append((node, "os.getuid/os.geteuid"))

    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "getattr"
            or len(node.args) < 2
            or not isinstance(node.args[0], ast.Name)
            or not isinstance(node.args[1], ast.Constant)
            or not isinstance(node.args[1].value, str)
        ):
            continue
        module = module_aliases.get(node.args[0].id)
        attribute = node.args[1].value
        if module == "sys" and attribute == "platform":
            violations.append((node, "sys.platform"))
        elif module == "os" and attribute == "name":
            violations.append((node, "os.name"))
        elif module == "os" and attribute in {"getuid", "geteuid"}:
            violations.append((node, "os.getuid/os.geteuid"))
    return tuple(violations)


def _name_components(*names: str) -> frozenset[str]:
    components: set[str] = set()
    for name in names:
        with_word_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
        components.update(
            component
            for component in re.split(r"[._-]", with_word_boundaries.casefold())
            if component
        )
    return frozenset(components)


def _import_bindings(
    tree: ast.AST,
) -> tuple[dict[str, str], dict[str, tuple[str, str]], frozenset[str]]:
    modules: dict[str, str] = {}
    symbols: dict[str, tuple[str, str]] = {}
    stars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", 1)[0]
                modules[bound_name] = alias.name if alias.asname else bound_name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    stars.add(module)
                else:
                    symbols[alias.asname or alias.name] = (module, alias.name)
    return modules, symbols, frozenset(stars)


def _resolved_access(
    node: ast.AST,
    modules: dict[str, str],
    symbols: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        module = modules.get(node.value.id)
        if module is not None:
            return module, node.attr
    if isinstance(node, ast.Name):
        return symbols.get(node.id)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        module = modules.get(node.args[0].id)
        if module is not None:
            return module, node.args[1].value
    return None


def _portable_rules_for_access(module: str, attribute: str) -> tuple[str, ...]:
    root = module.split(".", 1)[0]
    rules: list[str] = []
    if root == "socket" and attribute == "AF_UNIX":
        rules.append("AF_UNIX")
    if root == "os":
        if attribute in FORK_ATTRIBUTES:
            rules.append("fork/register_at_fork")
        if attribute.startswith("O_"):
            rules.append("os.O_* flags")
        if attribute in MODE_ATTRIBUTES:
            rules.append("chmod/fchmod")
        if attribute in DESCRIPTOR_ATTRIBUTES:
            rules.append("direct descriptor API")
        if attribute in {"major", "minor"}:
            rules.append("POSIX stat semantics")
    if root == "stat" and attribute in STAT_MODE_FUNCTIONS:
        rules.append("POSIX stat semantics")
    return tuple(rules)


def _portable_runtime_accesses(tree: ast.AST) -> tuple[tuple[ast.AST, str], ...]:
    modules, symbols, _ = _import_bindings(tree)
    violations: list[tuple[ast.AST, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        root = module.split(".", 1)[0]
        for alias in node.names:
            if alias.name == "*":
                wildcard_rules: tuple[str, ...] = ()
                if root == "socket":
                    wildcard_rules = ("AF_UNIX",)
                elif root == "os":
                    wildcard_rules = (
                        "fork/register_at_fork",
                        "os.O_* flags",
                        "chmod/fchmod",
                        "direct descriptor API",
                    )
                elif root == "stat":
                    wildcard_rules = ("POSIX stat semantics",)
                elif module.endswith("runtime.lease"):
                    wildcard_rules = ("retained SQLite descriptor API",)
                violations.extend((alias, rule) for rule in wildcard_rules)
                continue
            violations.extend(
                (alias, rule) for rule in _portable_rules_for_access(module, alias.name)
            )
            if alias.name in RETAINED_SQLITE_NAMES:
                violations.append((alias, "retained SQLite descriptor API"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in STAT_SEMANTIC_ATTRIBUTES:
                violations.append((node, "POSIX stat semantics"))
            if node.attr in RETAINED_SQLITE_NAMES:
                violations.append((node, "retained SQLite descriptor API"))
        elif isinstance(node, ast.Name) and node.id in RETAINED_SQLITE_NAMES:
            violations.append((node, "retained SQLite descriptor API"))

        access = _resolved_access(node, modules, symbols)
        if access is not None:
            violations.extend(
                (node, rule) for rule in _portable_rules_for_access(*access)
            )

        if not isinstance(node, ast.Call) or not node.args:
            continue
        callable_access = _resolved_access(node.func, modules, symbols)
        if callable_access != ("re", "compile"):
            continue
        pattern = node.args[0]
        if (
            isinstance(pattern, ast.Constant)
            and isinstance(pattern.value, str)
            and "[0-9a-f]" in pattern.value
            and "{8}" in pattern.value
            and ("{4}" in pattern.value or "{27}" in pattern.value)
        ):
            violations.append((node, "Linux process-marker regex"))

    return tuple(violations)


def _qualified_name(
    node: ast.AST,
    modules: dict[str, str],
    symbols: dict[str, tuple[str, str]],
) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in symbols:
            module, attribute = symbols[node.id]
            return f"{module}.{attribute}" if module else attribute
        return modules.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value, modules, symbols)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _has_platform_signal(
    node: ast.AST,
    modules: dict[str, str],
    symbols: dict[str, tuple[str, str]],
    *,
    platform_aliases: frozenset[str] = frozenset(),
    assigned_names: frozenset[str] = frozenset(),
) -> bool:
    condition_names = {
        "darwin",
        "linux",
        "macos",
        "nt",
        "osx",
        "posix",
        "unix",
        "win32",
        "windows",
    }
    for child in ast.walk(node):
        access = _resolved_access(child, modules, symbols)
        if access in {("os", "name"), ("sys", "platform")}:
            return True
        if access is not None and access[0].split(".", 1)[0] == "platform":
            return True
        if isinstance(child, ast.Name):
            if child.id in platform_aliases:
                return True
            if (
                child.id not in assigned_names
                and _name_components(child.id) & condition_names
            ):
                return True
        elif isinstance(child, ast.Attribute):
            if _name_components(child.attr) & condition_names:
                return True
        elif (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and _name_components(*re.findall(r"[A-Za-z0-9_-]+", child.value))
            & condition_names
        ):
            return True
    return False


def _enclosing_scope_is_module(
    node: ast.AST,
    links: dict[ast.AST, tuple[ast.AST, str, int | None]],
) -> bool:
    current = node
    while (link := links.get(current)) is not None:
        parent = link[0]
        if isinstance(parent, ast.Module):
            return True
        if isinstance(
            parent,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            return False
        current = parent
    return False


def _module_scope_assignments(
    tree: ast.AST,
    links: dict[ast.AST, tuple[ast.AST, str, int | None]],
) -> dict[str, tuple[ast.AST, ...]]:
    assignments: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if not _enclosing_scope_is_module(node, links):
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(value)
    return {name: tuple(values) for name, values in assignments.items()}


def _module_platform_aliases(
    tree: ast.AST,
    modules: dict[str, str],
    symbols: dict[str, tuple[str, str]],
    links: dict[ast.AST, tuple[ast.AST, str, int | None]],
) -> tuple[frozenset[str], frozenset[str]]:
    assignments = _module_scope_assignments(tree, links)
    assigned_names = frozenset(assignments)
    platform_aliases: set[str] = set()
    changed = True
    while changed:
        changed = False
        known_aliases = frozenset(platform_aliases)
        for name, values in assignments.items():
            if name in platform_aliases:
                continue
            if any(
                _has_platform_signal(
                    value,
                    modules,
                    symbols,
                    platform_aliases=known_aliases,
                    assigned_names=assigned_names,
                )
                for value in values
            ):
                platform_aliases.add(name)
                changed = True
    return frozenset(platform_aliases), assigned_names


def _platform_skip_calls(tree: ast.AST) -> tuple[tuple[ast.Call, str], ...]:
    modules, symbols, _ = _import_bindings(tree)
    links = _parent_links(tree)
    platform_aliases, assigned_names = _module_platform_aliases(
        tree, modules, symbols, links
    )
    violations: list[tuple[ast.Call, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qualified = _qualified_name(node.func, modules, symbols)
        if qualified is None:
            continue
        components = qualified.split(".")
        if (
            not components
            or components[-1] not in PLATFORM_SKIP_NAMES
            or components[0] != "pytest"
        ):
            continue
        platform_dependent = _has_platform_signal(
            node,
            modules,
            symbols,
            platform_aliases=platform_aliases,
            assigned_names=assigned_names,
        )
        current: ast.AST = node
        while not platform_dependent and (link := links.get(current)) is not None:
            parent = link[0]
            if isinstance(parent, (ast.If, ast.While)):
                platform_dependent = _has_platform_signal(
                    parent.test,
                    modules,
                    symbols,
                    platform_aliases=platform_aliases,
                    assigned_names=assigned_names,
                )
            current = parent
        if platform_dependent:
            violations.append((node, "platform-conditioned pytest skip"))
    return tuple(violations)


def _top_level_native_imports(tree: ast.AST) -> tuple[tuple[ast.alias, str], ...]:
    violations: list[tuple[ast.alias, str]] = []
    if not isinstance(tree, ast.Module):
        return ()
    links = _parent_links(tree)
    for node in ast.walk(tree):
        if not _enclosing_scope_is_module(node, links):
            continue
        if isinstance(node, ast.Import):
            aliases = node.names
            native_module = None
        elif isinstance(node, ast.ImportFrom):
            aliases = node.names
            native_module = (node.module or "").split(".", 1)[0]
        else:
            continue
        for alias in aliases:
            root = native_module or alias.name.split(".", 1)[0]
            if root in NATIVE_TEST_MODULES:
                violations.append((alias, "top-level native test import"))
    return tuple(violations)


def _import_violations(tree: ast.AST) -> tuple[tuple[ast.AST, str], ...]:
    violations: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = tuple((alias, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports = tuple(
                (alias, f"{module}.{alias.name}" if module else alias.name)
                for alias in node.names
            )
        else:
            continue
        for alias, imported_name in imports:
            root = imported_name.split(".", 1)[0]
            if root in PLATFORM_ONLY_STDLIB_MODULES:
                violations.append((alias, "platform-only stdlib import"))
                continue
            components = _name_components(imported_name, alias.asname or "")
            if components & PLATFORM_NAMES:
                violations.append((alias, "platform-specific runtime/readiness import"))
    return tuple(violations)


def _abstract_socket_calls(tree: ast.AST) -> tuple[ast.Call, ...]:
    assignments: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            assignments[node.target.id] = node.value

    def starts_with_null(value: ast.AST, seen: frozenset[str] = frozenset()) -> bool:
        if isinstance(value, ast.Constant) and isinstance(value.value, (bytes, str)):
            prefix = b"\0" if isinstance(value.value, bytes) else "\0"
            return value.value.startswith(prefix)
        if isinstance(value, ast.Name) and value.id not in seen:
            assigned = assignments.get(value.id)
            return assigned is not None and starts_with_null(
                assigned, seen | {value.id}
            )
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            return starts_with_null(value.left, seen)
        if isinstance(value, ast.JoinedStr) and value.values:
            return starts_with_null(value.values[0], seen)
        return False

    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"bind", "connect", "connect_ex"}
        and node.args
        and starts_with_null(node.args[0])
    )


def _source_violations_for(
    relative_path: str, source: str
) -> tuple[SourceViolation, ...]:
    violations: list[SourceViolation] = []
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=relative_path)

    def add(node: ast.AST, rule: str) -> None:
        line_number = node.lineno
        column = node.col_offset
        violations.append(
            SourceViolation(
                relative_path,
                rule,
                _source_identity(
                    tree,
                    source,
                    line_number,
                    column,
                    anchor=node,
                ),
                line_number,
                source_lines[line_number - 1].strip(),
            )
        )

    for line_number, line in enumerate(source_lines, start=1):
        for rule, pattern in FORBIDDEN_LINE_PATTERNS:
            for match in pattern.finditer(line):
                violations.append(
                    SourceViolation(
                        relative_path,
                        rule,
                        _source_identity(tree, source, line_number, match.start()),
                        line_number,
                        line.strip(),
                    )
                )

    for node, rule in (
        *_platform_accesses(tree),
        *_import_violations(tree),
        *_portable_runtime_accesses(tree),
        *_platform_skip_calls(tree),
        *_top_level_native_imports(tree),
    ):
        add(node, rule)

    for node in _abstract_socket_calls(tree):
        add(node, "Linux abstract socket address")

    return tuple(sorted(violations))


def _source_violations() -> tuple[SourceViolation, ...]:
    violations: list[SourceViolation] = []
    for path in _active_python_sources():
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        violations.extend(_source_violations_for(relative_path, source))

    return tuple(sorted(violations))


def test_production_source_matches_portability_migration_baseline() -> None:
    violations = _source_violations()
    actual = Counter(
        (violation.path, violation.rule, violation.identity) for violation in violations
    )
    expected = KNOWN_SOURCE_POLICY_VIOLATIONS

    assert actual == expected, (
        "portable source-policy baseline changed; remove fixed entries and reject "
        "new entries:\n"
        f"new={dict(actual - expected)!r}\n"
        f"fixed={dict(expected - actual)!r}\n"
        "current forbidden production source:\n"
        + "\n".join(violation.render() for violation in violations)
    )


@pytest.mark.parametrize(
    ("source", "expected_rule"),
    [
        ("import sys\nvalue = sys.platform\n", "sys.platform"),
        ("import sys as host_sys\nvalue = host_sys.platform\n", "sys.platform"),
        ("from sys import platform as host_platform\n", "sys.platform"),
        ("import os as host_os\nvalue = host_os.name\n", "os.name"),
        ("from os import name as host_name\n", "os.name"),
        ("from os import getuid as current_uid\n", "os.getuid/os.geteuid"),
        ('path = "/proc/self/fd/1"\n', "procfs path"),
        (
            'process_marker = "linux:boot-id:123"\n',
            "platform-specific process marker literal",
        ),
        ("libc.renameat2()\n", "renameat2"),
        ("import fcntl as locks\n", "direct fcntl"),
        ("from fcntl import flock\n", "direct fcntl"),
        ("os.open(path, flags, dir_fd=root)\n", "dir_fd"),
        ("Popen(argv, pass_fds=(fd,))\n", "pass_fds"),
        ("Popen(argv, start_new_session=True)\n", "start_new_session"),
        ("Popen(argv, creationflags=flags)\n", "Windows creationflags"),
        ("import msvcrt\n", "platform-only stdlib import"),
        (
            "from .runtime import windows\n",
            "platform-specific runtime/readiness import",
        ),
        (
            "from . import readiness_windows\n",
            "platform-specific runtime/readiness import",
        ),
        (
            "from .windows import ReadinessSignal\n",
            "platform-specific runtime/readiness import",
        ),
        (
            "from . import windows as readiness_backend\n",
            "platform-specific runtime/readiness import",
        ),
        (
            "import package.windows as runtime_backend\n",
            "platform-specific runtime/readiness import",
        ),
        (
            "from .osx import RuntimeLease\n",
            "platform-specific runtime/readiness import",
        ),
        ("import winsound\n", "platform-only stdlib import"),
        ("import pty\n", "platform-only stdlib import"),
        ("import syslog\n", "platform-only stdlib import"),
        ("import nt\n", "platform-only stdlib import"),
        (
            'import sys as host_sys\nvalue = getattr(host_sys, "platform")\n',
            "sys.platform",
        ),
        (
            'import os as host_os\nvalue = getattr(host_os, "name")\n',
            "os.name",
        ),
        ("from os import *\n", "os.name"),
        ("readiness_fd = 1\n", "platform-only readiness fd/handle"),
        ('argument = "--readiness-handle"\n', "platform-only readiness fd/handle"),
        ("startup = subprocess.STARTUPINFO()\n", "Windows STARTUPINFO"),
        ("startup.lpAttributeList = {'handle_list': []}\n", "Windows handle_list"),
        (
            'import socket\naddress = b"\\0runtime"\nsocket.socket().bind(address)\n',
            "Linux abstract socket address",
        ),
        (
            'import socket\naddress = "\\0runtime"\nsocket.socket().connect(address)\n',
            "Linux abstract socket address",
        ),
        ("import socket\nfamily = socket.AF_UNIX\n", "AF_UNIX"),
        ("import socket as net\nfamily = net.AF_UNIX\n", "AF_UNIX"),
        ("from socket import AF_UNIX as family\n", "AF_UNIX"),
        (
            'import socket as net\nfamily = getattr(net, "AF_UNIX")\n',
            "AF_UNIX",
        ),
        ("from socket import *\n", "AF_UNIX"),
        ("import os\npid = os.fork()\n", "fork/register_at_fork"),
        (
            "import os as host_os\nhost_os.register_at_fork(after_in_child=reset)\n",
            "fork/register_at_fork",
        ),
        ("from os import fork as spawn_child\n", "fork/register_at_fork"),
        (
            'import os as host_os\nspawn = getattr(host_os, "fork")\n',
            "fork/register_at_fork",
        ),
        ("import os\nflags = os.O_RDONLY | os.O_NOFOLLOW\n", "os.O_* flags"),
        ("from os import O_EXCL as exclusive\n", "os.O_* flags"),
        (
            'import os as host_os\nflag = getattr(host_os, "O_CLOEXEC", 0)\n',
            "os.O_* flags",
        ),
        ("import os\nos.chmod(path, 0o600)\n", "chmod/fchmod"),
        ("from os import fchmod as secure_descriptor\n", "chmod/fchmod"),
        ("owner = path.stat().st_uid\n", "POSIX stat semantics"),
        ("identity = (record.st_dev, record.st_ino)\n", "POSIX stat semantics"),
        ("mode = metadata.st_mode\n", "POSIX stat semantics"),
        ("links = metadata.st_nlink\n", "POSIX stat semantics"),
        ("import os\ndescriptor = os.open(path, flags)\n", "direct descriptor API"),
        ("from os import read as read_descriptor\n", "direct descriptor API"),
        (
            'import os as host_os\ncloser = getattr(host_os, "close")\n',
            "direct descriptor API",
        ),
        (
            'import re\n_BOOT_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f-]{27}$")\n',
            "Linux process-marker regex",
        ),
        (
            "owner: RetainedSQLiteFiles | None = None\n",
            "retained SQLite descriptor API",
        ),
        (
            "ledger = SQLiteLedger.open_retained(owner)\n",
            "retained SQLite descriptor API",
        ),
        (
            "owner = lease.retain_sqlite_files()\n",
            "retained SQLite descriptor API",
        ),
        (
            "from alice_brain_hermes.runtime.lease import "
            "RetainedSQLiteFiles as Owner\n",
            "retained SQLite descriptor API",
        ),
        (
            "from . import windows_backend\n",
            "platform-specific runtime/readiness import",
        ),
        (
            "from . import posix_impl\n",
            "platform-specific runtime/readiness import",
        ),
        (
            "from . import linux\n",
            "platform-specific runtime/readiness import",
        ),
        (
            "from . import darwin_support\n",
            "platform-specific runtime/readiness import",
        ),
        (
            '@pytest.mark.skipif(os.name == "nt", reason="POSIX only")\n'
            "def test_contract():\n    pass\n",
            "platform-conditioned pytest skip",
        ),
        (
            'if sys.platform == "win32":\n    pytest.xfail("not portable")\n',
            "platform-conditioned pytest skip",
        ),
        (
            'pytest.skip("unsupported on Windows")\n',
            "platform-conditioned pytest skip",
        ),
        (
            "import pytest as testing\nimport platform as host_platform\n"
            '@testing.mark.skipif(host_platform.system() == "Windows", '
            'reason="native")\ndef test_contract():\n    pass\n',
            "platform-conditioned pytest skip",
        ),
        (
            "from pytest import skip as skip_test\nimport os as host_os\n"
            'if host_os.name == "nt":\n    skip_test("native")\n',
            "platform-conditioned pytest skip",
        ),
        ("import fcntl as locks\n", "top-level native test import"),
        ("from msvcrt import locking\n", "top-level native test import"),
        (
            "if TYPE_CHECKING:\n    import fcntl as locks\n",
            "top-level native test import",
        ),
        (
            "try:\n    import msvcrt\nexcept ImportError:\n    pass\n",
            "top-level native test import",
        ),
        (
            "if use_native_backend:\n    from . import windows_backend\n",
            "platform-specific runtime/readiness import",
        ),
        (
            'import sys\nunsupported_host = sys.platform == "win32"\n'
            '@pytest.mark.skipif(unsupported_host, reason="native")\n'
            "def test_contract():\n    pass\n",
            "platform-conditioned pytest skip",
        ),
        (
            'import sys\nis_windows = sys.platform == "win32"\n'
            "unsupported_host = is_windows or forced_native_skip\n"
            '@pytest.mark.skipif(unsupported_host, reason="native")\n'
            "def test_contract():\n    pass\n",
            "platform-conditioned pytest skip",
        ),
        (
            'import sys\nunsupported_host = sys.platform == "win32"\n'
            "def test_contract():\n"
            "    if unsupported_host:\n"
            '        pytest.skip("native")\n',
            "platform-conditioned pytest skip",
        ),
    ],
)
def test_source_policy_recognizes_forbidden_bypasses(
    source: str, expected_rule: str
) -> None:
    violations = _source_violations_for("src/alice_brain_hermes/fixture.py", source)

    assert expected_rule in {violation.rule for violation in violations}


def test_abstract_socket_policy_ignores_unrelated_null_bytes() -> None:
    source = (
        'import socket\npadding = b"\\0" * 64\nsocket.socket().bind(("127.0.0.1", 0))\n'
    )

    violations = _source_violations_for("fixture.py", source)

    assert "Linux abstract socket address" not in {
        violation.rule for violation in violations
    }


def test_native_test_import_policy_ignores_function_local_import() -> None:
    source = "def test_lock():\n    import fcntl\n"

    violations = _source_violations_for("fixture.py", source)

    assert "top-level native test import" not in {
        violation.rule for violation in violations
    }


@pytest.mark.parametrize(
    "source",
    [
        (
            "feature_enabled = True\n"
            '@pytest.mark.skipif(feature_enabled, reason="optional feature")\n'
            "def test_contract():\n    pass\n"
        ),
        (
            "unsupported_host = False\n"
            "def test_contract():\n"
            "    if unsupported_host:\n"
            '        pytest.skip("optional feature")\n'
        ),
    ],
)
def test_platform_skip_policy_ignores_safe_nonplatform_controls(source: str) -> None:
    violations = _source_violations_for("fixture.py", source)

    assert "platform-conditioned pytest skip" not in {
        violation.rule for violation in violations
    }


def test_test_sources_have_no_uninventoried_portability_debt() -> None:
    test_rules = {
        "direct fcntl",
        "platform-conditioned pytest skip",
        "platform-only stdlib import",
        "platform-specific runtime/readiness import",
        "procfs path",
        "top-level native test import",
    }
    violations: list[SourceViolation] = []
    policy_path = Path(__file__).resolve()
    for path in sorted(TEST_ROOT.rglob("*.py")):
        if path.resolve() == policy_path:
            continue
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        violations.extend(
            violation
            for violation in _source_violations_for(relative_path, source)
            if violation.rule in test_rules
        )

    actual = Counter(
        (violation.path, violation.rule, violation.identity) for violation in violations
    )
    expected = KNOWN_TEST_POLICY_VIOLATIONS
    assert actual == expected, (
        "test portability-debt baseline changed; remove portable replacements "
        "and reject newly hidden tests:\n"
        f"new={dict(actual - expected)!r}\n"
        f"fixed={dict(expected - actual)!r}\n"
        "current test portability debt:\n"
        + "\n".join(violation.render() for violation in violations)
    )


def test_source_policy_scans_installed_package_entry_points() -> None:
    sources = set(_active_python_sources())

    assert SOURCE_ROOT == PROJECT_ROOT / "src"
    assert sources
    assert PROJECT_ROOT / "src" / "alice_brain_hermes" / "__init__.py" in sources
    assert PROJECT_ROOT / "src" / "alice_brain_hermes" / "cli.py" in sources


def _violation_identities(source: str) -> Counter[tuple[str, str]]:
    return Counter(
        (violation.rule, violation.identity)
        for violation in _source_violations_for("fixture.py", source)
    )


def test_source_policy_identity_ignores_formatting_and_unrelated_lines() -> None:
    original = "import os as host_os\nvalue=host_os.name\n"
    reformatted = (
        "# unrelated comment\n\n"
        "import os as host_os\n"
        "value = host_os.name  # formatting only\n"
    )

    assert _violation_identities(original) == _violation_identities(reformatted)


def test_source_policy_identity_rejects_same_count_statement_replacement() -> None:
    original = "import os as host_os\nfirst = host_os.name\n"
    replacement = "import os as host_os\nsecond = host_os.name\n"

    assert _violation_identities(original) != _violation_identities(replacement)


def test_source_policy_identity_includes_enclosing_control_flow() -> None:
    true_branch = "import os\nif enabled:\n    value = os.name\n"
    false_branch = "import os\nif enabled:\n    pass\nelse:\n    value = os.name\n"

    assert _violation_identities(true_branch) != _violation_identities(false_branch)


def test_source_policy_identity_preserves_duplicate_occurrence_counts() -> None:
    source = "import os as host_os\nvalue = host_os.name\nvalue = host_os.name\n"

    assert _violation_identities(source).total() == 2
    assert set(_violation_identities(source).values()) == {2}


@pytest.mark.parametrize(
    ("source", "expected_identity"),
    [
        ("value = host_os.name\n", "07d47991228a6ce7"),
        ('path = Path(f"/proc/{pid}/stat")\n', "cda4c98e887832ef"),
        (
            "def launch(*, readiness_fd: int | None = None):\n    pass\n",
            "84aabbe5b84d8220",
        ),
    ],
)
def test_source_policy_identity_is_stable_across_python_versions(
    source: str, expected_identity: str
) -> None:
    tree = ast.parse(source)

    assert _source_identity(tree, source, 1, 0) == expected_identity


def test_project_declares_only_approved_direct_dependencies() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        dependencies = tuple(tomllib.load(stream)["project"]["dependencies"])

    assert dependencies == EXPECTED_PROJECT_DEPENDENCIES


def test_uv_lock_matches_every_approved_direct_dependency() -> None:
    with (PROJECT_ROOT / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)

    requirements = {
        Requirement(dependency).name.casefold(): Requirement(dependency)
        for dependency in EXPECTED_PROJECT_DEPENDENCIES
    }
    packages = lock["package"]
    project_packages = [
        package for package in packages if package["name"] == "alice-brain-hermes"
    ]
    assert len(project_packages) == 1
    project_package = project_packages[0]
    assert {item["name"] for item in project_package["dependencies"]} == set(
        requirements
    )

    locked_metadata = {
        item["name"]: item.get("specifier", "")
        for item in project_package["metadata"]["requires-dist"]
        if "marker" not in item
    }
    assert set(locked_metadata) == set(requirements)
    for name, requirement in requirements.items():
        assert SpecifierSet(locked_metadata[name]) == requirement.specifier
        matching_packages = [package for package in packages if package["name"] == name]
        assert len(matching_packages) == 1
        assert Version(matching_packages[0]["version"]) in requirement.specifier

    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    assert SpecifierSet(lock["requires-python"]) == SpecifierSet(
        project["requires-python"]
    )
