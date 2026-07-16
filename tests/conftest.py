from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def make_symlink(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Path, Path, bool], bool]:
    path_type = type(Path.cwd())
    real_is_symlink = path_type.is_symlink
    reported: set[str] = set()

    def key(path: Path) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def is_symlink(path: Path) -> bool:
        return key(path) in reported or real_is_symlink(path)

    def make(path: Path, target: Path, target_is_directory: bool = False) -> bool:
        try:
            path.symlink_to(target, target_is_directory=target_is_directory)
        except (NotImplementedError, OSError):
            if target_is_directory:
                path.mkdir()
            else:
                path.touch()
            reported.add(key(path))
            return False
        return True

    monkeypatch.setattr(path_type, "is_symlink", is_symlink)
    return make
