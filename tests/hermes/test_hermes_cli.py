from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from alice_brain_hermes.hermes import cli as hermes_cli


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="hermes alice-brain",
        description="Alice-brain-Hermes consciousness runtime commands",
    )
    hermes_cli.setup_alice_brain_cli(result)
    return result


def test_bare_hermes_command_remains_a_side_effect_free_help_surface(
    capsys: pytest.CaptureFixture[str],
) -> None:
    command_parser = parser()
    arguments = command_parser.parse_args([])

    assert hermes_cli.handle_alice_brain_cli(arguments) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "usage: hermes alice-brain" in captured.out
    assert "Alice-brain-Hermes consciousness runtime commands" in captured.out


def test_hermes_handler_delegates_to_shared_python_control_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[argparse.Namespace] = []

    def run(arguments: argparse.Namespace) -> int:
        captured.append(arguments)
        return 17

    monkeypatch.setattr(hermes_cli, "run_control_namespace", run)
    home = tmp_path / "runtime home"
    arguments = parser().parse_args(
        [
            "--home",
            str(home),
            "trace",
            "--after-sequence",
            "4",
            "--limit",
            "7",
        ]
    )

    assert hermes_cli.handle_alice_brain_cli(arguments) == 17
    assert captured == [arguments]
    assert arguments.alice_brain_command == "trace"
    assert arguments.after_sequence == 4
    assert arguments.limit == 7


def test_status_does_not_autostart_or_create_the_runtime_home(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    home = tmp_path / "must remain absent"
    arguments = parser().parse_args(["--home", str(home), "status"])

    assert hermes_cli.handle_alice_brain_cli(arguments) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["command"] == "daemon.status"
    assert payload["data"]["running"] is False
    assert payload["data"]["status"] == "stopped"
    assert not home.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["start"],
        ["stop"],
        ["status"],
        ["doctor"],
        ["identity"],
        ["identity", "get"],
        ["trace"],
        ["trace", "list"],
        ["daemon", "run"],
        ["daemon", "start"],
        ["daemon", "stop"],
        ["daemon", "status"],
    ],
)
def test_hermes_parser_exposes_the_shared_control_commands(
    arguments: list[str],
) -> None:
    assert parser().parse_args(arguments).alice_brain_command == arguments[0]
