"""Hermes-native CLI adapter over the standalone Python control layer."""

from __future__ import annotations

import argparse

from alice_brain_hermes.cli import (
    configure_control_parser,
    run_control_namespace,
)

_PARSER_ATTRIBUTE = "_alice_brain_hermes_parser"


def setup_alice_brain_cli(parser: argparse.ArgumentParser) -> None:
    """Attach the shared control tree without contacting or starting a runtime."""

    configure_control_parser(parser, command_required=False)
    parser.set_defaults(**{_PARSER_ATTRIBUTE: parser})


def handle_alice_brain_cli(arguments: argparse.Namespace) -> int:
    """Run one command directly in Python; a bare command remains help-only."""

    if getattr(arguments, "alice_brain_command", None) is None:
        parser = getattr(arguments, _PARSER_ATTRIBUTE, None)
        if not isinstance(parser, argparse.ArgumentParser):
            raise RuntimeError("Alice-brain-Hermes CLI parser is unavailable")
        parser.print_help()
        return 0
    return run_control_namespace(arguments)


__all__ = ["handle_alice_brain_cli", "setup_alice_brain_cli"]
