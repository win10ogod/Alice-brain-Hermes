"""Inert Hermes Agent plugin registration boundary."""

from __future__ import annotations

from importlib import import_module, metadata
from threading import RLock
from types import MappingProxyType
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

APPROVED_HOOKS = (
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "pre_tool_call",
    "post_tool_call",
    "pre_approval_request",
    "post_approval_response",
    "subagent_start",
    "subagent_stop",
    "pre_verify",
)


SUPPORTED_HERMES = ">=0.18,<0.19"
_SUPPORTED_HERMES_SPECIFIER = SpecifierSet(SUPPORTED_HERMES)
_REGISTRATION_STATE_ATTRIBUTE = "_alice_brain_hermes_registration_v1"
_REGISTRATION_LOCK = RLock()


def on_session_start(**kwargs: Any) -> None:
    return None


def on_session_end(**kwargs: Any) -> None:
    return None


def on_session_finalize(**kwargs: Any) -> None:
    return None


def on_session_reset(**kwargs: Any) -> None:
    return None


def pre_llm_call(**kwargs: Any) -> None:
    return None


def post_llm_call(**kwargs: Any) -> None:
    return None


def pre_api_request(**kwargs: Any) -> None:
    return None


def post_api_request(**kwargs: Any) -> None:
    return None


def api_request_error(**kwargs: Any) -> None:
    return None


def pre_tool_call(**kwargs: Any) -> None:
    return None


def post_tool_call(**kwargs: Any) -> None:
    return None


def pre_approval_request(**kwargs: Any) -> None:
    return None


def post_approval_response(**kwargs: Any) -> None:
    return None


def subagent_start(**kwargs: Any) -> None:
    return None


def subagent_stop(**kwargs: Any) -> None:
    return None


def pre_verify(**kwargs: Any) -> None:
    return None


HOOK_CALLBACKS = MappingProxyType(
    {
        "on_session_start": on_session_start,
        "on_session_end": on_session_end,
        "on_session_finalize": on_session_finalize,
        "on_session_reset": on_session_reset,
        "pre_llm_call": pre_llm_call,
        "post_llm_call": post_llm_call,
        "pre_api_request": pre_api_request,
        "post_api_request": post_api_request,
        "api_request_error": api_request_error,
        "pre_tool_call": pre_tool_call,
        "post_tool_call": post_tool_call,
        "pre_approval_request": pre_approval_request,
        "post_approval_response": post_approval_response,
        "subagent_start": subagent_start,
        "subagent_stop": subagent_stop,
        "pre_verify": pre_verify,
    }
)


def _parse_hermes_version(version: object, *, source: str) -> Version:
    if not isinstance(version, str):
        raise RuntimeError(f"Hermes {source} version is invalid")
    try:
        return Version(version)
    except InvalidVersion as error:
        raise RuntimeError(f"Hermes {source} version {version!r} is invalid") from error


def resolve_hermes_version() -> str:
    """Resolve and cross-check the installed Hermes Agent host version."""

    distribution_version: str | None = None
    try:
        distribution_version = metadata.version("hermes-agent")
    except metadata.PackageNotFoundError:
        pass
    except Exception as error:
        raise RuntimeError(
            "Hermes Agent distribution version is unavailable"
        ) from error

    module_version: str | None = None
    try:
        module = import_module("hermes_cli")
    except ModuleNotFoundError as error:
        if error.name != "hermes_cli":
            raise RuntimeError("Hermes Agent module version is unavailable") from error
    except Exception as error:
        raise RuntimeError("Hermes Agent module version is unavailable") from error
    else:
        raw_module_version = getattr(module, "__version__", None)
        if raw_module_version is not None and not isinstance(raw_module_version, str):
            raise RuntimeError("Hermes module version is invalid")
        module_version = raw_module_version

    parsed_distribution = (
        _parse_hermes_version(distribution_version, source="distribution")
        if distribution_version is not None
        else None
    )
    parsed_module = (
        _parse_hermes_version(module_version, source="module")
        if module_version is not None
        else None
    )

    if parsed_distribution is not None and parsed_module is not None:
        if parsed_distribution != parsed_module:
            raise RuntimeError(
                "Hermes Agent distribution/module version mismatch: "
                f"{distribution_version!r} != {module_version!r}"
            )
        if distribution_version is None:
            raise RuntimeError("Hermes Agent distribution version is unavailable")
        return distribution_version
    if parsed_distribution is not None:
        if distribution_version is None:
            raise RuntimeError("Hermes Agent distribution version is unavailable")
        return distribution_version
    if parsed_module is not None:
        if module_version is None:
            raise RuntimeError("Hermes Agent module version is unavailable")
        return module_version
    raise RuntimeError("Hermes Agent is not installed or has no version metadata")


def require_supported_hermes(version: str | None) -> str:
    """Fail visibly unless *version* is in the verified Hermes release line."""

    if version is None:
        raise RuntimeError("Hermes Agent version is invalid")
    parsed = _parse_hermes_version(version, source="Agent")
    if parsed not in _SUPPORTED_HERMES_SPECIFIER:
        raise RuntimeError(
            f"Hermes Agent version {version!r} is unsupported; "
            f"required {SUPPORTED_HERMES}"
        )
    return version


def setup_alice_brain_cli(parser: Any) -> None:
    """Attach the shared runtime CLI lazily when Hermes builds its parser."""

    from alice_brain_hermes.hermes.cli import setup_alice_brain_cli as setup

    setup(parser)


def handle_alice_brain_cli(args: Any) -> int:
    """Dispatch the Hermes command through the shared in-process handler."""

    from alice_brain_hermes.hermes.cli import handle_alice_brain_cli as handle

    return handle(args)


def register(ctx: Any) -> None:
    """Register the inert Task 5 seam with a supported Hermes context."""

    with _REGISTRATION_LOCK:
        state = getattr(ctx, _REGISTRATION_STATE_ATTRIBUTE, None)
        if state == "registered":
            return
        if state == "registering":
            raise RuntimeError("Alice-brain-Hermes registration is re-entrant")
        if state == "failed":
            raise RuntimeError(
                "Alice-brain-Hermes registration previously failed for this context"
            )
        if state is not None:
            raise RuntimeError("Alice-brain-Hermes registration state is invalid")

        require_supported_hermes(resolve_hermes_version())
        register_hook = getattr(ctx, "register_hook", None)
        register_cli_command = getattr(ctx, "register_cli_command", None)
        if not callable(register_hook) or not callable(register_cli_command):
            raise RuntimeError("Hermes plugin context lacks registration callables")

        setattr(ctx, _REGISTRATION_STATE_ATTRIBUTE, "registering")
        try:
            for hook_name in APPROVED_HOOKS:
                register_hook(hook_name, HOOK_CALLBACKS[hook_name])
            register_cli_command(
                name="alice-brain",
                help="Inspect and control the Alice-brain-Hermes runtime",
                setup_fn=setup_alice_brain_cli,
                handler_fn=handle_alice_brain_cli,
                description="Alice-brain-Hermes consciousness runtime commands",
            )
            setattr(ctx, _REGISTRATION_STATE_ATTRIBUTE, "registered")
        except BaseException as registration_error:
            try:
                setattr(ctx, _REGISTRATION_STATE_ATTRIBUTE, "failed")
            except BaseException as state_error:
                raise registration_error from state_error
            raise
