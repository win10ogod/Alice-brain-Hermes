"""Exact synchronous Hermes 0.18.x hook callback surface."""

from __future__ import annotations

import threading
from typing import Any

from alice_brain_hermes.hermes.bridge import HookBridge, default_runtime_home


class HermesHooks:
    """Thin callback facade: bounded capture plus one atomic cache read."""

    def __init__(self, bridge: HookBridge) -> None:
        if type(bridge) is not HookBridge:
            raise TypeError("bridge must be an exact HookBridge")
        self.bridge = bridge

    def _observe(self, hook: str, kwargs: dict[str, Any]) -> None:
        self.bridge.capture(hook, kwargs)

    def on_session_start(self, **kwargs: Any) -> None:
        self._observe("on_session_start", kwargs)
        return None

    def on_session_end(self, **kwargs: Any) -> None:
        self._observe("on_session_end", kwargs)
        return None

    def on_session_finalize(self, **kwargs: Any) -> None:
        self._observe("on_session_finalize", kwargs)
        return None

    def on_session_reset(self, **kwargs: Any) -> None:
        self._observe("on_session_reset", kwargs)
        return None

    def pre_llm_call(self, **kwargs: Any) -> str | None:
        self._observe("pre_llm_call", kwargs)
        context = self.bridge.projections.read_context()
        return context if type(context) is str and context else None

    def post_llm_call(self, **kwargs: Any) -> None:
        self._observe("post_llm_call", kwargs)
        return None

    def pre_api_request(self, **kwargs: Any) -> None:
        self._observe("pre_api_request", kwargs)
        return None

    def post_api_request(self, **kwargs: Any) -> None:
        self._observe("post_api_request", kwargs)
        return None

    def api_request_error(self, **kwargs: Any) -> None:
        self._observe("api_request_error", kwargs)
        return None

    def pre_tool_call(self, **kwargs: Any) -> None:
        self._observe("pre_tool_call", kwargs)
        return None

    def post_tool_call(self, **kwargs: Any) -> None:
        self._observe("post_tool_call", kwargs)
        return None

    def pre_approval_request(self, **kwargs: Any) -> None:
        self._observe("pre_approval_request", kwargs)
        return None

    def post_approval_response(self, **kwargs: Any) -> None:
        self._observe("post_approval_response", kwargs)
        return None

    def subagent_start(self, **kwargs: Any) -> None:
        self._observe("subagent_start", kwargs)
        return None

    def subagent_stop(self, **kwargs: Any) -> None:
        self._observe("subagent_stop", kwargs)
        return None

    def pre_verify(self, **kwargs: Any) -> None:
        self._observe("pre_verify", kwargs)
        return None


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_HOOKS: HermesHooks | None = None


def default_hooks() -> HermesHooks:
    """Create the process bridge lazily on the first actual callback."""

    global _DEFAULT_HOOKS
    hooks = _DEFAULT_HOOKS
    if hooks is not None:
        return hooks
    with _DEFAULT_LOCK:
        hooks = _DEFAULT_HOOKS
        if hooks is None:
            hooks = HermesHooks(HookBridge(default_runtime_home()))
            _DEFAULT_HOOKS = hooks
        return hooks


def dispatch_hook(hook: str, kwargs: dict[str, Any]) -> str | None:
    callback = getattr(default_hooks(), hook)
    result = callback(**kwargs)
    if hook == "pre_llm_call":
        return result if type(result) is str else None
    return None


__all__ = ["HermesHooks", "default_hooks", "dispatch_hook"]
