"""Automatic InfoLang memory for Strands agents via the hook system.

:class:`InfoLangMemoryHook` implements the Strands ``HookProvider`` protocol so
memory works with *zero* changes to the agent's control flow:

* On :class:`~strands.hooks.BeforeInvocationEvent` it recalls memory relevant to
  the latest user turn and injects it as a marked context message.
* On :class:`~strands.hooks.AfterInvocationEvent` it stores the user's question
  and the assistant's answer for future recall.

Register it with ``Agent(hooks=[InfoLangMemoryHook(memory)])``. Memory failures
are best-effort by default (logged, not raised) so an InfoLang outage never
breaks a live agent; set ``raise_on_error=True`` to opt out.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

from infolang import InfoLang
from strands.hooks import (
    AfterInvocationEvent,
    BeforeInvocationEvent,
    HookProvider,
    HookRegistry,
)

from .adapter import InfoLangMemory, coerce_memory

_log = logging.getLogger("infolang_strands")

MEMORY_MARKER = "[InfoLang memory]"


class InfoLangMemoryHook(HookProvider):
    """Wire InfoLang recall/remember into a Strands agent's lifecycle.

    Parameters
    ----------
    memory:
        An :class:`~infolang_strands.adapter.InfoLangMemory` or raw
        :class:`infolang.InfoLang` client.
    auto_recall:
        Recall + inject context before each invocation.
    auto_remember:
        Store the question/answer pair after each invocation.
    top_k:
        Override the number of chunks to recall (defaults to the adapter's).
    source:
        ``source`` label attached to auto-stored memories.
    inject_role:
        Role of the injected context message (``"user"`` or ``"assistant"``).
    raise_on_error:
        Re-raise InfoLang errors instead of logging and continuing.
    """

    def __init__(
        self,
        memory: InfoLangMemory | InfoLang,
        *,
        auto_recall: bool = True,
        auto_remember: bool = True,
        top_k: int | None = None,
        source: str | None = None,
        inject_role: str = "user",
        raise_on_error: bool = False,
    ) -> None:
        self.memory = coerce_memory(memory, source=source)
        self.auto_recall = auto_recall
        self.auto_remember = auto_remember
        self.top_k = top_k
        self.source = source
        self.inject_role = inject_role
        self.raise_on_error = raise_on_error

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        if self.auto_recall:
            registry.add_callback(BeforeInvocationEvent, self._on_before_invocation)
        if self.auto_remember:
            registry.add_callback(AfterInvocationEvent, self._on_after_invocation)

    # --- callbacks -------------------------------------------------------

    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        messages = _messages(event.agent)
        query = latest_text(messages, "user", skip_prefix=MEMORY_MARKER)
        if not query:
            return
        try:
            result = self.memory.recall(query, top_k=self.top_k)
        except Exception:
            self._handle_error("recall")
            return
        if not result.chunks:
            return
        context = self.memory.format_chunks(result)
        messages.append(context_message(self.inject_role, context))

    def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        messages = _messages(event.agent)
        query = latest_text(messages, "user", skip_prefix=MEMORY_MARKER)
        answer = latest_text(messages, "assistant", skip_prefix=MEMORY_MARKER)
        if not query or not answer:
            return
        text = f"User asked: {query}\nAssistant answered: {answer}"
        try:
            self.memory.remember(text, source=self.source)
        except Exception:
            self._handle_error("remember")

    def _handle_error(self, op: str) -> None:
        if self.raise_on_error:
            raise
        _log.warning("InfoLang %s failed; continuing without memory.", op, exc_info=True)


# --- pure helpers --------------------------------------------------------


def _messages(agent: Any) -> list[Any]:
    """Return the agent's mutable message list (empty list if absent)."""

    messages = getattr(agent, "messages", None)
    return messages if isinstance(messages, list) else []


def message_text(message: Any) -> str:
    """Concatenate the text of every text content block in a message."""

    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, Sequence):
        return ""
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "\n".join(parts).strip()


def latest_text(
    messages: Sequence[Any],
    role: str,
    *,
    skip_prefix: str | None = None,
) -> str | None:
    """Return the text of the most recent message with ``role``.

    Messages whose text starts with ``skip_prefix`` (e.g. injected memory
    blocks) are ignored so genuine user/assistant turns are found.
    """

    for message in reversed(list(messages)):
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        text = message_text(message)
        if not text:
            continue
        if skip_prefix and text.startswith(skip_prefix):
            continue
        return text
    return None


def context_message(role: str, context: str) -> Any:
    """Build a marked context message for injection into the agent transcript."""

    return cast(
        Any,
        {"role": role, "content": [{"text": f"{MEMORY_MARKER}\n{context}"}]},
    )
