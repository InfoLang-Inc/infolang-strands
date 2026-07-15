"""Tests for the automatic InfoLang memory hook."""

from __future__ import annotations

from typing import Any

import pytest

from infolang_strands import MEMORY_MARKER, InfoLangMemory, InfoLangMemoryHook
from infolang_strands.hooks import (
    context_message,
    latest_text,
    message_text,
)

from .conftest import (
    FakeAgent,
    FakeInfoLang,
    assistant_msg,
    make_chunk,
    make_recall,
    user_msg,
)


class RecordingRegistry:
    """Captures add_callback registrations for assertions."""

    def __init__(self) -> None:
        self.registered: list[type] = []

    def add_callback(self, event_type: type, callback: Any, *, order: float = 0) -> None:
        self.registered.append(event_type)


def _before_event(agent: FakeAgent) -> Any:
    return type("Evt", (), {"agent": agent})()


def _after_event(agent: FakeAgent) -> Any:
    return type("Evt", (), {"agent": agent})()


# --- pure helpers --------------------------------------------------------


def test_message_text_joins_blocks() -> None:
    msg = {"role": "user", "content": [{"text": "a"}, {"text": "b"}]}
    assert message_text(msg) == "a\nb"


def test_message_text_ignores_non_text_blocks() -> None:
    msg = {"role": "user", "content": [{"toolUse": {}}, {"text": "hi"}]}
    assert message_text(msg) == "hi"


def test_message_text_non_dict() -> None:
    assert message_text("nope") == ""


def test_message_text_bad_content() -> None:
    assert message_text({"role": "user", "content": 42}) == ""


def test_latest_text_returns_most_recent() -> None:
    messages = [user_msg("first"), user_msg("second")]
    assert latest_text(messages, "user") == "second"


def test_latest_text_role_filter() -> None:
    messages = [user_msg("q"), assistant_msg("a")]
    assert latest_text(messages, "assistant") == "a"


def test_latest_text_skips_prefix() -> None:
    messages = [user_msg("real question"), user_msg(f"{MEMORY_MARKER}\ninjected")]
    assert latest_text(messages, "user", skip_prefix=MEMORY_MARKER) == "real question"


def test_latest_text_none_when_absent() -> None:
    assert latest_text([assistant_msg("a")], "user") is None


def test_latest_text_skips_empty() -> None:
    messages = [user_msg("real"), {"role": "user", "content": []}]
    assert latest_text(messages, "user") == "real"


def test_context_message_shape() -> None:
    msg = context_message("user", "some context")
    assert msg["role"] == "user"
    assert msg["content"][0]["text"].startswith(MEMORY_MARKER)
    assert "some context" in msg["content"][0]["text"]


# --- registration --------------------------------------------------------


def test_register_both_callbacks(memory: InfoLangMemory) -> None:
    from infolang_strands.hooks import AfterInvocationEvent, BeforeInvocationEvent

    reg = RecordingRegistry()
    InfoLangMemoryHook(memory).register_hooks(reg)  # type: ignore[arg-type]
    assert BeforeInvocationEvent in reg.registered
    assert AfterInvocationEvent in reg.registered


def test_register_recall_only(memory: InfoLangMemory) -> None:
    from infolang_strands.hooks import BeforeInvocationEvent

    reg = RecordingRegistry()
    InfoLangMemoryHook(memory, auto_remember=False).register_hooks(reg)  # type: ignore[arg-type]
    assert reg.registered == [BeforeInvocationEvent]


def test_register_remember_only(memory: InfoLangMemory) -> None:
    from infolang_strands.hooks import AfterInvocationEvent

    reg = RecordingRegistry()
    InfoLangMemoryHook(memory, auto_recall=False).register_hooks(reg)  # type: ignore[arg-type]
    assert reg.registered == [AfterInvocationEvent]


# --- before-invocation (auto recall) -------------------------------------


def test_before_injects_context(memory: InfoLangMemory) -> None:
    memory.client.recall_result = make_recall(make_chunk(text="berlin", score=0.9))  # type: ignore[attr-defined]
    agent = FakeAgent([user_msg("where did she move?")])
    hook = InfoLangMemoryHook(memory)
    hook._on_before_invocation(_before_event(agent))
    assert len(agent.messages) == 2
    injected = agent.messages[-1]
    assert injected["content"][0]["text"].startswith(MEMORY_MARKER)
    assert "berlin" in injected["content"][0]["text"]


def test_before_recalls_latest_user_query(memory: InfoLangMemory) -> None:
    agent = FakeAgent([user_msg("old"), assistant_msg("a"), user_msg("new query")])
    InfoLangMemoryHook(memory)._on_before_invocation(_before_event(agent))
    _name, args, _kwargs = memory.client.calls[0]  # type: ignore[attr-defined]
    assert args[0] == "new query"


def test_before_no_user_message(memory: InfoLangMemory) -> None:
    agent = FakeAgent([assistant_msg("only assistant")])
    InfoLangMemoryHook(memory)._on_before_invocation(_before_event(agent))
    assert memory.client.calls == []  # type: ignore[attr-defined]
    assert len(agent.messages) == 1


def test_before_no_chunks_no_injection(memory: InfoLangMemory) -> None:
    memory.client.recall_result = make_recall()  # type: ignore[attr-defined]
    agent = FakeAgent([user_msg("q")])
    InfoLangMemoryHook(memory)._on_before_invocation(_before_event(agent))
    assert len(agent.messages) == 1


def test_before_top_k_override(memory: InfoLangMemory) -> None:
    agent = FakeAgent([user_msg("q")])
    InfoLangMemoryHook(memory, top_k=3)._on_before_invocation(_before_event(agent))
    assert memory.client.kwargs_for("recall")["top_k"] == 3  # type: ignore[attr-defined]


def test_before_skips_previously_injected(memory: InfoLangMemory) -> None:
    agent = FakeAgent(
        [user_msg("real"), user_msg(f"{MEMORY_MARKER}\nold context")]
    )
    InfoLangMemoryHook(memory)._on_before_invocation(_before_event(agent))
    _name, args, _kwargs = memory.client.calls[0]  # type: ignore[attr-defined]
    assert args[0] == "real"


# --- after-invocation (auto remember) ------------------------------------


def test_after_remembers_exchange(memory: InfoLangMemory) -> None:
    agent = FakeAgent([user_msg("what is x?"), assistant_msg("x is y")])
    InfoLangMemoryHook(memory)._on_after_invocation(_after_event(agent))
    _name, args, _kwargs = memory.client.calls[0]  # type: ignore[attr-defined]
    assert "what is x?" in args[0]
    assert "x is y" in args[0]


def test_after_ignores_injected_user_message(memory: InfoLangMemory) -> None:
    agent = FakeAgent(
        [
            user_msg("real question"),
            user_msg(f"{MEMORY_MARKER}\ninjected"),
            assistant_msg("the answer"),
        ]
    )
    InfoLangMemoryHook(memory)._on_after_invocation(_after_event(agent))
    _name, args, _kwargs = memory.client.calls[0]  # type: ignore[attr-defined]
    assert "real question" in args[0]


def test_after_no_answer_skips(memory: InfoLangMemory) -> None:
    agent = FakeAgent([user_msg("q only")])
    InfoLangMemoryHook(memory)._on_after_invocation(_after_event(agent))
    assert memory.client.calls == []  # type: ignore[attr-defined]


def test_after_no_query_skips(memory: InfoLangMemory) -> None:
    agent = FakeAgent([assistant_msg("answer only")])
    InfoLangMemoryHook(memory)._on_after_invocation(_after_event(agent))
    assert memory.client.calls == []  # type: ignore[attr-defined]


def test_after_uses_source(memory: InfoLangMemory) -> None:
    agent = FakeAgent([user_msg("q"), assistant_msg("a")])
    InfoLangMemoryHook(memory, source="chat")._on_after_invocation(_after_event(agent))
    assert memory.client.kwargs_for("remember")["source"] == "chat"  # type: ignore[attr-defined]


# --- error handling ------------------------------------------------------


class BoomClient(FakeInfoLang):
    def recall(self, query: str, **kwargs: Any) -> Any:
        raise RuntimeError("boom recall")

    def remember(self, text: str, **kwargs: Any) -> Any:
        raise RuntimeError("boom remember")


def test_recall_error_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    mem = InfoLangMemory(BoomClient())  # type: ignore[arg-type]
    agent = FakeAgent([user_msg("q")])
    InfoLangMemoryHook(mem)._on_before_invocation(_before_event(agent))
    assert len(agent.messages) == 1  # no injection, no crash


def test_recall_error_raise_opt_in() -> None:
    mem = InfoLangMemory(BoomClient())  # type: ignore[arg-type]
    agent = FakeAgent([user_msg("q")])
    hook = InfoLangMemoryHook(mem, raise_on_error=True)
    with pytest.raises(RuntimeError, match="boom recall"):
        hook._on_before_invocation(_before_event(agent))


def test_remember_error_swallowed() -> None:
    mem = InfoLangMemory(BoomClient())  # type: ignore[arg-type]
    agent = FakeAgent([user_msg("q"), assistant_msg("a")])
    InfoLangMemoryHook(mem)._on_after_invocation(_after_event(agent))  # no crash


def test_remember_error_raise_opt_in() -> None:
    mem = InfoLangMemory(BoomClient())  # type: ignore[arg-type]
    agent = FakeAgent([user_msg("q"), assistant_msg("a")])
    hook = InfoLangMemoryHook(mem, raise_on_error=True)
    with pytest.raises(RuntimeError, match="boom remember"):
        hook._on_after_invocation(_after_event(agent))


def test_messages_missing_attr_is_safe(memory: InfoLangMemory) -> None:
    class NoMessages:
        pass

    event = type("Evt", (), {"agent": NoMessages()})()
    InfoLangMemoryHook(memory)._on_before_invocation(event)
    assert memory.client.calls == []  # type: ignore[attr-defined]


def test_inject_role_assistant(memory: InfoLangMemory) -> None:
    memory.client.recall_result = make_recall(make_chunk())  # type: ignore[attr-defined]
    agent = FakeAgent([user_msg("q")])
    InfoLangMemoryHook(memory, inject_role="assistant")._on_before_invocation(
        _before_event(agent)
    )
    assert agent.messages[-1]["role"] == "assistant"
