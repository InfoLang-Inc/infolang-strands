"""Tests for the Strands tool factory."""

from __future__ import annotations

import pytest

from infolang_strands import InfoLangMemory, infolang_tools

from .conftest import FakeInfoLang, make_chunk, make_recall


@pytest.fixture
def tools(memory: InfoLangMemory) -> dict[str, object]:
    return {t.tool_name: t for t in infolang_tools(memory)}  # type: ignore[attr-defined]


def test_exposes_five_named_tools(tools: dict[str, object]) -> None:
    assert set(tools) == {
        "infolang_recall",
        "infolang_investigate",
        "infolang_remember",
        "infolang_remember_batch",
        "infolang_forget",
    }


def test_tool_specs_have_descriptions_and_schema(tools: dict[str, object]) -> None:
    for tool in tools.values():
        spec = tool.tool_spec  # type: ignore[attr-defined]
        assert spec["description"]
        assert spec["inputSchema"]["json"]["type"] == "object"


def test_recall_required_param(tools: dict[str, object]) -> None:
    spec = tools["infolang_recall"].tool_spec  # type: ignore[attr-defined]
    assert spec["inputSchema"]["json"]["required"] == ["query"]


def test_recall_tool_returns_formatted(memory: InfoLangMemory) -> None:
    memory.client.recall_result = make_recall(make_chunk(text="berlin", score=0.9))  # type: ignore[attr-defined]
    tool = infolang_tools(memory)[0]
    out = tool("where?")  # type: ignore[operator]
    assert "berlin" in out


def test_recall_tool_top_k_default_from_adapter() -> None:
    client = FakeInfoLang()
    mem = InfoLangMemory(client, top_k=11)  # type: ignore[arg-type]
    tools = {t.tool_name: t for t in infolang_tools(mem)}  # type: ignore[attr-defined]
    tools["infolang_recall"]("q")  # type: ignore[operator]
    assert client.kwargs_for("recall")["top_k"] == 11


def test_recall_tool_top_k_arg(memory: InfoLangMemory) -> None:
    tools = {t.tool_name: t for t in infolang_tools(memory)}  # type: ignore[attr-defined]
    tools["infolang_recall"]("q", top_k=2)  # type: ignore[operator]
    assert memory.client.kwargs_for("recall")["top_k"] == 2  # type: ignore[attr-defined]


def test_investigate_tool(memory: InfoLangMemory) -> None:
    tools = {t.tool_name: t for t in infolang_tools(memory)}  # type: ignore[attr-defined]
    tools["infolang_investigate"]("why?")  # type: ignore[operator]
    assert memory.client.kwargs_for("investigate")["namespace_hint"] == "support"  # type: ignore[attr-defined]


def test_remember_tool_reports_id(memory: InfoLangMemory) -> None:
    tools = {t.tool_name: t for t in infolang_tools(memory)}  # type: ignore[attr-defined]
    out = tools["infolang_remember"]("a fact", tags="x")  # type: ignore[operator]
    assert "m1" in out
    assert memory.client.kwargs_for("remember")["tags"] == "x"  # type: ignore[attr-defined]


def test_remember_tool_without_id(memory: InfoLangMemory) -> None:
    from infolang import RememberResult

    memory.client.remember_result = RememberResult()  # type: ignore[attr-defined]
    tools = {t.tool_name: t for t in infolang_tools(memory)}  # type: ignore[attr-defined]
    out = tools["infolang_remember"]("a fact")  # type: ignore[operator]
    assert out == "Stored memory in InfoLang."


def test_remember_batch_tool(memory: InfoLangMemory) -> None:
    tools = {t.tool_name: t for t in infolang_tools(memory)}  # type: ignore[attr-defined]
    out = tools["infolang_remember_batch"](["a", "b"])  # type: ignore[operator]
    assert "Stored 2 memories" in out


def test_forget_tool(memory: InfoLangMemory) -> None:
    tools = {t.tool_name: t for t in infolang_tools(memory)}  # type: ignore[attr-defined]
    out = tools["infolang_forget"]("m5")  # type: ignore[operator]
    assert "m5" in out
    _name, args, _kwargs = memory.client.calls[-1]  # type: ignore[attr-defined]
    assert args[0] == "m5"


def test_accepts_raw_client() -> None:
    client = FakeInfoLang()
    tools = infolang_tools(client, namespace="ns", source="s", top_k=6)  # type: ignore[arg-type]
    tools[0]("q")  # type: ignore[operator]
    assert client.kwargs_for("recall")["namespace"] == "ns"
    assert client.kwargs_for("recall")["top_k"] == 6
