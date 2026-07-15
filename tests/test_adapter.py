"""Tests for the InfoLangMemory adapter."""

from __future__ import annotations

import pytest
from infolang import RememberResult

from infolang_strands import DEFAULT_SCORE_FLOOR, DEFAULT_TOP_K, InfoLangMemory, coerce_memory
from infolang_strands.adapter import InfoLangMemory as AdapterMemory

from .conftest import FakeInfoLang, make_chunk, make_recall


def test_recall_passes_namespace_and_top_k(memory: InfoLangMemory) -> None:
    memory.recall("who moved to Berlin?")
    kwargs = memory.client.kwargs_for("recall")  # type: ignore[attr-defined]
    assert kwargs["namespace"] == "support"
    assert kwargs["top_k"] == DEFAULT_TOP_K
    assert kwargs["filters"] is None


def test_recall_top_k_override(memory: InfoLangMemory) -> None:
    memory.recall("q", top_k=3)
    assert memory.client.kwargs_for("recall")["top_k"] == 3  # type: ignore[attr-defined]


def test_recall_filters_and_namespace_override(memory: InfoLangMemory) -> None:
    memory.recall("q", filters={"tag": "x"}, namespace="other")
    kwargs = memory.client.kwargs_for("recall")  # type: ignore[attr-defined]
    assert kwargs["filters"] == {"tag": "x"}
    assert kwargs["namespace"] == "other"


def test_investigate_uses_namespace_hint(memory: InfoLangMemory) -> None:
    memory.investigate("q", top_k=7)
    kwargs = memory.client.kwargs_for("investigate")  # type: ignore[attr-defined]
    assert kwargs["namespace_hint"] == "support"
    assert kwargs["top_k"] == 7


def test_remember_defaults_source(memory: InfoLangMemory) -> None:
    memory.remember("a fact", tags="t1,t2")
    kwargs = memory.client.kwargs_for("remember")  # type: ignore[attr-defined]
    assert kwargs["namespace"] == "support"
    assert kwargs["source"] == "unit-test"
    assert kwargs["tags"] == "t1,t2"


def test_remember_explicit_source_wins(memory: InfoLangMemory) -> None:
    memory.remember("a fact", source="override")
    assert memory.client.kwargs_for("remember")["source"] == "override"  # type: ignore[attr-defined]


def test_remember_batch(memory: InfoLangMemory) -> None:
    results = memory.remember_batch(["a", "b", "c"])
    assert len(results) == 3
    kwargs = memory.client.kwargs_for("remember_batch")  # type: ignore[attr-defined]
    assert kwargs["namespace"] == "support"
    assert kwargs["source"] == "unit-test"


def test_forget(memory: InfoLangMemory) -> None:
    memory.forget("m9")
    kwargs = memory.client.kwargs_for("forget")  # type: ignore[attr-defined]
    assert kwargs["namespace"] == "support"


def test_forget_namespace_override(memory: InfoLangMemory) -> None:
    memory.forget("m9", namespace="ns2")
    assert memory.client.kwargs_for("forget")["namespace"] == "ns2"  # type: ignore[attr-defined]


def test_defaults_when_no_namespace() -> None:
    client = FakeInfoLang()
    mem = InfoLangMemory(client)  # type: ignore[arg-type]
    assert mem.namespace is None
    assert mem.top_k == DEFAULT_TOP_K
    assert mem.score_floor == DEFAULT_SCORE_FLOOR
    mem.recall("q")
    assert client.kwargs_for("recall")["namespace"] is None


def test_format_chunks_empty(memory: InfoLangMemory) -> None:
    out = memory.format_chunks(make_recall())
    assert out == "No relevant memory found in InfoLang."


def test_format_chunks_custom_empty(memory: InfoLangMemory) -> None:
    assert memory.format_chunks(make_recall(), empty="nada") == "nada"


def test_format_chunks_with_scores(memory: InfoLangMemory) -> None:
    result = make_recall(make_chunk(text="fact one", score=0.91))
    out = memory.format_chunks(result)
    assert "Relevant memory from InfoLang (1 result(s)):" in out
    assert "[1] (score 0.91) fact one" in out


def test_format_chunks_without_scores(memory: InfoLangMemory) -> None:
    result = make_recall(make_chunk(text="fact one", score=0.91))
    out = memory.format_chunks(result, include_scores=False)
    assert "[1] fact one" in out
    assert "score" not in out


def test_format_chunks_missing_score(memory: InfoLangMemory) -> None:
    result = make_recall(make_chunk(text="no score", score=None))
    out = memory.format_chunks(result)
    assert "[1] no score" in out


def test_format_chunks_weak_match_caveat(memory: InfoLangMemory) -> None:
    result = make_recall(make_chunk(text="weak", score=0.5))
    out = memory.format_chunks(result)
    assert "Weak match" in out


def test_format_chunks_strong_no_caveat(memory: InfoLangMemory) -> None:
    result = make_recall(make_chunk(text="strong", score=0.99))
    assert "Weak match" not in memory.format_chunks(result)


def test_format_chunks_custom_header(memory: InfoLangMemory) -> None:
    out = memory.format_chunks(make_recall(make_chunk()), header="Recalled")
    assert out.startswith("Recalled (1 result(s)):")


def test_from_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_from_api_key(api_key: str, **kwargs: object) -> FakeInfoLang:
        captured["api_key"] = api_key
        captured.update(kwargs)
        return FakeInfoLang()

    monkeypatch.setattr("infolang_strands.adapter.InfoLang.from_api_key", fake_from_api_key)
    mem = InfoLangMemory.from_api_key(
        "il_live_x", namespace="ns", workspace="ws", source="s", top_k=9
    )
    assert captured == {"api_key": "il_live_x", "namespace": "ns", "workspace": "ws"}
    assert mem.namespace == "ns"
    assert mem.source == "s"
    assert mem.top_k == 9


def test_coerce_memory_passthrough(memory: InfoLangMemory) -> None:
    assert coerce_memory(memory) is memory


def test_coerce_memory_wraps_client() -> None:
    client = FakeInfoLang()
    mem = coerce_memory(client, namespace="ns", source="s", top_k=4)  # type: ignore[arg-type]
    assert isinstance(mem, AdapterMemory)
    assert mem.namespace == "ns"
    assert mem.source == "s"
    assert mem.top_k == 4


def test_remember_returns_result(memory: InfoLangMemory) -> None:
    result = memory.remember("x")
    assert isinstance(result, RememberResult)
    assert result.memory_id == "m1"
