"""Opt-in live tests against real InfoLang (and, optionally, a real model).

Run with::

    INFOLANG_API_KEY=il_live_... pytest --run-live -m live

These are skipped by default (see ``conftest.pytest_collection_modifyitems``).
"""

from __future__ import annotations

import os
import uuid

import pytest
from infolang import InfoLang

from infolang_strands import InfoLangMemory, infolang_tools

pytestmark = pytest.mark.live


def _memory() -> InfoLangMemory:
    api_key = os.environ.get("INFOLANG_API_KEY")
    if not api_key:
        pytest.skip("INFOLANG_API_KEY not set")
    namespace = os.environ.get("INFOLANG_TEST_NAMESPACE", "wp39-strands-live")
    client = InfoLang.from_api_key(api_key, namespace=namespace)
    return InfoLangMemory(client, namespace=namespace, source="wp39-live-test")


def test_live_remember_then_recall() -> None:
    memory = _memory()
    token = uuid.uuid4().hex
    memory.remember(f"The live canary token is {token}.")
    result = memory.recall(f"what is the live canary token {token}?", top_k=5)
    assert any(token in chunk.text for chunk in result.chunks)


def test_live_tools_roundtrip() -> None:
    memory = _memory()
    tools = {t.tool_name: t for t in infolang_tools(memory)}
    out = tools["infolang_recall"]("anything")
    assert isinstance(out, str)
