"""InfoLang integration for the Strands Agents SDK.

Two ways to give a Strands agent durable semantic memory:

Tools (manual, model-driven)::

    from infolang import InfoLang
    from strands import Agent
    from infolang_strands import infolang_tools

    il = InfoLang.from_api_key("il_live_...", namespace="support")
    agent = Agent(tools=infolang_tools(il))

Hook (automatic, transparent)::

    from infolang_strands import InfoLangMemory, InfoLangMemoryHook

    memory = InfoLangMemory.from_api_key("il_live_...", namespace="support")
    agent = Agent(hooks=[InfoLangMemoryHook(memory)])
"""

from __future__ import annotations

from ._version import __version__
from .adapter import DEFAULT_SCORE_FLOOR, DEFAULT_TOP_K, InfoLangMemory, coerce_memory
from .hooks import MEMORY_MARKER, InfoLangMemoryHook
from .tools import infolang_tools

__all__ = [
    "__version__",
    "InfoLangMemory",
    "InfoLangMemoryHook",
    "infolang_tools",
    "coerce_memory",
    "MEMORY_MARKER",
    "DEFAULT_TOP_K",
    "DEFAULT_SCORE_FLOOR",
]
