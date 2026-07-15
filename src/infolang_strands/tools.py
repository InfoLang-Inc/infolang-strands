"""Strands tools that expose InfoLang's four core memory operations.

``infolang_tools(...)`` returns a list of Strands ``@tool``-decorated callables
you drop straight into ``Agent(tools=[...])``. They give the model manual,
on-demand control over memory:

* ``infolang_recall`` / ``infolang_investigate`` \u2014 read
* ``infolang_remember`` (+ ``infolang_remember_batch``) \u2014 write
* ``infolang_forget`` \u2014 delete

For hands-off *automatic* recall/remember, use
:class:`infolang_strands.hooks.InfoLangMemoryHook` instead (or in addition).
"""

from __future__ import annotations

from infolang import InfoLang
from strands import tool
from strands.types.tools import AgentTool

from .adapter import DEFAULT_TOP_K, InfoLangMemory, coerce_memory


def infolang_tools(
    memory: InfoLangMemory | InfoLang,
    *,
    namespace: str | None = None,
    source: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[AgentTool]:
    """Build Strands memory tools bound to an InfoLang client.

    Parameters
    ----------
    memory:
        An :class:`~infolang_strands.adapter.InfoLangMemory` or a raw
        :class:`infolang.InfoLang` client (which is wrapped for you).
    namespace, source, top_k:
        Defaults applied when ``memory`` is a raw client.
    """

    mem = coerce_memory(memory, namespace=namespace, source=source, top_k=top_k)
    default_top_k = mem.top_k

    @tool
    def infolang_recall(query: str, top_k: int = default_top_k) -> str:
        """Recall relevant facts from InfoLang semantic memory.

        Args:
            query: What to search the memory bank for.
            top_k: Maximum number of memory chunks to return.
        """
        return mem.format_chunks(mem.recall(query, top_k=top_k))

    @tool
    def infolang_investigate(query: str) -> str:
        """Investigate a question against InfoLang memory (agent-style recall).

        Args:
            query: The question to investigate.
        """
        return mem.format_chunks(mem.investigate(query))

    @tool
    def infolang_remember(text: str, tags: str | None = None) -> str:
        """Store a fact worth keeping in InfoLang semantic memory.

        Args:
            text: The fact or note to remember.
            tags: Optional comma-separated tags to attach to the memory.
        """
        result = mem.remember(text, tags=tags)
        return _stored_message(result.memory_id)

    @tool
    def infolang_remember_batch(texts: list[str]) -> str:
        """Store several facts in InfoLang in a single call.

        Args:
            texts: A list of facts or notes to remember.
        """
        results = mem.remember_batch(texts)
        return f"Stored {len(results)} memories in InfoLang."

    @tool
    def infolang_forget(memory_id: str) -> str:
        """Delete a memory from InfoLang by its id.

        Args:
            memory_id: The id of the memory to forget.
        """
        mem.forget(memory_id)
        return f"Forgot memory {memory_id}."

    return [
        infolang_recall,
        infolang_investigate,
        infolang_remember,
        infolang_remember_batch,
        infolang_forget,
    ]


def _stored_message(memory_id: str | None) -> str:
    if memory_id:
        return f"Stored memory {memory_id} in InfoLang."
    return "Stored memory in InfoLang."
