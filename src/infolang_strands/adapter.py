"""Core adapter around the published InfoLang SDK.

This module is the single point of contact with the ``infolang`` package. Both
the Strands tools (:mod:`infolang_strands.tools`) and the automatic memory hook
(:mod:`infolang_strands.hooks`) delegate here so request shaping and result
formatting live in exactly one place.

It depends only on the public SDK surface: ``from infolang import InfoLang`` and
the documented ``recall`` / ``investigate`` / ``remember`` / ``memorize`` /
``remember_batch`` / ``forget`` operations. It never reimplements HTTP nor
imports runtime/engine internals.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from infolang import InfoLang, RecallResult, RememberResult

DEFAULT_TOP_K = 5
DEFAULT_SCORE_FLOOR = 0.85

RememberItem = str | dict[str, Any]


class InfoLangMemory:
    """A thin, framework-agnostic wrapper over an :class:`infolang.InfoLang` client.

    Parameters
    ----------
    client:
        A constructed :class:`infolang.InfoLang` client.
    namespace:
        Optional bank to scope every read/write to. When omitted the client's
        own default namespace (or the managed API key's namespace) is used.
    source:
        Default ``source`` label attached to writes.
    top_k:
        Default number of chunks to request on recall/investigate.
    score_floor:
        Confidence floor (matches the SDK's 0.85 weak-match threshold) used when
        rendering context so callers can flag low-confidence recalls.
    """

    def __init__(
        self,
        client: InfoLang,
        *,
        namespace: str | None = None,
        source: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        score_floor: float = DEFAULT_SCORE_FLOOR,
    ) -> None:
        self.client = client
        self.namespace = namespace
        self.source = source
        self.top_k = top_k
        self.score_floor = score_floor

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        namespace: str | None = None,
        workspace: str | None = None,
        source: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        score_floor: float = DEFAULT_SCORE_FLOOR,
    ) -> InfoLangMemory:
        """Build the adapter and its managed-cloud client in one call."""

        client = InfoLang.from_api_key(api_key, namespace=namespace, workspace=workspace)
        return cls(
            client,
            namespace=namespace,
            source=source,
            top_k=top_k,
            score_floor=score_floor,
        )

    # --- reads -----------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> RecallResult:
        """Semantic recall against the configured bank."""

        return self.client.recall(
            query,
            namespace=namespace or self.namespace,
            top_k=top_k if top_k is not None else self.top_k,
            filters=filters,
        )

    def investigate(
        self,
        query: str,
        *,
        top_k: int | None = None,
        namespace: str | None = None,
    ) -> RecallResult:
        """Agent-style recall (mirrors the SDK ``investigate`` helper)."""

        return self.client.investigate(
            query,
            namespace_hint=namespace or self.namespace,
            top_k=top_k if top_k is not None else self.top_k,
        )

    # --- writes ----------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        tags: str | None = None,
        source: str | None = None,
        namespace: str | None = None,
    ) -> RememberResult:
        """Store a single memory."""

        return self.client.remember(
            text,
            namespace=namespace or self.namespace,
            source=source or self.source,
            tags=tags,
        )

    def remember_batch(
        self,
        items: Sequence[RememberItem],
        *,
        source: str | None = None,
        namespace: str | None = None,
    ) -> list[RememberResult]:
        """Store many memories in one round-trip."""

        return self.client.remember_batch(
            list(items),
            namespace=namespace or self.namespace,
            source=source or self.source,
        )

    def forget(self, memory_id: str, *, namespace: str | None = None) -> None:
        """Delete a memory by id."""

        self.client.forget(memory_id, namespace=namespace or self.namespace)

    # --- rendering -------------------------------------------------------

    def format_chunks(
        self,
        result: RecallResult,
        *,
        include_scores: bool = True,
        header: str = "Relevant memory from InfoLang",
        empty: str = "No relevant memory found in InfoLang.",
    ) -> str:
        """Render a recall result as an LLM-friendly context block.

        Returns ``empty`` when there are no chunks. When the top score is below
        ``score_floor`` a weak-match caveat is appended so the model can weigh
        the evidence accordingly.
        """

        if not result.chunks:
            return empty
        lines = [f"{header} ({len(result.chunks)} result(s)):"]
        for i, chunk in enumerate(result.chunks, start=1):
            prefix = f"[{i}]"
            if include_scores and chunk.score is not None:
                prefix = f"[{i}] (score {chunk.score:.2f})"
            lines.append(f"{prefix} {chunk.text}")
        if result.weak:
            lines.append(
                "(Weak match: top score is below the "
                f"{self.score_floor:.2f} confidence floor \u2014 treat as a hint.)"
            )
        return "\n".join(lines)


def coerce_memory(
    memory: InfoLangMemory | InfoLang,
    *,
    namespace: str | None = None,
    source: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> InfoLangMemory:
    """Accept either an :class:`InfoLangMemory` or a raw client and normalise.

    Lets integration entry points be permissive about what callers hand them
    while keeping the rest of the code working with a single type.
    """

    if isinstance(memory, InfoLangMemory):
        return memory
    return InfoLangMemory(memory, namespace=namespace, source=source, top_k=top_k)
