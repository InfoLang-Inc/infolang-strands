"""Package-level smoke tests."""

from __future__ import annotations

import infolang_strands


def test_version_is_string() -> None:
    assert isinstance(infolang_strands.__version__, str)
    assert infolang_strands.__version__.count(".") >= 2


def test_public_exports() -> None:
    for name in (
        "InfoLangMemory",
        "InfoLangMemoryHook",
        "infolang_tools",
        "coerce_memory",
        "MEMORY_MARKER",
        "DEFAULT_TOP_K",
        "DEFAULT_SCORE_FLOOR",
    ):
        assert name in infolang_strands.__all__
        assert hasattr(infolang_strands, name)


def test_hook_is_hook_provider() -> None:
    from strands.hooks import HookProvider

    assert issubclass(infolang_strands.InfoLangMemoryHook, HookProvider)
