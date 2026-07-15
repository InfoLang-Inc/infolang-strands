# infolang-strands

InfoLang semantic memory for the [Strands Agents](https://strandsagents.com) SDK.

Give any Strands agent durable, cross-session memory backed by
[InfoLang](https://infolang.ai). Two integration styles, use either or both:

- **Tools** — the model decides when to recall, remember, or forget.
- **Hook** — memory is recalled and stored automatically, with no changes to
  your agent's control flow.

Built on the published InfoLang Python SDK (`infolang`); it never talks HTTP
directly or touches runtime internals.

## Install

```bash
pip install infolang-strands
```

This pulls in `infolang` (the public SDK) and `strands-agents`.

## Quickstart

### Tools (manual, model-driven)

```python
from infolang import InfoLang
from strands import Agent
from infolang_strands import infolang_tools

il = InfoLang.from_api_key("il_live_...", namespace="support", workspace="acme")
agent = Agent(tools=infolang_tools(il))

agent("Remember that our SLA is 99.9% and recall it whenever asked.")
```

The agent gets five tools: `infolang_recall`, `infolang_investigate`,
`infolang_remember`, `infolang_remember_batch`, and `infolang_forget`.

### Hook (automatic, transparent)

```python
from strands import Agent
from infolang_strands import InfoLangMemory, InfoLangMemoryHook

memory = InfoLangMemory.from_api_key("il_live_...", namespace="support")
agent = Agent(hooks=[InfoLangMemoryHook(memory)])

agent("What did we decide about the refund policy last week?")
```

Before each turn the hook recalls memory relevant to the user's message and
injects it as context; after each turn it stores the question/answer pair. A
recall/store failure is logged and swallowed by default so an InfoLang outage
never breaks a live agent — pass `raise_on_error=True` to opt out.

Mix both: expose the tools *and* run the hook.

```python
agent = Agent(
    tools=infolang_tools(memory),
    hooks=[InfoLangMemoryHook(memory, auto_remember=True, auto_recall=False)],
)
```

## Scoping

InfoLang scopes memory by `workspace` (tenant) and `namespace` (bank):

```python
memory = InfoLangMemory.from_api_key(
    "il_live_...",
    workspace="acme",        # tenant
    namespace="support",     # bank
    source="strands-agent",  # attached to every write
    top_k=5,
)
```

A managed API key honours `namespace` on both reads and writes, so one key can
serve many banks. Per-call overrides are available on every `InfoLangMemory`
method.

## API

| Symbol | Purpose |
|--------|---------|
| `InfoLangMemory(client, *, namespace, source, top_k, score_floor)` | Adapter over an `infolang.InfoLang` client |
| `InfoLangMemory.from_api_key(key, *, namespace, workspace, ...)` | Build adapter + managed client |
| `infolang_tools(memory, *, namespace, source, top_k)` | List of Strands tools for `Agent(tools=...)` |
| `InfoLangMemoryHook(memory, *, auto_recall, auto_remember, top_k, source, inject_role, raise_on_error)` | `HookProvider` for `Agent(hooks=...)` |

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```

Tests run fully offline against a fake InfoLang client. Live tests
(`-m live --run-live`) require `INFOLANG_API_KEY`.

## Verified against

- `strands-agents` 1.47.0
- `infolang` 0.2.0

## License

Apache-2.0
