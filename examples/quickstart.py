"""Runnable quickstart: a Strands agent with InfoLang memory.

Usage::

    export INFOLANG_API_KEY=il_live_...
    python examples/quickstart.py

Requires a model provider configured for Strands (e.g. AWS credentials for the
default Bedrock model). See https://strandsagents.com for provider setup.
"""

from __future__ import annotations

import os

from strands import Agent

from infolang_strands import InfoLangMemory, InfoLangMemoryHook, infolang_tools


def main() -> None:
    api_key = os.environ["INFOLANG_API_KEY"]
    memory = InfoLangMemory.from_api_key(
        api_key,
        namespace="strands-quickstart",
        source="quickstart",
    )

    # Manual tools + automatic recall/remember hook together.
    agent = Agent(
        tools=infolang_tools(memory),
        hooks=[InfoLangMemoryHook(memory)],
    )

    print(agent("Remember that the project codename is Northstar."))
    print(agent("What is the project codename?"))


if __name__ == "__main__":
    main()
