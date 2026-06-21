"""The LangGraph agent: a Claude-style think -> act -> observe loop over the vault.

The graph has two nodes:

* ``agent`` calls the LLM (bound to the vault tools).
* ``tools`` executes any tool calls and feeds results back.

It loops until the model answers with no further tool calls. References are then
collected deterministically from the notes the agent actually read.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .config import Settings
from .tools import build_tools
from .vault import Vault

SYSTEM_PROMPT = """\
You are a research assistant working over the user's Obsidian vault of company \
notes. The notes are interconnected via [[wikilinks]] and #tags.

Your job: answer the user's question using ONLY information found in the vault.

How to work:
1. Use the tools to explore. Typically: `search_notes` (or `search_by_tag`) to \
find candidates, `read_note` to read them in full, and `get_backlinks` / follow \
[[wikilinks]] to gather related context.
2. Read the actual note contents before answering. Do not guess.
3. If the notes do not contain the answer, say so plainly rather than inventing it.

How to answer:
- Be concise and concrete.
- Cite the relative path of every note you rely on, inline, like \
(source: folder/Note.md), next to the claim it supports.
- Prefer quoting or closely paraphrasing the notes over speculation.
"""


class AgentState(TypedDict):
    """Conversation state: an append-only list of messages."""

    messages: Annotated[list, add_messages]


@dataclass
class AgentResult:
    """Outcome of an agent run."""

    answer: str
    references: list[str]
    truncated: bool  # True if the loop hit its iteration limit


def build_graph(settings: Settings, vault: Vault, llm=None):
    """Compile the agent graph and return ``(graph, tools)``.

    ``llm`` lets callers inject a chat model (e.g. a fake model in tests); when
    omitted, a :class:`ChatOpenAI` is built from ``settings``.
    """
    tools = build_tools(vault)
    if llm is None:
        llm = ChatOpenAI(
            model=settings.model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
        )
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    builder.add_edge("tools", "agent")
    return builder.compile(), tools


def _collect_references(messages: list[BaseMessage]) -> list[str]:
    """Extract the relative paths of notes the agent actually read.

    ``read_note`` returns content whose first line is ``# <relative-path>``; we
    parse that so citations are canonical regardless of how the model named them.
    """
    refs: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "read_note":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            first = content.splitlines()[0] if content else ""
            if first.startswith("# ") and not first.startswith("# ERROR"):
                path = first[2:].strip()
                if path and path not in refs:
                    refs.append(path)
    return refs


def _final_answer(messages: list[BaseMessage]) -> str:
    """Return the content of the last AI message (the agent's final answer)."""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and not getattr(msg, "tool_calls", None):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def run_agent(
    query: str,
    settings: Settings,
    on_event: Callable[[str, object], None] | None = None,
    llm=None,
) -> AgentResult:
    """Run the agent loop for ``query`` and return the answer plus references.

    The graph is streamed once in ``updates`` mode; messages are accumulated as
    they arrive (so the loop runs exactly once). ``on_event(node, payload)`` is
    invoked for each step when provided, so callers can render the loop verbosely.
    ``llm`` lets tests inject a fake chat model.
    """
    vault = Vault(settings.resolved_vault())
    graph, _ = build_graph(settings, vault, llm=llm)

    inputs = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    }
    # recursion_limit counts super-steps; ~2 per agent/tools cycle, +1 to finish.
    config = {"recursion_limit": settings.max_iterations * 2 + 1}

    messages: list[BaseMessage] = list(inputs["messages"])
    truncated = False
    try:
        for update in graph.stream(inputs, config=config, stream_mode="updates"):
            for node, payload in update.items():
                if on_event is not None:
                    on_event(node, payload)
                if isinstance(payload, dict) and "messages" in payload:
                    messages.extend(payload["messages"])
    except GraphRecursionError:
        truncated = True

    answer = _final_answer(messages)
    if truncated and not answer:
        answer = (
            "I reached the tool-call limit before finishing. Try a narrower "
            "question or raise --max-iters."
        )
    return AgentResult(
        answer=answer,
        references=_collect_references(messages),
        truncated=truncated,
    )
