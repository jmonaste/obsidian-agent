"""The LangGraph agent: a Claude-style think -> act -> observe loop over the vault.

The graph has two nodes:

* ``agent`` calls the LLM (bound to the vault tools).
* ``tools`` executes any tool calls and feeds results back.

It loops until the model answers with no further tool calls. References are then
collected deterministically from the notes the agent actually read.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .config import Settings
from .tools import build_tools
from .vault import Vault

SYSTEM_PROMPT = """\
You are a research assistant over the user's Obsidian vault: Markdown notes \
linked by [[wikilinks]] and labeled with #tags and YAML `tags:` frontmatter. You \
can only READ the vault through the provided tools; you cannot see any note you \
have not read.

Answer ONLY from the vault. Workflow:
1. EXPLORE to find candidates: `search_notes` for content, `search_by_tag` for \
topics, `glob_notes` / `list_dir` for structure. Narrow before you read.
2. READ candidates with `read_note` before making any claim. Never guess.
3. Follow [[wikilinks]] or `get_backlinks` only when they add needed context.
4. STOP as soon as the notes you have read answer the question. Don't over-explore.

Tool output is bounded to protect a limited context window:
- Search/list tools may end with "(showing N of M)". If you need more, narrow the \
query or use a more specific tool — don't assume hidden items are irrelevant.
- `read_note` reports the note's total size; for a long note it returns a window \
plus a "[more: ...]" hint. Call `read_note` again with `offset`/`limit` ONLY if \
the missing part likely matters.

How to answer:
- Be concise and concrete; quote or closely paraphrase the notes.
- Cite the relative path of each note you relied on, inline, like \
(source: folder/Note.md), next to the claim it supports.
- Cite ONLY notes you actually read with `read_note`.
- If the vault lacks the answer, say so plainly rather than inventing it.
"""


class AgentState(TypedDict):
    """Conversation state: an append-only list of messages."""

    messages: Annotated[list, add_messages]


MAX_HISTORY_MESSAGES = 12  # carried-forward chat memory cap (~6 Q&A turns)


@dataclass
class AgentResult:
    """Outcome of an agent run."""

    answer: str
    references: list[str]
    truncated: bool  # True if the loop hit its iteration limit
    messages: list[BaseMessage] = field(default_factory=list)  # full run transcript


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


def trim_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Reduce a finished run to lean prior context for the next chat turn.

    Keeps the question/answer transcript and drops the bulky act/observe
    scratchpad so a multi-turn session stays small on a limited context window:

    * KEEP ``HumanMessage`` and final ``AIMessage`` (no tool calls).
    * DROP ``SystemMessage`` (re-added each turn), ``ToolMessage``, and any
      ``AIMessage`` that carried tool calls.

    Only the most recent ``MAX_HISTORY_MESSAGES`` are retained.
    """
    kept: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            kept.append(msg)
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            kept.append(msg)
    return kept[-MAX_HISTORY_MESSAGES:]


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
    history: list[BaseMessage] | None = None,
) -> AgentResult:
    """Run the agent loop for ``query`` and return the answer plus references.

    The graph is streamed once in ``updates`` mode; messages are accumulated as
    they arrive (so the loop runs exactly once). ``on_event(node, payload)`` is
    invoked for each step when provided, so callers can render the loop verbosely.
    ``llm`` lets tests inject a fake chat model. ``history`` carries prior chat
    turns (see :func:`trim_history`); references reflect only the current turn.
    """
    vault = Vault(settings.resolved_vault())
    graph, _ = build_graph(settings, vault, llm=llm)

    inputs = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            *(history or []),
            HumanMessage(content=query),
        ]
    }
    # recursion_limit counts super-steps; ~2 per agent/tools cycle, +1 to finish.
    config = {"recursion_limit": settings.max_iterations * 2 + 1}

    messages: list[BaseMessage] = list(inputs["messages"])
    turn_start = len(messages)  # references/answer come from this turn's messages
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

    new_messages = messages[turn_start:]
    answer = _final_answer(new_messages)
    if truncated and not answer:
        answer = (
            "I reached the tool-call limit before finishing. Try a narrower "
            "question or raise --max-iters."
        )
    return AgentResult(
        answer=answer,
        references=_collect_references(new_messages),
        truncated=truncated,
        messages=messages,
    )
