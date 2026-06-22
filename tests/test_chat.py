"""Tests for multi-turn chat: history trimming, per-turn references, and the CLI.

The graph is exercised with a fake tool-calling model (no LLM endpoint), and the
CLI is driven with typer's CliRunner.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from typer.testing import CliRunner

from obsidian_agent.agent import (
    MAX_HISTORY_MESSAGES,
    AgentResult,
    _collect_references,
    run_agent,
    trim_history,
)
from obsidian_agent.cli import app
from obsidian_agent.config import Settings

VAULT_DIR = Path(__file__).parent / "fixtures" / "vault"
runner = CliRunner()


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Fake model that supports ``bind_tools`` and replays responses in order."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, D102
        return self


def make_settings() -> Settings:
    return Settings(OBSIDIAN_VAULT_PATH=str(VAULT_DIR))


# --- trim_history (unit) -----------------------------------------------------
def test_trim_history_keeps_qa_drops_tools():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="q1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_note", "args": {"path": "x"}, "id": "c1"}],
        ),
        ToolMessage(content="# x\n   1 | hi", name="read_note", tool_call_id="c1"),
        AIMessage(content="final answer"),
    ]
    kept = trim_history(messages)
    assert [type(m).__name__ for m in kept] == ["HumanMessage", "AIMessage"]
    assert kept[0].content == "q1"
    assert kept[1].content == "final answer"


def test_trim_history_caps_length():
    messages: list = []
    for i in range(10):  # 10 Q&A pairs = 20 keepable messages
        messages.append(HumanMessage(content=f"q{i}"))
        messages.append(AIMessage(content=f"a{i}"))
    kept = trim_history(messages)
    assert len(kept) == MAX_HISTORY_MESSAGES
    assert kept[-1].content == "a9"  # most recent retained


# --- per-turn reference scoping ---------------------------------------------
def test_references_scoped_per_turn():
    messages = [
        ToolMessage(content="# Old/Note.md\n   1 | x", name="read_note", tool_call_id="0"),
        HumanMessage(content="q2"),
        ToolMessage(content="# New/Note.md\n   1 | y", name="read_note", tool_call_id="1"),
    ]
    turn_start = 1
    assert _collect_references(messages[turn_start:]) == ["New/Note.md"]


# --- two-turn run ------------------------------------------------------------
def _two_turn_model() -> ToolCallingFakeModel:
    return ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_note", "args": {"path": "Vendor-Onboarding"}, "id": "c1"}
                ],
            ),
            AIMessage(content="Answer one (source: Operations/Vendor-Onboarding.md)."),
            AIMessage(content="Answer two, no new sources."),
        ]
    )


def test_two_turn_history_carried_forward():
    settings = make_settings()
    llm = _two_turn_model()

    r1 = run_agent("How do we onboard a vendor?", settings, llm=llm)
    assert r1.references == ["Operations/Vendor-Onboarding.md"]

    history = trim_history(r1.messages)
    assert [type(m).__name__ for m in history] == ["HumanMessage", "AIMessage"]
    assert not any(isinstance(m, ToolMessage) for m in history)

    r2 = run_agent("Anything else?", settings, llm=llm, history=history)
    assert "Answer two" in r2.answer
    assert r2.references == []  # per-turn: no read_note this turn


# --- CLI ---------------------------------------------------------------------
def test_cli_chat_exits_cleanly():
    result = runner.invoke(app, ["--chat", "--vault", str(VAULT_DIR)], input="/exit\n")
    assert result.exit_code == 0


def test_cli_chat_reset_then_exit():
    result = runner.invoke(
        app, ["--chat", "--vault", str(VAULT_DIR)], input="/reset\n/exit\n"
    )
    assert result.exit_code == 0


def test_cli_oneshot_routes(monkeypatch):
    seen: dict = {}

    def fake_run(query, settings, on_event=None, llm=None, history=None):
        seen["query"] = query
        return AgentResult(answer="canned answer", references=[], truncated=False)

    monkeypatch.setattr("obsidian_agent.cli.run_agent", fake_run)
    result = runner.invoke(app, ["What is the vendor process", "--vault", str(VAULT_DIR)])
    assert result.exit_code == 0
    assert seen["query"] == "What is the vendor process"
    assert "canned answer" in result.stdout


def test_cli_requires_query_or_chat():
    result = runner.invoke(app, [])
    assert result.exit_code == 2
