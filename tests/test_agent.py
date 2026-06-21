"""End-to-end test of the agent loop using a fake tool-calling model.

This exercises the real StateGraph (agent -> tools -> agent) and the deterministic
reference collection, without contacting any LLM endpoint.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from obsidian_agent.agent import build_graph, run_agent
from obsidian_agent.config import Settings
from obsidian_agent.vault import Vault

VAULT_DIR = Path(__file__).parent / "fixtures" / "vault"


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Fake model that supports ``bind_tools`` (the base class does not).

    It simply returns its predefined responses in order, ignoring the tools.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, D102
        return self


def make_settings() -> Settings:
    return Settings(OBSIDIAN_VAULT_PATH=str(VAULT_DIR))


def fake_model() -> ToolCallingFakeModel:
    """A model that first calls read_note, then returns a final answer.

    Responses are returned in order on each invocation.
    """
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_note",
                    "args": {"path": "Vendor-Onboarding"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(
            content="Sign the master service agreement "
            "(source: Operations/Vendor-Onboarding.md)."
        ),
    ]
    return ToolCallingFakeModel(responses=responses)


def test_graph_routes_through_tools_then_ends():
    settings = make_settings()
    graph, _ = build_graph(settings, Vault(settings.resolved_vault()), llm=fake_model())
    nodes = list(graph.get_graph().nodes)
    assert {"agent", "tools"} <= set(nodes)


def test_run_agent_reads_note_and_cites_it():
    result = run_agent("How do we onboard a vendor?", make_settings(), llm=fake_model())
    assert "master service agreement" in result.answer
    # Reference is collected from the note the agent actually read.
    assert result.references == ["Operations/Vendor-Onboarding.md"]
    assert result.truncated is False


def test_run_agent_emits_events():
    events: list[str] = []
    run_agent(
        "How do we onboard a vendor?",
        make_settings(),
        on_event=lambda node, _payload: events.append(node),
        llm=fake_model(),
    )
    assert "agent" in events
    assert "tools" in events
